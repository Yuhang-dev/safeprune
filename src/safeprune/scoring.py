from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import iter_agent_stage_texts, load_agent_jsonl, load_preference_jsonl
from .pruning import ScoreWeights, combine_importance_scores, select_pruned_indices


@dataclass(frozen=True)
class LayerScores:
    layer: int
    attention: list[float]
    mlp: list[float]
    activation_attention: list[float] | None = None
    activation_mlp: list[float] | None = None
    loss_delta_attention: list[float] | None = None
    loss_delta_mlp: list[float] | None = None


@dataclass(frozen=True)
class FFNActivationStats:
    layer: int
    mean: list[float]
    mean_abs: list[float]
    second_moment: list[float]
    variance: list[float]
    token_count: int


def _get_decoder_layers(model) -> list[Any]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise ValueError("Unsupported model layout: expected model.model.layers or transformer.h")


def _infer_attention_shape(model, layer) -> tuple[int, int]:
    num_heads = int(getattr(model.config, "num_attention_heads"))
    hidden_size = int(getattr(model.config, "hidden_size"))
    return num_heads, hidden_size // num_heads


def _linear_out_group_norm(weight, groups: int) -> list[float]:
    torch = _require_torch()
    if weight.shape[0] % groups != 0:
        raise ValueError(f"Cannot split {weight.shape[0]} output rows into {groups} groups.")
    rows_per_group = weight.shape[0] // groups
    grouped = weight.detach().float().abs().reshape(groups, rows_per_group, weight.shape[1])
    return torch.linalg.vector_norm(grouped, ord=2, dim=(1, 2)).cpu().tolist()


def _linear_in_group_norm(weight, groups: int) -> list[float]:
    torch = _require_torch()
    if weight.shape[1] % groups != 0:
        raise ValueError(f"Cannot split {weight.shape[1]} input columns into {groups} groups.")
    cols_per_group = weight.shape[1] // groups
    grouped = weight.detach().float().abs().reshape(weight.shape[0], groups, cols_per_group)
    return torch.linalg.vector_norm(grouped, ord=2, dim=(0, 2)).cpu().tolist()


def _optional_out_group_norm(weight, groups: int) -> list[float] | None:
    if weight.shape[0] % groups != 0:
        return None
    return _linear_out_group_norm(weight, groups)


def compute_magnitude_scores(model) -> list[LayerScores]:
    layers = _get_decoder_layers(model)
    results: list[LayerScores] = []
    for layer_idx, layer in enumerate(layers):
        num_heads, _ = _infer_attention_shape(model, layer)
        self_attn = layer.self_attn
        mlp = layer.mlp

        components = [
            _linear_out_group_norm(self_attn.q_proj.weight, num_heads),
            _linear_in_group_norm(self_attn.o_proj.weight, num_heads),
        ]
        k = _optional_out_group_norm(self_attn.k_proj.weight, num_heads)
        v = _optional_out_group_norm(self_attn.v_proj.weight, num_heads)
        if k is not None:
            components.append(k)
        if v is not None:
            components.append(v)
        attention = [sum(values) / len(values) for values in zip(*components)]

        gate = mlp.gate_proj.weight.detach().float().abs().mean(dim=1).cpu().tolist()
        up = mlp.up_proj.weight.detach().float().abs().mean(dim=1).cpu().tolist()
        down = mlp.down_proj.weight.detach().float().abs().mean(dim=0).cpu().tolist()
        mlp_scores = [(gate_i + up_i + down_i) / 3.0 for gate_i, up_i, down_i in zip(gate, up, down)]
        results.append(LayerScores(layer=layer_idx, attention=attention, mlp=mlp_scores))
    return results


def compute_activation_scores(
    model,
    tokenizer,
    calibration_path: str | Path,
    max_length: int,
    max_batches: int | None = None,
) -> list[LayerScores]:
    torch = _require_torch()
    layers = _get_decoder_layers(model)
    records = load_preference_jsonl(calibration_path)
    prompts = [record.prompt for record in records]
    if max_batches is not None:
        prompts = prompts[:max_batches]

    attn_sums: list[Any] = []
    mlp_sums: list[Any] = []
    handles = []
    num_heads = int(getattr(model.config, "num_attention_heads"))
    hidden_size = int(getattr(model.config, "hidden_size"))
    head_dim = hidden_size // num_heads
    intermediate = int(getattr(model.config, "intermediate_size"))

    for _ in layers:
        attn_sums.append(torch.zeros(num_heads, device="cpu"))
        mlp_sums.append(torch.zeros(intermediate, device="cpu"))

    for layer_idx, layer in enumerate(layers):

        def q_hook(_module, _inputs, output, idx=layer_idx):
            view = output.detach().float().reshape(output.shape[0], output.shape[1], num_heads, head_dim)
            attn_sums[idx].add_(view.abs().mean(dim=(0, 1, 3)).cpu())

        def down_pre_hook(_module, inputs, idx=layer_idx):
            # Qwen SwiGLU sends SiLU(gate_proj(x)) * up_proj(x) into down_proj.
            # This tensor is the actual per-channel FFN contribution and is a
            # better pruning signal than gate_proj alone.
            hidden = inputs[0]
            mlp_sums[idx].add_(hidden.detach().float().abs().mean(dim=(0, 1)).cpu())

        handles.append(layer.self_attn.q_proj.register_forward_hook(q_hook))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(down_pre_hook))

    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            device = next(model.parameters()).device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            model(**encoded)

    for handle in handles:
        handle.remove()

    denom = max(1, len(prompts))
    return [
        LayerScores(
            layer=idx,
            attention=(attn_sums[idx] / denom).cpu().tolist(),
            mlp=(mlp_sums[idx] / denom).cpu().tolist(),
        )
        for idx in range(len(layers))
    ]


def compute_ffn_activation_stats(
    model,
    tokenizer,
    calibration_path: str | Path,
    max_length: int,
    max_batches: int | None = None,
) -> list[FFNActivationStats]:
    records = load_preference_jsonl(calibration_path)
    prompts = [record.prompt for record in records]
    if max_batches is not None:
        prompts = prompts[:max_batches]
    return compute_ffn_activation_stats_for_prompts(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_length=max_length,
    )


def compute_ffn_activation_stats_for_prompts(
    model,
    tokenizer,
    prompts: list[str],
    max_length: int,
) -> list[FFNActivationStats]:
    """Collect SwiGLU intermediate activation statistics.

    The hook is attached to ``down_proj`` input, which for Qwen-style SwiGLU is
    ``SiLU(gate_proj(x)) * up_proj(x)``. This is the channel being pruned.
    """

    torch = _require_torch()
    layers = _get_decoder_layers(model)
    intermediate = int(getattr(model.config, "intermediate_size"))
    sums = [torch.zeros(intermediate, device="cpu") for _ in layers]
    abs_sums = [torch.zeros(intermediate, device="cpu") for _ in layers]
    square_sums = [torch.zeros(intermediate, device="cpu") for _ in layers]
    counts = [0 for _ in layers]
    handles = []

    for layer_idx, layer in enumerate(layers):

        def down_pre_hook(_module, inputs, idx=layer_idx):
            hidden = inputs[0].detach().float()
            sums[idx].add_(hidden.sum(dim=(0, 1)).cpu())
            abs_sums[idx].add_(hidden.abs().sum(dim=(0, 1)).cpu())
            square_sums[idx].add_((hidden * hidden).sum(dim=(0, 1)).cpu())
            counts[idx] += int(hidden.shape[0] * hidden.shape[1])

        handles.append(layer.mlp.down_proj.register_forward_pre_hook(down_pre_hook))

    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            device = next(model.parameters()).device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            model(**encoded)

    for handle in handles:
        handle.remove()

    stats: list[FFNActivationStats] = []
    for idx in range(len(layers)):
        denom = max(1, counts[idx])
        mean = sums[idx] / denom
        mean_abs = abs_sums[idx] / denom
        second_moment = square_sums[idx] / denom
        variance = (second_moment - mean * mean).clamp_min(0.0)
        stats.append(
            FFNActivationStats(
                layer=idx,
                mean=mean.cpu().tolist(),
                mean_abs=mean_abs.cpu().tolist(),
                second_moment=second_moment.cpu().tolist(),
                variance=variance.cpu().tolist(),
                token_count=counts[idx],
            )
        )
    return stats


def activation_scores_from_stats(stats: list[FFNActivationStats]) -> list[LayerScores]:
    return [
        LayerScores(layer=item.layer, attention=[], mlp=list(item.mean_abs))
        for item in stats
    ]


def compute_wanda_channel_scores(model, stats: list[FFNActivationStats]) -> list[LayerScores]:
    torch = _require_torch()
    layers = _get_decoder_layers(model)
    results = []
    for item in stats:
        down_weight = layers[item.layer].mlp.down_proj.weight.detach().float()
        down_norm = torch.linalg.vector_norm(down_weight, ord=2, dim=0).cpu().tolist()
        activation_l2 = [
            math.sqrt(max(float(value), 0.0))
            for value in item.second_moment
        ]
        scores = [
            float(weight_norm) * float(act_norm)
            for weight_norm, act_norm in zip(down_norm, activation_l2, strict=True)
        ]
        results.append(LayerScores(layer=item.layer, attention=[], mlp=scores))
    return results


def compute_flap_fluctuation_scores(model, stats: list[FFNActivationStats]) -> list[LayerScores]:
    torch = _require_torch()
    layers = _get_decoder_layers(model)
    results = []
    for item in stats:
        down_weight = layers[item.layer].mlp.down_proj.weight.detach().float()
        down_norm = torch.linalg.vector_norm(down_weight, ord=2, dim=0).cpu().tolist()
        scores = [
            (float(weight_norm) ** 2) * max(float(variance), 0.0)
            for weight_norm, variance in zip(down_norm, item.variance, strict=True)
        ]
        results.append(LayerScores(layer=item.layer, attention=[], mlp=scores))
    return results


def compute_stage_activation_scores(
    model,
    tokenizer,
    agent_path: str | Path,
    max_length: int,
    stages: list[str] | None = None,
    max_batches_per_stage: int | None = None,
) -> dict[str, list[LayerScores]]:
    trajectories = load_agent_jsonl(agent_path)
    grouped = iter_agent_stage_texts(trajectories, stages=stages)
    return {
        stage: _compute_activation_scores_for_prompts(
            model=model,
            tokenizer=tokenizer,
            prompts=texts[:max_batches_per_stage] if max_batches_per_stage else texts,
            max_length=max_length,
        )
        for stage, texts in grouped.items()
        if texts
    }


def _compute_activation_scores_for_prompts(
    model,
    tokenizer,
    prompts: list[str],
    max_length: int,
) -> list[LayerScores]:
    torch = _require_torch()
    layers = _get_decoder_layers(model)
    attn_sums: list[Any] = []
    mlp_sums: list[Any] = []
    handles = []
    num_heads = int(getattr(model.config, "num_attention_heads"))
    hidden_size = int(getattr(model.config, "hidden_size"))
    head_dim = hidden_size // num_heads
    intermediate = int(getattr(model.config, "intermediate_size"))

    for _ in layers:
        attn_sums.append(torch.zeros(num_heads, device="cpu"))
        mlp_sums.append(torch.zeros(intermediate, device="cpu"))

    for layer_idx, layer in enumerate(layers):

        def q_hook(_module, _inputs, output, idx=layer_idx):
            view = output.detach().float().reshape(output.shape[0], output.shape[1], num_heads, head_dim)
            attn_sums[idx].add_(view.abs().mean(dim=(0, 1, 3)).cpu())

        def down_pre_hook(_module, inputs, idx=layer_idx):
            hidden = inputs[0]
            mlp_sums[idx].add_(hidden.detach().float().abs().mean(dim=(0, 1)).cpu())

        handles.append(layer.self_attn.q_proj.register_forward_hook(q_hook))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(down_pre_hook))

    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            device = next(model.parameters()).device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            model(**encoded)

    for handle in handles:
        handle.remove()

    denom = max(1, len(prompts))
    return [
        LayerScores(
            layer=idx,
            attention=(attn_sums[idx] / denom).cpu().tolist(),
            mlp=(mlp_sums[idx] / denom).cpu().tolist(),
        )
        for idx in range(len(layers))
    ]


def merge_score_components(
    magnitude_scores: list[LayerScores],
    activation_scores: list[LayerScores] | None = None,
    loss_delta_scores: list[LayerScores] | None = None,
) -> list[LayerScores]:
    merged: list[LayerScores] = []
    for idx, magnitude in enumerate(magnitude_scores):
        activation = activation_scores[idx] if activation_scores else None
        loss_delta = loss_delta_scores[idx] if loss_delta_scores else None
        merged.append(
            LayerScores(
                layer=magnitude.layer,
                attention=magnitude.attention,
                mlp=magnitude.mlp,
                activation_attention=activation.attention if activation else None,
                activation_mlp=activation.mlp if activation else None,
                loss_delta_attention=loss_delta.attention if loss_delta else None,
                loss_delta_mlp=loss_delta.mlp if loss_delta else None,
            )
        )
    return merged


def build_pruning_plan(
    scores: list[LayerScores],
    sparsity: float,
    weights: ScoreWeights,
    min_heads_per_layer: int,
    min_mlp_channels_per_layer: int,
    prune_attention_heads: bool = True,
    prune_mlp_channels: bool = True,
) -> dict[str, Any]:
    plan_layers = []
    for layer_scores in scores:
        attention_importance = combine_importance_scores(
            layer_scores.attention,
            layer_scores.activation_attention or [0.0] * len(layer_scores.attention),
            layer_scores.loss_delta_attention or [0.0] * len(layer_scores.attention),
            weights,
        )
        mlp_importance = combine_importance_scores(
            layer_scores.mlp,
            layer_scores.activation_mlp or [0.0] * len(layer_scores.mlp),
            layer_scores.loss_delta_mlp or [0.0] * len(layer_scores.mlp),
            weights,
        )
        plan_layers.append(
            {
                "layer": layer_scores.layer,
                "pruned_attention_heads": (
                    select_pruned_indices(attention_importance, sparsity, min_heads_per_layer)
                    if prune_attention_heads
                    else []
                ),
                "pruned_mlp_channels": (
                    select_pruned_indices(mlp_importance, sparsity, min_mlp_channels_per_layer)
                    if prune_mlp_channels
                    else []
                ),
                "num_attention_heads": len(attention_importance),
                "num_mlp_channels": len(mlp_importance),
            }
        )
    return {"sparsity": sparsity, "layers": plan_layers}


def build_layerwise_pruning_plan(
    scores: list[LayerScores],
    layer_sparsities: dict[int, float],
    weights: ScoreWeights,
    min_mlp_channels_per_layer: int,
    *,
    global_target: float | None = None,
    estimated_loss_delta: float | None = None,
    plan_name: str | None = None,
) -> dict[str, Any]:
    plan_layers = []
    normalized = {int(layer): float(value) for layer, value in layer_sparsities.items()}
    for layer_scores in scores:
        sparsity = normalized.get(int(layer_scores.layer), 0.0)
        if not 0.0 <= sparsity < 1.0:
            raise ValueError("Layer sparsity values must be in [0, 1).")
        mlp_importance = combine_importance_scores(
            layer_scores.mlp,
            layer_scores.activation_mlp or [0.0] * len(layer_scores.mlp),
            layer_scores.loss_delta_mlp or [0.0] * len(layer_scores.mlp),
            weights,
        )
        plan_layers.append(
            {
                "layer": layer_scores.layer,
                "pruned_attention_heads": [],
                "pruned_mlp_channels": select_pruned_indices(
                    mlp_importance,
                    sparsity,
                    min_mlp_channels_per_layer,
                ),
                "num_attention_heads": len(layer_scores.attention),
                "num_mlp_channels": len(mlp_importance),
            }
        )
    return {
        "sparsity": global_target,
        "global_target": global_target,
        "estimated_loss_delta": estimated_loss_delta,
        "name": plan_name,
        "mask_type": "ffn_channel_forward_mask",
        "allocation": {
            str(layer): sparsity
            for layer, sparsity in sorted(normalized.items())
            if sparsity > 0.0
        },
        "layers": plan_layers,
    }


def save_scores(scores: list[LayerScores], path: str | Path) -> None:
    serializable = [
        {
            "layer": score.layer,
            "attention": score.attention,
            "mlp": score.mlp,
            "activation_attention": score.activation_attention,
            "activation_mlp": score.activation_mlp,
            "loss_delta_attention": score.loss_delta_attention,
            "loss_delta_mlp": score.loss_delta_mlp,
        }
        for score in scores
    ]
    Path(path).write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def load_scores(path: str | Path) -> list[LayerScores]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        LayerScores(
            layer=item["layer"],
            attention=item["attention"],
            mlp=item["mlp"],
            activation_attention=item.get("activation_attention"),
            activation_mlp=item.get("activation_mlp"),
            loss_delta_attention=item.get("loss_delta_attention"),
            loss_delta_mlp=item.get("loss_delta_mlp"),
        )
        for item in raw
    ]


def save_activation_stats(stats: list[FFNActivationStats], path: str | Path) -> None:
    serializable = [
        {
            "layer": item.layer,
            "mean": item.mean,
            "mean_abs": item.mean_abs,
            "second_moment": item.second_moment,
            "variance": item.variance,
            "token_count": item.token_count,
        }
        for item in stats
    ]
    Path(path).write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def load_activation_stats(path: str | Path) -> list[FFNActivationStats]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        FFNActivationStats(
            layer=int(item["layer"]),
            mean=[float(value) for value in item["mean"]],
            mean_abs=[float(value) for value in item["mean_abs"]],
            second_moment=[float(value) for value in item["second_moment"]],
            variance=[float(value) for value in item["variance"]],
            token_count=int(item["token_count"]),
        )
        for item in raw
    ]


def save_plan(plan: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(plan, indent=2), encoding="utf-8")


def load_plan(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on remote env
        raise RuntimeError("Torch is required to compute pruning scores.") from exc
    return torch
