from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .stage_masks import active_mlp_ratio_from_plan


@dataclass(frozen=True)
class CompactLayerMetadata:
    layer: int
    original_channels: int
    remaining_channels: int
    pruned_channels: int
    has_bias_compensation: bool
    output_scale: float


def materialize_qwen_compact_mlp_subnet(
    model,
    plan: dict[str, Any],
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Physically slice Qwen-style FFN intermediate channels in-place.

    This keeps transformer hidden size unchanged and only shrinks each layer's
    FFN intermediate width. Attention, embeddings, hidden dimensions, and KV
    cache layout are untouched.
    """

    torch = _require_torch()
    layers = _get_decoder_layers(model)
    hidden_size = int(getattr(model.config, "hidden_size"))
    default_intermediate = int(getattr(model.config, "intermediate_size"))
    layer_plans = {
        int(layer_plan["layer"]): layer_plan
        for layer_plan in plan.get("layers", [])
    }
    metadata: list[CompactLayerMetadata] = []

    for layer_idx, layer in enumerate(layers):
        layer_plan = layer_plans.get(layer_idx, {})
        mlp = layer.mlp
        original_channels = int(mlp.gate_proj.weight.shape[0])
        if int(mlp.up_proj.weight.shape[0]) != original_channels:
            raise ValueError(f"Layer {layer_idx}: gate/up intermediate width mismatch.")
        if int(mlp.down_proj.weight.shape[1]) != original_channels:
            raise ValueError(f"Layer {layer_idx}: down_proj input width mismatch.")

        pruned = [int(idx) for idx in layer_plan.get("pruned_mlp_channels", [])]
        if strict:
            _validate_pruned_channels(pruned, original_channels, layer_idx)
        pruned_set = set(idx for idx in pruned if 0 <= idx < original_channels)
        keep = [idx for idx in range(original_channels) if idx not in pruned_set]
        if not keep:
            raise ValueError(f"Layer {layer_idx}: compact MLP would have zero channels.")

        keep_tensor = torch.tensor(
            keep,
            dtype=torch.long,
            device=mlp.gate_proj.weight.device,
        )
        compact_gate = _slice_linear_rows(mlp.gate_proj, keep_tensor)
        compact_up = _slice_linear_rows(mlp.up_proj, keep_tensor)
        compact_down = _slice_linear_columns(mlp.down_proj, keep_tensor)

        compensation = layer_plan.get("mlp_output_bias_compensation") or []
        if compensation and len(compensation) != hidden_size:
            raise ValueError(
                f"Layer {layer_idx}: bias compensation width {len(compensation)} "
                f"does not match hidden size {hidden_size}."
            )
        output_scale = float(layer_plan.get("mlp_output_scale", 1.0))

        layer.mlp = _make_compact_qwen_mlp(
            gate_proj=compact_gate,
            up_proj=compact_up,
            down_proj=compact_down,
            act_fn=getattr(mlp, "act_fn"),
            bias_compensation=compensation,
            output_scale=output_scale,
        )
        metadata.append(
            CompactLayerMetadata(
                layer=layer_idx,
                original_channels=original_channels,
                remaining_channels=len(keep),
                pruned_channels=len(pruned_set),
                has_bias_compensation=bool(compensation),
                output_scale=output_scale,
            )
        )

    total_original = sum(item.original_channels for item in metadata)
    total_remaining = sum(item.remaining_channels for item in metadata)
    total_pruned = sum(item.pruned_channels for item in metadata)
    return {
        "format": "safeprune.compact_subnet.v1",
        "note": "Runtime compact FFN intermediate-width subnet; hidden size is unchanged.",
        "plan_name": plan.get("plan_name") or plan.get("name"),
        "active_mlp_ratio_from_plan": active_mlp_ratio_from_plan(plan),
        "active_mlp_ratio_actual": total_remaining / total_original
        if total_original
        else 0.0,
        "total_original_channels": total_original,
        "total_remaining_channels": total_remaining,
        "total_pruned_channels": total_pruned,
        "default_intermediate_size": default_intermediate,
        "layers": [item.__dict__ for item in metadata],
    }


def write_compact_metadata(metadata: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def collect_prompts_from_jsonl(path: str | Path, limit: int | None = None) -> list[str]:
    prompts: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = _prompt_from_row(row)
            if prompt:
                prompts.append(prompt)
            if limit is not None and len(prompts) >= limit:
                break
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def compare_last_token_logits(
    *,
    model,
    tokenizer,
    prompts: list[str],
    max_length: int,
) -> list[dict[str, Any]]:
    torch = _require_torch()
    model.eval()
    results = []
    with torch.no_grad():
        for idx, prompt in enumerate(prompts):
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            device = next(model.parameters()).device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            logits = outputs.logits[:, -1, :].detach().float().cpu()
            results.append({"index": idx, "prompt": prompt, "logits": logits})
    return results


def summarize_logits_equivalence(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    torch = _require_torch()
    if len(baseline) != len(candidate):
        raise ValueError("Baseline and candidate result counts differ.")
    max_abs = 0.0
    total_abs = 0.0
    total_values = 0
    top1_matches = 0
    per_prompt = []
    for base, cand in zip(baseline, candidate, strict=True):
        base_logits = base["logits"]
        cand_logits = cand["logits"]
        if tuple(base_logits.shape) != tuple(cand_logits.shape):
            raise ValueError("Logit shape mismatch.")
        diff = (base_logits - cand_logits).abs()
        prompt_max = float(diff.max().item())
        prompt_mean = float(diff.mean().item())
        max_abs = max(max_abs, prompt_max)
        total_abs += float(diff.sum().item())
        total_values += int(diff.numel())
        base_top1 = int(torch.argmax(base_logits, dim=-1).item())
        cand_top1 = int(torch.argmax(cand_logits, dim=-1).item())
        match = base_top1 == cand_top1
        top1_matches += int(match)
        per_prompt.append(
            {
                "index": base["index"],
                "max_abs_diff": prompt_max,
                "mean_abs_diff": prompt_mean,
                "baseline_top1": base_top1,
                "candidate_top1": cand_top1,
                "top1_match": match,
            }
        )
    return {
        "count": len(baseline),
        "logits_max_abs_diff": max_abs,
        "logits_mean_abs_diff": total_abs / total_values if total_values else 0.0,
        "top1_match_rate": top1_matches / len(baseline) if baseline else 0.0,
        "per_prompt": per_prompt,
    }


def greedy_generations(
    *,
    model,
    tokenizer,
    prompts: list[str],
    max_length: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    torch = _require_torch()
    model.eval()
    rows = []
    with torch.no_grad():
        for idx, prompt in enumerate(prompts):
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            device = next(model.parameters()).device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output_ids = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
            rows.append(
                {
                    "index": idx,
                    "prompt": prompt,
                    "text": tokenizer.decode(output_ids[0], skip_special_tokens=True),
                }
            )
    return rows


def summarize_generation_equivalence(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(baseline) != len(candidate):
        raise ValueError("Baseline and candidate generation counts differ.")
    matches = 0
    per_prompt = []
    for base, cand in zip(baseline, candidate, strict=True):
        match = base["text"] == cand["text"]
        matches += int(match)
        per_prompt.append(
            {
                "index": base["index"],
                "exact_match": match,
                "baseline_text": base["text"],
                "candidate_text": cand["text"],
            }
        )
    return {
        "count": len(baseline),
        "exact_match_rate": matches / len(baseline) if baseline else 0.0,
        "per_prompt": per_prompt,
    }


def _slice_linear_rows(linear, keep_tensor):
    torch = _require_torch()
    out_features = int(keep_tensor.numel())
    in_features = int(linear.weight.shape[1])
    sliced = torch.nn.Linear(
        in_features,
        out_features,
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    with torch.no_grad():
        sliced.weight.copy_(linear.weight.index_select(0, keep_tensor))
        if linear.bias is not None:
            sliced.bias.copy_(linear.bias.index_select(0, keep_tensor))
    return sliced


def _slice_linear_columns(linear, keep_tensor):
    torch = _require_torch()
    out_features = int(linear.weight.shape[0])
    in_features = int(keep_tensor.numel())
    sliced = torch.nn.Linear(
        in_features,
        out_features,
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    with torch.no_grad():
        sliced.weight.copy_(linear.weight.index_select(1, keep_tensor))
        if linear.bias is not None:
            sliced.bias.copy_(linear.bias)
    return sliced


def _make_compact_qwen_mlp(
    *,
    gate_proj,
    up_proj,
    down_proj,
    act_fn,
    bias_compensation,
    output_scale,
):
    torch = _require_torch()

    class _CompactQwenMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = gate_proj
            self.up_proj = up_proj
            self.down_proj = down_proj
            self.act_fn = act_fn
            hidden_size = int(down_proj.weight.shape[0])
            bias = bias_compensation or [0.0] * hidden_size
            self.register_buffer(
                "mlp_output_bias_compensation",
                torch.tensor(bias, dtype=torch.float32).reshape(1, 1, hidden_size),
                persistent=True,
            )
            self.register_buffer(
                "mlp_output_scale",
                torch.tensor(float(output_scale), dtype=torch.float32),
                persistent=True,
            )

        def forward(self, x):
            hidden = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            output = self.down_proj(hidden)
            bias = self.mlp_output_bias_compensation.to(
                device=output.device,
                dtype=output.dtype,
            )
            scale = self.mlp_output_scale.to(device=output.device, dtype=output.dtype)
            return output * scale + bias

    return _CompactQwenMLP()


def _validate_pruned_channels(pruned: list[int], channel_count: int, layer_idx: int) -> None:
    if len(pruned) != len(set(pruned)):
        raise ValueError(f"Layer {layer_idx}: duplicate pruned channels.")
    invalid = [idx for idx in pruned if idx < 0 or idx >= channel_count]
    if invalid:
        raise ValueError(f"Layer {layer_idx}: out-of-range pruned channels: {invalid[:5]}")


def _prompt_from_row(row: dict[str, Any]) -> str | None:
    for key in ["prompt", "user_request", "text"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    messages = row.get("messages")
    if isinstance(messages, list):
        parts = []
        for message in messages:
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                parts.append(f"{message.get('role', 'user')}: {message['content']}")
        if parts:
            return "\n".join(parts)
    return None


def _get_decoder_layers(model) -> list[Any]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise ValueError("Unsupported model layout: expected model.model.layers or transformer.h")


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on remote env
        raise RuntimeError("Torch is required for compact subnet materialization.") from exc
    return torch
