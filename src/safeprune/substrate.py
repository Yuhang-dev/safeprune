from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

from .pruning import ScoreWeights, select_pruned_indices
from .scoring import (
    FFNActivationStats,
    LayerScores,
    build_layerwise_pruning_plan,
    _get_decoder_layers,
)


@dataclass(frozen=True)
class LayerBudgetOption:
    layer: int
    sparsity: float
    pruned_channels: int
    num_mlp_channels: int
    estimated_loss_delta: float


@dataclass(frozen=True)
class LayerwiseBudgetPlan:
    global_target: float
    target_pruned_channels: int
    total_channels: int
    selected_options: list[LayerBudgetOption]
    estimated_loss_delta: float

    @property
    def layer_sparsities(self) -> dict[int, float]:
        return {
            option.layer: option.sparsity
            for option in self.selected_options
            if option.sparsity > 0.0
        }

    @property
    def actual_sparsity(self) -> float:
        pruned = sum(option.pruned_channels for option in self.selected_options)
        return pruned / self.total_channels if self.total_channels else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_target": self.global_target,
            "target_pruned_channels": self.target_pruned_channels,
            "total_channels": self.total_channels,
            "actual_sparsity": self.actual_sparsity,
            "estimated_loss_delta": self.estimated_loss_delta,
            "layer_sparsities": {
                str(layer): sparsity
                for layer, sparsity in sorted(self.layer_sparsities.items())
            },
            "selected_options": [
                {
                    "layer": option.layer,
                    "sparsity": option.sparsity,
                    "pruned_channels": option.pruned_channels,
                    "num_mlp_channels": option.num_mlp_channels,
                    "estimated_loss_delta": option.estimated_loss_delta,
                }
                for option in self.selected_options
            ],
        }


def build_budget_options(
    *,
    scores: list[LayerScores],
    candidate_sparsities: list[float],
    min_mlp_channels_per_layer: int,
    loss_delta_by_layer: dict[int, dict[float, float]] | None = None,
) -> dict[int, list[LayerBudgetOption]]:
    options: dict[int, list[LayerBudgetOption]] = {}
    for layer_scores in scores:
        layer = int(layer_scores.layer)
        channel_count = len(layer_scores.mlp)
        layer_loss = loss_delta_by_layer.get(layer, {}) if loss_delta_by_layer else {}
        layer_options = []
        for sparsity in candidate_sparsities:
            pruned_channels = len(
                select_pruned_indices(
                    [1.0] * channel_count,
                    float(sparsity),
                    min_mlp_channels_per_layer,
                )
            )
            layer_options.append(
                LayerBudgetOption(
                    layer=layer,
                    sparsity=float(sparsity),
                    pruned_channels=pruned_channels,
                    num_mlp_channels=channel_count,
                    estimated_loss_delta=float(layer_loss.get(float(sparsity), float(sparsity))),
                )
            )
        options[layer] = sorted(layer_options, key=lambda item: item.sparsity)
    return options


def optimize_layerwise_budget(
    options_by_layer: dict[int, list[LayerBudgetOption]],
    global_target: float,
) -> LayerwiseBudgetPlan:
    if not 0.0 <= global_target < 1.0:
        raise ValueError("global_target must be in [0, 1).")
    if not options_by_layer:
        raise ValueError("At least one layer of budget options is required.")

    layers = sorted(options_by_layer)
    total_channels = sum(options_by_layer[layer][0].num_mlp_channels for layer in layers)
    target_pruned = math.ceil(total_channels * global_target)
    states: dict[int, tuple[float, tuple[LayerBudgetOption, ...]]] = {0: (0.0, tuple())}

    for layer in layers:
        next_states: dict[int, tuple[float, tuple[LayerBudgetOption, ...]]] = {}
        for pruned_so_far, (loss_so_far, chosen_so_far) in states.items():
            for option in options_by_layer[layer]:
                pruned = pruned_so_far + option.pruned_channels
                loss = loss_so_far + option.estimated_loss_delta
                chosen = (*chosen_so_far, option)
                existing = next_states.get(pruned)
                if existing is None or _budget_state_is_better(
                    loss,
                    chosen,
                    existing[0],
                    existing[1],
                ):
                    next_states[pruned] = (loss, chosen)
        states = next_states

    feasible = [
        (pruned, loss, chosen)
        for pruned, (loss, chosen) in states.items()
        if pruned >= target_pruned
    ]
    if not feasible:
        raise ValueError("No layer-wise budget allocation can satisfy the target.")
    pruned, loss, chosen = min(feasible, key=lambda item: (item[1], item[0]))
    return LayerwiseBudgetPlan(
        global_target=global_target,
        target_pruned_channels=target_pruned,
        total_channels=total_channels,
        selected_options=list(chosen),
        estimated_loss_delta=loss,
    )


def build_plan_from_budget(
    *,
    scores: list[LayerScores],
    budget: LayerwiseBudgetPlan,
    weights: ScoreWeights,
    min_mlp_channels_per_layer: int,
    plan_name: str | None = None,
) -> dict[str, Any]:
    return build_layerwise_pruning_plan(
        scores=scores,
        layer_sparsities=budget.layer_sparsities,
        weights=weights,
        min_mlp_channels_per_layer=min_mlp_channels_per_layer,
        global_target=budget.global_target,
        estimated_loss_delta=budget.estimated_loss_delta,
        plan_name=plan_name,
    )


def add_bias_compensation_to_plan(
    model,
    activation_stats: list[FFNActivationStats],
    plan: dict[str, Any],
) -> dict[str, Any]:
    torch = _require_torch()
    compensated = copy.deepcopy(plan)
    layers = _get_decoder_layers(model)
    stats_by_layer = {int(item.layer): item for item in activation_stats}
    for layer_plan in compensated.get("layers", []):
        layer_idx = int(layer_plan["layer"])
        pruned = [int(idx) for idx in layer_plan.get("pruned_mlp_channels", [])]
        if not pruned:
            layer_plan["mlp_output_bias_compensation"] = []
            continue
        stats = stats_by_layer[layer_idx]
        mean = torch.tensor(stats.mean, dtype=torch.float32, device=layers[layer_idx].mlp.down_proj.weight.device)
        pruned_tensor = torch.tensor(pruned, dtype=torch.long, device=mean.device)
        down_weight = layers[layer_idx].mlp.down_proj.weight.detach().float()
        compensation = down_weight[:, pruned_tensor] @ mean[pruned_tensor]
        layer_plan["mlp_output_bias_compensation"] = compensation.cpu().tolist()
    compensated["compensation"] = "bias"
    return compensated


def add_layer_output_scales(
    plan: dict[str, Any],
    layer_scales: dict[int, float],
) -> dict[str, Any]:
    scaled = copy.deepcopy(plan)
    for layer_plan in scaled.get("layers", []):
        layer_idx = int(layer_plan["layer"])
        if layer_idx in layer_scales:
            layer_plan["mlp_output_scale"] = float(layer_scales[layer_idx])
    scaled["scale_calibration"] = "layer_scalar"
    return scaled


def compute_per_layer_loss_delta_matrix(
    *,
    model,
    tokenizer,
    scores: list[LayerScores],
    prompts: list[str],
    candidate_sparsities: list[float],
    max_length: int,
    weights: ScoreWeights,
    min_mlp_channels_per_layer: int,
) -> dict[int, dict[float, float]]:
    from .masks import attach_qwen_switchable_mlp_masks

    base_loss = _average_lm_loss(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_length=max_length,
    )
    handle = attach_qwen_switchable_mlp_masks(model)
    matrix: dict[int, dict[float, float]] = {}
    try:
        for layer_scores in scores:
            layer_idx = int(layer_scores.layer)
            matrix[layer_idx] = {}
            for sparsity in candidate_sparsities:
                plan = build_layerwise_pruning_plan(
                    scores=scores,
                    layer_sparsities={layer_idx: float(sparsity)},
                    weights=weights,
                    min_mlp_channels_per_layer=min_mlp_channels_per_layer,
                )
                handle.set_plan(plan)
                loss = _average_lm_loss(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    max_length=max_length,
                )
                matrix[layer_idx][float(sparsity)] = float(loss - base_loss)
    finally:
        handle.remove()
    return matrix


def _average_lm_loss(
    *,
    model,
    tokenizer,
    prompts: list[str],
    max_length: int,
) -> float:
    torch = _require_torch()
    losses = []
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
            outputs = model(**encoded, labels=encoded["input_ids"])
            losses.append(float(outputs.loss.detach().float().cpu()))
    if not losses:
        raise ValueError("At least one prompt is required to compute loss.")
    return sum(losses) / len(losses)


def _budget_state_is_better(
    loss: float,
    chosen: tuple[LayerBudgetOption, ...],
    existing_loss: float,
    existing_chosen: tuple[LayerBudgetOption, ...],
) -> bool:
    if loss != existing_loss:
        return loss < existing_loss
    return tuple(option.sparsity for option in chosen) < tuple(
        option.sparsity for option in existing_chosen
    )


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on remote env
        raise RuntimeError("Torch is required for pruning substrate v2.") from exc
    return torch
