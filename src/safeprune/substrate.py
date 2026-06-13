from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

from .pruning import ScoreWeights, combine_importance_scores, select_pruned_indices
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


def build_nested_global_budget_plans(
    *,
    scores: list[LayerScores],
    target_budgets: list[float],
    weights: ScoreWeights,
    min_mlp_channels_per_layer: int,
    plan_name_prefix: str,
) -> dict[float, dict[str, Any]]:
    """Build globally ranked FFN channel plans with nested pruned sets."""

    if not target_budgets:
        raise ValueError("At least one target budget is required.")
    sorted_targets = sorted(float(target) for target in target_budgets)
    if any(target < 0.0 or target >= 1.0 for target in sorted_targets):
        raise ValueError("Target budgets must be in [0, 1).")

    layer_meta: dict[int, dict[str, Any]] = {}
    ranked_channels: list[tuple[float, int, int]] = []
    total_channels = 0
    for layer_scores in scores:
        layer = int(layer_scores.layer)
        importance = combine_importance_scores(
            layer_scores.mlp,
            layer_scores.activation_mlp or [0.0] * len(layer_scores.mlp),
            layer_scores.loss_delta_mlp or [0.0] * len(layer_scores.mlp),
            weights,
        )
        channel_count = len(importance)
        max_prunable = max(0, channel_count - min_mlp_channels_per_layer)
        layer_meta[layer] = {
            "scores": layer_scores,
            "num_mlp_channels": channel_count,
            "max_prunable": max_prunable,
        }
        total_channels += channel_count
        for channel, score in enumerate(importance):
            ranked_channels.append((float(score), layer, channel))

    ranked_channels.sort(key=lambda item: (item[0], item[1], item[2]))
    pruned_by_layer: dict[int, set[int]] = {layer: set() for layer in layer_meta}
    cursor = 0
    plans: dict[float, dict[str, Any]] = {}

    for target in sorted_targets:
        target_pruned = math.ceil(total_channels * target)
        while _total_pruned(pruned_by_layer) < target_pruned and cursor < len(ranked_channels):
            _score, layer, channel = ranked_channels[cursor]
            cursor += 1
            if len(pruned_by_layer[layer]) >= layer_meta[layer]["max_prunable"]:
                continue
            pruned_by_layer[layer].add(channel)

        plan = _plan_from_pruned_sets(
            scores=scores,
            pruned_by_layer=pruned_by_layer,
            target=target,
            total_channels=total_channels,
            plan_name=f"{plan_name_prefix}_{_budget_slug(target)}",
        )
        plans[target] = plan

    return plans


def validate_nested_pruned_sets(plans: dict[float, dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted((float(target), plan) for target, plan in plans.items())
    pairs = []
    warnings = []
    previous_target: float | None = None
    previous_set: set[tuple[int, int]] | None = None
    for target, plan in ordered:
        current_set = _global_pruned_set(plan)
        if previous_target is not None and previous_set is not None:
            missing = previous_set - current_set
            added = current_set - previous_set
            overlap = (
                len(previous_set & current_set) / len(previous_set)
                if previous_set
                else 1.0
            )
            pair = {
                "left_target": previous_target,
                "right_target": target,
                "left_pruned_count": len(previous_set),
                "right_pruned_count": len(current_set),
                "newly_pruned_count": len(added),
                "left_only_count": len(missing),
                "nesting_overlap": overlap,
                "left_subset_of_right": not missing,
            }
            pairs.append(pair)
            if missing:
                warnings.append(
                    f"{_budget_slug(previous_target)} is not nested in "
                    f"{_budget_slug(target)}: {len(missing)} missing channels"
                )
        previous_target = target
        previous_set = current_set
    return {"pairs": pairs, "warnings": warnings}


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


def _budget_slug(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _total_pruned(pruned_by_layer: dict[int, set[int]]) -> int:
    return sum(len(channels) for channels in pruned_by_layer.values())


def _plan_from_pruned_sets(
    *,
    scores: list[LayerScores],
    pruned_by_layer: dict[int, set[int]],
    target: float,
    total_channels: int,
    plan_name: str,
) -> dict[str, Any]:
    layers = []
    allocation = {}
    total_pruned = 0
    for layer_scores in scores:
        layer = int(layer_scores.layer)
        pruned = sorted(pruned_by_layer.get(layer, set()))
        channel_count = len(layer_scores.mlp)
        total_pruned += len(pruned)
        if pruned:
            allocation[str(layer)] = len(pruned) / channel_count if channel_count else 0.0
        layers.append(
            {
                "layer": layer_scores.layer,
                "pruned_attention_heads": [],
                "pruned_mlp_channels": pruned,
                "num_attention_heads": len(layer_scores.attention),
                "num_mlp_channels": channel_count,
            }
        )
    actual_sparsity = total_pruned / total_channels if total_channels else 0.0
    return {
        "sparsity": target,
        "global_target": target,
        "estimated_loss_delta": None,
        "name": plan_name,
        "plan_name": plan_name,
        "mask_type": "ffn_channel_forward_mask",
        "allocation": allocation,
        "layers": layers,
        "budget_plan": {
            "global_target": target,
            "target_pruned_channels": math.ceil(total_channels * target),
            "total_channels": total_channels,
            "actual_sparsity": actual_sparsity,
            "estimated_loss_delta": None,
            "nested_global_rank": True,
        },
    }


def _global_pruned_set(plan: dict[str, Any]) -> set[tuple[int, int]]:
    result = set()
    for layer in plan.get("layers", []):
        layer_idx = int(layer.get("layer", layer.get("layer_idx", 0)))
        for channel in layer.get("pruned_mlp_channels", []):
            result.add((layer_idx, int(channel)))
    return result


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
