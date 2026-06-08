from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ScoreWeights:
    magnitude: float = 0.40
    activation: float = 0.35
    loss_delta: float = 0.25

    @classmethod
    def from_mapping(cls, raw: dict[str, float]) -> "ScoreWeights":
        return cls(
            magnitude=float(raw.get("magnitude", 0.0)),
            activation=float(raw.get("activation", 0.0)),
            loss_delta=float(raw.get("loss_delta", 0.0)),
        )


def normalize_scores(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        return [0.0 for _ in values]
    return [(value - min_value) / (max_value - min_value) for value in values]


def combine_importance_scores(
    magnitude: Sequence[float],
    activation: Sequence[float],
    loss_delta: Sequence[float],
    weights: ScoreWeights,
) -> list[float]:
    if not (len(magnitude) == len(activation) == len(loss_delta)):
        raise ValueError("All score vectors must have the same length.")

    mag = normalize_scores(magnitude)
    act = normalize_scores(activation)
    delta = normalize_scores(loss_delta)
    return [
        weights.magnitude * mag_i + weights.activation * act_i + weights.loss_delta * delta_i
        for mag_i, act_i, delta_i in zip(mag, act, delta)
    ]


def select_pruned_indices(
    importance: Sequence[float],
    sparsity: float,
    min_remaining: int,
) -> list[int]:
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be in [0, 1).")
    total = len(importance)
    if total == 0:
        return []
    if min_remaining < 0:
        raise ValueError("min_remaining must be non-negative.")
    max_prunable = max(0, total - min_remaining)
    requested = int(total * sparsity)
    prune_count = min(requested, max_prunable)
    if prune_count == 0:
        return []
    ranked = sorted(range(total), key=lambda idx: (importance[idx], idx))
    return sorted(ranked[:prune_count])


def build_keep_mask(total: int, pruned_indices: Sequence[int]) -> list[bool]:
    pruned = set(pruned_indices)
    return [idx not in pruned for idx in range(total)]


def estimate_remaining(total: int, sparsity: float, min_remaining: int) -> int:
    return total - len(select_pruned_indices([1.0] * total, sparsity, min_remaining))


def require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on remote env
        raise RuntimeError("Torch is required for model pruning on the remote training host.") from exc
    return torch


def apply_linear_output_mask(linear, keep_mask: Sequence[bool]):
    torch = require_torch()
    mask = torch.tensor(keep_mask, dtype=torch.bool, device=linear.weight.device)
    linear.weight.data = linear.weight.data[mask, :].contiguous()
    if linear.bias is not None:
        linear.bias.data = linear.bias.data[mask].contiguous()
    linear.out_features = int(mask.sum().item())
    return linear


def apply_linear_input_mask(linear, keep_mask: Sequence[bool]):
    torch = require_torch()
    mask = torch.tensor(keep_mask, dtype=torch.bool, device=linear.weight.device)
    linear.weight.data = linear.weight.data[:, mask].contiguous()
    linear.in_features = int(mask.sum().item())
    return linear

