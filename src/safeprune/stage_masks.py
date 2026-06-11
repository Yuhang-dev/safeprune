from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pruning import ScoreWeights
from .scoring import LayerScores, build_pruning_plan


def sparsity_key(sparsity: float) -> str:
    return f"{float(sparsity):.2f}"


@dataclass(frozen=True)
class StageMaskBank:
    """Stage-indexed FFN-only structured pruning plans.

    The bank stores mask-based plans for algorithm validation. It does not imply
    physical tensor slicing or wall-clock speedup.
    """

    plans: dict[str, dict[str, dict[str, Any]]]

    @classmethod
    def from_stage_scores(
        cls,
        stage_scores: dict[str, list[LayerScores]],
        target_sparsities: list[float],
        weights: ScoreWeights,
        min_mlp_channels_per_layer: int,
    ) -> "StageMaskBank":
        plans: dict[str, dict[str, dict[str, Any]]] = {}
        for stage, scores in stage_scores.items():
            plans[stage] = {}
            for sparsity in target_sparsities:
                plan = build_pruning_plan(
                    scores=scores,
                    sparsity=sparsity,
                    weights=weights,
                    min_heads_per_layer=1,
                    min_mlp_channels_per_layer=min_mlp_channels_per_layer,
                    prune_attention_heads=False,
                    prune_mlp_channels=True,
                )
                plan["stage"] = stage
                plan["mask_type"] = "ffn_channel_forward_mask"
                plans[stage][sparsity_key(sparsity)] = plan
        return cls(plans=plans)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "StageMaskBank":
        if "plans" not in raw or not isinstance(raw["plans"], dict):
            raise ValueError("mask bank requires a 'plans' object")
        return cls(plans=raw["plans"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "safeprune.stage_mask_bank.v1",
            "note": "Mask-based FFN algorithm validation; not physical tensor slicing.",
            "plans": self.plans,
        }

    def select(self, stage: str, sparsity: float) -> dict[str, Any]:
        stage_plans = self.plans.get(stage)
        if not stage_plans:
            raise KeyError(f"No mask plans for stage {stage!r}")
        key = sparsity_key(sparsity)
        if key not in stage_plans:
            raise KeyError(f"No mask plan for stage={stage!r}, sparsity={key}")
        return stage_plans[key]

    def compose_layerwise_plan(
        self,
        stage: str,
        layer_sparsities: dict[int, float],
    ) -> dict[str, Any]:
        stage_plans = self.plans.get(stage)
        if not stage_plans:
            raise KeyError(f"No mask plans for stage {stage!r}")

        base_key = sorted(stage_plans)[0]
        base = deepcopy(stage_plans[base_key])
        layers = base.get("layers", [])
        layer_indices = {int(layer["layer"]) for layer in layers}

        source_by_sparsity: dict[str, dict[int, dict[str, Any]]] = {}
        normalized: dict[int, float] = {}
        for raw_layer_idx, raw_sparsity in layer_sparsities.items():
            layer_idx = int(raw_layer_idx)
            if layer_idx not in layer_indices:
                raise IndexError(f"Layer {layer_idx} is not present in stage {stage!r}")

            sparsity = float(raw_sparsity)
            if sparsity < 0.0:
                raise ValueError("Layer sparsity must be non-negative.")
            normalized[layer_idx] = sparsity

            if sparsity == 0.0:
                continue
            key = sparsity_key(sparsity)
            if key not in stage_plans:
                raise KeyError(f"No mask plan for stage={stage!r}, sparsity={key}")
            if key not in source_by_sparsity:
                source_by_sparsity[key] = {
                    int(layer["layer"]): layer
                    for layer in stage_plans[key].get("layers", [])
                }

        for layer in layers:
            layer_idx = int(layer["layer"])
            sparsity = normalized.get(layer_idx, 0.0)
            layer["pruned_attention_heads"] = []
            if sparsity == 0.0:
                layer["pruned_mlp_channels"] = []
                continue
            source = source_by_sparsity[sparsity_key(sparsity)][layer_idx]
            layer["pruned_mlp_channels"] = list(source.get("pruned_mlp_channels", []))

        base["stage"] = stage
        base["mask_type"] = "ffn_channel_forward_mask"
        base["allocation"] = {
            str(layer_idx): sparsity
            for layer_idx, sparsity in sorted(normalized.items())
            if sparsity > 0.0
        }
        return base

    def active_mlp_ratio(self, stage: str, sparsity: float) -> float:
        return active_mlp_ratio_from_plan(self.select(stage, sparsity))


def active_mlp_ratio_from_plan(plan: dict[str, Any]) -> float:
    total = 0
    active = 0
    for layer in plan["layers"]:
        channels = int(layer["num_mlp_channels"])
        pruned = len(layer.get("pruned_mlp_channels", []))
        total += channels
        active += channels - pruned
    return active / total if total else 0.0


def save_stage_mask_bank(mask_bank: StageMaskBank, path: str | Path) -> None:
    Path(path).write_text(json.dumps(mask_bank.to_dict(), indent=2), encoding="utf-8")


def load_stage_mask_bank(path: str | Path) -> StageMaskBank:
    return StageMaskBank.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
