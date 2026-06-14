from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from safeprune.config import load_config
from safeprune.modeling import load_causal_lm_and_tokenizer
from safeprune.pruning import ScoreWeights, combine_importance_scores
from safeprune.scoring import load_activation_stats, load_scores, save_plan
from safeprune.substrate import add_bias_compensation_to_plan, add_layer_output_scales


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build hardware-aligned FFN compact plans for latency experiments."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--scores-path", required=True)
    parser.add_argument("--activation-stats-path")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-budgets", default="0.10,0.20")
    parser.add_argument("--plan-prefix", default="aligned")
    parser.add_argument("--align-to", type=int, default=256)
    parser.add_argument("--min-remaining", type=int, default=4096)
    parser.add_argument("--weights", default="magnitude=1.0,activation=0.0,loss_delta=0.0")
    parser.add_argument("--with-bias-compensation", action="store_true")
    parser.add_argument("--with-layer-scale-placeholder", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.align_to <= 0:
        raise ValueError("--align-to must be positive.")
    if args.min_remaining < 0:
        raise ValueError("--min-remaining must be non-negative.")
    if args.with_bias_compensation and not args.activation_stats_path:
        raise ValueError("--with-bias-compensation requires --activation-stats-path.")

    config = load_config(args.config)
    scores = load_scores(args.scores_path)
    targets = _parse_float_list(args.target_budgets)
    weights = _parse_weights(args.weights)

    plans = build_hardware_aligned_plans(
        scores=scores,
        targets=targets,
        weights=weights,
        align_to=args.align_to,
        min_remaining=args.min_remaining,
        plan_prefix=args.plan_prefix,
    )

    model = None
    activation_stats = None
    if args.with_bias_compensation:
        activation_stats = load_activation_stats(args.activation_stats_path)
        model, _tokenizer = load_causal_lm_and_tokenizer(
            config.model.base_model,
            config.model,
            local_files_only=args.local_files_only,
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "safeprune.hardware_aligned_compact_plans.v1",
        "config": args.config,
        "scores_path": args.scores_path,
        "activation_stats_path": args.activation_stats_path,
        "target_budgets": targets,
        "plan_prefix": args.plan_prefix,
        "align_to": args.align_to,
        "min_remaining": args.min_remaining,
        "weights": weights.__dict__,
        "with_bias_compensation": args.with_bias_compensation,
        "with_layer_scale_placeholder": args.with_layer_scale_placeholder,
        "plans": {},
        "audit": audit_aligned_plans(plans, args.align_to, args.min_remaining),
        "note": (
            "Latency-oriented hardware-aligned plans. These are separate from "
            "the frozen P4 Real Tool active-cost plans."
        ),
    }

    for target, plan in sorted(plans.items()):
        output_plan = copy.deepcopy(plan)
        if args.with_bias_compensation and model is not None and activation_stats is not None:
            output_plan = add_bias_compensation_to_plan(model, activation_stats, output_plan)
        if args.with_layer_scale_placeholder:
            output_plan = add_layer_output_scales(
                output_plan,
                {int(layer["layer"]): 1.0 for layer in output_plan.get("layers", [])},
            )
        slug = _budget_slug(target)
        path = out_dir / f"budget_plan_{args.plan_prefix}_{slug}.json"
        save_plan(output_plan, path)
        manifest["plans"][slug] = {
            "target": target,
            "path": str(path),
            "name": output_plan.get("plan_name"),
            "actual_sparsity": output_plan["budget_plan"]["actual_sparsity"],
            "active_mlp_ratio": 1.0 - output_plan["budget_plan"]["actual_sparsity"],
        }

    (out_dir / "hardware_aligned_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    (out_dir / "hardware_aligned_audit.md").write_text(
        _audit_to_markdown(manifest),
        encoding="utf-8",
    )
    print(json.dumps(manifest["plans"], indent=2))
    print(f"Wrote hardware-aligned plans to {out_dir}", flush=True)


def build_hardware_aligned_plans(
    *,
    scores,
    targets: list[float],
    weights: ScoreWeights,
    align_to: int,
    min_remaining: int,
    plan_prefix: str,
) -> dict[float, dict[str, Any]]:
    if not targets:
        raise ValueError("At least one target budget is required.")
    targets = sorted(float(target) for target in targets)
    if any(target <= 0.0 or target >= 1.0 for target in targets):
        raise ValueError("Target budgets must be in (0, 1).")

    layer_meta: dict[int, dict[str, Any]] = {}
    blocks: list[dict[str, Any]] = []
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
        total_channels += channel_count
        min_remaining_layer = min(channel_count, max(min_remaining, align_to))
        if channel_count % align_to != 0:
            # Keep the original layer valid by requiring remaining widths to
            # share the same remainder as the dense layer.
            max_prunable = max(0, channel_count - min_remaining_layer)
            max_prunable -= max_prunable % align_to
        else:
            max_prunable = max(0, channel_count - min_remaining_layer)
            max_prunable -= max_prunable % align_to
        ranked = sorted(range(channel_count), key=lambda idx: (importance[idx], idx))
        layer_meta[layer] = {
            "channel_count": channel_count,
            "max_prunable": max_prunable,
            "ranked": ranked,
        }
        for start in range(0, max_prunable, align_to):
            block_channels = ranked[start : start + align_to]
            if len(block_channels) != align_to:
                continue
            blocks.append(
                {
                    "score": sum(float(importance[idx]) for idx in block_channels) / align_to,
                    "layer": layer,
                    "block_index": start // align_to,
                    "channels": block_channels,
                }
            )

    blocks.sort(key=lambda item: (item["score"], item["layer"], item["block_index"]))
    selected_by_layer: dict[int, set[int]] = {layer: set() for layer in layer_meta}
    cursor = 0
    plans: dict[float, dict[str, Any]] = {}
    for target in targets:
        target_pruned = math.ceil(total_channels * target)
        while _total_pruned(selected_by_layer) < target_pruned and cursor < len(blocks):
            block = blocks[cursor]
            cursor += 1
            layer = int(block["layer"])
            if len(selected_by_layer[layer]) + align_to > layer_meta[layer]["max_prunable"]:
                continue
            selected_by_layer[layer].update(int(channel) for channel in block["channels"])
        if _total_pruned(selected_by_layer) < target_pruned:
            raise ValueError(f"Cannot satisfy target {target} with aligned constraints.")
        plans[target] = _plan_from_pruned_sets(
            scores=scores,
            pruned_by_layer=selected_by_layer,
            target=target,
            total_channels=total_channels,
            plan_name=f"{plan_prefix}_{_budget_slug(target)}",
            align_to=align_to,
            min_remaining=min_remaining,
        )
    return plans


def audit_aligned_plans(
    plans: dict[float, dict[str, Any]],
    align_to: int,
    min_remaining: int,
) -> dict[str, Any]:
    ordered = sorted((float(target), plan) for target, plan in plans.items())
    plan_audits = []
    previous_set: set[tuple[int, int]] | None = None
    pairs = []
    for target, plan in ordered:
        widths = []
        non_aligned = []
        below_min = []
        duplicate_layers = []
        out_of_range_layers = []
        for layer in plan.get("layers", []):
            layer_idx = int(layer["layer"])
            channel_count = int(layer["num_mlp_channels"])
            pruned = [int(item) for item in layer.get("pruned_mlp_channels", [])]
            remaining = channel_count - len(set(pruned))
            widths.append(remaining)
            if remaining % align_to != channel_count % align_to:
                non_aligned.append(layer_idx)
            if remaining < min_remaining:
                below_min.append(layer_idx)
            if len(pruned) != len(set(pruned)):
                duplicate_layers.append(layer_idx)
            if any(item < 0 or item >= channel_count for item in pruned):
                out_of_range_layers.append(layer_idx)
        current_set = _global_pruned_set(plan)
        if previous_set is not None:
            missing = previous_set - current_set
            added = current_set - previous_set
            pairs.append(
                {
                    "left_subset_of_right": not missing,
                    "left_only_count": len(missing),
                    "newly_pruned_count": len(added),
                    "nesting_overlap": (
                        len(previous_set & current_set) / len(previous_set)
                        if previous_set
                        else 1.0
                    ),
                }
            )
        previous_set = current_set
        plan_audits.append(
            {
                "target": target,
                "plan_name": plan.get("plan_name"),
                "actual_sparsity": plan["budget_plan"]["actual_sparsity"],
                "active_mlp_ratio": 1.0 - plan["budget_plan"]["actual_sparsity"],
                "min_remaining": min(widths) if widths else None,
                "max_remaining": max(widths) if widths else None,
                "non_aligned_layers": non_aligned,
                "below_min_layers": below_min,
                "duplicate_layers": duplicate_layers,
                "out_of_range_layers": out_of_range_layers,
            }
        )
    return {"plans": plan_audits, "nested_pairs": pairs}


def _plan_from_pruned_sets(
    *,
    scores,
    pruned_by_layer: dict[int, set[int]],
    target: float,
    total_channels: int,
    plan_name: str,
    align_to: int,
    min_remaining: int,
) -> dict[str, Any]:
    layers = []
    allocation = {}
    total_pruned = 0
    for layer_scores in scores:
        layer = int(layer_scores.layer)
        channel_count = len(layer_scores.mlp)
        pruned = sorted(pruned_by_layer.get(layer, set()))
        total_pruned += len(pruned)
        if pruned:
            allocation[str(layer)] = len(pruned) / channel_count
        layers.append(
            {
                "layer": layer,
                "pruned_attention_heads": [],
                "pruned_mlp_channels": pruned,
                "num_attention_heads": len(layer_scores.attention),
                "num_mlp_channels": channel_count,
                "hardware_alignment": {
                    "align_to": align_to,
                    "min_remaining": min_remaining,
                    "remaining_channels": channel_count - len(pruned),
                },
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
            "hardware_aligned": True,
        },
    }


def _parse_float_list(raw: str) -> list[float]:
    return [float(item) for item in raw.split(",") if item.strip()]


def _parse_weights(raw: str) -> ScoreWeights:
    mapping: dict[str, float] = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        key, value = item.split("=", 1)
        mapping[key.strip()] = float(value)
    return ScoreWeights.from_mapping(mapping)


def _total_pruned(pruned_by_layer: dict[int, set[int]]) -> int:
    return sum(len(channels) for channels in pruned_by_layer.values())


def _global_pruned_set(plan: dict[str, Any]) -> set[tuple[int, int]]:
    result = set()
    for layer in plan.get("layers", []):
        layer_idx = int(layer["layer"])
        for channel in layer.get("pruned_mlp_channels", []):
            result.add((layer_idx, int(channel)))
    return result


def _budget_slug(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _audit_to_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Hardware-Aligned Compact Plan Audit",
        "",
        f"- align_to: {manifest['align_to']}",
        f"- min_remaining: {manifest['min_remaining']}",
        "",
        "| Plan | Target | Active MLP | Min remaining | Max remaining | Non-aligned | Below min | Duplicate | Out-of-range |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in manifest["audit"]["plans"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["plan_name"]),
                    f"{item['target']:.2f}",
                    f"{item['active_mlp_ratio']:.4f}",
                    str(item["min_remaining"]),
                    str(item["max_remaining"]),
                    str(len(item["non_aligned_layers"])),
                    str(len(item["below_min_layers"])),
                    str(len(item["duplicate_layers"])),
                    str(len(item["out_of_range_layers"])),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Nested Pairs")
    lines.append("")
    lines.append("| Pair | Subset | Left-only | Newly pruned | Overlap |")
    lines.append("|---:|---|---:|---:|---:|")
    for idx, pair in enumerate(manifest["audit"]["nested_pairs"], start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(pair["left_subset_of_right"]),
                    str(pair["left_only_count"]),
                    str(pair["newly_pruned_count"]),
                    f"{pair['nesting_overlap']:.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
