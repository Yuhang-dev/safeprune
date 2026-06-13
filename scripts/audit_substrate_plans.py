from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _layer_id(layer: dict[str, Any]) -> int:
    if "layer" in layer:
        return int(layer["layer"])
    if "layer_idx" in layer:
        return int(layer["layer_idx"])
    raise KeyError("Layer payload must contain 'layer' or 'layer_idx'.")


def _pruned_channels(layer: dict[str, Any]) -> list[int]:
    return [int(idx) for idx in layer.get("pruned_mlp_channels", [])]


def _bias_norm(layer: dict[str, Any]) -> float | None:
    values = layer.get("mlp_output_bias_compensation")
    if values is None:
        return None
    if not values:
        return 0.0
    return math.sqrt(sum(float(value) ** 2 for value in values))


def _valid_unique_pruned(layer: dict[str, Any]) -> set[int]:
    num_channels = int(layer.get("num_mlp_channels", 0))
    return {
        idx
        for idx in _pruned_channels(layer)
        if 0 <= idx < num_channels
    }


def _summarize_plan(name: str, plan: dict[str, Any]) -> dict[str, Any]:
    layers = []
    total_channels = 0
    total_pruned_raw = 0
    total_pruned_unique = 0
    duplicate_count = 0
    out_of_range_count = 0
    bias_layers = 0
    scale_layers = 0

    for layer in plan.get("layers", []):
        layer_idx = _layer_id(layer)
        num_channels = int(layer.get("num_mlp_channels", 0))
        pruned = _pruned_channels(layer)
        valid_unique = _valid_unique_pruned(layer)
        duplicate = len(pruned) - len(set(pruned))
        out_of_range = sum(1 for idx in pruned if idx < 0 or idx >= num_channels)
        bias_norm = _bias_norm(layer)
        has_scale = "mlp_output_scale" in layer

        total_channels += num_channels
        total_pruned_raw += len(pruned)
        total_pruned_unique += len(valid_unique)
        duplicate_count += duplicate
        out_of_range_count += out_of_range
        bias_layers += int(bias_norm is not None)
        scale_layers += int(has_scale)

        layers.append(
            {
                "layer": layer_idx,
                "num_mlp_channels": num_channels,
                "pruned_count": len(pruned),
                "unique_valid_pruned_count": len(valid_unique),
                "sparsity": len(valid_unique) / num_channels if num_channels else 0.0,
                "duplicate_count": duplicate,
                "out_of_range_count": out_of_range,
                "bias_compensation_norm": bias_norm,
                "scale": layer.get("mlp_output_scale"),
            }
        )

    actual_sparsity = total_pruned_unique / total_channels if total_channels else 0.0
    top_layers = sorted(layers, key=lambda item: (-item["sparsity"], item["layer"]))[:8]
    budget_plan = plan.get("budget_plan", {})
    return {
        "name": name,
        "plan_name": plan.get("plan_name"),
        "substrate_method": plan.get("substrate_method"),
        "global_target": plan.get("global_target", budget_plan.get("global_target")),
        "declared_actual_sparsity": budget_plan.get("actual_sparsity"),
        "total_layers": len(layers),
        "total_channels": total_channels,
        "total_pruned_raw": total_pruned_raw,
        "total_pruned_unique": total_pruned_unique,
        "actual_sparsity": actual_sparsity,
        "active_mlp_ratio": 1.0 - actual_sparsity,
        "duplicate_count": duplicate_count,
        "out_of_range_count": out_of_range_count,
        "bias_compensation_layers": bias_layers,
        "scale_layers": scale_layers,
        "top_sparsity_layers": [
            {
                "layer": item["layer"],
                "sparsity": item["sparsity"],
                "pruned_count": item["unique_valid_pruned_count"],
            }
            for item in top_layers
        ],
        "layers": layers,
    }


def _global_pruned_set(plan: dict[str, Any]) -> set[tuple[int, int]]:
    result = set()
    for layer in plan.get("layers", []):
        layer_idx = _layer_id(layer)
        for channel in _valid_unique_pruned(layer):
            result.add((layer_idx, channel))
    return result


def _pairwise_overlap(
    names: list[str],
    plans: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    pruned_sets = {name: _global_pruned_set(plans[name]) for name in names}
    overlap = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            left_set = pruned_sets[left]
            right_set = pruned_sets[right]
            intersection = left_set & right_set
            union = left_set | right_set
            left_only = left_set - right_set
            right_only = right_set - left_set
            overlap[f"{left}_vs_{right}"] = {
                "left": left,
                "right": right,
                "left_pruned_count": len(left_set),
                "right_pruned_count": len(right_set),
                "intersection_count": len(intersection),
                "left_only_count": len(left_only),
                "right_only_count": len(right_only),
                "jaccard": len(intersection) / len(union) if union else 1.0,
                "left_subset_of_right": not left_only,
                "right_subset_of_left": not right_only,
            }
    return overlap


def _layer_matrix(
    names: list[str],
    summaries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    all_layers = sorted(
        {
            int(layer["layer"])
            for summary in summaries.values()
            for layer in summary["layers"]
        }
    )
    by_name_layer = {
        name: {int(layer["layer"]): layer for layer in summary["layers"]}
        for name, summary in summaries.items()
    }
    rows = []
    for layer_idx in all_layers:
        row: dict[str, Any] = {"layer": layer_idx}
        for name in names:
            layer = by_name_layer.get(name, {}).get(layer_idx)
            if layer is None:
                row[name] = None
            else:
                row[name] = {
                    "sparsity": layer["sparsity"],
                    "pruned_count": layer["unique_valid_pruned_count"],
                    "bias_compensation_norm": layer["bias_compensation_norm"],
                }
        rows.append(row)
    return rows


def _warnings(
    names: list[str],
    summaries: dict[str, dict[str, Any]],
    pairwise: dict[str, dict[str, Any]],
) -> list[str]:
    warnings = []
    for name, summary in summaries.items():
        if summary["duplicate_count"]:
            warnings.append(f"{name}: duplicate pruned channel indices = {summary['duplicate_count']}")
        if summary["out_of_range_count"]:
            warnings.append(f"{name}: out-of-range pruned channel indices = {summary['out_of_range_count']}")
        declared = summary.get("declared_actual_sparsity")
        if declared is not None and abs(float(declared) - summary["actual_sparsity"]) > 1e-6:
            warnings.append(
                f"{name}: declared actual_sparsity {declared} differs from "
                f"computed {summary['actual_sparsity']:.8f}"
            )

    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            left_summary = summaries[left]
            right_summary = summaries[right]
            if left_summary["actual_sparsity"] <= right_summary["actual_sparsity"]:
                pair = pairwise[f"{left}_vs_{right}"]
                if not pair["left_subset_of_right"]:
                    warnings.append(
                        f"{left} is not nested in {right}: "
                        f"{pair['left_only_count']} channels appear only in the lower-budget plan"
                    )
    return warnings


def audit_plans(plan_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    names = list(plan_payloads)
    summaries = {
        name: _summarize_plan(name, plan)
        for name, plan in plan_payloads.items()
    }
    pairwise = _pairwise_overlap(names, plan_payloads)
    return {
        "plans": summaries,
        "layer_matrix": _layer_matrix(names, summaries),
        "pairwise_overlap": pairwise,
        "warnings": _warnings(names, summaries, pairwise),
    }


def _fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def to_markdown(audit: dict[str, Any]) -> str:
    lines = ["# Substrate Plan Audit", ""]
    lines.extend(
        [
            "## Plan Summary",
            "",
            "| Plan | Active MLP ratio | Actual sparsity | Pruned | Duplicate idx | Out-of-range idx | Bias layers | Scale layers | Top sparse layers |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for name, summary in audit["plans"].items():
        top = ", ".join(
            f"L{item['layer']}:{item['sparsity']:.2f}"
            for item in summary["top_sparsity_layers"]
        )
        lines.append(
            f"| {name} | {_fmt_float(summary['active_mlp_ratio'])} | "
            f"{_fmt_float(summary['actual_sparsity'])} | "
            f"{summary['total_pruned_unique']}/{summary['total_channels']} | "
            f"{summary['duplicate_count']} | {summary['out_of_range_count']} | "
            f"{summary['bias_compensation_layers']} | {summary['scale_layers']} | {top} |"
        )

    lines.extend(
        [
            "",
            "## Pairwise Overlap",
            "",
            "| Pair | Left pruned | Right pruned | Intersection | Left only | Right only | Jaccard | Left subset of right |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for pair_name, pair in audit["pairwise_overlap"].items():
        lines.append(
            f"| {pair_name} | {pair['left_pruned_count']} | {pair['right_pruned_count']} | "
            f"{pair['intersection_count']} | {pair['left_only_count']} | "
            f"{pair['right_only_count']} | {_fmt_float(pair['jaccard'])} | "
            f"{pair['left_subset_of_right']} |"
        )

    plan_names = list(audit["plans"])
    lines.extend(["", "## Per-Layer Allocation", ""])
    header = "| Layer | " + " | ".join(f"{name} sparsity" for name in plan_names) + " |"
    sep = "|---:" + "|---:" * len(plan_names) + "|"
    lines.extend([header, sep])
    for row in audit["layer_matrix"]:
        cells = [str(row["layer"])]
        for name in plan_names:
            payload = row.get(name)
            cells.append("" if payload is None else _fmt_float(payload["sparsity"]))
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(["", "## Bias Compensation Norm", ""])
    header = "| Layer | " + " | ".join(f"{name} bias norm" for name in plan_names) + " |"
    sep = "|---:" + "|---:" * len(plan_names) + "|"
    lines.extend([header, sep])
    for row in audit["layer_matrix"]:
        cells = [str(row["layer"])]
        for name in plan_names:
            payload = row.get(name)
            cells.append("" if payload is None else _fmt_float(payload["bias_compensation_norm"]))
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(["", "## Warnings", ""])
    if audit["warnings"]:
        lines.extend(f"- {warning}" for warning in audit["warnings"])
    else:
        lines.append("- none")
    return "\n".join(lines).rstrip() + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit substrate budget plans.")
    parser.add_argument("--plan", action="append", required=True, help="Budget plan JSON path.")
    parser.add_argument("--name", action="append", help="Plan name; must match --plan count if provided.")
    parser.add_argument("--output-json", help="Write audit JSON to this path.")
    parser.add_argument("--output-md", help="Write audit Markdown to this path.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plan_paths = [Path(path) for path in args.plan]
    if args.name is not None and len(args.name) != len(plan_paths):
        raise ValueError("--name count must match --plan count.")
    names = args.name or [path.stem.replace("budget_plan_", "plan_") for path in plan_paths]
    payloads = {
        name: _read_json(path)
        for name, path in zip(names, plan_paths, strict=True)
    }
    audit = audit_plans(payloads)
    markdown = to_markdown(audit)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if args.output_md:
        Path(args.output_md).write_text(markdown, encoding="utf-8")
    if not args.output_json and not args.output_md:
        print(markdown, end="")


if __name__ == "__main__":
    main()
