from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


METHOD_LABELS = {
    "dense": "Dense",
    "identity_hook": "Identity hook",
    "observe_failure_redense_global_balanced_approx_0.01": "Observe-failure-redense",
    "stage_reflect_dense_global_balanced_approx_0.01": "Stage-reflect-dense",
    "hidden_state_centroid_global_balanced_approx_0.01": "Hidden centroid",
    "hidden_state_centroid_reflect_dense_global_balanced_approx_0.01": "Hidden centroid reflect-dense",
    "hidden_state_centroid_event_reflect_dense_global_balanced_approx_0.01": "Hidden centroid + event",
}

SELECTED_PAIRS = [
    (
        "stage_reflect_dense_global_balanced_approx_0.01",
        "hidden_state_centroid_global_balanced_approx_0.01",
    ),
    (
        "stage_reflect_dense_global_balanced_approx_0.01",
        "hidden_state_centroid_reflect_dense_global_balanced_approx_0.01",
    ),
    (
        "stage_reflect_dense_global_balanced_approx_0.01",
        "hidden_state_centroid_event_reflect_dense_global_balanced_approx_0.01",
    ),
    (
        "observe_failure_redense_global_balanced_approx_0.01",
        "stage_reflect_dense_global_balanced_approx_0.01",
    ),
]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_analysis(
    *,
    metrics: dict[str, Any],
    pairwise: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "main_table": _main_table(metrics),
        "selected_pairwise": _selected_pairwise(pairwise),
        "detector_confusion": _detector_confusion(rows),
        "probe_cost_ratio": _probe_cost_ratio(metrics),
    }


def _main_table(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    table = []
    for method, row in metrics.items():
        table.append(
            {
                "method": method,
                "label": METHOD_LABELS.get(method, method),
                "success": f"{row.get('correct', 0)}/{row.get('total', 0)}",
                "failure_success": (
                    f"{row.get('failure_task_correct', 0)}/"
                    f"{row.get('failure_task_total', 0)}"
                ),
                "non_failure_success": (
                    f"{row.get('non_failure_task_correct', 0)}/"
                    f"{row.get('non_failure_task_total', 0)}"
                ),
                "raw_reflect_recall": row.get(
                    "raw_reflect_detection_recall",
                    row.get("reflect_detection_recall"),
                ),
                "effective_reflect_redense_recall": row.get(
                    "effective_reflect_redense_recall"
                ),
                "critical_reflect_miss_rate": row.get("critical_reflect_miss_rate"),
                "fallback_step_ratio": row.get("fallback_step_ratio"),
                "routing_probe_cost": row.get("routing_probe_cost"),
                "inclusive_cost_per_success": row.get(
                    "router_inclusive_cost_per_success"
                ),
            }
        )
    return table


def _selected_pairwise(pairwise: dict[str, Any]) -> dict[str, Any]:
    selected = {}
    for left, right in SELECTED_PAIRS:
        key = f"{left}_vs_{right}"
        reverse_key = f"{right}_vs_{left}"
        if key in pairwise:
            selected[key] = pairwise[key]
        elif reverse_key in pairwise:
            selected[reverse_key] = pairwise[reverse_key]
    return selected


def _detector_confusion(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    by_method: dict[str, Counter] = {}
    for row in rows:
        method = row.get("method")
        if not method:
            continue
        for trace in row.get("routing_trace", []):
            metadata = trace.get("route_metadata") or {}
            if "centroid_predicted_stage" not in metadata:
                continue
            counter = by_method.setdefault(method, Counter())
            actual_reflect = trace.get("stage") == "reflect"
            predicted_reflect = bool(metadata.get("centroid_reflect_predicted"))
            if actual_reflect and predicted_reflect:
                counter["tp"] += 1
            elif actual_reflect:
                counter["fn"] += 1
            elif predicted_reflect:
                counter["fp"] += 1
            else:
                counter["tn"] += 1
    return {
        method: {
            "tp": counts["tp"],
            "fn": counts["fn"],
            "fp": counts["fp"],
            "tn": counts["tn"],
        }
        for method, counts in sorted(by_method.items())
    }


def _probe_cost_ratio(metrics: dict[str, Any]) -> dict[str, float | None]:
    base = metrics.get("stage_reflect_dense_global_balanced_approx_0.01", {})
    base_cost = base.get("router_inclusive_cost_per_success")
    ratios = {}
    for method, row in metrics.items():
        cost = row.get("router_inclusive_cost_per_success")
        ratios[method] = (
            float(cost) / float(base_cost)
            if base_cost not in (None, 0) and cost is not None
            else None
        )
    return ratios


def _fmt_cell(value: Any) -> str:
    if value is None:
        return "N/A"
    return str(value)


def to_markdown(analysis: dict[str, Any]) -> str:
    lines = ["# Hidden-State P1 Analysis", ""]
    lines.extend(
        [
            "| Method | Success | Failure | Non-failure | Raw reflect recall | Effective redense recall | Critical miss | Fallback step | Probe cost | Inclusive cost/success |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in analysis["main_table"]:
        lines.append(
            "| {label} | {success} | {failure_success} | {non_failure_success} | "
            "{raw_reflect_recall} | {effective_reflect_redense_recall} | "
            "{critical_reflect_miss_rate} | {fallback_step_ratio} | "
            "{routing_probe_cost} | {inclusive_cost_per_success} |".format(
                **{key: _fmt_cell(value) for key, value in row.items()}
            )
        )
    lines.extend(["", "## Detector Confusion", ""])
    lines.extend(["| Method | TP | FN | FP | TN |", "|---|---:|---:|---:|---:|"])
    for method, counts in analysis["detector_confusion"].items():
        lines.append(
            f"| {METHOD_LABELS.get(method, method)} | {counts['tp']} | {counts['fn']} | "
            f"{counts['fp']} | {counts['tn']} |"
        )
    lines.extend(["", "## Selected Pairwise", ""])
    for pair, splits in analysis["selected_pairwise"].items():
        lines.append(f"### {pair}")
        lines.append("")
        lines.append("| Split | Both correct | Left only | Right only | Both wrong |")
        lines.append("|---|---:|---:|---:|---:|")
        for split, counts in splits.items():
            lines.append(
                f"| {split} | {counts.get('both_correct', 0)} | "
                f"{counts.get('left_only_correct', 0)} | "
                f"{counts.get('right_only_correct', 0)} | "
                f"{counts.get('both_wrong', 0)} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze hidden-state P1 results.")
    parser.add_argument(
        "--prefix",
        default=(
            "outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/"
            "real_tool_v1_p1_hidden_1000"
        ),
    )
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prefix = Path(args.prefix)
    metrics = _read_json(Path(f"{prefix}_metrics.json"))
    pairwise = _read_json(Path(f"{prefix}_pairwise.json"))
    rows = _read_jsonl(Path(f"{prefix}.jsonl"))
    analysis = build_analysis(metrics=metrics, pairwise=pairwise, rows=rows)
    output_json = Path(args.output_json or f"{prefix}_analysis.json")
    output_md = Path(args.output_md or f"{prefix}_analysis.md")
    output_json.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(to_markdown(analysis), encoding="utf-8")
    print(f"Wrote analysis JSON: {output_json}")
    print(f"Wrote analysis Markdown: {output_md}")


if __name__ == "__main__":
    main()
