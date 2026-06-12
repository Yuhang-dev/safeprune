from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DISPLAY_NAMES = {
    "dense": "Dense",
    "identity_hook": "Identity hook",
    "global_balanced_observe_approx_0.01": "Observe-only",
    "stage_global_balanced_approx_0.01": "Stage-aware",
    "failure_redense_global_balanced_approx_0.01": "Failure-redense",
    "observe_failure_redense_global_balanced_approx_0.01": "Observe-failure-redense",
    "stage_reflect_dense_global_balanced_approx_0.01": "Stage-reflect-dense",
    "stage_reflect_observe_global_balanced_approx_0.01": "Stage-reflect-observe",
}

METHOD_ORDER = [
    "dense",
    "identity_hook",
    "global_balanced_observe_approx_0.01",
    "stage_global_balanced_approx_0.01",
    "failure_redense_global_balanced_approx_0.01",
    "observe_failure_redense_global_balanced_approx_0.01",
    "stage_reflect_dense_global_balanced_approx_0.01",
    "stage_reflect_observe_global_balanced_approx_0.01",
]

PAIRWISE_KEYS = [
    "stage_global_balanced_approx_0.01_vs_stage_reflect_dense_global_balanced_approx_0.01",
    "failure_redense_global_balanced_approx_0.01_vs_stage_reflect_dense_global_balanced_approx_0.01",
    "observe_failure_redense_global_balanced_approx_0.01_vs_stage_reflect_dense_global_balanced_approx_0.01",
    "global_balanced_observe_approx_0.01_vs_stage_reflect_dense_global_balanced_approx_0.01",
    "stage_reflect_dense_global_balanced_approx_0.01_vs_stage_reflect_observe_global_balanced_approx_0.01",
]

FOCAL_EVENTS = [
    "timeout",
    "tool_error",
    "empty_observation",
    "premature_final",
    "schema_error",
    "invalid_argument",
    "format_error",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {line_number}: expected JSON object")
            rows.append(row)
    return rows


def _method_label(method: str) -> str:
    return DISPLAY_NAMES.get(method, method)


def _percent_reduction(old: float, new: float) -> float | None:
    if old == 0:
        return None
    return (old - new) / old


def build_analysis(
    *,
    predictions: Path,
    metrics: Path,
    pairwise: Path,
    failures: Path,
) -> dict[str, Any]:
    metrics_payload = _read_json(metrics)
    pairwise_payload = _read_json(pairwise)
    rows = _read_jsonl(predictions)
    failure_rows = _read_jsonl(failures) if failures.exists() else []

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(row)

    fallback_reduction = None
    if (
        "failure_redense_global_balanced_approx_0.01" in metrics_payload
        and "stage_reflect_dense_global_balanced_approx_0.01" in metrics_payload
    ):
        fallback_reduction = _percent_reduction(
            float(
                metrics_payload["failure_redense_global_balanced_approx_0.01"][
                    "fallback_step_ratio"
                ]
            ),
            float(
                metrics_payload["stage_reflect_dense_global_balanced_approx_0.01"][
                    "fallback_step_ratio"
                ]
            ),
        )

    return {
        "inputs": {
            "predictions": str(predictions),
            "metrics": str(metrics),
            "pairwise": str(pairwise),
            "failures": str(failures),
        },
        "main_table": _main_table(metrics_payload),
        "pairwise": _selected_pairwise(pairwise_payload),
        "failure_type_split": _failure_type_split(by_method),
        "failed_rows_event_split": _failed_rows_event_split(failure_rows),
        "fallback_comparison": {
            "failure_redense_step_ratio": metrics_payload[
                "failure_redense_global_balanced_approx_0.01"
            ]["fallback_step_ratio"],
            "stage_reflect_dense_step_ratio": metrics_payload[
                "stage_reflect_dense_global_balanced_approx_0.01"
            ]["fallback_step_ratio"],
            "relative_reduction": fallback_reduction,
        },
    }


def _main_table(metrics_payload: dict[str, Any]) -> list[dict[str, Any]]:
    table = []
    for method in METHOD_ORDER:
        if method not in metrics_payload:
            continue
        row = metrics_payload[method]
        table.append(
            {
                "method": method,
                "label": _method_label(method),
                "success": f"{row['correct']}/{row['total']}",
                "failure": f"{row['failure_task_correct']}/{row['failure_task_total']}",
                "non_failure": (
                    f"{row['non_failure_task_correct']}/{row['non_failure_task_total']}"
                ),
                "cost_per_success": row["cost_per_success"],
                "fallback_step_ratio": row.get("fallback_step_ratio", row.get("dense_fallback_rate")),
                "schema_validity_rate": row["schema_validity_rate"],
                "premature_final_rate": row.get("premature_final_rate"),
                "reflect_route_accuracy": row.get("reflect_route_accuracy"),
            }
        )
    return table


def _selected_pairwise(pairwise_payload: dict[str, Any]) -> dict[str, Any]:
    selected = {}
    for key in PAIRWISE_KEYS:
        if key not in pairwise_payload:
            continue
        selected[key] = pairwise_payload[key]
    return selected


def _failure_type_split(
    by_method: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    split: dict[str, dict[str, Any]] = {}
    for method in METHOD_ORDER:
        rows = by_method.get(method, [])
        failure_subset = [row for row in rows if row.get("failure_task")]
        by_first_event = Counter()
        event_occurrences = Counter()
        by_tool: dict[str, Counter] = defaultdict(Counter)
        for row in failure_subset:
            events = list(row.get("events") or [])
            if events:
                by_first_event[events[0]] += 1
            for event in events:
                event_occurrences[event] += 1
            tool = str(row.get("tool", "unknown"))
            by_tool[tool]["total"] += 1
            by_tool[tool]["correct"] += int(bool(row.get("success")))
        split[method] = {
            "first_event": {event: by_first_event[event] for event in FOCAL_EVENTS},
            "event_occurrences": {
                event: event_occurrences[event] for event in FOCAL_EVENTS
            },
            "by_tool": {
                tool: {
                    "correct": counts["correct"],
                    "total": counts["total"],
                    "accuracy": counts["correct"] / counts["total"]
                    if counts["total"]
                    else 0.0,
                }
                for tool, counts in sorted(by_tool.items())
            },
        }
    return split


def _failed_rows_event_split(failure_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, Counter] = defaultdict(Counter)
    terminal_by_method: dict[str, Counter] = defaultdict(Counter)
    for row in failure_rows:
        method = str(row.get("method", "unknown"))
        terminal_by_method[method][str(row.get("terminal_event"))] += 1
        for event in row.get("events") or []:
            by_method[method][str(event)] += 1
    return {
        method: {
            "terminal_events": dict(sorted(terminal_by_method[method].items())),
            "events": dict(sorted(counts.items())),
        }
        for method, counts in sorted(by_method.items())
    }


def to_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Real Tool v1 Analysis",
        "",
        "## Main Table",
        "",
        "| Method | Success | Failure | Non-failure | Cost / success | Fallback step ratio | Schema validity |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["main_table"]:
        lines.append(
            f"| {row['label']} | {row['success']} | {row['failure']} | "
            f"{row['non_failure']} | {row['cost_per_success']:.4f} | "
            f"{row['fallback_step_ratio']:.4f} | {row['schema_validity_rate']:.4f} |"
        )

    fallback = payload["fallback_comparison"]
    reduction = fallback["relative_reduction"]
    reduction_text = "n/a" if reduction is None else f"{reduction * 100:.1f}%"
    lines.extend(
        [
            "",
            "## Fallback Comparison",
            "",
            "| Comparison | Value |",
            "|---|---:|",
            f"| Failure-redense fallback step ratio | {fallback['failure_redense_step_ratio']:.4f} |",
            f"| Stage-reflect-dense fallback step ratio | {fallback['stage_reflect_dense_step_ratio']:.4f} |",
            f"| Relative reduction | {reduction_text} |",
            "",
            "## Selected Pairwise",
            "",
            "| Pair | Split | Both correct | Left only | Right only | Both wrong |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for pair_name, splits in payload["pairwise"].items():
        for split_name in ["all", "failure", "non_failure"]:
            split = splits[split_name]
            lines.append(
                f"| {pair_name} | {split_name} | {split['both_correct']} | "
                f"{split['left_only_correct']} | {split['right_only_correct']} | "
                f"{split['both_wrong']} |"
            )

    lines.extend(["", "## Failure Subset By Tool", ""])
    for method in METHOD_ORDER:
        if method not in payload["failure_type_split"]:
            continue
        lines.append(f"### {_method_label(method)}")
        lines.extend(
            [
                "",
                "| Tool | Correct | Total | Accuracy |",
                "|---|---:|---:|---:|",
            ]
        )
        for tool, counts in payload["failure_type_split"][method]["by_tool"].items():
            lines.append(
                f"| {tool} | {counts['correct']} | {counts['total']} | "
                f"{counts['accuracy']:.4f} |"
            )
        lines.append("")

    lines.extend(["## Failed Rows Event Occurrences", ""])
    for method in METHOD_ORDER:
        if method not in payload["failed_rows_event_split"]:
            continue
        event_counts = payload["failed_rows_event_split"][method]["events"]
        lines.append(f"### {_method_label(method)}")
        lines.extend(["", "| Event | Count |", "|---|---:|"])
        for event, count in event_counts.items():
            lines.append(f"| {event} | {count} |")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _default_paths(prefix: Path) -> dict[str, Path]:
    return {
        "predictions": prefix.with_suffix(".jsonl"),
        "metrics": Path(f"{prefix}_metrics.json"),
        "pairwise": Path(f"{prefix}_pairwise.json"),
        "failures": Path(f"{prefix}_failures.jsonl"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Real Tool v1 frozen results.")
    parser.add_argument(
        "--prefix",
        default=(
            "outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/"
            "real_tool_v1_bypass_1000"
        ),
        help="Artifact prefix without suffix.",
    )
    parser.add_argument("--predictions")
    parser.add_argument("--metrics")
    parser.add_argument("--pairwise")
    parser.add_argument("--failures")
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    args = parser.parse_args()

    prefix = Path(args.prefix)
    defaults = _default_paths(prefix)
    predictions = Path(args.predictions) if args.predictions else defaults["predictions"]
    metrics = Path(args.metrics) if args.metrics else defaults["metrics"]
    pairwise = Path(args.pairwise) if args.pairwise else defaults["pairwise"]
    failures = Path(args.failures) if args.failures else defaults["failures"]
    output_json = Path(args.output_json or f"{prefix}_analysis.json")
    output_md = Path(args.output_md or f"{prefix}_analysis.md")

    payload = build_analysis(
        predictions=predictions,
        metrics=metrics,
        pairwise=pairwise,
        failures=failures,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(to_markdown(payload), encoding="utf-8")
    print(f"Wrote analysis JSON: {output_json}")
    print(f"Wrote analysis Markdown: {output_md}")


if __name__ == "__main__":
    main()
