from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_PAIRS = [
    ("identity_hook", "dense"),
    ("global_balanced_observe_approx_0.01", "dense"),
    ("stage_global_balanced_approx_0.01", "dense"),
    ("failure_redense_global_balanced_approx_0.01", "dense"),
    ("stage_global_balanced_approx_0.01", "global_balanced_observe_approx_0.01"),
    (
        "failure_redense_global_balanced_approx_0.01",
        "stage_global_balanced_approx_0.01",
    ),
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: expected JSON object")
            rows.append(row)
    return rows


def _group_by_method(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["method"])][str(row["task_id"])] = row
    return dict(grouped)


def _is_success(row: dict[str, Any]) -> bool:
    if "success" in row:
        return bool(row["success"])
    if "strict_correct" in row:
        return bool(row["strict_correct"])
    raise KeyError("row must contain success or strict_correct")


def _pairwise_split(
    grouped: dict[str, dict[str, dict[str, Any]]],
    pairs: list[tuple[str, str]],
    max_examples: int,
) -> list[dict[str, Any]]:
    summaries = []
    for left, right in pairs:
        if left not in grouped or right not in grouped:
            continue
        counts = Counter()
        examples: dict[str, list[dict[str, Any]]] = {"left_only": [], "right_only": []}
        common_ids = sorted(set(grouped[left]) & set(grouped[right]))
        for task_id in common_ids:
            left_row = grouped[left][task_id]
            right_row = grouped[right][task_id]
            left_ok = _is_success(left_row)
            right_ok = _is_success(right_row)
            if left_ok and right_ok:
                counts["both_correct"] += 1
            elif left_ok and not right_ok:
                counts["left_only"] += 1
                if len(examples["left_only"]) < max_examples:
                    examples["left_only"].append(_example(task_id, left_row, right_row))
            elif right_ok and not left_ok:
                counts["right_only"] += 1
                if len(examples["right_only"]) < max_examples:
                    examples["right_only"].append(_example(task_id, left_row, right_row))
            else:
                counts["both_wrong"] += 1

        summaries.append(
            {
                "left": left,
                "right": right,
                "both_correct": counts["both_correct"],
                "left_only": counts["left_only"],
                "right_only": counts["right_only"],
                "both_wrong": counts["both_wrong"],
                "net_left_minus_right": counts["left_only"] - counts["right_only"],
                "examples": examples,
            }
        )
    return summaries


def _example(
    task_id: str,
    left_row: dict[str, Any],
    right_row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "tool": left_row.get("tool"),
        "expected": left_row.get("expected"),
        "left_prediction": _short_text(left_row.get("prediction") or left_row.get("final_answer")),
        "right_prediction": _short_text(right_row.get("prediction") or right_row.get("final_answer")),
        "failure_task": bool(left_row.get("failure_task")),
        "failure_events": left_row.get("failure_events") or left_row.get("events"),
    }


def _short_text(value: Any, max_len: int = 120) -> str:
    text = "" if value is None else str(value).replace("\n", " ")
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _by_tool(grouped: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for method, rows in sorted(grouped.items()):
        by_tool: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
        for row in rows.values():
            tool = str(row.get("tool", "unknown"))
            by_tool[tool]["total"] += 1
            by_tool[tool]["correct"] += int(_is_success(row))
        summary[method] = {
            tool: {
                "correct": counts["correct"],
                "total": counts["total"],
                "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0.0,
            }
            for tool, counts in sorted(by_tool.items())
        }
    return summary


def _failure_by_tool(grouped: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for method, rows in sorted(grouped.items()):
        groups: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {"total": 0, "correct": 0}
        )
        for row in rows.values():
            subset = "failure" if row.get("failure_task") else "non_failure"
            tool = str(row.get("tool", "unknown"))
            key = (subset, tool)
            groups[key]["total"] += 1
            groups[key]["correct"] += int(_is_success(row))
        summary[method] = {
            f"{subset}:{tool}": {
                "correct": counts["correct"],
                "total": counts["total"],
                "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0.0,
            }
            for (subset, tool), counts in sorted(groups.items())
        }
    return summary


def _to_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Agent Pairwise Error Analysis",
        "",
        f"Input: `{payload['input']}`",
        "",
        "## Pairwise Split",
        "",
        "| A | B | Both correct | A only | B only | Both wrong | Net A - B |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["pairwise"]:
        lines.append(
            "| {left} | {right} | {both_correct} | {left_only} | {right_only} | "
            "{both_wrong} | {net_left_minus_right} |".format(**row)
        )

    lines.extend(["", "## By Tool", ""])
    for method, tools in payload["by_tool"].items():
        lines.append(f"### {method}")
        lines.append("")
        lines.append("| Tool | Correct | Total | Accuracy |")
        lines.append("|---|---:|---:|---:|")
        for tool, counts in tools.items():
            lines.append(
                f"| {tool} | {counts['correct']} | {counts['total']} | "
                f"{counts['accuracy']:.4f} |"
            )
        lines.append("")

    lines.extend(["## Failure Split By Tool", ""])
    for method, groups in payload["failure_by_tool"].items():
        lines.append(f"### {method}")
        lines.append("")
        lines.append("| Subset:Tool | Correct | Total | Accuracy |")
        lines.append("|---|---:|---:|---:|")
        for key, counts in groups.items():
            lines.append(
                f"| {key} | {counts['correct']} | {counts['total']} | "
                f"{counts['accuracy']:.4f} |"
            )
        lines.append("")

    lines.extend(["## Examples", ""])
    for row in payload["pairwise"]:
        lines.append(f"### {row['left']} vs {row['right']}")
        lines.append("")
        for label in ["left_only", "right_only"]:
            lines.append(f"{label}:")
            for example in row["examples"][label]:
                lines.append(
                    "- `{task_id}` tool={tool} expected={expected} "
                    "failure={failure_task}".format(**example)
                )
            if not row["examples"][label]:
                lines.append("- none")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _parse_pairs(raw_pairs: list[str] | None) -> list[tuple[str, str]]:
    if not raw_pairs:
        return DEFAULT_PAIRS
    pairs = []
    for raw in raw_pairs:
        if ":" not in raw:
            raise ValueError(f"Pair must use LEFT:RIGHT format, got {raw!r}")
        left, right = raw.split(":", 1)
        pairs.append((left, right))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze pairwise Agent eval errors.")
    parser.add_argument("--input", required=True, help="Predictions JSONL path.")
    parser.add_argument("--output-json", help="Output JSON summary path.")
    parser.add_argument("--output-md", help="Output Markdown summary path.")
    parser.add_argument(
        "--pair",
        action="append",
        help="Pair in LEFT:RIGHT format. May be repeated. Defaults to controlled v1 pairs.",
    )
    parser.add_argument("--max-examples", type=int, default=5)
    args = parser.parse_args()

    input_path = Path(args.input)
    rows = _read_jsonl(input_path)
    grouped = _group_by_method(rows)
    payload = {
        "input": str(input_path),
        "pairwise": _pairwise_split(grouped, _parse_pairs(args.pair), args.max_examples),
        "by_tool": _by_tool(grouped),
        "failure_by_tool": _failure_by_tool(grouped),
    }

    output_json = Path(args.output_json or f"{input_path.with_suffix('')}_pairwise.json")
    output_md = Path(args.output_md or f"{input_path.with_suffix('')}_pairwise.md")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(_to_markdown(payload), encoding="utf-8")
    print(f"Wrote JSON summary: {output_json}")
    print(f"Wrote Markdown summary: {output_md}")


if __name__ == "__main__":
    main()
