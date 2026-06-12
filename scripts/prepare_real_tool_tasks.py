from __future__ import annotations

import argparse
import json
from pathlib import Path


FAILURE_EVENTS = ["timeout", "empty_observation", "tool_error"]


def build_tasks(count: int, failure_count: int) -> list[dict]:
    tasks = []
    failure_ids = set(_failure_indices(count, failure_count))
    for idx in range(count):
        tool_kind = idx % 3
        has_failure = idx in failure_ids
        if tool_kind == 0:
            task = _calculator_task(idx)
        elif tool_kind == 1:
            task = _unit_convert_task(idx)
        else:
            task = _lookup_task(idx)

        if has_failure:
            task["fault_schedule"] = [
                {
                    "tool": task["expected_tool"],
                    "attempt": 1,
                    "event": FAILURE_EVENTS[idx % len(FAILURE_EVENTS)],
                    "retryable": True,
                }
            ]
        else:
            task["fault_schedule"] = []
        task["metadata"]["has_injected_failure"] = has_failure
        tasks.append(task)
    return tasks


def _failure_indices(count: int, failure_count: int) -> list[int]:
    if failure_count <= 0:
        return []
    stride = max(count / failure_count, 1)
    return sorted({min(int(round(i * stride)), count - 1) for i in range(failure_count)})


def _calculator_task(idx: int) -> dict:
    a = idx + 17
    b = idx * 2 + 23
    result = a + b
    expression = f"{a} + {b}"
    return {
        "task_id": f"real_tool_{idx:04d}",
        "user_request": (
            f"Use the calculator tool to compute {expression}. "
            "Then answer with only the final number."
        ),
        "tool": "calculator",
        "expected_answer": str(result),
        "expected_tool": "calculator",
        "expected_arguments": {"expression": expression},
        "max_steps": 6,
        "metadata": {"category": "stateless_single_tool"},
    }


def _unit_convert_task(idx: int) -> dict:
    meters = idx + 3
    result = meters * 100
    return {
        "task_id": f"real_tool_{idx:04d}",
        "user_request": (
            f"Use the unit_convert tool to convert {meters} meters to centimeters. "
            "Then answer with only the numeric result."
        ),
        "tool": "unit_convert",
        "expected_answer": str(result),
        "expected_tool": "unit_convert",
        "expected_arguments": {"value": meters, "from_unit": "m", "to_unit": "cm"},
        "max_steps": 6,
        "metadata": {"category": "stateless_single_tool"},
    }


def _lookup_task(idx: int) -> dict:
    project_id = idx + 5
    return {
        "task_id": f"real_tool_{idx:04d}",
        "user_request": (
            f"Use the lookup tool to find the owner of project PRJ-{project_id}. "
            "Then answer with only the owner id."
        ),
        "tool": "lookup",
        "expected_answer": f"owner_{project_id}",
        "expected_tool": "lookup",
        "expected_arguments": {"project": f"PRJ-{project_id}"},
        "max_steps": 6,
        "metadata": {"category": "stateless_single_tool"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate real tool-execution Agent tasks.")
    parser.add_argument("--output", default="data/agent/real_tool_tasks_v1.jsonl")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--failure-count", type=int)
    args = parser.parse_args()

    failure_count = args.failure_count if args.failure_count is not None else args.count // 5
    rows = build_tasks(args.count, failure_count)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} real tool tasks to {output}")
    print(f"Injected deterministic failures: {sum(bool(row['fault_schedule']) for row in rows)}")


if __name__ == "__main__":
    main()
