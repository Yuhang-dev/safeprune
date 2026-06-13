from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safeprune.agent_loop import build_tool_system_prompt
from safeprune.tool_env import load_real_tool_tasks
from safeprune.tools import default_tool_registry


SCHEMA_TOKENS = [
    "type",
    "tool_call",
    "final",
    "name",
    "arguments",
    "answer",
    "calculator",
    "unit_convert",
    "lookup",
    "expression",
    "value",
    "from_unit",
    "to_unit",
    "project_id",
    "{",
    "}",
    "[",
    "]",
    '"',
    ",",
    ":",
]


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _tool_call_target(task) -> str:
    return _json_dumps(
        {
            "type": "tool_call",
            "name": task.expected_tool,
            "arguments": task.expected_arguments,
        }
    )


def _final_target(task) -> str:
    return _json_dumps({"type": "final", "answer": task.expected_answer})


def _prompt(system_prompt: str, user_request: str, turns: list[dict[str, str]] | None = None) -> str:
    lines = [
        "<system>",
        system_prompt,
        "</system>",
        "<user>",
        user_request,
        "</user>",
    ]
    for turn in turns or []:
        lines.extend([f"<{turn['role']}>", turn["content"], f"</{turn['role']}>"])
    return "\n".join(lines)


def build_schema_calibration_rows(tasks_path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    tasks = load_real_tool_tasks(tasks_path)
    if limit is not None:
        tasks = tasks[:limit]
    registry = default_tool_registry()
    system_prompt = build_tool_system_prompt(registry)
    rows: list[dict[str, Any]] = []
    for task in tasks:
        tool_call = _tool_call_target(task)
        rows.append(
            {
                "task_id": f"{task.task_id}_tool_call_initial",
                "tool": task.expected_tool,
                "schema_target_kind": "tool_call_initial",
                "prompt": _prompt(system_prompt, task.user_request),
                "assistant_target": tool_call,
                "schema_tokens": SCHEMA_TOKENS,
                "schema_target_only": True,
            }
        )

        ok_observation = {
            "tool": task.expected_tool,
            **registry.execute(task.expected_tool, task.expected_arguments).to_dict(),
        }
        rows.append(
            {
                "task_id": f"{task.task_id}_final_after_observation",
                "tool": task.expected_tool,
                "schema_target_kind": "final_after_observation",
                "prompt": _prompt(
                    system_prompt,
                    task.user_request,
                    [
                        {"role": "assistant", "content": tool_call},
                        {
                            "role": "user",
                            "content": "Tool observation:\n" + _json_dumps(ok_observation),
                        },
                    ],
                ),
                "assistant_target": _final_target(task),
                "schema_tokens": SCHEMA_TOKENS,
                "schema_target_only": True,
            }
        )

        failure_observation = {
            "tool": task.expected_tool,
            "ok": False,
            "output": {"error": "timeout"},
            "event": "timeout",
            "retryable": True,
        }
        rows.append(
            {
                "task_id": f"{task.task_id}_retry_after_failure",
                "tool": task.expected_tool,
                "schema_target_kind": "retry_after_failure",
                "prompt": _prompt(
                    system_prompt,
                    task.user_request,
                    [
                        {"role": "assistant", "content": tool_call},
                        {
                            "role": "user",
                            "content": "Tool observation:\n" + _json_dumps(failure_observation),
                        },
                    ],
                ),
                "assistant_target": tool_call,
                "schema_tokens": SCHEMA_TOKENS,
                "schema_target_only": True,
            }
        )
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare schema-heavy calibration snippets.")
    parser.add_argument("--tasks", default="data/agent/real_tool_tasks_v1.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tasks", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = build_schema_calibration_rows(args.tasks, limit=args.max_tasks)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} schema calibration rows to {output}")


if __name__ == "__main__":
    main()
