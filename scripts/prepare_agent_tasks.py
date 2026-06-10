from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_tasks(count: int) -> list[dict]:
    templates = [
        ("calculator", "What is {a} + {b}?", "{result}"),
        ("unit_convert", "Convert {a} meters to centimeters.", "{result}"),
        ("lookup", "Find the owner of project PRJ-{a}.", "owner_{a}"),
    ]
    tasks = []
    for idx in range(count):
        tool, prompt_template, expected_template = templates[idx % len(templates)]
        a = idx + 2
        b = idx + 7
        result = a + b if tool == "calculator" else a * 100 if tool == "unit_convert" else a
        expected = expected_template.format(a=a, b=b, result=result)
        event = "tool_error" if idx % 5 == 0 else "ok"
        tasks.append(
            {
                "task_id": f"controlled_{idx:04d}",
                "prompt": prompt_template.format(a=a, b=b),
                "steps": [
                    {
                        "stage": "plan",
                        "text": f"Plan the required {tool} call.",
                        "event": "ok",
                    },
                    {
                        "stage": "act",
                        "text": f"Call {tool} with the parsed arguments.",
                        "tool_name": tool,
                        "event": event,
                    },
                    {
                        "stage": "observe",
                        "text": "Read the tool response and detect whether it is usable.",
                        "event": "empty_observation" if event != "ok" else "ok",
                    },
                    {
                        "stage": "reflect",
                        "text": "Recover from the tool issue or confirm the result.",
                        "event": "ok",
                    },
                    {
                        "stage": "answer",
                        "text": f"Return the final answer: {expected}.",
                        "event": "ok",
                    },
                ],
                "answer": expected,
                "expected": expected,
                "source": "controlled_synthetic",
                "metadata": {"tool": tool, "has_injected_error": event != "ok"},
            }
        )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate controlled Agent trajectory tasks.")
    parser.add_argument("--output", default="data/agent/controlled_tasks.jsonl")
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = build_tasks(args.count)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} Agent tasks to {output}")


if __name__ == "__main__":
    main()
