import json
import tempfile
import unittest
from pathlib import Path

from safeprune.tool_env import LocalToolEnvironment, load_real_tool_tasks, validate_real_tool_task
from safeprune.tools import default_tool_registry


def _task(**overrides):
    row = {
        "task_id": "t1",
        "user_request": "Use calculator.",
        "tool": "calculator",
        "expected_answer": "5",
        "expected_tool": "calculator",
        "expected_arguments": {"expression": "2 + 3"},
        "max_steps": 6,
        "fault_schedule": [],
    }
    row.update(overrides)
    return row


class ToolEnvironmentTests(unittest.TestCase):
    def test_load_real_tool_tasks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tasks.jsonl"
            path.write_text(json.dumps(_task()) + "\n", encoding="utf-8")
            tasks = load_real_tool_tasks(path)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].expected_tool, "calculator")

    def test_fault_schedule_is_deterministic_per_attempt(self):
        task = validate_real_tool_task(
            _task(
                fault_schedule=[
                    {
                        "tool": "calculator",
                        "attempt": 1,
                        "event": "timeout",
                        "retryable": True,
                    }
                ]
            )
        )
        env = LocalToolEnvironment(task, default_tool_registry())
        first = env.execute("calculator", {"expression": "2 + 3"})
        second = env.execute("calculator", {"expression": "2 + 3"})

        self.assertFalse(first.ok)
        self.assertEqual(first.event, "timeout")
        self.assertTrue(second.ok)
        self.assertEqual(second.output["result"], 5)

    def test_success_requires_successful_tool_call_and_correct_answer(self):
        task = validate_real_tool_task(_task())
        env = LocalToolEnvironment(task, default_tool_registry())
        self.assertFalse(env.check_success("5"))

        env.execute("calculator", {"expression": "2 + 3"})
        self.assertTrue(env.check_success("5"))
        self.assertFalse(env.check_success("6"))


if __name__ == "__main__":
    unittest.main()
