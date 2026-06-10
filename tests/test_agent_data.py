import json
import tempfile
import unittest
from pathlib import Path

from safeprune.data import load_agent_jsonl, validate_agent_trajectory


class AgentDataTests(unittest.TestCase):
    def test_agent_task_loader_validates_schema(self):
        row = {
            "task_id": "t1",
            "prompt": "Use a calculator.",
            "steps": [{"stage": "act", "text": "Call calculator.", "event": "tool_error"}],
            "answer": "3",
            "expected": "3",
        }
        trajectory = validate_agent_trajectory(row)
        self.assertEqual(trajectory.steps[0].stage, "act")
        self.assertTrue(trajectory.steps[0].is_failure)

    def test_load_agent_jsonl(self):
        row = {
            "task_id": "t1",
            "prompt": "p",
            "steps": [{"stage": "plan", "text": "think"}],
            "answer": "a",
            "expected": "a",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "agent.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            trajectories = load_agent_jsonl(path)
        self.assertEqual(len(trajectories), 1)
        self.assertEqual(trajectories[0].task_id, "t1")


if __name__ == "__main__":
    unittest.main()
