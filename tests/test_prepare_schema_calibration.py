import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_schema_calibration import build_schema_calibration_rows


class PrepareSchemaCalibrationTests(unittest.TestCase):
    def test_builds_assistant_target_snippets_for_tool_final_and_retry(self):
        task = {
            "task_id": "t1",
            "user_request": "Convert 2 m to cm.",
            "tool": "unit_convert",
            "expected_answer": "200",
            "expected_tool": "unit_convert",
            "expected_arguments": {"value": 2, "from_unit": "m", "to_unit": "cm"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.jsonl"
            path.write_text(json.dumps(task) + "\n", encoding="utf-8")

            rows = build_schema_calibration_rows(path)

        self.assertEqual(len(rows), 3)
        kinds = {row["schema_target_kind"] for row in rows}
        self.assertEqual(
            kinds,
            {"tool_call_initial", "final_after_observation", "retry_after_failure"},
        )
        for row in rows:
            self.assertTrue(row["schema_target_only"])
            self.assertIn("assistant_target", row)
        tool_targets = [
            row for row in rows if row["schema_target_kind"] != "final_after_observation"
        ]
        self.assertTrue(all("unit_convert" in row["assistant_target"] for row in tool_targets))
        final = [row for row in rows if row["schema_target_kind"] == "final_after_observation"][0]
        self.assertIn('"type":"final"', final["assistant_target"])
        retry = [row for row in rows if row["schema_target_kind"] == "retry_after_failure"][0]
        self.assertIn("Tool observation", retry["prompt"])
        self.assertIn("timeout", retry["prompt"])


if __name__ == "__main__":
    unittest.main()
