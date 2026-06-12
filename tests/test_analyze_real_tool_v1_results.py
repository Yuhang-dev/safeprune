import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_real_tool_v1_results import build_analysis, to_markdown


class AnalyzeRealToolV1ResultsTests(unittest.TestCase):
    def test_build_analysis_summarizes_main_pairwise_and_events(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            predictions = root / "pred.jsonl"
            metrics = root / "metrics.json"
            pairwise = root / "pairwise.json"
            failures = root / "failures.jsonl"

            metrics.write_text(
                json.dumps(
                    {
                        "failure_redense_global_balanced_approx_0.01": {
                            "total": 2,
                            "correct": 2,
                            "failure_task_total": 1,
                            "failure_task_correct": 1,
                            "non_failure_task_total": 1,
                            "non_failure_task_correct": 1,
                            "cost_per_success": 1.1,
                            "fallback_step_ratio": 0.2,
                            "schema_validity_rate": 1.0,
                        },
                        "stage_reflect_dense_global_balanced_approx_0.01": {
                            "total": 2,
                            "correct": 1,
                            "failure_task_total": 1,
                            "failure_task_correct": 0,
                            "non_failure_task_total": 1,
                            "non_failure_task_correct": 1,
                            "cost_per_success": 1.5,
                            "fallback_step_ratio": 0.1,
                            "schema_validity_rate": 0.9,
                        },
                    }
                ),
                encoding="utf-8",
            )
            pairwise.write_text(
                json.dumps(
                    {
                        "failure_redense_global_balanced_approx_0.01_vs_stage_reflect_dense_global_balanced_approx_0.01": {
                            "all": {
                                "both_correct": 1,
                                "left_only_correct": 1,
                                "right_only_correct": 0,
                                "both_wrong": 0,
                            },
                            "failure": {
                                "both_correct": 0,
                                "left_only_correct": 1,
                                "right_only_correct": 0,
                                "both_wrong": 0,
                            },
                            "non_failure": {
                                "both_correct": 1,
                                "left_only_correct": 0,
                                "right_only_correct": 0,
                                "both_wrong": 0,
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            rows = [
                {
                    "method": "failure_redense_global_balanced_approx_0.01",
                    "task_id": "t1",
                    "tool": "lookup",
                    "success": True,
                    "failure_task": True,
                    "events": ["tool_error", "ok"],
                },
                {
                    "method": "stage_reflect_dense_global_balanced_approx_0.01",
                    "task_id": "t1",
                    "tool": "lookup",
                    "success": False,
                    "failure_task": True,
                    "events": ["tool_error", "schema_error"],
                    "terminal_event": "max_steps_exceeded",
                },
            ]
            predictions.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            failures.write_text(json.dumps(rows[1]) + "\n", encoding="utf-8")

            payload = build_analysis(
                predictions=predictions,
                metrics=metrics,
                pairwise=pairwise,
                failures=failures,
            )
            markdown = to_markdown(payload)

        self.assertEqual(payload["fallback_comparison"]["relative_reduction"], 0.5)
        self.assertIn("Failure-redense", markdown)
        self.assertIn("Stage-reflect-dense", markdown)
        self.assertEqual(
            payload["failure_type_split"][
                "stage_reflect_dense_global_balanced_approx_0.01"
            ]["event_occurrences"]["schema_error"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
