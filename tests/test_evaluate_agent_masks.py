import unittest

from scripts.evaluate_agent_masks import _summarize_rows


def _row(
    *,
    correct: bool,
    active_ffn_ratio: float,
    failure_task: bool = False,
    collapse: bool = False,
    routing_trace: list[dict] | None = None,
) -> dict:
    return {
        "tool": "calculator",
        "strict_correct": correct,
        "collapse": collapse,
        "latency_ms": 10.0,
        "active_ffn_ratio": active_ffn_ratio,
        "failure_task": failure_task,
        "routing_trace": routing_trace or [],
    }


class EvaluateAgentMasksSummaryTests(unittest.TestCase):
    def test_summary_includes_agent_cost_and_recovery_metrics(self):
        metrics = _summarize_rows(
            [
                _row(
                    correct=True,
                    active_ffn_ratio=0.998,
                    failure_task=True,
                    routing_trace=[{"selected_stage": "dense_fallback"}],
                ),
                _row(correct=False, active_ffn_ratio=0.990, failure_task=True),
                _row(correct=True, active_ffn_ratio=0.990),
                _row(correct=False, active_ffn_ratio=0.990, collapse=True),
            ]
        )

        self.assertEqual(metrics["total"], 4)
        self.assertEqual(metrics["correct"], 2)
        self.assertAlmostEqual(metrics["task_success_rate"], 0.5)
        self.assertAlmostEqual(metrics["generation_collapse_rate"], 0.25)
        self.assertAlmostEqual(metrics["active_ffn_ratio"], 0.992)
        self.assertAlmostEqual(metrics["average_active_ffn_ratio"], 0.992)
        self.assertAlmostEqual(metrics["active_ffn_cost"], 3.968)
        self.assertAlmostEqual(metrics["cost_per_success"], 1.984)
        self.assertEqual(metrics["failure_task_total"], 2)
        self.assertAlmostEqual(metrics["failure_task_success_rate"], 0.5)
        self.assertAlmostEqual(metrics["recovery_success_rate"], 0.5)
        self.assertEqual(metrics["non_failure_task_total"], 2)
        self.assertAlmostEqual(metrics["non_failure_task_success_rate"], 0.5)
        self.assertEqual(metrics["dense_fallback_task_count"], 1)

    def test_cost_per_success_is_none_when_no_task_succeeds(self):
        metrics = _summarize_rows(
            [
                _row(correct=False, active_ffn_ratio=0.7),
                _row(correct=False, active_ffn_ratio=0.8),
            ]
        )

        self.assertEqual(metrics["correct"], 0)
        self.assertAlmostEqual(metrics["active_ffn_cost"], 1.5)
        self.assertIsNone(metrics["cost_per_success"])


if __name__ == "__main__":
    unittest.main()
