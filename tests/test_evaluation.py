import unittest

from safeprune.evaluation import (
    TrajectoryRunResult,
    compute_trajectory_metrics,
    strict_agent_answer_correct,
)


class EvaluationTests(unittest.TestCase):
    def test_cost_per_success_handles_zero_success(self):
        metrics = compute_trajectory_metrics(
            [
                TrajectoryRunResult(
                    task_id="t1",
                    success=False,
                    schema_pass=False,
                    tool_calls=1,
                    tool_errors=1,
                    active_ffn_ratios=[0.7],
                )
            ]
        )
        self.assertEqual(metrics.successful_tasks, 0)
        self.assertIsNone(metrics.cost_per_success)
        self.assertEqual(metrics.recovery_success_rate, 0.0)

    def test_cost_per_success_uses_active_ffn_cost(self):
        metrics = compute_trajectory_metrics(
            [
                TrajectoryRunResult(task_id="t1", success=True, active_ffn_ratios=[0.5, 0.7]),
                TrajectoryRunResult(task_id="t2", success=True, active_ffn_ratios=[0.8]),
            ]
        )
        self.assertAlmostEqual(metrics.active_ffn_cost, 1.4)
        self.assertAlmostEqual(metrics.cost_per_success, 0.7)

    def test_strict_agent_answer_correct_avoids_numeric_substring_matches(self):
        self.assertTrue(strict_agent_answer_correct("calculator", "9", "Final answer: 9"))
        self.assertTrue(strict_agent_answer_correct("unit_convert", "900", "900 centimeters"))
        self.assertFalse(strict_agent_answer_correct("calculator", "9", "Final answer: 19"))
        self.assertFalse(strict_agent_answer_correct("unit_convert", "900", "9000 centimeters"))

    def test_strict_agent_answer_correct_avoids_lookup_substring_matches(self):
        self.assertTrue(strict_agent_answer_correct("lookup", "owner_4", "owner_4"))
        self.assertTrue(strict_agent_answer_correct("lookup", "owner_4", "Final answer: owner_4."))
        self.assertFalse(strict_agent_answer_correct("lookup", "owner_4", "owner_44"))
        self.assertFalse(strict_agent_answer_correct("lookup", "owner_4", "owner_49"))


if __name__ == "__main__":
    unittest.main()
