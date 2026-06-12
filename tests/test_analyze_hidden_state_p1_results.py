import unittest

from scripts.analyze_hidden_state_p1_results import build_analysis, to_markdown


class AnalyzeHiddenStateP1ResultsTests(unittest.TestCase):
    def test_builds_main_table_and_detector_confusion(self):
        metrics = {
            "stage_reflect_dense_global_balanced_approx_0.01": {
                "correct": 2,
                "total": 2,
                "failure_task_correct": 1,
                "failure_task_total": 1,
                "non_failure_task_correct": 1,
                "non_failure_task_total": 1,
                "router_inclusive_cost_per_success": 2.0,
            },
            "hidden_state_centroid_global_balanced_approx_0.01": {
                "correct": 1,
                "total": 2,
                "failure_task_correct": 0,
                "failure_task_total": 1,
                "non_failure_task_correct": 1,
                "non_failure_task_total": 1,
                "raw_reflect_detection_recall": 0.5,
                "effective_reflect_redense_recall": 0.0,
                "critical_reflect_miss_rate": 1.0,
                "fallback_step_ratio": 0.0,
                "routing_probe_cost": 4.0,
                "router_inclusive_cost_per_success": 6.0,
            },
        }
        pairwise = {
            "stage_reflect_dense_global_balanced_approx_0.01_vs_hidden_state_centroid_global_balanced_approx_0.01": {
                "all": {
                    "both_correct": 1,
                    "left_only_correct": 1,
                    "right_only_correct": 0,
                    "both_wrong": 0,
                }
            }
        }
        rows = [
            {
                "method": "hidden_state_centroid_global_balanced_approx_0.01",
                "routing_trace": [
                    {
                        "stage": "reflect",
                        "route_metadata": {
                            "centroid_predicted_stage": "answer",
                            "centroid_reflect_predicted": False,
                        },
                    },
                    {
                        "stage": "answer",
                        "route_metadata": {
                            "centroid_predicted_stage": "reflect",
                            "centroid_reflect_predicted": True,
                        },
                    },
                ],
            }
        ]

        analysis = build_analysis(metrics=metrics, pairwise=pairwise, rows=rows)
        markdown = to_markdown(analysis)

        self.assertEqual(analysis["detector_confusion"]["hidden_state_centroid_global_balanced_approx_0.01"]["fn"], 1)
        self.assertEqual(analysis["detector_confusion"]["hidden_state_centroid_global_balanced_approx_0.01"]["fp"], 1)
        self.assertEqual(analysis["probe_cost_ratio"]["hidden_state_centroid_global_balanced_approx_0.01"], 3.0)
        self.assertIn("Hidden centroid", markdown)
        self.assertIn("Selected Pairwise", markdown)


if __name__ == "__main__":
    unittest.main()
