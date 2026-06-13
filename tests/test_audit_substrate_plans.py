import unittest

from scripts.audit_substrate_plans import audit_plans, to_markdown


class AuditSubstratePlansTests(unittest.TestCase):
    def test_audits_overlap_validation_and_bias_norm(self):
        plan_3 = {
            "budget_plan": {"actual_sparsity": 0.1},
            "layers": [
                {
                    "layer": 0,
                    "num_mlp_channels": 10,
                    "pruned_mlp_channels": [1],
                    "mlp_output_bias_compensation": [3.0, 4.0],
                    "mlp_output_scale": 1.0,
                },
                {
                    "layer": 1,
                    "num_mlp_channels": 10,
                    "pruned_mlp_channels": [0],
                    "mlp_output_bias_compensation": [],
                    "mlp_output_scale": 1.0,
                },
            ],
        }
        plan_5 = {
            "budget_plan": {"actual_sparsity": 0.2},
            "layers": [
                {
                    "layer": 0,
                    "num_mlp_channels": 10,
                    "pruned_mlp_channels": [1, 2, 2, 99],
                    "mlp_output_bias_compensation": [0.0],
                },
                {
                    "layer": 1,
                    "num_mlp_channels": 10,
                    "pruned_mlp_channels": [0],
                    "mlp_output_bias_compensation": [],
                },
            ],
        }

        audit = audit_plans({"flap_0p03": plan_3, "flap_0p05": plan_5})
        markdown = to_markdown(audit)

        self.assertEqual(audit["plans"]["flap_0p05"]["duplicate_count"], 1)
        self.assertEqual(audit["plans"]["flap_0p05"]["out_of_range_count"], 1)
        self.assertAlmostEqual(
            audit["plans"]["flap_0p03"]["layers"][0]["bias_compensation_norm"],
            5.0,
        )
        pair = audit["pairwise_overlap"]["flap_0p03_vs_flap_0p05"]
        self.assertTrue(pair["left_subset_of_right"])
        self.assertEqual(pair["intersection_count"], 2)
        self.assertAlmostEqual(pair["nesting_overlap"], 1.0)
        self.assertIn("Substrate Plan Audit", markdown)
        self.assertIn("Duplicate idx", markdown)

    def test_warns_when_lower_budget_is_not_nested(self):
        plan_3 = {
            "layers": [
                {"layer": 0, "num_mlp_channels": 10, "pruned_mlp_channels": [1]},
            ],
        }
        plan_5 = {
            "layers": [
                {"layer": 0, "num_mlp_channels": 10, "pruned_mlp_channels": [2, 3]},
            ],
        }

        audit = audit_plans({"flap_0p03": plan_3, "flap_0p05": plan_5})

        self.assertFalse(
            audit["pairwise_overlap"]["flap_0p03_vs_flap_0p05"]["left_subset_of_right"]
        )
        self.assertTrue(any("not nested" in warning for warning in audit["warnings"]))

    def test_markdown_includes_adjacent_nested_ladder_section(self):
        plan_5 = {
            "layers": [
                {"layer": 0, "num_mlp_channels": 10, "pruned_mlp_channels": [1]},
            ],
        }
        plan_10 = {
            "layers": [
                {"layer": 0, "num_mlp_channels": 10, "pruned_mlp_channels": [1, 2]},
            ],
        }
        audit = audit_plans({"flap_0p05": plan_5, "flap_0p10": plan_10})
        audit["nested_validation"] = {
            "pairs": [
                {
                    "left_target": 0.05,
                    "right_target": 0.10,
                    "left_pruned_count": 1,
                    "right_pruned_count": 2,
                    "newly_pruned_count": 1,
                    "left_only_count": 0,
                    "nesting_overlap": 1.0,
                    "left_subset_of_right": True,
                }
            ],
            "warnings": [],
        }

        markdown = to_markdown(audit)

        self.assertIn("Adjacent Nested Ladder", markdown)
        self.assertIn("Newly pruned", markdown)


if __name__ == "__main__":
    unittest.main()
