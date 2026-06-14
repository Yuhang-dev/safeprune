import unittest
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from safeprune.pruning import ScoreWeights
from safeprune.scoring import LayerScores

from scripts.build_hardware_aligned_compact_plans import (
    audit_aligned_plans,
    build_hardware_aligned_plans,
)


class HardwareAlignedPlanTests(unittest.TestCase):
    def test_aligned_plans_are_nested_and_respect_width_constraints(self):
        scores = [
            LayerScores(layer=0, attention=[], mlp=[float(i) for i in range(1024)]),
            LayerScores(layer=1, attention=[], mlp=[float(i + 1024) for i in range(1024)]),
        ]
        plans = build_hardware_aligned_plans(
            scores=scores,
            targets=[0.25, 0.50],
            weights=ScoreWeights(magnitude=1.0, activation=0.0, loss_delta=0.0),
            align_to=128,
            min_remaining=256,
            plan_prefix="aligned",
        )
        audit = audit_aligned_plans(plans, align_to=128, min_remaining=256)

        self.assertEqual(len(audit["plans"]), 2)
        self.assertTrue(audit["nested_pairs"][0]["left_subset_of_right"])
        for item in audit["plans"]:
            self.assertEqual(item["non_aligned_layers"], [])
            self.assertEqual(item["below_min_layers"], [])
            self.assertEqual(item["duplicate_layers"], [])
            self.assertEqual(item["out_of_range_layers"], [])


if __name__ == "__main__":
    unittest.main()
