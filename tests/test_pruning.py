import unittest

from safeprune.pruning import (
    ScoreWeights,
    build_keep_mask,
    combine_importance_scores,
    normalize_scores,
    select_pruned_indices,
)
from safeprune.scoring import LayerScores, build_pruning_plan


class PruningTests(unittest.TestCase):
    def test_normalize_scores_handles_constant_values(self):
        self.assertEqual(normalize_scores([3.0, 3.0]), [0.0, 0.0])

    def test_select_pruned_indices_respects_min_remaining(self):
        pruned = select_pruned_indices([0.1, 0.2, 0.3, 0.4], sparsity=0.75, min_remaining=2)
        self.assertEqual(pruned, [0, 1])
        self.assertEqual(build_keep_mask(4, pruned), [False, False, True, True])

    def test_combine_importance_scores_prefers_high_weighted_values(self):
        scores = combine_importance_scores(
            magnitude=[1.0, 2.0],
            activation=[1.0, 1.0],
            loss_delta=[0.0, 10.0],
            weights=ScoreWeights(magnitude=0.5, activation=0.0, loss_delta=0.5),
        )
        self.assertGreater(scores[1], scores[0])

    def test_build_pruning_plan_can_disable_attention_pruning(self):
        plan = build_pruning_plan(
            scores=[LayerScores(layer=0, attention=[0.1, 0.2, 0.3, 0.4], mlp=[0.1, 0.2, 0.3, 0.4])],
            sparsity=0.5,
            weights=ScoreWeights(magnitude=1.0, activation=0.0, loss_delta=0.0),
            min_heads_per_layer=1,
            min_mlp_channels_per_layer=1,
            prune_attention_heads=False,
            prune_mlp_channels=True,
        )
        layer = plan["layers"][0]
        self.assertEqual(layer["pruned_attention_heads"], [])
        self.assertEqual(layer["pruned_mlp_channels"], [0, 1])


if __name__ == "__main__":
    unittest.main()
