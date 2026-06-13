import unittest

from safeprune.pruning import ScoreWeights
from safeprune.scoring import (
    FFNActivationStats,
    LayerScores,
    compute_flap_fluctuation_scores,
    compute_wanda_channel_scores,
)
from safeprune.stage_masks import active_mlp_ratio_from_plan
from safeprune.substrate import (
    add_bias_compensation_to_plan,
    build_budget_options,
    build_nested_global_budget_plans,
    build_plan_from_budget,
    optimize_layerwise_budget,
    validate_nested_pruned_sets,
)


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise unittest.SkipTest("torch is required for substrate v2 tests") from exc
    return torch


class _Mlp:
    def __init__(self, weight):
        torch = _torch()
        self.down_proj = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        self.down_proj.weight.data = weight.clone().float()


class _Layer:
    def __init__(self, weight):
        self.mlp = _Mlp(weight)


class _InnerModel:
    def __init__(self, layers):
        self.layers = layers


class _Model:
    def __init__(self, weights):
        self.model = _InnerModel([_Layer(weight) for weight in weights])


class SubstrateV2Tests(unittest.TestCase):
    def test_wanda_and_flap_scores_use_down_weight_and_activation_stats(self):
        torch = _torch()
        model = _Model([torch.tensor([[3.0, 0.0], [4.0, 2.0]])])
        stats = [
            FFNActivationStats(
                layer=0,
                mean=[1.0, 2.0],
                mean_abs=[1.0, 2.0],
                second_moment=[4.0, 9.0],
                variance=[0.25, 4.0],
                token_count=8,
            )
        ]

        wanda = compute_wanda_channel_scores(model, stats)[0].mlp
        flap = compute_flap_fluctuation_scores(model, stats)[0].mlp

        self.assertAlmostEqual(wanda[0], 5.0 * 2.0)
        self.assertAlmostEqual(wanda[1], 2.0 * 3.0)
        self.assertAlmostEqual(flap[0], 25.0 * 0.25)
        self.assertAlmostEqual(flap[1], 4.0 * 4.0)

    def test_budget_optimizer_satisfies_target_and_builds_plan(self):
        scores = [
            LayerScores(layer=0, attention=[], mlp=[0.1] * 10),
            LayerScores(layer=1, attention=[], mlp=[0.1] * 10),
        ]
        options = build_budget_options(
            scores=scores,
            candidate_sparsities=[0.0, 0.2],
            min_mlp_channels_per_layer=1,
            loss_delta_by_layer={
                0: {0.0: 0.0, 0.2: 10.0},
                1: {0.0: 0.0, 0.2: 1.0},
            },
        )

        budget = optimize_layerwise_budget(options, global_target=0.1)
        plan = build_plan_from_budget(
            scores=scores,
            budget=budget,
            weights=ScoreWeights(magnitude=1.0, activation=0.0, loss_delta=0.0),
            min_mlp_channels_per_layer=1,
        )

        self.assertEqual(budget.layer_sparsities, {1: 0.2})
        self.assertEqual(plan["allocation"], {"1": 0.2})
        self.assertAlmostEqual(active_mlp_ratio_from_plan(plan), 18 / 20)

    def test_bias_compensation_matches_removed_mean_contribution(self):
        torch = _torch()
        weight = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.0, 1.0]])
        model = _Model([weight])
        stats = [
            FFNActivationStats(
                layer=0,
                mean=[10.0, 20.0, 30.0],
                mean_abs=[10.0, 20.0, 30.0],
                second_moment=[100.0, 400.0, 900.0],
                variance=[0.0, 0.0, 0.0],
                token_count=3,
            )
        ]
        plan = {
            "layers": [
                {
                    "layer": 0,
                    "pruned_attention_heads": [],
                    "pruned_mlp_channels": [1, 2],
                    "num_attention_heads": 0,
                    "num_mlp_channels": 3,
                }
            ]
        }

        compensated = add_bias_compensation_to_plan(model, stats, plan)

        self.assertEqual(compensated["compensation"], "bias")
        self.assertEqual(compensated["layers"][0]["mlp_output_bias_compensation"], [130.0, 30.0])

    def test_nested_budget_ladder_monotonically_expands_pruned_sets(self):
        scores = [
            LayerScores(layer=0, attention=[], mlp=[0.1, 0.2, 0.3, 0.4, 0.5]),
        ]

        plans = build_nested_global_budget_plans(
            scores=scores,
            target_budgets=[0.2, 0.4, 0.6],
            weights=ScoreWeights(magnitude=1.0, activation=0.0, loss_delta=0.0),
            min_mlp_channels_per_layer=1,
            plan_name_prefix="flap",
        )
        validation = validate_nested_pruned_sets(plans)

        self.assertEqual(validation["warnings"], [])
        self.assertTrue(all(pair["left_subset_of_right"] for pair in validation["pairs"]))
        self.assertEqual([pair["newly_pruned_count"] for pair in validation["pairs"]], [1, 1])
        self.assertEqual(plans[0.2]["layers"][0]["pruned_mlp_channels"], [0])
        self.assertEqual(plans[0.4]["layers"][0]["pruned_mlp_channels"], [0, 1])
        self.assertAlmostEqual(active_mlp_ratio_from_plan(plans[0.6]), 0.4)


if __name__ == "__main__":
    unittest.main()
