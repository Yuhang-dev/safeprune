import tempfile
import unittest
from pathlib import Path

from safeprune.pruning import ScoreWeights
from safeprune.scoring import LayerScores
from safeprune.stage_masks import (
    StageMaskBank,
    active_mlp_ratio_from_plan,
    load_stage_mask_bank,
    save_stage_mask_bank,
)


class StageMaskBankTests(unittest.TestCase):
    def test_stage_mask_bank_builds_nested_masks(self):
        scores = {
            "plan": [
                LayerScores(
                    layer=0,
                    attention=[1.0, 1.0],
                    mlp=[0.1, 0.2, 0.3, 0.4],
                    activation_mlp=[0.1, 0.2, 0.3, 0.4],
                )
            ]
        }
        bank = StageMaskBank.from_stage_scores(
            stage_scores=scores,
            target_sparsities=[0.25, 0.50],
            weights=ScoreWeights(magnitude=1.0, activation=0.0, loss_delta=0.0),
            min_mlp_channels_per_layer=1,
        )
        self.assertEqual(bank.select("plan", 0.25)["layers"][0]["pruned_mlp_channels"], [0])
        self.assertEqual(bank.select("plan", 0.50)["layers"][0]["pruned_mlp_channels"], [0, 1])
        self.assertEqual(bank.active_mlp_ratio("plan", 0.50), 0.5)

    def test_stage_mask_bank_roundtrip(self):
        bank = StageMaskBank(
            plans={
                "answer": {
                    "0.30": {
                        "stage": "answer",
                        "sparsity": 0.3,
                        "layers": [
                            {
                                "layer": 0,
                                "pruned_attention_heads": [],
                                "pruned_mlp_channels": [0],
                                "num_attention_heads": 2,
                                "num_mlp_channels": 4,
                            }
                        ],
                    }
                }
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mask_bank.json"
            save_stage_mask_bank(bank, path)
            loaded = load_stage_mask_bank(path)
        self.assertEqual(loaded.select("answer", 0.30)["layers"][0]["pruned_mlp_channels"], [0])

    def test_compose_layerwise_plan_uses_requested_layer_sparsities(self):
        bank = StageMaskBank(
            plans={
                "observe": {
                    "0.25": {
                        "stage": "observe",
                        "layers": [
                            {
                                "layer": 0,
                                "pruned_attention_heads": [0],
                                "pruned_mlp_channels": [0],
                                "num_attention_heads": 2,
                                "num_mlp_channels": 4,
                            },
                            {
                                "layer": 1,
                                "pruned_attention_heads": [1],
                                "pruned_mlp_channels": [2],
                                "num_attention_heads": 2,
                                "num_mlp_channels": 4,
                            },
                        ],
                    },
                    "0.50": {
                        "stage": "observe",
                        "layers": [
                            {
                                "layer": 0,
                                "pruned_attention_heads": [0, 1],
                                "pruned_mlp_channels": [0, 1],
                                "num_attention_heads": 2,
                                "num_mlp_channels": 4,
                            },
                            {
                                "layer": 1,
                                "pruned_attention_heads": [0, 1],
                                "pruned_mlp_channels": [2, 3],
                                "num_attention_heads": 2,
                                "num_mlp_channels": 4,
                            },
                        ],
                    },
                }
            }
        )

        plan = bank.compose_layerwise_plan("observe", {0: 0.25, 1: 0.50})

        self.assertEqual(plan["layers"][0]["pruned_mlp_channels"], [0])
        self.assertEqual(plan["layers"][1]["pruned_mlp_channels"], [2, 3])
        self.assertEqual(plan["layers"][0]["pruned_attention_heads"], [])
        self.assertEqual(plan["allocation"], {"0": 0.25, "1": 0.5})
        self.assertAlmostEqual(active_mlp_ratio_from_plan(plan), 5 / 8)

    def test_compose_layerwise_plan_clears_unspecified_layers(self):
        bank = StageMaskBank(
            plans={
                "observe": {
                    "0.50": {
                        "stage": "observe",
                        "layers": [
                            {
                                "layer": 0,
                                "pruned_attention_heads": [],
                                "pruned_mlp_channels": [0, 1],
                                "num_attention_heads": 2,
                                "num_mlp_channels": 4,
                            },
                            {
                                "layer": 1,
                                "pruned_attention_heads": [],
                                "pruned_mlp_channels": [2, 3],
                                "num_attention_heads": 2,
                                "num_mlp_channels": 4,
                            },
                        ],
                    }
                }
            }
        )

        plan = bank.compose_layerwise_plan("observe", {0: 0.50})

        self.assertEqual(plan["layers"][0]["pruned_mlp_channels"], [0, 1])
        self.assertEqual(plan["layers"][1]["pruned_mlp_channels"], [])
        self.assertAlmostEqual(active_mlp_ratio_from_plan(plan), 6 / 8)

    def test_compose_layerwise_plan_validates_inputs(self):
        bank = StageMaskBank(
            plans={
                "observe": {
                    "0.25": {
                        "stage": "observe",
                        "layers": [
                            {
                                "layer": 0,
                                "pruned_attention_heads": [],
                                "pruned_mlp_channels": [0],
                                "num_attention_heads": 2,
                                "num_mlp_channels": 4,
                            }
                        ],
                    }
                }
            }
        )

        with self.assertRaises(KeyError):
            bank.compose_layerwise_plan("answer", {0: 0.25})
        with self.assertRaises(KeyError):
            bank.compose_layerwise_plan("observe", {0: 0.50})
        with self.assertRaises(IndexError):
            bank.compose_layerwise_plan("observe", {1: 0.25})
        with self.assertRaises(ValueError):
            bank.compose_layerwise_plan("observe", {0: -0.25})


if __name__ == "__main__":
    unittest.main()
