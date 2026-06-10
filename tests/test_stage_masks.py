import tempfile
import unittest
from pathlib import Path

from safeprune.pruning import ScoreWeights
from safeprune.scoring import LayerScores
from safeprune.stage_masks import StageMaskBank, load_stage_mask_bank, save_stage_mask_bank


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


if __name__ == "__main__":
    unittest.main()
