from pathlib import Path
import unittest

from safeprune.config import PruningConfig, SafePruneConfig, load_config


class ConfigTests(unittest.TestCase):
    def test_config_from_dict_minimal(self):
        config = SafePruneConfig.from_dict(
            {
                "experiment": {"name": "x"},
                "model": {"base_model": "Qwen/Qwen2.5-7B-Instruct"},
                "data": {"preference_train": "train.jsonl"},
            }
        )
        self.assertEqual(config.experiment.seed, 42)
        self.assertEqual(config.pruning.sparsity, 0.35)

    def test_pruning_config_rejects_invalid_sparsity(self):
        with self.assertRaisesRegex(ValueError, "sparsity"):
            PruningConfig(sparsity=1.0).validate()

    def test_4090_smoke_config_extends_base(self):
        config = load_config(Path("configs") / "safeprune_qwen2_5_1_5b_4090_smoke.yaml")
        self.assertEqual(config.model.base_model, "Qwen/Qwen2.5-1.5B-Instruct")
        self.assertEqual(config.pruning.target_sparsities, [0.25])
        self.assertEqual(config.data.preference_train, "data/preference/train.jsonl")

    def test_4090_main_config_extends_base(self):
        config = load_config(Path("configs") / "safeprune_qwen2_5_3b_4090.yaml")
        self.assertEqual(config.model.base_model, "Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(config.pruning.target_sparsities, [0.25, 0.35])
        self.assertEqual(config.recovery.lora_r, 16)


if __name__ == "__main__":
    unittest.main()
