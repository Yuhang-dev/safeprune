from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from safeprune.benchmarking import (
    build_benchmark_manifest,
    example_from_arc_easy,
    example_from_hellaswag,
    example_from_piqa,
    sha256_file,
    summarize_subnet_theory,
)


class BenchmarkingTests(unittest.TestCase):
    def test_manifest_records_plan_and_calibration_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = Path(tmp) / "plan.json"
            calib = Path(tmp) / "calib.jsonl"
            plan.write_text('{"layers":[]}', encoding="utf-8")
            calib.write_text('{"prompt":"x"}\n', encoding="utf-8")

            manifest = build_benchmark_manifest(
                model_revision="model@rev",
                plan_path=plan,
                calibration_paths=[calib],
                benchmark_version="p3_static_v1",
                seed=7,
                dtype="bfloat16",
                device="cuda:0",
            )

            self.assertEqual(manifest["model_revision"], "model@rev")
            self.assertEqual(manifest["plan_path"], str(plan))
            self.assertEqual(manifest["plan_sha256"], sha256_file(plan))
            self.assertIsNotNone(manifest["calibration_data_sha256"])
            self.assertEqual(manifest["seed"], 7)

    def test_multiple_choice_converters(self) -> None:
        piqa = example_from_piqa({"goal": "open a jar", "sol1": "twist", "sol2": "sleep", "label": 0})
        self.assertEqual(piqa.label, 0)
        self.assertEqual(len(piqa.choices), 2)

        hellaswag = example_from_hellaswag(
            {"ctx": "A person starts cooking.", "endings": ["They stir.", "They fly."], "label": "0"}
        )
        self.assertEqual(hellaswag.label, 0)

        arc = example_from_arc_easy(
            {
                "question": "Which is wet?",
                "choices": {"label": ["A", "B"], "text": ["water", "stone"]},
                "answerKey": "A",
            }
        )
        self.assertEqual(arc.label, 0)

    def test_subnet_theory_counts_pruned_channels(self) -> None:
        plan = {
            "layers": [
                {"layer": 0, "num_mlp_channels": 4, "pruned_mlp_channels": [1]},
                {"layer": 1, "num_mlp_channels": 4, "pruned_mlp_channels": [0, 3]},
            ]
        }
        summary = summarize_subnet_theory(
            plan=plan,
            hidden_size=8,
            intermediate_size=4,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
        self.assertEqual(summary["total_ffn_channels"], 8)
        self.assertEqual(summary["pruned_ffn_channels"], 3)
        self.assertEqual(summary["remaining_ffn_channels"], 5)
        self.assertAlmostEqual(summary["active_ffn_channel_ratio"], 5 / 8)
        self.assertEqual(summary["remaining_ffn_parameters"], 3 * 8 * 5)
        self.assertEqual(summary["ffn_macs_per_token"], 3 * 8 * 5)


if __name__ == "__main__":
    unittest.main()
