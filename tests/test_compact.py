import types
import unittest
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from safeprune.compact import materialize_qwen_compact_mlp_subnet
from safeprune.masks import attach_qwen_switchable_mlp_masks


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise unittest.SkipTest("torch is required for compact subnet tests") from exc
    return torch


class _ToyMlp:
    def __new__(cls):
        torch = _torch()

        class _Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = torch.nn.Linear(2, 3, bias=True)
                self.up_proj = torch.nn.Linear(2, 3, bias=True)
                self.down_proj = torch.nn.Linear(3, 2, bias=True)
                self.act_fn = torch.nn.Identity()
                self.gate_proj.weight.data = torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
                )
                self.gate_proj.bias.data = torch.tensor([0.5, 1.5, 2.5])
                self.up_proj.weight.data = torch.tensor(
                    [[2.0, 0.0], [0.0, 2.0], [2.0, 2.0]]
                )
                self.up_proj.bias.data = torch.tensor([0.25, 0.75, 1.25])
                self.down_proj.weight.data = torch.tensor(
                    [[1.0, 10.0, 100.0], [2.0, 20.0, 200.0]]
                )
                self.down_proj.bias.data = torch.tensor([7.0, 11.0])

            def forward(self, x):
                return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return _Module()


class _Layer:
    def __init__(self):
        self.mlp = _ToyMlp()


class _Model:
    def __init__(self):
        self.config = types.SimpleNamespace(intermediate_size=3, hidden_size=2)
        self.model = types.SimpleNamespace(layers=[_Layer()])


def _plan(pruned):
    return {
        "plan_name": "toy",
        "layers": [
            {
                "layer": 0,
                "num_mlp_channels": 3,
                "pruned_mlp_channels": pruned,
                "mlp_output_bias_compensation": [3.0, 5.0],
                "mlp_output_scale": 2.0,
            }
        ],
    }


class CompactSubnetTests(unittest.TestCase):
    def test_compact_mlp_matches_mask_hook_with_bias_and_scale(self):
        torch = _torch()
        model = _Model()
        plan = _plan([1])
        x = torch.tensor([[[1.0, 2.0]]])

        handle = attach_qwen_switchable_mlp_masks(model, initial_plan=plan)
        try:
            mask_output = model.model.layers[0].mlp(x)
        finally:
            handle.remove()

        metadata = materialize_qwen_compact_mlp_subnet(model, plan)
        compact_output = model.model.layers[0].mlp(x)

        self.assertTrue(torch.allclose(mask_output, compact_output))
        self.assertAlmostEqual(metadata["active_mlp_ratio_actual"], 2 / 3)
        self.assertEqual(metadata["layers"][0]["remaining_channels"], 2)
        self.assertEqual(tuple(model.model.layers[0].mlp.gate_proj.weight.shape), (2, 2))
        self.assertEqual(tuple(model.model.layers[0].mlp.down_proj.weight.shape), (2, 2))

    def test_compact_mlp_rejects_duplicate_pruned_channels(self):
        model = _Model()

        with self.assertRaisesRegex(ValueError, "duplicate"):
            materialize_qwen_compact_mlp_subnet(model, _plan([1, 1]))

    def test_compact_mlp_leaves_identity_layer_unwrapped(self):
        model = _Model()
        original_mlp = model.model.layers[0].mlp
        plan = {
            "plan_name": "identity",
            "layers": [
                {
                    "layer": 0,
                    "num_mlp_channels": 3,
                    "pruned_mlp_channels": [],
                    "mlp_output_bias_compensation": [],
                    "mlp_output_scale": 1.0,
                }
            ],
        }

        metadata = materialize_qwen_compact_mlp_subnet(model, plan)

        self.assertIs(model.model.layers[0].mlp, original_mlp)
        self.assertFalse(metadata["layers"][0]["replaced"])
        self.assertEqual(metadata["layers"][0]["remaining_channels"], 3)


if __name__ == "__main__":
    unittest.main()
