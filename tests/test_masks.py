import types
import unittest

from safeprune.masks import attach_qwen_switchable_mlp_masks


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise unittest.SkipTest("torch is required for mask tests") from exc
    return torch


class _Mlp:
    def __init__(self):
        torch = _torch()
        self.gate_proj = torch.nn.Linear(2, 2, bias=False)
        self.up_proj = torch.nn.Linear(2, 2, bias=False)
        self.down_proj = torch.nn.Linear(2, 2, bias=False)
        self.gate_proj.weight.data = torch.eye(2)
        self.up_proj.weight.data = torch.eye(2)
        self.down_proj.weight.data = torch.tensor([[1.0, 10.0], [2.0, 20.0]])


class _Layer:
    def __init__(self):
        self.mlp = _Mlp()


class _Model:
    def __init__(self):
        self.config = types.SimpleNamespace(intermediate_size=2, hidden_size=2)
        self.model = types.SimpleNamespace(layers=[_Layer()])


class MaskTests(unittest.TestCase):
    def test_switchable_mask_applies_bias_compensation_and_output_scale(self):
        torch = _torch()
        model = _Model()
        handle = attach_qwen_switchable_mlp_masks(model)
        try:
            handle.set_plan(
                {
                    "layers": [
                        {
                            "layer": 0,
                            "pruned_mlp_channels": [1],
                            "mlp_output_bias_compensation": [3.0, 5.0],
                            "mlp_output_scale": 2.0,
                        }
                    ]
                }
            )
            layer = model.model.layers[0]
            x = torch.ones(1, 1, 2)
            hidden = layer.mlp.gate_proj(x) * layer.mlp.up_proj(x)
            output = layer.mlp.down_proj(hidden)
        finally:
            handle.remove()

        self.assertEqual(output.tolist(), [[[5.0, 9.0]]])
        self.assertAlmostEqual(handle.active_mlp_ratio(), 0.5)


if __name__ == "__main__":
    unittest.main()
