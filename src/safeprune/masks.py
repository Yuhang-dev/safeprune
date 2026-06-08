from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ForwardMaskHandle:
    handles: list[Any]

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def attach_qwen_forward_masks(model, plan: dict[str, Any]) -> ForwardMaskHandle:
    """Attach structured masks without changing tensor shapes.

    This is the recovery-training implementation. It enforces pruned attention
    heads and MLP intermediate channels during forward/backward passes while
    keeping the original checkpoint shape compatible with PEFT and Transformers.
    Physical slicing for speed benchmarks can be added later for selected masks.
    """

    torch = _require_torch()
    layers = _get_decoder_layers(model)
    handles = []
    num_heads = int(getattr(model.config, "num_attention_heads"))
    hidden_size = int(getattr(model.config, "hidden_size"))
    head_dim = hidden_size // num_heads

    for layer_plan in plan["layers"]:
        layer = layers[int(layer_plan["layer"])]
        pruned_heads = set(layer_plan.get("pruned_attention_heads", []))
        pruned_channels = set(layer_plan.get("pruned_mlp_channels", []))

        if pruned_heads:
            head_mask = torch.ones(num_heads, head_dim, dtype=torch.float32)
            for head_idx in pruned_heads:
                head_mask[head_idx, :] = 0.0
            flat_head_mask = head_mask.reshape(1, 1, hidden_size)

            def mask_attention_input(_module, inputs, mask=flat_head_mask):
                hidden_states = inputs[0]
                masked = hidden_states * mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
                return (masked, *inputs[1:])

            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(mask_attention_input))

        if pruned_channels:
            intermediate = int(getattr(model.config, "intermediate_size"))
            channel_mask = torch.ones(intermediate, dtype=torch.float32)
            for channel_idx in pruned_channels:
                if channel_idx < intermediate:
                    channel_mask[channel_idx] = 0.0
            channel_mask = channel_mask.reshape(1, 1, intermediate)

            def mask_mlp_gate(_module, _inputs, output, mask=channel_mask):
                return output * mask.to(device=output.device, dtype=output.dtype)

            def mask_mlp_up(_module, _inputs, output, mask=channel_mask):
                return output * mask.to(device=output.device, dtype=output.dtype)

            handles.append(layer.mlp.gate_proj.register_forward_hook(mask_mlp_gate))
            handles.append(layer.mlp.up_proj.register_forward_hook(mask_mlp_up))

    return ForwardMaskHandle(handles=handles)


def _get_decoder_layers(model) -> list[Any]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise ValueError("Unsupported model layout: expected model.model.layers or transformer.h")


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on remote env
        raise RuntimeError("Torch is required to attach pruning masks.") from exc
    return torch
