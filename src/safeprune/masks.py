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


@dataclass
class SwitchableForwardMaskHandle:
    handles: list[Any]
    layer_masks: list[Any]
    layer_biases: list[Any]
    layer_scales: list[Any]
    intermediate_size: int
    hidden_size: int

    def set_plan(self, plan: dict[str, Any]) -> None:
        torch = _require_torch()
        for layer_mask in self.layer_masks:
            layer_mask.fill_(1.0)
        for layer_bias in self.layer_biases:
            layer_bias.zero_()
        for layer_scale in self.layer_scales:
            layer_scale.fill_(1.0)
        for layer_plan in plan["layers"]:
            layer_idx = int(layer_plan["layer"])
            if layer_idx >= len(self.layer_masks):
                continue
            mask = torch.ones(self.intermediate_size, dtype=torch.float32)
            for channel_idx in layer_plan.get("pruned_mlp_channels", []):
                if 0 <= int(channel_idx) < self.intermediate_size:
                    mask[int(channel_idx)] = 0.0
            self.layer_masks[layer_idx] = mask.reshape(1, 1, self.intermediate_size)
            compensation = layer_plan.get("mlp_output_bias_compensation") or []
            if compensation:
                if len(compensation) != self.hidden_size:
                    raise ValueError("MLP bias compensation width does not match hidden size.")
                self.layer_biases[layer_idx] = torch.tensor(
                    compensation,
                    dtype=torch.float32,
                ).reshape(1, 1, self.hidden_size)
            scale = layer_plan.get("mlp_output_scale")
            if scale is not None:
                self.layer_scales[layer_idx] = torch.tensor(
                    float(scale),
                    dtype=torch.float32,
                )

    def active_mlp_ratio(self) -> float:
        total = len(self.layer_masks) * self.intermediate_size
        if total == 0:
            return 0.0
        active = sum(float(mask.sum().item()) for mask in self.layer_masks)
        return active / total

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

        compensation = layer_plan.get("mlp_output_bias_compensation") or []
        output_scale = layer_plan.get("mlp_output_scale")
        if compensation or output_scale is not None:
            hidden_size = int(getattr(model.config, "hidden_size"))
            if compensation and len(compensation) != hidden_size:
                raise ValueError("MLP bias compensation width does not match hidden size.")
            bias = torch.tensor(compensation or [0.0] * hidden_size, dtype=torch.float32)
            scale = torch.tensor(float(output_scale) if output_scale is not None else 1.0)

            def compensate_mlp_output(_module, _inputs, output, bias=bias, scale=scale):
                device_bias = bias.to(device=output.device, dtype=output.dtype).reshape(1, 1, -1)
                device_scale = scale.to(device=output.device, dtype=output.dtype)
                return output * device_scale + device_bias

            handles.append(layer.mlp.down_proj.register_forward_hook(compensate_mlp_output))

    return ForwardMaskHandle(handles=handles)


def attach_qwen_switchable_mlp_masks(
    model,
    initial_plan: dict[str, Any] | None = None,
) -> SwitchableForwardMaskHandle:
    """Attach FFN channel masks that can switch between stage plans.

    This hook is for mask-based algorithm validation. It does not shrink model
    tensors and should not be used as evidence of physical speedup.
    """

    torch = _require_torch()
    layers = _get_decoder_layers(model)
    handles = []
    intermediate = int(getattr(model.config, "intermediate_size"))
    hidden_size = int(getattr(model.config, "hidden_size"))
    layer_masks = [
        torch.ones(1, 1, intermediate, dtype=torch.float32)
        for _ in layers
    ]
    layer_biases = [
        torch.zeros(1, 1, hidden_size, dtype=torch.float32)
        for _ in layers
    ]
    layer_scales = [
        torch.ones((), dtype=torch.float32)
        for _ in layers
    ]

    for layer_idx, layer in enumerate(layers):

        def mask_mlp_gate(_module, _inputs, output, idx=layer_idx):
            mask = layer_masks[idx].to(device=output.device, dtype=output.dtype)
            return output * mask

        def mask_mlp_up(_module, _inputs, output, idx=layer_idx):
            mask = layer_masks[idx].to(device=output.device, dtype=output.dtype)
            return output * mask

        def compensate_mlp_output(_module, _inputs, output, idx=layer_idx):
            bias = layer_biases[idx].to(device=output.device, dtype=output.dtype)
            scale = layer_scales[idx].to(device=output.device, dtype=output.dtype)
            return output * scale + bias

        handles.append(layer.mlp.gate_proj.register_forward_hook(mask_mlp_gate))
        handles.append(layer.mlp.up_proj.register_forward_hook(mask_mlp_up))
        handles.append(layer.mlp.down_proj.register_forward_hook(compensate_mlp_output))

    handle = SwitchableForwardMaskHandle(
        handles=handles,
        layer_masks=layer_masks,
        layer_biases=layer_biases,
        layer_scales=layer_scales,
        intermediate_size=intermediate,
        hidden_size=hidden_size,
    )
    if initial_plan is not None:
        handle.set_plan(initial_plan)
    return handle


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
