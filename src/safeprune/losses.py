from __future__ import annotations

from dataclasses import dataclass


def require_torch():
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - depends on remote env
        raise RuntimeError("Torch is required for SafePrune-DPO losses.") from exc
    return torch, functional


@dataclass(frozen=True)
class SafePruneLossWeights:
    dpo: float = 1.0
    consistency: float = 0.2
    safety_replay: float = 0.3


def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta):
    torch, functional = require_torch()
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (pi_logratios - ref_logratios)
    losses = -functional.logsigmoid(logits)
    rewards_chosen = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rewards_rejected = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    return losses.mean(), rewards_chosen, rewards_rejected


def kl_consistency_loss(student_logits, teacher_logits, attention_mask=None, temperature=1.0):
    torch, functional = require_torch()
    student_log_probs = functional.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = functional.softmax(teacher_logits / temperature, dim=-1)
    token_kl = functional.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    if attention_mask is not None:
        token_kl = token_kl * attention_mask
        denom = attention_mask.sum().clamp_min(1)
        return token_kl.sum() / denom * (temperature**2)
    return token_kl.mean() * (temperature**2)


def weighted_safeprune_loss(dpo, consistency, safety_replay, weights: SafePruneLossWeights):
    return weights.dpo * dpo + weights.consistency * consistency + weights.safety_replay * safety_replay

