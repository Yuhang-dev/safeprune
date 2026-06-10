from __future__ import annotations

from dataclasses import dataclass

from .data import AgentStep


@dataclass(frozen=True)
class RouterDecision:
    stage: str
    sparsity: float
    reason: str
    recovery_steps_remaining: int


class FFNMaskRouter:
    """Rule-based v1 router for stage-aware FFN mask selection."""

    def __init__(
        self,
        stage_sparsities: dict[str, float],
        default_sparsity: float,
        failure_sparsity: float,
        failure_events: list[str],
        recovery_window_steps: int = 2,
    ) -> None:
        self.stage_sparsities = dict(stage_sparsities)
        self.default_sparsity = float(default_sparsity)
        self.failure_sparsity = float(failure_sparsity)
        self.failure_events = set(failure_events)
        self.recovery_window_steps = int(recovery_window_steps)
        self._recovery_steps_remaining = 0

    def reset(self) -> None:
        self._recovery_steps_remaining = 0

    def route(self, step: AgentStep) -> RouterDecision:
        event = step.event or ""
        if step.is_failure or event in self.failure_events:
            self._recovery_steps_remaining = self.recovery_window_steps
            return RouterDecision(
                stage=step.stage,
                sparsity=self.failure_sparsity,
                reason=f"failure_event:{event or 'unknown'}",
                recovery_steps_remaining=self._recovery_steps_remaining,
            )

        if self._recovery_steps_remaining > 0:
            self._recovery_steps_remaining -= 1
            return RouterDecision(
                stage=step.stage,
                sparsity=self.failure_sparsity,
                reason="recovery_window",
                recovery_steps_remaining=self._recovery_steps_remaining,
            )

        return RouterDecision(
            stage=step.stage,
            sparsity=float(self.stage_sparsities.get(step.stage, self.default_sparsity)),
            reason="stage_default",
            recovery_steps_remaining=0,
        )
