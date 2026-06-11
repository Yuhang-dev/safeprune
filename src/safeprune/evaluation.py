from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TrajectoryRunResult:
    task_id: str
    success: bool
    schema_pass: bool = True
    tool_calls: int = 0
    tool_errors: int = 0
    recovered_errors: int = 0
    active_ffn_ratios: list[float] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass(frozen=True)
class TrajectoryMetrics:
    total_tasks: int
    successful_tasks: int
    task_success_rate: float
    schema_pass_rate: float | None
    tool_call_validity: float | None
    recovery_success_rate: float | None
    average_active_ffn_ratio: float | None
    total_latency_ms: float
    active_ffn_cost: float
    cost_per_success: float | None

    def to_dict(self) -> dict:
        return {
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "task_success_rate": self.task_success_rate,
            "schema_pass_rate": self.schema_pass_rate,
            "tool_call_validity": self.tool_call_validity,
            "recovery_success_rate": self.recovery_success_rate,
            "average_active_ffn_ratio": self.average_active_ffn_ratio,
            "total_latency_ms": self.total_latency_ms,
            "active_ffn_cost": self.active_ffn_cost,
            "cost_per_success": self.cost_per_success,
        }


def compute_trajectory_metrics(results: list[TrajectoryRunResult]) -> TrajectoryMetrics:
    total = len(results)
    successes = sum(1 for result in results if result.success)
    schema_total = total
    schema_passes = sum(1 for result in results if result.schema_pass)
    tool_calls = sum(result.tool_calls for result in results)
    tool_errors = sum(result.tool_errors for result in results)
    recovered_errors = sum(result.recovered_errors for result in results)
    total_latency = sum(result.latency_ms for result in results)

    per_task_active = [
        sum(result.active_ffn_ratios) / len(result.active_ffn_ratios)
        for result in results
        if result.active_ffn_ratios
    ]
    active_ffn_cost = sum(per_task_active)

    return TrajectoryMetrics(
        total_tasks=total,
        successful_tasks=successes,
        task_success_rate=successes / total if total else 0.0,
        schema_pass_rate=schema_passes / schema_total if schema_total else None,
        tool_call_validity=(tool_calls - tool_errors) / tool_calls if tool_calls else None,
        recovery_success_rate=recovered_errors / tool_errors if tool_errors else None,
        average_active_ffn_ratio=(
            sum(per_task_active) / len(per_task_active) if per_task_active else None
        ),
        total_latency_ms=total_latency,
        active_ffn_cost=active_ffn_cost,
        cost_per_success=active_ffn_cost / successes if successes else None,
    )


def strict_agent_answer_correct(tool: str, expected: str, prediction: str) -> bool:
    text = prediction.strip()
    if tool == "lookup":
        match = re.search(r"\bowner_\d+\b", text)
        return bool(match and match.group(0) == expected)

    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return False
    try:
        predicted = float(match.group(0))
        target = float(expected)
    except ValueError:
        return False
    return predicted == target


def is_generation_collapse(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    return len(stripped) >= 40 and len(set(stripped)) <= 4
