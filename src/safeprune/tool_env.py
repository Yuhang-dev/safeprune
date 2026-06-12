from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .evaluation import strict_agent_answer_correct
from .tools import ToolRegistry, default_tool_registry
from .tools.base import ToolResult


@dataclass(frozen=True)
class FaultSpec:
    tool: str
    attempt: int
    event: str
    retryable: bool = True
    output: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RealToolTask:
    task_id: str
    user_request: str
    tool: str
    expected_answer: str
    expected_tool: str
    expected_arguments: dict[str, Any]
    max_steps: int = 6
    requires_tool_success: bool = True
    fault_schedule: list[FaultSpec] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_injected_failure(self) -> bool:
        return bool(self.fault_schedule)


class LocalToolEnvironment:
    def __init__(
        self,
        task: RealToolTask,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.task = task
        self.registry = registry or default_tool_registry()
        self.attempts: dict[str, int] = {}
        self.tool_results: list[dict[str, Any]] = []

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        attempt = self.attempts.get(name, 0) + 1
        self.attempts[name] = attempt

        injected = self._injected_fault(name, attempt)
        if injected is not None:
            result = injected
        else:
            result = self.registry.execute(name, arguments)

        self.tool_results.append(
            {
                "name": name,
                "arguments": arguments,
                "attempt": attempt,
                **result.to_dict(),
            }
        )
        return result

    @property
    def has_successful_tool_observation(self) -> bool:
        return any(
            bool(result.get("ok"))
            and result.get("event") in {"ok", "success"}
            and result.get("output") not in (None, {})
            and bool(result.get("counts_as_successful_observation", True))
            for result in self.tool_results
        )

    def check_success(self, final_answer: str | None) -> bool:
        if final_answer is None:
            return False
        if self.task.requires_tool_success and not self.has_successful_tool_observation:
            return False
        return strict_agent_answer_correct(
            self.task.tool,
            self.task.expected_answer,
            final_answer,
        )

    def _injected_fault(self, name: str, attempt: int) -> ToolResult | None:
        for fault in self.task.fault_schedule:
            if fault.tool == name and fault.attempt == attempt:
                output = fault.output or {"error": fault.event}
                return ToolResult(
                    ok=False,
                    output=output,
                    event=fault.event,
                    retryable=fault.retryable,
                )
        return None


def validate_real_tool_task(raw: dict[str, Any], line_number: int | None = None) -> RealToolTask:
    prefix = f"line {line_number}: " if line_number is not None else ""
    required = {
        "task_id",
        "user_request",
        "tool",
        "expected_answer",
        "expected_tool",
        "expected_arguments",
    }
    missing = required - set(raw)
    if missing:
        raise ValueError(f"{prefix}missing keys: {sorted(missing)}")

    for key in ["task_id", "user_request", "tool", "expected_answer", "expected_tool"]:
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(f"{prefix}{key} must be a non-empty string")
    if not isinstance(raw["expected_arguments"], dict):
        raise ValueError(f"{prefix}expected_arguments must be an object")

    fault_schedule = [
        validate_fault_spec(fault, prefix=f"{prefix}fault {idx}: ")
        for idx, fault in enumerate(raw.get("fault_schedule", []), start=1)
    ]
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"{prefix}metadata must be an object")

    max_steps = int(raw.get("max_steps", 6))
    if max_steps < 1:
        raise ValueError(f"{prefix}max_steps must be positive")
    requires_tool_success = raw.get("requires_tool_success", True)
    if not isinstance(requires_tool_success, bool):
        raise ValueError(f"{prefix}requires_tool_success must be a boolean")

    return RealToolTask(
        task_id=raw["task_id"],
        user_request=raw["user_request"],
        tool=raw["tool"],
        expected_answer=raw["expected_answer"],
        expected_tool=raw["expected_tool"],
        expected_arguments=raw["expected_arguments"],
        max_steps=max_steps,
        requires_tool_success=requires_tool_success,
        fault_schedule=fault_schedule,
        metadata=metadata,
    )


def validate_fault_spec(raw: dict[str, Any], prefix: str = "") -> FaultSpec:
    required = {"tool", "attempt", "event"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"{prefix}missing fault keys: {sorted(missing)}")
    if not isinstance(raw["tool"], str) or not raw["tool"].strip():
        raise ValueError(f"{prefix}tool must be a non-empty string")
    if not isinstance(raw["event"], str) or not raw["event"].strip():
        raise ValueError(f"{prefix}event must be a non-empty string")
    attempt = int(raw["attempt"])
    if attempt < 1:
        raise ValueError(f"{prefix}attempt must be positive")
    output = raw.get("output", {})
    if not isinstance(output, dict):
        raise ValueError(f"{prefix}output must be an object")
    return FaultSpec(
        tool=raw["tool"],
        attempt=attempt,
        event=raw["event"],
        retryable=bool(raw.get("retryable", True)),
        output=output,
    )


def load_real_tool_tasks(path: str | Path) -> list[RealToolTask]:
    tasks = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: JSONL rows must be objects")
            tasks.append(validate_real_tool_task(row, line_number))
    return tasks
