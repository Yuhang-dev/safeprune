from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: dict[str, Any]
    event: str = "ok"
    retryable: bool = False
    counts_as_successful_observation: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "output": self.output,
            "event": self.event,
            "retryable": self.retryable,
            "counts_as_successful_observation": (
                self.ok and self.counts_as_successful_observation
            ),
        }


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., ToolResult]
