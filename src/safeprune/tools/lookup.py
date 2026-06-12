from __future__ import annotations

import re

from .base import ToolResult, ToolSpec


PROJECT_RE = re.compile(r"^PRJ-(\d+)$")


def lookup_tool() -> ToolSpec:
    return ToolSpec(
        name="lookup",
        description="Look up the owner of a project. Project PRJ-N is owned by owner_N.",
        parameters={
            "type": "object",
            "properties": {
                "project": {"type": "string"},
            },
            "required": ["project"],
            "additionalProperties": False,
        },
        handler=run_lookup,
    )


def run_lookup(project: str) -> ToolResult:
    match = PROJECT_RE.match(project.strip())
    if not match:
        return ToolResult(
            ok=False,
            output={"error": "project must use PRJ-N format"},
            event="invalid_argument",
            retryable=True,
        )
    return ToolResult(ok=True, output={"owner": f"owner_{match.group(1)}"})
