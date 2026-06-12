from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal


ActionType = Literal["tool_call", "final"]


@dataclass(frozen=True)
class AgentAction:
    type: ActionType
    name: str | None = None
    arguments: dict[str, Any] | None = None
    answer: str | None = None


@dataclass(frozen=True)
class ParseResult:
    ok: bool
    action: AgentAction | None = None
    event: str = "ok"
    error: str | None = None
    retryable: bool = True


def parse_agent_output(text: str) -> ParseResult:
    raw = text.strip()
    if not raw:
        return ParseResult(ok=False, event="format_error", error="empty output")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return ParseResult(ok=False, event="format_error", error=str(exc))

    if not isinstance(payload, dict):
        return ParseResult(ok=False, event="format_error", error="output must be a JSON object")

    kind = payload.get("type")
    if kind == "tool_call":
        expected_keys = {"type", "name", "arguments"}
        extra = set(payload) - expected_keys
        missing = expected_keys - set(payload)
        if extra or missing:
            return ParseResult(
                ok=False,
                event="schema_error",
                error=f"tool_call keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}",
            )
        if not isinstance(payload["name"], str) or not payload["name"].strip():
            return ParseResult(ok=False, event="schema_error", error="tool name must be a string")
        if not isinstance(payload["arguments"], dict):
            return ParseResult(ok=False, event="schema_error", error="arguments must be an object")
        return ParseResult(
            ok=True,
            action=AgentAction(
                type="tool_call",
                name=payload["name"],
                arguments=payload["arguments"],
            ),
        )

    if kind == "final":
        expected_keys = {"type", "answer"}
        extra = set(payload) - expected_keys
        missing = expected_keys - set(payload)
        if extra or missing:
            return ParseResult(
                ok=False,
                event="schema_error",
                error=f"final keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}",
            )
        answer = payload["answer"]
        if isinstance(answer, bool) or answer is None or isinstance(answer, dict | list):
            return ParseResult(
                ok=False,
                event="schema_error",
                error="answer must be a string or number",
            )
        answer_text = str(answer).strip()
        if not answer_text:
            return ParseResult(ok=False, event="schema_error", error="answer must be non-empty")
        return ParseResult(
            ok=True,
            action=AgentAction(type="final", answer=answer_text),
        )

    return ParseResult(ok=False, event="schema_error", error="type must be tool_call or final")


def tool_observation_message(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, sort_keys=True)
