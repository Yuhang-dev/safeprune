from __future__ import annotations

from typing import Any

from .base import ToolResult, ToolSpec


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Duplicate tool: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self._tools.values()
        ]

    def execute(self, name: str, args: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                ok=False,
                output={"error": f"unknown_tool: {name}"},
                event="unknown_tool",
                retryable=True,
            )

        validation_error = _validate_json_schema_args(args, tool.parameters)
        if validation_error is not None:
            return ToolResult(
                ok=False,
                output={"error": validation_error},
                event="schema_error",
                retryable=True,
            )

        try:
            return tool.handler(**args)
        except TypeError as exc:
            return ToolResult(
                ok=False,
                output={"error": str(exc)},
                event="schema_error",
                retryable=True,
            )


def _validate_json_schema_args(args: dict[str, Any], schema: dict[str, Any]) -> str | None:
    if schema.get("type") != "object":
        return "tool parameters schema must be an object"

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    missing = required - set(args)
    if missing:
        return f"missing required arguments: {sorted(missing)}"

    if schema.get("additionalProperties") is False:
        extra = set(args) - set(properties)
        if extra:
            return f"unexpected arguments: {sorted(extra)}"

    for name, value in args.items():
        if name not in properties:
            continue
        error = _validate_value(name, value, properties[name])
        if error is not None:
            return error
    return None


def _validate_value(name: str, value: Any, schema: dict[str, Any]) -> str | None:
    expected_type = schema.get("type")
    if expected_type == "string" and not isinstance(value, str):
        return f"{name} must be a string"
    if expected_type == "number" and not _is_number(value):
        return f"{name} must be a number"
    if expected_type == "integer" and not (isinstance(value, int) and not isinstance(value, bool)):
        return f"{name} must be an integer"
    if expected_type == "object" and not isinstance(value, dict):
        return f"{name} must be an object"

    enum = schema.get("enum")
    if enum is not None and value not in enum:
        return f"{name} must be one of {enum}"
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def default_tool_registry() -> ToolRegistry:
    from .calculator import calculator_tool
    from .lookup import lookup_tool
    from .unit_convert import unit_convert_tool

    registry = ToolRegistry()
    registry.register(calculator_tool())
    registry.register(unit_convert_tool())
    registry.register(lookup_tool())
    return registry
