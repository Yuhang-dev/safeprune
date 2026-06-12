from __future__ import annotations

from .base import ToolResult, ToolSpec


UNIT_TO_METER = {
    "mm": 0.001,
    "cm": 0.01,
    "m": 1.0,
    "km": 1000.0,
}


def unit_convert_tool() -> ToolSpec:
    return ToolSpec(
        name="unit_convert",
        description="Convert a numeric value between mm, cm, m, and km.",
        parameters={
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "from_unit": {"type": "string", "enum": sorted(UNIT_TO_METER)},
                "to_unit": {"type": "string", "enum": sorted(UNIT_TO_METER)},
            },
            "required": ["value", "from_unit", "to_unit"],
            "additionalProperties": False,
        },
        handler=run_unit_convert,
    )


def run_unit_convert(value: float, from_unit: str, to_unit: str) -> ToolResult:
    if from_unit not in UNIT_TO_METER or to_unit not in UNIT_TO_METER:
        return ToolResult(
            ok=False,
            output={"error": "unsupported unit"},
            event="invalid_argument",
            retryable=True,
        )

    meters = float(value) * UNIT_TO_METER[from_unit]
    converted = meters / UNIT_TO_METER[to_unit]
    return ToolResult(
        ok=True,
        output={"result": _clean_number(converted), "unit": to_unit},
    )


def _clean_number(value: float) -> int | float:
    if value.is_integer():
        return int(value)
    return value
