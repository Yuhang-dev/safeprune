from __future__ import annotations

import ast
import operator
from typing import Any

from .base import ToolResult, ToolSpec


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorExpressionError(ValueError):
    pass


def calculator_tool() -> ToolSpec:
    return ToolSpec(
        name="calculator",
        description="Evaluate a simple arithmetic expression with numbers and + - * / parentheses.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string"},
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
        handler=run_calculator,
    )


def run_calculator(expression: str) -> ToolResult:
    try:
        result = _eval_expression(expression)
    except CalculatorExpressionError as exc:
        return ToolResult(
            ok=False,
            output={"error": str(exc)},
            event="invalid_argument",
            retryable=True,
        )
    except ZeroDivisionError:
        return ToolResult(
            ok=False,
            output={"error": "division by zero"},
            event="invalid_argument",
            retryable=True,
        )

    return ToolResult(ok=True, output={"result": _clean_number(result)})


def _eval_expression(expression: str) -> float:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorExpressionError("invalid arithmetic syntax") from exc
    return float(_eval_node(tree.body))


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int | float) and not isinstance(node.value, bool):
            return float(node.value)
        raise CalculatorExpressionError("only numeric constants are allowed")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise CalculatorExpressionError("only + - * / operators are allowed")
        return float(_BIN_OPS[op_type](_eval_node(node.left), _eval_node(node.right)))

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise CalculatorExpressionError("only unary + and - are allowed")
        return float(_UNARY_OPS[op_type](_eval_node(node.operand)))

    raise CalculatorExpressionError(f"unsupported expression node: {type(node).__name__}")


def _clean_number(value: float) -> int | float:
    if value.is_integer():
        return int(value)
    return value
