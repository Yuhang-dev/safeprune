from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .evaluation import is_generation_collapse
from .tool_env import LocalToolEnvironment, RealToolTask
from .tool_protocol import ParseResult, parse_agent_output, tool_observation_message
from .tools import ToolRegistry, default_tool_registry


@dataclass(frozen=True)
class RouteOutcome:
    selected_stage: str
    reason: str
    active_ffn_ratio: float
    plan: dict[str, Any] | None = None
    router_prefill_cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


GenerateFn = Callable[[list[dict[str, str]]], str]
RouteFn = Callable[[str, str, list[dict[str, str]]], RouteOutcome]


def build_tool_system_prompt(registry: ToolRegistry) -> str:
    return (
        "You are a tool-using agent.\n"
        "You must output exactly one JSON object and no extra text.\n\n"
        "To call a tool:\n"
        '{"type":"tool_call","name":"<tool_name>","arguments":{...}}\n\n'
        "To answer the user:\n"
        '{"type":"final","answer":"<answer>"}\n\n'
        "Never invent tool results. After a tool observation, use it to decide the next action.\n"
        "Available tools:\n"
        f"{json.dumps(registry.schemas(), ensure_ascii=False)}"
    )


def run_real_tool_episode(
    task: RealToolTask,
    generate_fn: GenerateFn,
    route_fn: RouteFn,
    registry: ToolRegistry | None = None,
) -> dict[str, Any]:
    registry = registry or default_tool_registry()
    env = LocalToolEnvironment(task, registry)
    messages = [
        {"role": "system", "content": build_tool_system_prompt(registry)},
        {"role": "user", "content": task.user_request},
    ]
    last_event = "start"
    routing_trace = []
    tool_calls = []
    observations = []
    parse_errors = 0
    schema_errors = 0
    valid_tool_calls = 0
    successful_tool_calls = 0
    final_answer = None

    for step_id in range(task.max_steps):
        stage = infer_next_stage(last_event, observations)
        route = route_fn(stage, last_event, messages)
        raw_output = generate_fn(messages)
        parsed = parse_agent_output(raw_output)
        active_ratio = float(route.active_ffn_ratio)

        trace_row = {
            "step_id": step_id,
            "stage": stage,
            "event_before": last_event,
            "selected_stage": route.selected_stage,
            "reason": route.reason,
            "active_ffn_ratio": active_ratio,
            "router_prefill_cost": route.router_prefill_cost,
            "raw_output": raw_output,
            "route_metadata": route.metadata,
        }

        if not parsed.ok:
            parse_errors += int(parsed.event == "format_error")
            schema_errors += int(parsed.event == "schema_error")
            observation = _parse_error_observation(parsed)
            observations.append(observation)
            routing_trace.append({**trace_row, "event_after": parsed.event})
            messages.append({"role": "assistant", "content": raw_output})
            messages.append({"role": "user", "content": _observation_prompt(observation)})
            last_event = parsed.event
            continue

        action = parsed.action
        if action is None:
            raise RuntimeError("parse result ok without action")

        if action.type == "final":
            final_answer = action.answer
            success = env.check_success(final_answer)
            routing_trace.append({**trace_row, "event_after": "final"})
            return _episode_payload(
                task=task,
                success=success,
                terminal_event="final",
                final_answer=final_answer,
                messages=messages + [{"role": "assistant", "content": raw_output}],
                routing_trace=routing_trace,
                tool_calls=tool_calls,
                observations=observations,
                parse_errors=parse_errors,
                schema_errors=schema_errors,
                valid_tool_calls=valid_tool_calls,
                successful_tool_calls=successful_tool_calls,
            )

        tool_name = action.name or ""
        arguments = action.arguments or {}
        tool_result = env.execute(tool_name, arguments)
        valid_tool_calls += int(tool_result.event not in {"unknown_tool", "schema_error"})
        schema_errors += int(tool_result.event == "schema_error")
        successful_tool_calls += int(tool_result.ok)
        tool_call = {
            "name": tool_name,
            "arguments": arguments,
            "expected_tool": task.expected_tool,
            "tool_selection_correct": tool_name == task.expected_tool,
            "argument_exact_match": arguments == task.expected_arguments,
        }
        observation = {
            "tool": tool_name,
            **tool_result.to_dict(),
        }
        tool_calls.append(tool_call)
        observations.append(observation)
        routing_trace.append({**trace_row, "event_after": tool_result.event})
        messages.append({"role": "assistant", "content": raw_output})
        messages.append({"role": "user", "content": _observation_prompt(observation)})
        last_event = tool_result.event

    return _episode_payload(
        task=task,
        success=False,
        terminal_event="max_steps_exceeded",
        final_answer=final_answer,
        messages=messages,
        routing_trace=routing_trace,
        tool_calls=tool_calls,
        observations=observations,
        parse_errors=parse_errors,
        schema_errors=schema_errors,
        valid_tool_calls=valid_tool_calls,
        successful_tool_calls=successful_tool_calls,
    )


def infer_next_stage(last_event: str, observations: list[dict[str, Any]]) -> str:
    if last_event == "start":
        return "plan"
    if last_event in {
        "format_error",
        "schema_error",
        "invalid_argument",
        "unknown_tool",
        "timeout",
        "empty_observation",
        "tool_error",
    }:
        return "reflect"
    if observations and observations[-1].get("ok"):
        return "answer"
    return "observe"


def summarize_real_tool_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    successes = sum(1 for row in rows if row["success"])
    failure_rows = [row for row in rows if row["failure_task"]]
    non_failure_rows = [row for row in rows if not row["failure_task"]]
    generation_steps = sum(row["generation_steps"] for row in rows)
    active_ffn_cost = sum(float(row["task_active_ffn_cost"]) for row in rows)
    router_prefill_cost = sum(float(row.get("router_prefill_cost", 0.0)) for row in rows)
    inclusive_cost = active_ffn_cost + router_prefill_cost
    tool_calls = sum(row["tool_call_count"] for row in rows)
    valid_tool_calls = sum(row["valid_tool_call_count"] for row in rows)
    successful_tool_calls = sum(row["successful_tool_call_count"] for row in rows)
    schema_errors = sum(row["schema_error_count"] for row in rows)
    parse_errors = sum(row["parse_error_count"] for row in rows)
    dense_fallback_steps = sum(row["dense_fallback_step_count"] for row in rows)
    collapse = sum(1 for row in rows if row["collapse"])

    by_tool: dict[str, Counter] = {}
    for row in rows:
        tool = row["tool"]
        by_tool.setdefault(tool, Counter())
        by_tool[tool]["total"] += 1
        by_tool[tool]["correct"] += int(row["success"])

    return {
        "total": total,
        "correct": successes,
        "task_success_rate": successes / total if total else 0.0,
        "generation_collapse_rate": collapse / total if total else 0.0,
        "failure_task_total": len(failure_rows),
        "failure_task_correct": sum(1 for row in failure_rows if row["success"]),
        "failure_task_success_rate": _success_rate(failure_rows),
        "non_failure_task_total": len(non_failure_rows),
        "non_failure_task_correct": sum(1 for row in non_failure_rows if row["success"]),
        "non_failure_task_success_rate": _success_rate(non_failure_rows),
        "recovery_success_rate": _success_rate(failure_rows),
        "generation_steps": generation_steps,
        "average_steps": generation_steps / total if total else 0.0,
        "tool_call_count": tool_calls,
        "tool_call_validity": valid_tool_calls / tool_calls if tool_calls else None,
        "tool_execution_success_rate": (
            successful_tool_calls / tool_calls if tool_calls else None
        ),
        "schema_error_count": schema_errors,
        "parse_error_count": parse_errors,
        "schema_validity_rate": (
            1.0 - ((schema_errors + parse_errors) / generation_steps)
            if generation_steps
            else None
        ),
        "active_ffn_cost": active_ffn_cost,
        "average_active_ffn_ratio": (
            active_ffn_cost / generation_steps if generation_steps else None
        ),
        "cost_per_success": active_ffn_cost / successes if successes else None,
        "router_prefill_cost": router_prefill_cost,
        "router_inclusive_cost_per_success": (
            inclusive_cost / successes if successes else None
        ),
        "dense_fallback_step_count": dense_fallback_steps,
        "dense_fallback_rate": dense_fallback_steps / generation_steps if generation_steps else 0.0,
        "by_tool": {
            tool: {
                "total": counts["total"],
                "correct": counts["correct"],
                "accuracy": counts["correct"] / counts["total"],
            }
            for tool, counts in sorted(by_tool.items())
        },
    }


def _success_rate(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(1 for row in rows if row["success"]) / len(rows)


def _parse_error_observation(parsed: ParseResult) -> dict[str, Any]:
    return {
        "tool": None,
        "ok": False,
        "output": {"error": parsed.error or parsed.event},
        "event": parsed.event,
        "retryable": parsed.retryable,
    }


def _observation_prompt(observation: dict[str, Any]) -> str:
    return "Tool observation:\n" + tool_observation_message(observation)


def _episode_payload(
    *,
    task: RealToolTask,
    success: bool,
    terminal_event: str,
    final_answer: str | None,
    messages: list[dict[str, str]],
    routing_trace: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    parse_errors: int,
    schema_errors: int,
    valid_tool_calls: int,
    successful_tool_calls: int,
) -> dict[str, Any]:
    active_ratios = [float(row["active_ffn_ratio"]) for row in routing_trace]
    task_active_ffn_cost = sum(active_ratios)
    router_prefill_cost = sum(float(row.get("router_prefill_cost", 0.0)) for row in routing_trace)
    dense_fallback_steps = sum(
        1 for row in routing_trace if row.get("selected_stage") == "dense_fallback"
    )
    return {
        "task_id": task.task_id,
        "tool": task.tool,
        "expected": task.expected_answer,
        "final_answer": final_answer,
        "success": success,
        "collapse": any(is_generation_collapse(row.get("raw_output", "")) for row in routing_trace),
        "terminal_event": terminal_event,
        "failure_task": task.has_injected_failure,
        "failure_events": [fault.event for fault in task.fault_schedule],
        "messages": messages,
        "tool_calls": tool_calls,
        "observations": observations,
        "routing_trace": routing_trace,
        "events": [observation["event"] for observation in observations],
        "generation_steps": len(routing_trace),
        "active_ffn_ratio": (
            task_active_ffn_cost / len(active_ratios) if active_ratios else 0.0
        ),
        "task_active_ffn_cost": task_active_ffn_cost,
        "router_prefill_cost": router_prefill_cost,
        "tool_call_count": len(tool_calls),
        "valid_tool_call_count": valid_tool_calls,
        "successful_tool_call_count": successful_tool_calls,
        "parse_error_count": parse_errors,
        "schema_error_count": schema_errors,
        "dense_fallback_step_count": dense_fallback_steps,
    }
