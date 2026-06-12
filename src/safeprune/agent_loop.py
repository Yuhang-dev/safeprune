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
    selected_plan_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


GenerateFn = Callable[[list[dict[str, str]]], str]
RouteFn = Callable[[str, str, list[dict[str, str]]], RouteOutcome]

REFLECT_EVENTS = {
    "format_error",
    "schema_error",
    "invalid_argument",
    "unknown_tool",
    "timeout",
    "empty_observation",
    "tool_error",
    "premature_final",
}


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
            "selected_plan_name": route.selected_plan_name or route.selected_stage,
            "reason": route.reason,
            "active_ffn_ratio": active_ratio,
            "router_prefill_cost": route.router_prefill_cost,
            "fallback_remaining_before": route.metadata.get("fallback_remaining_before"),
            "fallback_remaining_after": route.metadata.get("fallback_remaining_after"),
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
            if task.requires_tool_success and not env.has_successful_tool_observation:
                observation = _premature_final_observation()
                observations.append(observation)
                routing_trace.append({**trace_row, "event_after": "premature_final"})
                messages.append({"role": "assistant", "content": raw_output})
                messages.append({"role": "user", "content": _observation_prompt(observation)})
                last_event = "premature_final"
                continue

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
    if last_event in REFLECT_EVENTS:
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
    dense_fallback_tasks = sum(1 for row in rows if row["dense_fallback_step_count"] > 0)
    premature_final_tasks = sum(1 for row in rows if row["premature_final_count"] > 0)
    reflect_rows = [row for row in rows if row["entered_reflect"]]
    reflect_success_rows = [row for row in reflect_rows if row["success"]]
    reflect_expected = sum(row["reflect_expected_count"] for row in rows)
    reflect_actual = sum(row["reflect_actual_count"] for row in rows)
    reflect_redense_steps = sum(
        1
        for row in rows
        for trace in row.get("routing_trace", [])
        if trace.get("stage") == "reflect" and trace.get("selected_stage") == "dense_fallback"
    )
    successful_observation_tasks = sum(
        1 for row in rows if row["has_successful_tool_observation"]
    )
    recovery_steps = [
        row["recovery_steps"]
        for row in reflect_success_rows
        if row.get("recovery_steps") is not None
    ]
    collapse = sum(1 for row in rows if row["collapse"])
    hidden_detection = _summarize_hidden_state_routing(rows)

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
        "routing_probe_cost": router_prefill_cost,
        "routing_probe_latency_ms": hidden_detection["routing_probe_latency_ms"],
        "routing_probe_latency_ms_mean": hidden_detection["routing_probe_latency_ms_mean"],
        "effective_cost_per_success": inclusive_cost / successes if successes else None,
        "reflect_detection_precision": hidden_detection["reflect_detection_precision"],
        "reflect_detection_recall": hidden_detection["reflect_detection_recall"],
        "reflect_detection_f1": hidden_detection["reflect_detection_f1"],
        "raw_reflect_detection_precision": hidden_detection["reflect_detection_precision"],
        "raw_reflect_detection_recall": hidden_detection["reflect_detection_recall"],
        "raw_reflect_detection_f1": hidden_detection["reflect_detection_f1"],
        "false_dense_fallback_rate": hidden_detection["false_dense_fallback_rate"],
        "missed_reflect_rate": hidden_detection["missed_reflect_rate"],
        "hidden_router_step_count": hidden_detection["hidden_router_step_count"],
        "hidden_reflect_predicted_count": hidden_detection["hidden_reflect_predicted_count"],
        "hidden_reflect_true_positive_count": hidden_detection[
            "hidden_reflect_true_positive_count"
        ],
        "hidden_selected_dense_count": hidden_detection["hidden_selected_dense_count"],
        "dense_fallback_step_count": dense_fallback_steps,
        "dense_fallback_task_count": dense_fallback_tasks,
        "dense_fallback_rate": dense_fallback_steps / generation_steps if generation_steps else 0.0,
        "fallback_trigger_rate": dense_fallback_tasks / total if total else 0.0,
        "fallback_step_ratio": dense_fallback_steps / generation_steps if generation_steps else 0.0,
        "premature_final_task_count": premature_final_tasks,
        "premature_final_rate": premature_final_tasks / total if total else 0.0,
        "reflect_expected_count": reflect_expected,
        "reflect_actual_count": reflect_actual,
        "reflect_route_accuracy": reflect_actual / reflect_expected if reflect_expected else None,
        "runner_reflect_transition_accuracy": (
            reflect_actual / reflect_expected if reflect_expected else None
        ),
        "effective_reflect_redense_recall": (
            reflect_redense_steps / reflect_actual if reflect_actual else None
        ),
        "critical_reflect_miss_rate": (
            1.0 - (reflect_redense_steps / reflect_actual) if reflect_actual else None
        ),
        "reflect_entry_rate": len(reflect_rows) / total if total else 0.0,
        "reflect_success_rate": (
            len(reflect_success_rows) / len(reflect_rows) if reflect_rows else None
        ),
        "recovery_steps_mean": _mean(recovery_steps),
        "recovery_steps_p95": _percentile(recovery_steps, 0.95),
        "successful_tool_observation_task_count": successful_observation_tasks,
        "successful_tool_observation_rate": (
            successful_observation_tasks / total if total else 0.0
        ),
        "by_tool": {
            tool: {
                "total": counts["total"],
                "correct": counts["correct"],
                "accuracy": counts["correct"] / counts["total"],
            }
            for tool, counts in sorted(by_tool.items())
        },
    }


def _summarize_hidden_state_routing(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hidden_steps = []
    for row in rows:
        for trace in row.get("routing_trace", []):
            metadata = trace.get("route_metadata") or {}
            if "centroid_predicted_stage" in metadata:
                hidden_steps.append(trace)

    if not hidden_steps:
        return {
            "routing_probe_latency_ms": 0.0,
            "routing_probe_latency_ms_mean": None,
            "reflect_detection_precision": None,
            "reflect_detection_recall": None,
            "reflect_detection_f1": None,
            "false_dense_fallback_rate": None,
            "missed_reflect_rate": None,
            "hidden_router_step_count": 0,
            "hidden_reflect_predicted_count": 0,
            "hidden_reflect_true_positive_count": 0,
            "hidden_selected_dense_count": 0,
        }

    predicted_reflect = 0
    actual_reflect = 0
    true_positive = 0
    selected_dense = 0
    false_dense = 0
    missed_reflect = 0
    non_reflect = 0
    probe_latency_ms = 0.0

    for trace in hidden_steps:
        metadata = trace.get("route_metadata") or {}
        is_predicted_reflect = bool(metadata.get("centroid_reflect_predicted"))
        is_actual_reflect = trace.get("stage") == "reflect"
        is_dense = trace.get("selected_stage") == "dense_fallback"
        probe_latency_ms += float(metadata.get("routing_probe_latency_ms", 0.0))

        predicted_reflect += int(is_predicted_reflect)
        actual_reflect += int(is_actual_reflect)
        non_reflect += int(not is_actual_reflect)
        true_positive += int(is_predicted_reflect and is_actual_reflect)
        selected_dense += int(is_dense)
        false_dense += int(is_dense and not is_actual_reflect)
        missed_reflect += int(is_actual_reflect and not is_dense)

    precision = true_positive / predicted_reflect if predicted_reflect else None
    recall = true_positive / actual_reflect if actual_reflect else None
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )

    return {
        "routing_probe_latency_ms": probe_latency_ms,
        "routing_probe_latency_ms_mean": probe_latency_ms / len(hidden_steps),
        "reflect_detection_precision": precision,
        "reflect_detection_recall": recall,
        "reflect_detection_f1": f1,
        "false_dense_fallback_rate": false_dense / non_reflect if non_reflect else None,
        "missed_reflect_rate": missed_reflect / actual_reflect if actual_reflect else None,
        "hidden_router_step_count": len(hidden_steps),
        "hidden_reflect_predicted_count": predicted_reflect,
        "hidden_reflect_true_positive_count": true_positive,
        "hidden_selected_dense_count": selected_dense,
    }


def _success_rate(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(1 for row in rows if row["success"]) / len(rows)


def _mean(values: list[int]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _percentile(values: list[int], quantile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * quantile))
    return ordered[index]


def _parse_error_observation(parsed: ParseResult) -> dict[str, Any]:
    return {
        "tool": None,
        "ok": False,
        "output": {"error": parsed.error or parsed.event},
        "event": parsed.event,
        "retryable": parsed.retryable,
    }


def _premature_final_observation() -> dict[str, Any]:
    return {
        "tool": "runner_guard",
        "ok": False,
        "output": {
            "error": "A successful tool execution is required before final answer.",
        },
        "event": "premature_final",
        "retryable": True,
    }


def _observation_prompt(observation: dict[str, Any]) -> str:
    return (
        "Tool observation:\n"
        + tool_observation_message(observation)
        + "\n\nNext action rules:\n"
        + "- If ok is true, answer with exactly one final JSON object using the observed result.\n"
        + "- If ok is false and retryable is true, call the needed tool again with valid arguments.\n"
        + "- Output no text outside the JSON object."
    )


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
    premature_final_count = sum(
        1 for observation in observations if observation["event"] == "premature_final"
    )
    reflect_expected_count = sum(
        1 for row in routing_trace if row.get("event_before") in REFLECT_EVENTS
    )
    reflect_actual_count = sum(
        1
        for row in routing_trace
        if row.get("event_before") in REFLECT_EVENTS and row.get("stage") == "reflect"
    )
    reflect_step_ids = [
        int(row["step_id"]) for row in routing_trace if row.get("stage") == "reflect"
    ]
    entered_reflect = bool(reflect_step_ids)
    recovery_steps = None
    if success and entered_reflect and routing_trace:
        recovery_steps = int(routing_trace[-1]["step_id"]) - min(reflect_step_ids) + 1
    successful_tool_observations = [
        observation
        for observation in observations
        if observation.get("ok")
        and observation.get("event") in {"ok", "success"}
        and observation.get("output") not in (None, {})
        and observation.get("counts_as_successful_observation", True)
    ]
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
        "premature_final_count": premature_final_count,
        "reflect_expected_count": reflect_expected_count,
        "reflect_actual_count": reflect_actual_count,
        "entered_reflect": entered_reflect,
        "recovery_steps": recovery_steps,
        "successful_tool_observation_count": len(successful_tool_observations),
        "has_successful_tool_observation": bool(successful_tool_observations),
    }
