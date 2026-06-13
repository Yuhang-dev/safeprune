from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from safeprune.config import load_config
from safeprune.data import AgentTrajectory, load_agent_jsonl
from safeprune.evaluation import (
    TrajectoryRunResult,
    compute_trajectory_metrics,
    is_generation_collapse,
    strict_agent_answer_correct,
)
from safeprune.masks import attach_qwen_switchable_mlp_masks
from safeprune.router import FFNMaskRouter
from safeprune.stage_masks import (
    StageMaskBank,
    active_mlp_ratio_from_plan,
    load_stage_mask_bank,
)


SAFE13 = [3, 6, 7, 8, 9, 10, 14, 15, 18, 23, 28, 30, 34]
NEXT48 = [1, 2, 4, 5, 11, 12, 13, 16, 19, 20, 26, 27, 29, 32, 33, 35]

METHODS = [
    "dense",
    "identity_hook",
    "full_stage_observe_0.01",
    "safe13_observe_0.03",
    "global_spread_observe_approx_0.01",
    "global_balanced_observe_approx_0.01",
    "global_concentrated_observe_approx_0.01",
    "stage_global_balanced_approx_0.01",
    "failure_redense_global_balanced_approx_0.01",
]
AGENT_STAGES = ["plan", "act", "observe", "reflect", "answer"]


def _dry_run_results() -> list[TrajectoryRunResult]:
    return [
        TrajectoryRunResult(
            task_id="dry_run_zero_success_guard",
            success=False,
            schema_pass=False,
            tool_calls=1,
            tool_errors=1,
            recovered_errors=0,
            active_ffn_ratios=[0.7],
            latency_ms=0.0,
        )
    ]


def _empty_plan_like(plan: dict[str, Any]) -> dict[str, Any]:
    import copy

    empty = copy.deepcopy(plan)
    for layer in empty["layers"]:
        layer["pruned_attention_heads"] = []
        layer["pruned_mlp_channels"] = []
        layer.pop("mlp_output_bias_compensation", None)
        layer.pop("mlp_output_scale", None)
    empty["allocation"] = {}
    return empty


def _full_stage_observe_plan(bank: StageMaskBank) -> dict[str, Any]:
    return bank.select("observe", 0.01)


def _safe13_observe_plan(bank: StageMaskBank) -> dict[str, Any]:
    return bank.compose_layerwise_plan("observe", {layer: 0.03 for layer in SAFE13})


def _global_spread_allocation() -> dict[int, float]:
    allocation = {layer: 0.02 for layer in SAFE13}
    allocation.update({layer: 0.01 for layer in NEXT48[:10]})
    return allocation


def _global_balanced_allocation() -> dict[int, float]:
    allocation = {layer: 0.03 for layer in SAFE13[:8]}
    allocation.update({layer: 0.01 for layer in [*SAFE13[8:], *NEXT48[:7]]})
    return allocation


def _global_concentrated_allocation() -> dict[int, float]:
    allocation = {layer: 0.05 for layer in SAFE13[:7]}
    allocation[SAFE13[7]] = 0.01
    return allocation


def _stage_plans(bank: StageMaskBank, allocation: dict[int, float]) -> dict[str, dict[str, Any]]:
    return {
        stage: bank.compose_layerwise_plan(stage, allocation)
        for stage in AGENT_STAGES
        if stage in bank.plans
    }


def _build_method_specs(bank: StageMaskBank, config) -> dict[str, dict[str, Any]]:
    observe_001 = bank.select("observe", 0.01)
    identity = _empty_plan_like(observe_001)
    balanced_stage_plans = _stage_plans(bank, _global_balanced_allocation())
    return {
        "dense": {"mode": "static", "plan": None},
        "identity_hook": {"mode": "static", "plan": identity},
        "full_stage_observe_0.01": {"mode": "static", "plan": _full_stage_observe_plan(bank)},
        "safe13_observe_0.03": {"mode": "static", "plan": _safe13_observe_plan(bank)},
        "global_spread_observe_approx_0.01": {
            "mode": "static",
            "plan": bank.compose_layerwise_plan("observe", _global_spread_allocation()),
        },
        "global_balanced_observe_approx_0.01": {
            "mode": "static",
            "plan": bank.compose_layerwise_plan("observe", _global_balanced_allocation()),
        },
        "global_concentrated_observe_approx_0.01": {
            "mode": "static",
            "plan": bank.compose_layerwise_plan("observe", _global_concentrated_allocation()),
        },
        "stage_global_balanced_approx_0.01": {
            "mode": "stage_dynamic",
            "stage_plans": balanced_stage_plans,
            "default_stage": "answer",
        },
        "failure_redense_global_balanced_approx_0.01": {
            "mode": "failure_redense_dynamic",
            "stage_plans": balanced_stage_plans,
            "fallback_plan": identity,
            "default_stage": "answer",
            "failure_events": list(config.agent.failure_events),
            "recovery_window_steps": config.agent.recovery_window_steps,
        },
    }


def _plan_summary(plan: dict[str, Any] | None) -> dict[str, Any]:
    if plan is None:
        return {
            "active_ratio": 1.0,
            "allocation": None,
            "pruned_channels": 0,
            "note": "No hook is attached for this method.",
        }
    return {
        "active_ratio": active_mlp_ratio_from_plan(plan),
        "allocation": plan.get("allocation"),
        "pruned_channels": sum(
            len(layer.get("pruned_mlp_channels", []))
            for layer in plan["layers"]
        ),
    }


def _plan_manifest(method_specs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    manifest = {}
    for method, spec in method_specs.items():
        if spec["mode"] == "static":
            manifest[method] = {"mode": "static", **_plan_summary(spec["plan"])}
            continue

        stage_plans = spec["stage_plans"]
        manifest[method] = {
            "mode": spec["mode"],
            "default_stage": spec["default_stage"],
            "stages": {
                stage: _plan_summary(plan)
                for stage, plan in sorted(stage_plans.items())
            },
        }
        if spec["mode"] == "failure_redense_dynamic":
            manifest[method]["fallback"] = _plan_summary(spec["fallback_plan"])
            manifest[method]["failure_events"] = spec["failure_events"]
            manifest[method]["recovery_window_steps"] = spec["recovery_window_steps"]
    manifest["_note"] = {
        "latency": "Forward hooks do not physically shrink GEMMs; latency is not speedup evidence."
    }
    return manifest


def _make_prompt(task: AgentTrajectory) -> str:
    tool = task.metadata.get("tool") if task.metadata else "unknown"
    rules = {
        "calculator": "Compute the arithmetic expression exactly.",
        "unit_convert": "Use 1 meter = 100 centimeters.",
        "lookup": "Use this lookup rule exactly: project PRJ-N is owned by owner_N.",
    }
    return (
        "You are solving a controlled tool-use task.\n"
        "Return only the final answer, with no explanation.\n\n"
        f"Tool type: {tool}\n"
        f"Rule: {rules.get(tool, 'Answer exactly.')}\n"
        f"Task: {task.prompt}\n"
        "Final answer:"
    )


def _torch_dtype(name: str):
    import torch

    mapping = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = str(name).lower()
    if key not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name!r}")
    return mapping[key]


def _load_model_and_tokenizer(config, local_files_only: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.base_model,
        trust_remote_code=config.model.trust_remote_code,
        local_files_only=local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model.base_model,
        torch_dtype=_torch_dtype(config.model.torch_dtype),
        device_map="auto",
        trust_remote_code=config.model.trust_remote_code,
        local_files_only=local_files_only,
    )
    model.eval()
    return tokenizer, model


def _generate(tokenizer, model, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    import torch

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][input_len:], skip_special_tokens=True)


def _artifact_paths(output_prefix: str) -> tuple[Path, Path, Path]:
    return (
        Path(f"{output_prefix}.jsonl"),
        Path(f"{output_prefix}_metrics.json"),
        Path(f"{output_prefix}_plans.json"),
    )


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row["strict_correct"])
    collapse = sum(1 for row in rows if row["collapse"])
    active_ratios = [float(row["active_ffn_ratio"]) for row in rows]
    active_ffn_cost = sum(active_ratios)
    failure_rows = [row for row in rows if row.get("failure_task", False)]
    non_failure_rows = [row for row in rows if not row.get("failure_task", False)]
    failure_correct = sum(1 for row in failure_rows if row["strict_correct"])
    non_failure_correct = sum(1 for row in non_failure_rows if row["strict_correct"])
    dense_fallback_count = sum(
        1
        for row in rows
        if any(
            trace.get("selected_stage") == "dense_fallback"
            for trace in row.get("routing_trace", [])
        )
    )
    by_tool: dict[str, dict[str, int]] = {}
    for row in rows:
        tool = row["tool"]
        by_tool.setdefault(tool, {"total": 0, "correct": 0})
        by_tool[tool]["total"] += 1
        by_tool[tool]["correct"] += int(row["strict_correct"])

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "task_success_rate": correct / total if total else 0.0,
        "collapse_count": collapse,
        "collapse_rate": collapse / total if total else 0.0,
        "generation_collapse_rate": collapse / total if total else 0.0,
        "average_latency_ms": (
            sum(float(row["latency_ms"]) for row in rows) / total if total else 0.0
        ),
        "active_ffn_ratio": active_ffn_cost / total if total else None,
        "average_active_ffn_ratio": (
            active_ffn_cost / total if total else None
        ),
        "active_ffn_cost": active_ffn_cost,
        "cost_per_success": active_ffn_cost / correct if correct else None,
        "failure_task_total": len(failure_rows),
        "failure_task_correct": failure_correct,
        "failure_task_success_rate": (
            failure_correct / len(failure_rows) if failure_rows else None
        ),
        "non_failure_task_total": len(non_failure_rows),
        "non_failure_task_correct": non_failure_correct,
        "non_failure_task_success_rate": (
            non_failure_correct / len(non_failure_rows) if non_failure_rows else None
        ),
        "recovery_success_rate": (
            failure_correct / len(failure_rows) if failure_rows else None
        ),
        "dense_fallback_task_count": dense_fallback_count,
        "dense_fallback_task_rate": dense_fallback_count / total if total else 0.0,
        "by_tool": {
            tool: {
                "total": values["total"],
                "correct": values["correct"],
                "accuracy": values["correct"] / values["total"],
            }
            for tool, values in sorted(by_tool.items())
        },
    }


def _stage_plan_for_task(
    task: AgentTrajectory,
    stage_plans: dict[str, dict[str, Any]],
    default_stage: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    trace = []
    selected_stage = default_stage
    ratios = []
    for step in task.steps:
        selected_stage = step.stage if step.stage in stage_plans else default_stage
        plan = stage_plans[selected_stage]
        ratio = active_mlp_ratio_from_plan(plan)
        ratios.append(ratio)
        trace.append(
            {
                "stage": step.stage,
                "event": step.event,
                "selected_stage": selected_stage,
                "reason": "stage_default",
                "active_ffn_ratio": ratio,
            }
        )
    if not trace:
        plan = stage_plans[default_stage]
        ratio = active_mlp_ratio_from_plan(plan)
        return plan, [], ratio
    final_plan = stage_plans[selected_stage]
    return final_plan, trace, sum(ratios) / len(ratios)


def _failure_redense_plan_for_task(
    task: AgentTrajectory,
    stage_plans: dict[str, dict[str, Any]],
    fallback_plan: dict[str, Any],
    default_stage: str,
    failure_events: list[str],
    recovery_window_steps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    router = FFNMaskRouter(
        stage_sparsities={stage: 0.01 for stage in stage_plans},
        default_sparsity=0.01,
        failure_sparsity=0.0,
        failure_events=failure_events,
        recovery_window_steps=recovery_window_steps,
    )
    trace = []
    selected_plan = stage_plans[default_stage]
    ratios = []
    for step in task.steps:
        decision = router.route(step)
        if decision.sparsity == 0.0:
            selected_stage = "dense_fallback"
            selected_plan = fallback_plan
        else:
            selected_stage = step.stage if step.stage in stage_plans else default_stage
            selected_plan = stage_plans[selected_stage]
        ratio = active_mlp_ratio_from_plan(selected_plan)
        ratios.append(ratio)
        trace.append(
            {
                "stage": step.stage,
                "event": step.event,
                "selected_stage": selected_stage,
                "reason": decision.reason,
                "recovery_steps_remaining": decision.recovery_steps_remaining,
                "active_ffn_ratio": ratio,
            }
        )
    if not trace:
        ratio = active_mlp_ratio_from_plan(selected_plan)
        return selected_plan, [], ratio
    return selected_plan, trace, sum(ratios) / len(ratios)


def _resolve_task_plan(
    task: AgentTrajectory,
    spec: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], float]:
    if spec["mode"] == "static":
        plan = spec["plan"]
        ratio = 1.0 if plan is None else active_mlp_ratio_from_plan(plan)
        return plan, [], ratio
    if spec["mode"] == "stage_dynamic":
        return _stage_plan_for_task(task, spec["stage_plans"], spec["default_stage"])
    if spec["mode"] == "failure_redense_dynamic":
        return _failure_redense_plan_for_task(
            task,
            spec["stage_plans"],
            spec["fallback_plan"],
            spec["default_stage"],
            spec["failure_events"],
            spec["recovery_window_steps"],
        )
    raise ValueError(f"Unsupported method mode: {spec['mode']}")


def _failure_events_for_task(task: AgentTrajectory) -> list[str]:
    return [
        step.event or "unknown"
        for step in task.steps
        if step.is_failure
    ]


def _evaluate_methods(
    tasks: list[AgentTrajectory],
    tokenizer,
    model,
    method_specs: dict[str, dict[str, Any]],
    methods: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    metrics = {}
    mask_handle = None

    try:
        for method in methods:
            print(f"\nMETHOD {method}", flush=True)
            spec = method_specs[method]
            method_rows = []
            correct = 0
            for idx, task in enumerate(tasks, start=1):
                plan, routing_trace, active_ratio = _resolve_task_plan(task, spec)
                if plan is None:
                    if mask_handle is not None:
                        mask_handle.remove()
                        mask_handle = None
                elif mask_handle is None:
                    mask_handle = attach_qwen_switchable_mlp_masks(model, initial_plan=plan)
                else:
                    mask_handle.set_plan(plan)

                tool = task.metadata.get("tool") if task.metadata else "unknown"
                prompt = _make_prompt(task)
                start = time.time()
                prediction = _generate(tokenizer, model, prompt)
                latency_ms = (time.time() - start) * 1000.0
                is_correct = strict_agent_answer_correct(tool, task.expected, prediction)
                correct += int(is_correct)
                failure_events = _failure_events_for_task(task)

                row = {
                    "method": method,
                    "task_id": task.task_id,
                    "tool": tool,
                    "prompt": task.prompt,
                    "expected": task.expected,
                    "prediction": prediction,
                    "strict_correct": is_correct,
                    "collapse": is_generation_collapse(prediction),
                    "latency_ms": latency_ms,
                    "active_ffn_ratio": active_ratio,
                    "failure_task": bool(failure_events),
                    "failure_events": failure_events,
                    "routing_trace": routing_trace,
                }
                method_rows.append(row)
                rows.append(row)

                if idx % 10 == 0:
                    print(f"{method}: {idx}/{len(tasks)} done, correct={correct}", flush=True)

            metrics[method] = _summarize_rows(method_rows)
            print(
                "RESULT",
                method,
                json.dumps(metrics[method], ensure_ascii=False),
                flush=True,
            )
    finally:
        if mask_handle is not None:
            mask_handle.remove()

    return rows, metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Agent FFN mask routing experiments.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mask-bank-path")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-prefix")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    out_dir = Path(config.evaluation.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_count = None
    if Path(config.agent.task_path).exists():
        task_count = len(load_agent_jsonl(config.agent.task_path))

    if args.dry_run:
        metrics = compute_trajectory_metrics(_dry_run_results()).to_dict()
        payload = {
            "status": "planned",
            "config": args.config,
            "task_path": config.agent.task_path,
            "task_count": task_count,
            "methods": args.methods,
            "primary_metric": "cost_per_success",
            "dry_run_metrics_zero_success_guard": metrics,
            "message": "Dry run only; no model generation or timing was executed.",
        }
        path = out_dir / "agent_eval_manifest.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote Agent eval manifest to {path}")
        return

    mask_bank_path = Path(
        args.mask_bank_path or Path(config.agent.mask_bank_dir) / "mask_bank.json"
    )
    output_prefix = args.output_prefix or str(out_dir / "agent_mask_eval")
    predictions_path, metrics_path, plans_path = _artifact_paths(output_prefix)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = load_agent_jsonl(config.agent.task_path)
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]

    print(f"Loading mask bank: {mask_bank_path}", flush=True)
    mask_bank = load_stage_mask_bank(mask_bank_path)
    method_specs = _build_method_specs(mask_bank, config)
    plan_manifest = _plan_manifest({method: method_specs[method] for method in args.methods})
    plans_path.write_text(json.dumps(plan_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote plan manifest to {plans_path}", flush=True)

    print(f"Loading model: {config.model.base_model}", flush=True)
    tokenizer, model = _load_model_and_tokenizer(config, args.local_files_only)
    rows, metrics = _evaluate_methods(tasks, tokenizer, model, method_specs, args.methods)

    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote predictions: {predictions_path}", flush=True)
    print(f"Wrote metrics: {metrics_path}", flush=True)
    print(f"Wrote plans: {plans_path}", flush=True)


if __name__ == "__main__":
    main()
