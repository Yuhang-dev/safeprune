from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safeprune.agent_loop import RouteOutcome, run_real_tool_episode, summarize_real_tool_rows
from safeprune.config import load_config
from safeprune.data import AgentStep
from safeprune.hidden_state_router import HiddenStateCentroidRouter
from safeprune.masks import attach_qwen_switchable_mlp_masks
from safeprune.router import FFNMaskRouter
from safeprune.stage_masks import active_mlp_ratio_from_plan, load_stage_mask_bank
from safeprune.tool_env import RealToolTask, load_real_tool_tasks
from safeprune.tools import default_tool_registry

from scripts.evaluate_agent_masks import (
    _build_method_specs,
    _empty_plan_like,
    _load_model_and_tokenizer,
    _plan_manifest,
    _plan_summary,
    _stage_plans,
    _global_balanced_allocation,
)


REAL_TOOL_METHODS = [
    "dense",
    "identity_hook",
    "global_balanced_observe_approx_0.01",
    "stage_global_balanced_approx_0.01",
    "failure_redense_global_balanced_approx_0.01",
    "observe_failure_redense_global_balanced_approx_0.01",
    "stage_reflect_dense_global_balanced_approx_0.01",
    "stage_reflect_observe_global_balanced_approx_0.01",
    "hidden_state_centroid_global_balanced_approx_0.01",
]
DEFAULT_REAL_TOOL_METHODS = [
    "dense",
    "identity_hook",
    "global_balanced_observe_approx_0.01",
    "stage_global_balanced_approx_0.01",
    "failure_redense_global_balanced_approx_0.01",
    "observe_failure_redense_global_balanced_approx_0.01",
    "stage_reflect_dense_global_balanced_approx_0.01",
    "stage_reflect_observe_global_balanced_approx_0.01",
]


@dataclass(frozen=True)
class RoutingPolicy:
    base_mode: str
    reflect_mode: str
    failure_window_mode: str = "normal"


def _artifact_paths(output_prefix: str) -> tuple[Path, Path, Path, Path, Path]:
    return (
        Path(f"{output_prefix}.jsonl"),
        Path(f"{output_prefix}_metrics.json"),
        Path(f"{output_prefix}_plans.json"),
        Path(f"{output_prefix}_pairwise.json"),
        Path(f"{output_prefix}_failures.jsonl"),
    )


def _generate_messages(
    tokenizer,
    model,
    messages: list[dict[str, str]],
    max_new_tokens: int,
) -> str:
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
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()


def _last_hidden_vector(tokenizer, model, messages: list[dict[str, str]]) -> list[float]:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    import torch

    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden = outputs.hidden_states[-1][0, -1].detach().float().cpu()
    return hidden.tolist()


def _build_hidden_centroid_router(
    *,
    tokenizer,
    model,
    tasks: list[RealToolTask],
    max_tasks: int,
) -> HiddenStateCentroidRouter:
    registry = default_tool_registry()
    from safeprune.agent_loop import build_tool_system_prompt

    vectors_by_stage: dict[str, list[list[float]]] = {
        "plan": [],
        "answer": [],
        "reflect": [],
        "observe": [],
    }
    for task in tasks[:max_tasks]:
        base = [
            {"role": "system", "content": build_tool_system_prompt(registry)},
            {"role": "user", "content": task.user_request},
        ]
        expected_call = {
            "type": "tool_call",
            "name": task.expected_tool,
            "arguments": task.expected_arguments,
        }
        ok_result = registry.execute(task.expected_tool, task.expected_arguments).to_dict()
        ok_observation = {
            "tool": task.expected_tool,
            **ok_result,
        }
        error_observation = {
            "tool": task.expected_tool,
            "ok": False,
            "output": {"error": "timeout"},
            "event": "timeout",
            "retryable": True,
        }

        vectors_by_stage["plan"].append(_last_hidden_vector(tokenizer, model, base))
        vectors_by_stage["observe"].append(
            _last_hidden_vector(
                tokenizer,
                model,
                [*base, {"role": "assistant", "content": json.dumps(expected_call)}],
            )
        )
        vectors_by_stage["answer"].append(
            _last_hidden_vector(
                tokenizer,
                model,
                [
                    *base,
                    {"role": "assistant", "content": json.dumps(expected_call)},
                    {"role": "user", "content": "Tool observation:\n" + json.dumps(ok_observation)},
                ],
            )
        )
        vectors_by_stage["reflect"].append(
            _last_hidden_vector(
                tokenizer,
                model,
                [
                    *base,
                    {"role": "assistant", "content": json.dumps(expected_call)},
                    {
                        "role": "user",
                        "content": "Tool observation:\n" + json.dumps(error_observation),
                    },
                ],
            )
        )
    return HiddenStateCentroidRouter.from_vectors(vectors_by_stage)


class _MethodRuntime:
    def __init__(self, model, tokenizer, spec: dict[str, Any], hidden_router=None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.spec = spec
        self.hidden_router = hidden_router
        self.mask_handle = None
        self.failure_router = None

    def close(self) -> None:
        if self.mask_handle is not None:
            self.mask_handle.remove()
            self.mask_handle = None

    def reset(self) -> None:
        self.failure_router = None

    def route(self, stage: str, event: str, messages: list[dict[str, str]]) -> RouteOutcome:
        if self.spec["mode"] == "static":
            plan = self.spec["plan"]
            self._set_plan(plan)
            ratio = 1.0 if plan is None else active_mlp_ratio_from_plan(plan)
            plan_name = self.spec.get("plan_name", "dense" if plan is None else "static")
            return RouteOutcome(
                selected_stage="dense" if plan is None else "static",
                reason="static",
                active_ffn_ratio=ratio,
                plan=plan,
                selected_plan_name=plan_name,
            )

        if self.spec["mode"] == "stage_dynamic":
            stage_plans = self.spec["stage_plans"]
            selected_stage = stage if stage in stage_plans else self.spec["default_stage"]
            plan = stage_plans[selected_stage]
            self._set_plan(plan)
            return RouteOutcome(
                selected_stage=selected_stage,
                reason="stage_default",
                active_ffn_ratio=active_mlp_ratio_from_plan(plan),
                plan=plan,
                selected_plan_name=selected_stage,
            )

        if self.spec["mode"] == "failure_redense_dynamic":
            return self._route_failure_redense(stage, event)

        if self.spec["mode"] == "routing_policy":
            return self._route_policy(stage, event)

        if self.spec["mode"] == "hidden_state_centroid":
            return self._route_hidden_centroid(messages)

        raise ValueError(f"Unsupported method mode: {self.spec['mode']}")

    def _route_failure_redense(self, stage: str, event: str) -> RouteOutcome:
        if self.failure_router is None:
            self.failure_router = FFNMaskRouter(
                stage_sparsities={stage_name: 0.01 for stage_name in self.spec["stage_plans"]},
                default_sparsity=0.01,
                failure_sparsity=0.0,
                failure_events=self.spec["failure_events"],
                recovery_window_steps=self.spec["recovery_window_steps"],
            )
        router_event = None if event == "start" else event
        fallback_remaining_before = self.failure_router._recovery_steps_remaining
        decision = self.failure_router.route(AgentStep(stage=stage, text="", event=router_event))
        if decision.sparsity == 0.0:
            selected_stage = "dense_fallback"
            selected_plan_name = "dense_fallback"
            plan = self.spec["fallback_plan"]
        else:
            selected_stage = (
                stage if stage in self.spec["stage_plans"] else self.spec["default_stage"]
            )
            selected_plan_name = selected_stage
            plan = self.spec["stage_plans"][selected_stage]
        self._set_plan(plan)
        return RouteOutcome(
            selected_stage=selected_stage,
            reason=decision.reason,
            active_ffn_ratio=active_mlp_ratio_from_plan(plan),
            plan=plan,
            selected_plan_name=selected_plan_name,
            metadata={
                "fallback_remaining_before": fallback_remaining_before,
                "fallback_remaining_after": decision.recovery_steps_remaining,
                "recovery_steps_remaining": decision.recovery_steps_remaining,
            },
        )

    def _route_policy(self, stage: str, event: str) -> RouteOutcome:
        policy: RoutingPolicy = self.spec["policy"]
        reason = "policy_default"
        fallback_remaining_before = 0
        fallback_remaining_after = 0

        if policy.failure_window_mode == "dense":
            if self.failure_router is None:
                self.failure_router = FFNMaskRouter(
                    stage_sparsities={stage_name: 0.01 for stage_name in self.spec["stage_plans"]},
                    default_sparsity=0.01,
                    failure_sparsity=0.0,
                    failure_events=self.spec["failure_events"],
                    recovery_window_steps=self.spec["recovery_window_steps"],
                )
            fallback_remaining_before = self.failure_router._recovery_steps_remaining
            router_event = None if event == "start" else event
            decision = self.failure_router.route(AgentStep(stage=stage, text="", event=router_event))
            fallback_remaining_after = decision.recovery_steps_remaining
            reason = decision.reason
            if decision.sparsity == 0.0:
                plan = self.spec["fallback_plan"]
                self._set_plan(plan)
                return RouteOutcome(
                    selected_stage="dense_fallback",
                    selected_plan_name="dense_fallback",
                    reason=reason,
                    active_ffn_ratio=active_mlp_ratio_from_plan(plan),
                    plan=plan,
                    metadata={
                        "policy": policy.__dict__,
                        "fallback_remaining_before": fallback_remaining_before,
                        "fallback_remaining_after": fallback_remaining_after,
                        "recovery_steps_remaining": decision.recovery_steps_remaining,
                    },
                )

        mode = policy.reflect_mode if stage == "reflect" else policy.base_mode
        selected_stage, selected_plan_name, plan = self._resolve_policy_plan(mode, stage)
        self._set_plan(plan)
        return RouteOutcome(
            selected_stage=selected_stage,
            selected_plan_name=selected_plan_name,
            reason=reason,
            active_ffn_ratio=active_mlp_ratio_from_plan(plan),
            plan=plan,
            metadata={
                "policy": policy.__dict__,
                "fallback_remaining_before": fallback_remaining_before,
                "fallback_remaining_after": fallback_remaining_after,
            },
        )

    def _resolve_policy_plan(
        self,
        mode: str,
        stage: str,
    ) -> tuple[str, str, dict[str, Any]]:
        if mode == "dense":
            return "dense_fallback", "reflect_dense", self.spec["fallback_plan"]
        if mode == "observe":
            return "observe", "observe", self.spec["observe_plan"]
        if mode == "stage":
            selected_stage = (
                stage if stage in self.spec["stage_plans"] else self.spec["default_stage"]
            )
            return selected_stage, selected_stage, self.spec["stage_plans"][selected_stage]
        raise ValueError(f"Unsupported routing policy mode: {mode}")

    def _route_hidden_centroid(self, messages: list[dict[str, str]]) -> RouteOutcome:
        if self.hidden_router is None:
            raise ValueError("hidden_state_centroid mode requires a centroid router")
        identity_plan = self.spec["identity_plan"]
        self._set_plan(identity_plan)
        vector = _last_hidden_vector(self.tokenizer, self.model, messages)
        centroid_route = self.hidden_router.route(vector)
        stage_plans = self.spec["stage_plans"]
        selected_stage = (
            centroid_route.stage
            if centroid_route.stage in stage_plans
            else self.spec["default_stage"]
        )
        plan = stage_plans[selected_stage]
        self._set_plan(plan)
        return RouteOutcome(
            selected_stage=selected_stage,
            reason="hidden_state_centroid",
            active_ffn_ratio=active_mlp_ratio_from_plan(plan),
            plan=plan,
            router_prefill_cost=1.0,
            selected_plan_name=selected_stage,
            metadata={"centroid_score": centroid_route.score},
        )

    def _set_plan(self, plan: dict[str, Any] | None) -> None:
        if plan is None:
            self.close()
            return
        if self.mask_handle is None:
            self.mask_handle = attach_qwen_switchable_mlp_masks(self.model, initial_plan=plan)
        else:
            self.mask_handle.set_plan(plan)


def _extend_method_specs(mask_bank, config) -> dict[str, dict[str, Any]]:
    specs = _build_method_specs(mask_bank, config)
    allocation = _global_balanced_allocation()
    balanced_stage_plans = _stage_plans(mask_bank, allocation)
    observe_plan = mask_bank.compose_layerwise_plan("observe", allocation)
    identity_plan = _empty_plan_like(mask_bank.select("observe", 0.01))
    specs["failure_redense_global_balanced_approx_0.01"] = {
        "mode": "routing_policy",
        "policy": RoutingPolicy("stage", "stage", "dense"),
        "stage_plans": balanced_stage_plans,
        "observe_plan": observe_plan,
        "fallback_plan": identity_plan,
        "default_stage": "answer",
        "failure_events": list(config.agent.failure_events),
        "recovery_window_steps": config.agent.recovery_window_steps,
    }
    specs["observe_failure_redense_global_balanced_approx_0.01"] = {
        "mode": "routing_policy",
        "policy": RoutingPolicy("observe", "observe", "dense"),
        "stage_plans": balanced_stage_plans,
        "observe_plan": observe_plan,
        "fallback_plan": identity_plan,
        "default_stage": "answer",
        "failure_events": list(config.agent.failure_events),
        "recovery_window_steps": config.agent.recovery_window_steps,
    }
    specs["stage_reflect_dense_global_balanced_approx_0.01"] = {
        "mode": "routing_policy",
        "policy": RoutingPolicy("stage", "dense", "normal"),
        "stage_plans": balanced_stage_plans,
        "observe_plan": observe_plan,
        "fallback_plan": identity_plan,
        "default_stage": "answer",
        "failure_events": list(config.agent.failure_events),
        "recovery_window_steps": config.agent.recovery_window_steps,
    }
    specs["stage_reflect_observe_global_balanced_approx_0.01"] = {
        "mode": "routing_policy",
        "policy": RoutingPolicy("stage", "observe", "normal"),
        "stage_plans": balanced_stage_plans,
        "observe_plan": observe_plan,
        "fallback_plan": identity_plan,
        "default_stage": "answer",
        "failure_events": list(config.agent.failure_events),
        "recovery_window_steps": config.agent.recovery_window_steps,
    }
    specs["hidden_state_centroid_global_balanced_approx_0.01"] = {
        "mode": "hidden_state_centroid",
        "stage_plans": balanced_stage_plans,
        "identity_plan": identity_plan,
        "default_stage": "answer",
    }
    return specs


def _plans_manifest(specs: dict[str, dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    manifest = {}
    regular_specs = {
        method: specs[method]
        for method in methods
        if specs[method]["mode"] in {"static", "stage_dynamic", "failure_redense_dynamic"}
    }
    manifest.update(_plan_manifest(regular_specs))
    for method in methods:
        spec = specs[method]
        if spec["mode"] == "routing_policy":
            manifest[method] = {
                "mode": "routing_policy",
                "policy": spec["policy"].__dict__,
                "default_stage": spec["default_stage"],
                "observe": _plan_summary(spec["observe_plan"]),
                "fallback": _plan_summary(spec["fallback_plan"]),
                "failure_events": spec["failure_events"],
                "recovery_window_steps": spec["recovery_window_steps"],
                "stages": {
                    stage: _plan_summary(plan)
                    for stage, plan in sorted(spec["stage_plans"].items())
                },
            }
            continue
        if spec["mode"] != "hidden_state_centroid":
            continue
        manifest[method] = {
            "mode": "hidden_state_centroid",
            "default_stage": spec["default_stage"],
            "router_prefill_cost": "1.0 per generation step, counted separately.",
            "stages": {
                stage: {
                    "active_ratio": active_mlp_ratio_from_plan(plan),
                    "allocation": plan.get("allocation"),
                }
                for stage, plan in sorted(spec["stage_plans"].items())
            },
        }
    return manifest


def _build_pairwise(rows: list[dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    by_method = {
        method: {
            row["task_id"]: row
            for row in rows
            if row["method"] == method
        }
        for method in methods
    }
    pairwise: dict[str, Any] = {}
    for left, right in combinations(methods, 2):
        common_task_ids = sorted(set(by_method[left]) & set(by_method[right]))
        pairwise[f"{left}_vs_{right}"] = {
            "all": _pairwise_counts(by_method[left], by_method[right], common_task_ids),
            "failure": _pairwise_counts(
                by_method[left],
                by_method[right],
                [
                    task_id
                    for task_id in common_task_ids
                    if by_method[left][task_id]["failure_task"]
                ],
            ),
            "non_failure": _pairwise_counts(
                by_method[left],
                by_method[right],
                [
                    task_id
                    for task_id in common_task_ids
                    if not by_method[left][task_id]["failure_task"]
                ],
            ),
        }
    return pairwise


def _pairwise_counts(
    left_rows: dict[str, dict[str, Any]],
    right_rows: dict[str, dict[str, Any]],
    task_ids: list[str],
) -> dict[str, int]:
    counts = {
        "total": 0,
        "both_correct": 0,
        "left_only_correct": 0,
        "right_only_correct": 0,
        "both_wrong": 0,
    }
    for task_id in task_ids:
        left_ok = bool(left_rows[task_id]["success"])
        right_ok = bool(right_rows[task_id]["success"])
        counts["total"] += 1
        if left_ok and right_ok:
            counts["both_correct"] += 1
        elif left_ok:
            counts["left_only_correct"] += 1
        elif right_ok:
            counts["right_only_correct"] += 1
        else:
            counts["both_wrong"] += 1
    return counts


def _evaluate_methods(
    *,
    tasks: list[RealToolTask],
    tokenizer,
    model,
    specs: dict[str, dict[str, Any]],
    methods: list[str],
    max_new_tokens: int,
    hidden_router: HiddenStateCentroidRouter | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    registry = default_tool_registry()
    all_rows = []
    metrics = {}
    for method in methods:
        print(f"\nMETHOD {method}", flush=True)
        runtime = _MethodRuntime(
            model=model,
            tokenizer=tokenizer,
            spec=specs[method],
            hidden_router=hidden_router
            if specs[method]["mode"] == "hidden_state_centroid"
            else None,
        )
        method_rows = []
        try:
            for idx, task in enumerate(tasks, start=1):
                runtime.reset()
                row = run_real_tool_episode(
                    task=task,
                    registry=registry,
                    route_fn=runtime.route,
                    generate_fn=lambda messages: _generate_messages(
                        tokenizer,
                        model,
                        messages,
                        max_new_tokens=max_new_tokens,
                    ),
                )
                row["method"] = method
                method_rows.append(row)
                all_rows.append(row)
                if idx % 10 == 0:
                    correct = sum(1 for item in method_rows if item["success"])
                    print(f"{method}: {idx}/{len(tasks)} done, success={correct}", flush=True)
        finally:
            runtime.close()
        metrics[method] = summarize_real_tool_rows(method_rows)
        print("RESULT", method, json.dumps(metrics[method], ensure_ascii=False), flush=True)
    return all_rows, metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate real tool-execution Agent Prune loop.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", default="data/agent/real_tool_tasks_v1.jsonl")
    parser.add_argument("--mask-bank-path")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=REAL_TOOL_METHODS,
        default=DEFAULT_REAL_TOOL_METHODS,
    )
    parser.add_argument("--failure-only", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-prefix")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-centroid-calibration-tasks", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    tasks = load_real_tool_tasks(args.tasks)
    if args.failure_only:
        tasks = [task for task in tasks if task.has_injected_failure]
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]

    mask_bank_path = Path(
        args.mask_bank_path or Path(config.agent.mask_bank_dir) / "mask_bank.json"
    )
    output_prefix = args.output_prefix or str(
        Path("outputs")
        / "agent_qwen2_5_3b_4090_low"
        / "real_tool_eval"
        / "real_tool_v1"
    )
    (
        predictions_path,
        metrics_path,
        plans_path,
        pairwise_path,
        failures_path,
    ) = _artifact_paths(output_prefix)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading mask bank: {mask_bank_path}", flush=True)
    mask_bank = load_stage_mask_bank(mask_bank_path)
    specs = _extend_method_specs(mask_bank, config)
    plans_path.write_text(
        json.dumps(_plans_manifest(specs, args.methods), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote plan manifest to {plans_path}", flush=True)

    print(f"Loading model: {config.model.base_model}", flush=True)
    tokenizer, model = _load_model_and_tokenizer(config, args.local_files_only)

    hidden_router = None
    if any(specs[method]["mode"] == "hidden_state_centroid" for method in args.methods):
        print("Building hidden-state centroid router", flush=True)
        hidden_router = _build_hidden_centroid_router(
            tokenizer=tokenizer,
            model=model,
            tasks=tasks,
            max_tasks=args.max_centroid_calibration_tasks,
        )

    rows, metrics = _evaluate_methods(
        tasks=tasks,
        tokenizer=tokenizer,
        model=model,
        specs=specs,
        methods=args.methods,
        max_new_tokens=args.max_new_tokens,
        hidden_router=hidden_router,
    )

    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    pairwise_path.write_text(
        json.dumps(_build_pairwise(rows, args.methods), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with failures_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if not row["success"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote predictions: {predictions_path}", flush=True)
    print(f"Wrote metrics: {metrics_path}", flush=True)
    print(f"Wrote plans: {plans_path}", flush=True)
    print(f"Wrote pairwise: {pairwise_path}", flush=True)
    print(f"Wrote failures: {failures_path}", flush=True)


if __name__ == "__main__":
    main()
