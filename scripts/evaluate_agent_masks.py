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
]


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


def _build_method_plans(bank: StageMaskBank) -> dict[str, dict[str, Any] | None]:
    observe_001 = bank.select("observe", 0.01)
    return {
        "dense": None,
        "identity_hook": _empty_plan_like(observe_001),
        "full_stage_observe_0.01": _full_stage_observe_plan(bank),
        "safe13_observe_0.03": _safe13_observe_plan(bank),
        "global_spread_observe_approx_0.01": bank.compose_layerwise_plan(
            "observe",
            _global_spread_allocation(),
        ),
        "global_balanced_observe_approx_0.01": bank.compose_layerwise_plan(
            "observe",
            _global_balanced_allocation(),
        ),
        "global_concentrated_observe_approx_0.01": bank.compose_layerwise_plan(
            "observe",
            _global_concentrated_allocation(),
        ),
    }


def _plan_manifest(method_plans: dict[str, dict[str, Any] | None]) -> dict[str, dict[str, Any]]:
    manifest = {}
    for method, plan in method_plans.items():
        if plan is None:
            manifest[method] = {
                "active_ratio": 1.0,
                "allocation": None,
                "pruned_channels": 0,
                "note": "No hook is attached for this method.",
            }
            continue
        manifest[method] = {
            "active_ratio": active_mlp_ratio_from_plan(plan),
            "allocation": plan.get("allocation"),
            "pruned_channels": sum(
                len(layer.get("pruned_mlp_channels", []))
                for layer in plan["layers"]
            ),
        }
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
        "collapse_count": collapse,
        "collapse_rate": collapse / total if total else 0.0,
        "average_latency_ms": (
            sum(float(row["latency_ms"]) for row in rows) / total if total else 0.0
        ),
        "active_ffn_ratio": rows[0]["active_ffn_ratio"] if rows else None,
        "by_tool": {
            tool: {
                "total": values["total"],
                "correct": values["correct"],
                "accuracy": values["correct"] / values["total"],
            }
            for tool, values in sorted(by_tool.items())
        },
    }


def _evaluate_methods(
    tasks: list[AgentTrajectory],
    tokenizer,
    model,
    method_plans: dict[str, dict[str, Any] | None],
    methods: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    metrics = {}
    mask_handle = None

    try:
        for method in methods:
            print(f"\nMETHOD {method}", flush=True)
            plan = method_plans[method]
            if plan is None:
                if mask_handle is not None:
                    mask_handle.remove()
                    mask_handle = None
                active_ratio = 1.0
            else:
                if mask_handle is None:
                    mask_handle = attach_qwen_switchable_mlp_masks(model, initial_plan=plan)
                else:
                    mask_handle.set_plan(plan)
                active_ratio = mask_handle.active_mlp_ratio()

            method_rows = []
            correct = 0
            for idx, task in enumerate(tasks, start=1):
                tool = task.metadata.get("tool") if task.metadata else "unknown"
                prompt = _make_prompt(task)
                start = time.time()
                prediction = _generate(tokenizer, model, prompt)
                latency_ms = (time.time() - start) * 1000.0
                is_correct = strict_agent_answer_correct(tool, task.expected, prediction)
                correct += int(is_correct)

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
    method_plans = _build_method_plans(mask_bank)
    plan_manifest = _plan_manifest({method: method_plans[method] for method in args.methods})
    plans_path.write_text(json.dumps(plan_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote plan manifest to {plans_path}", flush=True)

    print(f"Loading model: {config.model.base_model}", flush=True)
    tokenizer, model = _load_model_and_tokenizer(config, args.local_files_only)
    rows, metrics = _evaluate_methods(tasks, tokenizer, model, method_plans, args.methods)

    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote predictions: {predictions_path}", flush=True)
    print(f"Wrote metrics: {metrics_path}", flush=True)
    print(f"Wrote plans: {plans_path}", flush=True)


if __name__ == "__main__":
    main()
