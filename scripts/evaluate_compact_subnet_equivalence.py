from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from safeprune.compact import (
    collect_prompts_from_jsonl,
    compare_last_token_logits,
    greedy_generations,
    materialize_qwen_compact_mlp_subnet,
    summarize_generation_equivalence,
    summarize_logits_equivalence,
)
from safeprune.config import load_config
from safeprune.masks import attach_qwen_switchable_mlp_masks
from safeprune.modeling import load_causal_lm_and_tokenizer
from safeprune.scoring import load_plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare mask-hook FFN pruning against a compact FFN subnet."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-name")
    parser.add_argument("--prompts-jsonl", required=True)
    parser.add_argument("--max-prompts", type=int, default=8)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    max_length = args.max_length or config.data.max_length
    plan = load_plan(args.plan)
    if args.plan_name:
        plan["plan_name"] = args.plan_name
    prompts = collect_prompts_from_jsonl(args.prompts_jsonl, args.max_prompts)
    model, tokenizer = load_causal_lm_and_tokenizer(
        config.model.base_model,
        config.model,
        local_files_only=args.local_files_only,
    )

    print("Running mask-hook baseline", flush=True)
    handle = attach_qwen_switchable_mlp_masks(model, initial_plan=plan)
    try:
        baseline_logits = compare_last_token_logits(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_length=max_length,
        )
        baseline_generations = None
        if not args.skip_generation:
            baseline_generations = greedy_generations(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_length=max_length,
                max_new_tokens=args.max_new_tokens,
            )
    finally:
        handle.remove()

    print("Materializing compact subnet", flush=True)
    metadata = materialize_qwen_compact_mlp_subnet(model, plan)

    print("Running compact subnet", flush=True)
    compact_logits = compare_last_token_logits(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_length=max_length,
    )
    compact_generations = None
    if not args.skip_generation:
        compact_generations = greedy_generations(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_length=max_length,
            max_new_tokens=args.max_new_tokens,
        )

    result = {
        "format": "safeprune.compact_equivalence.v1",
        "config": args.config,
        "base_model": config.model.base_model,
        "plan_path": args.plan,
        "plan_name": plan.get("plan_name") or plan.get("name"),
        "prompt_count": len(prompts),
        "max_length": max_length,
        "max_new_tokens": 0 if args.skip_generation else args.max_new_tokens,
        "compact_metadata": metadata,
        "logits": summarize_logits_equivalence(baseline_logits, compact_logits),
        "generation": None,
    }
    if baseline_generations is not None and compact_generations is not None:
        result["generation"] = summarize_generation_equivalence(
            baseline_generations,
            compact_generations,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(_summary_for_stdout(result), indent=2))
    print(f"Wrote compact equivalence result to {output_path}", flush=True)


def _summary_for_stdout(result: dict) -> dict:
    generation = result.get("generation")
    return {
        "plan_name": result.get("plan_name"),
        "prompt_count": result.get("prompt_count"),
        "active_mlp_ratio_actual": result["compact_metadata"].get("active_mlp_ratio_actual"),
        "logits_max_abs_diff": result["logits"].get("logits_max_abs_diff"),
        "logits_mean_abs_diff": result["logits"].get("logits_mean_abs_diff"),
        "top1_match_rate": result["logits"].get("top1_match_rate"),
        "generation_exact_match_rate": generation.get("exact_match_rate")
        if generation
        else None,
    }


if __name__ == "__main__":
    main()
