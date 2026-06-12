from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safeprune.config import load_config
from safeprune.data import read_jsonl
from safeprune.masks import attach_qwen_switchable_mlp_masks
from safeprune.modeling import load_causal_lm_and_tokenizer
from safeprune.scoring import load_plan
from safeprune.stage_masks import active_mlp_ratio_from_plan


def _load_prompts(args) -> list[str]:
    if args.prompts_jsonl:
        return _load_prompt_rows(args.prompts_jsonl, args.max_prompts)
    if args.text_file:
        lines = [
            line.strip()
            for line in Path(args.text_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return lines[: args.max_prompts] if args.max_prompts else lines
    if args.dataset:
        from datasets import load_dataset

        dataset = load_dataset(args.dataset, args.dataset_name, split=args.split)
        prompts = [
            row[args.text_field]
            for row in dataset
            if isinstance(row.get(args.text_field), str) and row[args.text_field].strip()
        ]
        return prompts[: args.max_prompts] if args.max_prompts else prompts
    raise ValueError("Provide --prompts-jsonl, --text-file, or --dataset.")


def _load_prompt_rows(path: str | Path, limit: int | None = None) -> list[str]:
    prompts = []
    for row in read_jsonl(path):
        for key in ["prompt", "user_request", "text"]:
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                prompts.append(value)
                break
        else:
            steps = row.get("steps")
            if isinstance(steps, list):
                texts = [
                    step.get("text", "")
                    for step in steps
                    if isinstance(step, dict) and isinstance(step.get("text"), str)
                ]
                if texts:
                    prompts.append("\n".join(texts))
        if limit is not None and len(prompts) >= limit:
            break
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def _average_loss(model, tokenizer, prompts: list[str], max_length: int) -> float:
    import torch

    losses = []
    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            device = next(model.parameters()).device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded, labels=encoded["input_ids"])
            losses.append(float(outputs.loss.detach().float().cpu()))
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate mask-hook pruning PPL sanity.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan")
    parser.add_argument("--prompts-jsonl")
    parser.add_argument("--text-file")
    parser.add_argument("--dataset")
    parser.add_argument("--dataset-name")
    parser.add_argument("--split", default="test")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-prompts", type=int, default=256)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()

    config = load_config(args.config)
    prompts = _load_prompts(args)
    model, tokenizer = load_causal_lm_and_tokenizer(
        config.model.base_model,
        config.model,
        local_files_only=args.local_files_only,
    )
    max_length = args.max_length or config.data.max_length
    plan = load_plan(args.plan) if args.plan else None
    handle = None
    try:
        if plan is not None:
            handle = attach_qwen_switchable_mlp_masks(model, initial_plan=plan)
        loss = _average_loss(model, tokenizer, prompts, max_length)
    finally:
        if handle is not None:
            handle.remove()

    result = {
        "plan": args.plan,
        "prompt_count": len(prompts),
        "loss": loss,
        "ppl": math.exp(loss),
        "active_mlp_ratio": active_mlp_ratio_from_plan(plan) if plan else 1.0,
        "note": "Mask-hook PPL sanity; not wall-clock speedup evidence.",
    }
    payload = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
