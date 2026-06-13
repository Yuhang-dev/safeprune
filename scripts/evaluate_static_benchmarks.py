from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safeprune.benchmarking import (
    build_benchmark_manifest,
    dump_json,
    evaluate_multiple_choice_examples,
    evaluate_text_ppl,
    example_from_arc_easy,
    example_from_hellaswag,
    example_from_piqa,
)
from safeprune.config import load_config
from safeprune.masks import attach_qwen_switchable_mlp_masks
from safeprune.modeling import load_causal_lm_and_tokenizer
from safeprune.scoring import load_plan
from safeprune.stage_masks import active_mlp_ratio_from_plan


BENCHMARK_LOADERS: dict[str, tuple[str, str | None, str, Callable[[dict[str, Any]], Any]]] = {
    "piqa": ("piqa", None, "validation", example_from_piqa),
    "hellaswag": ("hellaswag", None, "validation", example_from_hellaswag),
    "arc_easy": ("ai2_arc", "ARC-Easy", "validation", example_from_arc_easy),
}


def _load_wikitext_texts(max_samples: int) -> list[str]:
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [
        str(row["text"]).strip()
        for row in dataset
        if isinstance(row.get("text"), str) and row["text"].strip()
    ]
    return texts[:max_samples]


def _load_mc_examples(benchmark: str, max_samples: int):
    from datasets import load_dataset

    dataset_name, subset, split, builder = BENCHMARK_LOADERS[benchmark]
    dataset = load_dataset(dataset_name, subset, split=split)
    examples = []
    for row in dataset:
        examples.append(builder(row))
        if len(examples) >= max_samples:
            break
    return examples


def _parse_plan_specs(args) -> list[dict[str, Any]]:
    if len(args.plan) != len(args.plan_name):
        raise ValueError("--plan-name must be provided once per --plan.")
    specs = []
    if not args.no_dense:
        specs.append({"name": "dense", "path": None, "plan": None})
    for path, name in zip(args.plan, args.plan_name):
        specs.append({"name": name, "path": path, "plan": load_plan(path)})
    return specs


def _to_markdown(rows: list[dict[str, Any]]) -> str:
    benchmark_names = sorted({name for row in rows for name in row["benchmarks"]})
    header = ["Plan", "Active MLP"] + benchmark_names
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|",
    ]
    for row in rows:
        values = [row["name"], f"{row['active_mlp_ratio']:.4f}"]
        for benchmark in benchmark_names:
            result = row["benchmarks"].get(benchmark)
            if result is None:
                values.append("N/A")
            elif result["metric"] == "perplexity":
                values.append(f"{result['ppl']:.4f}")
            else:
                values.append(f"{result['accuracy']:.4f}")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run static P3 benchmarks for dense and substrate plans."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", action="append", default=[])
    parser.add_argument("--plan-name", action="append", default=[])
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["wikitext2", "piqa", "hellaswag", "arc_easy"],
        choices=["wikitext2", "piqa", "hellaswag", "arc_easy"],
    )
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--benchmark-version", default="p3_static_v1")
    parser.add_argument("--calibration-path", action="append", default=[])
    parser.add_argument("--model-revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    model, tokenizer = load_causal_lm_and_tokenizer(
        config.model.base_model,
        config.model,
        local_files_only=args.local_files_only,
    )
    max_length = args.max_length or config.data.max_length
    specs = _parse_plan_specs(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    handle = attach_qwen_switchable_mlp_masks(model)
    results = []
    try:
        for spec in specs:
            handle.set_plan(spec["plan"] if spec["plan"] is not None else {"layers": []})
            model.eval()
            benchmarks: dict[str, Any] = {}
            if "wikitext2" in args.benchmarks:
                texts = _load_wikitext_texts(args.max_samples)
                benchmarks["wikitext2"] = evaluate_text_ppl(
                    model=model,
                    tokenizer=tokenizer,
                    texts=texts,
                    max_length=max_length,
                )
            for benchmark in ["piqa", "hellaswag", "arc_easy"]:
                if benchmark not in args.benchmarks:
                    continue
                examples = _load_mc_examples(benchmark, args.max_samples)
                benchmarks[benchmark] = evaluate_multiple_choice_examples(
                    model=model,
                    tokenizer=tokenizer,
                    examples=examples,
                    max_length=max_length,
                )

            device = str(next(model.parameters()).device)
            manifest = build_benchmark_manifest(
                model_revision=args.model_revision or config.model.base_model,
                plan_path=spec["path"],
                calibration_paths=args.calibration_path,
                benchmark_version=args.benchmark_version,
                seed=args.seed,
                dtype=str(config.model.torch_dtype),
                device=device,
                extra={
                    "benchmarks": args.benchmarks,
                    "max_samples": args.max_samples,
                    "max_length": max_length,
                    "note": "Static plan benchmark; Adaptive-B20 is reported only in Real Tool loop.",
                },
            )
            row = {
                "name": spec["name"],
                "plan": spec["path"],
                "active_mlp_ratio": active_mlp_ratio_from_plan(spec["plan"])
                if spec["plan"]
                else 1.0,
                "manifest": manifest,
                "benchmarks": benchmarks,
            }
            results.append(row)
            dump_json(row, output_dir / f"{spec['name']}_benchmark.json")
    finally:
        handle.remove()

    payload = {"results": results}
    dump_json(payload, output_dir / "static_benchmark_summary.json")
    (output_dir / "static_benchmark_summary.md").write_text(
        _to_markdown(results),
        encoding="utf-8",
    )
    print(f"Wrote static benchmark summary to {output_dir}")


if __name__ == "__main__":
    main()
