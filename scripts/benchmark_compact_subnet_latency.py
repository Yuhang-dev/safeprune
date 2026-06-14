from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from safeprune.compact import collect_prompts_from_jsonl, materialize_qwen_compact_mlp_subnet
from safeprune.config import load_config
from safeprune.modeling import load_causal_lm_and_tokenizer
from safeprune.scoring import load_plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark dense and compact Qwen FFN subnet latency."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", action="append", default=[])
    parser.add_argument("--plan-name", action="append", default=[])
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument(
        "--variant",
        choices=["all", "dense", "compact"],
        default="all",
        help=(
            "Benchmark all requested variants, only dense, or exactly one compact "
            "plan. Use dense/compact for single-model single-process latency checks."
        ),
    )
    parser.add_argument("--prompts-jsonl", required=True)
    parser.add_argument("--max-prompts", type=int, default=32)
    parser.add_argument("--batch-sizes", default="1")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--decode-new-tokens", type=int, default=32)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--measure-iters", type=int, default=5)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md")
    args = parser.parse_args()

    if len(args.plan) != len(args.plan_name):
        raise ValueError("--plan-name must be provided once per --plan.")
    if args.variant == "dense" and (args.no_dense or args.plan):
        raise ValueError("--variant dense must not be combined with --no-dense or --plan.")
    if args.variant == "compact" and len(args.plan) != 1:
        raise ValueError("--variant compact requires exactly one --plan/--plan-name pair.")
    if args.measure_iters <= 0:
        raise ValueError("--measure-iters must be positive.")
    if args.warmup_iters < 0:
        raise ValueError("--warmup-iters must be non-negative.")

    config = load_config(args.config)
    prompts = collect_prompts_from_jsonl(args.prompts_jsonl, args.max_prompts)
    batch_sizes = [int(item) for item in args.batch_sizes.split(",") if item.strip()]
    if any(size <= 0 for size in batch_sizes):
        raise ValueError("--batch-sizes must contain positive integers.")
    max_length = args.max_length or config.data.max_length

    specs = []
    if args.variant == "dense":
        specs.append({"name": "dense", "path": None, "plan": None})
    elif args.variant == "compact":
        specs.append(
            {
                "name": args.plan_name[0],
                "path": args.plan[0],
                "plan": load_plan(args.plan[0]),
            }
        )
    elif not args.no_dense:
        specs.append({"name": "dense", "path": None, "plan": None})
    if args.variant == "all":
        for path, name in zip(args.plan, args.plan_name, strict=True):
            specs.append({"name": name, "path": path, "plan": load_plan(path)})

    rows = []
    for spec in specs:
        print(f"\nPLAN {spec['name']}", flush=True)
        memory_before_load = _cuda_memory_snapshot()
        model, tokenizer = load_causal_lm_and_tokenizer(
            config.model.base_model,
            config.model,
            local_files_only=args.local_files_only,
        )
        memory_after_load = _cuda_memory_snapshot()
        metadata = None
        if spec["plan"] is not None:
            metadata = materialize_qwen_compact_mlp_subnet(model, spec["plan"])
            gc.collect()
            _empty_cuda_cache()
        memory_after_materialize = _cuda_memory_snapshot()
        model.eval()
        for batch_size in batch_sizes:
            print(f"Benchmarking {spec['name']} batch={batch_size}", flush=True)
            prompt_batch = _make_prompt_batch(prompts, batch_size)
            encoded = tokenizer(
                prompt_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            device = next(model.parameters()).device
            encoded = {key: value.to(device) for key, value in encoded.items()}
            prompt_token_lengths = [
                int(value)
                for value in encoded["attention_mask"].sum(dim=1).detach().cpu().tolist()
            ]
            prefill = _measure_prefill(
                model=model,
                encoded=encoded,
                warmup_iters=args.warmup_iters,
                measure_iters=args.measure_iters,
            )
            generation = _measure_generate(
                model=model,
                encoded=encoded,
                warmup_iters=args.warmup_iters,
                measure_iters=args.measure_iters,
                max_new_tokens=args.decode_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            approximate_decode_ms = max(
                0.0,
                generation["latency_ms_mean"] - prefill["latency_ms_mean"],
            )
            measured_new_tokens = generation.get("new_tokens_mean") or args.decode_new_tokens
            row = {
                "name": spec["name"],
                "plan_path": spec["path"],
                "batch_size": batch_size,
                "padded_prompt_tokens": int(encoded["input_ids"].shape[1]),
                "prompt_tokens": _value_stats(prompt_token_lengths),
                "decode_new_tokens_requested": args.decode_new_tokens,
                "active_mlp_ratio": (
                    metadata["active_mlp_ratio_actual"] if metadata else 1.0
                ),
                "compact_metadata": metadata,
                "memory_before_load": memory_before_load,
                "memory_after_load": memory_after_load,
                "memory_after_materialize": memory_after_materialize,
                "prefill": prefill,
                "generate": generation,
                "approx_decode_latency_ms_per_token": approximate_decode_ms
                / measured_new_tokens
                if measured_new_tokens
                else None,
                "approx_decode_tokens_per_second": (
                    batch_size * measured_new_tokens
                    / (approximate_decode_ms / 1000.0)
                    if approximate_decode_ms > 0
                    else None
                ),
                "peak_memory_bytes": _peak_cuda_memory(),
                "note": (
                    "Generate timing includes prefill; decode latency is approximated "
                    "as (generate - prefill) / decode_new_tokens."
                ),
            }
            rows.append(row)
        del model
        del tokenizer
        gc.collect()
        _empty_cuda_cache()

    payload = {
        "format": "safeprune.compact_latency.v1",
        "config": args.config,
        "base_model": config.model.base_model,
        "variant": args.variant,
        "prompts_jsonl": args.prompts_jsonl,
        "max_prompts": args.max_prompts,
        "max_length": max_length,
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
        "rows": rows,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.output_md:
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_md).write_text(_to_markdown(rows), encoding="utf-8")
    print(json.dumps(_summary(rows), indent=2))
    print(f"Wrote latency JSON to {output_json}", flush=True)


def _make_prompt_batch(prompts: list[str], batch_size: int) -> list[str]:
    if not prompts:
        raise ValueError("At least one prompt is required.")
    return [prompts[idx % len(prompts)] for idx in range(batch_size)]


def _measure_prefill(*, model, encoded, warmup_iters: int, measure_iters: int) -> dict[str, Any]:
    torch = _require_torch()
    with torch.no_grad():
        for _ in range(warmup_iters):
            _sync_cuda()
            _ = model(**encoded)
            _sync_cuda()
        timings = []
        _reset_cuda_peak_memory()
        for _ in range(measure_iters):
            _sync_cuda()
            start = time.perf_counter()
            _ = model(**encoded)
            _sync_cuda()
            timings.append((time.perf_counter() - start) * 1000.0)
    return _latency_stats(timings)


def _measure_generate(
    *,
    model,
    encoded,
    warmup_iters: int,
    measure_iters: int,
    max_new_tokens: int,
    pad_token_id: int | None,
    eos_token_id: int | list[int] | None,
) -> dict[str, Any]:
    torch = _require_torch()
    generate_kwargs = {
        **encoded,
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "pad_token_id": pad_token_id,
    }
    with torch.no_grad():
        for _ in range(warmup_iters):
            _sync_cuda()
            _ = model.generate(**generate_kwargs)
            _sync_cuda()
        timings = []
        new_token_counts = []
        eos_hits = 0
        total_sequences = 0
        _reset_cuda_peak_memory()
        for _ in range(measure_iters):
            _sync_cuda()
            start = time.perf_counter()
            output_ids = model.generate(**generate_kwargs)
            _sync_cuda()
            timings.append((time.perf_counter() - start) * 1000.0)
            counts, hits = _generated_token_counts(
                output_ids=output_ids,
                prompt_width=int(encoded["input_ids"].shape[1]),
                eos_token_id=eos_token_id,
            )
            new_token_counts.extend(counts)
            eos_hits += hits
            total_sequences += len(counts)
    stats = _latency_stats(timings)
    stats.update(
        {
            "new_tokens": _value_stats(new_token_counts),
            "new_tokens_mean": (
                statistics.fmean(new_token_counts) if new_token_counts else 0.0
            ),
            "new_tokens_min": min(new_token_counts) if new_token_counts else 0,
            "new_tokens_max": max(new_token_counts) if new_token_counts else 0,
            "eos_rate": eos_hits / total_sequences if total_sequences else 0.0,
        }
    )
    return stats


def _latency_stats(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "latency_ms_mean": statistics.fmean(values),
        "latency_ms_median": statistics.median(values),
        "latency_ms_p95": _percentile(ordered, 0.95),
        "latency_ms_min": min(values),
        "latency_ms_max": max(values),
        "samples": values,
    }


def _value_stats(values: list[int | float]) -> dict[str, Any]:
    if not values:
        return {"mean": 0.0, "min": 0, "max": 0, "samples": []}
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "samples": values,
    }


def _generated_token_counts(*, output_ids, prompt_width: int, eos_token_id) -> tuple[list[int], int]:
    eos_ids: set[int] = set()
    if isinstance(eos_token_id, int):
        eos_ids.add(eos_token_id)
    elif isinstance(eos_token_id, list):
        eos_ids.update(int(item) for item in eos_token_id)
    generated = output_ids[:, prompt_width:]
    counts: list[int] = []
    eos_hits = 0
    for row in generated.detach().cpu().tolist():
        count = len(row)
        for idx, token_id in enumerate(row):
            if token_id in eos_ids:
                count = idx + 1
                eos_hits += 1
                break
        counts.append(count)
    return counts, eos_hits


def _percentile(ordered: list[float], q: float) -> float:
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": row["name"],
            "batch_size": row["batch_size"],
            "active_mlp_ratio": row["active_mlp_ratio"],
            "prompt_tokens_mean": row["prompt_tokens"]["mean"],
            "new_tokens_mean": row["generate"]["new_tokens_mean"],
            "eos_rate": row["generate"]["eos_rate"],
            "prefill_ms_mean": row["prefill"]["latency_ms_mean"],
            "generate_ms_mean": row["generate"]["latency_ms_mean"],
            "approx_decode_ms_per_token": row["approx_decode_latency_ms_per_token"],
            "approx_decode_tokens_per_second": row["approx_decode_tokens_per_second"],
            "peak_memory_gb": (
                row["peak_memory_bytes"] / (1024**3)
                if row["peak_memory_bytes"] is not None
                else None
            ),
        }
        for row in rows
    ]


def _to_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Name | Batch | Active MLP | Prompt tok mean | New tok mean | EOS rate | Prefill ms | Generate ms | Approx decode ms/token | Approx decode tok/s | Peak GB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        peak = row["peak_memory_bytes"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["name"],
                    str(row["batch_size"]),
                    f"{row['active_mlp_ratio']:.4f}",
                    f"{row['prompt_tokens']['mean']:.1f}",
                    f"{row['generate']['new_tokens_mean']:.1f}",
                    f"{row['generate']['eos_rate']:.3f}",
                    f"{row['prefill']['latency_ms_mean']:.3f}",
                    f"{row['generate']['latency_ms_mean']:.3f}",
                    f"{row['approx_decode_latency_ms_per_token']:.3f}",
                    f"{row['approx_decode_tokens_per_second']:.3f}"
                    if row["approx_decode_tokens_per_second"] is not None
                    else "N/A",
                    f"{peak / (1024**3):.3f}" if peak is not None else "N/A",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _sync_cuda() -> None:
    torch = _require_torch()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_cuda_peak_memory() -> None:
    torch = _require_torch()
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(idx)


def _peak_cuda_memory() -> int | None:
    torch = _require_torch()
    if not torch.cuda.is_available():
        return None
    return max(torch.cuda.max_memory_allocated(idx) for idx in range(torch.cuda.device_count()))


def _cuda_memory_snapshot() -> dict[str, Any] | None:
    torch = _require_torch()
    if not torch.cuda.is_available():
        return None
    devices = []
    for idx in range(torch.cuda.device_count()):
        devices.append(
            {
                "device": idx,
                "allocated_bytes": int(torch.cuda.memory_allocated(idx)),
                "reserved_bytes": int(torch.cuda.memory_reserved(idx)),
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated(idx)),
            }
        )
    return {
        "devices": devices,
        "total_allocated_bytes": sum(item["allocated_bytes"] for item in devices),
        "total_reserved_bytes": sum(item["reserved_bytes"] for item in devices),
        "max_allocated_bytes": max(item["max_allocated_bytes"] for item in devices),
    }


def _empty_cuda_cache() -> None:
    torch = _require_torch()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on remote env
        raise RuntimeError("Torch is required for compact subnet latency benchmark.") from exc
    return torch


if __name__ == "__main__":
    main()
