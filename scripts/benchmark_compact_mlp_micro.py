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

from safeprune.compact import materialize_qwen_compact_mlp_subnet
from safeprune.config import load_config
from safeprune.modeling import load_causal_lm_and_tokenizer
from safeprune.scoring import load_plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Microbenchmark dense and compact Qwen MLP modules."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", action="append", default=[])
    parser.add_argument("--plan-name", action="append", default=[])
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--variant", choices=["all", "dense", "compact"], default="all")
    parser.add_argument("--layers", default="0,14,27")
    parser.add_argument("--shapes", default="1x1,1x128,4x128")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--measure-iters", type=int, default=50)
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

    config = load_config(args.config)
    layers = _parse_layers(args.layers)
    shapes = _parse_shapes(args.shapes)

    specs = []
    if args.variant == "dense":
        specs.append({"name": "dense", "path": None, "plan": None})
    elif args.variant == "compact":
        specs.append(
            {"name": args.plan_name[0], "path": args.plan[0], "plan": load_plan(args.plan[0])}
        )
    elif not args.no_dense:
        specs.append({"name": "dense", "path": None, "plan": None})
    if args.variant == "all":
        for plan_path, plan_name in zip(args.plan, args.plan_name, strict=True):
            specs.append({"name": plan_name, "path": plan_path, "plan": load_plan(plan_path)})

    rows = []
    for spec in specs:
        print(f"\nPLAN {spec['name']}", flush=True)
        model, tokenizer = load_causal_lm_and_tokenizer(
            config.model.base_model,
            config.model,
            local_files_only=args.local_files_only,
        )
        del tokenizer
        metadata = None
        if spec["plan"] is not None:
            metadata = materialize_qwen_compact_mlp_subnet(model, spec["plan"])
        model.eval()
        decoder_layers = _get_decoder_layers(model)
        for layer_idx in layers:
            mlp = decoder_layers[layer_idx].mlp
            width = int(mlp.gate_proj.weight.shape[0])
            hidden_size = int(mlp.gate_proj.weight.shape[1])
            device = mlp.gate_proj.weight.device
            dtype = mlp.gate_proj.weight.dtype
            for batch_size, seq_len in shapes:
                x = _require_torch().randn(
                    batch_size,
                    seq_len,
                    hidden_size,
                    device=device,
                    dtype=dtype,
                )
                timings = _measure_module(
                    module=mlp,
                    x=x,
                    warmup_iters=args.warmup_iters,
                    measure_iters=args.measure_iters,
                )
                row = {
                    "name": spec["name"],
                    "plan_path": spec["path"],
                    "active_mlp_ratio": metadata["active_mlp_ratio_actual"] if metadata else 1.0,
                    "layer": layer_idx,
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "hidden_size": hidden_size,
                    "intermediate_width": width,
                    "latency": timings,
                }
                rows.append(row)
        del model
        gc.collect()
        _empty_cuda_cache()

    payload = {
        "format": "safeprune.compact_mlp_microbenchmark.v1",
        "config": args.config,
        "base_model": config.model.base_model,
        "layers": layers,
        "shapes": shapes,
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
        "rows": rows,
        "summary": _summary(rows),
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.output_md:
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_md).write_text(_to_markdown(rows), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote MLP microbenchmark JSON to {output_json}", flush=True)


def _parse_layers(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _parse_shapes(value: str) -> list[tuple[int, int]]:
    shapes = []
    for item in value.split(","):
        if not item.strip():
            continue
        parts = item.lower().split("x")
        if len(parts) != 2:
            raise ValueError(f"Invalid shape {item!r}; expected BxS, e.g. 1x128.")
        shapes.append((int(parts[0]), int(parts[1])))
    return shapes


def _measure_module(*, module, x, warmup_iters: int, measure_iters: int) -> dict[str, Any]:
    torch = _require_torch()
    with torch.no_grad():
        for _ in range(warmup_iters):
            _sync_cuda()
            _ = module(x)
            _sync_cuda()
        timings = []
        for _ in range(measure_iters):
            _sync_cuda()
            start = time.perf_counter()
            _ = module(x)
            _sync_cuda()
            timings.append((time.perf_counter() - start) * 1000.0)
    ordered = sorted(timings)
    return {
        "latency_ms_mean": statistics.fmean(timings),
        "latency_ms_median": statistics.median(timings),
        "latency_ms_p95": _percentile(ordered, 0.95),
        "latency_ms_min": min(timings),
        "latency_ms_max": max(timings),
        "samples": timings,
    }


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[float]] = {}
    widths: dict[tuple[str, int, int], list[int]] = {}
    for row in rows:
        key = (row["name"], row["batch_size"], row["seq_len"])
        grouped.setdefault(key, []).append(row["latency"]["latency_ms_mean"])
        widths.setdefault(key, []).append(row["intermediate_width"])
    return [
        {
            "name": name,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "layer_count": len(values),
            "latency_ms_mean_across_layers": statistics.fmean(values),
            "intermediate_width_min": min(widths[(name, batch_size, seq_len)]),
            "intermediate_width_max": max(widths[(name, batch_size, seq_len)]),
        }
        for (name, batch_size, seq_len), values in sorted(grouped.items())
    ]


def _to_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Name | Layer | Batch | Seq | Width | MLP ms mean | MLP ms p95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["name"],
                    str(row["layer"]),
                    str(row["batch_size"]),
                    str(row["seq_len"]),
                    str(row["intermediate_width"]),
                    f"{row['latency']['latency_ms_mean']:.4f}",
                    f"{row['latency']['latency_ms_p95']:.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


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


def _get_decoder_layers(model) -> list[Any]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise ValueError("Unsupported model layout: expected model.model.layers or transformer.h")


def _sync_cuda() -> None:
    torch = _require_torch()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _empty_cuda_cache() -> None:
    torch = _require_torch()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on remote env
        raise RuntimeError("Torch is required for compact MLP microbenchmark.") from exc
    return torch


if __name__ == "__main__":
    main()
