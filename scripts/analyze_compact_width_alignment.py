from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from safeprune.config import load_config
from safeprune.scoring import load_plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit compact FFN remaining-width alignment for hardware-friendly GEMM."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", action="append", default=[])
    parser.add_argument("--plan-name", action="append", default=[])
    parser.add_argument("--alignment", default="8,16,32,64,128,256")
    parser.add_argument("--intermediate-size", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md")
    args = parser.parse_args()

    if len(args.plan) != len(args.plan_name):
        raise ValueError("--plan-name must be provided once per --plan.")
    if not args.plan:
        raise ValueError("At least one --plan is required.")

    config = load_config(args.config)
    intermediate_size = args.intermediate_size or _load_intermediate_size(
        config.model.base_model,
        trust_remote_code=config.model.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    alignment = [int(item) for item in args.alignment.split(",") if item.strip()]
    if any(item <= 0 for item in alignment):
        raise ValueError("--alignment values must be positive integers.")

    plans = []
    for plan_path, plan_name in zip(args.plan, args.plan_name, strict=True):
        plan = load_plan(plan_path)
        plans.append(
            _analyze_plan(
                plan=plan,
                plan_name=plan_name,
                plan_path=plan_path,
                intermediate_size=intermediate_size,
                alignment=alignment,
            )
        )

    payload = {
        "format": "safeprune.compact_width_alignment.v1",
        "config": args.config,
        "base_model": config.model.base_model,
        "intermediate_size": intermediate_size,
        "alignment": alignment,
        "plans": plans,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.output_md:
        Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_md).write_text(_to_markdown(plans, alignment), encoding="utf-8")
    print(json.dumps(_summary(plans), indent=2))
    print(f"Wrote alignment JSON to {output_json}", flush=True)


def _load_intermediate_size(
    model_name_or_path: str,
    *,
    trust_remote_code: bool,
    local_files_only: bool,
) -> int:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    return int(config.intermediate_size)


def _analyze_plan(
    *,
    plan: dict[str, Any],
    plan_name: str,
    plan_path: str,
    intermediate_size: int,
    alignment: list[int],
) -> dict[str, Any]:
    rows = []
    duplicate_layers = []
    out_of_range_layers = []
    for layer_plan in plan.get("layers", []):
        layer_idx = int(layer_plan["layer"])
        pruned = [int(item) for item in layer_plan.get("pruned_mlp_channels", [])]
        duplicates = len(pruned) - len(set(pruned))
        out_of_range = [
            item for item in pruned if item < 0 or item >= intermediate_size
        ]
        if duplicates:
            duplicate_layers.append(layer_idx)
        if out_of_range:
            out_of_range_layers.append(layer_idx)
        remaining = intermediate_size - len(set(item for item in pruned if 0 <= item < intermediate_size))
        rows.append(
            {
                "layer": layer_idx,
                "pruned_channels": len(set(pruned)),
                "remaining_channels": remaining,
                "alignment_remainders": {str(mod): remaining % mod for mod in alignment},
                "aligned": {str(mod): remaining % mod == 0 for mod in alignment},
                "duplicate_pruned_channels": duplicates,
                "out_of_range_pruned_channels": len(out_of_range),
            }
        )
    widths = [row["remaining_channels"] for row in rows]
    non_aligned_counts = {
        str(mod): sum(1 for width in widths if width % mod != 0) for mod in alignment
    }
    return {
        "name": plan_name,
        "plan_path": plan_path,
        "layer_count": len(rows),
        "intermediate_size": intermediate_size,
        "remaining_channels_min": min(widths) if widths else None,
        "remaining_channels_max": max(widths) if widths else None,
        "remaining_channels_unique": sorted(Counter(widths).items()),
        "non_aligned_layer_counts": non_aligned_counts,
        "duplicate_layer_count": len(duplicate_layers),
        "duplicate_layers": duplicate_layers,
        "out_of_range_layer_count": len(out_of_range_layers),
        "out_of_range_layers": out_of_range_layers,
        "layers": rows,
    }


def _summary(plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": plan["name"],
            "min_remaining": plan["remaining_channels_min"],
            "max_remaining": plan["remaining_channels_max"],
            "non_aligned_layer_counts": plan["non_aligned_layer_counts"],
            "duplicate_layer_count": plan["duplicate_layer_count"],
            "out_of_range_layer_count": plan["out_of_range_layer_count"],
        }
        for plan in plans
    ]


def _to_markdown(plans: list[dict[str, Any]], alignment: list[int]) -> str:
    lines = [
        "| Plan | Min width | Max width | Duplicate layers | Out-of-range layers | "
        + " | ".join(f"Non-aligned % {mod}" for mod in alignment)
        + " |",
        "|---|---:|---:|---:|---:|"
        + "|".join("---:" for _ in alignment)
        + "|",
    ]
    for plan in plans:
        lines.append(
            "| "
            + " | ".join(
                [
                    plan["name"],
                    str(plan["remaining_channels_min"]),
                    str(plan["remaining_channels_max"]),
                    str(plan["duplicate_layer_count"]),
                    str(plan["out_of_range_layer_count"]),
                    *[
                        str(plan["non_aligned_layer_counts"][str(mod)])
                        for mod in alignment
                    ],
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Per-Layer Widths")
    for plan in plans:
        lines.append("")
        lines.append(f"### {plan['name']}")
        lines.append("")
        lines.append("| Layer | Remaining channels | Pruned channels | " + " | ".join(f"% {mod}" for mod in alignment) + " |")
        lines.append("|---:|---:|---:|" + "|".join("---:" for _ in alignment) + "|")
        for row in plan["layers"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["layer"]),
                        str(row["remaining_channels"]),
                        str(row["pruned_channels"]),
                        *[
                            str(row["alignment_remainders"][str(mod)])
                            for mod in alignment
                        ],
                    ]
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
