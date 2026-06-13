from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safeprune.benchmarking import dump_json, summarize_subnet_theory, write_markdown_table
from safeprune.config import load_config
from safeprune.scoring import load_plan


def _load_model_shape(args):
    if args.hidden_size and args.intermediate_size and args.num_hidden_layers:
        return {
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "num_hidden_layers": args.num_hidden_layers,
            "num_attention_heads": args.num_attention_heads,
            "num_key_value_heads": args.num_key_value_heads,
        }
    from transformers import AutoConfig

    config = load_config(args.config)
    hf_config = AutoConfig.from_pretrained(
        config.model.base_model,
        local_files_only=args.local_files_only,
    )
    return {
        "hidden_size": int(hf_config.hidden_size),
        "intermediate_size": int(hf_config.intermediate_size),
        "num_hidden_layers": int(hf_config.num_hidden_layers),
        "num_attention_heads": int(hf_config.num_attention_heads),
        "num_key_value_heads": int(
            getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads)
        ),
    }


def _parse_specs(args):
    if len(args.plan) != len(args.plan_name):
        raise ValueError("--plan-name must be provided once per --plan.")
    specs = []
    if not args.no_dense:
        specs.append({"name": "dense", "path": None, "plan": None})
    for path, name in zip(args.plan, args.plan_name):
        specs.append({"name": name, "path": path, "plan": load_plan(path)})
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize theoretical FFN channels, parameters, MACs, and FLOPs."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", action="append", default=[])
    parser.add_argument("--plan-name", action="append", default=[])
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--intermediate-size", type=int)
    parser.add_argument("--num-hidden-layers", type=int)
    parser.add_argument("--num-attention-heads", type=int)
    parser.add_argument("--num-key-value-heads", type=int)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    shape = _load_model_shape(args)
    if not shape["num_attention_heads"]:
        raise ValueError("--num-attention-heads is required when shape is passed manually.")
    rows = []
    for spec in _parse_specs(args):
        summary = summarize_subnet_theory(
            plan=spec["plan"],
            hidden_size=shape["hidden_size"],
            intermediate_size=shape["intermediate_size"],
            num_hidden_layers=shape["num_hidden_layers"],
            num_attention_heads=shape["num_attention_heads"],
            num_key_value_heads=shape["num_key_value_heads"],
            sequence_length=args.sequence_length,
        )
        summary["name"] = spec["name"]
        summary["plan"] = spec["path"]
        rows.append(summary)

    payload = {"model_shape": shape, "sequence_length": args.sequence_length, "plans": rows}
    dump_json(payload, args.output_json)
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    write_markdown_table(rows, args.output_md)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
