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

from safeprune.compact import materialize_qwen_compact_mlp_subnet, write_compact_metadata
from safeprune.config import load_config
from safeprune.modeling import load_causal_lm_and_tokenizer
from safeprune.scoring import load_plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize a Qwen FFN intermediate-width compact subnet at runtime."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-name")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--save-state-dict",
        action="store_true",
        help="Also torch.save the compact model state_dict. This can be tens of GB.",
    )
    parser.add_argument("--state-dict-name", default="compact_model_state.pt")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    plan = load_plan(args.plan)
    if args.plan_name:
        plan["plan_name"] = args.plan_name

    model, _tokenizer = load_causal_lm_and_tokenizer(
        config.model.base_model,
        config.model,
        local_files_only=args.local_files_only,
    )
    model.eval()
    metadata = materialize_qwen_compact_mlp_subnet(model, plan)
    metadata.update(
        {
            "config": args.config,
            "base_model": config.model.base_model,
            "plan_path": args.plan,
            "output_dir": str(output_dir),
            "state_dict_saved": bool(args.save_state_dict),
        }
    )

    if args.save_state_dict:
        import torch

        state_path = output_dir / args.state_dict_name
        torch.save(model.state_dict(), state_path)
        metadata["state_dict_path"] = str(state_path)

    metadata_path = output_dir / "compact_metadata.json"
    write_compact_metadata(metadata, metadata_path)
    print(json.dumps(metadata, indent=2))
    print(f"Wrote compact metadata to {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
