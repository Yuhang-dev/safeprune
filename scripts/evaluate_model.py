from __future__ import annotations

import argparse
import json
from pathlib import Path

from safeprune.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    out_dir = Path(config.evaluation.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": args.model,
        "capability_tasks": config.evaluation.capability_tasks,
        "safety_tasks": config.evaluation.safety_tasks,
        "efficiency": config.evaluation.efficiency,
        "message": "Connect lm-eval, HarmBench, AdvBench refusal, and latency scripts here.",
    }
    (out_dir / "eval_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote eval manifest to {out_dir / 'eval_manifest.json'}")


if __name__ == "__main__":
    main()

