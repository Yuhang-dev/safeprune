from __future__ import annotations

import argparse
import json
from pathlib import Path

from safeprune.config import load_config
from safeprune.masks import attach_qwen_forward_masks
from safeprune.modeling import load_causal_lm_and_tokenizer
from safeprune.pruning import ScoreWeights
from safeprune.scoring import load_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--plan")
    parser.add_argument("--skip-model-load", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    out_dir = Path(config.pruning.pruned_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = Path(args.plan) if args.plan else Path(config.pruning.scores_dir) / (
        f"plan_s{config.pruning.sparsity:.2f}.json"
    )

    if not args.skip_model_load:
        plan = load_plan(plan_path)
        model, tokenizer = load_causal_lm_and_tokenizer(
            config.model.teacher_model or config.model.base_model,
            config.model,
        )
        attach_qwen_forward_masks(model, plan)
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)

    manifest = {
        "status": "planned" if args.skip_model_load else "complete",
        "message": "Forward structured masks are attached during recovery; checkpoint shape remains PEFT-compatible.",
        "sparsity": config.pruning.sparsity,
        "plan": str(plan_path),
        "score_weights": ScoreWeights.from_mapping(config.pruning.score_weights).__dict__,
    }
    (out_dir / "prune_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote prune manifest to {out_dir / 'prune_manifest.json'}")


if __name__ == "__main__":
    main()
