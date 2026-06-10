from __future__ import annotations

import argparse
import json
from pathlib import Path

from safeprune.config import load_config
from safeprune.modeling import load_causal_lm_and_tokenizer
from safeprune.pruning import ScoreWeights
from safeprune.scoring import (
    build_pruning_plan,
    compute_activation_scores,
    compute_magnitude_scores,
    merge_score_components,
    save_plan,
    save_scores,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-model-load", action="store_true")
    parser.add_argument("--with-activation", action="store_true")
    parser.add_argument("--max-calibration-batches", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    out_dir = Path(config.pruning.scores_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_model_load:
        model, _ = load_causal_lm_and_tokenizer(
            config.model.teacher_model or config.model.base_model,
            config.model,
        )
        tokenizer = None
        scores = compute_magnitude_scores(model)
        if args.with_activation:
            if not config.data.calibration:
                raise ValueError("--with-activation requires data.calibration in the config.")
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                config.model.teacher_model or config.model.base_model,
                trust_remote_code=config.model.trust_remote_code,
                use_fast=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            activation_scores = compute_activation_scores(
                model=model,
                tokenizer=tokenizer,
                calibration_path=config.data.calibration,
                max_length=config.data.max_length,
                max_batches=args.max_calibration_batches,
            )
            scores = merge_score_components(scores, activation_scores=activation_scores)
        save_scores(scores, out_dir / "scores.json")
        for sparsity in config.pruning.target_sparsities:
            plan = build_pruning_plan(
                scores=scores,
                sparsity=sparsity,
                weights=ScoreWeights.from_mapping(config.pruning.score_weights),
                min_heads_per_layer=config.pruning.min_heads_per_layer,
                min_mlp_channels_per_layer=config.pruning.min_mlp_channels_per_layer,
                prune_attention_heads=config.pruning.prune_attention_heads,
                prune_mlp_channels=config.pruning.prune_mlp_channels,
            )
            save_plan(plan, out_dir / f"plan_s{sparsity:.2f}.json")

    manifest = {
        "status": "planned" if args.skip_model_load else "complete",
        "message": "Magnitude scoring is implemented; activation scoring is enabled by --with-activation; loss-delta scores can be merged through scores.json.",
        "score_weights": config.pruning.score_weights,
        "calibration": config.data.calibration,
        "with_activation": args.with_activation,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote scoring manifest to {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
