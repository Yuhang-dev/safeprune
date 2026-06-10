from __future__ import annotations

import argparse
import json
from pathlib import Path

from safeprune.config import load_config
from safeprune.modeling import load_causal_lm_and_tokenizer
from safeprune.pruning import ScoreWeights
from safeprune.scoring import (
    compute_magnitude_scores,
    compute_stage_activation_scores,
    merge_score_components,
)
from safeprune.stage_masks import StageMaskBank, save_stage_mask_bank


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stage-specific FFN mask banks.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-model-load", action="store_true")
    parser.add_argument("--max-calibration-batches", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = Path(config.agent.mask_bank_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": args.config,
        "stages": config.agent.stages,
        "target_sparsities": config.agent.target_sparsities,
        "task_path": config.agent.task_path,
        "calibration_path": config.agent.calibration_path or config.agent.task_path,
        "mask_type": "ffn_channel_forward_mask",
        "note": "Mask-based algorithm validation only; no physical tensor slicing.",
    }

    if args.skip_model_load:
        manifest["status"] = "planned"
        manifest["message"] = "Skipped model load; no mask_bank.json was built."
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Wrote stage mask manifest to {out_dir / 'manifest.json'}")
        return

    model, tokenizer = load_causal_lm_and_tokenizer(config.model.base_model, config.model)
    magnitude_scores = compute_magnitude_scores(model)
    stage_activation = compute_stage_activation_scores(
        model=model,
        tokenizer=tokenizer,
        agent_path=config.agent.calibration_path or config.agent.task_path,
        max_length=config.data.max_length,
        stages=config.agent.stages,
        max_batches_per_stage=args.max_calibration_batches,
    )
    stage_scores = {
        stage: merge_score_components(magnitude_scores, activation_scores=activation_scores)
        for stage, activation_scores in stage_activation.items()
    }
    mask_bank = StageMaskBank.from_stage_scores(
        stage_scores=stage_scores,
        target_sparsities=config.agent.target_sparsities,
        weights=ScoreWeights.from_mapping(config.pruning.score_weights),
        min_mlp_channels_per_layer=config.pruning.min_mlp_channels_per_layer,
    )
    save_stage_mask_bank(mask_bank, out_dir / "mask_bank.json")
    manifest["status"] = "complete"
    manifest["mask_bank"] = str(out_dir / "mask_bank.json")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote stage mask bank to {out_dir / 'mask_bank.json'}")


if __name__ == "__main__":
    main()
