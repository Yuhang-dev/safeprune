from __future__ import annotations

import argparse
from pathlib import Path

from safeprune.config import load_config
from safeprune.pruning import ScoreWeights
from safeprune.scoring import build_pruning_plan, load_scores, save_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pruning plans from an existing scores.json file.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--scores")
    parser.add_argument("--sparsity", type=float, action="append", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    config = load_config(args.config)
    scores_path = Path(args.scores) if args.scores else Path(config.pruning.scores_dir) / "scores.json"
    output_dir = Path(args.output_dir) if args.output_dir else Path(config.pruning.scores_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = load_scores(scores_path)

    for sparsity in args.sparsity:
        plan = build_pruning_plan(
            scores=scores,
            sparsity=sparsity,
            weights=ScoreWeights.from_mapping(config.pruning.score_weights),
            min_heads_per_layer=config.pruning.min_heads_per_layer,
            min_mlp_channels_per_layer=config.pruning.min_mlp_channels_per_layer,
            prune_attention_heads=config.pruning.prune_attention_heads,
            prune_mlp_channels=config.pruning.prune_mlp_channels,
        )
        path = output_dir / f"plan_s{sparsity:.2f}.json"
        save_plan(plan, path)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
