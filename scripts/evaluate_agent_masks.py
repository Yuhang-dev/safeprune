from __future__ import annotations

import argparse
import json
from pathlib import Path

from safeprune.config import load_config
from safeprune.data import load_agent_jsonl
from safeprune.evaluation import TrajectoryRunResult, compute_trajectory_metrics


METHODS = [
    "dense",
    "static_global_mask",
    "stage_mask",
    "failure_redensification",
]


def _dry_run_results() -> list[TrajectoryRunResult]:
    return [
        TrajectoryRunResult(
            task_id="dry_run_zero_success_guard",
            success=False,
            schema_pass=False,
            tool_calls=1,
            tool_errors=1,
            recovered_errors=0,
            active_ffn_ratios=[0.7],
            latency_ms=0.0,
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Agent FFN mask routing experiments.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = Path(config.evaluation.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_count = None
    if Path(config.agent.task_path).exists():
        task_count = len(load_agent_jsonl(config.agent.task_path))

    if args.dry_run:
        metrics = compute_trajectory_metrics(_dry_run_results()).to_dict()
        payload = {
            "status": "planned",
            "config": args.config,
            "task_path": config.agent.task_path,
            "task_count": task_count,
            "methods": METHODS,
            "primary_metric": "cost_per_success",
            "dry_run_metrics_zero_success_guard": metrics,
            "message": "Dry run only; no model generation or timing was executed.",
        }
        path = out_dir / "agent_eval_manifest.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote Agent eval manifest to {path}")
        return

    raise NotImplementedError(
        "Full Agent generation evaluation is remote-only for now. Use --dry-run locally."
    )


if __name__ == "__main__":
    main()
