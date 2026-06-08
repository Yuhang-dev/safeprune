from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path

from safeprune.config import load_config, load_config_dict


def _write_yaml(path: Path, value: dict) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to write ablation configs.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-configs", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    raw = load_config_dict(args.config)

    methods = [
        ("safeprune_dpo", config.recovery.consistency_weight, config.recovery.safety_replay_weight),
        ("no_consistency", 0.0, config.recovery.safety_replay_weight),
        ("no_safety_replay", config.recovery.consistency_weight, 0.0),
        ("dpo_recovery", 0.0, 0.0),
    ]
    jobs = list(itertools.product(config.pruning.target_sparsities, methods))
    config_dir = Path(config.experiment.output_dir) / "grid_configs"
    for sparsity, method in jobs:
        name, consistency_weight, safety_replay_weight = method
        job_config = copy.deepcopy(raw)
        job_name = f"{name}_s{sparsity:.2f}".replace(".", "p")
        job_config["experiment"]["name"] = job_name
        job_config["pruning"]["sparsity"] = sparsity
        job_config["recovery"]["consistency_weight"] = consistency_weight
        job_config["recovery"]["safety_replay_weight"] = safety_replay_weight
        job_config["recovery"]["replay_ratio"] = 0.0 if safety_replay_weight == 0.0 else config.recovery.replay_ratio
        job_config["recovery"]["output_dir"] = str(Path(config.experiment.output_dir) / "recovered" / job_name)
        job_path = config_dir / f"{job_name}.yaml"
        if args.write_configs and not args.dry_run:
            _write_yaml(job_path, job_config)
        command = (
            "accelerate launch scripts/recover_safeprune.py "
            f"--config {job_path}"
        )
        print(command)
    if not args.write_configs and not args.dry_run:
        raise SystemExit("Pass --write-configs to create the printed config files.")


if __name__ == "__main__":
    main()
