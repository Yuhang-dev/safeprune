from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on remote env
        raise RuntimeError("PyYAML is required to load YAML configs. Install requirements.txt.") from exc

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return loaded


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config_dict(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    config = _load_yaml(path)
    parent = config.get("extends")
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
            if not parent_path.exists():
                parent_path = Path.cwd() / parent
        config = _deep_merge(load_config_dict(parent_path), config)
    return config


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int = 42
    output_dir: str = "outputs/default"


@dataclass(frozen=True)
class ModelConfig:
    base_model: str
    teacher_model: str | None = None
    trust_remote_code: bool = True
    torch_dtype: str = "bfloat16"
    use_flash_attention_2: bool = True


@dataclass(frozen=True)
class DataConfig:
    preference_train: str
    preference_eval: str | None = None
    safety_replay: str | None = None
    calibration: str | None = None
    max_length: int = 2048
    max_prompt_length: int = 1024


@dataclass(frozen=True)
class DPOConfig:
    beta: float = 0.1
    learning_rate: float = 5e-6
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 500


@dataclass(frozen=True)
class PruningConfig:
    sparsity: float = 0.35
    target_sparsities: list[float] = field(default_factory=lambda: [0.25, 0.35, 0.50])
    prune_attention_heads: bool = True
    prune_mlp_channels: bool = True
    score_weights: dict[str, float] = field(
        default_factory=lambda: {"magnitude": 0.4, "activation": 0.35, "loss_delta": 0.25}
    )
    min_heads_per_layer: int = 1
    min_mlp_channels_per_layer: int = 128
    calibration_batch_size: int = 1
    scores_dir: str = "outputs/prune_scores"
    pruned_dir: str = "outputs/pruned"

    def validate(self) -> None:
        if not 0.0 <= self.sparsity < 1.0:
            raise ValueError("pruning.sparsity must be in [0, 1).")
        if any(not 0.0 <= value < 1.0 for value in self.target_sparsities):
            raise ValueError("Every pruning.target_sparsities value must be in [0, 1).")
        missing = {"magnitude", "activation", "loss_delta"} - set(self.score_weights)
        if missing:
            raise ValueError(f"Missing pruning.score_weights: {sorted(missing)}")


@dataclass(frozen=True)
class RecoveryConfig:
    method: str = "safeprune_dpo"
    output_dir: str = "outputs/recovered"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=list)
    learning_rate: float = 1e-5
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    dpo_weight: float = 1.0
    consistency_weight: float = 0.2
    safety_replay_weight: float = 0.3
    replay_ratio: float = 0.2
    teacher_temperature: float = 1.0


@dataclass(frozen=True)
class EvaluationConfig:
    output_dir: str = "outputs/eval"
    capability_tasks: list[str] = field(default_factory=list)
    safety_tasks: list[str] = field(default_factory=list)
    efficiency: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentConfig:
    task_path: str = "data/agent/controlled_tasks.jsonl"
    calibration_path: str | None = None
    mask_bank_dir: str = "outputs/agent/mask_bank"
    stages: list[str] = field(
        default_factory=lambda: ["plan", "act", "observe", "reflect", "answer"]
    )
    target_sparsities: list[float] = field(default_factory=lambda: [0.10, 0.20, 0.30])
    stage_sparsities: dict[str, float] = field(
        default_factory=lambda: {
            "plan": 0.10,
            "act": 0.20,
            "observe": 0.30,
            "reflect": 0.10,
            "answer": 0.30,
        }
    )
    failure_sparsity: float = 0.10
    default_sparsity: float = 0.30
    recovery_window_steps: int = 2
    failure_events: list[str] = field(
        default_factory=lambda: [
            "tool_error",
            "invalid_json",
            "schema_error",
            "timeout",
            "empty_observation",
            "verifier_failure",
        ]
    )
    mask_granularity: int = 1

    def validate(self) -> None:
        if not self.stages:
            raise ValueError("agent.stages must not be empty.")
        invalid = [stage for stage in self.stage_sparsities if stage not in self.stages]
        if invalid:
            raise ValueError(f"agent.stage_sparsities contains unknown stages: {invalid}")
        for value in [*self.target_sparsities, *self.stage_sparsities.values(), self.failure_sparsity]:
            if not 0.0 <= float(value) < 1.0:
                raise ValueError("agent sparsity values must be in [0, 1).")
        if self.recovery_window_steps < 0:
            raise ValueError("agent.recovery_window_steps must be non-negative.")
        if self.mask_granularity < 1:
            raise ValueError("agent.mask_granularity must be positive.")


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool = False
    backend: str = "none"
    project: str = "safeprune-dpo"
    workspace: str | None = None
    experiment_name: str | None = None
    description: str | None = None
    mode: str | None = None
    logdir: str | None = None
    tags: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.backend not in {"none", "swanlab"}:
            raise ValueError("tracking.backend must be one of: none, swanlab.")
        if self.mode not in {None, "online", "local", "offline", "disabled"}:
            raise ValueError("tracking.mode must be one of: online, local, offline, disabled.")
        if any(not isinstance(tag, str) for tag in self.tags):
            raise ValueError("tracking.tags must contain only strings.")


@dataclass(frozen=True)
class SafePruneConfig:
    experiment: ExperimentConfig
    model: ModelConfig
    data: DataConfig
    dpo: DPOConfig = field(default_factory=DPOConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SafePruneConfig":
        config = cls(
            experiment=ExperimentConfig(**raw["experiment"]),
            model=ModelConfig(**raw["model"]),
            data=DataConfig(**raw["data"]),
            dpo=DPOConfig(**raw.get("dpo", {})),
            pruning=PruningConfig(**raw.get("pruning", {})),
            recovery=RecoveryConfig(**raw.get("recovery", {})),
            evaluation=EvaluationConfig(**raw.get("evaluation", {})),
            agent=AgentConfig(**raw.get("agent", {})),
            tracking=TrackingConfig(**raw.get("tracking", {})),
        )
        config.pruning.validate()
        config.agent.validate()
        config.tracking.validate()
        return config


def load_config(path: str | Path) -> SafePruneConfig:
    return SafePruneConfig.from_dict(load_config_dict(path))
