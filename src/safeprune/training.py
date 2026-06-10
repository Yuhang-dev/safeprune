from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

from .config import SafePruneConfig
from .data import load_preference_jsonl, to_hf_dpo_rows
from .masks import attach_qwen_forward_masks
from .modeling import attach_lora, load_causal_lm_and_tokenizer
from .scoring import load_plan


def ensure_output_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def train_dpo_teacher(config: SafePruneConfig) -> None:
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    output_dir = ensure_output_dir(Path(config.experiment.output_dir) / "dpo_teacher")
    records = load_preference_jsonl(config.data.preference_train)
    train_dataset = Dataset.from_list(to_hf_dpo_rows(records))

    eval_dataset = None
    if config.data.preference_eval:
        eval_records = load_preference_jsonl(config.data.preference_eval)
        eval_dataset = Dataset.from_list(to_hf_dpo_rows(eval_records))

    model, tokenizer = load_causal_lm_and_tokenizer(config.model.base_model, config.model)
    training_args = DPOConfig(
        output_dir=str(output_dir),
        learning_rate=config.dpo.learning_rate,
        num_train_epochs=config.dpo.num_train_epochs,
        per_device_train_batch_size=config.dpo.per_device_train_batch_size,
        gradient_accumulation_steps=config.dpo.gradient_accumulation_steps,
        warmup_ratio=config.dpo.warmup_ratio,
        logging_steps=config.dpo.logging_steps,
        save_steps=config.dpo.save_steps,
        bf16=config.model.torch_dtype in {"bfloat16", "bf16"},
        remove_unused_columns=False,
        report_to="none",
        beta=config.dpo.beta,
        max_length=config.data.max_length,
        max_prompt_length=config.data.max_prompt_length,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        callbacks=_build_tracking_callbacks(config, stage="dpo_teacher"),
    )
    started = time.time()
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "run_metadata.json").write_text(
        _json_dumps({"seconds": time.time() - started, "config": asdict(config)}),
        encoding="utf-8",
    )


def recover_with_lora(config: SafePruneConfig) -> None:
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer

    output_dir = ensure_output_dir(config.recovery.output_dir)
    records = load_preference_jsonl(config.data.preference_train)
    if config.data.safety_replay and config.recovery.replay_ratio > 0:
        replay = load_preference_jsonl(config.data.safety_replay)
        replay_count = max(1, int(len(records) * config.recovery.replay_ratio))
        records = records + replay[:replay_count]

    model, tokenizer = load_causal_lm_and_tokenizer(config.pruning.pruned_dir, config.model)
    plan_path = Path(config.pruning.scores_dir) / f"plan_s{config.pruning.sparsity:.2f}.json"
    if plan_path.exists():
        attach_qwen_forward_masks(model, load_plan(plan_path))
    model = attach_lora(model, config.recovery)

    ref_model, _ = load_causal_lm_and_tokenizer(config.pruning.pruned_dir, config.model)
    if plan_path.exists():
        attach_qwen_forward_masks(ref_model, load_plan(plan_path))

    train_dataset = Dataset.from_list(to_hf_dpo_rows(records))
    training_args = DPOConfig(
        output_dir=str(output_dir),
        learning_rate=config.recovery.learning_rate,
        num_train_epochs=config.recovery.num_train_epochs,
        per_device_train_batch_size=config.recovery.per_device_train_batch_size,
        gradient_accumulation_steps=config.recovery.gradient_accumulation_steps,
        logging_steps=config.dpo.logging_steps,
        save_steps=config.dpo.save_steps,
        bf16=config.model.torch_dtype in {"bfloat16", "bf16"},
        remove_unused_columns=False,
        report_to="none",
        beta=config.dpo.beta,
        max_length=config.data.max_length,
        max_prompt_length=config.data.max_prompt_length,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        callbacks=_build_tracking_callbacks(config, stage="recovery"),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def _build_tracking_callbacks(config: SafePruneConfig, stage: str) -> list:
    tracking = config.tracking
    if not tracking.enabled or tracking.backend == "none":
        return []
    if tracking.backend != "swanlab":
        raise ValueError(f"Unsupported tracking backend: {tracking.backend}")

    try:
        from swanlab.integration.transformers import SwanLabCallback
    except ImportError as exc:  # pragma: no cover - depends on remote env
        raise RuntimeError("SwanLab tracking is enabled. Install it with: pip install swanlab") from exc

    experiment_name = tracking.experiment_name or f"{config.experiment.name}_{stage}"
    callback_kwargs = {
        "project": tracking.project,
        "experiment_name": experiment_name,
        "description": tracking.description or f"{stage} run for {config.experiment.name}",
        "log_dir": tracking.logdir or str(Path(config.experiment.output_dir) / "swanlog"),
    }
    if tracking.workspace:
        callback_kwargs["workspace"] = tracking.workspace
    if tracking.mode:
        callback_kwargs["mode"] = tracking.mode
    if tracking.tags:
        callback_kwargs["tags"] = tracking.tags
    return [SwanLabCallback(**callback_kwargs)]


def _json_dumps(value) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True)
