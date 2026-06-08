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
    from transformers import TrainingArguments
    from trl import DPOTrainer

    output_dir = ensure_output_dir(Path(config.experiment.output_dir) / "dpo_teacher")
    records = load_preference_jsonl(config.data.preference_train)
    train_dataset = Dataset.from_list(to_hf_dpo_rows(records))

    eval_dataset = None
    if config.data.preference_eval:
        eval_records = load_preference_jsonl(config.data.preference_eval)
        eval_dataset = Dataset.from_list(to_hf_dpo_rows(eval_records))

    model, tokenizer = load_causal_lm_and_tokenizer(config.model.base_model, config.model)
    training_args = TrainingArguments(
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
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        beta=config.dpo.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=config.data.max_length,
        max_prompt_length=config.data.max_prompt_length,
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
    from transformers import TrainingArguments
    from trl import DPOTrainer

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

    ref_model_path = config.model.teacher_model or config.model.base_model
    ref_model, _ = load_causal_lm_and_tokenizer(ref_model_path, config.model)

    train_dataset = Dataset.from_list(to_hf_dpo_rows(records))
    training_args = TrainingArguments(
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
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        beta=config.dpo.beta,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_length=config.data.max_length,
        max_prompt_length=config.data.max_prompt_length,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def _json_dumps(value) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True)
