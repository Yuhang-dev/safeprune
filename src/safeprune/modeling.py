from __future__ import annotations

from typing import Any


def resolve_torch_dtype(dtype_name: str):
    import torch

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "auto": "auto",
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[dtype_name]


def load_causal_lm_and_tokenizer(
    model_name_or_path: str,
    model_config,
    load_in_4bit: bool = False,
    local_files_only: bool = False,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs: dict[str, Any] = {
        "trust_remote_code": model_config.trust_remote_code,
        "torch_dtype": resolve_torch_dtype(model_config.torch_dtype),
        "device_map": "auto",
    }
    if model_config.use_flash_attention_2:
        kwargs["attn_implementation"] = "flash_attention_2"
    if load_in_4bit:
        kwargs["load_in_4bit"] = True

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        use_fast=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs["local_files_only"] = local_files_only
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    model.config.use_cache = False
    return model, tokenizer


def attach_lora(model, recovery_config):
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=recovery_config.lora_r,
        lora_alpha=recovery_config.lora_alpha,
        lora_dropout=recovery_config.lora_dropout,
        target_modules=recovery_config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)
