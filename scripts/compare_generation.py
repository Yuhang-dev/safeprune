from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel

from safeprune.config import load_config
from safeprune.masks import attach_qwen_forward_masks
from safeprune.modeling import load_causal_lm_and_tokenizer
from safeprune.scoring import load_plan

PROMPTS = [
    "Explain pruning in neural networks in one paragraph.",
    "Write a short Python function to add two numbers.",
    "How can I stay safe during an earthquake?",
]


def _encode(tokenizer, prompt: str, use_chat_template: bool):
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    return tokenizer(prompt, return_tensors="pt")


def _generate(label: str, model, tokenizer, prompts: list[str], max_new_tokens: int, use_chat_template: bool) -> None:
    model.eval()
    print(f"\n===== {label} =====")
    for prompt in prompts:
        encoded = _encode(tokenizer, prompt, use_chat_template)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        print("\nPROMPT:", prompt)
        print(tokenizer.decode(output[0], skip_special_tokens=True))


def _attach_plan(config, model) -> None:
    plan_path = Path(config.pruning.scores_dir) / f"plan_s{config.pruning.sparsity:.2f}.json"
    attach_qwen_forward_masks(model, load_plan(plan_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare teacher, pruned, and recovered generation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--skip-recovered", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    prompts = args.prompt or PROMPTS

    teacher, tokenizer = load_causal_lm_and_tokenizer(config.model.teacher_model, config.model)
    _generate("teacher", teacher, tokenizer, prompts, args.max_new_tokens, args.chat_template)
    del teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pruned, tokenizer = load_causal_lm_and_tokenizer(config.pruning.pruned_dir, config.model)
    _attach_plan(config, pruned)
    _generate("pruned_masked", pruned, tokenizer, prompts, args.max_new_tokens, args.chat_template)
    del pruned
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not args.skip_recovered:
        recovered, tokenizer = load_causal_lm_and_tokenizer(config.pruning.pruned_dir, config.model)
        _attach_plan(config, recovered)
        recovered = PeftModel.from_pretrained(recovered, config.recovery.output_dir)
        _generate("recovered_masked_lora", recovered, tokenizer, prompts, args.max_new_tokens, args.chat_template)


if __name__ == "__main__":
    main()
