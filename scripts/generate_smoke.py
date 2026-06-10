from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel

from safeprune.config import load_config
from safeprune.masks import attach_qwen_forward_masks
from safeprune.modeling import load_causal_lm_and_tokenizer
from safeprune.scoring import load_plan


DEFAULT_PROMPTS = [
    "Explain pruning in neural networks in one paragraph.",
    "Write a short Python function to add two numbers.",
    "How can I stay safe during an earthquake?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a few samples from a masked LoRA recovery checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--chat-template", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    adapter_path = args.adapter or config.recovery.output_dir
    model, tokenizer = load_causal_lm_and_tokenizer(config.pruning.pruned_dir, config.model)

    plan_path = Path(config.pruning.scores_dir) / f"plan_s{config.pruning.sparsity:.2f}.json"
    if plan_path.exists():
        attach_qwen_forward_masks(model, load_plan(plan_path))
    else:
        raise FileNotFoundError(f"Missing pruning plan: {plan_path}")

    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    prompts = args.prompt or DEFAULT_PROMPTS
    for prompt in prompts:
        if args.chat_template and hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            encoded = tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        print("\nPROMPT:", prompt)
        print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
