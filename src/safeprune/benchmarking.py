from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_many(paths: list[str | Path]) -> str | None:
    existing = [Path(path) for path in paths if Path(path).exists()]
    if not existing:
        return None
    digest = hashlib.sha256()
    for path in sorted(existing, key=lambda item: str(item)):
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        file_digest = sha256_file(path)
        if file_digest is not None:
            digest.update(file_digest.encode("ascii"))
            digest.update(b"\0")
    return digest.hexdigest()


def build_benchmark_manifest(
    *,
    model_revision: str,
    plan_path: str | Path | None,
    calibration_paths: list[str | Path],
    benchmark_version: str,
    seed: int,
    dtype: str,
    device: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "model_revision": model_revision,
        "plan_path": str(plan_path) if plan_path else None,
        "plan_sha256": sha256_file(plan_path),
        "calibration_data_sha256": sha256_many(calibration_paths),
        "benchmark_version": benchmark_version,
        "seed": int(seed),
        "dtype": str(dtype),
        "device": str(device),
    }
    if extra:
        manifest.update(extra)
    return manifest


@dataclass(frozen=True)
class MultipleChoiceExample:
    prompt: str
    choices: list[str]
    label: int


def example_from_piqa(row: dict[str, Any]) -> MultipleChoiceExample:
    return MultipleChoiceExample(
        prompt=f"Question: {str(row['goal']).strip()}\nAnswer:",
        choices=[f" {row['sol1']}", f" {row['sol2']}"],
        label=int(row["label"]),
    )


def example_from_hellaswag(row: dict[str, Any]) -> MultipleChoiceExample:
    endings = row.get("endings") or []
    return MultipleChoiceExample(
        prompt=f"{str(row['ctx']).strip()}\nContinuation:",
        choices=[f" {ending}" for ending in endings],
        label=int(row["label"]),
    )


def example_from_arc_easy(row: dict[str, Any]) -> MultipleChoiceExample:
    choices = row.get("choices") or {}
    labels = [str(label) for label in choices.get("label", [])]
    texts = [str(text) for text in choices.get("text", [])]
    answer = str(row["answerKey"])
    try:
        label = labels.index(answer)
    except ValueError:
        normalized = [text.strip().lower() for text in texts]
        label = normalized.index(answer.strip().lower())
    return MultipleChoiceExample(
        prompt=f"Question: {str(row['question']).strip()}\nAnswer:",
        choices=[f" {text}" for text in texts],
        label=label,
    )


def score_continuation_mean_nll(
    *,
    model,
    tokenizer,
    prompt: str,
    continuation: str,
    max_length: int,
) -> float:
    import torch

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    continuation_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]
    if not continuation_ids:
        return float("inf")
    input_ids = list(prompt_ids) + list(continuation_ids)
    labels = [-100] * len(prompt_ids) + list(continuation_ids)
    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        labels = labels[-max_length:]
    valid_tokens = sum(1 for label in labels if label != -100)
    if valid_tokens == 0:
        return float("inf")

    device = next(model.parameters()).device
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    label_tensor = torch.tensor([labels], dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model(input_ids=input_tensor, labels=label_tensor)
    return float(outputs.loss.detach().float().cpu())


def evaluate_multiple_choice_examples(
    *,
    model,
    tokenizer,
    examples: list[MultipleChoiceExample],
    max_length: int,
) -> dict[str, Any]:
    correct = 0
    for example in examples:
        scores = [
            score_continuation_mean_nll(
                model=model,
                tokenizer=tokenizer,
                prompt=example.prompt,
                continuation=choice,
                max_length=max_length,
            )
            for choice in example.choices
        ]
        prediction = min(range(len(scores)), key=lambda idx: scores[idx])
        correct += int(prediction == example.label)
    total = len(examples)
    return {
        "metric": "accuracy",
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "scoring": "mean_continuation_nll",
    }


def evaluate_text_ppl(
    *,
    model,
    tokenizer,
    texts: list[str],
    max_length: int,
) -> dict[str, Any]:
    import torch

    total_loss = 0.0
    total_tokens = 0
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            input_ids = encoded["input_ids"].to(device)
            if input_ids.numel() < 2:
                continue
            outputs = model(input_ids=input_ids, labels=input_ids)
            tokens = int(input_ids.numel())
            total_loss += float(outputs.loss.detach().float().cpu()) * tokens
            total_tokens += tokens
    loss = total_loss / total_tokens if total_tokens else float("nan")
    return {
        "metric": "perplexity",
        "loss": loss,
        "ppl": math.exp(loss) if total_tokens else float("nan"),
        "total_tokens": total_tokens,
        "total": len(texts),
    }


def summarize_subnet_theory(
    *,
    plan: dict[str, Any] | None,
    hidden_size: int,
    intermediate_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int | None = None,
    sequence_length: int | None = None,
) -> dict[str, Any]:
    kv_heads = int(num_key_value_heads or num_attention_heads)
    head_dim = int(hidden_size // num_attention_heads)
    total_channels = intermediate_size * num_hidden_layers
    remaining_channels = total_channels
    pruned_channels = 0
    per_layer_remaining = []

    pruned_by_layer = _pruned_counts_by_layer(plan)
    for layer in range(num_hidden_layers):
        pruned = int(pruned_by_layer.get(layer, 0))
        remaining = max(0, intermediate_size - pruned)
        pruned_channels += pruned
        per_layer_remaining.append({"layer": layer, "remaining_ffn_channels": remaining})
    remaining_channels = total_channels - pruned_channels

    ffn_macs = 3 * hidden_size * remaining_channels
    dense_ffn_macs = 3 * hidden_size * total_channels
    attn_proj_macs_per_layer = (
        hidden_size * hidden_size
        + 2 * hidden_size * kv_heads * head_dim
        + hidden_size * hidden_size
    )
    attn_proj_macs = attn_proj_macs_per_layer * num_hidden_layers
    attention_context_macs = 0
    if sequence_length is not None:
        attention_context_macs = (
            2 * num_attention_heads * int(sequence_length) * head_dim * num_hidden_layers
        )
    total_macs = ffn_macs + attn_proj_macs + attention_context_macs

    return {
        "remaining_ffn_channels": remaining_channels,
        "total_ffn_channels": total_channels,
        "pruned_ffn_channels": pruned_channels,
        "active_ffn_channel_ratio": remaining_channels / total_channels
        if total_channels
        else 0.0,
        "remaining_ffn_parameters": ffn_macs,
        "dense_ffn_parameters": dense_ffn_macs,
        "ffn_macs_per_token": ffn_macs,
        "dense_ffn_macs_per_token": dense_ffn_macs,
        "ffn_flops_per_token": 2 * ffn_macs,
        "attention_projection_macs_per_token": attn_proj_macs,
        "attention_context_macs_per_token": attention_context_macs,
        "total_theoretical_macs_per_token": total_macs,
        "total_theoretical_flops_per_token": 2 * total_macs,
        "per_layer": per_layer_remaining,
        "notes": [
            "FFN parameters count gate/up/down matrix weights only.",
            "Total theoretical FLOPs include projection and FFN matmuls; "
            "attention context matmuls require --sequence-length.",
        ],
    }


def _pruned_counts_by_layer(plan: dict[str, Any] | None) -> dict[int, int]:
    if not plan:
        return {}
    counts = {}
    for layer in plan.get("layers", []):
        layer_idx = int(layer.get("layer", layer.get("layer_idx", 0)))
        counts[layer_idx] = len(layer.get("pruned_mlp_channels", []))
    return counts


def write_markdown_table(rows: list[dict[str, Any]], path: str | Path) -> None:
    lines = [
        "| Name | Active FFN | Remaining channels | FFN MACs/token | Total FLOPs/token |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['active_ffn_channel_ratio']:.4f} | "
            f"{row['remaining_ffn_channels']} | {row['ffn_macs_per_token']} | "
            f"{row['total_theoretical_flops_per_token']} |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def dump_json(payload: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
