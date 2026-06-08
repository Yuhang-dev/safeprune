from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable

ALLOWED_TAGS = {"helpfulness", "safety", "general"}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _write_jsonl(path: Path, rows: Iterable[dict[str, str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _completion_text(item: dict[str, Any]) -> str:
    for key in ("response", "text", "content", "answer", "completion"):
        value = _clean(item.get(key))
        if value:
            return value
    return ""


def _completion_score(item: dict[str, Any]) -> float:
    for key in ("overall_score", "score", "rating", "helpfulness_score"):
        value = item.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _rows_from_ultrafeedback(dataset, limit: int, seed: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    for idx in indices:
        raw = dataset[idx]
        prompt = _clean(raw.get("instruction") or raw.get("prompt"))
        completions = raw.get("completions") or raw.get("responses") or []
        if not prompt or not isinstance(completions, list) or len(completions) < 2:
            continue
        candidates = [
            (_completion_score(item), _completion_text(item))
            for item in completions
            if isinstance(item, dict) and _completion_text(item)
        ]
        if len(candidates) < 2:
            continue
        ranked = sorted(candidates, key=lambda pair: pair[0])
        rejected = ranked[0][1]
        chosen = ranked[-1][1]
        if chosen == rejected:
            continue
        rows.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "source": "ultrafeedback",
                "tag": "helpfulness",
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _rows_from_pair_dataset(
    dataset,
    limit: int,
    seed: int,
    source: str,
    tag: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    for idx in indices:
        raw = dataset[idx]
        prompt = _clean(raw.get("prompt") or raw.get("instruction") or raw.get("question"))
        chosen = _clean(raw.get("chosen") or raw.get("response_0") or raw.get("safer_response"))
        rejected = _clean(raw.get("rejected") or raw.get("response_1") or raw.get("unsafe_response"))
        if not prompt and chosen.startswith("Human:") and "Assistant:" in chosen:
            prompt = chosen.split("Assistant:", 1)[0].strip()
        if prompt and chosen and rejected and chosen != rejected:
            rows.append(
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "source": source,
                    "tag": tag,
                }
            )
        if len(rows) >= limit:
            break
    return rows


def _split(rows: list[dict[str, str]], train_count: int, eval_count: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    return rows[:train_count], rows[train_count : train_count + eval_count]


def _load_dataset(name: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets first: pip install datasets") from exc
    return load_dataset(name, split=split)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SafePrune-DPO JSONL files for 4090 feasibility runs.")
    parser.add_argument("--profile", choices=["smoke", "main"], default="smoke")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preference-dataset", default="openbmb/UltraFeedback")
    parser.add_argument("--preference-split", default="train")
    parser.add_argument("--safety-dataset", default="Anthropic/hh-rlhf")
    parser.add_argument("--safety-split", default="train")
    parser.add_argument("--train-count", type=int)
    parser.add_argument("--eval-count", type=int)
    parser.add_argument("--safety-count", type=int)
    parser.add_argument("--calibration-count", type=int)
    args = parser.parse_args()

    defaults = {
        "smoke": {"train": 500, "eval": 100, "safety": 100, "calibration": 128},
        "main": {"train": 3000, "eval": 500, "safety": 500, "calibration": 512},
    }[args.profile]
    train_count = args.train_count or defaults["train"]
    eval_count = args.eval_count or defaults["eval"]
    safety_count = args.safety_count or defaults["safety"]
    calibration_count = args.calibration_count or defaults["calibration"]

    preference_ds = _load_dataset(args.preference_dataset, args.preference_split)
    preference_rows = _rows_from_ultrafeedback(
        preference_ds,
        limit=train_count + eval_count + calibration_count,
        seed=args.seed,
    )
    if len(preference_rows) < train_count + eval_count:
        raise RuntimeError(f"Only built {len(preference_rows)} preference rows; need {train_count + eval_count}.")

    safety_ds = _load_dataset(args.safety_dataset, args.safety_split)
    safety_rows = _rows_from_pair_dataset(
        safety_ds,
        limit=safety_count + calibration_count,
        seed=args.seed,
        source=args.safety_dataset.split("/")[-1].lower().replace("-", "_"),
        tag="safety",
    )
    if len(safety_rows) < safety_count:
        raise RuntimeError(f"Only built {len(safety_rows)} safety rows; need {safety_count}.")

    train_rows, eval_rows = _split(preference_rows, train_count, eval_count)
    safety_replay = safety_rows[:safety_count]
    calibration = []
    calibration.extend(preference_rows[train_count + eval_count : train_count + eval_count + calibration_count // 2])
    calibration.extend(safety_rows[safety_count : safety_count + calibration_count - len(calibration)])
    for row in calibration:
        if row["tag"] not in ALLOWED_TAGS:
            row["tag"] = "general"

    out = Path(args.output_dir)
    counts = {
        "preference_train": _write_jsonl(out / "preference" / "train.jsonl", train_rows),
        "preference_eval": _write_jsonl(out / "preference" / "eval.jsonl", eval_rows),
        "safety_replay": _write_jsonl(out / "preference" / "safety_replay.jsonl", safety_replay),
        "calibration": _write_jsonl(out / "calibration" / "calibration.jsonl", calibration[:calibration_count]),
    }
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
