from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ALLOWED_TAGS = {"helpfulness", "safety", "general"}


@dataclass(frozen=True)
class PreferenceRecord:
    prompt: str
    chosen: str
    rejected: str
    source: str
    tag: str


def validate_record(raw: dict, line_number: int | None = None) -> PreferenceRecord:
    prefix = f"line {line_number}: " if line_number is not None else ""
    required = {"prompt", "chosen", "rejected", "source", "tag"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"{prefix}missing keys: {sorted(missing)}")

    values = {key: raw[key] for key in required}
    for key, value in values.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{prefix}{key} must be a non-empty string")

    tag = values["tag"]
    if tag not in ALLOWED_TAGS:
        raise ValueError(f"{prefix}tag must be one of {sorted(ALLOWED_TAGS)}, got {tag!r}")

    return PreferenceRecord(
        prompt=values["prompt"],
        chosen=values["chosen"],
        rejected=values["rejected"],
        source=values["source"],
        tag=tag,
    )


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: JSONL rows must be objects")
            rows.append(row)
    return rows


def load_preference_jsonl(path: str | Path) -> list[PreferenceRecord]:
    return [validate_record(row, idx) for idx, row in enumerate(read_jsonl(path), start=1)]


def summarize_records(records: Iterable[PreferenceRecord]) -> dict[str, int]:
    summary = {"total": 0, "helpfulness": 0, "safety": 0, "general": 0}
    for record in records:
        summary["total"] += 1
        summary[record.tag] += 1
    return summary


def to_hf_dpo_rows(records: Iterable[PreferenceRecord]) -> list[dict[str, str]]:
    return [
        {"prompt": record.prompt, "chosen": record.chosen, "rejected": record.rejected}
        for record in records
    ]

