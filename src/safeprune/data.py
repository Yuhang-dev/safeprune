from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ALLOWED_TAGS = {"helpfulness", "safety", "general"}
ALLOWED_AGENT_STAGES = {"plan", "act", "observe", "reflect", "answer"}


@dataclass(frozen=True)
class PreferenceRecord:
    prompt: str
    chosen: str
    rejected: str
    source: str
    tag: str


@dataclass(frozen=True)
class AgentStep:
    stage: str
    text: str
    event: str | None = None
    tool_name: str | None = None
    metadata: dict | None = None

    @property
    def is_failure(self) -> bool:
        return bool(self.event and self.event not in {"ok", "success", "none"})


@dataclass(frozen=True)
class AgentTrajectory:
    task_id: str
    prompt: str
    steps: list[AgentStep]
    answer: str
    expected: str
    source: str = "controlled"
    metadata: dict | None = None


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


def validate_agent_step(raw: dict, prefix: str = "") -> AgentStep:
    required = {"stage", "text"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"{prefix}missing step keys: {sorted(missing)}")

    stage = raw["stage"]
    text = raw["text"]
    if not isinstance(stage, str) or stage not in ALLOWED_AGENT_STAGES:
        raise ValueError(f"{prefix}stage must be one of {sorted(ALLOWED_AGENT_STAGES)}, got {stage!r}")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{prefix}step text must be a non-empty string")

    event = raw.get("event")
    if event is not None and not isinstance(event, str):
        raise ValueError(f"{prefix}event must be a string when provided")
    tool_name = raw.get("tool_name")
    if tool_name is not None and not isinstance(tool_name, str):
        raise ValueError(f"{prefix}tool_name must be a string when provided")
    metadata = raw.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError(f"{prefix}step metadata must be an object when provided")

    return AgentStep(
        stage=stage,
        text=text,
        event=event,
        tool_name=tool_name,
        metadata=metadata,
    )


def validate_agent_trajectory(raw: dict, line_number: int | None = None) -> AgentTrajectory:
    prefix = f"line {line_number}: " if line_number is not None else ""
    required = {"task_id", "prompt", "steps", "answer", "expected"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"{prefix}missing keys: {sorted(missing)}")

    for key in ["task_id", "prompt", "answer", "expected"]:
        if not isinstance(raw[key], str) or not raw[key].strip():
            raise ValueError(f"{prefix}{key} must be a non-empty string")

    raw_steps = raw["steps"]
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError(f"{prefix}steps must be a non-empty list")
    steps = [
        validate_agent_step(step, prefix=f"{prefix}step {idx}: ")
        for idx, step in enumerate(raw_steps, start=1)
    ]

    source = raw.get("source", "controlled")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"{prefix}source must be a non-empty string when provided")
    metadata = raw.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError(f"{prefix}metadata must be an object when provided")

    return AgentTrajectory(
        task_id=raw["task_id"],
        prompt=raw["prompt"],
        steps=steps,
        answer=raw["answer"],
        expected=raw["expected"],
        source=source,
        metadata=metadata,
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


def load_agent_jsonl(path: str | Path) -> list[AgentTrajectory]:
    return [
        validate_agent_trajectory(row, idx)
        for idx, row in enumerate(read_jsonl(path), start=1)
    ]


def iter_agent_stage_texts(
    trajectories: Iterable[AgentTrajectory],
    stages: Iterable[str] | None = None,
) -> dict[str, list[str]]:
    selected = set(stages) if stages is not None else set(ALLOWED_AGENT_STAGES)
    grouped = {stage: [] for stage in selected}
    for trajectory in trajectories:
        for step in trajectory.steps:
            if step.stage in selected:
                grouped.setdefault(step.stage, []).append(step.text)
    return grouped


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
