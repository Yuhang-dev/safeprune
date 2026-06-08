# Remote SafePrune-DPO Workflow

This file is the execution checklist for the remote GPU environment.

## 1. Install

```bash
cd /path/to/Prune
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,eval]"
```

If the remote machine already uses conda:

```bash
conda activate llm
pip install -e ".[dev,eval]"
```

## 2. Prepare Data

Create these files:

- `data/preference/train.jsonl`
- `data/preference/eval.jsonl`
- `data/preference/safety_replay.jsonl`
- `data/calibration/calibration.jsonl`

Validate each preference file:

```bash
python scripts/validate_data.py --path data/preference/train.jsonl
python scripts/validate_data.py --path data/preference/eval.jsonl
python scripts/validate_data.py --path data/preference/safety_replay.jsonl
```

## 3. Teacher

```bash
accelerate launch scripts/train_dpo_teacher.py \
  --config configs/safeprune_qwen2_5_7b.yaml
```

## 4. Pruning Scores and Plans

```bash
python scripts/compute_prune_scores.py \
  --config configs/safeprune_qwen2_5_7b.yaml
```

This writes:

- `outputs/safeprune_qwen2_5_7b/prune_scores/scores.json`
- `outputs/safeprune_qwen2_5_7b/prune_scores/plan_s0.25.json`
- `outputs/safeprune_qwen2_5_7b/prune_scores/plan_s0.35.json`
- `outputs/safeprune_qwen2_5_7b/prune_scores/plan_s0.50.json`

## 5. Ablations

Generate resolved configs:

```bash
python scripts/run_ablation_grid.py \
  --config configs/safeprune_qwen2_5_7b.yaml \
  --write-configs
```

Run the printed `accelerate launch` commands.

## 6. Evaluation

```bash
python scripts/evaluate_model.py \
  --config configs/safeprune_qwen2_5_7b.yaml \
  --model outputs/safeprune_qwen2_5_7b/recovered
```

Connect the generated manifest to the remote lm-eval, HarmBench, AdvBench
refusal, and latency scripts.

