# SafePrune-DPO

SafePrune-DPO is a research scaffold for studying whether a DPO-aligned
Qwen2.5-7B-Instruct model preserves safety and preference alignment after
25%-50% structured pruning.

The intended remote workflow is:

1. Train or load a DPO-aligned teacher.
2. Compute structured pruning scores from calibration data.
3. Apply attention-head and MLP-channel pruning masks.
4. Recover with LoRA using DPO loss, teacher consistency, and safety replay.
5. Evaluate capability, safety, and efficiency trade-offs.

This repository keeps heavy ML dependencies optional at import time so that
config, data, and pruning policy tests can run on a CPU-only machine.

## Remote Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,eval]"
```

For Linux remote servers:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,eval]"
```

If you already have a conda environment named `LLM` on Windows:

```powershell
$env:PYTHONPATH="D:\Prune\src"
& "D:\anaconda3\envs\LLM\python.exe" -m unittest discover -s tests
```

## Data Format

All preference files are JSONL:

```json
{"prompt":"...","chosen":"...","rejected":"...","source":"ultrafeedback","tag":"helpfulness"}
```

Allowed tags are `helpfulness`, `safety`, and `general`.

Validate data before training:

```bash
python scripts/validate_data.py --path data/preference/train.jsonl
```

## Main Commands

Train the DPO teacher:

```bash
accelerate launch scripts/train_dpo_teacher.py --config configs/safeprune_qwen2_5_7b.yaml
```

Compute pruning scores:

```bash
python scripts/compute_prune_scores.py --config configs/safeprune_qwen2_5_7b.yaml
```

Apply pruning masks:

```bash
python scripts/prune_model.py --config configs/safeprune_qwen2_5_7b.yaml
```

Recover the pruned model:

```bash
accelerate launch scripts/recover_safeprune.py --config configs/safeprune_qwen2_5_7b.yaml
```

Run an ablation grid:

```bash
python scripts/run_ablation_grid.py --config configs/safeprune_qwen2_5_7b.yaml --dry-run
```

## Outputs

The default config writes to:

- `outputs/dpo_teacher`
- `outputs/prune_scores`
- `outputs/pruned`
- `outputs/recovered`
- `outputs/eval`

Each run should preserve its resolved YAML config, metrics JSON, and timing
metadata for paper reproducibility.

## Experiment Roadmap

See `docs/experiment_roadmap.md` for the full SafePrune-DPO experiment route,
including current status markers and unfinished items.

For a smaller two-RTX-4090 feasibility run before the A100 experiments, see
`docs/rtx4090_feasibility_plan.md`.
