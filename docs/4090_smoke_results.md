# 4090 Smoke Results

Date: 2026-06-08

Remote host: `safeprune-4090`

Working directory:

```text
~/autodl-tmp/safeprune/code/Prune
```

## Environment

- GPU: 2 x RTX 4090, 24564 MiB each.
- Torch: `2.6.0+cu124`.
- Transformers: `4.44.2`.
- TRL: `0.10.1`.
- PEFT: `0.12.0`.
- Accelerate: `0.34.2`.
- SwanLab: `0.8.1`.

Data/cache/work directory:

```text
/root/autodl-tmp/safeprune
```

SwanLab project:

```text
https://swanlab.cn/@chengyuhang/safeprune-dpo-4090
```

## Experiment Process

### 1. Remote Environment

Actions completed:

1. Configured SSH alias `safeprune-4090`.
2. Verified two RTX 4090 GPUs with `nvidia-smi`.
3. Moved working/cache directories to the 50 GB data disk under `/root/autodl-tmp/safeprune`.
4. Installed compatible ML stack:

```bash
pip install \
  "transformers==4.44.2" \
  "tokenizers>=0.19,<0.20" \
  "accelerate==0.34.2" \
  "datasets==2.21.0" \
  "peft==0.12.0" \
  "trl==0.10.1" \
  "PyYAML>=6.0" \
  "swanlab>=0.4.0"
```

Important fixes:

- Upgraded PyTorch from `2.3.0+cu121` to `2.6.0+cu124`.
- Downgraded Transformers/TRL stack to avoid incompatible latest `transformers` float8 imports.
- Disabled Hugging Face Xet downloads with `HF_HUB_DISABLE_XET=1` after Xet CAS returned `401 Unauthorized`.
- Kept `use_flash_attention_2: false` because `flash-attn` is not installed.

### 2. Data Preparation

Command:

```bash
python scripts/prepare_4090_data.py --profile smoke
```

Generated and validated:

```text
data/preference/train.jsonl          500 rows, helpfulness
data/preference/eval.jsonl           100 rows, helpfulness
data/preference/safety_replay.jsonl  100 rows, safety
data/calibration/calibration.jsonl   128 rows, 64 helpfulness + 64 safety
```

### 3. DPO Teacher

Command:

```bash
accelerate launch --num_processes 1 scripts/train_dpo_teacher.py \
  --config configs/safeprune_qwen2_5_1_5b_4090_smoke.yaml
```

Output:

```text
outputs/safeprune_qwen2_5_1_5b_4090_smoke/dpo_teacher
```

SwanLab run:

```text
https://swanlab.cn/@chengyuhang/safeprune-dpo-4090/runs/7nyscszi
```

Status: completed. Teacher generation sanity check is normal.

### 4. Pruning Scores

Command:

```bash
python scripts/compute_prune_scores.py \
  --config configs/safeprune_qwen2_5_1_5b_4090_smoke.yaml \
  --with-activation \
  --max-calibration-batches 128
```

Output:

```text
outputs/safeprune_qwen2_5_1_5b_4090_smoke/prune_scores/scores.json
outputs/safeprune_qwen2_5_1_5b_4090_smoke/prune_scores/plan_s0.25.json
```

Additional plans were generated from the same `scores.json` without rerunning activation scoring:

```text
plan_s0.05.json
plan_s0.10.json
outputs/safeprune_qwen2_5_1_5b_4090_s0p10_mlp_only/prune_scores/plan_s0.10.json
```

Important fix:

- Activation hook accumulation was moved to CPU because `device_map="auto"` split layers across `cuda:0` and `cuda:1`.

### 5. Pruning And Recovery

Commands used:

```bash
python scripts/prune_model.py --config <config>

accelerate launch --num_processes 1 scripts/recover_safeprune.py \
  --config <config>
```

Important fixes:

- TRL `DPOTrainer` now uses `trl.DPOConfig` instead of `transformers.TrainingArguments`.
- Recovery DPO reference was changed from full teacher to pruned masked checkpoint. Using the full teacher as reference caused loss values around `90-124`.
- `prune_attention_heads: false` now actually disables attention-head pruning in generated plans.

### 6. Generation Sanity Checks

Generation comparison script:

```bash
python scripts/compare_generation.py \
  --config <config> \
  --chat-template \
  --max-new-tokens 80
```

Compared:

```text
teacher
pruned_masked
recovered_masked_lora
```

## Completed Runs

### DPO Teacher

Config:

```text
configs/safeprune_qwen2_5_1_5b_4090_smoke.yaml
```

Output:

```text
outputs/safeprune_qwen2_5_1_5b_4090_smoke/dpo_teacher
```

Status: completed.

Teacher generation sanity check: normal.

### 25% Mixed Head + MLP Pruning

Plan:

```text
outputs/safeprune_qwen2_5_1_5b_4090_smoke/prune_scores/plan_s0.25.json
```

Plan shape:

- Layers: 28.
- Attention: 12 heads per layer, 3 pruned per layer.
- MLP: 8960 channels per layer, 2240 pruned per layer.

Result:

- `pruned_masked`: collapsed into repeated numbers/fragments.
- `recovered_masked_lora`: still collapsed.

Conclusion: 25% mixed pruning fails for 1.5B smoke.

### 10% Mixed Head + MLP Pruning

Plan:

```text
outputs/safeprune_qwen2_5_1_5b_4090_smoke/prune_scores/plan_s0.10.json
```

Plan shape:

- Attention: 12 heads per layer, 1 pruned per layer.
- MLP: 8960 channels per layer, 896 pruned per layer.

Result:

- `pruned_masked`: near collapse. Python generation fails, pruning explanation repeats, safety answer degrades.

Conclusion: mixed 10% is already too aggressive for this 1.5B smoke setup.

### 5% MLP-Only Effective Pruning

Config:

```text
configs/safeprune_qwen2_5_1_5b_4090_s0p05.yaml
```

Plan shape:

- Attention: no heads pruned.
- MLP: 8960 channels per layer, 448 pruned per layer.

Recovery summary:

- Steps: 75.
- Average loss: about `0.700`.
- Last loss: about `0.681`.
- Last reward margin: about `0.036`.

Generation result:

- `pruned_masked`: coherent but degraded.
- `recovered_masked_lora`: code prompt improves; safety prompt remains weak.

Conclusion: 5% is the first non-collapsed point, but recovery is only partial.

### 10% MLP-Only Pruning

Config:

```text
configs/safeprune_qwen2_5_1_5b_4090_s0p10_mlp_only.yaml
```

Plan:

```text
outputs/safeprune_qwen2_5_1_5b_4090_s0p10_mlp_only/prune_scores/plan_s0.10.json
```

Plan shape:

- Attention: no heads pruned.
- MLP: 8960 channels per layer, 896 pruned per layer.

Recovery summary:

- Steps: 75.
- Average loss: about `0.687`.
- Last loss: about `0.666`.
- Last reward margin: about `0.139`.

Generation result:

- `pruned_masked`: no random-token collapse, but instruction following is badly degraded.
- `recovered_masked_lora`: training metrics look healthy, but generation remains degraded.

Conclusion: 10% MLP-only is more stable than mixed 10%, but still not usable after current recovery.

## Interpretation

The smoke run validates the remote training pipeline, SwanLab logging, score generation, mask attachment, pruning, LoRA recovery, and generation comparison scripts.

The current pruning method is too destructive for Qwen2.5-1.5B at moderate sparsity:

```text
5% MLP-only: no collapse, partial recovery
10% MLP-only: degraded, recovery metrics healthy but generation weak
10% mixed head+MLP: near collapse
25% mixed head+MLP: full collapse
```

Main finding for the next iteration:

```text
Attention head pruning is highly destabilizing in this setup.
MLP-only pruning is safer, but 10% still causes significant generation quality loss.
```

## Logs

Remote logs:

```text
logs/recover_s0p05_20260608_222342.log
logs/compare_s0p10_pruned_20260608_223427.log
logs/recover_s0p10_mlp_only_20260608_223924.log
logs/compare_s0p10_mlp_only_recovered_20260608_224228.log
```

SwanLab project:

```text
https://swanlab.cn/@chengyuhang/safeprune-dpo-4090
```

Additional remote artifacts:

```text
outputs/safeprune_qwen2_5_1_5b_4090_smoke/dpo_teacher
outputs/safeprune_qwen2_5_1_5b_4090_smoke/prune_scores
outputs/safeprune_qwen2_5_1_5b_4090_smoke/pruned
outputs/safeprune_qwen2_5_1_5b_4090_smoke/recovered
outputs/safeprune_qwen2_5_1_5b_4090_s0p05/pruned
outputs/safeprune_qwen2_5_1_5b_4090_s0p05/recovered
outputs/safeprune_qwen2_5_1_5b_4090_s0p10/pruned
outputs/safeprune_qwen2_5_1_5b_4090_s0p10_mlp_only/pruned
outputs/safeprune_qwen2_5_1_5b_4090_s0p10_mlp_only/recovered
```

## Final Conclusion

The 1.5B 4090 smoke experiment succeeded as an engineering feasibility check but failed as evidence that the current pruning/recovery method preserves behavior at moderate sparsity.

What is validated:

```text
remote environment
data preparation
DPO teacher training
SwanLab logging
magnitude + activation scoring
plan generation
mask attachment
LoRA recovery execution
generation comparison tooling
```

What is not validated:

```text
25% structured pruning quality
10% mixed head+MLP pruning quality
10% MLP-only recovery quality
real speedup or parameter reduction
full safety/capability benchmark results
```

Operational conclusion:

```text
Do not start 3B mixed head pruning directly.
First implement stronger recovery, especially teacher consistency or CE/SFT recovery.
If running 3B before that, start with MLP-only low-sparsity plans and generation sanity checks.
```

## Next Engineering Fixes

1. Add a teacher-consistency loss instead of relying only on DPO recovery.
2. Add a small CE/SFT recovery baseline on chosen answers.
3. Add MLP-only configs for 3B before trying mixed head pruning again.
4. Add automated small-prompt generation scoring to catch collapse before long recovery runs.
5. Revisit mask implementation and scoring before claiming any speedup or compression result.
