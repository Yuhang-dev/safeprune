# Remote Agent FFN Pruning Workflow

This is the current remote checklist. SafePrune-DPO training and recovery
scripts are legacy and are not the default workflow.

## 1. Install

Remote GPU install:

```bash
cd /path/to/Prune
conda activate llm
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
```

If Hugging Face Xet causes download auth errors:

```bash
export HF_HUB_DISABLE_XET=1
```

## 2. Local CPU / Remote Dry-Run Checks

These checks do not require a local GPU because they skip model loading.

```bash
python -m unittest discover -s tests

python scripts/build_stage_mask_bank.py \
  --config configs/agent_qwen2_5_1_5b_4090.yaml \
  --skip-model-load

python scripts/evaluate_agent_masks.py \
  --config configs/agent_qwen2_5_1_5b_4090.yaml \
  --dry-run
```

## 3. Prepare Controlled Agent Tasks

```bash
python scripts/prepare_agent_tasks.py \
  --output data/agent/controlled_tasks.jsonl \
  --count 1000
```

## 4. Build Stage FFN Mask Bank

```bash
python scripts/build_stage_mask_bank.py \
  --config configs/agent_qwen2_5_1_5b_4090.yaml \
  --max-calibration-batches 256
```

Output:

```text
outputs/agent_qwen2_5_1_5b_4090/mask_bank/mask_bank.json
outputs/agent_qwen2_5_1_5b_4090/mask_bank/manifest.json
```

Repeat with:

```bash
python scripts/build_stage_mask_bank.py \
  --config configs/agent_qwen2_5_3b_4090.yaml \
  --max-calibration-batches 512
```

## 5. Agent Evaluation

The local implementation currently supports dry-run manifests. Remote full
generation evaluation should write metrics with this method set:

```text
dense
static_global_mask
stage_mask
failure_redensification
hidden_state_only_router
```

Required metrics:

```text
task_success_rate
tool_call_validity
recovery_success_rate
average_active_ffn_ratio
latency_ms
cost_per_success
```

## 6. Real Tool-Execution Agent Loop

Generate a 100-task smoke set:

```bash
python scripts/prepare_real_tool_tasks.py \
  --output data/agent/real_tool_tasks_v1_smoke_100.jsonl \
  --count 100 \
  --failure-count 20
```

Run the smoke set:

```bash
python -u scripts/evaluate_real_tool_loop.py \
  --config configs/agent_qwen2_5_3b_4090.yaml \
  --tasks data/agent/real_tool_tasks_v1_smoke_100.jsonl \
  --mask-bank-path outputs/agent_qwen2_5_3b_4090_low/mask_bank/mask_bank.json \
  --local-files-only \
  --methods \
    dense \
    identity_hook \
    global_balanced_observe_approx_0.01 \
    stage_global_balanced_approx_0.01 \
    failure_redense_global_balanced_approx_0.01 \
  --output-prefix outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_smoke_100
```

If dense and identity_hook match, generate and run the 1000-task set:

```bash
python scripts/prepare_real_tool_tasks.py \
  --output data/agent/real_tool_tasks_v1.jsonl \
  --count 1000 \
  --failure-count 200

python -u scripts/evaluate_real_tool_loop.py \
  --config configs/agent_qwen2_5_3b_4090.yaml \
  --tasks data/agent/real_tool_tasks_v1.jsonl \
  --mask-bank-path outputs/agent_qwen2_5_3b_4090_low/mask_bank/mask_bank.json \
  --local-files-only \
  --output-prefix outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_1000
```

The default method set also includes:

```text
hidden_state_centroid_global_balanced_approx_0.01
```

For the first smoke run, it is acceptable to omit the centroid method and add it
after dense/identity/stage/failure are stable.

## 7. SliceGPT Reproduction

Use `docs/slicegpt_repro_commands.md`. Keep the external repository outside
this project or under an ignored workspace directory. Do not vendor it into
`src/`.
