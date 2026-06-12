# Agent FFN Pruning

This repository studies trajectory-event-aware FFN compute allocation for
reasoning and tool-using language agents.

The current research question is:

```text
Can Agent failure events, recovery stages, observations, and hidden states
identify where FFN compute should be preserved or temporarily restored?
```

The primary metric is not raw sparsity. It is cost per successful task, measured
with task success, tool-call validity, recovery rate, active FFN ratio, latency,
and memory.

## Current Direction

The default project route is:

1. Treat `docs/controlled_mask_validation_v1.md` as the frozen controlled-mask
   MVP result.
2. Treat `docs/real_tool_v1_results.md` as the frozen real tool-execution v1
   result.
3. Use Real Tool v1's central finding as the current paper story: failure
   recovery is a localized reflect-step compute bottleneck, and trajectory
   events can trigger targeted re-densification.
4. Add hidden-state dynamic routing baselines after Real Tool v1:
   `hidden_state_centroid_global_balanced_approx_0.01`,
   `hidden_state_centroid_reflect_dense_global_balanced_approx_0.01`, and
   `hidden_state_centroid_event_reflect_dense_global_balanced_approx_0.01`.
5. Start a bottom-up pruning budget ladder, targeting 3%, 5%, and 10% global FFN
   budgets with layer-wise allocation and compensation.
6. Keep SliceGPT/Probe Pruning as external baselines, not blockers for the
   current Agent Prune v1 loop.

The previous SafePrune-DPO alignment-preserving pruning path is now legacy. The
4090 smoke results remain useful as a negative signal: attention-head pruning was
unstable, and mask-only experiments must not be reported as real speedups.

## Environment

Remote smoke testing used this stack:

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install \
  transformers==4.44.2 \
  datasets==2.21.0 \
  peft==0.12.0 \
  trl==0.10.1 \
  accelerate==0.34.2 \
  PyYAML>=6.0 \
  swanlab==0.8.1
pip install -e .
```

For local CPU-only structural tests, do not load models and do not require a
GPU. Use an existing Python/conda environment or create a CPU test environment:

```powershell
conda create -n prune-agent-cpu python=3.10 -y
conda activate prune-agent-cpu
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.44.2 datasets==2.21.0 peft==0.12.0 trl==0.10.1 accelerate==0.34.2 PyYAML swanlab==0.8.1 pytest ruff
pip install -e .
```

Local commands should use `--skip-model-load` or `--dry-run` unless a remote GPU
host is available.

## Agent Data Format

Agent tasks are JSONL rows:

```json
{"task_id":"t1","prompt":"...","steps":[{"stage":"act","text":"...","event":"tool_error"}],"answer":"...","expected":"..."}
```

Allowed stages are:

```text
plan, act, observe, reflect, answer
```

Generate a small controlled set:

```bash
python scripts/prepare_agent_tasks.py \
  --output data/agent/controlled_tasks.jsonl \
  --count 100
```

## Main Commands

Build a stage-specific FFN mask bank without loading a model:

```bash
python scripts/build_stage_mask_bank.py \
  --config configs/agent_qwen2_5_1_5b_4090.yaml \
  --skip-model-load
```

Run an Agent evaluation dry run:

```bash
python scripts/evaluate_agent_masks.py \
  --config configs/agent_qwen2_5_1_5b_4090.yaml \
  --dry-run
```

Generate real tool-execution tasks:

```bash
export PYTHONPATH=$PWD:$PWD/src

python scripts/prepare_real_tool_tasks.py \
  --output data/agent/real_tool_tasks_v1.jsonl \
  --count 1000 \
  --failure-count 200
```

Run the real tool-execution Agent loop:

```bash
python scripts/evaluate_real_tool_loop.py \
  --config configs/agent_qwen2_5_3b_4090.yaml \
  --tasks data/agent/real_tool_tasks_v1.jsonl \
  --mask-bank-path outputs/agent_qwen2_5_3b_4090_low/mask_bank/mask_bank.json \
  --local-files-only \
  --output-prefix outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_bypass_1000
```

Analyze the frozen Real Tool v1 result:

```bash
python scripts/analyze_real_tool_v1_results.py
```

Run the next hidden-state centroid baseline with a separate calibration set:

```bash
python scripts/prepare_real_tool_tasks.py \
  --output data/agent/real_tool_tasks_v1_centroid_calib_500.jsonl \
  --count 500 \
  --failure-count 100 \
  --start-index 1000

python scripts/evaluate_real_tool_loop.py \
  --config configs/agent_qwen2_5_3b_4090.yaml \
  --tasks data/agent/real_tool_tasks_v1.jsonl \
  --centroid-calibration-tasks data/agent/real_tool_tasks_v1_centroid_calib_500.jsonl \
  --max-centroid-calibration-tasks 500 \
  --mask-bank-path outputs/agent_qwen2_5_3b_4090_low/mask_bank/mask_bank.json \
  --local-files-only \
  --methods \
    stage_reflect_dense_global_balanced_approx_0.01 \
    hidden_state_centroid_global_balanced_approx_0.01 \
    hidden_state_centroid_reflect_dense_global_balanced_approx_0.01 \
    hidden_state_centroid_event_reflect_dense_global_balanced_approx_0.01 \
  --save-centroid-router outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/hidden_state_centroid_router_v1.json \
  --output-prefix outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_hidden_centroid_1000
```

Build Pruning Substrate v2 budget plans:

```bash
python scripts/build_pruning_substrate_v2.py \
  --config configs/agent_qwen2_5_3b_4090.yaml \
  --calibration-path data/agent/controlled_tasks.jsonl \
  --max-calibration-prompts 512 \
  --candidate-sparsities 0,0.02,0.05,0.10,0.20,0.30 \
  --target-budgets 0.01,0.03,0.05,0.10,0.20 \
  --score-methods activation wanda flap \
  --with-bias-compensation \
  --with-layer-scale-placeholder \
  --local-files-only \
  --output-dir outputs/agent_qwen2_5_3b_4090_low/substrate_v2
```

Run a mask-hook PPL sanity check for a selected plan:

```bash
python scripts/evaluate_pruning_ppl.py \
  --config configs/agent_qwen2_5_3b_4090.yaml \
  --plan outputs/agent_qwen2_5_3b_4090_low/substrate_v2/flap/budget_plan_0p05.json \
  --prompts-jsonl data/agent/controlled_tasks.jsonl \
  --max-prompts 256 \
  --local-files-only \
  --output outputs/agent_qwen2_5_3b_4090_low/substrate_v2/flap/ppl_0p05.json
```

Run Real Tool smoke with external substrate plans:

```bash
python -u scripts/evaluate_real_tool_loop.py \
  --config configs/agent_qwen2_5_3b_4090.yaml \
  --tasks data/agent/real_tool_tasks_v1_smoke_100.jsonl \
  --mask-bank-path outputs/agent_qwen2_5_3b_4090_low/mask_bank/mask_bank.json \
  --local-files-only \
  --substrate-plan outputs/agent_qwen2_5_3b_4090_low/substrate_v2/flap/budget_plan_0p03.json \
  --substrate-name flap_0p03 \
  --substrate-plan outputs/agent_qwen2_5_3b_4090_low/substrate_v2/flap/budget_plan_0p05.json \
  --substrate-name flap_0p05 \
  --substrate-plan outputs/agent_qwen2_5_3b_4090_low/substrate_v2/flap/budget_plan_0p10.json \
  --substrate-name flap_0p10 \
  --methods \
    dense \
    substrate_flap_0p03_stage_reflect_dense \
    substrate_flap_0p05_stage_reflect_dense \
    substrate_flap_0p10_stage_reflect_dense \
    substrate_flap_0p05_observe_failure_redense \
    substrate_flap_0p10_observe_failure_redense \
  --output-prefix outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_substrate_flap_smoke_100
```

Run local unit tests:

```bash
python -m unittest discover -s tests
```

## Docs

- `docs/experiment_roadmap.md`: current Agent FFN pruning roadmap.
- `docs/rtx4090_feasibility_plan.md`: two-RTX-4090 execution plan.
- `docs/slicegpt_repro_commands.md`: external SliceGPT reproduction commands.
- `docs/4090_smoke_results.md`: legacy SafePrune-DPO smoke results and failure notes.
- `docs/controlled_mask_validation_v1.md`: frozen 1000-task controlled mask validation.
- `docs/real_tool_v1_results.md`: frozen 1000-task real tool-execution result.
