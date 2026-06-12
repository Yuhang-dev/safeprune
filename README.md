# Agent FFN Pruning

This repository studies trajectory-aware dynamic FFN pruning for reasoning and
tool-using language agents.

The current research question is:

```text
Can Agent stage labels, observations, failure events, and difficulty signals
allocate FFN compute better than fixed masks or hidden-state-only routing?
```

The primary metric is not raw sparsity. It is cost per successful task, measured
with task success, tool-call validity, recovery rate, active FFN ratio, latency,
and memory.

## Current Direction

The default project route is:

1. Treat `docs/controlled_mask_validation_v1.md` as the frozen controlled-mask
   MVP result.
2. Run Real Tool-Execution Agent Prune v1 on Qwen2.5-3B with local executable
   calculator, unit conversion, and lookup tools.
3. Compare dense, identity hook, observe-only global mask, stage-aware routing,
   and failure-triggered re-densification in the real tool loop.
4. Add `hidden_state_centroid_global_balanced_approx_0.01` as the first
   hidden-state-only dynamic routing baseline after the real tool smoke is
   stable.
5. Keep SliceGPT/Probe Pruning as external baselines, not blockers for the
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
  --output-prefix outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_1000
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
