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

Real Tool-Execution Agent Prune v1 已完成并冻结，不需要继续补同配置
`real_tool_v1_bypass_1000`。

冻结输出：

```text
outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_bypass_1000*
docs/real_tool_v1_results.md
```

本地分析命令：

```bash
python scripts/analyze_real_tool_v1_results.py
```

下一轮远端先跑 hidden-state centroid baseline，再进入 FLAP-style substrate v2
和 3% / 5% / 10% global FFN budget ladder。

### P1 hidden-state centroid baseline

先生成独立 calibration tasks，避免从冻结的 1000 条 eval tasks 拟合 centroid：

```bash
export PYTHONPATH=$PWD:$PWD/src

python scripts/prepare_real_tool_tasks.py \
  --output data/agent/real_tool_tasks_v1_centroid_calib_500.jsonl \
  --count 500 \
  --failure-count 100 \
  --start-index 1000
```

100 条 smoke：

```bash
python -u scripts/evaluate_real_tool_loop.py \
  --config configs/agent_qwen2_5_3b_4090.yaml \
  --tasks data/agent/real_tool_tasks_v1_smoke_100.jsonl \
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
  --output-prefix outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_hidden_centroid_smoke_100
```

1000 条正式：

```bash
python -u scripts/evaluate_real_tool_loop.py \
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
  --load-centroid-router outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/hidden_state_centroid_router_v1.json \
  --output-prefix outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_hidden_centroid_1000
```

重点读这些指标：

```text
reflect_detection_precision / recall / f1
false_dense_fallback_rate
missed_reflect_rate
routing_probe_cost
effective_cost_per_success
```

P1 1000 跑完后生成可贴进文档的分析表：

```bash
python scripts/analyze_hidden_state_p1_results.py \
  --prefix outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_p1_hidden_1000
```

## 6.1 Pruning Substrate v2

P2 不再继续手工调 `1%` mask。先构建 `1% / 3% / 5% / 10% / 20%`
global FFN budget plans：

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

输出重点：

```text
outputs/agent_qwen2_5_3b_4090_low/substrate_v2/*/budget_plan_0p03.json
outputs/agent_qwen2_5_3b_4090_low/substrate_v2/*/budget_plan_0p05.json
outputs/agent_qwen2_5_3b_4090_low/substrate_v2/*/budget_plan_0p10.json
outputs/agent_qwen2_5_3b_4090_low/substrate_v2/mask_bank_substrate_v2.json
outputs/agent_qwen2_5_3b_4090_low/substrate_v2/substrate_v2_manifest.json
```

先对候选 plan 做 PPL sanity：

```bash
python scripts/evaluate_pruning_ppl.py \
  --config configs/agent_qwen2_5_3b_4090.yaml \
  --plan outputs/agent_qwen2_5_3b_4090_low/substrate_v2/flap/budget_plan_0p05.json \
  --prompts-jsonl data/agent/controlled_tasks.jsonl \
  --max-prompts 256 \
  --local-files-only \
  --output outputs/agent_qwen2_5_3b_4090_low/substrate_v2/flap/ppl_0p05.json
```

只有通过 PPL sanity 的 `3% / 5% / 10%` plan 才接回 Real Tool 100 smoke。
当前 PPL sanity 是 controlled-task prompt sanity，不是 WikiText-2：

| Plan | Active MLP ratio | PPL | Relative PPL increase |
|---|---:|---:|---:|
| dense | 1.0000 | 23.2031 | 0.00% |
| flap 3% | 0.9697 | 23.6021 | +1.72% |
| flap 5% | 0.9497 | 23.6440 | +1.90% |
| flap 10% | 0.8998 | 24.7785 | +6.79% |
| flap 20% | 0.7998 | 24.0066 | +3.46% |

`flap` plans 已包含 bias compensation；layer scale 目前只是 `1.0`
placeholder，尚未做真实 scale calibration。

### P2 Real Tool smoke with external substrate plans

`evaluate_real_tool_loop.py` 支持用外部 substrate plan 注册动态方法。先跑核心 6 个方法：

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

这组已经跑完，结果如下：

| Method | Success | Failure | Non-failure | Cost / success | Fallback step ratio | Collapse |
|---|---:|---:|---:|---:|---:|---:|
| dense | 100/100 | 20/20 | 80/80 | 2.2000 | 0.0000 | 0.0000 |
| substrate_flap_0p03_stage_reflect_dense | 100/100 | 20/20 | 80/80 | 2.1395 | 0.0909 | 0.0000 |
| substrate_flap_0p05_stage_reflect_dense | 66/100 | 13/20 | 53/80 | 5.1615 | 0.5244 | 0.0000 |
| substrate_flap_0p10_stage_reflect_dense | 99/100 | 19/20 | 80/80 | 2.0399 | 0.0991 | 0.0000 |
| substrate_flap_0p05_observe_failure_redense | 66/100 | 13/20 | 53/80 | 5.1714 | 0.5616 | 0.0000 |
| substrate_flap_0p10_observe_failure_redense | 99/100 | 19/20 | 80/80 | 2.0602 | 0.1892 | 0.0000 |

结论：

```text
3% and 10% pass smoke.
10% is the current main candidate because it keeps 99/100 success and reduces
cost_per_success versus dense.
5% is anomalous: it has good controlled-prompt PPL but poor Agent success and a
large fallback step ratio. Do not run 5% 1000 before inspecting allocation and
failures.
At 10%, prefer stage-reflect-dense over observe-failure-redense because both are
99/100, but stage-reflect-dense uses fewer fallback steps.
```

下一步先用 audit 脚本看 5% 的 layer allocation 是否误伤关键层：

```bash
python scripts/audit_substrate_plans.py \
  --plan outputs/agent_qwen2_5_3b_4090_low/substrate_v2/flap/budget_plan_0p03.json \
  --name flap_0p03 \
  --plan outputs/agent_qwen2_5_3b_4090_low/substrate_v2/flap/budget_plan_0p05.json \
  --name flap_0p05 \
  --plan outputs/agent_qwen2_5_3b_4090_low/substrate_v2/flap/budget_plan_0p10.json \
  --name flap_0p10 \
  --plan outputs/agent_qwen2_5_3b_4090_low/substrate_v2/flap/budget_plan_0p20.json \
  --name flap_0p20 \
  --output-json outputs/agent_qwen2_5_3b_4090_low/substrate_v2/flap/plan_audit.json \
  --output-md outputs/agent_qwen2_5_3b_4090_low/substrate_v2/flap/plan_audit.md
```

20% pressure smoke 历史命令如下；该实验已经完成且失败，不需要重跑：

```bash
python -u scripts/evaluate_real_tool_loop.py \
  --config configs/agent_qwen2_5_3b_4090.yaml \
  --tasks data/agent/real_tool_tasks_v1_smoke_100.jsonl \
  --mask-bank-path outputs/agent_qwen2_5_3b_4090_low/mask_bank/mask_bank.json \
  --local-files-only \
  --substrate-plan outputs/agent_qwen2_5_3b_4090_low/substrate_v2/flap/budget_plan_0p20.json \
  --substrate-name flap_0p20 \
  --methods \
    dense \
    substrate_flap_0p20_stage_reflect_dense \
  --output-prefix outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_substrate_flap_0p20_smoke_100
```

20% pressure smoke 已完成，未通过主线门槛：

| Method | Success | Failure | Non-failure | Cost / success | Fallback step ratio | Schema validity |
|---|---:|---:|---:|---:|---:|---:|
| dense | 100/100 | 20/20 | 80/80 | 2.2000 | 0.0000 | 1.0000 |
| substrate_flap_0p20_stage_reflect_dense | 60/100 | 9/20 | 51/80 | 6.5194 | 0.6024 | 0.5247 |

20% 主要失败在 schema/tool protocol：

```text
unit_convert: 0/33
lookup: 26/33
calculator: 34/34
schema_error: 202
premature_final_rate: 0.67
```

评估代码审查没有发现会误判 20% 的 runner / parser / metrics bug。20% 不进入
1000 主表。

10% 主候选 1000 已经跑完：

| Method | Success | Failure | Non-failure | Cost / success | Fallback step ratio | Schema validity |
|---|---:|---:|---:|---:|---:|---:|
| dense | 997/1000 | 197/200 | 800/800 | 2.2066 | 0.0000 | 1.0000 |
| identity_hook | 997/1000 | 197/200 | 800/800 | 2.2066 | 0.0000 | 1.0000 |
| substrate_flap_0p10_stage_reflect_dense | 990/1000 | 195/200 | 795/800 | 2.0777 | 0.1156 | 0.9876 |
| substrate_flap_0p10_observe_failure_redense | 990/1000 | 195/200 | 795/800 | 2.0987 | 0.2114 | 0.9876 |

该结果可冻结为当前 P2 主结果：10% 不是无损，但稳定通过 1000，
cost_per_success 约降 5.8%，且 stage-reflect-dense 在相同成功率下比窗口式
observe-failure-redense 少约 45% dense fallback step。

历史命令如下；不要重复跑同一配置，除非需要复核：

```bash
python -u scripts/evaluate_real_tool_loop.py \
  --config configs/agent_qwen2_5_3b_4090.yaml \
  --tasks data/agent/real_tool_tasks_v1.jsonl \
  --mask-bank-path outputs/agent_qwen2_5_3b_4090_low/mask_bank/mask_bank.json \
  --local-files-only \
  --substrate-plan outputs/agent_qwen2_5_3b_4090_low/substrate_v2/flap/budget_plan_0p10.json \
  --substrate-name flap_0p10 \
  --methods \
    dense \
    identity_hook \
    substrate_flap_0p10_stage_reflect_dense \
    substrate_flap_0p10_observe_failure_redense \
  --output-prefix outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_substrate_flap_0p10_1000
```

不要把 `0p05` 或 `0p20` 加入 1000 主表。后续需要 nested budget ladder、
schema-aware calibration 和 stage-dependent substrate routing 后再重建这些档位。

### Historical v1 commands

Generate a 100-task smoke set:

```bash
export PYTHONPATH=$PWD:$PWD/src

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

Historical 1000-task command:

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
  --output-prefix outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_bypass_1000
```

The hidden-state centroid methods are not part of the frozen v1 default table yet:

```text
hidden_state_centroid_global_balanced_approx_0.01
hidden_state_centroid_reflect_dense_global_balanced_approx_0.01
hidden_state_centroid_event_reflect_dense_global_balanced_approx_0.01
```

For the first smoke run, it is acceptable to omit the centroid method and add it
after dense/identity/stage/failure are stable.

## 7. SliceGPT Reproduction

Use `docs/slicegpt_repro_commands.md`. Keep the external repository outside
this project or under an ignored workspace directory. Do not vendor it into
`src/`.
