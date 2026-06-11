# Agent FFN Pruning Handoff

日期：2026-06-11

本文用于新对话交接。新对话优先读本文，再按需读：

```text
docs/agent_mask_eval_results.md
docs/agent_ffn_progress_paths.md
```

## 1. 当前状态

当前主线是：

```text
Trajectory-Aware Dynamic FFN Pruning for Tool-Using Reasoning Agents
```

当前 Git 状态：

```text
branch: main
code baseline commit: 8846bb3 Add stage and failure agent mask eval
remote: https://github.com/Yuhang-dev/safeprune
local status: clean
```

远端路径：

```text
host: safeprune-4090
repo: /root/autodl-tmp/safeprune/code/Prune
venv: /root/autodl-tmp/safeprune/.venv_agent
HF cache: /root/autodl-tmp/safeprune/hf_cache
```

远端 repo 已 fast-forward 到 `origin/main`。远端仍有未跟踪实验产物和日志目录，这是正常的。

## 2. 已完成

### 2.1 工程化内容

已把临时 heredoc 小实验固化进正式代码：

```text
scripts/evaluate_agent_masks.py
src/safeprune/stage_masks.py
src/safeprune/evaluation.py
tests/test_stage_masks.py
tests/test_evaluation.py
```

正式 evaluator 现在支持：

```text
dense
identity_hook
full_stage_observe_0.01
safe13_observe_0.03
global_spread_observe_approx_0.01
global_balanced_observe_approx_0.01
global_concentrated_observe_approx_0.01
stage_global_balanced_approx_0.01
failure_redense_global_balanced_approx_0.01
```

新增能力：

```text
StageMaskBank.compose_layerwise_plan(stage, layer_sparsities)
active_mlp_ratio_from_plan(plan)
strict_agent_answer_correct(tool, expected, prediction)
routing_trace output for stage/failure dynamic methods
```

本地和远端测试均通过：

```text
python -m unittest discover -s tests
24 tests OK
```

### 2.2 关键实验结论

Qwen2.5-3B 对 naive full per-layer FFN channel mask 很敏感，但 stage/layer 信号有效。

低稀疏和 layer sensitivity 已证明：

```text
stage_observe_0.01 > static_plan_0.01
多数 single-layer 0.01 接近 dense
full 1% 失败主要来自 36 层同时剪枝的累积扰动
```

safe-layer-only 结果：

| Method | Correct | Active FFN ratio |
|---|---:|---:|
| dense | 97/100 | 1.0000 |
| identity_hook | 97/100 | 1.0000 |
| full_stage_observe_0.01 | 86/100 | 0.9900 |
| safe13_observe_0.01 | 96/100 | 0.9964 |
| safe13_observe_0.03 | 89/100 | 0.9892 |

global layer-wise 结果：

| Method | Correct | Active FFN ratio |
|---|---:|---:|
| full_stage_observe_0.01 | 86/100 | 0.9900 |
| global_spread_observe_approx_0.01 | 87/100 | 0.9900 |
| global_balanced_observe_approx_0.01 | 95/100 | 0.9900 |
| global_concentrated_observe_approx_0.01 | 95/100 | 0.9900 |
| dense | 97/100 | 1.0000 |

stage/failure 初步结果：

| Method | Correct | Active FFN ratio |
|---|---:|---:|
| dense | 97/100 | 1.0000 |
| identity_hook | 97/100 | 1.0000 |
| global_balanced_observe_approx_0.01 | 95/100 | 0.9900 |
| stage_global_balanced_approx_0.01 | 96/100 | 0.9900 |
| failure_redense_global_balanced_approx_0.01 | 97/100 | 0.9980 |

结论：

```text
per-layer fixed budget 是主要失败点。
global layer-wise allocation 是正确方向。
stage-aware dynamic plan 在同样 active ratio 下优于 observe-only global baseline。
failure re-densification 可以追平 dense，但计算成本上升。
```

注意：

```text
当前仍是 mask-based algorithm validation。
不能声称真实 wall-clock speedup。
latency 只作为运行记录，不作为压缩收益证据。
```

## 3. 远端产物

正式 global layer-wise：

```text
outputs/agent_qwen2_5_3b_4090_low/eval/global_layerwise_official_100.jsonl
outputs/agent_qwen2_5_3b_4090_low/eval/global_layerwise_official_100_metrics.json
outputs/agent_qwen2_5_3b_4090_low/eval/global_layerwise_official_100_plans.json
logs/global_layerwise_official_100_20260611_200517.log
```

正式 stage/failure v2：

```text
outputs/agent_qwen2_5_3b_4090_low/eval/stage_failure_official_v2_100.jsonl
outputs/agent_qwen2_5_3b_4090_low/eval/stage_failure_official_v2_100_metrics.json
outputs/agent_qwen2_5_3b_4090_low/eval/stage_failure_official_v2_100_plans.json
logs/stage_failure_official_v2_100_20260611_202211.log
```

输入产物：

```text
data/agent/controlled_tasks.jsonl
outputs/agent_qwen2_5_3b_4090_low/mask_bank/mask_bank.json
outputs/agent_qwen2_5_3b_4090_low/mask_bank/manifest.json
```

## 4. 复现实验命令

远端环境：

```bash
cd /root/autodl-tmp/safeprune/code/Prune
git pull --ff-only origin main
source /root/autodl-tmp/safeprune/.venv_agent/bin/activate

export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD/src
export HF_HOME=/root/autodl-tmp/safeprune/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/safeprune/hf_cache/hub
export HF_DATASETS_CACHE=/root/autodl-tmp/safeprune/hf_cache/datasets
export HF_HUB_DISABLE_XET=1
unset TRANSFORMERS_CACHE
```

全量测试：

```bash
python -m unittest discover -s tests
```

复跑 stage/failure 100 条：

```bash
python -u scripts/evaluate_agent_masks.py \
  --config configs/agent_qwen2_5_3b_4090.yaml \
  --mask-bank-path outputs/agent_qwen2_5_3b_4090_low/mask_bank/mask_bank.json \
  --max-tasks 100 \
  --local-files-only \
  --methods \
    dense \
    identity_hook \
    global_balanced_observe_approx_0.01 \
    stage_global_balanced_approx_0.01 \
    failure_redense_global_balanced_approx_0.01 \
  --output-prefix outputs/agent_qwen2_5_3b_4090_low/eval/stage_failure_official_v3_100
```

## 5. 下一步

下一步不是继续调 sparsity，也不是直接上 7B。下一步应正式化 Agent 指标。

### Step 1：把 evaluator metrics 补完整

在 `scripts/evaluate_agent_masks.py` 或 `src/safeprune/evaluation.py` 中补：

```text
task_success_rate
average_active_ffn_ratio
active_ffn_cost
cost_per_success
generation_collapse_rate
failure_task_success_rate
non_failure_task_success_rate
recovery_success_rate
```

当前 JSONL 已有 `routing_trace`，可直接从 trace 计算：

```text
是否经过 failure_event
是否使用 dense_fallback
每个 task 的平均 active_ffn_ratio
```

建议定义：

```text
failure task: 任一 step 的 event 不在 {ok, success, none}
recovery success: failure task 最终 strict_correct=True
cost_per_success: sum(task_active_ffn_ratio) / correct_count
```

### Step 2：跑 1000 条 controlled tasks

在 100 条结果稳定后，跑完整 1000 条：

```bash
python -u scripts/evaluate_agent_masks.py \
  --config configs/agent_qwen2_5_3b_4090.yaml \
  --mask-bank-path outputs/agent_qwen2_5_3b_4090_low/mask_bank/mask_bank.json \
  --local-files-only \
  --methods \
    dense \
    identity_hook \
    global_balanced_observe_approx_0.01 \
    stage_global_balanced_approx_0.01 \
    failure_redense_global_balanced_approx_0.01 \
  --output-prefix outputs/agent_qwen2_5_3b_4090_low/eval/stage_failure_official_1000
```

### Step 3：判断是否扩展

若 1000 条仍成立：

```text
stage_dynamic >= observe-only global baseline
failure_redense improves failure subset or cost_per_success tradeoff is clear
dense and identity_hook stay aligned
collapse_rate remains 0
```

再考虑：

```text
1. 更真实的 multi-step generation / tool execution
2. 7B confirmation
3. SliceGPT / Probe Pruning baseline
4. KD / compensation if target sparsity rises above 1%
```

## 6. 不要做

当前不要做：

```text
不要继续 full per-layer 20% / 30% mask。
不要重新尝试 attention head pruning。
不要声称 mask hook 有真实 wall-clock speedup。
不要直接上 7B 主实验。
不要把临时远端输出 vendoring 到 src/。
不要用 expected in prediction 这种 substring 判分。
```

## 7. 新对话建议首条指令

新对话可以直接说：

```text
读取 docs/agent_handoff_next_steps.md，
然后实现 Step 1：给 evaluate_agent_masks.py 补正式 Agent metrics，
包括 failure_task_success_rate、recovery_success_rate、active_ffn_cost 和 cost_per_success。
```
