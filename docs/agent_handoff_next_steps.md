# Agent FFN Pruning Handoff

日期：2026-06-13

本文是当前新对话交接入口。优先读本文，再按需读：

```text
docs/real_tool_v1_results.md
docs/controlled_mask_validation_v1.md
docs/agent_mask_eval_results.md
docs/agent_ffn_progress_paths.md
docs/remote_workflow.md
```

## 0. 最新状态

当前已经完成并冻结：

```text
Controlled Mask Validation v1
Real Tool-Execution Agent Prune v1
```

Real Tool v1 的正式结果在：

```text
docs/real_tool_v1_results.md
outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_bypass_1000*
```

主结论已经从“stage-aware mask switching 全面优于 observe-only”收紧为：

```text
Agent failure recovery 是局部计算瓶颈。
failure / reflect 轨迹信号可以定位高价值计算步骤。
只在 reflect recovery step 恢复 dense，可接近窗口式 redense 的鲁棒性，
并将 dense fallback step ratio 从 0.1829 降到 0.0926。
```

主表关键结果：

| Method | Success | Failure | Non-failure | Cost / success | Fallback step ratio |
|---|---:|---:|---:|---:|---:|
| dense | 997/1000 | 197/200 | 800/800 | 2.2066 | 0.0000 |
| identity_hook | 997/1000 | 197/200 | 800/800 | 2.2066 | 0.0000 |
| observe-only | 951/1000 | 151/200 | 800/800 | 2.4349 | 0.0000 |
| stage-aware | 933/1000 | 133/200 | 800/800 | 2.6878 | 0.0000 |
| failure-redense | 996/1000 | 196/200 | 800/800 | 2.1938 | 0.1829 |
| observe-failure-redense | 997/1000 | 197/200 | 800/800 | 2.1886 | 0.1818 |
| stage-reflect-dense | 996/1000 | 196/200 | 800/800 | 2.1918 | 0.0926 |

下一步不要继续补同配置 1000。优先级是：

```text
1. 跑 hidden-state centroid baseline：pure centroid、reflect-dense centroid、event hybrid。
2. 实现 FLAP-style scoring、global layer-wise budget、bias/scale compensation。
3. 做 3% / 5% / 10% global FFN budget ladder。
4. compact subnet / kernel 后再报告真实加速。
```

## 1. 当前阶段

当前主线已经收紧为：

```text
Trajectory-Event-Aware FFN Compute Allocation for Tool-Using Reasoning Agents
```

阶段已经从 real tool smoke 转到：

```text
Real Tool v1 frozen -> hidden-state baseline / budget ladder
```

也就是从：

```text
用户问题 -> 模型直接回答 -> strict exact match
```

升级为：

```text
用户问题 -> 模型生成 JSON tool_call -> Python 工具真实执行
-> observation 回填 -> 模型 final answer -> 自动判分
```

当前工作不是继续补同配置 1000，也不是上 7B，而是验证：

```text
hidden-state-only centroid router 是否能替代显式 failure / reflect 轨迹信号
更高 global FFN budget 是否能在 Real Tool v1 稳定路由下成立
```

## 2. 为什么要这样做

1000 条 controlled mask validation 已经完成 MVP 机制验证：

| Method | Correct | Active FFN | Cost / success |
|---|---:|---:|---:|
| dense | 913/1000 | 1.0000 | 1.0953 |
| identity_hook | 913/1000 | 1.0000 | 1.0953 |
| global_balanced_observe_approx_0.01 | 836/1000 | 0.9900 | 1.1842 |
| stage_global_balanced_approx_0.01 | 898/1000 | 0.9900 | 1.1025 |
| failure_redense_global_balanced_approx_0.01 | 900/1000 | 0.9916 | 1.1018 |

关键结论：

```text
stage-aware 在同样 active FFN ratio 下比 observe-only 多成功 62 条。
failure-redense 在 failure subset 上达到 189/200，追平 dense。
但当前 cost_per_success 仍未优于 dense，且 mask hook 不代表真实加速。
```

controlled 任务还有一个明显问题：

```text
dense calculator: 258/334 = 77.2%
```

这说明旧任务混入了模型心算能力，不完全是在测 Agent 工具使用。因此下一步必须做真实工具执行闭环。

## 3. 已经做了什么

### 3.1 Controlled Mask Validation v1 冻结

新增：

```text
docs/controlled_mask_validation_v1.md
scripts/analyze_agent_pairwise_errors.py
```

用途：

```text
把 1000 条 controlled direct-answer 实验冻结为 MVP 机制验证结果；
提供 pairwise wins/losses、by-tool、failure subset 和错误样例拆分。
```

### 3.2 Real Tool-Execution Agent Prune v1 scaffold

新增：

```text
src/safeprune/tool_protocol.py
src/safeprune/tool_env.py
src/safeprune/agent_loop.py
src/safeprune/tools/
scripts/prepare_real_tool_tasks.py
scripts/evaluate_real_tool_loop.py
```

实现内容：

```text
严格 JSON tool protocol
本地 calculator / unit_convert / lookup 工具
确定性 fault_schedule failure injection
自然闭环 Agent runner
step-level FFN mask routing
dense / identity / observe-only / stage-aware / failure-redense 方法集
tool validity、schema validity、recovery、active_ffn_cost、cost_per_success 指标
```

当前工具协议只接受：

```json
{"type":"tool_call","name":"calculator","arguments":{"expression":"2 + 3"}}
```

或：

```json
{"type":"final","answer":"5"}
```

### 3.3 Hidden-state-only centroid baseline

新增：

```text
src/safeprune/hidden_state_router.py
```

方法名：

```text
hidden_state_centroid_global_balanced_approx_0.01
hidden_state_centroid_reflect_dense_global_balanced_approx_0.01
hidden_state_centroid_event_reflect_dense_global_balanced_approx_0.01
```

用途：

```text
作为第一版 hidden-state dynamic router 对照；
pure centroid 检查 hidden state 是否能隐式识别阶段；
reflect-dense centroid 与 stage_reflect_dense 做公平比较；
event hybrid 检查显式 failure / reflect 信号是否能补强 hidden state。
```

注意：

```text
centroid router 需要额外 prefill hidden-state 计算；
metrics 中单独记录 routing_probe_cost、routing_probe_latency_ms、
reflect_detection_precision / recall / f1 和 effective_cost_per_success。
```

## 4. 当前远端状态

远端环境：

```text
host: connect.nmb1.seetacloud.com
repo: /root/autodl-tmp/safeprune/code/Prune
venv: /root/autodl-tmp/safeprune/.venv_agent
HF cache: /root/autodl-tmp/safeprune/hf_cache
```

用户已在远端生成 smoke 任务：

```text
data/agent/real_tool_tasks_v1_smoke_100.jsonl
count: 100
deterministic failures: 20
```

首次运行 `evaluate_real_tool_loop.py` 出现：

```text
ModuleNotFoundError: No module named 'scripts'
```

原因：

```text
python scripts/evaluate_real_tool_loop.py 运行时，
Python 默认只把 scripts/ 放入 sys.path，
不会把 repo root 当成可导入包。
```

已在本地修复：

```text
scripts/evaluate_real_tool_loop.py 启动时自动把 repo root 加到 sys.path。
README.md 和 docs/remote_workflow.md 也已改为 export PYTHONPATH=$PWD:$PWD/src。
```

远端临时绕过方式：

```bash
cd /root/autodl-tmp/safeprune/code/Prune
export PYTHONPATH=$PWD:$PWD/src
```

然后重跑 smoke。

### 4.1 Real tool smoke 100 结果

远端 100 条 smoke 已跑完，但没有通过进入 1000 的门槛。

汇总：

| Method | Correct | Task success | Schema validity | Avg steps | Active FFN cost | Cost / success |
|---|---:|---:|---:|---:|---:|---:|
| dense | 66/100 | 0.66 | 0.813 | 3.42 | 342.0 | 5.18 |
| identity_hook | 66/100 | 0.66 | 0.813 | 3.42 | 342.0 | 5.18 |
| global_balanced_observe_approx_0.01 | 33/100 | 0.33 | 0.620 | 4.60 | 455.4 | 13.80 |
| stage_global_balanced_approx_0.01 | 47/100 | 0.47 | 0.570 | 4.61 | 456.4 | 9.71 |
| failure_redense_global_balanced_approx_0.01 | 66/100 | 0.66 | 0.813 | 3.42 | 342.0 | 5.18 |

分工具看，dense / identity 最大问题来自 `unit_convert`：

| Method | calculator | lookup | unit_convert |
|---|---:|---:|---:|
| dense | 32/34 | 29/33 | 5/33 |
| identity_hook | 32/34 | 29/33 | 5/33 |
| global_balanced_observe_approx_0.01 | 4/34 | 26/33 | 3/33 |
| stage_global_balanced_approx_0.01 | 0/34 | 26/33 | 21/33 |
| failure_redense_global_balanced_approx_0.01 | 32/34 | 29/33 | 5/33 |

判断：

```text
dense 与 identity_hook 完全一致，说明 hook 本身没有污染。
dense task_success_rate=0.66 < 0.90，schema_validity_rate=0.813 < 0.95。
因此当前不能跑 1000，必须先修协议和 runner。
```

已定位两个主要实现问题：

```text
1. 模型经常输出 {"type":"final","answer":400} 这种数字 final。
   旧 parser 要求 answer 必须是 string，导致 schema_error 和重复工具调用。

2. failure-redense 在第一轮把 event=start 当成 failure。
   结果 dense_fallback_rate=1.0，方法退化成全程 dense fallback。
```

已在本地修复：

```text
src/safeprune/tool_protocol.py:
  final answer 接受 string / number，并统一转成 string 判分。

src/safeprune/agent_loop.py:
  observation prompt 明确 ok=true 后直接 final，retryable failure 后再重试工具。

scripts/evaluate_real_tool_loop.py:
  failure-redense route 时不再把 start event 传成 failure。
```

这次运行偏慢是预期内的，但也被上述问题放大了。100 条、5 个方法一共约 1947 次
`generate`，而不是旧 controlled direct-answer 的单轮生成；schema_error 重试又进一步增加
了轮数。修复后仍会比 controlled 慢，但不应再因为数字 final 被拒而大量空转。

## 5. 当前应执行的命令

远端环境变量：

```bash
cd /root/autodl-tmp/safeprune/code/Prune
source /root/autodl-tmp/safeprune/.venv_agent/bin/activate

export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD:$PWD/src
export HF_HOME=/root/autodl-tmp/safeprune/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/safeprune/hf_cache/hub
export HF_DATASETS_CACHE=/root/autodl-tmp/safeprune/hf_cache/datasets
export HF_HUB_DISABLE_XET=1
unset TRANSFORMERS_CACHE
```

real tool smoke：

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

如果 smoke 通过，再生成和跑 1000：

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

## 6. Smoke 判断标准

先看：

```text
dense 与 identity_hook 是否完全或近似一致
dense task_success_rate 是否 >= 0.90
dense schema_validity_rate 是否 >= 0.95
collapse_rate 是否为 0
```

再看：

```text
stage_global_balanced_approx_0.01 是否优于 global_balanced_observe_approx_0.01
failure_redense_global_balanced_approx_0.01 是否提升 failure_task_success_rate
```

若 dense 自身低于 0.90，优先调 prompt/protocol，不要直接解释成 pruning 失败。

## 7. 不要做

当前不要做：

```text
不要继续 full per-layer 20% / 30% mask。
不要重新尝试 attention head pruning。
不要声称 mask hook 有真实 wall-clock speedup。
不要直接上 7B 主实验。
不要把远端输出 vendoring 到 src/。
不要跳过 dense/identity 对齐检查。
不要在 real tool smoke 不稳定时加入复杂工具或 stateful DB。
```

## 8. 新对话建议首条指令

```text
读取 docs/agent_handoff_next_steps.md，
然后从 docs/real_tool_v1_results.md 继续：
不要补同配置 1000；
优先跑 hidden_state_centroid_global_balanced_approx_0.01 对照，
再规划 3% / 5% / 10% global FFN budget ladder。
```
