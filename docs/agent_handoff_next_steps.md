# Agent FFN Pruning Handoff

日期：2026-06-13

本文是当前新对话交接入口。优先读本文，再按需读：

```text
docs/p2_substrate_and_p2c_results.md
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
P1 hidden-state centroid baseline 1000
P2 FLAP 10% substrate 1000
P2c schema-aware nested adaptive routing 1000
```

当前下一步：

```text
标准 WikiText-2 / PIQA / HellaSwag / ARC-Easy sanity
compact FFN subnet / kernel for real wall-clock speedup
```

P2c 100-smoke 已通过：

| Method | Success | Failure | Non-failure | Cost / success | Schema validity |
|---|---:|---:|---:|---:|---:|
| dense | 100/100 | 20/20 | 80/80 | 2.2000 | 1.0000 |
| nested_uniform_10 | 100/100 | 20/20 | 80/80 | 2.0000 | 1.0000 |
| adaptive_A | 100/100 | 20/20 | 80/80 | 2.0000 | 1.0000 |
| adaptive_B15 | 100/100 | 20/20 | 80/80 | 1.9500 | 1.0000 |
| adaptive_B20 | 100/100 | 20/20 | 80/80 | 1.9000 | 1.0000 |

`adaptive_B20` routing trace 已确认：

```text
tool_call_or_retry -> nested_0p10
reflect_recovery   -> reflect_dense
final_answer       -> nested_0p20
```

P2c 1000 使用了最小解释矩阵：

```text
dense
identity_hook
nested_uniform_10
adaptive_B20
```

不要只跑 `adaptive_B20`，否则无法证明 generation-type routing 相比统一 10%
nested budget 的增益。

P2c 1000 已完成：

| Method | Success | Failure | Non-failure | Cost / success | Schema validity |
|---|---:|---:|---:|---:|---:|
| dense | 997/1000 | 197/200 | 800/800 | 2.2066 | 1.0000 |
| identity_hook | 997/1000 | 197/200 | 800/800 | 2.2066 | 1.0000 |
| nested_uniform_10 | 997/1000 | 197/200 | 800/800 | 2.0060 | 1.0000 |
| adaptive_B20 | 997/1000 | 197/200 | 800/800 | 1.9057 | 1.0000 |

解释：

```text
adaptive_B20 与 dense 成功率完全一致：997/1000。
相对 dense，cost_per_success 下降约 13.6%。
相对 nested_uniform_10，cost_per_success 进一步下降约 5.0%。
schema_validity_rate、premature_final_rate、generation_collapse_rate 均正常。
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

Real Tool v1 不要继续补同配置 1000。历史优先级中 P1/P2/P2c 已完成，现在优先级是：

```text
1. 补标准 WikiText-2 / PIQA / HellaSwag / ARC-Easy sanity。
2. 打包 P2c 1000 输出并更新论文表。
3. compact subnet / kernel 后再报告真实加速。
```

当前 P2 substrate v2 已开始，新增远端入口：

```text
scripts/build_pruning_substrate_v2.py
scripts/evaluate_pruning_ppl.py
scripts/evaluate_real_tool_loop.py --substrate-plan / --substrate-name
```

P2 第一阶段目标已经完成：

```text
不要继续手工调 1%。
已构建 activation / Wanda-channel / FLAP-style 的 1/3/5/10/20% budget plans。
已完成 controlled prompt PPL sanity，并把 3/5/10% 接回 Real Tool 100 smoke。
```

P2 controlled prompt PPL sanity 已完成：

| Plan | Active MLP ratio | PPL | Relative PPL increase |
|---|---:|---:|---:|
| dense | 1.0000 | 23.2031 | 0.00% |
| flap 3% | 0.9697 | 23.6021 | +1.72% |
| flap 5% | 0.9497 | 23.6440 | +1.90% |
| flap 10% | 0.8998 | 24.7785 | +6.79% |
| flap 20% | 0.7998 | 24.0066 | +3.46% |

注意：

```text
这不是标准 WikiText-2 PPL，而是 controlled-task prompt PPL sanity。
flap plans 已包含 bias compensation。
layer scale 目前只是 1.0 placeholder，还没有做真正 scale calibration。
20% PPL 不比 10% 差，说明 global layer-wise allocation 不是简单单调剪枝；
但 20% 仍先作为压力 smoke，不直接进正式 1000。
```

P2 Real Tool 100 smoke 已完成：

| Method | Success | Failure | Non-failure | Cost / success | Fallback step ratio | Collapse |
|---|---:|---:|---:|---:|---:|---:|
| dense | 100/100 | 20/20 | 80/80 | 2.2000 | 0.0000 | 0.0000 |
| flap 3% stage-reflect-dense | 100/100 | 20/20 | 80/80 | 2.1395 | 0.0909 | 0.0000 |
| flap 5% stage-reflect-dense | 66/100 | 13/20 | 53/80 | 5.1615 | 0.5244 | 0.0000 |
| flap 10% stage-reflect-dense | 99/100 | 19/20 | 80/80 | 2.0399 | 0.0991 | 0.0000 |
| flap 5% observe-failure-redense | 66/100 | 13/20 | 53/80 | 5.1714 | 0.5616 | 0.0000 |
| flap 10% observe-failure-redense | 99/100 | 19/20 | 80/80 | 2.0602 | 0.1892 | 0.0000 |

当前判断：

```text
3% 已通过，是安全档位。
10% 已通过，是当前主论文候选档位；stage-reflect-dense 达到 99/100，
failure subset 19/20，cost_per_success 低于 dense。
5% 是异常档位；PPL 很好但 Real Tool smoke 大幅退化，不能跑 5% 1000。
10% 下 stage-reflect-dense 比 observe-failure-redense fallback step ratio 更低，
因此优先作为 P2 主方法。
```

P2 plan audit 和 20% pressure smoke 也已完成。

Audit 结论：

```text
plan 文件本身没有明显损坏：
  duplicate idx = 0
  out-of-range idx = 0
  bias layers = 36
  scale layers = 36

真正问题是 allocator 非嵌套：
  flap_0p05_vs_flap_0p10: 5% 有 9353 个 channel 不在 10% 里
  flap_0p03_vs_flap_0p10: 3% 有 4400 个 channel 不在 10% 里

5% 异常不是 runner bug，也不是 plan loader 基础 bug；
它是当前独立 budget search 找到了一组坏 allocation。
```

20% pressure smoke 结果：

| Method | Success | Failure | Non-failure | Cost / success | Fallback step | Schema validity |
|---|---:|---:|---:|---:|---:|---:|
| dense | 100/100 | 20/20 | 80/80 | 2.2000 | 0.0000 | 1.0000 |
| flap 20% stage-reflect-dense | 60/100 | 9/20 | 51/80 | 6.5194 | 0.6024 | 0.5247 |

20% 失败形态：

```text
unit_convert: 0/33
lookup: 26/33
calculator: 34/34
schema_error: 202
premature_final_rate: 0.67
dense vs 20% pairwise: 20%_only_correct = 0
```

评估代码审查结论：

```text
未发现会把 20% 误判为失败的 runner / parser / metrics bug。
tool protocol 严格拒绝 {"type":"unit_convert", ...} 是正确行为；
正式协议只接受 {"type":"tool_call","name":"unit_convert","arguments":{...}}。
全量单测通过：76 tests OK, 3 skipped。
```

P2 10% Real Tool 1000 已完成：

| Method | Success | Failure | Non-failure | Cost / success | Fallback step | Schema validity |
|---|---:|---:|---:|---:|---:|---:|
| dense | 997/1000 | 197/200 | 800/800 | 2.2066 | 0.0000 | 1.0000 |
| identity_hook | 997/1000 | 197/200 | 800/800 | 2.2066 | 0.0000 | 1.0000 |
| substrate flap 10% stage-reflect-dense | 990/1000 | 195/200 | 795/800 | 2.0777 | 0.1156 | 0.9876 |
| substrate flap 10% observe-failure-redense | 990/1000 | 195/200 | 795/800 | 2.0987 | 0.2114 | 0.9876 |

关键解释：

```text
10% stage-reflect-dense 比 dense 少 7 条成功任务：997 -> 990。
但 cost_per_success 从 2.2066 降到 2.0777，约降低 5.8%。
active_ffn_cost 从 2200.0 降到 2056.9，约降低 6.5%。
与 observe-failure-redense 相比，成功率相同，但 fallback step ratio 从 0.2114 降到 0.1156，
dense fallback step 约少 45%。
failure recovery 没有崩：195/200；generation_collapse_rate = 0。
```

P1 hidden-state baseline 1000 也已完成：

| Method | Success | Failure | Non-failure | Cost / success | Router-inclusive cost | Fallback step |
|---|---:|---:|---:|---:|---:|---:|
| dense | 997/1000 | 197/200 | 800/800 | 2.2066 | 2.2066 | 0.0000 |
| observe-failure-redense | 997/1000 | 197/200 | 800/800 | 2.1886 | 2.1886 | 0.1818 |
| stage-reflect-dense | 996/1000 | 196/200 | 800/800 | 2.1918 | 2.1918 | 0.0926 |
| hidden centroid | 980/1000 | 180/200 | 800/800 | 2.3093 | 4.6420 | 0.0000 |
| hidden centroid reflect-dense | 983/1000 | 183/200 | 800/800 | 2.2609 | 4.5437 | 0.0392 |
| hidden centroid + event | 996/1000 | 196/200 | 800/800 | 2.1918 | 4.4036 | 0.0926 |

解释：

```text
hidden-state-only 不如显式 stage/event route。
event hybrid 能追平 stage-reflect-dense，但要额外 hidden-state prefill，
router-inclusive cost 约为 4.40，高于 stage-reflect-dense 的 2.19。
因此 P1 是审稿 baseline，不是主方法。
```

当前下一步固定为：

```text
1. 不跑 5% 1000。
2. 不跑 20% 1000。
3. 10% Real Tool 1000 已经完成，可冻结为 P2 当前主结果。
4. 后续 P2b/P2c 做 nested budget ladder + schema-aware calibration，修 5%/20%。
5. 再实现 stage-dependent substrate routing：tool-call safe budget、answer aggressive budget、
   reflect dense，而不是继续跑统一 20%。
```

P2c 已新增代码入口：

```text
scripts/prepare_schema_calibration.py
scripts/build_pruning_substrate_v2.py --nested-budget-ladder --schema-calibration-path
scripts/evaluate_real_tool_loop.py methods:
  nested_uniform_10
  adaptive_A
  adaptive_B15
  adaptive_B20
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

当前工作已经越过 hidden-state 和 3B P2c 验证阶段。现在要验证：

```text
P2c 是否保持通用 benchmark 能力
Adaptive-B20 是否能扩展到 Qwen2.5-7B smoke
active FFN cost 是否能通过 compact subnet 转成真实 latency 收益
```

最新重要修复：

```text
_empty_plan_like() 曾经只清空 pruned_mlp_channels，没有清空
mlp_output_bias_compensation / mlp_output_scale。
因此从带 bias compensation 的 substrate plan 派生出的 identity_hook
可能不是严格 dense，而是 Dense + stale bias compensation。
```

修复已完成：

```text
scripts/evaluate_agent_masks.py
tests/test_evaluate_real_tool_loop.py
```

这不会直接否定 Adaptive-B20 的机制，因为 10% / 20% substrate plan 的
bias compensation 是设计内行为；但所有依赖 `identity_hook == dense` 或
`reflect -> dense fallback` 的 sanity 结论都必须在修复后重跑确认。

当前下一步不要跑 7B 100/1000。先做：

```text
1. 3B P2c 最小矩阵 sanity rerun。
2. 7B dense / identity_hook repeat gate。
3. 7B 20-task minismoke matrix。
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
git pull origin main
source /root/autodl-tmp/safeprune/.venv_agent/bin/activate

export PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD:$PWD/src
export HF_HOME=/root/autodl-tmp/safeprune/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/safeprune/hf_cache/hub
export HF_DATASETS_CACHE=/root/autodl-tmp/safeprune/hf_cache/datasets
export HF_HUB_DISABLE_XET=1
unset TRANSFORMERS_CACHE
```

确认正式 1000 任务存在；如果没有再生成：

```bash
test -f data/agent/real_tool_tasks_v1.jsonl || \
python scripts/prepare_real_tool_tasks.py \
  --output data/agent/real_tool_tasks_v1.jsonl \
  --count 1000 \
  --failure-count 200
```

当前主命令：P2c schema-aware nested budget + generation-type routing。

生成 schema-heavy calibration snippets：

```bash
python scripts/prepare_schema_calibration.py \
  --tasks data/agent/real_tool_tasks_v1.jsonl \
  --output data/agent/schema_calibration_v1.jsonl \
  --max-tasks 500
```

构建 nested ladder：

```bash
python scripts/build_pruning_substrate_v2.py \
  --config configs/agent_qwen2_5_3b_4090.yaml \
  --calibration-path data/agent/controlled_tasks.jsonl \
  --schema-calibration-path data/agent/schema_calibration_v1.jsonl \
  --max-calibration-prompts 512 \
  --max-schema-calibration-prompts 1500 \
  --target-budgets 0.05,0.10,0.12,0.15,0.18,0.20 \
  --score-methods flap \
  --nested-budget-ladder \
  --schema-token-weight 1.0 \
  --with-bias-compensation \
  --with-layer-scale-placeholder \
  --local-files-only \
  --output-dir outputs/agent_qwen2_5_3b_4090_low/substrate_v2_schema_nested
```

查看 nested audit：

```bash
cat outputs/agent_qwen2_5_3b_4090_low/substrate_v2_schema_nested/flap/nested_budget_audit.md
```

跑 controlled prompt PPL sanity：

```bash
for b in 0p05 0p10 0p12 0p15 0p18 0p20; do
  python scripts/evaluate_pruning_ppl.py \
    --config configs/agent_qwen2_5_3b_4090.yaml \
    --plan outputs/agent_qwen2_5_3b_4090_low/substrate_v2_schema_nested/flap/budget_plan_${b}.json \
    --prompts-jsonl data/agent/controlled_tasks.jsonl \
    --max-prompts 256 \
    --local-files-only \
    --output outputs/agent_qwen2_5_3b_4090_low/substrate_v2_schema_nested/flap/ppl_${b}.json
done
```

跑 4 组 Real Tool 100 smoke：

```bash
python -u scripts/evaluate_real_tool_loop.py \
  --config configs/agent_qwen2_5_3b_4090.yaml \
  --tasks data/agent/real_tool_tasks_v1_smoke_100.jsonl \
  --mask-bank-path outputs/agent_qwen2_5_3b_4090_low/mask_bank/mask_bank.json \
  --local-files-only \
  --substrate-plan outputs/agent_qwen2_5_3b_4090_low/substrate_v2_schema_nested/flap/budget_plan_0p05.json \
  --substrate-name nested_0p05 \
  --substrate-plan outputs/agent_qwen2_5_3b_4090_low/substrate_v2_schema_nested/flap/budget_plan_0p10.json \
  --substrate-name nested_0p10 \
  --substrate-plan outputs/agent_qwen2_5_3b_4090_low/substrate_v2_schema_nested/flap/budget_plan_0p15.json \
  --substrate-name nested_0p15 \
  --substrate-plan outputs/agent_qwen2_5_3b_4090_low/substrate_v2_schema_nested/flap/budget_plan_0p20.json \
  --substrate-name nested_0p20 \
  --methods \
    dense \
    nested_uniform_10 \
    adaptive_A \
    adaptive_B15 \
    adaptive_B20 \
  --output-prefix outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_p2c_adaptive_smoke_100
```

打印 smoke 主指标：

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_p2c_adaptive_smoke_100_metrics.json")
metrics = json.loads(p.read_text())
keys = [
    "correct", "total", "failure_task_correct", "failure_task_total",
    "non_failure_task_correct", "non_failure_task_total",
    "cost_per_success", "fallback_step_ratio", "schema_validity_rate",
    "premature_final_rate", "generation_collapse_rate",
]
for method, row in metrics.items():
    print(method)
    for key in keys:
        print(f"  {key}: {row.get(key)}")
PY
```

## 6. P2c smoke 判断标准

先看：

```text
overall >= 98/100
failure >= 19/20
unit_convert >= 32/33
schema_validity >= 0.98
fallback_step_ratio 明显低于 1
generation_collapse_rate = 0
```

若 `adaptive_B20` 不稳，补跑 `adaptive_B15` 即可；不要扩大到 1000。

这条判断已完成：`adaptive_B20` 1000 稳定，已冻结为 P2c 主结果。

注意：P2c 主结果是在 `_empty_plan_like()` 修复前跑出的。由于 3B 的
`dense` 与 `identity_hook` 当时任务级结果一致，主结论大概率不变；但
论文级最终数字应以修复后的 sanity rerun 为准。

## 7. 不要做

当前不要做：

```text
不要继续 full per-layer 20% / 30% mask。
不要重新尝试 attention head pruning。
不要声称 mask hook 有真实 wall-clock speedup。
不要直接上 7B 1000；先跑 7B Real Tool 100 smoke。
不要在修复后的 dense / identity_hook gate 通过前跑 7B Real Tool 100 smoke。
不要把远端输出 vendoring 到 src/。
不要跳过 dense/identity 对齐检查。
不要在 real tool smoke 不稳定时加入复杂工具或 stateful DB。
不要把旧独立 5% 或旧 uniform 20% 加入当前主表。
```

## 8. 新对话建议首条指令

```text
读取 docs/agent_handoff_next_steps.md，
然后继续 P2：
5% allocation anomaly 已完成 audit；
20% pressure smoke 已失败；
10% Real Tool 1000 已完成，substrate_flap_0p10_stage_reflect_dense 为当前 P2 主结果；
下一步实现 schema-aware nested budget plans 和 stage-dependent substrate routing，
不要继续跑统一 20%。
```
