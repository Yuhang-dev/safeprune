# Real Tool-Execution Agent Prune v1

日期：2026-06-13

本文冻结当前真实工具闭环结果。该实验不再是 controlled direct-answer 任务，而是：

```text
user request
-> model JSON tool_call
-> local Python tool execution
-> observation returned to model
-> model final answer
-> task success / recovery / active FFN cost metrics
```

本轮结果足够作为 **Real Tool-Execution Agent Prune v1**。不需要继续补同配置 1000 条实验。

## 1. 结论摘要

当前最强结论不是：

```text
stage-aware mask switching 全面优于 observe-only mask。
```

真实工具闭环中，这个强结论不成立。更准确的结论是：

```text
Agent failure recovery 是局部计算瓶颈。
failure / reflect 轨迹信号能够定位高价值计算步骤。
只在 reflect recovery step 做局部 re-densification，可以接近 dense 的失败恢复能力，
同时比固定窗口式 re-densification 使用更少 dense fallback step。
```

主结果支持两条线：

| 方法 | 定位 | 结论 |
|---|---|---|
| `observe_failure_redense_global_balanced_approx_0.01` | 最佳工程策略 | 与 dense 同为 997/1000，且 cost per success 更低。 |
| `stage_reflect_dense_global_balanced_approx_0.01` | 最有机制解释力 | failure recovery 达到 196/200，fallback step ratio 只有 0.0926。 |

## 2. 实验设置

模型：

```text
Qwen/Qwen2.5-3B-Instruct
```

任务：

```text
1000 real tool tasks
calculator: 334
lookup: 333
unit_convert: 333
deterministic failure tasks: 200
non-failure tasks: 800
```

工具闭环：

```text
calculator: AST whitelist arithmetic
unit_convert: mm / cm / m / km
lookup: PRJ-N -> owner_N
```

冻结产物：

```text
outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_bypass_1000.jsonl
outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_bypass_1000_metrics.json
outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_bypass_1000_plans.json
outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_bypass_1000_pairwise.json
outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_bypass_1000_failures.jsonl
outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_bypass_1000_analysis.json
outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_bypass_1000_analysis.md
```

## 3. 主表

| Method | Success | Failure | Non-failure | Cost / success | Fallback step ratio | Schema validity |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 997/1000 | 197/200 | 800/800 | 2.2066 | 0.0000 | 1.0000 |
| Identity hook | 997/1000 | 197/200 | 800/800 | 2.2066 | 0.0000 | 1.0000 |
| Observe-only | 951/1000 | 151/200 | 800/800 | 2.4349 | 0.0000 | 0.9987 |
| Stage-aware | 933/1000 | 133/200 | 800/800 | 2.6878 | 0.0000 | 0.9191 |
| Failure-redense | 996/1000 | 196/200 | 800/800 | 2.1938 | 0.1829 | 0.9977 |
| Observe-failure-redense | 997/1000 | 197/200 | 800/800 | 2.1886 | 0.1818 | 1.0000 |
| Stage-reflect-dense | 996/1000 | 196/200 | 800/800 | 2.1918 | 0.0926 | 0.9977 |
| Stage-reflect-observe | 946/1000 | 146/200 | 800/800 | 2.4551 | 0.0000 | 0.9953 |

关键事实：

```text
dense == identity_hook: 997/1000
所有方法 non-failure: 800/800
差异集中在 200 条 failure subset
```

因此本轮评测主要测到的是 **failure recovery robustness**，不是普通 non-failure tool-use 能力。

## 4. Fallback 对比

| Comparison | Value |
|---|---:|
| Failure-redense fallback step ratio | 0.1829 |
| Stage-reflect-dense fallback step ratio | 0.0926 |
| Relative reduction | 49.4% |

解释：

```text
固定窗口式 failure-redense 在失败后连续多个 step 使用 dense fallback。
stage-reflect-dense 只在 reflect recovery step 使用 dense。
```

两者 failure success 几乎相同：

```text
failure-redense:       196/200
stage-reflect-dense:   196/200
```

但 `stage-reflect-dense` 的 dense fallback step 约少一半。这是本轮最重要的机制证据。

## 5. Pairwise 结果

| Pair | Split | Both correct | Left only | Right only | Both wrong |
|---|---|---:|---:|---:|---:|
| stage-aware vs stage-reflect-dense | all | 930 | 3 | 66 | 1 |
| stage-aware vs stage-reflect-dense | failure | 130 | 3 | 66 | 1 |
| stage-aware vs stage-reflect-dense | non-failure | 800 | 0 | 0 | 0 |
| failure-redense vs stage-reflect-dense | all | 996 | 0 | 0 | 4 |
| failure-redense vs stage-reflect-dense | failure | 196 | 0 | 0 | 4 |
| observe-failure-redense vs stage-reflect-dense | all | 996 | 1 | 0 | 3 |
| observe-failure-redense vs stage-reflect-dense | failure | 196 | 1 | 0 | 3 |
| observe-only vs stage-reflect-dense | all | 947 | 4 | 49 | 0 |
| observe-only vs stage-reflect-dense | failure | 147 | 4 | 49 | 0 |
| stage-reflect-dense vs stage-reflect-observe | all | 943 | 53 | 3 | 1 |
| stage-reflect-dense vs stage-reflect-observe | failure | 143 | 53 | 3 | 1 |

解读：

```text
stage-reflect-dense 相比 stage-aware 净恢复 63 条任务，几乎全部来自 failure subset。
stage-reflect-dense 相比 observe-only 净恢复 45 条任务，也全部来自 failure subset。
stage-reflect-dense 和 failure-redense 在任务正确性上几乎等价，但 fallback step 更少。
```

## 6. Failure subset by tool

| Method | Calculator | Lookup | Unit convert |
|---|---:|---:|---:|
| Dense | 64/67 | 67/67 | 66/66 |
| Identity hook | 64/67 | 67/67 | 66/66 |
| Observe-only | 63/67 | 22/67 | 66/66 |
| Stage-aware | 67/67 | 0/67 | 66/66 |
| Failure-redense | 64/67 | 66/67 | 66/66 |
| Observe-failure-redense | 64/67 | 67/67 | 66/66 |
| Stage-reflect-dense | 64/67 | 66/67 | 66/66 |
| Stage-reflect-observe | 62/67 | 18/67 | 66/66 |

`lookup` 是 failure recovery 的主要压力源。普通 stage-aware 在 lookup failure subset 上退化到 `0/67`，说明 reflect recovery 阶段的 mask 严重损害了工具重试能力。`stage-reflect-dense` 将其恢复到 `66/67`。

## 7. Claim 修正

Real Tool v1 不支持下面这个强 claim：

```text
stage-aware mask switching 全面优于 observe-only routing。
```

原因：

```text
observe-only failure: 151/200
stage-aware failure: 133/200
```

应改写为：

```text
Normal tool-use steps tolerate lightweight FFN masking, but recovery steps are more fragile.
Failure and reflect signals expose localized high-value compute nodes.
Targeted reflect-step re-densification recovers near-dense robustness while using fewer fallback steps than fixed-window redensification.
```

中文表述：

```text
正常工具调用路径对轻量 FFN mask 具有较强容忍度；
失败恢复阶段更脆弱，是局部计算瓶颈。
轨迹中的 failure / reflect 信号可以指导计算预算分配，
只在 recovery reflect step 恢复 dense 即可接近窗口式回退的鲁棒性，并显著减少 fallback step。
```

## 8. 论文意义

Real Tool v1 可以作为机制验证结果：

```text
Different Agent steps have non-uniform compute sensitivity.
Failure recovery is not a global compute bottleneck; it is concentrated in reflect steps.
Trajectory-event-aware compute allocation can recover robustness without dense execution everywhere.
```

当前仍不能声称：

```text
真实 wall-clock speedup
kernel-level acceleration
compact FFN subnet 已部署
5%-10% 结构化稀疏率已经稳定
```

当前只能报告：

```text
active FFN cost
fallback step ratio
cost per successful task
task success / failure recovery
```

## 9. 下一步

优先级：

1. `[NEXT]` 加入 `hidden_state_centroid_global_balanced_approx_0.01`，作为 hidden-state-only 动态 router baseline。
2. `[NEXT]` 比较 hidden-state-only 与 failure / reflect 显式轨迹信号，回答外部 Agent signal 是否提供额外价值。
3. `[P2]` 启动底层剪枝率升级线：global budget ladder `3% / 5% / 10%`。
4. `[P2]` 引入 FLAP-style fluctuation scoring、global layer-wise budget optimizer、bias / scale compensation。
5. `[P3]` compact subnet 或 kernel 实现后，才能报告真实 latency / throughput。

Real Tool v1 到此冻结，不继续补同配置 1000 条。
