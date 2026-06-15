# Agent Prune 论文实验总账

更新日期：2026-06-15

本文是当前项目用于论文写作、外部复核和结果追溯的统一实验说明。它回答：

1. 每一阶段做了什么实验；
2. 为什么需要该实验；
3. 方法和评测协议是什么；
4. 得到了什么数据；
5. 数据支持什么结论、不支持什么结论；
6. 每组实验在论文中承担什么角色。

详细命令、逐任务输出和历史调试过程仍保留在其他文档中。论文数字和 claim
应优先以本文及其引用的冻结文档为准。

## 1. 研究问题与最终方法

项目最初研究通用结构化剪枝，随后收敛到：

```text
Trajectory-event-aware FFN compute allocation for tool-using language agents.
```

核心观察是 Agent 轨迹中的计算敏感度并不均匀：

```text
tool_call_or_retry -> schema-sensitive，使用保守 sparse FFN
final_answer       -> schema risk 较低，可使用更激进的 FFN pruning
reflect_recovery   -> failure-sensitive，临时恢复 dense FFN
```

最终主方法 `Adaptive-B20` 的路由是：

```text
tool_call_or_retry -> schema-aware nested 10% plan
final_answer       -> schema-aware nested 20% plan
reflect_recovery   -> dense plan
```

本文中的百分比表示被删除的 FFN intermediate channels 比例。10% plan 的
active MLP ratio 约为 0.90，20% plan 的 active MLP ratio 约为 0.80。

## 2. 统一实验设置

### 2.1 模型

主实验模型：

```text
Qwen/Qwen2.5-3B-Instruct
Qwen/Qwen2.5-7B-Instruct
```

早期工程可行性实验还使用过 Qwen2.5-1.5B-Instruct，但它不进入论文主结果。

### 2.2 Real Tool 数据

正式 Real Tool 任务包含三个确定性本地工具：

| Tool | 行为 |
|---|---|
| `calculator` | 通过 AST 白名单执行数字、括号和 `+ - * /`。 |
| `unit_convert` | 在 `mm/cm/m/km` 之间执行确定性单位换算。 |
| `lookup` | 执行 `PRJ-N -> owner_N` 的本地查询。 |

1000-task 正式集包含：

```text
calculator: 约 1/3
unit_convert: 约 1/3
lookup: 约 1/3
failure tasks: 200
non-failure tasks: 800
```

failure task 使用 task 内固定 `fault_schedule`，注入：

```text
timeout
empty_observation
tool_error
```

同一任务在不同方法之间复用同一 failure schedule，保证 pairwise 比较公平。

### 2.3 严格工具协议

Agent 只能生成两种顶层 JSON：

```json
{"type":"tool_call","name":"calculator","arguments":{"expression":"2 + 3"}}
```

或：

```json
{"type":"final","answer":"5"}
```

在任务要求工具执行时，成功工具 observation 前的 `final` 会被 runner guard
拒绝并记录为：

```text
premature_final
```

该 observation 会回填给模型，下一轮进入 `reflect_recovery`。主评测没有使用
tolerant parser，没有自动修复 tool name、枚举值或额外字段。

### 2.4 Protocol 版本

| Version | 变化 | 是否改变 strict parser |
|---|---|---|
| v1 | 初始真实工具闭环和 strict JSON protocol。 | 否 |
| v1.1 | 明确 `type="tool_call"`，工具名必须放在 `name`；schema error 返回正确 wrapper 示例。 | 否 |
| v1.2 | 成功 observation 后明确要求精确复制工具结果，特别是 lookup 的 `owner_N`。 | 否 |

v1.1 和 v1.2 都是 prompt/feedback hardening，没有放宽 benchmark 判分。

### 2.5 FFN 通道与 scoring

Qwen 使用 SwiGLU：

```text
a = SiLU(gate_proj(x)) * up_proj(x)
FFN(x) = down_proj(a)
```

早期 scoring 统计过 `gate_proj(x)`，后来修正为从 `down_proj` input pre-hook
采集真实中间激活 `a`。

Substrate v2 使用的主要分数：

```text
Activation:
  score_j = activation statistic of a_j

Wanda-channel:
  score_j = ||W_down[:, j]||_2 * sqrt(E[a_j^2])

FLAP-style fluctuation:
  score_j = ||W_down[:, j]||_2^2 * Var(a_j)
```

删除通道的均值补偿为：

```text
bias_compensation = W_down[:, R] * mean(a_R)
```

其中 `R` 是被删除通道集合。layer scale 当前仍是 `1.0` placeholder，不应写成
已完成的 scale calibration。

### 2.6 核心指标

每次 generation 的 active FFN ratio 由当前 plan 决定。主要成本指标：

```text
active_ffn_cost
  = sum of active MLP ratios across all generation steps

cost_per_success
  = total active_ffn_cost / successful episodes
```

其他核心指标：

```text
task_success_rate
failure_task_success_rate
non_failure_task_success_rate
schema_validity_rate
recovery_success_rate
fallback_step_ratio
premature_final_rate
generation_collapse_rate
router_inclusive_cost_per_success
```

`active_ffn_cost` 是算法成本代理，不等于 wall-clock latency。

## 3. 实验证据总览

| Phase | 研究问题 | 主要结果 | 论文角色 |
|---|---|---|---|
| Legacy 1.5B smoke | 原始 mixed structured pruning 是否可行？ | 10%-25% mixed pruning 严重退化或 collapse。 | 负结果和工程排雷。 |
| Early 3B masks | naive per-layer FFN mask 为什么失败？ | 单层通常安全，36 层统一剪枝产生累积扰动。 | 推导 global layer-wise budget。 |
| Controlled v1 | stage/failure signal 是否有机制价值？ | stage-aware 898/1000 vs observe-only 836/1000；failure-redense 900/1000。 | 机制前置实验。 |
| Real Tool v1 | 真实 failure recovery 是否局部敏感？ | stage-reflect-dense 996/1000，约一半 fixed-window fallback steps。 | failure localization 主证据。 |
| P1 router baseline | hidden-state-only 能否替代显式 event？ | hidden-only 980/1000；event route 996/1000 且无 probe 开销。 | reviewer baseline。 |
| P2 substrate | 能否从约 1% 推到 10%？ | 10% stage-reflect-dense 990/1000，cost/success 降约 5.8%。 | 有意义预算验证。 |
| P2c 3B | generation type 是否能使用不同预算？ | Post-fix v1.2 Adaptive-B20 986/1000，failure 200/200，cost/success 降约 13.85%。 | 跨模型对齐结果；有效但非 Dense-equivalent。 |
| P4 7B | 方法是否扩展到更大模型？ | Adaptive-B20 1000/1000，cost/success 降约 15.1%。 | 当前论文主结果。 |
| Static sanity | 动态 20% 是否等于安全静态 20%？ | 20% static PPL 和多选准确率明显退化。 | 证明动态路由必要。 |
| P5 compact | active cost 能否直接转为速度？ | MLP-only 有信号，full-model fixed-decode 无显著加速。 | 部署限制分析。 |

## 4. Phase 0：旧 SafePrune-DPO 工程可行性

### 为什么做

在转向 Agent 之前，项目先验证远端训练、剪枝、LoRA recovery 和生成检查链路。

### 方法

Qwen2.5-1.5B 上比较：

```text
25% mixed attention-head + MLP pruning
10% mixed attention-head + MLP pruning
5% MLP-only pruning
10% MLP-only pruning
```

### 结果

| Setting | 结果 |
|---|---|
| 25% mixed | 完全 collapse，recovery 后仍 collapse。 |
| 10% mixed | 接近 collapse。 |
| 5% MLP-only | 不 collapse，但能力退化，recovery 只部分改善。 |
| 10% MLP-only | 训练指标看似健康，生成仍明显退化。 |

### 结论

1. attention-head pruning 在该 setup 下高度不稳定；
2. training loss 不能替代真实 generation 检查；
3. mask-based 实验不能声称真实结构压缩或速度收益；
4. 后续主线应先聚焦 FFN-only、低预算和 Agent 行为。

### 论文角色

该阶段是工程排雷，不进入主结果表。详细记录见
`docs/4090_smoke_results.md`。

## 5. Phase 1：早期 3B FFN mask 诊断

### 为什么做

需要判断性能下降来自 stage signal 无效，还是来自每层统一剪枝策略过于粗糙。

### 方法

1. 修正 SwiGLU activation scoring；
2. 使用严格 exact-match，而不是 substring 判分；
3. 扫描 1%-3% naive per-layer mask；
4. 比较不同 stage mask；
5. 做 single-layer sensitivity；
6. 构造 safe-layer-only 和 global layer-wise budget。

### 关键结果

低稀疏 100-task：

| Method | Success |
|---|---:|
| Dense | 93/100 |
| Identity hook | 93/100 |
| Static per-layer 1% | 65/100 |
| Stage-observe per-layer 1% | 79/100 |
| Stage-answer per-layer 1% | 72/100 |
| Stage-answer per-layer 2% | 53/100 |
| Stage-answer per-layer 3% | 46/100 |

single-layer sensitivity：

```text
dense: 49/50
most single-layer observe 1%: 48/50 or 49/50
worst single layer: 45/50
all-layer observe 1%: 38/50
```

global allocation 100-task：

| Method | Success | Active FFN |
|---|---:|---:|
| Dense | 97/100 | 1.0000 |
| Full per-layer observe 1% | 86/100 | 0.9900 |
| Safe13 observe 3% | 89/100 | 0.9892 |
| Global balanced observe about 1% | 95/100 | 0.9900 |
| Global concentrated observe about 1% | 95/100 | 0.9900 |

### 结论

stage activation 中存在真实信号，但 naive per-layer fixed budget 是主要失败点。
多数层单独剪枝接近 dense，所有层同时剪枝的累积扰动造成明显退化。该结果直接推动：

```text
per-layer fixed budget
-> global layer-wise allocation
-> FLAP-style substrate and compensation
```

### 论文角色

适合作为方法动机或附录诊断，不作为最终 Agent benchmark。

## 6. Phase 2：Controlled Mask Validation v1

### 为什么做

在引入真实 parser、tool execution 和 observation 前，先隔离两个机制问题：

1. stage-aware routing 是否优于 observe-only；
2. failure-triggered re-densification 是否恢复 failure subset。

### 方法

1000 条 direct-answer controlled tasks，比较：

```text
dense
identity_hook
global_balanced_observe_approx_0.01
stage_global_balanced_approx_0.01
failure_redense_global_balanced_approx_0.01
```

### 结果

| Method | Success | Failure | Non-failure | Active FFN | Cost / success |
|---|---:|---:|---:|---:|---:|
| Dense | 913/1000 | 189/200 | 724/800 | 1.0000 | 1.0953 |
| Identity hook | 913/1000 | 189/200 | 724/800 | 1.0000 | 1.0953 |
| Observe-only | 836/1000 | 178/200 | 658/800 | 0.9900 | 1.1842 |
| Stage-aware | 898/1000 | 187/200 | 711/800 | 0.9900 | 1.1025 |
| Failure-redense | 900/1000 | 189/200 | 711/800 | 0.9916 | 1.1018 |

### 结论

```text
stage-aware - observe-only = +62 successful tasks
failure-redense failure subset = dense failure subset = 189/200
```

因此 stage 和 failure signal 有机制价值。但该实验仍混入模型心算和直接回答能力，
且 cost per success 没有优于 dense。

### 论文角色

作为 controlled mechanism validation。冻结文档：
`docs/controlled_mask_validation_v1.md`。

## 7. Phase 3：Real Tool-Execution Agent Prune v1

### 为什么做

Controlled dense calculator 只有 `258/334`，说明直接回答任务混入模型心算能力。
真实 Agent 评测必须要求模型生成 tool call、实际执行工具并消费 observation。

### Runner 方法

自然闭环：

```text
prompt
-> JSON tool call
-> local tool execution
-> observation
-> retry or final
```

runner 增加：

```text
strict parser
fault_schedule
premature_final guard
reflect routing
step-level selected plan trace
failure/non-failure and fallback metrics
```

### 初始 smoke 暴露的问题

第一版 dense 只有 `66/100`，不是剪枝结论。原因包括：

1. numeric final 被错误拒绝；
2. successful observation 后 prompt 不够明确；
3. failure-redense 将初始 `start` 事件误判为 failure，退化成全程 dense。

修复后进入 bypass diagnosis。

### Bypass 1000 结果

| Method | Success | Failure | Non-failure | Cost / success | Fallback step | Schema validity |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 997/1000 | 197/200 | 800/800 | 2.2066 | 0.0000 | 1.0000 |
| Identity hook | 997/1000 | 197/200 | 800/800 | 2.2066 | 0.0000 | 1.0000 |
| Observe-only | 951/1000 | 151/200 | 800/800 | 2.4349 | 0.0000 | 0.9987 |
| Stage-aware | 933/1000 | 133/200 | 800/800 | 2.6878 | 0.0000 | 0.9191 |
| Failure-redense | 996/1000 | 196/200 | 800/800 | 2.1938 | 0.1829 | 0.9977 |
| Observe-failure-redense | 997/1000 | 197/200 | 800/800 | 2.1886 | 0.1818 | 1.0000 |
| Stage-reflect-dense | 996/1000 | 196/200 | 800/800 | 2.1918 | 0.0926 | 0.9977 |
| Stage-reflect-observe | 946/1000 | 146/200 | 800/800 | 2.4551 | 0.0000 | 0.9953 |

failure subset 的主要压力来自 lookup：

| Method | Lookup failure success |
|---|---:|
| Dense | 67/67 |
| Observe-only | 22/67 |
| Stage-aware | 0/67 |
| Failure-redense | 66/67 |
| Stage-reflect-dense | 66/67 |
| Stage-reflect-observe | 18/67 |

### 结论

Real Tool v1 推翻了“stage-aware 全面优于 observe-only”的强叙事。正确结论是：

1. non-failure 路径对轻量 mask 很稳定，所有方法均为 `800/800`；
2. 差异集中在 failure recovery；
3. reflect mask 是主要 failure bottleneck；
4. reflect-localized dense 与 fixed-window redense 几乎同样准确；
5. `0.0926` vs `0.1829` 表明 reflect-localized 方法约减少 49% dense fallback steps。

### 论文角色

这是 failure localization 的核心机制证据。冻结文档：
`docs/real_tool_v1_results.md`。

## 8. Phase 4：P1 Hidden-State Router Baseline

### 为什么做

审稿人会问显式 Agent event 是否必要，hidden state 是否已经隐含足够信息。因此需要
一个 hidden-state-only dynamic routing baseline。

### 方法

使用独立 500-task calibration set，采集当前 prompt 最后 token hidden state，
为自然闭环 stage 建 centroid。eval 时使用最近 centroid 选 plan。

比较：

```text
stage-reflect-dense explicit event route
hidden-state centroid
hidden-state centroid + reflect dense
hidden-state centroid + event override
```

hidden router 每次需要额外 prefill，因此同时报告 routing probe cost。

### 结果

| Method | Success | Failure | Cost / success | Router-inclusive cost | Fallback step |
|---|---:|---:|---:|---:|---:|
| Stage-reflect-dense | 996/1000 | 196/200 | 2.1918 | 2.1918 | 0.0926 |
| Hidden centroid | 980/1000 | 180/200 | 2.3093 | 4.6420 | 0.0000 |
| Hidden centroid reflect-dense | 983/1000 | 183/200 | 2.2609 | 4.5437 | 0.0392 |
| Hidden centroid + event | 996/1000 | 196/200 | 2.1918 | 4.4036 | 0.0926 |

### 结论

hidden-state-only centroid 没有替代显式 trajectory event。event hybrid 能追平
explicit route，但需要额外 hidden-state prefill，使 router-inclusive cost 明显更高。

### 论文角色

作为 reviewer baseline，支持：

```text
explicit failure/reflect events are reliable and nearly free routing signals
in this benchmark.
```

旧结果缺少部分 raw detection 字段时必须报告 `N/A`，不能推断为 0。

## 9. Phase 5：P2 Pruning Substrate v2

### 为什么做

约 1% mask 只能证明机制，缺乏有意义的压缩幅度。P2 目标是通过更好的 scoring、
global allocation 和 compensation 推进到 3%-20%。

### 方法

比较 activation、Wanda-channel 和 FLAP-style score；正式候选使用：

```text
FLAP-style fluctuation
global layer-wise budget allocation
bias compensation
```

第一版 budget 独立优化，因此不同预算之间不保证嵌套。

### Controlled-prompt PPL sanity

该 PPL 使用 controlled Agent prompts，不是标准 WikiText-2：

| Plan | Active MLP | PPL | Relative increase |
|---|---:|---:|---:|
| Dense | 1.0000 | 23.2031 | 0.00% |
| FLAP 3% | 0.9697 | 23.6021 | +1.72% |
| FLAP 5% | 0.9497 | 23.6440 | +1.90% |
| FLAP 10% | 0.8998 | 24.7785 | +6.79% |
| FLAP 20% | 0.7998 | 24.0066 | +3.46% |

该结果只能作为 collapse sanity，不能代替 Agent tool-use evaluation。

### Real Tool 100 smoke

| Method | Success | Failure | Non-failure | Cost / success |
|---|---:|---:|---:|---:|
| Dense | 100/100 | 20/20 | 80/80 | 2.2000 |
| FLAP 3% stage-reflect-dense | 100/100 | 20/20 | 80/80 | 2.1395 |
| FLAP 5% stage-reflect-dense | 66/100 | 13/20 | 53/80 | 5.1615 |
| FLAP 10% stage-reflect-dense | 99/100 | 19/20 | 80/80 | 2.0399 |
| FLAP 5% observe-failure-redense | 66/100 | 13/20 | 53/80 | 5.1714 |
| FLAP 10% observe-failure-redense | 99/100 | 19/20 | 80/80 | 2.0602 |

5% anomaly audit 没有发现 duplicate 或 out-of-range index，但发现预算非嵌套：

```text
9353 channels pruned by 5% were not pruned by the independent 10% plan.
```

因此不能将结果解释为“10% 天然优于 5%”，而应解释为独立 allocator 产生坏 plan。

### Uniform 20% pressure smoke

| Method | Success | Failure | Non-failure | Schema validity | Unit convert |
|---|---:|---:|---:|---:|---:|
| Dense | 100/100 | 20/20 | 80/80 | 1.0000 | 33/33 |
| FLAP 20% stage-reflect-dense | 60/100 | 9/20 | 51/80 | 0.5247 | 0/33 |

这说明 controlled-prompt PPL 严重低估了 tool schema 退化。

### 10% Real Tool 1000

| Method | Success | Failure | Non-failure | Cost / success | Active cost | Fallback step | Schema validity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense | 997/1000 | 197/200 | 800/800 | 2.2066 | 2200.0000 | 0.0000 | 1.0000 |
| FLAP 10% stage-reflect-dense | 990/1000 | 195/200 | 795/800 | 2.0777 | 2056.9205 | 0.1156 | 0.9876 |
| FLAP 10% observe-failure-redense | 990/1000 | 195/200 | 795/800 | 2.0987 | 2077.6726 | 0.2114 | 0.9876 |

### 结论

1. P2 将可用预算从约 1% 推到 10%；
2. 10% 不是无损：成功数从 997 降到 990；
3. stage-reflect-dense 的 cost/success 比 dense 低约 5.8%；
4. 同样 990/1000 时，reflect-localized route 比 fixed-window fallback 少约 45% dense steps；
5. PPL 不能代替 schema-sensitive Agent evaluation。

## 10. Phase 6：P2c Schema-Aware Nested Budgets

### 为什么做

P2 暴露两个问题：

1. 独立 budget search 导致 5%/10% plan 不嵌套；
2. uniform 20% 破坏 tool-call schema，但 final-answer 可能仍能承受更高预算。

### 方法

使用 schema-heavy assistant tool-call calibration snippets，并构建全局嵌套删除集合：

```text
M_5 subset M_10 subset M_12 subset M_15 subset M_18 subset M_20
```

其中 `M_r` 表示预算 `r` 的被删除通道集合。audit 要求：

```text
duplicate = 0
out-of-range = 0
adjacent nesting overlap = 100%
```

当前 schema calibration 是 schema-heavy samples；`--schema-token-weight` 接口存在，
但真正的 token-level weighted loss 尚未实现。

generation-type 方法：

| Method | Tool call/retry | Final answer | Reflect |
|---|---:|---:|---:|
| Nested uniform 10 | 10% | 10% | Dense |
| Adaptive A | 5% | 15% | Dense |
| Adaptive B15 | 10% | 15% | Dense |
| Adaptive B20 | 10% | 20% | Dense |

### 3B 100 smoke

| Method | Success | Failure | Non-failure | Cost / success | Schema validity |
|---|---:|---:|---:|---:|---:|
| Dense | 100/100 | 20/20 | 80/80 | 2.2000 | 1.0000 |
| Nested uniform 10 | 100/100 | 20/20 | 80/80 | 2.0000 | 1.0000 |
| Adaptive A | 100/100 | 20/20 | 80/80 | 2.0000 | 1.0000 |
| Adaptive B15 | 100/100 | 20/20 | 80/80 | 1.9500 | 1.0000 |
| Adaptive B20 | 100/100 | 20/20 | 80/80 | 1.9000 | 1.0000 |

### 3B historical 1000

| Method | Success | Failure | Non-failure | Cost / success | Schema validity |
|---|---:|---:|---:|---:|---:|
| Dense | 997/1000 | 197/200 | 800/800 | 2.2066 | 1.0000 |
| Identity hook | 997/1000 | 197/200 | 800/800 | 2.2066 | 1.0000 |
| Nested uniform 10 | 997/1000 | 197/200 | 800/800 | 2.0060 | 1.0000 |
| Adaptive B20 | 997/1000 | 197/200 | 800/800 | 1.9057 | 1.0000 |

历史结果支持：

```text
Adaptive-B20 vs Dense: cost/success -13.6%
Adaptive-B20 vs Nested uniform 10: cost/success -5.0%
```

### 重要可信度说明

该 3B 1000 表产生于 identity no-op 修复和 protocol v1.2 对齐之前。
`_empty_plan_like()` 曾清空 pruned channels，却保留：

```text
mlp_output_bias_compensation
mlp_output_scale
```

因此 derived identity/dense fallback 可能不是严格 no-op。该 bug 已修复并有回归测试。
10%/20% sparse plan 本身使用 compensation 是设计行为，不受“应清空”的影响；
但涉及 identity 和 derived dense fallback 的最终跨模型表仍需 post-fix v1.2 重跑。

### 3B post-fix protocol v1.2 1000

修复 identity no-op 并统一到 protocol v1.2 后，重新运行同一最小矩阵：

| Method | Success | Failure | Non-failure | Cost / success | Active cost | Fallback step | Schema validity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense | 993/1000 | 193/200 | 800/800 | 2.2367 | 2221.0000 | 0.0000 | 0.9937 |
| Identity hook | 993/1000 | 193/200 | 800/800 | 2.2367 | 2221.0000 | 0.0000 | 0.9937 |
| Nested uniform 10 | 957/1000 | 175/200 | 782/800 | 2.2371 | 2140.8990 | 0.1219 | 0.9557 |
| Adaptive B20 | 986/1000 | 200/200 | 786/800 | 1.9270 | 1899.9985 | 0.0909 | 1.0000 |

By tool：

| Method | Calculator | Lookup | Unit convert |
|---|---:|---:|---:|
| Dense | 327/334 | 333/333 | 333/333 |
| Identity hook | 327/334 | 333/333 | 333/333 |
| Nested uniform 10 | 291/334 | 333/333 | 333/333 |
| Adaptive B20 | 334/334 | 319/333 | 333/333 |

事件：

```text
Dense / identity:
  format_error = 14

Nested uniform 10:
  format_error = 104

Adaptive-B20:
  format_error = 0
```

成本变化：

| Comparison | Cost / success | Active FFN cost |
|---|---:|---:|
| Adaptive B20 vs Dense | -13.85% | -14.45% |
| Adaptive B20 vs Nested uniform 10 | -13.86% | -11.25% |
| Nested uniform 10 vs Dense | +0.02% | -3.61% |

该结果改变了 3B 的最终口径：

1. identity hook 与 Dense 完全一致，说明 no-op 修复有效；
2. Nested uniform 10 不是有效的最终 baseline：虽然 active cost 下降 3.61%，但
   `104` 次 format error 和成功率下降使 cost/success 略差于 Dense；
3. Adaptive-B20 不是 Dense-equivalent：总成功率低 7 条；
4. Adaptive-B20 将 failure subset 从 Dense 的 `193/200` 提高到 `200/200`；
5. 它的 14 条失败全部位于 non-failure subset，并在 aggregate by-tool 中集中于
   lookup；在查看逐条 failure trace 前，不能进一步断言具体错误格式；
6. Adaptive-B20 仍将 cost/success 降低约 13.85%，且 schema validity 为 1.0。

### 论文角色

旧 997/1000 表保留为 historical protocol result。跨模型最终表应使用 post-fix
v1.2 结果，并明确说明 3B Adaptive-B20 是 quality-cost trade-off，而 7B
Adaptive-B20 达到 Dense-equivalent success。

## 11. Phase 7：Qwen2.5-7B 扩展与 Protocol Gate

### 为什么做

验证方法能否扩展到更大模型，并排除 3B 特有现象。7B 必须重新进行 calibration、
FLAP scoring 和 nested plan 构建，不能复用 3B channel indices。

### 问题 1：Identity no-op compensation leak

初始 7B smoke 中 dense 与 identity 不一致。根因是 `_empty_plan_like()` 保留 stale
compensation。修复后 dense 与 identity 完全一致。

### 问题 2：Tool-call wrapper mismatch

修复 identity 后，dense 仍在 unit conversion 上生成：

```json
{"type":"unit_convert","arguments":{"value":7,"from_unit":"m","to_unit":"cm"}}
```

而 strict protocol 要求：

```json
{"type":"tool_call","name":"unit_convert","arguments":{"value":7,"from_unit":"m","to_unit":"cm"}}
```

Protocol v1.1 通过更明确的正反例和 schema-error feedback 修复该 baseline gate，
没有放宽 parser。

### v1.1 20-task gate

| Method | Success | Failure | Schema validity | Cost / success |
|---|---:|---:|---:|---:|
| Dense | 20/20 | 5/5 | 1.0000 | 2.3000 |
| Identity hook | 20/20 | 5/5 | 1.0000 | 2.3000 |
| Nested uniform 10 | 20/20 | 5/5 | 1.0000 | 2.0500 |
| Adaptive B20 | 20/20 | 5/5 | 1.0000 | 1.9500 |

### 问题 3：Lookup observation-to-final mismatch

v1.1 的首轮 100 smoke 中 dense/identity 仅 91/100。lookup tool call 和 observation
正确，但模型将：

```text
owner_3007
```

回答为：

```text
3007
```

Protocol v1.2 明确要求 final 精确复制 `owner_N`。strict evaluator 不变。

### v1.2 100 smoke

| Method | Success | Failure | Non-failure | Cost / success | Schema validity |
|---|---:|---:|---:|---:|---:|
| Dense | 100/100 | 20/20 | 80/80 | 2.2100 | 1.0000 |
| Identity hook | 100/100 | 20/20 | 80/80 | 2.2100 | 1.0000 |
| Nested uniform 10 | 100/100 | 20/20 | 80/80 | 2.0000 | 1.0000 |
| Adaptive B20 | 100/100 | 20/20 | 80/80 | 1.9000 | 1.0000 |

### v1.2 1000 主结果

| Method | Success | Failure | Non-failure | Cost / success | Active cost | Fallback step | Schema validity | Premature final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | 1000/1000 | 200/200 | 800/800 | 2.2390 | 2239.0000 | 0.0000 | 1.0000 | 0.0390 |
| Identity hook | 1000/1000 | 200/200 | 800/800 | 2.2390 | 2239.0000 | 0.0000 | 1.0000 | 0.0390 |
| Nested uniform 10 | 1000/1000 | 200/200 | 800/800 | 2.0000 | 1999.9970 | 0.0909 | 1.0000 | 0.0000 |
| Adaptive B20 | 1000/1000 | 200/200 | 800/800 | 1.9000 | 1899.9974 | 0.0909 | 1.0000 | 0.0000 |

三类工具对所有方法均为：

```text
calculator: 333/333
lookup: 333/333
unit_convert: 334/334
```

成本变化：

| Comparison | Cost / success reduction |
|---|---:|
| Nested uniform 10 vs Dense | 10.7% |
| Adaptive B20 vs Dense | 15.1% |
| Adaptive B20 vs Nested uniform 10 | 5.0% |

### 结论

这是当前最强主结果：

> Under strict Real Tool protocol v1.2, Adaptive-B20 matches Dense at
> 1000/1000 task success, including 200/200 injected-failure tasks, while
> reducing theoretical active-FFN cost per successful episode by about 15.1%.

该结果证明 generation-type routing 相比统一 10% plan 仍有额外价值。

## 12. Phase 8：静态通用能力 Sanity

### 为什么做

Real Tool 中把 20% 仅用于 final-answer 并不意味着 20% 是通用安全静态模型。
需要检查动态收益是否以严重通用能力退化为代价。

### 3B quick sanity，256 samples

| Plan | Active MLP | ARC-Easy | HellaSwag | PIQA | WikiText-2 PPL |
|---|---:|---:|---:|---:|---:|
| Dense | 1.0000 | 0.6836 | 0.5312 | 0.8008 | 13.8088 |
| Nested 10% | 0.9000 | 0.6719 | 0.5352 | 0.8008 | 13.9116 |
| Nested 15% | 0.8500 | 0.6328 | 0.5156 | 0.7891 | 26.1213 |
| Nested 20% | 0.8000 | 0.6445 | 0.4805 | 0.7539 | 36.4053 |

### 7B quick sanity，256 samples

| Plan | Active MLP | ARC-Easy | HellaSwag | PIQA | WikiText-2 PPL |
|---|---:|---:|---:|---:|---:|
| Dense | 1.0000 | 0.7891 | 0.5820 | 0.8242 | 11.7030 |
| Nested 10% | 0.9000 | 0.7773 | 0.5508 | 0.7656 | 15.5519 |
| Nested 20% | 0.8000 | 0.7031 | 0.5273 | 0.7461 | 33.8450 |

### 结论

1. 3B nested 10% 的通用 sanity 接近 dense；
2. 15%/20% static 明显退化；
3. 7B nested 10% 已出现非平凡退化；
4. 7B nested 20% 不是通用安全静态模型；
5. 这不反驳 Adaptive-B20，反而证明只在 final-answer 使用 20% 的动态设计是必要的。

### 口径限制

`max_samples=256` 只能写 quick sanity，不能作为最终 full benchmark table。
7B 15% 只有使用同一 calibration/scoring/nested ladder 构建并记录 hash 后才可加入。

## 13. Phase 9：P5 Compact Subnet Deployment Analysis

### 为什么做

mask hook 只减少理论 active channels，不减少原始 GEMM。必须检查 plan 是否能物化为
真实较窄的 FFN，以及理论成本能否转成 wall-clock speedup。

### Compact materialization

每层只裁 FFN intermediate width：

```text
gate_proj: keep selected output rows
up_proj:   keep selected output rows
down_proj: keep matching input columns
```

hidden size、attention 和 KV cache 不变。bias compensation 作为 MLP output
补偿保留。

### Functional equivalence

普通 compact logits-only，100 prompts：

| Plan | Active MLP | Mean logit diff | Top-1 match |
|---|---:|---:|---:|
| Nested 10% | 0.899998 | 0.0906 | 100/100 |
| Nested 20% | 0.799999 | 0.1062 | 100/100 |

hardware-aligned compact：

| Plan | Active MLP | Mean logit diff | Top-1 match |
|---|---:|---:|---:|
| Aligned 10% | 0.899614 | 0.0828 | 100/100 |
| Aligned 20% | 0.799710 | 0.0914 | 100/100 |

greedy exact match 较低，但不作为核心 pass/fail，因为 bf16 小差异会在自回归生成中累积。

### Naive compact 负结果

同一进程中的首轮 compact latency 更慢且显存更高。后续单 variant 检查发现 compact
materialization 时 dense 和 compact 权重同时存在，且不规则 intermediate widths 对
GEMM kernel 不友好。因此该数字不能作为正式速度结果。

### Hardware-aligned MLP-only microbenchmark

| Plan | B1 S1 | B1 S128 | B4 S128 |
|---|---:|---:|---:|
| Dense | 0.217 ms | 0.494 ms | 0.874 ms |
| Aligned 10% | 0.240 ms | 0.504 ms | 0.770 ms |
| Aligned 20% | 0.185 ms | 0.220 ms | 0.466 ms |

aligned 20% 在抽样 shape 上有清晰 MLP-level speed signal。

### Full-model fixed-decode

Batch 1：

| Plan | Decode ms/token | Tokens/s | Peak GB | Speedup |
|---|---:|---:|---:|---:|
| Dense | 26.226 | 38.130 | 8.266 | 0.0% |
| Aligned 10% | 25.497 | 39.220 | 7.993 | +2.8% |
| Aligned 20% | 26.680 | 37.481 | 7.857 | -1.7% |

Batch 4：

| Plan | Decode ms/token | Tokens/s | Peak GB | Speedup |
|---|---:|---:|---:|---:|
| Dense | 27.319 | 146.421 | 8.335 | 0.0% |
| Aligned 10% | 27.200 | 147.059 | 8.062 | +0.4% |
| Aligned 20% | 27.700 | 144.405 | 7.926 | -1.4% |

### 结论

```text
compact structure: pass
logits/top-1 equivalence: pass
hardware-aligned MLP-only speed signal: pass
meaningful full-model decode speedup: not demonstrated
dynamic Agent wall-clock speedup: not evaluated
```

当前 HF/PyTorch decode 中，attention、KV cache、launch overhead 和不均匀 layer width
抵消了 FFN GEMM 的理论收益。P5 应作为诚实的 deployment analysis，而不是主速度结果。

## 14. 主要正结果、负结果与修复

### 正结果

1. stage/failure signal 在 controlled setting 中有效；
2. Real Tool failure sensitivity 集中在 recovery/reflect；
3. reflect-localized dense 比 fixed-window redense 使用约一半 fallback steps；
4. explicit events 优于 hidden-state-only centroid；
5. FLAP substrate 将可用预算推进到 10%；
6. schema-aware nested plans 修复了独立预算 anomaly；
7. generation-type routing 允许 final-answer 使用 20%；
8. 7B Adaptive-B20 达到 1000/1000，并降低 15.1% theoretical cost/success。

### 重要负结果

1. mixed attention+MLP pruning 在早期 smoke 中不稳定；
2. naive per-layer 1% 也会明显掉分；
3. controlled PPL 无法预测 tool schema failure；
4. independent 5% allocator 产生异常坏 plan；
5. uniform 20% 在 Real Tool 中崩溃；
6. static 20% 通用 benchmark 明显退化；
7. compact MLP speed signal尚未转化为 full-model speedup。

### 关键修复

| 问题 | 修复 |
|---|---|
| 错误统计 gate activation | 改为统计真实 SwiGLU `down_proj` input。 |
| substring 判分误报 | 改为 tool-specific strict exact scoring。 |
| numeric final 被拒 | parser 接受 string/number，并统一比较。 |
| start 被当作 failure | 修正 failure-redense 初始事件。 |
| 成功 observation 后继续空转 | 明确 final-answer instruction。 |
| 提前 final | 增加 `premature_final` runner guard 和 reflect retry。 |
| identity hook 泄漏 compensation | `_empty_plan_like()` 清空 bias compensation 和 scale。 |
| 7B 将 tool name 写进 type | protocol v1.1 prompt 和结构化 schema feedback。 |
| lookup final 丢 `owner_` | protocol v1.2 observation-to-final hardening。 |
| compact 不规则 width 低效 | 构建 hardware-aligned compact plans。 |

## 15. 论文可用 Claim

### 主 Claim

> Tool-using Agent trajectories are compute-heterogeneous. Schema-sensitive
> tool-call/retry generations require conservative FFN budgets, final-answer
> generations tolerate more aggressive pruning, and failure recovery benefits
> from temporary dense FFN execution.

### 7B 实证 Claim

> On Qwen2.5-7B under strict Real Tool protocol v1.2, Adaptive-B20 matches
> Dense at 1000/1000 task success and reduces theoretical active-FFN cost per
> successful episode by about 15.1%.

### Failure Localization Claim

> Reflect-localized dense recovery retains near-dense recovery behavior while
> using roughly half as many dense fallback steps as fixed-window
> re-densification in the 3B Real Tool v1 experiment.

### Router Claim

> Explicit failure/reflect events outperform a hidden-state-only centroid
> router and avoid its additional prefill cost on this benchmark.

## 16. 当前不能声称

```text
15.1% wall-clock speedup
deployment-ready dynamic compact subnet
uniform static 20% pruning is safe
stage-aware masks universally outperform observe-only
PPL alone predicts Agent pruning quality
3B Adaptive-B20 is Dense-equivalent under protocol v1.2
the current local-tool benchmark covers stateful or open-world agents
```

## 17. 论文表格建议

### 主表：7B Real Tool v1.2

```text
Dense
Identity hook
Nested uniform 10
Adaptive-B20
```

字段：

```text
overall success
failure success
non-failure success
schema validity
premature final
fallback step ratio
active FFN cost
cost per success
```

### 机制表：3B Real Tool v1

```text
Observe-only
Stage-aware
Failure-redense
Observe-failure-redense
Stage-reflect-dense
Stage-reflect-observe
```

重点报告 failure subset 和 fallback step ratio。

### Router ablation

```text
Explicit stage/reflect route
Hidden centroid
Hidden centroid reflect-dense
Hidden centroid + event
```

必须包含 router-inclusive cost。

### Static sanity

```text
Dense
Nested 10%
Nested 20%
```

Adaptive-B20 不是单一静态模型，不能作为 static benchmark row。

### Deployment analysis

```text
mask-hook vs compact equivalence
irregular vs aligned MLP microbenchmark
full-model fixed-decode latency
```

## 18. 论文结构映射

### Introduction

- PPL-based pruning evaluation 会遗漏 Agent tool-schema failure；
- Agent generation types 具有不同计算敏感度；
- trajectory events 可以直接指导计算恢复。

### Problem Setup

- Real Tool protocol；
- deterministic fault schedule；
- active FFN cost 和 cost per successful episode。

### Method

- true SwiGLU channel statistics；
- FLAP-style scoring 和 bias compensation；
- schema-aware nested budget ladder；
- generation-type routing；
- reflect-localized dense recovery。

### Main Experiments

- 7B Real Tool v1.2 主表；
- 3B Real Tool failure localization；
- 3B/7B static sanity。

### Ablations

- dense vs identity；
- observe-only vs stage-aware；
- fixed-window vs reflect-localized recovery；
- hidden-state-only vs explicit events；
- uniform 10% vs Adaptive-B20；
- uniform static 20% pressure test。

### Deployment Analysis

- compact materialization；
- equivalence；
- alignment；
- MLP-only 和 full-model latency 差异。

### Limitations

- active cost 不等于速度；
- 256-sample static benchmark 只是 sanity；
- 工具集是确定性、无状态、本地工具；
- 3B v1.2 对齐已完成，但 Adaptive-B20 为 986/1000，不是 Dense-equivalent；
- dynamic compact switching 未实现。

## 19. 当前剩余工作

### 必做

1. 检查 3B Adaptive-B20 的 14 条 lookup failure trace，完成错误类型归因；
2. 从本文生成论文 main table、ablation table 和 method 描述；
3. 保存每个最终结果的 config、plan hash、calibration hash、seed、dtype 和 model revision。

### 可选增强

1. 将 static benchmark 从 256 扩展到 1024 或 full validation；
2. 使用同一 7B ladder 补 15% static point；
3. 加入更强 external structured-pruning baseline；
4. 使用更低层 kernel/runtime 重新研究 compact full-model speedup。

### 暂停

```text
继续手调 3B mask
重复 7B v1.2 1000
当前 HF wrapper 上直接做 dynamic Agent latency
扩展更多工具或 7B 更大实验矩阵
```

## 20. 结果与产物索引

| Experiment | 文档 / 产物 |
|---|---|
| Legacy 4090 smoke | `docs/4090_smoke_results.md` |
| Early mask diagnostics | `docs/agent_mask_eval_results.md` |
| Controlled v1 | `docs/controlled_mask_validation_v1.md` |
| Real Tool v1 | `docs/real_tool_v1_results.md` |
| P1/P2/P2c | `docs/p2_substrate_and_p2c_results.md` |
| 3B to 7B transition | `docs/3b_to_7b_transition_review.md` |
| Static benchmark commands/results | `docs/p4_static_benchmark_next_commands.md` |
| High-level evidence and outline | `docs/paper_evidence_and_outline.md` |
| Current handoff | `docs/agent_handoff_next_steps.md` |

主要输出前缀：

```text
outputs/agent_qwen2_5_3b_4090_low/eval/stage_failure_official_1000
outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_bypass_1000
outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_p1_hidden_1000
outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_substrate_flap_0p10_1000
outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_p2c_adaptive_1000
outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_p2c_adaptive_1000_protocol_v1_2
outputs/agent_qwen2_5_7b/real_tool_eval/real_tool_v1_7b_1000_protocol_v1_2
outputs/agent_qwen2_5_7b/static_benchmarks/p4_7b_static_quick_v1
outputs/agent_qwen2_5_7b/compact_subnets/hardware_aligned
```

## 21. 一句话总结

项目最终不是在证明一个静态 20% 剪枝模型普遍安全，而是在证明：

> Agent trajectory exposes when FFN capacity is valuable. By preserving a
> conservative schema-aware subnet for tool calls, using a more aggressive
> subnet for final answers, and restoring dense FFN only for recovery,
> Adaptive-B20 preserves Qwen2.5-7B Real Tool success while reducing
> theoretical active-FFN cost per successful episode by 15.1%; however, the
> current compact implementation does not yet convert this reduction into
> meaningful full-model wall-clock speedup.
