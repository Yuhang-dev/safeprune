# Agent FFN Mask 小实验记录与结论

日期：2026-06-11

本文回答四个问题：

1. 我们这轮小实验做了什么？
2. 为什么要做这些实验？
3. 得到了什么数据？
4. 这些数据说明什么，下一步该怎么做？

当前结论用于技术路线决策，不作为最终论文结果。

## 1. 实验目的

本项目当前方向是：

```text
Trajectory-Aware Dynamic FFN Pruning for Tool-Using Reasoning Agents
```

也就是验证：

```text
Agent 阶段、工具 observation、失败事件和难度信号，
是否能比固定 mask 或 hidden-state-only 路由更有效地分配 FFN 计算。
```

这轮小实验不直接证明完整方法，而是先回答一个更底层的问题：

```text
在 Qwen2.5-3B-Instruct 上，
最朴素的 stage-specific FFN channel mask 有没有可用边界？
```

如果最基础的 1%-5% FFN mask 都明显破坏 dense 能力，就不能继续盲目扩大 sparsity，也不能直接进入复杂 router、failure recovery 或 7B 实验。

## 2. 我们做了什么

### 2.1 修正 activation scoring

Qwen2.5 的 FFN 使用 SwiGLU：

```text
FFN(x) = down_proj(SiLU(gate_proj(x)) * up_proj(x))
```

之前 activation scoring 统计的是：

```text
gate_proj(x)
```

这不是实际进入 `down_proj` 的中间通道贡献。我们已把 scoring 改为统计：

```text
SiLU(gate_proj(x)) * up_proj(x)
```

实现上对应：

```text
src/safeprune/scoring.py
down_proj forward pre-hook input
```

这一步的目的：

```text
让 FFN channel importance 更贴近真实 SwiGLU 通道贡献。
```

### 2.2 重新生成低稀疏 mask bank

模型：

```text
Qwen/Qwen2.5-3B-Instruct
```

输出：

```text
outputs/agent_qwen2_5_3b_4090_low/mask_bank/mask_bank.json
outputs/agent_qwen2_5_3b_4090_low/mask_bank/manifest.json
```

低稀疏设置：

| Sparsity | 每层剪掉 FFN 通道数 | 每层 FFN 总通道数 | Active ratio |
|---:|---:|---:|---:|
| 0.01 | 110 | 11008 | 0.9900 |
| 0.02 | 220 | 11008 | 0.9800 |
| 0.03 | 330 | 11008 | 0.9700 |
| 0.05 | 550 | 11008 | 0.9500 |

已验证所有 stage 都生成了对应 mask：

```text
plan
act
observe
reflect
answer
```

### 2.3 改用严格判分

之前的评测使用过：

```python
ok = expected in prediction
```

这个判分会误判。例如：

```text
expected=9, pred=19      -> 会被 contains 误判正确
expected=900, pred=9000  -> 会被 contains 误判正确
```

本轮改成严格判分：

| 任务类型 | 判分方式 |
|---|---|
| calculator | 解析第一个数字，必须等于 expected |
| unit_convert | 解析第一个数字，必须等于 expected |
| lookup | 解析第一个 `owner_x`，必须等于 expected |

这一步的目的：

```text
避免旧评测高估 mask 后模型表现。
```

### 2.4 跑了两组 100 条评测

任务数据：

```text
data/agent/controlled_tasks.jsonl
first 100 tasks
```

任务类型：

```text
calculator: 34
lookup: 33
unit_convert: 33
```

第一组评测：低稀疏率扫描。

```text
dense
identity_hook
static_plan_0.01
stage_answer_0.01
stage_answer_0.02
stage_answer_0.03
```

远端保存路径：

```text
outputs/agent_qwen2_5_3b_4090_low/eval/strict_mask_compare_low_100.jsonl
outputs/agent_qwen2_5_3b_4090_low/eval/strict_mask_compare_low_100_metrics.json
logs/strict_mask_compare_3b_low_100_<timestamp>.log
```

第二组评测：固定 1% sparsity，对比不同 stage mask。

```text
dense
identity_hook
static_plan_0.01
stage_plan_0.01
stage_act_0.01
stage_observe_0.01
stage_reflect_0.01
stage_answer_0.01
```

远端保存路径：

```text
outputs/agent_qwen2_5_3b_4090_low/eval/stage_mask_0p01_compare_100.jsonl
outputs/agent_qwen2_5_3b_4090_low/eval/stage_mask_0p01_compare_100_metrics.json
logs/stage_mask_0p01_compare_100_<timestamp>.log
```

其中：

```text
*.jsonl        保存逐任务 prediction、expected、strict_correct、collapse 等明细
*_metrics.json 保存每个 method 的总准确率、collapse rate 和 by_tool 分项指标
*.log          保存远端 stdout/stderr 与 RESULT 汇总行
```

## 2.5 相关远端产物清单

本轮低稀疏实验相关产物应保留：

| 类型 | 路径 | 说明 |
|---|---|---|
| mask bank | `outputs/agent_qwen2_5_3b_4090_low/mask_bank/mask_bank.json` | 修正 SwiGLU scoring 后的低稀疏 stage mask bank |
| manifest | `outputs/agent_qwen2_5_3b_4090_low/mask_bank/manifest.json` | mask bank 构建配置与状态 |
| low-sparsity predictions | `outputs/agent_qwen2_5_3b_4090_low/eval/strict_mask_compare_low_100.jsonl` | 低稀疏扫描逐任务结果 |
| low-sparsity metrics | `outputs/agent_qwen2_5_3b_4090_low/eval/strict_mask_compare_low_100_metrics.json` | 低稀疏扫描汇总指标 |
| stage predictions | `outputs/agent_qwen2_5_3b_4090_low/eval/stage_mask_0p01_compare_100.jsonl` | 五个 stage mask 的逐任务结果 |
| stage metrics | `outputs/agent_qwen2_5_3b_4090_low/eval/stage_mask_0p01_compare_100_metrics.json` | 五个 stage mask 的汇总指标 |
| build log | `logs/build_stage_mask_3b_low_512_<timestamp>.log` | 低稀疏 mask bank 生成日志 |
| low-sparsity eval log | `logs/strict_mask_compare_3b_low_100_<timestamp>.log` | 低稀疏扫描日志 |
| stage eval log | `logs/stage_mask_0p01_compare_100_<timestamp>.log` | stage mask 对比日志 |

## 3. 得到的数据

### 3.1 链路校验结果

| Method | Correct | Accuracy | Collapse |
|---|---:|---:|---:|
| dense | 93/100 | 0.93 | 0 |
| identity_hook | 93/100 | 0.93 | 0 |

解释：

```text
dense 和 identity_hook 完全一致，说明 switchable MLP mask hook 本身没有污染模型。
```

这是后续所有 mask 对比可信的前提。

### 3.2 低稀疏扫描结果

| Method | Correct | Accuracy | Collapse |
|---|---:|---:|---:|
| dense | 93/100 | 0.93 | 0 |
| identity_hook | 93/100 | 0.93 | 0 |
| static_plan_0.01 | 65/100 | 0.65 | 0 |
| stage_answer_0.01 | 72/100 | 0.72 | 0 |
| stage_answer_0.02 | 53/100 | 0.53 | 0 |
| stage_answer_0.03 | 46/100 | 0.46 | 0 |

分工具结果：

| Method | calculator | lookup | unit_convert |
|---|---:|---:|---:|
| dense | 32/34 = 0.941 | 32/33 = 0.970 | 29/33 = 0.879 |
| identity_hook | 32/34 = 0.941 | 32/33 = 0.970 | 29/33 = 0.879 |
| static_plan_0.01 | 23/34 = 0.676 | 20/33 = 0.606 | 22/33 = 0.667 |
| stage_answer_0.01 | 23/34 = 0.676 | 19/33 = 0.576 | 30/33 = 0.909 |
| stage_answer_0.02 | 18/34 = 0.529 | 15/33 = 0.455 | 20/33 = 0.606 |
| stage_answer_0.03 | 13/34 = 0.382 | 13/33 = 0.394 | 20/33 = 0.606 |

### 3.3 Stage mask 1% 对比结果

| Method | Correct | Accuracy | Collapse |
|---|---:|---:|---:|
| dense | 93/100 | 0.93 | 0 |
| identity_hook | 93/100 | 0.93 | 0 |
| static_plan_0.01 | 65/100 | 0.65 | 0 |
| stage_plan_0.01 | 65/100 | 0.65 | 0 |
| stage_act_0.01 | 64/100 | 0.64 | 0 |
| stage_observe_0.01 | 79/100 | 0.79 | 0 |
| stage_reflect_0.01 | 51/100 | 0.51 | 0 |
| stage_answer_0.01 | 72/100 | 0.72 | 0 |

分工具结果：

| Method | calculator | lookup | unit_convert |
|---|---:|---:|---:|
| dense | 32/34 = 0.941 | 32/33 = 0.970 | 29/33 = 0.879 |
| identity_hook | 32/34 = 0.941 | 32/33 = 0.970 | 29/33 = 0.879 |
| static_plan_0.01 | 23/34 = 0.676 | 20/33 = 0.606 | 22/33 = 0.667 |
| stage_plan_0.01 | 23/34 = 0.676 | 20/33 = 0.606 | 22/33 = 0.667 |
| stage_act_0.01 | 20/34 = 0.588 | 24/33 = 0.727 | 20/33 = 0.606 |
| stage_observe_0.01 | 25/34 = 0.735 | 27/33 = 0.818 | 27/33 = 0.818 |
| stage_reflect_0.01 | 21/34 = 0.618 | 19/33 = 0.576 | 11/33 = 0.333 |
| stage_answer_0.01 | 23/34 = 0.676 | 19/33 = 0.576 | 30/33 = 0.909 |

## 4. 数据说明了什么

### 4.1 Hook 没问题

```text
dense:         93/100
identity_hook: 93/100
```

结论：

```text
模型加载、prompt、严格判分和 switchable mask hook 链路可信。
```

### 4.2 朴素 1% 每层固定 FFN mask 也会明显掉分

最好的 1% stage mask 是：

```text
stage_observe_0.01: 79/100
```

但它仍然比 dense 低：

```text
93/100 - 79/100 = 14 个百分点
```

结论：

```text
当前不能说 1% FFN mask 已经可用。
```

这说明 Qwen2.5-3B 对直接置零 FFN channel 很敏感，即使每层只剪 110 个通道，也会明显破坏精细任务能力。

### 4.3 Stage-specific mask 有真实信号

同样是 1% sparsity：

```text
static_plan_0.01:   65/100
stage_plan_0.01:    65/100
stage_act_0.01:     64/100
stage_observe_0.01: 79/100
stage_reflect_0.01: 51/100
stage_answer_0.01:  72/100
```

结论：

```text
不同 stage activation 生成的 mask 质量差异很大。
```

其中 `stage_observe_0.01` 明显优于 static mask：

```text
79/100 vs 65/100
```

这说明 stage 信息不是完全无效，至少在当前 controlled tasks 上，observe activation 选出的“低重要性通道”更不容易伤害整体任务。

### 4.4 但 stage 信号还不足以支撑方法成立

虽然 `stage_observe_0.01` 是当前最好结果，但它仍没有接近 dense。

结论：

```text
stage signal exists, but naive per-layer channel zeroing is too destructive.
```

这意味着当前失败的不是整个 Agent trajectory 方向，而是这个最朴素实现：

```text
每层固定剪同样比例
+ 36 层同时置零 FFN channel
+ 无 layer sensitivity
+ 无 layer-wise budget
+ 无 compensation
+ 无 recovery
```

### 4.5 问题不是乱码 collapse，而是能力偏移

所有实验中：

```text
collapse_count = 0
```

模型没有大规模乱码或循环输出。错误主要表现为：

```text
9 -> 19
owner_4 -> owner_44 / owner_49
600 -> 360 / 180
```

结论：

```text
mask 主要破坏了精细算术、检索映射和单位换算能力，
而不是简单造成生成崩溃。
```

## 5. 当前总判断

### 5.1 方向是否完全错了？

不是。

理由：

```text
stage_observe_0.01: 79/100
static_plan_0.01:   65/100
```

同样是 1% sparsity，`stage_observe_0.01` 明显优于 `static_plan_0.01`。
这说明 stage-specific activation 里确实有信号，而且当前最有价值的信号具体来自
`observe` 阶段，而不是泛泛地说 stage signal 存在。

### 5.2 当前 naive mask 方法是否可继续作为主方法？

不能。

理由：

```text
最好的 stage_observe_0.01 也只有 79/100，距离 dense 的 93/100 仍差 14 个点。
```

继续盲目扫 2%、3%、5%、10%、30% 没有意义。已有结果已经说明更高稀疏率会更差。

### 5.3 当前应该保留什么？

保留：

```text
stage-aware 研究问题
controlled Agent task 评测协议
strict exact-match validator
switchable MLP mask 实验框架
stage mask bank
```

停止把下面这套当主方法：

```text
每层固定比例剪通道
```

### 5.4 当前最正确的下一步

Layer sensitivity 已经把问题进一步定位清楚：

```text
多数 single-layer 0.01: 48/50 或 49/50，接近 dense。
full_stage_observe_0.01: 38/50，明显低于 dense。
```

因此 full 1% 失败的主要原因不是某个单层一剪就崩，也不是 stage signal 无效，
而是 36 层同时剪枝造成的累积扰动。

当前最正确的下一步是基于 `stage_observe` 的 safe-layer-only 评测：

```text
safe13_observe_0.01
safe13_observe_0.02
safe13_observe_0.03
```

跑完这三个对照后，再决定是否推进真正的 global layer-wise budget。

## 6. 下一步怎么做

### 6.1 下一步实验：Layer sensitivity

固定当前最好的 mask 来源：

```text
stage_observe_0.01
```

但不要 36 层同时剪。改成：

```text
每次只剪一个 layer 的 110 个通道，其余 35 层不剪。
```

建议保存路径：

```text
outputs/agent_qwen2_5_3b_4090_low/eval/layer_sensitivity_observe0p01_50.jsonl
outputs/agent_qwen2_5_3b_4090_low/eval/layer_sensitivity_observe0p01_50_metrics.json
logs/layer_sensitivity_observe0p01_50_<timestamp>.log
```

目的：

```text
找出哪些层一剪就明显掉分，哪些层相对安全。
```

如果发现部分层安全，下一步就做：

```text
layer-wise / global budget
```

也就是：

```text
敏感层剪 0
安全层剪更多
整体仍保持 1% 或更低的全局 budget
```

### 6.2 判断标准

Layer sensitivity 跑完后按下面决策：

| 结果 | 解释 | 下一步 |
|---|---|---|
| 少数层剪了就掉分，多数层安全 | per-layer fixed budget 是主要问题 | 实现 global layer-wise budget |
| 所有层单剪都明显掉分 | channel zeroing 本身太破坏 | 转 FLAP-style compensation / loss-delta scoring |
| 某些中后层特别安全 | 可以先做安全层优先剪枝 | 构造 safe-layer-only mask |
| observe mask 的安全层明显多 | stage signal 有继续价值 | 做 stage-aware layer-wise budget |

### 6.3 中期路线

如果 layer-wise budget 后 `stage_observe` 能接近 dense，再推进：

```text
plan / act / observe / reflect / answer 动态切换
+ failure-triggered re-densification
+ cost per successful task
```

如果 layer-wise budget 后仍然差，转向：

```text
FLAP-style fluctuation score
bias / scale compensation
loss-delta scoring
SliceGPT / Probe Pruning 复现对照
```

## 7. Layer sensitivity 结果

本轮固定使用当前最好的 mask 来源：

```text
stage_observe_0.01
```

但每次只剪一个 layer 的 110 个 FFN 通道，其余 35 层不剪。

远端保存路径：

```text
outputs/agent_qwen2_5_3b_4090_low/eval/layer_sensitivity_observe0p01_50.jsonl
outputs/agent_qwen2_5_3b_4090_low/eval/layer_sensitivity_observe0p01_50_metrics.json
logs/layer_sensitivity_observe0p01_50_<timestamp>.log
```

### 7.1 Baseline

| Method | Correct | Accuracy | Collapse |
|---|---:|---:|---:|
| dense | 49/50 | 0.98 | 0 |
| identity_hook | 49/50 | 0.98 | 0 |
| full_stage_observe_0.01 | 38/50 | 0.76 | 0 |

分工具结果：

| Method | calculator | lookup | unit_convert |
|---|---:|---:|---:|
| dense | 16/17 = 0.941 | 16/16 = 1.000 | 17/17 = 1.000 |
| identity_hook | 16/17 = 0.941 | 16/16 = 1.000 | 17/17 = 1.000 |
| full_stage_observe_0.01 | 10/17 = 0.588 | 12/16 = 0.750 | 16/17 = 0.941 |

### 7.2 单层最敏感与最安全层

最敏感层：

| Layer | Correct | Accuracy |
|---:|---:|---:|
| 22 | 45/50 | 0.90 |
| 00 | 46/50 | 0.92 |
| 31 | 46/50 | 0.92 |
| 17 | 47/50 | 0.94 |
| 21 | 47/50 | 0.94 |
| 24 | 47/50 | 0.94 |
| 25 | 47/50 | 0.94 |

最安全层：

| Layer | Correct | Accuracy |
|---:|---:|---:|
| 03 | 49/50 | 0.98 |
| 06 | 49/50 | 0.98 |
| 07 | 49/50 | 0.98 |
| 08 | 49/50 | 0.98 |
| 09 | 49/50 | 0.98 |
| 10 | 49/50 | 0.98 |
| 14 | 49/50 | 0.98 |
| 15 | 49/50 | 0.98 |
| 18 | 49/50 | 0.98 |
| 23 | 49/50 | 0.98 |
| 28 | 49/50 | 0.98 |
| 30 | 49/50 | 0.98 |
| 34 | 49/50 | 0.98 |

### 7.3 新结论

单层剪枝大多接近 dense：

```text
dense: 49/50
多数 single-layer 0.01: 48/50 或 49/50
最差 single-layer 0.01: 45/50
full_stage_observe_0.01: 38/50
```

这说明：

```text
没有单个 layer 一剪就直接崩。
主要问题是 36 层同时每层固定剪 110 个通道后，扰动逐层累积。
```

因此当前最重要的技术结论变为：

```text
per-layer fixed budget 是主要问题。
下一步应从“每层都剪 1%”改为“只在安全层剪，敏感层不剪”。
```

这支持继续做 layer-wise / global budget，而不是立刻放弃 Agent trajectory 方向。

### 7.4 下一步 safe-layer-only budget

根据单层 50 条结果，优先候选安全层为：

```text
[3, 6, 7, 8, 9, 10, 14, 15, 18, 23, 28, 30, 34]
```

这些层单层剪 0.01 时均保持 `49/50`，与 dense 持平。

下一步构造 safe-layer-only masks：

| Mask | 做法 | 近似全局 FFN 稀疏率 | 目的 |
|---|---|---:|---|
| safe13_0.01 | 只在 13 个安全层剪 1% | 0.36% | 验证只剪安全层是否接近 dense |
| safe13_0.02 | 只在 13 个安全层剪 2% | 0.72% | 在接近 1% 全局预算前测试 |
| safe13_0.03 | 只在 13 个安全层剪 3% | 1.08% | 与 full per-layer 1% 做近似同预算对比 |

关键比较：

```text
full_stage_observe_0.01: 86/100，active ratio 0.9900
safe13_observe_0.03:     89/100，active ratio 0.9892
```

`safe13_observe_0.03` 在近似相同 global FFN budget 下优于 full-stage 1%，
说明 layer-wise budget 方向是对的；但优势只有 3 个点，不能说明 safe-layer-only
已经完全解决问题。

更强的信号来自低 global budget：

```text
dense:                97/100
identity_hook:         97/100
safe13_observe_0.01:   96/100，active ratio 0.9964
```

这说明只剪安全层可以在很低全局稀疏率下基本贴近 dense。下一步应继续推进
global layer-wise budget，但需要更细的层间分配，而不是简单把 13 个安全层统一提高到 3%。

### 7.5 下一步具体要做

下一步不要继续扩大 full per-layer sparsity，而是按以下顺序推进。

第一步：构造并评测 safe-layer-only masks。

```text
safe layers: [3, 6, 7, 8, 9, 10, 14, 15, 18, 23, 28, 30, 34]
source mask: stage_observe
tasks: first 100 controlled tasks
```

评测方法：

| Method | 做法 | 近似全局 FFN 稀疏率 |
|---|---|---:|
| dense | 不挂 mask | 0 |
| identity_hook | 挂 hook 但不剪 | 0 |
| full_stage_observe_0.01 | 36 层每层剪 1% | 1.00% |
| safe13_observe_0.01 | 只在 13 个安全层剪 1% | 0.36% |
| safe13_observe_0.02 | 只在 13 个安全层剪 2% | 0.72% |
| safe13_observe_0.03 | 只在 13 个安全层剪 3% | 1.08% |

建议保存路径：

```text
outputs/agent_qwen2_5_3b_4090_low/eval/safe_layer_observe_compare_100.jsonl
outputs/agent_qwen2_5_3b_4090_low/eval/safe_layer_observe_compare_100_metrics.json
logs/safe_layer_observe_compare_100_<timestamp>.log
```

判断标准：

```text
如果 safe13_observe_0.03 接近 dense，说明全局约 1% budget 可通过 layer-wise 分配实现。
如果 safe13_observe_0.03 明显优于 full_stage_observe_0.01，但仍低于 dense，说明 layer-wise budget 有效但还需要 compensation。
如果 safe13_observe_0.01 也明显低于 dense，说明 channel zeroing 本身仍太破坏，需要换策略。
```

实际结果落在第二类和第一类之间：

```text
safe13_observe_0.01 接近 dense。
safe13_observe_0.03 明显高于 full_stage_observe_0.01，但仍低于 dense。
```

### 7.6 Safe-layer-only 结果

本轮修正了 lookup prompt，使 dense lookup 恢复到可评价状态。

远端保存路径：

```text
outputs/agent_qwen2_5_3b_4090_low/eval/safe_layer_observe_compare_100.jsonl
outputs/agent_qwen2_5_3b_4090_low/eval/safe_layer_observe_compare_100_metrics.json
outputs/agent_qwen2_5_3b_4090_low/eval/safe_layer_observe_compare_lookupfixed_100.jsonl
outputs/agent_qwen2_5_3b_4090_low/eval/safe_layer_observe_compare_lookupfixed_100_metrics.json
```

汇总结果：

| Method | Correct | Accuracy | Active FFN ratio | Collapse |
|---|---:|---:|---:|---:|
| dense | 97/100 | 0.97 | 1.0000 | 0 |
| identity_hook | 97/100 | 0.97 | 1.0000 | 0 |
| full_stage_observe_0.01 | 86/100 | 0.86 | 0.9900 | 0 |
| safe13_observe_0.01 | 96/100 | 0.96 | 0.9964 | 0 |
| safe13_observe_0.02 | 91/100 | 0.91 | 0.9928 | 0 |
| safe13_observe_0.03 | 89/100 | 0.89 | 0.9892 | 0 |

分工具结果：

| Method | calculator | lookup | unit_convert |
|---|---:|---:|---:|
| dense | 32/34 | 33/33 | 32/33 |
| identity_hook | 32/34 | 33/33 | 32/33 |
| full_stage_observe_0.01 | 29/34 | 32/33 | 25/33 |
| safe13_observe_0.01 | 31/34 | 33/33 | 32/33 |
| safe13_observe_0.02 | 33/34 | 28/33 | 30/33 |
| safe13_observe_0.03 | 32/34 | 29/33 | 28/33 |

结论：

```text
safe-layer-only 明显缓解了 36 层平均剪枝的累积扰动。
safe13_observe_0.01 几乎贴近 dense，说明安全层选择有效。
safe13_observe_0.03 在近似 1% global budget 下高于 full_stage_observe_0.01，
但仍低于 dense，说明下一步需要真正的 global layer-wise allocation。
```

### 7.7 Global layer-wise 结果

在 safe-layer-only 后，进一步构造了三个近似 1% global budget 的非均匀分配。
三者都使用 `stage_observe` mask，且 active FFN ratio 与 `full_stage_observe_0.01`
保持一致：

```text
active FFN ratio: 0.9900
pruned channels: 3960
```

远端保存路径：

```text
outputs/agent_qwen2_5_3b_4090_low/eval/global_layerwise_observe_compare_100.jsonl
outputs/agent_qwen2_5_3b_4090_low/eval/global_layerwise_observe_compare_100_metrics.json
outputs/agent_qwen2_5_3b_4090_low/eval/global_layerwise_observe_compare_100_plans.json
logs/global_layerwise_observe_compare_100_20260611_194733.log
```

方法定义：

| Method | Layer allocation |
|---|---|
| global_spread_observe_approx_0.01 | safe13 层剪 2%，另选 10 个次安全层剪 1% |
| global_balanced_observe_approx_0.01 | 最安全 8 层剪 3%，其余 12 个安全/次安全层剪 1% |
| global_concentrated_observe_approx_0.01 | 最安全 7 层剪 5%，再选 1 层剪 1% |

汇总结果：

| Method | Correct | Accuracy | Active FFN ratio | Collapse |
|---|---:|---:|---:|---:|
| dense | 97/100 | 0.97 | 1.0000 | 0 |
| identity_hook | 97/100 | 0.97 | 1.0000 | 0 |
| full_stage_observe_0.01 | 86/100 | 0.86 | 0.9900 | 0 |
| safe13_observe_0.03 | 89/100 | 0.89 | 0.9892 | 0 |
| global_spread_observe_approx_0.01 | 87/100 | 0.87 | 0.9900 | 0 |
| global_balanced_observe_approx_0.01 | 95/100 | 0.95 | 0.9900 | 0 |
| global_concentrated_observe_approx_0.01 | 95/100 | 0.95 | 0.9900 | 0 |

分工具结果：

| Method | calculator | lookup | unit_convert |
|---|---:|---:|---:|
| dense | 32/34 | 33/33 | 32/33 |
| identity_hook | 32/34 | 33/33 | 32/33 |
| full_stage_observe_0.01 | 29/34 | 32/33 | 25/33 |
| safe13_observe_0.03 | 32/34 | 29/33 | 28/33 |
| global_spread_observe_approx_0.01 | 32/34 | 26/33 | 29/33 |
| global_balanced_observe_approx_0.01 | 31/34 | 33/33 | 31/33 |
| global_concentrated_observe_approx_0.01 | 31/34 | 33/33 | 31/33 |

结论：

```text
global_balanced 和 global_concentrated 在同样 0.9900 active FFN ratio 下达到 95/100，
比 full_stage_observe_0.01 高 9 个点，
比 safe13_observe_0.03 高 6 个点，
只比 dense / identity_hook 低 2 个点。
```

这说明小实验的核心假设成立：

```text
full per-layer 1% 失败主要来自层间预算分配错误。
基于 layer sensitivity 的 global layer-wise allocation 可以显著缓解退化。
stage_observe 的通道重要性信号可以继续作为后续方法的基础。
```

同时，`global_spread` 只有 87/100，说明不是“多找一些层平均分配”就能解决问题。
真正有效的是优先把 budget 分给最安全层，并且避免把低置信度层纳入全局预算。

后续工程化已完成：临时 global allocation 已固化进正式 evaluator，并进一步补了
stage-aware 与 failure re-densification 的初步对照。

做法：

```text
1. 给每层一个 sensitivity penalty。
2. 在全局 FFN budget 下，优先剪单层 sensitivity 低的层。
3. 敏感层 0 pruning，安全层允许超过 1%。
4. 不再固定每层剪同样数量。
```

### 7.8 Stage-aware 与 failure re-densification 初步结果

正式 evaluator 已支持：

```text
stage_global_balanced_approx_0.01
failure_redense_global_balanced_approx_0.01
```

其中：

```text
stage_global_balanced_approx_0.01:
按 trajectory step 的 stage 选择对应 stage-specific global_balanced plan，
最终 answer 生成使用 answer stage plan，active cost 按全轨迹 step 平均。

failure_redense_global_balanced_approx_0.01:
使用 FFNMaskRouter。
正常 step 使用 stage-specific global_balanced plan。
failure / recovery window 内切到 dense fallback。
```

远端保存路径：

```text
outputs/agent_qwen2_5_3b_4090_low/eval/stage_failure_official_v2_100.jsonl
outputs/agent_qwen2_5_3b_4090_low/eval/stage_failure_official_v2_100_metrics.json
outputs/agent_qwen2_5_3b_4090_low/eval/stage_failure_official_v2_100_plans.json
logs/stage_failure_official_v2_100_20260611_202211.log
```

汇总结果：

| Method | Correct | Accuracy | Active FFN ratio | Collapse |
|---|---:|---:|---:|---:|
| dense | 97/100 | 0.97 | 1.0000 | 0 |
| identity_hook | 97/100 | 0.97 | 1.0000 | 0 |
| global_balanced_observe_approx_0.01 | 95/100 | 0.95 | 0.9900 | 0 |
| stage_global_balanced_approx_0.01 | 96/100 | 0.96 | 0.9900 | 0 |
| failure_redense_global_balanced_approx_0.01 | 97/100 | 0.97 | 0.9980 | 0 |

分工具结果：

| Method | calculator | lookup | unit_convert |
|---|---:|---:|---:|
| dense | 32/34 | 33/33 | 32/33 |
| identity_hook | 32/34 | 33/33 | 32/33 |
| global_balanced_observe_approx_0.01 | 31/34 | 33/33 | 31/33 |
| stage_global_balanced_approx_0.01 | 33/34 | 33/33 | 30/33 |
| failure_redense_global_balanced_approx_0.01 | 33/34 | 33/33 | 31/33 |

结论：

```text
stage-aware dynamic plan 在同样 0.9900 active FFN ratio 下达到 96/100，
高于 observe-only global_balanced 的 95/100。

failure re-densification 达到 97/100，追平 dense，
但 active FFN ratio 上升到 0.9980。
```

这说明 stage-aware 与 failure feedback 方向有继续价值；但当前实现仍是
mask-based algorithm validation，不能解释为真实 wall-clock speedup。
下一步应把 recovery_success_rate、cost_per_success 和失败子集指标正式化。

### 7.9 后续增强

引入更强 baseline 或恢复机制。

参考 Qwen2.5-3B 压缩顺序论文的启发，后续可以补：

```text
weight + activation 0.5/0.5 的 Qwen FFN baseline
teacher KL / KD recovery
failure-triggered re-densification
```

其中 KD / recovery 不作为当前小实验第一步，但如果目标回到 10%-30% FFN structured pruning，它会变成必要组件。

## 8. 一句话结论

这轮小实验已经得到一个可以收尾的结论：

```text
Qwen2.5-3B 对 naive per-layer FFN channel zeroing 极敏感，
1% 每层固定剪枝也会明显掉分。

但 stage-specific activation 不只是“有信号”：
stage_observe_0.01 明显优于 static_plan_0.01，
说明 observe 阶段选出的低重要性通道更不容易伤害当前 controlled tasks。

因此方向不应立即放弃，但当前主方法必须从“每层固定比例剪”
升级为 layer sensitivity + layer-wise/global budget，
否则继续扫 sparsity 没有意义。

Layer sensitivity 进一步说明：
多数单层剪枝接近 dense；
full 1% 失败主要来自 36 层同时剪枝的累积扰动。

safe-layer-only 进一步证明安全层选择有效：
safe13_observe_0.01 达到 96/100，几乎贴近 dense 的 97/100。

最终 global layer-wise 对照证明小实验成功：
global_balanced_observe_approx_0.01 和 global_concentrated_observe_approx_0.01
在同样 0.9900 active FFN ratio 下达到 95/100，
明显优于 full_stage_observe_0.01 的 86/100。

正式 evaluator 工程化后，stage/failure 初步对照继续支持该方向：
stage_global_balanced_approx_0.01 达到 96/100，
failure_redense_global_balanced_approx_0.01 以 0.9980 active FFN ratio 达到 97/100。

因此下一阶段应正式化 recovery_success_rate、cost_per_success 和失败子集指标，
再决定是否扩展到更大任务集或 7B。
```

## 9. 2026-06-12 更新：Controlled v1 已冻结，进入 Real Tool v1

### 9.1 做了什么

1000 条 controlled direct-answer 结果已冻结到：

```text
docs/controlled_mask_validation_v1.md
outputs/agent_qwen2_5_3b_4090_low/eval/stage_failure_official_1000.jsonl
outputs/agent_qwen2_5_3b_4090_low/eval/stage_failure_official_1000_metrics.json
outputs/agent_qwen2_5_3b_4090_low/eval/stage_failure_official_1000_plans.json
```

新增真实工具执行闭环：

```text
strict JSON tool protocol
local calculator / unit_convert / lookup execution
deterministic fault_schedule
natural loop Agent runner
real tool smoke / 1000 eval script
hidden-state centroid baseline
```

对应代码：

```text
src/safeprune/tool_protocol.py
src/safeprune/tool_env.py
src/safeprune/agent_loop.py
src/safeprune/tools/
src/safeprune/hidden_state_router.py
scripts/prepare_real_tool_tasks.py
scripts/evaluate_real_tool_loop.py
```

### 9.2 为什么要做

controlled direct-answer 任务已经足够证明 mask/routing 机制信号，但不是完整 Agent 工具使用。
旧流程是：

```text
prompt -> model final answer -> exact match
```

真实 Agent Prune 应该是：

```text
prompt -> JSON tool_call -> Python tool execution -> observation -> final answer
```

原因是 dense 在 controlled calculator 上只有：

```text
258/334 = 77.2%
```

这说明直接回答任务混入了模型心算能力。真实工具闭环可以把评测重点转到：

```text
工具选择
JSON/schema validity
参数正确性
observation 使用
失败恢复
每个成功任务的 active FFN cost
```

### 9.3 当前结果与状态

Controlled v1 1000 条结果：

| Method | Correct | Active FFN | Cost / success |
|---|---:|---:|---:|
| dense | 913/1000 | 1.0000 | 1.0953 |
| identity_hook | 913/1000 | 1.0000 | 1.0953 |
| global_balanced_observe_approx_0.01 | 836/1000 | 0.9900 | 1.1842 |
| stage_global_balanced_approx_0.01 | 898/1000 | 0.9900 | 1.1025 |
| failure_redense_global_balanced_approx_0.01 | 900/1000 | 0.9916 | 1.1018 |

结论：

```text
stage-aware routing 在同样 active FFN ratio 下明显优于 observe-only。
failure-redense 恢复 failure subset，但整体 cost_per_success 未超过 dense。
```

Real Tool v1 历史 smoke 诊断：

```text
本地实现已完成。
100 条 smoke task 已在远端生成并跑完一次。
第一次有效 smoke 未通过 gate：
  dense / identity_hook = 66/100
  dense schema_validity_rate = 0.813
  stage-aware = 47/100
  observe-only = 33/100
  failure-redense = 66/100，但 dense_fallback_rate=1.0
原因主要是协议和 runner 实现问题，而不是正式 pruning 结论：
  数字 final answer 被旧 parser 判成 schema_error；
  failure-redense 首轮 start event 被误判成 failure。
本地已修复 final parser、observation prompt、failure-redense start event。
下一步是远端 pull 后重跑 100 条 smoke。
```

当前不要继续扩展工具集、stateful DB、7B 或 RL router。先让 real tool smoke 的 dense/identity、schema validity 和 stage/failure 对照稳定。

## 10. 2026-06-13 更新：Real Tool v1 已冻结

真实工具闭环 1000 条结果已冻结到：

```text
docs/real_tool_v1_results.md
outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_bypass_1000.jsonl
outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_bypass_1000_metrics.json
outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_bypass_1000_pairwise.json
outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_bypass_1000_failures.jsonl
```

主表：

| Method | Success | Failure | Non-failure | Cost / success | Fallback step ratio |
|---|---:|---:|---:|---:|---:|
| dense | 997/1000 | 197/200 | 800/800 | 2.2066 | 0.0000 |
| identity_hook | 997/1000 | 197/200 | 800/800 | 2.2066 | 0.0000 |
| observe-only | 951/1000 | 151/200 | 800/800 | 2.4349 | 0.0000 |
| stage-aware | 933/1000 | 133/200 | 800/800 | 2.6878 | 0.0000 |
| failure-redense | 996/1000 | 196/200 | 800/800 | 2.1938 | 0.1829 |
| observe-failure-redense | 997/1000 | 197/200 | 800/800 | 2.1886 | 0.1818 |
| stage-reflect-dense | 996/1000 | 196/200 | 800/800 | 2.1918 | 0.0926 |
| stage-reflect-observe | 946/1000 | 146/200 | 800/800 | 2.4551 | 0.0000 |

结论：

```text
Real Tool v1 不支持 “stage-aware mask switching 全面优于 observe-only”。
stage-aware 在 non-failure 上稳定，但 failure recovery 弱于 observe-only。

真正成立的机制结论是：
failure recovery 是 reflect-step 局部计算瓶颈；
只在 reflect recovery step 做 dense fallback，
可以把 failure subset 恢复到 196/200，
同时把 fallback step ratio 从窗口式 redense 的 0.1829 降到 0.0926。
```

论文叙事应从：

```text
Stage-Aware Dynamic FFN Pruning
```

收紧为：

```text
Trajectory-Event-Aware FFN Compute Allocation
```

下一步：

```text
1. 跑 hidden-state centroid baseline。
2. 做 3% / 5% / 10% global FFN budget ladder。
3. 研究 FLAP-style scoring、layer-wise budget、bias/scale compensation。
```
