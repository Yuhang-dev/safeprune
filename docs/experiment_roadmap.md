# Agent 轨迹动态 FFN 剪枝实验路线图

本文档是 `D:\Prune` 当前主线。项目已从 SafePrune-DPO 对齐恢复转向：

```text
Trajectory-Aware Dynamic FFN Pruning for Tool-Using Reasoning Agents
```

核心问题：

> Agent 阶段、工具 observation、失败事件和难度信号，是否能比固定 mask 或 hidden-state-only 路由更有效地分配 FFN 计算？

主指标：

```text
cost per successful task = total active FFN cost / successful tasks
```

## 0. 当前状态

| 模块 | 状态 | 说明 |
|---|---|---|
| 旧 4090 smoke | `[DONE]` | 证明旧 DPO/pruning 链路可运行，但 mixed attention+MLP pruning 不稳定。 |
| Agent 数据 schema | `[DONE]` | 新增 `AgentStep` / `AgentTrajectory` JSONL loader。 |
| Stage mask bank | `[DONE]` | 新增 FFN-only stage mask bank 数据结构与 dry-run 构建脚本。 |
| Rule router | `[DONE]` | 新增 stage default + failure-triggered re-densification 路由器。 |
| Agent metrics | `[DONE]` | 新增 task success、tool validity、recovery rate、active FFN cost 指标。 |
| SliceGPT 复现 | `[TODO]` | 外部仓库复现，不 vendoring 到本项目。 |
| 真实 Agent generation eval | `[TODO]` | 远端接 Qwen 生成、工具执行和自动判分。 |
| compact FFN / kernel | `[TODO]` | 第一版只做 mask-based 算法验证，不声称真实加速。 |

## 1. 为什么转向 Agent 动态 FFN

旧 SafePrune-DPO 4090 smoke 的关键观察：

```text
5% MLP-only: 不崩，recovery 部分改善。
10% MLP-only: 训练指标健康，但生成明显退化。
10% mixed head+MLP: 接近 collapse。
25% mixed head+MLP: 完全 collapse。
```

因此当前主线不再直接推进 25%/35%/50% mixed structured pruning。第一版只研究 FFN channel mask，并把 attention head、KV cache、安全路径和 Triton kernel 放到后续扩展。

## 2. 最小复现目标：SliceGPT

第一阶段复现 SliceGPT，目的不是替代本项目方法，而是建立 2x4090 上的压缩与效率测量基线。

模型矩阵：

| 模型 | 用途 |
|---|---|
| `facebook/opt-125m` | 环境和流程 smoke。 |
| `facebook/opt-1.3b` | 小模型剪枝敏感性。 |
| `facebook/opt-2.7b` | 中等规模趋势确认。 |
| `microsoft/phi-2` | 跨架构确认。 |

剪枝率：

```text
10%, 20%, 25%, 30%
```

记录：

```text
WikiText-2 PPL
PIQA 或 ARC-Easy
模型权重大小
峰值显存
tokens/s
```

具体命令见 `docs/slicegpt_repro_commands.md`。

## 3. 本项目 MVP

方法：

```text
Stage-specific FFN mask bank
plus
Failure-triggered re-densification
```

输入信号：

| 信号 | 第一版用法 |
|---|---|
| `stage` | `plan/reflect` 使用低稀疏，`observe/answer` 使用高稀疏。 |
| `event` | `tool_error/schema_error/timeout` 触发临时 re-densification。 |
| `observation` | 第一版只通过 step text 和 event 进入统计，后续再建模。 |
| `difficulty` | 第一版暂不训练 router，后续加入 entropy/margin/verifier。 |

默认策略：

| Stage | Sparsity |
|---|---:|
| plan | 10% |
| act | 20% |
| observe | 30% |
| reflect | 10% |
| answer | 30% |
| failure window | 10% |

## 4. 实验顺序

1. `[LOCAL]` 跑单元测试和 dry-run：

```bash
python -m unittest discover -s tests
python scripts/build_stage_mask_bank.py --config configs/agent_qwen2_5_1_5b_4090.yaml --skip-model-load
python scripts/evaluate_agent_masks.py --config configs/agent_qwen2_5_1_5b_4090.yaml --dry-run
```

2. `[REMOTE]` 复现 SliceGPT 小模型矩阵。
3. `[REMOTE]` 生成受控 Agent 任务：

```bash
python scripts/prepare_agent_tasks.py --count 1000
```

4. `[REMOTE]` 用 Qwen2.5-1.5B 生成 stage mask bank。
5. `[REMOTE]` 比较 dense、static mask、stage mask、failure re-densification。
6. `[REMOTE]` 扩到 Qwen2.5-3B。
7. `[REMOTE]` 跑 Probe Pruning 作为 hidden-state-only 动态剪枝对照。

## 5. 必须对照

| 方法 | 目的 |
|---|---|
| Dense | 任务成功率上限。 |
| Static Global Mask | 固定 FFN mask 基线。 |
| Stage Mask | 检查阶段语义是否有价值。 |
| Failure Re-densification | 检查失败后恢复计算是否降低重试成本。 |
| Hidden-State-Only Router | 对齐 Probe Pruning 式动态信号。 |

## 6. 论文边界

当前贡献只能写成：

1. 提出 Agent 轨迹感知 FFN 动态稀疏问题。
2. 建立阶段敏感度画像和 stage mask bank。
3. 验证失败触发 re-densification 是否改善 recovery rate。
4. 用 cost per successful task 评价压缩后 Agent，而不是只看 PPL 或平均 sparsity。

第一版不声称：

```text
真实 wall-clock speedup
attention head 安全剪枝
KV cache 压缩
RL controller
Triton kernel 优化
```
