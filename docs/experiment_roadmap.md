# Agent 轨迹动态 FFN 剪枝实验路线图

本文档是 `D:\Prune` 当前主线。项目已从 SafePrune-DPO 对齐恢复转向：

```text
Trajectory-Event-Aware FFN Compute Allocation for Tool-Using Reasoning Agents
```

核心问题：

> Agent failure / reflect 事件、工具 observation、阶段和 hidden state，是否能定位高价值 FFN 计算步骤，并比固定 mask 或 hidden-state-only 路由更有效地分配恢复计算？

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
| Controlled Mask Validation v1 | `[DONE]` | 1000 条 direct-answer controlled mask 结果已冻结到 `docs/controlled_mask_validation_v1.md`。 |
| Real Tool-Execution Agent Prune v1 | `[DONE]` | 1000 条真实工具闭环结果已冻结到 `docs/real_tool_v1_results.md`。 |
| Hidden-state centroid router | `[IN PROGRESS]` | 加入 pure centroid、centroid reflect-dense、centroid+event hybrid 三个审稿基线。 |
| FLAP-style substrate v2 | `[NEXT]` | 升级底层 FFN scoring、global layer-wise budget 和 compensation。 |
| Budget ladder | `[NEXT]` | 在 substrate v2 上推进 3% / 5% / 10% global FFN budget。 |
| SliceGPT 复现 | `[BASELINE]` | 外部仓库复现，不 vendoring 到本项目；不再阻塞 Agent Prune v1。 |
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

## 2. 外部结构压缩基线：SliceGPT

SliceGPT 仍是外部结构压缩与效率测量基线，但不再阻塞当前主线。
当前主线已经进入 Real Tool-Execution Agent Prune v1。

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

Real Tool v1 后，论文故事从单纯 stage switching 收紧为：

```text
Trajectory-event-aware compute allocation
plus
Reflect-localized re-densification
```

输入信号：

| 信号 | 第一版用法 |
|---|---|
| `stage` | 标记 `plan/reflect/answer` 等步骤，但 Real Tool v1 不支持“stage-aware 全面优于 observe-only”的强结论。 |
| `event` | `tool_error/schema_error/timeout/premature_final` 定位 recovery bottleneck。 |
| `observation` | 第一版只通过 step text 和 event 进入统计，后续再建模。 |
| `reflect` | Real Tool v1 显示只在 reflect recovery step 做 dense fallback 即可接近固定窗口 redense。 |
| `hidden state` | P1 加入 centroid router、reflect-dense centroid 和 event hybrid，检验外部轨迹信号是否超过 hidden-state-only。 |

Real Tool v1 冻结策略：

| Method | 用途 | 结论 |
|---|---|---|
| `observe_failure_redense_global_balanced_approx_0.01` | 最佳工程策略 | 997/1000，cost per success 低于 dense。 |
| `stage_reflect_dense_global_balanced_approx_0.01` | 机制消融 | 只在 reflect step dense，fallback step ratio 比窗口式 redense 低约 49%。 |
| `stage_global_balanced_approx_0.01` | stage-aware baseline | non-failure 稳定，但 failure recovery 弱于 observe-only。 |

当前底层 mask 仍是约 1% global FFN budget。后续 3% / 5% / 10% 需要先升级 layer-wise budget 和 compensation，不直接恢复旧的 10%/20%/30% per-layer stage sparsity。

## 4. 实验顺序

1. `[DONE]` 冻结 controlled mask validation v1：

```text
docs/controlled_mask_validation_v1.md
outputs/agent_qwen2_5_3b_4090_low/eval/stage_failure_official_1000*.json
```

2. `[LOCAL]` 跑单元测试和 dry-run：

```bash
python -m unittest discover -s tests
python scripts/build_stage_mask_bank.py --config configs/agent_qwen2_5_1_5b_4090.yaml --skip-model-load
python scripts/evaluate_agent_masks.py --config configs/agent_qwen2_5_1_5b_4090.yaml --dry-run
```

3. `[REMOTE]` 生成真实工具执行 smoke 任务：

```bash
export PYTHONPATH=$PWD:$PWD/src
python scripts/prepare_real_tool_tasks.py \
  --output data/agent/real_tool_tasks_v1_smoke_100.jsonl \
  --count 100 \
  --failure-count 20
```

4. `[REMOTE]` 跑 Qwen2.5-3B real tool smoke：

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

5. `[DONE]` 跑 1000 条 real tool v1 bypass evaluation：

```text
outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_bypass_1000*
docs/real_tool_v1_results.md
```

6. `[IN PROGRESS]` 加入 hidden-state centroid router 对照：

```bash
python scripts/prepare_real_tool_tasks.py \
  --output data/agent/real_tool_tasks_v1_centroid_calib_500.jsonl \
  --count 500 \
  --failure-count 100 \
  --start-index 1000

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
  --save-centroid-router outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/hidden_state_centroid_router_v1.json \
  --output-prefix outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_hidden_centroid_1000
```

7. `[NEXT]` 实现 FLAP-style substrate v2。
8. `[NEXT]` 做 3% / 5% / 10% global FFN budget ladder。
9. `[BASELINE]` 复现 SliceGPT / Probe Pruning，不 vendoring 到本项目。

## 5. 必须对照

| 方法 | 目的 |
|---|---|
| Dense | 任务成功率上限。 |
| Static Global Mask | 固定 FFN mask 基线。 |
| Stage Mask | 检查阶段语义是否有价值；Real Tool v1 中它不是最强方法。 |
| Failure Re-densification | 检查失败后恢复计算是否降低重试成本。 |
| Reflect-localized Dense | 检查 recovery 计算是否集中在 reflect step。 |
| Hidden-State-Only Router | 对齐 Probe Pruning 式动态信号。 |

## 6. 论文边界

当前贡献只能写成：

1. 提出 Agent 轨迹事件感知 FFN 计算分配问题。
2. 建立阶段敏感度画像和 stage mask bank。
3. 验证失败触发与 reflect-localized re-densification 是否改善 recovery rate。
4. 用 cost per successful task 评价压缩后 Agent，而不是只看 PPL 或平均 sparsity。

第一版不声称：

```text
真实 wall-clock speedup
attention head 安全剪枝
KV cache 压缩
RL controller
Triton kernel 优化
```
