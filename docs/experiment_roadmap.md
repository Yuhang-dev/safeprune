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
| Hidden-state centroid router | `[DONE]` | 1000 条 P1 已完成；hidden-state-only 弱于显式 event，event hybrid 追平但 router-inclusive cost 高。 |
| FLAP-style substrate v2 | `[DONE]` | 10% Real Tool 1000 已完成；稳定但非无损，cost_per_success 约降 5.8%。 |
| Budget ladder | `[DONE]` | 10% Real Tool 1000 已完成；5% 是 allocation anomaly，20% pressure smoke 失败。 |
| P2c adaptive budget routing | `[DONE]` | 1000 最小矩阵已完成；adaptive_B20 追平 dense 997/1000，cost_per_success 降到 1.9057。 |
| Identity no-op fix | `[DONE]` | `_empty_plan_like()` 已修复 stale bias compensation；P2c final table 需用修复后 sanity rerun 确认。 |
| Real Tool protocol v1.1 | `[DONE]` | 7B trace 显示 `unit_convert` 被写进 `type`；已加强 prompt/error feedback，parser 仍保持 strict。 |
| P4 Qwen2.5-7B extension | `[NEXT]` | 先重跑 protocol v1.1 dense/identity gate 和 20 条 minismoke；通过后再跑 100/1000。 |
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

P2 substrate v2 已经把 controlled prompt PPL sanity 推到 20%：

| Plan | Active MLP ratio | PPL | Relative PPL increase |
|---|---:|---:|---:|
| dense | 1.0000 | 23.2031 | 0.00% |
| flap 3% | 0.9697 | 23.6021 | +1.72% |
| flap 5% | 0.9497 | 23.6440 | +1.90% |
| flap 10% | 0.8998 | 24.7785 | +6.79% |
| flap 20% | 0.7998 | 24.0066 | +3.46% |

这组结果只说明 substrate v2 在 controlled-task prompt PPL 上没有 collapse；
论文仍需补标准 WikiText-2 / PIQA / HellaSwag / ARC-Easy。

P2 Real Tool 100 smoke 最新结果：

| Method | Success | Failure | Non-failure | Cost / success | Fallback step ratio | Collapse |
|---|---:|---:|---:|---:|---:|---:|
| dense | 100/100 | 20/20 | 80/80 | 2.2000 | 0.0000 | 0.0000 |
| flap 3% stage-reflect-dense | 100/100 | 20/20 | 80/80 | 2.1395 | 0.0909 | 0.0000 |
| flap 5% stage-reflect-dense | 66/100 | 13/20 | 53/80 | 5.1615 | 0.5244 | 0.0000 |
| flap 10% stage-reflect-dense | 99/100 | 19/20 | 80/80 | 2.0399 | 0.0991 | 0.0000 |
| flap 5% observe-failure-redense | 66/100 | 13/20 | 53/80 | 5.1714 | 0.5616 | 0.0000 |
| flap 10% observe-failure-redense | 99/100 | 19/20 | 80/80 | 2.0602 | 0.1892 | 0.0000 |

结论：

```text
3% 是安全档位。
10% 是当前主论文候选档位：Real Tool smoke 接近 dense，cost_per_success 低于 dense。
5% 不是保守档位，而是异常档位；PPL 很好但 Agent loop 大幅退化，
需要优先检查 layer allocation 和 failure trace。
10% 下 stage-reflect-dense 与 observe-failure-redense 同为 99/100，
但 stage-reflect-dense fallback step ratio 更低，因此优先作为 10% 主方法。
```

P2 plan audit 和 20% pressure smoke 结论：

```text
audit 显示 5%/10%/20% plan 文件没有 duplicate/out-of-range channel index，
bias compensation 和 layer scale placeholder 也都存在。
异常来自当前独立 budget search 的非嵌套 allocation：
flap_0p05_vs_flap_0p10 中，5% 有 9353 个 channel 不在 10% 里。

20% pressure smoke 失败：
substrate_flap_0p20_stage_reflect_dense = 60/100
failure subset = 9/20
non-failure = 51/80
schema_validity_rate = 0.5247
unit_convert = 0/33
```

评估代码审查结论：

```text
未发现会把 20% 误判为失败的 runner / parser / metrics bug。
20% 失败是 tool protocol / schema 控制能力被破坏，不进入 1000 主表。
```

P2 10% Real Tool 1000 正式结果：

| Method | Success | Failure | Non-failure | Cost / success | Fallback step ratio | Schema validity |
|---|---:|---:|---:|---:|---:|---:|
| dense | 997/1000 | 197/200 | 800/800 | 2.2066 | 0.0000 | 1.0000 |
| identity_hook | 997/1000 | 197/200 | 800/800 | 2.2066 | 0.0000 | 1.0000 |
| substrate flap 10% stage-reflect-dense | 990/1000 | 195/200 | 795/800 | 2.0777 | 0.1156 | 0.9876 |
| substrate flap 10% observe-failure-redense | 990/1000 | 195/200 | 795/800 | 2.0987 | 0.2114 | 0.9876 |

结论：

```text
10% substrate 不是无损压缩：相对 dense 少 7 条成功任务。
但它稳定通过 1000：failure recovery 195/200，collapse=0，schema validity 0.9876。
stage-reflect-dense 将 cost_per_success 从 2.2066 降到 2.0777，约降低 5.8%。
同为 990/1000 时，stage-reflect-dense 比 observe-failure-redense 的 fallback step
少约 45%，说明 reflect-localized recovery 在更高 10% FFN budget 下仍成立。
```

P2c schema-aware nested 100 smoke 已通过：

| Method | Success | Failure | Non-failure | Cost / success | Schema validity |
|---|---:|---:|---:|---:|---:|
| dense | 100/100 | 20/20 | 80/80 | 2.2000 | 1.0000 |
| nested_uniform_10 | 100/100 | 20/20 | 80/80 | 2.0000 | 1.0000 |
| adaptive_A | 100/100 | 20/20 | 80/80 | 2.0000 | 1.0000 |
| adaptive_B15 | 100/100 | 20/20 | 80/80 | 1.9500 | 1.0000 |
| adaptive_B20 | 100/100 | 20/20 | 80/80 | 1.9000 | 1.0000 |

P2c 的核心解释：

```text
schema-sensitive tool-call / retry generation 使用 10% conservative budget；
final-answer generation 使用 20% aggressive budget；
failure reflect recovery 保持 dense。

这说明 uniform 20% 会破坏工具协议，但按 generation type 分配预算后，
answer step 可以承受更高剪枝率，且 smoke 中 unit_convert 恢复到 33/33。
```

P2c 1000 最小对照矩阵已完成：

```text
dense
identity_hook
nested_uniform_10
adaptive_B20
```

| Method | Success | Failure | Non-failure | Cost / success | Schema validity |
|---|---:|---:|---:|---:|---:|
| dense | 997/1000 | 197/200 | 800/800 | 2.2066 | 1.0000 |
| identity_hook | 997/1000 | 197/200 | 800/800 | 2.2066 | 1.0000 |
| nested_uniform_10 | 997/1000 | 197/200 | 800/800 | 2.0060 | 1.0000 |
| adaptive_B20 | 997/1000 | 197/200 | 800/800 | 1.9057 | 1.0000 |

`nested_uniform_10` 证明统一 10% nested substrate 已经稳定；`adaptive_B20`
证明 generation-type routing 在保持相同成功率时进一步降低 cost_per_success。
相对 dense，`adaptive_B20` 的 cost_per_success 下降约 13.6%；相对
`nested_uniform_10` 下降约 5.0%。

Post-freeze correction:

```text
_empty_plan_like() previously left mlp_output_bias_compensation and
mlp_output_scale in derived identity plans. That could make identity_hook a
Dense + stale compensation run instead of a true no-op hook.
```

The code is fixed. Final paper tables should use a post-fix sanity rerun for
any `identity_hook` or dense-fallback comparison.

7B dense after the identity fix still failed because the model generated:

```json
{"type":"unit_convert","arguments":{"value":7,"from_unit":"m","to_unit":"cm"}}
```

instead of the required:

```json
{"type":"tool_call","name":"unit_convert","arguments":{"value":7,"from_unit":"m","to_unit":"cm"}}
```

This is a protocol-shape adherence problem, not a pruning result. Real Tool
protocol v1.1 hardens the prompt and schema-error feedback while keeping the
strict parser unchanged.

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

6. `[DONE]` 加入 hidden-state centroid router 对照：

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

P1 1000 结果已经冻结到 `docs/p2_substrate_and_p2c_results.md`。结论是：

```text
hidden-state-only centroid 不如显式 event；
event hybrid 追平 stage-reflect-dense，但 router-inclusive cost 明显更高。
```

7. `[DONE]` 实现 FLAP-style substrate v2：

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

8. `[DONE]` 对 substrate v2 plans 做 controlled prompt PPL sanity：

```bash
python scripts/evaluate_pruning_ppl.py \
  --config configs/agent_qwen2_5_3b_4090.yaml \
  --plan outputs/agent_qwen2_5_3b_4090_low/substrate_v2/flap/budget_plan_0p05.json \
  --prompts-jsonl data/agent/controlled_tasks.jsonl \
  --max-prompts 256 \
  --local-files-only \
  --output outputs/agent_qwen2_5_3b_4090_low/substrate_v2/flap/ppl_0p05.json
```

9. `[DONE]` 把通过静态 sanity 的 3% / 5% / 10% 接回 Real Tool 100 smoke：

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

10. `[DONE]` 分析 5% anomaly，不跑 5% 1000：

```text
用 scripts/audit_substrate_plans.py 检查 budget_plan_0p05.json 的 layer allocation、
plan nesting、overlap、重复/越界 channel index 和 bias compensation norm。
抽取 5% stage-reflect-dense / observe-failure-redense 失败 trace。
确认是否某些关键层被 5% allocation 误剪，或 bias compensation 在某些层引入过强偏置。
```

11. `[DONE]` 单独跑 20% pressure smoke；结果未通过，不进入 1000 主表。
12. `[DONE]` 以 10% stage-reflect-dense 为主候选跑 1000，只保留必要方法：
`dense`、`identity_hook`、`substrate_flap_0p10_stage_reflect_dense`、
`substrate_flap_0p10_observe_failure_redense`。
13. `[DONE]` 跑 P2c 1000 最小矩阵：
`dense`、`identity_hook`、`nested_uniform_10`、`adaptive_B20`。
14. `[DONE]` 修复 `_empty_plan_like()` stale bias compensation，补单测。
15. `[DONE]` 定位 7B unit_convert schema 错误，加入 Real Tool protocol v1.1 hardening。
16. `[NEXT]` 重跑 3B P2c 最小矩阵 sanity 和 7B protocol v1.1 dense/identity gate。
17. `[BASELINE]` 复现 SliceGPT / Probe Pruning，不 vendoring 到本项目。

P2c 已实现的入口：

```text
scripts/prepare_schema_calibration.py
scripts/build_pruning_substrate_v2.py --nested-budget-ladder --schema-calibration-path
scripts/evaluate_real_tool_loop.py methods:
  nested_uniform_10
  adaptive_A
  adaptive_B15
  adaptive_B20
```

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
