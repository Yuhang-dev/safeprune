# Agent FFN 动态剪枝路径说明

日期：2026-06-13

本文结合两份输入材料整理当前项目路径：

- `C:/Users/Yuhang/Downloads/大模型动态剪枝_技术路线报告.docx`
- `C:/Users/Yuhang/Downloads/agent_trajectory_pruning_2x4090_experiment_plan.md`

用途是把当前项目中已经完成的工程、实验与文档路径，以及下一步必须完成的路径说明清楚，避免继续沿旧的 SafePrune-DPO 主线误跑。

## 0. 当前主线

项目当前主线是：

```text
Trajectory-Aware Dynamic FFN Pruning for Tool-Using Reasoning Agents
```

核心问题是：

```text
Agent 的阶段标签、工具 observation、失败事件和难度信号，是否能比固定 mask 或 hidden-state-only 路由更有效地分配 FFN 计算？
```

技术报告把总方向拆成两个可独立推进的子方向：

| 子方向 | 含义 | 本项目当前对应实现 |
|---|---|---|
| Stage-Aware Dynamic FFN Pruning | 按 `plan/act/observe/reflect/answer` 阶段切换 FFN 子网络或稀疏率 | 已有 stage mask bank、规则 router 和 stage sparsity 配置 |
| Adaptive Re-Densification | 默认较高稀疏率，困难步骤或失败事件后临时恢复更多通道 | 已有 failure-triggered router scaffold，真实 generation eval 待接入 |

第一版只做 FFN channel mask，不做 attention head、KV cache、安全 circuit、RL controller 或 Triton kernel。mask-based 实验只能报告 active FFN ratio 和任务指标，不能声称真实 wall-clock speedup。

当前阶段已经更新为：

```text
Controlled Mask Validation v1 已冻结。
Real Tool-Execution Agent Prune v1 已冻结。
```

这意味着当前不再把“补 evaluator metrics / 跑 controlled 1000 条”作为下一步；
这些已经完成。现在也不再补同配置 real-tool 1000。主线已经转为：

```text
hidden-state centroid / reflect-dense / event-hybrid baseline
FLAP-style substrate v2
3% / 5% / 10% global FFN budget ladder
```

## 0.1 当前做了什么、为什么、结果是什么

### 做了什么

已完成 controlled direct-answer 1000 条验证，并冻结到：

```text
docs/controlled_mask_validation_v1.md
```

新增 real tool-execution 代码：

```text
src/safeprune/tool_protocol.py
src/safeprune/tool_env.py
src/safeprune/agent_loop.py
src/safeprune/tools/
src/safeprune/hidden_state_router.py
scripts/prepare_real_tool_tasks.py
scripts/evaluate_real_tool_loop.py
scripts/analyze_agent_pairwise_errors.py
```

### 为什么要做

controlled direct-answer 任务已经证明 stage-aware routing 有信号，但它不是完整 Agent 工具闭环。
尤其 dense calculator 只有：

```text
258/334 = 77.2%
```

这说明旧任务混入了模型心算能力。真实 Agent Prune 必须让模型生成工具调用，
让 Python 工具真实执行，再把 observation 回填给模型。

### 当前结果

controlled v1 结果：

| Method | Correct | Active FFN | Cost / success |
|---|---:|---:|---:|
| dense | 913/1000 | 1.0000 | 1.0953 |
| global_balanced_observe_approx_0.01 | 836/1000 | 0.9900 | 1.1842 |
| stage_global_balanced_approx_0.01 | 898/1000 | 0.9900 | 1.1025 |
| failure_redense_global_balanced_approx_0.01 | 900/1000 | 0.9916 | 1.1018 |

结论：

```text
stage-aware 比 observe-only 净多成功 62 条。
failure-redense 在 failure subset 追平 dense。
但 cost_per_success 仍未优于 dense，且 mask hook 不代表真实加速。
```

real tool v1 当前状态：

```text
1000 条真实工具闭环结果已完成并冻结到 docs/real_tool_v1_results.md。
dense 与 identity_hook 完全一致：997/1000。
所有方法 non-failure 都是 800/800，差异集中在 failure recovery。
observe-failure-redense 达到 997/1000，是当前最佳工程策略。
stage-reflect-dense 达到 996/1000，failure subset 196/200，
并将 fallback step ratio 从窗口式 redense 的 0.1829 降到 0.0926。
```

论文故事也随之收紧：

```text
不是 stage-aware mask switching 全面优于 observe-only；
而是 failure / reflect 轨迹信号可以定位 recovery 计算瓶颈，
只在 reflect step 恢复 dense 即可接近固定窗口 redense 的鲁棒性，并减少 dense fallback steps。
```

P1 hidden-state baseline 已进入实现阶段。方法分为三档：

```text
hidden_state_centroid_global_balanced_approx_0.01
hidden_state_centroid_reflect_dense_global_balanced_approx_0.01
hidden_state_centroid_event_reflect_dense_global_balanced_approx_0.01
```

这一步要回答：

```text
显式 failure / reflect 事件是否比模型 hidden state 自身提供更可靠、更低成本的恢复计算分配信号？
```

新增指标包括：

```text
reflect_detection_precision / recall / f1
false_dense_fallback_rate
missed_reflect_rate
routing_probe_cost
routing_probe_latency_ms
effective_cost_per_success
```

P2 pruning substrate v2 已接入 Real Tool runner。当前结论：

```text
controlled-task prompt PPL sanity 显示 3% / 5% / 10% / 20% 均未 collapse。
Real Tool 100 smoke 显示 3% 和 10% 通过，5% 是异常档位。
20% pressure smoke 已失败，主要表现为 unit_convert 0/33 和 schema_validity_rate 0.5247。
10% Real Tool 1000 已完成：stage-reflect-dense 达到 990/1000，failure 195/200，
cost_per_success 从 dense 的 2.2066 降到 2.0777。
```

P2 Real Tool 100 smoke：

| Method | Success | Failure | Non-failure | Cost / success | Fallback step ratio |
|---|---:|---:|---:|---:|---:|
| dense | 100/100 | 20/20 | 80/80 | 2.2000 | 0.0000 |
| flap 3% stage-reflect-dense | 100/100 | 20/20 | 80/80 | 2.1395 | 0.0909 |
| flap 5% stage-reflect-dense | 66/100 | 13/20 | 53/80 | 5.1615 | 0.5244 |
| flap 10% stage-reflect-dense | 99/100 | 19/20 | 80/80 | 2.0399 | 0.0991 |
| flap 5% observe-failure-redense | 66/100 | 13/20 | 53/80 | 5.1714 | 0.5616 |
| flap 10% observe-failure-redense | 99/100 | 19/20 | 80/80 | 2.0602 | 0.1892 |

P2 Real Tool 1000：

| Method | Success | Failure | Non-failure | Cost / success | Fallback step ratio |
|---|---:|---:|---:|---:|---:|
| dense | 997/1000 | 197/200 | 800/800 | 2.2066 | 0.0000 |
| identity_hook | 997/1000 | 197/200 | 800/800 | 2.2066 | 0.0000 |
| flap 10% stage-reflect-dense | 990/1000 | 195/200 | 795/800 | 2.0777 | 0.1156 |
| flap 10% observe-failure-redense | 990/1000 | 195/200 | 795/800 | 2.0987 | 0.2114 |

下一步：

```text
不要跑 5% 1000。
10% stage-reflect-dense 已完成正式 1000，是当前 P2 主结果。
20% 不进入 1000 主表。
后续做 schema-aware nested budget plans，并实现 stage-dependent substrate routing：
tool-call generation 用 safe budget，answer generation 用 higher budget，reflect 用 dense。
```

## 1. 已完成路径

### 1.1 主线文档切换

已把项目叙事从 SafePrune-DPO 调整为 Agent trajectory-aware FFN pruning。

| 路径 | 状态 | 说明 |
|---|---|---|
| `README.md` | 已完成 | 项目简介、环境、主命令已切到 Agent FFN pruning |
| `docs/experiment_roadmap.md` | 已完成 | 当前总路线图，明确 SliceGPT 复现和 Agent FFN MVP |
| `docs/rtx4090_feasibility_plan.md` | 已完成 | 2 x RTX 4090 执行计划，区分本地 CPU dry-run 与远端 GPU |
| `docs/remote_workflow.md` | 已完成 | 远端安装、任务生成、mask bank 和 eval checklist |
| `docs/slicegpt_repro_commands.md` | 已完成 | SliceGPT 官方仓库复现命令，外部仓库不 vendoring |
| `docs/4090_smoke_results.md` | 保留 | 旧 SafePrune-DPO smoke 作为负面证据和排雷记录 |

### 1.2 Agent 数据与核心结构

| 路径 | 状态 | 说明 |
|---|---|---|
| `src/safeprune/data.py` | 已完成 | 新增 `AgentStep` / `AgentTrajectory` JSONL schema；旧 preference loader 保留为 legacy |
| `src/safeprune/stage_masks.py` | 已完成 | 新增 `StageMaskBank`，保存 `stage -> sparsity -> layer -> pruned_mlp_channels` |
| `src/safeprune/router.py` | 已完成 | 新增规则路由器，支持 stage default sparsity 和 failure-triggered re-densification |
| `src/safeprune/evaluation.py` | 已完成 | 新增 task success、tool-call validity、recovery rate、active FFN cost、cost per success 指标 |
| `src/safeprune/masks.py` | 已完成 | 新增 Qwen 可切换 MLP mask handle，避免每步重复 attach hook |
| `src/safeprune/config.py` | 已完成 | 新增 Agent 配置段，旧配置仍可加载 |

### 1.3 脚本与配置路径

| 路径 | 状态 | 说明 |
|---|---|---|
| `scripts/prepare_agent_tasks.py` | 已完成 | 生成受控工具调用任务，覆盖 calculator、unit_convert、lookup 和失败事件 |
| `scripts/build_stage_mask_bank.py` | 已完成 | 支持 `--skip-model-load` dry-run 和远端真实 calibration |
| `scripts/evaluate_agent_masks.py` | 部分完成 | 当前可 dry-run 并写 manifest；真实 Qwen generation loop 待接入 |
| `configs/agent_qwen2_5_1_5b_4090.yaml` | 已完成 | 1.5B 远端配置，默认 FFN-only，stage sparsity 为 10/20/30% |
| `configs/agent_qwen2_5_3b_4090.yaml` | 已完成 | 3B 远端配置，默认 FFN-only，stage sparsity 为 10/20/30% |

### 1.4 单元测试与本地 CPU 验证

| 路径 | 状态 | 说明 |
|---|---|---|
| `tests/test_agent_data.py` | 已完成 | Agent task schema validation |
| `tests/test_stage_masks.py` | 已完成 | Stage mask bank nested mask 构建 |
| `tests/test_router.py` | 已完成 | Stage default 和 failure re-densification |
| `tests/test_evaluation.py` | 已完成 | cost per success 的零成功保护等指标 |
| `tests/test_config.py` | 已更新 | Agent config 字段加载 |

已完成过本地 CPU 验证：

```bash
python -m unittest discover -s tests
python scripts/build_stage_mask_bank.py --config configs/agent_qwen2_5_1_5b_4090.yaml --skip-model-load
python scripts/evaluate_agent_masks.py --config configs/agent_qwen2_5_1_5b_4090.yaml --dry-run
```

### 1.5 GitHub 与远端路径

已推送到 GitHub：

```text
repo: https://github.com/Yuhang-dev/safeprune
branch: main
commit: 4554a22 Refactor toward agent trajectory FFN pruning
```

远端工作目录：

```text
/root/autodl-tmp/safeprune/code/Prune
```

推荐远端环境：

```text
/root/autodl-tmp/safeprune/.venv_agent
HF cache: /root/autodl-tmp/safeprune/hf_cache
```

### 1.6 远端已完成实验事实

受控任务数据已生成并检查：

```text
tasks: 1000
tools: calculator 334 / unit_convert 333 / lookup 333
stages: plan/act/observe/reflect/answer 各 1000
events: ok 4600 / tool_error 200 / empty_observation 200
```

1.5B stage mask bank 已生成并验证：

```text
output: outputs/agent_qwen2_5_1_5b_4090/mask_bank/mask_bank.json
layers: 28
mlp channels/layer: 8960
sparsities: 0.10 / 0.20 / 0.30
```

1.5B 阶段 Jaccard：

```text
s=0.10: 0.6175-0.6660
s=0.20: 0.6331-0.6847
s=0.30: 0.6621-0.7123
```

3B stage mask bank 已生成并验证：

```text
output: outputs/agent_qwen2_5_3b_4090/mask_bank/mask_bank.json
manifest: complete
file size: about 23M
layers: 36
mlp channels/layer: 11008
sparsities: 0.10 / 0.20 / 0.30
```

3B 阶段 Jaccard：

```text
s=0.10: avg=0.6381 min=0.6076 max=0.6873
s=0.20: avg=0.6732 min=0.6520 max=0.7172
s=0.30: avg=0.6975 min=0.6791 max=0.7421
```

这些结果说明 stage-specific mask 没有退化成单一 global mask，但这只证明阶段统计存在差异，不证明 mask 可用。

### 1.7 Dense 与 mask 可用性事实

3B dense controlled generation 基线正常：

```text
Qwen2.5-3B-Instruct dense: 20/20 = 1.0
```

mask 结果显示当前 naive FFN channel masking 很脆弱：

```text
static_30: collapse
stage_answer_30: collapse
static_20 / stage_20: near collapse
static_10 / stage_10: clearly degraded
```

关键判断：

```text
当前 naive FFN channel mask prototype 不能复现论文级结构化剪枝效果。
下一步必须先修正 scoring 和评测，再从 1%-5% 稀疏率重新找可用边界。
```

### 1.8 最新本地未提交修正

当前工作区有一个未提交修改：

```text
src/safeprune/scoring.py
```

修改内容：

```text
activation scoring 从 gate_proj 输出改为 down_proj forward pre-hook 输入。
```

原因：

```text
Qwen SwiGLU 真正送入 down_proj 的中间激活是 SiLU(gate_proj(x)) * up_proj(x)。
之前只统计 gate_proj 输出，会偏离真实 FFN 通道贡献。
```

该修正已在本地文件中，但仍待：

```text
1. 本地测试
2. commit / push
3. 远端 pull
4. 重新生成低稀疏 mask bank
```

## 2. 待完成路径

### 2.1 P0：先把评测与 scoring 修正闭环

这是当前最高优先级。

| 路径 | 状态 | 要做什么 |
|---|---|---|
| `src/safeprune/scoring.py` | 待测试 / 待提交 | 跑单测，确认 down_proj pre-hook 不破坏旧 scoring |
| `scripts/evaluate_agent_masks.py` | 待实现 | 接真实 Qwen generation、mask switching、自动判分 |
| 远端临时 strict eval 脚本 | 待固化 | 禁止 `expected in pred` 这种 substring 判分 |
| `outputs/agent_qwen2_5_3b_4090_s0p05/` | 待重跑 | 用修正后的 activation scoring 重新生成 5% mask bank |
| 低稀疏扫描 | 待执行 | 优先扫 1%、2%、3%、5%，不要继续 20/30% |

严格判分要求：

```text
calculator: 解析最终整数，必须 exact match
unit_convert: 解析最终数值，允许单位文本但数值必须 exact match
lookup: 解析 owner_x，必须 exact match
```

不要再使用：

```python
ok = expected in prediction
```

因为它会把 `expected=9, pred=19` 或 `expected=900, pred=9000` 误判为正确。

### 2.2 P0：重新定义可用稀疏率边界

当前实验已证明 10%-30% 不适合作为第一阶段主区间。已重跑低稀疏：

```text
1%, 2%, 3%, 5%
```

100 条严格评测结论：

```text
dense: 93/100
identity_hook: 93/100
static_plan_0.01: 65/100
stage_observe_0.01: 79/100
stage_answer_0.01: 72/100
stage_answer_0.02: 53/100
stage_answer_0.03: 46/100
```

结论：

```text
stage signal exists, but full per-layer 1% mask is still too destructive.
Do not continue sweeping full per-layer sparsity.
```

保存路径见：

```text
docs/agent_mask_eval_results.md
```

### 2.2.1 P0：safe-layer-only 与 global layer-wise budget

Layer sensitivity 已显示：

```text
dense: 49/50
identity_hook: 49/50
full_stage_observe_0.01: 38/50
single-layer observe 0.01: mostly 48/50 or 49/50
```

这说明主要问题是 36 层同时剪枝的累积扰动，而不是某一层单剪必崩。

已测试 safe-layer-only：

```text
safe layers: [3, 6, 7, 8, 9, 10, 14, 15, 18, 23, 28, 30, 34]
source mask: stage_observe
```

方法：

| Method | 做法 | 近似全局 FFN 稀疏率 |
|---|---|---:|
| safe13_observe_0.01 | 只在 13 个安全层剪 1% | 0.36% |
| safe13_observe_0.02 | 只在 13 个安全层剪 2% | 0.72% |
| safe13_observe_0.03 | 只在 13 个安全层剪 3% | 1.08% |

结果：

| Method | Correct | Active FFN ratio |
|---|---:|---:|
| dense | 97/100 | 1.0000 |
| identity_hook | 97/100 | 1.0000 |
| full_stage_observe_0.01 | 86/100 | 0.9900 |
| safe13_observe_0.01 | 96/100 | 0.9964 |
| safe13_observe_0.02 | 91/100 | 0.9928 |
| safe13_observe_0.03 | 89/100 | 0.9892 |

保存路径：

```text
outputs/agent_qwen2_5_3b_4090_low/eval/safe_layer_observe_compare_100.jsonl
outputs/agent_qwen2_5_3b_4090_low/eval/safe_layer_observe_compare_100_metrics.json
outputs/agent_qwen2_5_3b_4090_low/eval/safe_layer_observe_compare_lookupfixed_100.jsonl
outputs/agent_qwen2_5_3b_4090_low/eval/safe_layer_observe_compare_lookupfixed_100_metrics.json
logs/safe_layer_observe_compare_100_<timestamp>.log
```

判断：

```text
safe13_observe_0.01 几乎贴近 dense。
safe13_observe_0.03 高于 full_stage_observe_0.01，但仍明显低于 dense。
safe-layer-only 证明安全层选择有效，但简单把 13 个安全层统一拉到 3% 还不够。
```

随后已测试真正的 global layer-wise budget。三种 allocation 都保持近似 1% global budget：

```text
active FFN ratio: 0.9900
pruned channels: 3960
```

结果：

| Method | Correct | Active FFN ratio | 说明 |
|---|---:|---:|---|
| full_stage_observe_0.01 | 86/100 | 0.9900 | 36 层平均剪 1% |
| safe13_observe_0.03 | 89/100 | 0.9892 | 13 个安全层统一剪 3% |
| global_spread_observe_approx_0.01 | 87/100 | 0.9900 | safe13 剪 2%，另加 10 个次安全层剪 1% |
| global_balanced_observe_approx_0.01 | 95/100 | 0.9900 | 最安全 8 层剪 3%，其余 12 层剪 1% |
| global_concentrated_observe_approx_0.01 | 95/100 | 0.9900 | 最安全 7 层剪 5%，另 1 层剪 1% |
| dense | 97/100 | 1.0000 | baseline |

保存路径：

```text
outputs/agent_qwen2_5_3b_4090_low/eval/global_layerwise_observe_compare_100.jsonl
outputs/agent_qwen2_5_3b_4090_low/eval/global_layerwise_observe_compare_100_metrics.json
outputs/agent_qwen2_5_3b_4090_low/eval/global_layerwise_observe_compare_100_plans.json
logs/global_layerwise_observe_compare_100_20260611_194733.log
```

最终判断：

```text
小实验成功。
stage_observe 信号有效。
per-layer fixed budget 是主要失败点。
safe-layer-only 能显著缓解退化。
global_balanced / global_concentrated 在同样 1% global budget 下达到 95/100，
明显优于 full_stage_observe_0.01 的 86/100，接近 dense 的 97/100。
```

### 2.3 P1：实现真实 Agent evaluation loop

`scripts/evaluate_agent_masks.py` 当前只是 dry-run。应扩展为真实评测入口。

最小实现：

```text
1. 加载 tokenizer / model
2. 加载 StageMaskBank
3. attach_qwen_switchable_mlp_masks
4. 对每条 controlled task 构造 prompt
5. 按 method 设置 mask plan
6. generate
7. 严格判分
8. 写 JSONL 明细和 metrics summary
```

必须输出：

```text
outputs/<experiment>/eval/predictions.jsonl
outputs/<experiment>/eval/metrics.json
outputs/<experiment>/eval/agent_eval_manifest.json
```

必须包含指标：

```text
task_success_rate
tool_call_validity
recovery_success_rate
average_active_ffn_ratio
latency_ms
cost_per_success
generation_collapse_rate
```

### 2.4 P1：stage-aware 与 failure re-densification 初步对照

技术报告的核心不是“有 stage mask bank”，而是：

```text
stage 信息和失败反馈是否改善每个成功任务的成本？
```

已补初步对照：

| 对照 | 要回答的问题 |
|---|---|
| global_balanced_observe_approx_0.01 | observe-only global budget baseline |
| stage_global_balanced_approx_0.01 | stage-specific global budget 是否更好 |
| failure_redense_global_balanced_approx_0.01 | 失败/恢复窗口 dense fallback 是否恢复成功率 |
| dense / identity_hook | 上限与 hook 污染检查 |

远端保存路径：

```text
outputs/agent_qwen2_5_3b_4090_low/eval/stage_failure_official_v2_100.jsonl
outputs/agent_qwen2_5_3b_4090_low/eval/stage_failure_official_v2_100_metrics.json
outputs/agent_qwen2_5_3b_4090_low/eval/stage_failure_official_v2_100_plans.json
logs/stage_failure_official_v2_100_20260611_202211.log
```

结果：

| Method | Correct | Active FFN ratio |
|---|---:|---:|
| dense | 97/100 | 1.0000 |
| identity_hook | 97/100 | 1.0000 |
| global_balanced_observe_approx_0.01 | 95/100 | 0.9900 |
| stage_global_balanced_approx_0.01 | 96/100 | 0.9900 |
| failure_redense_global_balanced_approx_0.01 | 97/100 | 0.9980 |

判断：

```text
stage-aware dynamic plan 在同样 active FFN ratio 下优于 observe-only global baseline。
failure re-densification 追平 dense，但 active FFN ratio 上升。
下一步应正式化 recovery_success_rate、cost_per_success 和失败子集指标。
```

### 2.5 P1：SliceGPT 复现路径

实验计划要求先复现 SliceGPT，原因是它能建立真实结构压缩和效率基线。

外部路径：

```text
TransformerCompression/
```

不要放进：

```text
src/
```

复现矩阵：

```text
models: facebook/opt-125m, facebook/opt-1.3b, facebook/opt-2.7b, microsoft/phi-2
sparsity: 0.10, 0.20, 0.25, 0.30
metrics: WikiText-2 PPL, PIQA/ARC-Easy, model size, peak memory, tokens/s
```

仓库内记录路径：

```text
docs/slicegpt_repro_commands.md
```

建议新增远端结果记录：

```text
docs/slicegpt_2x4090_results.md
```

### 2.6 P2：增强 pruning 方法，而不是继续提高 sparsity

如果 1%-5% 仍明显退化，下一步不是继续调 prompt，而是增强剪枝策略。

候选增强：

| 方向 | 作用 |
|---|---|
| layer-wise budget | 避免每层固定比例同时扰动 |
| global budget allocation | 把稀疏率分配到低敏感层 |
| loss-delta scoring | 用输出 loss 变化估计通道重要性 |
| FLAP-style fluctuation | 加入 activation fluctuation 和补偿思想 |
| bias / scale compensation | 缓解置零导致的分布突变 |
| compact FFN slicing | 从 mask-based 验证转向真实结构压缩 |

### 2.7 P2：Probe Pruning / hidden-state-only 动态基线

在本项目 stage/failure MVP 稳定后，再引入 Probe Pruning 作为 hidden-state-only 动态剪枝基线。

路径：

```text
external Probe_Pruning repo -> LLaMA/OPT 原始复现 -> Qwen2.5 迁移
```

比较目标：

```text
trajectory-aware > hidden-state-only
```

如果 stage/failure 信号不能超过 hidden-state-only，就不能写成 Agent 轨迹信号有额外价值。

## 3. 当前不应继续做的路径

| 路径 | 原因 |
|---|---|
| 继续跑 20% / 30% Qwen FFN mask | 已经出现 collapse，浪费远端时间 |
| 重新尝试 attention head pruning | 旧 smoke 显示 mixed head+MLP 高风险 |
| 声称 mask-based wall-clock speedup | 通道乘零不减少 GEMM 计算 |
| 直接上 7B Agent 主实验 | 3B 的低稀疏可用边界还没找到 |
| 训练复杂 RL router | 规则 stage/failure MVP 尚未证明有效 |
| 把 SliceGPT / Probe Pruning 外部仓库 vendoring 到本项目 | 会污染项目边界和维护成本 |

## 4. 推荐下一步执行顺序

### 当前 Step 1：hidden-state centroid baseline

Real Tool v1 已冻结。下一步不再补同配置 1000，而是加入：

```text
hidden_state_centroid_global_balanced_approx_0.01
hidden_state_centroid_reflect_dense_global_balanced_approx_0.01
hidden_state_centroid_event_reflect_dense_global_balanced_approx_0.01
```

目的：

```text
回答 hidden-state-only routing 是否能替代显式 failure / reflect 轨迹信号，
以及显式 event override 是否能以接近零额外成本补强 hidden-state 路由。
```

### 历史 Step：real tool smoke

重跑前需要确认已经包含以下修复：

```text
final answer 接受 string / number。
observation prompt 指示 ok=true 后直接 final。
failure-redense 不把 start event 当作 failure。
```

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

判断：

```text
dense 与 identity_hook 对齐。
dense task_success_rate >= 0.90。
dense schema_validity_rate >= 0.95。
stage-aware 优于 observe-only。
failure-redense 提升 failure subset。
```

### 历史 Step：real tool 1000

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

Real Tool v1 1000 已完成并冻结，当前不要继续补同配置 1000。

### 当前 Step 2：运行 hidden-state centroid baseline

方法名：

```text
hidden_state_centroid_global_balanced_approx_0.01
hidden_state_centroid_reflect_dense_global_balanced_approx_0.01
hidden_state_centroid_event_reflect_dense_global_balanced_approx_0.01
```

目的：

```text
回答显式 stage/event 轨迹信号是否优于 hidden-state-only 动态路由。
```

### 历史 Step：本地收尾

```bash
python -m unittest discover -s tests
python scripts/build_stage_mask_bank.py --config configs/agent_qwen2_5_1_5b_4090.yaml --skip-model-load
python scripts/evaluate_agent_masks.py --config configs/agent_qwen2_5_1_5b_4090.yaml --dry-run
```

如果通过，提交 `src/safeprune/scoring.py` 的 SwiGLU activation scoring 修正。

### Step 2：远端同步

```bash
cd /root/autodl-tmp/safeprune/code/Prune
git pull origin main
source /root/autodl-tmp/safeprune/.venv_agent/bin/activate
source /etc/network_turbo || true
export PYTHONPATH=$PWD:$PWD/src
export HF_HOME=/root/autodl-tmp/safeprune/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/safeprune/hf_cache/hub
export HF_DATASETS_CACHE=/root/autodl-tmp/safeprune/hf_cache/datasets
export HF_HUB_DISABLE_XET=1
```

### Step 3：重建低稀疏 mask bank

先从 5% 开始，若仍差，再降到 1%-3%。

建议新增配置：

```text
configs/agent_qwen2_5_3b_4090_s0p01.yaml
configs/agent_qwen2_5_3b_4090_s0p02.yaml
configs/agent_qwen2_5_3b_4090_s0p03.yaml
configs/agent_qwen2_5_3b_4090_s0p05.yaml
```

### Step 4：跑严格评测

必须同时测：

```text
dense
identity_hook
static low sparsity
stage low sparsity
failure redensification
```

并使用 exact validator，不使用 substring contains。

### Step 5：决定论文路径是否成立

成立条件：

```text
1. low-sparsity stage mask 不明显破坏 dense baseline
2. stage mask 在相同 active FFN ratio 下优于 static mask
3. failure re-densification 能提高 recovery rate 或降低 cost per success
4. 结果在 3B 上成立，再考虑 7B 确认
```

不成立时的转向：

```text
1. 引入 layer-wise / loss-delta / FLAP-style scoring
2. 转向 SliceGPT/FLAP/Probe Pruning 复现与小模型敏感性分析
3. 把本项目定位为 Agent compression evaluation protocol，而不是新剪枝方法
```

## 5. 产物路径清单

### 仓库内应维护

```text
docs/agent_ffn_progress_paths.md          # 本文档
docs/experiment_roadmap.md                # 总路线
docs/rtx4090_feasibility_plan.md          # 2x4090 执行计划
docs/remote_workflow.md                   # 远端运行 checklist
docs/slicegpt_repro_commands.md           # SliceGPT 复现命令
docs/slicegpt_2x4090_results.md           # 待新增：SliceGPT 远端结果
docs/agent_mask_eval_results.md           # Agent mask 严格评测结果与下一步诊断
docs/agent_handoff_next_steps.md          # 新对话交接入口
```

### 远端应生成

```text
data/agent/controlled_tasks.jsonl
outputs/agent_qwen2_5_1_5b_4090/mask_bank/mask_bank.json
outputs/agent_qwen2_5_3b_4090/mask_bank/mask_bank.json
outputs/agent_qwen2_5_3b_4090_s0p05/mask_bank/mask_bank.json
outputs/agent_qwen2_5_3b_4090_s0p05/eval/predictions.jsonl
outputs/agent_qwen2_5_3b_4090_s0p05/eval/metrics.json
outputs/agent_qwen2_5_3b_4090_low/eval/safe_layer_observe_compare_lookupfixed_100.jsonl
outputs/agent_qwen2_5_3b_4090_low/eval/safe_layer_observe_compare_lookupfixed_100_metrics.json
outputs/agent_qwen2_5_3b_4090_low/eval/global_layerwise_observe_compare_100.jsonl
outputs/agent_qwen2_5_3b_4090_low/eval/global_layerwise_observe_compare_100_metrics.json
outputs/agent_qwen2_5_3b_4090_low/eval/global_layerwise_observe_compare_100_plans.json
logs/*.log
```

## 6. 一句话结论

目前项目已经完成从旧 SafePrune-DPO 到 Agent FFN 动态剪枝的主线切换，并冻结了 controlled mask validation v1 与 Real Tool-Execution Agent Prune v1。Real Tool v1 的主结论是：failure recovery 是局部 reflect-step 计算瓶颈，failure / reflect 轨迹信号可以用更少 fallback step 恢复鲁棒性。P2 substrate v2 已把底层 FFN 剪枝从 1% 推到 10% 正式 1000：`substrate_flap_0p10_stage_reflect_dense` 达到 990/1000、failure 195/200，cost_per_success 从 dense 的 2.2066 降到 2.0777；但它不是无损，且 20% 会破坏 tool schema。下一步是做 schema-aware nested budget plans 与 stage-dependent substrate routing，而不是继续跑统一 20%。
