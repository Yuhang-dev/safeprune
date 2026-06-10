# Agent FFN 动态剪枝路径说明

日期：2026-06-11

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

当前实验已证明 10%-30% 不适合作为第一阶段主区间。下一步应改为：

```text
1%, 2%, 3%, 5%
```

建议方法矩阵：

| 方法 | 目的 |
|---|---|
| dense | 上限和 prompt 自洽性 |
| identity_hook | 验证 hook 不改变模型 |
| static_01 / static_02 / static_03 / static_05 | 找 naive mask 可用边界 |
| stage_answer_01 / stage_answer_03 / stage_answer_05 | 看 stage mask 是否优于 static |
| failure_redensification_05_to_01 | 看失败恢复是否改善成本 |

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

### 2.4 P1：补 stage-aware 与 failure re-densification 的真实对照

技术报告的核心不是“有 stage mask bank”，而是：

```text
stage 信息和失败反馈是否改善每个成功任务的成本？
```

因此必须补以下对照：

| 对照 | 要回答的问题 |
|---|---|
| static global vs stage mask | 阶段标签本身是否有价值 |
| stage mask vs failure re-densification | 失败后恢复计算是否有价值 |
| fixed low sparsity vs dynamic fallback | 动态恢复是否只是始终低稀疏率的替代 |
| dense vs mask | mask 是否引入生成崩溃 |

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

### Step 1：本地收尾

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
export PYTHONPATH=$PWD/src
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
docs/agent_mask_eval_results.md           # 待新增：Agent mask 远端结果
```

### 远端应生成

```text
data/agent/controlled_tasks.jsonl
outputs/agent_qwen2_5_1_5b_4090/mask_bank/mask_bank.json
outputs/agent_qwen2_5_3b_4090/mask_bank/mask_bank.json
outputs/agent_qwen2_5_3b_4090_s0p05/mask_bank/mask_bank.json
outputs/agent_qwen2_5_3b_4090_s0p05/eval/predictions.jsonl
outputs/agent_qwen2_5_3b_4090_s0p05/eval/metrics.json
logs/*.log
```

## 6. 一句话结论

目前项目已经完成了从旧 SafePrune-DPO 到 Agent FFN 动态剪枝的代码骨架、文档路线、远端环境、任务数据和 1.5B/3B stage mask bank 生成。但真实方法有效性尚未证明；已有实验反而说明 Qwen2.5-3B 对 naive FFN channel masking 很敏感。下一步必须先完成 SwiGLU activation scoring 修正、严格判分和 1%-5% 低稀疏边界扫描，再判断 stage-aware 与 failure re-densification 是否有论文价值。
