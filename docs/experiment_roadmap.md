# SafePrune-DPO 实验路线图

本文档是 `D:\Prune` 项目的实验执行路线。状态标记如下：

- `[DONE]`：当前项目中已有代码、配置或文档。
- `[REMOTE]`：需要在远程 GPU 环境执行。
- `[TODO]`：当前尚未完成，需要补代码、数据、实验或论文结果。
- `[OPTIONAL]`：不是主线必需，但建议作为增强实验。

## 0. 当前项目状态

| 模块 | 状态 | 位置 | 说明 |
|---|---|---|---|
| 项目骨架 | `[DONE]` | `pyproject.toml`, `requirements.txt`, `README.md` | 已定义 Python 包、依赖和基础说明。 |
| 主配置 | `[DONE]` | `configs/safeprune_qwen2_5_7b.yaml` | 已配置 Qwen2.5-7B-Instruct、DPO、剪枝、恢复和评测字段。 |
| 消融配置 | `[DONE]` | `configs/ablations/*.yaml` | 已有 no consistency、no safety replay、prune-then-DPO 配置。 |
| 数据校验 | `[DONE]` | `scripts/validate_data.py`, `src/safeprune/data.py` | 已支持 preference JSONL schema 校验。 |
| DPO teacher 训练入口 | `[DONE]` | `scripts/train_dpo_teacher.py` | 远程安装 `trl/peft/accelerate` 后可运行。 |
| 剪枝打分与 plan | `[DONE]` | `scripts/compute_prune_scores.py`, `src/safeprune/scoring.py` | 已实现 magnitude + activation；loss-delta 字段预留。 |
| 结构化 mask | `[DONE]` | `src/safeprune/masks.py` | 已实现 attention head 和 MLP channel forward mask。 |
| 恢复训练入口 | `[DONE]` | `scripts/recover_safeprune.py`, `src/safeprune/training.py` | 已接入 LoRA + DPO recovery；完整 SafePrune loss 仍需加强。 |
| 评测入口 | `[TODO]` | `scripts/evaluate_model.py` | 当前只写 manifest，需要接入 lm-eval、HarmBench、AdvBench、latency。 |
| 真实实验结果 | `[TODO]` | `outputs/`, 论文表格 | 尚未远程运行，所有论文指标待补。 |

## 1. 实验目标

主问题：

> DPO 对齐后的 Qwen2.5-7B-Instruct 在 25%、35%、50% 结构化剪枝后，是否还能保持偏好边界和安全拒答边界？

需要回答三个子问题：

1. `[TODO]` 剪枝率到多少开始出现 alignment collapse？
2. `[TODO]` teacher consistency 和 safety replay 是否能修复剪枝后的安全退化？
3. `[TODO]` 在双 A100 预算下，能力、安全、效率的 Pareto trade-off 是什么？

## 2. 远程环境准备

目标环境：

- `[REMOTE]` 两张 NVIDIA A100，优先按单卡 60GB 执行；论文中报告时保留 40GB 可复现下限。
- `[REMOTE]` Python 3.10+。
- `[REMOTE]` CUDA + PyTorch + Transformers + TRL + PEFT + Accelerate。

推荐命令：

```bash
cd /path/to/Prune
conda activate llm
pip install -e ".[dev,eval]"
```

最小环境检查：

```bash
python -c "import torch, transformers, datasets, peft, trl, accelerate; print(torch.cuda.is_available())"
python -m unittest discover -s tests
```

状态：

- `[DONE]` 本地轻量单元测试已可跑。
- `[REMOTE]` 远程 ML 依赖和 GPU 可用性需要实际确认。

## 3. 数据准备

统一 JSONL schema：

```json
{"prompt": "...", "chosen": "...", "rejected": "...", "source": "...", "tag": "helpfulness|safety|general"}
```

需要准备的文件：

| 文件 | 状态 | 用途 |
|---|---|---|
| `data/preference/train.jsonl` | `[TODO]` | DPO teacher 和 recovery 主训练集。 |
| `data/preference/eval.jsonl` | `[TODO]` | DPO/recovery validation。 |
| `data/preference/safety_replay.jsonl` | `[TODO]` | SafePrune recovery 的安全 replay。 |
| `data/calibration/calibration.jsonl` | `[TODO]` | 剪枝打分 calibration。 |

建议数据构成：

- `[TODO]` Preference train：UltraFeedback 子集，建议 10k-50k pairs。
- `[TODO]` Preference eval：从 train 同源数据留出 500-2k pairs。
- `[TODO]` Safety replay：Harmful prompt + safe refusal pairs，建议 500-2k pairs。
- `[TODO]` Calibration：general/helpfulness/safety 混合，512-2048 prompts。

校验命令：

```bash
python scripts/validate_data.py --path data/preference/train.jsonl
python scripts/validate_data.py --path data/preference/eval.jsonl
python scripts/validate_data.py --path data/preference/safety_replay.jsonl
python scripts/validate_data.py --path data/calibration/calibration.jsonl
```

未完成项：

- `[TODO]` 数据下载/筛选脚本尚未实现。
- `[TODO]` safety replay 的 chosen/rejected 构造规则尚未固定。
- `[TODO]` 需要记录每个数据源 license 和过滤策略，供论文附录使用。

## 4. Stage A：DPO Teacher

目标：

- 得到未剪枝的 DPO-aligned teacher。
- 作为后续 pruning/recovery 的 teacher 和 DPO-only baseline。

命令：

```bash
accelerate launch scripts/train_dpo_teacher.py \
  --config configs/safeprune_qwen2_5_7b.yaml
```

输出：

```text
outputs/safeprune_qwen2_5_7b/dpo_teacher/
```

需要记录：

- `[REMOTE]` 训练 wall-clock time。
- `[REMOTE]` 总 GPU hours。
- `[REMOTE]` peak GPU memory。
- `[REMOTE]` final train/eval loss。
- `[REMOTE]` DPO reward margin。

未完成项：

- `[TODO]` 需要确认 `trl.DPOTrainer` 当前版本参数兼容性。
- `[TODO]` 需要补 teacher 的 baseline eval。
- `[TODO]` 需要保存 resolved config 和训练 metadata。

## 5. Stage B：Teacher Baseline Evaluation

目标：

- 建立 DPO-only baseline。
- 作为所有 pruned model 的能力、安全、效率比较基准。

能力评测：

```bash
lm_eval --model hf \
  --model_args pretrained=outputs/safeprune_qwen2_5_7b/dpo_teacher \
  --tasks mmlu,gsm8k,bbh \
  --batch_size auto \
  --output_path outputs/safeprune_qwen2_5_7b/eval/dpo_teacher_capability.json
```

安全评测：

```bash
python scripts/evaluate_model.py \
  --config configs/safeprune_qwen2_5_7b.yaml \
  --model outputs/safeprune_qwen2_5_7b/dpo_teacher
```

未完成项：

- `[TODO]` `scripts/evaluate_model.py` 尚未真正接入 HarmBench。
- `[TODO]` AdvBench refusal evaluator 尚未实现。
- `[TODO]` benign over-refusal set 尚未准备。
- `[TODO]` latency/tokens/s 脚本尚未实现。

## 6. Stage C：剪枝打分与 Pruning Plan

目标：

- 对每层 attention heads 和 MLP intermediate channels 打分。
- 生成 25%、35%、50% 三档剪枝 plan。

命令：

```bash
python scripts/compute_prune_scores.py \
  --config configs/safeprune_qwen2_5_7b.yaml \
  --with-activation \
  --max-calibration-batches 1024
```

输出：

```text
outputs/safeprune_qwen2_5_7b/prune_scores/scores.json
outputs/safeprune_qwen2_5_7b/prune_scores/plan_s0.25.json
outputs/safeprune_qwen2_5_7b/prune_scores/plan_s0.35.json
outputs/safeprune_qwen2_5_7b/prune_scores/plan_s0.50.json
```

当前打分：

- `[DONE]` Weight magnitude。
- `[DONE]` Activation sensitivity。
- `[TODO]` Calibration loss delta。

loss-delta 建议实现：

1. `[TODO]` 在 calibration batch 上计算 teacher 原始 loss。
2. `[TODO]` 对每个候选 head/channel 临时 mask。
3. `[TODO]` 重新 forward，记录 loss increase。
4. `[TODO]` normalize 后写入 `loss_delta_attention` / `loss_delta_mlp`。

剪枝计划检查：

- `[REMOTE]` 每层保留 head 数。
- `[REMOTE]` 每层保留 MLP channel 数。
- `[REMOTE]` 不允许某层完全塌缩。
- `[REMOTE]` 统计不同 tag 的 calibration 是否改变 prune plan。

## 7. Stage D：Prune-Only Baseline

目标：

- 检查不恢复时剪枝本身造成多少能力/安全退化。

命令：

```bash
python scripts/prune_model.py \
  --config configs/safeprune_qwen2_5_7b.yaml \
  --plan outputs/safeprune_qwen2_5_7b/prune_scores/plan_s0.25.json
```

对 0.25、0.35、0.50 各跑一次。

未完成项：

- `[TODO]` 当前 `prune_model.py` 保存的是 mask-compatible checkpoint；需要确认评测时 mask 是否被重新 attach。
- `[TODO]` 物理裁剪 tensor 的实现尚未完成，因此真实 speedup 需要额外实现或报告 mask active-structure reduction。
- `[TODO]` 需要补 `--sparsity` 或 per-plan 输出目录，避免三档结果互相覆盖。

## 8. Stage E：Recovery Baselines

需要跑四类 recovery：

| 名称 | DPO | Teacher KL | Safety Replay | 状态 |
|---|---:|---:|---:|---|
| Prune + CE recovery | no | no | no | `[TODO]` |
| Prune + DPO recovery | yes | no | no | `[REMOTE]` |
| No teacher KL | yes | no | yes | `[REMOTE]` |
| No safety replay | yes | yes | no | `[REMOTE]` |
| SafePrune-DPO | yes | yes | yes | `[REMOTE]` |

生成 ablation configs：

```bash
python scripts/run_ablation_grid.py \
  --config configs/safeprune_qwen2_5_7b.yaml \
  --write-configs
```

运行示例：

```bash
accelerate launch scripts/recover_safeprune.py \
  --config outputs/safeprune_qwen2_5_7b/grid_configs/safeprune_dpo_s0p25.yaml
```

未完成项：

- `[TODO]` 当前 recovery 主要依赖 TRL DPOTrainer；`teacher KL` 尚未实际合入 trainer loss。
- `[TODO]` `safety_replay_weight` 当前通过 replay 数据比例体现，尚未实现单独加权 loss。
- `[TODO]` CE recovery baseline 尚未实现。
- `[TODO]` 需要让 grid config 同时切换不同 `plan_s0.xx.json`，当前需要核对是否总是读 `config.pruning.sparsity` 对应 plan。

## 9. Stage F：主评测

每个 checkpoint 都需要跑三类评测。

### 9.1 Capability

任务：

- MMLU
- GSM8K
- BBH
- `[OPTIONAL]` 中文能力集，如 CMMLU/C-Eval。

输出表：

| Method | Prune | MMLU | GSM8K | BBH | Chinese Avg |
|---|---:|---:|---:|---:|---:|
| DPO only | 0% | `[TODO]` | `[TODO]` | `[TODO]` | `[OPTIONAL]` |
| Prune only | 25% | `[TODO]` | `[TODO]` | `[TODO]` | `[OPTIONAL]` |
| SafePrune-DPO | 25% | `[TODO]` | `[TODO]` | `[TODO]` | `[OPTIONAL]` |
| SafePrune-DPO | 35% | `[TODO]` | `[TODO]` | `[TODO]` | `[OPTIONAL]` |
| SafePrune-DPO | 50% | `[TODO]` | `[TODO]` | `[TODO]` | `[OPTIONAL]` |

### 9.2 Safety / Alignment

任务：

- HarmBench ASR，越低越好。
- AdvBench harmful compliance rate，越低越好。
- Refusal accuracy，越高越好。
- Benign over-refusal rate，越低越好。
- Preference win-rate vs teacher，越高越好。

输出表：

| Method | Prune | HarmBench ASR ↓ | AdvBench ASR ↓ | Refusal Acc ↑ | Over-refusal ↓ | Win-rate vs Teacher ↑ |
|---|---:|---:|---:|---:|---:|---:|
| DPO only | 0% | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Prune + DPO recovery | 25% | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| SafePrune-DPO | 25% | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| SafePrune-DPO | 35% | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| SafePrune-DPO | 50% | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

### 9.3 Efficiency

指标：

- Active attention heads。
- Active MLP channels。
- Parameter count after physical slicing。
- Peak GPU memory。
- Tokens/s。
- End-to-end latency。
- Training GPU hours。

输出表：

| Method | Prune | Active Heads | Active MLP Ch. | Peak Mem | Tokens/s | Latency | GPUh |
|---|---:|---:|---:|---:|---:|---:|---:|
| DPO only | 0% | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| SafePrune-DPO | 25% | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| SafePrune-DPO | 35% | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| SafePrune-DPO | 50% | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

未完成项：

- `[TODO]` latency benchmark 脚本。
- `[TODO]` physical slicing 后的真实 speedup。
- `[TODO]` 统一 metrics JSON 聚合脚本。

## 10. Stage G：消融实验

必须完成：

1. `[REMOTE]` Pruning rate：25%、35%、50%。
2. `[REMOTE]` Remove teacher KL。
3. `[REMOTE]` Remove safety replay。
4. `[REMOTE]` DPO recovery vs SafePrune-DPO。
5. `[REMOTE]` General-only calibration vs general+safety calibration。

建议完成：

1. `[OPTIONAL]` LoRA rank：8、16、32。
2. `[OPTIONAL]` Replay ratio：0%、10%、20%、40%。
3. `[OPTIONAL]` Qwen2.5-1.5B/3B sanity check。
4. `[OPTIONAL]` Random pruning baseline。

未完成项：

- `[TODO]` random pruning plan 生成器。
- `[TODO]` calibration source ablation 配置。
- `[TODO]` LoRA rank/replay ratio grid 配置。

## 11. Stage H：错误分析

需要人工或半自动分析以下失败类型：

| 类型 | 判断标准 | 状态 |
|---|---|---|
| Harmful compliance | 对 harmful prompt 给出可执行有害内容 | `[TODO]` |
| Benign over-refusal | 对普通请求错误拒答 | `[TODO]` |
| Reasoning degradation | 数学/推理能力明显退化 | `[TODO]` |
| Preference drift | 相比 teacher 更偏向 rejected answer | `[TODO]` |
| Format drift | 输出格式不稳定或 instruction following 失败 | `[TODO]` |

输出：

- `[TODO]` 每类失败 5-10 个代表样例。
- `[TODO]` 错误类型占比表。
- `[TODO]` 25%、35%、50% 的 collapse 边界分析。

## 12. Stage I：论文补表与结论

目标论文路径：

- `D:\SafePrune-dpo\main.tex`

需要补入：

1. `[TODO]` 主结果表。
2. `[TODO]` safety/alignment 表。
3. `[TODO]` efficiency 表。
4. `[TODO]` ablation 表。
5. `[TODO]` Pareto frontier 图。
6. `[TODO]` failure case 表。
7. `[TODO]` 最终 abstract 数字结果。
8. `[TODO]` conclusion 中的实证结论。

当前论文状态：

- `[DONE]` AAAI 2026 LaTeX 格式初稿已生成。
- `[DONE]` 正文结构已完整。
- `[TODO]` 实验结果全部待补。

## 13. 最小可投稿实验集

如果时间紧，至少完成以下实验：

1. `[REMOTE]` DPO teacher baseline。
2. `[REMOTE]` SafePrune-DPO 25%、35%、50%。
3. `[REMOTE]` Prune + DPO recovery 25%、35%。
4. `[REMOTE]` No safety replay 35%。
5. `[REMOTE]` MMLU/GSM8K + HarmBench/AdvBench + tokens/s。
6. `[TODO]` 失败案例分析，尤其是 50% collapse。

这组实验可以支撑核心论点：

> Moderate structured pruning can preserve most capability, but alignment safety degrades earlier unless recovery includes teacher consistency and safety replay.

该句目前仍是待验证假设，不能在论文中写成结论，直到实验结果支持。

## 14. 当前最高优先级 TODO

1. `[TODO]` 准备四个 JSONL 数据文件并通过 `validate_data.py`。
2. `[TODO]` 在远程环境安装 `peft/trl/accelerate/lm-eval/HarmBench`。
3. `[REMOTE]` 跑 DPO teacher。
4. `[TODO]` 实现或接入 HarmBench/AdvBench/refusal/latency 评测脚本。
5. `[REMOTE]` 跑 `compute_prune_scores.py --with-activation`。
6. `[TODO]` 补 loss-delta scoring。
7. `[TODO]` 补真正的 SafePrune weighted loss，而不是只依赖 DPOTrainer + replay mix。
8. `[REMOTE]` 跑 25%、35%、50% 主实验。
9. `[TODO]` 聚合 metrics，填入论文。

