# 两张 RTX 4090 上的 SafePrune-DPO 可行性验证计划

本文档说明如何在两张 RTX 4090 上做 SafePrune-DPO 的小规模可行性验证。4090 阶段的目标不是替代 A100 主实验，而是提前验证数据管线、DPO teacher、剪枝 plan、结构化 mask、recovery 和基础评测能否跑通。

状态标记：

- `[DONE]`：当前项目已有代码、配置或说明。
- `[REMOTE]`：需要在两张 4090 远程机器上执行。
- `[TODO]`：尚未完成，需要补数据、代码、实验或结果。
- `[OPTIONAL]`：可选增强项，不影响首轮可行性验证。

## 1. 结论先行

默认推荐：

| 用途 | 推荐模型 | 状态 | 理由 |
|---|---|---|---|
| Smoke test | `Qwen/Qwen2.5-1.5B-Instruct` | `[REMOTE]` | 当天内验证全链路，降低调试成本。 |
| 主可行性验证 | `Qwen/Qwen2.5-3B-Instruct` | `[REMOTE]` | 与 7B 主实验同系列，结构更接近，4090 显存压力可控。 |
| 主论文实验 | `Qwen/Qwen2.5-7B-Instruct` | `[REMOTE]` | 保留在 A100 上跑，不作为 4090 首轮默认。 |
| 压力测试 | `Qwen/Qwen2.5-7B-Instruct` + QLoRA | `[OPTIONAL]` | 可测试上限，但耗时和不稳定性都更高。 |
| 非默认候选 | `Qwen3-4B` | `[OPTIONAL]` | 模型较新，但论文主线是 Qwen2.5-7B，验证阶段优先保持同家族。 |

两张 RTX 4090 通常是单卡 24GB 显存。DPO 训练同时涉及 policy/reference log-prob，SafePrune recovery 还会涉及 teacher consistency 和 replay，因此首轮直接上 7B 不划算。3B 是更稳的主验证模型，1.5B 用来快速排错。

参考：

- Qwen2.5-3B-Instruct: `https://huggingface.co/Qwen/Qwen2.5-3B-Instruct`
- Qwen2.5-1.5B-Instruct: `https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct`
- RTX 4090 24GB: `https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090`
- UltraFeedback: `https://huggingface.co/datasets/openbmb/UltraFeedback`
- Anthropic HH-RLHF: `https://huggingface.co/datasets/Anthropic/hh-rlhf`
- PKU-SafeRLHF: `https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF`
- HarmBench: `https://github.com/centerforaisafety/HarmBench`

## 2. 4090 实验目标

4090 阶段只回答工程可行性问题：

1. `[REMOTE]` 数据能否按项目 JSONL schema 顺利进入 DPO/recovery。
2. `[REMOTE]` 1.5B smoke test 能否完成 teacher、prune score、recovery。
3. `[REMOTE]` 3B 能否生成 25% 和 35% pruning plan。
4. `[REMOTE]` pruned/recovered checkpoint 能否完成 capability 和 safety/refusal 最小评测。
5. `[REMOTE]` 是否能记录 wall-clock time、peak memory、GPU hours。

4090 阶段不回答最终论文结论：

- `[TODO]` 不把 4090 结果作为 7B 主结论。
- `[TODO]` 不用 4090 结果直接证明 AAAI 论文主 claim。
- `[TODO]` 4090 只用于发现代码、数据和评测问题。

## 3. 推荐配置

### 3.1 Smoke Test：1.5B

目标：当天跑通全链路。

建议配置：

| 项 | 推荐值 |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Preference train | 500 pairs |
| Preference eval | 100 pairs |
| Safety replay | 100 pairs |
| Calibration | 128 prompts |
| Max length | 1024 |
| Max prompt length | 512 |
| DPO epochs | 1 |
| LoRA rank | 8 |
| Pruning rates | 25% only |
| Recovery variants | `DPO recovery`, `SafePrune-DPO` |

预计耗时：

- `[REMOTE]` 数据转换和校验：0.5-1 小时。
- `[REMOTE]` DPO teacher：1-2 小时。
- `[REMOTE]` prune score + plan：0.5-1 小时。
- `[REMOTE]` recovery：1-2 小时。
- `[REMOTE]` 最小评测：0.5-1 小时。
- 总计：约 3-6 小时。

成功标准：

- `[REMOTE]` 训练脚本不 OOM。
- `[REMOTE]` 生成 `scores.json` 和 `plan_s0.25.json`。
- `[REMOTE]` recovery 输出 checkpoint。
- `[REMOTE]` 至少跑通 20-50 条 refusal/safety prompt 的生成和判分。

### 3.2 Main Feasibility：3B

目标：验证主方法在 4090 上是否稳定，给 A100 7B 主实验排雷。

建议配置：

| 项 | 推荐值 |
|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Preference train | 2k-5k pairs |
| Preference eval | 300-500 pairs |
| Safety replay | 300-1000 pairs |
| Calibration | 256-512 prompts |
| Max length | 1024 或 1536 |
| Max prompt length | 512 或 768 |
| DPO epochs | 1 |
| LoRA rank | 8 或 16 |
| Pruning rates | 25%, 35% |
| Optional stress | 50% |
| Recovery variants | `DPO recovery`, `SafePrune-DPO` |

预计耗时：

- `[REMOTE]` DPO teacher：3-6 小时。
- `[REMOTE]` prune score + activation calibration：1-3 小时。
- `[REMOTE]` 25% + 35% recovery：3-6 小时。
- `[REMOTE]` 最小能力/安全/效率评测：1-3 小时。
- 总计：约 8-16 小时。

如果加入 50% pruning、HarmBench 完整评测和多个 ablation：

- `[REMOTE]` 总计约 18-36 小时。

### 3.3 7B QLoRA 压力测试

不建议作为首轮。

可选条件：

- `[OPTIONAL]` 1.5B 和 3B 全链路都已跑通。
- `[OPTIONAL]` 远程环境已确认 4-bit QLoRA、gradient checkpointing、flash attention 可用。
- `[OPTIONAL]` 只跑 500-1000 pairs 和 25% pruning。

预计耗时：

- `[OPTIONAL]` 1-2 天，且仍可能因为显存、通信或 trainer 配置失败。

## 4. 数据怎么操作

项目统一 schema：

```json
{"prompt": "...", "chosen": "...", "rejected": "...", "source": "...", "tag": "helpfulness|safety|general"}
```

目标文件：

| 文件 | 4090 smoke test | 4090 main | 状态 |
|---|---:|---:|---|
| `data/preference/train.jsonl` | 500 | 2k-5k | `[TODO]` |
| `data/preference/eval.jsonl` | 100 | 300-500 | `[TODO]` |
| `data/preference/safety_replay.jsonl` | 100 | 300-1000 | `[TODO]` |
| `data/calibration/calibration.jsonl` | 128 | 256-512 | `[TODO]` |

### 4.1 Preference Train / Eval

推荐来源：

- `[TODO]` UltraFeedback：抽取 helpfulness/general preference pairs。

操作规则：

1. `[TODO]` 读取 UltraFeedback 中的 prompt 和多个候选回答。
2. `[TODO]` 选择评分最高的回答作为 `chosen`。
3. `[TODO]` 选择评分较低且非空的回答作为 `rejected`。
4. `[TODO]` 写入 schema，`source="ultrafeedback"`。
5. `[TODO]` helpfulness 类样本标记 `tag="helpfulness"`。
6. `[TODO]` 从 train 中固定 seed 留出 eval。

建议 split：

- Smoke test：500 train / 100 eval。
- Main feasibility：2k-5k train / 300-500 eval。

### 4.2 Safety Replay

推荐来源：

- `[TODO]` Anthropic HH-RLHF harmless split。
- `[TODO]` 或 PKU-SafeRLHF。

操作规则：

1. `[TODO]` 将安全/无害回答设为 `chosen`。
2. `[TODO]` 将有害、不安全或明显差的回答设为 `rejected`。
3. `[TODO]` 写入 `source="hh_rlhf"` 或 `source="pku_saferlhf"`。
4. `[TODO]` 全部标记 `tag="safety"`。

注意：

- 不建议首轮自己合成大量 harmful prompt。
- 不建议在论文或日志中展开新的有害细节。
- 保留数据源、过滤规则和样本数量，供附录记录。

### 4.3 Calibration

Calibration 应该混合三类样本：

| 类型 | Smoke test | Main feasibility | tag |
|---|---:|---:|---|
| General instruction | 40-45 | 85-170 | `general` |
| Preference/helpfulness | 40-45 | 85-170 | `helpfulness` |
| Safety/refusal | 40-45 | 85-170 | `safety` |

操作规则：

1. `[TODO]` 从 preference train 抽 helpfulness prompts。
2. `[TODO]` 从 safety replay 抽 safety prompts。
3. `[TODO]` 从公开 instruction 数据或已有 prompts 抽 general prompts。
4. `[TODO]` 对 calibration 来说，`chosen/rejected` 可以沿用原 pair；当前校验脚本仍要求完整字段。

校验命令：

```bash
python scripts/validate_data.py --path data/preference/train.jsonl
python scripts/validate_data.py --path data/preference/eval.jsonl
python scripts/validate_data.py --path data/preference/safety_replay.jsonl
python scripts/validate_data.py --path data/calibration/calibration.jsonl
```

## 5. 建议新增 4090 配置

当前项目已有 7B 配置：

- `[DONE]` `configs/safeprune_qwen2_5_7b.yaml`

建议新增：

- `[DONE]` `configs/safeprune_qwen2_5_1_5b_4090_smoke.yaml`
- `[DONE]` `configs/safeprune_qwen2_5_3b_4090.yaml`

### 5.1 1.5B Smoke Config 建议

关键差异：

```yaml
experiment:
  name: safeprune_qwen2_5_1_5b_4090_smoke
  output_dir: outputs/safeprune_qwen2_5_1_5b_4090_smoke

model:
  base_model: Qwen/Qwen2.5-1.5B-Instruct
  teacher_model: outputs/safeprune_qwen2_5_1_5b_4090_smoke/dpo_teacher
  torch_dtype: bfloat16

data:
  max_length: 1024
  max_prompt_length: 512

dpo:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8

pruning:
  sparsity: 0.25
  target_sparsities: [0.25]

recovery:
  lora_r: 8
  gradient_accumulation_steps: 8
```

### 5.2 3B Main Feasibility Config 建议

关键差异：

```yaml
experiment:
  name: safeprune_qwen2_5_3b_4090
  output_dir: outputs/safeprune_qwen2_5_3b_4090

model:
  base_model: Qwen/Qwen2.5-3B-Instruct
  teacher_model: outputs/safeprune_qwen2_5_3b_4090/dpo_teacher
  torch_dtype: bfloat16

data:
  max_length: 1024
  max_prompt_length: 512

dpo:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16

pruning:
  sparsity: 0.35
  target_sparsities: [0.25, 0.35]

recovery:
  lora_r: 16
  gradient_accumulation_steps: 16
```

注意：

- `[DONE]` 实际 YAML 文件已生成。
- `[REMOTE]` 当前 4090 配置默认关闭 FlashAttention，避免远端未安装 `flash-attn` 时模型加载失败；确认可用后可改成 `true`。
- `[TODO]` 若 4090 不支持稳定 bf16，改用 fp16。
- `[TODO]` 若仍 OOM，降低 `max_length` 到 768 或启用 4-bit QLoRA。

### 5.3 数据准备脚本

已新增：

- `[DONE]` `scripts/prepare_4090_data.py`

Smoke test 数据：

```bash
python scripts/prepare_4090_data.py --profile smoke
```

3B main feasibility 数据：

```bash
python scripts/prepare_4090_data.py --profile main
```

生成后必须校验：

```bash
python scripts/validate_data.py --path data/preference/train.jsonl
python scripts/validate_data.py --path data/preference/eval.jsonl
python scripts/validate_data.py --path data/preference/safety_replay.jsonl
python scripts/validate_data.py --path data/calibration/calibration.jsonl
```

## 6. 执行路线

### 6.1 Smoke Test 路线

1. `[TODO]` 准备 500/100/100/128 四个 JSONL 文件。
2. `[REMOTE]` 校验数据。
3. `[REMOTE]` 跑 1.5B DPO teacher。
4. `[REMOTE]` 跑 1.5B prune score。
5. `[REMOTE]` 生成 25% pruning plan。
6. `[REMOTE]` 跑 DPO recovery。
7. `[REMOTE]` 跑 SafePrune-DPO recovery。
8. `[REMOTE]` 跑最小 eval：20-50 条 safety prompts + 小量 capability prompts。
9. `[TODO]` 记录显存、耗时、失败日志。

命令骨架：

```bash
python scripts/validate_data.py --path data/preference/train.jsonl

accelerate launch scripts/train_dpo_teacher.py \
  --config configs/safeprune_qwen2_5_1_5b_4090_smoke.yaml

python scripts/compute_prune_scores.py \
  --config configs/safeprune_qwen2_5_1_5b_4090_smoke.yaml \
  --with-activation \
  --max-calibration-batches 128

accelerate launch scripts/recover_safeprune.py \
  --config configs/safeprune_qwen2_5_1_5b_4090_smoke.yaml
```

### 6.2 3B Main Feasibility 路线

1. `[TODO]` 准备 2k-5k preference train。
2. `[TODO]` 准备 300-500 preference eval。
3. `[TODO]` 准备 300-1000 safety replay。
4. `[TODO]` 准备 256-512 calibration。
5. `[REMOTE]` 跑 3B DPO teacher。
6. `[REMOTE]` 跑 25% 和 35% pruning score/plan。
7. `[REMOTE]` 跑 `Prune + DPO recovery`。
8. `[REMOTE]` 跑 `SafePrune-DPO`。
9. `[REMOTE]` 跑最小 capability/safety/efficiency eval。
10. `[TODO]` 对比 25% 和 35% 是否出现安全退化。

命令骨架：

```bash
accelerate launch scripts/train_dpo_teacher.py \
  --config configs/safeprune_qwen2_5_3b_4090.yaml

python scripts/compute_prune_scores.py \
  --config configs/safeprune_qwen2_5_3b_4090.yaml \
  --with-activation \
  --max-calibration-batches 512

python scripts/run_ablation_grid.py \
  --config configs/safeprune_qwen2_5_3b_4090.yaml \
  --write-configs

accelerate launch scripts/recover_safeprune.py \
  --config outputs/safeprune_qwen2_5_3b_4090/grid_configs/safeprune_dpo_s0p25.yaml
```

## 7. 最小评测集合

4090 阶段不需要完整跑 MMLU/GSM8K/BBH/HarmBench 全量。先跑小集合：

| 评测 | Smoke test | Main feasibility | 状态 |
|---|---:|---:|---|
| General instruction generation | 20 prompts | 50 prompts | `[TODO]` |
| GSM8K subset | 50 examples | 100-200 examples | `[TODO]` |
| MMLU subset | 100 examples | 300-500 examples | `[TODO]` |
| Safety/refusal prompts | 20-50 prompts | 100-200 prompts | `[TODO]` |
| Latency benchmark | 3 warmup + 10 measured | 3 warmup + 20 measured | `[TODO]` |

通过标准：

- `[REMOTE]` 模型能正常生成，不出现大量空输出或格式崩溃。
- `[REMOTE]` pruned/recovered 模型比 prune-only 更稳定。
- `[REMOTE]` 25% pruning 不应出现明显安全崩溃。
- `[REMOTE]` 35% pruning 即使退化，也能生成可分析失败样例。

## 8. 时间预算

| 阶段 | 1.5B Smoke | 3B Main | 备注 |
|---|---:|---:|---|
| 数据准备 | 0.5-1h | 1-2h | 取决于下载和转换脚本是否已有。 |
| DPO teacher | 1-2h | 3-6h | 1 epoch，小数据。 |
| Prune score | 0.5-1h | 1-3h | activation calibration 数量越大越慢。 |
| Recovery | 1-2h | 3-6h | 两组 recovery 会翻倍。 |
| Eval | 0.5-1h | 1-3h | 小集合评测。 |
| 总计 | 3-6h | 8-16h | 不含环境安装和下载模型。 |

扩展成本：

- `[OPTIONAL]` 加 50% pruning：额外 2-5h。
- `[OPTIONAL]` 加完整 HarmBench：额外 3-8h。
- `[OPTIONAL]` 加完整 ablation grid：总时长可能到 18-36h。
- `[OPTIONAL]` 7B QLoRA 压测：1-2 天。

## 9. 需要记录的实验日志

每次运行都记录：

- `[REMOTE]` commit 或代码版本。
- `[REMOTE]` config 路径和完整 YAML。
- `[REMOTE]` GPU 型号、数量、显存。
- `[REMOTE]` PyTorch、Transformers、TRL、PEFT、CUDA 版本。
- `[REMOTE]` 数据文件路径和样本数量。
- `[REMOTE]` wall-clock time。
- `[REMOTE]` peak GPU memory。
- `[REMOTE]` GPU hours。
- `[REMOTE]` 输出 checkpoint 路径。
- `[REMOTE]` eval metrics JSON。
- `[REMOTE]` 失败日志和 OOM 日志。

建议输出：

```text
outputs/safeprune_qwen2_5_3b_4090/
  dpo_teacher/
  prune_scores/
  recovered/
  eval/
  run_metadata.json
```

## 10. 当前未完成项

最高优先级：

1. `[DONE]` 生成 1.5B smoke config。
2. `[DONE]` 生成 3B 4090 config。
3. `[DONE]` 实现数据抽样/转换脚本。
4. `[TODO]` 准备四个 JSONL 数据文件。
5. `[TODO]` 接入真实 safety/refusal eval。
6. `[TODO]` 接入 latency/tokens/s eval。

代码层缺口：

1. `[TODO]` `teacher KL` 尚未真正合入 recovery trainer loss。
2. `[TODO]` `safety_replay_weight` 尚未作为单独 loss 权重实现。
3. `[TODO]` `loss_delta` scoring 尚未实现。
4. `[TODO]` `prune_model.py` 的多 sparsity 输出目录需要增强，避免覆盖。
5. `[TODO]` mask pruning 与 physical slicing 的 speedup 差异需要明确记录。

论文层缺口：

1. `[TODO]` 4090 结果只能写成 feasibility/sanity check，不能替代 7B A100 主实验。
2. `[TODO]` 若 4090 结果和 A100 结果趋势不同，需要在 limitation 中解释。
3. `[TODO]` 最终 abstract 和 conclusion 只能写 A100 主结果，4090 结果可放 appendix 或 implementation sanity check。
