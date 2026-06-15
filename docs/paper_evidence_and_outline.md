# Agent Prune Paper Evidence and Outline

本文用于收束论文主线、区分已冻结证据与待补实验，并防止将 theoretical
active-FFN cost reduction 误写成 wall-clock speedup。

## 1. Paper Thesis

核心命题：

```text
Tool-using Agent trajectories are compute-heterogeneous.

tool-call / retry generation is schema-sensitive and needs a conservative FFN budget;
final-answer generation can tolerate a more aggressive FFN budget;
reflect recovery is failure-sensitive and should temporarily restore dense FFN.
```

本文不是一篇“统一静态模型压缩”论文，也不主张当前实现已经获得端到端真实加速。

## 2. Main Result

当前最强、最适合作为主表的结果是 Qwen2.5-7B、Real Tool protocol v1.2、
1000 tasks：

| Method | Success | Failure | Non-failure | Schema validity | Cost / success |
|---|---:|---:|---:|---:|---:|
| Dense | 1000/1000 | 200/200 | 800/800 | 1.0000 | 2.2390 |
| Identity hook | 1000/1000 | 200/200 | 800/800 | 1.0000 | 2.2390 |
| Nested uniform 10 | 1000/1000 | 200/200 | 800/800 | 1.0000 | 2.0000 |
| Adaptive-B20 | 1000/1000 | 200/200 | 800/800 | 1.0000 | 1.9000 |

Adaptive-B20：

```text
tool_call_or_retry -> nested_0p10
final_answer       -> nested_0p20
reflect_recovery   -> dense
```

可以声明：

> Adaptive-B20 preserves Dense-equivalent task success under a strict Real Tool
> protocol while reducing theoretical active-FFN cost per successful episode by
> about 15.1% on Qwen2.5-7B.

不可以把该结果写成 15.1% latency speedup。

## 3. Evidence Chain

| Evidence | Result | Role in Paper |
|---|---|---|
| Controlled Mask Validation v1 | stage-aware 898/1000 vs observe-only 836/1000 | 早期机制证据，不作为最终主结果。 |
| Real Tool v1 | reflect-localized dense fallback 接近 fixed-window recovery，但 fallback steps 更少 | 证明 failure recovery 是局部计算瓶颈。 |
| Hidden-state centroid baseline | hidden-only 较弱；event hybrid 可追平但 router-inclusive cost 高 | 证明显式 trajectory event 更直接、更低成本。 |
| 3B P2 substrate | 10% substrate 990/1000，cost/success 降约 5.8% | 证明机制可以从 1% 扩展到有意义预算。 |
| 3B P2c | Adaptive-B20 997/1000，与 Dense 相同；cost/success 降约 13.6% | 3B generation-type routing 主结果，但 protocol v1.2 对齐仍待补。 |
| 7B P4 | Adaptive-B20 1000/1000；cost/success 降约 15.1% | 当前论文最强主结果。 |
| 7B static quick sanity | static 20% 的 PPL 和多选能力明显退化 | 说明 20% 只能动态用于 final-answer，不是通用静态压缩。 |
| P5 deployment analysis | compact 等价性通过；MLP-only 有速度信号；full-model decode 无显著收益 | 限制与系统分析，不作为主结果。 |

## 4. Core Ablation Matrix

论文至少需要一张统一 ablation 表：

| Ablation | Comparison | Question Answered |
|---|---|---|
| Hook sanity | Dense vs identity hook | hook 是否污染模型行为？ |
| Static high budget | uniform/static 20% vs Dense | 高比例统一剪枝是否破坏 tool schema 和通用能力？ |
| Conservative sparse baseline | nested uniform 10 vs Dense | 10% schema-safe plan 能否稳定运行？ |
| Generation-type routing | Adaptive-B20 vs nested uniform 10 | final-answer 使用 20% 是否进一步降低成本？ |
| Recovery allocation | no reflect dense / observe-only vs reflect dense | failure recovery 是否需要恢复更多计算？ |
| Failure localization | fixed-window redense vs reflect-localized dense | 显式 reflect 是否减少不必要 fallback steps？ |
| Router baseline | hidden-state centroid vs explicit event | trajectory event 是否比 hidden-state probe 更可靠、低成本？ |
| Static benchmark | nested 10 vs nested 20 | Real Tool 动态收益是否依赖静态不安全的 20% plan？ |
| Deployment analysis | irregular compact vs aligned compact | hardware alignment 是否是 MLP-level speed signal 的必要条件？ |

## 5. P5 Deployment Analysis

### Functional equivalence

| Plan | Active MLP | Mean logit diff | Top-1 match |
|---|---:|---:|---:|
| aligned_0p10 | 0.899614 | 0.0828 | 100/100 |
| aligned_0p20 | 0.799710 | 0.0914 | 100/100 |

### MLP-only latency

| Plan | B1 S1 | B1 S128 | B4 S128 |
|---|---:|---:|---:|
| Dense | 0.217 ms | 0.494 ms | 0.874 ms |
| aligned_0p10 | 0.240 ms | 0.504 ms | 0.770 ms |
| aligned_0p20 | 0.185 ms | 0.220 ms | 0.466 ms |

### Full-model fixed-decode latency

| Batch | Plan | Decode ms/token | Tok/s | Peak GB | Speedup vs Dense |
|---:|---|---:|---:|---:|---:|
| 1 | Dense | 26.226 | 38.130 | 8.266 | 0.0% |
| 1 | aligned_0p10 | 25.497 | 39.220 | 7.993 | +2.8% |
| 1 | aligned_0p20 | 26.680 | 37.481 | 7.857 | -1.7% |
| 4 | Dense | 27.319 | 146.421 | 8.335 | 0.0% |
| 4 | aligned_0p10 | 27.200 | 147.059 | 8.062 | +0.4% |
| 4 | aligned_0p20 | 27.700 | 144.405 | 7.926 | -1.4% |

P5 正式结论：

```text
compact materialization: pass
logits/top-1 equivalence: pass
hardware-aligned MLP-only speed signal: pass
meaningful full-model decode speedup: not demonstrated
dynamic Agent wall-clock speedup: not evaluated
```

正文中最多写：

> Hardware-aligned compact FFN subnets recover MLP-level speedups, but these
> gains do not yet translate into meaningful full-model decode acceleration
> under the current HuggingFace/PyTorch runtime.

## 6. Required Remaining Experiment

如果最终主表同时包含 3B 和 7B，则必须补：

```text
Qwen2.5-3B
Real Tool protocol v1.2
1000 tasks

dense
identity_hook
nested_uniform_10
adaptive_B20
```

原因：现有 3B P2c 与 7B 使用的 protocol version 不完全一致。未完成前，跨模型主表
必须显式标注 protocol 差异。

## 7. Optional Strengthening Experiments

### 7B static benchmark expansion

可将 quick sanity 从 256 samples 扩到 1024 或 full validation：

```text
Dense
Nested 10%
Nested 20%
```

`Nested 15%` 只有在从同一 calibration、同一 scoring 和同一 nested ladder 重新构建，
并记录 plan/calibration hash 时才可加入。不能临时生成一个 15% plan 后与已冻结
10%/20% 结果混用。

优先级：

```text
3B protocol v1.2 alignment > paper tables/ablation > static 1024/full > optional 15%
```

## 8. Paper Outline

### 1. Introduction

- 静态 PPL 和通用 benchmark 不能充分预测 Agent tool-schema failure。
- Agent trajectory 中不同 generation type 的计算敏感度不同。
- 提出 trajectory-event-aware FFN budget routing。

### 2. Agentic Pruning Failures

- tool-call schema breakage；
- premature final；
- failure recovery collapse；
- static 20% 在通用 benchmark 上退化。

### 3. Method

- schema-aware nested FFN substrate；
- generation-type routing；
- reflect-localized dense fallback；
- active FFN cost 和 cost per successful episode。

### 4. Experiments

- Real Tool benchmark 和 deterministic failure schedule；
- Qwen2.5-3B / 7B；
- main results；
- failure/non-failure split；
- schema validity；
- cost per success。

### 5. Ablations

- hook sanity；
- uniform vs adaptive budget；
- recovery strategy；
- hidden-state centroid baseline；
- static benchmark pressure test。

### 6. Deployment Analysis

- compact materialization equivalence；
- irregular vs hardware-aligned MLP；
- MLP-only speed signal；
- full-model latency limitation。

### 7. Limitations

- active FFN cost 不是 wall-clock speedup；
- 当前工具集是 deterministic local tools；
- 当前 dynamic routing 使用显式 trajectory event；
- compact runtime 仍需 kernel-aware engineering。

## 9. Contributions

1. 识别 PPL-based pruning evaluation 会遗漏 tool-schema 和 recovery failure。
2. 提出 trajectory-event-aware FFN budget routing。
3. 提出 schema-aware nested substrate，使 tool-call、final-answer 和 reflect 使用不同预算。
4. 在 3B/7B Real Tool 实验中验证动态预算；7B 达到 1000/1000，并降低约 15.1%
   theoretical active-FFN cost per successful episode。
5. 分析 compact deployment，证明 hardware alignment 对 MLP-level runtime signal
   是必要的，同时诚实报告当前 full-model speedup 尚未成立。

## 10. Stop List

当前不再优先：

```text
继续调当前 HF/PyTorch compact wrapper
直接做 dynamic Adaptive-B20 episode latency
扩展更多本地工具
重复同配置 7B Real Tool 1000
把 P5 写成真实加速结果
```

当前执行顺序：

```text
freeze P4/P5 evidence
-> run 3B protocol v1.2 alignment if using a cross-model main table
-> assemble main/ablation tables
-> write paper outline and method
-> optionally expand static benchmarks
```
