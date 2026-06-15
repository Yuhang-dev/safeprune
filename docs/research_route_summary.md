# Agent Prune Research Route Summary

This document summarizes the current research route and the evidence gathered
so far. Detailed commands and full tables live in:

```text
docs/paper_experiment_record.md
docs/experiment_roadmap.md
docs/paper_evidence_and_outline.md
docs/p2_substrate_and_p2c_results.md
docs/p4_static_benchmark_next_commands.md
docs/agent_handoff_next_steps.md
```

## 1. Core Thesis

The project has shifted from generic structured pruning to:

```text
Trajectory-event-aware FFN compute allocation for tool-using reasoning agents.
```

The central claim is not that one static FFN mask is universally safe. The claim
is that Agent trajectories expose where compute is valuable:

```text
tool-call / retry generation -> schema-sensitive, use conservative sparse FFN
final-answer generation      -> lower schema risk, can use more aggressive FFN pruning
reflect recovery             -> failure-sensitive, temporarily restore dense FFN
```

The main metric is:

```text
cost_per_success = total active FFN cost / successful tasks
```

This remains an active-FFN-cost metric. P5 compact measurements did not show a
meaningful full-model decode speedup, so the main result must not be described
as a wall-clock speedup.

## 2. Evidence Chain

| Stage | Question | Result | What It Proves |
|---|---|---|---|
| Controlled Mask Validation v1 | Does stage-aware masking preserve more task behavior than observe-only at the same tiny budget? | stage-aware 898/1000 vs observe-only 836/1000; failure-redense 900/1000 | Stage and failure signals are useful in a controlled setup. |
| Real Tool v1 | Does the mechanism survive real tool execution? | Dense 997/1000; observe-failure-redense 997/1000; stage-reflect-dense 996/1000 with about half the fallback steps of fixed-window redense | Failure recovery is a localized compute bottleneck; reflect-localized dense fallback is more targeted than fixed-window fallback. |
| P1 Hidden-State Baseline | Can hidden-state-only routing replace explicit event routing? | hidden-only underperforms; event hybrid catches up but router-inclusive cost is much higher | Explicit Agent trajectory events are cheaper and more reliable than hidden-state-only centroid routing for this benchmark. |
| P2 Substrate v2 | Can the substrate move beyond 1% pruning? | 10% FLAP substrate reaches 990/1000 on 3B Real Tool 1000, cost_per_success down about 5.8% | The mechanism works at a meaningful 10% FFN budget, but not losslessly. |
| P2c Generation-Type Routing | Can different generation types use different budgets? | Post-fix 3B v1.2 Adaptive-B20 reaches 986/1000 vs dense 993/1000, failure 200/200, and lowers cost_per_success about 13.85% | Tool-call and final-answer generations have different FFN budget tolerance, but the 3B trade-off is not lossless. |
| P4 7B Extension | Does the result scale to Qwen2.5-7B? | 7B Adaptive-B20 reaches 1000/1000 under strict protocol v1.2, cost_per_success 2.2390 -> 1.9000, about 15.1% lower | The strongest current result: dense-equivalent task success with lower active FFN cost. |
| P4 Static Benchmark | Is 20% safe as a static general model? | 7B nested_0p20 PPL rises from 11.7030 to 33.8450; MC accuracy drops | 20% is not a universal static compression setting; it should be used only as a dynamic final-answer budget. |
| P5 Compact Subnet | Can mask-hook cost become a physical subnet? | Aligned 10%/20% compact preserve top-1 logits; aligned MLP-only speedups appear, but full-model fixed-decode is flat or negative | Hardware alignment is necessary but insufficient for end-to-end speedup under the current HF/PyTorch runtime. P5 is deployment analysis, not a main speedup result. |

## 3. Current Best Result

Qwen2.5-7B under Real Tool protocol v1.2:

| Method | Success | Failure | Non-failure | Cost / success | Schema validity |
|---|---:|---:|---:|---:|---:|
| Dense | 1000/1000 | 200/200 | 800/800 | 2.2390 | 1.0000 |
| Identity hook | 1000/1000 | 200/200 | 800/800 | 2.2390 | 1.0000 |
| Nested uniform 10 | 1000/1000 | 200/200 | 800/800 | 2.0000 | 1.0000 |
| Adaptive B20 | 1000/1000 | 200/200 | 800/800 | 1.9000 | 1.0000 |

Adaptive-B20 routing:

```text
tool_call_or_retry -> nested_0p10
final_answer       -> nested_0p20
reflect_recovery   -> dense
```

Interpretation:

```text
Adaptive-B20 matches dense task success and reduces theoretical active-FFN
cost_per_success by about 15.1% on Qwen2.5-7B.
```

## 4. Static Benchmark Caveat

7B static quick sanity:

| Plan | Active MLP | ARC-Easy | HellaSwag | PIQA | WikiText-2 PPL |
|---|---:|---:|---:|---:|---:|
| dense | 1.0000 | 0.7891 | 0.5820 | 0.8242 | 11.7030 |
| nested_0p10 | 0.9000 | 0.7773 | 0.5508 | 0.7656 | 15.5519 |
| nested_0p20 | 0.8000 | 0.7031 | 0.5273 | 0.7461 | 33.8450 |

This is important for the paper narrative:

```text
Nested 20% is not generally safe as a static model.
Adaptive-B20 works because 20% is only used on final-answer generations.
```

This supports the need for generation-type routing rather than weakening it.

## 5. Compact Subnet Status

Implemented first-stage P5 tooling:

```text
src/safeprune/compact.py
scripts/materialize_compact_subnet.py
scripts/evaluate_compact_subnet_equivalence.py
scripts/benchmark_compact_subnet_latency.py
scripts/analyze_compact_width_alignment.py
scripts/benchmark_compact_mlp_micro.py
scripts/build_hardware_aligned_compact_plans.py
tests/test_compact.py
```

Current 7B compact equivalence:

| Plan | Prompts | Active actual | Logits max diff | Logits mean diff | Top-1 logits match | Greedy exact match |
|---|---:|---:|---:|---:|---:|---:|
| compact nested_0p10 | 32 | 0.899998 | 0.5938 | 0.0851 | 32/32 | 26/32 |
| compact nested_0p20 | 32 | 0.799999 | 0.7500 | 0.1066 | 32/32 | 23/32 |
| compact nested_0p10 | 100 | 0.899998 | 0.7260 | 0.0906 | 100/100 | skipped |
| compact nested_0p20 | 100 | 0.799999 | 0.7500 | 0.1062 | 100/100 | skipped |

Interpretation:

```text
compact structure: pass
compact logits/top1 equivalence: pass so far
compact greedy exact equivalence: not suitable as a strict pass/fail metric
compact task-level equivalence: not yet tested
naive latency: failed / slower than dense
```

Greedy exact match is brittle because small bf16 numerical differences can
change later tokens. The stronger signal is that compact subnets preserve the
planned active ratio and last-token top-1 logits so far.

## 6. Current Next Steps

P5 diagnostics are complete enough for the current paper:

```text
compact materialization and top-1 equivalence: pass
hardware-aligned MLP-only speed signal: pass
meaningful full-model fixed-decode speedup: not demonstrated
dynamic Agent latency: deferred
```

Current priorities:

1. Inspect the 14 non-failure lookup errors from the post-fix 3B protocol v1.2
   Adaptive-B20 run.
2. Build the final 7B main table, cross-model table, 3B failure-localization table, router ablation,
   static sanity table, and P5 deployment table.
3. Draft Method and Experiments using `docs/paper_experiment_record.md`.
4. Optionally expand static benchmarks from 256 samples to 1024 or full
   validation.
5. Resume compact runtime engineering only with a lower-level kernel/runtime or
   more hardware-friendly pruning granularity.

## 6A. Historical P5 Diagnostic Record

The following sequence records how P5 was diagnosed. It is not the current
execution plan.

Naive batch-1 single-subnet latency smoke:

| Model | Active MLP | Prefill ms | Decode ms/token | Tok/s | Peak GB |
|---|---:|---:|---:|---:|---:|
| dense | 1.0000 | 28.803 | 4.304 | 232.316 | 8.266 |
| nested_0p10 compact | 0.9000 | 41.532 | 6.403 | 156.184 | 14.093 |
| nested_0p20 compact | 0.8000 | 40.268 | 15.401 | 64.933 | 13.920 |

Interpretation:

```text
Do not claim wall-clock speedup.

The compact subnet is structurally valid, but the naive runtime path is slower
and uses more peak memory. Likely causes include same-process memory artifacts,
custom wrapper overhead, irregular GEMM shapes, and generation/token-length
measurement artifacts.
```

1. Re-run latency as one variant per Python process:

```text
dense
compact_10
compact_20
```

Goal:

```text
confirm whether the high compact memory and slow latency persist without
same-process dense -> compact -> compact interference
```

2. Run compact width-alignment audit:

```text
check per-layer remaining FFN width modulo 64/128/256
```

If widths are mostly unaligned, P5 should move to block-aligned plans before
serious latency claims.

3. Run MLP-only microbenchmark:

```text
dense MLP vs compact 10% MLP vs compact 20% MLP
input shapes [1,1,H], [1,128,H], [4,128,H]
```

If MLP-only is not faster, the issue is shape/kernel efficiency. If MLP-only is
faster but full-model generation is slower, the issue is wrapper/generation/
benchmark overhead.

Hardware-aligned MLP-only microbenchmark status:

| Plan | B1 S1 | B1 S128 | B4 S128 | Interpretation |
|---|---:|---:|---:|---|
| dense | 0.217 ms | 0.494 ms | 0.874 ms | Baseline |
| aligned_0p10 | 0.240 ms | 0.504 ms | 0.770 ms | Near dense for batch 1, faster for B4 S128 |
| aligned_0p20 | 0.185 ms | 0.220 ms | 0.466 ms | Faster than dense across sampled shapes |

After skipping no-op wrappers on dense-width layers, aligned_0p20 passes the
MLP-only latency gate. It is now reasonable to retry full-model single-subnet
latency for aligned_0p20 and to run aligned compact equivalence before any task
or dynamic latency claim.

Aligned logits-only compact equivalence on 100 prompts also passed:

```text
aligned_0p10 active_mlp_ratio_actual = 0.899614, mean diff = 0.0828, top1 = 1.0
aligned_0p20 active_mlp_ratio_actual = 0.799710, mean diff = 0.0914, top1 = 1.0
```

The first aligned_0p20 full-model latency run used HF `generate()` and produced
32 new tokens. Because dense can stop earlier on the same prompts, full-model
latency must now be measured with fixed-step decode before drawing a speedup
conclusion.

Fixed-decode batch-1 full-model latency:

| Plan | Decode ms/token | Tok/s | Peak GB | Interpretation |
|---|---:|---:|---:|---|
| dense | 26.226 | 38.130 | 8.266 | Baseline |
| aligned_0p10 | 25.497 | 39.220 | 7.993 | Slight speedup, about 2.8% |
| aligned_0p20 | 26.680 | 37.481 | 7.857 | Slightly slower than dense |

This means MLP-only speedups are not yet full-model speedups at batch-1 decode.
Batch-4 fixed-decode full-model latency:

| Plan | Decode ms/token | Tok/s | Peak GB | Interpretation |
|---|---:|---:|---:|---|
| dense | 27.319 | 146.421 | 8.335 | Baseline |
| aligned_0p10 | 27.200 | 147.059 | 8.062 | Noise-level speedup, about 0.4% |
| aligned_0p20 | 27.700 | 144.405 | 7.926 | Slower than dense |

P5 current status:

```text
MLP-only aligned compact speedup: positive
full-model fixed-decode speedup: not demonstrated
GPU memory: slightly lower for compact plans
dynamic Agent latency: do not run yet
```

This should be frozen as a partial systems result: hardware alignment recovers
MLP-level speed potential, but the current PyTorch/HF full-model decode path
does not convert that into meaningful end-to-end latency. Further speedup work
requires a lower-level kernel/runtime path, coarser block/group pruning, or a
workload where FFN GEMMs dominate more strongly.

4. Build or refine hardware-aligned compact plans:

```text
aligned_0p10 / aligned_0p20
remaining width aligned to 128 or 256
min_remaining >= 4096
nested pruned sets preserved
```

Use these plans only for P5 latency engineering. They do not replace the frozen
P4 Real Tool active-cost result. Before task or dynamic latency claims, validate
aligned mask-hook vs compact equivalence and then single-subnet full-model
latency.

5. Add compact task-level smoke after aligned latency diagnostics:

```text
compact_10 Real Tool smoke
compact_20 Real Tool smoke as pressure test
```

6. Only after that, implement dynamic Adaptive-B20 episode latency:

```text
tool-call/retry -> compact_10
final-answer    -> compact_20
reflect         -> dense
```

This final step must report switch overhead, memory cost, P50/P95 episode
latency, and successful-episode latency.

## 7. Do Not Claim Yet

Do not claim:

```text
wall-clock speedup
deployment-ready compact subnet
uniform 20% static compression is safe
attention-head or KV-cache pruning is safe
dynamic Adaptive-B20 latency improvement
```

Allowed current claim:

```text
On Qwen2.5-7B, Adaptive-B20 matches dense task success in a strict Real Tool
loop and reduces theoretical active-FFN cost_per_success by about 15.1%.
Static benchmarks show why this must be dynamic: 20% is not generally safe as a
static model, but can be reserved for final-answer generations.
```
