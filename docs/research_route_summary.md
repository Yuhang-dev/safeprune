# Agent Prune Research Route Summary

This document summarizes the current research route and the evidence gathered
so far. Detailed commands and full tables live in:

```text
docs/experiment_roadmap.md
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

This remains an active-FFN-cost metric. It is not a wall-clock speedup claim
until compact subnet latency is measured.

## 2. Evidence Chain

| Stage | Question | Result | What It Proves |
|---|---|---|---|
| Controlled Mask Validation v1 | Does stage-aware masking preserve more task behavior than observe-only at the same tiny budget? | stage-aware 898/1000 vs observe-only 836/1000; failure-redense 900/1000 | Stage and failure signals are useful in a controlled setup. |
| Real Tool v1 | Does the mechanism survive real tool execution? | Dense 997/1000; observe-failure-redense 997/1000; stage-reflect-dense 996/1000 with about half the fallback steps of fixed-window redense | Failure recovery is a localized compute bottleneck; reflect-localized dense fallback is more targeted than fixed-window fallback. |
| P1 Hidden-State Baseline | Can hidden-state-only routing replace explicit event routing? | hidden-only underperforms; event hybrid catches up but router-inclusive cost is much higher | Explicit Agent trajectory events are cheaper and more reliable than hidden-state-only centroid routing for this benchmark. |
| P2 Substrate v2 | Can the substrate move beyond 1% pruning? | 10% FLAP substrate reaches 990/1000 on 3B Real Tool 1000, cost_per_success down about 5.8% | The mechanism works at a meaningful 10% FFN budget, but not losslessly. |
| P2c Generation-Type Routing | Can different generation types use different budgets? | 3B Adaptive-B20 matches dense at 997/1000 and reduces cost_per_success by about 13.6% | Tool-call and final-answer generations have different FFN budget tolerance. |
| P4 7B Extension | Does the result scale to Qwen2.5-7B? | 7B Adaptive-B20 reaches 1000/1000 under strict protocol v1.2, cost_per_success 2.2390 -> 1.9000, about 15.1% lower | The strongest current result: dense-equivalent task success with lower active FFN cost. |
| P4 Static Benchmark | Is 20% safe as a static general model? | 7B nested_0p20 PPL rises from 11.7030 to 33.8450; MC accuracy drops | 20% is not a universal static compression setting; it should be used only as a dynamic final-answer budget. |
| P5 Compact Subnet | Can mask-hook cost become a physical subnet? | First-stage materialization implemented; 10%/20% compact preserve active ratio and 100% top-1 logits on 100 prompts; greedy exact drifts | Structure/logits equivalence is correct so far; single-subnet latency and task-level compact validation are next. |

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
latency: not yet tested
```

Greedy exact match is brittle because small bf16 numerical differences can
change later tokens. The stronger signal is that compact subnets preserve the
planned active ratio and last-token top-1 logits so far.

## 6. Immediate Next Steps

1. Run single-subnet latency smoke:

```text
dense
compact_10
compact_20
```

Report only:

```text
single-subnet latency / speed potential
```

Do not report Adaptive-B20 latency from single-subnet measurements.

2. Add compact task-level smoke after the compact runner exists:

```text
compact_10 Real Tool smoke
compact_20 Real Tool smoke as pressure test
```

3. Only after that, implement dynamic Adaptive-B20 episode latency:

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
