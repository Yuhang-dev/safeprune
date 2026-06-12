# Controlled Mask Validation v1

Date: 2026-06-11

This document freezes the controlled 1000-task mask validation run before
moving to the real tool-execution Agent Prune loop.

The run is still mask-hook algorithm validation. It does not claim wall-clock
speedup because the masked FFN channels are multiplied by zero inside the full
GEMM.

## What We Did And Why

We first validated Agent Prune in a controlled direct-answer setting:

```text
task prompt -> model final answer -> strict exact-match scoring
```

This setting intentionally avoids a full tool runtime so that we can isolate the
mask/routing question before adding tool-call parsing and environment effects.

The experiment asks three narrow questions:

1. Does a global layer-wise FFN budget avoid the collapse caused by naive
   per-layer pruning?
2. Does stage-aware routing improve over a fixed observe-only global mask at the
   same active FFN ratio?
3. Does failure-triggered re-densification recover injected failure tasks?

The controlled setting is now considered complete. Its role is to justify moving
to real tool execution, not to serve as the final Agent benchmark.

## Artifacts

Remote outputs:

```text
outputs/agent_qwen2_5_3b_4090_low/eval/stage_failure_official_1000.jsonl
outputs/agent_qwen2_5_3b_4090_low/eval/stage_failure_official_1000_metrics.json
outputs/agent_qwen2_5_3b_4090_low/eval/stage_failure_official_1000_plans.json
```

Methods:

```text
dense
identity_hook
global_balanced_observe_approx_0.01
stage_global_balanced_approx_0.01
failure_redense_global_balanced_approx_0.01
```

## Main Results

| Method | Correct | Task success | Active FFN | Active FFN cost | Cost / success | Failure success | Non-failure success | Collapse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 913/1000 | 0.913 | 1.0000 | 1000.0000 | 1.0953 | 189/200 | 724/800 | 0 |
| identity_hook | 913/1000 | 0.913 | 1.0000 | 1000.0000 | 1.0953 | 189/200 | 724/800 | 0 |
| global_balanced_observe_approx_0.01 | 836/1000 | 0.836 | 0.9900 | 990.0073 | 1.1842 | 178/200 | 658/800 | 0 |
| stage_global_balanced_approx_0.01 | 898/1000 | 0.898 | 0.9900 | 990.0073 | 1.1025 | 187/200 | 711/800 | 0 |
| failure_redense_global_balanced_approx_0.01 | 900/1000 | 0.900 | 0.9916 | 991.6061 | 1.1018 | 189/200 | 711/800 | 0 |

## Pairwise Split

| A | B | Both correct | A only | B only | Both wrong | Net A - B |
|---|---|---:|---:|---:|---:|---:|
| identity_hook | dense | 913 | 0 | 0 | 87 | 0 |
| global_balanced_observe_approx_0.01 | dense | 828 | 8 | 85 | 79 | -77 |
| stage_global_balanced_approx_0.01 | dense | 871 | 27 | 42 | 60 | -15 |
| failure_redense_global_balanced_approx_0.01 | dense | 878 | 22 | 35 | 65 | -13 |
| stage_global_balanced_approx_0.01 | global_balanced_observe_approx_0.01 | 823 | 75 | 13 | 89 | 62 |
| failure_redense_global_balanced_approx_0.01 | stage_global_balanced_approx_0.01 | 893 | 7 | 5 | 95 | 2 |

## By Tool

| Method | calculator | lookup | unit_convert |
|---|---:|---:|---:|
| dense | 258/334 | 330/333 | 325/333 |
| identity_hook | 258/334 | 330/333 | 325/333 |
| global_balanced_observe_approx_0.01 | 217/334 | 312/333 | 307/333 |
| stage_global_balanced_approx_0.01 | 274/334 | 317/333 | 307/333 |
| failure_redense_global_balanced_approx_0.01 | 270/334 | 320/333 | 310/333 |

## Failure Split By Tool

| Method | Subset | calculator | lookup | unit_convert |
|---|---|---:|---:|---:|
| dense | failure | 57/67 | 67/67 | 65/66 |
| dense | non_failure | 201/267 | 263/266 | 260/267 |
| global_balanced_observe_approx_0.01 | failure | 52/67 | 63/67 | 63/66 |
| global_balanced_observe_approx_0.01 | non_failure | 165/267 | 249/266 | 244/267 |
| stage_global_balanced_approx_0.01 | failure | 61/67 | 64/67 | 62/66 |
| stage_global_balanced_approx_0.01 | non_failure | 213/267 | 253/266 | 245/267 |
| failure_redense_global_balanced_approx_0.01 | failure | 57/67 | 67/67 | 65/66 |
| failure_redense_global_balanced_approx_0.01 | non_failure | 213/267 | 253/266 | 245/267 |

## Frozen Conclusions

The v1 controlled run validates two MVP claims:

```text
stage_global_balanced_approx_0.01 - global_balanced_observe_approx_0.01 = +62 tasks
```

At the same active FFN ratio, stage-aware routing substantially improves over
an observe-only global mask.

```text
failure_redense_global_balanced_approx_0.01 failure subset = dense failure subset = 189/200
```

Failure-triggered re-densification restores dense-level success on the injected
failure subset.

The run does not validate:

```text
Agent Prune beats dense on cost_per_success.
Mask hooks produce wall-clock speedup.
Controlled direct-answer results are final real tool-use benchmark results.
```

Next step: freeze this result and move to a real tool-execution loop where the
model must emit a structured tool call, the Python tool actually runs, and the
tool observation is fed back before the final answer.

## Current Follow-Up

The repository now contains the Real Tool-Execution Agent Prune v1 scaffold:

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

This follow-up exists because dense controlled calculator accuracy was only
`258/334`, meaning the old direct-answer task mixed Agent routing with the
model's own mental arithmetic. The real tool loop makes the model emit JSON
tool calls, executes local Python tools, returns observations, and scores the
final answer plus tool validity and recovery behavior.

Current real-tool stage:

```text
implementation: done locally
local structural tests: targeted tests pass
remote smoke: in progress
known issue fixed locally: evaluate_real_tool_loop.py now inserts repo root into sys.path;
remote can also use export PYTHONPATH=$PWD:$PWD/src
```
