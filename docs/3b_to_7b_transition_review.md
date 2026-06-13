# 3B to 7B Transition Review, Current State

This document is for external logic/data review. It summarizes the transition
from the Qwen2.5-3B P2c result to the current Qwen2.5-7B P4 state.

The focus is the 7B transition. The 3B result is included only as the anchor
that motivated the 7B test.

## 1. Current Status

Current state:

```text
3B P2c main result exists and motivates scale-up.
7B plans were rebuilt from 7B calibration data.
Initial 7B baseline gate failed.
Two issues were found and fixed:
  1. identity no-op plan leaked stale bias compensation.
  2. 7B used a different tool-call wrapper for unit_convert.
After fixes, 7B dense/identity gate passed.
After protocol v1.1, 7B 20-task full minismoke passed.
The first 7B 100-task smoke exposed a lookup final-answer formatting issue.
Protocol v1.2 hardens observation-to-final instructions.
After protocol v1.2, 7B 100-task smoke passed.
After protocol v1.2, 7B 1000-task evaluation passed.
```

Current 7B result is a 1000-task active-FFN-cost result, not a wall-clock
speedup result.

## 2. 3B as the Anchor, Not the Main Focus

The 3B P2c result showed the main idea:

```text
tool_call_or_retry -> nested 10% plan
final_answer       -> nested 20% plan
reflect_recovery   -> dense
```

On 1000 real-tool tasks, the reported 3B table was:

| Method | Success | Failure | Non-failure | Cost / success | Schema validity |
|---|---:|---:|---:|---:|---:|
| dense | 997/1000 | 197/200 | 800/800 | 2.2066 | 1.0000 |
| identity_hook | 997/1000 | 197/200 | 800/800 | 2.2066 | 1.0000 |
| nested_uniform_10 | 997/1000 | 197/200 | 800/800 | 2.0060 | 1.0000 |
| adaptive_B20 | 997/1000 | 197/200 | 800/800 | 1.9057 | 1.0000 |

Interpretation:

```text
adaptive_B20 matched dense task success and reduced cost_per_success by about
13.6% versus dense, and by about 5.0% versus nested_uniform_10.
```

Important caveat for review:

```text
This 3B table was produced before the identity no-op fix.
The adaptive_B20 mechanism itself is not automatically invalidated, because the
10% and 20% plans intentionally use bias compensation.
However, final paper tables should use a post-fix sanity rerun for identity_hook
and dense-fallback comparisons.
```

For the rest of this note, the important point is only:

```text
3B gave a strong reason to test whether the same routing idea scales to 7B.
```

## 3. Why 7B Needed a Separate Gate

7B cannot reuse 3B masks or channel indices. The 7B path rebuilt its own
schema-aware nested plans:

```text
schema-aware calibration
FLAP-style scoring
nested 10% / 20% plans
audit
controlled prompt PPL sanity
real-tool minismoke
```

Before any 7B 100 or 1000 run, the project used a 20-task minimum smoke:

```text
dense
identity_hook
nested_uniform_10
adaptive_B20
```

The gate checks:

```text
dense == identity_hook
schema_validity is high
unit_convert works
failure tasks recover
adaptive_B20 is not worse than nested_uniform_10 in success
adaptive_B20 has lower cost_per_success than nested_uniform_10
```

## 4. First 7B Problem: Identity Hook Was Not a True No-op

### Symptom

The first 7B minismoke showed `dense` and `identity_hook` did not match.

Representative numbers:

| Method | Success | Unit convert |
|---|---:|---:|
| dense | 14/20 | 1/7 |
| identity_hook | 18/20 | 5/7 |
| nested_uniform_10 | 14/20 | 1/7 |
| adaptive_B20 | 14/20 | 1/7 |

This violated the basic baseline gate. With deterministic decoding,
`identity_hook` should be a no-op version of dense.

### Root Cause

`_empty_plan_like()` cleared pruned channel lists, but did not clear:

```text
mlp_output_bias_compensation
mlp_output_scale
```

So a plan intended to be identity could become:

```text
Dense forward + stale MLP output bias compensation
```

instead of true dense.

### Fix

Commit:

```text
9148dfc Fix identity no-op plan compensation leak
```

Files:

```text
scripts/evaluate_agent_masks.py
tests/test_evaluate_real_tool_loop.py
```

After this fix, the 7B dense and identity runs matched. This converted the
problem from "hook may be polluted" to "7B dense baseline itself does not yet
follow the protocol".

## 5. Second 7B Problem: Tool-call Wrapper Mismatch

### Symptom After Identity Fix

After the no-op fix, 7B dense and identity matched, but both still failed
`unit_convert`:

| Method | Success | Unit convert | Schema validity |
|---|---:|---:|---:|
| dense | 14/20 | 1/7 | 0.5556 |
| identity_hook | 14/20 | 1/7 | 0.5556 |

The repeated failing output was:

```json
{"type":"unit_convert","arguments":{"value":7,"from_unit":"m","to_unit":"cm"}}
```

The strict protocol requires:

```json
{"type":"tool_call","name":"unit_convert","arguments":{"value":7,"from_unit":"m","to_unit":"cm"}}
```

So 7B knew the correct tool and arguments, but used a different wrapper:

```text
wrong: tool name in top-level type
missing: top-level name field
```

### Interpretation

This was not:

```text
a pruning failure
an identity-hook failure
a unit_convert implementation failure
a reason to relax validator
```

It was a dense-baseline protocol-shape failure.

## 6. Protocol v1.1 Fix

Protocol v1.1 keeps the strict parser but hardens the prompt and schema-error
feedback.

Commits:

```text
83c18e6 Harden real tool protocol prompt
5ac5979 Add structured tool-call schema feedback
```

Files:

```text
src/safeprune/agent_loop.py
src/safeprune/tool_protocol.py
tests/test_agent_loop.py
tests/test_tool_protocol.py
docs/p2_substrate_and_p2c_results.md
```

What changed:

```text
The system prompt explicitly says:
- top-level "type" must be exactly "tool_call" or "final"
- never put a tool name in "type"
- the tool name belongs in top-level "name"
- a successful tool call is mandatory before final

For tool-name-as-type schema errors, the observation includes expected_example:
{"type":"tool_call","name":"unit_convert","arguments":{...}}
```

What did not change:

```text
No tolerant parser.
No string-to-number coercion.
No enum relaxation.
No extra-field tolerance.
The main benchmark protocol remains strict.
```

## 7. 7B Result After Protocol v1.1: Dense / Identity Gate

Prefix:

```text
outputs/agent_qwen2_5_7b/real_tool_eval/real_tool_v1_7b_minismoke_20_protocol_v1_1_gate
```

| Method | Success | Failure | Non-failure | Schema validity | Premature final | Calculator | Lookup | Unit convert |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 20/20 | 5/5 | 15/15 | 1.0000 | 0.0500 | 7/7 | 6/6 | 7/7 |
| identity_hook | 20/20 | 5/5 | 15/15 | 1.0000 | 0.0500 | 7/7 | 6/6 | 7/7 |

Events for both methods:

```text
timeout: 2
ok: 20
empty_observation: 2
tool_error: 1
premature_final: 1
```

Conclusion:

```text
7B dense baseline now passes the strict protocol gate.
identity_hook == dense after the no-op fix.
unit_convert wrapper errors disappeared.
```

## 8. 7B Result After Protocol v1.1: Full 20-task Minismoke

Prefix:

```text
outputs/agent_qwen2_5_7b/real_tool_eval/real_tool_v1_7b_minismoke_20_protocol_v1_1
```

| Method | Success | Failure | Non-failure | Schema validity | Premature final | Fallback step | Cost / success | Active FFN cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 20/20 | 5/5 | 15/15 | 1.0000 | 0.0500 | 0.0000 | 2.3000 | 46.0000 |
| identity_hook | 20/20 | 5/5 | 15/15 | 1.0000 | 0.0500 | 0.0000 | 2.3000 | 46.0000 |
| nested_uniform_10 | 20/20 | 5/5 | 15/15 | 1.0000 | 0.0000 | 0.1111 | 2.0500 | 40.9999 |
| adaptive_B20 | 20/20 | 5/5 | 15/15 | 1.0000 | 0.0000 | 0.1111 | 1.9500 | 38.9999 |

By tool for all four methods:

| Tool | Correct |
|---|---:|
| calculator | 7/7 |
| lookup | 6/6 |
| unit_convert | 7/7 |

Cost reductions in this 7B 20-task smoke:

| Comparison | Cost / success reduction |
|---|---:|
| nested_uniform_10 vs dense | 10.9% |
| adaptive_B20 vs dense | 15.2% |
| adaptive_B20 vs nested_uniform_10 | 4.9% |

Current interpretation:

```text
The 7B 20-task minismoke now supports the same direction as 3B:
schema-safe nested 10% works, and generation-type routing with 20% final-answer
budget further reduces active FFN cost without losing success on this smoke.
```

## 9. 7B 100-task Smoke Issue: Lookup Final Answer Dropped Prefix

After the 20-task minismoke passed, the project started a 100-task 7B smoke
under protocol v1.1:

```text
outputs/agent_qwen2_5_7b/real_tool_eval/real_tool_v1_7b_smoke_100_protocol_v1_1
```

The dense and identity rows matched exactly, but both failed the 100-task gate:

| Method | Success | Failure | Non-failure | Schema validity | Calculator | Lookup | Unit convert |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 91/100 | 20/20 | 71/80 | 1.0000 | 34/34 | 24/33 | 33/33 |
| identity_hook | 91/100 | 20/20 | 71/80 | 1.0000 | 34/34 | 24/33 | 33/33 |

The lookup failures were not tool-call failures. The tool call was correct and
the observation was correct:

```json
{"tool":"lookup","ok":true,"output":{"owner":"owner_3007"},"event":"ok"}
```

But the final answer dropped the required `owner_` prefix:

```json
{"type":"final","answer":"3007"}
```

Expected:

```json
{"type":"final","answer":"owner_3007"}
```

Known lookup failures from the trace:

```text
dense / identity_hook: 9 lookup failures, all owner_N -> N final-answer errors.
nested_uniform_10: 0 lookup failures in the provided trace.
adaptive_B20: 2 lookup failures in the provided trace.
```

Interpretation:

```text
This is an observation-to-final formatting issue, not a tool-call protocol
issue, not a lookup tool implementation issue, and not a pruning result.
The evaluator is correct to reject "3007" when the expected answer is
"owner_3007".
```

### Protocol v1.2 Observation-to-final Hardening

Protocol v1.2 adds exact final-answer instructions after successful tool
observations.

For lookup observations, the next prompt now says:

```text
For lookup, the final answer must be exactly "owner_N".
Do not answer only the numeric project id; keep the owner_ prefix.
```

For calculator and unit_convert observations, it similarly points to the
observed `result` value. This does not change the strict parser or the
evaluator. It only clarifies how to transform a successful observation into a
final answer.

Files:

```text
src/safeprune/agent_loop.py
tests/test_agent_loop.py
```

Required next step:

```text
Rerun 7B 100-task smoke under the post-v1.2 code.
Do not run 7B 1000 until dense/identity and adaptive_B20 pass the 100-task gate.
```

## 10. 7B Protocol v1.2 100-task Smoke

After protocol v1.2, the 7B 100-task smoke passed.

Prefix:

```text
outputs/agent_qwen2_5_7b/real_tool_eval/real_tool_v1_7b_smoke_100_protocol_v1_2
```

| Method | Success | Failure | Non-failure | Schema validity | Premature final | Fallback step | Cost / success | Active FFN cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 100/100 | 20/20 | 80/80 | 1.0000 | 0.0100 | 0.0000 | 2.2100 | 221.0000 |
| identity_hook | 100/100 | 20/20 | 80/80 | 1.0000 | 0.0100 | 0.0000 | 2.2100 | 221.0000 |
| nested_uniform_10 | 100/100 | 20/20 | 80/80 | 1.0000 | 0.0000 | 0.0909 | 2.0000 | 199.9997 |
| adaptive_B20 | 100/100 | 20/20 | 80/80 | 1.0000 | 0.0000 | 0.0909 | 1.9000 | 189.9997 |

By tool for all methods:

| Tool | Correct |
|---|---:|
| calculator | 34/34 |
| lookup | 33/33 |
| unit_convert | 33/33 |

Cost reductions in the 7B 100-task smoke:

| Comparison | Cost / success reduction |
|---|---:|
| nested_uniform_10 vs dense | 9.5% |
| adaptive_B20 vs dense | 14.0% |
| adaptive_B20 vs nested_uniform_10 | 5.0% |

Interpretation:

```text
Protocol v1.2 resolved the lookup owner-prefix issue.
The 100-task smoke satisfied the 1000-entry gate:
dense == identity_hook, all methods reached 100/100, all tools were perfect,
schema validity was 1.0, and adaptive_B20 had the lowest cost_per_success.
```

## 11. 7B Protocol v1.2 1000-task Result

The 7B 1000-task run completed under protocol v1.2.

Prefix:

```text
outputs/agent_qwen2_5_7b/real_tool_eval/real_tool_v1_7b_1000_protocol_v1_2
```

| Method | Success | Failure | Non-failure | Schema validity | Premature final | Fallback step | Cost / success | Active FFN cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 1000/1000 | 200/200 | 800/800 | 1.0000 | 0.0390 | 0.0000 | 2.2390 | 2239.0000 |
| identity_hook | 1000/1000 | 200/200 | 800/800 | 1.0000 | 0.0390 | 0.0000 | 2.2390 | 2239.0000 |
| nested_uniform_10 | 1000/1000 | 200/200 | 800/800 | 1.0000 | 0.0000 | 0.0909 | 2.0000 | 1999.9970 |
| adaptive_B20 | 1000/1000 | 200/200 | 800/800 | 1.0000 | 0.0000 | 0.0909 | 1.9000 | 1899.9974 |

By tool for all methods:

| Tool | Correct |
|---|---:|
| calculator | 333/333 |
| lookup | 333/333 |
| unit_convert | 334/334 |

Events:

| Method family | Events |
|---|---|
| dense / identity_hook | empty_observation: 67, timeout: 67, tool_error: 66, premature_final: 39, ok: 1000 |
| nested_uniform_10 / adaptive_B20 | empty_observation: 67, timeout: 67, tool_error: 66, ok: 1000 |

Cost reductions in the 7B 1000-task run:

| Comparison | Cost / success reduction |
|---|---:|
| nested_uniform_10 vs dense | 10.7% |
| adaptive_B20 vs dense | 15.1% |
| adaptive_B20 vs nested_uniform_10 | 5.0% |

Interpretation:

```text
This is the current 7B main result.
Adaptive-B20 preserves dense-equivalent success: 1000/1000 overall,
200/200 failure, 800/800 non-failure, and schema_validity 1.0.
It reduces active-FFN cost_per_success from 2.2390 to 1.9000.
The improvement over nested_uniform_10 shows generation-type routing is useful
beyond simply using a uniform 10% nested budget.
```

This is still an active-FFN-cost result. It is not a wall-clock speedup claim.

## 12. What Is Proven So Far

Supported by current evidence:

```text
1. Initial 7B failure was not a pruning result; it was a baseline gate problem.
2. The identity no-op bug was real and fixed.
3. The remaining 7B dense issue was a strict tool-call wrapper mismatch.
4. Protocol v1.1 fixes the wrapper issue without relaxing the parser.
5. 7B dense/identity gate passes after v1.1.
6. 7B 20-task minismoke passes for dense, identity, nested_uniform_10, and adaptive_B20.
7. On 7B 20-task smoke, adaptive_B20 has lower cost_per_success than nested_uniform_10.
8. The first 7B 100-task failure is now localized to lookup final-answer
   formatting, and protocol v1.2 targets that issue.
9. Protocol v1.2 passes 7B 100-task smoke.
10. Protocol v1.2 passes 7B 1000-task evaluation.
11. Adaptive-B20 matches dense success on 7B 1000 while reducing
    cost_per_success by about 15.1%.
```

Not proven yet:

```text
1. Real wall-clock speedup.
2. Compact subnet equivalence with mask-hook plans.
3. Final cross-model comparison between 3B protocol v1 and 7B protocol v1.2.
4. Whether 3B should be rerun under protocol v1.2 before presenting a single
   cross-model table.
5. General benchmark retention for 7B nested 10% / 20% static plans.
```

## 13. Next Step

Do not rerun the same 7B 1000 matrix. The next work should move to:

```text
1. Freeze the 7B protocol v1.2 result in the main docs.
2. Run 7B static benchmark sanity for dense / nested 10% / nested 20%.
3. Decide whether to rerun 3B under protocol v1.2 for a clean cross-model table.
4. Start compact subnet equivalence and latency work.
```

## 14. Review Checklist

Ask the reviewer to check:

```text
1. Is the identity no-op fix sufficient to guarantee identity_hook is a true no-op?
2. Does protocol v1.1 preserve the strict benchmark definition?
3. Is it correct to classify the pre-v1.1 7B failure as a protocol gate failure?
4. Is it correct to classify the first 7B 100-task failure as
   observation-to-final formatting rather than pruning?
5. Does the 7B 1000 protocol v1.2 result support the stated active-FFN-cost claim?
6. Are the 3B and 7B results separated clearly by protocol version and task scale?
7. Should 3B be rerun under protocol v1.2 before presenting 3B and 7B in one final table?
8. What evidence is still needed before claiming real wall-clock speedup?
```

## 15. Files to Inspect

Core protocol and runner:

```text
src/safeprune/agent_loop.py
src/safeprune/tool_protocol.py
src/safeprune/tool_env.py
src/safeprune/tools/registry.py
scripts/evaluate_real_tool_loop.py
```

Identity/no-op and substrate:

```text
scripts/evaluate_agent_masks.py
src/safeprune/masks.py
src/safeprune/substrate.py
src/safeprune/stage_masks.py
```

Tests covering the recent fixes:

```text
tests/test_evaluate_real_tool_loop.py
tests/test_agent_loop.py
tests/test_tool_protocol.py
```
