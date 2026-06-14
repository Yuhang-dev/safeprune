# P2 Substrate and P2c Adaptive Routing Results

日期：2026-06-13

本文承接 `docs/real_tool_v1_results.md`。Real Tool v1 已经证明：

```text
failure / reflect trajectory events can localize high-value recovery compute.
```

P2/P2c 的目标不是继续手工调 1% mask，而是把底层 FFN 结构化剪枝率推到有论文意义的预算，并验证更激进预算能否按 generation type 安全分配。

## 1. 当前结论

当前已经形成三条主证据：

| Stage | Evidence | Result |
|---|---|---|
| Real Tool v1 | 1% trajectory-event-aware routing | failure recovery 是局部 reflect-step 计算瓶颈。 |
| P1 hidden-state baseline | centroid router 1000 | hidden-state-only 不如显式 event；event hybrid 追平但 router-inclusive cost 高。 |
| P2 10% substrate | FLAP substrate 1000 | 10% FFN pruning + reflect dense 稳定但非无损，cost/success 下降约 5.8%。 |
| P2c adaptive routing | schema-aware nested 1000 | tool-call 10%、answer 20%、reflect dense 达到 dense 同等 997/1000，cost/success 下降到 1.9057。 |

论文表述应保持克制：

```text
10% substrate is not lossless compression.
It preserves strong real-tool success while reducing active FFN cost.
P2c shows that final-answer generation can tolerate a more aggressive budget
than schema-sensitive tool-call generation.
```

当前仍不声称真实 wall-clock speedup；所有成本都是 mask-hook active FFN cost。

## 2. P1 Hidden-State Baseline

产物：

```text
outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_p1_hidden_1000*
```

| Method | Success | Failure | Non-failure | Cost / success | Router-inclusive cost | Fallback step |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 997/1000 | 197/200 | 800/800 | 2.2066 | 2.2066 | 0.0000 |
| Identity hook | 997/1000 | 197/200 | 800/800 | 2.2066 | 2.2066 | 0.0000 |
| Observe-failure-redense | 997/1000 | 197/200 | 800/800 | 2.1886 | 2.1886 | 0.1818 |
| Stage-reflect-dense | 996/1000 | 196/200 | 800/800 | 2.1918 | 2.1918 | 0.0926 |
| Hidden centroid | 980/1000 | 180/200 | 800/800 | 2.3093 | 4.6420 | 0.0000 |
| Hidden centroid reflect-dense | 983/1000 | 183/200 | 800/800 | 2.2609 | 4.5437 | 0.0392 |
| Hidden centroid + event | 996/1000 | 196/200 | 800/800 | 2.1918 | 4.4036 | 0.0926 |

Interpretation:

```text
Hidden-state-only centroid routing does not replace explicit Agent trajectory events.
The event hybrid matches stage-reflect-dense, but the router-inclusive cost is high
because the centroid router needs a hidden-state prefill.
```

This is a reviewer baseline, not the main method.

## 3. P2 Substrate v2

P2 introduces external substrate plans:

```text
scripts/build_pruning_substrate_v2.py
scripts/evaluate_pruning_ppl.py
scripts/evaluate_real_tool_loop.py --substrate-plan / --substrate-name
```

The first substrate v2 uses FLAP-style channel scoring, global budget allocation, and bias compensation. Layer scale is currently a `1.0` placeholder, not real scale calibration.

### Controlled prompt PPL sanity

This is controlled-task prompt PPL, not standard WikiText-2 PPL.

| Plan | Active MLP ratio | PPL | Relative PPL increase |
|---|---:|---:|---:|
| Dense | 1.0000 | 23.2031 | 0.00% |
| FLAP 3% | 0.9697 | 23.6021 | +1.72% |
| FLAP 5% | 0.9497 | 23.6440 | +1.90% |
| FLAP 10% | 0.8998 | 24.7785 | +6.79% |
| FLAP 20% | 0.7998 | 24.0066 | +3.46% |

This only says the substrate does not collapse on controlled prompts. The paper still needs standard WikiText-2 and common benchmark checks.

### Real Tool 100 smoke

| Method | Success | Failure | Non-failure | Cost / success | Fallback step |
|---|---:|---:|---:|---:|---:|
| Dense | 100/100 | 20/20 | 80/80 | 2.2000 | 0.0000 |
| FLAP 3% stage-reflect-dense | 100/100 | 20/20 | 80/80 | 2.1395 | 0.0909 |
| FLAP 5% stage-reflect-dense | 66/100 | 13/20 | 53/80 | 5.1615 | 0.5244 |
| FLAP 10% stage-reflect-dense | 99/100 | 19/20 | 80/80 | 2.0399 | 0.0991 |
| FLAP 5% observe-failure-redense | 66/100 | 13/20 | 53/80 | 5.1714 | 0.5616 |
| FLAP 10% observe-failure-redense | 99/100 | 19/20 | 80/80 | 2.0602 | 0.1892 |

Interpretation:

```text
3% is safe.
10% is the first meaningful substrate budget that remains stable enough for 1000.
5% is an allocation anomaly, not a conservative result.
```

### 5% audit and 20% pressure smoke

The audit found no duplicate or out-of-range channel indices. The 5% failure came from non-nested independent allocation:

```text
flap_0p05_vs_flap_0p10: 9353 channels pruned at 5% are not pruned at 10%.
```

Uniform 20% failed real-tool smoke and should not enter the 1000 table:

| Method | Success | Failure | Non-failure | Cost / success | Schema validity |
|---|---:|---:|---:|---:|---:|
| Dense | 100/100 | 20/20 | 80/80 | 2.2000 | 1.0000 |
| FLAP 20% stage-reflect-dense | 60/100 | 9/20 | 51/80 | 6.5194 | 0.5247 |

The failure is concentrated in protocol/schema behavior, especially `unit_convert = 0/33`.

## 4. P2 10% Real Tool 1000

产物：

```text
outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_substrate_flap_0p10_1000*
```

| Method | Success | Failure | Non-failure | Cost / success | Active FFN cost | Fallback step | Schema validity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense | 997/1000 | 197/200 | 800/800 | 2.2066 | 2200.0000 | 0.0000 | 1.0000 |
| Identity hook | 997/1000 | 197/200 | 800/800 | 2.2066 | 2200.0000 | 0.0000 | 1.0000 |
| FLAP 10% stage-reflect-dense | 990/1000 | 195/200 | 795/800 | 2.0777 | 2056.9205 | 0.1156 | 0.9876 |
| FLAP 10% observe-failure-redense | 990/1000 | 195/200 | 795/800 | 2.0987 | 2077.6726 | 0.2114 | 0.9876 |

Interpretation:

```text
10% stage-reflect-dense is stable but not lossless: success drops from 997 to 990.
Cost per success drops from 2.2066 to 2.0777, about 5.8%.
Active FFN cost drops from 2200.0 to 2056.9, about 6.5%.
Compared with 10% observe-failure-redense, success is identical but fallback step ratio
drops from 0.2114 to 0.1156, about 45% fewer dense fallback steps.
```

This preserves the Real Tool v1 story at a meaningful 10% FFN pruning budget.

## 5. P2c Schema-Aware Nested Budget Routing

P2c fixes the independent-plan problem and adds generation-type routing.

Implemented entry points:

```text
scripts/prepare_schema_calibration.py
scripts/build_pruning_substrate_v2.py --nested-budget-ladder --schema-calibration-path
scripts/evaluate_real_tool_loop.py methods:
  nested_uniform_10
  adaptive_A
  adaptive_B15
  adaptive_B20
```

Nested budget ladder:

```text
5%, 10%, 12%, 15%, 18%, 20%
```

If `M_r` is the deleted channel set, the builder enforces:

```text
M_5 subset M_10 subset M_12 subset M_15 subset M_18 subset M_20
```

Nested audit:

```text
duplicate channel indices: 0
out-of-range channel indices: 0
nesting overlap: 100% for adjacent budgets
```

Generation-type routing:

| Method | Tool-call / retry | Final answer | Reflect recovery |
|---|---:|---:|---:|
| `nested_uniform_10` | 10% | 10% | dense |
| `adaptive_A` | 5% | 15% | dense |
| `adaptive_B15` | 10% | 15% | dense |
| `adaptive_B20` | 10% | 20% | dense |

The first version uses schema-heavy assistant-target calibration snippets. The `--schema-token-weight` interface exists, but true token-level weighted loss is not yet implemented.

### P2c Real Tool 100 smoke

产物：

```text
outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_p2c_adaptive_smoke_100*
```

| Method | Success | Failure | Non-failure | Cost / success | Active FFN cost | Schema validity |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 100/100 | 20/20 | 80/80 | 2.2000 | 220.0000 | 1.0000 |
| Nested uniform 10 | 100/100 | 20/20 | 80/80 | 2.0000 | 199.9999 | 1.0000 |
| Adaptive A | 100/100 | 20/20 | 80/80 | 2.0000 | 199.9996 | 1.0000 |
| Adaptive B15 | 100/100 | 20/20 | 80/80 | 1.9500 | 194.9997 | 1.0000 |
| Adaptive B20 | 100/100 | 20/20 | 80/80 | 1.9000 | 189.9998 | 1.0000 |

By tool:

| Method | Calculator | Lookup | Unit convert |
|---|---:|---:|---:|
| Dense | 34/34 | 33/33 | 33/33 |
| Nested uniform 10 | 34/34 | 33/33 | 33/33 |
| Adaptive A | 34/34 | 33/33 | 33/33 |
| Adaptive B15 | 34/34 | 33/33 | 33/33 |
| Adaptive B20 | 34/34 | 33/33 | 33/33 |

`adaptive_B20` trace check:

```text
tool_call_or_retry -> nested_0p10
reflect_recovery   -> reflect_dense
final_answer       -> nested_0p20
```

Interpretation:

```text
Schema-aware nested plans fixed the uniform 20% protocol collapse seen in P2.
Adaptive B20 is the best 100-smoke candidate: same 100/100 success as dense and nested_uniform_10,
but lower cost per success.
```

## 6. P2c Real Tool 1000

产物：

```text
outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_p2c_adaptive_1000*
```

Minimal explanatory matrix:

```text
dense
identity_hook
nested_uniform_10
adaptive_B20
```

Main table:

| Method | Success | Failure | Non-failure | Cost / success | Active FFN cost | Fallback step | Schema validity | Premature final | Collapse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | 997/1000 | 197/200 | 800/800 | 2.2066 | 2200.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| Identity hook | 997/1000 | 197/200 | 800/800 | 2.2066 | 2200.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| Nested uniform 10 | 997/1000 | 197/200 | 800/800 | 2.0060 | 1999.9990 | 0.0909 | 1.0000 | 0.0000 | 0.0000 |
| Adaptive B20 | 997/1000 | 197/200 | 800/800 | 1.9057 | 1899.9985 | 0.0909 | 1.0000 | 0.0000 | 0.0000 |

By tool:

| Method | Calculator | Lookup | Unit convert |
|---|---:|---:|---:|
| Dense | 331/334 | 333/333 | 333/333 |
| Identity hook | 331/334 | 333/333 | 333/333 |
| Nested uniform 10 | 331/334 | 333/333 | 333/333 |
| Adaptive B20 | 331/334 | 333/333 | 333/333 |

Cost reductions:

| Comparison | Cost / success reduction | Active FFN cost reduction |
|---|---:|---:|
| Nested uniform 10 vs dense | 9.1% | 9.1% |
| Adaptive B20 vs dense | 13.6% | 13.6% |
| Adaptive B20 vs nested uniform 10 | 5.0% | 5.0% |

Interpretation:

```text
P2c reaches dense-equivalent task success: 997/1000 overall, 197/200 failure,
and 800/800 non-failure for both nested_uniform_10 and adaptive_B20.

Nested uniform 10 proves the schema-aware nested 10% substrate is stable.
Adaptive B20 proves generation-type routing adds value: same task success and
same fallback_step_ratio as nested_uniform_10, but lower cost_per_success by
about 5.0% because final-answer generations use the 20% plan.

Compared with dense, adaptive_B20 lowers cost_per_success by about 13.6%
without increasing schema errors, premature final, or generation collapse.
```

This is the current P2c main result. It is still an active-FFN-cost result, not a wall-clock speedup result.

### Post-freeze identity-hook fix

After the first 7B minismoke, we found a no-op-plan construction bug:
`_empty_plan_like()` cleared `pruned_mlp_channels` but did not remove
`mlp_output_bias_compensation` or `mlp_output_scale`. For substrate plans with
bias compensation, `identity_hook` could therefore become:

```text
Dense forward + stale MLP output bias compensation
```

instead of a true no-op hook. This directly explains why a 7B minismoke could
show `dense != identity_hook` under deterministic decoding.

The fix is in `scripts/evaluate_agent_masks.py`: `_empty_plan_like()` now clears
both compensation fields. The regression test is
`tests/test_evaluate_real_tool_loop.py::EvaluateRealToolLoopTests.test_empty_plan_like_removes_bias_compensation_and_scale`.

Impact on frozen 3B results:

```text
The main Adaptive-B20 mechanism is not automatically invalidated, because the
10% and 20% substrate plans intentionally use their own bias compensation.

However, every table row that uses identity_hook or a dense fallback generated
from _empty_plan_like() should be treated as requiring a sanity rerun after this
fix. In particular, the strict claim "identity_hook == dense" must be
reconfirmed before using the table as final paper evidence.
```

Required post-fix sanity rerun:

```text
3B: dense / identity_hook / nested_uniform_10 / adaptive_B20 on the P2c 1000 set.
7B: dense / identity_hook first, then the 20-task minismoke matrix.
```

### Real Tool protocol v1.1 hardening

The post-fix 7B dense/identity gate showed `dense == identity_hook`, but dense
still failed on 6/7 `unit_convert` tasks. The trace showed a concentrated schema
mistake:

```json
{"type":"unit_convert","arguments":{"value":7,"from_unit":"m","to_unit":"cm"}}
```

This is not a validator bug. The strict protocol requires:

```json
{"type":"tool_call","name":"unit_convert","arguments":{"value":7,"from_unit":"m","to_unit":"cm"}}
```

So v1.1 hardens only the prompt and error feedback:

```text
Do not put tool names in type.
The top-level type must be exactly tool_call or final.
If a previous JSON object was rejected, fix the shape instead of repeating it.
For tool-name-as-type errors, return an expected_example with the same arguments
wrapped as {"type":"tool_call","name":"<tool>","arguments":{...}}.
```

The parser remains strict; no string-to-number coercion, enum relaxation, or
extra-field tolerance is added to the main benchmark. If tolerant parsing is
needed later, it should be a separate ablation, not the main protocol.

### Real Tool protocol v1.2 observation-to-final hardening

The 7B protocol v1.1 20-task minismoke passed, but the first 100-task smoke
exposed a lookup-specific final-answer problem. Dense and identity matched, but
both reached only 91/100 because lookup succeeded at the tool level and then
the final answer dropped the `owner_` prefix:

```text
observation: {"owner":"owner_3007"}
wrong final: {"type":"final","answer":"3007"}
expected:    {"type":"final","answer":"owner_3007"}
```

This is an observation-to-final formatting issue, not a parser or tool-call
issue. Protocol v1.2 adds exact final-answer instructions after successful
observations. For lookup, the model is told to answer exactly the observed
`owner_N` value and not only the numeric project id. The strict parser and
strict evaluator remain unchanged.

### Qwen2.5-7B protocol v1.2 Real Tool 1000

After protocol v1.2, the 7B 100-task smoke passed, then the 7B 1000-task matrix
completed successfully.

Prefix:

```text
outputs/agent_qwen2_5_7b/real_tool_eval/real_tool_v1_7b_1000_protocol_v1_2
```

| Method | Success | Failure | Non-failure | Cost / success | Active FFN cost | Fallback step | Schema validity | Premature final | Collapse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | 1000/1000 | 200/200 | 800/800 | 2.2390 | 2239.0000 | 0.0000 | 1.0000 | 0.0390 | 0.0000 |
| Identity hook | 1000/1000 | 200/200 | 800/800 | 2.2390 | 2239.0000 | 0.0000 | 1.0000 | 0.0390 | 0.0000 |
| Nested uniform 10 | 1000/1000 | 200/200 | 800/800 | 2.0000 | 1999.9970 | 0.0909 | 1.0000 | 0.0000 | 0.0000 |
| Adaptive B20 | 1000/1000 | 200/200 | 800/800 | 1.9000 | 1899.9974 | 0.0909 | 1.0000 | 0.0000 | 0.0000 |

By tool:

| Method | Calculator | Lookup | Unit convert |
|---|---:|---:|---:|
| Dense | 333/333 | 333/333 | 334/334 |
| Identity hook | 333/333 | 333/333 | 334/334 |
| Nested uniform 10 | 333/333 | 333/333 | 334/334 |
| Adaptive B20 | 333/333 | 333/333 | 334/334 |

Cost reductions:

| Comparison | Cost / success reduction | Active FFN cost reduction |
|---|---:|---:|
| Nested uniform 10 vs dense | 10.7% | 10.7% |
| Adaptive B20 vs dense | 15.1% | 15.1% |
| Adaptive B20 vs nested uniform 10 | 5.0% | 5.0% |

Interpretation:

```text
Qwen2.5-7B protocol v1.2 is now the strongest current Agent Prune result.
Adaptive-B20 matches dense task success on 1000/1000 tasks, including
200/200 injected-failure tasks, with schema_validity 1.0 and no collapse.
It reduces active-FFN cost_per_success from 2.2390 to 1.9000, about 15.1%.
```

This remains an active-FFN-cost result, not a wall-clock latency result.

Command history and follow-up commands are centralized in
`docs/p4_static_benchmark_next_commands.md`. Do not repeat the same 7B
protocol v1.2 Real Tool 1000 matrix.

Relevant output prefixes:

```text
outputs/agent_qwen2_5_7b/real_tool_eval/real_tool_v1_7b_minismoke_20_protocol_v1_1_gate
outputs/agent_qwen2_5_7b/real_tool_eval/real_tool_v1_7b_minismoke_20_protocol_v1_1
outputs/agent_qwen2_5_7b/real_tool_eval/real_tool_v1_7b_smoke_100_protocol_v1_2
outputs/agent_qwen2_5_7b/real_tool_eval/real_tool_v1_7b_1000_protocol_v1_2
```

3B P2c historical reproduction command:

```bash
python -u scripts/evaluate_real_tool_loop.py \
  --config configs/agent_qwen2_5_3b_4090.yaml \
  --tasks data/agent/real_tool_tasks_v1.jsonl \
  --mask-bank-path outputs/agent_qwen2_5_3b_4090_low/mask_bank/mask_bank.json \
  --local-files-only \
  --substrate-plan outputs/agent_qwen2_5_3b_4090_low/substrate_v2_schema_nested/flap/budget_plan_0p05.json \
  --substrate-name nested_0p05 \
  --substrate-plan outputs/agent_qwen2_5_3b_4090_low/substrate_v2_schema_nested/flap/budget_plan_0p10.json \
  --substrate-name nested_0p10 \
  --substrate-plan outputs/agent_qwen2_5_3b_4090_low/substrate_v2_schema_nested/flap/budget_plan_0p15.json \
  --substrate-name nested_0p15 \
  --substrate-plan outputs/agent_qwen2_5_3b_4090_low/substrate_v2_schema_nested/flap/budget_plan_0p20.json \
  --substrate-name nested_0p20 \
  --methods dense identity_hook nested_uniform_10 adaptive_B20 \
  --output-prefix outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_p2c_adaptive_1000
```

## 7. Paper Story

The paper story can now be written as:

```text
1. PPL alone can miss Agent tool-schema degradation.
2. Uniform high pruning can break structured tool protocol.
3. Failure recovery is a localized reflect-step compute bottleneck.
4. Explicit Agent trajectory events are reliable, near-zero-overhead routing signals.
5. Schema-aware nested budgets allow conservative tool-call pruning,
   aggressive final-answer pruning, and dense reflect recovery.
6. Adaptive-B20 matches dense task success while reducing cost per successful task
   by 13.6% on the 3B P2c run and about 15.1% on the 7B protocol v1.2 run.
```

## 8. Next Work

P3 benchmark sanity:

```text
WikiText-2 PPL
PIQA
HellaSwag
ARC-Easy
```

Minimum benchmark matrix:

```text
Dense
Nested 10%
Nested 15% (optional)
Nested 20% pressure test
```

Run:

```bash
python -u scripts/evaluate_static_benchmarks.py \
  --config configs/agent_qwen2_5_3b_4090.yaml \
  --local-files-only \
  --calibration-path data/agent/controlled_tasks.jsonl \
  --calibration-path data/agent/schema_calibration_v1.jsonl \
  --plan outputs/agent_qwen2_5_3b_4090_low/substrate_v2_schema_nested/flap/budget_plan_0p10.json \
  --plan-name nested_0p10 \
  --plan outputs/agent_qwen2_5_3b_4090_low/substrate_v2_schema_nested/flap/budget_plan_0p15.json \
  --plan-name nested_0p15 \
  --plan outputs/agent_qwen2_5_3b_4090_low/substrate_v2_schema_nested/flap/budget_plan_0p20.json \
  --plan-name nested_0p20 \
  --benchmarks wikitext2 piqa hellaswag arc_easy \
  --max-samples 256 \
  --benchmark-version p3_static_v1 \
  --output-dir outputs/agent_qwen2_5_3b_4090_low/static_benchmarks/p3_static_v1
```

Result:

| Plan | Active MLP | ARC-Easy | HellaSwag | PIQA | WikiText-2 PPL |
|---|---:|---:|---:|---:|---:|
| Dense | 1.0000 | 0.6836 | 0.5312 | 0.8008 | 13.8088 |
| Nested 10% | 0.9000 | 0.6719 | 0.5352 | 0.8008 | 13.9116 |
| Nested 15% | 0.8500 | 0.6328 | 0.5156 | 0.7891 | 26.1213 |
| Nested 20% | 0.8000 | 0.6445 | 0.4805 | 0.7539 | 36.4053 |

Interpretation:

```text
Nested 10% preserves general benchmark sanity: WikiText-2 PPL is close to dense,
PIQA is unchanged on the 256-sample sanity set, and ARC-Easy / HellaSwag only move slightly.

Nested 15% and 20% are useful pressure points but should not be presented as
generally safe static models. Their WikiText-2 PPL degradation is large.

This supports the P2c routing story: 20% is acceptable only for selected
final-answer Agent generations, not as a uniform static model.
```

This answers whether the Real Tool gain comes with unacceptable general-capability loss.

P4 Qwen2.5-7B extension is now complete for Real Tool protocol v1.2:

```text
dense / identity_hook / nested_uniform_10 / adaptive_B20 all reached 1000/1000.
adaptive_B20 reduced active-FFN cost_per_success by about 15.1% versus dense.
```

Do not rerun the same 7B protocol v1.2 Real Tool 1000 matrix. The 7B static
quick sanity and subnet theory commands have completed.

```text
1. Run 3B protocol v1.2 1000 only if the final paper table compares 3B and 7B.
2. Otherwise proceed to P5 compact subnet materialization and latency.
3. Expand 7B static benchmark beyond 256 samples only if a final general
   benchmark table is needed.
```

Use `docs/p4_static_benchmark_next_commands.md` for the exact commands.

P4 7B static benchmark quick sanity:

| Plan | Active MLP | ARC-Easy | HellaSwag | PIQA | WikiText-2 PPL |
|---|---:|---:|---:|---:|---:|
| dense | 1.0000 | 0.7891 | 0.5820 | 0.8242 | 11.7030 |
| nested_0p10 | 0.9000 | 0.7773 | 0.5508 | 0.7656 | 15.5519 |
| nested_0p20 | 0.8000 | 0.7031 | 0.5273 | 0.7461 | 33.8450 |

P4 7B subnet theory:

| Name | Active FFN | Remaining channels | FFN MACs/token | Total FLOPs/token |
|---|---:|---:|---:|---:|
| dense | 1.0000 | 530432 | 5703204864 | 13050576896 |
| nested_0p10 | 0.9000 | 477388 | 5132875776 | 11909918720 |
| nested_0p20 | 0.8000 | 424345 | 4562557440 | 10769282048 |

Interpretation:

```text
7B static 20% is not a generally safe static model; WikiText-2 PPL and
multiple-choice accuracy degrade substantially.

This does not contradict the Real Tool result. It sharpens the claim:
adaptive_B20 is a dynamic Agent budget where 20% is reserved for final-answer
generations, while schema-sensitive tool-call/retry steps stay at 10% and
reflect recovery stays dense.
```

P5 compact subnet / real latency:

```text
First-stage materialization and equivalence scripts now exist:
scripts/materialize_compact_subnet.py
scripts/evaluate_compact_subnet_equivalence.py
scripts/benchmark_compact_subnet_latency.py
scripts/analyze_compact_width_alignment.py
scripts/benchmark_compact_mlp_micro.py
scripts/build_hardware_aligned_compact_plans.py

Materialize subnet_dense, subnet_10, and subnet_20.
Physically slice gate_proj rows, up_proj rows, and down_proj columns.
Check mask-hook vs compact equivalence before any latency claim.

Current compact equivalence status:

10% and 20% compact subnets preserve the planned active ratio and reach 100%
top-1 logits agreement with the corresponding mask-hook path on 100 prompts.
Greedy exact generation is treated as a diagnostic because small bf16 numeric
differences can drift later tokens.

Initial batch-1 naive latency is slower than dense and uses more peak memory.
This is a negative engineering result: it does not invalidate the active-FFN
cost result, but it blocks any wall-clock speedup claim until the runtime path
is optimized or replaced by a hardware-aligned implementation.

Hardware-aligned compact plans recover clear MLP-only speedups, especially for
the 20% subnet. However, fixed-decode full-model latency remains flat or
negative: aligned_0p10 gives only about 0.4% speedup at batch 4, and
aligned_0p20 is slower than dense. This freezes P5 as a partial systems result:
MLP-level speed potential exists, but the current HF/PyTorch full-model decode
path does not yet deliver meaningful wall-clock speedup.

Dynamic Agent episode latency runner is therefore deferred until a lower-level
kernel/runtime path, coarser block/group pruning, or more FFN-dominant workload
shows full-model speedup.
```

Only after this step can the project discuss real wall-clock speedup.

Theoretical subnet table command:

```bash
python scripts/summarize_subnet_theory.py \
  --config configs/agent_qwen2_5_3b_4090.yaml \
  --local-files-only \
  --plan outputs/agent_qwen2_5_3b_4090_low/substrate_v2_schema_nested/flap/budget_plan_0p10.json \
  --plan-name nested_0p10 \
  --plan outputs/agent_qwen2_5_3b_4090_low/substrate_v2_schema_nested/flap/budget_plan_0p20.json \
  --plan-name nested_0p20 \
  --output-json outputs/agent_qwen2_5_3b_4090_low/subnet_theory/p2c_subnet_theory.json \
  --output-md outputs/agent_qwen2_5_3b_4090_low/subnet_theory/p2c_subnet_theory.md
```

Do not do now:

```text
Do not run old independent 5% or uniform 20% 1000.
Do not keep tuning Qwen2.5-3B masks under the same setup.
Do not expand the local tool set.
Do not introduce TEAL or KV pruning yet.
Do not run more same-configuration 1000-task experiments.
Do not claim wall-clock speedup before compact subnet or kernel work.
```
