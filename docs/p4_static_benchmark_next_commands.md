# P4 Static Benchmark and Next Commands

This document is the current command sheet after freezing the Qwen2.5-7B
Real Tool protocol v1.2 result.

## Status

Do not rerun the same 7B Real Tool 1000 matrix. The frozen protocol v1.2
result is:

| Method | Success | Failure | Non-failure | Cost / success | Schema validity |
|---|---:|---:|---:|---:|---:|
| dense | 1000/1000 | 200/200 | 800/800 | 2.2390 | 1.0000 |
| identity_hook | 1000/1000 | 200/200 | 800/800 | 2.2390 | 1.0000 |
| nested_uniform_10 | 1000/1000 | 200/200 | 800/800 | 2.0000 | 1.0000 |
| adaptive_B20 | 1000/1000 | 200/200 | 800/800 | 1.9000 | 1.0000 |

`adaptive_B20` lowers theoretical active-FFN cost per successful episode by
about 15.1% versus dense. This is not a wall-clock speedup claim.

## 1. Remote Sync

```bash
cd /root/autodl-tmp/safeprune/code/Prune
source /root/autodl-tmp/safeprune/.venv_agent/bin/activate
git pull origin main
```

## 2. Check 7B Nested Plans

```bash
PLAN_DIR=outputs/agent_qwen2_5_7b/substrate_v2_schema_nested/flap

for p in \
  "$PLAN_DIR/budget_plan_0p10.json" \
  "$PLAN_DIR/budget_plan_0p15.json" \
  "$PLAN_DIR/budget_plan_0p20.json"
do
  test -f "$p" && echo "FOUND $p" || echo "MISSING $p"
done
```

If `budget_plan_0p15.json` is missing, recover it from the same 7B nested
ladder artifacts if possible. Do not silently mix a newly rebuilt 15% plan with
the already frozen 10%/20% Real Tool result.

If `budget_plan_0p15.json` is missing, continue the quick sanity with dense,
10%, and 20% only. The 15% point is useful but not required for the next
decision.

## 3. 7B Static Benchmark Quick Sanity

`evaluate_static_benchmarks.py` includes dense by default unless `--no-dense`
is passed. This run is only a quick sanity because `--max-samples 256` is too
small for final paper benchmark numbers.

```bash
python -u scripts/evaluate_static_benchmarks.py \
  --config configs/agent_qwen2_5_7b.yaml \
  --local-files-only \
  --calibration-path data/agent/controlled_tasks.jsonl \
  --calibration-path data/agent/schema_calibration_v1.jsonl \
  --plan outputs/agent_qwen2_5_7b/substrate_v2_schema_nested/flap/budget_plan_0p10.json \
  --plan-name nested_0p10 \
  --plan outputs/agent_qwen2_5_7b/substrate_v2_schema_nested/flap/budget_plan_0p20.json \
  --plan-name nested_0p20 \
  --benchmarks wikitext2 piqa hellaswag arc_easy \
  --max-samples 256 \
  --log-every 32 \
  --benchmark-version p4_7b_static_quick_v1 \
  --output-dir outputs/agent_qwen2_5_7b/static_benchmarks/p4_7b_static_quick_v1
```

View the table:

```bash
cat outputs/agent_qwen2_5_7b/static_benchmarks/p4_7b_static_quick_v1/static_benchmark_summary.md
```

If a verified `budget_plan_0p15.json` from the same 7B ladder is later found,
rerun the same command with:

```bash
  --plan outputs/agent_qwen2_5_7b/substrate_v2_schema_nested/flap/budget_plan_0p15.json \
  --plan-name nested_0p15 \
```

## 4. 7B Subnet Theory Table

```bash
mkdir -p outputs/agent_qwen2_5_7b/subnet_theory

python -u scripts/summarize_subnet_theory.py \
  --config configs/agent_qwen2_5_7b.yaml \
  --local-files-only \
  --plan outputs/agent_qwen2_5_7b/substrate_v2_schema_nested/flap/budget_plan_0p10.json \
  --plan-name nested_0p10 \
  --plan outputs/agent_qwen2_5_7b/substrate_v2_schema_nested/flap/budget_plan_0p20.json \
  --plan-name nested_0p20 \
  --output-json outputs/agent_qwen2_5_7b/subnet_theory/p4_7b_subnet_theory.json \
  --output-md outputs/agent_qwen2_5_7b/subnet_theory/p4_7b_subnet_theory.md
```

View the table:

```bash
cat outputs/agent_qwen2_5_7b/subnet_theory/p4_7b_subnet_theory.md
```

## 5. Results from 7B Quick Sanity

The 7B quick sanity has completed with dense, nested 10%, and nested 20%.
This is a 256-sample sanity run, not a final paper benchmark table.

| Plan | Active MLP | ARC-Easy | HellaSwag | PIQA | WikiText-2 PPL |
|---|---:|---:|---:|---:|---:|
| dense | 1.0000 | 0.7891 | 0.5820 | 0.8242 | 11.7030 |
| nested_0p10 | 0.9000 | 0.7773 | 0.5508 | 0.7656 | 15.5519 |
| nested_0p20 | 0.8000 | 0.7031 | 0.5273 | 0.7461 | 33.8450 |

The 7B subnet theory table is:

| Name | Active FFN | Remaining channels | FFN MACs/token | Total FLOPs/token |
|---|---:|---:|---:|---:|
| dense | 1.0000 | 530432 | 5703204864 | 13050576896 |
| nested_0p10 | 0.9000 | 477388 | 5132875776 | 11909918720 |
| nested_0p20 | 0.8000 | 424345 | 4562557440 | 10769282048 |

Interpretation:

```text
Nested 10% is usable as an Agent tool-call budget but already shows nontrivial
general-benchmark degradation on this quick sanity run.

Nested 20% is not a generally safe static compression setting: WikiText-2 PPL
and multiple-choice accuracy degrade substantially.

This strengthens the P2c claim that 20% should be treated as a dynamic
final-answer budget inside the Real Tool loop, not as a universal static model.
```

## 6. 3B Protocol v1.2 Alignment

Run this if the final paper table compares Qwen2.5-3B and Qwen2.5-7B directly.
Otherwise the 3B protocol v1 and 7B protocol v1.2 numbers must be clearly
marked as separate protocol versions.

```bash
python -u scripts/evaluate_real_tool_loop.py \
  --config configs/agent_qwen2_5_3b_4090.yaml \
  --tasks data/agent/real_tool_tasks_v1.jsonl \
  --mask-bank-path outputs/agent_qwen2_5_3b_4090_low/mask_bank/mask_bank.json \
  --local-files-only \
  --methods dense identity_hook nested_uniform_10 adaptive_B20 \
  --substrate-plan outputs/agent_qwen2_5_3b_4090_low/substrate_v2_schema_nested/flap/budget_plan_0p10.json \
  --substrate-name nested_0p10 \
  --substrate-plan outputs/agent_qwen2_5_3b_4090_low/substrate_v2_schema_nested/flap/budget_plan_0p20.json \
  --substrate-name nested_0p20 \
  --output-prefix outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_p2c_adaptive_1000_protocol_v1_2
```

Summary:

```bash
python - <<'PY'
import json
from pathlib import Path
from collections import defaultdict, Counter

prefix = "outputs/agent_qwen2_5_3b_4090_low/real_tool_eval/real_tool_v1_p2c_adaptive_1000_protocol_v1_2"
rows = [json.loads(x) for x in Path(prefix + ".jsonl").read_text().splitlines() if x.strip()]
metrics = json.loads(Path(prefix + "_metrics.json").read_text())

for method in ["dense", "identity_hook", "nested_uniform_10", "adaptive_B20"]:
    m = metrics[method]
    by_tool = defaultdict(lambda: [0, 0])
    events = Counter()
    for r in rows:
        if r["method"] != method:
            continue
        by_tool[r["tool"]][1] += 1
        by_tool[r["tool"]][0] += int(r["success"])
        for e in r.get("events") or []:
            events[e] += 1
    print("\n==", method, "==")
    print("success:", m["correct"], "/", m["total"])
    print("failure:", m["failure_task_correct"], "/", m["failure_task_total"])
    print("non_failure:", m["non_failure_task_correct"], "/", m["non_failure_task_total"])
    print("schema_validity:", m.get("schema_validity_rate"))
    print("premature_final_rate:", m.get("premature_final_rate"))
    print("fallback_step_ratio:", m.get("fallback_step_ratio"))
    print("cost_per_success:", m.get("cost_per_success"))
    print("active_ffn_cost:", m.get("active_ffn_cost"))
    print("by_tool:", {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_tool.items())})
    print("events:", dict(events))
PY
```

## Decision Rules

- Treat the 7B static benchmark as quick sanity only; expand to 1024 samples or
  full validation before using those numbers in the paper.
- If `nested_0p20` degrades on general benchmarks but `adaptive_B20` remains
  stable in Real Tool, describe 20% as a dynamic final-answer budget, not a
  universal static compression setting.
- Before P5, continue to claim only theoretical active-FFN cost reduction.
- P5 materialization and equivalence scripts exist for first-stage validation.
  Single-subnet latency smoke also exists. Dynamic Agent switching scripts are
  still future work.

## P5 First-Stage Compact Subnet Commands

P5 scripts now cover runtime materialization, mask-hook-vs-compact equivalence,
and single-subnet latency smoke. They do not implement dynamic Agent switching
yet.

Set the 7B cache first:

```bash
export HF_HOME=/root/autodl-tmp/safeprune/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/safeprune/hf_cache/hub
unset TRANSFORMERS_CACHE
```

Materialize metadata for 7B compact 10%:

```bash
python -u scripts/materialize_compact_subnet.py \
  --config configs/agent_qwen2_5_7b.yaml \
  --plan outputs/agent_qwen2_5_7b/substrate_v2_schema_nested/flap/budget_plan_0p10.json \
  --plan-name nested_0p10 \
  --local-files-only \
  --output-dir outputs/agent_qwen2_5_7b/compact_subnets/nested_0p10
```

Materialize metadata for 7B compact 20%:

```bash
python -u scripts/materialize_compact_subnet.py \
  --config configs/agent_qwen2_5_7b.yaml \
  --plan outputs/agent_qwen2_5_7b/substrate_v2_schema_nested/flap/budget_plan_0p20.json \
  --plan-name nested_0p20 \
  --local-files-only \
  --output-dir outputs/agent_qwen2_5_7b/compact_subnets/nested_0p20
```

Equivalence check for compact 10%:

```bash
python -u scripts/evaluate_compact_subnet_equivalence.py \
  --config configs/agent_qwen2_5_7b.yaml \
  --plan outputs/agent_qwen2_5_7b/substrate_v2_schema_nested/flap/budget_plan_0p10.json \
  --plan-name nested_0p10 \
  --prompts-jsonl data/agent/real_tool_tasks_v1_7b_min_smoke_20.jsonl \
  --max-prompts 8 \
  --max-new-tokens 16 \
  --local-files-only \
  --output outputs/agent_qwen2_5_7b/compact_subnets/nested_0p10/equivalence_8.json
```

Equivalence check for compact 20%:

```bash
python -u scripts/evaluate_compact_subnet_equivalence.py \
  --config configs/agent_qwen2_5_7b.yaml \
  --plan outputs/agent_qwen2_5_7b/substrate_v2_schema_nested/flap/budget_plan_0p20.json \
  --plan-name nested_0p20 \
  --prompts-jsonl data/agent/real_tool_tasks_v1_7b_min_smoke_20.jsonl \
  --max-prompts 8 \
  --max-new-tokens 16 \
  --local-files-only \
  --output outputs/agent_qwen2_5_7b/compact_subnets/nested_0p20/equivalence_8.json
```

If `data/agent/real_tool_tasks_v1_7b_min_smoke_20.jsonl` is absent, use
`data/agent/real_tool_tasks_v1_smoke_100.jsonl` with `--max-prompts 8`.

The expected first-stage acceptance is:

```text
active_mlp_ratio_actual matches the plan.
logits_mean_abs_diff is small.
top1_match_rate is high.
generation_exact_match_rate is high or failures are manually inspected.
```

Exact bitwise equality is not required because the compact path replaces
modules and applies compensation through a wrapper, but large logit or schema
behavior drift must be investigated before latency work.

## P5 Logits-Only Equivalence Result

The 100-prompt logits-only equivalence check has passed:

| Plan | Prompts | Active actual | Logits max diff | Logits mean diff | Top-1 match |
|---|---:|---:|---:|---:|---:|
| nested_0p10 | 100 | 0.899998 | 0.7260 | 0.0906 | 1.0000 |
| nested_0p20 | 100 | 0.799999 | 0.7500 | 0.1062 | 1.0000 |

This is enough to proceed to single-subnet latency smoke. Continue to treat
greedy exact match as a diagnostic, not the primary equivalence metric.

## P5 Single-Subnet Latency Smoke

Do not run this concurrently with any other GPU experiment.

```bash
python -u scripts/benchmark_compact_subnet_latency.py \
  --config configs/agent_qwen2_5_7b.yaml \
  --prompts-jsonl data/agent/real_tool_tasks_v1_7b_smoke_100.jsonl \
  --max-prompts 16 \
  --batch-sizes 1 \
  --decode-new-tokens 32 \
  --warmup-iters 2 \
  --measure-iters 5 \
  --local-files-only \
  --plan outputs/agent_qwen2_5_7b/substrate_v2_schema_nested/flap/budget_plan_0p10.json \
  --plan-name nested_0p10 \
  --plan outputs/agent_qwen2_5_7b/substrate_v2_schema_nested/flap/budget_plan_0p20.json \
  --plan-name nested_0p20 \
  --output-json outputs/agent_qwen2_5_7b/compact_subnets/latency/latency_smoke_b1.json \
  --output-md outputs/agent_qwen2_5_7b/compact_subnets/latency/latency_smoke_b1.md
```

If batch 1 is stable, optionally run batch 4:

```bash
python -u scripts/benchmark_compact_subnet_latency.py \
  --config configs/agent_qwen2_5_7b.yaml \
  --prompts-jsonl data/agent/real_tool_tasks_v1_7b_smoke_100.jsonl \
  --max-prompts 16 \
  --batch-sizes 4 \
  --decode-new-tokens 32 \
  --warmup-iters 2 \
  --measure-iters 5 \
  --local-files-only \
  --plan outputs/agent_qwen2_5_7b/substrate_v2_schema_nested/flap/budget_plan_0p10.json \
  --plan-name nested_0p10 \
  --plan outputs/agent_qwen2_5_7b/substrate_v2_schema_nested/flap/budget_plan_0p20.json \
  --plan-name nested_0p20 \
  --output-json outputs/agent_qwen2_5_7b/compact_subnets/latency/latency_smoke_b4.json \
  --output-md outputs/agent_qwen2_5_7b/compact_subnets/latency/latency_smoke_b4.md
```

Report these latency numbers only as single-subnet speed potential. They are
not Adaptive-B20 episode latency because dynamic switching overhead is not yet
measured.

## P5 Latency Smoke Result And Diagnostics

Initial batch-1 latency smoke did not show speedup:

| Model | Active MLP | Prefill ms | Decode ms/token | Tok/s | Peak GB |
|---|---:|---:|---:|---:|---:|
| dense | 1.0000 | 28.803 | 4.304 | 232.316 | 8.266 |
| nested_0p10 compact | 0.9000 | 41.532 | 6.403 | 156.184 | 14.093 |
| nested_0p20 compact | 0.8000 | 40.268 | 15.401 | 64.933 | 13.920 |

This is a negative P5 engineering result, not a speedup result. It means the
current runtime compact path is structurally valid but not performance-ready.
Do not run dynamic Adaptive-B20 latency until the single-subnet path is
understood.

### One-Variant-Per-Process Latency Checks

Dense only:

```bash
python -u scripts/benchmark_compact_subnet_latency.py \
  --config configs/agent_qwen2_5_7b.yaml \
  --variant dense \
  --prompts-jsonl data/agent/real_tool_tasks_v1_7b_smoke_100.jsonl \
  --max-prompts 16 \
  --batch-sizes 1 \
  --decode-new-tokens 32 \
  --warmup-iters 5 \
  --measure-iters 10 \
  --local-files-only \
  --output-json outputs/agent_qwen2_5_7b/compact_subnets/latency/latency_dense_only_b1.json \
  --output-md outputs/agent_qwen2_5_7b/compact_subnets/latency/latency_dense_only_b1.md
```

Compact 10 only:

```bash
python -u scripts/benchmark_compact_subnet_latency.py \
  --config configs/agent_qwen2_5_7b.yaml \
  --variant compact \
  --prompts-jsonl data/agent/real_tool_tasks_v1_7b_smoke_100.jsonl \
  --max-prompts 16 \
  --batch-sizes 1 \
  --decode-new-tokens 32 \
  --warmup-iters 5 \
  --measure-iters 10 \
  --local-files-only \
  --plan outputs/agent_qwen2_5_7b/substrate_v2_schema_nested/flap/budget_plan_0p10.json \
  --plan-name nested_0p10 \
  --output-json outputs/agent_qwen2_5_7b/compact_subnets/latency/latency_0p10_only_b1.json \
  --output-md outputs/agent_qwen2_5_7b/compact_subnets/latency/latency_0p10_only_b1.md
```

Compact 20 only:

```bash
python -u scripts/benchmark_compact_subnet_latency.py \
  --config configs/agent_qwen2_5_7b.yaml \
  --variant compact \
  --prompts-jsonl data/agent/real_tool_tasks_v1_7b_smoke_100.jsonl \
  --max-prompts 16 \
  --batch-sizes 1 \
  --decode-new-tokens 32 \
  --warmup-iters 5 \
  --measure-iters 10 \
  --local-files-only \
  --plan outputs/agent_qwen2_5_7b/substrate_v2_schema_nested/flap/budget_plan_0p20.json \
  --plan-name nested_0p20 \
  --output-json outputs/agent_qwen2_5_7b/compact_subnets/latency/latency_0p20_only_b1.json \
  --output-md outputs/agent_qwen2_5_7b/compact_subnets/latency/latency_0p20_only_b1.md
```

### Width Alignment Audit

```bash
python -u scripts/analyze_compact_width_alignment.py \
  --config configs/agent_qwen2_5_7b.yaml \
  --local-files-only \
  --plan outputs/agent_qwen2_5_7b/substrate_v2_schema_nested/flap/budget_plan_0p10.json \
  --plan-name nested_0p10 \
  --plan outputs/agent_qwen2_5_7b/substrate_v2_schema_nested/flap/budget_plan_0p20.json \
  --plan-name nested_0p20 \
  --output-json outputs/agent_qwen2_5_7b/compact_subnets/latency/width_alignment.json \
  --output-md outputs/agent_qwen2_5_7b/compact_subnets/latency/width_alignment.md
```

### MLP-Only Microbenchmark

```bash
python -u scripts/benchmark_compact_mlp_micro.py \
  --config configs/agent_qwen2_5_7b.yaml \
  --layers 0,14,27 \
  --shapes 1x1,1x128,4x128 \
  --warmup-iters 10 \
  --measure-iters 50 \
  --local-files-only \
  --plan outputs/agent_qwen2_5_7b/substrate_v2_schema_nested/flap/budget_plan_0p10.json \
  --plan-name nested_0p10 \
  --plan outputs/agent_qwen2_5_7b/substrate_v2_schema_nested/flap/budget_plan_0p20.json \
  --plan-name nested_0p20 \
  --output-json outputs/agent_qwen2_5_7b/compact_subnets/latency/mlp_micro.json \
  --output-md outputs/agent_qwen2_5_7b/compact_subnets/latency/mlp_micro.md
```

## P5 Hardware-Aligned Compact Plans

The naive compact plan uses irregular per-layer widths and is slower than dense
even in MLP-only microbenchmarks. The next P5 branch builds latency-oriented
plans with hardware-aligned remaining widths. These plans are separate from the
frozen P4 active-cost results.

Build aligned 10% / 20% plans:

```bash
python -u scripts/build_hardware_aligned_compact_plans.py \
  --config configs/agent_qwen2_5_7b.yaml \
  --scores-path outputs/agent_qwen2_5_7b/substrate_v2_schema_nested/flap/scores.json \
  --activation-stats-path outputs/agent_qwen2_5_7b/substrate_v2_schema_nested/ffn_activation_stats.json \
  --target-budgets 0.10,0.20 \
  --plan-prefix aligned \
  --align-to 256 \
  --min-remaining 4096 \
  --with-bias-compensation \
  --with-layer-scale-placeholder \
  --local-files-only \
  --output-dir outputs/agent_qwen2_5_7b/compact_subnets/hardware_aligned
```

If `ffn_activation_stats.json` is missing, run without bias compensation first
for latency diagnostics only:

```bash
python -u scripts/build_hardware_aligned_compact_plans.py \
  --config configs/agent_qwen2_5_7b.yaml \
  --scores-path outputs/agent_qwen2_5_7b/substrate_v2_schema_nested/flap/scores.json \
  --target-budgets 0.10,0.20 \
  --plan-prefix aligned \
  --align-to 256 \
  --min-remaining 4096 \
  --local-files-only \
  --output-dir outputs/agent_qwen2_5_7b/compact_subnets/hardware_aligned
```

Then audit the aligned plans:

```bash
python -u scripts/analyze_compact_width_alignment.py \
  --config configs/agent_qwen2_5_7b.yaml \
  --local-files-only \
  --plan outputs/agent_qwen2_5_7b/compact_subnets/hardware_aligned/budget_plan_aligned_0p10.json \
  --plan-name aligned_0p10 \
  --plan outputs/agent_qwen2_5_7b/compact_subnets/hardware_aligned/budget_plan_aligned_0p20.json \
  --plan-name aligned_0p20 \
  --output-json outputs/agent_qwen2_5_7b/compact_subnets/hardware_aligned/width_alignment.json \
  --output-md outputs/agent_qwen2_5_7b/compact_subnets/hardware_aligned/width_alignment.md
```

Only if aligned MLP-only latency improves should full-model latency be retried:

```bash
python -u scripts/benchmark_compact_mlp_micro.py \
  --config configs/agent_qwen2_5_7b.yaml \
  --layers 0,14,27 \
  --shapes 1x1,1x128,4x128 \
  --warmup-iters 10 \
  --measure-iters 50 \
  --local-files-only \
  --plan outputs/agent_qwen2_5_7b/compact_subnets/hardware_aligned/budget_plan_aligned_0p10.json \
  --plan-name aligned_0p10 \
  --plan outputs/agent_qwen2_5_7b/compact_subnets/hardware_aligned/budget_plan_aligned_0p20.json \
  --plan-name aligned_0p20 \
  --output-json outputs/agent_qwen2_5_7b/compact_subnets/hardware_aligned/mlp_micro.json \
  --output-md outputs/agent_qwen2_5_7b/compact_subnets/hardware_aligned/mlp_micro.md
```

Observed aligned MLP-only result:

| Plan | B1 S1 | B1 S128 | B4 S128 |
|---|---:|---:|---:|
| dense | 0.220 ms | 0.495 ms | 0.883 ms |
| aligned_0p10 | 0.477 ms | 0.585 ms | 1.401 ms |
| aligned_0p20 | 0.443 ms | 0.527 ms | 1.250 ms |

This improves over irregular compact but still does not beat dense. The
materializer now skips layers that have no pruned channels, no compensation, and
scale 1.0, avoiding wrapper overhead on dense-width layers. Re-run the aligned
MLP-only benchmark after pulling the commit before trying full-model latency.
