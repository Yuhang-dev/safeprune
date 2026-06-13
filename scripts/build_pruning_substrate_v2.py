from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from safeprune.config import load_config
from safeprune.data import read_jsonl
from safeprune.modeling import load_causal_lm_and_tokenizer
from safeprune.pruning import ScoreWeights
from safeprune.scoring import (
    activation_scores_from_stats,
    compute_flap_fluctuation_scores,
    compute_ffn_activation_stats_for_prompts,
    compute_magnitude_scores,
    compute_wanda_channel_scores,
    save_activation_stats,
    save_plan,
    save_scores,
)
from safeprune.substrate import (
    add_bias_compensation_to_plan,
    add_layer_output_scales,
    build_budget_options,
    build_nested_global_budget_plans,
    build_plan_from_budget,
    compute_per_layer_loss_delta_matrix,
    optimize_layerwise_budget,
    validate_nested_pruned_sets,
)


DEFAULT_CANDIDATE_SPARSITIES = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30]
DEFAULT_TARGET_BUDGETS = [0.01, 0.03, 0.05, 0.10, 0.20]
DEFAULT_SCORE_METHODS = ["activation", "wanda", "flap"]


def _parse_float_list(raw: str) -> list[float]:
    return [float(item) for item in raw.split(",") if item.strip()]


def _budget_slug(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _load_prompt_rows(path: str | Path, limit: int | None = None) -> list[str]:
    rows = read_jsonl(path)
    prompts = []
    for row in rows:
        prompt = _prompt_from_row(row)
        if prompt:
            prompts.append(prompt)
        if limit is not None and len(prompts) >= limit:
            break
    if not prompts:
        raise ValueError(f"No calibration prompts found in {path}")
    return prompts


def _prompt_from_row(row: dict) -> str | None:
    assistant_target = row.get("assistant_target")
    for key in ["prompt", "user_request", "text"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            if isinstance(assistant_target, str) and assistant_target.strip():
                return f"{value}\nAssistant target:\n{assistant_target}"
            return value
    chosen = row.get("chosen")
    if isinstance(chosen, str) and chosen.strip():
        prompt = row.get("prompt")
        return f"{prompt}\n{chosen}" if isinstance(prompt, str) else chosen
    steps = row.get("steps")
    if isinstance(steps, list):
        texts = [
            step.get("text", "")
            for step in steps
            if isinstance(step, dict) and isinstance(step.get("text"), str)
        ]
        if texts:
            return "\n".join(texts)
    return None


def _score_methods_payload(model, stats, methods: list[str]):
    magnitude_scores = compute_magnitude_scores(model)
    payload = {}
    if "activation" in methods:
        payload["activation"] = activation_scores_from_stats(stats)
    if "magnitude" in methods:
        payload["magnitude"] = magnitude_scores
    if "wanda" in methods:
        payload["wanda"] = compute_wanda_channel_scores(model, stats)
    if "flap" in methods:
        payload["flap"] = compute_flap_fluctuation_scores(model, stats)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Pruning Substrate v2 budget plans.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--calibration-path")
    parser.add_argument("--max-calibration-prompts", type=int, default=512)
    parser.add_argument("--candidate-sparsities", default=",".join(map(str, DEFAULT_CANDIDATE_SPARSITIES)))
    parser.add_argument("--target-budgets", default=",".join(map(str, DEFAULT_TARGET_BUDGETS)))
    parser.add_argument(
        "--score-methods",
        nargs="+",
        default=DEFAULT_SCORE_METHODS,
        choices=["activation", "magnitude", "wanda", "flap"],
    )
    parser.add_argument("--with-loss-delta", action="store_true")
    parser.add_argument("--loss-delta-prompts", type=int, default=32)
    parser.add_argument("--nested-budget-ladder", action="store_true")
    parser.add_argument("--schema-calibration-path")
    parser.add_argument("--max-schema-calibration-prompts", type=int, default=512)
    parser.add_argument("--schema-token-weight", type=float, default=1.0)
    parser.add_argument("--with-bias-compensation", action="store_true")
    parser.add_argument("--with-layer-scale-placeholder", action="store_true")
    parser.add_argument("--skip-model-load", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = Path(args.output_dir or Path(config.pruning.scores_dir) / "substrate_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_sparsities = _parse_float_list(args.candidate_sparsities)
    target_budgets = _parse_float_list(args.target_budgets)
    calibration_path = args.calibration_path or config.data.calibration or config.agent.calibration_path
    if calibration_path is None:
        raise ValueError("A calibration path is required.")

    manifest = {
        "config": args.config,
        "calibration_path": calibration_path,
        "candidate_sparsities": candidate_sparsities,
        "target_budgets": target_budgets,
        "score_methods": args.score_methods,
        "with_loss_delta": args.with_loss_delta,
        "nested_budget_ladder": args.nested_budget_ladder,
        "schema_calibration_path": args.schema_calibration_path,
        "schema_token_weight": args.schema_token_weight,
        "schema_weighting_note": (
            "v1 uses schema-heavy assistant-target calibration snippets; "
            "token-level weighting interface is reserved by --schema-token-weight."
        ),
        "with_bias_compensation": args.with_bias_compensation,
        "with_layer_scale_placeholder": args.with_layer_scale_placeholder,
        "note": "Mask-hook substrate plans; not physical speedup evidence.",
    }

    if args.skip_model_load:
        manifest["status"] = "planned"
        (out_dir / "substrate_v2_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote substrate v2 manifest to {out_dir / 'substrate_v2_manifest.json'}")
        return

    prompts = _load_prompt_rows(calibration_path, limit=args.max_calibration_prompts)
    if args.schema_calibration_path:
        schema_prompts = _load_prompt_rows(
            args.schema_calibration_path,
            limit=args.max_schema_calibration_prompts,
        )
        prompts = [*prompts, *schema_prompts]
        manifest["schema_calibration_count"] = len(schema_prompts)
    model, tokenizer = load_causal_lm_and_tokenizer(
        config.model.base_model,
        config.model,
        local_files_only=args.local_files_only,
    )
    stats = compute_ffn_activation_stats_for_prompts(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_length=config.data.max_length,
    )
    save_activation_stats(stats, out_dir / "ffn_activation_stats.json")

    scores_by_method = _score_methods_payload(model, stats, args.score_methods)
    weights = ScoreWeights(magnitude=1.0, activation=0.0, loss_delta=0.0)
    loss_delta_by_method = {}
    if args.with_loss_delta:
        loss_prompts = prompts[: args.loss_delta_prompts]
        for method, scores in scores_by_method.items():
            loss_delta_by_method[method] = compute_per_layer_loss_delta_matrix(
                model=model,
                tokenizer=tokenizer,
                scores=scores,
                prompts=loss_prompts,
                candidate_sparsities=candidate_sparsities,
                max_length=config.data.max_length,
                weights=weights,
                min_mlp_channels_per_layer=config.pruning.min_mlp_channels_per_layer,
            )
        (out_dir / "loss_delta_matrix.json").write_text(
            json.dumps(loss_delta_by_method, indent=2),
            encoding="utf-8",
        )

    plan_index = {}
    mask_bank_payload = {
        "format": "safeprune.substrate_v2_mask_bank.v1",
        "note": "Mask-hook substrate v2 budget plans; not physical speedup evidence.",
        "plans": {},
    }
    for method, scores in scores_by_method.items():
        method_dir = out_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        save_scores(scores, method_dir / "scores.json")
        mask_bank_payload["plans"][method] = {}
        plan_index[method] = {}
        if args.nested_budget_ladder:
            plans_by_target = build_nested_global_budget_plans(
                scores=scores,
                target_budgets=target_budgets,
                weights=weights,
                min_mlp_channels_per_layer=config.pruning.min_mlp_channels_per_layer,
                plan_name_prefix=method,
            )
            nested_validation = validate_nested_pruned_sets(plans_by_target)
        else:
            options = build_budget_options(
                scores=scores,
                candidate_sparsities=candidate_sparsities,
                min_mlp_channels_per_layer=config.pruning.min_mlp_channels_per_layer,
                loss_delta_by_layer=loss_delta_by_method.get(method),
            )
            plans_by_target = {}
            for target in target_budgets:
                budget = optimize_layerwise_budget(options, target)
                plan = build_plan_from_budget(
                    scores=scores,
                    budget=budget,
                    weights=weights,
                    min_mlp_channels_per_layer=config.pruning.min_mlp_channels_per_layer,
                    plan_name=f"{method}_{_budget_slug(target)}",
                )
                plan["budget_plan"] = budget.to_dict()
                plans_by_target[target] = plan
            nested_validation = None

        audit_payload = {}
        for target in target_budgets:
            plan = plans_by_target[float(target)]
            plan["substrate_method"] = method
            plan["nested_budget_ladder"] = bool(args.nested_budget_ladder)
            if args.with_bias_compensation:
                plan = add_bias_compensation_to_plan(model, stats, plan)
            if args.with_layer_scale_placeholder:
                plan = add_layer_output_scales(
                    plan,
                    {int(item.layer): 1.0 for item in scores},
                )
            plan_path = method_dir / f"budget_plan_{_budget_slug(target)}.json"
            save_plan(plan, plan_path)
            plan_index[method][str(target)] = str(plan_path)
            mask_bank_payload["plans"][method][_budget_slug(target)] = plan
            audit_payload[f"{method}_{_budget_slug(target)}"] = plan

        if args.nested_budget_ladder:
            from scripts.audit_substrate_plans import audit_plans, to_markdown

            audit = audit_plans(audit_payload)
            audit["nested_validation"] = nested_validation
            (method_dir / "nested_budget_audit.json").write_text(
                json.dumps(audit, indent=2),
                encoding="utf-8",
            )
            (method_dir / "nested_budget_audit.md").write_text(
                to_markdown(audit),
                encoding="utf-8",
            )

    manifest["status"] = "complete"
    manifest["plans"] = plan_index
    (out_dir / "mask_bank_substrate_v2.json").write_text(
        json.dumps(mask_bank_payload, indent=2),
        encoding="utf-8",
    )
    (out_dir / "substrate_v2_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote substrate v2 artifacts to {out_dir}")


if __name__ == "__main__":
    main()
