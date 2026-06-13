import unittest
import types
from unittest.mock import patch

from safeprune.agent_loop import summarize_real_tool_rows
from safeprune.hidden_state_router import HiddenStateCentroidRouter
from scripts.evaluate_real_tool_loop import (
    DEFAULT_REAL_TOOL_METHODS,
    REAL_TOOL_METHODS,
    RoutingPolicy,
    _MethodRuntime,
    _build_pairwise,
    _infer_substrate_name,
    _register_substrate_specs,
)


def _plan(pruned_count=0):
    return {
        "layers": [
            {
                "layer": 0,
                "num_mlp_channels": 100,
                "pruned_mlp_channels": list(range(pruned_count)),
            }
        ]
    }


def _runtime(policy):
    runtime = _MethodRuntime.__new__(_MethodRuntime)
    runtime.model = None
    runtime.tokenizer = None
    runtime.hidden_router = None
    runtime.mask_handle = None
    runtime.failure_router = None
    runtime.spec = {
        "mode": "routing_policy",
        "policy": policy,
        "stage_plans": {
            "plan": _plan(1),
            "observe": _plan(4),
            "reflect": _plan(2),
            "answer": _plan(3),
        },
        "observe_plan": _plan(4),
        "fallback_plan": _plan(0),
        "default_stage": "answer",
        "failure_events": ["timeout", "tool_error", "empty_observation", "premature_final"],
        "recovery_window_steps": 2,
    }
    runtime._set_plan = lambda _plan: None
    return runtime


def _hidden_runtime(*, reflect_mode="stage", event_override=False):
    runtime = _MethodRuntime.__new__(_MethodRuntime)
    runtime.model = None
    runtime.tokenizer = None
    runtime.mask_handle = None
    runtime.failure_router = None
    runtime.hidden_router = HiddenStateCentroidRouter.from_vectors(
        {
            "reflect": [[1.0, 0.0]],
            "answer": [[0.0, 1.0]],
        }
    )
    runtime.spec = {
        "mode": "hidden_state_centroid",
        "stage_plans": {
            "plan": _plan(1),
            "observe": _plan(4),
            "reflect": _plan(2),
            "answer": _plan(3),
        },
        "identity_plan": _plan(0),
        "fallback_plan": _plan(0),
        "reflect_mode": reflect_mode,
        "event_override": event_override,
        "router_prefill_cost": 0.25,
        "default_stage": "answer",
    }
    runtime._set_plan = lambda _plan: None
    return runtime


class EvaluateRealToolLoopTests(unittest.TestCase):
    def test_default_methods_include_bypass_methods_but_not_hidden_centroid(self):
        self.assertIn(
            "observe_failure_redense_global_balanced_approx_0.01",
            DEFAULT_REAL_TOOL_METHODS,
        )
        self.assertIn(
            "stage_reflect_dense_global_balanced_approx_0.01",
            DEFAULT_REAL_TOOL_METHODS,
        )
        self.assertIn(
            "stage_reflect_observe_global_balanced_approx_0.01",
            DEFAULT_REAL_TOOL_METHODS,
        )
        self.assertNotIn(
            "hidden_state_centroid_global_balanced_approx_0.01",
            DEFAULT_REAL_TOOL_METHODS,
        )
        self.assertIn(
            "hidden_state_centroid_reflect_dense_global_balanced_approx_0.01",
            REAL_TOOL_METHODS,
        )
        self.assertIn(
            "hidden_state_centroid_event_reflect_dense_global_balanced_approx_0.01",
            REAL_TOOL_METHODS,
        )

    def test_observe_failure_redense_triggers_dense_fallback_after_failure(self):
        runtime = _runtime(RoutingPolicy("observe", "observe", "dense"))

        route = runtime.route("reflect", "timeout", [])

        self.assertEqual(route.selected_stage, "dense_fallback")
        self.assertEqual(route.selected_plan_name, "dense_fallback")
        self.assertEqual(route.metadata["fallback_remaining_before"], 0)
        self.assertEqual(route.metadata["fallback_remaining_after"], 2)

    def test_stage_reflect_dense_uses_dense_only_for_reflect(self):
        runtime = _runtime(RoutingPolicy("stage", "dense", "normal"))

        plan_route = runtime.route("plan", "start", [])
        reflect_route = runtime.route("reflect", "timeout", [])

        self.assertEqual(plan_route.selected_stage, "plan")
        self.assertEqual(plan_route.selected_plan_name, "plan")
        self.assertEqual(reflect_route.selected_stage, "dense_fallback")
        self.assertEqual(reflect_route.selected_plan_name, "reflect_dense")
        self.assertAlmostEqual(reflect_route.active_ffn_ratio, 1.0)

    def test_stage_reflect_observe_uses_observe_plan_for_reflect(self):
        runtime = _runtime(RoutingPolicy("stage", "observe", "normal"))

        route = runtime.route("reflect", "timeout", [])

        self.assertEqual(route.selected_stage, "observe")
        self.assertEqual(route.selected_plan_name, "observe")
        self.assertAlmostEqual(route.active_ffn_ratio, 0.96)

    def test_substrate_name_infers_method_suffix_from_plan_path(self):
        self.assertEqual(
            _infer_substrate_name("outputs/substrate_v2/flap/budget_plan_0p05.json"),
            "flap_0p05",
        )

    def test_substrate_stage_reflect_dense_uses_sparse_plan_then_dense_reflect(self):
        substrate_plan = _plan(10)
        substrate_plan["substrate_method"] = "flap"
        substrate_plan["global_target"] = 0.10
        substrate_plan["budget_plan"] = {"actual_sparsity": 0.10}
        substrate_plan["layers"][0]["mlp_output_bias_compensation"] = [0.0]
        substrate_plan["layers"][0]["mlp_output_scale"] = 1.0
        specs = {}
        config = types.SimpleNamespace(
            agent=types.SimpleNamespace(
                stages=["plan", "observe", "reflect", "answer"],
                failure_events=["timeout"],
                recovery_window_steps=2,
            )
        )
        _register_substrate_specs(
            specs,
            substrate_plans=[("flap_0p10", substrate_plan, "plan.json")],
            identity_plan=_plan(0),
            config=config,
        )
        runtime = _MethodRuntime.__new__(_MethodRuntime)
        runtime.model = None
        runtime.tokenizer = None
        runtime.hidden_router = None
        runtime.mask_handle = None
        runtime.failure_router = None
        runtime.spec = specs["substrate_flap_0p10_stage_reflect_dense"]
        runtime._set_plan = lambda _plan: None

        plan_route = runtime.route("plan", "start", [])
        reflect_route = runtime.route("reflect", "timeout", [])

        self.assertEqual(plan_route.selected_plan_name, "flap_0p10")
        self.assertAlmostEqual(plan_route.active_ffn_ratio, 0.9)
        self.assertTrue(plan_route.metadata["bias_compensation_enabled"])
        self.assertTrue(plan_route.metadata["layer_scale_enabled"])
        self.assertEqual(reflect_route.selected_stage, "dense_fallback")
        self.assertAlmostEqual(reflect_route.active_ffn_ratio, 1.0)

    def test_generation_type_adaptive_policy_uses_tool_answer_and_reflect_plans(self):
        def substrate_plan(target, pruned):
            plan = _plan(pruned)
            plan["substrate_method"] = "flap"
            plan["global_target"] = target
            plan["budget_plan"] = {"global_target": target, "actual_sparsity": target}
            return plan

        specs = {}
        config = types.SimpleNamespace(
            agent=types.SimpleNamespace(
                stages=["plan", "observe", "reflect", "answer"],
                failure_events=["timeout"],
                recovery_window_steps=2,
            )
        )
        _register_substrate_specs(
            specs,
            substrate_plans=[
                ("flap_0p05", substrate_plan(0.05, 5), "p05.json"),
                ("flap_0p10", substrate_plan(0.10, 10), "p10.json"),
                ("flap_0p15", substrate_plan(0.15, 15), "p15.json"),
                ("flap_0p20", substrate_plan(0.20, 20), "p20.json"),
            ],
            identity_plan=_plan(0),
            config=config,
        )
        runtime = _MethodRuntime.__new__(_MethodRuntime)
        runtime.model = None
        runtime.tokenizer = None
        runtime.hidden_router = None
        runtime.mask_handle = None
        runtime.failure_router = None
        runtime.spec = specs["adaptive_A"]
        runtime._set_plan = lambda _plan: None

        tool_route = runtime.route("plan", "start", [], "tool_call_or_retry")
        answer_route = runtime.route("answer", "ok", [], "final_answer")
        reflect_route = runtime.route("reflect", "timeout", [], "reflect_recovery")

        self.assertEqual(tool_route.selected_plan_name, "flap_0p05")
        self.assertEqual(tool_route.metadata["generation_type"], "tool_call_or_retry")
        self.assertAlmostEqual(tool_route.active_ffn_ratio, 0.95)
        self.assertEqual(answer_route.selected_plan_name, "flap_0p15")
        self.assertAlmostEqual(answer_route.active_ffn_ratio, 0.85)
        self.assertEqual(reflect_route.selected_stage, "dense_fallback")
        self.assertAlmostEqual(reflect_route.active_ffn_ratio, 1.0)

    def test_hidden_centroid_can_use_sparse_reflect_plan(self):
        runtime = _hidden_runtime(reflect_mode="stage")

        with patch("scripts.evaluate_real_tool_loop._last_hidden_vector", return_value=[1.0, 0.0]):
            route = runtime.route("answer", "ok", [])

        self.assertEqual(route.selected_stage, "reflect")
        self.assertEqual(route.selected_plan_name, "hidden_reflect")
        self.assertAlmostEqual(route.active_ffn_ratio, 0.98)
        self.assertEqual(route.router_prefill_cost, 0.25)

    def test_hidden_centroid_reflect_dense_uses_dense_on_predicted_reflect(self):
        runtime = _hidden_runtime(reflect_mode="dense")

        with patch("scripts.evaluate_real_tool_loop._last_hidden_vector", return_value=[1.0, 0.0]):
            route = runtime.route("answer", "ok", [])

        self.assertEqual(route.selected_stage, "dense_fallback")
        self.assertEqual(route.selected_plan_name, "hidden_reflect_dense")
        self.assertAlmostEqual(route.active_ffn_ratio, 1.0)
        self.assertTrue(route.metadata["centroid_reflect_predicted"])

    def test_hidden_centroid_event_hybrid_uses_event_override(self):
        runtime = _hidden_runtime(reflect_mode="dense", event_override=True)

        with patch("scripts.evaluate_real_tool_loop._last_hidden_vector", return_value=[0.0, 1.0]):
            route = runtime.route("reflect", "timeout", [])

        self.assertEqual(route.selected_stage, "dense_fallback")
        self.assertEqual(route.selected_plan_name, "hidden_event_reflect_dense")
        self.assertEqual(route.metadata["centroid_predicted_stage"], "answer")
        self.assertTrue(route.metadata["event_override"])

    def test_hidden_router_metrics_measure_detection_and_probe_cost(self):
        rows = [
            {
                "task_id": "t1",
                "tool": "calculator",
                "success": True,
                "collapse": False,
                "failure_task": True,
                "generation_steps": 2,
                "task_active_ffn_cost": 2.0,
                "router_prefill_cost": 0.5,
                "tool_call_count": 1,
                "valid_tool_call_count": 1,
                "successful_tool_call_count": 1,
                "schema_error_count": 0,
                "parse_error_count": 0,
                "dense_fallback_step_count": 2,
                "premature_final_count": 0,
                "reflect_expected_count": 1,
                "reflect_actual_count": 1,
                "entered_reflect": True,
                "recovery_steps": 2,
                "has_successful_tool_observation": True,
                "routing_trace": [
                    {
                        "stage": "reflect",
                        "selected_stage": "dense_fallback",
                        "route_metadata": {
                            "centroid_predicted_stage": "reflect",
                            "centroid_reflect_predicted": True,
                            "routing_probe_latency_ms": 5.0,
                        },
                    },
                    {
                        "stage": "answer",
                        "selected_stage": "dense_fallback",
                        "route_metadata": {
                            "centroid_predicted_stage": "reflect",
                            "centroid_reflect_predicted": True,
                            "routing_probe_latency_ms": 7.0,
                        },
                    },
                ],
            }
        ]

        metrics = summarize_real_tool_rows(rows)

        self.assertEqual(metrics["routing_probe_cost"], 0.5)
        self.assertEqual(metrics["effective_cost_per_success"], 2.5)
        self.assertEqual(metrics["routing_probe_latency_ms"], 12.0)
        self.assertEqual(metrics["reflect_detection_precision"], 0.5)
        self.assertEqual(metrics["reflect_detection_recall"], 1.0)
        self.assertEqual(metrics["raw_reflect_detection_recall"], 1.0)
        self.assertEqual(metrics["runner_reflect_transition_accuracy"], 1.0)
        self.assertEqual(metrics["effective_reflect_redense_recall"], 1.0)
        self.assertEqual(metrics["critical_reflect_miss_rate"], 0.0)
        self.assertEqual(metrics["false_dense_fallback_rate"], 1.0)
        self.assertEqual(metrics["missed_reflect_rate"], 0.0)

    def test_pairwise_counts_all_failure_and_non_failure_splits(self):
        rows = [
            {"method": "a", "task_id": "t1", "success": True, "failure_task": True},
            {"method": "b", "task_id": "t1", "success": False, "failure_task": True},
            {"method": "a", "task_id": "t2", "success": False, "failure_task": False},
            {"method": "b", "task_id": "t2", "success": True, "failure_task": False},
            {"method": "a", "task_id": "t3", "success": True, "failure_task": False},
            {"method": "b", "task_id": "t3", "success": True, "failure_task": False},
        ]

        pairwise = _build_pairwise(rows, ["a", "b"])["a_vs_b"]

        self.assertEqual(pairwise["all"]["both_correct"], 1)
        self.assertEqual(pairwise["all"]["left_only_correct"], 1)
        self.assertEqual(pairwise["all"]["right_only_correct"], 1)
        self.assertEqual(pairwise["failure"]["left_only_correct"], 1)
        self.assertEqual(pairwise["non_failure"]["right_only_correct"], 1)


if __name__ == "__main__":
    unittest.main()
