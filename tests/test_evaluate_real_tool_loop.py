import unittest

from scripts.evaluate_real_tool_loop import (
    DEFAULT_REAL_TOOL_METHODS,
    RoutingPolicy,
    _MethodRuntime,
    _build_pairwise,
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
