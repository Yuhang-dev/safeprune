import unittest

from safeprune.data import AgentStep
from safeprune.router import FFNMaskRouter


class RouterTests(unittest.TestCase):
    def test_router_selects_stage_default_sparsity(self):
        router = FFNMaskRouter(
            stage_sparsities={"plan": 0.1, "answer": 0.3},
            default_sparsity=0.2,
            failure_sparsity=0.1,
            failure_events=["tool_error"],
        )
        decision = router.route(AgentStep(stage="answer", text="final", event="ok"))
        self.assertEqual(decision.sparsity, 0.3)
        self.assertEqual(decision.reason, "stage_default")

    def test_router_redensifies_on_failure_event(self):
        router = FFNMaskRouter(
            stage_sparsities={"observe": 0.3, "answer": 0.3},
            default_sparsity=0.3,
            failure_sparsity=0.1,
            failure_events=["tool_error"],
            recovery_window_steps=1,
        )
        failure = router.route(AgentStep(stage="observe", text="bad", event="tool_error"))
        self.assertEqual(failure.sparsity, 0.1)
        self.assertTrue(failure.reason.startswith("failure_event"))
        cooldown = router.route(AgentStep(stage="answer", text="retry", event="ok"))
        self.assertEqual(cooldown.sparsity, 0.1)
        self.assertEqual(cooldown.reason, "recovery_window")
        normal = router.route(AgentStep(stage="answer", text="final", event="ok"))
        self.assertEqual(normal.sparsity, 0.3)


if __name__ == "__main__":
    unittest.main()
