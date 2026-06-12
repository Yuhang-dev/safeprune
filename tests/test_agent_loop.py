import unittest

from safeprune.agent_loop import RouteOutcome, run_real_tool_episode, summarize_real_tool_rows
from safeprune.tool_env import validate_real_tool_task
from safeprune.tools import default_tool_registry


def _task(**overrides):
    row = {
        "task_id": "t1",
        "user_request": "Use calculator.",
        "tool": "calculator",
        "expected_answer": "5",
        "expected_tool": "calculator",
        "expected_arguments": {"expression": "2 + 3"},
        "max_steps": 6,
        "fault_schedule": [],
    }
    row.update(overrides)
    return validate_real_tool_task(row)


def _route(stage, event, _messages):
    selected = "dense_fallback" if event == "timeout" else stage
    ratio = 1.0 if selected == "dense_fallback" else 0.99
    return RouteOutcome(selected_stage=selected, reason="test", active_ffn_ratio=ratio)


class _SequenceGenerator:
    def __init__(self, outputs):
        self.outputs = list(outputs)

    def __call__(self, _messages):
        if not self.outputs:
            return '{"type":"final","answer":"missing"}'
        return self.outputs.pop(0)


class AgentLoopTests(unittest.TestCase):
    def test_successful_tool_loop(self):
        row = run_real_tool_episode(
            task=_task(),
            registry=default_tool_registry(),
            route_fn=_route,
            generate_fn=_SequenceGenerator(
                [
                    '{"type":"tool_call","name":"calculator","arguments":{"expression":"2 + 3"}}',
                    '{"type":"final","answer":"5"}',
                ]
            ),
        )

        self.assertTrue(row["success"])
        self.assertEqual(row["generation_steps"], 2)
        self.assertEqual(row["tool_call_count"], 1)
        self.assertEqual(row["successful_tool_call_count"], 1)

    def test_schema_error_then_retry(self):
        row = run_real_tool_episode(
            task=_task(),
            registry=default_tool_registry(),
            route_fn=_route,
            generate_fn=_SequenceGenerator(
                [
                    '{"type":"tool_call","name":"calculator"}',
                    '{"type":"tool_call","name":"calculator","arguments":{"expression":"2 + 3"}}',
                    '{"type":"final","answer":"5"}',
                ]
            ),
        )

        self.assertTrue(row["success"])
        self.assertEqual(row["schema_error_count"], 1)
        self.assertEqual(row["routing_trace"][1]["stage"], "reflect")

    def test_timeout_triggers_dense_fallback_on_next_step(self):
        row = run_real_tool_episode(
            task=_task(
                fault_schedule=[
                    {
                        "tool": "calculator",
                        "attempt": 1,
                        "event": "timeout",
                        "retryable": True,
                    }
                ]
            ),
            registry=default_tool_registry(),
            route_fn=_route,
            generate_fn=_SequenceGenerator(
                [
                    '{"type":"tool_call","name":"calculator","arguments":{"expression":"2 + 3"}}',
                    '{"type":"tool_call","name":"calculator","arguments":{"expression":"2 + 3"}}',
                    '{"type":"final","answer":"5"}',
                ]
            ),
        )

        self.assertTrue(row["success"])
        self.assertEqual(row["dense_fallback_step_count"], 1)
        self.assertEqual(row["failure_events"], ["timeout"])

    def test_max_steps_exceeded(self):
        row = run_real_tool_episode(
            task=_task(max_steps=1),
            registry=default_tool_registry(),
            route_fn=_route,
            generate_fn=_SequenceGenerator(
                ['{"type":"tool_call","name":"calculator","arguments":{"expression":"2 + 3"}}']
            ),
        )

        self.assertFalse(row["success"])
        self.assertEqual(row["terminal_event"], "max_steps_exceeded")

    def test_summary_metrics(self):
        success = run_real_tool_episode(
            task=_task(),
            registry=default_tool_registry(),
            route_fn=_route,
            generate_fn=_SequenceGenerator(
                [
                    '{"type":"tool_call","name":"calculator","arguments":{"expression":"2 + 3"}}',
                    '{"type":"final","answer":"5"}',
                ]
            ),
        )
        failure = run_real_tool_episode(
            task=_task(max_steps=1),
            registry=default_tool_registry(),
            route_fn=_route,
            generate_fn=_SequenceGenerator(["not json"]),
        )

        metrics = summarize_real_tool_rows([success, failure])
        self.assertEqual(metrics["total"], 2)
        self.assertEqual(metrics["correct"], 1)
        self.assertAlmostEqual(metrics["task_success_rate"], 0.5)
        self.assertIsNotNone(metrics["cost_per_success"])
        self.assertLess(metrics["schema_validity_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
