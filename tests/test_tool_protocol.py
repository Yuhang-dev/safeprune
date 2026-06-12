import unittest

from safeprune.tool_protocol import parse_agent_output


class ToolProtocolTests(unittest.TestCase):
    def test_parse_valid_tool_call(self):
        parsed = parse_agent_output(
            '{"type":"tool_call","name":"calculator","arguments":{"expression":"2 + 3"}}'
        )
        self.assertTrue(parsed.ok)
        self.assertEqual(parsed.action.type, "tool_call")
        self.assertEqual(parsed.action.name, "calculator")

    def test_parse_valid_final(self):
        parsed = parse_agent_output('{"type":"final","answer":"5"}')
        self.assertTrue(parsed.ok)
        self.assertEqual(parsed.action.type, "final")
        self.assertEqual(parsed.action.answer, "5")

    def test_reject_invalid_json(self):
        parsed = parse_agent_output("not json")
        self.assertFalse(parsed.ok)
        self.assertEqual(parsed.event, "format_error")

    def test_reject_extra_fields(self):
        parsed = parse_agent_output(
            '{"type":"final","answer":"5","extra":"bad"}'
        )
        self.assertFalse(parsed.ok)
        self.assertEqual(parsed.event, "schema_error")

    def test_reject_missing_fields(self):
        parsed = parse_agent_output('{"type":"tool_call","name":"calculator"}')
        self.assertFalse(parsed.ok)
        self.assertEqual(parsed.event, "schema_error")


if __name__ == "__main__":
    unittest.main()
