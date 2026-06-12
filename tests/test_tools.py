import unittest

from safeprune.tools import default_tool_registry


class ToolsTests(unittest.TestCase):
    def setUp(self):
        self.registry = default_tool_registry()

    def test_calculator_executes_safe_arithmetic(self):
        result = self.registry.execute("calculator", {"expression": "(2 + 3) * 4 / 2"})
        self.assertTrue(result.ok)
        self.assertEqual(result.output["result"], 10)

    def test_calculator_rejects_function_calls_and_names(self):
        result = self.registry.execute("calculator", {"expression": "__import__('os').system('x')"})
        self.assertFalse(result.ok)
        self.assertEqual(result.event, "invalid_argument")

        result = self.registry.execute("calculator", {"expression": "a + 1"})
        self.assertFalse(result.ok)
        self.assertEqual(result.event, "invalid_argument")

    def test_unit_convert(self):
        result = self.registry.execute(
            "unit_convert",
            {"value": 3, "from_unit": "m", "to_unit": "cm"},
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.output, {"result": 300, "unit": "cm"})

    def test_unit_convert_rejects_invalid_unit(self):
        result = self.registry.execute(
            "unit_convert",
            {"value": 3, "from_unit": "mile", "to_unit": "cm"},
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.event, "schema_error")

    def test_lookup(self):
        result = self.registry.execute("lookup", {"project": "PRJ-17"})
        self.assertTrue(result.ok)
        self.assertEqual(result.output["owner"], "owner_17")

    def test_lookup_rejects_unknown_project_format(self):
        result = self.registry.execute("lookup", {"project": "project_17"})
        self.assertFalse(result.ok)
        self.assertEqual(result.event, "invalid_argument")

    def test_registry_rejects_unknown_tool_and_extra_args(self):
        result = self.registry.execute("missing", {})
        self.assertFalse(result.ok)
        self.assertEqual(result.event, "unknown_tool")

        result = self.registry.execute("lookup", {"project": "PRJ-1", "extra": "bad"})
        self.assertFalse(result.ok)
        self.assertEqual(result.event, "schema_error")


if __name__ == "__main__":
    unittest.main()
