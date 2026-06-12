import unittest

from scripts.analyze_agent_pairwise_errors import _group_by_method, _pairwise_split


class AnalyzeAgentPairwiseErrorsTests(unittest.TestCase):
    def test_pairwise_split_uses_success_or_strict_correct(self):
        rows = [
            {"method": "a", "task_id": "t1", "strict_correct": True, "tool": "calculator"},
            {"method": "b", "task_id": "t1", "strict_correct": False, "tool": "calculator"},
            {"method": "a", "task_id": "t2", "success": False, "tool": "lookup"},
            {"method": "b", "task_id": "t2", "success": True, "tool": "lookup"},
        ]

        split = _pairwise_split(_group_by_method(rows), [("a", "b")], max_examples=2)[0]

        self.assertEqual(split["left_only"], 1)
        self.assertEqual(split["right_only"], 1)
        self.assertEqual(split["net_left_minus_right"], 0)


if __name__ == "__main__":
    unittest.main()
