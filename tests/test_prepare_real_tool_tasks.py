import unittest

from scripts.prepare_real_tool_tasks import build_tasks


class PrepareRealToolTasksTests(unittest.TestCase):
    def test_start_index_offsets_task_ids_and_arguments(self):
        rows = build_tasks(count=3, failure_count=1, start_index=1000)

        self.assertEqual(rows[0]["task_id"], "real_tool_1000")
        self.assertEqual(rows[1]["task_id"], "real_tool_1001")
        self.assertEqual(rows[2]["task_id"], "real_tool_1002")
        self.assertNotEqual(rows[0]["task_id"], "real_tool_0000")
        self.assertTrue(rows[0]["fault_schedule"])


if __name__ == "__main__":
    unittest.main()
