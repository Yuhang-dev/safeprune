import json
import tempfile
import unittest
from pathlib import Path

from safeprune.data import load_preference_jsonl, summarize_records, validate_record


class DataTests(unittest.TestCase):
    def test_validate_record_accepts_expected_schema(self):
        record = validate_record(
            {
                "prompt": "p",
                "chosen": "c",
                "rejected": "r",
                "source": "unit",
                "tag": "safety",
            }
        )
        self.assertEqual(record.tag, "safety")

    def test_load_preference_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "prompt": "p",
                        "chosen": "c",
                        "rejected": "r",
                        "source": "unit",
                        "tag": "general",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            records = load_preference_jsonl(path)
        self.assertEqual(
            summarize_records(records),
            {"total": 1, "helpfulness": 0, "safety": 0, "general": 1},
        )


if __name__ == "__main__":
    unittest.main()
