#!/usr/bin/env python3
import tempfile
import unittest
from pathlib import Path
from harness.benchmark import BenchmarkSummary, ItemEvaluation


class TestBenchmarkExport(unittest.TestCase):
    def test_csv_export(self):
        summary = BenchmarkSummary(
            total_items=1,
            valid_sql_count=1,
            valid_sql_rate=1.0,
            execution_match_count=1,
            execution_accuracy=1.0,
            exact_match_count=1,
            exact_match_rate=1.0,
            repairs_attempted=0,
            repairs_succeeded=0,
            repair_recovery_rate=1.0,
            avg_latency_ms=10.5,
            item_results=[
                ItemEvaluation(
                    item_id="item_01",
                    question="Select all items",
                    gold_sql="SELECT * FROM items;",
                    pred_sql="SELECT * FROM items;",
                    valid_sql=True,
                    execution_match=True,
                    exact_match=True,
                    was_repaired=False,
                    latency_ms=10.5,
                )
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_file = Path(tmpdir) / "eval.csv"
            summary.to_csv(str(csv_file))
            self.assertTrue(csv_file.exists())
            text = csv_file.read_text()
            self.assertIn("item_01", text)
            self.assertIn("Select all items", text)


if __name__ == "__main__":
    unittest.main()
