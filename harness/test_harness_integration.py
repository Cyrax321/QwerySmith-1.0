#!/usr/bin/env python3
"""
tests/test_harness_integration.py -- Demonstrates & tests using AgentMemoryEngine
as a standalone memory layer in benchmark and evaluation harnesses (e.g. SParC, CoSQL, Spider).
"""

import time
import unittest
from memory import AgentMemoryEngine


class TestHarnessMemoryIntegration(unittest.TestCase):
    """Verifies how an evaluation harness hooks into AgentMemoryEngine."""

    def setUp(self):
        # In evaluation harnesses, isolated in-memory or ephemeral databases are common
        self.mem = AgentMemoryEngine(":memory:")

    def tearDown(self):
        self.mem.close()

    def test_harness_spar_multi_turn_simulation(self):
        """
        Simulates an evaluation harness processing a multi-turn conversation
        (like SParC or CoSQL benchmarks) where subsequent questions depend on prior context.
        """
        dialogue = [
            {
                "turn": 1,
                "question": "Which employees work in the Sales department?",
                "gold_sql": "SELECT EmployeeId, FirstName, LastName FROM Employee WHERE Title LIKE '%Sales%';",
                "sample_cols": ["EmployeeId", "FirstName", "LastName"],
                "sample_rows": [(1, "Nancy", "Edwards"), (2, "Jane", "Peacock")],
            },
            {
                "turn": 2,
                "question": "And which of them were hired after 2010?",
                "gold_sql": "SELECT EmployeeId, FirstName, LastName FROM Employee WHERE Title LIKE '%Sales%' AND HireDate > '2010-01-01';",
                "sample_cols": ["EmployeeId", "FirstName", "LastName"],
                "sample_rows": [(2, "Jane", "Peacock")],
            },
            {
                "turn": 3,
                "question": "What is their total salary?",
                "gold_sql": "SELECT SUM(Salary) FROM Employee WHERE Title LIKE '%Sales%' AND HireDate > '2010-01-01';",
                "sample_cols": ["TotalSalary"],
                "sample_rows": [(75000,)],
            }
        ]

        session_id = "eval_harness_dialogue_01"
        db_name = "chinook.db"

        # Turn 1: Process and store
        t1 = dialogue[0]
        res1 = self.mem.recall(session_id, db_name, t1["question"])
        self.assertFalse(res1.is_followup)
        self.mem.commit(
            session_id=session_id,
            db_name=db_name,
            question=t1["question"],
            sql=t1["gold_sql"],
            columns=t1["sample_cols"],
            rows=t1["sample_rows"],
            success=True,
        )

        # Turn 2: Harness asserts follow-up resolution
        t2 = dialogue[1]
        t0 = time.perf_counter()
        res2 = self.mem.recall(session_id, db_name, t2["question"])
        recall_latency = (time.perf_counter() - t0) * 1000

        self.assertTrue(res2.is_followup, "Harness should detect 'And which of them' as follow-up")
        self.assertIsNotNone(res2.previous_turn)
        self.assertEqual(res2.previous_turn.question, t1["question"])
        self.assertIn("Nancy", res2.prompt_context)
        self.assertLess(recall_latency, 1.0, f"Harness recall exceeded 1ms: {recall_latency}ms")

        self.mem.commit(
            session_id=session_id,
            db_name=db_name,
            question=t2["question"],
            sql=t2["gold_sql"],
            columns=t2["sample_cols"],
            rows=t2["sample_rows"],
            success=True,
        )

        # Turn 3: Harness asserts chaining
        t3 = dialogue[2]
        res3 = self.mem.recall(session_id, db_name, t3["question"])
        self.assertTrue(res3.is_followup, "Harness should detect 'their total salary' as follow-up")
        self.assertEqual(res3.previous_turn.question, t2["question"])

    def test_harness_few_shot_exemplar_seeding(self):
        """
        Simulates pre-seeding the memory engine with Spider/BIRD training split exemplars,
        then evaluating few-shot recall on test questions.
        """
        training_exemplars = [
            {
                "db_name": "company.db",
                "question": "What is the average employee salary by department?",
                "sql": "SELECT department_id, AVG(salary) FROM employees GROUP BY department_id;",
            },
            {
                "db_name": "company.db",
                "question": "Find the highest spending customer in California",
                "sql": "SELECT c.id, c.name, SUM(o.total) AS spent FROM customers c JOIN orders o ON c.id = o.customer_id WHERE c.state = 'CA' GROUP BY c.id ORDER BY spent DESC LIMIT 1;",
            },
        ]

        # Harness pre-seeds memory
        imported_count = self.mem.import_dataset(training_exemplars)
        self.assertEqual(imported_count, 2)

        # Test query from test split
        test_q = "List the average salary per department"
        rec = self.mem.recall("test_sess", "company.db", test_q)

        self.assertGreater(len(rec.exemplars), 0)
        self.assertEqual(rec.exemplars[0].sql, training_exemplars[0]["sql"])
        self.assertIn("AVG(salary)", rec.prompt_context)


if __name__ == "__main__":
    unittest.main()
