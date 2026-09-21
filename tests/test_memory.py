#!/usr/bin/env python3
"""
tests/test_memory.py -- Comprehensive Unit & Benchmark Suite for QwerySmith AgentMemoryEngine
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from harness import AgentMemoryEngine, is_followup_question


class TestAgentMemoryEngine(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_memory.sqlite"
        self.mem = AgentMemoryEngine(self.db_path)

    def tearDown(self):
        self.mem.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_latency_sub_millisecond(self):
        """Verify that P95 read/write latency is strictly sub-millisecond (< 1.5ms)."""
        bench = self.mem.benchmark_latency(n_queries=100)
        self.assertLess(bench["read_p50_ms"], 1.0, f"Read P50 latency exceeded 1ms: {bench['read_p50_ms']}ms")
        self.assertLess(bench["read_p95_ms"], 5.0, f"Read P95 latency exceeded 5.0ms: {bench['read_p95_ms']}ms")
        self.assertLess(bench["write_p50_ms"], 1.0, f"Write P50 latency exceeded 1ms: {bench['write_p50_ms']}ms")

    def test_followup_question_heuristic(self):
        """Verify pronoun and conjunction follow-up detection."""
        self.assertTrue(is_followup_question("And how much did they spend?"))
        self.assertTrue(is_followup_question("What about those customers?"))
        self.assertTrue(is_followup_question("Show me their orders"))
        self.assertTrue(is_followup_question("Which of them are active?"))
        self.assertTrue(is_followup_question("How much did it cost?"))
        self.assertTrue(is_followup_question("And who was the manager?"))

        # Independent queries
        self.assertFalse(is_followup_question("List all tracks with genre Metal"))
        self.assertFalse(is_followup_question("What is the revenue for 2024?"))
        self.assertFalse(is_followup_question("Who are the top 5 artists?"))

    def test_multi_turn_followup_recall(self):
        """Verify multi-turn conversational context injection."""
        session_id = "sess_001"
        db_name = "chinook.db"

        # Turn 1: Independent query
        self.mem.commit(
            session_id=session_id,
            db_name=db_name,
            question="Which customers are from Brazil?",
            sql="SELECT CustomerId, FirstName, LastName, Country FROM Customer WHERE Country = 'Brazil';",
            columns=["CustomerId", "FirstName", "LastName", "Country"],
            rows=[(1, "Luís", "Gonçalves", "Brazil"), (10, "Eduardo", "Martins", "Brazil")],
            human_summary="Found 2 customers from Brazil.",
            success=True,
        )

        # Turn 2: Follow-up query
        rec = self.mem.recall(session_id, db_name, "And how much did they spend?")
        self.assertTrue(rec.is_followup)
        self.assertIsNotNone(rec.previous_turn)
        self.assertEqual(rec.previous_turn.question, "Which customers are from Brazil?")
        self.assertIn("Luís", rec.prompt_context)
        self.assertIn("Customer WHERE Country = 'Brazil'", rec.prompt_context)
        self.assertLess(rec.retrieval_ms, 2.0)

    def test_exemplar_retrieval_bm25(self):
        """Verify that FTS5 BM25 retrieves relevant past verified queries on the schema."""
        db_name = "chinook.db"
        self.mem.commit(
            session_id="s1",
            db_name=db_name,
            question="What is the top-selling track by invoice quantity?",
            sql="SELECT TrackId, SUM(Quantity) AS TotalQty FROM InvoiceLine GROUP BY TrackId ORDER BY TotalQty DESC LIMIT 1;",
            success=True,
        )
        self.mem.commit(
            session_id="s1",
            db_name=db_name,
            question="List all employees who are sales managers",
            sql="SELECT * FROM Employee WHERE Title LIKE '%Sales Manager%';",
            success=True,
        )

        # Query about tracks
        rec = self.mem.recall("s2", db_name, "Find the highest quantity track sold")
        self.assertGreater(len(rec.exemplars), 0)
        self.assertIn("TrackId", rec.exemplars[0].sql)
        self.assertIn("Verified Schema Exemplars", rec.prompt_context)

    def test_self_healed_experience_memory(self):
        """Verify that self-healed queries are marked and prioritized."""
        db_name = "chinook.db"
        failed_sql = "SELECT Name FROM Track JOIN Genre USING(GenreId);"
        repaired_sql = "SELECT Track.Name, Genre.Name FROM Track JOIN Genre ON Track.GenreId = Genre.GenreId;"

        self.mem.commit(
            session_id="repair_sess",
            db_name=db_name,
            question="Show tracks and their genres",
            sql=repaired_sql,
            success=True,
            repaired_from=failed_sql,
            error_msg="ambiguous column name: Name",
        )

        # Inspect stats
        st = self.mem.stats()
        self.assertEqual(st["total_self_healed_patterns"], 1)

        # Recall for similar query
        rec = self.mem.recall("new_sess", db_name, "List tracks with genres")
        self.assertTrue(any(e.was_repaired for e in rec.exemplars))
        self.assertIn("Self-healed & verified", rec.prompt_context)

    def test_persistence_across_reloads(self):
        """Verify memory survives full connection close and re-initialization."""
        self.mem.commit(
            session_id="persistent_session",
            db_name="store.db",
            question="Find total revenue for Q1",
            sql="SELECT SUM(total) FROM orders WHERE quarter = 1;",
            success=True,
        )
        self.mem.close()

        # Reopen with new instance pointing to same file
        mem2 = AgentMemoryEngine(self.db_path)
        last = mem2.get_last_turn("persistent_session")
        self.assertIsNotNone(last)
        self.assertEqual(last.question, "Find total revenue for Q1")
        self.assertEqual(mem2.stats()["total_verified_queries"], 1)
        mem2.close()

    def test_harness_dataset_export_and_import(self):
        """Verify harness dataset export and import for few-shot benchmark evaluation."""
        self.mem.commit(
            session_id="harness_sess",
            db_name="benchmark_db",
            question="Golden query 1",
            sql="SELECT * FROM table1;",
            success=True,
        )
        records = self.mem.export_dataset("benchmark_db")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["question"], "Golden query 1")

        # Import into fresh in-memory engine
        mem_harness = AgentMemoryEngine(":memory:")
        imported = mem_harness.import_dataset(records)
        self.assertEqual(imported, 1)
        self.assertEqual(mem_harness.stats()["total_verified_queries"], 1)
        mem_harness.close()


if __name__ == "__main__":
    unittest.main()
