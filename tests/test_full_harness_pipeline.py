#!/usr/bin/env python3
"""
tests/test_full_harness_pipeline.py -- End-to-end multi-stage pipeline integration test
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from harness import (
    BenchmarkEvaluator,
    BenchmarkItem,
    DialogueStateTracker,
    EventType,
    ExecutionGuidedSelector,
    HarnessConfig,
    SchemaLinker,
    SelfHealingEngine,
    TelemetryDispatcher,
    ValueGrounder,
    execute_sandboxed_query,
)


class TestFullHarnessPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "e2e_store.db"

        conn = sqlite3.connect(str(self.db_path))
        conn.executescript("""
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY,
                name TEXT,
                country TEXT,
                loyalty_tier TEXT
            );
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                customer_id INTEGER,
                order_date TEXT,
                total_amount REAL,
                status TEXT,
                FOREIGN KEY (customer_id) REFERENCES customers(id)
            );
            CREATE TABLE order_items (
                id INTEGER PRIMARY KEY,
                order_id INTEGER,
                product_name TEXT,
                quantity INTEGER,
                unit_price REAL,
                FOREIGN KEY (order_id) REFERENCES orders(id)
            );
            INSERT INTO customers VALUES (1, 'Sophia Chen', 'USA', 'Platinum');
            INSERT INTO customers VALUES (2, 'Kenji Sato', 'Japan', 'Gold');
            INSERT INTO orders VALUES (101, 1, '2024-01-15', 2500.0, 'Delivered');
            INSERT INTO orders VALUES (102, 2, '2024-02-10', 400.0, 'Pending');
            INSERT INTO order_items VALUES (1, 101, 'MacBook Pro', 1, 2500.0);
            INSERT INTO order_items VALUES (2, 102, 'Headphones', 1, 400.0);
        """)
        conn.commit()
        conn.close()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_full_pipeline_coordination(self):
        # 1. Config
        cfg = HarnessConfig(
            read_only=True,
            timeout_sec=2.0,
            enable_schema_pruning=True,
            enable_value_grounding=True,
            enable_dst=True,
        )

        # 2. Telemetry
        events_captured = []
        telemetry = TelemetryDispatcher()
        telemetry.subscribe(lambda ev: events_captured.append(ev.event_type))

        session_id = "test_pipeline_session"
        db_name = self.db_path.name

        telemetry.dispatch(EventType.TURN_START, session_id, db_name)

        # 3. Schema Linking
        linker = SchemaLinker(self.db_path)
        pruned = linker.link("Which customers bought a MacBook Pro?", max_tables=3)
        self.assertIn("customers", pruned.selected_tables)
        self.assertIn("order_items", pruned.selected_tables)
        telemetry.dispatch(EventType.SCHEMA_LINKED, session_id, db_name, tables=pruned.selected_tables)

        # 4. Value Grounding
        grounder = ValueGrounder(self.db_path)
        matches = grounder.ground("show platinum customers with delivered orders")
        hint_str = grounder.format_grounding_hints(matches)
        self.assertIn("customers.loyalty_tier = 'Platinum'", hint_str)
        self.assertIn("orders.status = 'Delivered'", hint_str)
        telemetry.dispatch(EventType.VALUE_GROUNDED, session_id, db_name)

        # 5. Dialogue State Tracker
        dst = DialogueStateTracker()
        turn1_q = "List all platinum customers"
        turn1_sql = "SELECT name FROM customers WHERE loyalty_tier = 'Platinum';"
        exec1 = execute_sandboxed_query(self.db_path, turn1_sql)
        dst.record_turn(session_id, db_name, turn1_q, turn1_sql, exec1["columns"], exec1["rows"])

        # Follow-up
        turn2_q = "and which of them live in the USA?"
        self.assertTrue(dst.is_followup(turn2_q, session_id))
        ctx_prompt = dst.build_context_prompt(session_id, turn2_q)
        self.assertIn("Sophia Chen", ctx_prompt)

        # 6. Execution Guided Selection & Sandboxing
        selector = ExecutionGuidedSelector()
        candidates = [
            "SELECT name FROM customers WHERE country = 'USA' AND loyalty_tier = 'Platinum';",
            "SELECT name FROM non_existent_table;",
        ]
        sel_res = selector.select(self.db_path, candidates, read_only=cfg.read_only)
        self.assertTrue(sel_res.success)
        self.assertEqual(sel_res.rows[0][0], "Sophia Chen")
        telemetry.dispatch(EventType.QUERY_EXECUTED, session_id, db_name)

        telemetry.dispatch(EventType.TURN_END, session_id, db_name)

        # Verify all telemetry events fired
        self.assertIn(EventType.TURN_START, events_captured)
        self.assertIn(EventType.SCHEMA_LINKED, events_captured)
        self.assertIn(EventType.VALUE_GROUNDED, events_captured)
        self.assertIn(EventType.QUERY_EXECUTED, events_captured)
        self.assertIn(EventType.TURN_END, events_captured)

    def test_benchmark_evaluator_summary(self):
        evaluator = BenchmarkEvaluator(timeout_sec=2.0)
        items = [
            BenchmarkItem(
                item_id="item_1",
                question="What is the total amount of order 101?",
                gold_sql="SELECT total_amount FROM orders WHERE id = 101;",
                db_path=str(self.db_path),
            ),
            BenchmarkItem(
                item_id="item_2",
                question="Find customer named Sophia Chen",
                gold_sql="SELECT id FROM customers WHERE name = 'Sophia Chen';",
                db_path=str(self.db_path),
            ),
        ]

        # Dummy agent that produces gold for item 1, and wrong table for item 2
        def dummy_agent(db_path: str, q: str):
            if "101" in q:
                return {"sql": "SELECT total_amount FROM orders WHERE id = 101;", "success": True}
            return {"sql": "SELECT id FROM wrong_table;", "success": False}

        summary = evaluator.evaluate(items, dummy_agent)
        self.assertEqual(summary.total_items, 2)
        self.assertEqual(summary.execution_match_count, 1)
        self.assertEqual(summary.execution_accuracy, 0.5)
        self.assertEqual(summary.valid_sql_rate, 0.5)


if __name__ == "__main__":
    unittest.main()
