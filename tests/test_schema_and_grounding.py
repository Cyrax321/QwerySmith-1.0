#!/usr/bin/env python3
"""
tests/test_schema_and_grounding.py -- Unit tests for SchemaLinker and ValueGrounder
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from harness.schema import SchemaLinker, ValueGrounder


class TestSchemaAndGrounding(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "enterprise.db"

        conn = sqlite3.connect(str(self.db_path))
        conn.executescript("""
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY,
                name TEXT,
                tier TEXT
            );
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                customer_id INTEGER,
                order_date TEXT,
                status TEXT,
                FOREIGN KEY (customer_id) REFERENCES customers(id)
            );
            CREATE TABLE order_items (
                id INTEGER PRIMARY KEY,
                order_id INTEGER,
                product_name TEXT,
                price REAL,
                FOREIGN KEY (order_id) REFERENCES orders(id)
            );
            CREATE TABLE audit_logs (
                log_id INTEGER PRIMARY KEY,
                action TEXT,
                timestamp TEXT
            );
            INSERT INTO customers VALUES (1, 'Sophia', 'Platinum');
            INSERT INTO customers VALUES (2, 'Kenji', 'Gold');
            INSERT INTO orders VALUES (101, 1, '2024-01-01', 'Delivered');
            INSERT INTO orders VALUES (102, 2, '2024-01-02', 'Pending');
            INSERT INTO order_items VALUES (1, 101, 'MacBook', 1999.99);
        """)
        conn.commit()
        conn.close()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_schema_linker_graph_expansion(self):
        linker = SchemaLinker(self.db_path)
        # 4 tables total, if max_tables=2
        pruned = linker.link("Which customers bought which products?", max_tables=3)
        # Should include customers and order_items, AND auto-include intermediate 'orders' table
        self.assertIn("customers", pruned.selected_tables)
        self.assertIn("order_items", pruned.selected_tables)
        self.assertIn("orders", pruned.selected_tables)
        self.assertNotIn("audit_logs", pruned.selected_tables)
        self.assertTrue(pruned.was_pruned)

    def test_value_grounder_casing_and_exact_match(self):
        grounder = ValueGrounder(self.db_path)
        # User query has lowercase 'platinum'
        matches = grounder.ground("show me all platinum customers who have orders delivered")
        matched_dict = {m.matched_term: (m.table_name, m.column_name, m.exact_db_value) for m in matches}

        self.assertIn("platinum", matched_dict)
        self.assertEqual(matched_dict["platinum"], ("customers", "tier", "Platinum"))

        self.assertIn("delivered", matched_dict)
        self.assertEqual(matched_dict["delivered"], ("orders", "status", "Delivered"))

        # Verify hint formatting
        hint_text = grounder.format_grounding_hints(matches)
        self.assertIn("customers.tier = 'Platinum'", hint_text)
        self.assertIn("orders.status = 'Delivered'", hint_text)


if __name__ == "__main__":
    unittest.main()
