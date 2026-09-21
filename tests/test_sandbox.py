#!/usr/bin/env python3
"""
tests/test_sandbox.py -- Tests for harness.security.sandbox
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from harness.security import (
    execute_sandboxed_query,
    get_safe_sqlite_connection,
    is_safe_read_only,
)


class TestSandboxSecurity(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "test_store.db"

        # Create test database
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("CREATE TABLE products (id INT PRIMARY KEY, name TEXT, price REAL);")
        conn.execute("INSERT INTO products VALUES (1, 'Widget', 19.99);")
        conn.execute("INSERT INTO products VALUES (2, 'Gadget', 29.99);")
        conn.commit()
        conn.close()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_safe_read_only_checker(self):
        # Valid queries
        safe, _ = is_safe_read_only("SELECT * FROM products;")
        self.assertTrue(safe)

        safe, _ = is_safe_read_only("WITH cte AS (SELECT id FROM products) SELECT * FROM cte;")
        self.assertTrue(safe)

        safe, _ = is_safe_read_only("EXPLAIN QUERY PLAN SELECT * FROM products WHERE price > 20;")
        self.assertTrue(safe)

        # Disallowed mutations
        safe, reason = is_safe_read_only("DROP TABLE products;")
        self.assertFalse(safe)
        self.assertIn("DROP", reason)

        safe, reason = is_safe_read_only("DELETE FROM products WHERE id = 1;")
        self.assertFalse(safe)
        self.assertIn("DELETE", reason)

        safe, reason = is_safe_read_only("UPDATE products SET price = 0;")
        self.assertFalse(safe)
        self.assertIn("UPDATE", reason)

        # Multi-statement injection
        safe, reason = is_safe_read_only("SELECT * FROM products; DROP TABLE products;")
        self.assertFalse(safe)

    def test_sandboxed_execution_blocks_mutations(self):
        res = execute_sandboxed_query(self.db_path, "DROP TABLE products;", read_only=True)
        self.assertFalse(res["success"])
        self.assertIn("Security Violation", res["error"])

        # Check that table still exists and data is intact
        res_read = execute_sandboxed_query(self.db_path, "SELECT COUNT(*) FROM products;")
        self.assertTrue(res_read["success"])
        self.assertEqual(res_read["rows"][0][0], 2)

    def test_sandboxed_execution_read_only_connection(self):
        # Even if a mutation somehow slipped past regex, SQLite URI mode=ro should prevent writes
        conn = get_safe_sqlite_connection(self.db_path, read_only=True)
        with self.assertRaises(sqlite3.OperationalError):
            conn.execute("INSERT INTO products VALUES (3, 'Doohickey', 9.99);")
        conn.close()


if __name__ == "__main__":
    unittest.main()
