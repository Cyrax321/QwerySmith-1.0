#!/usr/bin/env python3
"""
tests/test_adapters.py -- Tests for database adapters
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from harness.adapters import SQLiteAdapter


class TestSQLiteAdapter(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "adapter_test.db"

        conn = sqlite3.connect(str(self.db_path))
        conn.execute("CREATE TABLE users (id INT, email TEXT);")
        conn.execute("INSERT INTO users VALUES (1, 'alice@example.com');")
        conn.commit()
        conn.close()

        self.adapter = SQLiteAdapter(self.db_path)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_adapter_operations(self):
        tables = self.adapter.get_tables()
        self.assertEqual(tables, ["users"])

        counts = self.adapter.get_table_counts()
        self.assertEqual(counts["users"], 1)

        schema = self.adapter.get_schema()
        self.assertIn("CREATE TABLE users", schema)

        res = self.adapter.execute("SELECT email FROM users WHERE id = 1;")
        self.assertTrue(res.success)
        self.assertEqual(res.rows[0][0], "alice@example.com")


if __name__ == "__main__":
    unittest.main()
