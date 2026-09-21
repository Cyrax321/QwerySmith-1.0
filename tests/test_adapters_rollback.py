#!/usr/bin/env python3
import sqlite3
import tempfile
import unittest
from pathlib import Path
from harness.adapters import SQLiteAdapter


class TestAdapterTransaction(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "tx.db"
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("CREATE TABLE items (id INT, val TEXT);")
        conn.commit()
        conn.close()
        self.adapter = SQLiteAdapter(self.db_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_transaction_rollback_on_error(self):
        try:
            with self.adapter.transaction() as conn:
                conn.execute("INSERT INTO items VALUES (1, 'ok');")
                raise ValueError("Simulated failure")
        except ValueError:
            pass
        res = self.adapter.execute_query("SELECT COUNT(*) FROM items;")
        self.assertEqual(res["rows"][0][0], 0)


if __name__ == "__main__":
    unittest.main()
