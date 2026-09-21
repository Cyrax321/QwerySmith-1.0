#!/usr/bin/env python3
"""
tests/test_candidate_selection.py -- Tests for ExecutionGuidedSelector
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from harness.decoding import ExecutionGuidedSelector


class TestCandidateSelection(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "test.db"

        conn = sqlite3.connect(str(self.db_path))
        conn.execute("CREATE TABLE products (id INT, name TEXT, price REAL);")
        conn.execute("INSERT INTO products VALUES (1, 'Widget', 10.0);")
        conn.execute("INSERT INTO products VALUES (2, 'Gadget', 20.0);")
        conn.execute("INSERT INTO products VALUES (3, 'Doohickey', 30.0);")
        conn.commit()
        conn.close()

        self.selector = ExecutionGuidedSelector(timeout_sec=2.0)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_prunes_syntax_errors_and_picks_valid(self):
        candidates = [
            "SELECT * FORM products;",  # Syntax error ('FORM')
            "SELECT name, price FROM products WHERE price >= 20 ORDER BY price ASC;",  # Valid
            "SELECT non_existent_col FROM products;",  # Runtime column error
        ]
        res = self.selector.select(self.db_path, candidates)
        self.assertTrue(res.success)
        self.assertEqual(res.candidates_evaluated, 3)
        self.assertEqual(res.candidates_valid, 1)
        self.assertEqual(len(res.rows), 2)
        self.assertEqual(res.rows[0][0], "Gadget")

    def test_majority_voting_consensus(self):
        candidates = [
            "SELECT name FROM products WHERE price > 15;",  # Yields Gadget, Doohickey (Cluster A)
            "SELECT name FROM products WHERE price >= 20;",  # Yields Gadget, Doohickey (Cluster A)
            "SELECT name FROM products WHERE price = 10;",   # Yields Widget (Cluster B - minority)
        ]
        res = self.selector.select(self.db_path, candidates)
        self.assertTrue(res.success)
        self.assertEqual(res.candidates_valid, 3)
        self.assertEqual(res.consensus_count, 2)
        self.assertEqual(len(res.rows), 2)


if __name__ == "__main__":
    unittest.main()
