#!/usr/bin/env python3
import unittest
from harness.security.sandbox import is_system_table_query


class TestSecurityGuards(unittest.TestCase):
    def test_system_table_query(self):
        self.assertTrue(is_system_table_query("SELECT * FROM sqlite_master;"))
        self.assertTrue(is_system_table_query("SELECT * FROM sqlite_schema;"))
        self.assertFalse(is_system_table_query("SELECT * FROM products;"))


if __name__ == "__main__":
    unittest.main()
