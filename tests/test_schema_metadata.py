#!/usr/bin/env python3
import sqlite3
import tempfile
import unittest
from pathlib import Path
from harness.schema import SchemaLinker


class TestSchemaMetadata(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "meta.db"
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("CREATE TABLE tbl_alpha (id INT);")
        conn.execute("CREATE TABLE tbl_beta (id INT);")
        conn.commit()
        conn.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_table_descriptions_boost_score(self):
        descriptions = {"tbl_alpha": "Contains financial customer balance records"}
        linker = SchemaLinker(self.db_path, table_descriptions=descriptions)
        subgraph = linker.link("Show me financial customer records", max_tables=1)
        self.assertEqual(subgraph.selected_tables[0], "tbl_alpha")


if __name__ == "__main__":
    unittest.main()
