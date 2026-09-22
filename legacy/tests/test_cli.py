#!/usr/bin/env python3
import unittest
from harness.cli import build_parser, main


class TestCLI(unittest.TestCase):
    def test_parser_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["--db", "test.db"])
        self.assertEqual(args.db, "test.db")
        self.assertEqual(args.timeout, 3.0)
        self.assertTrue(args.read_only)

    def test_main_exit_code(self):
        self.assertEqual(main(["--db", "test.db"]), 0)


if __name__ == "__main__":
    unittest.main()
