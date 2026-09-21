#!/usr/bin/env python3
import unittest
from harness.decoding.selector import query_complexity


class TestDecodingTiebreaker(unittest.TestCase):
    def test_complexity_scorer(self):
        simple_q = "SELECT id FROM users;"
        complex_q = "SELECT id FROM users WHERE id IN (SELECT user_id FROM orders JOIN items ON 1=1);"
        self.assertLess(query_complexity(simple_q), query_complexity(complex_q))


if __name__ == "__main__":
    unittest.main()
