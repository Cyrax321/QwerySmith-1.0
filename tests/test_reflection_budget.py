#!/usr/bin/env python3
import unittest
from harness.reflection import SelfHealingEngine


class TestReflectionBudget(unittest.TestCase):
    def test_budget_exhaustion(self):
        healer = SelfHealingEngine()
        self.assertFalse(healer.is_budget_exhausted(1, max_retries=3))
        self.assertFalse(healer.is_budget_exhausted(2, max_retries=3))
        self.assertTrue(healer.is_budget_exhausted(3, max_retries=3))


if __name__ == "__main__":
    unittest.main()
