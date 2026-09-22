#!/usr/bin/env python3
import unittest
from harness.config import HarnessConfig


class TestHarnessConfig(unittest.TestCase):
    def test_round_trip_dict(self):
        cfg = HarnessConfig(timeout_sec=5.0, max_repair_attempts=2)
        d = cfg.to_dict()
        self.assertEqual(d["timeout_sec"], 5.0)
        self.assertEqual(d["max_repair_attempts"], 2)
        reconstructed = HarnessConfig.from_dict(d)
        self.assertEqual(reconstructed.timeout_sec, 5.0)
        self.assertEqual(reconstructed.max_repair_attempts, 2)


if __name__ == "__main__":
    unittest.main()
