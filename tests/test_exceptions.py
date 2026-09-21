#!/usr/bin/env python3
import unittest
from harness.exceptions import (
    HarnessError,
    DialectError,
    SchemaLinkingError,
    DecodingConsensusError,
    BenchmarkTimeoutError,
)


class TestHarnessExceptions(unittest.TestCase):
    def test_inheritance(self):
        self.assertTrue(issubclass(DialectError, HarnessError))
        self.assertTrue(issubclass(SchemaLinkingError, HarnessError))
        self.assertTrue(issubclass(DecodingConsensusError, HarnessError))
        self.assertTrue(issubclass(BenchmarkTimeoutError, HarnessError))

    def test_raise_and_catch(self):
        with self.assertRaises(HarnessError):
            raise SchemaLinkingError("Unable to link table: customers")


if __name__ == "__main__":
    unittest.main()
