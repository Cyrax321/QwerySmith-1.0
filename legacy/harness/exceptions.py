#!/usr/bin/env python3
"""
harness/exceptions.py -- Structured exception hierarchy for QwerySmith harness
"""


class HarnessError(Exception):
    """Base exception for all harness runtime failures."""
    pass


class DialectError(HarnessError):
    """Raised when query contains unsupported dialect syntax."""
    pass


class SchemaLinkingError(HarnessError):
    """Raised when schema linking fails to locate valid tables."""
    pass


class DecodingConsensusError(HarnessError):
    """Raised when candidate selection fails to find a valid execution candidate."""
    pass


class BenchmarkTimeoutError(HarnessError):
    """Raised when query execution exceeds configured timeout budget."""
    pass
