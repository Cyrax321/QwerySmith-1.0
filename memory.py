#!/usr/bin/env python3
"""
memory.py -- Ultra-Fast Persistent Agentic Memory Layer (Root Entrypoint)

Forwards to the modular harness package (`harness.memory`).
Provides 100% backward compatibility for evaluation harnesses and interactive scripts.
"""

from harness.memory import (
    AgentMemoryEngine,
    ExemplarRecord,
    MemoryRetrievalResult,
    MemoryTurn,
    is_followup_question,
    main_demo,
)

__all__ = [
    "AgentMemoryEngine",
    "MemoryTurn",
    "ExemplarRecord",
    "MemoryRetrievalResult",
    "is_followup_question",
    "main_demo",
]

if __name__ == "__main__":
    main_demo()
