"""
harness -- QwerySmith Agentic Execution, Memory, Self-Healing & Evaluation Harness Package.

Exposes:
- `QwerySmithAgent`: Autonomous conversational and text-to-SQL agent.
- `AgentMemoryEngine`: Ultra-fast (<1ms) persistent agentic memory layer.
- `SelfHealingEngine`: Reflection and error-repair engine.
- Tool Calls: `execute_query`, `get_schema`, `get_tables`, `get_table_counts`, `get_table_sample`, `init_sample_db`, `classify_intent`, `format_table`, `clean_sql`.
"""

from .agent import QwerySmithAgent, chat_loop, main
from .memory import (
    AgentMemoryEngine,
    ExemplarRecord,
    MemoryRetrievalResult,
    MemoryTurn,
    is_followup_question,
)
from .self_healing import RepairRecord, SelfHealingEngine
from .tools import (
    classify_intent,
    clean_sql,
    execute_query,
    format_table,
    get_schema,
    get_table_counts,
    get_table_sample,
    get_tables,
    init_sample_db,
)

__all__ = [
    "QwerySmithAgent",
    "chat_loop",
    "main",
    "AgentMemoryEngine",
    "MemoryTurn",
    "ExemplarRecord",
    "MemoryRetrievalResult",
    "is_followup_question",
    "SelfHealingEngine",
    "RepairRecord",
    "execute_query",
    "get_schema",
    "get_tables",
    "get_table_counts",
    "get_table_sample",
    "init_sample_db",
    "classify_intent",
    "format_table",
    "clean_sql",
]
