
from .exceptions import (
    HarnessError,
    DialectError,
    SchemaLinkingError,
    DecodingConsensusError,
    BenchmarkTimeoutError,
)
"""
harness -- QwerySmith Agentic Execution, Memory, Self-Healing & Evaluation Harness Package.

Exposes:
- `QwerySmithAgent`: Autonomous conversational and text-to-SQL agent.
- `AgentMemoryEngine`: Ultra-fast (<1ms) persistent agentic memory layer.
- `SelfHealingEngine`: Reflection and error-repair engine.
- Tool Calls: `execute_query`, `get_schema`, `get_tables`, `get_table_counts`, `get_table_sample`, `init_sample_db`, `classify_intent`, `format_table`, `clean_sql`.
"""

from .agent import QwerySmithAgent, chat_loop, main
from .config import HarnessConfig
from .conversation import DialogueStateTracker, DialogueTurn, SessionState
from .decoding import CandidateSelectionResult, ExecutionGuidedSelector
from .memory import (
    AgentMemoryEngine,
    ExemplarRecord,
    MemoryRetrievalResult,
    MemoryTurn,
    is_followup_question,
)
from .reflection import ErrorDiagnosis, MultiStepRepairTracker, RepairAttempt, SelfHealingEngine
from .schema import GroundedValueMatch, PrunedSchema, SchemaLinker, TableNode, ValueGrounder
from .security import (
    QueryTimeoutError,
    SecurityViolationError,
    execute_sandboxed_query,
    get_safe_sqlite_connection,
    is_safe_read_only,
)
from .telemetry import AgentEvent, EventType, LatencyBreakdown, TelemetryDispatcher
from .adapters import DatabaseAdapter, QueryResult, SQLiteAdapter
from .benchmark import BenchmarkEvaluator, BenchmarkItem, BenchmarkSummary
from .self_healing import RepairRecord
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
    # Core Agent & Config
    "QwerySmithAgent",
    "HarnessConfig",
    "chat_loop",
    "main",
    # Memory
    "AgentMemoryEngine",
    "MemoryTurn",
    "ExemplarRecord",
    "MemoryRetrievalResult",
    "is_followup_question",
    # Reflection & Self-Healing
    "SelfHealingEngine",
    "RepairRecord",
    "ErrorDiagnosis",
    "RepairAttempt",
    "MultiStepRepairTracker",
    # Schema & Value Grounding
    "SchemaLinker",
    "TableNode",
    "PrunedSchema",
    "ValueGrounder",
    "GroundedValueMatch",
    # Conversation & DST
    "DialogueStateTracker",
    "DialogueTurn",
    "SessionState",
    # Decoding & Selection
    "ExecutionGuidedSelector",
    "CandidateSelectionResult",
    # Security & Sandboxing
    "execute_sandboxed_query",
    "get_safe_sqlite_connection",
    "is_safe_read_only",
    "QueryTimeoutError",
    "SecurityViolationError",
    # Adapters
    "DatabaseAdapter",
    "HarnessError",
    "DialectError",
    "SchemaLinkingError",
    "DecodingConsensusError",
    "BenchmarkTimeoutError",
    "QueryResult",
    "SQLiteAdapter",
    # Telemetry & Benchmarks
    "TelemetryDispatcher",
    "AgentEvent",
    "EventType",
    "LatencyBreakdown",
    "BenchmarkEvaluator",
    "BenchmarkItem",
    "BenchmarkSummary",
    # Tools
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
