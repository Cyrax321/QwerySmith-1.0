# QwerySmith Enterprise Agentic Harness (`harness/`)

The `harness/` package is an enterprise-grade multi-stage Text-to-SQL agent harness combining execution-guided candidate selection, subgraph schema linking, column value grounding, multi-turn dialogue state tracking, and AST-aware self-healing reflection.

---

## Architecture Overview

```text
harness/
├── __init__.py               # Top-level exports with 100% backward compatibility
├── __main__.py               # CLI runner entrypoint (python -m harness)
├── agent.py                  # Orchestrator: Multi-stage reasoning & execution pipeline
├── config.py                 # Central configuration dataclass (HarnessConfig)
├── security/
│   ├── __init__.py
│   └── sandbox.py            # Strict read-only connections, timeouts & AST mutation guards
├── schema/
│   ├── __init__.py
│   ├── linker.py             # Subgraph schema linking, relevance scoring & FK expansion
│   └── value_grounding.py    # Categorical literal grounding & case-normalization index
├── decoding/
│   ├── __init__.py
│   └── selector.py           # Execution-guided candidate selection & majority result voting
├── conversation/
│   ├── __init__.py
│   └── state_tracker.py      # Dialogue State Tracking (DST) for multi-turn sessions (SParC/CoSQL)
├── reflection/
│   ├── __init__.py
│   └── self_healing.py       # Multi-step AST error reflection, fuzzy suggestions & retry budget
├── adapters/
│   ├── __init__.py
│   ├── base.py               # Abstract DatabaseAdapter interface
│   ├── sqlite_adapter.py     # SQLite driver with sandboxing & timeout
│   └── duckdb_adapter.py     # In-process analytical DuckDB adapter
├── telemetry/
│   ├── __init__.py
│   └── events.py             # Pub/Sub event dispatcher & latency breakdown profiling
├── benchmark/
│   ├── __init__.py
│   └── evaluator.py          # Benchmark evaluator for Spider / BIRD / custom test sets
├── memory.py                 # Sub-millisecond persistent memory engine (<1ms)
├── self_healing.py           # Backward-compatible reflection engine export
├── tools.py                  # Database sandboxed execution, inspection & utilities
└── README.md                 # Complete documentation
```

---

## 1. Core Modules

### A. Strict Sandboxing & Security (`harness.security`)
- **Read-Only Enforcement**: Connections opened in SQLite URI `mode=ro` preventing accidental or malicious writes.
- **AST Mutation Blocker**: Blocks `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `REPLACE`, `ATTACH`, `PRAGMA`.
- **Query Timeout Handler**: Uses opcode progress handler interrupts to cancel slow queries or Cartesian products without blocking the main Python thread.

### B. Subgraph Schema Linking & Value Grounding (`harness.schema`)
- **Relevance Ranking & Subgraph Expansion**: Scores tables and columns against question tokens, automatically including intermediate bridge tables along foreign key paths.
- **Column Value Grounding**: Catalogs categorical column values (e.g. `'Platinum'`, `'Shipped'`), mapping colloquial user literals to exact database casing.

### C. Execution-Guided Decoding & Selection (`harness.decoding`)
- **Candidate Evaluation**: Executes generated queries inside the sandbox to prune syntax errors.
- **Majority Consensus Voting**: Clusters queries by semantic result-set hash, selecting the consensus candidate that produces verified, non-empty results.

### D. Dialogue State Tracking (`harness.conversation`)
- **Multi-Turn Context**: Tracks conversation turns, active entities, and active filters across multi-turn sessions (SParC/CoSQL style).
- **Anaphoric Resolution**: Detects references (*"they"*, *"those"*, *"that"*, *"and in 2024"*) and injects prior-turn state.

### E. Multi-Step AST Self-Healing (`harness.reflection`)
- **Fuzzy Identifier Suggestions**: Uses Levenshtein distance against the schema to recommend exact column and table spellings.
- **Progressive Retry Budget**: Maintains multi-step attempt histories to prevent oscillating errors across retries ($K \le 3$).

### F. Pluggable Database Dialects (`harness.adapters`)
- Standardized `DatabaseAdapter` interface supporting `SQLiteAdapter` and `DuckDBAdapter`.

### G. Telemetry & Quantitative Benchmarking (`harness.telemetry`, `harness.benchmark`)
- **Telemetry Event Dispatcher**: Publishes `TURN_START`, `SCHEMA_LINKED`, `VALUE_GROUNDED`, `SQL_GENERATED`, `QUERY_EXECUTED`, `REPAIR_ATTEMPTED`, `TURN_END` events with sub-millisecond latency breakdowns.
- **Benchmark Evaluator**: Computes Execution Accuracy (`EX`), Valid Rate (`VR`), and Exact Match (`EM`) on benchmark suites.

---

## 2. Usage Examples

### Running with Centralized Config
```python
from harness import QwerySmithAgent, HarnessConfig

config = HarnessConfig(
    read_only=True,
    timeout_sec=3.0,
    enable_schema_pruning=True,
    enable_value_grounding=True,
    max_repair_attempts=3,
)

agent = QwerySmithAgent(model_path="Cyrax321/QwerySmith-1.1", config=config)
result = agent.query("company_store.db", "Which customers bought a MacBook Pro?")
print("Human Answer:", result["human_answer"])
print("Executed SQL:", result["sql"])
print("Latency Breakdown:", result["latency_breakdown"])
```

### Running Automated Benchmarks
```python
from harness import BenchmarkEvaluator, BenchmarkItem

evaluator = BenchmarkEvaluator(timeout_sec=2.0)
items = [
    BenchmarkItem("1", "Total sales in 2024?", "SELECT SUM(total_amount) FROM orders WHERE order_date >= '2024-01-01';", "store.db"),
]

summary = evaluator.evaluate(items, agent.query)
print("Execution Accuracy:", summary.execution_accuracy)
```

---

## 3. Interactive Cookbook Recipes

### Multi-Turn Dialogue Tracking
```python
from harness.conversation import DialogueStateTracker

tracker = DialogueStateTracker()
# Turn 1
tracker.record_turn(
    session_id="user_123",
    db_name="company_store.db",
    question="Which customers are in California?",
    sql="SELECT * FROM customers WHERE state = 'CA';",
    columns=["id", "name", "state"],
    rows=[(1, "Alice", "CA")],
)
# Context injection for follow-up questions
prompt_ctx = tracker.build_context_prompt("user_123", "And who among them spent over $500?")
```

### Analytical Acceleration with DuckDB
```python
from harness.adapters import DuckDBAdapter

adapter = DuckDBAdapter("chinook.db")
result = adapter.execute_query("SELECT AVG(total) FROM invoices;")
print("Average Invoice:", result["rows"][0][0])
```

### Production Security Hardening
All executions default to `mode=ro` with SQLite progress handler interrupts guarding against runaway cartesian products.
