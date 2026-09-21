# QwerySmith Agentic Harness (`harness/`)

The `harness/` package houses the runtime execution harness: autonomous agent loop, sub-millisecond persistent memory, self-healing reflection, and isolated database tools.

---

## Architecture Overview

```text
harness/
├── __init__.py               # Package exports (QwerySmithAgent, AgentMemoryEngine, etc.)
├── __main__.py               # CLI runner entrypoint (python -m harness)
├── agent.py                  # Dual-mode autonomous SQL & conversational agent
├── memory.py                 # Sub-millisecond persistent memory engine (<1ms)
├── self_healing.py           # AST & reflection error diagnosis and auto-repair
├── tools.py                  # Database execution sandbox, schema inspection, tables & intent
└── README.md                 # Package documentation
```

---

## 1. Components

### A. Autonomous Agent (`harness.agent.QwerySmithAgent`)
- **Dual-Mode Adapter Switching**: Enables QLoRA weights for SQL generation and dynamically disables adapter weights for conversational chit-chat, conceptual SQL explanations, and natural human summaries.
- **Real-Time Grounding**: Uses live database schemas and system timestamps for temporal queries.

### B. Persistent Memory Engine (`harness.memory.AgentMemoryEngine`)
- **Dual-Index Architecture**:
  - Compound B-Tree on `(session_id, turn_index)` for ~23 µs short-term context retrieval.
  - Compiled SQLite FTS5 virtual table with BM25 ranking for ~240 µs long-term verified exemplar recall.
- **Anaphoric Pronoun Resolution**: Detects follow-ups (*"they"*, *"those"*, *"that"*, *"it"*) and injects previous turn context.
- **Self-Healing Persistence**: Caches repaired queries into `qwerysmith_memory.sqlite` so the system never makes the same syntax error twice.

### C. Self-Healing Reflection Engine (`harness.self_healing.SelfHealingEngine`)
- Diagnoses runtime execution tracebacks (e.g. `ambiguous column name`, `no such column`, `syntax error`).
- Generates actionable repair prompts instructing the LLM on exact schema-qualifications needed.

### D. Database Tool Calls (`harness.tools`)
- `execute_query(conn_or_path, sql)`: Safe isolated SQL execution sandbox with timing.
- `get_schema(conn_or_path)`: DDL extraction for all active tables.
- `get_tables(conn_or_path)`: Table catalog inspection.
- `get_table_counts(conn_or_path)`: Row counting per table.
- `get_table_sample(conn_or_path, table_name)`: Sample rows preview.
- `classify_intent(text, table_names)`: Intent classification router.
- `format_table(columns, rows)`: ASCII table rendering.

---

## 2. Usage Examples

### Quick Import
```python
from harness import QwerySmithAgent, AgentMemoryEngine, execute_query, get_schema

# Initialize Agent
agent = QwerySmithAgent(model_path="Cyrax321/QwerySmith-1.1")

# Run End-to-End Query
result = agent.query("chinook.db", "Which customers are from Brazil?")
print("Human Answer:", result["human_answer"])
print("Executed SQL:", result["sql"])
```

### Evaluation Harness with Memory
```python
from harness import AgentMemoryEngine

mem = AgentMemoryEngine(":memory:")
mem.import_dataset([
    {"db_name": "chinook.db", "question": "Total sales?", "sql": "SELECT SUM(Total) FROM Invoice;"}
])

res = mem.recall(session_id="eval_01", db_name="chinook.db", question="What were the total sales?")
print("Recalled exemplar:", res.exemplars[0].sql)
```
