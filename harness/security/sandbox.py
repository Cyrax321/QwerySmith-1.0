#!/usr/bin/env python3
"""
harness/security/sandbox.py -- Strict Read-Only Sandbox, AST Safety & Timeout Guard
"""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class SecurityViolationError(Exception):
    """Raised when a query attempts mutating, administrative, or unauthorized actions."""
    pass


class QueryTimeoutError(Exception):
    """Raised when query execution exceeds the configured timeout budget."""
    pass


MUTATION_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH|VACUUM|REINDEX)\b",
    re.IGNORECASE,
)

PRAGMA_MUTATION_PATTERN = re.compile(
    r"\bPRAGMA\b\s*(\w+)?\s*=",
    re.IGNORECASE,
)


def is_safe_read_only(sql: str) -> Tuple[bool, Optional[str]]:
    """
    Analyzes SQL statement to guarantee it is strictly read-only.
    Allows: SELECT, WITH ... SELECT, EXPLAIN.
    Blocks: DDL, DML, ATTACH, dangerous PRAGMAs, multiple stacked statements.
    """
    stripped = sql.strip()
    if not stripped:
        return False, "Query is empty."

    # Remove string literals to avoid false positives inside quotes
    cleaned = re.sub(r"'[^']*'", "''", stripped)
    cleaned = re.sub(r'"[^"]*"', '""', cleaned)
    # Remove comments
    cleaned = re.sub(r"--[^\n]*", "", cleaned)
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)

    # Check for disallowed mutation keywords
    match = MUTATION_PATTERN.search(cleaned)
    if match:
        return False, f"Disallowed mutation statement: '{match.group(1).upper()}' is forbidden in read-only mode."

    # Check for mutating pragmas
    if PRAGMA_MUTATION_PATTERN.search(cleaned):
        return False, "Modifying database PRAGMAs is forbidden."

    # Ensure query starts with safe read prefix (SELECT, WITH, EXPLAIN)
    norm_start = cleaned.strip().upper()
    if not (norm_start.startswith("SELECT") or norm_start.startswith("WITH") or norm_start.startswith("EXPLAIN")):
        return False, "Query must begin with SELECT, WITH, or EXPLAIN."

    # Check for multiple statements separated by semicolons (prevent injection)
    statements = [s.strip() for s in cleaned.split(";") if s.strip()]
    if len(statements) > 1:
        return False, "Multiple stacked SQL statements separated by semicolons are not permitted."

    return True, None


def get_safe_sqlite_connection(
    db_path: Union[str, Path],
    read_only: bool = True,
    timeout_sec: float = 3.0,
) -> sqlite3.Connection:
    """
    Opens a SQLite connection wrapped with read-only flags and an opcode progress handler timeout.
    """
    resolved = Path(db_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Database file not found: {resolved}")

    if read_only:
        # Use SQLite URI mode=ro
        uri_path = f"file:{resolved}?mode=ro"
        conn = sqlite3.connect(uri_path, uri=True, check_same_thread=False)
    else:
        conn = sqlite3.connect(str(resolved), check_same_thread=False)

    # Configure timeout via progress handler
    if timeout_sec > 0:
        start_time = [time.perf_counter()]

        def progress_handler() -> int:
            if (time.perf_counter() - start_time[0]) > timeout_sec:
                # Return non-zero to interrupt query execution
                return 1
            return 0

        # Progress handler is called every 10,000 SQLite opcodes
        conn.set_progress_handler(progress_handler, 10000)

    return conn


def execute_sandboxed_query(
    conn_or_path: Union[sqlite3.Connection, str, Path],
    sql: str,
    max_rows: int = 100,
    timeout_sec: float = 3.0,
    read_only: bool = True,
) -> Dict[str, Any]:
    """
    Executes a SQL query safely inside a sandboxed environment with strict timing,
    read-only enforcement, and error trapping.
    """
    # 1. AST / Pattern Safety Verification
    if read_only:
        is_safe, reason = is_safe_read_only(sql)
        if not is_safe:
            return {
                "columns": [],
                "rows": [],
                "latency_exec_ms": 0.0,
                "error": f"Security Violation: {reason}",
                "success": False,
                "timed_out": False,
            }

    # 2. Connection Acquisition
    should_close = False
    if isinstance(conn_or_path, sqlite3.Connection):
        conn = conn_or_path
    else:
        try:
            conn = get_safe_sqlite_connection(conn_or_path, read_only=read_only, timeout_sec=timeout_sec)
            should_close = True
        except Exception as e:
            return {
                "columns": [],
                "rows": [],
                "latency_exec_ms": 0.0,
                "error": f"Connection error: {e}",
                "success": False,
                "timed_out": False,
            }

    # 3. Execution with wall-clock timing and timeout trapping
    t0 = time.perf_counter()
    try:
        cur = conn.cursor()
        cur.execute(sql)
        columns = [desc[0] for desc in cur.description] if cur.description else []
        rows = cur.fetchmany(max_rows)
        latency = (time.perf_counter() - t0) * 1000

        if should_close:
            conn.close()

        return {
            "columns": columns,
            "rows": rows,
            "latency_exec_ms": latency,
            "error": None,
            "success": True,
            "timed_out": False,
        }
    except sqlite3.OperationalError as e:
        latency = (time.perf_counter() - t0) * 1000
        err_msg = str(e)
        timed_out = "interrupted" in err_msg.lower()
        if timed_out:
            err_msg = f"Query timed out after {timeout_sec:.2f}s (execution budget exceeded)."

        if should_close:
            try:
                conn.close()
            except Exception:
                pass

        return {
            "columns": [],
            "rows": [],
            "latency_exec_ms": latency,
            "error": err_msg,
            "success": False,
            "timed_out": timed_out,
        }
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        if should_close:
            try:
                conn.close()
            except Exception:
                pass

        return {
            "columns": [],
            "rows": [],
            "latency_exec_ms": latency,
            "error": str(e),
            "success": False,
            "timed_out": False,
        }
