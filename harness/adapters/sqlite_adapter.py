#!/usr/bin/env python3
"""
harness/adapters/sqlite_adapter.py -- SQLite Database Driver with Sandboxing
"""

from __future__ import annotations
from contextlib import contextmanager

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..security.sandbox import execute_sandboxed_query, get_safe_sqlite_connection
from .base import DatabaseAdapter, QueryResult


class SQLiteAdapter(DatabaseAdapter):
    """SQLite dialect adapter incorporating read-only sandboxing and statement timeouts."""

    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path).resolve()
        if not self.db_path.exists():
            raise FileNotFoundError(f"SQLite database file does not exist: {self.db_path}")

    def execute(
        self,
        sql: str,
        timeout_sec: float = 3.0,
        max_rows: int = 100,
        read_only: bool = True,
    ) -> QueryResult:
        res = execute_sandboxed_query(
            self.db_path,
            sql,
            max_rows=max_rows,
            timeout_sec=timeout_sec,
            read_only=read_only,
        )
        return QueryResult(
            columns=res["columns"],
            rows=res["rows"],
            latency_exec_ms=res["latency_exec_ms"],
            error=res["error"],
            success=res["success"],
            timed_out=res.get("timed_out", False),
        )

    def get_schema(self) -> str:
        conn = get_safe_sqlite_connection(self.db_path, read_only=True)
        cur = conn.cursor()
        cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [row[0] for row in cur.fetchall() if row[0]]
        conn.close()
        return "\n".join(tables)

    def get_tables(self) -> List[str]:
        conn = get_safe_sqlite_connection(self.db_path, read_only=True)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [r[0] for r in cur.fetchall()]
        conn.close()
        return tables

    def get_table_counts(self) -> Dict[str, int]:
        tables = self.get_tables()
        conn = get_safe_sqlite_connection(self.db_path, read_only=True)
        counts = {}
        cur = conn.cursor()
        for t in tables:
            cur.execute(f"SELECT COUNT(*) FROM {t};")
            counts[t] = cur.fetchone()[0]
        conn.close()
        return counts

    def get_table_sample(self, table_name: str, limit: int = 3) -> Dict[str, Any]:
        conn = get_safe_sqlite_connection(self.db_path, read_only=True)
        cur = conn.cursor()
        try:
            cur.execute(f"SELECT * FROM {table_name} LIMIT {limit};")
            columns = [d[0] for d in cur.description]
            rows = cur.fetchall()
            conn.close()
            return {"table": table_name, "columns": columns, "rows": rows, "error": None}
        except Exception as e:
            conn.close()
            return {"table": table_name, "columns": [], "rows": [], "error": str(e)}

    def close(self) -> None:
        pass

    @contextmanager
    def transaction(self):
        """Context manager providing an atomic transaction with rollback on failure."""
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
