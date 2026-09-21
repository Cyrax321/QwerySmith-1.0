#!/usr/bin/env python3
"""
harness/adapters/duckdb_adapter.py -- Optional In-Process DuckDB Analytical Adapter
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .base import DatabaseAdapter, QueryResult


class DuckDBAdapter(DatabaseAdapter):
    """Adapter for DuckDB analytical databases."""

    def __init__(self, db_path: Union[str, Path]):
        try:
            import duckdb
        except ImportError:
            raise ImportError("DuckDB is not installed. Run 'pip install duckdb' to use DuckDBAdapter.")

        self.db_path = str(Path(db_path).resolve())
        self.duckdb = duckdb
        # Test read-only connection
        conn = duckdb.connect(self.db_path, read_only=True)
        conn.close()

    def execute(
        self,
        sql: str,
        timeout_sec: float = 3.0,
        max_rows: int = 100,
        read_only: bool = True,
    ) -> QueryResult:
        import time
        t0 = time.perf_counter()
        try:
            conn = self.duckdb.connect(self.db_path, read_only=read_only)
            res = conn.execute(sql)
            columns = [col[0] for col in res.description] if res.description else []
            rows = res.fetchmany(max_rows)
            latency = (time.perf_counter() - t0) * 1000
            conn.close()
            return QueryResult(
                columns=columns,
                rows=rows,
                latency_exec_ms=latency,
                error=None,
                success=True,
            )
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000
            return QueryResult(
                columns=[],
                rows=[],
                latency_exec_ms=latency,
                error=str(e),
                success=False,
            )

    def get_schema(self) -> str:
        conn = self.duckdb.connect(self.db_path, read_only=True)
        tables = conn.execute("SHOW TABLES;").fetchall()
        schema_parts = []
        for t in tables:
            tbl_name = t[0]
            schema_parts.append(f"-- Table: {tbl_name}")
            cols = conn.execute(f"DESCRIBE {tbl_name};").fetchall()
            for c in cols:
                schema_parts.append(f"  {c[0]} {c[1]}")
        conn.close()
        return "\n".join(schema_parts)

    def get_tables(self) -> List[str]:
        conn = self.duckdb.connect(self.db_path, read_only=True)
        tables = [t[0] for t in conn.execute("SHOW TABLES;").fetchall()]
        conn.close()
        return tables

    def get_table_counts(self) -> Dict[str, int]:
        tables = self.get_tables()
        conn = self.duckdb.connect(self.db_path, read_only=True)
        counts = {}
        for t in tables:
            counts[t] = conn.execute(f"SELECT COUNT(*) FROM {t};").fetchone()[0]
        conn.close()
        return counts

    def get_table_sample(self, table_name: str, limit: int = 3) -> Dict[str, Any]:
        conn = self.duckdb.connect(self.db_path, read_only=True)
        try:
            res = conn.execute(f"SELECT * FROM {table_name} LIMIT {limit};")
            columns = [c[0] for c in res.description]
            rows = res.fetchall()
            conn.close()
            return {"table": table_name, "columns": columns, "rows": rows, "error": None}
        except Exception as e:
            conn.close()
            return {"table": table_name, "columns": [], "rows": [], "error": str(e)}

    def close(self) -> None:
        pass
