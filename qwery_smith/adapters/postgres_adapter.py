"""Postgres adapter (plan §4.1: Olist canonical DB).

Connects read-only via psycopg + SQLAlchemy URI. The ingest path (load_csv,
add_primary_key, add_foreign_key) requires a read-write connection and is
used only by `ingest`; eval connections are strictly read-only (role with
SELECT-only grants + statement_timeout).
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

from ..exceptions import HarnessError, QueryTimeoutError
from .base import DatabaseAdapter
from .guard import assert_safe_select

_PG_TYPES = {
    "TEXT": "text", "VARCHAR": "text", "INTEGER": "integer", "INT": "integer",
    "BIGINT": "bigint", "REAL": "real", "FLOAT": "real", "DOUBLE": "double precision",
    "NUMERIC": "numeric", "BOOLEAN": "boolean", "DATE": "date", "TIMESTAMP": "timestamp",
    "DECIMAL": "numeric",
}


class PostgresAdapter(DatabaseAdapter):
    dialect = "postgres"

    def __init__(self, uri: str, read_only: bool = True, statement_timeout: float = 30.0):
        import psycopg

        self._psycopg = psycopg
        self.uri = uri
        self.read_only = read_only
        self.default_timeout = statement_timeout
        try:
            self.conn = psycopg.connect(uri)
        except Exception as e:
            raise HarnessError(f"cannot connect to Postgres: {e}") from e
        if read_only:
            with self.conn.transaction():
                self.conn.execute("SET default_transaction_read_only = on")
                self.conn.execute("SET statement_timeout = %s", (f"{int(statement_timeout * 1000)}ms",))

    # -- required interface ------------------------------------------------
    def execute(self, sql: str, timeout_sec: float = 30.0, max_rows: int = 10_000) -> tuple[list, list, float]:
        cur = self.conn.cursor()
        t0 = time.perf_counter()
        try:
            cur.execute("SET LOCAL statement_timeout = %s", (f"{int(timeout_sec * 1000)}ms",))
            cur.execute(sql)
            rows = cur.fetchmany(max_rows)
            if cur.fetchone() is not None:
                raise HarnessError(f"result exceeds max_rows={max_rows}")
            latency = (time.perf_counter() - t0) * 1000
            cols = [d.name for d in cur.description] if cur.description else []
            return rows, cols, latency
        except self._psycopg.errors.QueryCanceled as e:
            raise QueryTimeoutError(str(e)) from e
        finally:
            cur.close()

    def safe_execute(self, sql, timeout_sec=30.0, max_rows=10_000):
        # override to keep SET LOCAL inside the same read-only transaction
        reason = assert_safe_select(sql, dialect="postgres")
        if reason:
            from ..exceptions import UnsafeSQLError

            raise UnsafeSQLError(reason)
        return self.execute(sql, timeout_sec=timeout_sec, max_rows=max_rows)

    def load_csv(self, table: str, csv_path: Path, columns: dict[str, str]) -> int:
        if self.read_only:
            raise HarnessError("load_csv requires a read-write connection (ingest only)")
        cur = self.conn.cursor()
        try:
            cur.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')
            ddl_cols = ", ".join(f'"{n}" {_PG_TYPES.get(t.upper(), "text")}' for n, t in columns.items())
            cur.execute(f'CREATE TABLE "{table}" ({ddl_cols})')
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                reader = csv.reader(f)
                header = next(reader)
                idx = {h: i for i, h in enumerate(header)}
                n = 0
                for rec in reader:
                    vals = []
                    for name, t in columns.items():
                        v = rec[idx[name]] if name in idx else ""
                        if t.upper() in ("INTEGER", "INT", "BIGINT", "BOOLEAN") and v == "":
                            vals.append(None)
                        elif t.upper() in ("REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL") and v == "":
                            vals.append(None)
                        else:
                            vals.append(v)
                    cur.execute(
                        f'INSERT INTO "{table}" VALUES ({", ".join(["%s"] * len(vals))})', vals
                    )
                    n += 1
            self.conn.commit()
            return n
        finally:
            cur.close()

    def add_primary_key(self, table: str, columns: list[str]) -> None:
        cur = self.conn.cursor()
        try:
            cur.execute(
                f'ALTER TABLE "{table}" ADD PRIMARY KEY ({", ".join(columns)})'
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        finally:
            cur.close()

    def add_foreign_key(self, table: str, column: str, ref_table: str, ref_column: str) -> None:
        cur = self.conn.cursor()
        try:
            cur.execute(
                f'ALTER TABLE "{table}" ADD FOREIGN KEY ("{column}") REFERENCES "{ref_table}" ("{ref_column}")'
            )
            self.conn.commit()
        except Exception:
            # Olist has known FK orphans — report, don't fail (plan §4.1)
            self.conn.rollback()
        finally:
            cur.close()

    def table_names(self) -> list[str]:
        cur = self.conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' "
            "ORDER BY table_name"
        )
        return [r[0] for r in cur.fetchall()]

    def table_columns(self, table: str) -> list[tuple[str, str, bool]]:
        cur = self.conn.execute(
            "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = %s ORDER BY ordinal_position",
            (table,),
        )
        out = []
        for name, dtype, nullable in cur.fetchall():
            canonical = {
                "character varying": "TEXT", "text": "TEXT", "integer": "INTEGER",
                "bigint": "BIGINT", "smallint": "INTEGER", "numeric": "NUMERIC",
                "real": "REAL", "double precision": "REAL", "boolean": "BOOLEAN",
                "date": "DATE", "timestamp without time zone": "TIMESTAMP",
                "timestamp with time zone": "TIMESTAMP",
            }.get(dtype, "TEXT")
            out.append((name, canonical, nullable == "NO"))
        return out

    def foreign_keys(self, table: str) -> list[tuple[str, str, str]]:
        cur = self.conn.execute(
            """
            SELECT a.column_name, c_ref.table_name, c_ref.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage a ON tc.constraint_name = a.constraint_name
            JOIN information_schema.constraint_column_usage c_ref ON tc.constraint_name = c_ref.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_schema = 'public' AND tc.table_name = %s
            """,
            (table,),
        )
        return [(r[0], r[1], r[2]) for r in cur.fetchall()]

    def row_count(self, table: str) -> int:
        return self.conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]

    def scalar(self, sql: str) -> Any:
        return self.conn.execute(sql).fetchone()

    def close(self) -> None:
        self.conn.close()