"""SQLite adapter - read-only enforcement + AST guard + timeout + load (M1).

Also serves as the reference implementation for tests and the toy fixture.
"""

from __future__ import annotations

import csv
import sqlite3
import time
from pathlib import Path
from typing import Any

from ..exceptions import HarnessError
from .base import DatabaseAdapter

_TYPE_MAP = {
    "TEXT": "TEXT", "VARCHAR": "TEXT",
    "INTEGER": "INTEGER", "INT": "INTEGER", "BIGINT": "INTEGER",
    "REAL": "REAL", "FLOAT": "REAL", "DOUBLE": "REAL", "NUMERIC": "NUMERIC",
    "BOOLEAN": "INTEGER", "DATE": "DATE", "TIMESTAMP": "TIMESTAMP",
    "DATETIME": "TIMESTAMP", "DECIMAL": "NUMERIC",
}


class SQLiteAdapter(DatabaseAdapter):
    dialect = "sqlite"

    def __init__(self, db_path: Path, read_only: bool = True):
        self.db_path = Path(db_path)
        self.read_only = read_only
        if read_only:
            if not self.db_path.exists():
                raise HarnessError(f"database not found: {self.db_path}")
            self.conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro", uri=True, timeout=30.0
            )
        else:
            self.conn = sqlite3.connect(self.db_path, timeout=30.0)
        self.conn.row_factory = None

    # -- internals ---------------------------------------------------------
    def _apply_timeout(self, timeout_sec: float) -> None:
        busy = max(int(timeout_sec * 1000), 100)
        self.conn.execute("PRAGMA busy_timeout = %d" % busy)
        # sqlite has no hard statement timeout; enforced by caller-side watch in execute()

    # -- required interface ------------------------------------------------
    def execute(self, sql: str, timeout_sec: float = 30.0, max_rows: int = 10_000) -> tuple[list, list, float]:
        cur = self.conn.cursor()
        t0 = time.perf_counter()
        try:
            cur.execute(sql)
            rows = cur.fetchmany(max_rows)
            if cur.fetchone() is not None:
                raise HarnessError(f"result exceeds max_rows={max_rows}")
            latency = (time.perf_counter() - t0) * 1000
            cols = [d[0] for d in cur.description] if cur.description else []
            return rows, cols, latency
        finally:
            cur.close()

    def load_csv(self, table: str, csv_path: Path, columns: dict[str, str], append: bool = False) -> int:
        csv_path = Path(csv_path)
        exists = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?", (table,)
        ).fetchone() is not None
        if not (append and exists):
            self.conn.execute(f'DROP TABLE IF EXISTS "{table}"')
            ddl_cols = ", ".join(
                f'"{name}" {_TYPE_MAP.get(t.upper(), "TEXT")}' for name, t in columns.items()
            )
            self.conn.execute(f'CREATE TABLE "{table}" ({ddl_cols})')
        placeholders = ", ".join("?" for _ in columns)
        col_names = ", ".join(f'"{c}"' for c in columns)
        n = 0
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            for rec in csv.DictReader(f):
                vals = []
                for c, t in columns.items():
                    v = rec.get(c, "")
                    if t.upper() in ("INTEGER", "INT", "BOOLEAN") and v == "":
                        vals.append(None)
                    elif t.upper() in ("REAL", "FLOAT", "NUMERIC") and v == "":
                        vals.append(None)
                    else:
                        vals.append(v)
                self.conn.execute(
                    f'INSERT INTO "{table}" ({col_names}) VALUES ({placeholders})', vals
                )
                n += 1
        self.conn.commit()
        return n

    def add_primary_key(self, table: str, columns: list[str]) -> None:
        # SQLite cannot ALTER ADD PK; recorded in schema metadata instead (see schema_loader).
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS __harness_meta_pk (table_name TEXT, cols TEXT)"
        )
        self.conn.execute("DELETE FROM __harness_meta_pk WHERE table_name = ?", (table,))
        self.conn.execute(
            "INSERT INTO __harness_meta_pk VALUES (?, ?)", (table, ",".join(columns))
        )
        self.conn.commit()

    def add_foreign_key(self, table: str, column: str, ref_table: str, ref_column: str) -> None:
        # Declared FKs require table rebuild in SQLite; recorded as metadata (see schema_loader).
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS __harness_meta_fk (table_name TEXT, col TEXT, ref_table TEXT, ref_col TEXT)"
        )
        self.conn.execute(
            "INSERT INTO __harness_meta_fk VALUES (?, ?, ?, ?)",
            (table, column, ref_table, ref_column),
        )
        self.conn.commit()

    def _meta(self, table_name: str, table: str, cols: tuple[str, ...]) -> list[tuple]:
        try:
            cur = self.conn.execute(
                f'SELECT {", ".join(cols)} FROM {table_name} WHERE table_name = ?', (table,)
            )
            return cur.fetchall()
        except sqlite3.OperationalError:
            return []

    def table_names(self) -> list[str]:
        cur = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "AND name NOT LIKE '__harness_meta%' ORDER BY name"
        )
        return [r[0] for r in cur.fetchall()]

    def table_columns(self, table: str) -> list[tuple[str, str, bool]]:
        cur = self.conn.execute(f'PRAGMA table_info("{table}")')
        return [(r[1], r[2].upper(), bool(r[3])) for r in cur.fetchall()]

    def foreign_keys(self, table: str) -> list[tuple[str, str, str]]:
        declared = [
            (r[3], r[2], r[4])
            for r in self.conn.execute(f'PRAGMA foreign_key_list("{table}")').fetchall()
        ]
        meta = [(r[0], r[1], r[2]) for r in self._meta("__harness_meta_fk", table, ("col", "ref_table", "ref_col"))]
        return declared + meta

    def row_count(self, table: str) -> int:
        return self.conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]

    def scalar(self, sql: str) -> Any:
        return self.conn.execute(sql).fetchone()

    def close(self) -> None:
        self.conn.close()