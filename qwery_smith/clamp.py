"""Time-clamped shadow database (plan §4.4) — the mechanical window check.

Instead of injecting cutoff predicates into gold SQL (fragile under
subqueries/CTEs/NOT EXISTS), we execute gold SQL against a *clamped* shadow:
the holdout table filtered to rows before the cutoff, and fact children
(tables with FK -> holdout table) clamped to their parent's surviving rows.
Dimension tables pass through untouched.

A question is window-dependent iff its result differs between the full and
clamped databases. That single primitive drives BOTH:
  - split tagging at authoring time (dependent => heldout)
  - the §4.4 leak check at validate time (train_ok + dependent => leak)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .adapters.guard import assert_safe_select
from .config import HoldoutConfig
from .exceptions import HarnessError, UnsafeSQLError
from .questions import canonical_rows_hash
from .schema_loader import Schema


class _ClampedExecutor:
    """Read-only executor over the clamped shadow. safe_execute only —
    never use for introspection or eval scoring."""

    dialect = "sqlite"

    def __init__(self, conn):
        self.conn = conn

    def safe_execute(self, sql: str, timeout_sec: float = 30.0, max_rows: int = 10_000):
        reason = assert_safe_select(sql, dialect=self.dialect)
        if reason:
            raise UnsafeSQLError(reason)
        cur = self.conn.cursor()
        try:
            cur.execute(sql)
            rows = cur.fetchmany(max_rows)
            if cur.fetchone() is not None:
                raise HarnessError(f"result exceeds max_rows={max_rows}")
            cols = [d[0] for d in cur.description] if cur.description else []
            return rows, cols, 0.0
        finally:
            cur.close()

    def close(self) -> None:
        self.conn.close()


def build_clamped_sqlite(
    source_path: Path, schema: Schema, holdout: HoldoutConfig, cutoff_iso: str
) -> _ClampedExecutor:
    """In-memory shadow with the holdout window removed.

    Copies data (original DDL preserved => column affinity preserved =>
    canonical hashes comparable) rather than using cross-db views, which
    SQLite forbids. Dim tables copy in full; the holdout table filters to
    pre-cutoff rows; fact children filter to surviving parents.
    """
    cutoff_lit = "'" + cutoff_iso.replace("'", "''") + "'"
    src = sqlite3.connect(f"file:{source_path}?mode=ro", uri=True)
    mem = sqlite3.connect(":memory:")

    fact_children = {
        e.table: e for e in schema.fk_edges if e.ref_table == holdout.table
    }

    try:
        for tname in schema.tables:
            ddl_row = src.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name = ?", (tname,)
            ).fetchone()
            if ddl_row is None or not ddl_row[0]:
                continue
            mem.execute(ddl_row[0])

            if tname == holdout.table:
                sel = f'SELECT * FROM "{tname}" WHERE "{holdout.col}" < {cutoff_lit}'
            elif tname in fact_children:
                e = fact_children[tname]
                sel = (
                    f'SELECT * FROM "{tname}" WHERE "{e.column}" IN '
                    f'(SELECT "{e.ref_column}" FROM "{holdout.table}" '
                    f'WHERE "{holdout.col}" < {cutoff_lit})'
                )
            else:
                sel = f'SELECT * FROM "{tname}"'

            rows = src.execute(sel).fetchall()
            n_cols = len(rows[0]) if rows else len(src.execute(f'PRAGMA table_info("{tname}")').fetchall())
            if rows:
                mem.executemany(
                    f'INSERT INTO "{tname}" VALUES ({", ".join("?" * n_cols)})', rows
                )
    finally:
        src.close()
    return _ClampedExecutor(mem)


class _PostgresClampedExecutor(_ClampedExecutor):
    dialect = "postgres"

    def safe_execute(self, sql: str, timeout_sec: float = 30.0, max_rows: int = 10_000):
        reason = assert_safe_select(sql, dialect=self.dialect)
        if reason:
            raise UnsafeSQLError(reason)
        cur = self.conn.cursor()
        try:
            cur.execute(sql)
            rows = cur.fetchmany(max_rows)
            if cur.fetchone() is not None:
                raise HarnessError(f"result exceeds max_rows={max_rows}")
            cols = [d.name for d in cur.description] if cur.description else []
            return rows, cols, 0.0
        finally:
            cur.close()


def build_clamped_postgres(
    uri: str, schema: Schema, holdout: HoldoutConfig, cutoff_iso: str
) -> _PostgresClampedExecutor:
    import psycopg

    conn = psycopg.connect(uri)
    # unqualified names resolve pg_temp first => temp views shadow public tables
    conn.execute("SET search_path TO pg_temp, public")
    conn.execute("SET statement_timeout = 30000")
    conn.execute(
        f'CREATE TEMP VIEW "{holdout.table}" AS '
        f'SELECT * FROM public."{holdout.table}" WHERE "{holdout.col}" < %s',
        (cutoff_iso,),
    )
    for edge in schema.fk_edges:
        if edge.ref_table == holdout.table:
            conn.execute(
                f'CREATE TEMP VIEW "{edge.table}" AS '
                f'SELECT * FROM public."{edge.table}" WHERE "{edge.column}" IN '
                f'(SELECT "{edge.ref_column}" FROM public."{holdout.table}" '
                f'WHERE "{holdout.col}" < %s)',
                (cutoff_iso,),
            )
    return _PostgresClampedExecutor(conn)


def build_clamped(adapter, schema: Schema, holdout: HoldoutConfig, cutoff_iso: str):
    """Dispatch by adapter dialect. SQLite is the dev/test path; Postgres is
    the canonical-run path (same semantics)."""
    if adapter.dialect == "sqlite":
        return build_clamped_sqlite(adapter.db_path, schema, holdout, cutoff_iso)
    if adapter.dialect == "postgres":
        return build_clamped_postgres(adapter.uri, schema, holdout, cutoff_iso)
    raise HarnessError(f"no clamped executor for dialect: {adapter.dialect}")


def window_dependent(gold_sql: str, full_sha256: str, clamped) -> bool:
    """True iff executing gold against the clamped shadow changes the result.

    This is THE mechanical window rule (plan §4.4): no judgment, no predicate
    injection — just execution on both sides of the cutoff.
    """
    rows, cols, _ = clamped.safe_execute(gold_sql)
    clamped_sha, _n = canonical_rows_hash(rows, cols)
    return clamped_sha != full_sha256