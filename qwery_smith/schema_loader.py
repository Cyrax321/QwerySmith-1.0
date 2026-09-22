"""Schema loader (plan §3.1): introspection -> canonical DDL + FK graph.

Deterministic output (sorted tables, stable column order) so the DDL text is
SHA-256 hashable — the schema shown to the model is a versioned artifact.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .adapters.base import DatabaseAdapter

# Canonical type aliases (plan §3.1)
_CANON = {
    "VARCHAR": "TEXT", "CHAR": "TEXT", "TEXT": "TEXT",
    "INT": "INTEGER", "INTEGER": "INTEGER", "BIGINT": "BIGINT",
    "SMALLINT": "INTEGER", "REAL": "REAL", "FLOAT": "REAL",
    "DOUBLE": "REAL", "NUMERIC": "NUMERIC", "DECIMAL": "NUMERIC",
    "BOOLEAN": "BOOLEAN", "DATE": "DATE", "TIMESTAMP": "TIMESTAMP",
    "DATETIME": "TIMESTAMP",
}


@dataclass(frozen=True)
class TableSchema:
    name: str
    columns: tuple[tuple[str, str, bool], ...]   # (name, canonical_type, not_null)
    primary_key: tuple[str, ...] = ()
    row_count: int = 0


@dataclass(frozen=True)
class FKEdge:
    table: str
    column: str
    ref_table: str
    ref_column: str


@dataclass
class Schema:
    tables: dict[str, TableSchema] = field(default_factory=dict)
    fk_edges: tuple[FKEdge, ...] = ()

    def ddl_text(self) -> str:
        """The exact text shown to the model (plan §3.1) — deterministic."""
        blocks: list[str] = []
        for tname in sorted(self.tables):
            t = self.tables[tname]
            cols = []
            pk = set(self._pk_index(t))
            for name, ctype, not_null in t.columns:
                spec = f"  {name} {ctype}"
                if not_null:
                    spec += " NOT NULL"
                if name in pk:
                    spec += " -- pk"
                cols.append(spec)
            for edge in sorted(self.fk_edges, key=lambda e: (e.table, e.column)):
                if edge.table == tname:
                    cols.append(f"  -- fk: {edge.column} -> {edge.ref_table}.{edge.ref_column}")
            blocks.append(f"CREATE TABLE {tname} (\n" + ",\n".join(cols) + "\n);")
        return "\n\n".join(blocks)

    def ddl_sha256(self) -> str:
        return hashlib.sha256(self.ddl_text().encode()).hexdigest()

    def fingerprint(self) -> str:
        """Hash of DDL + per-table row counts (question manifest's db_fingerprint)."""
        h = hashlib.sha256()
        h.update(self.ddl_text().encode())
        for tname in sorted(self.tables):
            h.update(f"{tname}:{self.tables[tname].row_count}".encode())
        return h.hexdigest()

    def _pk_index(self, t: TableSchema) -> list[str]:
        return list(t.primary_key)

    def fk_graph(self) -> dict[str, list[FKEdge]]:
        out: dict[str, list[FKEdge]] = {}
        for e in self.fk_edges:
            out.setdefault(e.table, []).append(e)
        return out


def load_schema(adapter: DatabaseAdapter) -> Schema:
    """Introspect a prepared database into a canonical Schema."""
    tables: dict[str, TableSchema] = {}
    for tname in adapter.table_names():
        cols = tuple(
            (name, _CANON.get(dtype.upper(), "TEXT"), not_null)
            for name, dtype, not_null in adapter.table_columns(tname)
        )
        pk = _read_pk(adapter, tname)
        tables[tname] = TableSchema(
            name=tname,
            columns=cols,
            primary_key=tuple(pk),
            row_count=adapter.row_count(tname),
        )

    edges: list[FKEdge] = []
    for tname in sorted(tables):
        for col, ref_table, ref_col in adapter.foreign_keys(tname):
            edges.append(FKEdge(table=tname, column=col, ref_table=ref_table, ref_column=ref_col))

    return Schema(tables=tables, fk_edges=tuple(edges))


def _read_pk(adapter: DatabaseAdapter, tname: str) -> list[str]:
    """Read declared PKs. SQLite: PRAGMA pk index or __harness_meta_pk; PG: pg_index."""
    pk: list[str] = []
    if adapter.dialect == "sqlite":
        conn = getattr(adapter, "conn", None)
        if conn is not None:
            pk = [r[1] for r in conn.execute(f'PRAGMA table_info("{tname}")').fetchall() if r[5] > 0]
        if not pk:
            meta = getattr(adapter, "_meta", None)
            if meta:
                rows = adapter._meta("__harness_meta_pk", tname, ("cols",))
                if rows:
                    pk = [c for c in rows[0][0].split(",") if c]
    else:
        row = adapter.scalar(
            "SELECT a.attname FROM pg_index i "
            "JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
            "WHERE i.indrelid = %s::regclass AND i.indisprimary",
        ) if hasattr(adapter, "dialect") and adapter.dialect == "postgres" else None
        if row:
            pk = [r[0] for r in (row if isinstance(row, list) else [row])]
    return pk


def parse_pk_meta(adapter: DatabaseAdapter) -> dict[str, tuple[str, ...]]:
    """Read declared PKs (SQLite metadata-table variant used at ingest)."""
    out: dict[str, tuple[str, ...]] = {}
    for tname in adapter.table_names():
        try:
            row = adapter.scalar(
                f"SELECT cols FROM __harness_meta_pk WHERE table_name = '{tname}'"
            ) if hasattr(adapter, "_meta") else None
        except Exception:
            row = None
        if row and row[0]:
            out[tname] = tuple(row[0].split(","))
    return out