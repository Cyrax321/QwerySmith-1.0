"""
harness/adapters -- Pluggable Database Dialect Drivers (SQLite, DuckDB, etc.).
"""

from .base import DatabaseAdapter, QueryResult
from .sqlite_adapter import SQLiteAdapter
from .duckdb_adapter import DuckDBAdapter

__all__ = [
    "DatabaseAdapter",
    "QueryResult",
    "SQLiteAdapter",
    "DuckDBAdapter",
]
