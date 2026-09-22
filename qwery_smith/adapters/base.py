"""Abstract database adapter — the seam that keeps the harness dataset-agnostic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..exceptions import UnsafeSQLError
from .guard import assert_safe_select


class DatabaseAdapter(ABC):
    """Read-only access to a prepared database.

    Implementations MUST enforce:
      - SELECT-only (AST-verified, not regex)
      - read-only transactions
      - statement timeout
    """

    dialect: str = "abstract"

    def safe_execute(
        self,
        sql: str,
        timeout_sec: float = 30.0,
        max_rows: int = 10_000,
    ) -> tuple[list[tuple[Any, ...]], list[str], float]:
        """Validate + execute SQL read-only. Returns (rows, column_names, latency_ms)."""
        reason = assert_safe_select(sql, dialect=self.dialect)
        if reason:
            raise UnsafeSQLError(reason)
        return self.execute(sql, timeout_sec=timeout_sec, max_rows=max_rows)

    @abstractmethod
    def execute(
        self,
        sql: str,
        timeout_sec: float = 30.0,
        max_rows: int = 10_000,
    ) -> tuple[list[tuple[Any, ...]], list[str], float]:
        ...

    @abstractmethod
    def load_csv(self, table: str, csv_path: Path, columns: dict[str, str], append: bool = False) -> int:
        """Load one CSV into `table` (created from `columns` name->DDL type).

        append=True: append to an existing table (grouped multi-file loads,
        e.g. UCI Online Retail II's two sheets); first file creates.
        Returns row count loaded from THIS file.
        """
        ...

    @abstractmethod
    def add_primary_key(self, table: str, columns: list[str]) -> None:
        ...

    @abstractmethod
    def add_foreign_key(self, table: str, column: str, ref_table: str, ref_column: str) -> None:
        """May no-op if orphans exist — implementations report but do not fail."""
        ...

    @abstractmethod
    def table_names(self) -> list[str]:
        ...

    @abstractmethod
    def table_columns(self, table: str) -> list[tuple[str, str, bool]]:
        """[(name, ddl_type, not_null)] in ordinal order."""
        ...

    @abstractmethod
    def foreign_keys(self, table: str) -> list[tuple[str, str, str]]:
        """[(column, ref_table, ref_column)] declared FKs."""
        ...

    @abstractmethod
    def row_count(self, table: str) -> int:
        ...

    @abstractmethod
    def scalar(self, sql: str) -> Any:
        """Execute a single-row/single-col admin/PRAGMA query (trusted, internal)."""
        ...

    @abstractmethod
    def close(self) -> None:
        ...