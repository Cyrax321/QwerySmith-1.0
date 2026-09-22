#!/usr/bin/env python3
"""
harness/adapters/base.py -- Abstract Database Adapter Interface
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class QueryResult:
    columns: List[str]
    rows: List[Tuple[Any, ...]]
    latency_exec_ms: float
    error: Optional[str] = None
    success: bool = True
    timed_out: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "columns": self.columns,
            "rows": self.rows,
            "latency_exec_ms": self.latency_exec_ms,
            "error": self.error,
            "success": self.success,
            "timed_out": self.timed_out,
        }


class DatabaseAdapter(ABC):
    """Abstract interface defining required capabilities for any SQL database backend."""

    @abstractmethod
    def execute(
        self,
        sql: str,
        timeout_sec: float = 3.0,
        max_rows: int = 100,
        read_only: bool = True,
    ) -> QueryResult:
        """Executes a SQL query within the sandbox."""
        pass

    @abstractmethod
    def get_schema(self) -> str:
        """Returns table definitions and DDL for all user tables."""
        pass

    @abstractmethod
    def get_tables(self) -> List[str]:
        """Returns list of active table names."""
        pass

    @abstractmethod
    def get_table_counts(self) -> Dict[str, int]:
        """Returns row counts for each active table."""
        pass

    @abstractmethod
    def get_table_sample(self, table_name: str, limit: int = 3) -> Dict[str, Any]:
        """Fetches preview rows for a specific table."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Releases all open resources and connections."""
        pass
