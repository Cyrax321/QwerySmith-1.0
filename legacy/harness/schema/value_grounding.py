#!/usr/bin/env python3
"""
harness/schema/value_grounding.py -- Column Value Introspection & Literal Grounding Index
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


@dataclass
class GroundedValueMatch:
    matched_term: str
    table_name: str
    column_name: str
    exact_db_value: str
    confidence: float


class ValueGrounder:
    """
    Introspects low-to-medium cardinality database columns to catalog distinct string literals.
    Matches terms in natural language questions to exact database values.
    """

    def __init__(
        self,
        conn_or_path: Union[sqlite3.Connection, str, Path],
        max_distinct_per_col: int = 50,
        max_val_len: int = 60,
    ):
        self.value_index: Dict[str, List[Tuple[str, str, str]]] = {}  # lower_val -> [(table, col, exact_val)]
        self._build_index(conn_or_path, max_distinct_per_col, max_val_len)

    def _build_index(
        self,
        conn_or_path: Union[sqlite3.Connection, str, Path],
        max_distinct: int,
        max_len: int,
    ) -> None:
        """Indexes categorical string columns across all user tables."""
        should_close = False
        if isinstance(conn_or_path, sqlite3.Connection):
            conn = conn_or_path
        else:
            conn = sqlite3.connect(str(conn_or_path))
            should_close = True

        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [r[0] for r in cur.fetchall()]

        for tbl in tables:
            cur.execute(f"PRAGMA table_info({tbl});")
            cols = cur.fetchall()
            for col in cols:
                col_name = col[1]
                col_type = (col[2] or "").upper()

                # Focus on text/char/varchar/string columns
                if any(t in col_type for t in ["TEXT", "CHAR", "VARCHAR", "CLOB"]) or col_type == "":
                    try:
                        # Check cardinality
                        cur.execute(f"SELECT COUNT(DISTINCT {col_name}) FROM {tbl};")
                        distinct_count = cur.fetchone()[0]
                        if 0 < distinct_count <= max_distinct:
                            cur.execute(f"SELECT DISTINCT {col_name} FROM {tbl} WHERE {col_name} IS NOT NULL LIMIT {max_distinct};")
                            vals = [r[0] for r in cur.fetchall() if r[0] is not None]
                            for v in vals:
                                str_v = str(v).strip()
                                if 1 <= len(str_v) <= max_len:
                                    key = str_v.lower()
                                    if key not in self.value_index:
                                        self.value_index[key] = []
                                    self.value_index[key].append((tbl, col_name, str_v))
                    except Exception:
                        continue

        if should_close:
            conn.close()

    def ground(self, question: str) -> List[GroundedValueMatch]:
        """
        Searches user question for occurrences of indexed database values (up to 4-grams).
        """
        words = re.findall(r"[a-zA-Z0-9_'-]+", question.lower())
        matches: List[GroundedValueMatch] = []
        seen = set()

        n_words = len(words)
        # Check n-grams from 4 down to 1
        for n in range(min(4, n_words), 0, -1):
            for i in range(n_words - n + 1):
                ngram = " ".join(words[i:i + n])
                if ngram in self.value_index:
                    for tbl, col, exact_val in self.value_index[ngram]:
                        key = (tbl, col, exact_val)
                        if key not in seen:
                            seen.add(key)
                            matches.append(
                                GroundedValueMatch(
                                    matched_term=ngram,
                                    table_name=tbl,
                                    column_name=col,
                                    exact_db_value=exact_val,
                                    confidence=1.0 if ngram == exact_val.lower() else 0.85,
                                )
                            )

        return matches

    def format_grounding_hints(self, matches: List[GroundedValueMatch]) -> str:
        """Formats grounding matches as a prompt injection snippet."""
        if not matches:
            return ""

        hints = []
        for m in matches[:6]:  # Keep top 6 most relevant
            hints.append(f"- '{m.matched_term}' corresponds to {m.table_name}.{m.column_name} = '{m.exact_db_value}'")

        return "Database Value Grounding Hints:\n" + "\n".join(hints)
