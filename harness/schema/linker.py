#!/usr/bin/env python3
"""
harness/schema/linker.py -- Subgraph Schema Linking & Relevance Pruning
"""

from __future__ import annotations

import collections
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


@dataclass
class ForeignKeyRelation:
    from_table: str
    from_col: str
    to_table: str
    to_col: str


@dataclass
class TableNode:
    name: str
    ddl: str
    columns: List[str] = field(default_factory=list)
    foreign_keys: List[ForeignKeyRelation] = field(default_factory=list)


@dataclass
class PrunedSchema:
    selected_tables: List[str]
    pruned_ddl: str
    foreign_key_hints: List[str]
    total_tables_count: int
    was_pruned: bool


class SchemaLinker:
    """
    Analyzes questions against database catalogs, scores relevance of tables and columns,
    expands the foreign key graph to ensure join-completeness, and prunes unused schema noise.
    """

    def __init__(self, conn_or_path: Union[sqlite3.Connection, str, Path], table_descriptions: Optional[Dict[str, str]] = None):
        self.table_descriptions: Dict[str, str] = table_descriptions or {}
        self.tables: Dict[str, TableNode] = {}
        self.adj_graph: Dict[str, Set[str]] = collections.defaultdict(set)
        self.fk_relations: List[ForeignKeyRelation] = []
        self._introspect(conn_or_path)

    def _introspect(self, conn_or_path: Union[sqlite3.Connection, str, Path]) -> None:
        """Extracts schema, columns, and foreign keys from the target SQLite database."""
        should_close = False
        if isinstance(conn_or_path, sqlite3.Connection):
            conn = conn_or_path
        else:
            conn = sqlite3.connect(str(conn_or_path))
            should_close = True

        cur = conn.cursor()
        cur.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        rows = cur.fetchall()

        for tbl_name, ddl in rows:
            if not ddl:
                continue

            # Get columns
            cur.execute(f"PRAGMA table_info({tbl_name});")
            col_info = cur.fetchall()
            columns = [c[1] for c in col_info]

            # Get foreign keys
            cur.execute(f"PRAGMA foreign_key_list({tbl_name});")
            fk_info = cur.fetchall()
            fks = []
            for fk in fk_info:
                # fk format: (id, seq, to_table, from_col, to_col, on_update, on_delete, match)
                to_tbl = fk[2]
                from_col = fk[3]
                to_col = fk[4]
                rel = ForeignKeyRelation(
                    from_table=tbl_name,
                    from_col=from_col,
                    to_table=to_tbl,
                    to_col=to_col,
                )
                fks.append(rel)
                self.fk_relations.append(rel)
                self.adj_graph[tbl_name].add(to_tbl)
                self.adj_graph[to_tbl].add(tbl_name)

            self.tables[tbl_name] = TableNode(
                name=tbl_name,
                ddl=ddl.strip(),
                columns=columns,
                foreign_keys=fks,
            )

        if should_close:
            conn.close()

    def _tokenize(self, text: str) -> Set[str]:
        """Alphanumeric tokenizer splitting on whitespace and underscores with lowercase normalization."""
        words = re.findall(r"[a-zA-Z0-9]+", text.lower())
        tokens = set(words)
        for w in list(tokens):
            if w.endswith("s") and len(w) > 3:
                tokens.add(w[:-1])
        return tokens

    def score_table_relevance(self, question: str, table_node: TableNode) -> float:
        """Scores how relevant a table is to the user's question."""
        q_tokens = self._tokenize(question)
        score = 0.0

        tbl_tokens = self._tokenize(table_node.name)
        if tbl_tokens & q_tokens:
            score += 5.0  # Strong signal if table name matches question

        col_matches = 0
        for col in table_node.columns:
            c_tokens = self._tokenize(col)
            if c_tokens & q_tokens:
                col_matches += 1

        score += col_matches * 1.5
        
        if getattr(self, "table_descriptions", None):
            desc = self.table_descriptions.get(table_node.name, "")
            if desc and (self._tokenize(desc) & q_tokens):
                score += 4.0

        return score

    def _shortest_path(self, start: str, target: str) -> List[str]:
        """Finds shortest path in the foreign key graph between two tables using BFS."""
        if start == target:
            return [start]
        queue = collections.deque([[start]])
        visited = {start}
        while queue:
            path = queue.popleft()
            node = path[-1]
            for neighbor in self.adj_graph.get(node, []):
                if neighbor == target:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        return []

    def link(self, question: str, max_tables: int = 8) -> PrunedSchema:
        """
        Selects relevant tables for the question, ensuring relational connectivity.
        """
        all_table_names = list(self.tables.keys())
        total_count = len(all_table_names)

        # If total tables is small or <= max_tables, keep all tables
        if total_count <= max_tables:
            full_ddl = "\n\n".join(t.ddl for t in self.tables.values())
            hints = [
                f"{fk.from_table}.{fk.from_col} -> {fk.to_table}.{fk.to_col}"
                for fk in self.fk_relations
            ]
            return PrunedSchema(
                selected_tables=all_table_names,
                pruned_ddl=full_ddl,
                foreign_key_hints=hints,
                total_tables_count=total_count,
                was_pruned=False,
            )

        # Score tables
        scores: List[Tuple[str, float]] = []
        for name, node in self.tables.items():
            s = self.score_table_relevance(question, node)
            scores.append((name, s))

        scores.sort(key=lambda x: x[1], reverse=True)

        # Pick candidate seed tables with score > 0
        selected = set()
        seed_limit = max_tables
        for name, s in scores:
            if s > 0 and len(selected) < seed_limit:
                selected.add(name)

        # Fallback: if no score > 0, pick top 2
        if not selected:
            selected.update(t[0] for t in scores[:2])

        # Graph expansion: add connecting tables between selected pairs
        selected_list = list(selected)
        for i in range(len(selected_list)):
            for j in range(i + 1, len(selected_list)):
                path = self._shortest_path(selected_list[i], selected_list[j])
                for node in path:
                    selected.add(node)
                    if len(selected) >= max_tables:
                        break
            if len(selected) >= max_tables:
                break

        final_tables = [t for t in all_table_names if t in selected]
        pruned_ddl = "\n\n".join(self.tables[t].ddl for t in final_tables)

        # Foreign key hints for selected tables
        hints = []
        for fk in self.fk_relations:
            if fk.from_table in selected and fk.to_table in selected:
                hints.append(f"{fk.from_table}.{fk.from_col} -> {fk.to_table}.{fk.to_col}")

        return PrunedSchema(
            selected_tables=final_tables,
            pruned_ddl=pruned_ddl,
            foreign_key_hints=hints,
            total_tables_count=total_count,
            was_pruned=len(final_tables) < total_count,
        )
