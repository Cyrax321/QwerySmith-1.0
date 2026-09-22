#!/usr/bin/env python3
"""
harness/decoding/selector.py -- Execution-Guided Candidate Selection & Result Consensus Voting
"""

from __future__ import annotations

import collections
import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..security.sandbox import execute_sandboxed_query


@dataclass
class CandidateEvaluation:
    sql: str
    success: bool
    columns: List[str]
    rows: List[Tuple[Any, ...]]
    error: Optional[str]
    latency_ms: float
    result_hash: str


@dataclass
class CandidateSelectionResult:
    chosen_sql: str
    columns: List[str]
    rows: List[Tuple[Any, ...]]
    success: bool
    error: Optional[str]
    candidates_evaluated: int
    candidates_valid: int
    consensus_count: int
    consensus_ratio: float
    latency_exec_ms: float
    all_evaluations: List[CandidateEvaluation] = field(default_factory=list)


def _compute_result_hash(columns: List[str], rows: List[Tuple[Any, ...]]) -> str:
    """Computes a normalized hash representing the semantic result set."""
    if not rows:
        return "EMPTY_RESULT_SET"

    # Normalize values for hashing
    norm_rows = []
    for r in rows:
        norm_r = tuple(str(x) if x is not None else "NULL" for x in r)
        norm_rows.append(norm_r)

    # Sort rows to be invariant to ORDER BY differences unless order was significant
    norm_rows.sort()
    serialized = json.dumps([columns, norm_rows], sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


class ExecutionGuidedSelector:
    """
    Evaluates candidate SQL queries inside the database sandbox and selects the best candidate
    using execution-guided validation, empty-set penalization, and majority consensus voting.
    """

    def __init__(self, timeout_sec: float = 3.0, max_rows: int = 100):
        self.timeout_sec = timeout_sec
        self.max_rows = max_rows

    def select(
        self,
        conn_or_path: Union[sqlite3.Connection, str, Path],
        candidates: List[str],
        read_only: bool = True,
    ) -> CandidateSelectionResult:
        """
        Executes and ranks candidates, selecting the consensus or highest-validity query.
        """
        if not candidates:
            return CandidateSelectionResult(
                chosen_sql="",
                columns=[],
                rows=[],
                success=False,
                error="No candidate queries provided.",
                candidates_evaluated=0,
                candidates_valid=0,
                consensus_count=0,
                consensus_ratio=0.0,
                latency_exec_ms=0.0,
            )

        evaluations: List[CandidateEvaluation] = []
        valid_evals: List[CandidateEvaluation] = []

        for sql in candidates:
            exec_res = execute_sandboxed_query(
                conn_or_path,
                sql,
                max_rows=self.max_rows,
                timeout_sec=self.timeout_sec,
                read_only=read_only,
            )

            res_hash = _compute_result_hash(exec_res["columns"], exec_res["rows"]) if exec_res["success"] else "EXEC_ERROR"
            ev = CandidateEvaluation(
                sql=sql,
                success=exec_res["success"],
                columns=exec_res["columns"],
                rows=exec_res["rows"],
                error=exec_res["error"],
                latency_ms=exec_res["latency_exec_ms"],
                result_hash=res_hash,
            )
            evaluations.append(ev)
            if ev.success:
                valid_evals.append(ev)

        # 1. If no valid candidates, return the first failure
        if not valid_evals:
            first_fail = evaluations[0]
            return CandidateSelectionResult(
                chosen_sql=first_fail.sql,
                columns=[],
                rows=[],
                success=False,
                error=first_fail.error,
                candidates_evaluated=len(evaluations),
                candidates_valid=0,
                consensus_count=0,
                consensus_ratio=0.0,
                latency_exec_ms=first_fail.latency_ms,
                all_evaluations=evaluations,
            )

        # 2. Fast-path: single valid candidate
        if len(valid_evals) == 1:
            best = valid_evals[0]
            return CandidateSelectionResult(
                chosen_sql=best.sql,
                columns=best.columns,
                rows=best.rows,
                success=True,
                error=None,
                candidates_evaluated=len(evaluations),
                candidates_valid=1,
                consensus_count=1,
                consensus_ratio=1.0,
                latency_exec_ms=best.latency_ms,
                all_evaluations=evaluations,
            )

        # 3. Majority Consensus Voting on Non-Empty Result Sets
        # Group valid candidates by result hash
        hash_clusters: Dict[str, List[CandidateEvaluation]] = collections.defaultdict(list)
        for ev in valid_evals:
            hash_clusters[ev.result_hash].append(ev)

        # Rank clusters: prefer non-empty clusters first, then larger cluster size
        def cluster_key(item: Tuple[str, List[CandidateEvaluation]]):
            r_hash, ev_list = item
            is_non_empty = 1 if r_hash != "EMPTY_RESULT_SET" and ev_list[0].rows else 0
            return (is_non_empty, len(ev_list), -ev_list[0].latency_ms)

        sorted_clusters = sorted(hash_clusters.items(), key=cluster_key, reverse=True)
        winning_hash, winning_candidates = sorted_clusters[0]

        # Pick candidate within winning cluster that has cleanest syntax (e.g. shortest or fastest)
        chosen = min(winning_candidates, key=lambda c: (len(c.sql), c.latency_ms))

        return CandidateSelectionResult(
            chosen_sql=chosen.sql,
            columns=chosen.columns,
            rows=chosen.rows,
            success=True,
            error=None,
            candidates_evaluated=len(evaluations),
            candidates_valid=len(valid_evals),
            consensus_count=len(winning_candidates),
            consensus_ratio=len(winning_candidates) / len(valid_evals),
            latency_exec_ms=chosen.latency_ms,
            all_evaluations=evaluations,
        )


def query_complexity(sql: str) -> int:
    """Computes a structural complexity score based on tokens, clauses, and subqueries."""
    tokens = sql.strip().split()
    score = len(tokens)
    upper = sql.upper()
    score += upper.count("JOIN") * 5
    score += upper.count("SELECT") * 3
    score += upper.count("WHERE") * 2
    return score
