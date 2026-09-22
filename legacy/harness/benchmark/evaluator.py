#!/usr/bin/env python3
"""
harness/benchmark/evaluator.py -- Text-to-SQL Benchmark Evaluation Harness
"""

from __future__ import annotations

import csv
import json
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from ..security.sandbox import execute_sandboxed_query


@dataclass
class BenchmarkItem:
    item_id: str
    question: str
    gold_sql: str
    db_path: str


@dataclass
class ItemEvaluation:
    item_id: str
    question: str
    gold_sql: str
    pred_sql: str
    valid_sql: bool
    execution_match: bool
    exact_match: bool
    was_repaired: bool
    latency_ms: float
    error: Optional[str] = None


@dataclass
class BenchmarkSummary:
    total_items: int
    valid_sql_count: int
    valid_sql_rate: float
    execution_match_count: int
    execution_accuracy: float
    exact_match_count: int
    exact_match_rate: float
    repairs_attempted: int
    repairs_succeeded: int
    repair_recovery_rate: float
    avg_latency_ms: float
    item_results: List[ItemEvaluation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_items": self.total_items,
            "valid_sql_rate": round(self.valid_sql_rate * 100, 2),
            "execution_accuracy": round(self.execution_accuracy * 100, 2),
            "exact_match_rate": round(self.exact_match_rate * 100, 2),
            "repair_recovery_rate": round(self.repair_recovery_rate * 100, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }
    def to_csv(self, filepath: str) -> None:
        """Dumps detailed per-sample benchmark results to a CSV file."""
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["item_id", "question", "gold_sql", "pred_sql", "valid_sql", "execution_match", "exact_match", "latency_ms"])
            for r in self.item_results:
                writer.writerow([r.item_id, r.question, r.gold_sql, r.pred_sql, r.valid_sql, r.execution_match, r.exact_match, r.latency_ms])



def _compare_results(res_gold: Dict[str, Any], res_pred: Dict[str, Any]) -> bool:
    """Compares two query result sets for set equivalence."""
    if not res_gold["success"] or not res_pred["success"]:
        return False

    rows_gold = res_gold["rows"]
    rows_pred = res_pred["rows"]

    if len(rows_gold) != len(rows_pred):
        return False

    # Convert rows to string sets for safe comparison across numeric types
    set_gold = collections_counter(rows_gold)
    set_pred = collections_counter(rows_pred)
    return set_gold == set_pred


def collections_counter(rows: List[Tuple[Any, ...]]) -> Dict[str, int]:
    cnt = {}
    for r in rows:
        key = str(tuple(str(x) if x is not None else "NULL" for x in r))
        cnt[key] = cnt.get(key, 0) + 1
    return cnt


class BenchmarkEvaluator:
    """
    Automated evaluation harness for running Text-to-SQL agents against standard benchmarks
    and computing standard metrics (Execution Accuracy, Valid Rate, Latency).
    """

    def __init__(self, timeout_sec: float = 3.0):
        self.timeout_sec = timeout_sec

    def evaluate(
        self,
        items: List[BenchmarkItem],
        agent_predict_fn: Callable[[str, str], Dict[str, Any]],
    ) -> BenchmarkSummary:
        """
        Runs evaluation over items.
        agent_predict_fn takes (db_path, question) and returns agent query dict.
        """
        total = len(items)
        if total == 0:
            return BenchmarkSummary(0, 0, 0.0, 0, 0.0, 0, 0.0, 0, 0, 0.0, 0.0)

        results: List[ItemEvaluation] = []
        repairs_attempted = 0
        repairs_succeeded = 0

        for it in items:
            t0 = time.perf_counter()
            agent_res = agent_predict_fn(it.db_path, it.question)
            pred_latency = (time.perf_counter() - t0) * 1000

            pred_sql = agent_res.get("sql", "")
            was_repaired = bool(agent_res.get("repaired_from"))

            if agent_res.get("was_repaired"):
                repairs_attempted += 1
                if agent_res.get("success"):
                    repairs_succeeded += 1

            # Execute gold
            gold_res = execute_sandboxed_query(it.db_path, it.gold_sql, timeout_sec=self.timeout_sec)
            pred_res = execute_sandboxed_query(it.db_path, pred_sql, timeout_sec=self.timeout_sec)

            valid_sql = pred_res["success"]
            ex_match = _compare_results(gold_res, pred_res)
            em_match = (
                pred_sql.strip().rstrip(";").lower() == it.gold_sql.strip().rstrip(";").lower()
            )

            results.append(
                ItemEvaluation(
                    item_id=it.item_id,
                    question=it.question,
                    gold_sql=it.gold_sql,
                    pred_sql=pred_sql,
                    valid_sql=valid_sql,
                    execution_match=ex_match,
                    exact_match=em_match,
                    was_repaired=was_repaired,
                    latency_ms=pred_latency,
                    error=pred_res["error"],
                )
            )

        valid_count = sum(1 for r in results if r.valid_sql)
        ex_count = sum(1 for r in results if r.execution_match)
        em_count = sum(1 for r in results if r.exact_match)
        avg_lat = sum(r.latency_ms for r in results) / total

        return BenchmarkSummary(
            total_items=total,
            valid_sql_count=valid_count,
            valid_sql_rate=valid_count / total,
            execution_match_count=ex_count,
            execution_accuracy=ex_count / total,
            exact_match_count=em_count,
            exact_match_rate=em_count / total,
            repairs_attempted=repairs_attempted,
            repairs_succeeded=repairs_succeeded,
            repair_recovery_rate=(repairs_succeeded / repairs_attempted) if repairs_attempted > 0 else 1.0,
            avg_latency_ms=avg_lat,
            item_results=results,
        )
