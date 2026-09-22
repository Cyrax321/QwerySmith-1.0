"""Validator (plan §3.2): a release gate, not a linter.

Checks, in order:
  1. gold SQL executes in the sandbox (=> scorable denominator)
  2. expected_rows hash matches re-canonicalized gold result (drift detection)
  3. holdout-leak check (plan §4.4): every train_ok question's gold SQL,
     guarded with a cutoff predicate, must return identical rows
  4. manifest counts match question file
  5. enum fields legal; temporal-term lint on train_ok questions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .adapters.base import DatabaseAdapter
from .config import HoldoutConfig
from .exceptions import HoldoutLeakError
from .profiler import compute_holdout_cutoff
from .questions import (
    CATEGORIES,
    DIFFICULTIES,
    SOURCES,
    SPLITS,
    Question,
    QuestionSet,
    canonical_rows_hash,
)

_TEMPORAL_TERMS = ("last ", "recent", "latest", "past ", "current month", "current year")


@dataclass
class QuestionFailure:
    id: str
    check: str
    detail: str


@dataclass
class ValidationResult:
    scorable: int
    excluded: list[QuestionFailure] = field(default_factory=list)
    warnings: list[QuestionFailure] = field(default_factory=list)
    split_counts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return len(self.excluded) == 0

    def summary(self) -> str:
        lines = [
            f"scorable={self.scorable} excluded={len(self.excluded)} warnings={len(self.warnings)}"
        ]
        for f in self.excluded:
            lines.append(f"  EXCLUDED {f.id} [{f.check}]: {f.detail}")
        for f in self.warnings:
            lines.append(f"  warn {f.id} [{f.check}]: {f.detail}")
        return "\n".join(lines)


def _leak_guarded_sql(q: Question, holdout: HoldoutConfig, cutoff: str) -> str:
    """Plan §4.4 check #1: inject a cutoff guard into gold SQL via sqlglot.

    Builds `... AND <table>.<col> < cutoff` on the outermost SELECT (adds a
    WHERE if absent). If the guarded result differs from the stored result,
    the question can see the held-out window => leak.
    """
    import sqlglot
    from sqlglot import exp

    tree = sqlglot.parse_one(q.gold_sql, read="postgres")
    select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if select is None:
        raise HoldoutLeakError(f"{q.id}: cannot find SELECT to guard")

    lt = exp.LT(
        this=exp.column(holdout.col, table=holdout.table),
        expression=exp.Literal.string(cutoff),
    )
    where = select.args.get("where")
    if where is None:
        select.where(lt, copy=False)
    else:
        select.where(exp.and_(where.this, lt), copy=False)
    return tree.sql(dialect="postgres")


def check_holdout_leak(
    qs: QuestionSet,
    adapter: DatabaseAdapter,
    holdout: HoldoutConfig,
    cutoff_iso: str,
) -> list[QuestionFailure]:
    """§4.4: for each train_ok question, guarded gold must match stored hash."""
    failures: list[QuestionFailure] = []
    for q in qs.questions:
        if q.split != "train_ok":
            continue
        try:
            guarded = _leak_guarded_sql(q, holdout, cutoff_iso)
            rows, cols, _ = adapter.safe_execute(guarded, timeout_sec=30.0)
            sha, _n = canonical_rows_hash(rows, cols)
        except HoldoutLeakError as e:
            failures.append(QuestionFailure(q.id, "holdout_leak", str(e)))
            continue
        except Exception as e:
            failures.append(QuestionFailure(q.id, "holdout_leak", f"guarded gold failed: {type(e).__name__}: {e}"))
            continue
        if sha != q.expected_rows.sha256:
            failures.append(QuestionFailure(
                q.id, "holdout_leak",
                "guarded result differs from stored => question reads the held-out window",
            ))
    return failures


def validate_question_set(
    qs: QuestionSet,
    adapter: DatabaseAdapter,
    holdout: Optional[HoldoutConfig] = None,
    cutoff_iso: Optional[str] = None,
) -> ValidationResult:
    res = ValidationResult(scorable=0)

    if not qs.ids_unique():
        dupes = sorted({q.id for q in qs.questions if [x.id for x in qs.questions].count(q.id) > 1})
        res.excluded.append(QuestionFailure("—", "duplicate_ids", f"{len(dupes)} duplicate ids: {dupes[:5]}"))

    if cutoff_iso is None and holdout is not None:
        _max, cutoff_iso = compute_holdout_cutoff(adapter, holdout)

    for q in qs.questions:
        # enums
        if q.difficulty not in DIFFICULTIES:
            res.excluded.append(QuestionFailure(q.id, "difficulty", q.difficulty))
            continue
        if q.split not in SPLITS:
            res.excluded.append(QuestionFailure(q.id, "split", q.split))
            continue
        if q.category not in CATEGORIES:
            res.excluded.append(QuestionFailure(q.id, "category", q.category))
            continue
        if q.source not in SOURCES:
            res.excluded.append(QuestionFailure(q.id, "source", q.source))
            continue

        # 1. scorable check + 2. drift check
        try:
            rows, cols, _lat = adapter.safe_execute(q.gold_sql, timeout_sec=30.0)
        except Exception as e:
            res.excluded.append(QuestionFailure(q.id, "gold_sql", f"{type(e).__name__}: {e}"))
            continue
        sha, n = canonical_rows_hash(rows, cols)
        if sha != q.expected_rows.sha256:
            res.excluded.append(QuestionFailure(
                q.id, "expected_rows_drift",
                f"stored={q.expected_rows.sha256[:12]} recomputed={sha[:12]}",
            ))
            continue

        # 5. temporal-term lint on train_ok
        if q.split == "train_ok" and any(t in q.question.lower() for t in _TEMPORAL_TERMS):
            res.warnings.append(QuestionFailure(q.id, "temporal_term", "train_ok question mentions relative time"))

        res.scorable += 1
        res.split_counts[q.split] = res.split_counts.get(q.split, 0) + 1

    # 3. holdout-leak check (§4.4) — train_ok questions only
    if holdout is not None:
        res.excluded.extend(check_holdout_leak(qs, adapter, holdout, cutoff_iso))
        # recompute scorable: leaked questions must not count
        leaked_ids = {f.id for f in res.excluded if f.check == "holdout_leak"}
        if leaked_ids:
            res.scorable = sum(1 for q in qs.questions if q.id not in leaked_ids)
            for split in list(res.split_counts):
                res.split_counts[split] = sum(
                    1 for q in qs.questions
                    if q.split == split and q.id not in leaked_ids
                )

    return res


# -- ingest ---------------------------------------------------------------


def ingest_dataset(cfg, adapter) -> dict[str, int]:
    """Load raw CSVs into the database per datasource config (plan §4.1).

    Returns {table: row_count}. PKs/FKs are dataset config, not code.
    """
    import csv as _csv
    from pathlib import Path

    counts: dict[str, int] = {}
    for csv_stem, table in cfg.datasource.table_map.items():
        csv_path = Path(cfg.datasource.csv_dir) / f"{csv_stem}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        cols = cfg.datasource.extra.get("columns", {}).get(table)
        if not cols:
            with open(csv_path, newline="", encoding="utf-8-sig") as f:
                cols = {h: "TEXT" for h in next(_csv.reader(f))}
        counts[table] = adapter.load_csv(table, csv_path, cols)

    for table, pk_cols in cfg.datasource.extra.get("primary_keys", {}).items():
        adapter.add_primary_key(table, pk_cols)
    for spec in cfg.datasource.extra.get("foreign_keys", []):
        adapter.add_foreign_key(spec["table"], spec["column"], spec["ref_table"], spec["ref_column"])

    return counts