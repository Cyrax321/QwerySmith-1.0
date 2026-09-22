"""Scoring semantics (plan §6): EX, error taxonomy, McNemar, consistency.

Every system — base, fine-tuned, 30B, frontier — goes through the SAME
evaluate_one() path with the SAME frozen pack. No system-specific code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from .adapters.base import DatabaseAdapter
from .questions import Question, canonical_rows_hash
from .retrieval import EvidencePack
from .triples import parse_output, verify_citations

# ------------------------------------------------------------ taxonomy -----

# precedence-ordered (plan §6.5): first match wins
ERROR_CLASSES = (
    "timeout",
    "execution_error",
    "invalid_output",
    "wrong_result_missing_rows",
    "wrong_result_extra_rows",
    "wrong_result_mismatched_rows",
    "missing_citation",
    "invalid_citation",
    "unexpected_refusal",
)


@dataclass
class ScoredResult:
    question_id: str
    correct: bool
    error_class: Optional[str]          # None => correct
    reason: str                        # one-line reason (failure folders)
    pred_sql: Optional[str] = None
    pred_answer: Optional[str] = None
    latency_ms: float = 0.0
    pred_hash: Optional[str] = None

    def to_json(self) -> dict[str, Any]:
        return self.__dict__


def evaluate_one(
    q: Question,
    raw_output: str,
    pack: EvidencePack,
    adapter: DatabaseAdapter,
    timeout_sec: float = 30.0,
    latency_ms: float = 0.0,
) -> ScoredResult:
    """Score one raw model output against gold. Pure w.r.t. system identity."""
    parsed = parse_output(raw_output)

    if parsed["kind"] == "invalid_output":
        return ScoredResult(q.id, False, "invalid_output",
                            "output does not follow SQL:/ANSWER:/REFUSAL: contract",
                            latency_ms=latency_ms)

    if parsed["kind"] == "refusal":
        # refusal on an answerable question = wrong (plan §0/§6.5)
        return ScoredResult(q.id, False, "unexpected_refusal",
                            f"refused an answerable question: {parsed['refusal'][:60]}",
                            latency_ms=latency_ms)

    sql = parsed["sql"]
    answer = parsed["answer"]

    # citation check BEFORE execution (uncited counts as wrong even if rows match)
    cites = verify_citations(answer, pack)
    # any citation at all?
    from .triples import extract_citations

    if not extract_citations(answer):
        return ScoredResult(q.id, False, "missing_citation",
                             "answer carries no [table:row_id] citation",
                             pred_sql=sql, pred_answer=answer, latency_ms=latency_ms)
    if cites:
        return ScoredResult(q.id, False, "invalid_citation",
                             f"citations not in evidence pack: {', '.join(cites[:3])}",
                             pred_sql=sql, pred_answer=answer, latency_ms=latency_ms)

    try:
        rows, cols, lat = adapter.safe_execute(sql, timeout_sec=timeout_sec)
    except Exception as e:
        name = type(e).__name__
        if "Timeout" in name:
            return ScoredResult(q.id, False, "timeout",
                                 f"query exceeded {timeout_sec}s",
                                 pred_sql=sql, pred_answer=answer, latency_ms=latency_ms)
        return ScoredResult(q.id, False, "execution_error",
                             f"{name}: {str(e)[:80]}",
                             pred_sql=sql, pred_answer=answer, latency_ms=latency_ms)

    sha, n = canonical_rows_hash(rows, cols)
    if sha != q.expected_rows.sha256:
        # classify missing/extra/mismatched by row count when possible
        if n < q.expected_rows.n_rows:
            cls = "wrong_result_missing_rows"
        elif n > q.expected_rows.n_rows:
            cls = "wrong_result_extra_rows"
        else:
            cls = "wrong_result_mismatched_rows"
        return ScoredResult(q.id, False, cls,
                            f"returned {n} rows, gold {q.expected_rows.n_rows}; hash differs",
                            pred_sql=sql, pred_answer=answer, latency_ms=lat,
                            pred_hash=sha)
    return ScoredResult(q.id, True, None, "correct",
                        pred_sql=sql, pred_answer=answer, latency_ms=lat, pred_hash=sha)


# ------------------------------------------------------------- mcnemar -----


def mcnemar_exact(b: int, c: int) -> float:
    """Exact binomial two-sided p on discordant pairs b (A-only correct) and
    c (B-only correct). Compared against scipy's binomtest at test time."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided exact: 2 * P(X <= k), capped at 1
    def binom_pmf(i: int, n: int, p: float = 0.5) -> float:
        return math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))

    p = 2.0 * sum(binom_pmf(i, n) for i in range(0, k + 1))
    return min(p, 1.0)


def mcnemar_from_results(a: list[ScoredResult], b_: list[ScoredResult]) -> dict[str, float]:
    """Paired comparison: same questions, two systems."""
    assert len(a) == len(b_)
    b_only = 0  # system A correct, B wrong
    c_only = 0  # A wrong, B correct
    for ra, rb in zip(a, b_, strict=True):
        if ra.correct != rb.correct:
            if ra.correct:
                b_only += 1
            else:
                c_only += 1
    return {
        "b": b_only,           # A-only correct
        "c": c_only,           # B-only correct
        "p_value": mcnemar_exact(b_only, c_only),
    }


# ---------------------------------------------------------- consistency ----


def agreement_rate(runs: list[bool]) -> float:
    """Share of runs matching the majority correctness label (plan §0)."""
    if not runs:
        return 0.0
    trues = sum(runs)
    majority = trues > len(runs) / 2 or (trues == len(runs) / 2 and runs[0])
    matches = sum(1 for r in runs if r == majority)
    return matches / len(runs)


def flip_rate(runs: list[bool]) -> float:
    """Fraction of adjacent run-pairs whose correctness changes (plan §0)."""
    if len(runs) < 2:
        return 0.0
    flips = sum(1 for i in range(1, len(runs)) for a, b in [(runs[i - 1], runs[i])] if a != b)
    return flips / (len(runs) - 1)


@dataclass
class ConsistencyBlock:
    question_id: str
    headline: bool
    runs: list[bool] = field(default_factory=list)

    @property
    def agreement(self) -> float:
        return agreement_rate(self.runs)

    @property
    def flip(self) -> float:
        return flip_rate(self.runs)