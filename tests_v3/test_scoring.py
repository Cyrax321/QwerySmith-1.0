"""Scoring tests: taxonomy, McNemar (against R/scipy reference), consistency."""

import math

import pytest

from qwery_smith.questions import ExpectedRows, Question
from qwery_smith.retrieval import EvidencePack
from qwery_smith.scoring import (
    agreement_rate,
    evaluate_one,
    flip_rate,
    mcnemar_exact,
    mcnemar_from_results,
)
from qwery_smith.scoring import ScoredResult

PACK = EvidencePack(
    question_id="q1", question="q", schema_sha256="x",
    rows=[
        {"table": "orders", "row_id": "o01", "columns": ["order_id", "status"], "values": ["o01", "shipped"]},
    ],
)


def _q(sha, n_rows=1):
    return Question(
        id="q1", question="q", gold_sql="SELECT 1",
        expected_rows=ExpectedRows(sha256=sha, n_rows=n_rows),
        date="d", difficulty="easy", category="aggregation", split="heldout", source="human",
    )


GOOD_OUT = "SQL:\nSELECT order_id, status FROM orders WHERE order_id = 'o01'\nANSWER:\nOrder o01 is shipped [orders:o01]"


def _gold_hash(toy_db):
    rows, cols, _ = toy_db.safe_execute("SELECT order_id, status FROM orders WHERE order_id = 'o01'")
    from qwery_smith.questions import canonical_rows_hash

    return canonical_rows_hash(rows, cols)[0]


def test_correct_answer_scores_correct(toy_db):
    q = _q(_gold_hash(toy_db))
    r = evaluate_one(q, GOOD_OUT, PACK, toy_db)
    assert r.correct, r.reason
    assert r.error_class is None


def test_missing_citation_wrong_even_if_rows_right(toy_db):
    q = _q(_gold_hash(toy_db))
    out = "SQL:\nSELECT order_id, status FROM orders WHERE order_id = 'o01'\nANSWER:\nOrder o01 is shipped"
    r = evaluate_one(q, out, PACK, toy_db)
    assert not r.correct
    assert r.error_class == "missing_citation"


def test_invalid_citation_wrong(toy_db):
    q = _q(_gold_hash(toy_db))
    out = GOOD_OUT.replace("[orders:o01]", "[orders:oZZ]")
    r = evaluate_one(q, out, PACK, toy_db)
    assert r.error_class == "invalid_citation"


def test_wrong_rows_classified(toy_db):
    gold = _gold_hash(toy_db)
    q = _q(gold, n_rows=1)
    # returns 12 rows (no WHERE) while gold is 1 -> extra rows
    out = "SQL:\nSELECT order_id, status FROM orders\nANSWER:\nAll orders listed [orders:o01]"
    r = evaluate_one(q, out, PACK, toy_db)
    assert not r.correct
    assert r.error_class == "wrong_result_extra_rows"


def test_unexpected_refusal(toy_db):
    q = _q(_gold_hash(toy_db))
    r = evaluate_one(q, "REFUSAL:\nNo records.", PACK, toy_db)
    assert r.error_class == "unexpected_refusal"


def test_invalid_output_contract(toy_db):
    q = _q(_gold_hash(toy_db))
    r = evaluate_one(q, "I think the answer is 42", PACK, toy_db)
    assert r.error_class == "invalid_output"


def test_execution_error(toy_db):
    q = _q("f" * 64)
    out = "SQL:\nSELECT * FROM nope\nANSWER:\nBroken [orders:o01]"
    r = evaluate_one(q, out, PACK, toy_db)
    assert r.error_class == "execution_error"


# ---------------------------------------------------------------- mcnemar --

# Reference values computed with R: binom.test(b, b+c)$p.value
@pytest.mark.parametrize(
    ("b", "c", "expected_p"),
    [
        (10, 0, 0.001953125 * 2 * 0.5),  # 2*P(X<=0), n=10 => 2*(0.5^10)=0.001953125
        (8, 2, 2 * sum(math.comb(10, i) * 0.5 ** 10 for i in range(0, 3))),
        (5, 5, 1.0),
        (0, 0, 1.0),
        (1, 0, 2 * 0.5),   # 2*P(X<=0), n=1 => 1.0
    ],
)
def test_mcnemar_exact(b, c, expected_p):
    got = mcnemar_exact(b, c)
    assert math.isclose(got, min(expected_p, 1.0), rel_tol=1e-12), (b, c, got, expected_p)


def test_mcnemar_matches_binomtest():
    try:
        from scipy.stats import binomtest
    except ImportError:
        pytest.skip("scipy not installed")
    import random

    rng = random.Random(0)
    for _ in range(20):
        b = rng.randint(0, 20)
        c = rng.randint(0, 20)
        ours = mcnemar_exact(b, c)
        k, n = min(b, c), b + c
        ref = binomtest(k, n, 0.5).pvalue if n else 1.0
        assert math.isclose(ours, ref, rel_tol=1e-9), (b, c, ours, ref)


def test_mcnemar_from_results():
    a = [ScoredResult("1", True, None, ""), ScoredResult("2", False, "x", ""),
         ScoredResult("3", True, None, ""), ScoredResult("4", False, "x", "")]
    b = [ScoredResult("1", True, None, ""), ScoredResult("2", True, None, ""),
         ScoredResult("3", False, "x", ""), ScoredResult("4", False, "x", "")]
    m = mcnemar_from_results(a, b)
    assert m["b"] == 1 and m["c"] == 1 and m["p_value"] == 1.0


# ------------------------------------------------------------ consistency --

def test_agreement_rate():
    assert agreement_rate([True] * 5) == 1.0
    assert agreement_rate([True, True, True, False, False]) == 0.6
    assert agreement_rate([]) == 0.0
    assert agreement_rate([True, False]) == 0.5  # tie: majority = first label


def test_flip_rate():
    assert flip_rate([True] * 5) == 0.0
    assert flip_rate([True, False, True, False, True]) == 1.0
    assert flip_rate([True, True, False, False]) == 1 / 3  # 1 transition, 3 adjacent pairs
    assert flip_rate([True]) == 0.0