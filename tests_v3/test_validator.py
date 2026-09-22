"""Validator + holdout-leak tests (plan §4.4) — the enforcement machinery."""


from qwery_smith.config import HoldoutConfig
from qwery_smith.questions import ExpectedRows, Question, QuestionSet, canonical_rows_hash
from qwery_smith.validate import validate_question_set

HOLDOUT = HoldoutConfig(column="orders.purchase_ts", months=6)
# computed by profiler on fixture: cutoff = 2018-03-01
CUTOFF = "2018-03-01"


def _mk_q(qid, sql, split="train_ok"):
    _rows, _cols, _ = None, None, None
    return qid, sql, split


def _make_qs(adapter, specs):
    qs = QuestionSet()
    for qid, sql, split in specs:
        rows, cols, _ = adapter.safe_execute(sql)
        sha, n = canonical_rows_hash(rows, cols)
        qs.add(Question(
            id=qid,
            question=f"question for {qid}",
            gold_sql=sql,
            expected_rows=ExpectedRows(sha256=sha, n_rows=n),
            date="2026-09-22",
            difficulty="easy",
            category="per_order_lookup" if "WHERE" not in sql.upper() else "aggregation",
            split=split,
            source="human",
        ))
    return qs


def test_clean_train_question_passes(toy_db):
    qs = _make_qs(toy_db, [("q1", "SELECT COUNT(*) FROM orders WHERE purchase_ts < '2018-03-01'", "train_ok")])
    res = validate_question_set(qs, toy_db, holdout=HOLDOUT, cutoff_iso=CUTOFF)
    assert res.ok, res.summary()
    assert res.scorable == 1


def test_leaking_train_question_is_caught(toy_db):
    # gold SQL with no time filter -> guard changes rows -> leak detected
    qs = _make_qs(toy_db, [("q2", "SELECT COUNT(*) FROM orders", "train_ok")])
    res = validate_question_set(qs, toy_db, holdout=HOLDOUT, cutoff_iso=CUTOFF)
    assert not res.ok
    assert any(f.check == "holdout_leak" for f in res.excluded), res.summary()


def test_boundary_exact_train_ok(toy_db):
    # a question whose result is unchanged by the cutoff guard is safe
    qs = _make_qs(toy_db, [(
        "q3",
        "SELECT COUNT(*) FROM orders WHERE purchase_ts >= '2018-03-01' AND purchase_ts < '2018-09-01'",
        "train_ok",  # NB: this queries the held-out window! must leak by the >= guard flipping
    )])
    res = validate_question_set(qs, toy_db, holdout=HOLDOUT, cutoff_iso=CUTOFF)
    # the injected guard (AND < cutoff) contradicts >= 2018-03-01 -> empty result != stored -> LEAK
    assert any(f.check == "holdout_leak" for f in res.excluded)


def test_heldout_questions_bypass_leak_check(toy_db):
    # heldout questions are SUPPOSED to read the window — no leak check applies
    qs = _make_qs(toy_db, [("q4", "SELECT COUNT(*) FROM orders WHERE purchase_ts >= '2018-03-01'", "heldout")])
    res = validate_question_set(qs, toy_db, holdout=HOLDOUT, cutoff_iso=CUTOFF)
    assert res.ok, res.summary()
    assert res.scorable == 1


def test_gold_failure_excludes_and_logs(toy_db):
    # broken gold SQL: construct directly with a fabricated hash (it can't execute)
    qs = QuestionSet()
    qs.add(Question(
        id="q5", question="broken", gold_sql="SELECT * FROM nonexistent_table",
        expected_rows=ExpectedRows(sha256="f" * 64, n_rows=1),
        date="2026-09-22", difficulty="easy", category="aggregation",
        split="train_ok", source="human",
    ))
    res = validate_question_set(qs, toy_db, holdout=None, cutoff_iso=None)
    assert not res.ok
    assert any(f.check == "gold_sql" for f in res.excluded)


def test_expected_rows_drift_detected(toy_db):
    qs = _make_qs(toy_db, [("q6", "SELECT COUNT(*) FROM orders", "train_ok")])
    # tamper with stored hash
    q = qs.questions[0]
    qs.questions[0] = q.with_changes(expected_rows={"sha256": "deadbeef" * 8, "n_rows": 99})
    res = validate_question_set(qs, toy_db, holdout=None, cutoff_iso=None)
    assert any(f.check == "expected_rows_drift" for f in res.excluded)


def test_temporal_term_lint_warns(toy_db):
    qs = _make_qs(toy_db, [("q7", "SELECT COUNT(*) FROM orders WHERE purchase_ts < '2017-06-01'", "train_ok")])
    q = qs.questions[0]
    qs.questions[0] = q.with_changes(question="orders in the last 6 months of 2017")
    res = validate_question_set(qs, toy_db, holdout=None, cutoff_iso=None)
    assert any(f.check == "temporal_term" for f in res.warnings)


def test_jsonl_roundtrip(tmp_path):
    p = tmp_path / "q.jsonl"
    qs = QuestionSet()
    qs.add(Question(
        id="x1", question="q?", gold_sql="SELECT 1",
        expected_rows=ExpectedRows(sha256="a" * 64, n_rows=1),
        date="2026-09-22", difficulty="easy", category="aggregation",
        split="heldout", source="human",
    ))
    qs.save(p)
    loaded = QuestionSet.load(p)
    assert loaded.questions[0] == qs.questions[0]
    assert loaded.counts()["total"] == 1