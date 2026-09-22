"""End-to-end: runner + gate + report + failure folders with fake systems."""

import json

from qwery_smith.evaluate import SystemSpec, run_system, write_run
from qwery_smith.questions import ExpectedRows, Question, QuestionSet, canonical_rows_hash
from qwery_smith.retrieval import EvidencePack, build_index, build_pack
from qwery_smith.report import evaluate_gate, render_report, write_failure_folders
from qwery_smith.scoring import ScoredResult
from qwery_smith.schema_loader import load_schema

PACK = EvidencePack(
    question_id="q1", question="q", schema_sha256="x",
    rows=[{"table": "orders", "row_id": "o01", "columns": ["order_id", "status"], "values": ["o01", "shipped"]}],
)


def _make_question(toy_db, qid, sql):
    rows, cols, _ = toy_db.safe_execute(sql)
    sha, n = canonical_rows_hash(rows, cols)
    return Question(
        id=qid, question="question " + qid, gold_sql=sql,
        expected_rows=ExpectedRows(sha256=sha, n_rows=n),
        date="2026-09-22", difficulty="easy", category="per_order_lookup",
        split="heldout", source="human",
    )


def test_full_eval_loop(toy_db, tmp_path):
    schema = load_schema(toy_db)
    q1 = _make_question(toy_db, "q1", "SELECT order_id, status FROM orders WHERE order_id = 'o01'")
    q2 = _make_question(toy_db, "q2", "SELECT COUNT(*) FROM orders WHERE status = 'shipped'")

    def perfect_model(prompt: str) -> str:
        # a "system" that reads the question and returns contract-perfect output
        if "q1" in prompt:
            return "SQL:\nSELECT order_id, status FROM orders WHERE order_id = 'o01'\nANSWER:\nOrder o01 is shipped [orders:o01]"
        return "SQL:\nSELECT COUNT(*) FROM orders WHERE status = 'shipped'\nANSWER:\n2 shipped orders [orders:o03] [orders:o11]"

    def broken_model(prompt: str) -> str:
        return "REFUSAL:\nNothing found."

    packs = {}
    idx = build_index(toy_db, schema)
    for q in (q1, q2):
        p = build_pack(q.id, q.question, idx, schema, 5)
        # force o01/o03/o11 rows into pack so citations verify
        packs[q.id] = EvidencePack(
            question_id=q.id, question=q.question, schema_sha256=p.schema_sha256,
            rows=p.rows + [
                {"table": "orders", "row_id": "o03", "columns": ["order_id", "status"], "values": ["o03", "shipped"]},
                {"table": "orders", "row_id": "o11", "columns": ["order_id", "status"], "values": ["o11", "shipped"]},
                {"table": "orders", "row_id": "o01", "columns": ["order_id", "status"], "values": ["o01", "shipped"]},
            ],
        )

    good = run_system(
        SystemSpec("perfect", "candidate", None, n_consistency_runs=5),
        perfect_model, [q1, q2], packs, schema.ddl_text(), toy_db,
    )
    bad = run_system(
        SystemSpec("broken", "baseline", None, n_consistency_runs=5),
        broken_model, [q1, q2], packs, schema.ddl_text(), toy_db,
    )
    assert good.ex_heldout() == 1.0
    assert bad.ex_heldout() == 0.0
    assert bad.refusal_rate() == 1.0
    assert good.mean_agreement() == 1.0
    assert good.mean_flip() == 0.0

    # persistence
    s1 = write_run(good, tmp_path, [q1, q2])
    s2 = write_run(bad, tmp_path, [q1, q2])
    assert s1["ex_heldout"] == 1.0 and s2["ex_heldout"] == 0.0

    # report + gate
    gate = evaluate_gate(
        ex_row2=good.ex_heldout(), flip_row2=good.mean_flip(),
        ex_row4=good.ex_heldout(), flip_row4=bad.mean_flip() + 0.01,
    )
    assert gate.row2_passes
    md = render_report(
        "toy",
        [s2, s1],
        {"perfect": good.headline, "broken": bad.headline},
        gate=gate,
        row_labels={"perfect": "2. Qwen3-8B-FT (n=3)", "broken": "1. Qwen3-8B base"},
    )
    assert "GATE: PASS" in md
    assert "McNemar" in md

    # failure folders: broken system has 2 failures, perfect has 0
    counts = write_failure_folders(
        tmp_path, {q.id: q for q in (q1, q2)},
        {"perfect": good.headline, "broken": bad.headline},
    )
    assert counts["broken"] == 2 and counts["perfect"] == 0
    f = tmp_path / "failures" / "broken" / "q1.md"
    assert "unexpected_refusal" in f.read_text()


def test_gate_fail_triggers_row3_reporting():
    gate = evaluate_gate(ex_row2=0.40, flip_row2=0.10, ex_row4=0.50, flip_row4=0.05)
    assert not gate.row2_passes
    assert "GATE: FAIL" in gate.verdict_line()


def test_gate_boundary_exactly_5_points():
    # exactly 5 pts below => within margin => passes EX check (>= boundary)
    gate = evaluate_gate(ex_row2=0.45, flip_row2=0.0, ex_row4=0.50, flip_row4=0.05)
    assert gate.margin_met and gate.flip_met and gate.row2_passes


def test_gate_flip_worse_fails_even_if_ex_ok():
    gate = evaluate_gate(ex_row2=0.60, flip_row2=0.20, ex_row4=0.50, flip_row4=0.10)
    assert not gate.row2_passes
    assert gate.margin_met and not gate.flip_met