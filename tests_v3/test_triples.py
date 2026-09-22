"""Triples tests: mix ratios, distractor exclusion, refusal, ledger."""

import json
import random

from qwery_smith.questions import ExpectedRows, Question, QuestionSet
from qwery_smith.retrieval import build_index, load_pack
from qwery_smith.schema_loader import load_schema
from qwery_smith.triples import (
    build_triples,
    build_training_target,
    extract_citations,
    parse_output,
    sample_distractors,
    verify_citations,
    write_triples,
)


def _qs_with_trains():
    qs = QuestionSet()
    qs.add(Question(
        id="t1", question="how many shipped orders",
        gold_sql="SELECT COUNT(*) FROM orders WHERE status = 'shipped'",
        expected_rows=ExpectedRows(sha256="a" * 64, n_rows=1),
        date="2026-09-22", difficulty="easy", category="aggregation",
        split="train_ok", source="human",
    ))
    qs.add(Question(
        id="h1", question="delivered orders count",
        gold_sql="SELECT COUNT(*) FROM orders WHERE status = 'delivered'",
        expected_rows=ExpectedRows(sha256="b" * 64, n_rows=1),
        date="2026-09-22", difficulty="easy", category="aggregation",
        split="heldout", source="human",
    ))
    return qs


def _toy_cfg():
    from qwery_smith.config import DatasetConfig, DatasourceConfig, HoldoutConfig
    from pathlib import Path

    return DatasetConfig(
        name="toy", license="CC0", source_url="fixture",
        question_file=Path("questions_v1.jsonl"),
        holdout=HoldoutConfig(column="orders.purchase_ts", months=6),
        datasource=DatasourceConfig(csv_dir=Path("raw"), csv_pattern="*.csv",
                                    table_map={}, uri="sqlite:///", dialect="sqlite"),
    )


def test_heldout_never_reaches_triples(toy_db):
    cfg = _toy_cfg()
    qs = _qs_with_trains()
    schema = load_schema(toy_db)
    idx = build_index(toy_db, schema)
    from qwery_smith.retrieval import build_pack

    packs = {q.id: build_pack(q.id, q.question, idx, schema, 5) for q in qs.questions}
    triples = build_triples(cfg, qs.questions, idx, schema.ddl_text(), packs,
                            rng=random.Random(0))
    qids = {t.question_id for t in triples}
    assert "h1" not in qids, "held-out question leaked into training triples"


def test_mix_and_kinds(toy_db):
    cfg = _toy_cfg()
    qs = _qs_with_trains()
    schema = load_schema(toy_db)
    idx = build_index(toy_db, schema)
    from qwery_smith.retrieval import build_pack

    packs = {q.id: build_pack(q.id, q.question, idx, schema, 5) for q in qs.questions}
    # force all three kinds deterministically: build many triples from 1 question
    triples = []
    rng = random.Random(1)
    kinds = set()
    for _ in range(60):
        t = build_triples(cfg, qs.questions[:1], idx, schema.ddl_text(),
                          {qs.questions[0].id: packs[qs.questions[0].id]}, rng=rng)
        triples.extend(t)
        kinds.update(x.kind for x in t)
    assert kinds == {"grounded", "refusal", "schema_only"}


def test_distractors_exclude_positives(toy_db):
    schema = load_schema(toy_db)
    idx = build_index(toy_db, schema)
    positives = {"orders:o01"}
    rng = random.Random(7)
    picks = sample_distractors(idx.docs, positives, rng, 5)
    for d in picks:
        assert f"{d.table}:{d.row_id}" != "orders:o01"
    assert picks, "expected distractors"


def test_refusal_triple_shape(toy_db):
    cfg = _toy_cfg()
    qs = _qs_with_trains()
    schema = load_schema(toy_db)
    idx = build_index(toy_db, schema)
    from qwery_smith.retrieval import build_pack

    packs = {q.id: build_pack(q.id, q.question, idx, schema, 5) for q in qs.questions}
    rng = random.Random(0)
    for _ in range(40):
        for t in build_triples(cfg, qs.questions[:1], idx, schema.ddl_text(),
                                {qs.questions[0].id: packs[qs.questions[0].id]}, rng=rng):
            if t.kind == "refusal":
                assert t.target.startswith("REFUSAL:")
                assert "SQL:" not in t.target
                return
    assert False, "no refusal triple sampled in 40 draws"


def test_grounding_target_has_sql_and_citation(toy_db):
    q = _qs_with_trains().questions[0]
    target = build_training_target(
        q, "There are 2 shipped orders.",
        [{"table": "orders", "row_id": "o03"}, {"table": "orders", "row_id": "o11"}],
    )
    parsed = parse_output(target)
    assert parsed["kind"] == "answer"
    assert "COUNT(*)" in parsed["sql"]
    cites = extract_citations(parsed["answer"])
    assert ("orders", "o03") in cites and ("orders", "o11") in cites


def test_parse_output_contract():
    ok = parse_output("SQL:\nSELECT 1\nANSWER:\nOne row. [orders:o01]")
    assert ok["kind"] == "answer" and ok["sql"] == "SELECT 1"
    ref = parse_output("REFUSAL:\nNo matching records in the evidence.")
    assert ref["kind"] == "refusal"
    bad = parse_output("The answer is SELECT 1")
    assert bad["kind"] == "invalid_output"


def test_verify_citations():
    from qwery_smith.retrieval import EvidencePack

    pack = EvidencePack(
        question_id="q", question="q", schema_sha256="x",
        rows=[{"table": "orders", "row_id": "o01", "columns": ["order_id"], "values": ["o01"]}],
    )
    assert verify_citations("answer [orders:o01]", pack) == []
    bad = verify_citations("answer [orders:oXX] [order_items:o01]", pack)
    assert len(bad) == 2


def test_ledger_written(toy_db, tmp_path):
    cfg = _toy_cfg()
    qs = _qs_with_trains()
    schema = load_schema(toy_db)
    idx = build_index(toy_db, schema)
    from qwery_smith.retrieval import build_pack

    packs = {q.id: build_pack(q.id, q.question, idx, schema, 5) for q in qs.questions}
    triples = build_triples(cfg, qs.questions, idx, schema.ddl_text(), packs,
                            rng=random.Random(3))
    meta = write_triples(triples, tmp_path / "t.jsonl")
    ledger = json.loads(open(meta["ledger"]).read())
    assert ledger["n_triples"] == len(triples)
    assert set(ledger["by_kind"]) <= {"grounded", "refusal", "schema_only"}
    assert isinstance(ledger["evidence_row_ids"], list)