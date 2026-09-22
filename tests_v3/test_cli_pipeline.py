"""CLI integration: eval -> report through config alone (no code changes)."""

import json

import pytest
from typer.testing import CliRunner

from qwery_smith.cli import app

runner = CliRunner()


@pytest.fixture
def toy_dataset(tmp_path):
    """A complete dataset dir driven ONLY by config (the §8.1 reusability shape)."""
    import csv

    ds = tmp_path / "datasets" / "toy"
    (ds / "raw").mkdir(parents=True)
    (ds / "prepared").mkdir()

    with open(ds / "raw" / "orders.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order_id", "customer_id", "status", "purchase_ts"])
        w.writerow(["o01", "c01", "shipped", "2017-01-15 10:00:00"])
        w.writerow(["o02", "c02", "delivered", "2018-09-17 21:10:00"])

    (ds / "config.yaml").write_text(
        """
name: toy
version: 1.0.0
license: "CC0"
source_url: "fixture"
question_file: "questions_v1.jsonl"
created_with_seed: 42
holdout:
  column: "orders.purchase_ts"
  months: 6
datasource:
  dialect: "sqlite"
  uri: "sqlite:///prepared/toy.db"
  csv_dir: "raw"
  csv_pattern: "*.csv"
  table_map:
    orders: orders
  extra:
    columns:
      orders: {order_id: TEXT, customer_id: TEXT, status: TEXT, purchase_ts: TIMESTAMP}
    primary_keys:
      orders: [order_id]
""",
        encoding="utf-8",
    )

    # questions: one train_ok, one heldout
    import hashlib

    def rowhash(rows):
        payload = json.dumps({"n": len(rows), "rows": rows}, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    (ds / "questions_v1.jsonl").write_text(
        json.dumps({
            "id": "t1", "question": "count shipped orders",
            "gold_sql": "SELECT COUNT(*) FROM orders WHERE status = 'shipped'",
            "expected_rows": {"sha256": "0" * 64, "n_rows": 1},
            "date": "2026-09-22", "difficulty": "easy", "category": "aggregation",
            "split": "train_ok", "source": "human",
        }, ensure_ascii=False) + "\n" +
        json.dumps({
            "id": "h1", "question": "order o01 status",
            "gold_sql": "SELECT status FROM orders WHERE order_id = 'o01'",
            "expected_rows": {"sha256": "0" * 64, "n_rows": 1},
            "date": "2026-09-22", "difficulty": "easy", "category": "per_order_lookup",
            "split": "heldout", "source": "human",
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # fake systems via script driver — pure config, proves driver pluggability
    script = tmp_path / "fake_system.sh"
    script.write_text('#!/bin/bash\ncat\n', encoding="utf-8")  # placeholder; replaced below
    (ds / "systems.yaml").write_text(
        """
systems:
  - name: echo-model
    role: baseline
    driver: openai_compatible
    base_url: "http://localhost:9999/v1"
    model_id: "fake"
    hardware: "test"
""",
        encoding="utf-8",
    )
    return tmp_path


def _fix_hashes(tmp_path):
    """Fill expected_rows hashes by executing gold against the ingested DB."""
    from qwery_smith.adapters import open_adapter
    from qwery_smith.config import DatasetConfig
    from qwery_smith.questions import QuestionSet, canonical_rows_hash

    cfg = DatasetConfig.load("toy", root=tmp_path)
    a = open_adapter(cfg.datasource.uri)
    qs = QuestionSet.load(cfg.question_file)
    lines = []
    for q in qs.questions:
        rows, cols, _ = a.safe_execute(q.gold_sql)
        sha, n = canonical_rows_hash(rows, cols)
        d = q.to_json()
        d["expected_rows"] = {"sha256": sha, "n_rows": n}
        lines.append(json.dumps(d, ensure_ascii=False))
    cfg.question_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    a.close()


def test_cli_pipeline_full(toy_dataset, tmp_path):
    root = toy_dataset

    # ingest + profile first
    for stage in ["ingest", "profile"]:
        r = runner.invoke(app, [stage, "toy", "--root", str(root)])
        assert r.exit_code == 0, f"{stage}: {r.output}"

    # fill expected_rows hashes from the ingested DB, then validate
    _fix_hashes(root)
    for stage in ["validate", "retrieve", "triples"]:
        r = runner.invoke(app, [stage, "toy", "--root", str(root)])
        assert r.exit_code == 0, f"{stage}: {r.output}"

    _fix_hashes(root)
    r = runner.invoke(app, ["validate", "toy", "--root", str(root)])
    assert r.exit_code == 0, r.output

    # eval with a live OpenAI-compatible server is not possible here;
    # exercise eval wiring via a script-driver system instead
    sysyaml = root / "datasets" / "toy" / "systems.yaml"
    good_py = root / "good_system.py"
    good_py.write_text(
        "import sys\n"
        "out = 'SQL:' + chr(10) + \"SELECT status FROM orders WHERE order_id = 'o01'\" + chr(10) + 'ANSWER:' + chr(10) + 'shipped [orders:o01]'\n"
        "print(out)\n",
        encoding="utf-8",
    )
    sysyaml.write_text(
        f"""
systems:
  - name: perfect
    role: candidate
    driver: script
    model_id: "python3 {good_py}"
    hardware: "test"
  - name: frontier
    role: reference
    driver: script
    model_id: "python3 {good_py}"
    hardware: "test"
""",
        encoding="utf-8",
    )

    r = runner.invoke(app, ["eval", "toy", "--root", str(root), "--split", "heldout"])
    assert r.exit_code == 0, r.output
    assert "EX=" in r.output

    r = runner.invoke(app, ["report", "toy", "--root", str(root)])
    assert r.exit_code == 0, r.output
    assert "GATE" in r.output

    # report.md + failure structure exist
    runs = sorted((root / "runs" / "toy").glob("*"))
    assert runs
    assert (runs[-1] / "report.md").exists()