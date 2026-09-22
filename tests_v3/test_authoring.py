"""Authoring tests: generators, facts discovery, mechanical split tagging."""

import json

from typer.testing import CliRunner

from qwery_smith.authoring import discover_facts, generate_candidates
from qwery_smith.cli import app
from qwery_smith.config import HoldoutConfig
from qwery_smith.schema_loader import load_schema

runner = CliRunner()

HOLDOUT = HoldoutConfig(column="orders.purchase_ts", months=6)


def _cfg(tmp_path):
    from pathlib import Path

    from qwery_smith.config import DatasetConfig, DatasourceConfig

    tmp_path = tmp_path or Path(".")
    return DatasetConfig(
        name="toy", license="CC0", source_url="fixture",
        question_file=tmp_path / "questions_v1.jsonl",
        holdout=HOLDOUT,
        datasource=DatasourceConfig(csv_dir=tmp_path, csv_pattern="*.csv",
                                    table_map={}, uri="sqlite:///", dialect="sqlite"),
    )


def test_discover_facts(toy_db):
    facts = discover_facts(toy_db, load_schema(toy_db))
    assert "orders" in facts.pk_by_table
    assert facts.pk_by_table["orders"] == ("order_id",)
    assert ("order_items", "order_id", "orders", "order_id") in facts.fk_edges
    assert "status" in facts.categorical_cols.get("orders", [])  # 3 distinct statuses
    assert "price" in facts.numeric_cols.get("order_items", [])
    assert facts.sample_pks["orders"]  # sampled PK values exist


def test_generate_candidates_all_categories(toy_db):
    cfg = _cfg(None)
    cands = generate_candidates(toy_db, load_schema(toy_db), cfg, n=12, seed=3)
    cats = {c.category for c in cands}
    assert cats <= {"per_order_lookup", "aggregation", "multi_table_join", "review_text"}
    assert len(cands) >= 4  # got something from multiple generators
    for c in cands:
        rows, cols, _ = toy_db.safe_execute(c.gold_sql)  # all gold executes
        assert rows, f"empty result: {c.gold_sql}"


def test_generate_candidates_dedupes(toy_db):
    cfg = _cfg(None)
    cands = generate_candidates(toy_db, load_schema(toy_db), cfg, n=30, seed=3)
    sqls = [c.gold_sql for c in cands]
    assert len(sqls) == len(set(sqls))


def test_author_cli_mechanical_splits(toy_db, tmp_path, monkeypatch):
    # point the dataset root at tmp; copy fixture db as the prepared db
    ds = tmp_path / "datasets" / "toy"
    ds.mkdir(parents=True)
    import shutil

    shutil.copy(toy_db.db_path, ds / "prepared.db")
    (ds / "config.yaml").write_text(
        f"""
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
  uri: "sqlite:///prepared.db"
  csv_dir: "raw"
  csv_pattern: "*.csv"
  table_map: {{}}
""",
        encoding="utf-8",
    )

    r = runner.invoke(app, ["author", "toy", "--root", str(tmp_path), "--n", "12",
                            "--seed", "3", "--force"])
    assert r.exit_code == 0, r.output

    draft = ds / "prepared" / "questions_draft.jsonl"
    assert draft.exists()
    lines = [json.loads(l) for l in draft.read_text().splitlines() if l.strip()]
    assert len(lines) >= 4
    splits = {l["split"] for l in lines}
    assert splits <= {"train_ok", "heldout"}
    # mechanics: a no-filter aggregate MUST be tagged heldout if present
    for l in lines:
        if l["gold_sql"].strip() == "SELECT COUNT(*) FROM orders":
            assert l["split"] == "heldout"
    # every question has a real hash (not placeholder)
    assert all(l["expected_rows"]["sha256"] != "0" * 64 for l in lines)
    assert all(l["source"] == "template+human-verified" for l in lines)