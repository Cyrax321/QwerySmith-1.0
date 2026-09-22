"""Coverage: real paths not previously exercised."""

from pathlib import Path

import pytest

from qwery_smith.config import ConfigError, DatasetConfig
from qwery_smith.questions import QuestionSet, load_manifest, write_manifest


def _toy_cfg(root: Path, name="toycfg"):
    ds = root / "datasets" / name
    ds.mkdir(parents=True)
    (ds / "config.yaml").write_text(
        """
name: toycfg
version: 1.0.0
license: "CC0"
source_url: "fixture"
question_file: "questions_v1.jsonl"
created_with_seed: 7
holdout:
  column: "orders.purchase_ts"
  months: 6
datasource:
  dialect: "sqlite"
  uri: "sqlite:///prepared/toy.db"
  csv_dir: "raw"
  csv_pattern: "*.csv"
  table_map: {}
""",
        encoding="utf-8",
    )
    return ds


def test_config_missing_dataset(tmp_path):
    with pytest.raises(ConfigError):
        DatasetConfig.load("nonexistent", root=tmp_path)


def test_config_missing_key(tmp_path):
    ds = _toy_cfg(tmp_path, "broken")
    (ds / "config.yaml").write_text(
        "name: broken\ndatasource:\n  dialect: sqlite\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError):
        DatasetConfig.load("broken", root=tmp_path)


def test_config_fields_parsed(tmp_path):
    _toy_cfg(tmp_path)
    cfg = DatasetConfig.load("toycfg", root=tmp_path)
    assert cfg.name == "toycfg"
    assert cfg.seed == 7
    assert cfg.holdout.months == 6
    assert cfg.holdout.table == "orders"
    assert cfg.holdout.col == "purchase_ts"
    assert cfg.datasource.dialect == "sqlite"
    # absolute-path sqlite URI resolution (4-slash hostless form)
    assert cfg.datasource.uri.startswith("sqlite:////")


def test_manifest_roundtrip(tmp_path):
    m = tmp_path / "manifest.yaml"
    write_manifest(
        m,
        name="olist",
        version="1.0.0",
        license_="CC BY-NC-SA 4.0",
        source_url="https://kaggle/...",
        seed=42,
        questions_path=Path("questions_v1.jsonl"),
        db_fingerprint="abc123",
        holdout={"column": "orders.order_purchase_timestamp", "cutoff": "2018-03-01"},
        counts={"total": 100, "by_split": {"train_ok": 62, "heldout": 38}},
    )
    loaded = load_manifest(m)
    assert loaded["name"] == "olist"
    assert loaded["created_with_seed"] == 42
    assert loaded["counts"]["total"] == 100
    assert loaded["holdout"]["cutoff"] == "2018-03-01"
    assert loaded["license"].startswith("CC BY-NC-SA")


def test_adapter_factory_sqlite_abs(tmp_path):
    from qwery_smith.adapters import open_adapter
    from qwery_smith.adapters.sqlite_adapter import SQLiteAdapter

    db = tmp_path / "a.db"
    a = SQLiteAdapter(db, read_only=False)
    a.conn.execute("CREATE TABLE t (x INTEGER)")
    a.conn.commit()
    a.close()
    b = open_adapter(f"sqlite:////{db}")  # absolute 4-slash form
    assert b.dialect == "sqlite"
    rows, cols, _ = b.safe_execute("SELECT COUNT(*) AS n FROM t")
    assert rows == [(0,)]
    b.close()


def test_adapter_factory_unknown_scheme():
    from qwery_smith.adapters import open_adapter

    with pytest.raises(ValueError):
        open_adapter("mysql://whatever")


def test_questionset_duplicate_ids_flagged(tmp_path):
    from qwery_smith.questions import ExpectedRows, Question

    qs = QuestionSet()
    for _ in range(2):
        qs.add(Question(
            id="same", question="q", gold_sql="SELECT 1",
            expected_rows=ExpectedRows(sha256="a" * 64, n_rows=1),
            date="2026-09-22", difficulty="easy", category="aggregation",
            split="train_ok", source="human",
        ))
    assert not qs.ids_unique()
    p = tmp_path / "q.jsonl"
    qs.save(p)
    assert QuestionSet.load(p).counts()["total"] == 2


def test_parse_output_multiline_sql_and_wrapped_answer():
    from qwery_smith.triples import parse_output

    out = "SQL:\nSELECT a,\n  b\n  FROM t\nANSWER:\nLine one.\nLine two [orders:o01]."
    parsed = parse_output(out)
    assert parsed["kind"] == "answer"
    assert "SELECT a," in parsed["sql"]
    assert parsed["answer"].endswith("[orders:o01].")


def test_parse_output_refusal_with_trailing_text():
    from qwery_smith.triples import parse_output

    parsed = parse_output("REFUSAL:\nThe evidence does not contain order status.")
    assert parsed["kind"] == "refusal"
    assert "order status" in parsed["refusal"]


def test_scoring_taxonomy_order_timeout_first():
    """timeout precedes execution_error in the taxonomy (plan §6.5)."""
    from qwery_smith.scoring import ERROR_CLASSES

    assert ERROR_CLASSES.index("timeout") < ERROR_CLASSES.index("execution_error")
    assert ERROR_CLASSES.index("invalid_output") < ERROR_CLASSES.index("wrong_result_missing_rows")
    assert ERROR_CLASSES.index("missing_citation") < ERROR_CLASSES.index("unexpected_refusal")


def test_wilson_ci_bounds():
    from qwery_smith.report import _wilson_ci

    lo, hi = _wilson_ci(0.5, 10)
    assert 0.0 < lo < 0.5 < hi < 1.0
    assert _wilson_ci(0.5, 0) == (0.0, 0.0)  # degenerate n handled


def test_gate_margin_is_exactly_five_points():
    from qwery_smith.report import GATE_EX_MARGIN_POINTS

    assert GATE_EX_MARGIN_POINTS == 5.0  # pre-registered; frozen (plan §7.1)