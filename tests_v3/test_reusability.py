"""Part 3 reusability proof (plan §8.1): the full pipeline runs on a second
dataset with config changes only - enforced, not asserted.

The toy #2 dataset is deliberately a different shape from toy #1:
  - single flat table, no FK graph
  - grouped load (two CSV files -> one table)
  - different holdout column and window
  - different text/keyword vocabulary
If any fix were needed in qwery_smith/ to make this pass, that fix had to land
in the harness (it did, in the pre-test hardening) - this test pins it.
"""

import json
from pathlib import Path

from typer.testing import CliRunner

from qwery_smith.cli import app

runner = CliRunner()


def _make_toy2(root: Path) -> Path:
    """Retail-shaped dataset: two CSV sheets, single table, config only."""
    ds = root / "datasets" / "retail_toy"
    raw = ds / "raw"
    raw.mkdir(parents=True)

    import csv

    # sheet 1 (older)
    with open(raw / "Year 2009-2010.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"])
        w.writerow(["50001", "85099B", "RED WOOLY HEART", 12, "2010-03-14 10:00:00", 2.95, "17850", "United Kingdom"])
        w.writerow(["50002", "22633", "VINTAGE GIFT SET", 6, "2010-05-02 11:30:00", 4.25, "17850", "United Kingdom"])
        w.writerow(["50003", "85099B", "RED WOOLY HEART", 3, "2010-08-19 09:15:00", 2.95, "13085", "France"])
    # sheet 2 (newer - includes the holdout window)
    with open(raw / "Year 2010-2011.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"])
        w.writerow(["50004", "85099B", "RED WOOLY HEART", 10, "2011-02-11 14:00:00", 2.95, "17850", "United Kingdom"])
        w.writerow(["50005", "22633", "VINTAGE GIFT SET", 2, "2011-04-03 16:45:00", 4.25, "15311", "Germany"])
        w.writerow(["50006", "85099B", "RED WOOLY HEART", 24, "2011-06-30 10:05:00", 2.95, "17850", "United Kingdom"])
        w.writerow(["50007", "22633", "VINTAGE GIFT SET", 5, "2011-09-12 12:20:00", 4.25, "17850", "United Kingdom"])

    (ds / "config.yaml").write_text(
        """
name: retail_toy
version: 1.0.0
license: "CC0 (synthetic)"
source_url: "fixture"
question_file: "questions_v1.jsonl"
created_with_seed: 42
holdout:
  column: "online_retail.InvoiceDate"
  months: 6
datasource:
  dialect: "sqlite"
  uri: "sqlite:///prepared/retail_toy.db"
  csv_dir: "raw"
  csv_pattern: "*.csv"
  table_map:
    online_retail:
      - "Year 2009-2010"
      - "Year 2010-2011"
  extra:
    columns:
      online_retail:
        InvoiceNo: TEXT
        StockCode: TEXT
        Description: TEXT
        Quantity: INTEGER
        InvoiceDate: TIMESTAMP
        UnitPrice: REAL
        CustomerID: TEXT
        Country: TEXT
    primary_keys:
      online_retail: [InvoiceNo, StockCode]
    authoring:
      review_keywords: ["RED", "VINTAGE", "HEART"]
""",
        encoding="utf-8",
    )
    return ds


def test_part3_reusability_config_only(tmp_path):
    """Full pipeline on a differently-shaped dataset: config changes only."""
    _make_toy2(tmp_path)

    # 1. ingest - grouped load (two files -> one table)
    r = runner.invoke(app, ["ingest", "retail_toy", "--root", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert "7" in r.output  # 3 + 4 rows across both sheets

    # 2. profile - cutoff computed mechanically: max 2011-09-12 -> 2011-03-01
    r = runner.invoke(app, ["profile", "retail_toy", "--root", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert "2011-03-01" in r.output

    # 3. author - generators must work with no FK graph + grouped table
    r = runner.invoke(app, ["author", "retail_toy", "--root", str(tmp_path),
                            "--n", "8", "--seed", "1", "--force"])
    assert r.exit_code == 0, r.output
    draft = tmp_path / "datasets" / "retail_toy" / "prepared" / "questions_draft.jsonl"
    lines = [json.loads(l) for l in draft.read_text().splitlines() if l.strip()]
    assert len(lines) >= 4
    # every gold executes and hashes; splits mechanically tagged
    assert all(l["expected_rows"]["n_rows"] >= 1 for l in lines)

    # 4. validate the draft as if reviewed (promote to questions_v1.jsonl)
    cfg_dir = tmp_path / "datasets" / "retail_toy"
    reviewed = cfg_dir / "questions_v1.jsonl"
    out_lines = []
    for l in lines:
        l["source"] = "human"  # pretend review happened
        out_lines.append(json.dumps(l, ensure_ascii=False))
    reviewed.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    r = runner.invoke(app, ["validate", "retail_toy", "--root", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert "FAILED" not in r.output

    # 5. retrieve + triples - packs and RAFT triples build on the new shape
    for stage in ["retrieve", "triples"]:
        r = runner.invoke(app, [stage, "retail_toy", "--root", str(tmp_path)])
        assert r.exit_code == 0, f"{stage}: {r.output}"

    # 6. eval + report - script-driver systems, same as Olist path
    good_py = tmp_path / "good_system.py"
    good_py.write_text(
        "import sys\n"
        "out = 'SQL:' + chr(10) + \"SELECT COUNT(*) FROM online_retail WHERE Description LIKE '%RED%'\" + chr(10) + 'ANSWER:' + chr(10) + '3 red lines [online_retail:50001]'\n"
        "print(out)\n",
        encoding="utf-8",
    )
    sysyaml = cfg_dir / "systems.yaml"
    sysyaml.write_text(
        f"""
systems:
  - name: candidate-model
    role: candidate
    driver: script
    model_id: "python3 {good_py}"
    hardware: "test"
  - name: reference-model
    role: reference
    driver: script
    model_id: "python3 {good_py}"
    hardware: "test"
""",
        encoding="utf-8",
    )
    r = runner.invoke(app, ["eval", "retail_toy", "--root", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert "EX=" in r.output

    r = runner.invoke(app, ["report", "retail_toy", "--root", str(tmp_path)])
    assert r.exit_code == 0, r.output
    assert "GATE" in r.output

    # same table format delivered (plan §6: 'deliver the same results table')
    run_dirs = sorted((tmp_path / "runs" / "retail_toy").glob("*"))
    report_md = (run_dirs[-1] / "report.md").read_text()
    assert "| System |" in report_md
    assert "EX held-out" in report_md


def test_grouped_load_row_counts(tmp_path):
    """Grouped load specifically: 3 + 4 = 7 rows in one table."""
    _make_toy2(tmp_path)
    r = runner.invoke(app, ["ingest", "retail_toy", "--root", str(tmp_path)])
    assert r.exit_code == 0, r.output

    from qwery_smith.adapters import open_adapter
    from qwery_smith.config import DatasetConfig

    cfg = DatasetConfig.load("retail_toy", root=tmp_path)
    a = open_adapter(cfg.datasource.uri)
    rows, _, _ = a.safe_execute("SELECT COUNT(*) FROM online_retail")
    assert rows == [(7,)]
    # both sheets' data present
    rows, _, _ = a.safe_execute(
        "SELECT COUNT(DISTINCT InvoiceNo) FROM online_retail"
    )
    assert rows == [(7,)]
    a.close()


def test_olist_config_still_loads():
    """The Olist config (9-table, FK-rich shape) still parses after the
    grouped-load table_map generalization."""
    from qwery_smith.config import DatasetConfig

    cfg = DatasetConfig.load("olist")
    assert cfg.name == "olist"
    assert len(cfg.datasource.table_map) == 9
    assert all(isinstance(v, str) for v in cfg.datasource.table_map.values())
    assert cfg.holdout.column == "orders.order_purchase_timestamp"