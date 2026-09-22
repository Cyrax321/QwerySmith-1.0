"""3-seed protocol tests: aggregation, multi-run report merge, train prep."""

import json

from typer.testing import CliRunner

from qwery_smith.cli import app
from qwery_smith.report import aggregate_seeds, evaluate_gate, render_report
from qwery_smith.scoring import ScoredResult

runner = CliRunner()


def _summary(name, role, ex, flip, n=10, **kw):
    return {
        "system": name, "role": role, "hardware": "T4", "n_questions": n,
        "ex_heldout": ex, "refusal_rate": 0.0, "agreement": 1.0,
        "flip_rate": flip, "p50_latency_ms": 100.0, "n_failures": 0,
        "generated_at": "t", **kw,
    }


def _results_for(n_questions, correct_flags):
    out = []
    for i, ok in enumerate(correct_flags):
        out.append(ScoredResult(f"q{i}", ok, None if ok else "wrong_result_missing_rows",
                                "correct" if ok else "wrong", latency_ms=1.0))
    assert len(out) == n_questions
    return out


def test_aggregate_seeds_mean_and_median():
    seeds = [
        _summary("ft-s1", "candidate_seed", 0.40, 0.05),
        _summary("ft-s2", "candidate_seed", 0.50, 0.10),
        _summary("ft-s3", "candidate_seed", 0.60, 0.15),
    ]
    cand, rep = aggregate_seeds(seeds, {})
    assert abs(cand["ex_heldout"] - 0.5) < 1e-9          # mean
    assert cand["ex_range"] == (0.4, 0.6)                # range
    assert abs(cand["flip_rate"] - 0.10) < 1e-9          # mean flip
    assert rep == "ft-s2"                               # median-EX seed


def test_aggregate_seeds_passthrough_single():
    cand = _summary("ft", "candidate", 0.55, 0.02)
    out, rep = aggregate_seeds([cand], {})
    assert out is cand and rep == "ft"


def test_gate_uses_seed_mean():
    seeds = [
        _summary("ft-s1", "candidate_seed", 0.40, 0.05),
        _summary("ft-s2", "candidate_seed", 0.50, 0.10),
        _summary("ft-s3", "candidate_seed", 0.60, 0.15),
    ]
    ref = _summary("frontier", "reference", 0.52, 0.12)
    cand, rep = aggregate_seeds(seeds, {})
    gate = evaluate_gate(cand["ex_heldout"], cand["flip_rate"], ref["ex_heldout"], ref["flip_rate"])
    assert gate.row2_passes  # mean 0.50 within 5pts of 0.52; flip 0.10 <= 0.12
    # but the worst single seed would have failed — aggregation is load-bearing
    gate_worst = evaluate_gate(0.40, 0.15, ref["ex_heldout"], ref["flip_rate"])
    assert not gate_worst.row2_passes


def test_render_report_with_seed_rows():
    seeds = [
        _summary("ft-s1", "candidate_seed", 0.40, 0.05),
        _summary("ft-s2", "candidate_seed", 0.50, 0.10),
    ]
    others = [
        _summary("qwen3-8b-base", "baseline", 0.30, 0.20),
        _summary("frontier", "reference", 0.52, 0.12),
        _summary("qwen3-30b", "onprem", 0.45, 0.08),
    ]
    flags = {  # 10 questions each
        "ft-s2": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        "qwen3-8b-base": [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        "frontier": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    }
    headline = {k: _results_for(10, v) for k, v in flags.items()}
    cand, rep = aggregate_seeds(others + seeds, headline)
    gate = evaluate_gate(cand["ex_heldout"], cand["flip_rate"], 0.52, 0.12)
    md = render_report("toy", others + seeds, headline, gate=gate)
    assert "mean ± range" in md
    assert "median-EX seed" in md
    assert "per-seed" in md           # per-seed McNemar rows present
    assert "GATE" in md


def test_train_prep_three_seed_configs(tmp_path):
    """`train` (prep mode) writes 3 pinned configs + manifest; execute refused on CPU."""
    import csv

    ds = tmp_path / "datasets" / "toy"
    (ds / "raw").mkdir(parents=True)
    (ds / "prepared").mkdir()
    with open(ds / "raw" / "orders.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order_id", "status", "purchase_ts"])
        w.writerow(["o01", "shipped", "2017-01-15 10:00:00"])

    (ds / "config.yaml").write_text(
        """
name: toy
version: 1.0.0
license: "CC0"
source_url: "fixture"
question_file: "questions_v1.jsonl"
created_with_seed: 42
datasource:
  dialect: "sqlite"
  uri: "sqlite:///prepared/toy.db"
  csv_dir: "raw"
  csv_pattern: "*.csv"
  table_map:
    orders: orders
  extra:
    columns:
      orders: {order_id: TEXT, status: TEXT, purchase_ts: TIMESTAMP}
""",
        encoding="utf-8",
    )

    # a minimal triples file
    triples = ds / "prepared" / "triples_seed42.jsonl"
    triples.write_text(
        json.dumps({"question_id": "t1", "kind": "grounded",
                    "prompt": "P", "target": "T", "evidence_row_ids": []}) + "\n",
        encoding="utf-8",
    )

    r = runner.invoke(app, ["train", "toy", "--root", str(tmp_path), "--seeds", "1,2,3",
                            "--dry-run"])
    assert r.exit_code == 0, r.output

    runs = sorted((tmp_path / "runs" / "toy").glob("train_*"))
    assert len(runs) == 1
    run_dir = runs[0]
    for seed in (1, 2, 3):
        cfg_path = run_dir / f"qlora_seed{seed}.yaml"
        assert cfg_path.exists()
        import yaml

        cfg = yaml.safe_load(cfg_path.read_text())
        assert cfg["seed"] == seed
        assert cfg["lora"] == {"r": 16, "alpha": 32, "dropout": 0.05,
                               "target_modules": "all-linear"}
        assert cfg["decoding"]["temperature"] == 0.7     # pinned, not improvised
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["seeds"] == [1, 2, 3]
    assert manifest["n_triples"] == 1

    # --execute without CUDA must fail cleanly
    r = runner.invoke(app, ["train", "toy", "--root", str(tmp_path), "--seeds", "1",
                            "--execute"])
    assert r.exit_code == 2
    assert "CUDA" in r.output