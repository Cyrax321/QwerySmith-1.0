"""Publish stage tests: card content, figures staging, dry-run CLI."""

import json
from pathlib import Path

from typer.testing import CliRunner

from qwery_smith.cli import app
from qwery_smith.publish import (
    PIPELINE_DIAGRAM,
    PublishSpec,
    build_model_card,
    _loss_curve_figure,
    _results_figure,
)

runner = CliRunner()

RECORD = {
    "seed": 1,
    "n_triples": 180,
    "final_loss": 0.42,
    "hardware": {"gpu": "NVIDIA T4", "vram_mb": 15360,
                  "packages": {"torch": "2.4.0", "unsloth": None}},
    "log_history": [
        {"step": 10, "loss": 1.2, "learning_rate": 1e-4},
        {"step": 20, "loss": 0.8, "learning_rate": 8e-5},
        {"step": 30, "loss": 0.5, "learning_rate": 3e-5},
    ],
}
QLORA_CFG = {
    "base_model": "Qwen/Qwen3-8B",
    "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": "all-linear"},
    "optim": {"lr": 1.0e-4, "schedule": "cosine", "warmup_ratio": 0.03, "epochs": 3},
    "batch": {"per_device": 1, "grad_accum": 16, "max_len": 4096},
    "decoding": {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
}
COUNTS = {"total": 100, "by_split": {"train_ok": 62, "heldout": 38}}


def test_card_contains_everything():
    card = build_model_card(
        model_name="QwerySmith-2.0-seed1",
        base_model="Qwen/Qwen3-8B",
        dataset="olist",
        dataset_url="https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce",
        dataset_license="CC BY-NC-SA 4.0",
        question_counts=COUNTS,
        cutoff="2018-03-01",
        qlora_config=QLORA_CFG,
        train_record=RECORD,
        report_md="| System | EX |\n|---|---|\n| r2 | 40% |\n\nGATE: PASS — within margin",
        gate_pass=True,
        repo_url="https://github.com/Cyrax321/QwerySmith-1.0",
    )
    # identity + claims policy
    assert "behaviour-tuned" in card
    assert "GATE: PASS" in card
    assert "stores no data" in card
    # recipe pinned
    assert "r=16" in card and "alpha=32" in card and "cosine" in card
    assert "T=0.7" in card
    # dataset + holdout
    assert "2018-03-01" in card
    assert "train_ok" in card and "held-out" in card
    # evaluation embedded from captured artifacts
    assert "40%" in card
    # hardware stated
    assert "NVIDIA T4" in card
    # diagram + usage + limitations + license
    assert "PIPELINE DIAGRAM" not in card  # the text is embedded, not the var name
    assert "CREATE TABLE" in card          # diagram content present
    assert "PeftModel.from_pretrained" in card
    assert "Limitations" in card
    assert "apache-2.0" in card
    # provenance
    assert "github.com/Cyrax321" in card


def test_card_gate_fail_says_so():
    card = build_model_card(
        model_name="QwerySmith-2.0-seed1", base_model="Qwen/Qwen3-8B",
        dataset="olist", dataset_url="u", dataset_license="l",
        question_counts=COUNTS, cutoff=None, qlora_config=QLORA_CFG,
        train_record=RECORD, gate_pass=False,
    )
    assert "GATE: FAIL" in card
    assert "not as a recommended model" in card


def test_card_pending_eval_placeholder():
    card = build_model_card(
        model_name="QwerySmith-2.0-seed1", base_model="Qwen/Qwen3-8B",
        dataset="olist", dataset_url="u", dataset_license="l",
        question_counts=COUNTS, cutoff=None, qlora_config=QLORA_CFG,
        train_record=RECORD, report_md=None, gate_pass=None,
    )
    assert "Evaluation pending" in card


def test_loss_figure(tmp_path):
    try:
        import matplotlib  # noqa

        ok = _loss_curve_figure(RECORD["log_history"], tmp_path / "loss.png", 1)
        assert ok and (tmp_path / "loss.png").stat().st_size > 5000
    except ImportError:
        # matplotlib absent locally: figure gracefully skipped on Colab it runs
        assert not _loss_curve_figure([], tmp_path / "loss.png", 1)


def test_results_figure(tmp_path):
    try:
        import matplotlib  # noqa

        md = "# Results\n\n| System | EX |\n|---|---|\n| r1 | 30% |\n| r2 | 40% |\n"
        ok = _results_figure(md, tmp_path / "table.png")
        assert ok and (tmp_path / "table.png").stat().st_size > 3000
    except ImportError:
        assert not _results_figure("", tmp_path / "table.png")


def test_publish_cli_dry_run(tmp_path, monkeypatch):
    """Dry-run stages cards locally without HF token or network."""
    import yaml

    ds = tmp_path / "datasets" / "toy"
    (ds / "prepared").mkdir(parents=True)
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
  table_map: {}
""",
        encoding="utf-8",
    )
    (ds / "questions_v1.jsonl").write_text(
        json.dumps({
            "id": "t1", "question": "q", "gold_sql": "SELECT 1",
            "expected_rows": {"sha256": "0" * 64, "n_rows": 1},
            "date": "2026-09-22", "difficulty": "easy", "category": "aggregation",
            "split": "train_ok", "source": "human",
        }) + "\n",
        encoding="utf-8",
    )
    (ds / "prepared" / "profile.yaml").write_text(
        "holdout: {cutoff: '2018-03-01'}\n", encoding="utf-8",
    )

    run_dir = tmp_path / "runs" / "toy" / "train_20260922"
    adapters = run_dir / "adapters" / "adapter_seed1"
    adapters.mkdir(parents=True)
    (adapters / "adapter_model.safetensors").write_bytes(b"weights")
    (adapters / "train_record.json").write_text(json.dumps(RECORD))
    (run_dir / "qlora_seed1.yaml").write_text(yaml.safe_dump(QLORA_CFG))

    r = runner.invoke(app, ["publish", "toy", "--root", str(tmp_path),
                            "--hf-user", "Cyrax321", "--dry-run"])
    assert r.exit_code == 0, r.output
    preview = run_dir / "hf_cards_preview"
    card = (preview / "card_seed1.md").read_text()
    assert "QwerySmith-2.0-seed1" in card
    assert "2018-03-01" in card            # cutoff pulled from frozen profile
    assert "NVIDIA T4" in card             # hardware from train_record
    assert "train_ok" in card              # counts from question file


def test_publish_requires_adapters(tmp_path):
    ds = tmp_path / "datasets" / "toy"
    (ds / "prepared").mkdir(parents=True)
    (ds / "config.yaml").write_text(
        """
name: toy
version: 1.0.0
license: "CC0"
source_url: "f"
question_file: "questions_v1.jsonl"
datasource:
  dialect: "sqlite"
  uri: "sqlite:///prepared/toy.db"
  csv_dir: "raw"
  csv_pattern: "*.csv"
  table_map: {}
""",
        encoding="utf-8",
    )
    r = runner.invoke(app, ["publish", "toy", "--root", str(tmp_path),
                            "--hf-user", "x", "--dry-run"])
    assert r.exit_code == 2