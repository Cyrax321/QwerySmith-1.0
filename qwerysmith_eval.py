#!/usr/bin/env python3
"""
Qwerysmith 1.0 - evaluation suite: metrics, confusion matrices and figures for a finished run.

The training is already done. This script does NOT train and does NOT re-evaluate by
default: it reads what the finished run left on disk

    <out>/config.json                   the exact CLI arguments of that run
    <out>/train_log.json                per-step loss / lr / grad-norm history
    <out>/preds/<system>__<set>.json    cached generations {"raw": [...], "pred": [...]}
    <out>/adapter/                      the LoRA adapter (only the GPU stages load it)

and turns it into every number and figure needed to argue "how good is QwerySmith 1.0".

Metrics produced
----------------
  accuracy            valid-SQL rate, exact match, execution accuracy (result-set match),
                      token F1 (with and without literals), edit similarity, clause /
                      component match, soft clause Jaccard, schema-linking P/R/F1
  confusion matrices  schema linking (table-level, column-level 2x2), clause-level 2x2 for
                      every clause type, SQL-signature multiclass matrix, outcome matrix
  agreement           Cohen's kappa (unweighted / linear / quadratic), Fleiss' kappa,
                      Krippendorff's alpha, Gwet's AC1, PABAK, percent agreement
  significance        Wilson score intervals, paired bootstrap CIs, McNemar (exact and
                      continuity corrected), Holm-Bonferroni, odds ratio
  LLM-specific        per-answer log-probability confidence, calibration (ECE / MCE /
                      Brier / NLL) with a reliability diagram, ROC-AUC and PR-AUC for
                      "is this answer correct", risk-coverage and selective accuracy,
                      self-consistency (pass@k, majority-vote denotation accuracy)
  training / data     loss, learning-rate and grad-norm curves, hyper-parameters, sequence
                      length distributions, accuracy by SQL complexity and schema size

Stages (--stage)
----------------
  locate            find run directories and cached predictions (local disk or Drive)
  verify            cross-check this script's scorer against QwerySmith.py and scikit-learn
  figures           all CPU metrics, tables, figures and REPORT.md     (default)
  confidence        GPU: token log-probabilities of the cached answers -> calibration figures
  selfconsistency   GPU: k sampled answers per question -> pass@k, majority-vote accuracy
  robustness        GPU: re-generate under perturbed schemas -> drop-in accuracy
  pack              zip everything under <out>/eval for download
  all               locate + verify + figures + pack      (CPU only, safe anywhere)

Colab usage (from the folder where the model was trained, or with --out pointing at Drive)

  !python qwerysmith_eval.py --stage locate
  !python qwerysmith_eval.py --stage figures        # seconds, CPU only, uses the cached preds
  !python qwerysmith_eval.py --stage confidence     # optional, needs the GPU
  !python qwerysmith_eval.py --stage figures        # adds calibration / ROC / PR / selective
  !python qwerysmith_eval.py --stage all

Everything is written to <out>/eval: figures/*.png|.svg|.pdf, tables/*.md|.csv|.tex,
metrics_all.json, per_item_scores.csv, REPORT.md and qwerysmith_eval.zip.

Only numpy + matplotlib are required on top of the standard library (both ship with Colab).
scikit-learn is optional and used by `--stage verify` only. unsloth/torch are imported by
the GPU stages only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import sqlite3
import sys
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

# --------------------------------------------------------------------------
# Constants (kept identical to QwerySmith.py so cached artefacts line up)
# --------------------------------------------------------------------------
SEED = 42
MODEL_NAME = "Qwerysmith-1.0"
RUN_DIR_DEFAULT = "runs/qwerysmith-1.0"
MODEL_DEFAULT = "unsloth/Qwen3-4B"
IN_DIST_DATASET = "b-mc2/sql-create-context"
EXTERNAL_DATASET = "gretelai/synthetic_text_to_sql"
SYSTEMS = ["base_zeroshot", "base_fewshot", "finetuned"]
SET_NAMES = ["in_dist", "external"]

SYSTEM = (
    "You are a text-to-SQL assistant. Given a database schema and a question, "
    "reply with exactly one SQL query and nothing else."
)
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
END = "<|im_end|>"

SYSTEM_LABEL = {
    "base_zeroshot": "base - 0-shot",
    "base_fewshot": "base - 3-shot",
    "finetuned": "QwerySmith - fine-tuned",
}
SET_LABEL = {"in_dist": "in-distribution", "external": "external (OOD)"}
PALETTE = {"base_zeroshot": "#8c8c8c", "base_fewshot": "#4c72b0", "finetuned": "#dd8452"}
ACCENT = "#55a868"
WARN_C = "#c44e52"

OUTCOME_LABELS = ["executed_exact", "executed_match", "valid_wrong", "invalid"]
OUTCOME_TITLE = {
    "executed_exact": "correct + same text",
    "executed_match": "correct, different text",
    "valid_wrong": "runs, wrong answer",
    "invalid": "does not run",
}
OUTCOME_COLOR = {
    "executed_exact": "#2e7d32",
    "executed_match": "#8bc34a",
    "valid_wrong": "#f0ad4e",
    "invalid": "#c0392b",
}

# Clause flags used for the component-level confusion matrices
CLAUSES = [
    ("distinct", "SELECT DISTINCT"),
    ("aggregate", "aggregate fn"),
    ("join", "JOIN"),
    ("where", "WHERE"),
    ("group_by", "GROUP BY"),
    ("having", "HAVING"),
    ("order_by", "ORDER BY"),
    ("limit", "LIMIT"),
    ("subquery", "subquery"),
    ("set_op", "UNION/EXCEPT"),
    ("cte", "WITH (CTE)"),
    ("case_when", "CASE WHEN"),
]
AGG_FUNCS = ("count", "sum", "avg", "min", "max", "total", "group_concat")
PERTURBATIONS = [
    "table_order",
    "rename_tables",
    "distractor_tables",
    "uppercase_schema",
    "comment_noise",
]

# --------------------------------------------------------------------------
# Small helpers, logging and optional dependencies
# --------------------------------------------------------------------------
def log(msg: Any = "") -> None:
    print(msg, flush=True)


def rule(title: str = "", width: int = 78) -> None:
    if title:
        log("\n" + "-" * 4 + f" {title} " + "-" * max(0, width - len(title) - 6))
    else:
        log("-" * width)


def json_dump(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))


def json_load(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001
        return default


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default


def pct(x: float | None, nd: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{nd}f}%"


def mean(xs: Iterable[float]) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "replace")).hexdigest()[:12]


def human_time(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


def flatten(xss: Iterable[Iterable[Any]]) -> list:
    out: list = []
    for xs in xss:
        out.extend(xs)
    return out


try:  # numpy and matplotlib ship with Colab; metrics-only stages work without them
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

if plt is not None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": ":",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )


def require_plotting() -> None:
    if np is None or plt is None:
        sys.exit(
            "Plotting dependencies are missing. Install them first:\n"
            "  !pip install -q numpy matplotlib\n"
            "(metrics are still computed without them, only the figures need them)"
        )

# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------
@dataclass
class ItemScore:
    """Everything measurable about one (item, system) pair."""

    set_name: str
    system: str
    idx: int
    question: str
    gold: str
    pred: str
    raw: str
    valid: bool
    em: bool
    gold_scorable: bool
    ex: bool | None
    outcome: str
    complexity: str
    gold_sig: str
    pred_sig: str
    gold_sig_full: str
    pred_sig_full: str
    gold_flags: dict = field(default_factory=dict)
    pred_flags: dict = field(default_factory=dict)
    gold_tables: list = field(default_factory=list)
    pred_tables: list = field(default_factory=list)
    gold_cols: list = field(default_factory=list)
    pred_cols: list = field(default_factory=list)
    cand_tables: list = field(default_factory=list)
    cand_cols: list = field(default_factory=list)
    token_f1: float = 0.0
    token_f1_nolit: float = 0.0
    edit_sim: float = 0.0
    clause_jaccard: float = 0.0
    component_f1: float = 0.0
    link_p: float = 0.0
    link_r: float = 0.0
    link_f: float = 0.0
    error: str = ""
    conf: float | None = None
    conf_extra: dict = field(default_factory=dict)
    sc_pass: bool | None = None
    sc_majority: bool | None = None
    sc_agree: float | None = None
    sc_latency: float | None = None
    conf_extra_robustness: dict = field(default_factory=dict)
    robust_extra: dict = field(default_factory=dict)
    extra_robustness: dict = field(default_factory=dict)
    robust: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        return {
            "set": self.set_name,
            "system": self.system,
            "idx": self.idx,
            "question": self.question,
            "gold": self.gold,
            "pred": self.pred,
            "valid_sql": int(self.valid),
            "exact_match": int(self.em),
            "exec_scorable": int(self.gold_scorable),
            "exec_correct": "" if self.ex is None else int(self.ex),
            "outcome": self.outcome,
            "complexity": self.complexity,
            "gold_signature": self.gold_sig,
            "pred_signature": self.pred_sig,
            "token_f1": round(self.token_f1, 4),
            "token_f1_no_literals": round(self.token_f1_nolit, 4),
            "edit_similarity": round(self.edit_sim, 4),
            "clause_jaccard": round(self.clause_jaccard, 4),
            "component_f1": round(self.component_f1, 4),
            "schema_link_precision": round(self.link_p, 4),
            "schema_link_recall": round(self.link_r, 4),
            "schema_link_f1": round(self.link_f, 4),
            "error_type": self.error,
            "confidence": "" if self.conf is None else round(self.conf, 6),
            "self_consistency_pass_at_k": "" if self.sc_pass is None else int(self.sc_pass),
            "self_consistency_majority": "" if self.sc_majority is None else int(self.sc_majority),
            "self_consistency_agreement": "" if self.sc_agree is None else round(self.sc_agree, 4),
        }


@dataclass
class RunPaths:
    out: Path
    eval_dir: Path
    preds_dir: Path
    adapter_dir: Path
    config: dict
    train_log: list
    preds: dict = field(default_factory=dict)
    manifest: dict = field(default_factory=dict)

    @property
    def figures_dir(self) -> Path:
        return self.eval_dir / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.eval_dir / "tables"


@dataclass
class Ctx:
    """Everything the metric, table and figure builders need."""

    paths: RunPaths
    items: dict
    scores: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    pooled: dict = field(default_factory=dict)
    confidence: dict = field(default_factory=dict)
    self_consistency: dict = field(default_factory=dict)
    robustness: dict = field(default_factory=dict)
    complexity_rows: dict = field(default_factory=dict)
    tables: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    verify: dict = field(default_factory=dict)
    figures: list = field(default_factory=list)
    args: Any = None

    def systems_in(self, set_name: str) -> list:
        return [s for s in SYSTEMS if (set_name, s) in self.scores]

    def sets_in(self) -> list:
        return [s for s in SET_NAMES if any((s, y) in self.scores for y in SYSTEMS)]

    @property
    def all_systems(self) -> list:
        return [s for s in SYSTEMS if s in self.pooled]

# --------------------------------------------------------------------------
