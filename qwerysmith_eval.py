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
# 1. Scoring engine
#    Vendored from QwerySmith.py so this suite runs standalone in Colab.
#    `--stage verify` proves it agrees with the original file and with sklearn.
# --------------------------------------------------------------------------
def split_statements(sql: str) -> list[str]:
    out, buf = [], ""
    for part in sql.split(";"):
        buf += part
        if sqlite3.complete_statement(buf + ";"):
            if buf.strip():
                out.append(buf.strip())
            buf = ""
        else:
            buf += ";"
    if buf.strip():
        out.append(buf.strip().rstrip(";"))
    return out


def schema_only(context: str) -> str:
    """Keep only CREATE TABLE / CREATE VIEW statements (the model never sees INSERT rows)."""
    keep = [s + ";" for s in split_statements(context) if re.match(r"(?is)^create\s+(table|view)\b", s)]
    return "\n".join(keep) if keep else context.strip()


def _literals(sql: str):
    strs = re.findall(r"'([^']*)'", sql)
    nums = []
    for x in re.findall(r"(?<![\w.])\d+(?:\.\d+)?", sql):
        nums.append(float(x) if "." in x else int(x))
    return strs, nums


def populate_empty_tables(conn: sqlite3.Connection, gold: str, seed: int, n_rows: int = 40) -> None:
    """Fill empty tables with random data seeded with the gold query's own literals."""
    rng = random.Random(seed)
    strs, nums = _literals(gold)
    text_pool = strs + ["alpha", "beta", "gamma", "delta", "north", "south", "x", "y"]
    date_pool = [s for s in strs if re.match(r"\d{4}-\d{2}", s)]
    int_pool = [n for n in nums if isinstance(n, int)] + list(range(1, 11))
    real_pool = [float(n) for n in nums] + [round(rng.uniform(1, 100), 2) for _ in range(10)]
    try:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
    except sqlite3.Error:
        return
    for t in tables:
        try:
            if conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] > 0:
                continue
            cols = conn.execute(f'PRAGMA table_info("{t}")').fetchall()
        except sqlite3.Error:
            continue
        rows = []
        for _ in range(n_rows):
            row = []
            for c in cols:
                ty = (c[2] or "").upper()
                if "INT" in ty:
                    row.append(rng.choice(int_pool))
                elif any(k in ty for k in ("REAL", "FLOA", "DOUB", "DEC", "NUM")):
                    row.append(rng.choice(real_pool))
                elif "DATE" in ty or "TIME" in ty:
                    if date_pool and rng.random() < 0.5:
                        row.append(rng.choice(date_pool))
                    else:
                        row.append(f"20{rng.randint(20, 24)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}")
                else:
                    row.append(rng.choice(text_pool))
            rows.append(row)
        try:
            conn.executemany(f'INSERT INTO "{t}" VALUES ({",".join("?" * len(cols))})', rows)
        except sqlite3.Error:
            pass


def make_db(context: str, gold: str, seed: int) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    for stmt in split_statements(context):
        try:
            conn.execute(stmt)
        except sqlite3.Error:
            pass
    populate_empty_tables(conn, gold, seed=seed)
    return conn


def run_query(conn: sqlite3.Connection, sql: str, timeout: float = 3.0):
    if not re.match(r"(?is)^\s*(select|with)\b", sql or ""):
        return False, None
    start = time.time()
    conn.set_progress_handler(lambda: 1 if time.time() - start > timeout else 0, 10000)
    try:
        return True, conn.execute(sql).fetchmany(1000)
    except Exception:  # noqa: BLE001
        return False, None
    finally:
        conn.set_progress_handler(None, 0)


def _canon(rows) -> list[str]:
    return sorted(repr(tuple(round(v, 4) if isinstance(v, float) else v for v in r)) for r in rows)


def _norm(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";").lower()).strip()


def clean_sql(text: str) -> str:
    """Lenient extraction so the base model is not punished for chatty formatting."""
    text = re.sub(r"(?s)<think>.*?</think>", "", text)
    m = re.search(r"(?is)```(?:sql)?\s*(.*?)```", text)
    if m:
        text = m.group(1)
    text = text.strip()
    m = re.search(r"(?is)\b(select|with)\b", text)
    if m:
        text = text[m.start():]
    stmts = split_statements(text)
    return (stmts[0] if stmts else text).strip()


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _loader_args(ns) -> argparse.Namespace:
    """Minimal namespace the vendored dataset loaders need."""
    return argparse.Namespace(n_train=int(ns.n_train or 0), n_test=int(ns.n_test or 200), n_external=0)


def load_in_dist(args) -> tuple[list[dict], list[dict]]:
    """Verbatim mirror of QwerySmith.load_in_dist (same seed, same order)."""
    from datasets import load_dataset

    log(f"Loading {IN_DIST_DATASET} ...")
    ds = load_dataset(IN_DIST_DATASET, split="train")
    rows = [
        {
            "question": r["question"],
            "context": r["context"],
            "schema": schema_only(r["context"]),
            "gold": r["answer"].strip(),
        }
        for r in ds
    ]
    random.Random(SEED).shuffle(rows)
    test, rest = rows[: args.n_test], rows[args.n_test:]
    test_q = {r["question"] for r in test}
    train = [r for r in rest if r["question"] not in test_q]
    if args.n_train:
        train = train[: args.n_train]
    return train, test


def load_external(args) -> list[dict]:
    """Verbatim mirror of QwerySmith.load_external (same seed, same order)."""
    from datasets import load_dataset

    log(f"Loading {EXTERNAL_DATASET} (external test) ...")
    try:
        ds = load_dataset(EXTERNAL_DATASET, split="test")
    except Exception as e:  # noqa: BLE001
        log(f"WARNING: could not load external set, skipping it ({e})")
        return []
    idx = list(range(len(ds)))
    random.Random(SEED).shuffle(idx)
    items = []
    for i in idx:
        r = ds[i]
        it = {
            "question": r["sql_prompt"],
            "context": r["sql_context"],
            "schema": schema_only(r["sql_context"]),
            "gold": r["sql"].strip(),
        }
        conn = make_db(it["context"], it["gold"], seed=0)
        ok, rows = run_query(conn, it["gold"])
        conn.close()
        if ok and rows:
            items.append(it)
        if len(items) >= args.n_external:
            break
    with_rows = sum("insert into" in it["context"].lower() for it in items)
    log(f"external items usable: {len(items)} ({with_rows} ship with their own INSERT rows)")
    return items


def derive_items(args, cfg: dict) -> dict:
    """Re-build the evaluation sets exactly as the training run did."""
    ns = _loader_args(cfg)
    ns.n_train = int(cfg.get("n_train", 0) or 0)
    if args.n_test:
        ns.n_test = args.n_test
    n_ext = int(cfg.get("n_external", 0) or 0) if args.n_external is None else args.n_external
    ns.n_external = n_ext
    _, in_test = load_in_dist(ns)
    items = {"in_dist": in_test}
    if n_ext:
        ext = load_external(ns)
        if ext:
            items["external"] = ext
    return items


def items_from_json(path: Path) -> dict:
    """Load a saved items dump: {"in_dist": [...], ...} or a flat list with a "set" key."""
    data = json_load(path, None)
    if data is None:
        return {}
    if isinstance(data, dict) and "in_dist" in data:
        return {k: v for k, v in data.items() if isinstance(v, list)}
    if isinstance(data, list):
        out: dict = defaultdict(list)
        for row in data:
            out[row.get("set", "in_dist")].append(row)
        return dict(out)
    return {}


def resolve_items(args, paths: RunPaths) -> dict:
    """Item lists in the exact order the cached predictions were produced.

    Priority: --items-json  ->  <out>/eval/items.json  ->  re-derive from Hugging Face.
    """
    auto = paths.eval_dir / "items.json"
    items: dict = {}
    if args.items_json:
        items = items_from_json(Path(args.items_json))
        if items:
            log(f"Evaluation items: --items-json {args.items_json}")
    if not items and auto.exists():
        items = items_from_json(auto)
        if items:
            log(f"Evaluation items: {auto}")
    if not items:
        if getattr(args, "offline", False):
            sys.exit(
                "No cached items found and --offline was given.\n"
                f"Expected {auto} or pass --items-json. Run once without --offline to build it."
            )
        items = derive_items(args, paths.config)
        log("Evaluation items: re-derived from the Hugging Face datasets")
        if not getattr(args, "no_dump_items", False):
            json_dump(auto, items)
            log(f"  cached to {auto} so later runs never download anything again")

    for set_name in list(items):
        for system in SYSTEMS:
            payload = paths.preds.get((system, set_name))
            if not payload:
                continue
            n_pred, n_item = len(payload.get("pred", [])), len(items[set_name])
            if n_pred > n_item:
                log(
                    f"WARNING: {system}/{set_name} has {n_pred} predictions but only {n_item} items:\n"
                    f"         the cached run used a larger --n-test/--n-external than this script.\n"
                    f"         Re-run:  python QwerySmith.py --stage eval --out {paths.out}"
                )
                if not args.allow_mismatch:
                    sys.exit("Refusing to guess the alignment. Use --allow-mismatch to continue anyway.")
            elif n_pred < n_item:
                log(f"note: {system}/{set_name} covers {n_pred}/{n_item} items (partial run); scoring that prefix")
                items[set_name] = items[set_name][:n_pred]
    return items


def find_pipeline(start: Path | None = None) -> Path | None:
    """Locate QwerySmith.py (explicit --pipeline, next to this file, or the cwd)."""
    if start and Path(start).exists():
        return Path(start)
    here = Path(__file__).resolve().parent
    for cand in (here / "QwerySmith.py", Path.cwd() / "QwerySmith.py", here.parent / "QwerySmith.py"):
        if cand.exists():
            return cand
    return None


def load_pipeline_module(path: Path):
    """Import QwerySmith.py by path. Its heavy imports are function-local, so this is cheap."""
    try:
        spec = importlib.util.spec_from_file_location("qwerysmith_pipeline", path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:  # noqa: BLE001
        log(f"(could not import {path}: {e})")
        return None


def user_msg(it: dict) -> dict:
    return {"role": "user", "content": f"Schema:\n{it['schema']}\n\nQuestion: {it['question']}"}


def build_messages(it: dict, shots=()) -> list:
    msgs = [{"role": "system", "content": SYSTEM}]
    for s in shots:
        msgs += [user_msg(s), {"role": "assistant", "content": s["gold"]}]
    msgs.append(user_msg(it))
    return msgs


def render_prompt(tok, messages: list) -> str:
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:  # chat templates without the thinking switch
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# --------------------------------------------------------------------------
# 3. SQL analysis: text normalisation, clause flags, signatures, schema linking
# --------------------------------------------------------------------------
_STR_RE = re.compile(r"'(?:[^']|'')*'")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUM_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|[^\sA-Za-z0-9_]")
_KEYWORDS = {
    "select", "from", "where", "group", "by", "order", "having", "limit", "offset", "join", "inner",
    "left", "right", "full", "outer", "cross", "on", "as", "and", "or", "not", "in", "is", "null",
    "distinct", "union", "all", "except", "intersect", "with", "case", "when", "then", "else", "end",
    "asc", "desc", "between", "like", "exists", "using", "natural", "values", "recursive",
} | set(AGG_FUNCS)


def strip_literals(sql: str) -> str:
    """Blank out string literals so keyword searches never fire inside data."""
    return _STR_RE.sub(" '' ", sql or "")


def tokenize(sql: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(sql or "")]


def tokenize_no_literals(sql: str) -> list[str]:
    """Token list with literal contents replaced by a placeholder (structure-aware F1)."""
    s = _STR_RE.sub(" '?' ", sql or "")
    s = _NUM_RE.sub(" 0 ", s)
    return tokenize(s)


def clause_flags(sql: str) -> dict:
    """Which SQL constructs does this query use? Works on any dialect-ish text."""
    s = strip_literals(sql or "")
    low = s.lower()
    flags = {
        "distinct": bool(re.search(r"\bdistinct\b", low)),
        "aggregate": bool(re.search(r"\b(" + "|".join(AGG_FUNCS) + r")\s*\(", low)),
        "join": bool(re.search(r"\bjoin\b", low)) or bool(re.search(r",\s*\w+\s+on\b", low)),
        "where": bool(re.search(r"\bwhere\b", low)),
        "group_by": bool(re.search(r"\bgroup\s+by\b", low)),
        "having": bool(re.search(r"\bhaving\b", low)),
        "order_by": bool(re.search(r"\border\s+by\b", low)),
        "limit": bool(re.search(r"\blimit\b", low)) or bool(re.search(r"\bfetch\s+(first|next)\b", low)),
        "subquery": len(re.findall(r"\bselect\b", low)) > 1 or bool(re.search(r"\bexists\s*\(", low)),
        "set_op": bool(re.search(r"\b(union|intersect|except)\b", low)),
        "cte": bool(re.match(r"\s*with\b", low)),
        "case_when": bool(re.search(r"\bcase\b", low)) and bool(re.search(r"\bwhen\b", low)),
        "star": bool(re.search(r"select\s+(distinct\s+)?\*", low)),
        "offset": bool(re.search(r"\boffset\b", low)),
        "string_literal": bool(_STR_RE.search(sql or "")),
        "numeric_literal": bool(_NUM_RE.search(strip_literals(sql or ""))),
    }
    return flags


def signature_full(sql: str) -> str:
    """Ordered, human readable signature of the structural constructs."""
    flags = clause_flags(sql)
    parts = []
    if flags["cte"]:
        parts.append("CTE")
    if flags["set_op"]:
        parts.append("SETOP")
    if flags["subquery"]:
        parts.append("SUBQ")
    parts.append("AGG" if flags["aggregate"] else "PLAIN")
    if flags["distinct"]:
        parts.append("DISTINCT")
    if flags["join"]:
        parts.append("JOIN")
    if flags["where"]:
        parts.append("WHERE")
    if flags["group_by"]:
        parts.append("GROUP")
    if flags["having"]:
        parts.append("HAVING")
    if flags["order_by"] or flags["limit"]:
        parts.append("ORDERLIMIT")
    if flags["case_when"]:
        parts.append("CASE")
    return "+".join(parts)


def classify_complexity(sql: str) -> str:
    """Coarse task family of a query, used for the 'accuracy by complexity' breakdown."""
    f = clause_flags(sql)
    if f["set_op"]:
        return "set operation"
    if f["cte"] or f["subquery"]:
        return "subquery / CTE"
    if f["aggregate"] and f["group_by"]:
        return "aggregation + grouping"
    if f["aggregate"]:
        return "simple aggregation"
    if f["join"]:
        return "join"
    if f["distinct"]:
        return "distinct projection"
    if f["order_by"] or f["limit"]:
        return "ordering / top-k"
    if f["where"]:
        return "filter"
    return "plain projection"


def _split_top_level(s: str) -> list[str]:
    out, depth, buf = [], 0, ""
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(buf)
            buf = ""
        else:
            buf += ch
    if buf.strip():
        out.append(buf)
    return out


def parse_schema(schema: str) -> dict:
    """Tables -> columns, plus view names, parsed out of the CREATE statements."""
    tables: dict = {}
    views: list = []
    for stmt in split_statements(schema or ""):
        m = re.match(
            r"(?is)^\s*create\s+table\s+(?:if\s+not\s+exists\s+)?[`\"\[]?(\w+)[`\"\]]?\s*\((.*)\)\s*$",
            stmt.strip(),
        )
        if m:
            cols = []
            for part in _split_top_level(m.group(2)):
                part = part.strip()
                if not part:
                    continue
                first = re.sub(r'[`"\[\]]', "", part.split()[0])
                if first.lower() in {"primary", "foreign", "unique", "constraint", "check", "key", "index"}:
                    continue
                if re.match(r"^[A-Za-z_]\w*$", first):
                    cols.append(first.lower())
            tables[m.group(1).lower()] = cols
            continue
        m = re.match(r"(?is)^\s*create\s+view\s+[`\"\[]?(\w+)", stmt.strip())
        if m:
            views.append(m.group(1).lower())
    return {
        "tables": tables,
        "views": views,
        "names": set(tables) | set(views),
        "columns": {c for cols in tables.values() for c in cols},
    }


def extract_refs(sql: str, schema: dict) -> dict:
    """Which schema tables/columns does this query reference? (this is schema linking)"""
    s = strip_literals(sql or "")
    names, cols = schema["names"], schema["columns"]
    aliases: dict = {}
    for m in re.finditer(
        r"(?is)\b(?:from|join|into|update)\s+[`\"\[]?(\w{1,64})[`\"\]]?"
        r"(?:\s+(?:as\s+)?[`\"\[]?(\w{1,64})[`\"\]]?)?",
        s,
    ):
        t = m.group(1).lower()
        alias = (m.group(2) or "").lower()
        if t in names and alias and alias not in _KEYWORDS and alias != t:
            aliases[alias] = t
    tables: set = set(aliases.values())
    used: set = set()
    for m in re.finditer(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b", s):
        qual, col = m.group(1).lower(), m.group(2).lower()
        if col in cols:
            used.add(col)
        resolved = aliases.get(qual, qual)
        if resolved in names:
            tables.add(resolved)
    for tok in _IDENT_RE.findall(s.lower()):
        if tok in names:
            tables.add(tok)
        if tok in cols:
            used.add(tok)
    return {"tables": sorted(tables), "columns": sorted(used), "aliases": aliases}


def levenshtein(a: str, b: str) -> int:
    """Character-level edit distance (two-row DP)."""
    if a == b:
        return 0
    if not a or not b:
        return len(a) or len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def edit_similarity(a: str, b: str) -> float:
    a, b = _norm(a or ""), _norm(b or "")
    return 1.0 - safe_div(levenshtein(a, b), max(len(a), len(b), 1))


def prf(tp: float, fp: float, fn: float) -> tuple:
    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    return p, r, safe_div(2 * p * r, p + r)


def token_f1(pred: str, gold: str, no_literals: bool = False) -> float:
    """Multiset token F1 between two SQL strings."""
    fn = tokenize_no_literals if no_literals else tokenize
    a, b = fn(pred), fn(gold)
    if not a or not b:
        return 0.0
    ca, cb = Counter(a), Counter(b)
    overlap = sum((ca & cb).values())
    if not overlap:
        return 0.0
    p, r = overlap / len(a), overlap / len(b)
    return safe_div(2 * p * r, p + r)


def clause_jaccard(fa: dict, fb: dict) -> float:
    """Soft overlap of the construct sets (Jaccard over the tracked clauses)."""
    keys = [k for k, _ in CLAUSES]
    inter = sum(1 for k in keys if fa.get(k) and fb.get(k))
    union = sum(1 for k in keys if fa.get(k) or fb.get(k))
    return safe_div(inter, union)


def component_f1(fa: dict, fb: dict) -> float:
    """Micro F1 over clause flags: did the model reproduce the required constructs?"""
    keys = [k for k, _ in CLAUSES]
    tp = sum(1 for k in keys if fa.get(k) and fb.get(k))
    fp = sum(1 for k in keys if fb.get(k) and not fa.get(k))
    fn = sum(1 for k in keys if fa.get(k) and not fb.get(k))
    return prf(tp, fp, fn)[2]


def schema_link(gold_refs: dict, pred_refs: dict) -> tuple:
    """Precision/recall/F1 over the set of schema tables + columns used."""
    counts = {"tp": 0, "fp": 0, "fn": 0}
    for key in ("tables", "columns"):
        g, p = set(gold_refs[key]), set(pred_refs[key])
        counts["tp"] += len(g & p)
        counts["fp"] += len(p - g)
        counts["fn"] += len(g - p)
    p, r, f = prf(counts["tp"], counts["fp"], counts["fn"])
    return p, r, f, counts


def _mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    """Matthews correlation coefficient (0 when a whole row or column is empty)."""
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return safe_div(tp * tn - fp * fn, den)


def cm2(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    """Binary confusion matrix with the usual derived scores."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    p, r, f1 = prf(tp, fp, fn)
    n = tp + tn + fp + fn
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": n,
        "precision": p, "recall": r, "f1": f1,
        "accuracy": safe_div(tp + tn, n),
        "specificity": safe_div(tn, tn + fp),
        "npv": safe_div(tn, tn + fn),
        "balanced_accuracy": 0.5 * (safe_div(tp, tp + fn) + safe_div(tn, tn + fp)),
        "mcc": _mcc(tp, tn, fp, fn),
    }


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] | None = None) -> dict:
    """Multiclass confusion matrix with per-class scores and macro/weighted/micro F1."""
    labels = list(labels) if labels is not None else sorted(set(list(y_true) + list(y_pred)))
    idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    mat = [[0] * k for _ in range(k)]
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            mat[idx[t]][idx[p]] += 1
    per = {}
    for i, lab in enumerate(labels):
        tp = mat[i][i]
        fp = sum(mat[r][i] for r in range(k)) - tp
        fn = sum(mat[i]) - tp
        p, r, f1 = prf(tp, fp, fn)
        per[lab] = {
            "precision": p, "recall": r, "f1": f1,
            "support": sum(mat[i]), "tp": tp, "fp": fp, "fn": fn,
        }
    support = {lab: per[lab]["support"] for lab in labels}
    tp_all = sum(mat[i][i] for i in range(k))
    fp_all = sum(mat[i][j] for i in range(k) for j in range(k) if i != j)
    fn_all = fp_all
    _, _, micro_f1 = prf(tp_all, fp_all, fn_all)
    total = sum(support.values())
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return {
        "labels": labels,
        "matrix": mat,
        "per_class": per,
        "support": support,
        "accuracy": safe_div(correct, total),
        "macro_f1": mean([per[l]["f1"] for l in labels]),
        "weighted_f1": safe_div(sum(per[l]["f1"] * support[l] for l in labels), total),
        "micro_f1": micro_f1,
        "balanced_accuracy": mean([per[l]["recall"] for l in labels]),
        "cohen_kappa": cohen_kappa(list(y_true), list(y_pred), "none"),
        "cohen_kappa_linear": cohen_kappa(list(y_true), list(y_pred), "linear"),
        "cohen_kappa_quadratic": cohen_kappa(list(y_true), list(y_pred), "quadratic"),
        "krippendorff_alpha": 0.0,
        "n": total,
    }

# --------------------------------------------------------------------------
# 4. Statistics: chance-corrected agreement, tests, resampling, curve metrics
# --------------------------------------------------------------------------
def _label_order(labels: Sequence[str]) -> list[str]:
    return sorted(set(labels))


def cohen_kappa(a: Sequence[str], b: Sequence[str], weights: str = "none") -> float:
    """Cohen's kappa with optional linear/quadratic weights (matches sklearn)."""
    pairs = list(zip(a, b))
    if not pairs:
        return 0.0
    labels = _label_order([x for p in pairs for x in p])
    idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    n = len(pairs)
    obs = [[0] * k for _ in range(k)]
    for x, y in pairs:
        obs[idx[x]][idx[y]] += 1
    row = [sum(r) for r in obs]
    col = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    if weights == "none":
        w = [[0 if i == j else 1 for j in range(k)] for i in range(k)]
    else:
        w = [
            [(abs(i - j) / (k - 1)) ** (1 if weights == "linear" else 2) if k > 1 else 0 for j in range(k)]
            for i in range(k)
        ]
    p_obs = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k)) / n
    p_exp = sum(w[i][j] * row[i] * col[j] for i in range(k) for j in range(k)) / (n * n)
    if p_exp == 0:
        return 1.0 if p_obs == 0 else 0.0
    return 1.0 - p_obs / p_exp


def percent_agreement(a: Sequence[str], b: Sequence[str]) -> float:
    pairs = list(zip(a, b))
    return safe_div(sum(1 for x, y in pairs if x == y), len(pairs))


def pabak(a: Sequence[str], b: Sequence[str]) -> float:
    """Prevalence-adjusted bias-adjusted kappa."""
    return max(-1.0, min(1.0, 2 * percent_agreement(a, b) - 1))


def gwet_ac1(a: Sequence[str], b: Sequence[str]) -> float:
    """Gwet's AC1: agreement coefficient that stays stable under class imbalance."""
    pairs = list(zip(a, b))
    if not pairs:
        return 0.0
    n = len(pairs)
    labels = _label_order([x for p in pairs for x in p])
    p_g = {}
    for lab in labels:
        p_g[lab] = sum(1 for p in pairs for x in p if x == lab) / (2 * n)
    p_e = sum(p * (1 - p) for p in p_g.values())
    po = percent_agreement(a, b)
    if p_e == 0:
        return 1.0 if po == 1 else 0.0
    return (po - p_e) / (1 - p_e)


def fleiss_kappa(ratings: Sequence[Sequence[str]], labels: Sequence[str] | None = None) -> float:
    """Fleiss' kappa over items x raters (raters = the systems being compared)."""
    rows = [r for r in ratings if r]
    if not rows:
        return 0.0
    labels = list(labels) if labels else _label_order([x for r in rows for x in r])
    idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    n = len(rows)
    counts = []
    for r in rows:
        cnt = [0] * k
        for x in r:
            if x in idx:
                cnt[idx[x]] += 1
        counts.append(cnt)
    m = mean([sum(c) for c in counts])
    if m < 2:
        return 0.0
    p_i = safe_div(sum(sum(c * c for c in cnt) for cnt in counts) - n * m, n * m * (m - 1))
    p_j = [safe_div(sum(cnt[j] for cnt in counts), n * m) for j in range(k)]
    p_e = sum(p * p for p in p_j)
    if p_e >= 1:
        return 1.0 if p_i >= 1 else 0.0
    return (p_i - p_e) / (1 - p_e)


def krippendorff_alpha(ratings: Sequence[Sequence[str]]) -> float:
    """Krippendorff's alpha for nominal data (coincidence-matrix formulation)."""
    rows = [r for r in ratings if len(r) >= 2]
    if not rows:
        return 0.0
    labels = _label_order([x for r in rows for x in r])
    idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    o = [[0.0] * k for _ in range(k)]
    for r in rows:
        m_u = len(r)
        cnt = [0] * k
        for x in r:
            cnt[idx[x]] += 1
        for c in range(k):
            for j in range(k):
                if c == j:
                    o[c][j] += cnt[c] * (cnt[c] - 1) / (m_u - 1)
                else:
                    o[c][j] += cnt[c] * cnt[j] / (m_u - 1)
    total = sum(sum(r) for r in o)
    if total <= 1:
        return 0.0
    do = sum(o[c][j] for c in range(k) for j in range(k) if c != j) / total
    n_c = [sum(o[c]) for c in range(k)]
    de = safe_div(sum(n_c[c] * (total - n_c[c]) for c in range(k)), total * (total - 1))
    if de == 0:
        return 1.0 if do == 0 else 0.0
    return 1.0 - do / de


def binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test (used by McNemar's exact test)."""
    if n == 0:
        return 1.0

    def pmf(i: int) -> float:
        return math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))

    observed = pmf(k)
    total = sum(pmf(i) for i in range(n + 1) if pmf(i) <= observed + 1e-12)
    return min(1.0, total)


def _chi2_sf_1df(x: float) -> float:
    """Survival function of a chi-square distribution with 1 degree of freedom."""
    return 1.0 if x <= 0 else math.erfc(math.sqrt(x / 2.0))


def mcnemar(a_correct: Sequence[bool | None], b_correct: Sequence[bool | None]) -> dict:
    """Paired comparison of two systems on exactly the same items."""
    pairs = [(bool(x), bool(y)) for x, y in zip(a_correct, b_correct) if x is not None and y is not None]
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    n = b + c
    exact = binom_two_sided(min(b, c), n) if n else 1.0
    chi2 = safe_div((abs(b - c) - 1) ** 2, n)
    return {
        "n_pairs": len(pairs),
        "a_only_correct": b,
        "b_only_correct": c,
        "ties_correct": sum(1 for x, y in pairs if x and y),
        "ties_wrong": sum(1 for x, y in pairs if not x and not y),
        "exact_p": exact,
        "chi2": chi2,
        "chi2_p": _chi2_sf_1df(chi2) if n else 1.0,
        "odds_ratio": safe_div(b, c, default=float("inf") if b else 0.0),
        "significant_05": bool(exact < 0.05) if n else False,
    }


def holm_bonferroni(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p-values, preserving the input keys."""
    items = sorted(((k, v) for k, v in pvals.items() if v is not None), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, v) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * v))
        running = adj
        out[k] = adj
    return out


def bootstrap_mean_ci(values: Sequence[float], n_boot: int = 2000, seed: int = SEED) -> tuple:
    """Percentile bootstrap CI of the mean (pure Python is fast enough at these sizes)."""
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 2:
        return (vals[0] if vals else 0.0, vals[0] if vals else 0.0)
    rng = random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(n_boot):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * n_boot)], means[min(n_boot - 1, int(0.975 * n_boot))]


def paired_bootstrap_delta(a: Sequence, b: Sequence, n_boot: int = 2000, seed: int = SEED) -> dict:
    """Bootstrap distribution of mean(a) - mean(b) over paired items."""
    pairs = [(float(x), float(y)) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 2:
        return {"delta": 0.0, "lo": 0.0, "hi": 0.0, "p_two_sided": 1.0, "prob_better": 0.5, "draws": []}
    rng = random.Random(seed)
    n = len(pairs)
    draws = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(n):
            i = rng.randrange(n)
            total += pairs[i][0] - pairs[i][1]
        draws.append(total / n)
    draws.sort()
    delta = mean([x - y for x, y in pairs])
    frac_pos = safe_div(sum(1 for d in draws if d > 0), len(draws))
    return {
        "delta": delta,
        "lo": draws[int(0.025 * n_boot)],
        "hi": draws[min(n_boot - 1, int(0.975 * n_boot))],
        "p_two_sided": min(1.0, 2 * min(frac_pos, 1 - frac_pos)),
        "prob_better": frac_pos,
        "draws": draws,
    }


def auc_roc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """AUC-ROC via the rank (Mann-Whitney) formula; ties get average ranks."""
    pairs = [(float(s), int(y)) for s, y in zip(scores, labels) if s is not None]
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = sum(1 for _, y in pairs if y == 0)
    if not n_pos or not n_neg:
        return 0.0
    ranked = sorted(enumerate(pairs), key=lambda it: it[1][0])
    ranks = [0.0] * len(ranked)
    i = 0
    while i < len(ranked):
        j = i
        while j + 1 < len(ranked) and ranked[j + 1][1][0] == ranked[i][1][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    rank_sum_pos = sum(ranks[k] for k, (_, (_, y)) in enumerate(ranked) if y == 1)
    return safe_div(rank_sum_pos - n_pos * (n_pos + 1) / 2.0, n_pos * n_neg)


def auc_pr(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Average precision, i.e. the area under the precision-recall curve."""
    pairs = sorted([(float(s), int(y)) for s, y in zip(scores, labels) if s is not None], key=lambda p: -p[0])
    total_pos = sum(y for _, y in pairs)
    if not pairs or total_pos == 0:
        return 0.0
    tp, ap, prev_recall = 0, 0.0, 0.0
    for i, (_, y) in enumerate(pairs, 1):
        tp += y
        if y == 1:
            recall = tp / total_pos
            ap += (recall - prev_recall) * (tp / i)
            prev_recall = recall
    return ap


def calibration(y_true: Sequence[int], conf: Sequence[float], n_bins: int = 15) -> dict:
    """Reliability-diagram data plus ECE / MCE / Brier / NLL / AUC."""
    pairs = [(float(c), int(y)) for c, y in zip(conf, y_true) if c is not None]
    if not pairs:
        return {}
    buckets: list = [[] for _ in range(n_bins)]
    for c, y in pairs:
        buckets[min(n_bins - 1, max(0, int(c * n_bins)))].append((c, y))
    rows, ece, mce = [], 0.0, 0.0
    for b, items in enumerate(buckets):
        if not items:
            continue
        acc = mean([y for _, y in items])
        avg_conf = mean([c for c, _ in items])
        gap = abs(acc - avg_conf)
        ece += gap * len(items) / len(pairs)
        mce = max(mce, gap)
        rows.append(
            {"bin": b, "lo": b / n_bins, "hi": (b + 1) / n_bins, "n": len(items),
             "confidence": avg_conf, "accuracy": acc, "gap": gap}
        )
    clipped = [min(1 - 1e-6, max(1e-6, c)) for c, _ in pairs]
    ys = [y for _, y in pairs]
    brier = mean([(c - y) ** 2 for c, y in zip(clipped, ys)])
    nll = -mean([y * math.log(c) + (1 - y) * math.log(1 - c) for c, y in zip(clipped, ys)])
    return {
        "n": len(pairs),
        "bins": rows,
        "ece": ece,
        "mce": mce,
        "brier": brier,
        "nll": nll,
        "auc_roc": auc_roc([c for c, _ in pairs], ys),
        "auc_pr": auc_pr([c for c, _ in pairs], ys),
        "mean_confidence": mean([c for c, _ in pairs]),
        "accuracy": mean(ys),
    }


def risk_coverage(y_true: Sequence[int], conf: Sequence[float], grid: int = 50) -> dict:
    """Selective prediction: accuracy of the answers that are kept, most confident first."""
    pairs = sorted([(float(c), int(y)) for c, y in zip(conf, y_true) if c is not None], key=lambda p: -p[0])
    n = len(pairs)
    if n < 2:
        return {}
    coverages, accuracies, risks = [], [], []
    cum_correct = 0
    step = max(1, n // grid)
    for i, (_, y) in enumerate(pairs, 1):
        cum_correct += y
        if i % step == 0 or i == n:
            coverages.append(i / n)
            accuracies.append(cum_correct / i)
            risks.append(1 - cum_correct / i)
    at50 = accuracies[min(range(len(coverages)), key=lambda i: abs(coverages[i] - 0.5))] if coverages else 0.0
    return {
        "coverage": coverages,
        "accuracy": accuracies,
        "risk": risks,
        "aurc": mean(risks),
        "at_50pct": at50,
        "n": n,
    }

# --------------------------------------------------------------------------
# 5. Finding the finished run on disk
# --------------------------------------------------------------------------
def parse_manifest(out_dir: Path) -> dict:
    """QwerySmith writes manifest.json with an 'artifacts' list of hashes."""
    data = json_load(out_dir / "manifest.json", {}) or {}
    out = {}
    for a in data.get("artifacts", []) if isinstance(data, dict) else []:
        if isinstance(a, dict) and a.get("name"):
            out[a["name"]] = a
    return out


def _looks_like_run(p: Path) -> bool:
    return any((p / f).exists() for f in ("config.json", "train_log.json", "preds", "adapter", "manifest.json"))


def _match_pred_name(stem: str):
    """'finetuned__in_dist' / 'preds_finetuned_external' / 'finetuned' -> (system, set)"""
    low = stem.lower()
    for system in SYSTEMS:
        if system not in low:
            continue
        rest = low.replace(system, " ").strip(" _-")
        for set_name in SET_NAMES:
            if set_name in rest:
                return system, set_name
        return system, "in_dist"
    return None


def discover_runs(args) -> list:
    """Directories that look like a QwerySmith run, best candidate first."""
    found = []

    def add(p: Path) -> None:
        p = p.resolve()
        if _looks_like_run(p) and p not in found:
            found.append(p)

    if getattr(args, "out", None) and not getattr(args, "all_runs", False):
        add(Path(args.out))
    for raw in list(getattr(args, "search", []) or []):
        root = Path(raw)
        if _looks_like_run(root):
            add(root)
        for p in sorted(root.glob("*")):
            if p.is_dir() and _looks_like_run(p):
                add(p)
    if not found or getattr(args, "all_runs", False):
        cwd = Path.cwd()
        for base in (cwd, cwd / "runs", Path(__file__).resolve().parent, Path(__file__).resolve().parent / "runs"):
            add(base)
            if base.exists():
                for p in sorted(base.glob("*")):
                    if p.is_dir() and _looks_like_run(p):
                        add(p)
    if not found:
        sys.exit(
            "No QwerySmith run found.\n"
            "Point the script at it explicitly, e.g.\n"
            "  python qwerysmith_eval.py --stage figures --out /content/drive/MyDrive/QwerySmith/runs/qwerysmith-1.0\n"
            "or, to re-generate the predictions first,\n"
            "  python QwerySmith.py --stage eval --out runs/qwerysmith-1.0"
        )
    if len(found) > 1 and not getattr(args, "all_runs", False):
        log("Several runs found:")
        for i, p in enumerate(found):
            log(f"  [{i}] {p}")
        log(f"Using the first one: {found[0]}  (--out/--all-runs to choose)")
    return found


def load_run(out_dir: Path, ns) -> RunPaths:
    """Read config.json, train_log.json and every cached prediction file."""
    out_dir = Path(out_dir).resolve()
    cfg = json_load(out_dir / "config.json", {}) or {}
    for key in ("n_train", "n_test", "n_external", "model", "max_len", "seed"):
        if key in cfg and cfg.get(key) is not None:
            continue
        if hasattr(ns, key) and getattr(ns, key) is not None:
            cfg[key] = getattr(ns, key)
    train_log = json_load(out_dir / "train_log.json", None)
    if not isinstance(train_log, list):
        train_log = cfg.get("train_log") if isinstance(cfg.get("train_log"), list) else []

    preds: dict = {}
    preds_dir = out_dir / "preds"
    if preds_dir.exists():
        for f in sorted(preds_dir.glob("*.json")):
            payload = json_load(f, None)
            if not isinstance(payload, dict) or "pred" not in payload:
                continue
            key = _match_pred_name(f.stem)
            if key is None:
                log(f"note: ignoring {f.name} (cannot tell which system/set it belongs to)")
                continue
            payload["_file"] = f.name
            preds[key] = payload
    adapter_dir = out_dir / "adapter"
    if not adapter_dir.exists():
        for alt in ("final_adapter", "lora_adapter", "model", "final_model"):
            if (out_dir / alt).exists():
                adapter_dir = out_dir / alt
                break
    return RunPaths(
        out=out_dir,
        eval_dir=out_dir / "eval",
        preds_dir=preds_dir,
        adapter_dir=adapter_dir,
        config=cfg,
        train_log=train_log,
        preds=preds,
        manifest=parse_manifest(out_dir),
    )

def score_item(
    set_name: str,
    system: str,
    idx: int,
    item: dict,
    raw: str,
    pred: str,
    gold_refs: dict,
    gold_flags: dict,
    schema: dict,
    conn: sqlite3.Connection,
    gold_scorable: bool,
    gold_rows,
    complexity: str,
) -> ItemScore:
    """All per-item measurements for one (item, system) pair."""
    valid, rows = run_query(conn, pred)
    em = _norm(pred) == _norm(item["gold"])
    ex = None
    if gold_scorable:
        ex = bool(valid and _canon(rows) == _canon(gold_rows))
    outcome = (
        "invalid" if not valid
        else "executed_exact" if (em and ex)
        else "executed_match" if ex
        else "valid_wrong"
    )
    pred_flags = clause_flags(pred)
    pred_refs = extract_refs(pred, schema)
    link_p, link_r, link_f, _ = schema_link(gold_refs, pred_refs)
    return ItemScore(
        set_name=set_name,
        system=system,
        idx=idx,
        question=item.get("question", ""),
        gold=item["gold"],
        pred=pred,
        raw=raw,
        valid=valid,
        em=em,
        gold_scorable=gold_scorable,
        ex=ex,
        outcome=outcome,
        complexity=complexity,
        gold_sig=signature_full(item["gold"]) if gold_scorable else "",
        pred_sig=signature_full(pred),
        gold_sig_full=signature_full(item["gold"]) if gold_scorable else "",
        pred_sig_full=signature_full(pred),
        gold_flags=gold_flags,
        pred_flags=pred_flags,
        gold_tables=gold_refs["tables"],
        pred_tables=pred_refs["tables"],
        gold_cols=gold_refs["columns"],
        pred_cols=pred_refs["columns"],
        cand_tables=sorted(schema["names"]),
        cand_cols=sorted(schema["columns"]),
        token_f1=token_f1(pred, item["gold"]),
        token_f1_nolit=token_f1(pred, item["gold"], no_literals=True),
        edit_sim=edit_similarity(pred, item["gold"]),
        clause_jaccard=clause_jaccard(gold_flags, pred_flags) if gold_scorable else 0.0,
        component_f1=component_f1(gold_flags, pred_flags) if gold_scorable else 0.0,
        link_p=link_p,
        link_r=link_r,
        link_f=link_f,
        error="" if valid else "invalid SQL",
    )


def score_all(ctx: Ctx, quiet: bool = False) -> None:
    """Build the per-item score table for every system that has cached predictions."""
    for set_name, items in ctx.items.items():
        # ---- reference side: schema, gold references, gold result set, complexity
        refs, flags, conns, complexities, scorable, gold_rows = [], [], [], [], [], []
        for i, it in enumerate(items):
            it.setdefault("schema", schema_only(it["context"]))
            schema = parse_schema(it["schema"])
            refs.append(extract_refs(it["gold"], schema))
            flags.append(clause_flags(it["gold"]))
            complexities.append(classify_complexity(it["gold"]))
            conn = make_db(it["context"], it["gold"], seed=i)
            ok, rows = run_query(conn, it["gold"])
            conns.append(conn)
            scorable.append(bool(ok and rows))
            gold_rows.append(rows or [])
        for system in SYSTEMS:
            payload = ctx.paths.preds.get((system, set_name))
            if not payload:
                continue
            preds = payload.get("pred") or []
            raws = payload.get("raw") or preds
            scores = []
            for i in range(min(len(preds), len(items))):
                scores.append(
                    score_item(
                        set_name, system, i, items[i],
                        raws[i] if i < len(raws) else preds[i],
                        clean_sql(preds[i]) or preds[i],
                        refs[i], flags[i], parse_schema(items[i]["schema"]), conns[i],
                        scorable[i], gold_rows[i], complexities[i],
                    )
                )
            ctx.scores[(set_name, system)] = scores
            if not quiet:
                ex = [s for s in scores if s.ex is not None]
                log(
                    f"scored {system:14s} {set_name:9s} n={len(scores):4d}  "
                    f"valid={pct(mean([s.valid for s in scores]), 1)}  "
                    f"EM={pct(mean([s.em for s in scores]), 1)}  "
                    f"EX={pct(mean([s.ex for s in ex]), 1)}"
                )
        for conn in conns:
            conn.close()

def outcome_counts(rows: Sequence[ItemScore]) -> dict:
    """Kept for backwards compatibility with older notebooks: outcome histogram of a set."""
    c = {k: 0 for k in OUTCOME_LABELS}
    c["n_scorable"] = 0
    for r in rows:
        c[r.outcome] += 1
        c["n_scorable"] += int(r.gold_scorable)
    c["n"] = len(rows)
    return c




# --------------------------------------------------------------------------
# 7. Figure infrastructure
# --------------------------------------------------------------------------
class Fig:
    """A single figure: accumulates one or more axes, then saves png + svg + pdf."""

    def __init__(self, ctx: Ctx, key: str, title: str, caption: str, nrows: int = 1, ncols: int = 1, figsize=None,
                 polar: bool = False):
        require_plotting()
        self.ctx = ctx
        self.key = key
        self.title = title
        self.caption = caption
        size = figsize or (max(6.0, 7.0 * ncols), 4.2 * nrows)
        kw = {"subplot_kw": {"projection": "polar"}} if polar else {}
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=size, **kw)
        self.axes_grid = self.axes
        self.fig.suptitle(f"{MODEL_NAME} - {title}", fontsize=12, fontweight="bold")

    def ax(self, row: int = 0, col: int = 0):
        if hasattr(self.axes_grid, "shape"):
            arr = self.axes_grid if hasattr(self.axes_grid, "__len__") else [self.axes_grid]
            try:
                return arr[row][col] if len(getattr(self.axes_grid, "shape", ())) == 2 else arr[col]
            except Exception:  # noqa: BLE001
                return self.axes_grid
        return self.axes_grid

    def each(self):
        if hasattr(self.axes_grid, "flat"):
            return list(self.axes_grid.flat)
        return [self.axes_grid]

    def save(self) -> None:
        out = self.ctx.paths.figures_dir
        out.mkdir(parents=True, exist_ok=True)
        self.fig.tight_layout(rect=(0, 0.03, 1, 0.96))
        base = out / self.key
        for ext in ("png", "svg"):
            self.fig.savefig(f"{base}.{ext}", bbox_inches="tight")
        plt.close(self.fig)
        self.ctx.figures.append({"key": self.key, "title": self.title, "caption": self.caption})
        log(f"  [fig] {self.key}.png / .svg")


def style_axis(ax, title: str = "", xlabel: str = "", ylabel: str = "", xtick_rot: float = 0.0) -> None:
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xtick_rot:
        plt.sca(ax)
        plt.xticks(rotation=xtick_rot, ha="right" if xtick_rot else "center")


def bar_labels(ax, bars, values, fmt="{:.1f}%", dy=0.01) -> None:
    for bar, val in zip(bars, values):
        if val is None:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + dy,
            fmt.format(100 * val if "%" in fmt else val),
            ha="center", va="bottom", fontsize=8,
        )


def heatmap(ax, matrix, row_labels, col_labels, title="", cmap="Blues", fmt="{:.0f}", vmin=None, vmax=None,
            xlabel="", ylabel="", cbar_label="") -> None:
    arr = np.asarray(matrix, dtype=float)
    if arr.size and np.nanmax(arr) == np.nanmin(arr):
        vmax = np.nanmax(arr) or 1.0
        vmin = 0.0
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    total = arr.sum() if arr.size else 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if not arr.size or np.isnan(val):
                continue
            frac = safe_div(val, total)
            txt = fmt.format(int(val)) if fmt == "{:.0f}" else fmt.format(val)
            if frac > 0.005:
                txt += f"\n{100 * frac:.0f}%"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if arr.max() and val > 0.6 * arr.max() else "black")
    ax.grid(False)
    if cbar_label:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)

def compute_metrics(ctx: Ctx) -> None:
    """Headline metrics, confusion matrices, error matrices and significance tests."""
    for set_name in ctx.sets_in():
        for system in ctx.systems_in(set_name):
            rows = ctx.scores[(set_name, system)]
            m = headline_metrics(rows)
            m["per_clause"] = per_clause_agreement(rows)
            m["signature_matrix"] = sig_matrix(rows)
            m["outcome_matrix"] = {"labels": OUTCOME_LABELS, "counts": [m["outcomes"][k] for k in OUTCOME_LABELS]}
            m["table_link_matrix"] = cm2(
                [1 if r.gold_tables else 0 for r in rows],
                [1 if r.pred_tables else 0 for r in rows],
            )
            ctx.metrics.setdefault(set_name, {})[system] = m
            ctx.complexity_rows[(set_name, system)] = [
                {
                    "idx": r.idx,
                    "complexity": r.complexity,
                    "n_tables": len(r.cand_tables),
                    "n_cols": len(r.cand_cols),
                    "question_len": len(r.question.split()),
                    "gold_len": len(r.gold.split()),
                    "pred_len": len(r.pred.split()),
                    "valid": int(r.valid),
                    "em": int(r.em),
                    "exec": "" if r.ex is None else int(r.ex),
                    "token_f1": r.token_f1,
                    "link_f1": r.link_f,
                }
                for r in rows
            ]

    for system in SYSTEMS:
        rows = [r for set_name in ctx.sets_in() for r in ctx.scores.get((set_name, system), [])]
        if rows:
            ctx.pooled[system] = headline_metrics(rows)


def _align(a: Sequence[ItemScore], b: Sequence[ItemScore]) -> list:
    """Pair two systems' item scores by index (both cover the same items)."""
    by_idx = {r.idx: r for r in b}
    return [(r, by_idx[r.idx]) for r in a if r.idx in by_idx]


def add_comparisons(ctx: Ctx) -> None:
    """Paired significance tests and agreement coefficients between systems."""
    store = ctx.metrics.setdefault("comparisons", {})
    for set_name in ctx.sets_in():
        systems = ctx.systems_in(set_name)
        for i, sys_a in enumerate(systems):
            for sys_b in systems[i + 1:]:
                pairs = _align(ctx.scores[(set_name, sys_a)], ctx.scores[(set_name, sys_b)])
                base = f"{set_name} | {sys_a} vs {sys_b}"
                store[f"{base} (execution accuracy)"] = mcnemar([p[0].ex for p in pairs], [p[1].ex for p in pairs])
                store[f"{base} (exact match)"] = mcnemar([p[0].em for p in pairs], [p[1].em for p in pairs])
                store[f"{base} (valid SQL)"] = mcnemar([p[0].valid for p in pairs], [p[1].valid for p in pairs])
                scorable = [p for p in pairs if p[0].gold_scorable]
                boot = paired_bootstrap_delta([p[0].token_f1 for p in scorable], [p[1].token_f1 for p in scorable])
                store[f"{base} (token F1 bootstrap)"] = {k: v for k, v in boot.items() if k != "draws"}
                store[f"{base} (bootstrap draws)"] = boot["draws"]
                a_out = [p[0].outcome for p in pairs]
                b_out = [p[1].outcome for p in pairs]
                a_bin = ["correct" if p[0].ex else "wrong" for p in pairs]
                b_bin = ["correct" if p[1].ex else "wrong" for p in pairs]
                store[f"{base} (agreement)"] = {
                    "n": len(pairs),
                    "percent_agreement_binary": percent_agreement(a_bin, b_bin),
                    "cohen_kappa_binary": cohen_kappa(a_bin, b_bin),
                    "gwet_ac1_binary": gwet_ac1(a_bin, b_bin),
                    "pabak_binary": pabak(a_bin, b_bin),
                    "cohen_kappa_outcome": cohen_kappa(a_out, b_out),
                    "cohen_kappa_outcome_linear": cohen_kappa(a_out, b_out, "linear"),
                    "percent_agreement_outcome": percent_agreement(a_out, b_out),
                    "prediction_identity": mean([p[0].pred == p[1].pred for p in pairs]),
                    "token_f1_between_systems": mean([token_f1(p[0].pred, p[1].pred) for p in pairs]),
                }
        if len(systems) >= 3:
            common = set.intersection(*[{r.idx for r in ctx.scores[(set_name, s)]} for s in systems])
            by_sys = {s: {r.idx: r for r in ctx.scores[(set_name, s)]} for s in systems}
            ratings_outcome, ratings_binary = [], []
            for idx in sorted(common):
                ratings_outcome.append([by_sys[s][idx].outcome for s in systems])
                ratings_binary.append(["correct" if by_sys[s][idx].ex else "wrong" for s in systems])
            store[f"{set_name} | all systems (agreement)"] = {
                "n_items_rated_by_all": len(ratings_outcome),
                "fleiss_kappa_outcome": fleiss_kappa(ratings_outcome, OUTCOME_LABELS),
                "fleiss_kappa_binary": fleiss_kappa(ratings_binary, ["correct", "wrong"]),
                "krippendorff_alpha_outcome": krippendorff_alpha(ratings_outcome),
                "krippendorff_alpha_binary": krippendorff_alpha(ratings_binary),
                "unanimous_items": sum(1 for r in ratings_binary if len(set(r)) == 1),
                "unanimously_correct": sum(1 for r in ratings_binary if set(r) == {"correct"}),
                "unanimously_wrong": sum(1 for r in ratings_binary if set(r) == {"wrong"}),
            }
        raw = {
            k: v["exact_p"]
            for k, v in store.items()
            if k.startswith(set_name) and isinstance(v, dict) and "exact_p" in v
        }
        for k, adj in holm_bonferroni(raw).items():
            store[k]["exact_p_holm"] = adj


def attach_sidecars(ctx: Ctx) -> None:
    """Fold in the optional GPU-stage artefacts when they are present."""
    ev = ctx.paths.eval_dir
    ctx.confidence = json_load(ev / "confidence.json", {}) or {}
    ctx.self_consistency = json_load(ev / "self_consistency.json", {}) or {}
    ctx.robustness = json_load(ev / "robustness.json", {}) or {}
    for store, kind in (
        (ctx.confidence, "conf"),
        (ctx.self_consistency, "sc"),
        (ctx.robustness, "rob"),
    ):
        for system, payload in (store or {}).items():
            if not isinstance(payload, dict):
                continue
            for set_name, recs in payload.items():
                rows = ctx.scores.get((set_name, system))
                if not rows or not isinstance(recs, dict):
                    continue
                for idx_str, rec in recs.items():
                    try:
                        idx = int(idx_str)
                    except (TypeError, ValueError):
                        continue
                    if not (0 <= idx < len(rows)) or not isinstance(rec, dict):
                        continue
                    row = rows[idx]
                    if kind == "conf":
                        row.conf = rec.get("confidence")
                        row.conf_extra = {k: v for k, v in rec.items() if k != "confidence"}
                    elif kind == "sc":
                        row.sc_pass = rec.get("pass_at_k")
                        row.sc_majority = rec.get("majority_correct")
                        row.sc_agree = rec.get("agreement")
                        row.sc_latency = rec.get("seconds")
                    else:
                        row.conf_extra["robust"] = rec


def pairwise_outcome_matrix(ctx: Ctx, set_name: str, sys_a: str, sys_b: str) -> dict:
    """How does system B's answer relate to system A's answer, item by item?"""
    pairs = _align(ctx.scores[(set_name, sys_a)], ctx.scores[(set_name, sys_b)])
    return confusion_matrix([p[0].outcome for p in pairs], [p[1].outcome for p in pairs], OUTCOME_LABELS)


def fixed_broken_matrix(ctx: Ctx, set_name: str, sys_a: str, sys_b: str) -> dict:
    """Did B fix what A broke? 2x2 over execution correctness."""
    pairs = _align(ctx.scores[(set_name, sys_a)], ctx.scores[(set_name, sys_b)])
    y_true = ["correct" if p[0].ex else "wrong" for p in pairs]
    y_pred = ["correct" if p[1].ex else "wrong" for p in pairs]
    both = sum(1 for t, p in zip(y_true, y_pred) if t == "correct" and p == "correct")
    fixed = sum(1 for t, p in zip(y_true, y_pred) if t == "wrong" and p == "correct")
    broke = sum(1 for t, p in zip(y_true, y_pred) if t == "correct" and p == "wrong")
    neither = sum(1 for t, p in zip(y_true, y_pred) if t == "wrong" and p == "wrong")
    return {
        "sys_a": sys_a,
        "sys_b": sys_b,
        "n": len(pairs),
        "both_correct": both,
        "b_fixed": fixed,
        "b_broke": broke,
        "neither": neither,
        "net_gain": fixed - broke,
        "matrix": [[both, broke], [fixed, neither]],
    }


# --------------------------------------------------------------------------
# 8. Figures
# --------------------------------------------------------------------------
METRIC_KEYS = [
    ("valid_rate", "valid SQL"),
    ("exact_match", "exact match"),
    ("execution_accuracy", "execution"),
    ("token_f1", "token F1"),
    ("token_f1_no_literals", "token F1 (no literals)"),
    ("edit_similarity", "edit similarity"),
    ("clause_jaccard", "clause Jaccard"),
    ("component_f1", "clause F1"),
    ("schema_link_f1", "schema-linking F1"),
]


def fig_accuracy_comparison(ctx: Ctx) -> None:
    """fig01 - the headline numbers, base 0-shot / base 3-shot / fine-tuned."""
    sets = ctx.sets_in()
    f = Fig(
        ctx,
        "fig01_accuracy_comparison",
        "Headline accuracy: valid SQL, exact match, execution accuracy",
        "Per evaluation set. 'valid SQL' = the query parses and runs, 'exact match' = textually "
        "identical to the gold query, 'execution' = returned exactly the gold result set. Error "
        "bars are 95% Wilson score intervals; execution accuracy only counts items whose gold "
        "query is itself executable and non-empty.",
        ncols=max(1, len(sets)),
        figsize=(6.4 * max(1, len(sets)), 4.6),
    )
    panels = [("valid_rate", "valid SQL"), ("exact_match", "exact match"), ("execution_accuracy", "execution")]
    for j, set_name in enumerate(sets):
        ax = f.ax(0, j)
        systems = ctx.systems_in(set_name)
        width = 0.8 / max(1, len(systems))
        for k, system in enumerate(systems):
            m = ctx.metrics[set_name][system]
            xs = [i + (k - (len(systems) - 1) / 2) * width for i in range(len(panels))]
            vals = [m[key] for key, _ in panels]
            lo = [v - m[f"{key}_ci"][0] for v, (key, _) in zip(vals, panels)]
            hi = [m[f"{key}_ci"][1] - v for v, (key, _) in zip(vals, panels)]
            bars = ax.bar(xs, vals, width * 0.9, yerr=[lo, hi], capsize=3, ecolor="#444444",
                          color=PALETTE[system], label=SYSTEM_LABEL[system])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.035, f"{100 * v:.1f}",
                        ha="center", fontsize=8)
        ax.set_xticks(range(len(panels)))
        ax.set_xticklabels([t for _, t in panels])
        ax.set_ylim(0, 1.14)
        ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
        if j == 0:
            ax.set_ylabel("percent of items")
            ax.legend(loc="upper center", fontsize=8, ncols=len(systems))
        n = ctx.metrics[set_name][systems[0]]["n"]
        ax.set_title(f"{SET_LABEL.get(set_name, set_name)} (n={n}, scorable={ctx.metrics[set_name][systems[0]]['n_scorable']})")
    f.save()


def fig_metric_radar(ctx: Ctx) -> None:
    """fig02 - one polygon per system over nine pooled quality dimensions."""
    systems = ctx.all_systems
    if len(systems) < 2:
        return
    f = Fig(
        ctx,
        "fig02_metric_radar",
        "Quality profile across all evaluation items",
        "Every axis is a 0-100% score pooled over both evaluation sets, so a larger polygon is "
        "better. Exact match and execution accuracy are the strict measures; token F1, edit "
        "similarity, clause F1 and schema-linking F1 give partial credit.",
        figsize=(6.6, 6.0),
        polar=True,
    )
    keys = METRIC_KEYS
    angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    angles += angles[:1]
    for system in systems:
        m = ctx.pooled[system]
        vals = [m.get(key, 0.0) for key, _ in keys]
        vals += vals[:1]
        ax = f.ax()
        ax.plot(angles, vals, color=PALETTE[system], linewidth=2, label=SYSTEM_LABEL[system])
        ax.fill(angles, vals, color=PALETTE[system], alpha=0.15)
    ax = f.ax()
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([t for _, t in keys], fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=8)
    f.save()


def fig_complexity_breakdown(ctx: Ctx) -> None:
    """fig03 - execution accuracy by SQL task family."""
    sets = ctx.sets_in()
    f = Fig(
        ctx,
        "fig03_complexity_breakdown",
        "Execution accuracy by query complexity",
        "Items are bucketed by the constructs their gold query needs (filters, joins, "
        "aggregations, sub-queries, set operations...). Buckets with fewer than 5 items are "
        "hidden to keep the bars meaningful.",
        ncols=max(1, len(sets)),
        figsize=(7.6 * max(1, len(sets)), 5.0),
    )
    for j, set_name in enumerate(sets):
        ax = f.ax(0, j)
        systems = ctx.systems_in(set_name)
        buckets: dict = defaultdict(lambda: defaultdict(list))
        for system in systems:
            for row in ctx.complexity_rows[(set_name, system)]:
                if row["exec"] != "":
                    buckets[row["complexity"]][system].append(row["exec"])
        order = sorted(buckets, key=lambda k: -len(buckets[k][systems[0]]))
        keep = [k for k in order if len(buckets[k][systems[0]]) >= 5]
        order = keep or order
        width = 0.8 / max(1, len(systems))
        for k, system in enumerate(systems):
            vals = [mean(buckets[c][system]) if buckets[c][system] else 0.0 for c in order]
            xs = [i + (k - (len(systems) - 1) / 2) * width for i in range(len(order))]
            bars = ax.bar(xs, vals, width * 0.9, color=PALETTE[system], label=SYSTEM_LABEL[system])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{100 * v:.0f}%",
                        ha="center", fontsize=7)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{c} (n={len(buckets[c][systems[0]])})" for c in order], rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 1.14)
        ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
        ax.set_title(SET_LABEL.get(set_name, set_name))
        if j == 0:
            ax.set_ylabel("execution accuracy")
            ax.legend(fontsize=8)
    f.save()


# @@FIG04@@

def fig_outcome_distribution(ctx: Ctx) -> None:
    """fig04 - from 'runs but wrong' to 'identical text', per system."""
    sets = ctx.sets_in()
    f = Fig(
        ctx,
        "fig04_outcome_distribution",
        "What the answers actually are",
        "Every generated query is classified as: correct and textually identical to the gold query, "
        "correct but written differently (execution accuracy - exact match), executable but "
        "returning the wrong rows, or not executable at all.",
        ncols=max(1, len(sets)),
        figsize=(6.6 * max(1, len(sets)), 4.8),
    )
    for j, set_name in enumerate(sets):
        ax = f.ax(0, j)
        systems = ctx.systems_in(set_name)
        ypos = np.arange(len(systems))
        left = np.zeros(len(systems))
        for key in OUTCOME_LABELS:
            vals = np.array([
                ctx.metrics[set_name][s]["outcomes"][key] / max(1, ctx.metrics[set_name][s]["n"])
                for s in systems
            ])
            ax.barh(ypos, vals, left=left, color=OUTCOME_COLOR[key], label=OUTCOME_TITLE[key], height=0.6)
            for y, (v, l) in enumerate(zip(vals, left)):
                if v > 0.05:
                    ax.text(l + v / 2, y, f"{100 * v:.0f}%", ha="center", va="center", fontsize=8, color="white")
            left += vals
        ax.set_yticks(ypos)
        ax.set_yticklabels([SYSTEM_LABEL[s] for s in systems], fontsize=9)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
        ax.set_title(SET_LABEL.get(set_name, set_name))
        ax.grid(axis="y", visible=False)
        if j == 0:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=2, fontsize=8)
    f.save()


def fig_outcome_confusion(ctx: Ctx) -> None:
    """fig05 - baseline versus fine-tuned: what got fixed, what got broken."""
    sets = ctx.sets_in()
    base = "base_zeroshot" if "base_zeroshot" in ctx.all_systems else (ctx.all_systems[0] if ctx.all_systems else None)
    if not base or "finetuned" not in ctx.all_systems:
        return
    f = Fig(
        ctx,
        "fig05_outcome_confusion",
        f"Outcome transitions: {SYSTEM_LABEL[base]} (rows) vs {SYSTEM_LABEL['finetuned']} (columns)",
        "Each cell counts items by (baseline outcome, fine-tuned outcome). The interesting cells "
        "are the off-diagonal ones: 'invalid -> correct' are the fixes, 'correct -> invalid' are "
        "the regressions caused by fine-tuning.",
        ncols=max(1, len(sets)),
        figsize=(7.6 * max(1, len(sets)), 6.4),
    )
    lines = []
    for j, set_name in enumerate(sets):
        ax = f.ax(0, j)
        fb = fixed_broken_matrix(ctx, set_name, base, "finetuned")
        mat = pairwise_outcome_matrix(ctx, set_name, base, "finetuned")["matrix"]
        heatmap(
            ax, mat,
            [OUTCOME_TITLE[k].replace(", ", " -\n") for k in OUTCOME_LABELS],
            [OUTCOME_TITLE[k].replace(", ", " -\n") for k in OUTCOME_LABELS],
            title=f"{SET_LABEL.get(set_name, set_name)}\nfixed: {fb['b_fixed']}   regressed: {fb['b_broke']}",
            xlabel="fine-tuned outcome", ylabel="baseline outcome", cmap="viridis_r",
        )
        lines.append(f"{set_name}: fixed={fb['b_fixed']} regressed={fb['b_broke']} both_correct={fb['both_correct']} neither={fb['neither']}")
        ctx.metrics.setdefault("fixed_broken", {})[set_name] = fb
    ctx.notes.append("fig05 " + "; ".join(lines))
    f.save()


def fig_signature_confusion(ctx: Ctx) -> None:
    """fig06 - which structural family of query the model produces for each gold family."""
    sets = ctx.sets_in()
    systems = [s for s in ("finetuned", "base_fewshot", "base_zeroshot") if s in ctx.all_systems]
    if not sets or not systems:
        return
    system = systems[0]
    f = Fig(
        ctx,
        "fig06_signature_confusion",
        f"SQL family confusion matrix - {SYSTEM_LABEL[system]}",
        "Rows are the structural signature of the gold query (which clauses the task needs), "
        "columns the signature of the generated query. The diagonal means 'used exactly the "
        "constructs the task asked for'; off-diagonal mass shows constructs added or dropped.",
        ncols=max(1, len(sets)),
        figsize=(8.4 * max(1, len(sets)), 7.0),
    )
    for j, set_name in enumerate(sets):
        if system not in ctx.systems_in(set_name):
            continue
        ax = f.ax(0, j)
        mat = ctx.metrics[set_name][system]["signature_matrix"]
        heatmap(
            ax, mat["matrix"], mat["labels"], mat["labels"],
            title=f"{SET_LABEL.get(set_name, set_name)}  (family accuracy {100 * mat['accuracy']:.1f}%, macro F1 {100 * mat['macro_f1']:.1f}%)",
            xlabel="predicted signature", ylabel="gold signature", cmap="Blues",
        )
    f.save()


def fig_clause_heatmap(ctx: Ctx) -> None:
    """fig07 - per-construct F1 for every system: where fine-tuning actually helped."""
    sets = ctx.sets_in()
    f = Fig(
        ctx,
        "fig07_clause_heatmap",
        "Per-clause F1 (construct flagged in the prediction when the gold query needs it)",
        "One row per SQL construct, one column per system. F1 penalises both 'the construct is "
        "missing' and 'the construct was invented', the raw confusion counts are in "
        "tables/clause_metrics.md.",
        ncols=max(1, len(sets)),
        figsize=(5.6 * max(1, len(sets)), 6.2),
    )
    for j, set_name in enumerate(sets):
        ax = f.ax(0, j)
        systems = ctx.systems_in(set_name)
        present = [
            (key, label) for key, label in CLAUSES
            if any(key in ctx.metrics[set_name][s]["per_clause"] for s in systems)
        ]
        mat = [
            [ctx.metrics[set_name][s]["per_clause"].get(key, {}).get("f1", 0.0) for s in systems]
            for key, _ in present
        ]
        heatmap(
            ax, mat, [label for _, label in present], [SYSTEM_LABEL[s] for s in systems],
            title=SET_LABEL.get(set_name, set_name), cmap="YlGn", fmt="{:.2f}", vmin=0.0, vmax=1.0,
        )
    f.save()


# @@FIG08@@

def fig_verdict_matrix(ctx: Ctx) -> None:
    """fig08 - raw 2x2 confusion matrices between the systems' correctness verdicts."""
    sets = ctx.sets_in()
    systems = ctx.all_systems
    pairs = [(a, b) for i, a in enumerate(systems) for b in systems[i + 1:]]
    if not pairs:
        return
    f = Fig(
        ctx,
        "fig08_confusion_between_systems",
        "2x2 agreement matrices between the systems (execution correctness)",
        "Rows = system A verdict, columns = system B verdict on the very same items. How much the "
        "models agree is turned into kappa / alpha / AC1 / PABAK in fig09 and "
        "tables/agreement_metrics.md.",
        nrows=len(sets), ncols=max(1, len(pairs)),
        figsize=(4.2 * max(1, len(pairs)), 4.3 * len(sets)),
    )
    for i, set_name in enumerate(sets):
        for j, (a, b) in enumerate(pairs):
            ax = f.ax(i, j)
            if a not in ctx.systems_in(set_name) or b not in ctx.systems_in(set_name):
                ax.axis("off")
                continue
            cm = pairwise_confusion(ctx, set_name, a, b, "correct")
            mat = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
            res = agreement_bundle(ctx, set_name, a, b, "correct")
            ti = "" if i == 0 else f"{SYSTEM_LABEL[a]} vs {SYSTEM_LABEL[b]} ({SET_LABEL.get(set_name, set_name)})"
            tile_text(ax, mat, ["B wrong", "B right"], ["A wrong", "A right"],
                      f"kappa={res['unweighted_kappa']:.2f}\naccuracy={100 * cm['accuracy']:.0f}%  n={cm['n']}", ti)
    f.save()


def fig_kappa_matrix(ctx: Ctx) -> None:
    """fig09 - every chance-corrected agreement coefficient, pair by pair."""
    sets = ctx.sets_in()
    systems = ctx.all_systems
    pairs = [(a, b) for i, a in enumerate(systems) for b in systems[i + 1:]]
    if not pairs:
        return
    f = Fig(
        ctx,
        "fig09_kappa_matrix",
        "Agreement between systems: Cohen's kappa (and friends)",
        "Kappa interpretation (Landis & Koch): <0 worse than chance, 0-0.20 slight, 0.21-0.40 "
        "fair, 0.41-0.60 moderate, 0.61-0.80 substantial, 0.81-1.00 almost perfect. Weighted "
        "kappa is computed over the ordered outcome labels; PABAK and Gwet's AC1 compensate for "
        "the class imbalance that makes plain kappa look low even at high accuracy.",
        ncols=max(1, len(sets)),
        figsize=(6.8 * max(1, len(sets)), 3.6 + 0.55 * len(pairs)),
    )
    lines = []
    for j, set_name in enumerate(sets):
        ax = f.ax(0, j)
        rows, labels = [], []
        for a, b in pairs:
            if a not in ctx.systems_in(set_name) or b not in ctx.systems_in(set_name):
                continue
            r = agreement_bundle(ctx, set_name, a, b, "correct")
            rows.append([r["unweighted_kappa"], r["linear_kappa"], r["quadratic_kappa"],
                         r["gwet_ac1"], r["pabak"], r["krippendorff_alpha"]])
            labels.append(f"{SYSTEM_LABEL[a]} vs\n{SYSTEM_LABEL[b]}")
            lines.append(
                f"{set_name} {a} vs {b}: kappa={r['unweighted_kappa']:.3f} "
                f"acc={100 * r['percent_agreement']:.1f}% AC1={r['gwet_ac1']:.3f} "
                f"PABAK={r['pabak']:.3f} alpha={r['krippendorff_alpha']:.3f}"
            )
        if not rows:
            ax.axis("off")
            continue
        heatmap(
            ax, rows, labels,
            ["Cohen's\nkappa", "linear\nkappa", "quadratic\nkappa", "Gwet\nAC1", "PABAK", "Krippendorff\nalpha"],
            title=SET_LABEL.get(set_name, set_name), cmap="RdYlGn", fmt="{:.2f}", vmin=0.0, vmax=1.0,
        )
    ctx.notes.append("fig09 " + "; ".join(lines))
    f.save()

def fig_kappa_gauge(ctx: Ctx) -> None:
    """fig10 - the Landis-Koch interpretation of the pairwise kappa values."""
    systems = ctx.all_systems
    pairs = [(a, b) for i, a in enumerate(systems) for b in systems[i + 1:]]
    rows = []
    for set_name in ctx.sets_in():
        for a, b in pairs:
            if a not in ctx.systems_in(set_name) or b not in ctx.systems_in(set_name):
                continue
            r = agreement_bundle(ctx, set_name, a, b, "correct")
            rows.append((f"{SET_LABEL.get(set_name, set_name)} - {SYSTEM_LABEL[a]} vs {SYSTEM_LABEL[b]}", r))
    if not rows:
        return
    f = Fig(
        ctx,
        "fig10_kappa_gauge",
        "Chance-corrected agreement on the Landis & Koch scale",
        "One bar per system pair: plain Cohen's kappa (filled) with the linear- and "
        "quadratic-weighted values marked on top. The dashed lines are the conventional "
        "interpretation thresholds.",
        figsize=(7.8, 0.42 * len(rows) + 3.2),
    )
    ax = f.ax()
    ypos = np.arange(len(rows))
    strength = np.array([r["unweighted_kappa"] for _, r in rows])
    ax.barh(ypos, strength, color=[kappa_color(v) for v in strength], height=0.62, label="unweighted kappa")
    ax.scatter([r["linear_kappa"] for _, r in rows], ypos, marker="|", s=260, color="#222222", zorder=4,
               label="linear-weighted")
    ax.scatter([r["quadratic_kappa"] for _, r in rows], ypos, marker="|", s=120, color="#777777", zorder=4,
               label="quadratic-weighted")
    bands = [(0.0, 0.20, "slight"), (0.20, 0.40, "fair"), (0.40, 0.60, "moderate"),
             (0.60, 0.80, "substantial"), (0.80, 1.0, "almost perfect")]
    for lo, hi, name in bands:
        ax.axvspan(lo, hi, color="#000000", alpha=0.03, zorder=0)
        ax.text((lo + hi) / 2, len(rows) - 0.4, name, ha="center", fontsize=7, color="#555555")
    for lo, _, _ in bands[1:]:
        ax.axvline(lo, color="#888888", linestyle="--", linewidth=0.8, zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([lab for lab, _ in rows], fontsize=8)
    ax.set_xlim(min(0.0, float(min(strength)) - 0.05), 1.0)
    ax.set_xlabel("agreement coefficient")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8, loc="lower left")
    f.save()


def fig_training_curves(ctx: Ctx) -> None:
    """fig11 - loss / learning-rate / grad-norm history of the finished run."""
    log = ctx.paths.train_log
    if not log:
        return
    f = Fig(
        ctx,
        "fig11_training_curves",
        "Training history of the exported adapter",
        "Read straight out of train_log.json inside the run directory, no re-training. The noisy "
        "line is the per-step training loss, the dark line its running mean, the dashed line the "
        "mean of the last 20% of the steps.",
        ncols=3,
        figsize=(15.0, 4.0),
    )
    steps = [r.get("step") for r in log]
    names = list(log[-1].keys())
    panels = [k for k in ("loss", "lr", "grad_norm", "epoch") if k in names][:3] or names[:3]
    for j, key in enumerate(panels):
        ax = f.ax(0, j)
        ys = [r.get(key) for r in log]
        ax.plot(steps, ys, color=PALETTE["finetuned"], linewidth=1.0, alpha=0.55, label=key)
        if len(ys) >= 5:
            k = max(2, len(ys) // 25)
            ax.plot(steps, [mean(ys[max(0, i - k):i + 1]) for i in range(len(ys))],
                    color="#333333", linewidth=1.8, label="running mean")
        if key == "loss" and len(ys) >= 10:
            tail = mean(ys[int(0.8 * len(ys)):])
            ax.axhline(tail, color=WARN_C, linestyle="--", linewidth=1.2, label=f"final mean {tail:.4f}")
        ax.set_xlabel("step")
        ax.set_title(key)
        ax.legend(fontsize=8)
        if j == 0:
            ax.set_ylabel(key)
    f.save()

def fig_throughput(ctx: Ctx) -> None:
    """fig12 - sequence length evidence and generation cost, straight from the run."""
    log = ctx.paths.train_log
    lengths = {}
    for row in log:
        if isinstance(row.get("lengths"), dict):
            lengths = row["lengths"]
            break
    f = Fig(
        ctx,
        "fig12_throughput_and_lengths",
        "Training sequence lengths and generation cost",
        "Left: the sequence-length histogram recorded by the training loop (with the longest "
        "bucket it kept). Right: wall-clock cost of the evaluation generations, from the run "
        "manifest. Both panels come from the finished run, nothing is re-trained.",
        ncols=2,
        figsize=(12.5, 4.2),
    )
    ax = f.ax(0, 0)
    if lengths:
        keys = sorted((int(k), v) for k, v in lengths.items() if k != "samples" and int(k) > 0)
        if keys:
            xs = [k for k, _ in keys]
            ys = [v for _, v in keys]
            ax.bar(range(len(xs)), ys, color=PALETTE["finetuned"], width=0.8)
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels([str(k) for k in xs], fontsize=8)
            ax.set_xlabel("sequence length bucket (tokens)")
            ax.set_ylabel("sequences")
            ax.set_title(f"training sequences (n={lengths.get('samples', sum(ys))})")
        else:
            ax.axis("off")
    else:
        stats = ctx.paths.manifest.get("length_stats") or {}
        if stats:
            ax.barh(["mean", "p95", "max"], [stats.get("mean", 0), stats.get("p95", 0), stats.get("max", 0)],
                    color=PALETTE["finetuned"])
            ax.axvline(ctx.paths.config.get("max_len", 0), color="#333333", linestyle=":", label="max_len")
            ax.set_xlabel("tokens")
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "no sequence-length information\nwas recorded in this run",
                    ha="center", va="center", transform=ax.transAxes, color="#666666")
            ax.axis("off")

    ax2 = f.ax(0, 1)
    systems = [s for s in SYSTEMS if s in ctx.pooled]
    gen = [ctx.paths.manifest.get("systems", {}).get(s, {}) for s in systems]
    secs = [g.get("seconds_per_sample") or g.get("seconds", 0.0) for g in gen]
    tps = [g.get("tokens_per_second") or g.get("tok_per_s", 0.0) for g in gen]
    if any(secs):
        xs = np.arange(len(systems))
        ax2.bar(xs, secs, color=[PALETTE[s] for s in systems], width=0.55)
        for x, v in zip(xs, secs):
            ax2.text(x, v, f"{v:.2f}s", ha="center", va="bottom", fontsize=8)
        ax2.set_xticks(xs)
        ax2.set_xticklabels([SYSTEM_LABEL[s].replace(" - ", "\n") for s in systems], fontsize=8)
        ax2.set_ylabel("seconds per answer")
        for x, v in zip(xs, tps):
            if v:
                ax2.text(x, max(secs) * 0.88, f"{v:.0f} tok/s", ha="center", fontsize=8, color="#333333")
        ax2.set_title("generation speed")
    else:
        ax2.text(0.5, 0.5, "no timing information was recorded\nin this run",
                 ha="center", va="center", transform=ax2.transAxes, color="#666666")
        ax2.axis("off")
    f.save()


def fig_length_analysis(ctx: Ctx) -> None:
    """fig13 - how answer quality depends on question / schema / gold-query length."""
    sets = ctx.sets_in()
    f = Fig(
        ctx,
        "fig13_length_analysis",
        "Accuracy versus input length",
        "Items are split into quartiles of question length, schema size and gold-query length and "
        "the bars are execution accuracy per quartile. This shows where the model starts to "
        "struggle: long questions, wide schemas, long target queries.",
        nrows=max(1, len(sets)), ncols=3,
        figsize=(15.0, 3.9 * max(1, len(sets))),
    )
    systems = [s for s in ("finetuned", "base_fewshot", "base_zeroshot") if s in ctx.all_systems]
    for i, set_name in enumerate(sets):
        items = ctx.items[set_name]
        specs = [
            ("question length (words)", lambda it: len(str(it.get("question", "")).split())),
            ("schema size (characters)", lambda it: len(str(it.get("schema", "")))),
            ("gold query length (words)", lambda it: len(str(it.get("gold", "")).split())),
        ]
        for j, (label, fn) in enumerate(specs):
            ax = f.ax(i, j)
            values = [fn(it) for it in items]
            order = sorted(range(len(items)), key=lambda k: values[k])
            quart = [0] * len(items)
            for rank, idx in enumerate(order):
                quart[idx] = min(3, int(4 * rank / max(1, len(items))))
            for s in systems:
                scores = ctx.scores.get((set_name, s))
                if not scores:
                    continue
                vals = []
                for q in range(4):
                    sub = [sc.ex for sc in scores if sc.ex is not None and quart[sc.idx] == q]
                    vals.append(mean([bool(v) for v in sub]) if sub else 0.0)
                xs = np.arange(4) + (systems.index(s) - (len(systems) - 1) / 2) * (0.8 / len(systems))
                ax.bar(xs, vals, 0.8 / len(systems) * 0.9, color=PALETTE[s], label=SYSTEM_LABEL[s])
            edges = sorted(values)[:: max(1, len(items) // 4)][:4]
            ax.set_xticks(range(4))
            ax.set_xticklabels([f"Q{q + 1}\nfrom {edges[q] if q < len(edges) else 0:.0f}" for q in range(4)], fontsize=7)
            ax.set_ylim(0, 1.12)
            ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
            ax.set_title(f"{SET_LABEL.get(set_name, set_name)} - {label}")
            if j == 0:
                ax.set_ylabel("execution accuracy")
                ax.legend(fontsize=7)
    f.save()


def fig_calibration(ctx: Ctx) -> None:
    """fig14 - is the model's confidence trustworthy? (needs --stage confidence first)"""
    conf = ctx.confidence
    if not conf:
        return
    sets = [s for s in ctx.sets_in() if s in conf]
    if not sets:
        return
    f = Fig(
        ctx,
        "fig14_calibration",
        "Confidence calibration of the fine-tuned model",
        "Left: reliability diagram - the mean token log-probability of each answer turned into a "
        "probability, bucketed, against how often those answers were actually correct. The diagonal "
        "is perfect calibration and the marker size is the bucket population. Right: risk-coverage "
        "- if only the n% most confident questions are answered, how accurate are those answers? "
        "ECE / MCE / Brier / NLL are in tables/confidence_metrics.md.",
        ncols=2 * len(sets), figsize=(6.4 * 2 * len(sets), 4.4),
    )
    for i, set_name in enumerate(sets):
        entry = conf[set_name]
        systems = [s for s in SYSTEMS if s in entry]
        ax = f.ax(0, 2 * i)
        ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1, label="perfect calibration")
        for s in systems:
            bins = entry[s].get("calibration", {}).get("bins", [])
            if not bins:
                continue
            xs = [b["confidence"] for b in bins]
            ys = [b["accuracy"] for b in bins]
            ns = [b["n"] for b in bins]
            ax.plot(xs, ys, marker="o", color=PALETTE[s],
                    label=f"{SYSTEM_LABEL[s]} (ECE {entry[s]['calibration']['ece']:.3f})")
            ax.scatter(xs, ys, s=[20 + 1.2 * n for n in ns], color=PALETTE[s], alpha=0.35)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("predicted confidence")
        ax.set_ylabel("observed accuracy")
        ax.set_title(f"Reliability - {SET_LABEL.get(set_name, set_name)}")
        ax.legend(fontsize=7, loc="upper left")

        ax2 = f.ax(0, 2 * i + 1)
        for s in systems:
            rc = entry[s].get("risk_coverage", {})
            if rc:
                ax2.plot(rc["coverage"], rc["accuracy"], color=PALETTE[s],
                         label=f"{SYSTEM_LABEL[s]} (AURC {rc['aurc']:.3f})")
        ax2.set_xlabel("coverage (fraction of most confident answers kept)")
        ax2.set_ylabel("accuracy on the kept answers")
        ax2.set_ylim(0, 1.02)
        ax2.set_title(f"Selective accuracy - {SET_LABEL.get(set_name, set_name)}")
        ax2.legend(fontsize=7, loc="lower left")
    f.save()

def fig_confidence_summary(ctx: Ctx) -> None:
    """fig15 - confidence distributions, accuracy by decile and the aggregate scores."""
    conf = ctx.confidence
    if not conf:
        return
    sets = [s for s in ctx.sets_in() if s in conf]
    systems = [s for s in SYSTEMS if any(s in conf[st] for st in sets)]
    if not sets or not systems:
        return
    f = Fig(
        ctx,
        "fig15_confidence_summary",
        "Confidence distributions, accuracy by confidence and summary scores",
        "Left: distribution of the confidence score for correct (solid) and wrong (hatched) "
        "answers - good separation means the score can be trusted to reject bad answers. Middle: "
        "accuracy inside each confidence decile. Right: aggregate scores for 'this answer is "
        "correct' (ROC-AUC, PR-AUC) and calibration quality (1 - ECE, higher is better).",
        ncols=3, figsize=(15.5, 4.4),
    )
    ax = f.ax(0, 0)
    for s in systems:
        for st in sets:
            rows = conf[st].get(s, {}).get("rows") or []
            if not rows:
                continue
            ax.hist([r["conf"] for r in rows if r["correct"]], bins=15, range=(0, 1),
                    color=PALETTE[s], alpha=0.6, label=f"{SYSTEM_LABEL[s]} correct")
            ax.hist([r["conf"] for r in rows if not r["correct"]], bins=15, range=(0, 1),
                    color=PALETTE[s], alpha=0.3, histtype="stepfilled", hatch="//",
                    edgecolor="white", label=f"{SYSTEM_LABEL[s]} wrong")
    ax.set_xlabel("confidence")
    ax.set_ylabel("items")
    ax.set_title("correct vs wrong answers")
    ax.legend(fontsize=7)

    ax2 = f.ax(0, 1)
    width = 0.8 / max(1, len(systems))
    for k, s in enumerate(systems):
        dec = None
        for st in sets:
            dec = conf[st].get(s, {}).get("by_decile") or dec
        if not dec:
            continue
        xs = [d["decile"] + (k - (len(systems) - 1) / 2) * width for d in dec]
        ax2.bar(xs, [d["accuracy"] for d in dec], width * 0.9, color=PALETTE[s], label=SYSTEM_LABEL[s])
    ax2.set_xticks(range(1, 11))
    ax2.set_xlabel("confidence decile (1 = least confident)")
    ax2.set_ylabel("execution accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("accuracy rises with confidence")
    ax2.legend(fontsize=7)

    ax3 = f.ax(0, 2)
    rows = []
    for s in systems:
        merged = merge_confidence([conf[st][s] for st in sets if s in conf[st]])
        if merged:
            rows.append((s, merged["calibration"], merged["auc"]))
    if rows:
        xs = np.arange(len(rows))
        w = 0.27
        ax3.bar(xs - w, [r[1].get("auc_roc", 0) for r in rows], w, color="#4c72b0", label="ROC-AUC")
        ax3.bar(xs, [r[1].get("auc_pr", 0) for r in rows], w, color=ACCENT, label="PR-AUC")
        ax3.bar(xs + w, [1 - r[1].get("ece", 0) for r in rows], w, color="#dd8452", label="1 - ECE")
        ax3.set_xticks(xs)
        ax3.set_xticklabels([SYSTEM_LABEL[r[0]].replace(" - ", "\n") for r in rows], fontsize=8)
        ax3.set_ylim(0, 1.1)
        ax3.set_title("summary scores (higher = better)")
        ax3.legend(fontsize=8)
        for x, r in zip(xs, rows):
            ax3.text(x, 1.04, f"n={r[1].get('n', 0)}", ha="center", fontsize=7, color="#555555")
    f.save()

# --------------------------------------------------------------------------
# 9. Tables, manifest, report
# --------------------------------------------------------------------------
def fig_self_consistency(ctx: Ctx) -> None:
    """fig16 - sampling k answers: how stable is the model, and does majority voting help?"""
    sc = ctx.self_consistency
    if not sc:
        return
    sets = [s for s in ctx.sets_in() if s in sc]
    if not sets:
        return
    f = Fig(
        ctx,
        "fig16_self_consistency",
        "Self-consistency: pass@k and majority voting",
        "Each question was sampled k times at a non-zero temperature. Left: share of questions "
        "with at least one correct sample (pass@k) and with all k samples correct. Middle: "
        "per-question sample agreement against the probability that the majority-voted answer is "
        "correct. Right: greedy accuracy versus majority-vote accuracy plus the mean agreement.",
        ncols=3, figsize=(15.5, 4.4),
    )
    rows = []
    for st in sets:
        for s in SYSTEMS:
            if s in sc[st]:
                rows.append((st, s, sc[st][s]))
    if not rows:
        return
    ax = f.ax(0, 0)
    xs = np.arange(len(rows))
    w = 0.38
    ax.bar(xs - w / 2, [r[2].get("pass_at_k", 0) for r in rows], w, color=ACCENT, label="pass@k (any correct)")
    ax.bar(xs + w / 2, [r[2].get("all_correct", 0) for r in rows], w, color="#4c72b0", label="all k correct")
    for x, r in zip(xs, rows):
        ax.text(x, 1.02, f"k={r[2].get('k', 0)}", ha="center", fontsize=7, color="#555555")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{SYSTEM_LABEL[r[1]]}\n{SET_LABEL.get(r[0], r[0])}".replace(" - ", "\n") for r in rows], fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("fraction of questions")
    ax.set_title("coverage of k samples")
    ax.legend(fontsize=8)

    ax2 = f.ax(0, 1)
    for st, s, entry in rows:
        dec = entry.get("by_agreement")
        if not dec:
            continue
        ax2.plot([d["agreement"] for d in dec], [d["majority_accuracy"] for d in dec],
                 marker="o", color=PALETTE[s], label=f"{SYSTEM_LABEL[s]} - {SET_LABEL.get(st, st)}")
    ax2.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
    ax2.set_xlabel("mean agreement between the k samples")
    ax2.set_ylabel("majority-vote accuracy")
    ax2.set_ylim(0, 1.02)
    ax2.set_title("agreement predicts correctness")
    ax2.legend(fontsize=7)

    ax3 = f.ax(0, 2)
    ax3.bar(xs - w, [r[2].get("greedy_accuracy", 0) for r in rows], w, color="#8c8c8c", label="greedy")
    ax3.bar(xs, [r[2].get("majority_accuracy", 0) for r in rows], w, color=ACCENT, label="majority vote")
    ax3.bar(xs + w, [r[2].get("mean_agreement", 0) for r in rows], w, color="#4c72b0", label="mean agreement")
    ax3.set_xticks(xs)
    ax3.set_xticklabels([f"{SYSTEM_LABEL[r[1]]}\n{SET_LABEL.get(r[0], r[0])}".replace(" - ", "\n") for r in rows], fontsize=7)
    ax3.set_ylim(0, 1.08)
    ax3.set_title("majority voting gain")
    ax3.legend(fontsize=8)
    f.save()


def fig_robustness(ctx: Ctx) -> None:
    """fig17 - accuracy when the schema presentation is perturbed (GPU stage optional)."""
    rob = ctx.robustness
    if not rob:
        return
    systems = [s for s in SYSTEMS if any(s in rob[st] for st in rob)]
    sets = [st for st in rob if any(s in rob[st] for s in systems)]
    if not systems or not sets:
        return
    f = Fig(
        ctx,
        "fig17_schema_robustness",
        "Schema-presentation robustness",
        "The same questions answered again after perturbing only the schema text: reordering the "
        "tables, renaming the tables (the question is untouched), adding irrelevant distractor "
        "tables, upper-casing the schema and padding it with SQL comments. The first bar is the "
        "cached clean accuracy and the labels are the drop versus that reference.",
        ncols=max(1, len(sets)), figsize=(8.4 * max(1, len(sets)), 4.6),
    )
    for j, set_name in enumerate(sets):
        ax = f.ax(0, j)
        order = ["clean"] + [
            p for p in PERTURBATIONS
            if any(p in rob[set_name][s].get("perturbations", {}) for s in systems)
        ]
        width = 0.8 / max(1, len(systems))
        for k, s in enumerate(systems):
            entry = rob[set_name].get(s)
            if not entry:
                continue
            vals = [entry.get("clean_accuracy", 0.0)]
            for p in order[1:]:
                vals.append(entry.get("perturbations", {}).get(p, {}).get("accuracy", 0.0))
            xs = [i + (k - (len(systems) - 1) / 2) * width for i in range(len(order))]
            bars = ax.bar(xs, vals, width * 0.9, color=PALETTE[s], label=SYSTEM_LABEL[s])
            for b, v, p in zip(bars[1:], vals[1:], order[1:]):
                drop = entry.get("perturbations", {}).get(p, {}).get("drop", 0.0)
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{100 * drop:+.0f}",
                        ha="center", fontsize=7, color=WARN_C if drop > 0.03 else "#555555")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([o.replace("_", "\n") for o in order], fontsize=8)
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
        ax.set_title(f"{SET_LABEL.get(set_name, set_name)} - labels: drop vs clean")
        if j == 0:
            ax.set_ylabel("execution accuracy")
            ax.legend(fontsize=8)
    f.save()


def run_figure_builders(ctx: Ctx) -> None:
    """Every figure builder in report order. Builders without data return immediately."""
    builders = [
        fig_accuracy_comparison,
        fig_metric_radar,
        fig_complexity_breakdown,
        fig_outcome_distribution,
        fig_outcome_confusion,
        fig_signature_confusion,
        fig_clause_heatmap,
        fig_verdict_matrix,
        fig_kappa_matrix,
        fig_kappa_gauge,
        fig_training_curves,
        fig_throughput,
        fig_length_analysis,
        fig_calibration,
        fig_confidence_summary,
        fig_self_consistency,
        fig_robustness,
    ]
    for fn in builders:
        try:
            fn(ctx)
        except Exception as e:  # noqa: BLE001
            log(f"  ! {fn.__name__} skipped: {type(e).__name__}: {e}")
    log(f"figures: {len(ctx.figures)} written to {ctx.paths.figures_dir}")


def md_table(headers: Sequence, rows: Sequence[Sequence], align: Sequence | None = None) -> str:
    """Render a GitHub-flavoured markdown table (pandas-free)."""
    align = list(align or [])
    sep = []
    for i in range(len(headers)):
        a = align[i] if i < len(align) else "l"
        sep.append({"c": ":---:", "r": "---:", "l": ":---"}.get(a, "---"))
    out = ["| " + " | ".join(str(h) for h in headers) + " |", "| " + " | ".join(sep) + " |"]
    for row in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(out) + "\n"


def table_to(ctx: Ctx, name: str, headers: Sequence, rows: Sequence[Sequence], caption: str = "",
             align: Sequence | None = None, note: str = "") -> None:
    """Write one table three times: markdown (for the report), CSV (for Excel/Sheets) and LaTeX."""
    ctx.paths.tables_dir.mkdir(parents=True, exist_ok=True)
    md = f"### {caption or name}\n\n" if caption else ""
    if note:
        md += f"_{note}_\n\n"
    md += md_table(headers, rows, align)
    (ctx.paths.tables_dir / f"{name}.md").write_text(md)
    with (ctx.paths.tables_dir / f"{name}.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([str(h) for h in headers])
        for row in rows:
            writer.writerow([("" if v is None else (f"{v:.6f}" if isinstance(v, float) else v)) for v in row])
    tex = [
        "\\begin{table}[t]\\centering",
        f"\\caption{{{caption or name}}}",
        "\\begin{tabular}{" + ("l" + "r" * (len(headers) - 1)) + "}",
        "\\toprule",
        " & ".join(str(h).replace("%", "\\%").replace("_", "\\_") for h in headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        cells = []
        for v in row:
            sv = "" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
            cells.append(sv.replace("%", "\\%").replace("_", "\\_").replace("&", "\\&"))
        tex.append(" & ".join(cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (ctx.paths.tables_dir / f"{name}.tex").write_text("\n".join(tex))
    ctx.tables[name] = {"headers": list(headers), "rows": [list(r) for r in rows], "caption": caption, "note": note}


def ci_cell(m: dict, key: str, nd: int = 1) -> str:
    """'87.3% [83.0, 90.7]' - point estimate with its 95% Wilson interval."""
    if key not in m:
        return "n/a"
    lo, hi = m.get(f"{key}_ci", (None, None))
    if lo is None:
        return pct(m[key], nd)
    return f"{pct(m[key], nd)} [{100 * lo:.{nd}f}, {100 * hi:.{nd}f}]"


def make_overall_table(ctx: Ctx) -> None:
    """tables/overall_metrics - the 'how good is it' table."""
    headers = ["system", "set", "n", "valid SQL", "exact match", "execution", "token F1", "F1 (no literals)",
               "edit similarity", "clause F1", "schema-linking F1"]
    rows = []
    for set_name in ctx.sets_in():
        for system in ctx.systems_in(set_name):
            m = ctx.metrics[set_name][system]
            rows.append([
                SYSTEM_LABEL[system], SET_LABEL.get(set_name, set_name), m["n"],
                ci_cell(m, "valid_rate"), ci_cell(m, "exact_match"), ci_cell(m, "execution_accuracy"),
                f"{m['token_f1']:.4f}", f"{m['token_f1_no_literals']:.4f}", f"{m['edit_similarity']:.4f}",
                f"{m['component_f1']:.4f}", f"{m['schema_link_f1']:.4f}",
            ])
    for system in ctx.all_systems:
        m = ctx.pooled[system]
        rows.append([
            SYSTEM_LABEL[system], "both sets (pooled)", m["n"],
            ci_cell(m, "valid_rate"), ci_cell(m, "exact_match"), ci_cell(m, "execution_accuracy"),
            f"{m['token_f1']:.4f}", f"{m['token_f1_no_literals']:.4f}", f"{m['edit_similarity']:.4f}",
            f"{m['component_f1']:.4f}", f"{m['schema_link_f1']:.4f}",
        ])
    table_to(
        ctx, "overall_metrics", headers, rows,
        caption="Overall metrics per system and evaluation set",
        align=["l", "l", "r"] + ["r"] * (len(headers) - 3),
        note="Percentages are followed by the 95% Wilson score interval. 'execution' is the "
             "execution accuracy, i.e. the generated query returns exactly the gold result set; "
             "only items whose gold query itself runs and returns rows are counted.",
    )


def make_confusion_tables(ctx: Ctx) -> None:
    """tables/per_class_confusion + tables/outcome_matrix - every 2x2 / k x k matrix."""
    rows = []
    for set_name in ctx.sets_in():
        for system in ctx.systems_in(set_name):
            m = ctx.metrics[set_name][system]
            cm = m["outcome_confusion"]
            for label in cm["labels"]:
                pc = cm["per_class"][label]
                rows.append([
                    SYSTEM_LABEL[system], SET_LABEL.get(set_name, set_name), OUTCOME_TITLE.get(label, label),
                    pc["tp"], pc["fp"], pc["fn"], pc["support"],
                    f"{pc['precision']:.3f}", f"{pc['recall']:.3f}", f"{pc['f1']:.3f}",
                ])
            rows.append([
                SYSTEM_LABEL[system], SET_LABEL.get(set_name, set_name), "*overall*",
                "", "", "", cm["n"],
                f"acc {cm['accuracy']:.3f}", f"macro F1 {cm['macro_f1']:.3f}",
                f"bal.acc {cm['balanced_accuracy']:.3f}",
            ])
            rows.append([
                SYSTEM_LABEL[system], SET_LABEL.get(set_name, set_name), "*kappa*",
                "", "", "", m["signature_matrix"]["n"],
                f"unweighted {cm['cohen_kappa']:.3f}",
                f"linear {cm['cohen_kappa_linear']:.3f}",
                f"quadratic {cm['cohen_kappa_quadratic']:.3f}",
            ])
    table_to(
        ctx, "per_class_confusion", ["system", "set", "class", "TP", "FP", "FN", "support",
                                     "precision", "recall", "F1"],
        rows, caption="Per-class confusion statistics of the four outcome classes",
        align=["l", "l", "l"] + ["r"] * 7,
        note="Outcome classes: executed_exact (correct and identical text), executed_match "
             "(correct but differently written), valid_wrong (executes, wrong rows), invalid "
             "(does not execute). Precision/recall are computed one-vs-rest.",
    )

    rows = []
    for set_name in ctx.sets_in():
        for system in ctx.systems_in(set_name):
            cm = ctx.metrics[set_name][system]["outcome_confusion"]
            for i, label in enumerate(cm["labels"]):
                rows.append([SYSTEM_LABEL[system], SET_LABEL.get(set_name, set_name), OUTCOME_TITLE.get(label, label)]
                            + list(cm["matrix"][i]))
    table_to(
        ctx, "outcome_matrix", ["system", "set", "gold/true outcome"] + [OUTCOME_TITLE[k] for k in OUTCOME_LABELS],
        rows, caption="Outcome confusion matrix (rows = reference outcome, columns = predicted outcome)",
        align=["l", "l", "l"] + ["r"] * len(OUTCOME_LABELS),
    )

    rows = []
    for set_name in ctx.sets_in():
        for system in ctx.systems_in(set_name):
            mat = ctx.metrics[set_name][system]["signature_matrix"]
            for i, label in enumerate(mat["labels"]):
                rows.append([SYSTEM_LABEL[system], SET_LABEL.get(set_name, set_name), label] + list(mat["matrix"][i]))
    labels = []
    for set_name in ctx.sets_in():
        for system in ctx.systems_in(set_name):
            labels = ctx.metrics[set_name][system]["signature_matrix"]["labels"]
            break
        break
    table_to(
        ctx, "signature_matrix", ["system", "set", "gold signature"] + labels,
        rows, caption="SQL-family (structural signature) confusion matrix",
        align=["l", "l", "l"] + ["r"] * len(labels),
        note="A signature such as AGG+JOIN+WHERE+GROUP means the query aggregates over a join and "
             "groups the result. The diagonal counts the queries that used exactly the required "
             "constructs; off-diagonal cells show added or dropped constructs.",
    )


def make_agreement_table(ctx: Ctx) -> None:
    """tables/agreement - Cohen's kappa, Fleiss' kappa, Krippendorff's alpha, AC1, PABAK."""
    rows = []
    for set_name in ctx.sets_in():
        panel = ctx.metrics.get("agreement", {}).get(set_name, {})
        for pair, d in panel.get("pairs", {}).items():
            a, b = pair.split("|")
            rows.append([
                SET_LABEL.get(set_name, set_name), SYSTEM_LABEL[a], SYSTEM_LABEL[b], d["n"],
                f"{d['percent_agreement']:.3f}", f"{d['cohen_kappa']:.3f}",
                f"{d['cohen_kappa_linear']:.3f}", f"{d['cohen_kappa_quadratic']:.3f}",
                f"{d['gwet_ac1']:.3f}", f"{d['pabak']:.3f}",
                f"{d['fleiss_kappa']:.3f}", f"{d['krippendorff_alpha']:.3f}",
            ])
        rows.append([
            SET_LABEL.get(set_name, set_name), "*all systems*", "*all systems*", panel.get("n", 0),
            "", "", "", "", "", "",
            f"{panel.get('fleiss_kappa', 0.0):.3f}", f"{panel.get('krippendorff_alpha', 0.0):.3f}",
        ])
    table_to(
        ctx, "agreement", ["set", "system A", "system B", "n items", "raw agreement", "Cohen kappa",
                           "kappa linear", "kappa quadratic", "Gwet AC1", "PABAK", "Fleiss kappa",
                           "Krippendorff alpha"],
        rows, caption="Chance-corrected agreement between the systems (outcome classes)",
        align=["l", "l", "l"] + ["r"] * 9,
        note="These coefficients measure whether the systems classify items the same way *beyond "
             "chance*. Landis & Koch: <0.20 slight, 0.21-0.40 fair, 0.41-0.60 moderate, 0.61-0.80 "
             "substantial, 0.81-1.00 almost perfect. Low kappa with high raw agreement means the "
             "task is easy (the models agree by default); high kappa means they make the same "
             "distinctions.",
    )


def make_verdict_tables(ctx: Ctx) -> None:
    """tables/verdict_matrix + tables/binary_confusions + tables/significance_tests."""
    rows, tested = [], []
    for set_name in ctx.sets_in():
        vm = ctx.metrics.get("verdict", {}).get(set_name, {})
        for pair, d in vm.get("pairs", {}).items():
            a, b = pair.split("|")
            rows.append([
                SET_LABEL.get(set_name, set_name), SYSTEM_LABEL[a], SYSTEM_LABEL[b], d["n"],
                f"{d['a_only_correct']}", f"{d['b_only_correct']}",
                f"{d['exact_p']:.4g}", f"{d['chi2_p']:.4g}",
                f"{d['odds_ratio']:.2f}" if d["odds_ratio"] not in (float("inf"),) else "inf",
                "**yes**" if d["significant_05_adjusted"] else ("yes (uncorrected)" if d["significant_05"] else "no"),
            ])
            tested.append((set_name, a, b))
    table_to(
        ctx, "verdict_matrix", ["set", "system A", "system B", "n paired", "A right / B wrong",
                                "B right / A wrong", "McNemar exact p", "chi2 p",
                                "odds ratio", "significant (Holm, 5%)"],
        rows, caption="Pairwise significance of the execution-accuracy differences (McNemar test)",
        align=["l", "l", "l"] + ["r"] * 7,
        note="Only items where the two systems disagree contribute to McNemar's test. "
             "p is exact (binomial); Holm-Bonferroni multiplies it to correct for the number of "
             "comparisons made inside each evaluation set.",
    )

    rows = []
    for set_name in ctx.sets_in():
        bc = ctx.metrics.get("binary_confusions", {}).get(set_name, {})
        for (a, b), kinds in bc.items():
            for kind, label, _ in KINDS:
                d = kinds.get(kind, {})
                if not d:
                    continue
                rows.append([
                    SET_LABEL.get(set_name, set_name), SYSTEM_LABEL[a], SYSTEM_LABEL[b], label,
                    d["tp"], d["fp"], d["fn"], d["tn"],
                    f"{d['precision']:.3f}", f"{d['recall']:.3f}", f"{d['f1']:.3f}",
                    f"{d['accuracy']:.3f}", f"{d['mcc']:.3f}", f"{d['cohen_kappa']:.3f}",
                ])
    table_to(
        ctx, "binary_confusions", ["set", "reference", "model", "measure", "TP", "FP", "FN", "TN",
                                   "precision", "recall", "F1", "accuracy", "MCC", "Cohen kappa"],
        rows, caption="2x2 confusion matrices treating each system as the reference annotator",
        align=["l", "l", "l", "l"] + ["r"] * 10,
        note="The baseline is treated as the reference and the fine-tuned model as the prediction, "
             "so 'TP' = both correct, 'FP' = only the fine-tuned model correct, 'FN' = only the "
             "baseline correct. This is the usual way to report a model's agreement with a "
             "reference system, not the way to report absolute quality.",
    )

    rows = []
    for set_name in ctx.sets_in():
        st = ctx.metrics.get("significance", {}).get(set_name, {})
        for metric, systems in st.items():
            for pair, d in systems.items():
                a, b = pair.split("|")
                rows.append([
                    SET_LABEL.get(set_name, set_name), METRIC_TITLE.get(metric, metric),
                    SYSTEM_LABEL[a], SYSTEM_LABEL[b],
                    f"{100 * d['delta']:+.2f}", f"[{100 * d['lo']:+.2f}, {100 * d['hi']:+.2f}]",
                    f"{d['p_two_sided']:.4g}", f"{d['p_adjusted']:.4g}",
                    f"{100 * d['prob_better']:.1f}%",
                ])
    table_to(
        ctx, "significance_tests", ["set", "metric", "system A", "system B", "delta (A-B)",
                                    "95% bootstrap CI", "bootstrap p", "Holm-adjusted p",
                                    "P(A better)"],
        rows, caption="Paired bootstrap comparison of every metric between every pair of systems",
        align=["l", "l", "l", "l"] + ["r"] * 5,
        note="2000 paired bootstrap resamples over the evaluation items. The CI is the percentile "
             "interval of the difference; the p-value is the two-sided bootstrap tail probability "
             "and is Holm-corrected across the comparisons within each set.",
    )


def make_clause_table(ctx: Ctx) -> None:
    """tables/clause_metrics - per-construct detection and per-construct accuracy."""
    rows = []
    for set_name in ctx.sets_in():
        for system in ctx.systems_in(set_name):
            per = ctx.metrics[set_name][system]["per_clause"]
            for key, label in CLAUSES:
                d = per.get(key)
                if not d:
                    continue
                rows.append([
                    SET_LABEL.get(set_name, set_name), SYSTEM_LABEL[system], label,
                    d["gold_n"], d["pred_n"], d["tp"], d["fp"], d["fn"],
                    f"{d['precision']:.3f}", f"{d['recall']:.3f}", f"{d['f1']:.3f}",
                    f"{d['presence_accuracy']:.3f}", f"{d['accuracy_when_required']:.3f}",
                ])
    table_to(
        ctx, "clause_metrics", ["set", "system", "clause", "gold n", "pred n", "TP", "FP", "FN",
                                "precision", "recall", "F1", "presence accuracy", "accuracy when required"],
        rows, caption="Component-level metrics per SQL construct",
        align=["l", "l", "l"] + ["r"] * 10,
        note="'presence accuracy' = how often the construct is present/absent exactly as in the "
             "gold query. 'accuracy when required' = execution accuracy restricted to the items "
             "whose gold query uses that construct, i.e. how well the model solves that kind of task.",
    )


def make_training_table(ctx: Ctx) -> None:
    """tables/training_summary - hyper-parameters plus the loss trajectory."""
    cfg = ctx.paths.cfg_view()
    rows = [[k, cfg[k]] for k in sorted(cfg)]
    table_to(ctx, "training_config", ["setting", "value"], rows, caption="Hyper-parameters of the finished run (config.json)")

    log = ctx.paths.train_log
    if not log:
        return
    step = lambda r: r.get("step", r.get("global_step", 0))  # noqa: E731
    n = len(log)
    idxs = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1}) if n else []
    rows = []
    for i in idxs:
        r = log[i]
        rows.append([step(r), r.get("loss", ""), r.get("lr", ""), r.get("grad_norm", ""),
                     r.get("epoch", ""), r.get("tokens_seen", ""), r.get("elapsed_s", "")])
    table_to(
        ctx, "training_curves", ["step", "loss", "learning rate", "grad norm", "epoch",
                                 "tokens seen", "elapsed s"],
        rows, caption="Training progress (quartiles of the logged history)",
        align=["r"] * 7,
        note="loss is the mean of the last loss window, not a smoothed value; it is logged from the "
             "training loop itself so it is comparable between runs with the same configuration.",
    )
    losses = [r.get("loss") for r in log if isinstance(r.get("loss"), (int, float))]
    if losses:
        first, last = mean(losses[: max(1, len(losses) // 20)]), mean(losses[-max(1, len(losses) // 20):])
        table_to(
            ctx, "training_summary",
            ["logged steps", "first loss (mean of first 5%)", "last loss (mean of last 5%)", "absolute drop"],
            [[len(log), f"{first:.4f}", f"{last:.4f}", f"{first - last:+.4f}"]],
            caption="Training loss summary", align=["r"] * 4,
        )


def make_calibration_table(ctx: Ctx) -> None:
    """tables/calibration - confidence metrics and the reliability diagram numbers."""
    if not ctx.confidence:
        return
    rows = []
    for set_name, systems in ctx.confidence.items():
        for system, entry in systems.items():
            cal = entry.get("metrics", {})
            if not cal:
                continue
            rows.append([
                SET_LABEL.get(set_name, set_name), SYSTEM_LABEL.get(system, system), cal["n"],
                f"{cal['ece']:.4f}", f"{cal['mce']:.4f}", f"{cal['brier']:.4f}",
                f"{cal['nll']:.4f}", f"{cal['auc_roc']:.4f}", f"{cal['auc_pr']:.4f}",
                f"{cal['mean_confidence']:.4f}", f"{cal['accuracy']:.4f}",
                f"{entry.get('risk_coverage', {}).get('aurc', float('nan')):.4f}",
                f"{entry.get('risk_coverage', {}).get('at_50pct', float('nan')):.4f}",
            ])
    table_to(
        ctx, "calibration", ["set", "system", "n answers", "ECE", "MCE", "Brier", "NLL",
                             "AUC-ROC", "AUC-PR", "mean confidence", "accuracy", "AURC",
                             "accuracy at 50% coverage"],
        rows, caption="Confidence calibration and selective-prediction metrics",
        align=["l", "l"] + ["r"] * 11,
        note="ECE/MCE = expected/maximum calibration error between confidence and accuracy. "
             "AUC-ROC and AUC-PR treat confidence as a score for 'the answer is correct'. AURC is "
             "the area under the risk-coverage curve, i.e. how fast accuracy drops as less "
             "confident answers are added back in - lower is better.",
    )
    rows = []
    for set_name, systems in ctx.confidence.items():
        for system, entry in systems.items():
            for b in entry.get("metrics", {}).get("bins", []):
                rows.append([
                    SET_LABEL.get(set_name, set_name), SYSTEM_LABEL.get(system, system),
                    f"{b['lo']:.2f}-{b['hi']:.2f}", b["n"],
                    f"{b['confidence']:.4f}", f"{b['accuracy']:.4f}", f"{b['gap']:.4f}",
                ])
    table_to(
        ctx, "reliability_diagram", ["set", "system", "confidence bin", "n", "mean confidence",
                                     "accuracy", "gap"],
        rows, caption="Reliability diagram data (15 equal-width confidence bins)",
        align=["l", "l", "l"] + ["r"] * 4,
    )


# @@TB4@@

# @@REPORT@@

# @@VERIFY@@

# --------------------------------------------------------------------------
