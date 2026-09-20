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
