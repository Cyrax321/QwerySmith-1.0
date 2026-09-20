#!/usr/bin/env python3
"""
Qwerysmith 1.1: text-to-SQL fine-tuning pipeline (Qwen3-4B + QLoRA via Unsloth).

Stages (run one with --stage, or everything with --stage all):
  baseline  evaluate the untouched base model (zero-shot and 3-shot)
  train     QLoRA fine-tune, save the adapter
  eval      evaluate the fine-tuned adapter
  report    build the comparison table, CSV of predictions, and a chart
  export    optional: merged 16-bit model, GGUF, push to the Hugging Face Hub

Colab usage (T4 is enough):
  !pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" trl datasets matplotlib
  !python Qwerysmith_V11.py --smoke            # 5-10 min sanity check first
  !python Qwerysmith_V11.py --mix sql_create_context:5000,gretel:5000 --heldout sqale,large_schema

v1.1 changes from v1.0
-----------------------
v1.0 trained on 10k rows of a single dataset (b-mc2/sql-create-context) and,
while it improved sharply on that dataset's own held-out split, it REGRESSED
on an external text-to-SQL set relative to the untouched base model (fine
detail in runs/qwerysmith-1.0/results.md). That is single-source overfitting:
the model learned that dataset's narrow conventions (single-table schemas,
string-quoted numerics) rather than SQL generation in general.

v1.1 fixes this at the data layer, via data_sources.py:
  - training now draws from a configurable MIX of sources (--mix), so no
    single dataset's quirks can dominate
  - eval sets are carved out of every source BEFORE the training mix is
    built, so an eval question can never leak into training
  - an explicit --heldout list (comma-separated source names NEVER put in
    --mix) stays a true generalization check, even after gretel moves from
    "external" to "in-mix"; each named source gets its own heldout_<name>
    eval set
  - lora_dropout and checkpointing are now configurable (--dropout,
    --save-steps) to make the recipe ablation (gentler LR/dropout) separable
    from the data-mix ablation

See data_sources.py's module docstring for the source registry and the
exact run plan (run B = data mix only, run C = data mix + gentler recipe).
Your original v1.0 run directory and predictions remain valid as run A --
do not regenerate them, they are the baseline this version is measured
against.

Everything is cached in --out, so if Colab dies you can re-run the same
command and finished steps are skipped. Point --out at Google Drive
(from google.colab import drive; drive.mount("/content/drive")) so the
results survive the session.

Library APIs change often. If a call errors, check the Unsloth / TRL docs first.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from data_sources import build_sets

SEED = 42
MODEL_NAME = "Qwerysmith-1.1"
MODEL_DEFAULT = "unsloth/Qwen3-4B"

SYSTEM = (
    "You are a text-to-SQL assistant. Given a database schema and a question, "
    "reply with exactly one SQL query and nothing else."
)

# Qwen3 chat-format markers, used to (a) end each training example and (b) mask
# the loss so the model is only trained on the SQL answer. If you switch to a
# different model family, change these three.
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
END = "<|im_end|>"

SYSTEMS = ["base_zeroshot", "base_fewshot", "finetuned"]




# --------------------------------------------------------------------------
# 1. SQLite helpers (used for execution-based scoring)
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
    keep = []
    for s in split_statements(context):
        m = re.search(r"(?is)\bcreate\s+(table|view)\b", s)
        if m:
            keep.append(s[m.start():].strip() + ";")
    return "\n".join(keep) if keep else context.strip()


def _literals(sql: str):
    strs = re.findall(r"'([^']*)'", sql)
    nums = []
    for x in re.findall(r"(?<![\w.])\d+(?:\.\d+)?", sql):
        nums.append(float(x) if "." in x else int(x))
    return strs, nums


def populate_empty_tables(conn: sqlite3.Connection, gold: str, seed: int, n_rows: int = 40) -> None:
    """Fill tables that have no rows with random data, seeded with the gold
    query's own literals so that WHERE filters and joins actually match rows."""
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
        m = re.search(r"(?is)\b(create\s+(table|view)|insert\s+into)\b", stmt)
        if m:
            stmt = stmt[m.start():]
        try:
            conn.execute(stmt)
        except sqlite3.Error:
            pass
    populate_empty_tables(conn, gold, seed)
    return conn


def run_query(conn: sqlite3.Connection, sql: str, timeout: float = 3.0):
    if not re.match(r"(?is)^\s*(select|with)\b", sql or ""):
        return False, None  # empty output, chatter, or non-SELECT statements count as invalid
    start = time.time()
    conn.set_progress_handler(lambda: 1 if time.time() - start > timeout else 0, 10000)
    try:
        return True, conn.execute(sql).fetchmany(1000)
    except Exception:
        return False, None
    finally:
        conn.set_progress_handler(None, 0)


def _canon(rows) -> list[str]:
    return sorted(
        repr(tuple(round(v, 4) if isinstance(v, float) else v for v in r)) for r in rows
    )


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


def score(items: list[dict], preds: list[str]):
    """Returns (metrics, per_item). Execution accuracy only counts items whose
    gold query executes and returns at least one row."""
    valid = em = ex = scored = 0
    per = []
    for i, (it, pred) in enumerate(zip(items, preds)):
        conn = make_db(it["context"], it["gold"], seed=i)
        ok_p, rows_p = run_query(conn, pred)
        ok_g, rows_g = run_query(conn, it["gold"])
        conn.close()
        v = bool(ok_p)
        e = _norm(pred) == _norm(it["gold"])
        x = None
        if ok_g and rows_g:
            scored += 1
            x = bool(ok_p and _canon(rows_p) == _canon(rows_g))
            ex += int(x)
        valid += int(v)
        em += int(e)
        per.append({"valid": v, "em": e, "ex": x})
    n = len(items)
    metrics = {
        "n": n,
        "valid": valid / n if n else 0.0,
        "exact_match": em / n if n else 0.0,
        "exact_match_ci": wilson(em, n),
        "exec_acc": ex / scored if scored else 0.0,
        "exec_scored": scored,
        "exec_ci": wilson(ex, scored),
    }
    return metrics, per


# --------------------------------------------------------------------------
# 2. Data
#
# v1.1: single-source load_in_dist()/load_external() have moved to
# data_sources.py as a source registry + build_sets(), so training can draw
# from a configurable mix (--mix) while eval sets are carved out first and
# never leak into training. See that module's docstring for the registry
# and the run plan. schema_only/make_db/run_query above are imported back
# into data_sources.py, so this file stays the single source of truth for
# the SQL-execution logic.
# --------------------------------------------------------------------------
def user_msg(it: dict) -> dict:
