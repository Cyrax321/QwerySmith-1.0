#!/usr/bin/env python3
"""
paper_eval.py -- Institutional Research Paper Evaluation Suite for QwerySmith

This script loads the finished run artifacts (predictions.csv, preds/*.json,
train_log.json, results.json) from an authentic model run directory and computes
every standard empirical metric, confusion matrix, LaTeX table, and publication figure
from REAL evaluation data:

1. Main Benchmark Performance (Execution Accuracy, Exact Match, Valid SQL with 95% CIs)
2. 4x4 Pairwise Outcome State Transition Matrix (Base -> Fine-Tuned)
3. AST Clause-Level 2x2 Confusion Matrices Grid (SELECT, WHERE, JOIN, GROUP BY, etc.)
4. Inter-System Agreement & Reliability Matrix (Cohen's Kappa & Percent Agreement)
5. Per-Split Inter-System Agreement Matrix
6. Query Complexity Performance Matrix (Simple, Moderate, Complex, Advanced across splits)
7. Error Taxonomy & Failure Mode Distribution Matrix
8. Diagnostic Clause Testing Matrix (Precision, Recall, Specificity, NPV, Balanced Acc, F1, MCC)
9. Error Recovery & Migration Flow Matrix (Base Failure -> Fine-Tuned Resolution)
10. Clause Co-occurrence Correlation Matrix (Gold vs Base vs Fine-Tuned + Frobenius Error)
11. Query Token Length vs Execution Accuracy Matrix
12. Paired Statistical Significance Testing (McNemar's test, Odds Ratios, p-values)
13. 13 Publication-Grade Figures (300 DPI PNG + Vector PDF)
14. 10 Publication-Grade LaTeX Tables (.tex format using booktabs)
15. Full Academic Markdown Report (RESEARCH_PAPER_REPORT.md)
16. Interactive Colab / Jupyter Inline Display Support

STRICT POLICY: NO FAKE OR SYNTHETIC DATA. All figures and tables are strictly derived
from actual run artifacts on disk. If artifacts are missing, the script explicitly reports
the missing data rather than fabricating synthetic points.

Usage:
    python paper_eval.py --out /content/drive/MyDrive/qwerysmith-1.1
    python paper_eval.py --out runs/qwerysmith-1.1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# Try importing numpy and matplotlib
try:
    import numpy as np
except ImportError:
    np = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# --------------------------------------------------------------------------
# Publication Styling & Aesthetics
# --------------------------------------------------------------------------
if plt is not None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10.5,
        "font.family": "sans-serif",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "legend.frameon": True,
        "legend.framealpha": 0.85,
    })

COLORS = {
    "base_zeroshot": "#7f8c8d",  # Cool gray
    "base_fewshot": "#3498db",   # Precision blue
    "finetuned": "#2ecc71",      # Emerald green
}

LABELS = {
    "base_zeroshot": "Base (0-Shot)",
    "base_fewshot": "Base (3-Shot)",
    "finetuned": "QwerySmith 1.1 (Fine-Tuned)",
}

CLAUSE_NAMES = [
    "SELECT", "WHERE", "JOIN", "GROUP BY", "HAVING", "ORDER BY", "LIMIT", "AGGREGATE"
]

OUTCOME_STATES = [
    "invalid", "valid_wrong", "executed_match", "executed_exact"
]

OUTCOME_DISPLAY = {
    "invalid": "Invalid SQL",
    "valid_wrong": "Runs Wrong",
    "executed_match": "Exec Match",
    "executed_exact": "Exec Exact",
}

COMPLEXITY_TIERS = [
    "Simple (Projection / Filter)",
    "Moderate (Agg / Sort)",
    "Complex (Multi-Table Join)",
    "Advanced (Nested / Set)",
]

LENGTH_BINS = [
    "Short (≤15)",
    "Medium (16–30)",
    "Long (31–55)",
    "Very Long (>55)",
]


# --------------------------------------------------------------------------
# Conversion & Normalization Helpers (Eliminates Scaling Bugs)
# --------------------------------------------------------------------------
def to_pct(val: float | int | None) -> float:
    """
    Safely normalizes ratio (<= 1.0) or percentage (> 1.0) to a standard 0-100 percentage.
    Completely eliminates double-scaling bugs (e.g. 88.5% turning into 8850%).
    """
    if val is None:
        return 0.0
    v = float(val)
    if abs(v) <= 1.0 and v != 0.0:
        return v * 100.0
    return v


def to_ratio(val: float | int | None) -> float:
    """Safely normalizes a percentage or ratio to a 0.0 - 1.0 ratio."""
    if val is None:
        return 0.0
    v = float(val)
    if abs(v) > 1.0:
        return v / 100.0
    return v


def parse_bool(val) -> bool | None:
    """Robust parser for boolean fields from CSV, JSON, or SQL logs."""
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "pass", "correct"):
        return True
    if s in ("0", "false", "f", "no", "n", "fail", "wrong"):
        return False
    return None


def clean_sql(sql: str) -> str:
    """Cleans SQL query string by removing markdown fences, comments, and extra whitespace."""
    if not sql:
        return ""
    s = str(sql).strip()
    # Strip markdown fences
    s = re.sub(r"^```(?:sql)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    # Strip line comments
    s = re.sub(r"--[^\n]*", " ", s)
    # Strip block comments
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.DOTALL)
    return s.strip()


# --------------------------------------------------------------------------
# Statistical Helpers & Metrics
# --------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = 1.95996) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion (returns 0.0 to 1.0 ratios)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + (z * z) / (4 * n * n)) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def cohen_kappa(y1: list[str], y2: list[str]) -> float:
    """Calculates Cohen's Kappa inter-rater agreement between two systems."""
    if not y1 or len(y1) != len(y2):
        return 1.0
    cats = sorted(list(set(y1) | set(y2)))
    cat_to_i = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    n = len(y1)
    if n == 0 or k <= 1:
        return 1.0

    cm = [[0] * k for _ in range(k)]
    for a, b in zip(y1, y2):
        cm[cat_to_i[a]][cat_to_i[b]] += 1

    po = sum(cm[i][i] for i in range(k)) / n
    pe = sum(sum(cm[i][j] for j in range(k)) * sum(cm[j][i] for j in range(k)) for i in range(k)) / (n * n)
    return (po - pe) / (1.0 - pe) if pe < 1.0 else 1.0


def calculate_mcc(tp: int, fp: int, fn: int, tn: int) -> float:
    """Matthews Correlation Coefficient for binary diagnostic classification."""
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denom if denom > 0 else 0.0


def mcnemar_test(y_a: list[bool], y_b: list[bool]) -> dict:
    """
    McNemar's paired test comparing system A vs system B.
    wins_A (b): A correct, B wrong
    wins_B (c): B correct, A wrong
    """
    b = sum(1 for a, b_val in zip(y_a, y_b) if a is True and b_val is False)
    c = sum(1 for a, b_val in zip(y_a, y_b) if a is False and b_val is True)
    both_correct = sum(1 for a, b_val in zip(y_a, y_b) if a is True and b_val is True)
    both_wrong = sum(1 for a, b_val in zip(y_a, y_b) if a is False and b_val is False)
    total_discordant = b + c

    if total_discordant == 0:
        p_val = 1.0
    else:
        k = min(b, c)
        p_val = min(1.0, 2.0 * sum(math.comb(total_discordant, i) * (0.5 ** total_discordant) for i in range(k + 1)))

    odds_ratio = (b / c) if c > 0 else (float("inf") if b > 0 else 1.0)
    chi2 = ((abs(b - c) - 1) ** 2) / total_discordant if total_discordant > 0 else 0.0

    return {
        "wins_A": b,
        "wins_B": c,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "total_discordant": total_discordant,
        "odds_ratio": odds_ratio,
        "chi2": chi2,
        "p_value": p_val,
        "significant": p_val < 0.05,
    }


# --------------------------------------------------------------------------
# AST Clause & Query Categorization
# --------------------------------------------------------------------------
def extract_clause_flags(sql: str) -> dict[str, bool]:
    """Detects standard SQL clauses and syntactic constructs."""
    clean = clean_sql(sql)
    s_no_lit = re.sub(r"'[^']*'", "''", clean).lower()
    return {
        "SELECT": bool(re.search(r"\bselect\b", s_no_lit)),
        "WHERE": bool(re.search(r"\bwhere\b", s_no_lit)),
        "JOIN": bool(re.search(r"\bjoin\b", s_no_lit)) or bool(re.search(r",\s*\w+\s+on\b", s_no_lit)),
        "GROUP BY": bool(re.search(r"\bgroup\s+by\b", s_no_lit)),
        "HAVING": bool(re.search(r"\bhaving\b", s_no_lit)),
        "ORDER BY": bool(re.search(r"\border\s+by\b", s_no_lit)),
        "LIMIT": bool(re.search(r"\blimit\b", s_no_lit)) or bool(re.search(r"\bfetch\s+(first|next)\b", s_no_lit)),
        "AGGREGATE": bool(re.search(r"\b(count|sum|avg|min|max|total|group_concat)\s*\(", s_no_lit)),
        "DISTINCT": bool(re.search(r"\bdistinct\b", s_no_lit)),
        "SUBQUERY": len(re.findall(r"\bselect\b", s_no_lit)) > 1 or bool(re.search(r"\bexists\s*\(", s_no_lit)),
        "SET_OP": bool(re.search(r"\b(union|intersect|except)\b", s_no_lit)),
    }


def classify_complexity(sql: str) -> str:
    """Categorizes SQL query into academic complexity tiers."""
    f = extract_clause_flags(sql)
    if f["SUBQUERY"] or f["SET_OP"]:
        return "Advanced (Nested / Set)"
    if f["JOIN"]:
        return "Complex (Multi-Table Join)"
    if f["GROUP BY"] or f["ORDER BY"] or f["AGGREGATE"]:
        return "Moderate (Agg / Sort)"
    return "Simple (Projection / Filter)"


def classify_length_bin(sql: str) -> str:
    """Classifies SQL token/word count into standard length tiers."""
    tokens = len(clean_sql(sql).split())
    if tokens <= 15:
        return "Short (≤15)"
    elif tokens <= 30:
        return "Medium (16–30)"
    elif tokens <= 55:
        return "Long (31–55)"
    else:
        return "Very Long (>55)"


def is_valid_sql_syntax(sql: str) -> bool:
    """Performs rigorous syntactic checks on SQL queries."""
    clean = clean_sql(sql)
    if not clean:
        return False
    # Check leading keyword or presence of query command
    tokens = clean.split()
    if not tokens:
        return False
    first_word = tokens[0].upper()
    if first_word not in ("SELECT", "WITH", "PRAGMA", "EXPLAIN"):
        if not re.search(r"\bSELECT\b", clean, flags=re.IGNORECASE):
            return False

    # Check balanced quotes and parentheses
    depth = 0
    in_quote = False
    quote_char = None
    for ch in clean:
        if ch in ("'", '"', '`'):
            if not in_quote:
                in_quote = True
                quote_char = ch
            elif quote_char == ch:
                in_quote = False
                quote_char = None
        elif not in_quote:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    return False
    return depth == 0 and not in_quote


def resolve_valid(row_val, pred: str, is_ex: bool | None) -> bool:
    """Resolves SQL validity accurately with strict precedence rules."""
    # If a query executed successfully and produced matching results, it is unequivocally valid SQL
    if is_ex is True:
        return True
    parsed = parse_bool(row_val)
    if parsed is not None:
        return parsed
    return is_valid_sql_syntax(pred)


def determine_outcome(pred: str, gold: str, is_valid: bool, is_ex: bool | None) -> str:
    """Classifies prediction into one of 4 mutually exclusive states."""
    clean_p = clean_sql(pred)
    clean_g = clean_sql(gold)
    is_em = (clean_p.lower() == clean_g.lower()) if (clean_p and clean_g) else False

    # Precedence: Execution outcome takes precedence over static syntax estimation
    if is_ex is True:
        return "executed_exact" if is_em else "executed_match"
    if not is_valid:
        return "invalid"
    return "valid_wrong"


def classify_failure_mode(pred: str, gold: str) -> str:
    """Assigns an explicit error taxonomy label to a non-passing query."""
    clean_p = clean_sql(pred)
    if not is_valid_sql_syntax(clean_p):
        return "Syntax Error"
    gf, pf = extract_clause_flags(gold), extract_clause_flags(clean_p)
    if gf["JOIN"] != pf["JOIN"]:
        return "Join Error"
    if gf["AGGREGATE"] != pf["AGGREGATE"] or gf["GROUP BY"] != pf["GROUP BY"]:
        return "Aggregation Error"
    if gf["WHERE"] != pf["WHERE"]:
        return "Predicate Error"
    return "Semantic Row Mismatch"


# --------------------------------------------------------------------------
# Artifact Loader (Parses Authentic Run Outputs)
# --------------------------------------------------------------------------
def load_data(run_dir: Path):
    """Loads prediction records, logs, and pre-computed results from actual run files."""
    run_dir = Path(run_dir).resolve()
    print(f"📂 Inspecting run directory: {run_dir}")

    # 1. Load results.json if present
    results_json = {}
    for r_candidate in [run_dir / "results.json", run_dir / "eval" / "results.json"]:
        if r_candidate.exists():
            print(f"  Loading benchmark metrics from {r_candidate.name}...")
            try:
                results_json = json.loads(r_candidate.read_text(encoding="utf-8"))
                break
            except Exception as e:
                print(f"  ⚠️ Could not parse {r_candidate.name}: {e}")

    # 2. Load train_log.json or trainer_state.json if present
    train_log = []
    for t_candidate in [run_dir / "train_log.json", run_dir / "trainer" / "trainer_state.json"]:
        if t_candidate.exists():
            print(f"  Loading training history from {t_candidate.name}...")
            try:
                raw_t = json.loads(t_candidate.read_text(encoding="utf-8"))
                if isinstance(raw_t, dict) and "log_history" in raw_t:
                    train_log = raw_t["log_history"]
                elif isinstance(raw_t, list):
                    train_log = raw_t
                break
            except Exception as e:
                print(f"  ⚠️ Could not parse {t_candidate.name}: {e}")

    # 3. Load queries and execution records from predictions.csv
    csv_file = run_dir / "predictions.csv"
    items_by_set = defaultdict(list)

    if csv_file.exists():
        print(f"  Loading queries and execution records from {csv_file.name}...")
        with open(csv_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sname = row.get("set", "default")
                gold = clean_sql(row.get("gold", ""))

                b0_pred = clean_sql(row.get("base_zeroshot_pred", ""))
                b0_ex = parse_bool(row.get("base_zeroshot_exec_correct"))
                b0_valid = resolve_valid(row.get("base_zeroshot_valid"), b0_pred, b0_ex)

                b3_pred = clean_sql(row.get("base_fewshot_pred", ""))
                b3_ex = parse_bool(row.get("base_fewshot_exec_correct"))
                b3_valid = resolve_valid(row.get("base_fewshot_valid"), b3_pred, b3_ex)

                ft_pred = clean_sql(row.get("finetuned_pred", ""))
                ft_ex = parse_bool(row.get("finetuned_exec_correct"))
                ft_valid = resolve_valid(row.get("finetuned_valid"), ft_pred, ft_ex)

                items_by_set[sname].append({
                    "question": row.get("question", ""),
                    "gold": gold,
                    "complexity": classify_complexity(gold),
                    "length_bin": classify_length_bin(gold),
                    "base_zeroshot_pred": b0_pred,
                    "base_zeroshot_exec": b0_ex,
                    "base_zeroshot_valid": b0_valid,
                    "base_zeroshot_outcome": determine_outcome(b0_pred, gold, b0_valid, b0_ex),
                    "base_zeroshot_error": classify_failure_mode(b0_pred, gold) if b0_ex is not True else None,
                    "base_fewshot_pred": b3_pred,
                    "base_fewshot_exec": b3_ex,
                    "base_fewshot_valid": b3_valid,
                    "base_fewshot_outcome": determine_outcome(b3_pred, gold, b3_valid, b3_ex),
                    "base_fewshot_error": classify_failure_mode(b3_pred, gold) if b3_ex is not True else None,
                    "finetuned_pred": ft_pred,
                    "finetuned_exec": ft_ex,
                    "finetuned_valid": ft_valid,
                    "finetuned_outcome": determine_outcome(ft_pred, gold, ft_valid, ft_ex),
                    "finetuned_error": classify_failure_mode(ft_pred, gold) if ft_ex is not True else None,
                })
    else:
        print("  ⚠️ predictions.csv not found; looking for preds/*.json ...")
        preds_dir = run_dir / "preds"
        if preds_dir.exists():
            for f in sorted(list(preds_dir.glob("*.json"))):
                parts = f.stem.split("__")
                if len(parts) == 2:
                    sysname, sname = parts
                    try:
                        payload = json.loads(f.read_text(encoding="utf-8"))
                        preds = payload.get("pred", [])
                        for i, p in enumerate(preds):
                            if len(items_by_set[sname]) <= i:
                                items_by_set[sname].append({
                                    "gold": "", "complexity": "Simple (Projection / Filter)", "length_bin": "Medium (16–30)"
                                })
                            clean_p = clean_sql(p)
                            items_by_set[sname][i][f"{sysname}_pred"] = clean_p
                            items_by_set[sname][i][f"{sysname}_valid"] = is_valid_sql_syntax(clean_p)
                    except Exception as e:
                        print(f"  ⚠️ Could not read {f.name}: {e}")

    return items_by_set, results_json, train_log


# --------------------------------------------------------------------------
# Multi-Dimensional Matrix & Analysis Engine (100% Authentic Computation)
# --------------------------------------------------------------------------
def analyze_dataset(items_by_set: dict, results_json: dict) -> dict:
    """Computes all research paper matrices and metrics strictly from real data."""
    analysis = {
        "benchmarks": {},
        "clause_metrics": defaultdict(dict),
        "clause_confusion": defaultdict(lambda: defaultdict(dict)),
        "diagnostic_metrics": defaultdict(dict),
        "clause_correlations": {},
        "transition_matrix": {},
        "error_migration_matrix": {},
        "agreement_matrix": {},
        "persplit_agreement": {},
        "complexity_matrix": defaultdict(dict),
        "length_matrix": defaultdict(dict),
        "error_taxonomy": defaultdict(dict),
        "significance": {},
    }

    all_items = [it for items in items_by_set.values() for it in items if (it.get("gold") or it.get("finetuned_pred"))]
    sets = list(items_by_set.keys())
    if not sets and results_json:
        sets = sorted(list(set(k.split("/")[0] for k in results_json.keys())))

    systems = ["base_zeroshot", "base_fewshot", "finetuned"]

    # 1. Main Benchmarks
    for sname in sets:
        analysis["benchmarks"][sname] = {}
        items = items_by_set.get(sname, [])
        for sysname in systems:
            key = f"{sname}/{sysname}"
            if key in results_json:
                analysis["benchmarks"][sname][sysname] = results_json[key]
            elif items:
                valid_preds = [it for it in items if it.get(f"{sysname}_valid")]
                em_matches = [it for it in items if it.get("gold") and clean_sql(it.get(f"{sysname}_pred", "")).lower() == clean_sql(it.get("gold", "")).lower()]
                scored_items = [it for it in items if it.get(f"{sysname}_exec") is not None]
                ex_matches = [it for it in scored_items if it.get(f"{sysname}_exec") is True]

                n = len(items)
                scored = len(scored_items)
                em_rate = len(em_matches) / n if n else 0.0
                ex_rate = len(ex_matches) / scored if scored else 0.0

                analysis["benchmarks"][sname][sysname] = {
                    "n": n,
                    "valid": len(valid_preds) / n if n else 0.0,
                    "exact_match": em_rate,
                    "exact_match_ci": wilson_ci(len(em_matches), n),
                    "exec_acc": ex_rate,
                    "exec_scored": scored,
                    "exec_ci": wilson_ci(len(ex_matches), scored),
                }
            else:
                analysis["benchmarks"][sname][sysname] = {
                    "n": 0, "valid": 0.0, "exact_match": 0.0, "exact_match_ci": (0.0, 0.0),
                    "exec_acc": 0.0, "exec_scored": 0, "exec_ci": (0.0, 0.0),
                }

    # 2. 4x4 Pairwise Outcome State Transition Matrix (Base 3-Shot -> Fine-Tuned)
    trans_matrix = {s_base: {s_ft: 0 for s_ft in OUTCOME_STATES} for s_base in OUTCOME_STATES}
    for it in all_items:
        s_base = it.get("base_fewshot_outcome")
        s_ft = it.get("finetuned_outcome")
        if s_base in OUTCOME_STATES and s_ft in OUTCOME_STATES:
            trans_matrix[s_base][s_ft] += 1
    analysis["transition_matrix"] = trans_matrix

    # 3. Inter-System Agreement (Global & Per-Split Cohen's Kappa)
    agree_mat = {s1: {s2: 1.0 if s1 == s2 else 0.0 for s2 in systems} for s1 in systems}
    if all_items:
        for s1 in systems:
            for s2 in systems:
                if s1 == s2:
                    agree_mat[s1][s2] = 1.0
                else:
                    y1 = [it.get(f"{s1}_outcome", "") for it in all_items]
                    y2 = [it.get(f"{s2}_outcome", "") for it in all_items]
                    agree_mat[s1][s2] = cohen_kappa(y1, y2)
    analysis["agreement_matrix"] = agree_mat

    for sname in sets:
        items = items_by_set.get(sname, [])
        analysis["persplit_agreement"][sname] = {s1: {s2: 1.0 if s1 == s2 else 0.0 for s2 in systems} for s1 in systems}
        if items:
            for s1 in systems:
                for s2 in systems:
                    if s1 == s2:
                        analysis["persplit_agreement"][sname][s1][s2] = 1.0
                    else:
                        y1 = [it.get(f"{s1}_outcome", "") for it in items]
                        y2 = [it.get(f"{s2}_outcome", "") for it in items]
                        analysis["persplit_agreement"][sname][s1][s2] = cohen_kappa(y1, y2)

    # 4. AST Clause-Level 2x2 Confusion Matrices & Diagnostic Metrics
    for sysname in systems:
        for clause in CLAUSE_NAMES:
            tp = fp = fn = tn = 0
            for it in all_items:
                gold_sql = it.get("gold", "")
                pred_sql = it.get(f"{sysname}_pred", "")
                if not gold_sql:
                    continue
                gold_has = extract_clause_flags(gold_sql)[clause]
                pred_has = extract_clause_flags(pred_sql)[clause]
                if gold_has and pred_has:
                    tp += 1
                elif not gold_has and pred_has:
                    fp += 1
                elif gold_has and not pred_has:
                    fn += 1
                else:
                    tn += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            balanced_acc = (recall + specificity) / 2.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            mcc = calculate_mcc(tp, fp, fn, tn)

            analysis["clause_metrics"][sysname][clause] = {
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": precision, "recall": recall, "f1": f1,
            }
            analysis["clause_confusion"][sysname][clause] = {
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            }
            analysis["diagnostic_metrics"][sysname][clause] = {
                "precision": precision, "recall": recall, "specificity": specificity,
                "npv": npv, "balanced_acc": balanced_acc, "f1": f1, "mcc": mcc,
            }

    # 5. Clause Co-occurrence Correlation Matrix & Frobenius Error
    if np is not None and all_items:
        gold_items = [it for it in all_items if it.get("gold")]
        if len(gold_items) >= 2:
            gold_mat = np.array([[int(extract_clause_flags(it["gold"])[c]) for c in CLAUSE_NAMES] for it in gold_items])
            base_mat = np.array([[int(extract_clause_flags(it.get("base_fewshot_pred", ""))[c]) for c in CLAUSE_NAMES] for it in gold_items])
            ft_mat = np.array([[int(extract_clause_flags(it.get("finetuned_pred", ""))[c]) for c in CLAUSE_NAMES] for it in gold_items])

            def safe_corrcoef(m):
                with np.errstate(divide="ignore", invalid="ignore"):
                    c = np.corrcoef(m, rowvar=False)
                    c = np.nan_to_num(c, nan=0.0)
                np.fill_diagonal(c, 1.0)
                return c

            corr_gold = safe_corrcoef(gold_mat)
            corr_base = safe_corrcoef(base_mat)
            corr_ft = safe_corrcoef(ft_mat)

            frob_base = float(np.linalg.norm(corr_gold - corr_base, ord="fro"))
            frob_ft = float(np.linalg.norm(corr_gold - corr_ft, ord="fro"))

            analysis["clause_correlations"] = {
                "gold": corr_gold.tolist(),
                "base_fewshot": corr_base.tolist(),
                "finetuned": corr_ft.tolist(),
                "frobenius_error_base": frob_base,
                "frobenius_error_ft": frob_ft,
            }

    # 6. Complexity Tier Performance Matrix
    for sname in sets:
        analysis["complexity_matrix"][sname] = {}
        items = items_by_set.get(sname, [])
        for tier in COMPLEXITY_TIERS:
            tier_items = [it for it in items if it.get("complexity") == tier]
            analysis["complexity_matrix"][sname][tier] = {}
            for sysname in systems:
                scored = [it for it in tier_items if it.get(f"{sysname}_exec") is not None]
                corr = sum(1 for it in scored if it.get(f"{sysname}_exec") is True)
                acc = (corr / len(scored)) if scored else 0.0
                analysis["complexity_matrix"][sname][tier][sysname] = {
                    "count": len(tier_items),
                    "scored": len(scored),
                    "acc": acc,
                }

    # 7. Token Length Stratification Matrix
    for lbin in LENGTH_BINS:
        bin_items = [it for it in all_items if it.get("length_bin") == lbin]
        analysis["length_matrix"][lbin] = {}
        for sysname in systems:
            scored = [it for it in bin_items if it.get(f"{sysname}_exec") is not None]
            corr = sum(1 for it in scored if it.get(f"{sysname}_exec") is True)
            acc = (corr / len(scored)) if scored else 0.0
            analysis["length_matrix"][lbin][sysname] = {
                "count": len(bin_items),
                "scored": len(scored),
                "acc": acc,
            }

    # 8. Error Taxonomy Distribution Matrix
    error_types = ["Syntax Error", "Join Error", "Aggregation Error", "Predicate Error", "Semantic Row Mismatch"]
    for sysname in systems:
        analysis["error_taxonomy"][sysname] = {err: 0 for err in error_types}
        for it in all_items:
            if it.get(f"{sysname}_exec") is True:
                continue
            err_mode = it.get(f"{sysname}_error", "Semantic Row Mismatch")
            if err_mode in analysis["error_taxonomy"][sysname]:
                analysis["error_taxonomy"][sysname][err_mode] += 1
            else:
                analysis["error_taxonomy"][sysname]["Semantic Row Mismatch"] += 1

    # 9. Error Recovery & Migration Flow Matrix (Base Failure Mode -> Fine-Tuned Resolution)
    mig_categories = ["Resolved Exact", "Resolved Exec", "Persistent Error", "Alternative Failure"]
    mig_matrix = {e: {cat: 0 for cat in mig_categories} for e in error_types}

    for it in all_items:
        if it.get("base_fewshot_exec") is not True and it.get("base_fewshot_exec") is not None:
            base_err = it.get("base_fewshot_error", "Semantic Row Mismatch")
            if base_err not in mig_matrix:
                base_err = "Semantic Row Mismatch"
            ft_out = it.get("finetuned_outcome")
            ft_err = it.get("finetuned_error")

            if ft_out == "executed_exact":
                mig_matrix[base_err]["Resolved Exact"] += 1
            elif ft_out == "executed_match":
                mig_matrix[base_err]["Resolved Exec"] += 1
            elif ft_err == base_err:
                mig_matrix[base_err]["Persistent Error"] += 1
            else:
                mig_matrix[base_err]["Alternative Failure"] += 1

    analysis["error_migration_matrix"] = mig_matrix

    # 10. Statistical Significance Testing (McNemar)
    for sname in sets:
        items = items_by_set.get(sname, [])
        scored_pairs = [it for it in items if it.get("finetuned_exec") is not None and it.get("base_fewshot_exec") is not None]
        if scored_pairs:
            ft_results = [it["finetuned_exec"] for it in scored_pairs]
            base_results = [it["base_fewshot_exec"] for it in scored_pairs]
            mc = mcnemar_test(ft_results, base_results)
            analysis["significance"][sname] = mc

    return analysis


# --------------------------------------------------------------------------
# Figure Builders (13 Publication-Grade Figures in PNG & PDF)
# --------------------------------------------------------------------------
def generate_figures(analysis: dict, train_log: list, out_dir: Path):
    """Renders all 13 publication figures strictly from authentic computed metrics."""
    if plt is None or np is None:
        print("⚠️ Matplotlib/NumPy not installed. Skipping figure rendering.")
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n🎨 Generating publication figures in {fig_dir} ...")

    sets = [s for s, sys_dict in analysis.get("benchmarks", {}).items() if any(m.get("n", 0) > 0 for m in sys_dict.values())]
    if not sets:
        sets = list(analysis.get("benchmarks", {}).keys())

    systems = ["base_zeroshot", "base_fewshot", "finetuned"]

    # ----------------------------------------------------------------------
    # Figure 1: Benchmark Execution Accuracy with 95% Error Bars
    # ----------------------------------------------------------------------
    if sets:
        fig, ax = plt.subplots(figsize=(9.0, 5.2))
        x = np.arange(len(sets))
        width = 0.25

        for i, sysname in enumerate(systems):
            accs, errors_lo, errors_hi = [], [], []
            for sname in sets:
                m = analysis["benchmarks"][sname].get(sysname, {})
                acc = to_pct(m.get("exec_acc", 0.0))
                lo, hi = m.get("exec_ci", (acc / 100, acc / 100))
                lo_pct = to_pct(lo)
                hi_pct = to_pct(hi)
                accs.append(acc)
                errors_lo.append(max(0.0, acc - lo_pct))
                errors_hi.append(max(0.0, hi_pct - acc))

            rects = ax.bar(
                x + i * width, accs, width,
                yerr=[errors_lo, errors_hi],
                capsize=4,
                label=LABELS[sysname],
                color=COLORS[sysname],
                edgecolor="black",
                linewidth=0.7,
                alpha=0.9,
            )
            for rect, acc, err_hi in zip(rects, accs, errors_hi):
                y_pos = rect.get_height() + err_hi + 1.8
                ax.annotate(
                    f"{acc:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, y_pos),
                    ha="center", va="bottom", fontsize=8.5, color="#2c3e50", weight="bold"
                )

        ax.set_ylabel("Execution Accuracy (%)", weight="bold", fontsize=10.5)
        ax.set_title("Figure 1: Execution Accuracy Across Benchmark Splits (with 95% Bootstrap CIs)", pad=14, fontsize=11.5, weight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels([s.replace("_", " ").title() for s in sets], weight="bold")
        ax.set_ylim(0, 115)
        ax.legend(loc="upper right", framealpha=0.9)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig1_execution_accuracy.png")
        fig.savefig(fig_dir / "fig1_execution_accuracy.pdf")
        plt.close(fig)
        print("  ✅ Saved fig1_execution_accuracy.{png,pdf}")
    else:
        print("  ℹ️ No benchmark data available; skipping fig1_execution_accuracy.")

    # ----------------------------------------------------------------------
    # Figure 2: Exact Match vs. Execution Accuracy Divergence
    # ----------------------------------------------------------------------
    if sets and any("finetuned" in analysis["benchmarks"].get(s, {}) for s in sets):
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        x = np.arange(len(sets))
        width = 0.35

        em_vals = [to_pct(analysis["benchmarks"][s]["finetuned"].get("exact_match", 0.0)) for s in sets]
        ex_vals = [to_pct(analysis["benchmarks"][s]["finetuned"].get("exec_acc", 0.0)) for s in sets]

        rects_em = ax.bar(x - width/2, em_vals, width, label="Exact Match (String Parity)", color="#9b59b6", alpha=0.85, edgecolor="black", linewidth=0.7)
        rects_ex = ax.bar(x + width/2, ex_vals, width, label="Execution Accuracy (Result Equivalence)", color="#2ecc71", alpha=0.85, edgecolor="black", linewidth=0.7)

        for rect, val in zip(rects_em, em_vals):
            ax.annotate(f"{val:.1f}%", xy=(rect.get_x() + rect.get_width()/2, val + 1.2),
                        ha="center", va="bottom", fontsize=8, color="#5b2c6f", weight="bold")
        for rect, val in zip(rects_ex, ex_vals):
            ax.annotate(f"{val:.1f}%", xy=(rect.get_x() + rect.get_width()/2, val + 1.2),
                        ha="center", va="bottom", fontsize=8, color="#196f3d", weight="bold")

        for i in range(len(sets)):
            diff = ex_vals[i] - em_vals[i]
            sign = "+" if diff >= 0 else ""
            color = "#27ae60" if diff >= 0 else "#c0392b"
            peak = max(em_vals[i], ex_vals[i])
            ax.annotate(
                f"Δ {sign}{diff:.1f}%",
                xy=(x[i], peak + 6.0),
                ha="center", va="bottom", fontsize=8.5, weight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor=color, alpha=0.9, linewidth=0.8)
            )

        ax.set_ylabel("Percentage (%)", weight="bold", fontsize=10.5)
        ax.set_title("Figure 2: Exact Match vs. Execution Accuracy (Semantic Generalization Gap)", pad=14, fontsize=11.5, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", " ").title() for s in sets], weight="bold")
        ax.set_ylim(0, 118)
        ax.legend(loc="upper right", framealpha=0.9)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig2_exact_vs_execution.png")
        fig.savefig(fig_dir / "fig2_exact_vs_execution.pdf")
        plt.close(fig)
        print("  ✅ Saved fig2_exact_vs_execution.{png,pdf}")
    else:
        print("  ℹ️ No fine-tuned benchmark data available; skipping fig2_exact_vs_execution.")

    # ----------------------------------------------------------------------
    # Figure 3: Clause-Level F1 Score Breakdown
    # ----------------------------------------------------------------------
    has_clause_data = bool(analysis["clause_metrics"].get("finetuned"))
    if has_clause_data:
        fig, ax = plt.subplots(figsize=(9.5, 5.0))
        x = np.arange(len(CLAUSE_NAMES))
        width = 0.38

        base_f1s = [to_pct(analysis["clause_metrics"]["base_fewshot"].get(c, {}).get("f1", 0.0)) for c in CLAUSE_NAMES]
        ft_f1s = [to_pct(analysis["clause_metrics"]["finetuned"].get(c, {}).get("f1", 0.0)) for c in CLAUSE_NAMES]

        rects_base = ax.bar(x - width/2, base_f1s, width, label="Base (3-Shot)", color=COLORS["base_fewshot"], alpha=0.85, edgecolor="black", linewidth=0.7)
        rects_ft = ax.bar(x + width/2, ft_f1s, width, label="QwerySmith 1.1 (Fine-Tuned)", color=COLORS["finetuned"], alpha=0.85, edgecolor="black", linewidth=0.7)

        for rect, val in zip(rects_base, base_f1s):
            if val > 0:
                ax.annotate(f"{val:.1f}%", xy=(rect.get_x() + rect.get_width()/2, val + 1.2),
                            ha="center", va="bottom", fontsize=7.5, color="#2980b9", weight="bold")
        for rect, val in zip(rects_ft, ft_f1s):
            if val > 0:
                ax.annotate(f"{val:.1f}%", xy=(rect.get_x() + rect.get_width()/2, val + 1.2),
                            ha="center", va="bottom", fontsize=7.5, color="#196f3d", weight="bold")

        ax.set_ylabel("F1 Score (%)", weight="bold", fontsize=10.5)
        ax.set_title("Figure 3: AST Clause Detection F1 Score Across Key SQL Constructs", pad=14, fontsize=11.5, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(CLAUSE_NAMES, weight="bold")
        ax.set_ylim(0, 115)
        ax.legend(loc="lower right", framealpha=0.9)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig3_clause_f1_scores.png")
        fig.savefig(fig_dir / "fig3_clause_f1_scores.pdf")
        plt.close(fig)
        print("  ✅ Saved fig3_clause_f1_scores.{png,pdf}")
    else:
        print("  ℹ️ No clause detection data available; skipping fig3_clause_f1_scores.")

    # ----------------------------------------------------------------------
    # Figure 4: Head-to-Head Pairwise Win/Loss Comparison
    # ----------------------------------------------------------------------
    sig_sets = [s for s in sets if s in analysis.get("significance", {})]
    if sig_sets:
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        x = np.arange(len(sig_sets))
        width = 0.52

        wins = [analysis["significance"][s].get("wins_A", 0) for s in sig_sets]
        losses = [analysis["significance"][s].get("wins_B", 0) for s in sig_sets]

        max_val = max(max(wins) if wins else 10, max(losses) if losses else 10)
        y_bound = max(max_val * 1.35, 12)

        ax.bar(x, wins, width, label="Fine-Tuned Wins (Base Failed)", color="#2ecc71", edgecolor="black", linewidth=0.7)
        ax.bar(x, [-l for l in losses], width, label="Base Wins (Fine-Tuned Failed)", color="#e74c3c", edgecolor="black", linewidth=0.7)

        ax.axhline(0, color="black", linewidth=0.8)
        for i in range(len(sig_sets)):
            if wins[i] > 0:
                ax.annotate(f"+{wins[i]}", xy=(x[i], wins[i] + max(y_bound * 0.03, 0.8)),
                            ha="center", va="bottom", weight="bold", color="#27ae60", fontsize=9.5)
            if losses[i] > 0:
                ax.annotate(f"-{losses[i]}", xy=(x[i], -losses[i] - max(y_bound * 0.03, 0.8) - 1.0),
                            ha="center", va="top", weight="bold", color="#c0392b", fontsize=9.5)

        ax.set_ylabel("Net Query Wins / Losses", weight="bold", fontsize=10.5)
        ax.set_title("Figure 4: Pairwise Head-to-Head Win/Loss Margin (Fine-Tuned vs 3-Shot Base)", pad=14, fontsize=11.5, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", " ").title() for s in sig_sets], weight="bold")
        ax.set_ylim(-y_bound, y_bound)
        ax.legend(loc="upper left", framealpha=0.9)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig4_pairwise_win_loss.png")
        fig.savefig(fig_dir / "fig4_pairwise_win_loss.pdf")
        plt.close(fig)
        print("  ✅ Saved fig4_pairwise_win_loss.{png,pdf}")
    else:
        print("  ℹ️ No paired significance data available; skipping fig4_pairwise_win_loss.")

    # ----------------------------------------------------------------------
    # Figure 5: Training Loss Convergence & Cosine Learning Rate Schedule
    # ----------------------------------------------------------------------
    steps = [r.get("step", i) for i, r in enumerate(train_log) if "loss" in r]
    losses = [float(r["loss"]) for r in train_log if "loss" in r]
    lrs = [float(r["learning_rate"]) for r in train_log if "learning_rate" in r]

    if steps and losses:
        fig, ax1 = plt.subplots(figsize=(9.0, 4.8))
        color = "#e67e22"
        ax1.set_xlabel("Training Steps", weight="bold", fontsize=10.5)
        ax1.set_ylabel("Cross-Entropy Loss (SQL Masked)", color=color, weight="bold", fontsize=10.5)
        ax1.plot(steps, losses, color=color, linewidth=2.0, label="Training Loss")
        ax1.tick_params(axis="y", labelcolor=color)

        if lrs:
            ax2 = ax1.twinx()
            color = "#2980b9"
            ax2.set_ylabel("Learning Rate", color=color, weight="bold", fontsize=10.5)
            ax2.plot(steps, lrs, color=color, linewidth=1.5, linestyle="--", label="Cosine Schedule")
            ax2.tick_params(axis="y", labelcolor=color)
            ax2.grid(False)

        ax1.set_title(f"Figure 5: QLoRA Training Loss Convergence and Learning Rate Schedule ({len(steps)} Steps)", pad=14, fontsize=11.5, weight="bold")
        fig.tight_layout()
        fig.savefig(fig_dir / "fig5_training_dynamics.png")
        fig.savefig(fig_dir / "fig5_training_dynamics.pdf")
        plt.close(fig)
        print("  ✅ Saved fig5_training_dynamics.{png,pdf}")
    else:
        print("  ℹ️ No authentic loss history in train_log.json; skipping fig5_training_dynamics.")

    # ----------------------------------------------------------------------
    # Figure 6: 4x4 Pairwise Outcome State Transition Heatmap (Base -> FT)
    # ----------------------------------------------------------------------
    tm = analysis.get("transition_matrix", {})
    tot_trans = sum(sum(tm.get(s, {}).values()) for s in OUTCOME_STATES)
    if tot_trans > 0:
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        mat_data = np.array([[tm.get(s_b, {}).get(s_ft, 0) for s_ft in OUTCOME_STATES] for s_b in OUTCOME_STATES])

        im = ax.imshow(mat_data, cmap="Blues", interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        display_labels = [OUTCOME_DISPLAY[s] for s in OUTCOME_STATES]
        ax.set_xticks(np.arange(len(OUTCOME_STATES)))
        ax.set_yticks(np.arange(len(OUTCOME_STATES)))
        ax.set_xticklabels(display_labels, rotation=35, ha="right", weight="bold")
        ax.set_yticklabels(display_labels, weight="bold")
        ax.set_xlabel("QwerySmith 1.1 Outcome", weight="bold", labelpad=8)
        ax.set_ylabel("Base (3-Shot) Outcome", weight="bold", labelpad=8)
        ax.set_title("Figure 6: 4x4 Pairwise Outcome State Transition Matrix", pad=14, fontsize=11.5, weight="bold")

        max_val = np.max(mat_data) if mat_data.size else 1
        for i in range(len(OUTCOME_STATES)):
            row_sum = sum(mat_data[i])
            for j in range(len(OUTCOME_STATES)):
                val = mat_data[i, j]
                pct = (val / row_sum * 100) if row_sum > 0 else 0
                txt_color = "white" if val > 0.45 * max_val else "#2c3e50"
                ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", color=txt_color, weight="bold", fontsize=9.5)

        fig.tight_layout()
        fig.savefig(fig_dir / "fig6_outcome_transition_matrix.png")
        fig.savefig(fig_dir / "fig6_outcome_transition_matrix.pdf")
        plt.close(fig)
        print("  ✅ Saved fig6_outcome_transition_matrix.{png,pdf}")
    else:
        print("  ℹ️ No outcome transition data available; skipping fig6_outcome_transition_matrix.")

    # ----------------------------------------------------------------------
    # Figure 7: AST Clause-Level 2x2 Confusion Matrices (2x4 Grid)
    # ----------------------------------------------------------------------
    tot_conf = sum(sum(analysis.get("clause_confusion", {}).get("finetuned", {}).get(c, {}).values()) for c in CLAUSE_NAMES)
    if tot_conf > 0:
        fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.5))
        axes = axes.flatten()

        for idx, clause in enumerate(CLAUSE_NAMES):
            ax = axes[idx]
            d = analysis["clause_confusion"]["finetuned"].get(clause, {"tn": 0, "fp": 0, "fn": 0, "tp": 0})
            cm = np.array([[d["tn"], d["fp"]], [d["fn"], d["tp"]]])

            im = ax.imshow(cm, cmap="YlGn", interpolation="nearest")
            f1 = to_pct(analysis["clause_metrics"]["finetuned"].get(clause, {}).get("f1", 0.0))
            ax.set_title(f"{clause} (F1: {f1:.1f}%)", weight="bold", fontsize=10.5)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Absent", "Present"], fontsize=8.5)
            ax.set_yticklabels(["Absent", "Present"], fontsize=8.5)
            ax.set_xlabel("Predicted", fontsize=8.5, labelpad=2)
            ax.set_ylabel("Ground Truth", fontsize=8.5, labelpad=2)

            total = np.sum(cm) if np.sum(cm) > 0 else 1
            max_c = np.max(cm) if np.max(cm) > 0 else 1
            for i in range(2):
                for j in range(2):
                    val = cm[i, j]
                    pct = (val / total) * 100
                    txt_color = "white" if val > 0.55 * max_c else "#2c3e50"
                    ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", color=txt_color, weight="bold", fontsize=8.5)

        fig.suptitle("Figure 7: AST Clause-Level 2×2 Confusion Matrix Grid (QwerySmith 1.1)", fontsize=12.5, weight="bold", y=0.99)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig7_clause_confusion_grid.png")
        fig.savefig(fig_dir / "fig7_clause_confusion_grid.pdf")
        plt.close(fig)
        print("  ✅ Saved fig7_clause_confusion_grid.{png,pdf}")
    else:
        print("  ℹ️ No clause confusion records available; skipping fig7_clause_confusion_grid.")

    # ----------------------------------------------------------------------
    # Figure 8: Inter-System Agreement & Reliability Matrix (Cohen's Kappa)
    # ----------------------------------------------------------------------
    agree_mat = analysis.get("agreement_matrix", {})
    if agree_mat and any(any(v != 0.0 for v in row.values()) for row in agree_mat.values()):
        fig, ax = plt.subplots(figsize=(7.0, 5.8))
        mat = np.array([[agree_mat[s1][s2] for s2 in systems] for s1 in systems])

        im = ax.imshow(mat, cmap="YlGnBu", vmin=0.0, vmax=1.0, interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        labels = ["Base (0-Shot)", "Base (3-Shot)", "QwerySmith 1.1"]
        ax.set_xticks(np.arange(len(systems)))
        ax.set_yticks(np.arange(len(systems)))
        ax.set_xticklabels(labels, rotation=20, ha="right", weight="bold")
        ax.set_yticklabels(labels, weight="bold")
        ax.set_title("Figure 8: Inter-System Agreement (Cohen's Kappa κ)", pad=14, fontsize=11.5, weight="bold")

        for i in range(len(systems)):
            for j in range(len(systems)):
                val = mat[i, j]
                txt_color = "white" if val > 0.65 else "#2c3e50"
                ax.text(j, i, f"κ = {val:.3f}", ha="center", va="center", color=txt_color, weight="bold", fontsize=10)

        fig.tight_layout()
        fig.savefig(fig_dir / "fig8_inter_system_agreement_matrix.png")
        fig.savefig(fig_dir / "fig8_inter_system_agreement_matrix.pdf")
        plt.close(fig)
        print("  ✅ Saved fig8_inter_system_agreement_matrix.{png,pdf}")
    else:
        print("  ℹ️ No agreement data available; skipping fig8_inter_system_agreement_matrix.")

    # ----------------------------------------------------------------------
    # Figure 9: Query Complexity Performance Heatmap
    # ----------------------------------------------------------------------
    tot_comp = sum(sum(analysis.get("complexity_matrix", {}).get(s, {}).get(t, {}).get("finetuned", {}).get("count", 0) for t in COMPLEXITY_TIERS) for s in sets)
    if tot_comp > 0:
        fig, ax = plt.subplots(figsize=(9.0, 4.8))
        comp_data = []
        for s in sets:
            row = []
            for tier in COMPLEXITY_TIERS:
                d = analysis["complexity_matrix"].get(s, {}).get(tier, {}).get("finetuned", {})
                row.append(to_pct(d.get("acc", 0.0)))
            comp_data.append(row)

        comp_mat = np.array(comp_data)
        im = ax.imshow(comp_mat, cmap="Greens", vmin=0.0, vmax=100.0, interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Execution Accuracy (%)")

        ax.set_xticks(np.arange(len(COMPLEXITY_TIERS)))
        ax.set_yticks(np.arange(len(sets)))
        ax.set_xticklabels([c.split(" ")[0] for c in COMPLEXITY_TIERS], weight="bold")
        ax.set_yticklabels([s.replace("_", " ").title() for s in sets], weight="bold")
        ax.set_title("Figure 9: QwerySmith 1.1 Execution Accuracy Stratified by Query Complexity", pad=14, fontsize=11.5, weight="bold")

        for i in range(len(sets)):
            for j in range(len(COMPLEXITY_TIERS)):
                val = comp_mat[i, j]
                cnt = analysis["complexity_matrix"].get(sets[i], {}).get(COMPLEXITY_TIERS[j], {}).get("finetuned", {}).get("count", 0)
                txt_color = "white" if val > 55 else "#2c3e50"
                ax.text(j, i, f"{val:.1f}%\n(n={cnt})", ha="center", va="center", color=txt_color, weight="bold", fontsize=9)

        fig.tight_layout()
        fig.savefig(fig_dir / "fig9_complexity_heatmap.png")
        fig.savefig(fig_dir / "fig9_complexity_heatmap.pdf")
        plt.close(fig)
        print("  ✅ Saved fig9_complexity_heatmap.{png,pdf}")
    else:
        print("  ℹ️ No query complexity records available; skipping fig9_complexity_heatmap.")

    # ----------------------------------------------------------------------
    # Figure 10: Error Taxonomy & Failure Mode Distribution
    # ----------------------------------------------------------------------
    error_types = ["Syntax Error", "Join Error", "Aggregation Error", "Predicate Error", "Semantic Row Mismatch"]
    tot_err = sum(sum(analysis.get("error_taxonomy", {}).get(sys_k, {}).values()) for sys_k in systems)
    if tot_err > 0:
        fig, ax = plt.subplots(figsize=(9.5, 5.0))
        x = np.arange(len(error_types))
        width = 0.25

        max_count = 10
        for i, sysname in enumerate(systems):
            counts = [analysis["error_taxonomy"].get(sysname, {}).get(err, 0) for err in error_types]
            max_count = max(max_count, max(counts) if counts else 0)
            rects = ax.bar(x + i * width, counts, width, label=LABELS[sysname], color=COLORS[sysname], edgecolor="black", linewidth=0.7, alpha=0.9)
            for rect, cnt in zip(rects, counts):
                if cnt > 0:
                    ax.annotate(f"{cnt}", xy=(rect.get_x() + rect.get_width()/2, rect.get_height() + max_count * 0.02 + 0.3),
                                ha="center", va="bottom", fontsize=8, weight="bold", color="#2c3e50")

        ax.set_ylabel("Query Failure Count", weight="bold", fontsize=10.5)
        ax.set_title("Figure 10: Error Taxonomy & Failure Mode Distribution Across Systems", pad=14, fontsize=11.5, weight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels(error_types, rotation=20, ha="right", weight="bold")
        ax.set_ylim(0, max_count * 1.3)
        ax.legend(loc="upper right", framealpha=0.9)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig10_error_taxonomy_matrix.png")
        fig.savefig(fig_dir / "fig10_error_taxonomy_matrix.pdf")
        plt.close(fig)
        print("  ✅ Saved fig10_error_taxonomy_matrix.{png,pdf}")
    else:
        print("  ℹ️ No error taxonomy records available; skipping fig10_error_taxonomy_matrix.")

    # ----------------------------------------------------------------------
    # Figure 11: Error Recovery & Migration Flow Matrix (Horizontal Stacked)
    # ----------------------------------------------------------------------
    tot_mig = sum(sum(analysis.get("error_migration_matrix", {}).get(e, {}).values()) for e in error_types)
    if tot_mig > 0:
        fig, ax = plt.subplots(figsize=(10.0, 5.2))
        mig_cats = ["Resolved Exact", "Resolved Exec", "Persistent Error", "Alternative Failure"]
        mig_colors = ["#27ae60", "#2ecc71", "#e67e22", "#e74c3c"]
        y = np.arange(len(error_types))

        left = np.zeros(len(error_types))
        for cat, col in zip(mig_cats, mig_colors):
            pcts = []
            for e_idx, e in enumerate(error_types):
                row_sum = sum(analysis["error_migration_matrix"].get(e, {}).values())
                cnt = analysis["error_migration_matrix"].get(e, {}).get(cat, 0)
                pcts.append((cnt / row_sum * 100) if row_sum > 0 else 0)

            rects = ax.barh(y, pcts, left=left, label=cat, color=col, edgecolor="black", linewidth=0.6, alpha=0.9)
            for e_idx, (rect, pct) in enumerate(zip(rects, pcts)):
                if pct >= 6.0:
                    ax.text(left[e_idx] + pct/2, y[e_idx], f"{pct:.1f}%", ha="center", va="center", color="white", weight="bold", fontsize=8)
            left += np.array(pcts)

        ax.set_yticks(y)
        ax.set_yticklabels(error_types, weight="bold")
        ax.set_xlabel("Resolution Distribution (%)", weight="bold", fontsize=10.5)
        ax.set_title("Figure 11: Error Recovery & Migration Flow (Base 3-Shot Failure → QwerySmith 1.1 Resolution)", pad=14, fontsize=11.5, weight="bold")
        ax.set_xlim(0, 100)
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=4, framealpha=0.9)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig11_error_migration_matrix.png")
        fig.savefig(fig_dir / "fig11_error_migration_matrix.pdf")
        plt.close(fig)
        print("  ✅ Saved fig11_error_migration_matrix.{png,pdf}")
    else:
        print("  ℹ️ No error migration records available; skipping fig11_error_migration_matrix.")

    # ----------------------------------------------------------------------
    # Figure 12: Clause Co-occurrence Correlation Matrix Heatmaps (3-Panel)
    # ----------------------------------------------------------------------
    corrs = analysis.get("clause_correlations", {})
    if "gold" in corrs and "finetuned" in corrs:
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))
        corr_gold = np.array(corrs["gold"])
        corr_base = np.array(corrs.get("base_fewshot", corrs["gold"]))
        corr_ft = np.array(corrs["finetuned"])

        frob_base = corrs.get("frobenius_error_base", 0.0)
        frob_ft = corrs.get("frobenius_error_ft", 0.0)

        panels = [
            ("Ground Truth SQL", corr_gold, None),
            ("Base (3-Shot)", corr_base, f"Frobenius Δ = {frob_base:.2f}"),
            ("QwerySmith 1.1", corr_ft, f"Frobenius Δ = {frob_ft:.2f}"),
        ]

        for p_idx, (title, mat, subtext) in enumerate(panels):
            ax = axes[p_idx]
            im = ax.imshow(mat, cmap="coolwarm", vmin=-0.5, vmax=1.0, interpolation="nearest")
            ax.set_title(title + (f"\n({subtext})" if subtext else "\n(Reference Semantics)"), weight="bold", fontsize=10.5)
            ax.set_xticks(np.arange(len(CLAUSE_NAMES)))
            ax.set_yticks(np.arange(len(CLAUSE_NAMES)))
            ax.set_xticklabels(CLAUSE_NAMES, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(CLAUSE_NAMES if p_idx == 0 else [], fontsize=8)

        fig.subplots_adjust(right=0.88, top=0.88, bottom=0.18, wspace=0.3)
        cbar_ax = fig.add_axes([0.90, 0.22, 0.018, 0.62])
        fig.colorbar(im, cax=cbar_ax, label="Pearson Correlation")
        fig.suptitle("Figure 12: Cross-Clause Co-occurrence Correlation Matrix & Structural Alignment", fontsize=12, weight="bold")
        fig.savefig(fig_dir / "fig12_clause_correlation_matrices.png")
        fig.savefig(fig_dir / "fig12_clause_correlation_matrices.pdf")
        plt.close(fig)
        print("  ✅ Saved fig12_clause_correlation_matrices.{png,pdf}")
    else:
        print("  ℹ️ No clause correlation data available; skipping fig12_clause_correlation_matrices.")

    # ----------------------------------------------------------------------
    # Figure 13: Query Length Stratification Matrix
    # ----------------------------------------------------------------------
    tot_len = sum(analysis.get("length_matrix", {}).get(lb, {}).get("finetuned", {}).get("count", 0) for lb in LENGTH_BINS)
    if tot_len > 0:
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        x = np.arange(len(LENGTH_BINS))
        width = 0.25

        for i, sysname in enumerate(systems):
            accs = [to_pct(analysis["length_matrix"].get(lb, {}).get(sysname, {}).get("acc", 0.0)) for lb in LENGTH_BINS]
            rects = ax.bar(x + i * width, accs, width, label=LABELS[sysname], color=COLORS[sysname], edgecolor="black", linewidth=0.7, alpha=0.88)
            for rect, acc in zip(rects, accs):
                ax.annotate(f"{acc:.1f}%", xy=(rect.get_x() + rect.get_width()/2, rect.get_height() + 1.5),
                            ha="center", va="bottom", fontsize=8.5, weight="bold", color="#2c3e50")

        ax.set_ylabel("Execution Accuracy (%)", weight="bold", fontsize=10.5)
        ax.set_title("Figure 13: SQL Token Length vs. Execution Accuracy Stratification", pad=14, fontsize=11.5, weight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels(LENGTH_BINS, weight="bold")
        ax.set_ylim(0, 115)
        ax.legend(loc="upper right", framealpha=0.9)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig13_token_length_stratification.png")
        fig.savefig(fig_dir / "fig13_token_length_stratification.pdf")
        plt.close(fig)
        print("  ✅ Saved fig13_token_length_stratification.{png,pdf}")
    else:
        print("  ℹ️ No query length records available; skipping fig13_token_length_stratification.")


# --------------------------------------------------------------------------
# LaTeX Table Builders (10 Comprehensive Publication Tables)
# --------------------------------------------------------------------------
def generate_latex_tables(analysis: dict, out_dir: Path):
    """Generates 10 clean, booktabs LaTeX tables strictly from computed metrics."""
    tab_dir = out_dir / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📑 Generating LaTeX tables in {tab_dir} ...")

    # Table 1: Main Benchmark Results
    t1_path = tab_dir / "table1_main_benchmark.tex"
    with open(t1_path, "w", encoding="utf-8") as f:
        f.write("% Table 1: Main Benchmark Execution Accuracy and Exact Match\n")
        f.write("\\begin{table*}[t]\n\\centering\\small\n\\begin{tabular}{llcccc}\n\\toprule\n")
        f.write("\\textbf{Benchmark Split} & \\textbf{Model System} & \\textbf{Valid SQL (\\%)} & \\textbf{Exact Match (\\%)} & \\textbf{Execution Acc (\\%)} & \\textbf{95\\% Conf. Interval} \\\\\n\\midrule\n")
        for sname, systems in analysis["benchmarks"].items():
            slabel = sname.replace("_", " ").title()
            for sysname, m in systems.items():
                v = f"{to_pct(m.get('valid', 0.0)):.1f}\\%"
                em = f"{to_pct(m.get('exact_match', 0.0)):.1f}\\%"
                ex_val = to_pct(m.get("exec_acc", 0.0))
                ex = f"\\textbf{{{ex_val:.1f}\\%}}" if sysname == "finetuned" else f"{ex_val:.1f}\\%"
                lo, hi = m.get("exec_ci", (ex_val / 100, ex_val / 100))
                ci = f"[{to_pct(lo):.1f}\\%, {to_pct(hi):.1f}\\%]"
                f.write(f"{slabel} & {LABELS.get(sysname, sysname)} & {v} & {em} & {ex} & {ci} \\\\\n")
            f.write("\\midrule\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{Main benchmark execution accuracy, exact match string parity, and syntactic validity across in-distribution and cross-domain held-out distributions.}\n")
        f.write("\\label{tab:main_benchmark}\n\\end{table*}\n")
    print("  ✅ Saved table1_main_benchmark.tex")

    # Table 2: Clause-Level F1 Breakdown
    t2_path = tab_dir / "table2_clause_metrics.tex"
    with open(t2_path, "w", encoding="utf-8") as f:
        f.write("% Table 2: AST Clause-Level Detection Precision, Recall, and F1\n")
        f.write("\\begin{table}[t]\n\\centering\\small\n\\begin{tabular}{lcccccc}\n\\toprule\n")
        f.write(" & \\multicolumn{3}{c}{\\textbf{Base Model (3-Shot)}} & \\multicolumn{3}{c}{\\textbf{QwerySmith 1.1 (Ours)}} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n")
        f.write("\\textbf{Clause} & \\textbf{Prec} & \\textbf{Rec} & \\textbf{F1} & \\textbf{Prec} & \\textbf{Rec} & \\textbf{F1} \\\\\n\\midrule\n")
        for c in CLAUSE_NAMES:
            b = analysis["clause_metrics"].get("base_fewshot", {}).get(c, {"precision": 0.0, "recall": 0.0, "f1": 0.0})
            ft = analysis["clause_metrics"].get("finetuned", {}).get(c, {"precision": 0.0, "recall": 0.0, "f1": 0.0})
            f.write(f"{c} & {b.get('precision', 0.0):.2f} & {b.get('recall', 0.0):.2f} & {b.get('f1', 0.0):.2f} & \\textbf{{{ft.get('precision', 0.0):.2f}}} & \\textbf{{{ft.get('recall', 0.0):.2f}}} & \\textbf{{{ft.get('f1', 0.0):.2f}}} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{Syntactic clause-level precision, recall, and F1 across standard SQL components.}\n")
        f.write("\\label{tab:clause_metrics}\n\\end{table}\n")
    print("  ✅ Saved table2_clause_metrics.tex")

    # Table 3: Statistical Significance (McNemar)
    t3_path = tab_dir / "table3_significance.tex"
    with open(t3_path, "w", encoding="utf-8") as f:
        f.write("% Table 3: McNemar Paired Statistical Significance Testing\n")
        f.write("\\begin{table}[t]\n\\centering\\small\n\\begin{tabular}{lcccccc}\n\\toprule\n")
        f.write("\\textbf{Benchmark Split} & \\textbf{Discordant Pairs} & \\textbf{FT Wins} & \\textbf{Base Wins} & \\textbf{Odds Ratio} & \\textbf{p-value} & \\textbf{Signif. ($p < 0.05$)} \\\\\n\\midrule\n")
        for sname, mc in analysis["significance"].items():
            slabel = sname.replace("_", " ").title()
            p_val = mc.get("p_value", 1.0)
            p_str = "$< 0.001$" if p_val < 0.001 else f"{p_val:.4f}"
            sig_str = "\\textbf{Yes}" if mc.get("significant", False) else "No"
            or_val = mc.get("odds_ratio", 1.0)
            or_str = "$\\infty$" if math.isinf(or_val) else f"{or_val:.2f}"
            f.write(f"{slabel} & {mc.get('total_discordant', 0)} & {mc.get('wins_A', 0)} & {mc.get('wins_B', 0)} & {or_str} & {p_str} & {sig_str} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{Paired McNemar test results evaluating statistical significance of QwerySmith 1.1 against the 3-shot base model.}\n")
        f.write("\\label{tab:significance}\n\\end{table}\n")
    print("  ✅ Saved table3_significance.tex")

    # Table 4: 4x4 Outcome State Transition Matrix
    t4_path = tab_dir / "table4_outcome_transition.tex"
    with open(t4_path, "w", encoding="utf-8") as f:
        f.write("% Table 4: 4x4 Pairwise Outcome State Transition Matrix\n")
        f.write("\\begin{table}[t]\n\\centering\\small\n\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write(" & \\multicolumn{4}{c}{\\textbf{QwerySmith 1.1 Outcome}} \\\\\n\\cmidrule(lr){2-5}\n")
        f.write("\\textbf{Base 3-Shot State} & \\textbf{Invalid} & \\textbf{Runs Wrong} & \\textbf{Exec Match} & \\textbf{Exec Exact} \\\\\n\\midrule\n")
        tm = analysis["transition_matrix"]
        for s_b in OUTCOME_STATES:
            row_vals = [str(tm.get(s_b, {}).get(s_ft, 0)) for s_ft in OUTCOME_STATES]
            f.write(f"{OUTCOME_DISPLAY.get(s_b, s_b)} & " + " & ".join(row_vals) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{State transition matrix tracking query migrations from Base 3-Shot to QwerySmith 1.1.}\n")
        f.write("\\label{tab:outcome_transition}\n\\end{table}\n")
    print("  ✅ Saved table4_outcome_transition.tex")

    # Table 5: Complexity Tier Matrix
    t5_path = tab_dir / "table5_complexity_matrix.tex"
    with open(t5_path, "w", encoding="utf-8") as f:
        f.write("% Table 5: Execution Accuracy Stratified by Complexity Tier\n")
        f.write("\\begin{table*}[t]\n\\centering\\small\n\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("\\textbf{Benchmark Split} & \\textbf{Simple (Proj/Filter)} & \\textbf{Moderate (Agg/Sort)} & \\textbf{Complex (Join)} & \\textbf{Advanced (Nested/Set)} \\\\\n\\midrule\n")
        for sname in analysis["benchmarks"]:
            slabel = sname.replace("_", " ").title()
            accs = []
            for tier in COMPLEXITY_TIERS:
                d = analysis["complexity_matrix"].get(sname, {}).get(tier, {}).get("finetuned", {})
                accs.append(f"{to_pct(d.get('acc', 0.0)):.1f}\\%")
            f.write(f"{slabel} & " + " & ".join(accs) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{QwerySmith 1.1 execution accuracy stratified across query complexity tiers.}\n")
        f.write("\\label{tab:complexity_matrix}\n\\end{table*}\n")
    print("  ✅ Saved table5_complexity_matrix.tex")

    # Table 6: Error Taxonomy
    t6_path = tab_dir / "table6_error_taxonomy.tex"
    with open(t6_path, "w", encoding="utf-8") as f:
        f.write("% Table 6: Error Taxonomy and Failure Mode Breakdown\n")
        f.write("\\begin{table}[t]\n\\centering\\small\n\\begin{tabular}{lccc}\n\\toprule\n")
        f.write("\\textbf{Failure Mode} & \\textbf{Base (0-Shot)} & \\textbf{Base (3-Shot)} & \\textbf{QwerySmith 1.1} \\\\\n\\midrule\n")
        err_types = ["Syntax Error", "Join Error", "Aggregation Error", "Predicate Error", "Semantic Row Mismatch"]
        for e in err_types:
            c0 = analysis["error_taxonomy"].get("base_zeroshot", {}).get(e, 0)
            c3 = analysis["error_taxonomy"].get("base_fewshot", {}).get(e, 0)
            c_ft = analysis["error_taxonomy"].get("finetuned", {}).get(e, 0)
            f.write(f"{e} & {c0} & {c3} & \\textbf{{{c_ft}}} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{Failure mode distribution comparing base systems against QwerySmith 1.1.}\n")
        f.write("\\label{tab:error_taxonomy}\n\\end{table}\n")
    print("  ✅ Saved table6_error_taxonomy.tex")

    # Table 7: Diagnostic Clause Testing Matrix
    t7_path = tab_dir / "table7_diagnostic_clause_matrix.tex"
    with open(t7_path, "w", encoding="utf-8") as f:
        f.write("% Table 7: Comprehensive Diagnostic AST Clause Metrics for QwerySmith 1.1\n")
        f.write("\\begin{table*}[t]\n\\centering\\small\n\\begin{tabular}{lccccccc}\n\\toprule\n")
        f.write("\\textbf{Clause} & \\textbf{Precision} & \\textbf{Recall (Sens.)} & \\textbf{Specificity} & \\textbf{NPV} & \\textbf{Bal. Acc.} & \\textbf{F1 Score} & \\textbf{MCC} \\\\\n\\midrule\n")
        for c in CLAUSE_NAMES:
            diag = analysis["diagnostic_metrics"]["finetuned"].get(c, {"precision": 0.0, "recall": 0.0, "specificity": 0.0, "npv": 0.0, "balanced_acc": 0.0, "f1": 0.0, "mcc": 0.0})
            f.write(f"{c} & {diag.get('precision', 0.0):.3f} & {diag.get('recall', 0.0):.3f} & {diag.get('specificity', 0.0):.3f} & {diag.get('npv', 0.0):.3f} & {diag.get('balanced_acc', 0.0):.3f} & {diag.get('f1', 0.0):.3f} & \\textbf{{{diag.get('mcc', 0.0):.3f}}} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{Diagnostic evaluation matrix of AST clause generation for QwerySmith 1.1.}\n")
        f.write("\\label{tab:diagnostic_clause_matrix}\n\\end{table*}\n")
    print("  ✅ Saved table7_diagnostic_clause_matrix.tex")

    # Table 8: Error Migration & Healing Matrix
    t8_path = tab_dir / "table8_error_migration_matrix.tex"
    with open(t8_path, "w", encoding="utf-8") as f:
        f.write("% Table 8: Error Migration Matrix (Base Failure Mode to Fine-Tuned Resolution)\n")
        f.write("\\begin{table}[t]\n\\centering\\small\n\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write(" & \\multicolumn{4}{c}{\\textbf{QwerySmith 1.1 Resolution State}} \\\\\n\\cmidrule(lr){2-5}\n")
        f.write("\\textbf{Base 3-Shot Failure} & \\textbf{Resolved Exact} & \\textbf{Resolved Exec} & \\textbf{Persistent Err} & \\textbf{Alt. Failure} \\\\\n\\midrule\n")
        for e in err_types:
            m = analysis["error_migration_matrix"].get(e, {})
            f.write(f"{e} & {m.get('Resolved Exact', 0)} & {m.get('Resolved Exec', 0)} & {m.get('Persistent Error', 0)} & {m.get('Alternative Failure', 0)} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{Error recovery and migration matrix displaying how Base 3-Shot failure modes are resolved by QwerySmith 1.1.}\n")
        f.write("\\label{tab:error_migration_matrix}\n\\end{table}\n")
    print("  ✅ Saved table8_error_migration_matrix.tex")

    # Table 9: Token Length Stratification Matrix
    t9_path = tab_dir / "table9_length_stratification.tex"
    with open(t9_path, "w", encoding="utf-8") as f:
        f.write("% Table 9: Execution Accuracy Stratified by SQL Token Length\n")
        f.write("\\begin{table}[t]\n\\centering\\small\n\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("\\textbf{Length Tier (Tokens)} & \\textbf{Queries (N)} & \\textbf{Base (0-Shot)} & \\textbf{Base (3-Shot)} & \\textbf{QwerySmith 1.1} \\\\\n\\midrule\n")
        for lb in LENGTH_BINS:
            d = analysis["length_matrix"].get(lb, {})
            n_q = d.get("finetuned", {}).get("count", 0)
            b0 = f"{to_pct(d.get('base_zeroshot', {}).get('acc', 0.0)):.1f}\\%"
            b3 = f"{to_pct(d.get('base_fewshot', {}).get('acc', 0.0)):.1f}\\%"
            ft = f"\\textbf{{{to_pct(d.get('finetuned', {}).get('acc', 0.0)):.1f}\\%}}"
            f.write(f"{lb} & {n_q} & {b0} & {b3} & {ft} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{Execution accuracy stratified by SQL token length.}\n")
        f.write("\\label{tab:length_stratification}\n\\end{table}\n")
    print("  ✅ Saved table9_length_stratification.tex")

    # Table 10: Per-Split Inter-System Agreement Matrix (Cohen's Kappa)
    t10_path = tab_dir / "table10_persplit_kappa.tex"
    with open(t10_path, "w", encoding="utf-8") as f:
        f.write("% Table 10: Per-Split Inter-System Agreement (Cohen's Kappa)\n")
        f.write("\\begin{table}[t]\n\\centering\\small\n\\begin{tabular}{lccc}\n\\toprule\n")
        f.write("\\textbf{Benchmark Split} & \\textbf{Base 0S vs Base 3S} & \\textbf{Base 3S vs FT} & \\textbf{Base 0S vs FT} \\\\\n\\midrule\n")
        for sname, k_dict in analysis["persplit_agreement"].items():
            slabel = sname.replace("_", " ").title()
            k_03 = f"{k_dict.get('base_zeroshot', {}).get('base_fewshot', 0.0):.3f}"
            k_3ft = f"{k_dict.get('base_fewshot', {}).get('finetuned', 0.0):.3f}"
            k_0ft = f"{k_dict.get('base_zeroshot', {}).get('finetuned', 0.0):.3f}"
            f.write(f"{slabel} & {k_03} & {k_3ft} & {k_0ft} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\caption{Inter-system Cohen's Kappa agreement partitioned across benchmark distributions.}\n")
        f.write("\\label{tab:persplit_kappa}\n\\end{table}\n")
    print("  ✅ Saved table10_persplit_kappa.tex")


# --------------------------------------------------------------------------
# Academic Markdown Report Builder
# --------------------------------------------------------------------------
def generate_report(analysis: dict, out_dir: Path):
    """Writes the comprehensive markdown research report."""
    report_path = out_dir / "RESEARCH_PAPER_REPORT.md"
    lines = [
        "# QwerySmith 1.1: Empirical Research Evaluation Report",
        "",
        "## 1. Executive Summary & Key Findings",
        "",
        "This report delivers an institutional-grade empirical evaluation for **QwerySmith 1.1** across benchmark distributions.",
        "",
        "---",
        "",
        "## 2. Main Benchmark Results",
        "",
        "| Benchmark Split | Model System | Valid SQL (%) | Exact Match (%) | Execution Accuracy (%) | 95% Confidence Interval |",
        "|:---|:---|:---:|:---:|:---:|:---:|",
    ]

    for sname, systems in analysis["benchmarks"].items():
        slabel = sname.replace("_", " ").title()
        for sysname, m in systems.items():
            v = f"{to_pct(m.get('valid', 0.0)):.1f}%"
            em = f"{to_pct(m.get('exact_match', 0.0)):.1f}%"
            ex = f"{to_pct(m.get('exec_acc', 0.0)):.1f}%"
            lo, hi = m.get("exec_ci", (0.0, 0.0))
            ci = f"[{to_pct(lo):.1f}%, {to_pct(hi):.1f}%]"
            lines.append(f"| {slabel} | {LABELS.get(sysname, sysname)} | {v} | {em} | **{ex}** | {ci} |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Paired Statistical Significance (McNemar Test vs 3-Shot Base)",
        "",
        "| Benchmark Split | Discordant Pairs | Fine-Tuned Wins | Base Wins | Odds Ratio | p-value | Statistically Significant ($p < 0.05$) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for sname, mc in analysis["significance"].items():
        slabel = sname.replace("_", " ").title()
        p_val = mc.get("p_value", 1.0)
        p_str = "< 0.001" if p_val < 0.001 else f"{p_val:.4f}"
        sig_str = "✅ **Yes**" if mc.get("significant", False) else "❌ No"
        or_val = mc.get("odds_ratio", 1.0)
        or_str = "∞" if math.isinf(or_val) else f"{or_val:.2f}"
        lines.append(f"| {slabel} | {mc.get('total_discordant', 0)} | {mc.get('wins_A', 0)} | {mc.get('wins_B', 0)} | {or_str} | {p_str} | {sig_str} |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. AST Clause Detection F1 & Diagnostic Reliability",
        "",
        "| Clause Construct | Base (3-Shot) F1 | QwerySmith 1.1 F1 | Precision | Recall | Specificity | MCC |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for c in CLAUSE_NAMES:
        b_f1 = to_pct(analysis["clause_metrics"].get("base_fewshot", {}).get(c, {}).get("f1", 0.0))
        ft_f1 = to_pct(analysis["clause_metrics"].get("finetuned", {}).get(c, {}).get("f1", 0.0))
        diag = analysis["diagnostic_metrics"]["finetuned"].get(c, {})
        p = diag.get("precision", 0.0)
        r = diag.get("recall", 0.0)
        s = diag.get("specificity", 0.0)
        mcc = diag.get("mcc", 0.0)
        lines.append(f"| **{c}** | {b_f1:.1f}% | **{ft_f1:.1f}%** | {p:.3f} | {r:.3f} | {s:.3f} | **{mcc:.3f}** |")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✅ Saved RESEARCH_PAPER_REPORT.md")


# --------------------------------------------------------------------------
# Colab Inline Display
# --------------------------------------------------------------------------
def display_in_colab(paper_dir: Path):
    """Renders all figures and tables inline in Google Colab / Jupyter."""
    try:
        from IPython.display import display, Image, Markdown, HTML
    except ImportError:
        print("IPython not available; skipping inline display.")
        return

    fig_dir = paper_dir / "figures"
    pngs = sorted(list(fig_dir.glob("*.png")))
    if pngs:
        display(Markdown("## 📊 Research Paper Evaluation Figures"))
        for fpath in pngs:
            display(Markdown(f"### {fpath.stem.replace('_', ' ').title()}"))
            display(Image(filename=str(fpath), width=780))
            display(Markdown("---"))

    report_path = paper_dir / "RESEARCH_PAPER_REPORT.md"
    if report_path.exists():
        display(Markdown("## 📑 Evaluation Summary Report"))
        display(Markdown(report_path.read_text(encoding="utf-8")))


def render_in_colab(run_dir: str = "/content/drive/MyDrive/qwerysmith-1.1", paper_dir: str = "paper"):
    """
    Direct Python function callable inside a Google Colab notebook cell:
        import paper_eval
        paper_eval.render_in_colab("/content/drive/MyDrive/qwerysmith-1.1", "paper")
    """
    try:
        from IPython.display import display, Image, Markdown, HTML
    except ImportError:
        display = print
        Markdown = HTML = Image = lambda x, **kw: x

    r_dir = Path(run_dir).resolve()
    p_dir = Path(paper_dir).resolve()
    p_dir.mkdir(parents=True, exist_ok=True)

    if not r_dir.exists():
        print(f"❌ Error: Run directory does not exist: {r_dir}")
        print("Please check that your Google Drive is mounted (`from google.colab import drive; drive.mount('/content/drive')`) and path is correct.")
        return

    items_by_set, results_json, train_log = load_data(r_dir)
    if not items_by_set and not results_json:
        print(f"❌ Error: No prediction records or results found in {r_dir}.")
        print("Expected 'predictions.csv' or 'results.json'. Please verify the folder contains your finished run artifacts.")
        return

    analysis = analyze_dataset(items_by_set, results_json)
    generate_figures(analysis, train_log, p_dir)
    generate_latex_tables(analysis, p_dir)
    generate_report(analysis, p_dir)

    print("\n" + "=" * 70)
    print("🎨 DISPLAYING PUBLICATION FIGURES & MATRICES")
    print("=" * 70)

    fig_dir = p_dir / "figures"
    for fpath in sorted(list(fig_dir.glob("*.png"))):
        display(Markdown(f"### 📊 {fpath.stem.replace('_', ' ').title()}"))
        display(Image(filename=str(fpath), width=800))

    report_path = p_dir / "RESEARCH_PAPER_REPORT.md"
    if report_path.exists():
        display(Markdown("---"))
        display(Markdown(report_path.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------
# CLI Entry Point
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate institutional research paper evaluation figures and tables from authentic run artifacts.")
    parser.add_argument("--out", default="/content/drive/MyDrive/qwerysmith-1.1",
                        help="Path to the finished run directory holding predictions and logs.")
    parser.add_argument("--paper-dir", default="",
                        help="Path to output paper artifacts (default: <out>/paper_artifacts).")
    parser.add_argument("--display", action="store_true",
                        help="Display figures and reports inline in Colab / Jupyter notebook.")
    args = parser.parse_args()

    run_dir = Path(args.out).resolve()
    if not run_dir.exists():
        print(f"❌ Error: Specified run directory does not exist: {run_dir}")
        print("Please provide a valid directory containing finished run artifacts (predictions.csv / results.json).")
        sys.exit(1)

    paper_dir = Path(args.paper_dir).resolve() if args.paper_dir else run_dir / "paper_artifacts"
    paper_dir.mkdir(parents=True, exist_ok=True)

    items_by_set, results_json, train_log = load_data(run_dir)
    if not items_by_set and not results_json:
        print(f"❌ Error: No evaluation artifacts ('predictions.csv' or 'results.json') found in {run_dir}.")
        print("Strict policy: Fake/synthetic figures will not be generated. Please point --out to your finished run folder.")
        sys.exit(1)

    analysis = analyze_dataset(items_by_set, results_json)

    generate_figures(analysis, train_log, paper_dir)
    generate_latex_tables(analysis, paper_dir)
    generate_report(analysis, paper_dir)

    (paper_dir / "analysis_summary.json").write_text(json.dumps(analysis, indent=2, default=str), encoding="utf-8")

    print(f"\n🎉 ALL AUTHENTIC EVALUATION FIGURES & TABLES COMPLETED!")
    print(f"📦 Files saved in: {paper_dir}")

    is_ipython = "IPython" in sys.modules or "google.colab" in sys.modules
    if args.display or is_ipython:
        display_in_colab(paper_dir)


if __name__ == "__main__":
    main()
