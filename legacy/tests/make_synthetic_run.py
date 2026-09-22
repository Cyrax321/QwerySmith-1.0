#!/usr/bin/env python3
"""
Build a synthetic QwerySmith 1.0 run directory so the evaluation suite can be tested end
to end without a GPU, without downloading a dataset and without a trained model.

It writes exactly the artefacts a real run leaves behind, plus the optional sidecar files
that the GPU stages normally produce:

    <out>/config.json               arguments of a hypothetical run
    <out>/train_log.json            loss / lr / grad-norm history
    <out>/preds/*__*.json           cached generations for the three systems
    <out>/eval/items.json           the evaluation items (question/context/schema/gold)
    <out>/eval/confidence.json      token log-probabilities (correlated with correctness)
    <out>/eval/self_consistency.json
    <out>/eval/robustness.json

Usage
-----
    python tests/make_synthetic_run.py --out /tmp/qw_synth/run
    python qwerysmith_eval.py --stage figures --out /tmp/qw_synth/run
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

SEED = 42
SYSTEMS = ["base_zeroshot", "base_fewshot", "finetuned"]

DBS = {
    "shop": {
        "tables": {
            "customers": [("id", "INTEGER"), ("name", "TEXT"), ("country", "TEXT")],
            "orders": [("id", "INTEGER"), ("customer_id", "INTEGER"), ("amount", "REAL"), ("order_date", "TEXT")],
            "products": [("id", "INTEGER"), ("name", "TEXT"), ("price", "REAL"), ("stock", "INTEGER")],
        },
        "rows": {
            "customers": [(1, "Ada", "UK"), (2, "Grace", "US"), (3, "Alan", "UK"), (4, "Linus", "FI")],
            "orders": [
                (1, 1, 120.5, "2023-01-05"),
                (2, 1, 80.0, "2023-02-11"),
                (3, 2, 250.25, "2023-02-20"),
                (4, 3, 15.0, "2023-03-01"),
                (5, 4, 410.0, "2023-04-18"),
            ],
            "products": [(1, "Lamp", 25.5, 12), (2, "Chair", 99.0, 4), (3, "Desk", 210.0, 2)],
        },
    },
    "hr": {
        "tables": {
            "employees": [
                ("id", "INTEGER"),
                ("name", "TEXT"),
                ("dept_id", "INTEGER"),
                ("salary", "INTEGER"),
                ("hired", "TEXT"),
            ],
            "departments": [("id", "INTEGER"), ("name", "TEXT"), ("city", "TEXT")],
            "projects": [("id", "INTEGER"), ("dept_id", "INTEGER"), ("budget", "REAL")],
        },
        "rows": {
            "employees": [
                (1, "Ada", 1, 90000, "2019-04-01"),
                (2, "Grace", 2, 120000, "2018-06-15"),
                (3, "Alan", 1, 85000, "2021-09-30"),
                (4, "Linus", 3, 70000, "2022-01-10"),
                (5, "Barbara", 2, 130000, "2017-03-03"),
            ],
            "departments": [(1, "Engineering", "London"), (2, "Research", "Cambridge"), (3, "Support", "Leeds")],
            "projects": [(1, 1, 50000.0), (2, 1, 20000.0), (3, 2, 90000.0), (4, 3, 5000.0)],
        },
    },
}

# @@SYNTH_CASES@@

# @@PATTERNS@@

# @@BUILDERS@@

# @@SYNTH_MAIN@@
