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

