#!/usr/bin/env python3
"""
data_sources.py -- drop-in data mixing for QwerySmith v1.1

Put this file next to QwerySmith.py. It replaces load_in_dist() / load_external()
with a source registry so you can mix training data and keep several eval sets,
including one that never appears in training.

WHAT TO CHANGE IN QwerySmith.py
-------------------------------
1. At the top, after the other imports:

       from data_sources import build_sets

2. Delete load_in_dist() and load_external() (or just stop calling them).

3. In main(), replace:

       train_items, in_test = load_in_dist(args)
       sets = {"in_dist": in_test}
       if args.n_external:
           ext = load_external(args)
           if ext:
               sets["external"] = ext

   with:

       train_items, sets = build_sets(args)

4. Add these to parse_args():

       p.add_argument("--mix", default="sql_create_context:5000,gretel:5000",
                      help="comma-separated source:count pairs (see SOURCES)")
       p.add_argument("--heldout", default="",
                      help="registry name of a source NEVER used for training, "
                           "e.g. sqale. Empty = skip.")
       p.add_argument("--n-heldout", type=int, default=300)
       p.add_argument("--save-steps", type=int, default=0,
                      help="checkpoint every N steps so you can pick the best one")

   and change the defaults for the recipe ablation (run C):

       p.add_argument("--lr", type=float, default=1e-4)      # was 2e-4
       p.add_argument("--dropout", type=float, default=0.05) # new

5. In stage_train(), three edits:

       lora_dropout=args.dropout,                 # was 0
       ...
       save_strategy="steps" if args.save_steps else "no",
       save_steps=args.save_steps or 500,
       save_total_limit=6,

6. In the --smoke branch of parse_args(), also shrink the mix:

       args.mix = "sql_create_context:250,gretel:250"

RUN PLAN
--------
  # run B -- isolates the data mix, everything else identical to v1.0
  !python QwerySmith.py --out runs/v11-B --mix sql_create_context:5000,gretel:5000 \
      --lr 2e-4 --dropout 0 --heldout sqale,large_schema

  # run C -- adds the gentler recipe on top of B's data
  !python QwerySmith.py --out runs/v11-C --mix sql_create_context:5000,gretel:5000 \
      --lr 1e-4 --dropout 0.05 --save-steps 100 --heldout sqale,large_schema

Your v1.0 predictions stay valid as run A -- do not regenerate them.

A NOTE ON THE HELD-OUT SET
--------------------------
Once `gretel` is in --mix, gretel_test is no longer an *external* set: it is a
held-out split of a distribution you trained on. Useful, but not a
generalization number. --heldout takes a comma-separated list of registry
names that must NEVER also appear in --mix; each becomes its own eval set
(heldout_<name>), so you can compare several independent checks at once
instead of trusting a single one. Column mappings below have been checked
against each dataset's live viewer, but re-verify if a source's schema
changes upstream.

Sources currently registered:
  sql_create_context  WikiSQL+Spider derived (b-mc2). NOTE: Spider rows may
                       already be inside this dataset, so raw Spider is NOT
                       a safe held-out source once sql_create_context is in
                       --mix -- you cannot tell overlap from generalization.
  gretel / gretel_test synthetic, single distribution, ~100 domains.
  sqale                real-world schemas via SchemaPile. Schemas run large
                       (median ~91 tables); _executable_only's schema-size
                       cap means only its smaller-schema tail survives at
                       --max-len 2048.
  large_schema         VikramPal/large-schema-text2sql-20k. Median schema is
                       95 tables / ~28K prompt chars -- this dataset's whole
                       point is stress-testing schema linking at that scale,
                       which the size cap below defeats. What you'll actually
                       get held out here is its small-schema tail (a handful
                       of tables), not the large-schema behavior it's known
                       for. Only single_sql rows are used (multi_dialect_json
                       rows have JSON-object targets, not bare SQL); 7 rows
                       with known corrupted identifiers are excluded by id.
"""
from __future__ import annotations

import random

# Row ids with documented corruption (mojibake / malformed DDL) in
# VikramPal/large-schema-text2sql-20k -- see that dataset's card, "Known
# limitations" #9.
_LARGE_SCHEMA_BAD_IDS = {298, 9259, 9908, 11666, 12141, 12713, 17311}

# Every entry maps a HF dataset to the three fields we need, plus an optional
# "filter" callable(raw_row) -> bool for sources that mix task types or ship
# known-bad rows. To add a source, open it in the dataset viewer and fill in
# the real column names.
SOURCES = {
    "sql_create_context": dict(
        path="b-mc2/sql-create-context", split="train",
        q="question", ctx="context", sql="answer",
    ),
    "gretel": dict(
        path="gretelai/synthetic_text_to_sql", split="train",
        q="sql_prompt", ctx="sql_context", sql="sql",
    ),
    "gretel_test": dict(
        path="gretelai/synthetic_text_to_sql", split="test",
        q="sql_prompt", ctx="sql_context", sql="sql",
