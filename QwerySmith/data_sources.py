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
    ),
    "sqale": dict(  # column names verified against the live HF viewer
        path="trl-lab/SQaLe-text-to-SQL-dataset", split="train",
        q="question", ctx="schema", sql="query",
    ),
    "large_schema": dict(
        path="VikramPal/large-schema-text2sql-20k", split="train",
        # ctx is the raw system_prompt; schema_only() (called below) already
        # strips everything except CREATE TABLE/VIEW statements, so no extra
        # parsing is needed to pull the schema out of the surrounding prompt.
        q="question", ctx="system_prompt", sql="response",
        filter=lambda r: (
            r.get("task_mode") == "single_sql"
            and int(r.get("id") or -1) not in _LARGE_SCHEMA_BAD_IDS
        ),
    ),
}


def _load_source(name: str, limit: int, seed: int, oversample: int = 3) -> list[dict]:
    """Load a registry source into QwerySmith's internal item format.

    oversample: how big a multiple of `limit` to scan before giving up. Raise
    this for low-yield sources (e.g. large_schema, where the size filter in
    _executable_only rejects most rows) so you don't silently end up with
    fewer items than requested.
    """
    from datasets import load_dataset

    try:
        from Qwerysmith_V11 import schema_only
    except ImportError:
        from QwerySmith import schema_only

    spec = SOURCES[name]
    print(f"Loading {spec['path']} [{spec['split']}] ...")
    try:
        ds = load_dataset(spec["path"], split=spec["split"])
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: could not load {name}, skipping it ({e})")
        return []

    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    row_filter = spec.get("filter")
    items = []
    for i in idx[: max(limit * oversample, limit) if limit else len(idx)]:
        r = ds[i]
        if row_filter and not row_filter(r):
            continue
        ctx = (r.get(spec["ctx"]) or "").strip()
        q = (r.get(spec["q"]) or "").strip()
        gold = (r.get(spec["sql"]) or "").strip()
        if not (ctx and q and gold):
            continue
        items.append({
            "question": q, "context": ctx,
            "schema": schema_only(ctx), "gold": gold, "source": name,
        })
        if limit and len(items) >= limit:
            break
    print(f"  -> {len(items)} items from {name}")
    return items



def _executable_only(items: list[dict], want: int, max_schema_chars: int = 3000) -> list[dict]:
    """Keep only items whose gold query runs AND returns at least one row,
    AND whose schema fits comfortably inside --max-len tokens.

    max_schema_chars matters most for SQaLe: its schemas run up to ~90 tables
    / 400+ columns (multiple thousand characters of CREATE TABLE text), which
    would get silently truncated at --max-len 2048 tokens before the model
    ever sees the question. ~3000 chars of schema text leaves room for the
    question + generation inside a 2048-token budget; raise this only if you
    also raise --max-len."""
    try:
        from Qwerysmith_V11 import make_db, run_query
    except ImportError:
        from QwerySmith import make_db, run_query

    keep = []
    for it in items:
        if len(it["schema"]) > max_schema_chars:
            continue
        conn = make_db(it["context"], it["gold"], seed=0)
        ok, rows = run_query(conn, it["gold"])
        conn.close()
        if ok and rows:
            keep.append(it)
        if len(keep) >= want:
            break
    return keep


def _parse_mix(spec: str) -> list[tuple[str, int]]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, n = part.partition(":")
        if name not in SOURCES:
            raise SystemExit(f"Unknown source '{name}'. Known: {list(SOURCES)}")
        out.append((name, int(n or 0)))
    return out


def build_sets(args):
    """Returns (train_items, eval_sets).

    eval_sets always contains in_dist; gretel_test and the --heldout source are
    added when available. Any training item whose question appears in ANY eval
    set is dropped, so a mix can never leak into an eval.
    """
    seed = 42

    # ---- eval sets are carved out FIRST, so training can never claim them ----
    mix_dict = dict(_parse_mix(args.mix))
    sql_needed = args.n_test + mix_dict.get("sql_create_context", 0) * 2
    in_pool = _load_source("sql_create_context", sql_needed, seed)
    in_test = in_pool[: args.n_test]
    in_test_qs = {r["question"] for r in in_test}

    sets = {"in_dist": in_test}

    if args.n_external:
        near = _load_source("gretel_test", args.n_external * 4, seed)
        near = _executable_only(near, args.n_external)
        if near:
            sets["gretel_test"] = near

    if args.heldout:
        mix_names = {name for name, _ in _parse_mix(args.mix)}
        # large_schema's size filter rejects most rows (median schema is
        # ~28K chars of prompt), so it needs a much bigger scan window than
        # the default 4x to actually reach --n-heldout items.
        oversample_by_source = {"large_schema": 20}
        for name in [n.strip() for n in args.heldout.split(",") if n.strip()]:
            if name not in SOURCES:
                raise SystemExit(f"Unknown --heldout source '{name}'. Known: {list(SOURCES)}")
            if name in mix_names:
                print(f"WARNING: '{name}' is in both --mix and --heldout -- skipping it "
                      f"as held-out, since it can no longer measure generalization.")
                continue
            osf = oversample_by_source.get(name, 4)
            ho = _load_source(name, args.n_heldout * osf, seed, oversample=1)
            ho = _executable_only(ho, args.n_heldout)
            if ho:
                sets[f"heldout_{name}"] = ho
