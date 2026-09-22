# QwerySmith v3.0 — Reusable Text-to-SQL Evaluation & Fine-Tuning Harness

Turns any relational database with history into a question-answering system
that answers with citations, and measures small fine-tuned models against
larger ones — on identical questions and identical retrieval.

**Status:** harness complete and tested. Awaiting Olist raw data for the
first full dataset run. No result numbers are claimed here; results land in
`runs/<dataset>/` with a report generated from measured output only.

## The design constraint

The v1.1 evaluation showed fine-tuning teaches style, not facts. So in v3:
**facts come from retrieval; fine-tuning is for behaviour only** — output
contract, terminology, decision rules, and calibrated refusal when the
retrieved evidence doesn't contain the answer. Every system in the
comparison sees byte-identical evidence packs.

## Pipeline (one command per stage; `all` chains them)

```bash
uv run python -m qwery_smith author   olist --n 100   # draft questions, mechanical splits
uv run python -m qwery_smith ingest   olist          # raw CSVs -> Postgres/SQLite
uv run python -m qwery_smith profile  olist          # row counts, date ranges, holdout cutoff
uv run python -m qwery_smith validate olist          # scorable denominator + leak checks
uv run python -m qwery_smith retrieve olist          # freeze evidence packs (question text only)
uv run python -m qwery_smith triples  olist          # RAFT triples from train_ok questions
uv run python -m qwery_smith train    olist --seeds 1,2,3   # QLoRA config + T4 execution
uv run python -m qwery_smith eval     olist          # 4-system matrix, identical questions
uv run python -m qwery_smith report   olist          # results table + gate + failure folders
```

## What the harness enforces (not asserts)

| Guarantee | Mechanism |
|---|---|
| Held-out data never trains | Clamped-shadow DB: a question is `heldout` iff its gold result changes when the time window is removed. Same primitive re-checks every `train_ok` question at validate time. Tested against subqueries, `NOT EXISTS`, `HAVING` shifts. |
| Uncited answers are wrong | Citation parser + pack-membership verification in the scorer; correct-rows-but-uncited scores `missing_citation`. |
| Identical questions & retrieval | Evidence packs frozen per question (SHA-256, gzip), loaded read-only by every system. |
| SQL safety | sqlglot AST guard (SELECT-only; blocks CTE-hidden mutations, `SELECT INTO`, stacked statements) + read-only role + statement timeout. |
| One result of record | Reports render from captured run artifacts (`*__summary.json`, `headline_results.json`), never recompute from memory. |
| Same gate everywhere | Pre-registered: row 2 passes iff within 5 EX points of frontier AND no worse on flip rate. Same table format for every dataset. |
| Reusable on a new DB | `tests_v3/test_reusability.py` runs the full pipeline on a differently-shaped dataset (single flat table, grouped multi-file load, no FK graph) from config alone. |

## Eval matrix (defined in `datasets/<name>/systems.yaml`)

| # | System | Role |
|---|---|---|
| 1 | Qwen3-8B + retrieval | baseline |
| 2 | Qwen3-8B + QLoRA adapter (3 seeds) | candidate |
| 3 | Qwen3-30B-A3B-Instruct-2507 AWQ + retrieval | on-prem alternative |
| 4 | Frontier model + retrieval (public data only) | reference |

All four run through one code path (`qwery_smith.evaluate.run_system`);
a "system" is just a callable from a prompt to raw output.

## Datasets

- `datasets/olist/` — Brazilian e-commerce, 9 tables. Place the Kaggle CSVs
  in `raw/` (the one manual step). Holdout: last 6 months of orders, cutoff
  computed mechanically at profile time.
- `datasets/online_retail_ii/` — UCI Online Retail II, single flat table,
  grouped two-sheet load. Same pipeline, config changes only.

## Development

```bash
uv sync --group dev --extra postgres
uv run pytest tests_v3          # 91 tests
```

Adapters: SQLite (dev default) and Postgres (canonical) behind one
interface; the eval matrix is dialect-agnostic.

## Runbook: from raw data to the gate (the full operational sequence)

**Local (CPU, any machine):**

```bash
# 0. one-time: place the 9 Olist CSVs in datasets/olist/raw/ (Kaggle creds)
# 1. draft the 100 questions with mechanical split tagging
uv run python -m qwery_smith author olist --n 100
# 2. HUMAN REVIEW: rewrite phrasing, verify gold SQL by eye, set source='human',
#    promote prepared/questions_draft.jsonl -> datasets/olist/questions_v1.jsonl
# 3. build everything checkable
uv run python -m qwery_smith ingest olist
uv run python -m qwery_smith profile olist        # records the frozen cutoff
uv run python -m qwery_smith validate olist       # scorable denominator + leak checks
uv run python -m qwery_smith retrieve olist        # freeze evidence packs
uv run python -m qwery_smith triples olist         # RAFT triples (train_ok only)
```

**Colab T4 (`notebooks/t4_train.ipynb`):**

```bash
# 4. train the 3 seed adapters — one command, pinned config, hardware captured
python -m qwery_smith train olist --seeds 1,2,3 --execute
#    -> runs/olist/train_<ts>/adapters/adapter_seed{1,2,3} + train_record.json each
```

**Colab L4 (`notebooks/eval_servers.ipynb`):**

```bash
# 5. serve the matrix (vLLM): 8B base + LoRA hot-swap on :8000, 30B-AWQ on :8001
# 6. run eval — 3 passes for the candidate row (one per served adapter):
uv run python -m qwery_smith eval olist --roles baseline,onprem,reference,candidate_seed --tag seed1
#    ...restart server with adapter_seed2, repeat with --tag seed2, etc.
# 7. aggregate the seed series + gate (mean EX, median-EX seed for McNemar):
uv run python -m qwery_smith report olist --run runs/olist
#    -> report.md (fixed table + gate verdict) + failures/<system>/ per wrong answer
```

**Publish (any machine with the HF token):**

```bash
# 8. push adapters to Hugging Face with full model cards
#    local: token lives in .env (auto-loaded, gitignored)
#    Colab: token in the secret manager as HF_TOKEN
uv run python -m qwery_smith publish olist --hf-user Cyrax321                    # per-seed repos
uv run python -m qwery_smith publish olist --hf-user Cyrax321 --canonical-seed 2 \
    --gate-pass <verdict>                 # canonical QwerySmith-2.0 (median-EX seed)
uv run python -m qwery_smith publish olist --hf-user Cyrax321 --dry-run        # preview cards
```

Per-seed repos `QwerySmith-2.0-seed{1,2,3}` carry: adapter weights, pinned
recipe, dataset + mechanical holdout rule, loss figure, hardware manifest,
pipeline diagram, limitations. The canonical repo additionally embeds the
measured results table and states the gate verdict plainly (a FAIL card
reads "published for provenance, not as a recommended model").

Every step writes its artifacts under `runs/` or `datasets/<name>/prepared/`;
`report` reads only captured output — nothing is recomputed from memory.

## Repository layout

```
qwery_smith/        dataset-agnostic harness (no Olist string in source)
datasets/           configs + question sets — the ONLY dataset specifics
runs/               per-run artifacts: raw outputs, summaries, failure folders
tests_v3/           unit + integration + reusability proof
notebooks/          T4/Colab training cells (pinned-config execution)
```

## Provenance

Fine-tune recipe and evaluation discipline follow the lessons of the
QwerySmith 1.0/1.1 review: single-variable comparisons, 3 seeds, scorable
denominators, paired tests, and failure folders with one-line reasons.
See `PLAN.md` for the full design document.