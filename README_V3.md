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
uv run pytest tests_v3          # 86 tests
```

Adapters: SQLite (dev default) and Postgres (canonical) behind one
interface; the eval matrix is dialect-agnostic.

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