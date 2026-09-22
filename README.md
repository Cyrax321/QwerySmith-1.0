# QwerySmith v3.0 - Reusable Text-to-SQL Evaluation & Fine-Tuning Harness

[![tests](https://github.com/Cyrax321/QwerySmith-1.0/actions/workflows/tests.yml/badge.svg)](https://github.com/Cyrax321/QwerySmith-1.0/actions/workflows/tests.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

Turns any relational database with history into a question-answering system
that answers with citations, and measures small fine-tuned models against
larger ones - on identical questions and identical retrieval.

**This repo = the harness + the QwerySmith model family.** Weights live on
Hugging Face ([Cyrax321](https://huggingface.co/Cyrax321)); the family
document - every release, recipe, and measurement - is
[MODELS.md](MODELS.md).

**Status:** harness complete, lint-clean, 110 tests green, CI on every
push. Publish path live (Hugging Face, token in env/Colab secret).
Awaiting Olist raw data for the first full dataset run - no result numbers
are claimed here; results land in `runs/<dataset>/` as reports generated
from measured output only, and the gate verdict (whatever it turns out to
be) is stated by code, not prose.

## Quickstart

```bash
# 1. clone + env
git clone https://github.com/Cyrax321/QwerySmith-1.0.git && cd QwerySmith-1.0
uv sync --group dev --extra postgres

# 2. prove the harness on the bundled toy shape (no data download needed)
uv run pytest tests_v3 -q                         # 110 green
uv run pytest tests_v3/test_reusability.py -q      # the config-only proof

# 3. first real run: drop the 9 Olist CSVs into datasets/olist/raw/, then
uv run python -m qwery_smith ingest  olist
uv run python -m qwery_smith profile olist         # prints the frozen holdout cutoff
```

The harness is dataset-agnostic: the only place a dataset's specifics live
is `datasets/<name>/config.yaml` (+ its question file). The full runbook is
at the bottom of this README.

## The design constraint

The v1.1 evaluation showed fine-tuning teaches style, not facts. So in v3:
**facts come from retrieval; fine-tuning is for behaviour only** - output
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
uv run python -m qwery_smith publish  olist --hf-user <you>   # adapters -> Hugging Face, full cards
```

## Metrics (what the numbers mean)

| Metric | Definition |
|---|---|
| **EX (execution accuracy)** | % of scorable questions whose generated SQL runs in the sandbox and returns the expected row set under canonicalization. **Uncited or invalidly-cited answers count as wrong, even if the rows are right.** |
| **Scorable N** | Held-out questions minus any whose gold SQL fails validation (each exclusion logged; target zero). |
| **Agreement** | Per question: share of the 5 consistency runs matching the majority correctness label. |
| **Flip rate** | Per question: fraction of adjacent run-pairs whose correctness changes. Gate requires the candidate be no worse than the frontier here. |
| **McNemar (exact)** | Paired test on discordant question pairs vs baseline and frontier; row 2 represented by its median-EX seed, per-seed p-values also reported. |
| **Refusal rate** | Share of questions answered with `REFUSAL:` - refusal on an answerable question scores wrong, but the rate itself is reported as calibrated-refusal behaviour. |

Canonicalization (§6.4): rows compared as sorted bags, column order by name,
floats to 2dp, NFC + casefold + whitespace-collapsed strings, NULL distinct
from empty; row order matters only for `ORDER BY ... LIMIT` gold.

## What the harness enforces (not asserts)

| Guarantee | Mechanism |
|---|---|
| Held-out data never trains | Clamped-shadow DB: a question is `heldout` iff its gold result changes when the time window is removed. Same primitive re-checks every `train_ok` question at validate time. Tested against subqueries, `NOT EXISTS`, `HAVING` shifts. |
| Uncited answers are wrong | Citation parser + pack-membership verification in the scorer; correct-rows-but-uncited scores `missing_citation`. |
| Identical questions & retrieval | Evidence packs frozen per question (SHA-256, gzip), loaded read-only by every system. |
| Consistency is real, not pinned | Each of the 5 consistency runs sends a distinct seed to the driver and the seed is recorded in the raw output - flip rate can't be silently zero. |
| Question-set identity | `validate` writes the versioned manifest: seed, counts, licence, DB fingerprint, frozen cutoff - drift between it and the questions is detectable. |
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
a "system" is just a callable from a prompt to raw output. Identical
questions, byte-identical frozen evidence packs, identical sandbox and
scorer - the only variable between row 1 and row 2 is the adapter weights
(the v1.1 confound, closed). Row 2 runs 3 seeds: the report folds them per
plan §7.1 (mean EX ± range; median-EX seed carries the McNemar row).

**Gate (pre-registered, frozen):** row 2 passes iff within 5 EX points of
row 4 AND no worse flip rate. On FAIL, row 3 vs row 4 is reported on the
same measures. Any amendment requires a written change before any run
exists.

## Hardware (plan §8.3)

| Workload | Machine | Notes |
|---|---|---|
| Prep stages (ingest→triples) | any CPU | minutes |
| Row 1-2 inference (8B) | T4 16 GB | 4-bit; LoRA hot-swapped per seed |
| QLoRA training ×3 seeds | T4 16 GB | ~40-60 min/seed (Colab) |
| Row 3 (30B-A3B AWQ) | L4 24 GB | ~17 GB weights; sequential with 8B, never co-resident |
| Frontier row | API | public data only; model version pinned in the manifest |

Every run writes `train_record.json` / `__summary.json` with GPU, VRAM and
package versions - "state what ran on what" is a captured artifact, not a
claim.

## Datasets

- `datasets/olist/` - Brazilian e-commerce, 9 tables (Kaggle; place the CSVs
  in `raw/` - the one manual step). Holdout: last 6 months of orders, cutoff
  computed mechanically at profile time and recorded in the manifest.
  Systems matrix in `systems.yaml`: 8B base / 8B+adapter (3 seeds) /
  30B-A3B AWQ / frontier (public data only).
- `datasets/online_retail_ii/` - UCI Online Retail II, single flat table,
  grouped two-sheet load (export the Excel sheets as
  `raw/Year 2009-2010.csv` and `raw/Year 2010-2011.csv` - names match
  `table_map`). Same pipeline, config changes only; the reusability test
  proves the shape on a synthetic twin before real data arrives.

## Adding a dataset (config only - no code changes)

1. `datasets/<name>/config.yaml` - name, licence, source URL, holdout rule
   (`column: "table.date_col"`, `months: N`), datasource block
   (dialect, URI, csv_dir, `table_map` - use a **list value** to append
   multiple CSVs into one table), explicit column types, PKs/FKs,
   authoring keywords.
2. Drop raw CSVs in `datasets/<name>/raw/`.
3. `author <name> --n 30` → review → `questions_v1.jsonl`.
4. Copy `datasets/olist/systems.yaml`, point the endpoints at your servers.
5. Run the pipeline; `test_reusability.py` is the template for proving it.

If any step needs a `qwery_smith/` code edit, that's a harness bug - fix it
in the harness, never in the config.

## Development

```bash
uv sync --group dev --extra postgres
uv run pytest tests_v3          # 110 tests
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
# 4. train the 3 seed adapters - one command, pinned config, hardware captured
python -m qwery_smith train olist --seeds 1,2,3 --execute
#    -> runs/olist/train_<ts>/adapters/adapter_seed{1,2,3} + train_record.json each
```

**Colab L4 (`notebooks/eval_servers.ipynb`):**

```bash
# 5. serve the matrix (vLLM): 8B base + LoRA hot-swap on :8000, 30B-AWQ on :8001
# 6. run eval - 3 passes for the candidate row (one per served adapter):
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
`report` reads only captured output - nothing is recomputed from memory.

## Secrets & security

- **Tokens never live in the repo.** Local: `.env` (gitignored, auto-loaded;
  real environment variables win). Colab: the secret manager (`HF_TOKEN`,
  `FRONTIER_API_KEY`) - read via `userdata.get`, never pasted into cells.
- **Model-generated SQL executes sandboxed**: read-only role,
  `statement_timeout` (30 s), sqlglot AST guard (SELECT-only; blocks
  CTE-hidden mutations, `SELECT INTO`, stacked statements) before any
  execution. Guarded against comment-obfuscation and string-literal
  false-positives by tests.
- **Eval isolation**: gold SQL / expected rows are physically separate from
  the retrieval path - the retriever sees question text only.

## Repository layout

```
qwery_smith/        dataset-agnostic harness - 11 CLI stages, no Olist
                    string in source. Core modules: schema_loader, profiler,
                    authoring, clamp (holdout enforcement), retrieval, triples,
                    training, scoring, evaluate, report, publish, systems.
datasets/           configs + question sets - the ONLY dataset specifics
                    (olist: 9 tables; online_retail_ii: grouped 2-file load)
runs/               per-run artifacts: adapters + train records, raw outputs,
                    summaries, report.md, failure folders
tests_v3/           110 tests: guard bypasses, clamp window-dependence,
                    scoring vs R reference values, CLI loops, reusability
                    proof, publish card content
notebooks/          t4_train.ipynb (prepare→train→publish→zip)
                    eval_servers.ipynb (serve→eval passes→report→publish)
.github/            CI: tests + reusability proof on every push/PR
legacy/             QwerySmith 1.x, preserved verbatim
PLAN.md             the design document (gate fixed before any run)
MODELS.md           the QwerySmith model family: releases, recipes, measurements
CHANGELOG.md        what shipped, per release line
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `no config found at datasets/<name>/config.yaml` | run from repo root, or pass `--root` |
| validate fails on `expected_rows_drift` for every question | DB was rebuilt but questions hold old hashes - regenerate (`author --force`) or re-fix hashes from the current DB |
| `--execute requires CUDA` | you're on CPU; `train` without `--execute` prepares configs, Colab trains |
| Colab cell dies in unsloth install | version drift - pin per the notebook comment, or `%pip install -q unsloth` alone then restart runtime |
| vLLM server never healthy | check `/content/vllm_*.log` tail printed by the cell; most common: VRAM (30B needs the 8B server stopped first) |
| pack integrity failure at eval | packs were rebuilt with a different index - delete `prepared/packs/` and re-run `retrieve` for ALL systems (identical-retrieval rule) |
| publish: 401 | HF token lacks `repo.write`, or wasn't set as `HF_TOKEN` |

## Provenance

Fine-tune recipe and evaluation discipline follow the lessons of the
QwerySmith 1.0/1.1 review: single-variable comparisons, 3 seeds, scorable
denominators, paired tests, and failure folders with one-line reasons.
See `PLAN.md` for the full design document, `CHANGELOG.md` for what
shipped, and `notebooks/t4_train.ipynb` / `notebooks/eval_servers.ipynb`
for the operational execution path (T4 training, L4 eval).