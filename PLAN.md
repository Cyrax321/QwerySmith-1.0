# QwerySmith v3.0 - Design Document & Execution Plan
## Reusable Text-to-SQL Evaluation and Fine-Tuning Harness, proven on Olist

**Prepared by:** Anandhu · **For:** Adheesh · **Date:** 2026-09-22
**Status:** Awaiting sign-off. The gate in §7 is fixed on approval and frozen before the first evaluation run.

---

# 0. One-Page Plan (the requested "before starting" page)

**What I will measure - held-out data only** (the last six months of orders, defined as a date cutoff computed mechanically from the data, are frozen out of all training artifacts and used only for final scoring):

| Metric | Operationalization |
|---|---|
| **Execution accuracy (EX)** | Fraction of *scorable* held-out questions whose generated SQL executes in the sandbox and whose result bag equals gold under the canonicalization in §6.4. **Uncited or invalidly-cited answers count as wrong**, even if the rows are correct. |
| **Scorable denominator** | All held-out questions minus those whose gold SQL fails at validation time. Each exclusion is logged with the Postgres error. Target: zero exclusions; any exclusion blocks the run until resolved. |
| **McNemar paired tests** | Exact binomial on discordant pairs, row 2 vs row 4 (primary) and row 2 vs row 1 (ablation), computed on the headline run with per-seed breakdowns. |
| **Consistency** | 5 runs/question at fixed sampling settings, distinct sample seeds. **Agreement** = share of runs matching the majority correctness label per question. **Flip rate** = fraction of the 4 adjacent run-pairs whose correctness changes. |

**What counts as failure** (mutually exclusive, precedence-ordered): sandbox rejection or SQL exception → `execution_error`; >30 s → `timeout`; parser contract violation → `invalid_output`; executable but wrong rows → `wrong_result_{missing,extra,mismatched}_rows`; correct rows without a verifiable citation → `missing_citation` / `invalid_citation`; refusal on an answerable question → `unexpected_refusal`. **Gate failure** = row 2 >5 EX points below row 4, or strictly worse flip rate.

**The hard part:** *retrieval grounding without leakage.* Facts come from retrieval; fine-tuning teaches only behavior - format, terminology, decision rules, refusal. So the training triples must contain distractor rows plausible enough that the model learns to *filter the retrieved set rather than pattern-match*, and the evaluation retriever must be built from question text alone with gold SQL nowhere in the retrieval path. Second hardest: holdout hygiene - proving, not asserting, that no training artifact touches the held-out window. Both are addressed with enforcement machinery (§4.4, §5.2), not intentions.

---

# 1. Purpose and Design Constraint

Turn a company process with historical records into a system that answers questions from that history, correctly and with citations, on infrastructure we control. Olist is the first dataset; **the harness ships dataset-agnostic and is re-proven on UCI Online Retail II with configuration changes only.**

The v1.1 result is the design constraint: fine-tuning taught style, not facts (external eval fell below base). Therefore:

- **Facts come from retrieval.** The model is never expected to know data it was not shown in context. Every run shows every system the same retrieved evidence pack.
- **Fine-tuning is only for behavior:** the output contract (§5.3), our terminology, decision rules, and **calibrated refusal** - refusing when the retrieved context does not contain the answer, rather than hallucinating.

## 1.1 Closing the three review findings

| Review finding | Closed by |
|---|---|
| In-distribution-only reporting is misleading | All headline metrics are held-out only. The retrieval pack is the same distribution for every system, so no system gets an in-distribution advantage. In-distribution numbers, if produced, are quarantined to a clearly-labelled appendix. |
| README/paper incompatibility | Delivery README states exactly what was measured, with denominators, and nothing else. No "production-ready"/"100%" language. §9 makes the README a release-gated artifact. |
| 1.1 confounded (mixture+LR+dropout) and single-seed | Row 1 vs row 2 differ by **exactly one variable**: the adapter weights (§5.6). Row 2 runs **3 seeds**. Retrieval packs are frozen per question (§4.3), so retrieval variance cannot confound the comparison. |

*Carried-over action (from the prior review):* the 1.0 post is not published in its current form; it is either folded into the 1.1 story or amended to include the external result and the 61/200 scorable denominator.

---

# 2. System Architecture

```text
                         ┌─────────────────────────────────────────────────────┐
                         │              qwery_smith/ (Python pkg)              │
                         │   dataset-agnostic - no Olist string in source      │
                         └─────────────────────────────────────────────────────┘
  ┌──────────────┐   prepare   ┌──────────────┐  retrieve  ┌──────────────┐  train  ┌──────────────┐
  │ raw CSVs     │────────────▶│ Postgres     │───────────▶│ frozen       │────────▶│ QLoRA        │
  │ (gitignored) │             │ 16 (Docker)  │            │ evidence     │         │ adapter      │
  └──────────────┘             └──────┬───────┘            │ packs (gzip) │         └──────────────┘
                                      │ introspect         └──────┬───────┘                │
                                      ▼                           │ same pack to           ▼
                              ┌──────────────┐                    │ every system    ┌──────────────┐
                              │ schema DDL   │                    ▼                 │ adapters     │
                              │ + FK graph   │            ┌─────────────────────────┴─────────┐
                              │ + profile    │            │ eval matrix: rows 1-4             │
                              └──────────────┘            │ identical questions + retrieval   │
  ┌──────────────┐   validate                             └──────────────┬──────────────────┘
  │ questions_   │──────────────────────────────────────────────────────▶│ sandboxed exec
  │ v1.jsonl     │                                                       ▼
  └──────────────┘                                              ┌──────────────────┐
                                                                │ report.py        │
                                                                │ table · gate ·   │
                                                                │ failure folders  │
                                                                └──────────────────┘
```

**Pipeline stages** (each is a CLI verb; `all` runs the DAG top-to-bottom):
`ingest → validate → retrieve → train → eval → report`

Everything a run depends on - dataset version hash, question-set hash, pack hashes, model IDs + weight hashes, decoding config, container digests - is written to a **run manifest** (§8.2). Two runs with the same manifest are bit-comparable.

## 2.1 Reuse from the existing codebase (minimal diff, not greenfield)

The v1.x `harness/` package contains three components I will **extract and harden** rather than rewrite:

| Existing asset | Reused as | Gap to close |
|---|---|---|
| `harness/security/sandbox.py` - read-only AST guard (already blocks DDL/DML, stacked statements, mutating PRAGMAs) | Eval SQL sandbox policy (§6.3) | Port the guard from SQLite string-regex to a **sqlglot-parsed AST** so it works dialect-correctly on Postgres; add row-count guardrails |
| `harness/benchmark/evaluator.py` - per-item EX loop with latency capture | Scoring loop skeleton (§6) | Add canonicalization, citation check, consistency runner, McNemar |
| `harness/adapters/base.py` - `DatabaseAdapter` interface | DB abstraction | Add a Postgres adapter alongside the existing SQLite/DuckDB ones |

Everything else (schema loader, question format, retrieval, training, report) is new.

---

# 3. Part 1 - The Harness (five components, none dataset-specific)

### 3.1 Schema loader

- Connects over SQLAlchemy URIs: `postgresql+psycopg://…` or `sqlite:////…`. Read-only role only.
- Introspects tables/columns/types/PK/FKs via `information_schema` (PG) / `pragma_*` (SQLite) behind the adapter interface.
- Emits the **exact DDL text shown to the model**: `CREATE TABLE` stubs, canonical type aliases (e.g. `numeric(10,2) → NUMERIC`), inline PK, and `-- fk: order_items.order_id → orders.order_id` comments. Tables sorted alphabetically; output is SHA-256 hashed and stored - schema text is a versioned artifact, so prompt drift is impossible.
- Emits a **profile** (row counts, min/max per date column, distinct-count estimates per categorical column) in YAML. The profiler output is what the question author uses to assign `split` (§4.1).

### 3.2 Question-set format (one file for all datasets)

Two files per dataset, `datasets/<name>/`:

`manifest.yaml`
```yaml
name: olist
version: 1.0.0
license: "CC BY-NC-SA 4.0 (Olist, via Kaggle)"
source_url: "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"
created_with_seed: 42
db_fingerprint: "sha256:…"          # hash of DDL text + per-table row counts
holdout:
  column: orders.order_purchase_timestamp
  cutoff: "2018-03-01T00:00:00"     # computed by profiler, verified here
counts: {total: 100, heldout: 38, train_ok: 62,
         by_difficulty: {easy: 30, medium: 40, hard: 30}}
```

`questions_v1.jsonl` - one JSON object per line:
```json
{"id": "olist-0041", "question": "Total payment value by customer state for delivered orders in H1 2017.", "gold_sql": "SELECT c.customer_state, SUM(p.payment_value) ...", "expected_rows": {"sha256": "…", "n_rows": 20}, "date": "2026-09-25", "difficulty": "medium", "category": "aggregation", "split": "train_ok", "source": "human"}
```

- `expected_rows` stores a **canonicalized hash + row count**, not raw rows (keeps the file small; raw rows live in `prepared/snapshots/` for the failure folders).
- `source`: `human` or `template+human-verified`. This enables provenance auditing later.
- **Versioning contract:** any change to questions, gold SQL, or expected rows bumps `version`, and the manifest records counts + seed + licence. Eval runs pin the version; reports cite it.

**Validator** (`qwery_smith validate`) - a release gate, not a linter:
1. Executes every gold SQL in the sandbox; failure ⇒ question excluded and logged (this *is* the scorable-denominator computation).
2. Re-canonicalizes `expected_rows` and compares against the stored hash ⇒ catches DB drift.
3. **Holdout-leak check (§4.4)** for every `train_ok` question.
4. Checks split/category/difficulty counts match the manifest.

### 3.3 Training script (RAFT triples + QLoRA, one command)

`python -m qwery_smith train --dataset olist --seeds 1 2 3`

**Triple construction** from `train_ok` question + its frozen positive rows:

| Triple type | Share | Context given to model | Target |
|---|---|---|---|
| Grounded answer | ~70% | positives + *hard-negative distractors* | Contract output (§5.3) citing the supporting rows |
| Calibrated refusal | ~15% | distractors only - positives withheld | `REFUSAL: the retrieved records do not contain …` |
| Schema-only answer | ~15% | schema block, no rows (context-retrieval miss) | Grounded answer if answerable from schema semantics, else refusal (teaches the boundary) |

**Hard-negative distractors** (seeded, per-question deterministic): rows from a table in `tables_touched`, sharing ≥1 categorical value with the positives but in a disjoint time window or adjacent category - i.e. *plausible but wrong*. Distractor count ~ uniform(2-5). This is the RAFT mechanism that forces *filtering* behavior.

**Example training instance (abbreviated):**
```
SYSTEM: You answer questions about the company database using only the
        RETRIEVED EVIDENCE. Cite every claim [table:pk]. If the evidence
        does not contain the answer, say so.
SCHEMA: <DDL stubs>
QUESTION: What was the total freight for order 4e7cd1…?
RETRIEVED EVIDENCE:
  orders(order_id, customer_id, status, purchase_ts)  - 
    (4e7cd1…, c_881, delivered, 2017-06-14 09:12)          ← relevant
    (9b2aa0…, c_881, delivered, 2018-05-02 18:40)          ← distractor: right customer, wrong order/period
  order_items(order_id, item, product_id, price, freight)  - 
    (4e7cd1…, 1, p_2210, 189.90, 23.45)                    ← relevant
    (9b2aa0…, 1, p_0934, 59.00, 18.10)                     ← distractor
ASSISTANT:
  SQL:
    SELECT SUM(freight_value) FROM order_items WHERE order_id = '4e7cd1…';
  ANSWER:
    Total freight was R$23.45 [order_items:4e7cd1…, item 1].
```

**QLoRA config** (T4-safe, fully pinned):
```yaml
base_model: Qwen/Qwen3-8B            # instruct model; non-thinking mode (below)
load_in_4bit: true                   # NF4, double-quant, bf16 compute
lora: {r: 16, alpha: 32, dropout: 0.05, target: all-linear}
optim: {lr: 1.0e-4, schedule: cosine, warmup_ratio: 0.03, epochs: 3}
batch: {per_device: 1, grad_accum: 16, max_len: 4096}
packing: true
seed: <per-run>                      # data order + dropout + distractor sampling
```

**Decoding is pinned from Qwen's official cards, not improvised** (this is the kind of detail the 1.1 confound taught): Qwen3-8B supports explicit thinking/non-thinking modes, and Qwen warns **greedy decoding degrades thinking-mode output**. Therefore all systems run **non-thinking** (`enable_thinking=False`) with `T=0.7, top_p=0.8, top_k=20` (Qwen's non-thinking guidance), `max_new_tokens=1024`, `presence_penalty=0`. Headline EX uses the same settings with a fixed sampling seed (greedy is off the table per the card); consistency uses 5 distinct seeds at these settings. Row 3 (`Qwen3-30B-A3B-Instruct-2507`) is **non-thinking-only already** (the card states it emits no `<think>` blocks) and uses its card defaults `T=0.7, top_p=0.8`. So the whole matrix shares one decoding contract.

### 3.4 Evaluation harness

- **Identical questions, identical retrieval:** packs are frozen per question once (§4.3) and mmap-loaded read-only for every row of the matrix. The retriever reads `question` only; `gold_sql`/`expected_rows` are physically in a different file the eval process cannot open (enforced by a permissions split in `prepared/`).
- **Sandbox:** read-only PG role + `idle_in_transaction_session_timeout` + **sqlglot-AST SELECT-only guard** + `statement_timeout = 30s`. See §6.3.
- **Scoring:** §6.4 canonicalization → bag-equality. Uncited ⇒ wrong.
- **Consistency runner:** 5 samples/question, sample seeds `1000+q_index+k`, captures the 6 outputs (1 headline + 5) for the report.
- **All output captured raw** per run: prompt hash, full completion, latency, parsed SQL, result hash, citation list, classifier output. Nothing is recomputed at report time that wasn't captured at run time.

### 3.5 Results report

- Renders the fixed table (§7.2) + gate verdict + per-system **failure folders**: `failures/<system>/<question_id>.md` each containing question, full output, error class, and a **one-line reason** (auto-classified per §6.5, human spot-checked on a 20% sample).
- Also emits `report.json` so tables are machine-diffable across datasets.

---

# 4. Data - Olist (all nine tables, Postgres)

### 4.1 Ingest

Dockerized **Postgres 16** (`postgres:16-alpine`, digest pinned). All nine Kaggle CSVs loaded via `COPY … FROM PROGRAM` with explicit column types in the DDL (not inferred): `orders, order_items, order_payments, order_reviews, customers, sellers, products, geolocation, product_category_name_translation`. Constraints added *after* load (PKs; FKs where clean - Olist has known orphans, the loader reports them rather than failing). A `docker-compose.yml` + `make db` brings the database up identically anywhere.

### 4.2 Holdout rule - computed mechanically, frozen in manifest

```
cutoff = date_trunc('month', max(order_purchase_timestamp)) - interval '6 months'
```
The profiler computes it (Olist's max is 2018-10-17, so the cutoff lands 2018-04-01 - verified at prepare time and recorded). A question is `heldout` iff its gold SQL can read any row at/after the cutoff. **The rule is a function of data, not of my judgment**, which is what makes it defensible.

### 4.3 Frozen retrieval packs

Per question, built once at `retrieve` time from question text only (BM25 over a flattened row-text index built from the *full* dataset - the model must still filter; the pack is not a leak because it is what a production system would serve): top-K rows (K=8 default) across candidate tables + the schema block. Serialized gzipped JSON, SHA-256, stored in `prepared/packs/`. **Every system consumes the same file.** Packs regenerate only on explicit re-retrieval with a bumped index version.

### 4.4 Holdout-leak enforcement (prove, don't assert)

Three independent checks; any failure is a hard error:
1. **Expected-rows intersection:** for each `train_ok` question, re-execute gold with an `AND order_purchase_timestamp < cutoff` guard injected via sqlglot; if the guarded result ≠ stored result, the question can see the window → fail.
2. **Training-corpus audit:** the trainer writes every triple's evidence row IDs to a ledger; a post-build check asserts zero ledger rows appear in any `heldout` question's positive set.
3. **Temporal-term lint:** `train_ok` questions containing `last|recent|latest|past N months` etc. are flagged for manual re-review (window semantics are leak-prone).

---

# 5. Part 2 - Olist Evaluation

### 5.1 The 100-question set

| Category | n | Difficulty split (e/m/h) | Must exercise (named SQL/feature patterns) |
|---|---|---|---|
| Per-order lookups | 24 | 10/9/5 | equality filter on PK; multi-table enrichment of one order (items+payments+review) |
| Aggregates | 32 | 8/16/8 | `SUM/COUNT/AVG` + `GROUP BY` state/category; date-range predicates; `%` share-of-total; `HAVING` |
| 3+ table joins | 24 | 4/12/8 | star joins orders→items→products→sellers→customers; late-delivery correlation with review score (join + aggregate + condition) |
| Review text | 20 | 8/8/4 | `review_comment_message ILIKE '%…%'` with curated PT keyword sets; join text hits back to category/state aggregates; complement queries (orders with score ≤2 *without* comments) |

- Expected rows snapshotted by execution at validation; 100% of questions pass a second-author checklist (correct grain, correct filter, correct join keys, correct null handling). 
- Held-out target ≈38 questions. **Statistical power note (pre-registered):** at n=38 held-out, a McNemar exact test with ≥10 discordant pairs detects a 20-point gap at α=0.05 with ~80% power; a 5-point gap will NOT reach significance at this n. I therefore report (a) the gate as the decision rule (it does not require significance) and (b) McNemar as the evidence about *where* differences come from, with discordant-pair counts alongside p-values - I will not claim equivalence from a non-significant McNemar. If a tighter read on the 5-point margin is wanted, the honest fix is a bigger held-out set, stated here before any run.

### 5.2 Systems compared - identical questions, identical retrieval

| # | System | Role | Serving |
|---|---|---|---|
| 1 | Qwen3-8B + retrieval | baseline | vLLM, bf16, 4-bit KV cache, T4 |
| 2 | Qwen3-8B + QLoRA (§3.3), **3 seeds** | candidate | same server; adapter hot-swapped (LoRA-on-base), so rows 1/2 share weights/process |
| 3 | **Qwen3-30B-A3B-Instruct-2507** + retrieval, inference-only | on-prem alternative | vLLM, BF16 on L4/A10-24GB (30.5B→~61GB bf16 ⇒ **AWQ 4-bit ≈17GB**; pinned artifact `Qwen/Qwen3-30B-A3B-Instruct-2507-AWQ`) |
| 4 | Frontier + retrieval | reference | API, snapshot/model-version recorded; data is public so API use is sanctioned |

**Why 30B-A3B (MoE, 3.3B active) as the primary row 3:** it is the strongest *deployable-on-one-24GB-card* Qwen class, its card shows it competitive with GPT-4o-0327 on agent/coding slices and ahead of the older non-thinking 30B on every row, and inference cost is ~10× lower than dense 32B at equal quality. `Qwen3-32B` (dense, AWQ) is the sanctioned fallback if MoE quant proves unstable; the choice + artifact hash are recorded in the manifest either way.

### 5.3 Output contract (what the fine-tune buys)

```text
SQL:
<exactly one SELECT statement, no prose>
ANSWER:
<≤80 words; every factual claim carries ≥1 citation [table:primary_key]>
```
or, when evidence is insufficient:
```text
REFUSAL:
<one sentence naming what is missing from the retrieved evidence>
```
Parsed by a strict grammar; anything else ⇒ `invalid_output` ⇒ wrong. Citation verifier: each cited `[table:pk]` must appear in that question's evidence pack. This is the machine-checkable version of "uncited answers count as wrong."

### 5.4 Retrieval at eval (the single shared path)

Question → BM25 over row-text index → top-K evidence rows → rendered with the schema block into the prompt template (identical for all rows). No few-shot examples with Olist content (few-shots, 2, are synthetic and dataset-neutral - formatting demos only, recorded in the manifest).

---

# 6. Scoring Semantics (exact definitions)

### 6.3 Sandbox
`SET TRANSACTION READ ONLY; SET LOCAL statement_timeout = '30s'`, connection role has `SELECT` only (no `CONNECT`-escalation), sqlglot AST walk rejects any non-SELECT node, CTE-mutations, `FOR UPDATE`, `INTO`, set-returning foot-guns. PG error + wall-clock captured per execution.

### 6.4 Canonicalization (what "same rows" means)
`rows → sorted bag of tuples`; column order ignored (projected-column sets must match by name); numerics rounded to 2dp; strings NFC-normalized, casefolded, whitespace-collapsed; timestamps truncated to seconds; `NULL` canonical. Row order respected **only** when gold has `ORDER BY … LIMIT`. The canonicalizer is unit-tested against a 40-case fixture (intentional false-positive/false-negative traps from the v1.x eval experience).

### 6.5 Error taxonomy (precedence-ordered; first match wins)
`timeout → execution_error → invalid_output → wrong_result_{missing,extra,mismatched}_rows → missing_citation → invalid_citation → unexpected_refusal`. Refusal on *unanswerable-from-evidence* questions is scored correct on a small injected probe subset (n≈8 held-out questions whose packs deliberately omit positives) - this measures calibrated refusal without polluting the EX denominator.

---

# 7. Gate and Results Format

### 7.1 Gate - fixed on plan approval, frozen before the first run

> **Row 2 PASSES iff** `EX_heldout(row2_mean_3seeds) ≥ EX_heldout(row4) - 5.0` **AND** `flip_rate(row2_pooled) ≤ flip_rate(row4)`.
> PASS ⇒ row 2 becomes the private-data candidate. FAIL ⇒ report row 3 vs row 4 on the same measures.

Mechanics, pre-registered: row-2 EX = mean of the 3 per-seed EX values; McNemar row2-vs-row4 uses the *median-EX seed* as the representative run (per-seed p-values also reported); flip rate pooled across the 15 runs/question. **Any amendment to this paragraph requires written sign-off before any eval run exists.**

### 7.2 Results table (identical for every dataset)

| System | EX held-out (95% CI) | Scorable N | McNemar vs r1 (b/c, p) | McNemar vs r4 (b/c, p) | Agreement | Flip | Refusal% | p50 lat. (ms) | Hardware |
|---|---|---|---|---|---|---|---|---|---|
| r1 Qwen3-8B | | | - | | | | | | T4 |
| r2 Qwen3-8B-FT (mean ± range, n=3) | | | | | | | | | T4 |
| r3 Qwen3-30B-A3B-Instruct-2507 AWQ | | | | | | | | | L4/A10 |
| r4 Frontier (<pinned version>) | | | | | | | | | API |

`b/c` = discordant pair counts (row-correct-only / comparator-correct-only). CIs: Wilson. Plus: **GATE: PASS/FAIL**, the failure folders, and `report.json`.

---

# 8. Reusability (Part 3) and Reproducibility

### 8.1 Online Retail II - config-only test

`datasets/online_retail_ii/` = manifest + 30 questions + raw CSV. Schema differs structurally (single fact table, invoice-line grain, `InvoiceDate` holdout column, null `CustomerID` semantics) - chosen precisely because it stresses the dataset-agnostic claims: different date column (holdout config points elsewhere), different join topology (retrieval must handle wide-table evidence), different text field (`Description`, EN).

**Automated proof:** `tests/test_reusability.py` runs the full DAG on Online Retail II in a temp workdir where the *only* added files are the dataset dir; if any `qwery_smith/` source edit is required, the test fails and the fix lands in the harness, never the config. This test runs in CI before delivery. Same results table + gate delivered.

### 8.2 One command + run manifest

```bash
python -m qwery_smith all --dataset olist            # raw CSV → report.md + failure folders
python -m qwery_smith all --dataset online_retail_ii
```
Prereqs: Docker (Postgres) + the raw CSVs present locally (Kaggle/UCI need user credentials; this is the only manual step). Everything else - ingest, validate, retrieve, train×3 seeds, eval×4 systems, report - is the single command, idempotent and resumable at stage boundaries (`--from eval`).

`runs/<dataset>/<ts>/manifest.json`:
```json
{"dataset": {"name": "olist", "version": "1.0.0", "questions_sha256": "…", "holdout_cutoff": "2018-04-01", "packs_sha256": "…"},
 "models": {"r1": {"id": "Qwen/Qwen3-8B", "weights_sha256": "…"}, "r2": {"adapter_sha256_by_seed": {"1": "…", "2": "…", "3": "…"}}, "r3": {"id": "Qwen/Qwen3-30B-A3B-Instruct-2507-AWQ", "weights_sha256": "…"}, "r4": {"api_model": "<pinned>", "snapshot": "2026-…"}},
 "decoding": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_new_tokens": 1024, "seeds": {"headline": 42, "consistency": "1000+q+k"}},
 "hardware": {"r1_r2": "T4 16GB (Colab)", "r3": "L4 24GB", "r4": "API"},
 "containers": {"postgres": "postgres:16-alpine@sha256:…"}}
```

### 8.3 Hardware plan

| Workload | Target | Fallback | Est. time |
|---|---|---|---|
| r1 + r2 inference (8B bf16, 4-bit KV) | Colab T4 16GB | L4 | 100 q × 6 runs × ~4-8 s ≈ 1.5 h |
| r2 QLoRA train ×3 seeds (~2.2k triples) | Colab T4 | Kaggle P100 | ~40-60 min/seed |
| r3 (AWQ ≈17GB weights) | Colab L4 24GB (paid) / A10G | A100 via spot | ~45 min |
| r4 | Frontier API | - | minutes; budget cap set |
| ingest/validate/retrieve/unit tests | local CPU | - | - |

**What ran on what** is stated in the manifest and the report; quant artifacts and runtimes included. Budget note: L4 hours + API ≈ small, capped, recorded.

---

# 9. Delivery, Verification, and Milestones

**Release gates** (each blocks the next):
- **G1 data ready:** Postgres up, 9 tables, profiler green, cutoff recorded, orphans reported.
- **G2 questions ready:** 100 validated, 0 scorable exclusions, holdout-leak checks green, 100% second-author review.
- **G3 training ready:** 20-question smoke train converges; triple audit ledger clean.
- **G4 eval ready:** dry run of all four systems on 10 train_ok questions; report renders; failure folders work; canonicalizer fixtures pass.
- **G5 final:** full matrix ×3 seeds; gate verdict; failure folders; Online Retail II config-only test green; clean-env one-command rebuild verified (fresh clone on a second machine → identical table); delivery README written *from* `report.json` (numbers are templated, so the README cannot diverge from the measurement).

| # | Milestone | Exit artifact |
|---|---|---|
| M1 | Harness skeleton + Postgres + profiler | G1 green |
| M2 | Question format + validator + retrieval + packs | G2 green (first 30 questions), leak checks live |
| M3 | Training pipeline + triple audit | G3 green |
| M4 | Full 100 questions + eval harness + report | G4 green |
| M5 | Full Olist matrix | G5 table for Olist |
| M6 | Online Retail II + clean-rebuild + delivery README | G5 fully green |

**Testing:** unit (schema loader over PG+SQLite fixtures; sqlglot guard bypass attempts incl. comment-obfuscated `DROP`; canonicalizer 40-case fixture; McNemar against R reference values), integration (end-to-end on a 5-question toy dataset in CI), regression (golden outputs for 10 questions catch harness changes silently shifting scores). All tests run in the one command's `--check` mode before any GPU work.

---

# 10. Risks (engineering-honest)

| Risk | Likelihood | Mitigation |
|---|---|---|
| Held-out slice too small for the 5-pt margin (§5.1 power note) | medium | Pre-registered interpretation rule; optionally grow held-out to ~50 by re-tagging *before* training starts |
| AWQ 30B-A3B quality regression vs bf16 | low-med | 10-question sanity delta bf16-vs-AWQ recorded; if >2 pts, record both and gate on bf16 numbers with a hardware upgrade |
| T4 OOM at 4096 ctx during training | med | pack budget enforced at triple build (≤3.3k tok/instance measured); seq len 3072 fallback is a config flag, reported if used |
| Frontier API drift between snapshot and run | low | pin explicit model version string; record response `model` field per call |
| Review-text questions are ambiguously answerable (PT keyword nuance) | med | curated keyword sets frozen in the manifest; second-author answerability pass on all 20 |
| Distractors too easy/too hard → behavior tune teaches shortcuts or guesswork | **high - core risk** | 20-question ablation before full training: grounded vs no-distractor vs easy-distractor triples; pick by held-out-subset flip rate; decision recorded |

---

# 11. Judging-Criteria Map

| Judged on | Where |
|---|---|
| Held-out Olist results only | §4.2 rule, §7 headline metrics, in-distribution quarantined |
| Part 3 runs without code changes | §8.1 enforced by CI test |
| The failure folders | §3.5 + taxonomy §6.5, one-line reason per wrong answer |
| One command rebuilds everything | §8.2 + clean-env verification at G5 |
| Hardware stated | §8.3 + run manifest |

# 12. Decisions I'd like confirmed

1. **Row 3 = Qwen3-30B-A3B-Instruct-2507 (AWQ)** as primary with dense-32B fallback - ok?
2. **Frontier:** strongest pinned coding-capable frontier model with a budget cap (~$40 across both datasets). Name your constraint and I'll pin the version.
3. **Held-out size:** keep ≈38 with the pre-registered power interpretation, or grow to ≈50 before training starts (my recommendation, costs only question authoring time)?
4. **L4/A10 budget** for row 3 (~a few dollars of GPU time) - approved?

*On approval, M1-M2 start immediately: Postgres + profiler + schema loader, and the first 30 validated questions with the leak-check live.*
