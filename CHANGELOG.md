# Changelog

All notable changes to the QwerySmith v3 harness. Format: Keep a Changelog;
this project follows the assignment's design document (PLAN.md) sections.

## [3.0.0] — 2026-09-22

The reusable text-to-SQL evaluation and fine-tuning harness, proven on
Olist (config-only re-proven on UCI Online Retail II). See PLAN.md.

### Added
- Eleven-stage CLI: `author ingest profile validate retrieve triples train
  eval report publish all` — one command per pipeline stage, `all` chains them.
- Schema loader: Postgres + SQLite introspection → deterministic CREATE TABLE
  text + FK graph + SHA-256 fingerprint (the DDL shown to the model is a
  versioned artifact).
- Profiler: row counts, date ranges, and the mechanical holdout cutoff
  (month-floor rule, plan §4.2) — computed from data, never authored.
- Question-set format: JSONL + versioned manifest (seed, counts, licence,
  db fingerprint, frozen cutoff). `validate` is a release gate: scorable
  denominator, expected-rows drift detection, holdout-leak enforcement.
- Time-clamped shadow database (clamp.py): window-dependence decided by
  execution, not predicate injection — correct under subqueries, NOT EXISTS,
  HAVING-threshold shifts. Drives both split tagging and the leak check.
- Question authoring: schema-facts discovery + four category generators
  (per-order lookup, aggregate, multi-table join, review text) with
  validated executable gold, dedup, and mechanical split assignment.
- Retrieval: inverted-index BM25 over serialized rows; evidence packs frozen
  per question (gzip + SHA-256, tamper-detection); retrieval uses question
  text only — gold SQL is never in the retrieval path.
- RAFT triples: grounded/refusal/schema-only mix (~70/15/15), seeded
  hard-negative distractors, refusal teaching, evidence ledger for the
  training-corpus audit. Held-out questions are structurally excluded.
- QLoRA training: pinned config generator + T4 executor (`train --execute`),
  trl API-drift fallbacks, hardware manifest, adapter file hashes, full loss
  history → `train_record.json` per seed.
- Eval harness: system-agnostic runner (any callable), identical questions
  and byte-identical packs for every system, per-run distinct consistency
  seeds, raw-output capture; sandboxed execution (read-only role, timeout,
  sqlglot AST SELECT-only guard).
- Scoring: §6.4 canonicalization (NFC, casefold, 2-dp floats, NULL rules),
  precedence-ordered 9-class error taxonomy, exact McNemar (verified against
  R reference values), Wilson CIs, agreement/flip consistency metrics.
- Report: fixed table format for every dataset, pre-registered gate
  (row 2 within 5 EX points of frontier, no worse flip), 3-seed aggregation
  (mean EX ± range, median-EX seed for McNemar, per-seed p-values),
  failure folders with one-line reasons per wrong answer per system.
- Publish: per-seed HF repos + canonical release, full model cards (pinned
  recipe, dataset + holdout rule, measured results embedded, hardware,
  pipeline diagram, limitations), loss/results figures; gate verdict stated
  plainly — a FAIL card reads "published for provenance, not recommended".
- Reusability proof: `tests_v3/test_reusability.py` runs the full pipeline
  on a differently-shaped dataset (single flat table, grouped two-file
  load, no FK graph) from config alone.
- CI: harness tests + the reusability proof on every push/PR.

### Fixed
- Consistency runs originally passed a pinned seed to every request — all 5
  samples would have been identical and flip rate fake-zero. Runner now
  derives per-question, per-run-distinct seeds and records them in raw output.
- Dataset-relative SQLite URI resolution on symlinked absolute paths
  (macOS /private/var): hostless 4-slash form built and parsed correctly.
- Config loader: missing keys raise ConfigError naming the key (was a raw
  KeyError); `created_with_seed` read per the manifest schema.
- Ingest path had undefined names when invoked outside the CLI (ruff F821).

### Historical
- QwerySmith 1.x (fine-tuning assignment, paper, v1 harness) preserved
  verbatim under `legacy/` with review context. The v1.0 README's
  "production-ready / 100% success rate" claims are kept as
  `legacy/README_1.0.md` under a historical-document banner; the live root
  README states measured artifacts only.