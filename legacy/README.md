# Legacy: QwerySmith 1.x (the fine-tuning assignment)

This directory contains the **completed 1.0/1.1 assignment** — the multi-source
curriculum fine-tune of Qwen3-4B, its evaluation, and the paper — exactly as
reviewed. It is kept intact as history and provenance for the v1.1 findings
that motivate v3.

**Do not run anything from here for the current assignment.** The live
harness is `../qwery_smith/` (see `../README.md`).

Contents:

- `QwerySmith/` — 1.0 and 1.1 fine-tuning pipelines
- `harness/` — v1.x agentic harness (sandbox, schema linking, self-healing,
  consensus voting). Several concepts were extracted and hardened into v3:
  the AST read-only guard became `qwery_smith/adapters/guard.py`; the
  evaluator loop became `qwery_smith/scoring.py`.
- `paper/`, `paper_eval.py`, `qwerysmith_eval.py` — the 1.1 paper and its
  leak-proof held-out evaluation (in-distribution vs external results).
- `tests/` — v1.x test suite (34 passing at review time).

## Context: what v1.x established (and what it didn't)

The 1.1 evaluation showed the fine-tuned model **did not beat the base model
on data it hadn't seen** — fine-tuning taught a style, not facts. That result
is the design constraint for v3: facts come from retrieval; fine-tuning is
only for behaviour. The v1.0 in-distribution-only reporting and the
README-vs-paper claims mismatch were also review findings — addressed in v3
by held-out-only headline metrics and reports rendered from captured
artifacts.

See `../PLAN.md` §1.1 for the full mapping of review findings to v3
mechanisms.