# The QwerySmith Family

Text-to-SQL models that answer from retrieved evidence with verifiable
citations, plus the harness that trains and evaluates them. Open weights
on Hugging Face; one-command reproducibility from raw data.

The family principle, established by the 1.1 evaluation: **facts come from
retrieval; fine-tuning shapes behaviour.** Every QwerySmith model is a
behaviour tune of a strong open base: it learns the output contract,
terminology, decision rules, and calibrated refusal. It never memorizes
data. This page documents each release, what it is, and exactly how it was
produced and measured.

## Family tree

```text
                        QWEN3 OPEN BASES
                              |
              +---------------+----------------+
              |                                |
     QwerySmith 1.x (legacy)           QwerySmith 2.0 (current)
     Qwen3-4B curriculum FT           Qwen3-8B QLoRA behaviour tune
     style, not facts                 contract + citations + refusal
              |                                |
     "fine-tuning taught style,        retrieval-grounded answers,
      not facts" (the finding          3-seed release, pre-registered
      that defines 2.0)                gate, measured-only claims
```

| Release | Base | Method | What it learns | Status |
|---|---|---|---|---|
| QwerySmith 1.0 | Qwen3-4B | curriculum SFT (50/50 synthetic + enterprise SQL) | SQL style | superseded; preserved in `legacy/` |
| QwerySmith 1.1 | Qwen3-4B | curriculum SFT, tuned recipe | SQL style | superseded; its external evaluation (below base on unseen data) is the finding that defines 2.0 |
| **QwerySmith 2.0** | Qwen3-8B | QLoRA r16 a32, NF4, 3 seeds | output contract, `[table:row_id]` citations, refusal when evidence is absent | current release |
| QwerySmith 2.0 seed 1/2/3 | Qwen3-8B | same recipe, seed variance | same | per-seed provenance repos |

Weights: [huggingface.co/Cyrax321](https://huggingface.co/Cyrax321)
(`QwerySmith-2.0` canonical + `QwerySmith-2.0-seed{1,2,3}`; 1.0/1.1 under
`Cyrax321/QwerySmith-1.0` and `-1.1`).

## QwerySmith 2.0 in detail

### What it does

Given a database schema and retrieved evidence rows, the model emits:

```
SQL:
<one SELECT statement>
ANSWER:
<answer text; every factual claim carries [table:primary_key] citations>
```

or, when the evidence does not contain the answer:

```
REFUSAL:
<one sentence naming what is missing>
```

Generated SQL is meant to execute in a read-only sandbox (timeout, AST
SELECT-only guard). Citations are machine-verifiable against the evidence
pack; uncited answers score as wrong in the harness, by design.

### Training recipe (fully pinned, reproducible)

| Parameter | Value |
|---|---|
| Base | Qwen/Qwen3-8B (Apache-2.0) |
| Method | QLoRA: 4-bit NF4, double quantization |
| LoRA | r=16, alpha=32, dropout=0.05, all-linear targets |
| Optimizer | lr 1e-4, cosine schedule, warmup 3%, 3 epochs |
| Batch | 1 per device x 16 grad accumulation, 4096 ctx |
| Decoding | non-thinking mode, T=0.7, top_p=0.8, top_k=20 |
| Data | RAFT-style triples: ~70% grounded / 15% refusal / 15% schema-only, hard-negative distractors |
| Seeds | 1, 2, 3 (identical recipe; spread reported) |

### How it was produced

```text
raw CSVs -> Postgres/SQLite -> schema DDL + FK graph + fingerprint
         -> questions (gold SQL + expected rows, mechanical holdout split)
         -> frozen evidence packs (SHA-256; question text only, no gold leakage)
         -> RAFT triples (train_ok questions only; ledger for corpus audit)
         -> QLoRA x 3 seeds (T4; hardware + hashes + loss curve captured)
         -> eval matrix: base / FT / 30B / frontier, identical questions + packs
         -> report + gate verdict + failure folders (measured artifacts only)
         -> Hugging Face release (card embeds the measured table)
```

Every stage is one CLI command (`python -m qwery_smith <stage> olist`);
`all` chains them. The dataset, its questions, and the frozen holdout
cutoff are versioned in a manifest; the harness re-derives everything from
raw data on a clean machine.

### Evaluation

Four systems, identical questions, byte-identical evidence packs:

| # | System | Role |
|---|---|---|
| 1 | Qwen3-8B + retrieval | baseline |
| 2 | Qwen3-8B + QwerySmith 2.0 adapter (3 seeds) | candidate |
| 3 | Qwen3-30B-A3B-Instruct-2507 AWQ + retrieval | on-prem alternative |
| 4 | Frontier + retrieval | reference (public data only) |

Gate (pre-registered before any run): the candidate passes iff within 5
execution-accuracy points of the frontier AND no worse on flip rate. A
failed gate is stated on the model card as "published for provenance, not
as a recommended model". Results tables live in `runs/<dataset>/report.md`
and are embedded in the HF cards; no number is hand-written.

First dataset: Olist Brazilian e-commerce (9 tables, Postgres/SQLite).
Second dataset (reusability proof, config changes only): UCI Online
Retail II.

### Honest scope

- Behaviour tune for the harness prompt contract; other formats are out of
  distribution.
- Correctness depends on retrieval, same as every system in the matrix;
  the model refuses rather than guessing when evidence is thin.
- Evaluated on Olist + Online Retail II. Generalization beyond relational
  QA is not claimed.

## Model card contents (every release)

Each HF repo ships: adapter weights + tokenizer, the pinned training
config, `train_record.json` (GPU, VRAM, package versions, adapter file
hashes, loss curve), the pipeline diagram, loss and results figures, the
measured evaluation table, limitations, and the full lineage. See
`python -m qwery_smith publish --help`.

## Citation

```bibtex
@software{qwerysmith2,
  title  = {QwerySmith 2.0: Retrieval-Grounded Text-to-SQL Behaviour Tuning},
  author = {Anandhu (Cyrax321)},
  year   = {2026},
  url    = {https://github.com/Cyrax321/QwerySmith-1.0},
  note   = {QLoRA behaviour tune of Qwen3-8B; trained and evaluated by the QwerySmith v3 harness}
}
```