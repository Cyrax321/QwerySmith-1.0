"""Publish stage: save the trained model to a Hugging Face account.

Produces, per seed:
  - adapter weights + tokenizer (huggingface_hub upload_folder)
  - MODEL_CARD.md — full: architecture, recipe, dataset, evaluation (from
    captured run artifacts only), citation format, limitations
  - figures/ — loss curve (per-seed), results table render, pipeline diagram
  - train_record.json + the pinned QLoRA config

Claims policy (v1.x review lesson): the card states exactly what was measured,
with denominators. No 'production-ready'. No in-distribution-only numbers as
headlines. If the gate FAILED, the card says so on the front.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .config import DatasetConfig


# ------------------------------------------------------------- figures -----

def _loss_curve_figure(log_history: list[dict], out_path: Path, seed: int) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    steps = [e["step"] for e in log_history if "loss" in e]
    losses = [e["loss"] for e in log_history if "loss" in e]
    lrs = [e["learning_rate"] for e in log_history if "learning_rate" in e]
    if not steps:
        return False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    ax1.plot(steps, losses, color="#2563eb", lw=1.5)
    ax1.set_xlabel("step"); ax1.set_ylabel("train loss"); ax1.set_title(f"seed {seed}: loss")
    ax1.grid(alpha=0.3)
    if lrs:
        ax2.plot([s for s, e in zip(steps, log_history, strict=False) if "learning_rate" in e], lrs, color="#dc2626", lw=1.5)
        ax2.set_xlabel("step"); ax2.set_ylabel("lr"); ax2.set_title("cosine schedule")
        ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def _results_figure(report_md: str, out_path: Path) -> bool:
    """Render the report's table lines into a PNG table."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    rows = [l for l in report_md.splitlines() if l.startswith("|")]
    if len(rows) < 3:
        return False
    header = [c.strip() for c in rows[0].strip("|").split("|")]
    body = [[c.strip() for c in r.strip("|").split("|")] for r in rows[2:]]
    fig, ax = plt.subplots(figsize=(12, 0.5 * (len(body) + 1)))
    ax.axis("off")
    table = ax.table(cellText=body, colLabels=header, loc="upper center",
                     cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


PIPELINE_DIAGRAM = """```text
                    QWERYSMITH 2.0 — ARCHITECTURE & PIPELINE
================================================================================

  RAW CSVs          PREPARE (CPU, local)                TRAIN (T4, Colab)
============      =========================            ==================
 9 Olist tables -> ingest   -> Postgres/SQLite  RAFT   triples jsonl -> QLoRA
                  profile  -> cutoff frozen    TRIPLE       |
                  validate -> leak checks       BUILDER      v
                  retrieve -> evidence packs    ~70/15/15  Qwen3-8B 4-bit
                  triples  -> training data   grounded/   + LoRA r16 a32
                              (train_ok only)  refusal/    lr 1e-4 cosine
                                               schema      3 seeds
================================================================================

  EVAL MATRIX (L4, Colab) — identical questions + identical frozen packs
===============================================================================
   row 1  Qwen3-8B base        + packs --> scorer --> EX / flip / refusal
   row 2  Qwen3-8B + adapter   + packs --> scorer --> mean of 3 seeds
   row 3  Qwen3-30B-A3B AWQ   + packs --> scorer --> on-prem alternative
   row 4  frontier (API)      + packs --> scorer --> reference
                                        |
                     McNemar + GATE (pre-registered) <- report.md + failures/

================================================================================

  INFERENCE CONTRACT (what the fine-tune teaches — behaviour only)
===============================================================================
   prompt = SYSTEM contract + SCHEMA (CREATE TABLE text)
          + QUESTION + RETRIEVED EVIDENCE [table:row_id] rows
                 |
                 v
   SQL:  <one SELECT statement>
   ANSWER: <text with [table:row_id] citations>     or   REFUSAL: <why>
                 |
                 v
   sandboxed execution (read-only role, 30s timeout, sqlglot AST guard)
   facts ALWAYS come from retrieval — the adapter never stores data
```"""


# ----------------------------------------------------------- model card ----

def _fmt_pct(x: Any) -> str:
    try:
        return f"{float(x) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def build_model_card(
    *,
    model_name: str,                      # e.g. "QwerySmith-2.0-seed1"
    base_model: str,
    dataset: str,                         # "olist"
    dataset_url: str,
    dataset_license: str,
    question_counts: dict[str, Any],
    cutoff: Optional[str],
    qlora_config: dict[str, Any],
    train_record: dict[str, Any],
    report_md: Optional[str] = None,
    gate_pass: Optional[bool] = None,
    repo_url: Optional[str] = None,
    plan_url: Optional[str] = None,
    is_canonical: bool = False,
) -> str:
    hw = train_record.get("hardware", {})
    lora = qlora_config.get("lora", {})
    optim = qlora_config.get("optim", {})
    decoding = qlora_config.get("decoding", {})
    counts = question_counts or {}

    gate_line = (
        "See the results table below."
        if gate_pass is None else
        ("**GATE: PASS** — the candidate is within the pre-registered margin of the frontier reference."
         if gate_pass else
         "**GATE: FAIL** — the candidate did NOT meet the pre-registered gate. It is published for provenance, not as a recommended model.")
    )

    card = f"""---
license: apache-2.0
base_model: {base_model}
tags:
- text-to-sql
- qwen3
- qlora
- retrieval-augmented
- citation-grounded
- olist
library_name: peft
---

# {model_name}

A **behaviour-tuned** text-to-SQL adapter: it answers questions about a
relational database from retrieved evidence, with machine-verifiable
citations, and refuses when the evidence does not contain the answer.
**Facts come from retrieval — this adapter stores no data.**

{gate_line}

## What it is / is not

- **Is:** a QLoRA adapter (PEFT) for `{base_model}` teaching the output
  contract (`SQL:` / `ANSWER:` with `[table:row_id]` citations, or
  `REFUSAL:`), schema terminology, and calibrated refusal.
- **Is not:** a knowledge store. Unmodified `{base_model}` + the same
  retrieval achieves the fact-bearing part; per the v1.1 finding,
  fine-tuning taught style, not facts — this release is built on that
  constraint.

## Training recipe (fully pinned)

| Parameter | Value |
|---|---|
| Base model | `{base_model}` |
| Method | QLoRA, 4-bit NF4, double quantization |
| LoRA | r={lora.get("r")}, alpha={lora.get("alpha")}, dropout={lora.get("dropout")}, targets={lora.get("target_modules")} |
| Optimizer | lr={optim.get("lr")}, {optim.get("schedule")} schedule, warmup {optim.get("warmup_ratio")}, {optim.get("epochs")} epochs |
| Batch | {qlora_config.get("batch", {}).get("per_device")} per device x {qlora_config.get("batch", {}).get("grad_accum")} grad accum, max_len {qlora_config.get("batch", {}).get("max_len")} |
| Decoding (pinned) | non-thinking mode, T={decoding.get("temperature")}, top_p={decoding.get("top_p")}, top_k={decoding.get("top_k") or "-"} |
| Training data | RAFT-style triples from {counts.get("total", "?")} questions ({counts.get("by_split", {}).get("train_ok", "?")} train_ok; held-out questions never trained) |
| Mix | ~70% grounded / ~15% refusal / ~15% schema-only, hard-negative distractors |
| Seed | {train_record.get("seed")} |

Training data construction, distractor sampling, and the refusal mix are
produced by the harness (`triples` stage) and are reproducible from the repo.

## Dataset

- **{dataset}** — [{dataset_url}]({dataset_url}), {dataset_license}
- {counts.get("total", "?")} questions with gold SQL + expected rows;
  difficulty-tagged; split mechanically:
  {counts.get("by_split", {}).get("train_ok", "?")} train_ok /
  {counts.get("by_split", {}).get("heldout", "?")} held-out
- **Held-out rule:** questions whose gold result changes when the last
  {("6" if not cutoff else "6")} months of data are removed are tagged
  held-out (cutoff {cutoff or "computed at profile time"}) and are never
  used in training. Enforced by a clamped-shadow-database execution check.

## Evaluation — measured artifacts only

All numbers below come from captured run artifacts
(`report.md`, `failures/`), produced by the same harness on identical
questions with byte-identical frozen evidence packs for every system.
"""
    if report_md:
        card += "\n" + report_md.rstrip() + "\n"
    else:
        card += """
*(Evaluation pending — this card is generated from the training run only.
The results table lands here after the eval matrix runs.)*
"""
    card += f"""

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{base_model}", device_map="auto")
model = PeftModel.from_pretrained(base, "{model_name}")
tok = AutoTokenizer.from_pretrained("{base_model}")

# The model expects the harness prompt contract:
#   SYSTEM contract + SCHEMA (CREATE TABLE text)
#   + QUESTION + RETRIEVED EVIDENCE rows as [table:row_id] lines
# and emits either:
#   SQL: <one SELECT>  /  ANSWER: <text with [table:row_id] citations>
#   or REFUSAL: <what is missing>
```

Execute generated SQL read-only, with a statement timeout and an AST
SELECT-only guard (the harness ships one).

## Figures

| | |
|---|---|
| `figures/loss_curve_seed{train_record.get("seed")}.png` | training loss + LR schedule |
| `figures/results_table.png` | the evaluation table, as measured |
| `figures/` + card diagram | pipeline & inference contract (below) |

{PIPELINE_DIAGRAM}

## Limitations (stated plainly)

- Evaluation is on **{dataset}** only; generalization to other databases is
  not claimed by this card.
- The adapter emits SQL for the schema it is shown; correctness depends on
  retrieval quality — the same dependency every system in the matrix has.
- Fine-tuned for the harness output contract; other prompt formats are
  out of distribution.
- Single-dataset behaviour tune; the 3-seed spread (see sibling repos)
  indicates run-to-run variance.

## Provenance

- Trained by the QwerySmith v3 harness (one-command reproducible):
  {f"[{repo_url}]({repo_url})" if repo_url else "see repository"}
  {f"— design document: [{plan_url}]({plan_url})" if plan_url else ""}
- Hardware: {hw.get("gpu", "unknown")}, {hw.get("vram_mb", "?")} MB VRAM;
  packages: {json.dumps(hw.get("packages", {}), indent=None)}
- v1.x lineage (1.0/1.1 fine-tuning assignment and its evaluation):
  see the repository's `legacy/` directory.

## License

Apache-2.0 (base model license). Dataset retains {dataset_license}.
"""
    return card


# ------------------------------------------------------------- publish -----

@dataclass
class PublishSpec:
    hf_user: str
    model_prefix: str                     # "QwerySmith-2.0"
    adapter_dir: Path
    train_record_path: Path
    qlora_config_path: Path
    repo_url: Optional[str] = None
    plan_url: Optional[str] = None
    dataset: str = "olist"
    dataset_url: str = ""
    dataset_license: str = ""
    question_counts: dict = None
    cutoff: Optional[str] = None
    report_md_path: Optional[Path] = None
    gate_pass: Optional[bool] = None


def publish_seed(spec: PublishSpec, seed: int, is_canonical: bool = False) -> dict[str, Any]:
    """Assemble the HF repo dir for one seed and upload it."""
    import shutil

    from huggingface_hub import HfApi

    token = None
    import os

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    record = json.loads(Path(spec.train_record_path).read_text())
    import yaml

    qlora_cfg = yaml.safe_load(Path(spec.qlora_config_path).read_text())

    repo_id = f"{spec.hf_user}/{spec.model_prefix}-seed{seed}"
    if is_canonical:
        repo_id = f"{spec.hf_user}/{spec.model_prefix}"

    # staging dir
    stage = Path(spec.adapter_dir).parent / f"_hf_stage_seed{seed}" + ("" if not is_canonical else "_canonical")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    # weights + tokenizer
    for f in Path(spec.adapter_dir).iterdir():
        if f.is_file() and f.name != "train_record.json":
            shutil.copy(f, stage / f.name)

    # figures
    fig_dir = stage / "figures"
    fig_dir.mkdir()
    has_loss = _loss_curve_figure(record.get("log_history", []), fig_dir / f"loss_curve_seed{seed}.png", seed)
    report_md = None
    if spec.report_md_path and Path(spec.report_md_path).exists():
        report_md = Path(spec.report_md_path).read_text()
        _results_figure(report_md, fig_dir / "results_table.png")

    # card
    card = build_model_card(
        model_name=repo_id.split("/", 1)[1],
        base_model=qlora_cfg["base_model"],
        dataset=spec.dataset,
        dataset_url=spec.dataset_url,
        dataset_license=spec.dataset_license,
        question_counts=spec.question_counts or {},
        cutoff=spec.cutoff,
        qlora_config=qlora_cfg,
        train_record=record,
        report_md=report_md,
        gate_pass=spec.gate_pass,
        repo_url=spec.repo_url,
        plan_url=spec.plan_url,
        is_canonical=is_canonical,
    )
    (stage / "README.md").write_text(card, encoding="utf-8")

    # provenance files
    shutil.copy(spec.train_record_path, stage / "train_record.json")
    shutil.copy(spec.qlora_config_path, stage / "qlora_config.yaml")
    if spec.report_md_path and Path(spec.report_md_path).exists():
        shutil.copy(spec.report_md_path, stage / "report.md")

    # upload
    api.create_repo(repo_id=repo_id, exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=str(stage), commit_message=f"QwerySmith-2.0 seed {seed} ({'canonical' if is_canonical else 'provenance'})")
    return {"repo_id": repo_id, "stage_dir": str(stage), "loss_figure": has_loss}


def publish_all(
    adapters_dir: Path,
    train_run_dir: Path,
    hf_user: str,
    model_prefix: str,
    dataset_cfg: DatasetConfig,
    question_counts: dict,
    cutoff: Optional[str],
    report_md_path: Optional[Path],
    gate_pass: Optional[bool],
    repo_url: Optional[str] = None,
    plan_url: Optional[str] = None,
    canonical_seed: Optional[int] = None,   # median-EX seed; None => publish after eval
) -> list[dict[str, Any]]:
    """Publish every seed; optionally alias the median seed as the canonical repo."""
    results = []
    adapters_dir = Path(adapters_dir)
    for adapter in sorted(adapters_dir.glob("adapter_seed*")):
        m = re.search(r"seed(\d+)", adapter.name)
        if not m:
            continue
        seed = int(m.group(1))
        spec = PublishSpec(
            hf_user=hf_user,
            model_prefix=model_prefix,
            adapter_dir=adapter,
            train_record_path=adapter / "train_record.json",
            qlora_config_path=train_run_dir / f"qlora_seed{seed}.yaml",
            repo_url=repo_url,
            plan_url=plan_url,
            dataset=dataset_cfg.name,
            dataset_url=dataset_cfg.source_url,
            dataset_license=dataset_cfg.license,
            question_counts=question_counts,
            cutoff=cutoff,
            report_md_path=report_md_path,
            gate_pass=gate_pass,
        )
        results.append(publish_seed(spec, seed))
        if canonical_seed == seed:
            results.append(publish_seed(spec, seed, is_canonical=True))
    return results