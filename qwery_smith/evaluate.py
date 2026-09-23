"""Eval runner (plan §3.4): every system on identical questions + packs.

A System is anything callable: (prompt) -> raw_output text. Base/FT models
are vLLM/transformers calls; frontier is an API call. The runner never knows
which is which - that's the point.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from .questions import Question
from .retrieval import EvidencePack
from .scoring import ConsistencyBlock, ScoredResult, evaluate_one
from .triples import render_prompt
from .adapters.base import DatabaseAdapter

SystemFn = Callable[[str], str]


@dataclass
class SystemSpec:
    name: str
    role: str                       # baseline | candidate | onprem | reference
    call: SystemFn
    n_consistency_runs: int = 5
    hardware: str = ""


@dataclass
class SystemRun:
    spec: SystemSpec
    headline: list[ScoredResult] = field(default_factory=list)
    consistency: dict[str, ConsistencyBlock] = field(default_factory=dict)
    raw_outputs: list[dict[str, Any]] = field(default_factory=list)

    def ex_heldout(self) -> float:
        if not self.headline:
            return 0.0
        return sum(1 for r in self.headline if r.correct) / len(self.headline)

    def refusal_rate(self) -> float:
        if not self.headline:
            return 0.0
        n = sum(1 for r in self.headline if r.error_class == "unexpected_refusal")
        return n / len(self.headline)

    def mean_agreement(self) -> float:
        if not self.consistency:
            return 0.0
        return sum(b.agreement for b in self.consistency.values()) / len(self.consistency)

    def mean_flip(self) -> float:
        if not self.consistency:
            return 0.0
        return sum(b.flip for b in self.consistency.values()) / len(self.consistency)

    def p50_latency_ms(self) -> float:
        lats = sorted(r.latency_ms for r in self.headline)
        if not lats:
            return 0.0
        mid = len(lats) // 2
        return lats[mid] if len(lats) % 2 else (lats[mid - 1] + lats[mid]) / 2


def run_system(
    spec: SystemSpec | Any,          # duck-typed: needs .name/.role/.hardware/.n_consistency_runs
    call: SystemFn,
    questions: list[Question],
    packs: dict[str, EvidencePack],
    schema_ddl: str,
    adapter: DatabaseAdapter,
    timeout_sec: float = 30.0,
    consistency_seeds: Optional[list[int]] = None,
) -> SystemRun:
    """Run headline (1 deterministic-seed call) + consistency (5 sampled calls).

    Consistency runs pass DISTINCT seeds to the driver (plan §0) - otherwise
    a pinned-seed API returns identical outputs and flip rate is fake 0.
    """
    run = SystemRun(spec=spec)

    for q_index, q in enumerate(questions):
        pack = packs[q.id]
        prompt = render_prompt(schema_ddl, q.question, pack.render_evidence())
        n_runs = getattr(spec, "n_consistency_runs", 5)
        seeds = consistency_seeds or [1000 + q_index * 10 + k for k in range(n_runs)]
        headline_seed = seeds[0] - 1  # distinct from every consistency seed

        # headline
        t0 = time.perf_counter()
        raw = call(prompt, headline_seed)
        lat = (time.perf_counter() - t0) * 1000
        scored = evaluate_one(q, raw, pack, adapter, timeout_sec=timeout_sec, latency_ms=lat)
        run.headline.append(scored)
        run.raw_outputs.append({
            "question_id": q.id, "kind": "headline", "seed": headline_seed, "raw": raw,
            "scored": scored.to_json(),
        })

        # consistency block: seeds actually vary per run
        block = ConsistencyBlock(question_id=q.id, headline=scored.correct)
        for seed_k in seeds[:n_runs]:
            t0 = time.perf_counter()
            raw_k = call(prompt, seed_k)
            lat_k = (time.perf_counter() - t0) * 1000
            scored_k = evaluate_one(q, raw_k, pack, adapter, timeout_sec=timeout_sec, latency_ms=lat_k)
            block.runs.append(scored_k.correct)
            run.raw_outputs.append({
                "question_id": q.id, "kind": "consistency", "seed": seed_k, "raw": raw_k,
                "scored": scored_k.to_json(),
            })
        run.consistency[q.id] = block

    return run


def write_run(
    run: SystemRun,
    out_dir: Path,
    questions: list[Question],
) -> dict[str, Any]:
    """Persist raw outputs + summary. The report reads these, never recomputes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    name = run.spec.name

    (out_dir / f"{name}__raw.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in run.raw_outputs) + "\n",
        encoding="utf-8",
    )

    failures = [r for r in run.headline if not r.correct]
    summary = {
        "system": name,
        "role": run.spec.role,
        "hardware": run.spec.hardware,
        "n_questions": len(run.headline),
        "ex_heldout": run.ex_heldout(),
        "refusal_rate": run.refusal_rate(),
        "agreement": run.mean_agreement(),
        "flip_rate": run.mean_flip(),
        "p50_latency_ms": run.p50_latency_ms(),
        "n_failures": len(failures),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / f"{name}__summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary