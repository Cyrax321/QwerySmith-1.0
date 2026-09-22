"""Results report (plan §3.5/§7.2): same table + same gate for every dataset.

GATE (pre-registered, plan §7.1) — do not touch without written amendment:
  row 2 PASSES iff
    EX_heldout(row2_mean_3seeds) >= EX_heldout(row4) - 5.0
    AND flip_rate(row2_pooled) <= flip_rate(row4)
  FAIL => report row 3 vs row 4 on the same measures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .questions import Question
from .scoring import ScoredResult, mcnemar_from_results

GATE_EX_MARGIN_POINTS = 5.0


@dataclass
class GateResult:
    row2_passes: bool
    ex_row2: float
    ex_row4: float
    flip_row2: float
    flip_row4: float
    margin_met: bool
    flip_met: bool

    def verdict_line(self) -> str:
        if self.row2_passes:
            return (
                f"GATE: PASS — row 2 EX {self.ex_row2:.1%} is within "
                f"{GATE_EX_MARGIN_POINTS:.0f} pts of row 4 ({self.ex_row4:.1%}); "
                f"flip {self.flip_row2:.3f} <= {self.flip_row4:.3f}"
            )
        parts = []
        if not self.margin_met:
            parts.append(
                f"row 2 EX {self.ex_row2:.1%} is >{GATE_EX_MARGIN_POINTS:.0f} pts below row 4 ({self.ex_row4:.1%})"
            )
        if not self.flip_met:
            parts.append(f"flip {self.flip_row2:.3f} > row 4 {self.flip_row4:.3f}")
        return "GATE: FAIL — " + "; ".join(parts) + ". Reporting row 3 vs row 4."


def evaluate_gate(
    ex_row2: float, flip_row2: float, ex_row4: float, flip_row4: float
) -> GateResult:
    margin_met = ex_row2 >= ex_row4 - GATE_EX_MARGIN_POINTS / 100.0
    flip_met = flip_row2 <= flip_row4
    return GateResult(
        row2_passes=margin_met and flip_met,
        ex_row2=ex_row2, ex_row4=ex_row4,
        flip_row2=flip_row2, flip_row4=flip_row4,
        margin_met=margin_met, flip_met=flip_met,
    )


def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return (max(0.0, center - half), min(1.0, center + half))


def aggregate_seeds(
    system_summaries: list[dict[str, Any]],
    headline_results: dict[str, list[ScoredResult]],
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """Fold role='candidate_seed' rows into one candidate row (plan §7.1).

    Candidate row: EX = mean over seeds; range = min..max; flip = mean;
    agreement = mean; representative seed (median EX) used for McNemar.
    Returns (synthesized_candidate_summary, representative_system_name).
    """
    seed_rows = [s for s in system_summaries if s.get("role") == "candidate_seed"]
    if not seed_rows:
        cand = next((s for s in system_summaries if s.get("role") == "candidate"), None)
        return cand, (cand or {}).get("system")

    exs = [s["ex_heldout"] for s in seed_rows]
    mean_ex = sum(exs) / len(exs)
    rep = sorted(seed_rows, key=lambda s: s["ex_heldout"])[len(seed_rows) // 2]  # median
    candidate = {
        "system": rep["system"],          # representative; label shows the aggregate
        "role": "candidate",
        "hardware": rep.get("hardware", ""),
        "n_questions": rep["n_questions"],
        "ex_heldout": mean_ex,
        "ex_range": (min(exs), max(exs)),
        "refusal_rate": sum(s.get("refusal_rate", 0.0) for s in seed_rows) / len(seed_rows),
        "agreement": sum(s.get("agreement", 0.0) for s in seed_rows) / len(seed_rows),
        "flip_rate": sum(s.get("flip_rate", 0.0) for s in seed_rows) / len(seed_rows),
        "p50_latency_ms": sum(s.get("p50_latency_ms", 0.0) for s in seed_rows) / len(seed_rows),
        "n_failures": sum(s.get("n_failures", 0) for s in seed_rows),
        "seed_systems": [s["system"] for s in seed_rows],
        "representative_seed": rep["system"],
        "generated_at": rep.get("generated_at"),
    }
    return candidate, rep["system"]


def render_report(
    dataset: str,
    system_summaries: list[dict[str, Any]],   # in row order 1..4; row2 may be seed-mean
    headline_results: dict[str, list[ScoredResult]],   # name -> per-question headline
    gate: Optional[GateResult] = None,
    row_labels: Optional[dict[str, str]] = None,
) -> str:
    labels = row_labels or {}

    # fold 3-seed rows into the single candidate row when present (§7.1)
    candidate, rep_seed = aggregate_seeds(system_summaries, headline_results)
    if candidate is not None:
        displayed = [s for s in system_summaries if s.get("role") != "candidate_seed"]
        if candidate not in displayed:
            displayed.append(candidate)
        # McNemar/gate use the representative seed's per-question results
        headline_results = dict(headline_results)
        if rep_seed and rep_seed in headline_results and candidate["system"] == rep_seed:
            pass  # representative name == candidate system name; already correct
    else:
        displayed = system_summaries

    lines = [f"# Results — {dataset}", ""]
    lines.append("| System | EX held-out (95% CI) | Scorable N | Agr. | Flip | Refusal% | p50 lat (ms) | Hardware |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for s in displayed:
        n = s["n_questions"]
        ex = s["ex_heldout"]
        lo, hi = _wilson_ci(ex, n)
        name = labels.get(s["system"], s["system"])
        if "ex_range" in s:
            name = f"{name} (mean ± range, n={len(s.get('seed_systems', []))})"
            lo, hi = s["ex_range"]
        lines.append(
            f"| {name} | {ex:.1%} ({lo:.0%}–{hi:.0%}) | {n} | "
            f"{s.get('agreement', 0):.2f} | {s.get('flip_rate', 0):.3f} | "
            f"{s.get('refusal_rate', 0):.1%} | {s.get('p50_latency_ms', 0):.0f} | "
            f"{s.get('hardware', '')} |"
        )

    lines.append("")

    # McNemar rows — representative seed vs comparators (§7.1)
    r2 = next((s for s in displayed if s.get("role") == "candidate"), None)
    r1 = next((s for s in displayed if s.get("role") == "baseline"), None)
    r4 = next((s for s in displayed if s.get("role") == "reference"), None)
    rep_name = r2.get("representative_seed", r2["system"]) if r2 else None

    def _results_for(system_summary):
        name = system_summary.get("representative_seed", system_summary["system"])
        return headline_results.get(name, headline_results.get(system_summary["system"]))

    if r2 and r1 and _results_for(r2) and _results_for(r1):
        m = mcnemar_from_results(_results_for(r2), _results_for(r1))
        lines.append(
            f"McNemar row2 ({rep_name}, median-EX seed) vs row1: b={m['b']} c={m['c']} p={m['p_value']:.4f}"
        )
    if r2 and r4 and _results_for(r2) and _results_for(r4):
        m = mcnemar_from_results(_results_for(r2), _results_for(r4))
        lines.append(
            f"McNemar row2 ({rep_name}, median-EX seed) vs row4: b={m['b']} c={m['c']} p={m['p_value']:.4f}"
        )
    # per-seed p-values also reported (plan §7.1)
    seed_rows = [s for s in system_summaries if s.get("role") == "candidate_seed"]
    if r4 and seed_rows:
        r4_results = _results_for(r4)
        for s in seed_rows:
            sr = headline_results.get(s["system"])
            if sr and r4_results:
                m = mcnemar_from_results(sr, r4_results)
                lines.append(
                    f"  per-seed {s['system']}: b={m['b']} c={m['c']} p={m['p_value']:.4f}"
                )

    if gate:
        lines.append("")
        lines.append(gate.verdict_line())
        onprem = next((s for s in displayed if s.get("role") == "onprem"), None)
        if not gate.row2_passes and onprem and r4:
            lines.append(
                f"  row 3 ({onprem['system']}): EX {onprem['ex_heldout']:.1%}, "
                f"flip {onprem.get('flip_rate', 0):.3f} vs row 4 EX {r4['ex_heldout']:.1%}, "
                f"flip {r4.get('flip_rate', 0):.3f}"
            )

    return "\n".join(lines) + "\n"


def write_failure_folders(
    out_dir: Path,
    questions: dict[str, Question],
    headline_results: dict[str, list[ScoredResult]],
) -> dict[str, int]:
    """One markdown file per wrong answer per system, one-line reason (plan §3.5)."""
    counts: dict[str, int] = {}
    for system, results in headline_results.items():
        sys_dir = out_dir / "failures" / system
        sys_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for r in results:
            if r.correct:
                continue
            q = questions[r.question_id]
            content = (
                f"# {r.question_id} — {r.error_class}\n\n"
                f"**Question:** {q.question}\n\n"
                f"**Reason:** {r.reason}\n\n"
                f"**Predicted SQL:**\n```sql\n{r.pred_sql or '—'}\n```\n\n"
                f"**Predicted answer:** {r.pred_answer or '—'}\n\n"
                f"**Gold SQL:**\n```sql\n{q.gold_sql}\n```\n"
            )
            (sys_dir / f"{r.question_id}.md").write_text(content, encoding="utf-8")
            n += 1
        counts[system] = n
    return counts