"""Results report (plan §3.5/§7.2): same table + same gate for every dataset.

GATE (pre-registered, plan §7.1) — do not touch without written amendment:
  row 2 PASSES iff
    EX_heldout(row2_mean_3seeds) >= EX_heldout(row4) - 5.0
    AND flip_rate(row2_pooled) <= flip_rate(row4)
  FAIL => report row 3 vs row 4 on the same measures.
"""

from __future__ import annotations

import json
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


def render_report(
    dataset: str,
    system_summaries: list[dict[str, Any]],   # in row order 1..4; row2 may be seed-mean
    headline_results: dict[str, list[ScoredResult]],   # name -> per-question headline
    gate: Optional[GateResult] = None,
    row_labels: Optional[dict[str, str]] = None,
) -> str:
    labels = row_labels or {}

    lines = [f"# Results — {dataset}", ""]
    lines.append("| System | EX held-out (95% CI) | Scorable N | Agr. | Flip | Refusal% | p50 lat (ms) | Hardware |")
    lines.append("|---|---|---|---|---|---|---|---|")

    by_role = {}
    for s in system_summaries:
        by_role[s.get("role", s["system"])] = s

    for s in system_summaries:
        n = s["n_questions"]
        ex = s["ex_heldout"]
        lo, hi = _wilson_ci(ex, n)
        name = labels.get(s["system"], s["system"])
        lines.append(
            f"| {name} | {ex:.1%} ({lo:.0%}–{hi:.0%}) | {n} | "
            f"{s.get('agreement', 0):.2f} | {s.get('flip_rate', 0):.3f} | "
            f"{s.get('refusal_rate', 0):.1%} | {s.get('p50_latency_ms', 0):.0f} | "
            f"{s.get('hardware', '')} |"
        )

    lines.append("")

    # McNemar rows
    r2 = by_role.get("candidate")
    r1 = by_role.get("baseline")
    r4 = by_role.get("reference")
    if r2 and r1:
        m = mcnemar_from_results(
            headline_results[r2["system"]], headline_results[r1["system"]]
        )
        lines.append(
            f"McNemar row2 vs row1: b={m['b']} c={m['c']} p={m['p_value']:.4f}"
        )
    if r2 and r4:
        m = mcnemar_from_results(
            headline_results[r2["system"]], headline_results[r4["system"]]
        )
        lines.append(
            f"McNemar row2 vs row4: b={m['b']} c={m['c']} p={m['p_value']:.4f}"
        )

    if gate:
        lines.append("")
        lines.append(gate.verdict_line())
        if not gate.row2_passes and "onprem" in by_role and r4:
            lines.append(
                f"  row 3 ({by_role['onprem']['system']}): EX {by_role['onprem']['ex_heldout']:.1%}, "
                f"flip {by_role['onprem'].get('flip_rate', 0):.3f} vs row 4 EX {r4['ex_heldout']:.1%}, "
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