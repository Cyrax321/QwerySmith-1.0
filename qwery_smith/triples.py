"""RAFT triple construction (plan §3.3) + prompt rendering.

Triples: (question, retrieved rows including distractors, answer with citation).
Grounded ~70% / refusal ~15% / schema-only ~15% (config-driven, not hardcoded).
Distractors are seeded hard negatives: same table, disjoint value on a filtered
column, adjacent category/period - plausible but wrong. The model must learn
to FILTER, not pattern-match.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .config import DatasetConfig
from .questions import Question
from .retrieval import EvidencePack, IndexDoc

# ---------------------------------------------------------------- prompt ---

SYSTEM_PROMPT = """You answer questions about the company database using only the
RETRIEVED EVIDENCE. Output format, exactly:
SQL:
<one SELECT statement answering the question>
ANSWER:
<answer in <=80 words; every factual claim carries >=1 citation [table:row_id]>
If the retrieved evidence does not contain the answer, output exactly:
REFUSAL:
<one sentence naming what is missing>"""

OUTPUT_CONTRACT = ("SQL:", "ANSWER:")


def render_prompt(schema_ddl: str, question_text: str, evidence_text: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"SCHEMA:\n{schema_ddl}\n\n"
        f"QUESTION: {question_text}\n\n"
        f"RETRIEVED EVIDENCE:\n{evidence_text}"
    )


def parse_output(raw: str) -> dict[str, str]:
    """Parse the output contract (SQL:/ANSWER: or REFUSAL:). Strict-ish: first
    marker wins; missing markers => invalid_output."""
    raw = raw.strip()
    if raw.startswith("REFUSAL:"):
        return {"kind": "refusal", "refusal": raw[len("REFUSAL:"):].strip()}
    if not raw.startswith("SQL:"):
        return {"kind": "invalid_output", "raw": raw}
    sql_part, sep, rest = raw[len("SQL:"):].partition("\nANSWER:")
    if not sep:
        # tolerate 'SQL:' on its own line
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[0].strip() == "SQL:" and "ANSWER:" in raw:
            try:
                ai = next(i for i, l in enumerate(lines) if l.strip().startswith("ANSWER:"))
                sql_part = "\n".join(lines[1:ai])
                rest = "\n".join(lines[ai + 1:])
                return {"kind": "answer", "sql": sql_part.strip(), "answer": rest.strip()}
            except StopIteration:
                return {"kind": "invalid_output", "raw": raw}
        return {"kind": "invalid_output", "raw": raw}
    return {"kind": "answer", "sql": sql_part.strip(), "answer": rest.strip()}


# --------------------------------------------------------------- triples ---

@dataclass
class Triple:
    question_id: str
    kind: str                    # grounded | refusal | schema_only
    prompt: str
    target: str
    evidence_row_ids: list[str]  # ledger entry for the §4.4 training-corpus audit

    def to_jsonl(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


CITATION_RE = None  # compiled lazily


def extract_citations(text: str) -> list[tuple[str, str]]:
    """[table:row_id] citations - machine-verifiable (plan §5.3)."""
    import re

    global CITATION_RE
    if CITATION_RE is None:
        CITATION_RE = re.compile(r"\[([A-Za-z_][A-Za-z0-9_]*):([^\]\s]+)\]")
    return [(m.group(1), m.group(2)) for m in CITATION_RE.finditer(text)]


def verify_citations(answer: str, pack: EvidencePack) -> list[str]:
    """Return list of invalid citations (empty => all valid)."""
    valid = pack.row_ids()
    bad = []
    for table, rid in extract_citations(answer):
        if f"{table}:{rid}" not in valid:
            bad.append(f"{table}:{rid}")
    return bad


def build_training_target(
    q: Question,
    answer_sentence: str,
    evidence_rows: list[dict[str, Any]],
    n_cite: int = 2,
) -> str:
    """Compose the gold answer text with citations to the top evidence rows."""
    cites = []
    for r in evidence_rows[:n_cite]:
        cites.append(f"[{r['table']}:{r['row_id']}]")
    cite_str = " " + " ".join(cites) if cites else ""
    return (
        f"SQL:\n{q.gold_sql}\n"
        f"ANSWER:\n{answer_sentence}{cite_str}"
    )


def sample_distractors(
    index_docs: list[IndexDoc],
    positives: set[str],          # "table:row_id" strings
    rng: random.Random,
    n: int,
    same_table: bool = True,
) -> list[IndexDoc]:
    """Seeded hard-negative sampling: prefer same-table rows not in positives."""
    pool = [
        d for d in index_docs
        if f"{d.table}:{d.row_id}" not in positives
    ]
    if not pool:
        return []
    # weighted: same tables as positives first, then rest - deterministic under rng
    pos_tables = {p.split(":", 1)[0] for p in positives}
    near = [d for d in pool if d.table in pos_tables]
    far = [d for d in pool if d.table not in pos_tables] if same_table else pool
    picked: list[IndexDoc] = []
    if near:
        picked.extend(rng.sample(near, min(n, len(near))))
    while len(picked) < n and far:
        extra = rng.sample(far, min(n - len(picked), len(far)))
        picked.extend(extra)
        far = [d for d in far if d not in extra]
    return picked[:n]


def build_triples(
    cfg: DatasetConfig,
    questions: list[Question],
    index,
    schema_ddl: str,
    eval_packs: dict[str, EvidencePack],
    answer_texts: Optional[dict[str, str]] = None,
    rng: Optional[random.Random] = None,
    mix: dict[str, float] | None = None,
) -> list[Triple]:
    """Build RAFT triples from train_ok questions only.

    mix: {grounded, refusal, schema_only} shares, default 70/15/15 (plan §3.3).
    answer_texts: qid -> gold answer sentence. If absent, generated as
    "The query returns N row(s)." - datasets author richer sentences later.
    """
    rng = rng or random.Random(cfg.seed)
    m = mix or {"grounded": 0.70, "refusal": 0.15, "schema_only": 0.15}
    docs = index.docs
    triples: list[Triple] = []

    for q in questions:
        if q.split != "train_ok":
            continue  # held-out questions NEVER reach the trainer
        pack = eval_packs.get(q.id)
        if pack is None:
            continue
        # positive rows = the pack rows the gold answer cites
        positive_ids = {f"{r['table']}:{r['row_id']}" for r in pack.rows}
        positives = [r for r in pack.rows]
        answer_sentence = (answer_texts or {}).get(q.id, f"The query returns {q.expected_rows.n_rows} row(s).")

        roll = rng.random()
        if roll < m["grounded"]:
            evidence = positives + [
                {"table": d.table, "row_id": d.row_id, "columns": list(d.columns),
                 "values": [str(v) if v is not None else None for v in d.values]}
                for d in sample_distractors(docs, positive_ids, rng, rng.randint(2, 5))
            ]
            rng.shuffle(evidence)
            ev_text = "\n".join(
                f"[{r['table']}:{r['row_id']}] " + " | ".join(
                    f"{c}: {v}" for c, v in zip(r["columns"], r["values"], strict=False) if v not in (None, "")
                )
                for r in evidence
            )
            prompt = render_prompt(schema_ddl, q.question, ev_text)
            target = build_training_target(q, answer_sentence, positives)
            triples.append(Triple(
                question_id=q.id,
                kind="grounded",
                prompt=prompt,
                target=target,
                evidence_row_ids=sorted(positive_ids | {f"{r['table']}:{r['row_id']}" for r in evidence}),
            ))
        elif roll < m["grounded"] + m["refusal"]:
            # distractors only -> refusal
            distractors = sample_distractors(docs, positive_ids, rng, max(len(positives), 3))
            ev_text = "\n".join(d.render() for d in distractors)
            prompt = render_prompt(schema_ddl, q.question, ev_text)
            target = "REFUSAL:\nThe retrieved evidence does not contain records matching this question."
            triples.append(Triple(
                question_id=q.id, kind="refusal", prompt=prompt, target=target,
                evidence_row_ids=sorted({f"{d.table}:{d.row_id}" for d in distractors}),
            ))
        else:
            # schema-only: no rows
            prompt = render_prompt(schema_ddl, q.question, "(no rows retrieved)")
            target = build_training_target(q, answer_sentence, positives)
            triples.append(Triple(
                question_id=q.id, kind="schema_only", prompt=prompt, target=target,
                evidence_row_ids=[],
            ))

    return triples


def write_triples(triples: list[Triple], path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in triples:
            f.write(t.to_jsonl() + "\n")
    # ledger for the §4.4 training-corpus audit
    ledger = {
        "n_triples": len(triples),
        "by_kind": {},
        "evidence_row_ids": sorted({rid for t in triples for rid in t.evidence_row_ids}),
    }
    for t in triples:
        ledger["by_kind"][t.kind] = ledger["by_kind"].get(t.kind, 0) + 1
    ledger_path = path.with_suffix(".ledger.json")
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False), encoding="utf-8")
    return {"triples": str(path), "ledger": str(ledger_path), "n": len(triples)}