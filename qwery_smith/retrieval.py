"""Retrieval (plan §4.3): inverted-index BM25 over serialized rows.

Builds a row-text corpus from any prepared database, retrieves top-K evidence
rows from question text ONLY (gold SQL is never in the retrieval path), and
freezes evidence packs to disk so every system in the eval matrix sees
byte-identical context.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .adapters.base import DatabaseAdapter
from .exceptions import HarnessError
from .schema_loader import Schema, TableSchema

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_K1 = 1.5
_B = 0.75


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _row_id(pk: tuple[str, ...], values: dict[str, Any], ordinal: int) -> str:
    if pk:
        return "#".join(str(values.get(c)) for c in pk)
    return f"rowidx:{ordinal}"


@dataclass
class IndexDoc:
    table: str
    row_id: str
    columns: tuple[str, ...]
    values: tuple[Any, ...]
    tokens: tuple[str, ...] = field(default_factory=tuple)

    def render(self) -> str:
        cells = " | ".join(f"{c}: {v}" for c, v in zip(self.columns, self.values, strict=False) if v not in (None, ""))
        return f"[{self.table}:{self.row_id}] {cells}"


@dataclass
class BM25Index:
    docs: list[IndexDoc]
    _postings: dict[str, list[int]] = field(default_factory=dict)
    _df: dict[str, int] = field(default_factory=dict)
    _avgdl: float = 0.0

    def search(self, query: str, top_k: int) -> list[tuple[float, IndexDoc]]:
        q_tokens = tokenize(query)
        if not q_tokens or not self.docs:
            return []
        candidates: set[int] = set()
        for t in q_tokens:
            candidates.update(self._postings.get(t, ()))
        if not candidates:
            return []
        n = len(self.docs)
        scored: list[tuple[float, int]] = []
        for doc_i in candidates:
            doc = self.docs[doc_i]
            score = 0.0
            dl = len(doc.tokens)
            for t in q_tokens:
                tf = 0
                for dt in doc.tokens:
                    if dt == t:
                        tf += 1
                if tf == 0:
                    continue
                df = self._df.get(t, 0)
                if df == 0:
                    continue
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                score += idf * (tf * (_K1 + 1)) / (tf + _K1 * (1 - _B + _B * dl / self._avgdl))
            scored.append((score, doc_i))
        scored.sort(key=lambda x: (-x[0], x[1]))  # deterministic tie-break by doc order
        return [(s, self.docs[i]) for s, i in scored[:top_k]]


def build_index(
    adapter: DatabaseAdapter,
    schema: Schema,
    row_caps: Optional[dict[str, int]] = None,
) -> BM25Index:
    caps = row_caps or {}
    docs: list[IndexDoc] = []
    for tname in sorted(schema.tables):
        t: TableSchema = schema.tables[tname]
        cols = [c for c, _t, _nn in t.columns]
        limit = caps.get(tname)
        sql = f'SELECT * FROM "{tname}"'
        if limit:
            sql += f" LIMIT {int(limit)}"
        cur_rows, _c, _l = adapter.execute(sql, max_rows=max(t.row_count, limit or 0) + 1)
        col_index = {c: i for i, c in enumerate(cols)}
        for ordinal, row in enumerate(cur_rows):
            values = {c: row[col_index[c]] for c in cols}
            rid = _row_id(t.primary_key, values, ordinal)
            text = f"{tname} " + " ".join(str(v) for v in row if v not in (None, ""))
            docs.append(IndexDoc(
                table=tname,
                row_id=rid,
                columns=tuple(cols),
                values=tuple(row),
                tokens=tuple(tokenize(text)),
            ))
    idx = BM25Index(docs=docs)
    idx._df = {}
    for i, d in enumerate(docs):
        seen: set[str] = set()
        for tok in d.tokens:
            if tok in seen:
                continue
            seen.add(tok)
            idx._postings.setdefault(tok, []).append(i)
            idx._df[tok] = idx._df.get(tok, 0) + 1
    idx._avgdl = (sum(len(d.tokens) for d in docs) / len(docs)) if docs else 0.0
    return idx


@dataclass
class EvidencePack:
    question_id: str
    question: str
    schema_sha256: str
    rows: list[dict[str, Any]]      # [{table, row_id, columns, values}]

    def row_ids(self) -> set[str]:
        return {f"{r['table']}:{r['row_id']}" for r in self.rows}

    def to_json(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "schema_sha256": self.schema_sha256,
            "rows": self.rows,
        }

    def render_evidence(self) -> str:
        lines = []
        for r in self.rows:
            cells = " | ".join(
                f"{c}: {v}" for c, v in zip(r["columns"], r["values"], strict=False) if v not in (None, "")
            )
            lines.append(f"[{r['table']}:{r['row_id']}] {cells}")
        return "\n".join(lines)


def build_pack(
    question_id: str,
    question_text: str,
    index: BM25Index,
    schema: Schema,
    top_k: int,
) -> EvidencePack:
    hits = index.search(question_text, top_k)
    return EvidencePack(
        question_id=question_id,
        question=question_text,
        schema_sha256=schema.ddl_sha256(),
        rows=[
            {
                "table": d.table,
                "row_id": d.row_id,
                "columns": list(d.columns),
                "values": [str(v) if v is not None else None for v in d.values],
            }
            for _s, d in hits
        ],
    )


def freeze_pack(pack: EvidencePack, packs_dir: Path) -> dict[str, str]:
    packs_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(pack.to_json(), ensure_ascii=False, sort_keys=True)
    sha = hashlib.sha256(payload.encode()).hexdigest()
    path = packs_dir / f"{pack.question_id}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(payload)
    return {"path": str(path), "sha256": sha}


def load_pack(packs_dir: Path, question_id: str, expected_sha256: Optional[str] = None) -> EvidencePack:
    path = packs_dir / f"{question_id}.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        payload = f.read()
    if expected_sha256 is not None:
        got = hashlib.sha256(payload.encode()).hexdigest()
        if got != expected_sha256:
            raise HarnessError(f"pack integrity failure for {question_id}: {got[:12]} != {expected_sha256[:12]}")
    d = json.loads(payload)
    return EvidencePack(
        question_id=d["question_id"],
        question=d["question"],
        schema_sha256=d["schema_sha256"],
        rows=d["rows"],
    )