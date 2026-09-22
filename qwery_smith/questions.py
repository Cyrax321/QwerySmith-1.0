"""Question set format (plan §3.2): one file format for all datasets.

questions_v1.jsonl - one JSON object per line:
  id, question, gold_sql, expected_rows {sha256, n_rows}, date, difficulty,
  category, split, source

manifest.yaml - name, version, licence, seed, counts, db_fingerprint,
holdout cutoff (filled from profiler output at validate time).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date as date_cls
from pathlib import Path
from typing import Any, Optional

import yaml

from .exceptions import ValidationError

DIFFICULTIES = {"easy", "medium", "hard"}
SPLITS = {"train_ok", "heldout"}
CATEGORIES = {"per_order_lookup", "aggregation", "multi_table_join", "review_text"}
SOURCES = {"human", "template+human-verified"}
_TEMPORAL_TERMS = ("last ", "recent", "latest", "past ", "current month", "current year")


@dataclass(frozen=True)
class ExpectedRows:
    sha256: str
    n_rows: int


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    gold_sql: str
    expected_rows: ExpectedRows
    date: str                        # authorship date (ISO)
    difficulty: str                  # easy | medium | hard
    category: str                    # per_order_lookup | aggregation | multi_table_join | review_text
    split: str                       # train_ok | heldout
    source: str                      # human | template+human-verified

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["expected_rows"] = asdict(self.expected_rows)
        return d

    def with_changes(self, **changes: Any) -> "Question":
        """Copy with modifications; accepts expected_rows as dict or ExpectedRows."""
        d = self.to_json()
        d.update(changes)
        return Question.from_json(d)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Question":
        return cls(
            id=d["id"],
            question=d["question"],
            gold_sql=d["gold_sql"],
            expected_rows=ExpectedRows(**d["expected_rows"]),
            date=d["date"],
            difficulty=d["difficulty"],
            category=d["category"],
            split=d["split"],
            source=d["source"],
        )


def canonical_rows_hash(rows: list[tuple], columns: list[str]) -> tuple[str, int]:
    """Plan §6.4 canonicalization core: bag-of-tuples, sorted, JSON-serialized.

    Column ORDER is ignored (names carried separately); row order ignored;
    numerics rounded to 2dp; strings NFC-normalized, casefolded,
    whitespace-collapsed; NULL canonical; timestamps to second precision.
    """
    import unicodedata

    def canon_cell(v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, float):
            return round(v, 2)
        if isinstance(v, int):
            return v
        if hasattr(v, "isoformat"):
            return v.isoformat(sep=" ", timespec="seconds") if not isinstance(v, date_cls) else v.isoformat()
        if isinstance(v, (list, tuple)):
            return [canon_cell(x) for x in v]
        s = unicodedata.normalize("NFC", str(v))
        s = " ".join(s.split()).casefold()
        return s

    canon = sorted([tuple(canon_cell(c) for c in row) for row in rows], key=repr)
    payload = json.dumps({"n": len(canon), "rows": [list(r) for r in canon]}, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest(), len(canon)


@dataclass
class QuestionSet:
    questions: list[Question] = field(default_factory=list)

    def add(self, q: Question) -> None:
        self.questions.append(q)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for q in self.questions:
                f.write(json.dumps(q.to_json(), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: Path) -> "QuestionSet":
        qs = cls()
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                try:
                    qs.questions.append(Question.from_json(d))
                except KeyError as e:
                    raise ValidationError(f"{path}:{i}: missing key {e}") from e
        return qs

    def counts(self) -> dict[str, Any]:
        by_diff: dict[str, int] = {}
        by_cat: dict[str, int] = {}
        by_split: dict[str, int] = {}
        for q in self.questions:
            by_diff[q.difficulty] = by_diff.get(q.difficulty, 0) + 1
            by_cat[q.category] = by_cat.get(q.category, 0) + 1
            by_split[q.split] = by_split.get(q.split, 0) + 1
        return {
            "total": len(self.questions),
            "by_difficulty": by_diff,
            "by_category": by_cat,
            "by_split": by_split,
        }

    def ids_unique(self) -> bool:
        ids = [q.id for q in self.questions]
        return len(ids) == len(set(ids))


def write_manifest(
    path: Path,
    *,
    name: str,
    version: str,
    license_: str,
    source_url: str,
    seed: int,
    questions_path: Path,
    db_fingerprint: str,
    holdout: Optional[dict[str, Any]],
    counts: dict[str, Any],
) -> None:
    """Write the manifest (plan §3.2). Holdout cutoff comes from profiler  - 
    mechanical, not authored."""
    manifest = {
        "name": name,
        "version": version,
        "license": license_,
        "source_url": source_url,
        "created_with_seed": seed,
        "questions_file": questions_path.name,
        "db_fingerprint": db_fingerprint,
        "holdout": holdout,
        "counts": counts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest, sort_keys=True), encoding="utf-8")


def load_manifest(path: Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())