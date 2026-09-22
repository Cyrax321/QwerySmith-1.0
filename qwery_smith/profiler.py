"""Profiler (plan §3.1): row counts, date min/max, categorical distinct counts,
holdout cutoff. Output is the YAML artifact question authors use to tag splits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from typing import Any, Optional

from .adapters.base import DatabaseAdapter
from .config import HoldoutConfig
from .exceptions import ProfileError


@dataclass
class DateColumnProfile:
    table: str
    column: str
    min: Optional[str]
    max: Optional[str]


@dataclass
class DatasetProfile:
    tables: list[dict[str, Any]]              # [{name, row_count, columns: [{name, type, not_null}]}]
    fk_edges: list[dict[str, str]]
    date_columns: list[DateColumnProfile]
    holdout: Optional[dict[str, Any]]         # {column, cutoff, rule}
    generated_at: str

    def to_yaml_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "tables": self.tables,
            "fk_edges": self.fk_edges,
            "date_columns": [asdict(d) for d in self.date_columns],
            "holdout": self.holdout,
        }


_DATE_TYPES = {"DATE", "TIMESTAMP"}


def _month_floor(d: date) -> date:
    return d.replace(day=1)


def compute_holdout_cutoff(
    adapter: DatabaseAdapter, holdout: HoldoutConfig
) -> tuple[str, str]:
    """Plan §4.2: cutoff = start_of_month(max(<date col>)) - N months.

    Returns (max_value, cutoff) as ISO strings. Pure function of data — not of
    anyone's judgment (that's the whole point).
    """
    q = f'SELECT MIN("{holdout.col}"), MAX("{holdout.col}") FROM "{holdout.table}"'
    row = adapter.scalar(q)
    lo, hi = row[0], row[1]
    if hi is None:
        raise ProfileError(f"holdout column {holdout.column} has no values")

    def _to_date(v: Any) -> date:
        if isinstance(v, datetime):
            return v.date()
        if isinstance(v, date):
            return v
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d"):
            try:
                return datetime.strptime(str(v).strip(), fmt).date()
            except ValueError:
                continue
        raise ProfileError(f"unparseable date value in {holdout.column}: {v!r}")

    hi_d = _to_date(hi)
    cutoff = _month_floor(hi_d) - timedelta(days=30 * holdout.months)
    # month arithmetic without relativedelta: step back N calendar months
    y, m = hi_d.year, hi_d.month
    for _ in range(holdout.months):
        m -= 1
        if m == 0:
            m = 12
            y -= 1
    cutoff = date(y, m, 1)
    return hi_d.isoformat(), cutoff.isoformat()


def profile_database(
    adapter: DatabaseAdapter, holdout: Optional[HoldoutConfig] = None
) -> DatasetProfile:
    from .schema_loader import load_schema

    schema = load_schema(adapter)

    tables_out: list[dict[str, Any]] = []
    for tname in sorted(schema.tables):
        t = schema.tables[tname]
        tables_out.append({
            "name": tname,
            "row_count": t.row_count,
            "columns": [
                {"name": n, "type": ty, "not_null": nn}
                for n, ty, nn in t.columns
            ],
        })

    date_cols: list[DateColumnProfile] = []
    for tname in sorted(schema.tables):
        for cname, ctype, _nn in schema.tables[tname].columns:
            if ctype in _DATE_TYPES:
                row = adapter.scalar(f'SELECT MIN("{cname}"), MAX("{cname}") FROM "{tname}"')
                lo_s = row[0].isoformat() if hasattr(row[0], "isoformat") else (str(row[0]) if row[0] is not None else None)
                hi_s = row[1].isoformat() if hasattr(row[1], "isoformat") else (str(row[1]) if row[1] is not None else None)
                date_cols.append(DateColumnProfile(table=tname, column=cname, min=lo_s, max=hi_s))

    holdout_out: Optional[dict[str, Any]] = None
    if holdout is not None:
        hi_iso, cutoff_iso = compute_holdout_cutoff(adapter, holdout)
        holdout_out = {
            "column": holdout.column,
            "months": holdout.months,
            "data_max": hi_iso,
            "cutoff": cutoff_iso,
            "rule": "date_trunc(month, max(col)) - N months",
        }

    return DatasetProfile(
        tables=tables_out,
        fk_edges=[
            {"table": e.table, "column": e.column, "ref_table": e.ref_table, "ref_column": e.ref_column}
            for e in schema.fk_edges
        ],
        date_columns=date_cols,
        holdout=holdout_out,
        generated_at=datetime.now().isoformat(timespec="seconds"),
    )