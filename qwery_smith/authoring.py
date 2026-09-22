"""Question authoring (plan §5.1): generators produce (question, gold_sql)
candidates per category; the author reviews, edits, and the harness computes
expected_rows + split mechanically.

Design: templates are functions over schema facts (discovered from the
profiler), NOT hardcoded Olist values. The dataset config may pin a few
template parameters (e.g. specific order IDs for per-order lookups) via
`extra.authoring` - those live in config, never code.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .adapters.base import DatabaseAdapter
from .config import DatasetConfig


@dataclass
class Candidate:
    question: str
    gold_sql: str
    category: str
    difficulty: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SchemaFacts:
    """Discoverable facts used by question generators - dataset-agnostic."""
    pk_by_table: dict[str, tuple[str, ...]]
    fk_edges: list[tuple[str, str, str, str]]     # (table, col, ref_table, ref_col)
    columns: dict[str, list[tuple[str, str]]]      # table -> [(col, type)]
    date_cols: dict[str, list[str]]                # table -> [date-ish col names]
    text_cols: dict[str, list[str]]                # table -> [text col names]
    numeric_cols: dict[str, list[str]]             # table -> [numeric col names]
    categorical_cols: dict[str, list[str]]         # low-cardinality text cols
    sample_pks: dict[str, list[tuple]]             # table -> sampled pk values


def discover_facts(
    adapter: DatabaseAdapter, schema, max_cardinality: int = 60
) -> SchemaFacts:
    facts = SchemaFacts(
        pk_by_table={}, fk_edges=[], columns={}, date_cols={}, text_cols={},
        numeric_cols={}, categorical_cols={}, sample_pks={},
    )
    for tname in sorted(schema.tables):
        t = schema.tables[tname]
        facts.pk_by_table[tname] = t.primary_key
        facts.columns[tname] = [(c, ty) for c, ty, _ in t.columns]
        for c, ty, _ in t.columns:
            if ty in ("DATE", "TIMESTAMP"):
                facts.date_cols.setdefault(tname, []).append(c)
            elif ty in ("REAL", "NUMERIC"):
                facts.numeric_cols.setdefault(tname, []).append(c)
            else:
                facts.text_cols.setdefault(tname, []).append(c)
        # categorical: distinct count under threshold
        for c, ty in facts.columns[tname]:
            if ty in ("TEXT",):
                n = adapter.scalar(f'SELECT COUNT(DISTINCT "{c}") FROM "{tname}"')
                if n is not None and n[0] is not None and n[0] <= max_cardinality and n[0] >= 2:
                    facts.categorical_cols.setdefault(tname, []).append(c)
        # sample pks (handles multi-column pk tuples and scalars)
        if t.primary_key:
            pk_cols = ", ".join(f'"{c}"' for c in t.primary_key)
            rows = adapter.execute(
                f'SELECT {pk_cols} FROM "{tname}" ORDER BY RANDOM() LIMIT 3',
                max_rows=3,
            )[0]
            facts.sample_pks[tname] = [
                tuple(r) if len(t.primary_key) > 1 else r[0] for r in rows
            ]
    facts.fk_edges = [(e.table, e.column, e.ref_table, e.ref_column) for e in schema.fk_edges]
    return facts


# ------------------------------------------------------------ generators ---
# Each generator: (facts, adapter, cfg, rng) -> Optional[Candidate]


def _quote(v: Any) -> str:
    if isinstance(v, (int, float)):
        return str(v)
    return "'" + str(v).replace("'", "''") + "'"


def gen_per_order_lookup(facts: SchemaFacts, adapter, cfg, rng) -> Optional[Candidate]:
    """A lookup about one entity: 'status of order X' - key equality filter.

    Handles both single-column PKs and single-column KEYS on flat tables
    (e.g. Invoice on a line-grain table): the lookup target is one entity
    id from the key column, projected attributes from the same row-set.
    """
    # 1) classic single-column pk path
    for tname in facts.sample_pks:
        pk = facts.pk_by_table[tname]
        if len(pk) != 1:
            continue
        cols = [c for c, _ in facts.columns[tname] if c != pk[0]]
        if not cols:
            continue
        target = rng.choice(cols)
        vals = [v for v in facts.sample_pks[tname] if not isinstance(v, tuple)]
        if not vals:
            continue
        pk_val = rng.choice(vals)
        sql = f'SELECT "{target}" FROM "{tname}" WHERE "{pk[0]}" = {_quote(pk_val)}'
        q = f"What is the {target.replace('_', ' ')} of {tname} {_quote(pk_val).strip(chr(39))}?"
        return Candidate(q, sql, "per_order_lookup", "easy", {"table": tname, "pk": pk_val, "col": target})

    # 2) flat-table path: pick a low-cardinality text column as the entity key
    #    (invoice-like), project 1-2 attributes for one sampled value
    for tname, cats in facts.categorical_cols.items():
        key_cols = [
            c for c, ty in facts.columns[tname]
            if ty == "TEXT" and c not in cats and c not in facts.numeric_cols.get(tname, [])
        ]
        if not key_cols:
            continue
        key = rng.choice(key_cols)
        # distinct non-empty values, bounded query
        row = adapter.scalar(
            f'SELECT COUNT(DISTINCT "{key}") FROM "{tname}" '
            f'WHERE "{key}" IS NOT NULL AND "{key}" != \'\''
        )
        n_distinct = row[0] if row else 0
        if not n_distinct or n_distinct > 200000:
            continue
        srow = adapter.scalar(
            f'SELECT "{key}" FROM "{tname}" WHERE "{key}" IS NOT NULL AND "{key}" != \'\' '
            f'ORDER BY RANDOM() LIMIT 1'
        )
        if not srow or srow[0] is None:
            continue
        key_val = srow[0]
        proj = rng.sample([c for c, _t in facts.columns[tname] if c != key], k=min(2, len(facts.columns[tname]) - 1))
        sel = ", ".join(f'"{c}"' for c in proj)
        sql = f'SELECT {sel} FROM "{tname}" WHERE "{key}" = {_quote(key_val)}'
        key_label = key.replace("_", " ").lower().replace(" id", "")
        q = (
            f"For {key_label} {str(key_val)}, list "
            + " and ".join(c.replace('_', ' ').lower() for c in proj) + "."
        )
        return Candidate(q, sql, "per_order_lookup", "easy",
                         {"table": tname, "key": key, "key_val": key_val})
    return None


def gen_aggregate(facts: SchemaFacts, adapter, cfg, rng) -> Optional[Candidate]:
    """SUM/COUNT/AVG grouped by a categorical; variants: plain, month-windowed,
    HAVING-filtered, per-entity count. Randomized across a real parameter
    space so flat tables yield diverse aggregates, not one repeated shape."""
    for tname in facts.categorical_cols:
        cat = rng.choice(facts.categorical_cols[tname])
        numeric = facts.numeric_cols.get(tname) or []
        dcols = facts.date_cols.get(tname) or []
        variant = rng.choice(["plain", "windowed", "having", "count"])
        if variant == "windowed" and dcols:
            dcol = rng.choice(dcols)
            # sample a real month from the data
            srow = adapter.scalar(
                f"SELECT STRFTIME('%Y-%m', \"{dcol}\") FROM \"{tname}\" "
                f"WHERE \"{dcol}\" IS NOT NULL ORDER BY RANDOM() LIMIT 1"
            )
            month = srow[0] if srow and srow[0] else None
            if month:
                if numeric:
                    m = rng.choice(numeric)
                    sql = (
                        f'SELECT "{cat}", SUM("{m}") AS total_{m}\n'
                        f'FROM "{tname}"\n'
                        f'WHERE STRFTIME(\'%Y-%m\', "{dcol}") = \'{month}\'\n'
                        f'GROUP BY "{cat}" ORDER BY total_{m} DESC'
                    )
                    q = f"Total {m.replace('_', ' ')} by {cat.replace('_', ' ')} in {month}."
                else:
                    sql = (
                        f'SELECT "{cat}", COUNT(*) AS n\n'
                        f'FROM "{tname}"\n'
                        f'WHERE STRFTIME(\'%Y-%m\', "{dcol}") = \'{month}\'\n'
                        f'GROUP BY "{cat}" ORDER BY n DESC'
                    )
                    q = f"Number of records by {cat.replace('_', ' ')} in {month}."
                return Candidate(q, sql, "aggregation", "medium",
                                 {"table": tname, "group": cat, "month": month})
        if variant == "having":
            if numeric:
                m = rng.choice(numeric)
                sql = (
                    f'SELECT "{cat}", COUNT(*) AS n, SUM("{m}") AS total_{m}\n'
                    f'FROM "{tname}" GROUP BY "{cat}" HAVING COUNT(*) > 1000 '
                    f'ORDER BY total_{m} DESC'
                )
                q = (
                    f"By {cat.replace('_', ' ')}: groups with over 1000 records, "
                    f"with total {m.replace('_', ' ')}."
                )
                return Candidate(q, sql, "aggregation", "hard",
                                 {"table": tname, "group": cat, "having": True})
        if numeric and variant == "plain":
            m = rng.choice(numeric)
            fn = rng.choice(["SUM", "AVG"])
            label = f"{fn.lower()}_{m}"
            sql = (
                f'SELECT "{cat}", {fn}("{m}") AS {label} FROM "{tname}" '
                f'GROUP BY "{cat}" ORDER BY {label} DESC'
            )
            q = f"{fn} {m.replace('_', ' ')} by {cat.replace('_', ' ')}."
            return Candidate(q, sql, "aggregation", "medium", {"table": tname, "group": cat, "fn": fn})
        # count fallback
        sql = f'SELECT "{cat}", COUNT(*) AS n FROM "{tname}" GROUP BY "{cat}" ORDER BY n DESC'
        q = f"Number of records by {cat.replace('_', ' ')}."
        return Candidate(q, sql, "aggregation", "medium", {"table": tname, "group": cat})
    return None


def gen_multi_table_join(facts: SchemaFacts, adapter, cfg, rng) -> Optional[Candidate]:
    """Join a fact table to two or more dimension tables, aggregate."""
    # find a fact table: one with >= 2 outgoing FKs
    fk_by_table: dict[str, list[tuple[str, str, str]]] = {}
    for (tb, col, rt, rc) in facts.fk_edges:
        fk_by_table.setdefault(tb, []).append((col, rt, rc))
    facts_tables = [t for t in fk_by_table if len(fk_by_table[t]) >= 2]
    if not facts_tables:
        return gen_multi_table_join_flat(facts, adapter, cfg, rng)
    ft = rng.choice(facts_tables)
    dims = fk_by_table[ft]
    d1, d2 = rng.sample(dims, 2)
    (c1, t1, rc1), (c2, t2, rc2) = d1, d2
    facts.pk_by_table.get(t1, (rc1,))
    facts.pk_by_table.get(t2, (rc2,))
    cat1 = facts.categorical_cols.get(t1) or [rc1]
    cat2 = facts.categorical_cols.get(t2) or [rc2]
    g1, g2 = rng.choice(cat1), rng.choice(cat2)
    numeric = facts.numeric_cols.get(ft)
    measure = rng.choice(numeric) if numeric else "*"
    agg = f'SUM(f."{measure}")' if measure != "*" else "COUNT(*)"
    label = f"total_{measure}" if measure != "*" else "n"
    sql = (
        f'SELECT d1."{g1}" AS {g1}, d2."{g2}" AS {g2}, {agg} AS {label}\n'
        f'FROM "{ft}" f\n'
        f'JOIN "{t1}" d1 ON f."{c1}" = d1."{rc1}"\n'
        f'JOIN "{t2}" d2 ON f."{c2}" = d2."{rc2}"\n'
        f'GROUP BY d1."{g1}", d2."{g2}"\n'
        f'ORDER BY {label} DESC'
    )
    q = (
        f"{agg.split('(')[0]} of {measure if measure != '*' else 'records'} in {ft} "
        f"by {g1.replace('_', ' ')} and {g2.replace('_', ' ')}."
    )
    return Candidate(q, sql, "multi_table_join", "hard", {"fact": ft, "dims": [t1, t2]})


def gen_multi_table_join_flat(facts: SchemaFacts, adapter, cfg, rng) -> Optional[Candidate]:
    """No-FK fallback: two-level grouping + subquery on a single wide table.

    Still exercises the hard patterns (nested aggregation, comparison against
    a global statistic) without a join graph to walk.
    """
    for tname in facts.categorical_cols:
        cats = facts.categorical_cols[tname]
        numeric = facts.numeric_cols.get(tname) or []
        if len(cats) < 1 or not numeric:
            continue
        cat = rng.choice(cats)
        m = rng.choice(numeric)
        variant = rng.choice(["above_global_avg", "top_by_subgroup"])
        if variant == "above_global_avg":
            sql = (
                f'SELECT "{cat}", SUM("{m}") AS total_{m}\n'
                f'FROM "{tname}"\n'
                f'GROUP BY "{cat}"\n'
                f'HAVING SUM("{m}") > (SELECT AVG(total) FROM '
                f'(SELECT SUM("{m}") AS total FROM "{tname}" GROUP BY "{cat}"))\n'
                f'ORDER BY total_{m} DESC'
            )
            q = (
                f"Which {cat.replace('_', ' ')} have total {m.replace('_', ' ')} "
                f"above the overall group average?"
            )
            return Candidate(q, sql, "multi_table_join", "hard",
                             {"table": tname, "variant": variant})
        # top_by_subgroup: two distinct categoricals if possible, else month x cat
        dcols = facts.date_cols.get(tname) or []
        if dcols:
            dcol = rng.choice(dcols)
            other_cats = [c for c in cats if c != cat]
            if other_cats:
                cat2 = rng.choice(other_cats)
                g2_sql = f'"{cat2}"'
                g2_label = cat2.replace('_', ' ')
            else:
                cat2 = None
                g2_sql = f'STRFTIME(\'%Y-%m\', "{dcol}") AS month'
                g2_label = "month"
            sql = (
                f'SELECT "{cat}", {g2_sql}, SUM("{m}") AS total_{m}\n'
                f'FROM "{tname}"\n'
                f'WHERE STRFTIME(\'%Y\', "{dcol}") = \'2011\'\n'
                f'GROUP BY "{cat}", {g2_sql} ORDER BY total_{m} DESC LIMIT 20'
            )
            q = (
                f"Top 20 {cat.replace('_', ' ')} by {g2_label} "
                f"combinations for total {m.replace('_', ' ')} in 2011."
            )
            return Candidate(q, sql, "multi_table_join", "hard",
                             {"table": tname, "variant": variant})
    return None


def gen_review_text(facts: SchemaFacts, adapter, cfg, rng) -> Optional[Candidate]:
    """Questions requiring review/comment text: LIKE on a text column with a
    curated keyword, joined to a category aggregate."""
    # find a table with long text columns that is FK-linked to a parent
    for tname in facts.text_cols:
        text_col = rng.choice(facts.text_cols[tname])
        adapter_rows, _, _ = adapter.execute(
            f'SELECT COUNT(*) FROM "{tname}" WHERE "{text_col}" IS NOT NULL', max_rows=1
        )
        # find an FK from this table to elsewhere (to build a join)
        fks = [(c, rt, rc) for (tb, c, rt, rc) in facts.fk_edges if tb == tname]
        if not adapter_rows or not adapter_rows[0][0]:
            continue
        if not fks:
            # single-table datasets (no FK graph): category aggregate over
            # text-matching rows of the same table - still a review-text question
            keywords = cfg.extra.get("authoring", {}).get("review_keywords") or [
                "broken", "late", "good", "never arrived",
            ]
            kw = _safe_keyword(rng.choice(keywords))
            cats = facts.categorical_cols.get(tname)
            if not cats:
                return None
            cat = rng.choice(cats)
            sql = (
                f'SELECT "{cat}", COUNT(*) AS n\n'
                f'FROM "{tname}"\n'
                f'WHERE "{text_col}" LIKE \'%{kw}%\'\n'
                f'GROUP BY "{cat}"\n'
                f'ORDER BY n DESC'
            )
            q = f"Count of {tname} records whose {text_col.replace('_', ' ')} mention '{kw}', by {cat.replace('_', ' ')}."
            return Candidate(q, sql, "review_text", "hard", {"table": tname, "keyword": kw})
        c, rt, rc = rng.choice(fks)
        keywords = cfg.extra.get("authoring", {}).get("review_keywords") or [
            "broken", "late", "good", "never arrived",
        ]
        kw = _safe_keyword(rng.choice(keywords))
        parent_cat = (facts.categorical_cols.get(rt) or [rc])[0]
        sql = (
            f"SELECT d.\"{parent_cat}\", COUNT(*) AS n\n"
            f"FROM \"{tname}\" r\n"
            f"JOIN \"{rt}\" d ON r.\"{c}\" = d.\"{rc}\"\n"
            f"WHERE r.\"{text_col}\" ILIKE '%{kw}%'\n"
            f"GROUP BY d.\"{parent_cat}\"\n"
            f"ORDER BY n DESC"
        )
        q = f"Count of {tname} records whose {text_col.replace('_', ' ')} mention '{kw}', by {parent_cat.replace('_', ' ')}."
        return Candidate(q, sql, "review_text", "hard", {"table": tname, "keyword": kw})
    return None


def _safe_keyword(kw: str) -> str:
    """Keywords are inlined into LIKE patterns - strip quote wildcards."""
    return str(kw).replace("'", "").replace("%", "").replace("_", "")


GENERATORS: dict[str, Callable] = {
    "per_order_lookup": gen_per_order_lookup,
    "aggregation": gen_aggregate,
    "multi_table_join": gen_multi_table_join,
    "review_text": gen_review_text,
}


def generate_candidates(
    adapter, schema, cfg: DatasetConfig, n: int, seed: Optional[int] = None, categories=None
) -> list[Candidate]:
    rng = random.Random(cfg.seed if seed is None else seed)
    facts = discover_facts(adapter, schema)
    cats = categories or list(GENERATORS)
    out: list[Candidate] = []
    seen_sql: set[str] = set()
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        cat = rng.choice(cats)
        cand = GENERATORS[cat](facts, adapter, cfg, rng)
        if cand is None:
            continue
        # validate: executes + non-empty result
        try:
            rows, cols, _ = adapter.safe_execute(cand.gold_sql)
        except Exception:
            continue
        if not rows:
            continue
        if cand.gold_sql in seen_sql:
            continue
        seen_sql.add(cand.gold_sql)
        out.append(cand)
    return out