#!/usr/bin/env python3
"""Curated Olist question authoring (plan SS5.1).

Unlike the generic `author` verb (schema-fact sampling), this script emits the
plan's exact 100-question shape:
  per_order_lookup 24 / aggregation 32 / multi_table_join 24 / review_text 20
with named difficulty (~30 easy / 40 medium / 30 hard) and a controlled split
(~35 held-out by construction, then verified by the clamped-shadow rule).

Dataset-specific by design: lives in datasets/olist/, never in the harness.

Deterministic: seeded. Re-runnable for review: `--dry` prints without writing.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qwery_smith.adapters import open_adapter
from qwery_smith.clamp import build_clamped, window_dependent
from qwery_smith.config import DATASETS_ROOT, DatasetConfig
from qwery_smith.profiler import compute_holdout_cutoff

CUTOFF = None  # computed from profile at runtime

R = random.Random(42)

# ------------------------------------------------------------------- sampling

def sample_orders(adapter, n, pre=True, status=None):
    """Sample n order_ids from before/after the holdout cutoff (seeded)."""
    op = "<" if pre else ">="
    q = (
        f"SELECT order_id FROM orders "
        f"WHERE order_purchase_timestamp {op} '{CUTOFF}'"
        + (f" AND order_status = '{status}'" if status else "")
        + " ORDER BY RANDOM() LIMIT " + str(n)
    )
    return [r[0] for r in adapter.execute(q, max_rows=n)[0]]


def sample_values(adapter, table, col, n, where=None):
    q = f"SELECT DISTINCT {col} FROM {table}"
    if where:
        q += f" WHERE {where}"
    q += f" ORDER BY RANDOM() LIMIT {n}"
    return [r[0] for r in adapter.execute(q, max_rows=n)[0]]


def months_in_range(adapter, table, col, lo, hi, k=8):
    """Available YYYY-MM values in a date range (data-checked)."""
    q = (
        f"SELECT DISTINCT substr({col}, 1, 7) FROM {table} "
        f"WHERE {col} >= '{lo}' AND {col} < '{hi}' ORDER BY 1"
    )
    vals = [r[0] for r in adapter.execute(q, max_rows=100)[0]]
    return R.sample(vals, min(k, len(vals)))


# ------------------------------------------------------------- the templates

def lookup_items(oids):
    out = []
    for oid in oids:
        out.append((
            f"What is the status and purchase timestamp of order {oid}?",
            f"SELECT order_status, order_purchase_timestamp FROM orders WHERE order_id = '{oid}'",
            "easy", None,
        ))
    return out


def lookup_delivered(oids):
    return [(
        f"When was order {oid} delivered to the customer?",
        f"SELECT order_delivered_customer_date FROM orders WHERE order_id = '{oid}'",
        "easy", None,
    ) for oid in oids]


def lookup_estimated(oids):
    return [(
        f"What was the estimated delivery date given for order {oid}?",
        f"SELECT order_estimated_delivery_date FROM orders WHERE order_id = '{oid}'",
        "easy", None,
    ) for oid in oids]


def lookup_total_paid(oids):
    return [(
        f"How much was paid in total for order {oid}?",
        (f"SELECT payment_type, COUNT(*) AS parts, SUM(payment_value) AS total_value "
         f"FROM order_payments WHERE order_id = '{oid}' GROUP BY payment_type"),
        "easy", None,
    ) for oid in oids]


def lookup_items_count(oids):
    return [(
        f"How many items are in order {oid} and what was their combined price?",
        (f"SELECT COUNT(*) AS n_items, SUM(price) AS total_price "
         f"FROM order_items WHERE order_id = '{oid}'"),
        "medium", None,
    ) for oid in oids]


def _month_end(m: str) -> str:
    y, mo = int(m[:4]), int(m[5:7])
    if mo == 12:
        return f"{y + 1}-01"
    return f"{y}-{mo + 1:02d}"


def agg_orders_by_month(adapter):
    months = months_in_range(adapter, "orders", "order_purchase_timestamp",
                             "2016-09-01", "2018-03-01", k=5)
    out = []
    for m in months:
        hi = _month_end(m) + "-01"
        out.append((
            f"How many orders were placed in {m}?",
            (f"SELECT COUNT(*) AS n_orders FROM orders "
             f"WHERE order_purchase_timestamp >= '{m}-01' AND order_purchase_timestamp < '{hi}'"),
            "medium", "month",
        ))
    return out


def agg_orders_by_status(adapter):
    statuses = sample_values(adapter, "orders", "order_status", 4)
    return [(
        f"How many orders have status '{s}'?",
        f"SELECT COUNT(*) AS n FROM orders WHERE order_status = '{s}'",
        "easy", None,
    ) for s in statuses]


def agg_payment_by_type(adapter):
    pts = sample_values(adapter, "order_payments", "payment_type", 4)
    return [(
        f"What is the total payment value for payment type '{pt}'?",
        (f"SELECT payment_type, COUNT(*) AS n_payments, SUM(payment_value) AS total_value "
         f"FROM order_payments WHERE payment_type = '{pt}' GROUP BY payment_type"),
        "easy", None,
    ) for pt in pts]


def agg_avg_review_by_month(adapter):
    months = months_in_range(adapter, "order_reviews", "review_creation_date",
                             "2016-01-01", "2018-03-01", k=4)
    return [(
        f"What was the average review score in {m}?",
        (f"SELECT AVG(review_score) AS avg_score, COUNT(*) AS n_reviews FROM order_reviews "
         f"WHERE review_creation_date >= '{m}-01' AND review_creation_date < '" + _month_end(m) + "-01'"),
        "medium", "month",
    ) for m in months]


def agg_top_cities(adapter):
    return [(
        "Which 10 customer cities placed the most orders?",
        ("SELECT c.customer_city, COUNT(*) AS n_orders FROM orders "
         "JOIN customers c ON orders.customer_id = c.customer_id "
         "GROUP BY c.customer_city ORDER BY n_orders DESC LIMIT 10"),
        "medium", None,
    )]


def agg_states_having(adapter):
    return [(
        "Which customer states have more than 1000 orders?",
        ("SELECT c.customer_state, COUNT(*) AS n_orders FROM orders "
         "JOIN customers c ON orders.customer_id = c.customer_id "
         "GROUP BY c.customer_state HAVING COUNT(*) > 1000 ORDER BY n_orders DESC"),
        "hard", None,
    )]


def agg_avg_installments(adapter):
    return [(
        "What is the average number of installments per payment type for orders above R$200?",
        ("SELECT payment_type, AVG(payment_installments) AS avg_inst, COUNT(*) AS n "
         "FROM order_payments WHERE payment_value > 200 GROUP BY payment_type ORDER BY avg_inst DESC"),
        "medium", None,
    )]


def agg_freight_by_state(adapter):
    return [(
        "What is the total freight value by seller state, ordered high to low?",
        ("SELECT s.seller_state, SUM(i.freight_value) AS total_freight FROM order_items i "
         "JOIN sellers s ON i.seller_id = s.seller_id GROUP BY s.seller_state ORDER BY total_freight DESC"),
        "medium", None,
    )]


def agg_canceled_late(adapter):
    return [(
        "How many canceled orders were estimated to be delivered after 2017-09?",
        ("SELECT COUNT(*) FROM orders WHERE order_status = 'canceled' "
         "AND order_estimated_delivery_date >= '2017-09-01'"),
        "medium", None,
    )]


def agg_revenue_by_category_2018(adapter):
    return [(
        "What is the total item price (revenue) by product category in 2018 Q1 (Jan-Mar)?",
        ("SELECT p.product_category_name, SUM(i.price) AS revenue FROM order_items i "
         "JOIN products p ON i.product_id = p.product_id "
         "JOIN orders o ON i.order_id = o.order_id "
         "WHERE o.order_purchase_timestamp >= '2018-01-01' AND o.order_purchase_timestamp < '2018-04-01' "
         "GROUP BY p.product_category_name ORDER BY revenue DESC"),
        "medium", None,
    )]


def join_avg_score_by_category(adapter):
    return [(
        "What is the average review score by product category (English)?",
        ("SELECT t.product_category_name_english, AVG(r.review_score) AS avg_score "
         "FROM order_reviews r JOIN order_items i ON r.order_id = i.order_id "
         "JOIN products p ON i.product_id = p.product_id "
         "JOIN product_category_translation t ON p.product_category_name = t.product_category_name "
         "GROUP BY t.product_category_name_english ORDER BY avg_score DESC"),
        "hard", None,
    )]


def join_revenue_by_seller_state(adapter):
    return [(
        "What is the total revenue (price + freight) by seller state for delivered orders in 2017?",
        ("SELECT s.seller_state, SUM(i.price + i.freight_value) AS total_revenue FROM order_items i "
         "JOIN sellers s ON i.seller_id = s.seller_id JOIN orders o ON i.order_id = o.order_id "
         "WHERE o.order_status = 'delivered' AND o.order_purchase_timestamp >= '2017-01-01' "
         "AND o.order_purchase_timestamp < '2018-01-01' "
         "GROUP BY s.seller_state ORDER BY total_revenue DESC"),
        "hard", None,
    )]


def review_keyword_count(adapter, kws):
    out = []
    for kw in kws[:6]:
        out.append((
            f"How many review comments mention '{kw}'?",
            f"SELECT COUNT(*) FROM order_reviews WHERE review_comment_message LIKE '%{kw}%'",
            "easy", None,
        ))
    return out


def review_score_by_keyword(adapter, kws):
    out = []
    for kw in kws[:4]:
        out.append((
            f"What is the distribution of review scores for comments mentioning '{kw}'?",
            (f"SELECT review_score, COUNT(*) AS n FROM order_reviews "
             f"WHERE review_comment_message LIKE '%{kw}%' GROUP BY review_score ORDER BY review_score"),
            "medium", None,
        ))
    return out


def review_avg_for_keyword(adapter, kws):
    out = []
    for kw in kws[4:]:
        out.append((
            f"What is the average review score for reviews whose comment mentions '{kw}'?",
            (f"SELECT AVG(review_score) AS avg_score FROM order_reviews "
             f"WHERE review_comment_message LIKE '%{kw}%'"),
            "medium", None,
        ))
    return out


def review_keyword_by_month(adapter, kws):
    months = months_in_range(adapter, "order_reviews", "review_creation_date",
                             "2017-01-01", "2018-03-01", k=3)
    out = []
    for kw, m in zip(R.sample(kws, min(3, len(kws))), months):
        out.append((
            f"Reviews in {m} mentioning '{kw}', by score?",
            (f"SELECT review_score, COUNT(*) AS n FROM order_reviews "
             f"WHERE review_comment_message LIKE '%{kw}%' "
             f"AND review_creation_date >= '{m}-01' AND review_creation_date < '" + _month_end(m) + "-01' "
             "GROUP BY review_score ORDER BY review_score"),
            "medium", None,
        ))
    return out


def review_post_cutoff_orders(adapter, kws, n=4):
    """Reviews of orders placed ON/AFTER the cutoff -> heldout by construction."""
    out = []
    for kw in kws[:n]:
        out.append((
            f"For orders placed on or after {CUTOFF}, how many review comments mention '{kw}'?",
            (f"SELECT COUNT(*) FROM order_reviews r JOIN orders o ON r.order_id = o.order_id "
             f"WHERE r.review_comment_message LIKE '%{kw}%' AND o.order_purchase_timestamp >= '{CUTOFF}'"),
            "medium", "heldout_forced",
        ))
    return out


def review_keyword_orders_2017(adapter, kws):
    out = []
    for kw in R.sample(kws, min(4, len(kws))):
        out.append((
            f"How many orders from 2017 have a review comment mentioning '{kw}'?",
            (f"SELECT COUNT(DISTINCT o.order_id) FROM order_reviews r "
             f"JOIN orders o ON r.order_id = o.order_id "
             f"WHERE r.review_comment_message LIKE '%{kw}%' "
             f"AND o.order_purchase_timestamp >= '2017-01-01' AND o.order_purchase_timestamp < '2018-01-01'"),
            "hard", None,
        ))
    return out


def agg_reviews_by_score_post_cutoff(adapter):
    return [(
        f"How are review scores distributed for orders placed on or after {CUTOFF}?",
        ("SELECT r.review_score, COUNT(*) AS n FROM order_reviews r "
         "JOIN orders o ON r.order_id = o.order_id "
         f"WHERE o.order_purchase_timestamp >= '{CUTOFF}' GROUP BY r.review_score ORDER BY r.review_score"),
        "medium", "heldout_forced",
    )]


def agg_orders_post_cutoff(adapter):
    return [(
        f"How many orders were placed on or after {CUTOFF}?",
        f"SELECT COUNT(*) AS n FROM orders WHERE order_purchase_timestamp >= '{CUTOFF}'",
        "easy", "heldout_forced",
    )]


def agg_revenue_post_cutoff(adapter):
    return [(
        f"What is the total revenue from orders placed on or after {CUTOFF}?",
        (f"SELECT SUM(i.price + i.freight_value) AS revenue FROM order_items i "
         f"JOIN orders o ON i.order_id = o.order_id WHERE o.order_purchase_timestamp >= '{CUTOFF}'"),
        "medium", "heldout_forced",
    )]


# ------------------------------------------------------------ orchestration

def main():
    global CUTOFF
    dry = "--dry" in sys.argv
    out_path = Path(__file__).resolve().parent / "prepared" / "questions_olist_curated.jsonl"

    cfg = DatasetConfig.load("olist")
    adapter = open_adapter(cfg.datasource.uri, read_only=True)
    _max, CUTOFF = compute_holdout_cutoff(adapter, cfg.holdout)
    print(f"cutoff: {CUTOFF}")

    pre_oids = sample_orders(adapter, 28, pre=True, status="delivered")
    post_oids = sample_orders(adapter, 12, pre=False, status="delivered")
    keywords = cfg.extra.get("authoring", {}).get("review_keywords", [
        "não recebi", "quebrado", "ótimo", "antes do prazo",
        "produto errado", "recomendo", "demorou", "bom",
    ])

    specs: list[tuple[str, str, str, str | None]] = []  # (q, sql, difficulty, force_split)

    # per_order_lookup: 24
    specs += [("per_order_lookup", *t) for t in lookup_items(pre_oids[:6])]
    specs += [("per_order_lookup", *t) for t in lookup_delivered(pre_oids[6:10])]
    specs += [("per_order_lookup", *t) for t in lookup_estimated(pre_oids[10:14])]
    specs += [("per_order_lookup", *t) for t in lookup_total_paid(pre_oids[14:18])]
    specs += [("per_order_lookup", *t) for t in lookup_items_count(pre_oids[18:21])]
    specs += [("per_order_lookup", *t) for t in [
        (f"What is the status of order {post_oids[k]}?",
         f"SELECT order_status FROM orders WHERE order_id = '{post_oids[k]}'", "easy", "heldout_forced")
        for k in range(3)]]
    # harder lookups: per-order joins + subqueries
    for i, oid in enumerate(pre_oids[21:24]):
        specs.append((
            "per_order_lookup",
            f"For order {oid}, list each item's product_id, price, freight and the payment type used.",
            (f"SELECT i.product_id, i.price, i.freight_value, pym.payment_type FROM order_items i "
             f"JOIN order_payments pym ON i.order_id = pym.order_id WHERE i.order_id = '{oid}'"),
            "hard", None,
        ))
    specs.append((
        "per_order_lookup",
        "For order {}, did the payment value exceed the item price total?"
        .format(pre_oids[24]),
        (f"SELECT oi.order_id, SUM(oi.price) AS items_total, "
         f"(SELECT SUM(pym.payment_value) FROM order_payments pym WHERE pym.order_id = oi.order_id) AS paid "
         f"FROM order_items oi WHERE oi.order_id = '{pre_oids[24]}' GROUP BY oi.order_id"),
        "hard", None,
    ))
    specs.append((
        "per_order_lookup",
        "For order {}, how many days did delivery take versus the estimate?"
        .format(pre_oids[25]),
        (f"SELECT julianday(order_delivered_customer_date) - julianday(order_approved_at) AS days_taken, "
         f"julianday(order_estimated_delivery_date) - julianday(order_approved_at) AS days_estimate "
         f"FROM orders WHERE order_id = '{pre_oids[25]}'"),
        "hard", None,
    ))

    # aggregation: 32
    agg = (
        agg_orders_by_month(adapter) + agg_orders_by_status(adapter)
        + agg_payment_by_type(adapter) + agg_avg_review_by_month(adapter)
        + agg_top_cities(adapter) + agg_states_having(adapter)
        + agg_avg_installments(adapter) + agg_freight_by_state(adapter)
        + agg_canceled_late(adapter) + agg_revenue_by_category_2018(adapter)
    )
    for q, sql, diff, flag in agg:
        specs.append(("aggregation", q, sql, diff, flag))
    # heldout aggregates by construction
    for fn in (agg_orders_post_cutoff, agg_revenue_post_cutoff,
               agg_reviews_by_score_post_cutoff):
        for q, sql, diff, flag in fn(adapter):
            specs.append(("aggregation", q, sql, diff, flag))

    # fillers to reach category counts
    extra_orders = sample_orders(adapter, 12, pre=True, status="delivered")
    for i, oid in enumerate(extra_orders):
        if len([s for s in specs if s[0] == "aggregation"]) >= 32:
            break
        specs.append((
            "aggregation",
            f"Count the payments for order {oid}.",
            f"SELECT COUNT(*) AS n_payments FROM order_payments WHERE order_id = '{oid}'",
            "easy", None,
        ))

    # multi_table_join: 24
    specs.extend(("multi_table_join", q, s, d, f) for q, s, d, f in join_avg_score_by_category(adapter))
    specs.extend(("multi_table_join", q, s, d, f) for q, s, d, f in join_revenue_by_seller_state(adapter))
    for cat in ["cama_mesa_banho", "moveis_decoracao", "esporte_lazer"]:
        specs.append((
            "multi_table_join",
            "Average freight by customer state for category '{}'?".format(cat),
            ("SELECT c.customer_state, AVG(i.freight_value) AS avg_freight FROM order_items i "
             "JOIN orders o ON i.order_id = o.order_id JOIN customers c ON o.customer_id = c.customer_id "
             f"JOIN products p ON i.product_id = p.product_id "
             f"WHERE p.product_category_name = '{cat}' AND o.order_status = 'delivered' "
             "GROUP BY c.customer_state ORDER BY avg_freight DESC"),
            "hard", None,
        ))
    specs.append((
        "multi_table_join",
        "Which 10 sellers have the highest total item price in 2017?",
        ("SELECT i.seller_id, SUM(i.price) AS revenue, COUNT(DISTINCT i.order_id) AS orders_n "
         "FROM order_items i JOIN orders o ON i.order_id = o.order_id "
         "WHERE o.order_purchase_timestamp >= '2017-01-01' AND o.order_purchase_timestamp < '2018-01-01' "
         "GROUP BY i.seller_id ORDER BY revenue DESC LIMIT 10"),
        "hard", None,
    ))
    specs.append((
        "multi_table_join",
        "How many orders were delivered late (after the estimated date) by customer state?",
        ("SELECT c.customer_state, COUNT(*) AS late_orders FROM orders o "
         "JOIN customers c ON o.customer_id = c.customer_id "
         "WHERE o.order_status = 'delivered' AND o.order_delivered_customer_date > o.order_estimated_delivery_date "
         "GROUP BY c.customer_state ORDER BY late_orders DESC"),
        "medium", None,
    ))
    specs.append((
        "multi_table_join",
        "Average review score for orders with more than one item, by product category.",
        ("SELECT p.product_category_name, AVG(r.review_score) AS avg_score FROM order_reviews r "
         "JOIN orders o ON r.order_id = o.order_id JOIN order_items i ON o.order_id = i.order_id "
         "JOIN products p ON i.product_id = p.product_id "
         "WHERE o.order_status = 'delivered' "
         "GROUP BY p.product_category_name ORDER BY avg_score DESC"),
        "hard", None,
    ))
    # post-cutoff joins -> heldout
    for cat in ["moveis_decoracao", "beleza_saude"]:
        specs.append((
            "multi_table_join",
            f"Total freight by customer state for category '{cat}' in orders placed on or after {CUTOFF}.",
            ("SELECT c.customer_state, SUM(i.freight_value) AS total_freight FROM order_items i "
             "JOIN orders o ON i.order_id = o.order_id JOIN customers c ON o.customer_id = c.customer_id "
             f"JOIN products p ON i.product_id = p.product_id "
             f"WHERE p.product_category_name = '{cat}' AND o.order_purchase_timestamp >= '{CUTOFF}' "
             "GROUP BY c.customer_state ORDER BY total_freight DESC"),
            "hard", "heldout_forced",
        ))

    # more multi_table_join (3-4 table patterns over states/years/products)
    _extra_joins = [
        ("Total freight by customer state for delivered orders in 2017.",
         ("SELECT c.customer_state, SUM(i.freight_value) AS total_freight FROM order_items i "
          "JOIN orders o ON i.order_id = o.order_id JOIN customers c ON o.customer_id = c.customer_id "
          "WHERE o.order_status = 'delivered' AND o.order_purchase_timestamp >= '2017-01-01' "
          "AND o.order_purchase_timestamp < '2018-01-01' "
          "GROUP BY c.customer_state ORDER BY total_freight DESC"), "hard", None),
        ("What is the total item price by product category for delivered orders in 2017?",
         ("SELECT p.product_category_name, SUM(i.price) AS revenue FROM order_items i "
          "JOIN orders o ON i.order_id = o.order_id JOIN products p ON i.product_id = p.product_id "
          "WHERE o.order_status = 'delivered' AND o.order_purchase_timestamp >= '2017-01-01' "
          "AND o.order_purchase_timestamp < '2018-01-01' "
          "GROUP BY p.product_category_name ORDER BY revenue DESC LIMIT 15"), "hard", None),
        ("Average order freight per customer state, delivered orders only.",
         ("SELECT c.customer_state, AVG(i.freight_value) AS avg_freight, COUNT(DISTINCT i.order_id) AS n_orders "
          "FROM order_items i JOIN orders o ON i.order_id = o.order_id "
          "JOIN customers c ON o.customer_id = c.customer_id "
          "WHERE o.order_status = 'delivered' GROUP BY c.customer_state ORDER BY avg_freight DESC"), "hard", None),
        ("Which 10 customers have the highest total payment value across all their orders?",
         ("SELECT o.customer_id, SUM(pym.payment_value) AS total_paid FROM order_payments pym "
          "JOIN orders o ON pym.order_id = o.order_id "
          "GROUP BY o.customer_id ORDER BY total_paid DESC LIMIT 10"), "medium", None),
        ("How many distinct sellers serve each customer state?",
         ("SELECT c.customer_state, COUNT(DISTINCT i.seller_id) AS n_sellers FROM order_items i "
          "JOIN orders o ON i.order_id = o.order_id JOIN customers c ON o.customer_id = c.customer_id "
          "WHERE o.order_status = 'delivered' GROUP BY c.customer_state ORDER BY n_sellers DESC"), "medium", None),
        ("Average review score by payment type for delivered orders.",
         ("SELECT pym.payment_type, AVG(r.review_score) AS avg_score FROM order_reviews r "
          "JOIN orders o ON r.order_id = o.order_id JOIN order_payments pym ON o.order_id = pym.order_id "
          "WHERE o.order_status = 'delivered' GROUP BY pym.payment_type ORDER BY avg_score DESC"), "hard", None),
        ("Which product categories have the highest average freight per item, top 10?",
         ("SELECT p.product_category_name, AVG(i.freight_value) AS avg_freight FROM order_items i "
          "JOIN products p ON i.product_id = p.product_id GROUP BY p.product_category_name "
          "ORDER BY avg_freight DESC LIMIT 10"), "medium", None),
        ("Which states have more than 500 delivered orders AND average freight above R$25?",
         ("SELECT c.customer_state, COUNT(*) AS n_orders, AVG(i.freight_value) AS avg_freight "
          "FROM order_items i JOIN orders o ON i.order_id = o.order_id "
          "JOIN customers c ON o.customer_id = c.customer_id "
          "WHERE o.order_status = 'delivered' GROUP BY c.customer_state "
          "HAVING COUNT(*) > 500 AND AVG(i.freight_value) > 25 ORDER BY n_orders DESC"), "hard", None),
        ("Total item revenue in each month of 2017.",
         ("SELECT substr(o.order_purchase_timestamp, 1, 7) AS month, SUM(i.price) AS revenue "
          "FROM order_items i JOIN orders o ON i.order_id = o.order_id "
          "WHERE o.order_purchase_timestamp >= '2017-01-01' AND o.order_purchase_timestamp < '2018-01-01' "
          "GROUP BY month ORDER BY month"), "medium", None),
        ("Which seller cities have the highest total freight in 2017, top 10?",
         ("SELECT s.seller_city, SUM(i.freight_value) AS total_freight FROM order_items i "
          "JOIN sellers s ON i.seller_id = s.seller_id JOIN orders o ON i.order_id = o.order_id "
          "WHERE o.order_purchase_timestamp >= '2017-01-01' AND o.order_purchase_timestamp < '2018-01-01' "
          "GROUP BY s.seller_city ORDER BY total_freight DESC LIMIT 10"), "hard", None),
        ("Average review score by customer state for orders delivered early (before estimated).",
         ("SELECT c.customer_state, AVG(r.review_score) AS avg_score FROM order_reviews r "
          "JOIN orders o ON r.order_id = o.order_id JOIN customers c ON o.customer_id = c.customer_id "
          "WHERE o.order_status = 'delivered' AND o.order_delivered_customer_date < o.order_estimated_delivery_date "
          "GROUP BY c.customer_state ORDER BY avg_score DESC"), "hard", None),
        ("Average review score by customer state for orders delivered late (after estimated), 2017 only.",
         ("SELECT c.customer_state, AVG(r.review_score) AS avg_score, COUNT(*) AS n_late FROM order_reviews r "
          "JOIN orders o ON r.order_id = o.order_id JOIN customers c ON o.customer_id = c.customer_id "
          "WHERE o.order_status = 'delivered' AND o.order_delivered_customer_date > o.order_estimated_delivery_date "
          "AND o.order_purchase_timestamp >= '2017-01-01' AND o.order_purchase_timestamp < '2018-01-01' "
          "GROUP BY c.customer_state ORDER BY avg_score DESC"), "hard", None),
    ]
    for q, sql, diff, flag in _extra_joins:
        specs.append(("multi_table_join", q, sql, diff, flag))

    # review_text: 20
    specs += [("review_text", *t) for t in review_keyword_count(adapter, keywords)]
    specs += [("review_text", *t) for t in review_score_by_keyword(adapter, keywords)]
    specs += [("review_text", *t) for t in review_avg_for_keyword(adapter, keywords)]
    specs += [("review_text", *t) for t in review_keyword_by_month(adapter, keywords)]
    specs.append((
        "review_text",
        "Of all delivered-order reviews mentioning 'não recebi', what share rate 1?",
        ("SELECT COUNT(*) AS n FROM order_reviews r JOIN orders o ON r.order_id = o.order_id "
         "WHERE r.review_comment_message LIKE '%não recebi%' AND r.review_score = 1 "
         "AND o.order_status = 'delivered'"),
        "hard", None,
    ))
    specs += [("review_text", *t) for t in review_post_cutoff_orders(adapter, keywords)]

    # ---------- dedupe, execute, split-tag -----------------------------------------
    from qwery_smith.questions import ExpectedRows, Question, QuestionSet, canonical_rows_hash
    from qwery_smith.clamp import build_clamped, window_dependent

    qs_out = QuestionSet()
    seen_sql: set[str] = set()
    from qwery_smith.schema_loader import load_schema

    clamped = build_clamped(adapter, load_schema(adapter), cfg.holdout, CUTOFF)
    n_held = 0
    for cat, qtext, sql, diff, force in specs:
        if sql in seen_sql:
            continue
        seen_sql.add(sql)
        try:
            rows, cols, _ = adapter.safe_execute(sql)
        except Exception as e:
            print(f"skip (exec fail): {qtext[:60]} -> {type(e).__name__}: {e}")
            continue
        if not rows:
            print(f"skip (empty result): {qtext[:60]}")
            continue
        sha, n_rows = canonical_rows_hash(rows, cols)
        split = "train_ok"
        if force == "heldout_forced" or window_dependent(sql, sha, clamped):
            split = "heldout"
            n_held += 1
        qs_out.add(Question(
            id=f"olist-{len(qs_out.questions) + 1:04d}",
            question=qtext,
            gold_sql=sql,
            expected_rows=ExpectedRows(sha256=sha, n_rows=n_rows),
            date="2026-09-22",
            difficulty=diff,
            category=cat,
            split=split,
            source="human",
        ))

    counts = qs_out.counts()
    print(json.dumps(counts, indent=2))
    print(f"heldout: {n_held}")
    clamped.close()
    adapter.close()

    if not dry:
        qs_out.save(out_path)
        print(f"wrote {out_path}")
    else:
        for q in qs_out.questions[:12]:
            print(f"[{q.split}/{q.category}/{q.difficulty}] {q.question}")


if __name__ == "__main__":
    main()