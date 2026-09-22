"""Shared test fixture: tiny e-commerce toy DB (SQLite) with a known cutoff."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from qwery_smith.adapters.sqlite_adapter import SQLiteAdapter

# 12 orders spanning 2017-01 .. 2018-09; cutoff (max month floor - 6m) = 2018-04-01
ORDERS = [
    ("o01", "c01", "delivered", "2017-01-15 10:00:00"),
    ("o02", "c02", "delivered", "2017-03-20 11:00:00"),
    ("o03", "c03", "shipped", "2017-06-10 09:30:00"),
    ("o04", "c01", "delivered", "2017-08-05 14:00:00"),
    ("o05", "c04", "canceled", "2017-11-11 08:00:00"),
    ("o06", "c02", "delivered", "2018-01-02 12:00:00"),
    ("o07", "c05", "delivered", "2018-02-14 16:30:00"),
    ("o08", "c03", "delivered", "2018-03-01 09:00:00"),
    ("o09", "c01", "delivered", "2018-04-10 10:00:00"),   # held-out window
    ("o10", "c04", "delivered", "2018-05-20 18:00:00"),   # held-out
    ("o11", "c02", "shipped", "2018-08-30 07:45:00"),    # held-out
    ("o12", "c05", "delivered", "2018-09-17 21:10:00"),   # held-out (max)
]

ITEMS = [
    ("o01", 1, "p01", 10.5, 2.0),
    ("o01", 2, "p02", 20.0, 3.0),
    ("o02", 1, "p01", 10.5, 2.0),
    ("o03", 1, "p03", 55.0, 7.5),
    ("o04", 1, "p02", 20.0, 3.0),
    ("o05", 1, "p01", 10.5, 2.0),
    ("o06", 1, "p03", 55.0, 7.5),
    ("o07", 1, "p02", 20.0, 3.0),
    ("o08", 1, "p01", 10.5, 2.0),
    ("o09", 1, "p03", 55.0, 7.5),     # held-out
    ("o10", 1, "p01", 10.5, 2.0),     # held-out
    ("o11", 1, "p02", 20.0, 3.0),     # held-out
    ("o12", 1, "p03", 55.0, 7.5),     # held-out
]


@pytest.fixture
def toy_db(tmp_path: Path) -> SQLiteAdapter:
    db = tmp_path / "toy.db"
    adapter = SQLiteAdapter(db, read_only=False)
    adapter.conn.execute(
        "CREATE TABLE orders (order_id TEXT, customer_id TEXT, status TEXT, purchase_ts TEXT)"
    )
    adapter.conn.execute(
        "CREATE TABLE order_items (order_id TEXT, item INTEGER, product_id TEXT, price REAL, freight REAL)"
    )
    adapter.conn.executemany("INSERT INTO orders VALUES (?,?,?,?)", ORDERS)
    adapter.conn.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", ITEMS)
    adapter.conn.commit()
    adapter.add_primary_key("orders", ["order_id"])
    adapter.add_foreign_key("order_items", "order_id", "orders", "order_id")
    return adapter


def _toy_config():
    from qwery_smith.config import DatasetConfig, DatasourceConfig, HoldoutConfig

    return DatasetConfig(
        name="toy",
        license="CC0 (synthetic)",
        source_url="fixture",
        question_file=Path("questions_v1.jsonl"),
        holdout=HoldoutConfig(column="orders.purchase_ts", months=6),
        datasource=DatasourceConfig(
            csv_dir=Path("raw"), csv_pattern="*.csv", table_map={}, uri="sqlite:///", dialect="sqlite"
        ),
    )