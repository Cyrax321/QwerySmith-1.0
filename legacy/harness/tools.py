#!/usr/bin/env python3
"""
harness/tools.py -- Core Database Tool Calls and Inspection Utilities for QwerySmith

Provides:
- Database execution sandbox with latency profiling and error trapping (`execute_query`)
- Schema introspection and DDL extraction (`get_schema`, `get_tables`, `get_table_counts`, `get_table_sample`)
- Synthetic enterprise database generator (`init_sample_db`)
- Intent router (`classify_intent`)
- Table formatting and SQL sanitization (`format_table`, `clean_sql`)
"""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# --------------------------------------------------------------------------
# 1. Sample Enterprise Database Generator
# --------------------------------------------------------------------------
def init_sample_db(db_path: str | Path = "company_store.db") -> Path:
    """Creates a rich, multi-table e-commerce enterprise database for testing."""
    path = Path(db_path).resolve()
    if path.exists() and path.stat().st_size > 1000:
        return path

    print(f"[Init] Initializing enterprise demo database: {path.name} ...")
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()

    cur.executescript("""
        DROP TABLE IF EXISTS reviews;
        DROP TABLE IF EXISTS order_items;
        DROP TABLE IF EXISTS orders;
        DROP TABLE IF EXISTS products;
        DROP TABLE IF EXISTS customers;

        CREATE TABLE customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            city TEXT NOT NULL,
            country TEXT NOT NULL,
            loyalty_tier TEXT CHECK(loyalty_tier IN ('Bronze', 'Silver', 'Gold', 'Platinum')),
            created_at DATE NOT NULL
        );

        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price DECIMAL(10, 2) NOT NULL,
            stock INTEGER NOT NULL,
            supplier_country TEXT NOT NULL
        );

        CREATE TABLE orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            order_date DATE NOT NULL,
            status TEXT CHECK(status IN ('Pending', 'Shipped', 'Delivered', 'Cancelled')),
            total_amount DECIMAL(10, 2) NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        );

        CREATE TABLE order_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price DECIMAL(10, 2) NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        );

        CREATE TABLE reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            customer_id INTEGER NOT NULL,
            rating INTEGER CHECK(rating BETWEEN 1 AND 5),
            comment TEXT,
            review_date DATE NOT NULL,
            FOREIGN KEY (product_id) REFERENCES products(id),
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        );
    """)

    customers = [
        (1, "Sophia Chen", "sophia@example.com", "San Francisco", "USA", "Platinum", "2021-03-15"),
        (2, "Kenji Sato", "kenji@example.jp", "Tokyo", "Japan", "Platinum", "2020-07-22"),
        (3, "Amara Okafor", "amara@example.ng", "Lagos", "Nigeria", "Gold", "2022-01-10"),
        (4, "Lukas Weber", "lukas@example.de", "Berlin", "Germany", "Gold", "2021-11-05"),
        (5, "Elena Rostova", "elena@example.ru", "Moscow", "Russia", "Silver", "2023-04-18"),
        (6, "Mateo Silva", "mateo@example.br", "São Paulo", "Brazil", "Silver", "2022-09-30"),
        (7, "Aisha Al-Mansoor", "aisha@example.ae", "Dubai", "UAE", "Bronze", "2023-08-12"),
        (8, "Liam O'Connor", "liam@example.ie", "Dublin", "Ireland", "Bronze", "2024-01-02"),
    ]
    cur.executemany("INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?, ?);", customers)

    products = [
        (1, "MacBook Pro 16", "Electronics", 2499.00, 15, "USA"),
        (2, "Noise-Cancelling Headphones", "Electronics", 349.50, 45, "Japan"),
        (3, "Ergonomic Standing Desk", "Furniture", 750.00, 10, "Germany"),
        (4, "Mechanical Keyboard", "Accessories", 129.99, 80, "Taiwan"),
        (5, "Ultra-Wide 34-inch Monitor", "Electronics", 899.00, 25, "South Korea"),
        (6, "Wireless Ergonomic Mouse", "Accessories", 69.00, 120, "China"),
        (7, "Leather Executive Chair", "Furniture", 450.00, 18, "Italy"),
        (8, "USB-C Multi-Port Hub", "Accessories", 49.99, 200, "Vietnam"),
    ]
    cur.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?);", products)

    orders = [
        (1, 1, "2023-05-12", "Delivered", 2568.00),
        (2, 2, "2023-06-01", "Delivered", 2848.50),
        (3, 2, "2023-09-14", "Delivered", 400.50),
        (4, 3, "2023-11-20", "Delivered", 1248.50),
        (5, 4, "2024-01-15", "Delivered", 750.00),
        (6, 6, "2024-02-10", "Shipped", 129.99),
        (7, 7, "2024-03-05", "Pending", 899.00),
        (8, 1, "2024-03-18", "Delivered", 118.99),
    ]
    cur.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?);", orders)

    order_items = [
        (1, 1, 1, 1, 2499.00),
        (2, 1, 6, 1, 69.00),
        (3, 2, 1, 1, 2499.00),
        (4, 2, 2, 1, 349.50),
        (5, 3, 4, 1, 129.99),
        (6, 3, 8, 1, 49.99),
        (7, 4, 5, 1, 899.00),
        (8, 4, 2, 1, 349.50),
        (9, 5, 3, 1, 750.00),
        (10, 6, 4, 1, 129.99),
        (11, 7, 5, 1, 899.00),
        (12, 8, 6, 1, 69.00),
        (13, 8, 8, 1, 49.99),
    ]
    cur.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?, ?);", order_items)

    reviews = [
        (1, 1, 1, 5, "Unmatched battery life and performance for software engineering.", "2023-05-20"),
        (2, 2, 2, 4, "Great noise cancellation on long flights.", "2023-06-15"),
        (3, 5, 4, 5, "Amazing screen real-estate for productivity.", "2023-12-01"),
        (4, 3, 4, 5, "Motor is quiet and desk frame is sturdy.", "2024-01-22"),
        (5, 4, 6, 4, "Clicky switches with solid tactile feedback.", "2024-02-18"),
    ]
    cur.executemany("INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?);", reviews)

    conn.commit()
    conn.close()
    print(f"[OK] Demo database '{path.name}' created with 5 tables and populated data.\n")
    return path


# --------------------------------------------------------------------------
# 2. SQL Sanitization & Table Formatter
# --------------------------------------------------------------------------
def clean_sql(raw: str) -> str:
    """Strips Markdown blocks, comments, and trailing semicolons to leave pure SQL."""
    raw = re.sub(r"```(?:sql)?\s*", "", raw, flags=re.IGNORECASE)
    raw = raw.replace("```", "")
    raw = re.sub(r"^--.*$", "", raw, flags=re.MULTILINE)

    match = re.search(r"(SELECT\b.+)", raw, re.DOTALL | re.IGNORECASE)
    if match:
        raw = match.group(1)

    raw = raw.split(";")[0].strip()
    return raw + ";"


def format_table(columns: List[str], rows: List[Tuple[Any, ...]], max_rows: int = 50) -> str:
    """Renders tabular database records as an ASCII table."""
    if not columns and not rows:
        return "  (Empty result set)"

    str_rows = [[str(val) if val is not None else "NULL" for val in r] for r in rows[:max_rows]]
    col_widths = [len(c) for c in columns]
    for row in str_rows:
        for i, val in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(val))

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header = "| " + " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(columns)) + " |"

    lines = [sep, header, sep]
    for row in str_rows:
        line = "| " + " | ".join(row[i].ljust(col_widths[i]) if i < len(row) else "".ljust(col_widths[i]) for i in range(len(columns))) + " |"
        lines.append(line)
    lines.append(sep)

    if len(rows) > max_rows:
        lines.append(f"  ... ({len(rows) - max_rows} additional rows omitted)")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 3. Intent Classification Router
# --------------------------------------------------------------------------
GREETINGS = {
    "hi", "hey", "hello", "hola", "yo", "sup", "good morning", "good afternoon",
    "good evening", "who are you", "what are you", "what can you do", "help", ":help"
}


def classify_intent(text: str, table_names: Optional[List[str]] = None) -> str:
    """Classifies user input into:
    - 'COMMAND': Starts with ':' or exit keyword
    - 'REALTIME_SYS': Current time, date, environment status
    - 'DB_META': Asking about tables, schema, database row counts
    - 'DB_QUERY': Asking to query, filter, aggregate, or calculate database records
    - 'CONVERSATIONAL': General chit-chat, greetings, SQL concepts, reasoning
    """
    clean = text.strip().lower()
    clean_alpha = re.sub(r"[^\w\s]", "", clean)

    if clean.startswith(":") or clean in ["exit", "quit", "q"]:
        return "COMMAND"

    if clean_alpha in GREETINGS:
        return "CONVERSATIONAL"

    time_keywords = ["what time", "current time", "what date", "todays date", "today's date", "what day is it"]
    if any(tk in clean for tk in time_keywords):
        return "REALTIME_SYS"

    meta_keywords = ["what tables", "list tables", "show tables", "database schema", "table list", "how many tables"]
    if any(mk in clean for mk in meta_keywords):
        return "DB_META"

    db_terms = {
        "customer", "customers", "order", "orders", "product", "products", "item", "items",
        "review", "reviews", "sales", "revenue", "price", "stock", "spent", "spending",
        "bought", "buy", "purchase", "purchases", "highest", "lowest", "average", "total",
        "sum", "count", "top", "tier", "platinum", "gold", "silver", "bronze", "country"
    }
    if table_names:
        for t in table_names:
            db_terms.add(t.lower())

    words = set(clean_alpha.split())
    if words & db_terms:
        return "DB_QUERY"

    query_phrases = [
        "who spent", "how many", "which customer", "which product", "list all", "show me all",
        "find all", "calculate the", "what is the total", "what are the top", "rank the"
    ]
    if any(qp in clean for qp in query_phrases):
        return "DB_QUERY"

    return "CONVERSATIONAL"


# --------------------------------------------------------------------------
# 4. Database Sandbox Execution & Introspection Tool Calls
# --------------------------------------------------------------------------
def _get_connection(conn_or_path: Union[sqlite3.Connection, str, Path]) -> Tuple[sqlite3.Connection, bool]:
    """Returns (conn, should_close)."""
    if isinstance(conn_or_path, sqlite3.Connection):
        return conn_or_path, False
    return sqlite3.connect(str(conn_or_path)), True


def execute_query(
    conn_or_path: Union[sqlite3.Connection, str, Path],
    sql: str,
    max_rows: int = 100,
    timeout_sec: float = 3.0,
    read_only: bool = False,
) -> Dict[str, Any]:
    """
    Executes a SQL query safely inside an isolated transaction with microsecond timing.
    Returns: {columns, rows, latency_exec_ms, error, success}
    """
    from .security.sandbox import execute_sandboxed_query
    return execute_sandboxed_query(
        conn_or_path=conn_or_path,
        sql=sql,
        max_rows=max_rows,
        timeout_sec=timeout_sec,
        read_only=read_only,
    )


def get_schema(conn_or_path: Union[sqlite3.Connection, str, Path]) -> str:
    """Extracts clean CREATE TABLE DDL definitions for all user tables in the SQLite database."""
    conn, should_close = _get_connection(conn_or_path)
    cur = conn.cursor()
    cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [row[0] for row in cur.fetchall() if row[0]]
    if should_close:
        conn.close()
    return "\n".join(tables)


def get_tables(conn_or_path: Union[sqlite3.Connection, str, Path]) -> List[str]:
    """Returns list of active table names."""
    conn, should_close = _get_connection(conn_or_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [r[0] for r in cur.fetchall()]
    if should_close:
        conn.close()
    return tables


def get_table_counts(conn_or_path: Union[sqlite3.Connection, str, Path]) -> Dict[str, int]:
    """Returns row counts for each active table."""
    conn, should_close = _get_connection(conn_or_path)
    tables = get_tables(conn)
    counts = {}
    cur = conn.cursor()
    for t in tables:
        cur.execute(f"SELECT COUNT(*) FROM {t};")
        counts[t] = cur.fetchone()[0]
    if should_close:
        conn.close()
    return counts


def get_table_sample(
    conn_or_path: Union[sqlite3.Connection, str, Path],
    table_name: str,
    limit: int = 3,
) -> Dict[str, Any]:
    """Fetches sample rows and column definitions for a given table."""
    conn, should_close = _get_connection(conn_or_path)
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT * FROM {table_name} LIMIT {limit};")
        columns = [d[0] for d in cur.description]
        rows = cur.fetchall()
        if should_close:
            conn.close()
        return {"table": table_name, "columns": columns, "rows": rows, "error": None}
    except Exception as e:
        if should_close:
            conn.close()
        return {"table": table_name, "columns": [], "rows": [], "error": str(e)}
