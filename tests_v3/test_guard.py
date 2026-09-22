"""AST guard tests - the adversarial cases a regex guard would miss."""

import pytest

from qwery_smith.adapters.guard import assert_safe_select

SAFE = [
    "SELECT 1",
    "SELECT * FROM orders",
    "WITH x AS (SELECT * FROM orders) SELECT * FROM x",
    "SELECT * FROM orders UNION SELECT * FROM orders",
    "select count(*) from orders where status = 'delivered'",
]

UNSAFE = [
    # regex-guard bypasses from the v1.x threat model
    ("DROP TABLE orders", "root"),
    ("SELECT 1; DROP TABLE orders", "multiple"),
    ("UPDATE orders SET status = 'x'", "root"),
    ("DELETE FROM orders", "root"),
    ("INSERT INTO orders VALUES (1)", "root"),
    ("CREATE TABLE evil (x int)", "root"),
    ("ALTER TABLE orders ADD COLUMN x int", "root"),
    ("TRUNCATE TABLE orders", "root"),
    ("SELECT * INTO evil FROM orders", "INTO"),
    ("WITH x AS (DELETE FROM orders RETURNING *) SELECT * FROM x", "CTE"),
    # NOTE: pg_sleep(100) passes the AST guard BY DESIGN - it's a latency DoS,
    # handled by the next defense layers (statement_timeout=30s + read-only
    # role + read-only transaction), not by structural mutation checking.
]


@pytest.mark.parametrize("sql", SAFE)
def test_safe(sql):
    assert assert_safe_select(sql) is None


@pytest.mark.parametrize(("sql", "tag"), UNSAFE)
def test_unsafe(sql, tag):
    reason = assert_safe_select(sql)
    assert reason is not None, f"{tag}: should be rejected"


def test_comment_obfuscated_drop():
    # classic regex-bypass: mutation hidden behind a comment
    reason = assert_safe_select("SELECT * FROM orders; -- DROP TABLE orders")
    assert reason is not None  # rejected: stacked statements

def test_string_literal_hides_keyword():
    # 'DROP' inside a literal must NOT trip the guard (false-positive check)
    assert assert_safe_select("SELECT 'DROP TABLE' AS msg") is None

def test_empty():
    assert assert_safe_select("") is not None
    assert assert_safe_select("   ") is not None