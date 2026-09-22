"""AST SELECT-only guard via sqlglot (plan §6.3).

Regex guards are bypassable (comments, string literals, CTE-hidden mutations);
this walks the parsed AST and rejects anything that is not a plain read.
"""

from __future__ import annotations

from typing import Optional

import sqlglot
from sqlglot import exp


def assert_safe_select(sql: str, dialect: str = "sqlite") -> Optional[str]:
    """Return None if safe, else a one-line rejection reason."""
    if not sql or not sql.strip():
        return "empty SQL"

    try:
        statements = sqlglot.parse(sql, read=dialect)
    except sqlglot.errors.ParseError as e:
        return f"parse error: {str(e).splitlines()[0]}"

    if not statements:
        return "no statements parsed"
    if len(statements) > 1:
        return "multiple stacked statements"

    tree = statements[0]
    if tree is None:
        return "no statements parsed"

    if not isinstance(tree, exp.Select) and not (
        isinstance(tree, exp.Union) or isinstance(tree, (exp.Except, exp.Intersect))
    ):
        return f"root must be SELECT/WITH-SELECT/set-op, got {type(tree).__name__}"

    for node in tree.walk():
        node = node[0] if isinstance(node, tuple) else node
        if isinstance(node, (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Alter,
                             exp.Create, exp.TruncateTable, exp.Merge, exp.Command)):
            return f"forbidden node: {type(node).__name__}"
        if isinstance(node, exp.Into):
            return "SELECT ... INTO is forbidden"
        if isinstance(node, (exp.Insert,)):
            return "INSERT forbidden"
        # Locking reads
        if isinstance(node, exp.Lock) or (isinstance(node, exp.Select) and node.args.get("lock")):
            return "locking reads (FOR UPDATE) forbidden"

    return None