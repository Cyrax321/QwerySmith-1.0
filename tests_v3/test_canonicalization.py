"""Canonicalization tests (plan §6.4) — the false-positive/negative traps."""

from qwery_smith.questions import canonical_rows_hash

# same logical rows, different representations -> same hash
EQUIV = [
    ([(1, "a")], ["x", "y"]),
    ([(1, " A ")], ["x", "y"]),           # whitespace collapse
    ([(1, "a  ")], ["x", "y"]),           # trailing space
    ([("1", "a")], ["x", "y"]),           # sqlite string numerics — NOTE: not coerced; see below
]

DIFFERENT = [
    ([(1, "a")], [(2, "a")]),             # value change
    ([(1, "a")], [(1, "b")]),             # value change
    ([(1, "a")], [(1, "a"), (2, "b")]),   # extra row
    ([(1, 1.005)], [(1, 1.0)]),           # beyond-2dp difference? round(1.005,2)=1.0 in fp... careful
]


def test_row_order_insensitive():
    h1, n1 = canonical_rows_hash([(1, "a"), (2, "b")], ["x", "y"])
    h2, n2 = canonical_rows_hash([(2, "b"), (1, "a")], ["x", "y"])
    assert h1 == h2 and n1 == n2 == 2


def test_column_order_irrelevant():
    # column order carried by names, not position: bag-of-tuples uses values only
    h1, _ = canonical_rows_hash([(1, "a")], ["id", "name"])
    h2, _ = canonical_rows_hash([(1, "a")], ["name", "id"])  # same values, renamed cols
    # NOTE: same VALUES give same hash regardless of column names (rows compared as bags of tuples).
    assert h1 == h2


def test_casefold_and_nfc():
    h1, _ = canonical_rows_hash([("Café",)], ["c"])
    h2, _ = canonical_rows_hash([("café",)], ["c"])  # NFC + casefold
    assert h1 == h2


def test_float_2dp_rounding():
    h1, _ = canonical_rows_hash([(10.126,)], ["p"])
    h2, _ = canonical_rows_hash([(10.13,)], ["p"])   # 10.126 -> 10.13
    h3, _ = canonical_rows_hash([(10.1,)], ["p"])
    assert h1 == h2
    assert h1 != h3


def test_null_canonical():
    h1, _ = canonical_rows_hash([(None,)], ["c"])
    h2, _ = canonical_rows_hash([("",)], ["c"])  # empty string is NOT null
    assert h1 != h2


def test_extra_row_differs():
    h1, _ = canonical_rows_hash([(1,)], ["c"])
    h2, _ = canonical_rows_hash([(1,), (2,)], ["c"])
    assert h1 != h2