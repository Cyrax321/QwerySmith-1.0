"""M2 tests: retrieval, packs, triples, citations, prompt contract."""

from qwery_smith.retrieval import (
    build_index,
    build_pack,
    freeze_pack,
    load_pack,
    tokenize,
)
from qwery_smith.schema_loader import load_schema


def test_tokenize():
    assert tokenize("Total freight for order e7cd1…? (A&B)") == [
        "total", "freight", "for", "order", "e7cd1", "a", "b"
    ]


def test_index_retrieval_ranks_relevant_row_first(toy_db):
    schema = load_schema(toy_db)
    idx = build_index(toy_db, schema)
    hits = idx.search("status shipped", top_k=5)
    assert hits, "no hits"
    top_table, top_rid = hits[0][1].table, hits[0][1].row_id
    assert top_table == "orders"
    # shipped orders are o03 and o11
    assert top_rid in {"o03", "o11"}


def test_product_id_lookup(toy_db):
    schema = load_schema(toy_db)
    idx = build_index(toy_db, schema)
    hits = idx.search("orders containing product p03", top_k=4)
    got_tables = [h[1].table for h in hits]
    assert "order_items" in got_tables
    assert all(h[1].values[2] == "p03" for h in hits if h[1].table == "order_items")


def test_pack_deterministic(toy_db, tmp_path):
    schema = load_schema(toy_db)
    idx = build_index(toy_db, schema)
    p1 = build_pack("q1", "status shipped orders", idx, schema, top_k=5)
    p2 = build_pack("q1", "status shipped orders", idx, schema, top_k=5)
    assert [r["row_id"] for r in p1.rows] == [r["row_id"] for r in p2.rows]


def test_freeze_load_roundtrip_and_integrity(toy_db, tmp_path):
    schema = load_schema(toy_db)
    idx = build_index(toy_db, schema)
    pack = build_pack("qX", "freight for o02", idx, schema, top_k=3)
    meta = freeze_pack(pack, tmp_path)
    loaded = load_pack(tmp_path, "qX", expected_sha256=meta["sha256"])
    assert loaded.question == pack.question
    assert [r["row_id"] for r in loaded.rows] == [r["row_id"] for r in pack.rows]
    # tamper detection
    import gzip, json as _json, shutil

    src = tmp_path / "qX.json.gz"
    raw = gzip.decompress(src.read_bytes())
    tampered = raw.replace(b"freight", b"FREIGHT")
    with gzip.open(src, "wb") as f:
        f.write(tampered)
    from qwery_smith.exceptions import HarnessError

    try:
        load_pack(tmp_path, "qX", expected_sha256=meta["sha256"])
        assert False, "tamper not detected"
    except HarnessError:
        pass


def test_evidence_render(toy_db):
    schema = load_schema(toy_db)
    idx = build_index(toy_db, schema)
    pack = build_pack("qY", "product p01 orders", idx, schema, top_k=2)
    text = pack.render_evidence()
    assert text.count("[") >= 1
    assert "order_items" in text or "orders" in text