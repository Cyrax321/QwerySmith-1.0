"""Schema loader + profiler + cutoff tests."""


from qwery_smith.config import HoldoutConfig
from qwery_smith.profiler import compute_holdout_cutoff, profile_database
from qwery_smith.schema_loader import load_schema

HOLDOUT = HoldoutConfig(column="orders.purchase_ts", months=6)


def test_schema_ddl_deterministic(toy_db):
    s1 = load_schema(toy_db)
    text1 = s1.ddl_text()
    s2 = load_schema(toy_db)  # reload
    assert text1 == s2.ddl_text()
    assert s1.ddl_sha256() == s2.ddl_sha256()
    assert "CREATE TABLE order_items" in text1
    assert "-- fk: order_id -> orders.order_id" in text1


def test_fingerprint_includes_row_counts(toy_db):
    s = load_schema(toy_db)
    fp1 = s.fingerprint()
    toy_db.conn.execute("INSERT INTO orders VALUES ('o99','c99','x','2017-01-01 00:00:00')")
    toy_db.conn.commit()
    fp2 = load_schema(toy_db).fingerprint()
    assert fp1 != fp2


def test_cutoff_math(toy_db):
    # max purchase = 2018-09-17 -> month floor 2018-09-01 -> minus 6 months = 2018-03-01
    hi, cutoff = compute_holdout_cutoff(toy_db, HOLDOUT)
    assert hi == "2018-09-17"
    assert cutoff == "2018-03-01"


def test_cutoff_december_rollover():
    # pure month arithmetic: from 2018-01 floor, minus 6 -> 2017-07
    class FakeAdapter:
        def scalar(self, q):
            return ("2018-01-05 10:00:00", "2018-01-05 10:00:00")  # (MIN, MAX)

    hi, cutoff = compute_holdout_cutoff(FakeAdapter(), HOLDOUT)
    assert cutoff == "2017-07-01"


def test_profile_has_holdout_block(toy_db):
    prof = profile_database(toy_db, holdout=HOLDOUT)
    assert prof.holdout["cutoff"] == "2018-03-01"
    assert prof.holdout["data_max"] == "2018-09-17"
    names = [t["name"] for t in prof.tables]
    assert names == sorted(names)  # deterministic


def test_pk_meta_roundtrip(toy_db):
    from qwery_smith.schema_loader import parse_pk_meta

    pks = parse_pk_meta(toy_db)
    assert pks.get("orders") == ("order_id",)