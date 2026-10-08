import json

import numpy as np
import polars as pl
import pytest

import svy

from svy.errors import MethodError


@pytest.fixture
def frame():
    rng = np.random.default_rng(5)
    n = 300
    df = pl.DataFrame(
        {
            "str": rng.integers(1, 4, n),
            "psu": np.arange(n) // 10,
            "w": rng.uniform(1, 3, n),
            "classe": rng.integers(1, 4, n),
            "internet": rng.integers(0, 2, n),
            **{f"r{i}": rng.uniform(1, 3, n) for i in range(1, 11)},
        }
    )
    # Class 1 all online: a row share of 100% and an empty cell.
    return df.with_columns(
        internet=pl.when(pl.col("classe") == 1).then(1).otherwise(pl.col("internet"))
    )


@pytest.fixture
def sample(frame):
    return svy.Sample(frame, svy.Design(stratum="str", psu="psu", wgt="w"))


def _cells(table):
    return {(c.rowvar, c.colvar): c for c in table.estimates}


def _check_against_prop(cells, est, *, row_is_by, scale=1.0):
    assert len(cells) == len(est.estimates)
    for p in est.estimates:
        by, lvl = str(p.by_level[0]), str(p.y_level)
        c = cells[(by, lvl) if row_is_by else (lvl, by)]
        assert c.est == pytest.approx(p.est * scale)
        assert c.se == pytest.approx(p.se * scale)
        if np.isfinite(p.lci):
            assert c.lci == pytest.approx(p.lci * scale)
            assert c.uci == pytest.approx(p.uci * scale)
        assert c.n == p.n


def test_row_shares_are_prop_by_row(sample):
    t = sample.categorical.tabulate("classe", "internet", share_of="row")
    est = sample.estimation.prop("internet", by="classe")
    _check_against_prop(_cells(t), est, row_is_by=True)


def test_row_percent_scales_estimates_and_intervals(sample):
    t = sample.categorical.tabulate("classe", "internet", units="percent", share_of="row")
    est = sample.estimation.prop("internet", by="classe")
    _check_against_prop(_cells(t), est, row_is_by=True, scale=100.0)
    assert _cells(t)[("1", "1")].est == pytest.approx(100.0)
    assert _cells(t)[("1", "0")].est == 0.0


def test_column_shares_are_prop_by_column(sample):
    t = sample.categorical.tabulate("classe", "internet", share_of="col")
    est = sample.estimation.prop("classe", by="internet")
    _check_against_prop(_cells(t), est, row_is_by=False)


def test_rows_sum_to_one(sample):
    t = sample.categorical.tabulate("classe", "internet", share_of="row")
    totals = {}
    for c in t.estimates:
        totals[c.rowvar] = totals.get(c.rowvar, 0.0) + c.est
    assert all(v == pytest.approx(1.0) for v in totals.values())


def test_rao_scott_test_does_not_depend_on_share_of(sample):
    base = sample.categorical.tabulate("classe", "internet")
    for share in ("row", "col"):
        t = sample.categorical.tabulate("classe", "internet", share_of=share)
        assert t.stats == base.stats


def test_total_is_the_default(sample):
    default = sample.categorical.tabulate("classe", "internet")
    total = sample.categorical.tabulate("classe", "internet", share_of="total")
    assert default.share_of == total.share_of == "total"
    assert [c.est for c in default.estimates] == [c.est for c in total.estimates]


def test_where_matches_prop_where(sample):
    where = svy.col("str") != 3
    t = sample.categorical.tabulate("classe", "internet", share_of="row", where=where)
    est = sample.estimation.prop("internet", by="classe", where=where)
    _check_against_prop(_cells(t), est, row_is_by=True)


def test_replication_matches_prop_replication(frame):
    d = svy.Design(wgt="w", rep_wgts=svy.BootstrapWgts(prefix="r", n_reps=10))
    s = svy.Sample(frame, d)
    t = s.categorical.tabulate("classe", "internet", share_of="row", method="replication")
    est = s.estimation.prop("internet", by="classe", method="replication")
    _check_against_prop(_cells(t), est, row_is_by=True)


def test_title_names_the_denominator(sample):
    row = sample.categorical.tabulate("classe", "internet", share_of="row")
    col = sample.categorical.tabulate("classe", "internet", share_of="col")
    assert row.__plain_str__().startswith("Table: classe × internet (row shares)")
    assert "(column shares)" in col.__plain_str__()
    assert "(row shares)" in str(row)
    assert "shares)" not in sample.categorical.tabulate("classe", "internet").__plain_str__()


def test_value_labels_still_show(sample):
    sample.set_value_labels("classe", {1: "A", 2: "B", 3: "C"})
    out = sample.categorical.tabulate("classe", "internet", share_of="row").__plain_str__()
    assert "\n  A " in out and "\n  C " in out


def test_boundary_note_is_carried(sample):
    t = sample.categorical.tabulate("classe", "internet", share_of="row")
    assert any(f.code == "PROP_CI_BOUNDARY" for f in t.findings)


@pytest.mark.parametrize(
    "kw",
    [
        {"colvar": None, "share_of": "row"},
        {"colvar": "internet", "share_of": "row", "units": "count"},
        {"colvar": "internet", "share_of": "col", "count_total": 1000},
    ],
)
def test_combinations_that_make_no_sense_raise(sample, kw):
    with pytest.raises(MethodError) as e:
        sample.categorical.tabulate("classe", **kw)
    assert e.value.code == "METHOD_NOT_APPLICABLE"


def test_unknown_share_of_raises(sample):
    with pytest.raises(MethodError) as e:
        sample.categorical.tabulate("classe", "internet", share_of="rows")
    assert e.value.code == "INVALID_CHOICE"


def test_saved_table_keeps_share_of(sample):
    t = sample.categorical.tabulate("classe", "internet", share_of="row")
    back = svy.serialize.from_json(svy.serialize.to_json(t))
    assert back.share_of == "row"


def test_older_payload_reads_as_total(sample):
    raw = json.loads(svy.serialize.to_json(sample.categorical.tabulate("classe", "internet")))
    raw.pop("share_of")
    raw["schema_version"] = "svy-result/0.7"
    back = svy.serialize.from_json(json.dumps(raw).encode())
    assert back.share_of == "total"
