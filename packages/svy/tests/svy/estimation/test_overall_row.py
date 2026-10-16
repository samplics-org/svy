# tests/svy/estimation/test_overall_row.py
"""overall=True: the by= result with the by=None rows after it."""

import numpy as np
import polars as pl
import pytest

import svy

from svy.errors import MethodError
from svy.serialize import from_json, to_json, to_polars


@pytest.fixture(scope="module")
def sample() -> svy.Sample:
    rng = np.random.default_rng(7)
    n = 300
    df = pl.DataFrame(
        {
            "st": np.repeat([1, 2, 3], n // 3),
            "psu": np.arange(n) // 10,
            "w": rng.uniform(1, 3, n),
            "region": rng.choice(["E", "N", "W"], n),
            "sex": rng.choice(["f", "m"], n),
            "y": rng.normal(50, 10, n),
            "x": rng.uniform(1, 2, n),
        }
    )
    return svy.Sample(df, svy.Design(stratum="st", psu="psu", wgt="w"))


CALLS = {
    "mean": lambda e, **k: e.mean("y", **k),
    "total": lambda e, **k: e.total("y", **k),
    "prop": lambda e, **k: e.prop("sex", **k),
    "ratio": lambda e, **k: e.ratio("y", "x", **k),
    "median": lambda e, **k: e.median("y", **k),
    "quantile": lambda e, **k: e.quantile("y", p=0.25, **k),
}


def _rows(result):
    return [(p.by_level, p.y_level, p.est, p.se) for p in result.estimates]


@pytest.mark.parametrize("name", CALLS)
def test_rows_are_the_two_calls(sample, name):
    call = CALLS[name]
    combined = call(sample.estimation, by="region", overall=True)
    grouped = call(sample.estimation, by="region")
    whole = call(sample.estimation)
    assert _rows(combined) == _rows(grouped) + _rows(whole)


def test_two_domains_and_where(sample):
    e = sample.estimation
    kw = dict(by=["region", "sex"], where=pl.col("x") > 1.2)
    combined = e.mean("y", overall=True, **kw)
    assert _rows(combined) == _rows(e.mean("y", **kw)) + _rows(e.mean("y", where=kw["where"]))


def test_replication(sample):
    rep = sample.weighting.create_bs_wgts(n_reps=20, rep_prefix="bs", rstate=1)
    e = rep.estimation
    combined = e.mean("y", by="region", overall=True, method="replication")
    whole = e.mean("y", method="replication")
    assert _rows(combined)[-1] == _rows(whole)[0]


def test_lists_get_an_overall_row_per_member(sample):
    res = sample.estimation.quantile("y", p=(0.25, 0.75), by="region", overall=True)
    assert all(m.estimates[-1].by is None for m in res)
    res = sample.estimation.mean(["y", "x"], by="region", overall=True)
    assert [m.estimates[-1].by for m in res] == [None, None]


def test_table_lists_overall_last_with_null_domain(sample):
    df = sample.estimation.prop("sex", by="region", overall=True).to_polars()
    assert df["region"].to_list()[-2:] == [None, None]
    assert df["region"].to_list()[:-2] == ["E", "E", "N", "N", "W", "W"]


def test_printed_as_all(sample):
    text = str(sample.estimation.mean("y", by=["region", "sex"], overall=True))
    assert text.count("All") == 2


def test_serialized_table_matches(sample):
    res = sample.estimation.mean("y", by="region", overall=True)
    data = from_json(to_json(res))
    assert data.estimates[-1].by is None
    assert to_polars(data).equals(res.to_polars())


def test_needs_by(sample):
    with pytest.raises(MethodError) as exc:
        sample.estimation.mean("y", overall=True)
    assert exc.value.code == "OVERALL_WITHOUT_BY"


def test_contrast_is_refused_and_points_to_the_by_result(sample):
    with pytest.raises(MethodError, match="overall=True") as exc:
        sample.estimation.mean("y", by="region", overall=True).contrast({"E": 1, "N": -1})
    assert exc.value.code == "CONTRAST_NO_COVARIANCE"
    sample.estimation.mean("y", by="region").contrast({"E": 1, "N": -1})
