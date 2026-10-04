"""Contrasts serialize: rows, method, alpha, df and their source estimate; no covariance."""

import json

import numpy as np
import polars as pl
import pytest

import svy

from svy.errors import SerializationError
from svy.serialize import ContrastData, from_json, serialize, to_json, to_polars


@pytest.fixture(scope="module")
def sample() -> svy.Sample:
    rng = np.random.default_rng(3)
    n = 120
    return svy.Sample(
        pl.DataFrame(
            {
                "stratum": np.repeat([1, 2, 3, 4], n // 4),
                "psu": np.arange(n) // 4,
                "w": rng.uniform(1, 3, n),
                "y": rng.normal(10, 2, n),
                "b": rng.integers(0, 2, n),
                "g": rng.choice(["A", "B", "C"], n),
            }
        ),
        svy.Design(stratum="stratum", psu="psu", wgt="w"),
    )


def test_round_trip_and_table(sample):
    c = sample.estimation.mean("y", by="g").contrast(svy.estd("A") - svy.estd("B"))
    d = from_json(to_json(c))
    assert isinstance(d, ContrastData)
    assert (d.param, d.y, d.method, d.alpha) == ("Mean", "y", "Taylor", 0.05)
    assert d.df == pytest.approx(c.df)
    assert to_polars(d).equals(c.to_polars())
    assert serialize(c) == d


def test_named_and_nonlinear_contrasts(sample):
    c = sample.estimation.mean("y", by="g").contrast(
        {"a_b": svy.estd("A") - svy.estd("B"), "ratio": svy.estd("A") / svy.estd("C")}
    )
    d = serialize(c)
    assert [e.contrast for e in d.estimates] == ["a_b", "ratio"]
    assert d.estimates[1].est == pytest.approx(c.estimates[1].est)


def test_proportion_and_replication(sample):
    rep = sample.weighting.create_jk_wgts()
    c = rep.estimation.prop("b", by="g", method="replication").contrast(
        svy.estd(("A", 1)) - svy.estd(("B", 1))
    )
    d = from_json(to_json(c))
    assert d.param == "Proportion" and d.y == "b"
    assert d.method != "Taylor"
    assert to_polars(d).equals(c.to_polars())


def test_alpha_is_kept(sample):
    c = sample.estimation.mean("y", by="g").contrast(svy.estd("A") - svy.estd("B"), alpha=0.1)
    assert serialize(c).alpha == 0.1


def test_covariance_is_not_saved(sample):
    c = sample.estimation.mean("y", by="g").contrast(
        {"a_b": svy.estd("A") - svy.estd("B"), "a_c": svy.estd("A") - svy.estd("C")}
    )
    raw = json.loads(to_json(c))
    assert "covariance" not in raw
    assert raw["kind"] == "contrast"


def test_row_index(sample):
    c = sample.estimation.mean("y", by="g").contrast(svy.estd("A") - svy.estd("B"))
    df = to_polars(serialize(c), row_index="row")
    assert df.columns[0] == "row" and df["row"].to_list() == [0]


def test_bare_contrast_has_no_source():
    from svy.estimation.contrast import Contrast, ContrastEst

    row = ContrastEst("c", 1.0, 0.5, 0.5, 0.0, 2.0, 2.0, 0.05, 10.0)
    d = serialize(Contrast([row], np.eye(1), alpha=0.05, df=10.0, method="Taylor"))
    assert (d.param, d.y) == (None, None)


def test_estimate_still_refuses_a_bad_kind():
    with pytest.raises(SerializationError):
        serialize(object())
