# tests/svy/serialization/test_to_polars.py
"""
svy.serialize.to_polars: the table of a payload equals the live result's
``to_polars()``, after a JSON round trip, and ``row_index`` maps every table
row to exactly one payload row.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polars.testing import assert_frame_equal

import svy

from svy.errors import SerializationError
from svy.regression.base import GLM
from svy.serialize import from_json, serialize, to_json, to_polars


@pytest.fixture(scope="module")
def sample() -> svy.Sample:
    rng = np.random.default_rng(2026)
    n = 240
    stratum = np.repeat(["s1", "s2", "s3", "s4"], n // 4)
    psu = np.repeat(np.arange(n // 6), 6)
    region = rng.choice(["North", "South", "East", "West"], n)
    df = pl.DataFrame(
        {
            "stratum": stratum,
            "psu": psu,
            "w": rng.uniform(1, 5, n),
            "region": region,
            "zone": rng.integers(1, 4, n),
            "sex": rng.choice(["f", "m"], n),
            "y": rng.normal(50, 10, n),
            "z": rng.normal(0, 1, n),
            "d": rng.integers(0, 2, n),
        }
    )
    return svy.Sample(df, svy.Design(stratum="stratum", psu="psu", wgt="w"))


# Each case: (id, result builder, to_polars options)
CASES = [
    ("mean", lambda s: s.estimation.mean("y"), {}),
    ("mean_by", lambda s: s.estimation.mean("y", by="region"), {}),
    ("mean_by2", lambda s: s.estimation.mean("y", by=["region", "sex"]), {}),
    ("mean_by2_raw", lambda s: s.estimation.mean("y", by=["region", "sex"]), {"tidy": False}),
    ("mean_where", lambda s: s.estimation.mean("y", by="sex", where={"region": "North"}), {}),
    ("mean_as_factor", lambda s: s.estimation.mean("zone", as_factor=True), {}),
    ("mean_deff", lambda s: s.estimation.mean("y", by="zone", deff="wr"), {}),
    ("total_by_int", lambda s: s.estimation.total("y", by="zone"), {}),
    ("prop_by", lambda s: s.estimation.prop("sex", by="region"), {}),
    ("median_by", lambda s: s.estimation.median("y", by="region"), {}),
    ("quantiles", lambda s: s.estimation.quantile("y", p=(0.25, 0.75), by="sex"), {}),
    ("mean_list", lambda s: s.estimation.mean(["y", "z"], by="sex"), {}),
    ("ttest_one", lambda s: s.categorical.ttest("y", mean_h0=50), {}),
    (
        "ttest_one_est",
        lambda s: s.categorical.ttest("y", mean_h0=50),
        {"component": "estimates"},
    ),
    ("ttest_two", lambda s: s.categorical.ttest("y", group="sex"), {}),
    (
        "ttest_two_est",
        lambda s: s.categorical.ttest("y", group="sex"),
        {"component": "estimates"},
    ),
    (
        "ttest_two_est_raw",
        lambda s: s.categorical.ttest("y", group="sex"),
        {"component": "estimates", "tidy": False},
    ),
    ("table_one", lambda s: s.categorical.tabulate("region"), {}),
    ("table_two", lambda s: s.categorical.tabulate("region", "sex"), {}),
    ("table_two_raw", lambda s: s.categorical.tabulate("region", "sex"), {"tidy": False}),
    ("glm", lambda s: s.glm.fit("y", x=["z", svy.Cat("region")]), {}),
    (
        "glm_logit_exp",
        lambda s: s.glm.fit("d", x=["z"], family="binomial"),
        {"exponentiate": True},
    ),
    (
        "glm_pred",
        lambda s: s.glm.fit("y", x=["z"]).predict(pl.DataFrame({"z": [-1.0, 0.0, 1.0]})),
        {},
    ),
]


def _live(result):
    return result.fitted if isinstance(result, GLM) else result


def _payload(result):
    return from_json(to_json(result))


@pytest.mark.parametrize("build,opts", [c[1:] for c in CASES], ids=[c[0] for c in CASES])
def test_matches_live_table(sample, build, opts):
    result = _live(build(sample))
    assert_frame_equal(to_polars(_payload(result), **opts), result.to_polars(**opts))


def _payload_rows(data, opts) -> list:
    kind = data.kind
    if kind == "estimate_list":
        return [p for m in data.estimates for p in m.estimates]
    if kind.startswith("ttest"):
        return data.estimates if opts.get("component") == "estimates" else data.diff
    if kind == "glm_fit":
        return data.coefs
    if kind == "glm_pred":
        return list(range(len(data.yhat)))
    return data.estimates


@pytest.mark.parametrize("build,opts", [c[1:] for c in CASES], ids=[c[0] for c in CASES])
def test_row_index_maps_each_row_to_one_payload_row(sample, build, opts):
    data = _payload(_live(build(sample)))
    plain = to_polars(data, **opts)
    indexed = to_polars(data, row_index="row", **opts)

    assert indexed.columns[0] == "row"
    assert indexed["row"].dtype == pl.UInt32
    assert_frame_equal(indexed.drop("row"), plain)
    rows = _payload_rows(data, opts)
    assert sorted(indexed["row"].to_list()) == list(range(len(rows)))


def test_row_index_follows_the_display_sort(sample):
    """Estimate tables are sorted for display; the index points at the payload row."""
    data = _payload(sample.estimation.mean("y", by=["region", "sex"]))
    table = to_polars(data, row_index="row")
    for rec in table.iter_rows(named=True):
        p = data.estimates[rec["row"]]
        assert [str(v) for v in p.by_level] == [rec["region"], rec["sex"]]
        assert p.est == rec["est"]


def test_estimate_list_row_index_counts_through_members(sample):
    data = _payload(sample.estimation.mean(["y", "z"], by="sex"))
    table = to_polars(data, row_index="row")
    flat = [p for m in data.estimates for p in m.estimates]
    assert [flat[i].est for i in table["row"]] == table["est"].to_list()


def test_nan_interval_survives(sample):
    """A domain with one PSU has a NaN interval; the table still round-trips."""
    result = sample.estimation.mean("y", by="psu")
    table = to_polars(_payload(result))
    assert table["lci"].is_nan().any()
    assert_frame_equal(table, result.to_polars())


def test_payload_without_table_raises(sample):
    data = serialize(sample.categorical.tabulate("region", "sex").stats.chisq)
    with pytest.raises(SerializationError) as exc:
        to_polars(data)
    assert exc.value.code == "PAYLOAD_NO_TABLE"
