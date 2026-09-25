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


@pytest.fixture(scope="module")
def labelled(sample) -> svy.Sample:
    s = svy.Sample(sample.data, sample.design)
    s.meta.set_label("zone", "Zone of residence")
    s.meta.set_value_labels("zone", {1: "Urban", 2: "Rural", 3: "Peri-urban"})
    s.meta.set_value_labels("sex", {"f": "Female", "m": "Male"})
    return s


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


LABELLED_CASES = [
    ("mean_by", lambda s: s.estimation.mean("y", by=["zone", "sex"]), {}),
    ("mean_by_raw", lambda s: s.estimation.mean("y", by=["zone", "sex"]), {"tidy": False}),
    ("mean_by_off", lambda s: s.estimation.mean("y", by="zone"), {"use_labels": False}),
    ("prop", lambda s: s.estimation.prop("zone", by="sex"), {}),
    ("mean_as_factor", lambda s: s.estimation.mean("zone", as_factor=True), {}),
    ("unlabelled_by", lambda s: s.estimation.mean("y", by="region"), {}),
    ("mean_list", lambda s: s.estimation.mean(["y", "z"], by="zone"), {}),
]


@pytest.mark.parametrize(
    "build,opts", [c[1:] for c in LABELLED_CASES], ids=[c[0] for c in LABELLED_CASES]
)
def test_labels_match_live_table(labelled, build, opts):
    result = build(labelled)
    assert_frame_equal(to_polars(_payload(result), **opts), result.to_polars(**opts))


def test_label_columns_follow_codes(labelled):
    table = labelled.estimation.mean("y", by=["zone", "sex"]).to_polars()
    assert table.columns[:4] == ["zone", "zone_label", "sex", "sex_label"]
    pairs = dict(zip(table["zone"], table["zone_label"]))
    assert pairs == {"1": "Urban", "2": "Rural", "3": "Peri-urban"}
    assert table["zone"].to_list() == sorted(table["zone"].to_list())


def test_no_label_column_without_value_labels(labelled):
    table = labelled.estimation.mean("y", by="region").to_polars()
    assert "region_label" not in table.columns


def test_labels_off_gives_codes_only(labelled):
    table = labelled.estimation.mean("y", by="zone").to_polars(use_labels=False)
    assert table.columns[0] == "zone" and "zone_label" not in table.columns


def test_raw_rows_carry_label_lists(labelled):
    table = labelled.estimation.mean("y", by=["zone", "sex"]).to_polars(tidy=False)
    by_label = table["by_label"].to_list()
    assert all(len(v) == 2 for v in by_label)
    assert {v[1] for v in by_label} == {"Female", "Male"}


def test_printing_still_shows_labels_in_place(labelled):
    printable = labelled.estimation.mean("y", by="zone").to_polars_printable()
    assert printable.columns[0] == "Zone of residence"
    assert set(printable[printable.columns[0]]) == {"Urban", "Rural", "Peri-urban"}


def test_payload_stores_only_present_levels(labelled):
    data = serialize(labelled.estimation.mean("y", by="zone", where={"zone": [1, 2]}))
    (zone,) = [v for v in data.labels if v.var == "zone"]
    assert zone.var_label == "Zone of residence"
    assert {lv.label for lv in zone.values} == {"Urban", "Rural"}


def test_label_column_clash_raises(sample):
    from svy.errors import MethodError

    s = svy.Sample(sample.data.with_columns(zone_label=pl.col("sex")), sample.design)
    s.meta.set_value_labels("zone", {1: "Urban", 2: "Rural", 3: "Peri-urban"})
    with pytest.raises(MethodError) as exc:
        s.estimation.mean("y", by=["zone", "zone_label"]).to_polars()
    assert exc.value.code == "LABEL_COLUMN_CLASH"


def test_nan_deff_survives(sample):
    """A requested deff that is NaN comes back NaN; an unrequested one stays absent."""
    s = svy.Sample(
        sample.data.with_columns(c=(pl.col("stratum") > "s2").cast(pl.Float64)), sample.design
    )
    result = s.estimation.mean("c", by="stratum", deff="wr")
    assert any(np.isnan(p.deff) for p in result.estimates)
    data = _payload(result)
    assert all(p.deff is not None for p in data.estimates)
    assert_frame_equal(to_polars(data), result.to_polars())
    assert all(p.deff is None for p in _payload(s.estimation.mean("c", by="stratum")).estimates)
