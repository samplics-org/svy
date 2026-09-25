# tests/svy/serialization/test_to_polars.py
"""
svy.serialize.to_polars: the table of a payload equals the live result's
``to_polars()``, after a JSON round trip, and ``row_index`` maps every table
row to exactly one payload row.
"""

from __future__ import annotations

import datetime as dt
import json

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
    ("prop_list", lambda s: s.estimation.prop(["sex", "region"]), {}),
    ("ratio_by", lambda s: s.estimation.ratio("y", "w", by="sex"), {}),
    ("ratios_x", lambda s: s.estimation.ratio("y", ["w", "psu"]), {}),
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
    (
        "ttest_two_where",
        lambda s: s.categorical.ttest("y", group="sex", where={"region": "North"}),
        {},
    ),
    ("ttest_two_by", lambda s: s.categorical.ttest("y", group="sex", by="zone")[1], {}),
    ("table_one", lambda s: s.categorical.tabulate("region"), {}),
    ("table_two", lambda s: s.categorical.tabulate("region", "sex"), {}),
    ("table_two_raw", lambda s: s.categorical.tabulate("region", "sex"), {"tidy": False}),
    (
        "table_two_where",
        lambda s: s.categorical.tabulate("region", "sex", where={"zone": [1, 2]}),
        {},
    ),
    ("chi_square", lambda s: s.categorical.tabulate("region", "sex").stats.chisq, {}),
    ("describe", lambda s: s.describe(["y", "region", "zone", "d"]), {}),
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
    if kind == "chi_square":
        return [data]
    if kind == "describe":
        return data.items
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


def test_every_serializable_kind_has_a_table(sample):
    from svy.serialize.serializers import _SERIALIZERS
    from svy.serialize.tables import _TABLES

    covered = {type(serialize(_live(c[1](sample)))) for c in CASES}
    assert covered == set(_TABLES)
    assert len(_TABLES) == len(_SERIALIZERS)


def test_stacked_list_names_each_member(sample):
    """Several variables (or denominators) lead with y (or x)."""
    assert sample.estimation.mean(["y", "z"], by="sex").to_polars().columns[0] == "y"
    assert sample.estimation.ratio("y", ["w", "psu"]).to_polars().columns[0] == "x"
    assert "y" not in sample.estimation.quantile("y", p=(0.25, 0.75)).to_polars().columns


def test_empty_estimate():
    from svy import Estimate, PopParam

    assert to_polars(serialize(Estimate(PopParam.MEAN))).is_empty()


def test_unknown_payload_raises():
    with pytest.raises(SerializationError) as exc:
        to_polars("not a payload")  # type: ignore[arg-type]
    assert exc.value.code == "PAYLOAD_NO_TABLE"
    assert "EstimateData" in (exc.value.expected or [])


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
    assert pairs == {1: "Urban", 2: "Rural", 3: "Peri-urban"}
    assert table["zone"].to_list() == sorted(table["zone"].to_list())


def test_no_label_column_without_value_labels(labelled):
    table = labelled.estimation.mean("y", by="region").to_polars()
    assert "region_label" not in table.columns


def test_labels_off_gives_codes_only(labelled):
    table = labelled.estimation.mean("y", by="zone").to_polars(use_labels=False)
    assert table.columns[0] == "zone" and "zone_label" not in table.columns


def test_raw_rows_carry_label_lists(labelled):
    table = labelled.estimation.mean("y", by=["zone", "sex"]).to_polars(tidy=False)
    assert table.schema["by_level"] == pl.Struct({"zone": pl.Int64, "sex": pl.String})
    by_label = table["by_label"].to_list()
    assert {v["sex"] for v in by_label} == {"Female", "Male"}
    assert {v["zone"] for v in by_label} == {"Urban", "Rural", "Peri-urban"}


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


@pytest.mark.parametrize(
    "values",
    [
        [dt.date(2020, 1, 1), dt.date(2021, 6, 30)],
        [dt.datetime(2020, 1, 1, 8), dt.datetime(2020, 1, 1, 17, 30)],
        [
            dt.datetime(2020, 1, 1, 8, tzinfo=dt.timezone.utc),
            dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc),
        ],
        [dt.time(8, 0), dt.time(17, 30, 15)],
    ],
    ids=["date", "datetime", "datetime_tz", "time"],
)
def test_temporal_levels_round_trip(sample, values):
    n = sample.data.height
    s = svy.Sample(
        sample.data.with_columns(t=pl.Series([values[i % 2] for i in range(n)])), sample.design
    )
    result = s.estimation.mean("y", by="t")
    js = to_json(result)
    assert set(json.loads(js)["temporal"].values()) == {type(values[0]).__name__}
    data = from_json(js)
    assert [p.by_level for p in data.estimates] == [list(p.by_level) for p in result.estimates]
    assert_frame_equal(to_polars(data), result.to_polars())


def test_temporal_level_in_labels_and_prop(sample):
    days = [dt.date(2020, 1, 1), dt.date(2020, 1, 2)]
    s = svy.Sample(
        sample.data.with_columns(t=pl.Series([days[i % 2] for i in range(sample.data.height)])),
        sample.design,
    )
    s.meta.set_value_labels("t", {days[0]: "New year", days[1]: "Day after"})
    result = s.estimation.prop("t")
    data = from_json(to_json(result))
    assert {p.y_level for p in data.estimates} == set(days)
    assert {lv.code for v in data.labels for lv in v.values} == set(days)
    assert_frame_equal(to_polars(data), result.to_polars())


def test_duration_level_round_trip():
    """Durations cannot be a by variable yet; the payload still carries one exactly."""
    from svy import Estimate, ParamEst, PopParam

    est = Estimate(PopParam.MEAN, alpha=0.05)
    est.method, est.n_strata, est.n_psus = "Taylor", 1, 2
    span = dt.timedelta(hours=36, microseconds=5)
    est.estimates = [
        ParamEst(y="y", est=1.0, se=0.1, cv=0.1, lci=0.8, uci=1.2, by=("d",), by_level=(span,))
    ]
    js = to_json(est)
    assert json.loads(js)["temporal"] == {"/estimates/0/by_level/0": "duration"}
    assert from_json(js).estimates[0].by_level == [span]
