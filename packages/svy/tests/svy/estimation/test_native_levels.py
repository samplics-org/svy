# tests/svy/estimation/test_native_levels.py
"""Domain and category levels keep the source column's type.

The kernel groups on strings, so levels used to come back as text
(``"3"``, ``"false"``, ``"1.0"`` for an ``as_factor`` mean). They are now
mapped back to the column's own values; contrasts still accept the old
string keys.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from svy import Design, Sample
from svy.core.data_prep import level_lookup
from svy.errors import MethodError
from svy.estimation.contrast import linear_contrast
from svy.serialize import from_json, to_json


N = 240


@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    rng = np.random.default_rng(20260925)
    return pl.DataFrame(
        {
            "stratum": np.repeat(["s1", "s2"], N // 2),
            "psu": np.repeat(np.arange(24), N // 24),
            "w": rng.uniform(1, 3, N),
            "y": rng.normal(10, 2, N),
            "x": rng.uniform(1, 2, N),
            "zone": rng.integers(1, 4, N),
            "b": rng.integers(0, 2, N).astype(bool),
            "g": rng.choice(["a", "b"], N),
            "f": rng.choice([1.5, 2.5], N),
            "fi": rng.choice([1.0, 2.0], N),
            "d": pl.Series([dt.date(2020, 1, 1), dt.date(2021, 1, 1)])[rng.integers(0, 2, N)],
        }
    ).with_columns(pl.col("g").cast(pl.Categorical).alias("gc"))


@pytest.fixture(scope="module")
def sample(data) -> Sample:
    return Sample(data, Design(stratum="stratum", psu="psu", wgt="w"))


@pytest.fixture(scope="module")
def rep_sample(sample) -> Sample:
    return sample.weighting.create_jk_wgts()


def _levels(values) -> set:
    return {(type(v), v) for v in values}


def _source(data: pl.DataFrame, col: str) -> set:
    return _levels(data[col].unique().to_list())


def _by(result, i: int = 0) -> list:
    return [p.by_level[i] for p in result.estimates]


BY_COLS = ["zone", "b", "g", "gc", "f", "d"]


class TestByLevels:
    @pytest.mark.parametrize("by", BY_COLS)
    def test_mean(self, sample, data, by):
        assert _levels(_by(sample.estimation.mean("y", by=by))) == _source(data, by)

    @pytest.mark.parametrize("by", BY_COLS)
    def test_total(self, sample, data, by):
        assert _levels(_by(sample.estimation.total("y", by=by))) == _source(data, by)

    @pytest.mark.parametrize("by", ["zone", "b"])
    def test_ratio(self, sample, data, by):
        assert _levels(_by(sample.estimation.ratio("y", "x", by=by))) == _source(data, by)

    @pytest.mark.parametrize("by", ["zone", "b"])
    def test_prop(self, sample, data, by):
        r = sample.estimation.prop("g", by=by)
        assert _levels(_by(r)) == _source(data, by)
        assert _levels(p.y_level for p in r.estimates) == _source(data, "g")

    @pytest.mark.parametrize("by", ["zone", "b"])
    def test_median(self, sample, data, by):
        assert _levels(_by(sample.estimation.median("y", by=by))) == _source(data, by)

    def test_quantile(self, sample, data):
        for r in sample.estimation.quantile("y", p=[0.25, 0.75], by="zone"):
            assert _levels(_by(r)) == _source(data, "zone")

    def test_corr(self, sample, data):
        assert _levels(_by(sample.estimation.corr(["y", "x"], by="b"))) == _source(data, "b")

    def test_multiple_by(self, sample, data):
        r = sample.estimation.mean("y", by=["zone", "b"])
        expected = set(data.select("zone", "b").unique().rows())
        assert {p.by_level for p in r.estimates} == expected
        assert all(type(p.by_level[0]) is int and type(p.by_level[1]) is bool for p in r.estimates)
        assert set(r.domains) == expected

    def test_domains_single_by(self, sample):
        r = sample.estimation.mean("y", by="zone")
        assert sorted(r.domains) == [1, 2, 3]

    def test_where(self, sample):
        r = sample.estimation.mean("y", by="zone", where=pl.col("zone") != 2)
        assert sorted(_by(r)) == [1, 3]

    def test_where_multiple_by(self, sample):
        r = sample.estimation.mean("y", by=["zone", "b"], where=pl.col("b"))
        assert {p.by_level for p in r.estimates} == {(1, True), (2, True), (3, True)}


class TestReplicationByLevels:
    @pytest.mark.parametrize("by", ["zone", "b", "f"])
    def test_mean(self, rep_sample, data, by):
        assert _levels(_by(rep_sample.estimation.mean("y", by=by))) == _source(data, by)

    def test_multiple_by(self, rep_sample, data):
        r = rep_sample.estimation.total("y", by=["b", "zone"])
        assert {p.by_level for p in r.estimates} == set(data.select("b", "zone").unique().rows())

    def test_ratio(self, rep_sample, data):
        assert _levels(_by(rep_sample.estimation.ratio("y", "x", by="b"))) == _source(data, "b")

    def test_prop(self, rep_sample, data):
        r = rep_sample.estimation.prop("b", by="zone")
        assert _levels(_by(r)) == _source(data, "zone")
        assert _levels(p.y_level for p in r.estimates) == _source(data, "b")

    def test_median_and_quantile(self, rep_sample, data):
        assert _levels(_by(rep_sample.estimation.median("y", by="zone"))) == _source(data, "zone")
        for r in rep_sample.estimation.quantile("y", p=[0.25], by="zone"):
            assert _levels(_by(r)) == _source(data, "zone")

    def test_as_factor(self, rep_sample):
        r = rep_sample.estimation.mean("zone", as_factor=True)
        assert [p.y_level for p in r.estimates] == [1, 2, 3]


class TestYLevels:
    def test_as_factor_int(self, sample):
        r = sample.estimation.mean("zone", as_factor=True)
        assert [p.y_level for p in r.estimates] == [1, 2, 3]
        assert all(type(p.y_level) is int for p in r.estimates)

    def test_as_factor_bool(self, sample):
        r = sample.estimation.mean("b", as_factor=True)
        assert _levels(p.y_level for p in r.estimates) == {(bool, False), (bool, True)}

    def test_as_factor_by(self, sample, data):
        r = sample.estimation.mean("zone", by="b", as_factor=True)
        assert _levels(_by(r)) == _source(data, "b")
        assert _levels(p.y_level for p in r.estimates) == _source(data, "zone")

    @pytest.mark.parametrize("y", ["zone", "b", "g", "gc", "fi"])
    def test_prop(self, sample, data, y):
        r = sample.estimation.prop(y)
        assert _levels(p.y_level for p in r.estimates) == _source(data, y)

    def test_prop_string_codes_not_parsed(self):
        df = pl.DataFrame({"y": ["01", "02", "01", "10"], "w": [1.0, 2.0, 1.0, 1.0]})
        r = Sample(df, Design(wgt="w")).estimation.prop("y")
        assert sorted(p.y_level for p in r.estimates) == ["01", "02", "10"]


class TestTtest:
    def test_one_sample_by(self, sample):
        res = sample.categorical.ttest("y", by="zone", mean_h0=10)
        assert sorted(r.diff[0].by_level for r in res) == [1, 2, 3]
        assert all(type(r.estimates[0].by_level) is int for r in res)

    def test_two_sample_groups(self, sample):
        res = sample.categorical.ttest("y", group="b")
        assert res.groups.levels == (False, True)
        assert [e.group_level for e in res.estimates] == [False, True]

    def test_two_sample_by(self, sample):
        res = sample.categorical.ttest("y", group="b", by="zone")
        for r in res:
            assert type(r.diff[0].by_level) is int
            assert {e.group_level for e in r.estimates} == {False, True}


class TestContrastKeys:
    def test_keys_are_native(self, sample):
        assert sorted(sample.estimation.mean("y", by="zone").keys()) == [1, 2, 3]

    def test_native_and_string_keys_agree(self, sample):
        r = sample.estimation.mean("y", by="zone")
        native = r.contrast({3: 1, 1: -1}).estimates[0]
        legacy = r.contrast({"3": 1, "1": -1}).estimates[0]
        assert native.est == legacy.est
        assert native.se == legacy.se

    def test_bool_keys(self, sample):
        r = sample.estimation.mean("y", by="b")
        native = r.contrast({True: 1, False: -1}).estimates[0].est
        assert r.contrast({"true": 1, "false": -1}).estimates[0].est == native
        assert r.contrast({"True": 1, "False": -1}).estimates[0].est == native

    def test_multiple_by_keys(self, sample):
        r = sample.estimation.mean("y", by=["zone", "b"])
        native = r.contrast({(1, True): 1, (1, False): -1}).estimates[0].est
        assert r.contrast({("1", "true"): 1, ("1", "false"): -1}).estimates[0].est == native
        assert r.contrast({(1, "true"): 1, ("1", False): -1}).estimates[0].est == native

    def test_as_factor_float_spelling(self, sample):
        r = sample.estimation.mean("zone", as_factor=True)
        native = r.contrast({3: 1, 1: -1}).estimates[0].est
        assert r.contrast({"3.0": 1, "1.0": -1}).estimates[0].est == native
        assert r.contrast({"3": 1, "1": -1}).estimates[0].est == native

    def test_prop_by_keys(self, sample):
        r = sample.estimation.prop("zone", by="b")
        native = r.contrast({(True, 1): 1, (False, 1): -1}).estimates[0].est
        assert r.contrast({("true", "1"): 1, ("false", 1): -1}).estimates[0].est == native

    def test_unknown_key_still_raises(self, sample):
        r = sample.estimation.mean("y", by="zone")
        with pytest.raises(MethodError, match="Unknown contrast key"):
            r.contrast({4: 1, 1: -1})

    def test_ambiguous_string_key_raises(self):
        keys = [("1", 2), (1, "2")]
        with pytest.raises(MethodError, match="Unknown contrast key"):
            linear_contrast(
                keys,
                np.array([1.0, 2.0]),
                np.eye(2),
                {("1", "2"): 1},
                df=10.0,
                alpha=0.05,
                method="Taylor",
            )


class TestSerialization:
    @pytest.mark.parametrize("by", ["zone", "b", "g", "f"])
    def test_by_level_round_trip(self, sample, by):
        r = sample.estimation.mean("y", by=by)
        back = from_json(to_json(r))
        assert [tuple(p.by_level) for p in back.estimates] == [p.by_level for p in r.estimates]
        assert [type(p.by_level[0]) for p in back.estimates] == [
            type(p.by_level[0]) for p in r.estimates
        ]

    def test_y_level_round_trip(self, sample):
        r = sample.estimation.prop("zone", by="b")
        back = from_json(to_json(r))
        got = [(tuple(p.by_level), p.y_level) for p in back.estimates]
        assert got == [(p.by_level, p.y_level) for p in r.estimates]
        assert all(type(p.y_level) is int and type(p.by_level[0]) is bool for p in back.estimates)

    def test_ttest_groups_round_trip(self, sample):
        back = from_json(to_json(sample.categorical.ttest("y", group="b")))
        assert back.groups.levels == [False, True]


class TestDisplay:
    def test_bool_prints_lowercase(self, sample):
        df = sample.estimation.mean("y", by="b").to_polars_printable()
        assert sorted(df["b"].to_list()) == ["false", "true"]

    def test_as_factor_prints_integer(self, sample):
        df = sample.estimation.mean("zone", as_factor=True).to_polars_printable()
        assert df["zone"].to_list() == ["1", "2", "3"]

    def test_data_view_keeps_types(self, sample):
        assert sample.estimation.mean("y", by="b").to_polars().schema["b"] == pl.Boolean
        df = sample.estimation.mean("zone", as_factor=True).to_polars()
        assert df["zone"].to_list() == [1, 2, 3]


class TestLevelLookup:
    def test_null_maps_to_none(self):
        df = pl.DataFrame({"z": [1, None, 2]})
        assert level_lookup(df, ["z"])["__Null__"] is None

    def test_cast_spellings(self):
        df = pl.DataFrame({"i": [1, 2], "f": [1.0, 2.5], "b": [True, False]})
        assert level_lookup(df, ["i"])["1.0"] == 1
        lk = level_lookup(df, ["f"])
        assert lk["1"] == 1.0 and "2" not in lk
        assert level_lookup(df, ["b"])["0.0"] is False

    def test_multiple_columns(self):
        df = pl.DataFrame({"a": [1, None], "b": ["x", "y"]})
        lk = level_lookup(df, ["a", "b"])
        assert lk == {"1__by__x": (1, "x"), "__Null____by__y": (None, "y")}

    def test_lazy_and_missing_column(self):
        lf = pl.LazyFrame({"a": [1, 2]})
        assert level_lookup(lf, ["a"])["2"] == 2
        assert level_lookup(lf, ["nope"]) == {}
