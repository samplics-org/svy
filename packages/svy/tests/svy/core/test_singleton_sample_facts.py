# tests/svy/core/test_singleton_sample_facts.py
"""Singleton facts on the sample (``singletons``, ``n_singletons``,
``domain_singletons``) and the removed ``sample.singleton`` namespace."""

from __future__ import annotations

import datetime as dt
import warnings

import polars as pl
import pytest

import svy

from svy.core.design import Singleton
from svy.errors.singleton_errors import SingletonAPIRemoved, SingletonError
from svy.utils import deprecated


pytestmark = pytest.mark.filterwarnings("error")

SURVEY = pl.DataFrame(
    {
        "region": ["North"] * 4 + ["South"] * 4 + ["Center"] * 4 + ["East"] * 3 + ["West"] * 2,
        "zone": ["N"] * 4 + ["S"] * 4 + ["N"] * 4 + ["N"] * 3 + ["S"] * 2,
        "cluster": ["c01", "c01", "c02", "c02", "c03", "c03", "c04", "c04",
                    "c05", "c05", "c06", "c06", "c07", "c07", "c07", "c08", "c08"],
        "income": [980, 1220, 1050, 870, 760, 1180, 990, 1310,
                   1420, 1090, 880, 1200, 950, 1010, 760, 720, 860],
        "w": [20.0] * 17,
    }
)  # fmt: skip
DESIGN = dict(stratum="region", psu="cluster", wgt="w")


def sample(singleton=None, data: pl.DataFrame = SURVEY, **kw) -> svy.Sample:
    return svy.Sample(data, svy.Design(**{**DESIGN, **kw}, singleton=singleton))


# ---------------------------------------------------------------------------
# sample.singletons / sample.n_singletons
# ---------------------------------------------------------------------------


class TestSingletons:
    def test_without_a_rule(self):
        s = sample()
        assert s.singletons.to_dicts() == [
            {"region": "East", "cluster": "c07", "n": 3, "handled": None},
            {"region": "West", "cluster": "c08", "n": 2, "handled": None},
        ]
        assert s.n_singletons == 2
        assert s.singletons.schema == pl.Schema(
            {"region": pl.Utf8, "cluster": pl.Utf8, "n": pl.UInt32, "handled": pl.Utf8}
        )

    @pytest.mark.parametrize(
        "rule, handled",
        [
            ("center", ["center", "center"]),
            ("scale", ["scale", "scale"]),
            ("skip", ["skip", "skip"]),
            ("self_representing", ["self_representing", "self_representing"]),
            ("pool", ["pool -> __pooled__", "pool -> __pooled__"]),
            (Singleton("pool", name="others"), ["pool -> others", "pool -> others"]),
            (Singleton("collapse", within="zone"), ["collapse -> Center", "collapse -> South"]),
            (
                Singleton("collapse", using={"East": "North", "West": "South"}),
                ["collapse -> North", "collapse -> South"],
            ),
        ],
    )
    def test_how_the_rule_handled_each(self, rule, handled):
        assert sample(rule).singletons["handled"].to_list() == handled

    def test_a_rule_that_cannot_handle_them_leaves_handled_empty(self):
        s = sample(Singleton("collapse", using={"East": "Center"}))
        assert s.singletons["handled"].to_list() == [None, None]
        # Reading the facts never raises; the next Taylor analysis does.
        assert s.n_singletons == 2
        str(s)
        with pytest.raises(SingletonError, match="not in the collapse mapping: 'West'"):
            s.estimation.mean("income")

    def test_none(self):
        clean = SURVEY.filter(~pl.col("region").is_in(["East", "West"]))
        s = sample("center", clean)
        assert s.singletons.is_empty() and s.n_singletons == 0
        assert s.singletons.columns == ["region", "cluster", "n", "handled"]

    def test_follows_the_data(self):
        s = sample("pool")
        fewer = s.wrangling.filter_records(svy.col("cluster") != "c02")
        assert fewer.singletons["region"].to_list() == ["East", "North", "West"]
        assert fewer.singletons["handled"].to_list() == ["pool -> __pooled__"] * 3
        assert s.n_singletons == 2

    def test_tuple_strata_and_psus(self):
        s = sample(
            Singleton("collapse", within="zone"),
            stratum=("zone", "region"),
            psu=("region", "cluster"),
        )
        assert s.singletons.columns == ["zone", "region", "cluster", "n", "handled"]
        assert s.singletons["handled"].to_list() == [
            "collapse -> N, Center",
            "collapse -> S, South",
        ]

    def test_integer_valued_float_and_date_strata_as_the_data_shows_them(self):
        codes = {
            "North": 2001.0,
            "South": 2002.0,
            "Center": 2003.0,
            "East": 2004.0,
            "West": 2005.0,
        }
        data = SURVEY.with_columns(
            pl.col("region").replace_strict(codes).alias("code"),
            pl.col("region")
            .replace_strict({k: dt.date(2020, 1, int(v) - 2000) for k, v in codes.items()})
            .alias("day"),
        )
        floats = sample(
            Singleton("collapse", using={2004.0: 2001.0, 2005.0: 2002.0}), data, stratum="code"
        )
        assert floats.singletons["handled"].to_list() == ["collapse -> 2001", "collapse -> 2002"]
        dates = sample("collapse", data, stratum="day")
        # Rebalanced: once East joins 2020-01-01, West goes to the next smallest.
        assert dates.singletons["handled"].to_list() == [
            "collapse -> 2020-01-01",
            "collapse -> 2020-01-02",
        ]

    def test_element_design(self):
        rows = SURVEY.filter(
            pl.col("cluster").is_in(["c01", "c03", "c07"]).not_() | (pl.col("region") == "East")
        )
        s = svy.Sample(rows.head(13), svy.Design(stratum="region", wgt="w", singleton="skip"))
        assert s.singletons.columns == ["region", "n", "handled"]
        assert (s.singletons["n"] == 1).all()

    def test_a_design_without_strata(self):
        s = svy.Sample(SURVEY, svy.Design(psu="cluster", wgt="w"))
        assert s.singletons.is_empty() and s.n_singletons == 0

    def test_properties_not_methods(self):
        assert isinstance(type(sample()).singletons, property)
        assert isinstance(type(sample()).n_singletons, property)


# ---------------------------------------------------------------------------
# sample.domain_singletons
# ---------------------------------------------------------------------------


class TestDomainSingletons:
    def test_where(self):
        found = sample("center").domain_singletons(where=svy.col("income") < 1000)
        assert found.to_dicts() == [{"region": "Center", "cluster": "c06", "n": 1, "n_psus": 2}]

    def test_by(self):
        data = SURVEY.with_columns(high=pl.col("income") >= 1000)
        found = sample(data=data).domain_singletons(by="high")
        assert found.columns == ["high", "region", "cluster", "n", "n_psus"]
        assert found.height > 0 and (found["n_psus"] > 1).all()

    def test_matches_the_estimate_finding(self):
        low = svy.col("income") < 1000
        s = sample("center")
        found = s.domain_singletons(where=low)
        finding = s.estimation.mean("income", where=low).findings[0]
        assert [f"region={r}" for r in found["region"]] == finding.extra["pairs"]

    def test_is_a_method_that_needs_a_domain(self):
        assert sample().domain_singletons().is_empty()


# ---------------------------------------------------------------------------
# The removed namespace
# ---------------------------------------------------------------------------


class TestRemovedNamespace:
    def test_access_raises_a_guiding_error(self):
        s = sample()
        with pytest.raises(SingletonAPIRemoved) as err:
            s.singleton.center()
        e = err.value
        assert isinstance(e, AttributeError) and isinstance(e, SingletonError)
        assert e.code == "SINGLETON_API_REMOVED"
        assert 'svy.Design(..., singleton="center")' in e.detail
        assert "sample.singletons" in e.detail and "sample.domain_singletons" in e.detail
        assert "sample.wrangling.recode(psu_column" in e.detail
        assert '"self_representing"' in e.hint

    def test_hasattr_is_false(self):
        assert not hasattr(sample(), "singleton")

    def test_other_missing_attributes_are_plain(self):
        with pytest.raises(AttributeError) as err:
            sample().not_a_thing  # noqa: B018
        assert not isinstance(err.value, SingletonAPIRemoved)

    def test_removed_names(self):
        for name in ("SingletonFacet", "SingletonResult", "SingletonSummary", "SingletonHandling"):
            assert not hasattr(svy, name)
        assert svy.SingletonInfo and svy.StratumInfo  # a collapse callable receives them


# ---------------------------------------------------------------------------
# svy.utils.deprecated (release-v2026 §4.5)
# ---------------------------------------------------------------------------


def test_the_deprecation_helper():
    class K:
        @deprecated(since="1.0", remove_in="2.0", use="K.g()")
        def f(self, x):
            """Doc."""
            return x + 1

    with pytest.deprecated_call(match=r"K.f\(\) is deprecated since svy 1.0"):
        assert K().f(1) == 2
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        K().f(1)
    assert rec[0].filename == __file__
    assert K.f.__doc__.startswith(".. deprecated:: 1.0")
    assert "will be removed in 2.0. Use K.g() instead." in K.f.__deprecated__
