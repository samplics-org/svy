# tests/svy/core/test_singleton_edge_cases.py
"""Edge cases of the declared singleton rule: multistage samples, within=,
messages, replicate designs, analyses after a filter, panels, in_domains and
the error paths of the rule and the deprecated methods."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import svy

from svy.core.design import Singleton
from svy.errors import MethodError
from svy.errors.singleton_errors import SingletonError


def _result(sample):
    """What the design's singleton rule did to the current data (internal)."""
    sample._sync_parts()
    return sample._singleton_result


def _detected(sample):
    """The singleton strata of the current data (internal)."""
    from svy.core.singleton import _Engine

    sample._sync_parts()
    return _Engine(sample, _sync=False).detected()


def _keys(sample):
    """svy's keys of the singleton strata of the current data (internal)."""
    return [s.stratum_key for s in _detected(sample)]


pytestmark = pytest.mark.filterwarnings("error")

# Strata a, b, c with 3 PSUs, d and e with one; 3 rows per PSU.
STRATA = {"a": 3, "b": 3, "c": 3, "d": 1, "e": 1}
REGION = {"a": "N", "b": "N", "c": "S", "d": "N", "e": "S"}
CODE = {"a": 2001.0, "b": 2002.0, "c": 2003.0, "d": 2004.0, "e": 2005.0}


def make_frame(strata: dict[str, int] = STRATA) -> pl.DataFrame:
    rows = []
    for st, n_psu in strata.items():
        for p in range(n_psu):
            for r in range(3):
                i = len(rows) + 1
                rows.append(
                    {
                        "id": i,
                        "st": st,
                        "reg": REGION.get(st, "N"),
                        "urb": "u" if st in "ab" else "r",
                        "code": CODE.get(st, 2009.0),
                        "psu": f"{st}{p}",
                        "y": float((i * 37) % 17 + 3 * p),
                        "x": int((i * 7) % 3 == 0),
                        "g": "u" if i % 2 else "v",
                        "h": i % 3,
                        "w": 1.0 + (i % 4) * 0.5,
                    }
                )
    return pl.DataFrame(rows)


DATA = make_frame()
DESIGN = dict(stratum="st", psu="psu", wgt="w")


def sample(data: pl.DataFrame = DATA, singleton=None, **kw) -> svy.Sample:
    return svy.Sample(data, svy.Design(**{**DESIGN, **kw}, singleton=singleton))


def resolved(s: svy.Sample):
    from svy.core.singleton import _Engine

    lr, f = _result(s), _Engine(s, _sync=False)
    if lr is None:
        return None
    if isinstance(lr.applied, dict) and lr.method == "collapse":
        keys = list(lr.applied)
        return dict(zip(f._key_values(keys), f._key_values([lr.applied[k] for k in keys])))
    return f._key_values([s.stratum_key for s in lr.detected])


def problem(s: svy.Sample) -> SingletonError:
    with pytest.raises(SingletonError) as err:
        s.estimation.mean("y")
    return err.value


def se(s: svy.Sample) -> float:
    return s.estimation.mean("y").estimates[0].se


# ---------------------------------------------------------------------------
# add_stage: the combined sample detects its singletons
# ---------------------------------------------------------------------------


def _stage1(singleton=None) -> svy.Sample:
    ea = pl.DataFrame(
        {
            "ea": [1, 2, 3, 4, 5, 6],
            "region": ["N", "N", "N", "S", "S", "S"],
            "mos": [100.0, 200.0, 300.0, 150.0, 250.0, 350.0],
        }
    )
    design = svy.Design(mos="mos", stratum="region", psu="ea", singleton=singleton)
    return svy.Sample(ea, design).sampling.pps_sys(n=2, by="region", rstate=5)


def _stage2(selected: list[int]) -> svy.Sample:
    ea = [e for e in range(1, 7) for _ in range(5)]
    hh = pl.DataFrame(
        {
            "hid": list(range(len(ea))),
            "ea": ea,
            "region": ["N" if e <= 3 else "S" for e in ea],
            "y": np.random.default_rng(3).normal(20, 4, len(ea)),
        }
    ).filter(pl.col("ea").is_in(selected))
    return svy.Sample(hh, svy.Design(stratum="region", psu="ea")).sampling.srs(
        n=3, by="ea", rstate=9
    )


class TestAddStage:
    def _combined(self, singleton=None) -> tuple[svy.Sample, int]:
        s1 = _stage1(singleton)
        selected = s1._data["ea"].to_list()
        return s1.sampling.add_stage(_stage2(selected)), selected[-1]

    def test_the_combined_sample_is_checked_like_a_built_one(self):
        combined, _ = self._combined()
        fresh = svy.Sample(combined.data, combined.design)
        assert combined.n_singletons == fresh.n_singletons == 0
        assert combined._internal_design["stratum"] is not None

    def test_a_singleton_is_detected_without_a_rule(self):
        combined, last = self._combined()
        one_psu = combined.wrangling.filter_records(svy.col("ea") != last)
        assert _keys(one_psu) == ["S"]
        with pytest.raises(SingletonError) as err:
            one_psu.estimation.mean("y")
        assert err.value.code == "SINGLETON_ERROR"

    def test_the_carried_rule_handles_it(self):
        combined, last = self._combined("center")
        one_psu = combined.wrangling.filter_records(svy.col("ea") != last)
        assert resolved(one_psu) == ["S"]
        fresh = svy.Sample(one_psu.data, one_psu.design)
        assert se(one_psu) == pytest.approx(se(fresh), rel=1e-12)


# ---------------------------------------------------------------------------
# within=
# ---------------------------------------------------------------------------


class TestWithin:
    def test_nulls_are_ignored(self):
        with_nulls = DATA.with_columns(
            pl.when(pl.col("id").is_in([1, 20, 31])).then(None).otherwise("reg").alias("reg")
        )
        clean = sample(singleton=Singleton("collapse", within="reg"))
        holey = sample(with_nulls, Singleton("collapse", within="reg"))
        assert resolved(holey) == resolved(clean) == {"d": "a", "e": "c"}
        assert se(holey) == pytest.approx(se(clean), rel=1e-12)

    def test_a_singleton_with_no_known_value_joins_only_such_strata(self):
        unknown = DATA.with_columns(
            pl.when(pl.col("st").is_in(["d", "b"])).then(None).otherwise("reg").alias("reg")
        )
        s = sample(unknown, Singleton("collapse", within="reg"))
        assert resolved(s) == {"d": "b", "e": "c"}
        none = DATA.with_columns(
            pl.when(pl.col("st") == "d").then(None).otherwise("reg").alias("reg")
        )
        assert problem(sample(none, Singleton("collapse", within="reg"))).code == (
            "NO_MERGE_TARGETS"
        )

    def test_a_column_that_varies_within_a_stratum_raises(self):
        err = problem(sample(singleton=Singleton("collapse", within="g")))
        assert err.code == "SINGLETON_WITHIN_INVALID"
        assert "'g' takes several values in 5 strata (e.g. 'a', 'b', 'c')" in err.detail

    def test_several_columns(self):
        s = sample(singleton=Singleton("collapse", within=["reg", "urb"]))
        # d (N, r) has no N/r stratum with several PSUs.
        assert problem(s).code == "NO_MERGE_TARGETS"
        t = sample(
            DATA.with_columns(
                pl.when(pl.col("st") == "d").then(pl.lit("u")).otherwise("urb").alias("urb")
            ),
            Singleton("collapse", within=["reg", "urb"]),
        )
        assert resolved(t) == {"d": "a", "e": "c"}

    def test_tuple_strata(self):
        s = sample(singleton=Singleton("collapse", within="reg"), stratum=("urb", "st"))
        assert resolved(s) == {("r", "d"): ("u", "a"), ("r", "e"): ("r", "c")}

    def test_a_mapping_across_within_raises(self):
        rule = Singleton("collapse", using={"d": "a", "e": "a"}, within="reg")
        err = problem(sample(singleton=rule))
        assert err.code == "SINGLETON_TARGET_OUTSIDE_WITHIN"
        assert err.detail == "the collapse mapping merges strata with different reg: 'e' -> 'a'."

    def test_a_mapping_inside_within(self):
        rule = Singleton("collapse", using={"d": "b", "e": "c"}, within="reg")
        assert resolved(sample(singleton=rule)) == {"d": "b", "e": "c"}

    def test_a_missing_within_column_is_named(self):
        with pytest.raises(ValueError, match="within/order_by read nope"):
            sample(singleton=Singleton("collapse", within="nope"))


# ---------------------------------------------------------------------------
# Values in messages as the data shows them
# ---------------------------------------------------------------------------


class TestMessages:
    def _floats(self, rule) -> svy.Sample:
        return sample(singleton=rule, stratum="code")

    def test_unmapped(self):
        err = problem(self._floats(Singleton("collapse", using={2004.0: 2001.0})))
        assert err.detail == "1 singleton stratum is not in the collapse mapping: 2005."

    def test_target_missing(self):
        rule = Singleton("collapse", using={2004.0: 2001.0, 2005.0: 1999.0})
        assert problem(self._floats(rule)).detail.endswith("no longer has: 1999.")

    def test_unused_and_changed(self):
        s = self._floats(
            Singleton("collapse", using={2004.0: 2001.0, 2005.0: 2003.0, 7.0: 2001.0})
        )
        (w,) = [w for w in s.warnings if w.code == "SINGLETON_MAPPING_UNUSED"]
        assert w.detail.endswith("so not collapsed: 7.")
        c = self._floats(Singleton("collapse"))
        r = c.wrangling.filter_records(svy.col("st") != "a")
        (w,) = [w for w in r.warnings if w.code == "SINGLETON_COLLAPSE_CHANGED"]
        assert "2004 -> 2002, 2005 -> 2003" in w.detail

    def test_singleton_error(self):
        err = problem(sample(stratum="code"))
        assert "code=2004 (PSU=d0, n=3)" in err.detail

    def test_tuple_strata(self):
        s = sample(
            singleton=Singleton("collapse", using={("N", 2004.0): ("N", 2001.0)}),
            stratum=("reg", "code"),
        )
        assert problem(s).detail.endswith("mapping: ('S', 2005).")


# ---------------------------------------------------------------------------
# Replicate designs: Taylor analyses need the rule, replication does not
# ---------------------------------------------------------------------------


@pytest.fixture
def rep_no_rule() -> svy.Sample:
    built = sample(singleton="skip").weighting.create_jk_wgts()
    return built.update_design(singleton=None)


class TestReplicateDesigns:
    def test_replication_needs_no_rule(self, rep_no_rule):
        assert rep_no_rule.estimation.mean("y", method="replication").estimates[0].se > 0
        assert rep_no_rule.glm.fit("y", x=["x"]).fitted.coefs[0].se > 0

    @pytest.mark.parametrize(
        "analysis",
        [
            lambda s: s.estimation.mean("y"),
            lambda s: s.categorical.tabulate("x"),
            lambda s: s.categorical.ttest("y", mean_h0=3),
            lambda s: s.categorical.ranktest("y", group="g", method="kruskal-wallis"),
        ],
        ids=["mean", "tabulate", "ttest", "ranktest"],
    )
    def test_taylor_analyses_need_a_rule(self, rep_no_rule, analysis):
        with pytest.raises(SingletonError) as err:
            analysis(rep_no_rule)
        assert err.value.code == "SINGLETON_ERROR"
        analysis(rep_no_rule.update_design(singleton="center"))


# ---------------------------------------------------------------------------
# Every analysis sees singletons a filter creates
# ---------------------------------------------------------------------------

ANALYSES = {
    "mean": lambda s: s.estimation.mean("y"),
    "tabulate": lambda s: s.categorical.tabulate("x"),
    "crosstab": lambda s: s.categorical.tabulate("x", "g"),
    "ttest": lambda s: s.categorical.ttest("y", mean_h0=3),
    "ttest2": lambda s: s.categorical.ttest("y", group="g"),
    "ranktest": lambda s: s.categorical.ranktest("y", group="g", method="kruskal-wallis"),
    "glm": lambda s: s.glm.fit("y", x=["x"]),
}


@pytest.mark.parametrize("analysis", list(ANALYSES))
def test_a_filter_that_creates_a_singleton_reaches_every_analysis(analysis):
    s = sample(make_frame({"a": 3, "b": 2, "c": 2}))
    ANALYSES[analysis](s)
    one_psu = s.wrangling.filter_records(svy.col("psu") != "b1")
    with pytest.raises(SingletonError) as err:
        ANALYSES[analysis](one_psu)
    assert err.value.code == "SINGLETON_ERROR"


@pytest.mark.parametrize("analysis", list(ANALYSES))
def test_a_rule_declared_ahead_reaches_every_analysis(analysis):
    s = sample(make_frame({"a": 3, "b": 2, "c": 2}), "center")
    one_psu = s.wrangling.filter_records(svy.col("psu") != "b1")
    ANALYSES[analysis](one_psu)
    assert resolved(one_psu) == ["b"]


# ---------------------------------------------------------------------------
# Collapse findings over refreshes
# ---------------------------------------------------------------------------


def test_a_generator_rstate_records_a_change_only_when_the_mapping_moves():
    rule = Singleton("collapse", using="largest", rstate=np.random.default_rng(1))
    s = sample(singleton=rule)
    before = resolved(s)
    r = s.wrangling.filter_records(svy.col("id") != 1)
    changed = [w for w in r.warnings if w.code == "SINGLETON_COLLAPSE_CHANGED"]
    assert bool(changed) is (resolved(r) != before)


def test_an_unused_entry_is_recorded_once():
    s = sample(singleton=Singleton("collapse", using={"d": "a", "e": "c", "zz": "a"}))
    for i in range(3):
        s = s.wrangling.filter_records(svy.col("id") != 1000 + i)
        s.estimation.mean("y")
    assert [w.code for w in s.warnings].count("SINGLETON_MAPPING_UNUSED") == 1


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def _wave(w: int) -> pl.DataFrame:
    return DATA.with_columns(pl.col("y") + w)


def test_a_panel_with_a_rule():
    long = pl.concat(
        [_wave(1).with_columns(wave=pl.lit(1)), _wave(2).with_columns(wave=pl.lit(2))]
    )
    s = svy.Sample(
        long,
        svy.Design(stratum="st", wgt="w", case_id="id", wave="wave", singleton="center"),
    )
    assert s.estimation.mean("y", by="wave").estimates[0].se > 0
    assert _result(s) is None  # the case is the PSU: no singletons


def test_combine_samples_panel_carries_an_explicit_mapping():
    rule = Singleton("collapse", using={"d": "a", "e": "c"})
    waves = [sample(_wave(w), rule) for w in (1, 2)]
    combined = svy.combine_samples(waves, kind="panel", case_id="id")
    assert combined.design.singleton == rule
    # The panel's stratum is the one-column tuple ("st",).
    assert resolved(combined) == {("d",): ("a",), ("e",): ("c",)}
    fresh = svy.Sample(combined.data, combined.design)
    assert se(combined) == pytest.approx(se(fresh), rel=1e-12)


# ---------------------------------------------------------------------------
# in_domains
# ---------------------------------------------------------------------------


class TestDomainSingletons:
    def test_several_by_columns_and_tuple_strata(self):
        s = sample(stratum=("reg", "st"))
        found = s.domain_singletons(by=["g", "h"])
        assert found.columns == ["g", "h", "reg", "st", "psu", "n", "n_psus"]
        assert found.height > 0
        assert (found["n_psus"] > 1).all() and (found["n"] >= 1).all()

    def test_a_null_by_level_is_a_domain(self):
        holey = DATA.with_columns(
            pl.when(pl.col("psu") == "a0").then(None).otherwise("g").alias("g")
        )
        found = sample(holey).domain_singletons(by="g")
        assert None in found["g"].to_list()

    def test_by_and_where(self):
        s = sample()
        both = s.domain_singletons(by="g", where=svy.col("h") == 0)
        alone = s.domain_singletons(where=(svy.col("h") == 0) & (svy.col("g") == "u"))
        assert both.filter(pl.col("g") == "u")["st"].to_list() == alone["st"].to_list()

    def test_a_design_without_strata(self):
        found = svy.Sample(DATA, svy.Design(psu="psu", wgt="w")).domain_singletons(by="g")
        assert found.is_empty() and found.columns == ["g", "psu", "n", "n_psus"]


# ---------------------------------------------------------------------------
# on_domain_singletons="error" wording
# ---------------------------------------------------------------------------


def test_error_for_a_where_domain_counts_strata():
    s = sample(singleton=Singleton("skip", on_domain_singletons="error"))
    with pytest.raises(SingletonError) as err:
        s.estimation.mean("y", where=svy.col("psu").is_in(["a0", "b0", "c0", "d0", "e0"]))
    assert err.value.detail.startswith("3 strata have one PSU in the domain:")


def test_error_lists_five_pairs_and_counts_the_rest():
    s = sample(singleton=Singleton("skip", on_domain_singletons="error"))
    with pytest.raises(SingletonError) as err:
        s.estimation.mean("y", by="id")
    lines = err.value.detail.splitlines()
    assert lines[0].endswith("domain × stratum pairs have one PSU in the domain:")
    assert lines[5].startswith("  5. ") and lines[6].startswith("  ... and ")


# ---------------------------------------------------------------------------
# Error paths of the rule and the deprecated methods
# ---------------------------------------------------------------------------


def test_descending_must_be_a_bool():
    with pytest.raises(MethodError, match="descending is True or False"):
        Singleton("collapse", descending="yes")  # type: ignore[arg-type]


def test_a_callable_rule_prints_its_name():
    def nearest(singleton, candidates):
        return candidates[0]

    assert repr(Singleton("collapse", using=nearest)) == "Singleton('collapse', using=nearest)"
