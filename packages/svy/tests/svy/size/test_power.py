"""One-sample test sizing and power given n.

Expected values come from base R (qnorm/pnorm closed forms, ``power.t.test``) and from Stata
``power onemean ..., knownsd`` / ``power oneproportion``; none is read from svy.
"""

from __future__ import annotations

import warnings

import pytest

import svy

from svy.errors import Code, ExperimentalWarning, SelectionError, UsageError
from svy.size import Power, PowerRow, SampleSize, TargetPower, TargetTestMean, TargetTestProp


pytestmark = pytest.mark.filterwarnings("ignore::svy.errors.ExperimentalWarning")


def _code(excinfo) -> str:
    return str(excinfo.value.code)


# =============================================================================
# Sample size
# =============================================================================


class TestSizeMean:
    @pytest.mark.parametrize(
        "kwargs, n",
        [
            ({}, 32),  # R: (qnorm(.975)+qnorm(.8))^2/.25 = 31.40; Stata knownsd N = 32
            ({"two_sides": False}, 25),  # 24.73; Stata onesided N = 25
            ({"alpha": 0.01, "power": 0.9}, 60),  # ((qnorm(.995)+qnorm(.9))/.5)^2 = 59.5
            ({"power": 0.5}, 16),  # z_b = 0: (qnorm(.975)/.5)^2 = 15.4
        ],
    )
    def test_matches_r_and_stata(self, kwargs, n):
        assert SampleSize().test_mean(0, 0.5, 1, **kwargs).n == n

    def test_superiority_margin(self):
        # R: ceiling(((qnorm(.95)+qnorm(.8))/(.5-.2))^2) = 69
        assert SampleSize().test_mean(0, 0.5, 1, two_sides=False, delta=0.2).n == 69

    def test_non_inferiority_margin(self):
        # epsilon = 0, delta = -0.2: ceiling(((qnorm(.95)+qnorm(.8))/.2)^2) = 155
        assert SampleSize().test_mean(0, 0, 1, two_sides=False, delta=-0.2).n == 155

    def test_equivalence(self):
        # eps = 0.1, delta = 0.5, z_b = qnorm((1 + .8)/2): ceiling(((qnorm(.95)+z_b)/.4)^2) = 54
        assert SampleSize().test_mean(0, 0.1, 1, delta=0.5).n == 54

    def test_negative_effect_gives_same_n(self):
        assert SampleSize().test_mean(0, -0.5, 1).n == 32

    def test_scales_with_sigma_and_effect(self):
        assert SampleSize().test_mean(10, 11, 2).n == SampleSize().test_mean(0, 0.5, 1).n

    def test_pipeline_deff_fpc_nonresponse(self):
        # n0 32 -> x2 = 64 -> N=500: ceil(500*64/563) = 57 -> /0.8 = 71.25 -> 72
        size = SampleSize().test_mean(0, 0.5, 1, deff=2, pop_size=500, resp_rate=0.8).size
        assert (size.n0, size.n1_deff, size.n2_fpc, size.n) == (32, 64, 57, 72)

    def test_stratified_equals_scalar_per_stratum(self):
        out = SampleSize().test_mean(
            mu0=0, mu1={"a": 0.5, "b": 0.25}, sigma={"a": 1, "b": 2}, deff={"a": 1, "b": 1.5}
        )
        a = SampleSize().test_mean(0, 0.5, 1)
        b = SampleSize().test_mean(0, 0.25, 2, deff=1.5)
        assert out.n == {"a": a.n, "b": b.n}

    def test_target_records_the_call(self):
        t = SampleSize().test_mean(0, 0.5, 1, deff=2).target
        assert isinstance(t, TargetTestMean)
        assert (t.mu0, t.mu1, t.sigma, t.deff, t.method) == (0, 0.5, 1, 2, "wald")

    def test_replaces_the_goal_held(self):
        ss = SampleSize().allocate(20, pop_size={"a": 10, "b": 30})
        assert ss.test_mean(0, 0.5, 1).n == 32 and ss.allocation is None


class TestSizeProp:
    @pytest.mark.parametrize(
        "method, n",
        [
            ("wald", 189),  # 188.37; Stata test(wald) N = 189
            ("score", 194),  # 193.85; Stata default N = 194
            ("arcsine", 194),  # 193.58: ((qnorm(.975)+qnorm(.8))/h)^2, h = ES.h(.6, .5)
        ],
    )
    def test_two_sided(self, method, n):
        assert SampleSize().test_prop(0.5, 0.6, method=method).n == n

    @pytest.mark.parametrize(
        "method, n", [("wald", 149), ("score", 153), ("arcsine", 153)]
    )  # 148.38, 152.46, 152.49
    def test_one_sided(self, method, n):
        assert SampleSize().test_prop(0.5, 0.6, method=method, two_sides=False).n == n

    def test_default_method_is_wald(self):
        assert SampleSize().test_prop(0.5, 0.6).target.method == "wald"

    def test_decreasing_proportion(self):
        # R: ((qnorm(.975)+qnorm(.8)) * sqrt(.2*.8) / .1)^2 = 125.58
        assert SampleSize().test_prop(0.3, 0.2).n == 126

    def test_arcsine_is_symmetric_in_direction(self):
        assert SampleSize().test_prop(0.8, 0.9, method="arcsine").n == 98  # R: 97.9

    def test_pipeline(self):
        # R: n0 189 -> ceil(1.5 * 189) = 284 -> ceil(2000*284/2283) = 249 -> ceil(249/.9) = 277
        size = SampleSize().test_prop(0.5, 0.6, deff=1.5, pop_size=2000, resp_rate=0.9).size
        assert (size.n0, size.n1_deff, size.n2_fpc, size.n) == (189, 284, 249, 277)

    def test_stratified(self):
        out = SampleSize().test_prop({"a": 0.5, "b": 0.3}, {"a": 0.6, "b": 0.2})
        assert out.n == {"a": 189, "b": 126}

    def test_target(self):
        t = SampleSize().test_prop(0.5, 0.6, method="score").target
        assert isinstance(t, TargetTestProp) and t.method == "score"


class TestSizeErrors:
    @pytest.mark.parametrize(
        "call",
        [
            lambda: SampleSize().test_mean(0, 0, 1),  # eps == 0
            lambda: SampleSize().test_mean(0, 0.2, 1, two_sides=False, delta=0.2),  # eps == delta
            lambda: SampleSize().test_mean(0, 0.1, 1, two_sides=False, delta=0.3),  # eps < delta
            lambda: SampleSize().test_mean(0, 0.6, 1, delta=0.5),  # |eps| >= delta, equivalence
            lambda: SampleSize().test_prop(0.5, 0.5),
            lambda: SampleSize().test_prop(0.5, 0.5, method="arcsine"),
        ],
    )
    def test_infeasible(self, call):
        with pytest.raises(SelectionError) as e:
            call()
        assert _code(e) == Code.INFEASIBLE_DESIGN

    def test_infeasible_names_the_method(self):
        with pytest.raises(SelectionError, match="SampleSize.test_mean"):
            SampleSize().test_mean(0, 0.1, 1, two_sides=False, delta=0.3)

    @pytest.mark.parametrize("method", ["score", "arcsine"])
    def test_margin_needs_wald(self, method):
        with pytest.raises(SelectionError) as e:
            SampleSize().test_prop(0.5, 0.6, method=method, delta=0.05)
        assert _code(e) == Code.SIZE_METHOD_MARGIN
        assert "delta" in str(e.value)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"sigma": 0},
            {"sigma": -1},
            {"sigma": float("nan")},
            {"sigma": float("inf")},
            {"power": 0},
            {"power": 1},
            {"deff": 0},
            {"deff": float("inf")},
            {"resp_rate": 0},
            {"resp_rate": 1.1},
            {"pop_size": 0},
            {"pop_size": -5},
        ],
    )
    def test_mean_ranges(self, kwargs):
        args = {"mu0": 0, "mu1": 0.5, "sigma": 1, **kwargs}
        with pytest.raises(SelectionError) as e:
            SampleSize().test_mean(**args)
        assert _code(e) == Code.INVALID_RANGE
        assert next(iter(kwargs)) in str(e.value)

    @pytest.mark.parametrize(
        "p0, p1, bad",
        [(0, 0.5, "p0"), (1, 0.5, "p0"), (0.5, 0, "p1"), (0.5, 1, "p1"), (0.5, 1.2, "p1")],
    )
    def test_proportions_stay_inside_the_unit_interval(self, p0, p1, bad):
        with pytest.raises(SelectionError) as e:
            SampleSize().test_prop(p0, p1)
        assert _code(e) == Code.INVALID_RANGE and bad in str(e.value)

    def test_alpha(self):
        with pytest.raises(UsageError) as e:
            SampleSize().test_mean(0, 0.5, 1, alpha=1.0)
        assert _code(e) == Code.INVALID_RANGE

    @pytest.mark.parametrize("bad", [True, "0.5", None, [0.5]])
    def test_numbers_only(self, bad):
        with pytest.raises(SelectionError) as e:
            SampleSize().test_mean(0, bad, 1)
        assert _code(e) == Code.INVALID_TYPE

    def test_method_choices(self):
        with pytest.raises(SelectionError) as e:
            SampleSize().test_prop(0.5, 0.6, method="fleiss")
        assert _code(e) == Code.INVALID_CHOICE
        with pytest.raises(SelectionError) as e:
            SampleSize().test_mean(0, 0.5, 1, method="score")
        assert _code(e) == Code.INVALID_CHOICE

    def test_method_is_case_insensitive(self):
        assert SampleSize().test_prop(0.5, 0.6, method="Score").n == 194

    @pytest.mark.parametrize("two_sides", [1, "yes", None])
    def test_two_sides_is_a_bool(self, two_sides):
        with pytest.raises(SelectionError) as e:
            SampleSize().test_mean(0, 0.5, 1, two_sides=two_sides)
        assert _code(e) == Code.INVALID_TYPE

    def test_mapping_keys_must_match(self):
        with pytest.raises(UsageError) as e:
            SampleSize().test_mean(0, {"a": 0.5, "b": 0.5}, {"a": 1, "c": 1})
        assert _code(e) == Code.MAPPING_KEYS_MISMATCH

    def test_empty_mapping(self):
        with pytest.raises(UsageError) as e:
            SampleSize().test_mean(0, {}, 1)
        assert _code(e) == Code.MAPPING_KEYS_MISMATCH


# =============================================================================
# Power
# =============================================================================


class TestPowerMean:
    def test_two_sided(self):
        # R: pnorm(.5*sqrt(32) - qnorm(.975)) + pnorm(-.5*sqrt(32) - qnorm(.975)); Stata 0.8074
        assert Power().test_mean(0, 0.5, 1, 32).power == pytest.approx(0.807430419433, abs=1e-10)

    def test_one_sided(self):
        # R: pnorm(.5*sqrt(32) - qnorm(.95)); Stata onesided 0.8817
        p = Power().test_mean(0, 0.5, 1, 32, two_sides=False).power
        assert p == pytest.approx(0.881709031778, abs=1e-10)

    def test_one_sided_side_follows_the_effect(self):
        up = Power().test_mean(0, 0.5, 1, 32, two_sides=False).power
        down = Power().test_mean(0, -0.5, 1, 32, two_sides=False).power
        assert down == pytest.approx(up, abs=1e-14)

    def test_two_sided_symmetric_in_effect(self):
        assert Power().test_mean(0, -0.5, 1, 32).power == pytest.approx(
            Power().test_mean(0, 0.5, 1, 32).power, abs=1e-14
        )

    @pytest.mark.parametrize("two_sides, alpha", [(True, 0.05), (False, 0.05), (True, 0.2)])
    def test_no_effect_gives_alpha(self, two_sides, alpha):
        p = Power().test_mean(3, 3, 1, 50, two_sides=two_sides, alpha=alpha).power
        assert p == pytest.approx(alpha, abs=1e-12)

    def test_noninferiority(self):
        # R: pnorm((0 + .2)*sqrt(100) - qnorm(.95))
        p = Power().test_mean(0, 0, 1, 100, two_sides=False, delta=-0.2).power
        assert p == pytest.approx(0.638760031312, abs=1e-10)

    def test_superiority_margin(self):
        # R: pnorm((.5 - .2)*10 - qnorm(.95))
        p = Power().test_mean(0, 0.5, 1, 100, two_sides=False, delta=0.2).power
        assert p == pytest.approx(0.91231453675, abs=1e-10)

    def test_effect_at_the_margin_gives_alpha(self):
        p = Power().test_mean(0, 0.2, 1, 100, two_sides=False, delta=0.2).power
        assert p == pytest.approx(0.05, abs=1e-12)

    def test_effect_inside_a_one_sided_margin_is_not_an_error(self):
        p = Power().test_mean(0, 0.1, 1, 100, two_sides=False, delta=0.3).power
        assert 0.0 < p < 0.05

    def test_equivalence_is_exact(self):
        # R: pnorm((.5-.1)*10 - qnorm(.95)) + pnorm((.5+.1)*10 - qnorm(.95)) - 1
        p = Power().test_mean(0, 0.1, 1, 100, delta=0.5).power
        assert p == pytest.approx(0.990735645725, abs=1e-10)

    def test_equivalence_outside_the_margin_stays_in_the_unit_interval(self):
        p = Power().test_mean(0, 0.6, 1, 100, delta=0.5).power
        assert p == pytest.approx(0.00408631306003, abs=1e-10)

    def test_deff_fpc_nonresponse_are_undone_exactly(self):
        # R: n_eff = 72*.8*(500-1)/(500-72*.8)/2 = 32.4846292948
        r = Power().test_mean(0, 0.5, 1, 72, deff=2, pop_size=500, resp_rate=0.8).rows[0]
        assert r.n == 72 and r.n_eff == pytest.approx(32.4846292948, abs=1e-9)
        assert r.power == pytest.approx(0.813214232808, abs=1e-10)

    def test_census_has_no_sampling_variance(self):
        p = Power().test_mean(0, 0.01, 1, 100, pop_size=100).power
        assert p == pytest.approx(1.0, abs=1e-15)

    def test_census_without_effect_gives_alpha(self):
        assert Power().test_mean(0, 0, 1, 100, pop_size=100).power == pytest.approx(0.05)

    def test_more_respondents_than_population(self):
        with pytest.raises(SelectionError) as e:
            Power().test_mean(0, 0.5, 1, 101, pop_size=100)
        assert _code(e) == Code.POWER_N_EXCEEDS_POPULATION
        assert "pop_size" in str(e.value)

    def test_nonresponse_can_bring_n_under_the_population(self):
        assert Power().test_mean(0, 0.5, 1, 110, pop_size=100, resp_rate=0.9).power > 0.99

    def test_fractional_and_tiny_n(self):
        p = Power().test_mean(0, 0.5, 1, 0.5).power
        assert p == pytest.approx(0.06443896766, abs=1e-9)  # R, both tails at n = 0.5

    def test_power_rises_with_n_and_effect(self):
        by_n = [Power().test_mean(0, 0.3, 1, n).power for n in (10, 20, 40, 80, 160)]
        by_e = [Power().test_mean(0, e, 1, 40).power for e in (0.05, 0.1, 0.2, 0.4, 0.8)]
        assert by_n == sorted(by_n) and by_e == sorted(by_e)

    def test_close_to_t_test_at_large_n(self):
        # R: power.t.test(n = 2000, delta = .05, sd = 1, type = "one.sample")$power = 0.608353
        p = Power().test_mean(0, 0.05, 1, 2000).power
        assert p == pytest.approx(0.608353, abs=1e-3)

    def test_extreme_sigma(self):
        assert Power().test_mean(0, 0.5, 1e9, 32).power == pytest.approx(0.05, abs=1e-6)
        assert Power().test_mean(0, 0.5, 1e-9, 32).power == pytest.approx(1.0)

    def test_stratified_equals_scalar_per_stratum(self):
        out = Power().test_mean(
            0,
            mu1={"a": 0.5, "b": 0.25},
            sigma={"a": 1, "b": 2},
            n={"a": 32, "b": 100},
            pop_size={"a": 500, "b": None},
        )
        a = Power().test_mean(0, 0.5, 1, 32, pop_size=500).power
        b = Power().test_mean(0, 0.25, 2, 100).power
        assert out.power == pytest.approx({"a": a, "b": b}, abs=1e-14)
        assert out.effect == {"a": 0.5, "b": 0.25} and out.n == {"a": 32, "b": 100}

    def test_scalar_n_broadcasts_to_strata(self):
        out = Power().test_mean(0, {"a": 0.5, "b": 0.25}, 1, 32)
        assert out.n == {"a": 32, "b": 32}

    def test_alpha_by_stratum(self):
        out = Power().test_mean(0, 0.5, 1, 32, alpha={"a": 0.05, "b": 0.01})
        assert out.power["a"] > out.power["b"]


class TestPowerProp:
    @pytest.mark.parametrize(
        "method, power",
        [
            ("wald", 0.811424884857),  # Stata test(wald) 0.8114
            ("score", 0.800313838422),  # Stata default 0.8003
            ("arcsine", 0.800841470806),  # pwr.p.test form
        ],
    )
    def test_two_sided(self, method, power):
        assert Power().test_prop(0.5, 0.6, 194, method=method).power == pytest.approx(
            power, abs=1e-10
        )

    @pytest.mark.parametrize(
        "method, power",
        [("wald", 0.88459340176), ("score", 0.877858625487), ("arcsine", 0.876922187265)],
    )
    def test_one_sided(self, method, power):
        p = Power().test_prop(0.5, 0.6, 194, method=method, two_sides=False).power
        assert p == pytest.approx(power, abs=1e-10)

    def test_effect_is_on_the_proportion_scale(self):
        assert Power().test_prop(0.5, 0.6, 194, method="arcsine").effect == pytest.approx(0.1)

    def test_n_eff_with_deff(self):
        r = Power().test_prop(0.5, 0.6, 194, deff=2).rows[0]
        assert r.n_eff == pytest.approx(97.0)
        assert r.power == Power().test_prop(0.5, 0.6, 97).power

    def test_margin_needs_wald(self):
        with pytest.raises(SelectionError) as e:
            Power().test_prop(0.5, 0.6, 194, method="score", delta=0.05)
        assert _code(e) == Code.SIZE_METHOD_MARGIN

    def test_stratified(self):
        out = Power().test_prop({"a": 0.5, "b": 0.4}, 0.6, 194)
        assert out.power["a"] == pytest.approx(0.811424884857, abs=1e-10)
        assert out.power["b"] > 0.99

    @pytest.mark.parametrize("p", [0, 1, -0.1, 1.1])
    def test_unit_interval(self, p):
        with pytest.raises(SelectionError) as e:
            Power().test_prop(p, 0.5, 100)
        assert _code(e) == Code.INVALID_RANGE

    @pytest.mark.parametrize("n", [0, -3, float("nan"), float("inf")])
    def test_n_must_be_positive_and_finite(self, n):
        with pytest.raises(SelectionError) as e:
            Power().test_prop(0.5, 0.6, n)
        assert _code(e) == Code.INVALID_RANGE and "n" in str(e.value)

    def test_extreme_proportions(self):
        assert Power().test_prop(0.5, 1 - 1e-9, 50).power == pytest.approx(1.0)
        assert Power().test_prop(0.5, 1e-9, 50, method="arcsine").power == pytest.approx(1.0)


class TestPowerOutput:
    def test_target(self):
        t = Power().test_mean(0, 0.5, 1, 32, deff=2).target
        assert isinstance(t, TargetPower)
        assert (t.goal, t.solve) == ("test_mean", "power")
        assert t.args["n"] == 32 and t.args["deff"] == 2 and t.args["mu1"] == 0.5

    def test_rows_and_properties_unstratified(self):
        pw = Power().test_prop(0.5, 0.6, 194)
        assert pw.rows == [
            PowerRow(stratum=None, n=194, n_eff=194, effect=pytest.approx(0.1), power=pw.power)
        ]
        assert isinstance(pw.power, float) and pw.n == 194

    def test_a_new_goal_replaces_the_held_one(self):
        pw = Power()
        pw.test_mean(0, 0.5, 1, 32)
        pw.test_prop(0.5, 0.6, 194)
        assert pw.target.goal == "test_prop" and len(pw.rows) == 1

    def test_empty(self):
        pw = Power()
        assert pw.power is None and pw.n is None and pw.effect is None and pw.rows == []
        assert "Power" in repr(pw) and "no power" in pw.__plain_str__()

    def test_to_polars(self):
        df = Power().test_mean(0, {"a": 0.5, "b": 0.25}, 1, 32).to_polars()
        assert df.columns == ["stratum", "n", "n_eff", "effect", "power"]
        assert df["stratum"].to_list() == ["a", "b"]
        assert df["power"][0] == pytest.approx(0.807430419433, abs=1e-10)

    def test_to_polars_unstratified_has_no_stratum_column(self):
        assert Power().test_mean(0, 0.5, 1, 32).to_polars().columns == [
            "n",
            "n_eff",
            "effect",
            "power",
        ]

    def test_print(self):
        out = str(Power().test_mean(0, {"a": 0.5, "b": 0.25}, 1, 32))
        assert "Power" in out and "0.8074" in out and "stratum" in out
        assert repr(Power().test_mean(0, 0.5, 1, 32)) == (
            "Power(stratified=False, strata=1, goal=test_mean)"
        )

    def test_is_exported(self):
        assert svy.Power is Power and "Power" in svy.__all__


class TestExperimental:
    def test_warns_once_and_says_so(self):
        import sys

        exp = sys.modules["svy.utils.experimental"]
        exp._WARNED.discard("Power.test_mean")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Power().test_mean(0, 0.5, 1, 32)
            Power().test_mean(0, 0.5, 1, 32)
        assert [type(x.message) for x in w] == [ExperimentalWarning]
        assert "Experimental" in Power.test_mean.__doc__
        assert Power.test_prop.__svy_experimental__ is True

    def test_sample_size_goals_are_stable(self):
        assert not hasattr(SampleSize.test_mean, "__svy_experimental__")


# =============================================================================
# Round trip: the power at the size sized for
# =============================================================================


GRID = [
    {},
    {"two_sides": False},
    {"power": 0.9},
    {"alpha": 0.01},
    {"deff": 1.7, "resp_rate": 0.8},
    {"deff": 2.0, "pop_size": 500, "resp_rate": 0.8},
    {"pop_size": 120},
]


class TestRoundTrip:
    @pytest.mark.parametrize("kw", GRID)
    def test_mean(self, kw):
        n = SampleSize().test_mean(0, 0.4, 1, **kw).n
        target = kw.get("power", 0.8)
        pw = {k: v for k, v in kw.items() if k != "power"}
        assert Power().test_mean(0, 0.4, 1, n, **pw).power >= target

    @pytest.mark.parametrize("method", ["wald", "score", "arcsine"])
    @pytest.mark.parametrize("kw", GRID)
    def test_prop(self, kw, method):
        n = SampleSize().test_prop(0.3, 0.42, method=method, **kw).n
        target = kw.get("power", 0.8)
        pw = {k: v for k, v in kw.items() if k != "power"}
        assert Power().test_prop(0.3, 0.42, n, method=method, **pw).power >= target

    def test_without_ceilings_it_is_the_target(self):
        # n_eff = (z_a + z_b)^2 / eps^2 exactly gives the one-tail target
        from scipy.stats import norm

        n_eff = ((norm.isf(0.025) + norm.ppf(0.8)) / 0.5) ** 2
        p = Power().test_mean(0, 0.5, 1, n_eff).power
        assert p == pytest.approx(0.8, abs=1e-3)  # the far tail adds < 1e-3

    def test_equivalence_sizing_is_conservative(self):
        n = SampleSize().test_mean(0, 0.1, 1, delta=0.5).n
        assert Power().test_mean(0, 0.1, 1, n, delta=0.5).power >= 0.8
