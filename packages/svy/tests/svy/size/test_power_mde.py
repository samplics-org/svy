"""Minimum detectable effect given n and power.

Expected values come from base R (``uniroot`` at tol 1e-14 on the closed-form power, and
closed forms for the one-sided means) and from Stata ``power onemean ..., knownsd`` /
``power oneproportion``; none is read from svy.
"""

from __future__ import annotations

import pytest

from svy.errors import Code, SelectionError, UsageError
from svy.size import Power


pytestmark = pytest.mark.filterwarnings("ignore::svy.errors.ExperimentalWarning")

LAM = 2.80158179  # solves pnorm(l - qnorm(.975)) + pnorm(-l - qnorm(.975)) = .8


def _code(excinfo) -> str:
    return str(excinfo.value.code)


class TestMdeMean:
    def test_two_sided(self):
        # R: LAM / sqrt(32) = 0.495254369912; Stata power onemean 0, sd(1) knownsd n(32) power(.8): 0.4953
        r = Power().mde_mean(0, 1, 32)
        assert r.effect == pytest.approx(0.495254369912, abs=1e-10)
        assert r.power == 0.8 and r.n == 32

    def test_lower_is_the_mirror(self):
        assert Power().mde_mean(0, 1, 32, direction="lower").effect == pytest.approx(
            -0.495254369912, abs=1e-10
        )

    def test_one_sided(self):
        # R: (qnorm(.95) + qnorm(.8)) / sqrt(32)
        assert Power().mde_mean(0, 1, 32, two_sides=False).effect == pytest.approx(
            0.439550808782, abs=1e-10
        )

    def test_non_inferiority_margin(self):
        # one-sided, delta = -.2: eps = delta + (qnorm(.95) + qnorm(.8)) / sqrt(32)
        r = Power().mde_mean(0, 1, 32, two_sides=False, delta=-0.2)
        assert r.effect == pytest.approx(0.239550808782, abs=1e-10)

    def test_alpha_and_power(self):
        r = Power().mde_mean(0, 1, 100, alpha=0.01, power=0.9)
        assert r.effect == pytest.approx(0.385738086874, abs=1e-10)

    def test_deff_fpc_nonresponse(self):
        # R: n_eff = 32.4846292948, effect = LAM / sqrt(n_eff)
        r = Power().mde_mean(0, 1, 72, deff=2, pop_size=500, resp_rate=0.8)
        assert r.effect == pytest.approx(0.491546205461, abs=1e-10)

    def test_scales_with_sigma(self):
        assert Power().mde_mean(5, 3, 32).effect == pytest.approx(3 * 0.495254369912, abs=1e-9)

    def test_census_detects_any_effect(self):
        assert Power().mde_mean(0, 1, 100, pop_size=100).effect == 0.0
        assert Power().mde_mean(0, 1, 100, pop_size=100, two_sides=False, delta=0.1).effect == 0.1

    def test_stratified(self):
        out = Power().mde_mean(0, {"a": 1, "b": 2}, {"a": 32, "b": 128})
        assert out.effect == pytest.approx(
            {"a": 0.495254369912, "b": 2 * LAM / 128**0.5}, abs=1e-8
        )

    def test_row_and_target(self):
        pw = Power().mde_mean(0, 1, 32)
        assert pw.target.goal == "mde_mean" and pw.target.solve == "effect"
        assert pw.target.args["power"] == 0.8 and pw.rows[0].n_eff == 32

    @pytest.mark.parametrize("two_sides", [True, False])
    @pytest.mark.parametrize("power", [0.2, 0.5, 0.8, 0.95, 0.999])
    def test_power_at_the_mde_is_the_target(self, two_sides, power):
        e = Power().mde_mean(1, 2, 57, power=power, two_sides=two_sides, deff=1.4).effect
        got = Power().test_mean(1, 1 + e, 2, 57, two_sides=two_sides, deff=1.4).power
        assert got == pytest.approx(power, abs=1e-9)


class TestMdeProp:
    @pytest.mark.parametrize(
        "method, effect",
        [
            ("wald", 0.0985961958285),
            (
                "score",
                0.0999610328551,
            ),  # Stata power oneproportion .5, n(194) power(.8): delta 0.1000
            ("arcsine", 0.0998941450796),
        ],
    )
    def test_upper_and_lower(self, method, effect):
        up = Power().mde_prop(0.5, 194, method=method).effect
        down = Power().mde_prop(0.5, 194, method=method, direction="lower").effect
        assert up == pytest.approx(effect, abs=1e-10)
        assert down == pytest.approx(-effect, abs=1e-10)

    def test_the_two_sides_differ_away_from_one_half(self):
        # R, score, p0 = .3, n = 100
        up = Power().mde_prop(0.3, 100, method="score").effect
        down = Power().mde_prop(0.3, 100, method="score", direction="lower").effect
        assert up == pytest.approx(0.131500422613, abs=1e-10)
        assert down == pytest.approx(-0.122009259858, abs=1e-10)

    @pytest.mark.parametrize("method", ["wald", "score", "arcsine"])
    @pytest.mark.parametrize("direction", ["upper", "lower"])
    @pytest.mark.parametrize("two_sides", [True, False])
    def test_power_at_the_mde_is_the_target(self, method, direction, two_sides):
        kw = dict(method=method, two_sides=two_sides, deff=1.3, pop_size=900)
        e = Power().mde_prop(0.35, 120, direction=direction, **kw).effect
        got = Power().test_prop(0.35, 0.35 + e, 120, **kw).power
        assert got == pytest.approx(0.8, abs=1e-8)

    def test_larger_n_detects_smaller_effects(self):
        effects = [Power().mde_prop(0.4, n).effect for n in (50, 100, 200, 400, 800)]
        assert effects == sorted(effects, reverse=True)

    def test_census(self):
        assert Power().mde_prop(0.4, 100, pop_size=100).effect == 0.0

    def test_stratified(self):
        out = Power().mde_prop({"a": 0.5, "b": 0.3}, 100, method="score")
        assert out.effect["b"] == pytest.approx(0.131500422613, abs=1e-10)
        assert out.effect["a"] == Power().mde_prop(0.5, 100, method="score").effect

    @pytest.mark.parametrize("method", ["score", "arcsine"])
    def test_too_few_units_cannot_reach_the_power(self, method):
        with pytest.raises(SelectionError) as e:
            Power().mde_prop(0.5, 2, method=method)
        assert _code(e) == Code.POWER_UNREACHABLE

    def test_margin_beyond_one(self):
        with pytest.raises(SelectionError) as e:
            Power().mde_prop(0.9, 100, two_sides=False, delta=0.2)
        assert _code(e) == Code.POWER_UNREACHABLE

    def test_margin_needs_wald(self):
        with pytest.raises(SelectionError) as e:
            Power().mde_prop(0.5, 100, two_sides=False, delta=0.05, method="score")
        assert _code(e) == Code.SIZE_METHOD_MARGIN

    def test_wald_near_the_boundary_is_still_a_proportion(self):
        e = Power().mde_prop(0.97, 20).effect
        assert 0 < e < 0.03


class TestMdeComparisons:
    def test_means_equal_groups(self):
        # R: LAM * sqrt(2 / 64)
        assert Power().mde_means(0, 1, 64).effect == pytest.approx(0.495254369912, abs=1e-10)

    def test_means_unequal_sd_and_sizes(self):
        # R: LAM * sqrt(1/40 + 4/80)
        r = Power().mde_means(0, 1, (40, 80), sigma2=2)
        assert r.effect == pytest.approx(0.767244770721, abs=1e-10)
        assert r.n == (40, 80)

    def test_means_full_overlap_is_paired(self):
        # R: LAM * sqrt((1 + 1 - 2 * .5) / 40)
        r = Power().mde_means(0, 1, 40, rho=0.5, overlap=1)
        assert r.effect == pytest.approx(0.44296897491, abs=1e-10)

    def test_means_lower(self):
        assert Power().mde_means(0, 1, 64, direction="lower").effect == pytest.approx(
            -0.495254369912, abs=1e-10
        )

    @pytest.mark.parametrize(
        "var_mode, effect", [("alt-props", 0.0999483338313), ("pooled-prop", 0.100451330974)]
    )
    def test_props(self, var_mode, effect):
        # R: uniroot on the two-sided power at 385 per group
        assert Power().mde_props(0.5, 385, var_mode=var_mode).effect == pytest.approx(
            effect, abs=1e-10
        )

    def test_props_deff_pair(self):
        # R: n_eff = (300 / 1.5, 400 / 2) = (200, 200), p1 = .3
        up = Power().mde_props(0.3, (300, 400), deff=(1.5, 2)).effect
        down = Power().mde_props(0.3, (300, 400), deff=(1.5, 2), direction="lower").effect
        assert up == pytest.approx(0.133715954154, abs=1e-10)
        assert down == pytest.approx(-0.118611015902, abs=1e-10)

    @pytest.mark.parametrize("var_mode", ["alt-props", "pooled-prop", "max-var"])
    @pytest.mark.parametrize("direction", ["upper", "lower"])
    def test_props_power_at_the_mde_is_the_target(self, var_mode, direction):
        kw = dict(var_mode=var_mode, rho=0.3, overlap=0.5, resp_rate=(0.9, 0.8))
        e = Power().mde_props(0.35, (120, 150), direction=direction, **kw).effect
        got = Power().compare_props(0.35, 0.35 + e, (120, 150), **kw).power
        assert got == pytest.approx(0.8, abs=1e-8)

    def test_means_power_at_the_mde_is_the_target(self):
        kw = dict(sigma2=1.5, alloc_ratio=1.0, deff=(1.2, 1.6), two_sides=False)
        e = Power().mde_means(2, 1, (80, 60), **kw).effect
        assert Power().compare_means(2, 2 + e, 1, (80, 60), **kw).power == pytest.approx(
            0.8, abs=1e-8
        )

    def test_census_of_both_groups(self):
        assert Power().mde_props(0.4, 100, pop_size=100).effect == 0.0

    def test_stratified_pairs(self):
        out = Power().mde_means(0, 1, {"a": (64, 64), "b": (40, 80)}, sigma2={"a": None, "b": 2})
        assert out.effect["a"] == pytest.approx(0.495254369912, abs=1e-10)
        assert out.effect["b"] == pytest.approx(0.767244770721, abs=1e-10)


class TestErrors:
    @pytest.mark.parametrize("power", [0.05, 0.01, 1, 1.2, 0, float("nan")])
    def test_power_must_exceed_alpha_and_stay_below_one(self, power):
        with pytest.raises(SelectionError) as e:
            Power().mde_mean(0, 1, 32, power=power)
        assert _code(e) == Code.INVALID_RANGE and "power" in str(e.value)

    def test_power_below_a_stratum_alpha(self):
        with pytest.raises(SelectionError) as e:
            Power().mde_mean(0, 1, 32, power=0.5, alpha={"a": 0.05, "b": 0.6})
        assert _code(e) == Code.INVALID_RANGE

    def test_equivalence_has_no_minimum_effect(self):
        with pytest.raises(SelectionError) as e:
            Power().mde_mean(0, 1, 32, delta=0.5)
        assert _code(e) == Code.INVALID_VALUE and "equivalence" in str(e.value)

    def test_lower_needs_no_margin(self):
        with pytest.raises(SelectionError) as e:
            Power().mde_mean(0, 1, 32, two_sides=False, delta=0.1, direction="lower")
        assert _code(e) == Code.INVALID_VALUE and "direction" in str(e.value)

    @pytest.mark.parametrize("direction", ["up", "both", None, "Upper"])
    def test_direction_choices(self, direction):
        with pytest.raises(SelectionError) as e:
            Power().mde_prop(0.5, 100, direction=direction)
        assert _code(e) == Code.INVALID_CHOICE

    def test_inputs_are_checked_like_the_power_goals(self):
        with pytest.raises(SelectionError) as e:
            Power().mde_mean(0, 0, 32)
        assert _code(e) == Code.INVALID_RANGE and "sigma" in str(e.value)
        with pytest.raises(SelectionError) as e:
            Power().mde_prop(1, 32)
        assert _code(e) == Code.INVALID_RANGE
        with pytest.raises(SelectionError) as e:
            Power().mde_mean(0, 1, 101, pop_size=100)
        assert _code(e) == Code.POWER_N_EXCEEDS_POPULATION
        with pytest.raises(UsageError) as e:
            Power().mde_means(0, 1, (40, 80), alloc_ratio=2)
        assert _code(e) == Code.PARAM_GIVEN_TWICE

    def test_method_names_the_caller(self):
        with pytest.raises(SelectionError, match="Power.mde_props"):
            Power().mde_props(0.5, 100, var_mode="pooled")


def test_the_mde_methods_are_experimental():
    for name in ("mde_mean", "mde_prop", "mde_means", "mde_props"):
        assert getattr(Power, name).__svy_experimental__ is True
