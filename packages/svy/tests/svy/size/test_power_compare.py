"""Power given n for two-group comparisons.

Expected values come from base R closed forms and from Stata ``power twomeans ..., knownsd``
and ``power pairedmeans ..., knownsd`` (Stata's ``n()`` is the total of both groups); none is
read from svy.
"""

from __future__ import annotations

import pytest

from svy.errors import Code, SelectionError, UsageError
from svy.size import Power, SampleSize


pytestmark = pytest.mark.filterwarnings("ignore::svy.errors.ExperimentalWarning")


def _code(excinfo) -> str:
    return str(excinfo.value.code)


class TestCompareMeans:
    def test_equal_groups(self):
        # Stata power twomeans 0 .5, sd(1) knownsd n(64): 0.5160 (32 per group)
        p = Power().compare_means(0, 0.5, 1, 32).power
        assert p == pytest.approx(0.516005273976, abs=1e-10)

    def test_one_sided(self):
        # Stata ... n(64) onesided: 0.6388
        assert Power().compare_means(0, 0.5, 1, 32, two_sides=False).power == pytest.approx(
            0.6388, abs=5e-5
        )

    def test_unequal_sd_and_sizes(self):
        # Stata sd1(1) sd2(2) n1(40) n2(80): 0.4467
        p = Power().compare_means(0, 0.5, 1, (40, 80), sigma2=2).power
        assert p == pytest.approx(0.446690101475, abs=1e-10)

    def test_alloc_ratio(self):
        # Stata n1(30) nratio(2): 0.6088
        p = Power().compare_means(0, 0.5, 1, 30, alloc_ratio=2).power
        assert p == pytest.approx(0.6088, abs=5e-5)

    def test_equal_n_matches_one_sample_of_half_the_size(self):
        assert Power().compare_means(0, 0.5, 1, 64).power == pytest.approx(
            Power().test_mean(0, 0.5, 1, 32).power, abs=1e-14
        )

    def test_full_overlap_is_a_paired_test(self):
        # Stata power pairedmeans 0 .5, sd1(1) sd2(1) corr(.5) knownsd n(40): 0.8854; onesided 0.9354
        two = Power().compare_means(0, 0.5, 1, 40, rho=0.5, overlap=1).power
        one = Power().compare_means(0, 0.5, 1, 40, rho=0.5, overlap=1, two_sides=False).power
        assert two == pytest.approx(0.885379140762, abs=1e-10)
        assert one == pytest.approx(0.935420170048, abs=1e-10)

    def test_partial_overlap(self):
        # Var(d) = 1/100 + 1/100 - 2 * .5 * (.5 * 100) / 100^2 = .015
        p = Power().compare_means(0, 0.3, 1, 100, rho=0.5, overlap=0.5).power
        assert p == pytest.approx(0.687770420076, abs=1e-10)

    def test_partial_overlap_unequal_sizes(self):
        # m = .6 * min(100, 200); Var = 1/100 + 4/200 - 2 * .4 * 60 * sqrt(4) / (100 * 200)
        p = Power().compare_means(0, 0.4, 1, (100, 200), sigma2=2, rho=0.4, overlap=0.6).power
        assert p == pytest.approx(0.712195521609, abs=1e-10)

    def test_non_inferiority(self):
        # one-sided, delta = -.1, eps = 0, Var = 2/200: pnorm(.1/sqrt(.01) - qnorm(.95))
        p = Power().compare_means(0, 0, 1, 200, two_sides=False, delta=-0.1).power
        assert p == pytest.approx(0.259511022841, abs=1e-10)

    def test_equivalence(self):
        p = Power().compare_means(0, 0.05, 1, 300, delta=0.3).power
        assert p == pytest.approx(0.917635882464, abs=1e-10)

    def test_effect_is_group_2_minus_group_1(self):
        assert Power().compare_means(1, 0.4, 1, 50).effect == pytest.approx(-0.6)

    def test_rows_hold_both_groups(self):
        r = Power().compare_means(0, 0.5, 1, 30, alloc_ratio=2, deff=(1, 2)).rows[0]
        assert r.n == (30, 60) and r.n_eff == (30, 30)

    def test_n_pair_from_the_sizing_result(self):
        ss = SampleSize().compare_means(0, 0.5, 1, deff=1.5, resp_rate=0.8)
        pw = Power().compare_means(0, 0.5, 1, ss.n, deff=1.5, resp_rate=0.8)
        assert pw.n == ss.n and pw.power >= 0.8

    def test_census(self):
        assert Power().compare_means(0, 0.1, 1, 100, pop_size=100).power == pytest.approx(1.0)

    def test_census_of_one_group(self):
        # group 1 is exhausted: Var = sigma^2 / n2 only
        p = Power().compare_means(0, 0.5, 1, (50, 40), pop_size=(50, None)).power
        assert p == pytest.approx(0.885379140762, abs=1e-10)  # R, both tails at n_eff = 40

    def test_stratified_with_pairs_in_a_mapping(self):
        out = Power().compare_means(0, {"a": 0.5, "b": 0.3}, 1, {"a": (32, 32), "b": (100, 100)})
        assert out.power["a"] == pytest.approx(0.516005273976, abs=1e-10)
        assert out.power["b"] == Power().compare_means(0, 0.3, 1, 100).power
        assert out.n == {"a": (32, 32), "b": (100, 100)}

    def test_scalar_deff_is_both_groups(self):
        assert Power().compare_means(0, 0.5, 1, 64, deff=2).power == pytest.approx(
            Power().compare_means(0, 0.5, 1, 32).power, abs=1e-14
        )


class TestCompareProps:
    @pytest.mark.parametrize(
        "var_mode, power",
        [
            ("alt-props", 0.800413455043),
            ("pooled-prop", 0.796421309226),
            ("max-var", 0.7924430173),
        ],
    )
    def test_two_sided(self, var_mode, power):
        # R closed form at 385 per group; the sizing at the default alt-props gives n = 385
        p = Power().compare_props(0.5, 0.6, 385, var_mode=var_mode).power
        assert p == pytest.approx(power, abs=1e-10)

    def test_one_sided(self):
        assert Power().compare_props(0.5, 0.6, 385, two_sides=False).power == pytest.approx(
            0.876609994701, abs=1e-10
        )

    def test_pooled_variance_follows_the_group_sizes(self):
        p = Power().compare_props(0.5, 0.6, (300, 600), var_mode="pooled-prop").power
        assert p == pytest.approx(0.814325075312, abs=1e-10)

    def test_deff_fpc_nonresponse_per_group(self):
        # R: n_eff1 = 300*.9*(2000-1)/(2000-270)/1.5 = 207.9884, n_eff2 = 400*.8/2 = 160
        pw = Power().compare_props(
            0.5, 0.6, (300, 400), deff=(1.5, 2), resp_rate=(0.9, 0.8), pop_size=(2000, None)
        )
        assert pw.rows[0].n_eff == pytest.approx((207.988439306, 160.0), abs=1e-9)
        assert pw.power == pytest.approx(0.485624086114, abs=1e-10)

    def test_no_effect_gives_alpha(self):
        assert Power().compare_props(0.4, 0.4, 200).power == pytest.approx(0.05, abs=1e-12)

    def test_effect_sign_does_not_matter_two_sided_alt_props(self):
        assert Power().compare_props(0.6, 0.5, 385).power == pytest.approx(
            Power().compare_props(0.5, 0.6, 385).power, abs=1e-14
        )

    def test_stratified(self):
        out = Power().compare_props({"a": 0.5, "b": 0.2}, {"a": 0.6, "b": 0.4}, 385)
        assert out.power["a"] == pytest.approx(0.800413455043, abs=1e-10)
        assert out.power["b"] > 0.99

    def test_target_and_goal(self):
        t = Power().compare_props(0.5, 0.6, 385, var_mode="max-var").target
        assert (t.goal, t.solve, t.args["var_mode"], t.args["n"]) == (
            "compare_props",
            "power",
            "max-var",
            385,
        )

    def test_to_polars_has_both_groups(self):
        df = Power().compare_props(0.5, 0.6, (300, 600)).to_polars()
        assert df.columns == ["n1", "n2", "n_eff1", "n_eff2", "effect", "power"]
        assert df.row(0)[:2] == (300.0, 600.0)

    def test_print(self):
        out = str(Power().compare_props({"a": 0.5, "b": 0.2}, {"a": 0.6, "b": 0.4}, 385))
        assert "n_eff2" in out and "stratum" in out


class TestErrors:
    def test_pair_and_ratio_disagree(self):
        with pytest.raises(UsageError) as e:
            Power().compare_means(0, 0.5, 1, (40, 80), alloc_ratio=2)
        assert _code(e) == Code.PARAM_GIVEN_TWICE

    def test_pair_with_default_ratio_is_fine(self):
        assert Power().compare_means(0, 0.5, 1, (40, 80), alloc_ratio=1.0).power > 0

    @pytest.mark.parametrize("n", [(1, 2, 3), (40,), (0, 40), (40, -1)])
    def test_bad_pairs(self, n):
        with pytest.raises(SelectionError):
            Power().compare_means(0, 0.5, 1, n)

    def test_n_must_be_positive(self):
        with pytest.raises(SelectionError) as e:
            Power().compare_means(0, 0.5, 1, 0)
        assert _code(e) == Code.INVALID_RANGE

    def test_alloc_ratio_must_be_positive(self):
        with pytest.raises(SelectionError) as e:
            Power().compare_means(0, 0.5, 1, 40, alloc_ratio=0)
        assert _code(e) == Code.INVALID_RANGE and "alloc_ratio" in str(e.value)

    def test_group_larger_than_its_population(self):
        with pytest.raises(SelectionError) as e:
            Power().compare_props(0.5, 0.6, (50, 90), pop_size=(100, 80))
        assert _code(e) == Code.POWER_N_EXCEEDS_POPULATION

    def test_no_variance_left(self):
        with pytest.raises(SelectionError) as e:
            Power().compare_means(0, 0.5, 1, 40, rho=1, overlap=1)
        assert _code(e) == Code.SIZE_NO_VARIANCE and "Power.compare_means" in str(e.value)

    @pytest.mark.parametrize(
        "kw", [{"rho": 1.5}, {"rho": -1.1}, {"overlap": 1.2}, {"overlap": -0.1}]
    )
    def test_rho_and_overlap_ranges(self, kw):
        with pytest.raises(SelectionError) as e:
            Power().compare_means(0, 0.5, 1, 40, **kw)
        assert _code(e) == Code.INVALID_RANGE

    def test_var_mode_choice_names_the_method(self):
        with pytest.raises(SelectionError) as e:
            Power().compare_props(0.5, 0.6, 100, var_mode="pooled")
        assert _code(e) == Code.INVALID_CHOICE and "Power.compare_props" in str(e.value)

    def test_method_choice(self):
        with pytest.raises(SelectionError) as e:
            Power().compare_props(0.5, 0.6, 100, method="score")
        assert _code(e) == Code.INVALID_CHOICE

    @pytest.mark.parametrize("p", [0, 1, 1.5])
    def test_proportions_stay_inside_the_unit_interval(self, p):
        with pytest.raises(SelectionError) as e:
            Power().compare_props(p, 0.5, 100)
        assert _code(e) == Code.INVALID_RANGE

    def test_sigma2_must_be_positive(self):
        with pytest.raises(SelectionError) as e:
            Power().compare_means(0, 0.5, 1, 40, sigma2=0)
        assert "sigma2" in str(e.value)

    def test_deff_pair_of_three(self):
        with pytest.raises(SelectionError) as e:
            Power().compare_means(0, 0.5, 1, 40, deff=(1, 2, 3))
        assert _code(e) == Code.INVALID_TYPE

    def test_mapping_keys_must_match(self):
        with pytest.raises(UsageError) as e:
            Power().compare_means(0, {"a": 0.5}, {"b": 1}, 40)
        assert _code(e) == Code.MAPPING_KEYS_MISMATCH


GRID = [
    {},
    {"two_sides": False},
    {"alpha": 0.01},
    {"deff": (1.7, 1.2), "resp_rate": (0.8, 0.9)},
    {"deff": 2.0, "pop_size": 3000, "resp_rate": 0.8},
    {"alloc_ratio": 2.0},
    {"rho": 0.4, "overlap": 0.5},
]


class TestRoundTrip:
    @pytest.mark.parametrize("kw", GRID)
    def test_means(self, kw):
        for power in (0.8, 0.9):
            ss = SampleSize().compare_means(0, 0.3, 1, power=power, **kw)
            got = Power().compare_means(0, 0.3, 1, ss.n, **{**kw, "alloc_ratio": 1.0}).power
            assert got >= power

    @pytest.mark.parametrize("var_mode", ["alt-props", "pooled-prop", "max-var"])
    @pytest.mark.parametrize("kw", GRID)
    def test_props(self, kw, var_mode):
        ss = SampleSize().compare_props(0.3, 0.42, var_mode=var_mode, **kw)
        got = (
            Power()
            .compare_props(0.3, 0.42, ss.n, var_mode=var_mode, **{**kw, "alloc_ratio": 1.0})
            .power
        )
        assert got >= 0.8
