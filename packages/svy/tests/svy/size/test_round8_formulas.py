"""
Round 8 size & power formula fixes (findings SZ1-SZ10).

Reference values are hand-computed from the Chow-Shao-Wang / Cochran
formulas with z-quantiles from scipy (za2 = 1.959964, za1 = 1.644854,
zb80 = 0.841621); each test states its derivation.
"""

import numpy as np
import pytest

from svy import SampleSize
from svy.core.enumerations import MeanVarMode
from svy.engine.size_and_power.power import calculate_power
from svy.engine.size_and_power.size import (
    _wald_sample_size_one_mean_scalar,
    _wald_sample_size_two_means_scalar,
)
from svy.errors import MethodError


# =============================================================================
# SZ1 — compare_means is implemented
# =============================================================================


class TestCompareMeans:
    def test_equal_var(self):
        """n/group = (za2+zb)^2 * 2*sigma^2 / eps^2 = 7.848886*50/4 -> 99."""
        ss = SampleSize().compare_means(mu1=10.0, mu2=12.0, sigma1=5.0)
        assert ss.size.n0 == (99, 99)

    def test_unequal_var(self):
        """n1 = (za2+zb)^2 (s1^2 + s2^2/k) / eps^2 = 7.848886*(16+36)/4 -> 103."""
        ss = SampleSize().compare_means(mu1=10.0, mu2=12.0, sigma1=4.0, sigma2=6.0)
        assert ss.size.n0 == (103, 103)

    def test_non_inferiority(self):
        """CSW NI: eps=-0.2, delta=-0.5 -> denom=0.3.
        n1 = (1.959964+0.841621)^2 * 8 / 0.09 -> 698."""
        ss = SampleSize().compare_means(
            mu1=10.0,
            mu2=9.8,
            sigma1=2.0,
            delta=-0.5,
            two_sides=False,
            alpha=0.025,
            power=0.80,
        )
        assert ss.size.n0 == (698, 698)

    def test_alloc_ratio(self):
        ss = SampleSize().compare_means(mu1=10.0, mu2=12.0, sigma1=5.0, alloc_ratio=2.0)
        n1, n2 = ss.size.n0
        assert n2 == 2 * n1

    def test_stratified(self):
        ss = SampleSize().compare_means(
            mu1={"r1": 10.0, "r2": 20.0},
            mu2={"r1": 12.0, "r2": 24.0},
            sigma1={"r1": 5.0, "r2": 10.0},
        )
        sizes = {s.stratum: s for s in ss.size}
        assert sizes["r1"].n0 == (99, 99)
        # r2 has doubled eps and sigma -> identical n
        assert sizes["r2"].n0 == (99, 99)

    def test_requires_sigma(self):
        with pytest.raises(TypeError):
            SampleSize().compare_means(mu1=10.0, mu2=12.0)

    def test_invalid_method_raises(self):
        with pytest.raises(MethodError):
            SampleSize().compare_means(mu1=10.0, mu2=12.0, sigma1=5.0, method="fleiss")


# =============================================================================
# SZ2 — one-mean power-size denominator
# =============================================================================


def test_one_mean_power_size_two_sided():
    """Two-sided delta=0 superiority: n = (za2+zb)^2 sigma^2 / eps^2 = 32
    (the old clamp gave ~7.85e24)."""
    n = _wald_sample_size_one_mean_scalar(
        two_sides=True, epsilon=0.5, delta=0.0, sigma=1.0, alpha=0.05, power=0.80
    )
    assert n == 32


def test_one_mean_power_size_negative_epsilon():
    """Signed epsilon: same magnitude -> same n."""
    n = _wald_sample_size_one_mean_scalar(
        two_sides=True, epsilon=-0.5, delta=0.0, sigma=1.0, alpha=0.05, power=0.80
    )
    assert n == 32


# =============================================================================
# SZ3 — one-sided power direction follows sign(delta)
# =============================================================================


class TestOneSidedPowerDirection:
    def test_scalar_matches_array(self):
        scalar = calculate_power(False, -1.0, 1.0, 25, 0.05)
        arr = calculate_power(
            False, np.array([-1.0]), np.array([1.0]), np.array([25.0]), 0.05
        )
        assert scalar == pytest.approx(float(arr[0]), rel=1e-12)
        # Phi(-zc + 5) = 0.99960 (the old hardcoded "greater" gave ~1.5e-11)
        assert scalar == pytest.approx(0.9996034, abs=1e-6)

    def test_scalar_sign_symmetry(self):
        up = calculate_power(False, 1.0, 1.0, 25, 0.05)
        down = calculate_power(False, -1.0, 1.0, 25, 0.05)
        assert up == pytest.approx(down, rel=1e-12)

    def test_map_matches_scalar(self):
        out = calculate_power(
            False, {"a": -1.0, "b": 1.0}, {"a": 1.0, "b": 1.0}, {"a": 25, "b": 25}, 0.05
        )
        assert out["a"] == pytest.approx(out["b"], rel=1e-12)
        assert out["a"] == pytest.approx(0.9996034, abs=1e-6)


# =============================================================================
# SZ4 — non-inferiority uses signed epsilon (denominator eps - delta)
# =============================================================================


class TestNonInferioritySigned:
    def test_two_props_ni(self):
        """CSW: p1=.60, p2=.58 (eps=-.02), delta=-.05 -> denom=.03.
        var = .6*.4 + .58*.42 = .4836; n1 = .4836*((za2+zb)/.03)^2 -> 4218
        (the old |eps| denominator gave 775/group: ~22% achieved power)."""
        ss = SampleSize().compare_props(
            p1=0.60,
            p2=0.58,
            delta=-0.05,
            two_sides=False,
            alpha=0.025,
            power=0.80,
        )
        assert ss.size.n0 == (4218, 4218)

    def test_superiority_unchanged(self):
        """delta=0: signed eps changes nothing (denominator squared)."""
        ss = SampleSize().compare_props(p1=0.3, p2=0.5, two_sides=False)
        assert ss.size.n0 == (72, 72)

    def test_equivalence_branch_keeps_delta_minus_abs_eps(self):
        """Two-sided delta!=0 equivalence: denom = delta - |eps| = 0.4.
        n1 = (za1 + z_{(1+b)/2})^2 * 2 / 0.16 -> 108."""
        n1, n2 = _wald_sample_size_two_means_scalar(
            two_sides=True,
            epsilon=0.1,
            delta=0.5,
            sigma_1=1.0,
            kappa=1.0,
            alpha=0.05,
            power=0.80,
        )
        assert (n1, n2) == (108, 108)

    def test_equivalence_infeasible_raises(self):
        """|eps| >= delta cannot reach equivalence power."""
        with pytest.raises(MethodError):
            _wald_sample_size_two_means_scalar(
                two_sides=True,
                epsilon=0.6,
                delta=0.5,
                sigma_1=1.0,
                kappa=1.0,
                alpha=0.05,
                power=0.80,
            )


# =============================================================================
# SZ5 — p/moe validation
# =============================================================================


class TestParamValidation:
    @pytest.mark.parametrize("p", [-0.2, 0.0, 1.0, 1.5])
    def test_estimate_prop_invalid_p_raises(self, p):
        with pytest.raises(MethodError):
            SampleSize().estimate_prop(p=p, moe=0.05)

    @pytest.mark.parametrize("moe", [-0.1, 0.0, 5.0])
    def test_estimate_prop_invalid_moe_raises(self, moe):
        with pytest.raises(MethodError):
            SampleSize().estimate_prop(p=0.5, moe=moe)

    def test_compare_props_invalid_p_raises(self):
        """The old code silently clipped p1=1.5 to ~1."""
        with pytest.raises(MethodError):
            SampleSize().compare_props(p1=1.5, p2=0.5)


# =============================================================================
# SZ7 — pooled-proportion weights under unequal allocation
# =============================================================================


def test_pooled_prop_unequal_allocation():
    """kappa = n2/n1 = 3: pooled p = (p1 + 3*p2)/4 = 0.525.
    var_factor = v + v/3 = 0.3325; n1 = 0.3325*((za2+zb)/0.1)^2 -> 261
    (the swapped weights gave 256)."""
    ss = SampleSize().compare_props(
        p1=0.6, p2=0.5, alloc_ratio=3.0, var_mode="pooled-prop"
    )
    assert ss.size.n0 == (261, 783)


# =============================================================================
# SZ8 — clear errors instead of cryptic crashes
# =============================================================================


class TestCrashGuards:
    def test_power_out_of_range_raises(self):
        """power=80 used to crash with 'cannot convert float NaN to integer'."""
        with pytest.raises(MethodError):
            SampleSize().compare_props(p1=0.3, p2=0.5, power=80)

    def test_delta_equals_epsilon_raises(self):
        """delta == eps used to crash with OverflowError."""
        with pytest.raises(MethodError):
            SampleSize().compare_props(
                p1=0.6, p2=0.5, delta=-0.10, two_sides=False
            )


# =============================================================================
# SZ9 — no silent parameter sanitization
# =============================================================================


class TestNoSilentSanitization:
    def test_resp_rate_percent_raises(self):
        """resp_rate=90 (percent instead of fraction) used to be ignored."""
        with pytest.raises(MethodError):
            SampleSize().estimate_prop(p=0.5, moe=0.05, resp_rate=90)

    def test_negative_deff_raises(self):
        with pytest.raises(MethodError):
            SampleSize().estimate_prop(p=0.5, moe=0.05, deff=-2.0)

    def test_negative_sigma_raises(self):
        """sigma=-5 used to be clamped to ~0, giving n0=1."""
        with pytest.raises(MethodError):
            SampleSize().estimate_mean(sigma=-5, moe=1)

    def test_comparison_resp_rate_raises(self):
        with pytest.raises(MethodError):
            SampleSize().compare_props(p1=0.3, p2=0.5, resp_rate=85)


# =============================================================================
# SZ10 — optimal allocation ratio orientation (kappa = n2/n1)
# =============================================================================


def test_optimal_kappa_two_means_unequal_var():
    """With kappa unset, the total-n-minimizing ratio is sigma2/sigma1:
    sigma1=2, sigma2=6 -> kappa=3, so n2 = 3*n1."""
    n1, n2 = _wald_sample_size_two_means_scalar(
        two_sides=True,
        epsilon=1.0,
        delta=0.0,
        sigma_1=2.0,
        sigma_2=6.0,
        kappa=None,
        alpha=0.05,
        power=0.80,
        var_mode=MeanVarMode.UNEQUAL_VAR,
    )
    assert n2 == 3 * n1
    # n1 = (za2+zb)^2 (4 + 36/3) / 1 = 7.848886*16 -> 126
    assert n1 == 126
