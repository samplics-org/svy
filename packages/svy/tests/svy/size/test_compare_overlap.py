# tests/svy/size/test_compare_overlap.py
"""
compare_props / compare_means with units in both rounds (rho=, overlap=).

Full overlap is a paired design; Stata 19.5 `power pairedmeans ..., knownsd`
gives N (pairs) for the z test. Partial overlap is checked against Kish's
formula computed directly.
"""

import math

import pytest

from scipy import stats

from svy import SampleSize
from svy.errors._method import MethodError


def _z(alpha=0.05, power=0.8, two_sides=True):
    za = stats.norm.ppf(1 - alpha / 2 if two_sides else 1 - alpha)
    return (za + stats.norm.ppf(power)) ** 2


def _kish_n1(v1, v2, eps, *, rho, overlap, k=1.0, **z):
    var = v1 + v2 / k - 2 * rho * overlap * min(1.0, k) * math.sqrt(v1 * v2) / k
    return math.ceil(_z(**z) * var / eps**2)


@pytest.mark.parametrize(
    "kw, stata_n",
    [
        (dict(mu1=0, mu2=2, sigma1=10, rho=0.6), 157),
        (dict(mu1=0, mu2=2, sigma1=10, sigma2=14, rho=0.3), 416),
        (dict(mu1=0, mu2=3, sigma1=8, rho=0.0, two_sides=False), 88),
    ],
)
def test_full_overlap_is_stata_paired_means(kw, stata_n):
    assert SampleSize().compare_means(**kw, overlap=1).n == (stata_n, stata_n)


@pytest.mark.parametrize("rho, overlap", [(0.0, 0.7), (0.5, 0.0), (0.0, 0.0)])
def test_no_correlation_or_no_overlap_is_independent(rho, overlap):
    base = SampleSize().compare_props(0.3, 0.4)
    assert SampleSize().compare_props(0.3, 0.4, rho=rho, overlap=overlap).n == base.n
    base = SampleSize().compare_means(0, 2, 10)
    assert SampleSize().compare_means(0, 2, 10, rho=rho, overlap=overlap).n == base.n


def test_partial_overlap_props_is_kish():
    n = SampleSize().compare_props(0.3, 0.4, rho=0.5, overlap=0.5).n
    expected = _kish_n1(0.21, 0.24, 0.1, rho=0.5, overlap=0.5)
    assert n == (expected, expected)


def test_partial_overlap_with_unequal_allocation_is_kish():
    n1, n2 = SampleSize().compare_means(0, 2, 10, alloc_ratio=2, rho=0.4, overlap=0.8).n
    expected = _kish_n1(100, 100, 2, rho=0.4, overlap=0.8, k=2.0)
    assert (n1, n2) == (expected, math.ceil(2 * expected))


def test_positive_correlation_lowers_n_negative_raises_it():
    base = SampleSize().compare_means(0, 2, 10).n[0]
    assert SampleSize().compare_means(0, 2, 10, rho=0.5, overlap=1).n[0] < base
    assert SampleSize().compare_means(0, 2, 10, rho=-0.5, overlap=1).n[0] > base


def test_per_stratum_inputs():
    ss = SampleSize().compare_props(
        {"a": 0.3, "b": 0.2}, {"a": 0.4, "b": 0.3}, rho={"a": 0.5, "b": 0.0}, overlap=1
    )
    alone_a = SampleSize().compare_props(0.3, 0.4, rho=0.5, overlap=1).n
    alone_b = SampleSize().compare_props(0.2, 0.3).n
    assert ss.n == {"a": alone_a, "b": alone_b}


def test_deff_scales_the_overlap_adjusted_size():
    plain = SampleSize().compare_means(0, 2, 10, rho=0.6, overlap=1)
    with_deff = SampleSize().compare_means(0, 2, 10, rho=0.6, overlap=1, deff=2.0)
    assert with_deff.n[0] == pytest.approx(2 * plain.n[0], abs=2)


def test_inputs_are_recorded():
    ss = SampleSize().compare_props(0.3, 0.4, rho=0.5, overlap=0.25)
    assert (ss._target.rho, ss._target.overlap) == (0.5, 0.25)


@pytest.mark.parametrize(
    "kw, param", [(dict(rho=1.5), "rho"), (dict(rho=-2), "rho"), (dict(overlap=1.2), "overlap")]
)
def test_out_of_range_inputs_name_the_parameter(kw, param):
    with pytest.raises(MethodError) as exc:
        SampleSize().compare_means(0, 2, 10, **kw)
    assert exc.value.param == param


def test_no_variance_left_raises():
    with pytest.raises(MethodError) as exc:
        SampleSize().compare_means(0, 2, 10, rho=1, overlap=1)
    assert exc.value.code == "SIZE_NO_VARIANCE"
