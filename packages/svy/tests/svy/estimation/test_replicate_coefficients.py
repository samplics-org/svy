# tests/svy/estimation/test_replicate_coefficients.py
"""Per-replicate variance coefficients now come from the RepWeights variant.

They used to be re-derived inside the kernel from a method label plus a
fay_coef that crossed the FFI boundary for every method. These tests pin the
behaviour that move has to preserve, including the estimation-time Fay
override, which nothing covered before.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import svy

from svy.core.repwgts import BootstrapWgts, BrrWgts, JackknifeWgts, SdrWgts


@pytest.fixture
def brr_sample():
    rng = np.random.default_rng(3)
    data = pl.DataFrame(
        {
            "stratum": [1, 1, 2, 2, 3, 3, 4, 4],
            "psu": [1, 2, 3, 4, 5, 6, 7, 8],
            "wgt": [10.0] * 8,
            "y": rng.normal(100, 10, 8),
        }
    )
    s = svy.Sample(data=data, design=svy.Design(wgt="wgt", stratum="stratum", psu="psu"))
    return s.weighting.create_brr_wgts(fay_coef=0.5)


def _se(sample, **kwargs) -> float:
    return sample.estimation.mean("y", method="replication", **kwargs).to_dicts()[0]["se"]


def test_design_fay_coefficient_is_used_by_default(brr_sample):
    assert brr_sample._design.rep_wgts.fay_coef == 0.5
    assert _se(brr_sample) > 0


def test_estimation_time_fay_overrides_the_design(brr_sample):
    """BRR scale is 1/(B(1-f)^2), so the SE scales as 1/(1-f) exactly."""
    at_design = _se(brr_sample)  # f = 0.5
    at_override = _se(brr_sample, fay_coef=0.25)
    assert at_design / at_override == pytest.approx((1 - 0.25) / (1 - 0.5), rel=1e-12)


def test_fay_coef_zero_means_no_override_not_fay_zero(brr_sample):
    """0.0 is the sentinel for 'unset', so it must not reset the design's 0.5."""
    assert _se(brr_sample, fay_coef=0.0) == pytest.approx(_se(brr_sample), rel=1e-15)


def test_fay_override_is_ignored_for_methods_without_one(brr_sample):
    """A bootstrap has no Fay coefficient; passing one must not change anything."""
    data = brr_sample._data
    boot = svy.Sample(
        data=data,
        design=svy.Design(wgt="wgt", stratum="stratum", psu="psu"),
    ).weighting.create_bs_wgts(n_reps=16, rep_prefix="bs", rstate=5)
    assert _se(boot, fay_coef=0.4) == pytest.approx(_se(boot), rel=1e-15)


@pytest.mark.parametrize(
    "variant, expected",
    [
        (BootstrapWgts(prefix="w", n_reps=64), [1.0 / 64] * 64),
        (SdrWgts(prefix="w", n_reps=80), [4.0 / 80] * 80),
        (JackknifeWgts(prefix="w", n_reps=20), [19.0 / 20] * 20),
        (BrrWgts(prefix="w", n_reps=32), [1.0 / 32] * 32),
        (BrrWgts(prefix="w", n_reps=32, fay_coef=0.3), [1.0 / (32 * 0.7**2)] * 32),
    ],
)
def test_variant_coefficients_match_the_kernel_formulas(variant, expected):
    """Mirrors replicate_coefficients() in svy-rs/src/estimation/replication.rs."""
    assert variant.coefficients() == pytest.approx(expected, rel=1e-15)


def test_user_scale_takes_precedence_over_the_method_default():
    rs = tuple(np.linspace(0.5, 1.5, 10))
    assert BootstrapWgts(prefix="w", n_reps=10, scale=rs).coefficients() == pytest.approx(
        list(rs)
    )


def test_coefficients_follow_the_resolved_replicate_count(brr_sample):
    """The column count wins over a stale n_reps recorded on the design."""
    rw = brr_sample._design.rep_wgts
    assert len(rw.coefficients()) == rw.n_reps
