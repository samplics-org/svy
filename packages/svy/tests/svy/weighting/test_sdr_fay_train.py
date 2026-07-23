# tests/svy/weighting/test_sdr_fay_train.py
"""
Fay-Train correctness of SDR weight creation (round 8 validation phase).

For a single stratum in systematic order with unit weights, replicate
factors f = 1 + 2^{-3/2}(h_a - h_b) over cyclic Hadamard rows 1..R-1 make
the SDR variance of a total (coefficient 4/R, all R replicates) equal the
successive-difference form exactly:

    v(total) = 1/2 * ( z_1^2 + z_n^2 + sum_k (z_{k+1} - z_k)^2 )

for z_i = w_i * y_i, provided n <= R-1 (no Hadamard row reuse). This is
the algebraic identity from the Hadamard orthogonality relations
(Fay & Train 1995; Ash 2014).
"""

import numpy as np
import polars as pl
import pytest

from svy.core.sample import Design, Sample


@pytest.fixture
def sdr_sample():
    y = [3.0, 7.0, 2.0, 9.0, 4.0]
    df = pl.DataFrame(
        {
            "y": y,
            "w": [1.0] * len(y),
            "sort_order": list(range(len(y))),
        }
    )
    return Sample(df, Design(wgt="w")), np.array(y)


def test_sdr_total_matches_successive_difference_identity(sdr_sample):
    sample, z = sdr_sample
    out = sample.weighting.create_sdr_wgts(n_reps=8, order_col="sort_order")

    est = out.estimation.total(y="y", method="replication")
    v_hat = est.estimates[0].se ** 2

    diffs = np.diff(z)
    v_expected = 0.5 * (z[0] ** 2 + z[-1] ** 2 + np.sum(diffs**2))
    np.testing.assert_allclose(v_hat, v_expected, rtol=1e-10)


def test_sdr_replicate_means_preserve_totals(sdr_sample):
    """Mean replicate weight over all R replicates equals the full-sample
    weight, so the replicate totals average to the full-sample total."""
    sample, z = sdr_sample
    out = sample.weighting.create_sdr_wgts(n_reps=8, order_col="sort_order")
    rep_cols = out.design.rep_wgts.columns
    reps = out.data.select(rep_cols).to_numpy()
    np.testing.assert_allclose(reps.mean(axis=1), np.ones(len(z)), rtol=1e-12)

    rep_totals = (reps * z[:, None]).sum(axis=0)
    np.testing.assert_allclose(rep_totals.mean(), z.sum(), rtol=1e-12)


def test_sdr_factors_in_fay_train_set(sdr_sample):
    sample, _ = sdr_sample
    out = sample.weighting.create_sdr_wgts(n_reps=8, order_col="sort_order")
    reps = out.data.select(out.design.rep_wgts.columns).to_numpy()
    allowed = np.array([1.0, 1.0 + 2**-0.5, 1.0 - 2**-0.5])
    for f in np.unique(np.round(reps, 12)):
        assert np.min(np.abs(allowed - f)) < 1e-10
