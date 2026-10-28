"""Weighting kernels give the same answer whatever the units of the weights and auxiliaries."""

import numpy as np
import pytest

import svy_rs as ps

# Pure arithmetic on a handful of values: rounding error only.
REL = 1e-12
# A 4x4 solve accumulated over 200 rows; the unit-diagonal system is well conditioned.
CAL_REL = 1e-9
SCALES = [1e-6, 1e-3, 1.0, 1e3, 1e6]


def _design(scale):
    rng = np.random.default_rng(20261028)
    n = 200
    w = rng.uniform(1, 3, n)
    x1 = rng.uniform(1, 2, n)
    x2 = rng.uniform(0, 1, n)
    # The intercept stays at 1 while the other columns follow `scale`; x1 + x2 is redundant.
    X = np.column_stack([np.ones(n), x1 * scale, x2 * scale, (x1 + x2) * scale])
    return w * scale, X


def _totals(w, X, consistent):
    t = (X[:, :3].T @ w) * 1.05
    redundant = t[1] + t[2]
    return np.append(t, redundant if consistent else redundant * 1.01)


def _g_reference(w, X, totals):
    """Linear calibration's g with numpy's minimum-norm least squares on the unit-diagonal system."""
    a = X.T @ (w[:, None] * X)
    d = 1 / np.sqrt(np.diag(a))
    y, *_ = np.linalg.lstsq(a * np.outer(d, d), (totals - X.T @ w) * d, rcond=None)
    return 1 + X @ (y * d)


@pytest.mark.parametrize(
    "consistent", [True, False], ids=["consistent", "inconsistent"]
)
@pytest.mark.parametrize("scale", SCALES)
def test_redundant_calibration_does_not_depend_on_units(scale, consistent):
    w, X = _design(scale)
    totals = _totals(w, X, consistent)
    g = ps.calibrate(w[:, None], X, totals)[:, 0] / w
    np.testing.assert_allclose(g, _g_reference(w, X, totals), rtol=CAL_REL)
    w1, X1 = _design(1.0)
    g1 = ps.calibrate(w1[:, None], X1, _totals(w1, X1, consistent))[:, 0] / w1
    np.testing.assert_allclose(g, g1, rtol=CAL_REL)


@pytest.mark.parametrize("scale", SCALES)
def test_consistent_redundant_controls_are_met(scale):
    w, X = _design(scale)
    totals = _totals(w, X, consistent=True)
    new_w = ps.calibrate(w[:, None], X, totals)[:, 0]
    np.testing.assert_allclose(X.T @ new_w, totals, rtol=CAL_REL)


def test_normalize_tiny_weights():
    out = ps.normalize(np.array([[1e-12], [3e-12]]), control=np.array([8e-12]))
    np.testing.assert_allclose(out[:, 0], [2e-12, 6e-12], rtol=REL)


def test_rake_tiny_weights():
    out = ps.rake(
        np.full((4, 1), 1e-12),
        [np.array([0, 0, 1, 1], dtype=np.int64)],
        [np.array([1e-12, 3e-12])],
    )
    np.testing.assert_allclose(
        out[:, 0], [0.5e-12, 0.5e-12, 1.5e-12, 1.5e-12], rtol=REL
    )


def test_nonresponse_tiny_weights():
    # Two respondents carry the nonrespondent's weight: factor (2 + 2) / 2.
    out = ps.adjust_nr(
        np.array([[1e-12], [1e-12], [2e-12]]),
        np.zeros(3, dtype=np.int64),
        np.array([0, 0, 1], dtype=np.int64),
        True,
    )
    np.testing.assert_allclose(out[:, 0], [2e-12, 2e-12, 0.0], rtol=REL, atol=0)
