import numpy as np
import pytest

from numpy.testing import assert_allclose

import svy.svy_rs


# --- Test Fixtures ---


@pytest.fixture
def sample_data():
    """Provides a consistent dataset for testing Taylor variance."""
    # (n=10, k=2) score matrix
    y_score = np.array(
        [
            [0.1, 0.2],
            [0.3, -0.1],
            [0.5, 0.4],
            [-0.2, 0.3],
            [0.0, 0.1],
            [-0.4, -0.2],
            [0.6, 0.1],
            [0.2, -0.3],
            [-0.1, 0.5],
            [0.3, 0.0],
        ],
        dtype=np.float64,
    )

    w = np.full(10, 2.0, dtype=np.float64)
    stratum = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2], dtype=np.int64)
    psu = np.array([1, 1, 2, 2, 3, 4, 4, 5, 5, 5], dtype=np.int64)

    return {"y_score": y_score, "w": w, "stratum": stratum, "psu": psu}


# --- Python Implementations for Ground Truth ---


def python_scores_total(y, w):
    """Reference implementation for scores_total."""
    return w[:, np.newaxis] * y


def python_variance_stratum_between(y_score_s, psu_s=None):
    """Reference implementation for _variance_stratum_between."""
    y = y_score_s
    n, k = y.shape
    if n <= 1:
        return np.zeros((k, k))

    if psu_s is not None:
        unique_psus, inv = np.unique(psu_s, return_inverse=True)
        m = len(unique_psus)
        if m <= 1:
            return np.zeros((k, k))

        T = np.zeros((m, k))
        np.add.at(T, inv, y)

        mean = T.mean(axis=0, keepdims=True)
        diff = T - mean
        return (m / (m - 1.0)) * (diff.T @ diff)
    else:
        mean = y.mean(axis=0, keepdims=True)
        diff = y - mean
        return (n / (n - 1.0)) * (diff.T @ diff)


def python_taylor_variance(y_score, stratum=None, psu=None):
    """Reference implementation for _taylor_variance."""
    if stratum is None:
        return python_variance_stratum_between(y_score, psu)

    total_cov = np.zeros((y_score.shape[1], y_score.shape[1]))
    unique_strata = np.unique(stratum)

    for s_val in unique_strata:
        mask = stratum == s_val
        y_s = y_score[mask]
        psu_s = psu[mask] if psu is not None else None
        total_cov += python_variance_stratum_between(y_s, psu_s)

    return total_cov


# --- Tests for Rust Functions ---


def test_scores_total_py(sample_data):
    """Verify that the Rust scores_total function matches the NumPy equivalent."""
    y, w = sample_data["y_score"], sample_data["w"]

    expected = python_scores_total(y, w)
    # Call the function from the imported `estimation` module
    actual = svy.svy_rs.taylor.scores_total_py(y=y, wgt=w)

    assert_allclose(actual, expected, rtol=1e-12)


def test_scores_total_py_raises_error_on_shape_mismatch():
    """Ensure ValueError is raised for incompatible shapes."""
    y = np.zeros((10, 2))
    w_wrong_shape = np.zeros(9)

    with pytest.raises(ValueError, match="y and w must have compatible first dimension"):
        svy.svy_rs.estimation.taylor.scores_total_py(y=y, wgt=w_wrong_shape)


@pytest.mark.parametrize(
    "design",
    [
        "unstratified_unclustered",
        "unstratified_clustered",
        "stratified_unclustered",
        "stratified_clustered",
    ],
    ids=[
        "No Stratum, No PSU",
        "No Stratum, With PSU",
        "With Stratum, No PSU",
        "With Stratum, With PSU",
    ],
)
def test_taylor_variance(sample_data, design):
    """
    Test taylor_variance against the Python reference implementation
    for all major survey design configurations.
    """
    y_score = sample_data["y_score"]
    w = sample_data["w"]
    stratum = sample_data["stratum"] if "stratified" in design else None
    psu = sample_data["psu"] if "clustered" in design else None

    expected_cov = python_taylor_variance(y_score, stratum=stratum, psu=psu)
    # Call the function from the imported `estimation` module
    actual_cov = svy.svy_rs.estimation.taylor.taylor_variance(
        y_score=y_score, wgt=w, stratum=stratum, psu=psu
    )

    assert_allclose(actual_cov, expected_cov, rtol=1e-12)


def test_taylor_variance_raises_error_on_shape_mismatch(sample_data):
    """Ensure ValueError is raised for mismatched input lengths."""
    y_score = sample_data["y_score"]
    w = sample_data["w"]
    stratum_wrong_shape = np.array([1, 2, 3])

    with pytest.raises(ValueError, match="stratum must have same length as y_score"):
        svy.svy_rs.estimation.taylor.taylor_variance(
            y_score=y_score, wgt=w, stratum=stratum_wrong_shape
        )
