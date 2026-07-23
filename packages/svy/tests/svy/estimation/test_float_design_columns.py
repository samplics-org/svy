# tests/svy/estimation/test_float_design_columns.py
"""
Float-typed design columns (regression).

Numeric stratum/PSU codes read from CSVs frequently arrive as Float64
(e.g. MEPS VARSTR/VARPSU). The factorized-design cache used to cast the
column straight to Categorical, which polars forbids for floats:
`conversion from f64 to cat failed`. Float columns now go through Utf8.
"""

import numpy as np
import polars as pl
import pytest

from svy.core.sample import Design, Sample


@pytest.fixture
def float_design_sample():
    rng = np.random.default_rng(3)
    n = 80
    return Sample(
        pl.DataFrame(
            {
                "VARSTR": np.repeat([2019.0, 2020.0, 2033.0, 2055.0], n // 4),
                "VARPSU": np.tile(np.repeat([1.0, 2.0], n // 8), 4),
                "w": rng.uniform(1.0, 3.0, n),
                "y": rng.normal(100.0, 10.0, n),
            }
        ),
        Design(stratum="VARSTR", psu="VARPSU", wgt="w"),
    )


def test_total_with_float_design_columns(float_design_sample):
    est = float_design_sample.estimation.total(y="y")
    res = est.estimates[0]
    assert np.isfinite(res.est) and np.isfinite(res.se) and res.se > 0


def test_mean_with_float_design_columns(float_design_sample):
    est = float_design_sample.estimation.mean(y="y")
    res = est.estimates[0]
    assert np.isfinite(res.est) and np.isfinite(res.se) and res.se > 0


def test_float_matches_int_design_columns(float_design_sample):
    """Float-coded and int-coded design columns give identical results."""
    df_int = float_design_sample.data.with_columns(
        pl.col("VARSTR").cast(pl.Int64), pl.col("VARPSU").cast(pl.Int64)
    )
    s_int = Sample(df_int, Design(stratum="VARSTR", psu="VARPSU", wgt="w"))
    a = float_design_sample.estimation.mean(y="y").estimates[0]
    b = s_int.estimation.mean(y="y").estimates[0]
    np.testing.assert_allclose([a.est, a.se], [b.est, b.se], rtol=1e-12)
