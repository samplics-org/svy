import math

import numpy as np
import polars as pl
import pytest

import svy

from svy.core.containers import ChiSquare, FDist, TDist


@pytest.fixture
def sample():
    rng = np.random.default_rng(1)
    n = 400
    df = pl.DataFrame(
        {
            "str": rng.integers(1, 5, n),
            "psu": np.arange(n) // 10,
            "w": rng.uniform(1, 3, n),
            "a": rng.integers(1, 4, n),
            "b": rng.integers(1, 3, n),
        }
    )
    return svy.Sample(df, svy.Design(stratum="str", psu="psu", wgt="w"))


class TestStatisticStrings:
    def test_f(self):
        f = FDist(df_num=3.29895777, df_den=49.4844, value=11.6691, p_value=0.0203)
        assert str(f) == "F(3.30, 49.48) = 11.67, p = 0.020"

    def test_chi_square_integer_df(self):
        assert str(ChiSquare(df=3, value=35.214, p_value=2e-7)) == "chi2(3) = 35.21, p < 0.001"

    def test_t(self):
        assert str(TDist(df=29.0, value=-2.1, p_value=0.0444)) == "t(29) = -2.10, p = 0.044"

    def test_nan(self):
        s = str(ChiSquare(df=2, value=math.nan, p_value=math.nan))
        assert s == "chi2(2) = nan, p = nan"

    def test_repr_unchanged(self):
        assert repr(ChiSquare(df=3, value=1.0, p_value=0.5)).startswith("ChiSquare(df=3")


class TestTablePrint:
    def test_crosstab_ends_with_both_tests(self, sample):
        t = sample.categorical.tabulate("a", "b")
        out = t.__plain_str__().splitlines()
        assert out[-2] == f"Rao-Scott {t.stats.f}"
        assert out[-1] == f"Rao-Scott {t.stats.chisq}"

    def test_rich_print_shows_the_tests(self, sample):
        t = sample.categorical.tabulate("a", "b")
        assert f"Rao-Scott {t.stats.f}" in str(t)
        assert f"Rao-Scott {t.stats.chisq}" in str(t)

    def test_one_way_has_no_test(self, sample):
        t = sample.categorical.tabulate("a")
        assert "Rao-Scott" not in t.__plain_str__()
        assert "Rao-Scott" not in str(t)
