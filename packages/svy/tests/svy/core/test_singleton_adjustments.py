# tests/svy/core/test_singleton_adjustments.py
import polars as pl
import pytest
import svy_rs as ps

import svy

from svy.core.constants import SVY_ROW_INDEX
from svy.core.enumerations import PopParam, SingletonHandling
from svy.core.singleton import _VAR_EXCLUDE_COL
from svy.estimation.base import Estimation


@pytest.fixture
def adjustment_sample():
    """
    Sample with 50% singletons (2 of 4 strata).
    Strata: A(1), B(2), C(1), D(2).
    """
    rows = [
        (0, "A", "101", 10.0, 1.0),
        (1, "A", "101", 12.0, 1.0),
        (2, "B", "201", 20.0, 1.0),
        (3, "B", "202", 22.0, 1.0),
        (4, "C", "301", 30.0, 1.0),
        (5, "D", "401", 40.0, 1.0),
        (6, "D", "402", 42.0, 1.0),
    ]
    df = pl.DataFrame(
        rows,
        schema=[SVY_ROW_INDEX, "stratum", "cluster", "income", "weight"],
        orient="row",
    )
    design = svy.Design(stratum="stratum", psu="cluster", wgt="weight")
    return svy.Sample(data=df, design=design)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS FOR SCALE METHOD
# ══════════════════════════════════════════════════════════════════════════════


def test_scale_config_correctness(adjustment_sample):
    sample = adjustment_sample.singleton.scale()
    result = sample.singleton.last_result
    assert result.method == SingletonHandling.SCALE
    assert result.config.singleton_fraction == 0.5
    assert result.config.var_exclude_col == _VAR_EXCLUDE_COL


def test_scale_excludes_singletons_from_data(adjustment_sample):
    sample = adjustment_sample.singleton.scale()
    df = sample._data
    assert df.filter(pl.col("stratum") == "A").get_column(_VAR_EXCLUDE_COL).all()
    assert not df.filter(pl.col("stratum") == "B").get_column(_VAR_EXCLUDE_COL).any()


def test_scale_inflation_logic_explicit_params(adjustment_sample):
    """Test both Total (inflation) and Mean (deflation) logic."""
    sample = adjustment_sample.singleton.scale()
    est = sample.estimation
    # Ensure cache is populated
    _ = est._get_polars_design_info()

    mock_result = pl.DataFrame({"est": [50.0], "var": [100.0], "se": [10.0]})

    # 1. TOTAL: Expect Inflation (Factor = 1/(1-0.5) = 2.0)
    res_total = est._adjust_variance_for_singletons(mock_result, param=PopParam.TOTAL)
    assert res_total["var"][0] == pytest.approx(200.0)  # 100 * 2.0

    # 2. MEAN: Expect Deflation (Factor = 1-0.5 = 0.5)
    res_mean = est._adjust_variance_for_singletons(mock_result, param=PopParam.MEAN)
    assert res_mean["var"][0] == pytest.approx(50.0)  # 100 * 0.5


def test_scale_estimation_flow_integration(adjustment_sample, monkeypatch):
    """
    Verify integration for MEAN calculation.
    """
    sample = adjustment_sample.singleton.scale()

    def mock_taylor_mean(*args, **kwargs):
        return pl.DataFrame(
            {
                "y": ["income"],
                "est": [1.0],
                "se": [1.0],
                "var": [1.0],
                "df": [10],
                "n": [100],
                "deff": [1.0],
            }
        )

    monkeypatch.setattr(ps, "taylor_mean", mock_taylor_mean)

    result = sample.estimation.mean("income")

    # Logic for Mean: Factor = (1 - 0.5) = 0.5
    # Original Mock Variance = 1.0
    # Expected Adjusted Variance = 0.5
    est_var = result.estimates[0].se ** 2
    assert est_var == pytest.approx(0.5)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS FOR CENTER METHOD
# ══════════════════════════════════════════════════════════════════════════════


def test_center_config_correctness(adjustment_sample):
    sample = adjustment_sample.singleton.center()
    result = sample.singleton.last_result
    assert result.method == SingletonHandling.CENTER
    assert result.config.singleton_fraction is None


def test_center_does_not_exclude_rows(adjustment_sample):
    sample = adjustment_sample.singleton.center()
    df = sample._data
    assert not df.get_column(_VAR_EXCLUDE_COL).any()


def test_center_arg_passing(adjustment_sample, monkeypatch):
    sample = adjustment_sample.singleton.center()
    captured_kwargs = {}

    def mock_taylor_mean(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return pl.DataFrame(
            {
                "y": ["income"],
                "est": [1.0],
                "se": [1.0],
                "var": [1.0],
                "df": [10],
                "n": [100],
                "deff": [1.0],
            }
        )

    monkeypatch.setattr(ps, "taylor_mean", mock_taylor_mean)
    sample.estimation.mean("income")
    assert captured_kwargs.get("singleton_method") == "center"


def test_center_idempotent_if_not_configured(adjustment_sample, monkeypatch):
    sample = adjustment_sample
    captured_kwargs = {}

    def mock_taylor_mean(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return pl.DataFrame(
            {
                "y": ["income"],
                "est": [1.0],
                "se": [1.0],
                "var": [1.0],
                "df": [10],
                "n": [100],
                "deff": [1.0],
            }
        )

    monkeypatch.setattr(ps, "taylor_mean", mock_taylor_mean)
    sample.estimation.mean("income")
    assert captured_kwargs.get("singleton_method") is None


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICATION TEST (Regression against R)
# ══════════════════════════════════════════════════════════════════════════════


def test_verify_singleton_methods():
    csv_path = "test_data/singleton_test_20012026.csv"
    try:
        data = pl.read_csv(csv_path)
    except FileNotFoundError:
        try:
            data = pl.read_csv("tests/test_data/singleton_test_20012026.csv")
        except FileNotFoundError:
            pytest.skip("Verification CSV not found. Run generation script first.")

    design = svy.Design(stratum="stratum", psu="psu", wgt="weight")
    sample = svy.Sample(data, design)

    R_EXPECTED = {
        "MEAN": 24.86417079,
        "DF": 2.00000000,
        "SCALE_SE": 0.07411897,
        "CENTER_SE": 3.93156714,
    }

    # SCALE
    s_scale = sample.singleton.scale()
    res_scale = s_scale.estimation.mean("y")
    est_scale = res_scale.estimates[0]

    assert est_scale.est == pytest.approx(R_EXPECTED["MEAN"], abs=1e-6)
    assert est_scale.se == pytest.approx(R_EXPECTED["SCALE_SE"], abs=1e-6)
    assert res_scale.degrees_freedom == pytest.approx(R_EXPECTED["DF"], abs=1e-6)

    # CENTER
    s_center = sample.singleton.center()
    res_center = s_center.estimation.mean("y")
    est_center = res_center.estimates[0]

    assert est_center.est == pytest.approx(R_EXPECTED["MEAN"], abs=1e-6)
    assert est_center.se == pytest.approx(R_EXPECTED["CENTER_SE"], abs=1e-6)
    assert res_center.degrees_freedom == pytest.approx(R_EXPECTED["DF"], abs=1e-6)
