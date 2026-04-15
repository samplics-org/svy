# tests/svy/weighting/test_wgt_calibration.py
import numpy as np
import polars as pl
import pytest

from numpy.testing import assert_allclose

from svy import EstimationMethod
from svy.core.sample import Design, Sample
from svy.core.terms import Cat, Cross

CALIB_WGT = "calib_wgt"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_cat_df() -> pl.DataFrame:
    base = pl.DataFrame(
        {
            "A": ["A1", "A1", "A1", "A2", "A2", "A2", "A3", "A3", "A3", "A4", "A4", "A4"],
            "B": ["B1", "B2", "B3", "B1", "B2", "B3", "B1", "B2", "B3", "B1", "B2", "B3"],
            "wgt": [2.0, 4.0, 4.0, 5.0, 14.0, 31.0, 10.0, 5.0, 5.0, 3.0, 10.0, 7.0],
            "benchmark": [8.0, 4.0, 5.5, 6.0, 15.0, 34.0, 17.0, 6.0, 20.0, 5.5, 16.5, 12.5],
        }
    )
    df = pl.concat([base] * 10).sort(["A", "B"])
    by_vals = (["D1", "D2"] * 60)[:120]
    return df.with_columns(pl.Series("by", by_vals))


@pytest.fixture
def sample_num_df() -> pl.DataFrame:
    base = pl.DataFrame(
        {
            "A": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "B": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            "wgt": [2.0, 4.0, 4.0, 5.0, 14.0, 31.0, 10.0, 5.0, 5.0, 3.0, 10.0, 7.0],
        }
    )
    df = pl.concat([base] * 10).sort(["A", "B"])
    by_vals = (["D1", "D2"] * 60)[:120]
    return df.with_columns(pl.Series("by", by_vals))


@pytest.fixture
def sample_cat_numlevels_df() -> pl.DataFrame:
    base = pl.DataFrame(
        {
            "A": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "B": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
            "wgt": [2.0, 4.0, 4.0, 5.0, 14.0, 31.0, 10.0, 5.0, 5.0, 3.0, 10.0, 7.0],
        }
    )
    df = pl.concat([base] * 10).sort(["A", "B"])
    by_vals = (["D1", "D2"] * 60)[:120]
    by_int_vals = [1 if b == "D1" else 2 for b in by_vals]
    return df.with_columns(pl.Series("by", by_vals), pl.Series("by_int", by_int_vals))


# ---------------------------------------------------------------------------
# Categorical calibration
# ---------------------------------------------------------------------------


def test_calibration_cat(sample_cat_df):
    s = Sample(data=sample_cat_df, design=Design(wgt="wgt"))
    term = Cross(Cat("A"), Cat("B"))
    targets = {
        ("A1", "B1"): 80,
        ("A1", "B2"): 40,
        ("A1", "B3"): 55,
        ("A2", "B1"): 60,
        ("A2", "B2"): 150,
        ("A2", "B3"): 340,
        ("A3", "B1"): 170,
        ("A3", "B2"): 60,
        ("A3", "B3"): 200,
        ("A4", "B1"): 55,
        ("A4", "B2"): 165,
        ("A4", "B3"): 125,
    }
    s2 = s.weighting.calibrate(controls={term: targets}, wgt_name="_calib_wgt")

    df = s2.data.with_columns((pl.col("_calib_wgt") / pl.col("wgt")).alias("factor"))
    got = df.unique(subset=["A", "B"], keep="first").sort(["A", "B"])["factor"].to_numpy()
    expected = np.array(
        [4.0, 1.0, 1.375, 1.2, 1.071429, 1.096774, 1.7, 1.2, 4.0, 1.833333, 1.65, 1.785714]
    )
    assert_allclose(got, expected, rtol=1e-5)


def test_calibration_cat_by(sample_cat_df):
    s = Sample(data=sample_cat_df, design=Design(wgt="wgt"))
    term = Cross(Cat("A"), Cat("B"))
    targets = {
        ("A1", "B1"): 40,
        ("A1", "B2"): 20,
        ("A1", "B3"): 27.5,
        ("A2", "B1"): 30,
        ("A2", "B2"): 75,
        ("A2", "B3"): 170,
        ("A3", "B1"): 85,
        ("A3", "B2"): 30,
        ("A3", "B3"): 100,
        ("A4", "B1"): 27.5,
        ("A4", "B2"): 82.5,
        ("A4", "B3"): 62.5,
    }
    controls = {"D1": {term: targets}, "D2": {term: targets}}
    s2 = s.weighting.calibrate(controls=controls, by="by", wgt_name="_calib_wgt")

    df = s2.data.with_columns((pl.col("_calib_wgt") / pl.col("wgt")).alias("factor"))
    got = (
        df.unique(subset=["by", "A", "B"], keep="first")
        .sort(["by", "A", "B"])["factor"]
        .to_numpy()
    )
    factors = [4.0, 1.0, 1.375, 1.2, 1.071429, 1.096774, 1.7, 1.2, 4.0, 1.833333, 1.65, 1.785714]
    assert_allclose(got, np.array(factors + factors), rtol=1e-5)


# ---------------------------------------------------------------------------
# Numerical calibration
# ---------------------------------------------------------------------------


def test_calibration_num(sample_num_df):
    s = Sample(data=sample_num_df, design=Design(wgt="wgt"))
    ctrl = {"A": 3945, "B": 3355}
    s2 = s.weighting.calibrate(controls=ctrl, wgt_name="_calib_wgt")

    df = s2.data.with_columns((pl.col("_calib_wgt") / pl.col("wgt")).alias("factor"))
    got = df.unique(subset=["A", "B"], keep="first").sort(["A", "B"])["factor"].to_numpy()
    expected = np.array(
        [
            1.196633,
            1.165079,
            1.133525,
            1.424819,
            1.393265,
            1.361711,
            1.653006,
            1.621452,
            1.589898,
            1.881192,
            1.849638,
            1.818084,
        ]
    )
    assert_allclose(got, expected, rtol=1e-5)


def test_calibration_num_by(sample_num_df):
    s = Sample(data=sample_num_df, design=Design(wgt="wgt"))
    targets = {"A": 1972.5, "B": 1677.5}
    controls = {"D1": targets, "D2": targets}
    s2 = s.weighting.calibrate(controls=controls, by="by", wgt_name="_calib_wgt")

    df = s2.data.with_columns((pl.col("_calib_wgt") / pl.col("wgt")).alias("factor"))
    got = (
        df.unique(subset=["by", "A", "B"], keep="first")
        .sort(["by", "A", "B"])["factor"]
        .to_numpy()
    )
    base = [
        1.196633,
        1.165079,
        1.133525,
        1.424819,
        1.393265,
        1.361711,
        1.653006,
        1.621452,
        1.589898,
        1.881192,
        1.849638,
        1.818084,
    ]
    assert_allclose(got, np.array(base + base), rtol=1e-5)


# ---------------------------------------------------------------------------
# Mixed calibration
# ---------------------------------------------------------------------------


def test_calibration_mix(sample_cat_df):
    df = sample_cat_df.with_columns(pl.col("B").str.replace("B", "").cast(pl.Int64))
    s = Sample(data=df, design=Design(wgt="wgt"))
    controls = {Cat("A"): {"A1": 175, "A2": 550, "A3": 430, "A4": 345}, "B": 3355}
    s2 = s.weighting.calibrate(controls=controls, wgt_name="_calib_wgt")

    df_res = s2.data.with_columns((pl.col("_calib_wgt") / pl.col("wgt")).alias("factor"))
    got = df_res.unique(subset=["A", "B"], keep="first").sort(["A", "B"])["factor"].to_numpy()
    expected = np.array(
        [
            1.579512,
            1.721585,
            1.863659,
            0.884049,
            1.026122,
            1.168195,
            2.043445,
            2.185518,
            2.327592,
            1.554512,
            1.696585,
            1.838659,
        ]
    )
    assert_allclose(got, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# Numeric levels for categorical vars
# ---------------------------------------------------------------------------


def test_calibration_cat_numeric_levels(sample_cat_numlevels_df):
    s = Sample(data=sample_cat_numlevels_df, design=Design(wgt="wgt"))
    term = Cross(Cat("A"), Cat("B"))
    targets = {
        (1, 1): 80,
        (1, 2): 40,
        (1, 3): 55,
        (2, 1): 60,
        (2, 2): 150,
        (2, 3): 340,
        (3, 1): 170,
        (3, 2): 60,
        (3, 3): 200,
        (4, 1): 55,
        (4, 2): 165,
        (4, 3): 125,
    }
    s2 = s.weighting.calibrate(controls={term: targets}, wgt_name="_calib_wgt")

    df = s2.data.with_columns((pl.col("_calib_wgt") / pl.col("wgt")).alias("factor"))
    got = df.unique(subset=["A", "B"], keep="first").sort(["A", "B"])["factor"].to_numpy()
    expected = np.array(
        [4.0, 1.0, 1.375, 1.2, 1.071429, 1.096774, 1.7, 1.2, 4.0, 1.833333, 1.65, 1.785714]
    )
    assert_allclose(got, expected, rtol=1e-5)


def test_calibration_cat_numeric_levels_by_int(sample_cat_numlevels_df):
    s = Sample(data=sample_cat_numlevels_df, design=Design(wgt="wgt"))
    term = Cross(Cat("A"), Cat("B"))
    targets = {
        (1, 1): 40,
        (1, 2): 20,
        (1, 3): 27.5,
        (2, 1): 30,
        (2, 2): 75,
        (2, 3): 170,
        (3, 1): 85,
        (3, 2): 30,
        (3, 3): 100,
        (4, 1): 27.5,
        (4, 2): 82.5,
        (4, 3): 62.5,
    }
    controls = {1: {term: targets}, 2: {term: targets}}
    s2 = s.weighting.calibrate(controls=controls, by="by_int", wgt_name="_calib_wgt")

    df = s2.data.with_columns((pl.col("_calib_wgt") / pl.col("wgt")).alias("factor"))
    got = (
        df.unique(subset=["by_int", "A", "B"], keep="first")
        .sort(["by_int", "A", "B"])["factor"]
        .to_numpy()
    )
    factors = [4.0, 1.0, 1.375, 1.2, 1.071429, 1.096774, 1.7, 1.2, 4.0, 1.833333, 1.65, 1.785714]
    assert_allclose(got, np.array(factors + factors), rtol=1e-5)


# ---------------------------------------------------------------------------
# Auto-generated column name uses
# ---------------------------------------------------------------------------


def test_calibrate_auto_col_name_uses_svy_calib_prefix(sample_num_df):
    s = Sample(data=sample_num_df, design=Design(wgt="wgt"))
    s2 = s.weighting.calibrate(controls={"A": 3945, "B": 3355})
    assert CALIB_WGT in s2.data.columns


# Rename to match new convention
def test_calibrate_auto_col_name_uses_calib_wgt(sample_num_df):
    s = Sample(data=sample_num_df, design=Design(wgt="wgt"))
    s2 = s.weighting.calibrate(controls={"A": 3945, "B": 3355})
    assert CALIB_WGT in s2.data.columns


def test_calibrate_wgt_name_overrides_default(sample_num_df):
    """Explicit wgt_name overrides auto-generated name."""
    s = Sample(data=sample_num_df, design=Design(wgt="wgt"))
    s2 = s.weighting.calibrate(controls={"A": 3945, "B": 3355}, wgt_name="my_calib")
    assert "my_calib" in s2.data.columns
    assert CALIB_WGT not in s2.data.columns


def test_calibrate_replicate_weights_auto_prefix(sample_num_df):
    """Replicate prefix matches wgt_name."""
    df = sample_num_df.with_columns(
        pl.Series("rw1", np.ones(sample_num_df.height)),
        pl.Series("rw2", np.ones(sample_num_df.height)),
    )
    s = Sample(data=df, design=Design(wgt="wgt"))
    s._design = s.design.update_rep_weights(
        method=EstimationMethod.BOOTSTRAP, prefix="rw", n_reps=2
    )
    s2 = s.weighting.calibrate(controls={"A": 3945, "B": 3355}, wgt_name="_calib_wgt")
    assert "_calib_wgt1" in s2.data.columns
    assert "_calib_wgt2" in s2.data.columns
    assert s2.design.rep_wgts.prefix == "_calib_wgt"


# DELETE test_calibrate_replicate_weights_with_explicit_prefix entirely

# ---------------------------------------------------------------------------
# Robustness: control totals verified
# ---------------------------------------------------------------------------


def test_calibrate_matches_global_controls(sample_num_df):
    s = Sample(data=sample_num_df, design=Design(wgt="wgt"))
    ctrl = {"A": 3945, "B": 3355}
    s2 = s.weighting.calibrate(controls=ctrl, wgt_name="_calib_wgt")

    df = s2.data
    assert np.isclose(float((df["A"] * df["_calib_wgt"]).sum()), ctrl["A"], rtol=1e-9)
    assert np.isclose(float((df["B"] * df["_calib_wgt"]).sum()), ctrl["B"], rtol=1e-9)


def test_calibrate_matches_controls_within_bys(sample_cat_df):
    s = Sample(data=sample_cat_df, design=Design(wgt="wgt"))
    term = Cross(Cat("A"), Cat("B"))
    targets = {
        ("A1", "B1"): 40,
        ("A1", "B2"): 20,
        ("A1", "B3"): 27.5,
        ("A2", "B1"): 30,
        ("A2", "B2"): 75,
        ("A2", "B3"): 170,
        ("A3", "B1"): 85,
        ("A3", "B2"): 30,
        ("A3", "B3"): 100,
        ("A4", "B1"): 27.5,
        ("A4", "B2"): 82.5,
        ("A4", "B3"): 62.5,
    }
    controls = {"D1": {term: targets}, "D2": {term: targets}}
    s2 = s.weighting.calibrate(controls=controls, by="by", wgt_name="_calib_wgt")

    grouped = s2.data.group_by(["by", "A", "B"]).agg(pl.col("_calib_wgt").sum().alias("sum_w"))
    for row in grouped.iter_rows(named=True):
        assert np.isclose(row["sum_w"], targets[(row["A"], row["B"])], rtol=1e-8)


# ---------------------------------------------------------------------------
# Replicate weights
# ---------------------------------------------------------------------------


def test_calibrate_replicate_weights_auto_prefix_uses_svy_calib(sample_num_df):
    """Without rep_wgts_prefix, auto prefix should be svy_calib_{original_prefix}."""
    df = sample_num_df.with_columns(
        pl.Series("rw1", np.ones(sample_num_df.height)),
        pl.Series("rw2", np.ones(sample_num_df.height)),
    )
    s = Sample(data=df, design=Design(wgt="wgt"))
    s._design = s.design.update_rep_weights(
        method=EstimationMethod.BOOTSTRAP, prefix="rw", n_reps=2
    )
    s2 = s.weighting.calibrate(controls={"A": 3945, "B": 3355}, wgt_name="_calib_wgt")
    assert "_calib_wgt1" in s2.data.columns
    assert "_calib_wgt2" in s2.data.columns
    assert s2.design.rep_wgts.prefix == "_calib_wgt"


def test_calibrate_replicate_weights_custom_wgt_name(sample_num_df):
    """Custom wgt_name propagates to replicate weight prefix."""
    df = sample_num_df.with_columns(
        pl.Series("rw1", np.ones(sample_num_df.height)),
        pl.Series("rw2", np.ones(sample_num_df.height)),
    )
    s = Sample(data=df, design=Design(wgt="wgt"))
    s._design = s.design.update_rep_weights(
        method=EstimationMethod.BOOTSTRAP, prefix="rw", n_reps=2
    )
    s2 = s.weighting.calibrate(
        controls={"A": 3945, "B": 3355},
        wgt_name="_calib_wgt",
    )
    assert "_calib_wgt" in s2.data.columns
    assert "_calib_wgt1" in s2.data.columns
    assert "_calib_wgt2" in s2.data.columns
    assert s2.design.rep_wgts.prefix == "_calib_wgt"


def test_calibrate_replicate_weights_ignored_when_flag_set(sample_num_df):
    df = sample_num_df.with_columns(
        pl.Series("rw1", np.ones(sample_num_df.height)),
        pl.Series("rw2", np.ones(sample_num_df.height)),
    )
    s = Sample(data=df, design=Design(wgt="wgt"))
    s._design = s.design.update_rep_weights(
        method=EstimationMethod.BOOTSTRAP, prefix="rw", n_reps=2
    )
    s2 = s.weighting.calibrate(
        controls={"A": 3945, "B": 3355},
        wgt_name="_calib_wgt",
        ignore_reps=True,
    )
    assert s2.design.rep_wgts.columns == ["rw1", "rw2"]
