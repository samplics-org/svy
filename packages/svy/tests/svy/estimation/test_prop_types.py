# tests/svy/estimation/test_prop_types.py

import numpy as np
import polars as pl
import pytest

import svy
from svy.core.enumerations import EstimationMethod


# ============================================================================
# Shared fixtures
# ============================================================================


@pytest.fixture
def base_df():
    """10-row DataFrame with design columns and a string 'answer' column."""
    return pl.DataFrame(
        {
            "stratum": ["A"] * 5 + ["B"] * 5,
            "psu": [1, 1, 2, 2, 3, 4, 4, 5, 5, 6],
            "wgt": [2.0] * 10,
            "answer": ["yes", "no", "yes", "yes", "no", "no", "no", "yes", "no", "yes"],
        }
    )


@pytest.fixture
def string_prop_sample(base_df):
    design = svy.Design(stratum="stratum", psu="psu", wgt="wgt")
    return svy.Sample(data=base_df, design=design)


@pytest.fixture
def rep_sample(base_df):
    """Same data with 8 BRR replicate weight columns."""
    rng = np.random.default_rng(42)
    n = len(base_df)
    n_reps = 8
    rep_cols = [pl.Series(f"rep{r + 1}", rng.uniform(0.5, 1.5, n) * 2.0) for r in range(n_reps)]
    df = base_df.with_columns(rep_cols)
    rep_wgts = svy.RepWeights(prefix="rep", method=EstimationMethod.BRR, n_reps=n_reps)
    design = svy.Design(stratum="stratum", psu="psu", wgt="wgt", rep_wgts=rep_wgts)
    return svy.Sample(data=df, design=design)


def _make_rep_sample(df):
    """Helper: add 8 BRR rep weights to an arbitrary DataFrame."""
    rng = np.random.default_rng(99)
    n, n_reps = len(df), 8
    rep_cols = [pl.Series(f"rep{r + 1}", rng.uniform(0.5, 1.5, n) * 2.0) for r in range(n_reps)]
    df2 = df.with_columns(rep_cols)
    rep_wgts = svy.RepWeights(prefix="rep", method=EstimationMethod.BRR, n_reps=n_reps)
    design = svy.Design(stratum="stratum", psu="psu", wgt="wgt", rep_wgts=rep_wgts)
    return svy.Sample(data=df2, design=design)


def _taylor_sample(df):
    return svy.Sample(data=df, design=svy.Design(stratum="stratum", psu="psu", wgt="wgt"))


# ============================================================================
# String
# ============================================================================


def test_prop_string_taylor(string_prop_sample):
    result = string_prop_sample.estimation.prop("answer")
    levels = {p.y_level for p in result.estimates}
    assert levels == {"yes", "no"}
    ests = {p.y_level: p.est for p in result.estimates}
    assert ests["yes"] == pytest.approx(0.5, abs=1e-10)
    assert ests["no"] == pytest.approx(0.5, abs=1e-10)
    assert sum(ests.values()) == pytest.approx(1.0, abs=1e-10)


def test_prop_string_replication(rep_sample):
    result = rep_sample.estimation.prop("answer")
    levels = {p.y_level for p in result.estimates}
    assert levels == {"yes", "no"}
    assert sum(p.est for p in result.estimates) == pytest.approx(1.0, abs=1e-10)


def test_prop_string_three_levels(base_df):
    """String y with more than two levels."""
    df = base_df.with_columns(
        pl.Series("grade", ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"])
    )
    result = _taylor_sample(df).estimation.prop("grade")
    assert {p.y_level for p in result.estimates} == {"A", "B", "C"}
    assert sum(p.est for p in result.estimates) == pytest.approx(1.0, abs=1e-10)


def test_prop_string_with_by(string_prop_sample):
    """String y with a by= grouping."""
    result = string_prop_sample.estimation.prop("answer", by="stratum")
    pairs = {(p.by_level[0], p.y_level) for p in result.estimates}
    assert ("A", "yes") in pairs
    assert ("B", "no") in pairs
    for stratum in ("A", "B"):
        stratum_ests = [p.est for p in result.estimates if p.by_level[0] == stratum]
        assert sum(stratum_ests) == pytest.approx(1.0, abs=1e-10)


# ============================================================================
# Categorical
# ============================================================================


def test_prop_categorical_taylor(base_df):
    df = base_df.with_columns(pl.col("answer").cast(pl.Categorical))
    result = _taylor_sample(df).estimation.prop("answer")
    assert {p.y_level for p in result.estimates} == {"yes", "no"}
    assert sum(p.est for p in result.estimates) == pytest.approx(1.0, abs=1e-10)


def test_prop_categorical_replication(base_df):
    df = base_df.with_columns(pl.col("answer").cast(pl.Categorical))
    result = _make_rep_sample(df).estimation.prop("answer")
    assert {p.y_level for p in result.estimates} == {"yes", "no"}
    assert sum(p.est for p in result.estimates) == pytest.approx(1.0, abs=1e-10)


# ============================================================================
# Enum
# ============================================================================


def test_prop_enum_taylor(base_df):
    df = base_df.with_columns(pl.col("answer").cast(pl.Enum(["no", "yes"])))
    result = _taylor_sample(df).estimation.prop("answer")
    assert {p.y_level for p in result.estimates} == {"yes", "no"}
    assert sum(p.est for p in result.estimates) == pytest.approx(1.0, abs=1e-10)


def test_prop_enum_replication(base_df):
    df = base_df.with_columns(pl.col("answer").cast(pl.Enum(["no", "yes"])))
    result = _make_rep_sample(df).estimation.prop("answer")
    assert {p.y_level for p in result.estimates} == {"yes", "no"}
    assert sum(p.est for p in result.estimates) == pytest.approx(1.0, abs=1e-10)


# ============================================================================
# Boolean
# ============================================================================


def test_prop_boolean_taylor(base_df):
    df = base_df.with_columns((pl.col("answer") == "yes").alias("answered_yes"))
    result = _taylor_sample(df).estimation.prop("answered_yes")
    # Taylor converts bool → "true"/"false" string levels via cast(&DataType::String)
    assert sum(p.est for p in result.estimates) == pytest.approx(1.0, abs=1e-10)


def test_prop_boolean_replication(base_df):
    df = base_df.with_columns((pl.col("answer") == "yes").alias("answered_yes"))
    result = _make_rep_sample(df).estimation.prop("answered_yes")
    # Replication converts bool → 0/1 integer levels
    assert sum(p.est for p in result.estimates) == pytest.approx(1.0, abs=1e-10)


# ============================================================================
# Integer (non-Int64 variants)
# ============================================================================


@pytest.mark.parametrize("int_dtype", [pl.Int8, pl.Int16, pl.Int32, pl.UInt8, pl.UInt32])
def test_prop_integer_taylor(base_df, int_dtype):
    df = base_df.with_columns(
        pl.Series("category", [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]).cast(int_dtype)
    )
    result = _taylor_sample(df).estimation.prop("category")
    assert len(result.estimates) == 2
    assert sum(p.est for p in result.estimates) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("int_dtype", [pl.Int8, pl.Int16, pl.Int32, pl.UInt8, pl.UInt32])
def test_prop_integer_replication(base_df, int_dtype):
    df = base_df.with_columns(
        pl.Series("category", [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]).cast(int_dtype)
    )
    result = _make_rep_sample(df).estimation.prop("category")
    assert len(result.estimates) == 2
    assert sum(p.est for p in result.estimates) == pytest.approx(1.0, abs=1e-10)


# ============================================================================
# Float — should raise on both paths
# ============================================================================


def test_prop_float_taylor_raises(base_df):
    df = base_df.with_columns(
        pl.Series("score", [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5])
    )
    with pytest.raises((TypeError, Exception), match="[Ff]loat|prop"):
        _taylor_sample(df).estimation.prop("score")


def test_prop_float_replication_raises(base_df):
    df = base_df.with_columns(
        pl.Series("score", [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5])
    )
    with pytest.raises((TypeError, Exception), match="[Ff]loat|prop"):
        _make_rep_sample(df).estimation.prop("score")
