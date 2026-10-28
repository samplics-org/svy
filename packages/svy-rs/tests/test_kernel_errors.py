"""Kernel failures raise a coded ValueError (bad input) or RuntimeError (numerical)."""

import numpy as np
import polars as pl
import pytest

import svy_rs as ps


@pytest.fixture
def df():
    return pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "x": [2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
            "w": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            "g3": ["a", "a", "b", "b", "c", "c"],
            "g1": ["a"] * 6,
        }
    )


INVALID_INPUT = [
    (
        lambda d: ps.taylor_mean(d, "y", "w", deff_ref="srswor"),
        "INVALID_CHOICE",
        "unknown deff reference 'srswor'; expected 'wor' or 'wr'",
    ),
    (
        lambda d: ps.taylor_assoc(d, ["y"], ["x"], "spearman", "w"),
        "INVALID_CHOICE",
        "unknown association kind 'spearman'",
    ),
    (
        lambda d: ps.taylor_assoc(d, ["y", "x"], ["x"], "corr", "w"),
        "SHAPE_MISMATCH",
        "pair columns must be equal length, got 2 and 1",
    ),
    (
        lambda d: ps.replicate_assoc(d, ["y", "x"], ["x"], "corr", "w", ["w"], [1.0]),
        "SHAPE_MISMATCH",
        "pair columns must be equal length, got 2 and 1",
    ),
    (
        lambda d: ps.replicate_mean(d, "y", "w", ["w", "x"], [1.0]),
        "REP_COEFS_LENGTH_MISMATCH",
        "rep_coefs has 1 entries but there are 2 replicate weight columns",
    ),
    (
        lambda d: ps.replicate_mean(d, "y", "w", ["w"], [1.0], center="median"),
        "INVALID_CHOICE",
        "Unknown center: median. Use 'rep_mean' or 'estimate'",
    ),
    (
        lambda d: ps.ttest_rs(d, "y", "w", group_col="g3"),
        "TEST_GROUP_COUNT",
        "Two-sample t-test requires exactly 2 groups, found 3",
    ),
    (
        lambda d: ps.ranktest_rs(d, "y", "g1", "w"),
        "TEST_GROUP_COUNT",
        "Rank test requires at least 2 groups with positive weight, found 1",
    ),
    (
        lambda d: ps.ttest_rs(d, "y", "w", rep_weight_cols=["w"]),
        "INVALID_VALUE",
        "rep_coefs is required with rep_weight_cols",
    ),
    (
        lambda d: ps.ttest_rs(d, "y", "w", rep_weight_cols=["w", "x"], rep_coefs=[1.0]),
        "REP_COEFS_LENGTH_MISMATCH",
        "rep_coefs has 1 entries but there are 2 replicate weight columns",
    ),
    (
        lambda d: ps.ranktest_rs(d, "y", "g3", "w", scores_given=True),
        "INVALID_VALUE",
        "scores_given requires rep_weight_cols",
    ),
    (
        lambda d: ps.weighted_quantile_at(np.array([1.0, 2.0]), np.array([0.5]), [0.5]),
        "SHAPE_MISMATCH",
        "values_sorted has 2 entries but cdf has 1",
    ),
]


@pytest.mark.parametrize(("call", "code", "message"), INVALID_INPUT)
def test_invalid_input_is_a_coded_value_error(df, call, code, message):
    with pytest.raises(ps.KernelValueError) as info:
        call(df)
    assert isinstance(info.value, ValueError)
    assert not isinstance(info.value, RuntimeError)
    assert info.value.code == code
    assert message in str(info.value)


def test_polars_failure_is_a_coded_runtime_error(df):
    with pytest.raises(ps.KernelRuntimeError) as info:
        ps.taylor_mean(df, "absent", "w")
    assert isinstance(info.value, RuntimeError)
    assert not isinstance(info.value, ValueError)
    assert info.value.code == "KERNEL_INTERNAL"
    assert "absent" in str(info.value)
