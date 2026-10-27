"""A missing design value raises at estimation, with or without drop_nulls."""

from __future__ import annotations

import math

import polars as pl
import pytest

from svy import BrrWgts, Design, Sample
from svy.errors import Code, DataError


@pytest.fixture
def data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "stratum": ["a"] * 6 + ["b"] * 6,
            "psu": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            "wgt": [10.0, 12.0, 9.0, 11.0, 10.0, 8.0, 14.0, 13.0, 9.0, 10.0, 12.0, 11.0],
            "y": [1.0, 3.0, 2.0, 5.0, 4.0, 2.0, 6.0, 3.0, 5.0, 4.0, 2.0, 1.0],
            "x": [2.0, 1.0, 3.0, 2.0, 1.0, 4.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0],
            "g": ["u", "v"] * 6,
        }
    )


def _set(df: pl.DataFrame, col: str, value, rows: tuple[int, ...] = (4, 6)) -> pl.DataFrame:
    # Two rows in different PSUs, so a null stratum does not also make a singleton.
    return df.with_columns(
        pl.when(pl.int_range(pl.len()).is_in(rows))
        .then(pl.lit(value, dtype=df.schema[col]))
        .otherwise(pl.col(col))
        .alias(col)
    )


CALLS = {
    "mean": lambda s, **kw: s.estimation.mean("y", **kw),
    "total": lambda s, **kw: s.estimation.total("y", **kw),
    "prop": lambda s, **kw: s.estimation.prop("g", **kw),
    "ratio": lambda s, **kw: s.estimation.ratio("y", "x", **kw),
    "median": lambda s, **kw: s.estimation.median("y", **kw),
    "corr": lambda s, **kw: s.estimation.corr(["y", "x"], **kw),
    "mean_by": lambda s, **kw: s.estimation.mean("y", by="g", **kw),
    "ttest": lambda s, **kw: s.categorical.ttest("y", mean_h0=3.0, **kw),
    "glm": lambda s, **kw: s.glm.fit("y", x=["x"], **kw),
}

MISSING = [
    ("stratum", None, "stratum (NULL)"),
    ("psu", None, "psu (NULL)"),
    ("wgt", None, "wgt (NULL)"),
    ("wgt", math.nan, "wgt (NULL)"),  # Sample stores a NaN weight as null
    ("wgt", math.inf, "wgt (±∞)"),
]


def _assert_design_error(exc: pytest.ExceptionInfo, affected: str) -> None:
    err = exc.value
    assert err.code == Code.MISSING_VALUES
    assert err.got == [affected]
    assert affected in str(err)
    assert "no place in the design" in err.hint
    assert "drop_nulls=True" in err.hint


@pytest.mark.parametrize("call", CALLS)
@pytest.mark.parametrize("col, value, affected", MISSING)
@pytest.mark.parametrize("drop_nulls", [True, False])
def test_missing_design_value_raises(data, call, col, value, affected, drop_nulls):
    s = Sample(_set(data, col, value), Design(stratum="stratum", psu="psu", wgt="wgt"))
    with pytest.raises(DataError.ValueError) as exc:
        CALLS[call](s, drop_nulls=drop_nulls)
    _assert_design_error(exc, affected)


@pytest.mark.parametrize("drop_nulls", [True, False])
def test_missing_design_value_outside_where_domain_raises(data, drop_nulls):
    # Rows 4 and 6 are in stratum "a"; the domain is stratum "b".
    s = Sample(_set(data, "psu", None), Design(stratum="stratum", psu="psu", wgt="wgt"))
    with pytest.raises(DataError.ValueError) as exc:
        s.estimation.mean("y", where=pl.col("stratum") == "b", drop_nulls=drop_nulls)
    _assert_design_error(exc, "psu (NULL)")


def test_missing_weight_raises_on_a_replicate_design(data):
    reps = {f"r{i}": pl.col("wgt") * (1.5 if i % 2 else 0.5) for i in range(1, 5)}
    df = _set(data.with_columns(**reps), "wgt", None)
    s = Sample(
        df, Design(stratum="stratum", psu="psu", wgt="wgt", rep_wgts=BrrWgts(prefix="r", n_reps=4))
    )
    with pytest.raises(DataError.ValueError) as exc:
        s.estimation.mean("y", method="replication", drop_nulls=True)
    _assert_design_error(exc, "wgt (NULL)")


def test_design_and_analysis_values_missing_reports_design_first(data):
    df = _set(_set(data, "wgt", None, rows=(4,)), "y", None, rows=(7,))
    s = Sample(df, Design(stratum="stratum", psu="psu", wgt="wgt"))
    with pytest.raises(DataError.ValueError) as exc:
        s.estimation.mean("y")
    _assert_design_error(exc, "wgt (NULL)")


@pytest.mark.parametrize("where", [None, pl.col("stratum") == "a"])
def test_missing_analysis_value_hint_says_rows_stay_in_the_variance(data, where):
    s = Sample(_set(data, "y", None), Design(stratum="stratum", psu="psu", wgt="wgt"))
    with pytest.raises(DataError.ValueError) as exc:
        s.estimation.mean("y", where=where)
    err = exc.value
    assert err.code == Code.MISSING_VALUES
    assert err.got == ["y (NULL)"]
    assert err.hint == (
        "Set drop_nulls=True to exclude those rows from the estimate; "
        "they keep their place in the variance."
    )


def test_missing_value_in_unused_column_is_ignored(data):
    df = data.with_columns(sex=pl.Series([None] * 12, dtype=pl.String))
    s = Sample(df, Design(stratum="stratum", psu="psu", wgt="wgt"))
    # Weighted mean of y over all rows, by hand: sum(w*y) / sum(w).
    w, y = data["wgt"].to_list(), data["y"].to_list()
    expected = sum(a * b for a, b in zip(w, y)) / sum(w)
    assert s.estimation.mean("y").estimates[0].est == pytest.approx(expected, rel=1e-12)


def test_selection_hint_is_unchanged(data):
    s = Sample(_set(data, "g", None), Design(stratum="stratum"))
    with pytest.raises(DataError.ValueError) as exc:
        s.sampling.srs(2, by="g", rstate=1)
    assert exc.value.hint == "Set drop_nulls=True to leave those rows out."
