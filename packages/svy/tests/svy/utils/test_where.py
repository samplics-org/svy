# tests/utils/test_where.py
"""Tests for ``svy.utils.where._compile_where``.

Covers every WhereArg form documented in the compiler docstring:

  * None                                     → None
  * Mapping[str, scalar]                     → col == scalar
  * Mapping[str, list|tuple|set]             → col.is_in(values)
  * Mapping[str, pl.Series]                  → col.is_in(series)
  * Sequence[ExprLike]                       → AND of expressions
  * ExprLike (pl.Expr or wrapper)            → pass-through
"""

from __future__ import annotations

import polars as pl
import pytest
import svy

from svy.utils.where import _compile_where


# --------------------------------------------------------------------------- #
# None / empty
# --------------------------------------------------------------------------- #


class TestEmpty:
    def test_none_returns_none(self):
        assert _compile_where(None) is None

    def test_empty_mapping_returns_none(self):
        assert _compile_where({}) is None

    def test_empty_sequence_returns_none(self):
        assert _compile_where([]) is None


# --------------------------------------------------------------------------- #
# Mapping shorthand
# --------------------------------------------------------------------------- #


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "region": ["north", "south", "east", "west", "north"],
            "age": [20, 35, 45, 60, 25],
            "id": [1, 2, 3, 4, 5],
        }
    )


class TestMappingScalar:
    def test_scalar_equality(self, df):
        pred = _compile_where({"region": "north"})
        result = df.filter(pred)
        assert len(result) == 2
        assert set(result["region"].unique()) == {"north"}

    def test_scalar_equality_numeric(self, df):
        pred = _compile_where({"age": 35})
        result = df.filter(pred)
        assert result["id"].to_list() == [2]

    def test_multiple_keys_are_anded(self, df):
        pred = _compile_where({"region": "north", "age": 20})
        result = df.filter(pred)
        assert result["id"].to_list() == [1]


class TestMappingCollection:
    @pytest.mark.parametrize("ctor", [list, tuple, set])
    def test_list_tuple_set_means_is_in(self, df, ctor):
        pred = _compile_where({"region": ctor(["north", "south"])})
        result = df.filter(pred)
        assert set(result["region"].unique()) == {"north", "south"}
        assert len(result) == 3

    def test_string_is_treated_as_scalar_not_as_iterable(self, df):
        """A string value must be equality, not is_in over characters."""
        pred = _compile_where({"region": "north"})
        result = df.filter(pred)
        # If string were treated as iterable we'd get is_in(['n','o','r','t','h'])
        # which would match nothing.  Confirm we got proper equality.
        assert len(result) == 2

    def test_bytes_is_treated_as_scalar(self):
        """Bytes should not be treated as an is_in collection either."""
        # Just confirm no exception from trying to iterate bytes as values.
        d = pl.DataFrame({"x": [b"a", b"b"]})
        pred = _compile_where({"x": b"a"})
        assert len(d.filter(pred)) == 1


class TestMappingSeries:
    """pl.Series as mapping value — the case that motivated this patch."""

    def test_series_means_is_in(self, df):
        values = pl.Series(["north", "east"])
        pred = _compile_where({"region": values})
        result = df.filter(pred)
        assert set(result["region"].unique()) == {"north", "east"}
        assert len(result) == 3

    def test_series_of_ints(self, df):
        values = pl.Series([2, 4])
        pred = _compile_where({"id": values})
        result = df.filter(pred)
        assert result["id"].to_list() == [2, 4]

    def test_empty_series(self, df):
        values = pl.Series([], dtype=pl.Int64)
        pred = _compile_where({"id": values})
        result = df.filter(pred)
        assert len(result) == 0

    def test_series_combined_with_scalar_in_same_mapping(self, df):
        values = pl.Series(["north", "south"])
        pred = _compile_where({"region": values, "age": 20})
        result = df.filter(pred)
        assert result["id"].to_list() == [1]


# --------------------------------------------------------------------------- #
# Sequence of expressions
# --------------------------------------------------------------------------- #


class TestSequenceOfExprs:
    def test_list_of_svy_exprs_is_anded(self, df):
        """The canonical form — users write svy.col, not pl.col."""
        pred = _compile_where([svy.col("age") >= 30, svy.col("region") != "west"])
        result = df.filter(pred)
        assert result["id"].to_list() == [2, 3]

    def test_list_of_pl_exprs_also_works(self, df):
        """Raw pl.Expr is also supported for users already comfortable with Polars."""
        pred = _compile_where([pl.col("age") >= 30, pl.col("region") != "west"])
        result = df.filter(pred)
        assert result["id"].to_list() == [2, 3]

    def test_mixed_svy_and_pl_exprs(self, df):
        """svy.Expr and pl.Expr should compose cleanly in the same list."""
        pred = _compile_where([svy.col("age") >= 30, pl.col("region") != "west"])
        result = df.filter(pred)
        assert result["id"].to_list() == [2, 3]

    def test_single_item_sequence(self, df):
        pred = _compile_where([svy.col("age") > 40])
        result = df.filter(pred)
        assert result["id"].to_list() == [3, 4]


# --------------------------------------------------------------------------- #
# Single ExprLike
# --------------------------------------------------------------------------- #


class TestSingleExpr:
    def test_svy_expr_passthrough(self, df):
        """The canonical form."""
        pred = _compile_where(svy.col("age") > 30)
        result = df.filter(pred)
        assert result["id"].to_list() == [2, 3, 4]

    def test_pl_expr_passthrough(self, df):
        """Raw pl.Expr is also accepted."""
        pred = _compile_where(pl.col("age") > 30)
        result = df.filter(pred)
        assert result["id"].to_list() == [2, 3, 4]

    def test_svy_col_is_in_with_series(self, df):
        """The form users should reach for when they have a pl.Series of values."""
        values = pl.Series(["north", "south"])
        pred = _compile_where(svy.col("region").is_in(values))
        result = df.filter(pred)
        assert len(result) == 3

    def test_svy_col_is_in_with_series(self, df):
        """svy.col routes through the wrapper, which handles implode internally."""
        import svy
        values = pl.Series(["north", "south"])
        pred = _compile_where(svy.col("region").is_in(values))
        result = df.filter(pred)
        assert len(result) == 3
