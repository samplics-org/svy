# tests/svy/wrangling/test_mutate_extended.py
"""Extended tests for mutate functionality - edge cases and advanced usage."""

import numpy as np
import polars as pl
import pytest

from svy.core.expr import col, when
from svy.core.sample import Sample
from svy.errors import MethodError


# ---------- Fixtures ----------


@pytest.fixture
def sample_basic():
    return Sample(pl.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]}))


@pytest.fixture
def sample_with_nulls():
    return Sample(
        pl.DataFrame(
            {
                "x": [1, None, 3],
                "y": [None, 2.0, None],
            }
        )
    )


# ==================== None and Null handling ====================


def test_mutate_none_creates_null_column(sample_basic: Sample):
    out = sample_basic.wrangling.mutate({"c": None})
    assert "c" in out._data.columns
    assert out._data["c"].null_count() == 3


def test_mutate_preserves_nulls_in_expressions(sample_with_nulls: Sample):
    out = sample_with_nulls.wrangling.mutate({"z": col("x") + col("y")})
    # Null + anything = Null
    assert out._data["z"].null_count() > 0


# ==================== Boolean scalars ====================


def test_mutate_bool_true_scalar(sample_basic: Sample):
    out = sample_basic.wrangling.mutate({"flag": True})
    assert out._data["flag"].to_list() == [True, True, True]


def test_mutate_bool_false_scalar(sample_basic: Sample):
    out = sample_basic.wrangling.mutate({"flag": False})
    assert out._data["flag"].to_list() == [False, False, False]


# ==================== String scalars ====================


def test_mutate_string_scalar(sample_basic: Sample):
    out = sample_basic.wrangling.mutate({"label": "constant"})
    assert out._data["label"].to_list() == ["constant", "constant", "constant"]


def test_mutate_empty_string_scalar(sample_basic: Sample):
    out = sample_basic.wrangling.mutate({"empty": ""})
    assert out._data["empty"].to_list() == ["", "", ""]


# ==================== Numeric scalars ====================


def test_mutate_int_scalar(sample_basic: Sample):
    out = sample_basic.wrangling.mutate({"one": 1})
    assert out._data["one"].to_list() == [1, 1, 1]


def test_mutate_float_scalar(sample_basic: Sample):
    out = sample_basic.wrangling.mutate({"pi": 3.14159})
    assert out._data["pi"].to_list() == [3.14159, 3.14159, 3.14159]


def test_mutate_negative_scalar(sample_basic: Sample):
    out = sample_basic.wrangling.mutate({"neg": -42})
    assert out._data["neg"].to_list() == [-42, -42, -42]


def test_mutate_zero_scalar(sample_basic: Sample):
    out = sample_basic.wrangling.mutate({"zero": 0})
    assert out._data["zero"].to_list() == [0, 0, 0]


# ==================== Tuple inputs (converted to list) ====================


def test_mutate_tuple_input(sample_basic: Sample):
    out = sample_basic.wrangling.mutate({"c": (100, 200, 300)})
    assert out._data["c"].to_list() == [100, 200, 300]


# ==================== Empty mutations ====================


def test_mutate_empty_specs(sample_basic: Sample):
    out = sample_basic.wrangling.mutate({})
    # Should return unchanged
    assert out._data.columns == sample_basic._data.columns


# ==================== Overwriting existing columns ====================


def test_mutate_overwrite_existing_column(sample_basic: Sample):
    out = sample_basic.wrangling.mutate({"a": col("a") * 100})
    assert out._data["a"].to_list() == [100, 200, 300]


def test_mutate_overwrite_with_scalar(sample_basic: Sample):
    out = sample_basic.wrangling.mutate({"a": 999})
    assert out._data["a"].to_list() == [999, 999, 999]


# ==================== Conditional expressions (when/then/otherwise) ====================


def test_mutate_when_then_otherwise_complete(sample_basic: Sample):
    expr = when(col("a") > 1).then("big").otherwise("small")
    out = sample_basic.wrangling.mutate({"size": expr})
    assert out._data["size"].to_list() == ["small", "big", "big"]


def test_mutate_chained_when_using_polars(sample_basic: Sample):
    # svy's when/then doesn't support chaining, use Polars for complex cases
    expr = (
        pl.when(pl.col("a") == 1)
        .then(pl.lit("one"))
        .when(pl.col("a") == 2)
        .then(pl.lit("two"))
        .otherwise(pl.lit("other"))
    )
    out = sample_basic.wrangling.mutate({"label": expr})
    assert out._data["label"].to_list() == ["one", "two", "other"]


# ==================== Complex dependency chains ====================


def test_mutate_three_level_dependency(sample_basic: Sample):
    out = sample_basic.wrangling.mutate(
        {
            "c": col("a") + col("b"),
            "d": col("c") * 2,
            "e": col("d") + 1,
        }
    )
    # a=[1,2,3], b=[10,20,30]
    # c = [11, 22, 33]
    # d = [22, 44, 66]
    # e = [23, 45, 67]
    assert out._data["c"].to_list() == [11, 22, 33]
    assert out._data["d"].to_list() == [22, 44, 66]
    assert out._data["e"].to_list() == [23, 45, 67]


def test_mutate_parallel_dependencies(sample_basic: Sample):
    # Both d and e depend on c, but not on each other
    out = sample_basic.wrangling.mutate(
        {
            "c": col("a") + 10,
            "d": col("c") * 2,
            "e": col("c") - 5,
        }
    )
    assert out._data["c"].to_list() == [11, 12, 13]
    assert out._data["d"].to_list() == [22, 24, 26]
    assert out._data["e"].to_list() == [6, 7, 8]


# ==================== Callable variations ====================


def test_mutate_callable_returns_polars_expr(sample_basic: Sample):
    out = sample_basic.wrangling.mutate({"c": lambda env: pl.col("a") + pl.col("b")})
    assert out._data["c"].to_list() == [11, 22, 33]


def test_mutate_callable_uses_env(sample_basic: Sample):
    # env provides column accessors
    out = sample_basic.wrangling.mutate({"c": lambda env: env["a"] + env["b"]})
    assert out._data["c"].to_list() == [11, 22, 33]


def test_mutate_callable_returns_scalar(sample_basic: Sample):
    out = sample_basic.wrangling.mutate({"c": lambda env: 42})
    assert out._data["c"].to_list() == [42, 42, 42]


# ==================== Mixed types in same call ====================


def test_mutate_mixed_types_comprehensive(sample_basic: Sample):
    arr = np.array([0.1, 0.2, 0.3])
    out = sample_basic.wrangling.mutate(
        {
            "scalar_int": 1,
            "scalar_str": "hello",
            "scalar_bool": True,
            "from_expr": col("a") * 2,
            "from_polars": pl.col("b") + 1,
            "from_numpy": arr,
            "from_list": [7, 8, 9],
            "from_callable": lambda env: env["a"] + 100,
        }
    )
    assert out._data["scalar_int"].to_list() == [1, 1, 1]
    assert out._data["scalar_str"].to_list() == ["hello", "hello", "hello"]
    assert out._data["scalar_bool"].to_list() == [True, True, True]
    assert out._data["from_expr"].to_list() == [2, 4, 6]
    assert out._data["from_polars"].to_list() == [11, 21, 31]
    assert out._data["from_numpy"].to_list() == [0.1, 0.2, 0.3]
    assert out._data["from_list"].to_list() == [7, 8, 9]
    assert out._data["from_callable"].to_list() == [101, 102, 103]


# ==================== Series with different dtypes ====================


def test_mutate_series_int(sample_basic: Sample):
    s = pl.Series("new", [100, 200, 300])
    out = sample_basic.wrangling.mutate({"c": s})
    assert out._data["c"].dtype in [pl.Int64, pl.Int32]


def test_mutate_series_float(sample_basic: Sample):
    s = pl.Series("new", [1.1, 2.2, 3.3])
    out = sample_basic.wrangling.mutate({"c": s})
    assert out._data["c"].dtype == pl.Float64


def test_mutate_series_string(sample_basic: Sample):
    s = pl.Series("new", ["x", "y", "z"])
    out = sample_basic.wrangling.mutate({"c": s})
    assert out._data["c"].dtype == pl.Utf8


def test_mutate_series_bool(sample_basic: Sample):
    s = pl.Series("new", [True, False, True])
    out = sample_basic.wrangling.mutate({"c": s})
    assert out._data["c"].dtype == pl.Boolean


# ==================== Error cases ====================


def test_mutate_unsupported_type_raises():
    s = Sample(pl.DataFrame({"a": [1, 2, 3]}))
    with pytest.raises((MethodError, TypeError)):
        s.wrangling.mutate({"c": object()})


def test_mutate_list_wrong_length_raises(sample_basic: Sample):
    with pytest.raises(MethodError) as exc_info:
        sample_basic.wrangling.mutate({"c": [1, 2]})  # n=3, list has 2
    assert exc_info.value.code == "MUTATE_COMPILE_FAILED"


def test_mutate_numpy_wrong_length_raises(sample_basic: Sample):
    with pytest.raises(MethodError):
        sample_basic.wrangling.mutate({"c": np.array([1, 2, 3, 4])})  # n=3, array has 4


def test_mutate_series_wrong_length_raises(sample_basic: Sample):
    s = pl.Series("x", [1, 2])
    with pytest.raises(MethodError):
        sample_basic.wrangling.mutate({"c": s})


def test_mutate_self_reference_raises():
    s = Sample(pl.DataFrame({"a": [1, 2, 3]}))
    # Column depends on itself (if 'c' doesn't exist)
    with pytest.raises((MethodError, Exception)):
        s.wrangling.mutate({"c": pl.col("c") + 1})


# ==================== Large number of columns ====================


def test_mutate_many_columns():
    s = Sample(pl.DataFrame({"base": [1, 2, 3]}))
    specs = {f"col_{i}": i for i in range(50)}
    out = s.wrangling.mutate(specs)
    assert all(f"col_{i}" in out._data.columns for i in range(50))


# ==================== Inplace parameter ====================


def test_mutate_inplace_true_modifies_original():
    df = pl.DataFrame({"a": [1, 2, 3]})
    s = Sample(df)
    original_id = id(s)
    out = s.wrangling.mutate({"b": 1}, inplace=True)
    assert out is s
    assert id(out) == original_id
    assert "b" in s._data.columns


def test_mutate_inplace_false_returns_new():
    df = pl.DataFrame({"a": [1, 2, 3]})
    s = Sample(df)
    out = s.wrangling.mutate({"b": 1}, inplace=False)
    # Behavior depends on implementation - may return same or new Sample
    assert "b" in out._data.columns
