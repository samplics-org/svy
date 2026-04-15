# tests/svy/core/test_expr.py
from __future__ import annotations

import math

from datetime import date, datetime

import polars as pl
import pytest

from svy.core.expr import (
    Expr,
    all_horizontal,
    any_horizontal,
    coalesce,
    col,
    cols,
    concat_str,
    lit,
    max_horizontal,
    min_horizontal,
    struct,
    sum_horizontal,
    to_polars_expr,
    when,
)


# =============================================================================
# Test Fixtures
# =============================================================================


def _df_basic() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [0, 1, 2, 3, 4],
            "y": [10, 20, 30, 40, 50],
            "g": ["a", "b", "c", "a", "b"],
            "s": ["alpha", "beta", "gamma", "alphabet", "bet"],
        }
    )


def _df_nulls() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": [1, None, 3, None, 5],
            "b": [None, 2, None, 4, None],
            "c": [10, 20, 30, 40, 50],
        }
    )


def _df_floats() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": [1.0, 4.0, 9.0, 16.0, 25.0],
            "y": [0.0, 1.0, -1.0, float("nan"), float("inf")],
        }
    )


def _df_dates() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "dt": [
                datetime(2023, 1, 15, 10, 30, 45),
                datetime(2023, 6, 20, 14, 0, 0),
                datetime(2024, 12, 31, 23, 59, 59),
            ],
            "d": [date(2023, 1, 15), date(2023, 6, 20), date(2024, 12, 31)],
        }
    )


def _df_lists() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "nums": [[1, 2, 3], [4, 5], [6, 7, 8, 9]],
            "strs": [["a", "b"], ["c"], ["d", "e", "f"]],
        }
    )


# =============================================================================
# to_polars_expr normalization
# =============================================================================


def test_to_polars_expr_accepts_Expr_and_pl_expr():
    e = col("x") + 1
    p = to_polars_expr(e)
    assert isinstance(p, pl.Expr)

    p2 = to_polars_expr(pl.col("x") * 2)
    assert isinstance(p2, pl.Expr)


def test_to_polars_expr_incomplete_when_raises_typeerror():
    partial = when(col("x") > 0).then(1)
    with pytest.raises(TypeError, match="Incomplete conditional expression"):
        _ = to_polars_expr(partial)


# =============================================================================
# __repr__
# =============================================================================


def test_repr():
    e = col("x")
    assert "Expr" in repr(e)
    assert "col" in repr(e)

    w = when(col("x") > 0)
    assert "_When" in repr(w)

    t = when(col("x") > 0).then(1)
    assert "_Then" in repr(t)


# =============================================================================
# Input Validation
# =============================================================================


def test_col_requires_string():
    with pytest.raises(TypeError, match="expects a string"):
        col(123)


def test_when_requires_expr():
    with pytest.raises(TypeError, match="expects an Expr"):
        when("not an expr")


def test_coalesce_requires_at_least_one_arg():
    with pytest.raises(ValueError, match="at least one"):
        coalesce()


# =============================================================================
# Arithmetic Operations
# =============================================================================


def test_arithmetic_basic():
    df = _df_basic()
    out = df.select(
        (col("x") + 1).alias("add")._e,
        (col("x") - 1).alias("sub")._e,
        (col("x") * 2).alias("mul")._e,
        (col("y") / 10).alias("div")._e,
        (col("x") ** 2).alias("pow")._e,
    )
    assert out["add"].to_list() == [1, 2, 3, 4, 5]
    assert out["sub"].to_list() == [-1, 0, 1, 2, 3]
    assert out["mul"].to_list() == [0, 2, 4, 6, 8]
    assert out["div"].to_list() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert out["pow"].to_list() == [0, 1, 4, 9, 16]


def test_arithmetic_floordiv_mod():
    df = pl.DataFrame({"x": [7, 8, 9, 10, 11]})
    out = df.select(
        (col("x") // 3).alias("floordiv")._e,
        (col("x") % 3).alias("mod")._e,
    )
    assert out["floordiv"].to_list() == [2, 2, 3, 3, 3]
    assert out["mod"].to_list() == [1, 2, 0, 1, 2]


def test_arithmetic_reverse_operations():
    df = pl.DataFrame({"x": [1, 2, 3]})
    out = df.select(
        (10 + col("x")).alias("radd")._e,
        (10 - col("x")).alias("rsub")._e,
        (10 * col("x")).alias("rmul")._e,
        (10 / col("x")).alias("rtruediv")._e,
        (10 // col("x")).alias("rfloordiv")._e,
        (10 % col("x")).alias("rmod")._e,
        (2 ** col("x")).alias("rpow")._e,
    )
    assert out["radd"].to_list() == [11, 12, 13]
    assert out["rsub"].to_list() == [9, 8, 7]
    assert out["rmul"].to_list() == [10, 20, 30]
    assert out["rtruediv"].to_list() == [10.0, 5.0, 10 / 3]
    assert out["rfloordiv"].to_list() == [10, 5, 3]
    assert out["rmod"].to_list() == [0, 0, 1]
    assert out["rpow"].to_list() == [2, 4, 8]


def test_unary_neg_and_abs():
    df = pl.DataFrame({"x": [-2, -1, 0, 1, 2]})
    out = df.select(
        (-col("x")).alias("neg")._e,
        abs(col("x")).alias("abs")._e,
    )
    assert out["neg"].to_list() == [2, 1, 0, -1, -2]
    assert out["abs"].to_list() == [2, 1, 0, 1, 2]


# =============================================================================
# Comparisons
# =============================================================================


def test_comparisons():
    df = _df_basic()
    out = df.select(
        (col("x") < 2).alias("lt")._e,
        (col("x") <= 2).alias("le")._e,
        (col("x") > 2).alias("gt")._e,
        (col("x") >= 2).alias("ge")._e,
        (col("x") == 2).alias("eq")._e,
        (col("x") != 2).alias("ne")._e,
    )
    assert out["lt"].to_list() == [True, True, False, False, False]
    assert out["le"].to_list() == [True, True, True, False, False]
    assert out["gt"].to_list() == [False, False, False, True, True]
    assert out["ge"].to_list() == [False, False, True, True, True]
    assert out["eq"].to_list() == [False, False, True, False, False]
    assert out["ne"].to_list() == [True, True, False, True, True]


# =============================================================================
# Boolean Composition
# =============================================================================


def test_boolean_and_or_not():
    df = _df_basic()
    expr = ((col("x") >= 1) & (col("x") <= 3)) | (col("y") == 50)
    out = df.select(expr.alias("mask")._e)
    expected = [(1 <= x <= 3) or (y == 50) for x, y in zip(df["x"], df["y"])]
    assert out["mask"].to_list() == expected

    out_not = df.select((~(col("x") < 2)).alias("not_lt2")._e)
    expected_not = [not (x < 2) for x in df["x"]]
    assert out_not["not_lt2"].to_list() == expected_not


def test_boolean_xor():
    df = pl.DataFrame({"a": [True, True, False, False], "b": [True, False, True, False]})
    out = df.select((col("a") ^ col("b")).alias("xor")._e)
    assert out["xor"].to_list() == [False, True, True, False]


# =============================================================================
# Naming / Aliasing
# =============================================================================


def test_alias():
    df = _df_basic()
    out = df.select((col("x") + 1).alias("new_name")._e)
    assert "new_name" in out.columns


def test_name_prefix_suffix():
    df = _df_basic()
    out = df.select(
        col("x").name_prefix("pre_")._e,
        col("y").name_suffix("_suf")._e,
    )
    assert "pre_x" in out.columns
    assert "y_suf" in out.columns


# =============================================================================
# Null Handling
# =============================================================================


def test_fill_null():
    df = _df_nulls()
    out = df.select(col("a").fill_null(0).alias("filled")._e)
    assert out["filled"].to_list() == [1, 0, 3, 0, 5]


def test_fill_null_with_expr():
    df = _df_nulls()
    out = df.select(col("a").fill_null(col("c")).alias("filled")._e)
    assert out["filled"].to_list() == [1, 20, 3, 40, 5]


def test_is_null_is_not_null():
    df = _df_nulls()
    out = df.select(
        col("a").is_null().alias("is_null")._e,
        col("a").is_not_null().alias("is_not_null")._e,
    )
    assert out["is_null"].to_list() == [False, True, False, True, False]
    assert out["is_not_null"].to_list() == [True, False, True, False, True]


def test_is_nan_is_not_nan():
    df = _df_floats()
    out = df.select(
        col("y").is_nan().alias("is_nan")._e,
        col("y").is_not_nan().alias("is_not_nan")._e,
    )
    assert out["is_nan"].to_list() == [False, False, False, True, False]
    assert out["is_not_nan"].to_list() == [True, True, True, False, True]


def test_is_finite_is_infinite():
    df = _df_floats()
    out = df.select(
        col("y").is_finite().alias("is_finite")._e,
        col("y").is_infinite().alias("is_infinite")._e,
    )
    assert out["is_finite"].to_list() == [True, True, True, False, False]
    assert out["is_infinite"].to_list() == [False, False, False, False, True]


def test_fill_nan():
    df = _df_floats()
    out = df.select(col("y").fill_nan(999.0).alias("filled")._e)
    result = out["filled"].to_list()
    assert result[3] == 999.0


def test_drop_nulls():
    df = _df_nulls()
    out = df.select(col("a").drop_nulls().alias("dropped")._e)
    assert out["dropped"].to_list() == [1, 3, 5]


# =============================================================================
# Mathematical Functions
# =============================================================================


def test_math_sqrt_cbrt():
    df = pl.DataFrame({"x": [1.0, 4.0, 9.0, 27.0]})
    out = df.select(
        col("x").sqrt().alias("sqrt")._e,
        col("x").cbrt().alias("cbrt")._e,
    )
    assert out["sqrt"].to_list() == [1.0, 2.0, 3.0, pytest.approx(5.196, rel=0.01)]
    assert out["cbrt"].to_list() == [
        1.0,
        pytest.approx(1.587, rel=0.01),
        pytest.approx(2.08, rel=0.01),
        3.0,
    ]


def test_math_log_exp():
    df = pl.DataFrame({"x": [1.0, math.e, math.e**2]})
    out = df.select(
        col("x").log().alias("log")._e,
        col("x").exp().alias("exp")._e,
    )
    assert out["log"].to_list() == pytest.approx([0.0, 1.0, 2.0], rel=0.01)


def test_math_log10_log2():
    df = pl.DataFrame({"x": [1.0, 10.0, 100.0, 2.0, 8.0]})
    out = df.select(
        col("x").log10().alias("log10")._e,
    )
    assert out["log10"].to_list()[:3] == pytest.approx([0.0, 1.0, 2.0])


def test_math_round_floor_ceil():
    df = pl.DataFrame({"x": [1.2, 1.5, 1.8, -1.2, -1.8]})
    out = df.select(
        col("x").round().alias("round")._e,
        col("x").floor().alias("floor")._e,
        col("x").ceil().alias("ceil")._e,
    )
    assert out["round"].to_list() == [1.0, 2.0, 2.0, -1.0, -2.0]
    assert out["floor"].to_list() == [1.0, 1.0, 1.0, -2.0, -2.0]
    assert out["ceil"].to_list() == [2.0, 2.0, 2.0, -1.0, -1.0]


def test_math_sign():
    df = pl.DataFrame({"x": [-5, -1, 0, 1, 5]})
    out = df.select(col("x").sign().alias("sign")._e)
    assert out["sign"].to_list() == [-1, -1, 0, 1, 1]


def test_math_clip():
    df = pl.DataFrame({"x": [1, 5, 10, 15, 20]})
    out = df.select(col("x").clip(5, 15).alias("clipped")._e)
    assert out["clipped"].to_list() == [5, 5, 10, 15, 15]


def test_math_pow_method():
    df = pl.DataFrame({"x": [1, 2, 3, 4]})
    out = df.select(col("x").pow(2).alias("squared")._e)
    assert out["squared"].to_list() == [1.0, 4.0, 9.0, 16.0]


# =============================================================================
# Trigonometric Functions
# =============================================================================


def test_trig_sin_cos_tan():
    df = pl.DataFrame({"x": [0.0, math.pi / 2, math.pi]})
    out = df.select(
        col("x").sin().alias("sin")._e,
        col("x").cos().alias("cos")._e,
    )
    assert out["sin"].to_list() == pytest.approx([0.0, 1.0, 0.0], abs=1e-10)
    assert out["cos"].to_list() == pytest.approx([1.0, 0.0, -1.0], abs=1e-10)


def test_trig_radians_degrees():
    df = pl.DataFrame({"deg": [0.0, 90.0, 180.0], "rad": [0.0, math.pi / 2, math.pi]})
    out = df.select(
        col("deg").radians().alias("to_rad")._e,
        col("rad").degrees().alias("to_deg")._e,
    )
    assert out["to_rad"].to_list() == pytest.approx([0.0, math.pi / 2, math.pi], rel=0.01)
    assert out["to_deg"].to_list() == pytest.approx([0.0, 90.0, 180.0], rel=0.01)


# =============================================================================
# Type Casting
# =============================================================================


def test_cast():
    df = pl.DataFrame({"x": [1, 2, 3]})
    out = df.select(col("x").cast(pl.Float64).alias("float")._e)
    assert out["float"].dtype == pl.Float64


def test_to_float_to_int_to_str():
    df = pl.DataFrame({"x": [1, 2, 3]})
    out = df.select(
        col("x").to_float().alias("float")._e,
        col("x").to_str().alias("str")._e,
    )
    assert out["float"].dtype == pl.Float64
    assert out["str"].dtype == pl.Utf8
    assert out["str"].to_list() == ["1", "2", "3"]


def test_to_bool():
    df = pl.DataFrame({"x": [0, 1, 2]})
    out = df.select(col("x").to_bool().alias("bool")._e)
    assert out["bool"].to_list() == [False, True, True]


# =============================================================================
# Membership / Range Checks
# =============================================================================


def test_is_in_and_between():
    df = _df_basic()
    out_in = df.select(col("g").is_in(["a", "c"]).alias("in_ac")._e)
    assert out_in["in_ac"].to_list() == [True, False, True, True, False]

    out_between = df.select(col("x").between(1, 3).alias("between_1_3")._e)
    assert out_between["between_1_3"].to_list() == [False, True, True, True, False]


def test_is_not_in():
    df = _df_basic()
    out = df.select(col("g").is_not_in(["a", "c"]).alias("not_in")._e)
    assert out["not_in"].to_list() == [False, True, False, False, True]


def test_isin_alias():
    df = _df_basic()
    out = df.select(col("g").isin(["a"]).alias("isin")._e)
    assert out["isin"].to_list() == [True, False, False, True, False]


# =============================================================================
# Aggregations
# =============================================================================


def test_aggregations_basic():
    df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
    assert df.select(col("x").sum()._e).item() == 15
    assert df.select(col("x").mean()._e).item() == 3.0
    assert df.select(col("x").median()._e).item() == 3.0
    assert df.select(col("x").min()._e).item() == 1
    assert df.select(col("x").max()._e).item() == 5
    assert df.select(col("x").count()._e).item() == 5
    assert df.select(col("x").first()._e).item() == 1
    assert df.select(col("x").last()._e).item() == 5


def test_aggregations_std_var():
    df = pl.DataFrame({"x": [2, 4, 4, 4, 5, 5, 7, 9]})
    std = df.select(col("x").std()._e).item()
    var = df.select(col("x").var()._e).item()
    assert var == pytest.approx(std**2, rel=0.01)


def test_aggregations_n_unique_unique():
    df = pl.DataFrame({"x": [1, 2, 2, 3, 3, 3]})
    assert df.select(col("x").n_unique()._e).item() == 3
    assert sorted(df.select(col("x").unique()._e).to_series().to_list()) == [1, 2, 3]


def test_aggregations_null_count():
    df = _df_nulls()
    assert df.select(col("a").null_count()._e).item() == 2


def test_aggregations_quantile():
    df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
    q50 = df.select(col("x").quantile(0.5)._e).item()
    assert q50 == 3.0


def test_aggregations_product():
    df = pl.DataFrame({"x": [1, 2, 3, 4]})
    assert df.select(col("x").product()._e).item() == 24


def test_aggregations_arg_min_arg_max():
    df = pl.DataFrame({"x": [3, 1, 4, 1, 5]})
    assert df.select(col("x").arg_min()._e).item() == 1
    assert df.select(col("x").arg_max()._e).item() == 4


# =============================================================================
# Window / Grouping Functions
# =============================================================================


def test_over():
    df = pl.DataFrame({"g": ["a", "a", "b", "b"], "x": [1, 2, 3, 4]})
    out = df.with_columns(col("x").sum().over("g").alias("group_sum")._e)
    assert out["group_sum"].to_list() == [3, 3, 7, 7]


def test_cum_sum_cum_max_cum_min():
    df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
    out = df.select(
        col("x").cum_sum().alias("cum_sum")._e,
        col("x").cum_max().alias("cum_max")._e,
        col("x").cum_min().alias("cum_min")._e,
    )
    assert out["cum_sum"].to_list() == [1, 3, 6, 10, 15]
    assert out["cum_max"].to_list() == [1, 2, 3, 4, 5]
    assert out["cum_min"].to_list() == [1, 1, 1, 1, 1]


def test_shift():
    df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
    out = df.select(col("x").shift(2).alias("shifted")._e)
    assert out["shifted"].to_list() == [None, None, 1, 2, 3]


def test_diff():
    df = pl.DataFrame({"x": [1, 3, 6, 10]})
    out = df.select(col("x").diff().alias("diff")._e)
    assert out["diff"].to_list() == [None, 2, 3, 4]


def test_pct_change():
    df = pl.DataFrame({"x": [100.0, 110.0, 121.0]})
    out = df.select(col("x").pct_change().alias("pct")._e)
    assert out["pct"].to_list()[1:] == pytest.approx([0.1, 0.1], rel=0.01)


def test_rank():
    df = pl.DataFrame({"x": [3, 1, 4, 1, 5]})
    out = df.select(col("x").rank().alias("rank")._e)
    assert out["rank"].to_list() == [3, 1, 4, 2, 5]


def test_rolling_mean():
    df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
    out = df.select(col("x").rolling_mean(3).alias("rolling")._e)
    assert out["rolling"].to_list() == [None, None, 2.0, 3.0, 4.0]


# =============================================================================
# String Operations
# =============================================================================


def test_string_lower_upper_title():
    df = pl.DataFrame({"s": ["Hello World", "PYTHON", "test"]})
    out = df.select(
        col("s").lower().alias("lower")._e,
        col("s").upper().alias("upper")._e,
        col("s").title().alias("title")._e,
    )
    assert out["lower"].to_list() == ["hello world", "python", "test"]
    assert out["upper"].to_list() == ["HELLO WORLD", "PYTHON", "TEST"]
    assert out["title"].to_list() == ["Hello World", "Python", "Test"]


def test_string_strip():
    df = pl.DataFrame({"s": ["  hello  ", "world   ", "   test"]})
    out = df.select(
        col("s").strip().alias("strip")._e,
        col("s").lstrip().alias("lstrip")._e,
        col("s").rstrip().alias("rstrip")._e,
    )
    assert out["strip"].to_list() == ["hello", "world", "test"]
    assert out["lstrip"].to_list() == ["hello  ", "world   ", "test"]
    assert out["rstrip"].to_list() == ["  hello", "world", "   test"]


def test_string_contains_startswith_endswith():
    df = _df_basic()
    out = df.select(
        col("s").contains("alp").alias("contains_alp")._e,
        col("s").startswith("al").alias("startswith_al")._e,
        col("s").endswith("et").alias("endswith_et")._e,
    )
    assert out["contains_alp"].to_list() == [True, False, False, True, False]
    assert out["startswith_al"].to_list() == [True, False, False, True, False]
    assert out["endswith_et"].to_list() == [False, False, False, True, True]


def test_string_replace():
    df = pl.DataFrame({"s": ["hello world", "world world"]})
    out = df.select(
        col("s").replace("world", "python").alias("replace_first")._e,
        col("s").replace_all("world", "python").alias("replace_all")._e,
    )
    assert out["replace_first"].to_list() == ["hello python", "python world"]
    assert out["replace_all"].to_list() == ["hello python", "python python"]


def test_string_slice_head_tail():
    df = pl.DataFrame({"s": ["abcdefgh"]})
    out = df.select(
        col("s").slice(2, 3).alias("slice")._e,
        col("s").head(3).alias("head")._e,
        col("s").tail(3).alias("tail")._e,
    )
    assert out["slice"].to_list() == ["cde"]
    assert out["head"].to_list() == ["abc"]
    assert out["tail"].to_list() == ["fgh"]


def test_string_len():
    df = pl.DataFrame({"s": ["a", "abc", "abcde"]})
    out = df.select(col("s").str_len().alias("len")._e)
    assert out["len"].to_list() == [1, 3, 5]


def test_string_split():
    df = pl.DataFrame({"s": ["a,b,c", "d,e", "f"]})
    out = df.select(col("s").split(",").alias("split")._e)
    assert out["split"].to_list() == [["a", "b", "c"], ["d", "e"], ["f"]]


def test_string_pad():
    df = pl.DataFrame({"s": ["a", "ab", "abc"]})
    out = df.select(
        col("s").pad_left(5, "x").alias("pad_left")._e,
        col("s").pad_right(5, "x").alias("pad_right")._e,
    )
    assert out["pad_left"].to_list() == ["xxxxa", "xxxab", "xxabc"]
    assert out["pad_right"].to_list() == ["axxxx", "abxxx", "abcxx"]


def test_string_extract():
    df = pl.DataFrame({"s": ["abc123", "def456", "ghi"]})
    out = df.select(col("s").extract(r"(\d+)", 1).alias("nums")._e)
    assert out["nums"].to_list() == ["123", "456", None]


def test_string_count_matches():
    df = pl.DataFrame({"s": ["ababa", "bbb", "ccc"]})
    out = df.select(col("s").count_matches("b").alias("count")._e)
    assert out["count"].to_list() == [2, 3, 0]


# =============================================================================
# Date/Time Operations
# =============================================================================


def test_dt_components():
    df = _df_dates()
    out = df.select(
        col("dt").dt_year().alias("year")._e,
        col("dt").dt_month().alias("month")._e,
        col("dt").dt_day().alias("day")._e,
        col("dt").dt_hour().alias("hour")._e,
        col("dt").dt_minute().alias("minute")._e,
        col("dt").dt_second().alias("second")._e,
    )
    assert out["year"].to_list() == [2023, 2023, 2024]
    assert out["month"].to_list() == [1, 6, 12]
    assert out["day"].to_list() == [15, 20, 31]
    assert out["hour"].to_list() == [10, 14, 23]
    assert out["minute"].to_list() == [30, 0, 59]
    assert out["second"].to_list() == [45, 0, 59]


def test_dt_weekday_quarter():
    df = _df_dates()
    out = df.select(
        col("dt").dt_weekday().alias("weekday")._e,
        col("dt").dt_quarter().alias("quarter")._e,
    )
    # Polars weekday: Monday=1, Sunday=7
    # 2023-01-15 is Sunday (7), 2023-06-20 is Tuesday (2), 2024-12-31 is Tuesday (2)
    assert out["weekday"].to_list() == [7, 2, 2]
    assert out["quarter"].to_list() == [1, 2, 4]


def test_dt_strftime():
    df = _df_dates()
    out = df.select(col("dt").dt_strftime("%Y-%m-%d").alias("formatted")._e)
    assert out["formatted"].to_list() == ["2023-01-15", "2023-06-20", "2024-12-31"]


# =============================================================================
# List Operations
# =============================================================================


def test_list_len():
    df = _df_lists()
    out = df.select(col("nums").list_len().alias("len")._e)
    assert out["len"].to_list() == [3, 2, 4]


def test_list_get_first_last():
    df = _df_lists()
    out = df.select(
        col("nums").list_first().alias("first")._e,
        col("nums").list_last().alias("last")._e,
        col("nums").list_get(1).alias("second")._e,
    )
    assert out["first"].to_list() == [1, 4, 6]
    assert out["last"].to_list() == [3, 5, 9]
    assert out["second"].to_list() == [2, 5, 7]


def test_list_sum_mean():
    df = _df_lists()
    out = df.select(
        col("nums").list_sum().alias("sum")._e,
        col("nums").list_mean().alias("mean")._e,
    )
    assert out["sum"].to_list() == [6, 9, 30]
    assert out["mean"].to_list() == [2.0, 4.5, 7.5]


def test_list_contains():
    df = _df_lists()
    out = df.select(col("nums").list_contains(5).alias("has_5")._e)
    assert out["has_5"].to_list() == [False, True, False]


def test_list_join():
    df = _df_lists()
    out = df.select(col("strs").list_join("-").alias("joined")._e)
    assert out["joined"].to_list() == ["a-b", "c", "d-e-f"]


def test_list_sort_reverse():
    df = pl.DataFrame({"nums": [[3, 1, 2], [6, 4, 5]]})
    out = df.select(
        col("nums").list_sort().alias("sorted")._e,
        col("nums").list_reverse().alias("reversed")._e,
    )
    assert out["sorted"].to_list() == [[1, 2, 3], [4, 5, 6]]
    assert out["reversed"].to_list() == [[2, 1, 3], [5, 4, 6]]


def test_explode():
    df = pl.DataFrame({"nums": [[1, 2], [3, 4, 5]]})
    out = df.select(col("nums").explode()._e)
    assert out["nums"].to_list() == [1, 2, 3, 4, 5]


# =============================================================================
# Sorting
# =============================================================================


def test_sort():
    df = pl.DataFrame({"x": [3, 1, 4, 1, 5]})
    out = df.select(
        col("x").sort().alias("asc")._e,
        col("x").sort(descending=True).alias("desc")._e,
    )
    assert out["asc"].to_list() == [1, 1, 3, 4, 5]
    assert out["desc"].to_list() == [5, 4, 3, 1, 1]


def test_reverse():
    df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
    out = df.select(col("x").reverse().alias("rev")._e)
    assert out["rev"].to_list() == [5, 4, 3, 2, 1]


def test_arg_sort():
    df = pl.DataFrame({"x": [3, 1, 4]})
    out = df.select(col("x").arg_sort().alias("idx")._e)
    assert out["idx"].to_list() == [1, 0, 2]


def test_sample():
    df = pl.DataFrame({"x": list(range(100))})
    out = df.select(col("x").sample(n=5, seed=42).alias("sampled")._e)
    assert len(out["sampled"]) == 5


# =============================================================================
# when/then/otherwise
# =============================================================================


def test_when_then_otherwise_basic():
    df = _df_basic()
    expr = when(col("x") > 2).then(lit(100)).otherwise(lit(-1))
    out = df.select(expr.alias("flag")._e)
    assert out["flag"].to_list() == [-1, -1, -1, 100, 100]


def test_when_then_chained():
    df = pl.DataFrame({"age": [15, 25, 45, 70]})
    expr = (
        when(col("age") < 18).then("child").when(col("age") < 65).then("adult").otherwise("senior")
    )
    out = df.select(expr.alias("group")._e)
    assert out["group"].to_list() == ["child", "adult", "adult", "senior"]


def test_when_then_multiple_chains():
    df = pl.DataFrame({"score": [0, 50, 70, 85, 95]})
    expr = (
        when(col("score") >= 90)
        .then("A")
        .when(col("score") >= 80)
        .then("B")
        .when(col("score") >= 70)
        .then("C")
        .when(col("score") >= 60)
        .then("D")
        .otherwise("F")
    )
    out = df.select(expr.alias("grade")._e)
    assert out["grade"].to_list() == ["F", "F", "C", "B", "A"]


# =============================================================================
# Public Constructors
# =============================================================================


def test_cols():
    result = cols("a", "b", "c")
    assert len(result) == 3
    assert all(isinstance(e, Expr) for e in result)


def test_lit():
    df = pl.DataFrame({"x": [1, 2, 3]})
    out = df.with_columns(lit(100).alias("const")._e)
    assert out["const"].to_list() == [100, 100, 100]


def test_coalesce():
    df = _df_nulls()
    out = df.select(coalesce(col("a"), col("b"), col("c")).alias("result")._e)
    assert out["result"].to_list() == [1, 2, 3, 4, 5]


def test_concat_str():
    df = pl.DataFrame({"first": ["John", "Jane"], "last": ["Doe", "Smith"]})
    out = df.select(concat_str(col("first"), lit(" "), col("last")).alias("full")._e)
    assert out["full"].to_list() == ["John Doe", "Jane Smith"]


# =============================================================================
# Horizontal Operations
# =============================================================================


def test_sum_horizontal():
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    out = df.select(sum_horizontal(col("a"), col("b"), col("c")).alias("sum")._e)
    assert out["sum"].to_list() == [9, 12]


def test_min_max_horizontal():
    df = pl.DataFrame({"a": [1, 5], "b": [3, 2], "c": [2, 8]})
    out = df.select(
        min_horizontal(col("a"), col("b"), col("c")).alias("min")._e,
        max_horizontal(col("a"), col("b"), col("c")).alias("max")._e,
    )
    assert out["min"].to_list() == [1, 2]
    assert out["max"].to_list() == [3, 8]


def test_all_any_horizontal():
    df = pl.DataFrame({"a": [True, True, False], "b": [True, False, False]})
    out = df.select(
        all_horizontal(col("a"), col("b")).alias("all")._e,
        any_horizontal(col("a"), col("b")).alias("any")._e,
    )
    assert out["all"].to_list() == [True, False, False]
    assert out["any"].to_list() == [True, True, False]


# =============================================================================
# Struct
# =============================================================================


def test_struct():
    df = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    out = df.select(struct(col("a"), col("b")).alias("s")._e)
    assert out["s"].to_list() == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]


def test_struct_with_names():
    df = pl.DataFrame({"a": [1, 2]})
    out = df.select(struct(x=col("a"), y=lit(100)).alias("s")._e)
    assert out["s"].to_list() == [{"x": 1, "y": 100}, {"x": 2, "y": 100}]


# =============================================================================
# Mixing with Native Polars Expressions
# =============================================================================


def test_mix_expr_and_polars_expr():
    df = _df_basic()
    mixed = col("x") + pl.col("y")
    out = df.select(mixed.alias("sum")._e)
    assert out["sum"].to_list() == [x + y for x, y in zip(df["x"], df["y"])]


def test_when_accepts_polars_expr():
    df = pl.DataFrame({"x": [1, 2, 3]})
    expr = when(pl.col("x") > 1).then("big").otherwise("small")
    out = df.select(expr.alias("size")._e)
    assert out["size"].to_list() == ["small", "big", "big"]
