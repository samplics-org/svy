# tests/svy/wrangling/test_input_errors.py
"""Bad input to sample.wrangling raises DataError, also the builtin it replaces."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import svy

from svy.errors import Code, DataError, SvyError


N = 12


@pytest.fixture
def wr():
    df = pl.DataFrame(
        {
            "id": np.arange(N),
            "y": np.arange(N, dtype=float),
            "g": np.tile(["a", "b"], N // 2),
            "txt": ["t"] * N,
            "yn": [None, *np.arange(1.0, N)],
            "w": np.ones(N),
        }
    )
    return svy.Sample(df, svy.Design(wgt="w")).wrangling


def _raises(kind, code, call, *, param=None, says=None):
    with pytest.raises(DataError) as err:
        call()
    e = err.value
    assert type(e) is getattr(DataError, kind.__name__)
    assert isinstance(e, kind) and isinstance(e, SvyError)
    assert e.code == code
    if param is not None:
        assert e.param == param
    if says is not None:
        assert says in e.detail
    return e


@pytest.mark.parametrize(
    "call, param",
    [
        (lambda w: w.rename_columns({"nope": "x"}), "renames"),
        (lambda w: w.remove_columns("nope"), "columns"),
        (lambda w: w.keep_columns(["y", "nope"]), "columns"),
        (lambda w: w.top_code({"nope": 1.0}), "top_codes"),
        (lambda w: w.bottom_code({"nope": 1.0}), "bottom_codes"),
        (lambda w: w.bottom_and_top_code({"nope": (0.0, 1.0)}), "bottom_and_top_codes"),
        (lambda w: w.recode("nope", {1: [0]}), "cols"),
        (lambda w: w.categorize("nope", bins=[0, 5, 20]), "col"),
        (lambda w: w.cast("nope", "Int64"), "cols"),
        (lambda w: w.fill_null("nope", 0), "cols"),
        (lambda w: w.order_by("nope"), "cols"),
        (lambda w: w.distinct("nope"), "cols"),
    ],
    ids=[
        "rename",
        "remove",
        "keep",
        "top_code",
        "bottom_code",
        "bottom_and_top_code",
        "recode",
        "categorize",
        "cast",
        "fill_null",
        "order_by",
        "distinct",
    ],
)
def test_missing_column_names_the_parameter(wr, call, param):
    e = _raises(KeyError, Code.MISSING_COLUMNS, lambda: call(wr), param=param)
    assert e.got == ["nope"]
    assert not [c for c in e.extra["available_preview"] if c.startswith("__svy_")]


@pytest.mark.parametrize(
    "call, param",
    [
        (lambda w: w.top_code({"txt": 1.0}), "top_codes"),
        (lambda w: w.categorize("txt", bins=[0, 5, 20]), "col"),
    ],
    ids=["top_code", "categorize"],
)
def test_text_column_where_a_number_is_needed(wr, call, param):
    e = _raises(TypeError, Code.COLUMN_NOT_NUMERIC, lambda: call(wr), param=param)
    assert e.got == "String"


@pytest.mark.parametrize(
    "call, param",
    [
        (lambda w: w.top_code({"y": "a"}), "top_codes"),
        (lambda w: w.bottom_code({"y": None}), "bottom_codes"),
        (lambda w: w.bottom_and_top_code({"y": (0.0, "z")}), "bottom_and_top_codes"),
        (lambda w: w.bottom_and_top_code({"y": 1.0}), "bottom_and_top_codes"),
        (lambda w: w.top_code(["y"]), "top_codes"),
        (lambda w: w.rename_columns("y"), "renames"),
        (lambda w: w.mutate("z"), "specs"),
    ],
    ids=[
        "top bound str",
        "bottom bound None",
        "pair with str",
        "pair not a pair",
        "top_code list",
        "rename str",
        "mutate str",
    ],
)
def test_argument_of_wrong_type(wr, call, param):
    _raises(TypeError, Code.INVALID_TYPE, lambda: call(wr), param=param)


@pytest.mark.parametrize(
    "call, param, allowed",
    [
        (lambda w: w.clean_names(case_style="screaming"), "case_style", None),
        (lambda w: w.clean_names(letter_case="shout"), "letter_case", None),
        (
            lambda w: w.fill_null("yn", strategy="median"),
            "strategy",
            ["forward", "backward", "mean", "min", "max", "zero", "one"],
        ),
        (lambda w: w.distinct(keep="all"), "keep", ["first", "last", "any", "none"]),
    ],
    ids=["case_style", "letter_case", "strategy", "keep"],
)
def test_unknown_option(wr, call, param, allowed):
    e = _raises(ValueError, Code.INVALID_CHOICE, lambda: call(wr), param=param)
    if allowed is not None:
        assert e.expected == allowed


@pytest.mark.parametrize("param", ["case_style", "letter_case"])
def test_naming_option_of_wrong_type(wr, param):
    _raises(TypeError, Code.INVALID_TYPE, lambda: wr.clean_names(**{param: 3}), param=param)


def test_unsorted_bins(wr):
    _raises(
        ValueError,
        Code.INVALID_VALUE,
        lambda: wr.categorize("y", bins=[20, 5, 0]),
        param="bins",
        says="strictly increasing",
    )


def test_sorted_bins_label_every_row(wr):
    out = wr.categorize("y", bins=[0, 5, 11], into="c").data["c"]
    assert out.null_count() == 0 and out.n_unique() == 2


@pytest.mark.parametrize("percentiles", [0, 1, True], ids=["zero", "one", "bool"])
def test_too_few_percentile_groups(wr, percentiles):
    _raises(
        ValueError,
        Code.INVALID_RANGE,
        lambda: wr.categorize("y", percentiles=percentiles),
        param="percentiles",
    )


def test_fill_null_value_and_strategy(wr):
    _raises(
        ValueError,
        Code.INVALID_VALUE,
        lambda: wr.fill_null("yn", 9.0, strategy="zero"),
        param="strategy",
        says="not both",
    )


def test_fill_null_value_alone_still_fills(wr):
    assert wr.fill_null("yn", 9.0).data["yn"][0] == 9.0


def test_order_by_descending_length(wr):
    _raises(
        ValueError,
        Code.INVALID_VALUE,
        lambda: wr.order_by(["y", "g"], descending=[True]),
        param="descending",
        says="1 entries for 2 columns",
    )


def test_row_index_name_taken(wr):
    _raises(
        ValueError,
        Code.INVALID_VALUE,
        lambda: wr.with_row_index("y"),
        param="name",
        says="'y' already exists",
    )


def test_strict_cast_that_fails(wr):
    _raises(
        ValueError,
        Code.CAST_FAILED,
        lambda: wr.cast("g", "Int64"),
        param="cols",
        says="column 'g'",
    )


def test_lenient_cast_gives_nulls(wr):
    assert wr.cast("g", "Int64", strict=False).data["g"].null_count() == N
