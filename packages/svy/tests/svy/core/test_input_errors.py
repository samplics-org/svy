# tests/svy/core/test_input_errors.py
"""Bad input to Design, Sample, replicate weights and expressions raises svy errors."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import svy

from svy.errors import (
    Code,
    DataError,
    DesignError,
    SingletonError,
    SvyError,
    UsageError,
    WeightingError,
)


N = 20


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "y": np.arange(N, dtype=float),
            "s": np.repeat(["a", "b"], N // 2),
            "p": np.arange(N) // 2,
            "w": np.ones(N),
            "txt": ["t"] * N,
            "r1": np.ones(N),
            "r2": np.ones(N),
        }
    )


def _raises(area, kind, code, call, *, param=None, says=None):
    with pytest.raises(area) as err:
        call()
    e = err.value
    assert type(e) is getattr(area, kind.__name__)
    assert isinstance(e, kind) and isinstance(e, SvyError)
    assert e.code == code
    if param is not None:
        assert e.param == param
    if says is not None:
        assert says in e.detail
    return e


# ---- Design ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kw, param, kind, code",
    [
        ({"stratum": 3}, "stratum", TypeError, Code.INVALID_TYPE),
        ({"psu": ["ea", 2]}, "psu", TypeError, Code.INVALID_TYPE),
        ({"stratum": ""}, "stratum", ValueError, Code.INVALID_VALUE),
        ({"stratum": []}, "stratum", ValueError, Code.INVALID_VALUE),
        ({"wgt": ["w"]}, "wgt", TypeError, Code.INVALID_TYPE),
        ({"wgt": ""}, "wgt", ValueError, Code.INVALID_VALUE),
        ({"wr": "yes"}, "wr", TypeError, Code.INVALID_TYPE),
        ({"pop_size": -1}, "pop_size", TypeError, Code.INVALID_TYPE),
        ({"pop_size": ""}, "pop_size", ValueError, Code.INVALID_VALUE),
        ({"rep_wgts": "r"}, "rep_wgts", TypeError, Code.INVALID_TYPE),
    ],
    ids=[
        "stratum int",
        "psu mixed",
        "stratum empty",
        "stratum empty list",
        "wgt list",
        "wgt empty",
        "wr str",
        "pop_size int",
        "pop_size empty",
        "rep_wgts str",
    ],
)
def test_design_field(kw, param, kind, code):
    _raises(DesignError, kind, code, lambda: svy.Design(**kw), param=param)


def test_design_update_checks_fields_too():
    _raises(
        DesignError,
        ValueError,
        Code.INVALID_VALUE,
        lambda: svy.Design(wgt="w").update(stratum=[]),
        param="stratum",
    )


def test_design_is_frozen():
    d = svy.Design(wgt="w")
    _raises(
        DesignError,
        AttributeError,
        Code.DESIGN_FROZEN,
        lambda: setattr(d, "wgt", "x"),
        param="wgt",
        says="Design is frozen",
    )
    _raises(DesignError, AttributeError, Code.DESIGN_FROZEN, lambda: delattr(d, "wgt"))
    assert d.wgt == "w"


def test_replicates_for_another_weight():
    e = _raises(
        DesignError,
        ValueError,
        Code.REP_WGTS_WGT_MISMATCH,
        lambda: svy.Design(wgt="w", rep_wgts=svy.BrrWgts(prefix="r", n_reps=2, wgt="w0")),
        param="rep_wgts",
    )
    assert (e.expected, e.got) == ("w", "w0")


def test_singleton_rule_without_strata():
    _raises(
        DesignError,
        ValueError,
        Code.SINGLETON_WITHOUT_STRATUM,
        lambda: svy.Design(psu="p", singleton="center"),
        param="singleton",
    )


# ---- replicate weights -----------------------------------------------------------------


@pytest.mark.parametrize("cls", [svy.BrrWgts, svy.JackknifeWgts, svy.BootstrapWgts, svy.SdrWgts])
@pytest.mark.parametrize(
    "kw, param, kind, code",
    [
        ({"prefix": "r", "n_reps": 1}, "n_reps", ValueError, Code.INVALID_RANGE),
        ({"prefix": "r", "n_reps": -1}, "n_reps", ValueError, Code.INVALID_RANGE),
        ({"prefix": "r", "n_reps": "2"}, "n_reps", TypeError, Code.INVALID_TYPE),
        ({"prefix": "r", "n_reps": True}, "n_reps", TypeError, Code.INVALID_TYPE),
        ({"prefix": " ", "n_reps": 2}, "prefix", ValueError, Code.INVALID_VALUE),
        ({"prefix": "r", "n_reps": 2, "df": 0}, "df", ValueError, Code.INVALID_RANGE),
        ({"prefix": "r", "n_reps": 2, "psu": ""}, "psu", ValueError, Code.INVALID_VALUE),
        ({"prefix": "r", "n_reps": 2, "scale": [1.0]}, "scale", ValueError, Code.INVALID_VALUE),
        ({"prefix": "r", "n_reps": 2, "scale": "x"}, "scale", TypeError, Code.INVALID_TYPE),
    ],
    ids=[
        "n_reps 1",
        "n_reps negative",
        "n_reps str",
        "n_reps bool",
        "prefix blank",
        "df zero",
        "psu empty",
        "scale length",
        "scale str",
    ],
)
def test_replicate_weight_spec(cls, kw, param, kind, code):
    _raises(DesignError, kind, code, lambda: cls(**kw), param=param)


def test_two_replicates_is_the_minimum():
    assert svy.BrrWgts(prefix="r", n_reps=2).n_reps == 2


def test_negative_fay_coefficient():
    _raises(
        DesignError,
        ValueError,
        Code.INVALID_RANGE,
        lambda: svy.BrrWgts(prefix="r", n_reps=2, fay_coef=-0.1),
        param="fay_coef",
    )


@pytest.mark.parametrize(
    "method, says",
    [("taylor", "is not a valid replication method"), ("bss", "Unknown replication method")],
)
def test_unknown_replication_method(method, says):
    _raises(
        DesignError,
        ValueError,
        Code.INVALID_CHOICE,
        lambda: svy.RepWeights(method=method, prefix="r", n_reps=2),
        param="method",
        says=says,
    )


# ---- Sample: the data against the design ----------------------------------------------


def test_sample_design_of_wrong_type(df):
    _raises(DesignError, TypeError, Code.INVALID_TYPE, lambda: svy.Sample(df, "w"), param="design")


@pytest.mark.parametrize(
    "design, param",
    [
        (svy.Design(wgt="nope"), "design"),
        (svy.Design(mos="nope"), "design"),
        (svy.Design(wgt="w", rep_wgts=svy.BrrWgts(prefix="q", n_reps=2)), "design"),
    ],
    ids=["wgt", "mos", "replicates"],
)
def test_sample_missing_design_column(df, design, param):
    e = _raises(
        DataError,
        ValueError,
        Code.MISSING_COLUMNS,
        lambda: svy.Sample(df, design),
        param=param,
        says="Design references columns not found in data",
    )
    assert e.got


def test_sample_missing_stratum_column(df):
    e = _raises(
        DataError,
        KeyError,
        Code.MISSING_COLUMNS,
        lambda: svy.Sample(df, svy.Design(stratum="nope")),
        param="stratum",
    )
    assert e.got == ["nope"]
    assert not [c for c in e.extra["available_preview"] if c.startswith("__svy_")]


@pytest.mark.parametrize(
    "design, param, code",
    [
        (svy.Design(wgt="txt"), "wgt", Code.WEIGHT_NOT_NUMERIC),
        (svy.Design(mos="txt"), "mos", Code.COLUMN_NOT_NUMERIC),
        (svy.Design(prob="txt"), "prob", Code.COLUMN_NOT_NUMERIC),
        (svy.Design(pop_size="txt"), "pop_size", Code.COLUMN_NOT_NUMERIC),
    ],
    ids=["wgt", "mos", "prob", "pop_size"],
)
def test_sample_text_column_where_a_number_is_needed(df, design, param, code):
    e = _raises(DataError, TypeError, code, lambda: svy.Sample(df, design), param=param)
    assert e.got == "String"


@pytest.mark.parametrize(
    "rows, says",
    [
        ({"hh": [1, 1], "w": [1.0, 1.0]}, "must be unique across rows"),
        ({"hh": [1, None], "w": [1.0, 1.0]}, "contains nulls in ['hh']"),
    ],
    ids=["duplicated", "nulls"],
)
def test_case_id(rows, says):
    _raises(
        DesignError,
        ValueError,
        Code.CASE_ID_INVALID,
        lambda: svy.Sample(pl.DataFrame(rows), svy.Design(wgt="w", case_id="hh")),
        param="case_id",
        says=says,
    )


def test_case_id_column_missing():
    _raises(
        DataError,
        ValueError,
        Code.MISSING_COLUMNS,
        lambda: svy.Sample(
            pl.DataFrame({"hh": [1, 2], "w": [1.0, 1.0]}), svy.Design(wgt="w", case_id="id")
        ),
        says="not found in data",
    )


def test_sample_print_width(df):
    s = svy.Sample(df, svy.Design(wgt="w"))
    _raises(
        DesignError, ValueError, Code.INVALID_RANGE, lambda: s.set_print_width(20), param="width"
    )
    _raises(
        DesignError,
        TypeError,
        Code.INVALID_TYPE,
        lambda: s.set_print_width("wide"),
        param="width",
    )
    assert s.set_print_width(21) is s


def test_describe_with_no_visible_column(df):
    s = svy.Sample(df, svy.Design(wgt="w"))
    _raises(
        DataError,
        ValueError,
        Code.INVALID_VALUE,
        lambda: s.describe(columns=["nope"]),
        param="columns",
        says="No visible columns",
    )


# ---- combine_samples -------------------------------------------------------------------


def test_combine_samples_unknown_argument(df):
    s = svy.Sample(df, svy.Design(wgt="w"))
    e = _raises(
        UsageError,
        TypeError,
        Code.UNKNOWN_ARGUMENT,
        lambda: svy.combine_samples([s, s], averge_wgts=True),
        says="did you mean average_wgts=?",
    )
    assert e.got == ["averge_wgts"] and "_legacy" not in e.expected


# ---- singleton collapse ----------------------------------------------------------------


@pytest.fixture(scope="module")
def singleton_df() -> pl.DataFrame:
    # Stratum "c" has one PSU; "a" and "b" have two.
    return pl.DataFrame(
        {
            "s": ["a", "a", "b", "b", "c"],
            "p": [1, 2, 3, 4, 5],
            "w": [1.0] * 5,
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


def _collapse(df, using):
    rule = svy.Singleton("collapse", using=using)
    return svy.Sample(
        df, svy.Design(stratum="s", psu="p", wgt="w", singleton=rule)
    ).estimation.mean("y")


@pytest.mark.parametrize(
    "using, code",
    [
        ({"x": "a"}, Code.SINGLETON_UNMAPPED),
        ({"c": "zz"}, Code.SINGLETON_TARGET_MISSING),
        (lambda single, cands: "zz", Code.SINGLETON_TARGET_MISSING),
    ],
    ids=["mapping without the singleton", "mapping to no stratum", "callable"],
)
def test_collapse_target(singleton_df, using, code):
    with pytest.raises(SingletonError) as err:
        _collapse(singleton_df, using)
    assert err.value.code == code


# ---- expressions and thresholds --------------------------------------------------------


@pytest.mark.parametrize(
    "call, param",
    [
        (lambda: svy.col(3), "name"),
        (lambda: svy.lit(object()), "value"),
        (lambda: svy.col("y") + object(), "value"),
        (lambda: svy.when(3), "condition"),
        (lambda: bool(svy.col("y") > 1), "expr"),
    ],
    ids=["col int", "lit object", "add object", "when int", "truth value"],
)
def test_expression_of_wrong_type(call, param):
    _raises(DataError, TypeError, Code.INVALID_TYPE, call, param=param)


def test_incomplete_when_is_named(df):
    s = svy.Sample(df, svy.Design(wgt="w"))
    _raises(
        DataError,
        TypeError,
        Code.INVALID_TYPE,
        lambda: s.estimation.mean("y", where=svy.when(svy.col("y") > 1).then(True)),
        says="Incomplete conditional expression",
    )


def test_coalesce_needs_an_expression():
    _raises(DataError, ValueError, Code.INVALID_VALUE, svy.coalesce, param="exprs")


@pytest.mark.parametrize(
    "call, says",
    [
        (lambda: svy.Cap("mode"), "Unsupported stat 'mode'"),
        (lambda: svy.Cap("median", 0), "k must be nonzero"),
        (lambda: svy.Cap.quantile(99), "For the 99th percentile use 0.99"),
        (lambda: svy.Cap("median", p=0.5), "p applies only to a quantile"),
        (lambda: svy.Cap.absolute(0), "must be > 0, got 0"),
    ],
    ids=["stat", "k zero", "percent", "p on median", "absolute zero"],
)
def test_threshold(call, says):
    _raises(WeightingError, ValueError, Code.THRESHOLD_INVALID, call, says=says)
