# tests/svy/core/test_set_data_lineage.py
"""set_data, update_data and clone(data=) hold a new frame to the weight lineage.

The weights, replicate weights and record columns of the design and its
history must keep their values; rows may change only on a sample without
lineage. A refused frame leaves the sample untouched.
"""

import warnings

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from numpy.testing import assert_allclose

from svy import Design, Sample, col
from svy.errors import MethodError


DATA_DIR = Path(__file__).resolve().parents[2] / "test_data"
STYPE_POP = {"E": 4421.0, "H": 755.0, "M": 1018.0}
CELLS = "__svy_cells_rk_1"


@contextmanager
def no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


@pytest.fixture(scope="module")
def apiclus1():
    return pl.read_csv(DATA_DIR / "apiclus1.csv", null_values=["NA"])


@pytest.fixture
def chain(apiclus1):
    """pw with bootstrap replicates -> rk (scoped rake) -> tr (trim)."""

    def build():
        s = Sample(apiclus1, Design(wgt="pw", stratum="stype", psu="dnum", pop_size="fpc"))
        s = s.weighting.create_bs_wgts(n_reps=6, rstate=np.random.default_rng(3))
        s = s.weighting.rake(
            controls={"stype": STYPE_POP}, wgt_name="rk", where=col("api00") > 500
        )
        return s.weighting.trim(upper=40.0, wgt_name="tr")

    return build


ROLES = {
    "current weight": ("tr", "'tr' is the design's weight"),
    "prev_wgt of the current record": ("rk", "'rk' is the weight 'tr' was adjusted from"),
    "history weight": ("pw", "'pw' is the weight 'rk' was adjusted from"),
    "cells snapshot": (CELLS, "cell snapshot the raking of 'rk' reads"),
    "current replicate": ("tr3", "'tr3' is a replicate weight of 'tr'"),
    "history replicate": ("pw3", "'pw3' is a replicate weight of 'pw'"),
}

SETTERS = {
    "set_data": lambda s, df: s.set_data(df),
    "update_data": lambda s, df: s.update_data(df),
    "clone": lambda s, df: s.clone(data=df),
}


def _state(s):
    return (
        s._data,
        s._design,
        s._design_history,
        s._data_version,
        sorted(s._metadata),
        len(s._warnings.list()),
    )


def _assert_untouched(s, state):
    data, design, history, version, meta, n_warn = state
    assert s._data is data
    assert s._design == design and s._design_history == history
    assert s._data_version == version
    assert sorted(s._metadata) == meta
    assert len(s._warnings.list()) == n_warn


def _se(s):
    return s.estimation.mean("api00").to_polars()["se"][0]


def _changed(df, column):
    c = pl.col(column)
    expr = pl.when(pl.int_range(pl.len()) == 0).then(c + 1).otherwise(c)
    return df.with_columns(expr.cast(df.schema[column]).alias(column))


@pytest.mark.parametrize("setter", list(SETTERS))
@pytest.mark.parametrize("role", list(ROLES))
def test_changed_values_are_refused(chain, role, setter):
    s = chain()
    column, phrase = ROLES[role]
    state = _state(s)
    with pytest.raises(MethodError) as ei:
        SETTERS[setter](s, _changed(s.data, column))
    err = ei.value
    assert err.code == "WEIGHT_OVERWRITE"
    assert err.where == setter
    assert phrase in err.detail
    assert f"rename_columns({{'{column}': '{column}_v1'}})" in err.hint
    _assert_untouched(s, state)


@pytest.mark.parametrize("setter", list(SETTERS))
def test_unchanged_values_are_accepted(chain, setter):
    s = chain()
    se = _se(s)
    df = s.data.with_columns(pl.lit(1).alias("unrelated"))
    with no_warnings():
        out = SETTERS[setter](s, df)
        assert_allclose(_se(out), se)
    assert "unrelated" in out.data.columns
    assert out.design == s.design
    assert out.design_history == s.design_history


@pytest.mark.parametrize("setter", ["set_data", "update_data"])
def test_strata_recode_is_accepted(chain, setter):
    s = chain()
    df = s.data.with_columns(
        pl.when(pl.col("stype") == "M").then(pl.lit("H")).otherwise(pl.col("stype")).alias("stype")
    )
    with no_warnings():
        getattr(s, setter)(df)
    assert set(s.data.get_column("stype").unique()) == {"E", "H"}


@pytest.mark.parametrize("setter", list(SETTERS))
def test_widening_cast_is_accepted_narrowing_is_not(chain, setter):
    s = chain()
    assert s.data.schema[CELLS] == pl.Int32
    with no_warnings():
        out = SETTERS[setter](s, s.data.with_columns(pl.col(CELLS).cast(pl.Int64)))
    assert out.data.schema[CELLS] == pl.Int64
    s = chain()
    with pytest.raises(MethodError, match="WEIGHT_OVERWRITE"):
        SETTERS[setter](s, s.data.with_columns(pl.col("tr").cast(pl.Float32)))


def test_float32_weight_widened_is_accepted(apiclus1):
    df = apiclus1.with_columns(pl.col("pw").cast(pl.Float32))
    s = Sample(df, Design(wgt="pw", psu="dnum"))
    with no_warnings():
        s.set_data(s.data.with_columns(pl.col("pw").cast(pl.Float64)))
    assert s.data.schema["pw"] == pl.Float64


@pytest.mark.parametrize("setter", list(SETTERS))
def test_rows_changed_with_lineage_are_refused(chain, setter):
    s = chain()
    state = _state(s)
    with pytest.raises(MethodError) as ei:
        SETTERS[setter](s, s.data.head(100))
    err = ei.value
    assert err.code == "DATA_ROWS_CHANGED"
    assert "filter_records" in err.hint and "combine_samples" in err.hint
    _assert_untouched(s, state)


@pytest.mark.parametrize(
    "build",
    [
        lambda df: Sample(df, Design(wgt="pw", psu="dnum")).weighting.create_bs_wgts(
            n_reps=4, rstate=np.random.default_rng(1)
        ),
        lambda df: Sample(df, Design(wgt="pw", psu="dnum")).weighting.normalize(),
        lambda df: Sample(df, Design(wgt="pw", psu="dnum")).update_design(psu=None),
    ],
    ids=["replicates", "record", "history"],
)
def test_each_kind_of_lineage_blocks_a_row_change(apiclus1, build):
    s = build(apiclus1)
    with pytest.raises(MethodError, match="DATA_ROWS_CHANGED"):
        s.set_data(s.data.head(100))


@pytest.mark.parametrize("setter", ["set_data", "update_data"])
def test_rows_changed_on_a_plain_design_are_accepted(apiclus1, setter):
    s = Sample(apiclus1, Design(wgt="pw", psu="dnum"))
    with no_warnings():
        getattr(s, setter)(s.data.head(100))
    assert s.n_records == 100


def test_plain_design_weight_is_still_protected(apiclus1):
    s = Sample(apiclus1, Design(wgt="pw", psu="dnum"))
    with pytest.raises(MethodError, match="WEIGHT_OVERWRITE"):
        s.set_data(_changed(s.data, "pw"))


@pytest.mark.parametrize("setter", list(SETTERS))
def test_missing_record_column_is_refused_before_anything_changes(chain, setter):
    s = chain()
    state = _state(s)
    with pytest.raises(ValueError, match="Design references columns not found in data"):
        SETTERS[setter](s, s.data.drop("rk"))
    _assert_untouched(s, state)


def test_dropping_a_history_only_column_is_accepted(chain):
    s = chain()
    with no_warnings():
        s.set_data(s.data.drop(CELLS))
    # The raking snapshot is gone, so the raked design is not restored: the
    # record is dropped and the replicates reset, as for any other weight.
    with pytest.warns(UserWarning, match="rep_wgts="):
        s.update_design(wgt="rk")
    assert s.design.wgt_adjustment is None
    assert s.design.rep_wgts is None


@pytest.mark.parametrize("setter", ["set_data", "update_data"])
def test_a_late_failure_restores_the_sample(chain, setter):
    s = chain()
    state = _state(s)
    with pytest.raises(TypeError, match="Population size column"):
        getattr(s, setter)(s.data.with_columns(pl.col("fpc").cast(pl.String)))
    data, design, history, _, meta, n_warn = state
    assert s._data.equals(data)
    assert s._design == design and s._design_history == history
    assert sorted(s._metadata) == meta and len(s._warnings.list()) == n_warn
    with no_warnings():
        _se(s)


def test_clone_carries_history_and_validates_columns(chain):
    s = chain()
    with no_warnings():
        c = s.clone(data=s.data)
        assert_allclose(_se(c), _se(s))
    assert c.design_history == s.design_history
    with pytest.raises(ValueError, match="not found in data"):
        s.clone(data=s.data.drop("rk"))
