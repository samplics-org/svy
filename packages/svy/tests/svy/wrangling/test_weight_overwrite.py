# tests/svy/wrangling/test_weight_overwrite.py
"""Wrangling does not write new values into a weight column under its name.

Protected: the design's weight, every earlier design's weight, replicate
weights (current and earlier), and the weight-adjustment record's columns
(current and earlier). Exact widening casts are allowed. The way out is a new
name, or a rename first, which the design and its history follow.
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
    """pw (bootstrap replicates) -> rake to rk (scoped, so the cells snapshot
    has nulls) -> trim to tr. Current weight tr; its record reads rk."""

    def build():
        s = Sample(apiclus1, Design(wgt="pw", psu="dnum", pop_size="fpc"))
        s = s.weighting.create_bs_wgts(n_reps=8, rstate=np.random.default_rng(5))
        s = s.weighting.rake(
            controls={"stype": STYPE_POP}, wgt_name="rk", where=col("api00") > 500
        )
        return s.weighting.trim(upper=40.0, wgt_name="tr")

    return build


CELLS = "__svy_cells_rk_1"

# role -> (column, phrase in the detail)
ROLES = {
    "current weight": ("tr", "'tr' is the design's weight"),
    "prev_wgt of the current record": ("rk", "'rk' is the weight 'tr' was adjusted from"),
    "history weight": ("pw", "'pw' is the weight 'rk' was adjusted from"),
    "cells snapshot": (CELLS, "cell snapshot the raking of 'rk' reads"),
    "current replicate": ("tr3", "'tr3' is a replicate weight of 'tr'"),
    "history replicate": ("pw3", "'pw3' is a replicate weight of 'pw'"),
}


def _first(s, c):
    return s.data.get_column(c)[0]


STEPS = {
    "mutate": lambda s, c, ip: s.wrangling.mutate({c: pl.col(c) * 2}, inplace=ip),
    "mutate_same_values": lambda s, c, ip: s.wrangling.mutate({c: pl.col(c)}, inplace=ip),
    "recode_replace": lambda s, c, ip: s.wrangling.recode(
        c, {-1: [_first(s, c)]}, replace=True, inplace=ip
    ),
    "fill_null": lambda s, c, ip: s.wrangling.fill_null(c, 0, inplace=ip),
    "top_code": lambda s, c, ip: s.wrangling.top_code({c: 1.0}, replace=True, inplace=ip),
    "bottom_code": lambda s, c, ip: s.wrangling.bottom_code({c: 1e6}, replace=True, inplace=ip),
    "bottom_and_top_code": lambda s, c, ip: s.wrangling.bottom_and_top_code(
        {c: (0.0, 1.0)}, replace=True, inplace=ip
    ),
    "categorize_replace": lambda s, c, ip: s.wrangling.categorize(
        c, bins=[-1e9, 1.0, 1e9], replace=True, inplace=ip
    ),
    "cast_narrowing": lambda s, c, ip: s.wrangling.cast(c, pl.Float32, inplace=ip),
    "cast_to_string": lambda s, c, ip: s.wrangling.cast(c, pl.String, inplace=ip),
}


def _snapshot(s):
    return (s.data, s.design, s.design_history, s._data_version)


def _assert_untouched(s, snap):
    data, design, history, version = snap
    assert s.data.equals(data)
    assert s.data.schema == data.schema
    assert s.design == design
    assert s.design_history == history
    assert s._data_version == version


@pytest.mark.parametrize("inplace", [False, True], ids=["fork", "inplace"])
@pytest.mark.parametrize("step", list(STEPS))
@pytest.mark.parametrize("role", list(ROLES))
def test_every_writing_step_is_refused_on_every_role(chain, role, step, inplace):
    s = chain()
    column, phrase = ROLES[role]
    snap = _snapshot(s)
    with pytest.raises(MethodError) as ei:
        STEPS[step](s, column, inplace)
    err = ei.value
    assert err.code == "WEIGHT_OVERWRITE"
    assert err.param == column
    assert phrase in err.detail
    _assert_untouched(s, snap)


@pytest.mark.parametrize("role", list(ROLES))
def test_writing_into_an_existing_name_via_into_is_refused(chain, role):
    s = chain()
    column, _ = ROLES[role]
    snap = _snapshot(s)
    for step in (
        lambda: s.wrangling.recode("stype", {"X": ["E"]}, into=column),
        lambda: s.wrangling.categorize("api00", bins=[0, 600, 1000], into=column),
        lambda: s.wrangling.top_code({"api00": 1.0}, into=column),
    ):
        with pytest.raises((MethodError, ValueError), match="already exists|WEIGHT_OVERWRITE"):
            step()
    _assert_untouched(s, snap)


def test_join_under_a_weight_name_is_refused(chain):
    s = chain()
    other = s.data.select("snum", pl.lit(1.0).alias("tr"))
    snap = _snapshot(s)
    with pytest.raises(MethodError):
        s.wrangling.join(other, on="snum")
    _assert_untouched(s, snap)


def test_error_message_and_hint(chain):
    s = chain()
    with pytest.raises(MethodError) as ei:
        s.wrangling.mutate({"rk": pl.col("rk") + 1})
    err = ei.value
    assert err.title == "Weight columns cannot be overwritten"
    assert err.where == "wrangling.mutate"
    assert "sample.wrangling.mutate({'rk_new': ...})" in err.hint
    assert "sample.wrangling.rename_columns({'rk': 'rk_v1'})" in err.hint
    assert "then write 'rk'" in err.hint


def test_several_columns_are_all_named(chain):
    s = chain()
    with pytest.raises(MethodError) as ei:
        s.wrangling.mutate({"tr": pl.col("tr") * 2, "pw": pl.col("pw") * 2, "api00": 1})
    assert set(ei.value.got) == {"tr", "pw"}
    assert "'tr' is the design's weight" in ei.value.detail


@pytest.mark.parametrize(
    "column,dtype",
    [("tr", pl.Float64), (CELLS, pl.Int64), (CELLS, pl.Float64)],
)
def test_exact_widening_cast_is_allowed_and_changes_nothing_else(chain, column, dtype):
    s = chain()
    assert s.data.schema[CELLS] == pl.Int32
    se = s.estimation.mean("api00").to_polars()["se"][0]
    with no_warnings():
        out = s.wrangling.cast(column, dtype)
        assert out.data.schema[column] == dtype
        assert out.design == s.design
        assert out.design_history == s.design_history
        assert_allclose(out.estimation.mean("api00").to_polars()["se"][0], se)


@pytest.mark.parametrize(
    "column,dtype",
    [(CELLS, pl.Int16), (CELLS, pl.UInt32), (CELLS, pl.Float32), ("pw", pl.Int64)],
)
def test_other_casts_are_refused(chain, column, dtype):
    s = chain()
    with pytest.raises(MethodError) as ei:
        s.wrangling.cast(column, dtype, strict=False)
    assert ei.value.code == "WEIGHT_OVERWRITE"


def test_new_names_and_design_columns_stay_writable(chain):
    s = chain()
    with no_warnings():
        out = s.wrangling.mutate({"tr_new": pl.col("tr") * 2, "api00": pl.col("api00") + 1})
        out = out.wrangling.recode("stype", {"EM": ["E", "M"]}, replace=True)
        out = out.wrangling.fill_null("api99", 0)
        out = out.wrangling.cast("dnum", pl.Float64)
    assert out.design.wgt == "tr"
    assert set(out.data.get_column("stype").unique()) == {"EM", "H"}


def test_strata_recode_is_still_allowed(apiclus1):
    s = Sample(apiclus1, Design(wgt="pw", stratum="stype", psu="dnum"))
    s = s.weighting.poststratify(STYPE_POP, cells="stype", wgt_name="ps")
    with no_warnings():
        out = s.wrangling.recode("stype", {"EM": ["E", "M"]}, replace=True)
        out.estimation.mean("api00")
    assert out.design.stratum == "stype"


def test_columns_outside_the_lineage_stay_writable(apiclus1):
    s = Sample(apiclus1, Design(wgt="pw", psu="dnum"))
    with no_warnings():
        out = s.wrangling.mutate({"api99": pl.col("api99") * 2})
    assert out.design.wgt == "pw"


def test_rename_then_write_keeps_the_chain(chain):
    s = chain()
    ses = {}
    for w in ("pw", "rk", "tr"):
        s.update_design(wgt=w)
        ses[w] = s.estimation.mean("api00").to_polars()["se"][0]
    with no_warnings():
        s = s.wrangling.rename_columns({"rk": "rk_v1"})
        s = s.wrangling.mutate({"rk": pl.col("rk_v1") * 0 + 1.0})
        assert s.design.wgt_adjustment.prev_wgt == "rk_v1"
        for w, name in (("pw", "pw"), ("rk", "rk_v1"), ("tr", "tr"), ("rk", "rk_v1")):
            s.update_design(wgt=name)
            assert_allclose(s.estimation.mean("api00").to_polars()["se"][0], ses[w])
        assert s.design.wgt_adjustment.kind == "raking"
        assert s.design.wgt_adjustment.new_wgt == "rk_v1"
        # The new 'rk' is an ordinary column: nothing reads it.
        s = s.wrangling.mutate({"rk": pl.col("rk") * 3})


def test_a_removed_history_column_is_no_longer_protected(chain):
    s = chain()
    with no_warnings():
        s = s.wrangling.remove_columns(CELLS)
        s = s.wrangling.mutate({CELLS: 1})
    with pytest.raises(MethodError):
        s.wrangling.mutate({"pw": 1.0})


def test_no_design_means_nothing_is_protected(apiclus1):
    s = Sample(apiclus1)
    with no_warnings():
        out = s.wrangling.mutate({"pw": pl.col("pw") * 2})
    assert_allclose(out.data["pw"].to_numpy(), apiclus1["pw"].to_numpy() * 2)
