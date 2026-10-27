# tests/svy/engine/test_input_errors.py
"""Checks that live in the engine raise svy errors through the public calls that reach them."""

from __future__ import annotations

import polars as pl
import pytest

import svy

from svy.errors import Code, DataError, SelectionError, SvyError


@pytest.fixture
def wr():
    df = pl.DataFrame({"a": [1.0, 2, 3, 4], "b": [1.0, 2, 3, 4], "w": [1.0] * 4})
    return svy.Sample(df, svy.Design(wgt="w")).wrangling


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


@pytest.mark.parametrize(
    "call, param, says",
    [
        (lambda w: w.rename_columns({"a": "x", "b": "x"}), "renames", "Duplicate destination"),
        (lambda w: w.categorize("a", bins=[5]), "bins", "at least 2 edges"),
        (lambda w: w.top_code({"a": 2.0, "b": 2.0}, into="q"), "into", "single input column"),
        (lambda w: w.recode(["a", "b"], {9: [1.0]}, into="q"), "into", "single input column"),
        (lambda w: w.top_code({"a": 2.0}, into={"zz": "q"}), "into", "['zz']"),
    ],
    ids=[
        "rename duplicate",
        "one bin edge",
        "into str top_code",
        "into str recode",
        "into unknown",
    ],
)
def test_wrangling_checks_in_the_engine(wr, call, param, says):
    _raises(DataError, ValueError, Code.INVALID_VALUE, lambda: call(wr), param=param, says=says)


def test_into_of_wrong_type(wr):
    _raises(
        DataError,
        TypeError,
        Code.INVALID_TYPE,
        lambda: wr.top_code({"a": 2.0}, into=3),
        param="into",
    )


def test_into_mapping_names_the_output(wr):
    out = wr.top_code({"a": 2.0}, into={"a": "a_capped"}).data
    assert out["a_capped"].to_list() == [1.0, 2.0, 2.0, 2.0]


@pytest.mark.parametrize("sd", [-1.0, float("nan"), float("inf")], ids=["negative", "nan", "inf"])
def test_neyman_sd_must_be_finite_and_non_negative(sd):
    e = _raises(
        SelectionError,
        ValueError,
        Code.ALLOCATION_INVALID,
        lambda: svy.SampleSize().allocate(
            10, pop_size={"a": 10, "b": 20}, method="neyman", sigma={"a": sd, "b": 1.0}
        ),
        says="group_sds must be finite and >= 0",
    )
    assert "'a'" in e.detail


def test_neyman_splits_by_n_times_sd():
    # N * S = 10 * 2 = 20 and 20 * 1 = 20, so n = 10 splits evenly.
    out = svy.SampleSize().allocate(
        10, pop_size={"a": 10, "b": 20}, method="neyman", sigma={"a": 2.0, "b": 1.0}
    )
    assert out.n == {"a": 5, "b": 5}
