# tests/svy/utils/test_input_errors.py
"""A column spec that is not names raises UsageError, wherever the call reaches it."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import svy

from svy.errors import Code, SvyError, UsageError


@pytest.fixture
def sample() -> svy.Sample:
    df = pl.DataFrame(
        {"y": np.arange(8, dtype=float), "g": np.tile(["a", "b"], 4), "w": np.ones(8)}
    )
    return svy.Sample(df, svy.Design(wgt="w"))


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.wrangling.remove_columns(3),
        lambda s: s.wrangling.keep_columns(3),
        lambda s: s.estimation.mean("y", by=3),
        lambda s: s.sampling.srs(n=1, by=3),
        lambda s: s.sampling.srs(n=1, order_by=3),
        lambda s: s.describe(columns=3),
        lambda s: s.sampling.srs(n=1, by=["g", 3]),
    ],
    ids=["remove", "keep", "mean by", "srs by", "srs order_by", "describe", "list with int"],
)
def test_column_spec_that_is_not_names(sample, call):
    with pytest.raises(UsageError) as err:
        call(sample)
    e = err.value
    assert type(e) is UsageError.TypeError and isinstance(e, SvyError)
    assert e.code == Code.INVALID_TYPE and e.param == "columns"


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.estimation.mean("y", by=["g"]),
        lambda s: s.estimation.mean("y", by=("g",)),
        lambda s: s.describe(columns=["y"]),
        lambda s: s.wrangling.remove_columns(["g"]),
    ],
    ids=["by list", "by tuple", "describe list", "remove list"],
)
def test_names_in_a_list_or_tuple_still_work(sample, call):
    call(sample)
