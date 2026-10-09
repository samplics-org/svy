"""A Sample built from a LazyFrame holds the collected frame."""

import warnings

import polars as pl
import pytest

import svy


DF = pl.DataFrame(
    {
        "st": ["a", "a", "b", "b"],
        "psu": [1, 2, 3, 4],
        "w": [1.0, 2.0, 1.0, 2.0],
        "y": [1.0, 2.0, 3.0, 4.0],
    }
)


@pytest.mark.parametrize("design", [None, svy.Design(stratum="st", psu="psu", wgt="w")])
def test_lazy_input_matches_eager_input(design):
    lazy = svy.Sample(DF.lazy(), design)
    eager = svy.Sample(DF, design)

    assert isinstance(lazy._data, pl.DataFrame)
    assert lazy.data.equals(eager.data)


def test_lazy_input_never_resolves_a_lazy_schema_later():
    s = svy.Sample(DF.lazy())
    with warnings.catch_warnings():
        warnings.simplefilter("error", pl.exceptions.PerformanceWarning)
        s.estimation.mean("y")
        s.wrangling.mutate({"z": svy.col("y") * 2})
        s.describe()
