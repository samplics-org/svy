# tests/svy/estimation/test_native_levels_state.py
"""
Native levels stay right when the sample changes between two estimates.

Estimation memoises the level lookup per data version; every wrangling step,
in place or not, and every design change must be seen by the next estimate.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from svy import Design, Sample


N = 240


def _sample() -> Sample:
    rng = np.random.default_rng(7)
    df = pl.DataFrame(
        {
            "stratum": np.repeat(["s1", "s2"], N // 2),
            "psu": np.repeat(np.arange(24), N // 24),
            "w": rng.uniform(1, 3, N),
            "w2": rng.uniform(1, 3, N),
            "y": rng.normal(10, 2, N),
            "zone": rng.integers(1, 4, N),
        }
    )
    return Sample(df, Design(stratum="stratum", psu="psu", wgt="w"))


def _levels(s: Sample, by: str = "zone") -> set:
    return {p.by_level[0] for p in s.estimation.mean("y", by=by).estimates}


def _prop_levels(s: Sample, y: str = "zone") -> set:
    return {p.y_level for p in s.estimation.prop(y).estimates}


def test_cast_in_place_between_estimates():
    s = _sample()
    assert _levels(s) == {1, 2, 3}
    s.wrangling.cast("zone", pl.String, inplace=True)
    assert _levels(s) == {"1", "2", "3"}
    s.wrangling.cast("zone", pl.Float64, inplace=True)
    assert _levels(s) == {1.0, 2.0, 3.0}
    assert all(isinstance(v, float) for v in _levels(s))


def test_fork_does_not_share_the_parent_cache():
    parent = _sample()
    assert _levels(parent) == {1, 2, 3}
    child = parent.wrangling.cast("zone", pl.String)
    assert _levels(child) == {"1", "2", "3"}
    # Interleave: each keeps its own data.
    assert _levels(parent) == {1, 2, 3}
    assert _levels(child) == {"1", "2", "3"}
    assert _prop_levels(parent) == {1, 2, 3}
    assert _prop_levels(child) == {"1", "2", "3"}


def test_recode_in_place_changes_the_levels():
    s = _sample()
    assert _levels(s) == {1, 2, 3}
    s.wrangling.recode("zone", {10: [1], 20: [2, 3]}, replace=True, inplace=True)
    assert _levels(s) == {10, 20}
    assert _prop_levels(s) == {10, 20}


def test_filter_records_drops_a_level():
    s = _sample()
    assert _levels(s) == {1, 2, 3}
    s.wrangling.filter_records(pl.col("zone") != 3, inplace=True)
    assert _levels(s) == {1, 2}


def test_rename_in_place_uses_the_new_column():
    s = _sample()
    assert _levels(s) == {1, 2, 3}
    s.wrangling.rename_columns({"zone": "area"}, inplace=True)
    assert _levels(s, by="area") == {1, 2, 3}
    with pytest.raises(Exception):
        _levels(s, by="zone")


def test_same_column_name_new_values_after_mutate():
    s = _sample()
    assert _levels(s) == {1, 2, 3}
    s.wrangling.mutate({"zone": pl.col("zone") * 100}, inplace=True)
    assert _levels(s) == {100, 200, 300}


def test_design_change_between_estimates():
    s = _sample()
    first = s.estimation.mean("y", by="zone")
    s2 = s.update_design(wgt="w2")
    second = s2.estimation.mean("y", by="zone")
    assert {p.by_level[0] for p in second.estimates} == {1, 2, 3}
    assert [p.est for p in first.estimates] != [p.est for p in second.estimates]


def test_where_and_several_by_after_cast():
    s = _sample()
    s.wrangling.cast("zone", pl.String, inplace=True)
    r = s.estimation.mean("y", by=["zone", "stratum"], where=pl.col("zone") != "2")
    assert {p.by_level for p in r.estimates} == {
        ("1", "s1"),
        ("1", "s2"),
        ("3", "s1"),
        ("3", "s2"),
    }


def test_contrast_keys_follow_the_new_type():
    s = _sample()
    r = s.estimation.mean("y", by="zone")
    assert set(r.keys()) == {1, 2, 3}
    s.wrangling.cast("zone", pl.String, inplace=True)
    r2 = s.estimation.mean("y", by="zone")
    assert set(r2.keys()) == {"1", "2", "3"}
    r2.contrast({"1": 1, "2": -1})


def test_cast_to_categorical_and_enum_in_place():
    s = _sample()
    s.wrangling.cast("zone", pl.String, inplace=True)
    s.wrangling.cast("zone", pl.Categorical, inplace=True)
    assert _levels(s) == {"1", "2", "3"}
    s.wrangling.cast("zone", pl.Enum(["3", "2", "1"]), inplace=True)
    assert _levels(s) == {"1", "2", "3"}


def test_null_levels_dropped_then_filled():
    s = _sample()
    s.wrangling.mutate(
        {"zone": pl.when(pl.col("zone") == 3).then(None).otherwise(pl.col("zone"))},
        inplace=True,
    )
    r = s.estimation.mean("y", by="zone", drop_nulls=True)
    assert {p.by_level[0] for p in r.estimates} == {1, 2}
    s.wrangling.fill_null("zone", 9, inplace=True)
    assert _levels(s) == {1, 2, 9}


def test_where_that_empties_a_level_after_recode():
    s = _sample()
    s.wrangling.recode("zone", {1: [1, 2], 3: [3]}, replace=True, inplace=True)
    r = s.estimation.mean("y", by="zone", where=pl.col("zone") == 1)
    assert {p.by_level[0] for p in r.estimates} <= {1, 3}
    assert all(isinstance(p.by_level[0], int) for p in r.estimates)
