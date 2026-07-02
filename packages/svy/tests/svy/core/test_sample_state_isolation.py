# tests/svy/core/test_sample_state_isolation.py
"""
Regression tests: derived samples (use_weight / _replace_data) must own
independent copies of mutable state (metadata, warnings) so that edits on
the derived sample never mutate the original.
"""

import polars as pl
import pytest

import svy


@pytest.fixture
def base_sample():
    df = pl.DataFrame(
        {
            "strat": ["A", "A", "B", "B"],
            "psu": ["1", "2", "3", "4"],
            "w": [1.0, 2.0, 3.0, 4.0],
            "w2": [1.5, 2.5, 3.5, 4.5],
            "age": [10, 20, 30, 40],
        }
    )
    design = svy.Design(stratum="strat", psu="psu", wgt="w")
    return svy.Sample(df, design)


def test_use_weight_metadata_is_isolated(base_sample):
    derived = base_sample.use_weight("w2")
    derived.set_var_label("age", "Age in years")

    assert base_sample.resolve_labels("age").var_label == ""
    assert derived.resolve_labels("age").var_label == "Age in years"

    base_sample.set_var_label("age", "Original label")
    assert derived.resolve_labels("age").var_label == "Age in years"


def test_use_weight_missing_defs_are_isolated(base_sample):
    derived = base_sample.use_weight("w2")
    derived.set_missing("age", codes=[-99])

    base_meta = base_sample.meta.get("age")
    assert base_meta is None or base_meta.missing is None


def test_use_weight_warnings_are_isolated(base_sample):
    derived = base_sample.use_weight("w2")
    n_before = len(base_sample.warnings)
    derived.warn(
        code="TEST",
        title="test",
        detail="test warning",
        where="tests",
    )
    assert len(base_sample.warnings) == n_before
    assert len(derived.warnings) == n_before + 1


def test_use_weight_updates_design_without_mutating_original(base_sample):
    derived = base_sample.use_weight("w2")
    assert derived.design.wgt == "w2"
    assert base_sample.design.wgt == "w"


def test_replace_data_isolates_metadata(base_sample):
    derived = base_sample._replace_data(base_sample._data)
    derived.set_var_label("age", "changed")
    assert base_sample.resolve_labels("age").var_label == ""
