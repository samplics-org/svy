# tests/svy/core/test_repweights_integration.py
"""
Integration tests for samples with replicate weights.

Regression tests for the stale ``RepWeights.wgts`` / ``.clone()`` API:
clone, column rename, keep/remove columns, and singleton handling must all
work on a replication design (previously these raised
``AttributeError: 'RepWeights' object has no attribute 'wgts'``).
"""

import polars as pl
import pytest

import svy

from svy.core.design import make_rep_weights
from svy.errors import MethodError


@pytest.fixture
def rep_sample():
    """
    Stratified sample with jackknife replicate weights rw1..rw4.

    Stratum C is a singleton (one PSU) so singleton handling can be exercised.
    """
    df = pl.DataFrame(
        {
            "strat": ["A", "A", "B", "B", "C"],
            "psu": ["1", "2", "3", "4", "5"],
            "w": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [10.0, 20.0, 30.0, 40.0, 50.0],
            "extra": [1, 2, 3, 4, 5],
            "rw1": [1.0] * 5,
            "rw2": [1.0] * 5,
            "rw3": [1.0] * 5,
            "rw4": [1.0] * 5,
        }
    )
    rw = make_rep_weights("jackknife", prefix="rw", n_reps=4)
    design = svy.Design(stratum="strat", psu="psu", wgt="w", rep_wgts=rw)
    return svy.Sample(df, design)


def test_clone_with_rep_weights(rep_sample):
    clone = rep_sample.clone()
    assert clone.design.rep_wgts is not None
    assert clone.design.rep_wgts.prefix == "rw"
    assert clone.design.rep_wgts.n_reps == 4


def test_rename_non_design_column_keeps_rep_weights(rep_sample):
    out = rep_sample.wrangling.rename_columns({"extra": "extra2"})
    assert "extra2" in out._data.columns
    assert out.design.rep_wgts is not None
    assert out.design.rep_wgts.prefix == "rw"


def test_rename_rep_weight_columns_updates_prefix(rep_sample):
    renames = {f"rw{i}": f"jk{i}" for i in range(1, 5)}
    out = rep_sample.wrangling.rename_columns(renames)
    assert out.design.rep_wgts is not None
    assert out.design.rep_wgts.prefix == "jk"
    assert set(out.design.rep_wgts.columns) <= set(out._data.columns)


def test_rename_rep_weight_column_dropping_suffix_raises(rep_sample):
    with pytest.raises(Exception, match="replicate"):
        rep_sample.wrangling.rename_columns({"rw1": "not_a_rep_weight"})


def test_clean_names_with_rep_weights(rep_sample):
    # Uppercase rename path goes through the same design-rename machinery.
    out = rep_sample.wrangling.clean_names(letter_case="upper")
    assert out.design.rep_wgts is not None
    assert out.design.rep_wgts.prefix == "RW"


def test_remove_unrelated_column_keeps_rep_weights(rep_sample):
    out = rep_sample.wrangling.remove_columns("extra")
    assert "extra" not in out._data.columns
    assert out.design.rep_wgts is not None


def test_remove_rep_weight_column_is_protected(rep_sample):
    with pytest.raises(MethodError):
        rep_sample.wrangling.remove_columns("rw1")


def test_remove_rep_weight_column_with_force_drops_rep_design(rep_sample):
    out = rep_sample.wrangling.remove_columns("rw1", force=True)
    assert "rw1" not in out._data.columns
    # A partial replicate set cannot be represented; the replicate design
    # must be dropped as a whole.
    assert out.design.rep_wgts is None


def test_keep_columns_with_design_columns_keeps_rep_weights(rep_sample):
    keep = ["strat", "psu", "w", "y", "rw1", "rw2", "rw3", "rw4"]
    out = rep_sample.wrangling.keep_columns(keep)
    for c in ("rw1", "rw2", "rw3", "rw4"):
        assert c in out._data.columns
    assert out.design.rep_wgts is not None


def test_keep_columns_force_drops_rep_design(rep_sample):
    out = rep_sample.wrangling.keep_columns(["y"], force=True)
    assert out.design.rep_wgts is None


def test_singleton_handling_with_rep_weights(rep_sample):
    assert rep_sample.singleton.exists
    pooled = rep_sample.singleton.pool()
    assert pooled.design.rep_wgts is not None
    assert pooled.singleton.last_result is not None
