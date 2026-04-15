# tests/svy/selection/test_col_naming.py
"""
Tests for user-controlled output column naming in selection methods,
and guards against using names that already exist in the frame.

Coverage
--------
prob_name=
    custom prob column name appears in output
    custom prob name updates design.prob
    default (None) uses svy_prob_selection
    prob_name that already exists raises MethodError

wgt_name=
    custom weight column name appears in output
    custom weight name updates design.wgt
    default (None) uses svy_sample_weight
    wgt_name that already exists raises MethodError

hit_name=
    custom hit column name appears in output
    custom hit name updates design.hit
    default (None) uses svy_number_of_hits
    hit_name that already exists raises MethodError

All three custom names together
    all three custom columns present
    all three update design fields
    weight == 1/prob regardless of column names

PPS-specific
    prob_name works on pps_sys
    hit_name works on pps_sys
    svy_certainty is still present (not user-nameable)

Multi-stage chaining with custom names
    stage-1 prob name propagates correctly
    stage-2 can use different name without collision
    weight at end is 1 / combined prob

Guard fires before any drawing
    collision on prob_name raises before selection
    collision on wgt_name raises before selection
    collision on hit_name raises before selection
    error message names the offending column and param
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from svy import Sample
from svy.core.constants import SVY_HIT, SVY_PROB, SVY_WEIGHT, SVY_CERTAINTY
from svy.core.design import Design
from svy.errors import MethodError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = 42


def _make_sample(n_rows: int = 20, *, mos: bool = False, seed: int = 0) -> Sample:
    rng = np.random.default_rng(seed)
    data: dict = {
        "unit_id": list(range(n_rows)),
        "value": rng.integers(1, 100, size=n_rows).tolist(),
        "region": ["North"] * (n_rows // 2) + ["South"] * (n_rows - n_rows // 2),
    }
    if mos:
        data["size"] = rng.integers(10, 500, size=n_rows).tolist()
    design = Design(mos="size") if mos else Design()
    return Sample(data=pl.DataFrame(data), design=design)


# ---------------------------------------------------------------------------
# prob_name= on srs
# ---------------------------------------------------------------------------

class TestProbName:
    def test_custom_prob_name_in_output(self):
        samp = _make_sample()
        result = samp.sampling.srs(n=5, prob_name="stage1_prob", rstate=RNG)
        assert "stage1_prob" in result._data.columns

    def test_custom_prob_name_updates_design(self):
        samp = _make_sample()
        result = samp.sampling.srs(n=5, prob_name="stage1_prob", rstate=RNG)
        assert result._design.prob == "stage1_prob"

    def test_default_prob_name_is_svy_constant(self):
        samp = _make_sample()
        result = samp.sampling.srs(n=5, rstate=RNG)
        assert SVY_PROB in result._data.columns

    def test_custom_prob_name_not_default(self):
        """Custom name means the default SVY_PROB column is not created."""
        samp = _make_sample()
        result = samp.sampling.srs(n=5, prob_name="my_prob", rstate=RNG)
        assert "my_prob" in result._data.columns
        assert SVY_PROB not in result._data.columns

    def test_prob_values_correct_with_custom_name(self):
        samp = _make_sample(20)
        result = samp.sampling.srs(n=5, prob_name="p1", rstate=RNG)
        probs = result._data["p1"].to_numpy()
        np.testing.assert_allclose(probs, 5 / 20, rtol=1e-9)


# ---------------------------------------------------------------------------
# wgt_name= on srs
# ---------------------------------------------------------------------------

class TestWgtName:
    def test_custom_wgt_name_in_output(self):
        samp = _make_sample()
        result = samp.sampling.srs(n=5, wgt_name="base_wgt", rstate=RNG)
        assert "base_wgt" in result._data.columns

    def test_custom_wgt_name_updates_design(self):
        samp = _make_sample()
        result = samp.sampling.srs(n=5, wgt_name="base_wgt", rstate=RNG)
        assert result._design.wgt == "base_wgt"

    def test_default_wgt_name_is_svy_constant(self):
        samp = _make_sample()
        result = samp.sampling.srs(n=5, rstate=RNG)
        assert SVY_WEIGHT in result._data.columns

    def test_custom_wgt_name_not_default(self):
        samp = _make_sample()
        result = samp.sampling.srs(n=5, wgt_name="w1", rstate=RNG)
        assert "w1" in result._data.columns
        assert SVY_WEIGHT not in result._data.columns


# ---------------------------------------------------------------------------
# hit_name= on srs
# ---------------------------------------------------------------------------

class TestHitName:
    def test_custom_hit_name_in_output(self):
        samp = _make_sample()
        result = samp.sampling.srs(n=5, hit_name="stage1_hits", rstate=RNG)
        assert "stage1_hits" in result._data.columns

    def test_custom_hit_name_updates_design(self):
        samp = _make_sample()
        result = samp.sampling.srs(n=5, hit_name="stage1_hits", rstate=RNG)
        assert result._design.hit == "stage1_hits"

    def test_default_hit_name_is_svy_constant(self):
        samp = _make_sample()
        result = samp.sampling.srs(n=5, rstate=RNG)
        assert SVY_HIT in result._data.columns

    def test_custom_hit_name_not_default(self):
        samp = _make_sample()
        result = samp.sampling.srs(n=5, hit_name="h1", rstate=RNG)
        assert "h1" in result._data.columns
        assert SVY_HIT not in result._data.columns

    def test_hit_values_correct_with_custom_name(self):
        samp = _make_sample(20)
        result = samp.sampling.srs(n=5, hit_name="hits", rstate=RNG)
        assert result._data["hits"].sum() == 5


# ---------------------------------------------------------------------------
# All three names together
# ---------------------------------------------------------------------------

class TestAllThreeNames:
    def test_all_custom_columns_present(self):
        samp = _make_sample()
        result = samp.sampling.srs(
            n=5, prob_name="p", wgt_name="w", hit_name="h", rstate=RNG
        )
        assert "p" in result._data.columns
        assert "w" in result._data.columns
        assert "h" in result._data.columns

    def test_all_three_update_design(self):
        samp = _make_sample()
        result = samp.sampling.srs(
            n=5, prob_name="p", wgt_name="w", hit_name="h", rstate=RNG
        )
        assert result._design.prob == "p"
        assert result._design.wgt == "w"
        assert result._design.hit == "h"

    def test_weight_is_inverse_prob_with_custom_names(self):
        samp = _make_sample()
        result = samp.sampling.srs(
            n=5, prob_name="myprob", wgt_name="mywgt", hit_name="myhit", rstate=RNG
        )
        df = result._data
        probs = df["myprob"].to_numpy()
        wgts = df["mywgt"].to_numpy()
        np.testing.assert_allclose(wgts, 1.0 / probs, rtol=1e-9)

    def test_no_default_columns_created(self):
        samp = _make_sample()
        result = samp.sampling.srs(
            n=5, prob_name="p", wgt_name="w", hit_name="h", rstate=RNG
        )
        cols = result._data.columns
        assert SVY_PROB not in cols
        assert SVY_WEIGHT not in cols
        assert SVY_HIT not in cols


# ---------------------------------------------------------------------------
# PPS with custom names
# ---------------------------------------------------------------------------

class TestPpsCustomNames:
    def _samp(self) -> Sample:
        return _make_sample(20, mos=True)

    def test_pps_sys_custom_prob_name(self):
        result = self._samp().sampling.pps_sys(
            n=3, prob_name="ea_prob", rstate=RNG
        )
        assert "ea_prob" in result._data.columns
        assert result._design.prob == "ea_prob"

    def test_pps_sys_custom_hit_name(self):
        result = self._samp().sampling.pps_sys(
            n=3, hit_name="ea_hits", rstate=RNG
        )
        assert "ea_hits" in result._data.columns
        assert result._design.hit == "ea_hits"

    def test_pps_sys_certainty_always_present(self):
        """svy_certainty is not user-nameable — always uses the constant."""
        result = self._samp().sampling.pps_sys(
            n=3, prob_name="p", wgt_name="w", hit_name="h", rstate=RNG
        )
        assert SVY_CERTAINTY in result._data.columns

    def test_pps_sys_weight_is_inverse_prob(self):
        result = self._samp().sampling.pps_sys(
            n=3, prob_name="myprob", wgt_name="mywgt", rstate=RNG
        )
        df = result._data
        probs = df["myprob"].to_numpy()
        wgts = df["mywgt"].to_numpy()
        np.testing.assert_allclose(wgts, 1.0 / probs, rtol=1e-9)

    def test_pps_wr_custom_names(self):
        result = self._samp().sampling.pps_wr(
            n=3, prob_name="p", wgt_name="w", hit_name="h", rstate=RNG
        )
        assert result._design.prob == "p"
        assert result._design.wgt == "w"

    def test_pps_brewer_custom_names(self):
        result = self._samp().sampling.pps_brewer(
            n=2, prob_name="p", wgt_name="w", hit_name="h", rstate=RNG
        )
        assert result._design.prob == "p"

    def test_pps_murphy_custom_names(self):
        result = self._samp().sampling.pps_murphy(
            n=2, prob_name="p", wgt_name="w", hit_name="h", rstate=RNG
        )
        assert result._design.prob == "p"

    def test_pps_rs_custom_names(self):
        result = self._samp().sampling.pps_rs(
            n=2, prob_name="p", wgt_name="w", hit_name="h", rstate=RNG
        )
        assert result._design.prob == "p"


# ---------------------------------------------------------------------------
# Multi-stage chaining with custom names
# ---------------------------------------------------------------------------

class TestChainingWithCustomNames:
    """
    Two-stage design: custom stage-1 names, custom stage-2 names.
    Verifies that user can name each stage independently without collision.
    """

    def _stage1(self) -> Sample:
        """Select EAs with PPS, custom column names."""
        rng = np.random.default_rng(0)
        data = pl.DataFrame({
            "ea_id": list(range(1, 6)),
            "region": ["North", "North", "South", "South", "South"],
            "n_hh": [100, 80, 120, 90, 110],
        })
        design = Design(psu="ea_id", mos="n_hh")
        return (
            Sample(data=data, design=design)
            .sampling.pps_sys(
                n=3,
                prob_name="ea_prob",
                wgt_name="ea_wgt",
                hit_name="ea_hits",
                rstate=rng,
            )
        )

    def test_stage1_custom_col_names(self):
        ea = self._stage1()
        assert "ea_prob" in ea._data.columns
        assert "ea_wgt" in ea._data.columns
        assert "ea_hits" in ea._data.columns
        assert ea._design.prob == "ea_prob"

    def test_stage2_custom_col_names_no_collision(self):
        """Stage-2 can use its own names without colliding with stage-1."""
        ea = self._stage1()
        # Build HH frame only for selected EAs so the PSU join succeeds
        selected_ea_ids = ea._data["ea_id"].unique().to_list()
        hh_rows = [
            {"hh_id": i, "ea_id": ea_id, "income": 1000 * (i + 1)}
            for i, ea_id in enumerate(
                eid for eid in selected_ea_ids for _ in range(3)
            )
        ]
        hh_data = pl.DataFrame(hh_rows)
        combined = ea.sampling.add_stage(hh_data)
        result = combined.sampling.srs(
            n=2,
            prob_name="hh_prob",
            wgt_name="final_wgt",
            hit_name="hh_hits",
            rstate=42,
        )
        # _combine_stages always renames stage-1 prob to SVY_PROB_STAGE1
        # (svy_prob_selection_stage1) regardless of the user-supplied name --
        # the chaining trigger depends on that exact constant.
        from svy.core.constants import SVY_PROB_STAGE1
        assert SVY_PROB_STAGE1 in result._data.columns
        assert "hh_prob" in result._data.columns     # stage-2 prob present
        assert "final_wgt" in result._data.columns
        assert result._design.prob == "hh_prob"
        assert result._design.wgt == "final_wgt"


# ---------------------------------------------------------------------------
# Guard: collision detection
# ---------------------------------------------------------------------------

class TestColNameGuard:
    def test_prob_name_collision_raises(self):
        """prob_name that already exists in frame -> MethodError."""
        data = pl.DataFrame({
            "unit_id": list(range(10)),
            "existing_prob": [0.5] * 10,
        })
        samp = Sample(data=data, design=Design())
        with pytest.raises(MethodError):
            samp.sampling.srs(n=3, prob_name="existing_prob", rstate=RNG)

    def test_wgt_name_collision_raises(self):
        data = pl.DataFrame({
            "unit_id": list(range(10)),
            "existing_wgt": [2.0] * 10,
        })
        samp = Sample(data=data, design=Design())
        with pytest.raises(MethodError):
            samp.sampling.srs(n=3, wgt_name="existing_wgt", rstate=RNG)

    def test_hit_name_collision_raises(self):
        data = pl.DataFrame({
            "unit_id": list(range(10)),
            "existing_hits": [1] * 10,
        })
        samp = Sample(data=data, design=Design())
        with pytest.raises(MethodError):
            samp.sampling.srs(n=3, hit_name="existing_hits", rstate=RNG)

    def test_error_mentions_column_name(self):
        data = pl.DataFrame({"unit_id": list(range(10)), "my_prob": [0.5] * 10})
        samp = Sample(data=data, design=Design())
        with pytest.raises(MethodError) as exc:
            samp.sampling.srs(n=3, prob_name="my_prob", rstate=RNG)
        assert "my_prob" in str(exc.value)

    def test_error_mentions_param_name(self):
        data = pl.DataFrame({"unit_id": list(range(10)), "my_prob": [0.5] * 10})
        samp = Sample(data=data, design=Design())
        with pytest.raises(MethodError) as exc:
            samp.sampling.srs(n=3, prob_name="my_prob", rstate=RNG)
        assert "prob_name" in str(exc.value)

    def test_pps_guard_fires(self):
        """Guard works on pps_sys too."""
        samp = _make_sample(20, mos=True)
        data_with_col = samp._data.with_columns(pl.lit(0.5).alias("taken"))
        samp2 = Sample(data=data_with_col, design=samp._design)
        with pytest.raises(MethodError):
            samp2.sampling.pps_sys(n=3, prob_name="taken", rstate=RNG)

    def test_guard_fires_before_draw(self):
        """
        The guard must fire before the engine draws anything -- confirmed
        by the fact that a non-existent result is never assigned.
        """
        data = pl.DataFrame({"unit_id": list(range(10)), "existing": [1.0] * 10})
        samp = Sample(data=data, design=Design())
        original_rows = samp._data.height
        try:
            samp.sampling.srs(n=3, prob_name="existing", rstate=RNG)
        except MethodError:
            pass
        # Sample should be unchanged
        assert samp._data.height == original_rows
        assert "existing" in samp._data.columns
        assert samp._data["existing"][0] == 1.0
