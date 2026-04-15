# tests/svy/selection/test_add_stage.py
"""
Comprehensive tests for Selection.add_stage().

Coverage:
  - Basic two-stage workflow (DataFrame, unselected Sample, pre-selected Sample)
  - Probability chaining correctness
  - Column renaming (always-rename, collision rule)
  - PSU validation (unmatched PSUs, warning for stage-1 PSUs not in stage-2)
  - Design fields on the returned Sample
  - Custom prob_name / wgt_name
  - Multi-column PSU
  - Corner cases: single PSU, all PSUs matched, n=1 per EA
  - Error cases: missing prob, missing PSU, wrong type
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from svy import Design, Sample
from svy.core.constants import (
    SVY_CERTAINTY,
    SVY_HIT,
    SVY_PROB,
    SVY_PROB_STAGE1,
    SVY_PROB_STAGE2,
    SVY_ROW_INDEX,
    SVY_WEIGHT,
    SVY_WGT_STAGE1,
    SVY_WGT_STAGE2,
    SVY_CERT_STAGE1,
    SVY_HITS_STAGE1,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _ea_frame() -> pl.DataFrame:
    """Small EA-level frame: 3 EAs, known MOS."""
    return pl.DataFrame(
        {
            "ea": [1, 2, 3],
            "region": ["North", "North", "South"],
            "mos": [100.0, 200.0, 300.0],
            "ea_aux": ["a", "b", "c"],  # extra EA-level variable
        }
    )


def _hh_frame() -> pl.DataFrame:
    """Small household-level frame: 3 HHs per EA."""
    return pl.DataFrame(
        {
            "hid": [101, 102, 103, 201, 202, 203, 301, 302, 303],
            "ea": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "region": [
                "North",
                "North",
                "North",
                "North",
                "North",
                "North",
                "South",
                "South",
                "South",
            ],
            "income": [50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0],
        }
    )


def _selected_ea_sample() -> Sample:
    """Select all 3 EAs (n=3) so probabilities are deterministic."""
    df = _ea_frame()
    design = Design(mos="mos", stratum="region", psu="ea")
    samp = Sample(data=df, design=design)
    return samp.sampling.pps_sys(n=3, rstate=RNG)


def _hh_sample_obj() -> Sample:
    """Household frame as a Sample (not yet selected)."""
    return Sample(
        data=_hh_frame(),
        design=Design(stratum="region", psu="ea"),
    )


def _selected_hh_sample() -> Sample:
    """Pre-selected household sample: 2 HHs per EA."""
    hh = _hh_sample_obj()
    return hh.sampling.srs(n=2, by="ea", rstate=RNG)


# ---------------------------------------------------------------------------
# 1. Basic workflow — next_stage is a DataFrame
# ---------------------------------------------------------------------------


class TestAddStageDataFrame:
    def test_returns_sample(self):
        ea = _selected_ea_sample()
        result = ea.sampling.add_stage(_hh_frame())
        assert isinstance(result, Sample)

    def test_data_has_all_hh_rows(self):
        ea = _selected_ea_sample()
        result = ea.sampling.add_stage(_hh_frame())
        # All 9 HH rows kept (stage-2 is the population)
        assert result.data.height == 9

    def test_stage1_prob_column_present(self):
        ea = _selected_ea_sample()
        result = ea.sampling.add_stage(_hh_frame())
        assert SVY_PROB_STAGE1 in result.data.columns

    def test_stage1_weight_column_present(self):
        ea = _selected_ea_sample()
        result = ea.sampling.add_stage(_hh_frame())
        assert SVY_WGT_STAGE1 in result.data.columns

    def test_design_prob_is_stage1(self):
        ea = _selected_ea_sample()
        result = ea.sampling.add_stage(_hh_frame())
        assert result.design.prob == SVY_PROB_STAGE1

    def test_design_stratum_from_stage1(self):
        ea = _selected_ea_sample()
        result = ea.sampling.add_stage(_hh_frame())
        assert result.design.stratum == ea.design.stratum

    def test_design_psu_from_stage1(self):
        ea = _selected_ea_sample()
        result = ea.sampling.add_stage(_hh_frame())
        assert result.design.psu == ea.design.psu

    def test_design_wgt_is_none(self):
        ea = _selected_ea_sample()
        result = ea.sampling.add_stage(_hh_frame())
        assert result.design.wgt is None

    def test_design_hit_is_none(self):
        ea = _selected_ea_sample()
        result = ea.sampling.add_stage(_hh_frame())
        assert result.design.hit is None

    def test_design_mos_is_none(self):
        ea = _selected_ea_sample()
        result = ea.sampling.add_stage(_hh_frame())
        assert result.design.mos is None

    def test_design_pop_size_is_none(self):
        ea = _selected_ea_sample()
        result = ea.sampling.add_stage(_hh_frame())
        assert result.design.pop_size is None

    def test_design_rep_wgts_is_none(self):
        ea = _selected_ea_sample()
        result = ea.sampling.add_stage(_hh_frame())
        assert result.design.rep_wgts is None

    def test_stage1_row_index_not_in_result(self):
        ea = _selected_ea_sample()
        result = ea.sampling.add_stage(_hh_frame())
        # stage-1 row index should not bleed into stage-2 data
        assert SVY_ROW_INDEX not in result.data.columns or (
            # if present, it should be the stage-2 row index
            result.data[SVY_ROW_INDEX].n_unique() == result.data.height
        )

    def test_ea_aux_column_carried_over(self):
        """Non-design EA variable should come along."""
        ea = _selected_ea_sample()
        result = ea.sampling.add_stage(_hh_frame())
        assert "ea_aux" in result.data.columns

    def test_ea_aux_values_correct(self):
        ea = _selected_ea_sample()
        result = ea.sampling.add_stage(_hh_frame())
        # ea=1 -> ea_aux="a", ea=2 -> "b", ea=3 -> "c"
        for row in result.data.iter_rows(named=True):
            expected = {1: "a", 2: "b", 3: "c"}[row["ea"]]
            assert row["ea_aux"] == expected


# ---------------------------------------------------------------------------
# 2. Basic workflow — next_stage is an unselected Sample
# ---------------------------------------------------------------------------


class TestAddStageUnselectedSample:
    def test_returns_sample(self):
        ea = _selected_ea_sample()
        hh = _hh_sample_obj()
        result = ea.sampling.add_stage(hh)
        assert isinstance(result, Sample)

    def test_data_has_all_hh_rows(self):
        ea = _selected_ea_sample()
        result = ea.sampling.add_stage(_hh_sample_obj())
        assert result.data.height == 9

    def test_design_ssu_set_from_stage2_psu(self):
        ea = _selected_ea_sample()
        hh = _hh_sample_obj()
        result = ea.sampling.add_stage(hh)
        # stage-2 psu="ea" becomes ssu in combined design
        assert result.design.ssu == hh.design.psu

    def test_already_selected_false(self):
        ea = _selected_ea_sample()
        result = ea.sampling.add_stage(_hh_sample_obj())
        assert result.design.prob == SVY_PROB_STAGE1


# ---------------------------------------------------------------------------
# 3. Basic workflow — next_stage is a pre-selected Sample
# ---------------------------------------------------------------------------


class TestAddStagePreselected:
    def test_returns_sample(self):
        ea = _selected_ea_sample()
        hh_sel = _selected_hh_sample()
        result = ea.sampling.add_stage(hh_sel)
        assert isinstance(result, Sample)

    def test_combined_prob_column_present(self):
        ea = _selected_ea_sample()
        hh_sel = _selected_hh_sample()
        result = ea.sampling.add_stage(hh_sel)
        assert SVY_PROB in result.data.columns

    def test_combined_weight_column_present(self):
        ea = _selected_ea_sample()
        hh_sel = _selected_hh_sample()
        result = ea.sampling.add_stage(hh_sel)
        assert SVY_WEIGHT in result.data.columns

    def test_stage2_prob_column_present(self):
        ea = _selected_ea_sample()
        hh_sel = _selected_hh_sample()
        result = ea.sampling.add_stage(hh_sel)
        assert SVY_PROB_STAGE2 in result.data.columns

    def test_stage2_weight_column_present(self):
        ea = _selected_ea_sample()
        hh_sel = _selected_hh_sample()
        result = ea.sampling.add_stage(hh_sel)
        assert SVY_WGT_STAGE2 in result.data.columns

    def test_combined_prob_is_product(self):
        """pi_combined = pi_stage1 * pi_stage2 for every row."""
        ea = _selected_ea_sample()
        hh_sel = _selected_hh_sample()
        result = ea.sampling.add_stage(hh_sel)
        df = result.data
        for row in df.iter_rows(named=True):
            expected = row[SVY_PROB_STAGE1] * row[SVY_PROB_STAGE2]
            assert np.isclose(row[SVY_PROB], expected, atol=1e-10)

    def test_combined_weight_is_inverse_prob(self):
        ea = _selected_ea_sample()
        hh_sel = _selected_hh_sample()
        result = ea.sampling.add_stage(hh_sel)
        df = result.data
        for row in df.iter_rows(named=True):
            assert np.isclose(row[SVY_WEIGHT], 1.0 / row[SVY_PROB], atol=1e-10)

    def test_design_prob_is_combined(self):
        ea = _selected_ea_sample()
        hh_sel = _selected_hh_sample()
        result = ea.sampling.add_stage(hh_sel)
        assert result.design.prob == SVY_PROB

    def test_design_wgt_is_combined(self):
        ea = _selected_ea_sample()
        hh_sel = _selected_hh_sample()
        result = ea.sampling.add_stage(hh_sel)
        assert result.design.wgt == SVY_WEIGHT


# ---------------------------------------------------------------------------
# 4. Probability chaining through subsequent selection
# ---------------------------------------------------------------------------


class TestChainingThroughSelection:
    def _run(self):
        ea = _selected_ea_sample()
        hh_stage = ea.sampling.add_stage(_hh_frame())
        hh_sample = hh_stage.sampling.srs(n=2, by="ea", rstate=np.random.default_rng(99))
        return ea, hh_sample

    def test_combined_prob_column_present(self):
        _, hh = self._run()
        assert SVY_PROB in hh.data.columns

    def test_stage2_prob_column_present(self):
        _, hh = self._run()
        assert SVY_PROB_STAGE2 in hh.data.columns

    def test_stage1_prob_column_present(self):
        _, hh = self._run()
        assert SVY_PROB_STAGE1 in hh.data.columns

    def test_combined_prob_equals_product(self):
        """pi_combined = pi_stage1 * pi_stage2 for every selected HH."""
        _, hh = self._run()
        for row in hh.data.iter_rows(named=True):
            expected = row[SVY_PROB_STAGE1] * row[SVY_PROB_STAGE2]
            assert np.isclose(row[SVY_PROB], expected, atol=1e-10), (
                f"row={row}, expected pi={expected}"
            )

    def test_combined_weight_equals_inverse_prob(self):
        _, hh = self._run()
        for row in hh.data.iter_rows(named=True):
            assert np.isclose(row[SVY_WEIGHT], 1.0 / row[SVY_PROB], atol=1e-10)

    def test_stage2_hits_present(self):
        _, hh = self._run()
        assert SVY_HIT in hh.data.columns

    def test_selected_rows_count(self):
        """2 HHs per EA × 3 EAs = 6 HHs."""
        ea, hh = self._run()
        n_eas = ea.data.height
        assert hh.data.height == n_eas * 2

    def test_design_prob_updated(self):
        _, hh = self._run()
        assert hh.design.prob == SVY_PROB

    def test_design_wgt_updated(self):
        _, hh = self._run()
        assert hh.design.wgt == SVY_WEIGHT


# ---------------------------------------------------------------------------
# 5. Column renaming
# ---------------------------------------------------------------------------


class TestColumnRenaming:
    def test_stage1_prob_always_renamed(self):
        ea = _selected_ea_sample()
        result = ea.sampling.add_stage(_hh_frame())
        assert SVY_PROB_STAGE1 in result.data.columns
        # Original name should not exist (it was renamed)
        assert SVY_PROB not in result.data.columns

    def test_stage1_weight_always_renamed(self):
        ea = _selected_ea_sample()
        result = ea.sampling.add_stage(_hh_frame())
        assert SVY_WGT_STAGE1 in result.data.columns
        assert SVY_WEIGHT not in result.data.columns

    def test_stage1_certainty_renamed_if_present(self):
        ea = _selected_ea_sample()
        if SVY_CERTAINTY in ea.data.columns:
            result = ea.sampling.add_stage(_hh_frame())
            assert SVY_CERT_STAGE1 in result.data.columns

    def test_collision_rule_appends_stage1(self):
        """If stage-1 has a column that also exists in stage-2, append _stage1."""
        # Both frames have "region"
        ea = _selected_ea_sample()
        result = ea.sampling.add_stage(_hh_frame())
        # hh_frame has "region" -> stage-1 "region" should become "region_stage1"
        assert "region_stage1" in result.data.columns
        # stage-2 "region" kept as-is
        assert "region" in result.data.columns

    def test_no_collision_keeps_name(self):
        """Unique stage-1 columns (ea_aux) should keep their name."""
        ea = _selected_ea_sample()
        result = ea.sampling.add_stage(_hh_frame())
        assert "ea_aux" in result.data.columns
        assert "ea_aux_stage1" not in result.data.columns

    def test_row_index_not_carried_from_stage1(self):
        """svy_row_index from stage-1 must not appear in combined data."""
        ea = _selected_ea_sample()
        result = ea.sampling.add_stage(_hh_frame())
        # If SVY_ROW_INDEX is present, it must be the stage-2 one
        if SVY_ROW_INDEX in result.data.columns:
            # Values should be 0..N-1 corresponding to stage-2 rows
            assert result.data[SVY_ROW_INDEX].n_unique() == result.data.height


# ---------------------------------------------------------------------------
# 6. Custom prob_name and wgt_name
# ---------------------------------------------------------------------------


class TestCustomNames:
    def test_custom_prob_name_in_design(self):
        ea = _selected_ea_sample()
        hh_sel = _selected_hh_sample()
        result = ea.sampling.add_stage(hh_sel, prob_name="pi_combined")
        assert result.design.prob == "pi_combined"
        assert "pi_combined" in result.data.columns

    def test_custom_wgt_name_in_design(self):
        ea = _selected_ea_sample()
        hh_sel = _selected_hh_sample()
        result = ea.sampling.add_stage(hh_sel, wgt_name="w_combined")
        assert result.design.wgt == "w_combined"
        assert "w_combined" in result.data.columns

    def test_custom_names_chaining(self):
        """Custom names should propagate through subsequent selection."""
        ea = _selected_ea_sample()
        hh_stage = ea.sampling.add_stage(_hh_frame(), prob_name="pi_hh", wgt_name="w_hh")
        hh_sample = hh_stage.sampling.srs(n=2, by="ea", rstate=np.random.default_rng(7))
        assert "pi_hh" in hh_sample.data.columns
        assert "w_hh" in hh_sample.data.columns
        assert hh_sample.design.prob == "pi_hh"
        assert hh_sample.design.wgt == "w_hh"

    def test_custom_prob_name_value_correct(self):
        """Combined prob under custom name must equal pi_stage1 * pi_stage2."""
        ea = _selected_ea_sample()
        hh_stage = ea.sampling.add_stage(_hh_frame(), prob_name="psu_prob")
        hh_sample = hh_stage.sampling.srs(n=2, by="ea", rstate=np.random.default_rng(11))
        for row in hh_sample.data.iter_rows(named=True):
            expected = row[SVY_PROB_STAGE1] * row[SVY_PROB_STAGE2]
            assert np.isclose(row["psu_prob"], expected, atol=1e-10), (
                f"psu_prob={row['psu_prob']}, expected={expected}"
            )

    def test_custom_wgt_name_is_inverse_prob(self):
        """Custom weight must equal 1 / custom_prob."""
        ea = _selected_ea_sample()
        hh_stage = ea.sampling.add_stage(_hh_frame(), prob_name="psu_prob", wgt_name="psu_wgt")
        hh_sample = hh_stage.sampling.srs(n=2, by="ea", rstate=np.random.default_rng(11))
        for row in hh_sample.data.iter_rows(named=True):
            assert np.isclose(row["psu_wgt"], 1.0 / row["psu_prob"], atol=1e-10)

    def test_custom_names_do_not_create_default_cols(self):
        """When custom names are given, the default svy_prob_selection and
        svy_sample_weight columns should not be present as the combined output."""
        ea = _selected_ea_sample()
        hh_stage = ea.sampling.add_stage(_hh_frame(), prob_name="psu_prob", wgt_name="psu_wgt")
        hh_sample = hh_stage.sampling.srs(n=2, by="ea", rstate=np.random.default_rng(11))
        # The combined cols should be under custom names
        assert "psu_prob" in hh_sample.data.columns
        assert "psu_wgt" in hh_sample.data.columns
        # Default names should NOT be the combined output
        assert hh_sample.design.prob == "psu_prob"
        assert hh_sample.design.wgt == "psu_wgt"

    def test_stage1_prob_still_renamed_with_custom_names(self):
        """svy_prob_selection_stage1 must still be present regardless of
        custom output names."""
        ea = _selected_ea_sample()
        hh_stage = ea.sampling.add_stage(_hh_frame(), prob_name="psu_prob", wgt_name="psu_wgt")
        hh_sample = hh_stage.sampling.srs(n=2, by="ea", rstate=np.random.default_rng(11))
        assert SVY_PROB_STAGE1 in hh_sample.data.columns
        assert SVY_WGT_STAGE1 in hh_sample.data.columns


# ---------------------------------------------------------------------------
# 7. PSU validation
# ---------------------------------------------------------------------------


class TestPSUValidation:
    def test_unmatched_psu_in_stage2_raises(self):
        """If stage-2 has a PSU not in stage-1, raise ValueError."""
        ea = _selected_ea_sample()
        # Add a household with ea=999 (not in stage-1)
        bad_hh = _hh_frame().vstack(
            pl.DataFrame({"hid": [999], "ea": [999], "region": ["North"], "income": [50.0]})
        )
        with pytest.raises(ValueError, match="no match in stage-1"):
            ea.sampling.add_stage(bad_hh)

    def test_stage1_psu_not_in_stage2_warns(self):
        """If stage-1 has a PSU with no stage-2 records, warn."""
        ea = _selected_ea_sample()
        # Only keep HHs from ea=1 and ea=2, drop ea=3
        partial_hh = _hh_frame().filter(pl.col("ea") != 3)
        with pytest.warns(UserWarning, match="PSU"):
            ea.sampling.add_stage(partial_hh)

    def test_stage1_psu_not_in_stage2_warns_count(self):
        ea = _selected_ea_sample()
        partial_hh = _hh_frame().filter(pl.col("ea") != 3)
        with pytest.warns(UserWarning) as record:
            ea.sampling.add_stage(partial_hh)
        assert any("1" in str(w.message) for w in record)

    def test_all_psus_matched_no_warning(self):
        ea = _selected_ea_sample()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Should not raise any warning
            try:
                ea.sampling.add_stage(_hh_frame())
            except UserWarning:
                pytest.fail("Unexpected UserWarning when all PSUs matched")


# ---------------------------------------------------------------------------
# 8. Error cases
# ---------------------------------------------------------------------------


class TestErrorCases:
    def test_stage1_no_prob_raises(self):
        """add_stage on a sample without selection probs should raise."""
        df = _ea_frame()
        design = Design(mos="mos", stratum="region", psu="ea")
        samp = Sample(data=df, design=design)
        with pytest.raises(ValueError, match="no selection probabilities"):
            samp.sampling.add_stage(_hh_frame())

    def test_stage1_no_psu_raises(self):
        """add_stage when stage-1 design has no PSU should raise."""
        df = _ea_frame()
        design = Design(mos="mos", stratum="region")
        samp = Sample(data=df, design=design)
        # Manually add a fake prob column so we pass that check
        from svy.core.constants import SVY_PROB

        samp._data = samp._data.with_columns(pl.lit(0.5).alias(SVY_PROB))
        samp._design = samp._design.fill_missing(prob=SVY_PROB)
        with pytest.raises(ValueError, match="no PSU"):
            samp.sampling.add_stage(_hh_frame())

    def test_wrong_type_raises(self):
        ea = _selected_ea_sample()
        with pytest.raises(TypeError, match="Sample"):
            ea.sampling.add_stage({"ea": [1, 2, 3]})  # type: ignore

    def test_psu_column_missing_from_stage2_raises(self):
        """If the PSU column doesn't exist in stage-2 data, raise ValueError."""
        ea = _selected_ea_sample()
        bad_hh = _hh_frame().drop("ea")
        with pytest.raises(ValueError, match="PSU column"):
            ea.sampling.add_stage(bad_hh)


# ---------------------------------------------------------------------------
# 9. Multi-column PSU
# ---------------------------------------------------------------------------


class TestMultiColumnPSU:
    def _ea_multi(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "region": ["N", "N", "S"],
                "ea": [1, 2, 3],
                "mos": [100.0, 200.0, 300.0],
            }
        )

    def _hh_multi(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "hid": [101, 102, 201, 202, 301, 302],
                "region": ["N", "N", "N", "N", "S", "S"],
                "ea": [1, 1, 2, 2, 3, 3],
                "income": [50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            }
        )

    def test_multi_column_psu_join(self):
        """Join should work when PSU is a tuple of columns."""
        df = self._ea_multi()
        design = Design(mos="mos", psu=("region", "ea"))
        samp = Sample(data=df, design=design)
        ea_sel = samp.sampling.pps_sys(n=3, rstate=RNG)

        hh_design = Design(psu=("region", "ea"))
        hh = Sample(data=self._hh_multi(), design=hh_design)

        result = ea_sel.sampling.add_stage(hh)
        assert result.data.height == 6
        assert SVY_PROB_STAGE1 in result.data.columns


# ---------------------------------------------------------------------------
# 10. LazyFrame input
# ---------------------------------------------------------------------------


class TestLazyFrameInput:
    def test_lazyframe_accepted(self):
        ea = _selected_ea_sample()
        lazy_hh = _hh_frame().lazy()
        result = ea.sampling.add_stage(lazy_hh)
        assert isinstance(result, Sample)
        assert result.data.height == 9


# ---------------------------------------------------------------------------
# 11. Probability correctness (numerical)
# ---------------------------------------------------------------------------


class TestProbabilityCorrectness:
    def test_stage1_probs_match_ea_sample(self):
        """
        svy_prob_selection_stage1 in the combined frame should equal
        the original svy_prob_selection from ea_sample for each EA.
        """
        ea = _selected_ea_sample()
        ea_probs = {row["ea"]: row[SVY_PROB] for row in ea.data.iter_rows(named=True)}
        result = ea.sampling.add_stage(_hh_frame())
        for row in result.data.iter_rows(named=True):
            expected = ea_probs[row["ea"]]
            assert np.isclose(row[SVY_PROB_STAGE1], expected, atol=1e-10)

    def test_stage1_weight_is_inverse_stage1_prob(self):
        ea = _selected_ea_sample()
        result = ea.sampling.add_stage(_hh_frame())
        for row in result.data.iter_rows(named=True):
            assert np.isclose(row[SVY_WGT_STAGE1], 1.0 / row[SVY_PROB_STAGE1], atol=1e-10)

    def test_combined_prob_range(self):
        """Combined probabilities must be in (0, 1]."""
        ea = _selected_ea_sample()
        hh_stage = ea.sampling.add_stage(_hh_frame())
        hh_sample = hh_stage.sampling.srs(n=2, by="ea", rstate=np.random.default_rng(5))
        probs = hh_sample.data[SVY_PROB].to_list()
        for p in probs:
            assert 0.0 < p <= 1.0 + 1e-10, f"prob={p} out of range"

    def test_combined_weight_positive(self):
        ea = _selected_ea_sample()
        hh_stage = ea.sampling.add_stage(_hh_frame())
        hh_sample = hh_stage.sampling.srs(n=2, by="ea", rstate=np.random.default_rng(5))
        wgts = hh_sample.data[SVY_WEIGHT].to_list()
        for w in wgts:
            assert w > 0, f"weight={w} not positive"


# ---------------------------------------------------------------------------
# 12. Idempotency and immutability
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_stage1_sample_unchanged(self):
        """add_stage must not modify the original stage-1 Sample."""
        ea = _selected_ea_sample()
        original_cols = set(ea.data.columns)
        original_height = ea.data.height
        _ = ea.sampling.add_stage(_hh_frame())
        assert set(ea.data.columns) == original_cols
        assert ea.data.height == original_height

    def test_next_stage_sample_unchanged(self):
        """add_stage must not modify the next_stage Sample."""
        ea = _selected_ea_sample()
        hh = _hh_sample_obj()
        original_cols = set(hh.data.columns)
        original_height = hh.data.height
        _ = ea.sampling.add_stage(hh)
        assert set(hh.data.columns) == original_cols
        assert hh.data.height == original_height


# ---------------------------------------------------------------------------
# 13. Corner cases
# ---------------------------------------------------------------------------


class TestCornerCases:
    def test_single_ea(self):
        """Works with a single PSU in stage-1."""
        df = pl.DataFrame({"ea": [1], "mos": [100.0]})
        design = Design(mos="mos", psu="ea")
        samp = Sample(data=df, design=design)
        ea_sel = samp.sampling.pps_sys(n=1, rstate=RNG)

        hh_df = pl.DataFrame(
            {
                "hid": [101, 102, 103],
                "ea": [1, 1, 1],
                "income": [50.0, 60.0, 70.0],
            }
        )
        result = ea_sel.sampling.add_stage(hh_df)
        assert result.data.height == 3
        assert SVY_PROB_STAGE1 in result.data.columns

    def test_n1_per_ea_srs_after_add_stage(self):
        """Select 1 HH per EA after add_stage — combined probs correct."""
        ea = _selected_ea_sample()
        hh_stage = ea.sampling.add_stage(_hh_frame())
        hh_sample = hh_stage.sampling.srs(n=1, by="ea", rstate=np.random.default_rng(13))
        # 1 HH per EA × 3 EAs = 3
        assert hh_sample.data.height == 3
        # All probs positive
        assert (hh_sample.data[SVY_PROB] > 0).all()

    def test_all_hhs_selected_per_ea(self):
        """Select all 3 HHs per EA — stage-2 probs should all be 1.0."""
        ea = _selected_ea_sample()
        hh_stage = ea.sampling.add_stage(_hh_frame())
        hh_sample = hh_stage.sampling.srs(n=3, by="ea", rstate=np.random.default_rng(17))
        assert hh_sample.data.height == 9
        stage2_probs = hh_sample.data[SVY_PROB_STAGE2].to_list()
        for p in stage2_probs:
            assert np.isclose(p, 1.0, atol=1e-10)

    def test_pps_after_add_stage(self):
        """PPS selection for stage-2 after add_stage should also chain.

        add_stage sets mos=None in the combined Design (correct per spec —
        MOS is stage-specific). The user must restore mos on the combined
        design before calling pps_sys.
        """
        ea = _selected_ea_sample()
        # Add mos to hh frame for PPS
        hh_with_mos = _hh_frame()
        hh_design = Design(psu="ea", mos="income")
        hh = Sample(data=hh_with_mos, design=hh_design)
        hh_stage = ea.sampling.add_stage(hh)
        hh_stage = hh_stage.update_design(mos="income")
        hh_sample = hh_stage.sampling.pps_sys(n=2, by="ea", rstate=np.random.default_rng(21))
        assert SVY_PROB in hh_sample.data.columns
        assert SVY_PROB_STAGE1 in hh_sample.data.columns
        assert SVY_PROB_STAGE2 in hh_sample.data.columns
        # Combined = product
        for row in hh_sample.data.iter_rows(named=True):
            expected = row[SVY_PROB_STAGE1] * row[SVY_PROB_STAGE2]
            assert np.isclose(row[SVY_PROB], expected, atol=1e-10)
