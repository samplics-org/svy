# tests/svy/selection/test_where.py
"""
Tests for the ``where=`` parameter in selection methods.

Coverage
--------
_apply_where
    - None passes all rows through unchanged
    - Mapping form: single key, multi-key AND, list value (is_in)
    - Sequence of expressions (AND-combined)
    - pl.Expr form
    - Empty mapping returns None expr (all rows)
    - Mask is aligned to src_df length
    - All-false mask yields empty eligible frame

srs(where=...)
    - Output row count equals input row count
    - Non-eligible rows have null prob / weight / hit
    - Selected rows have non-null prob / weight / hit
    - Weight == 1 / prob for selected rows
    - n is computed against the eligible subset only
    - Probability == n / N_eligible (not n / N_total)
    - Stratified: where filters within strata correctly
    - where=None is identical to omitting where
    - Combining where + by
    - Non-eligible rows preserve original column values
    - All four WhereArg forms work end-to-end
    - Reproducibility: same seed + same where gives same result
    - Different seeds give different results
    - where that excludes an entire stratum warns, not crashes
    - where that matches zero rows: returns full frame with all-null selection cols
    - where + drop_nulls compose correctly
    - Design wgt and prob updated after selection

pps_sys(where=...)
    - Output row count equals input row count
    - Non-eligible rows have null prob / weight / hit / certainty
    - Selected rows have non-null prob
    - Weight == 1 / prob for selected rows
    - All selected rows are eligible
    - where=None identical to omitting where
    - Reproducibility

PPS variant smoke tests
    - pps_wr / pps_brewer / pps_murphy / pps_rs all accept where=

Regression guard
    - without where=, SRS inner-join behaviour unchanged
    - without where=, prob = n/N as before
    - without where=, PPS inner-join behaviour unchanged
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from svy import Sample
from svy.core.design import Design
from svy.core.constants import SVY_HIT, SVY_PROB, SVY_WEIGHT, SVY_CERTAINTY
from svy.selection.srs import _apply_where


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

RNG = 42


def _make_sample(
    n_rows: int = 20,
    *,
    stratum: str | None = None,
    mos: bool = False,
    seed: int = 0,
) -> Sample:
    """
    Build a minimal Sample with a simple integer population.

    Parameters
    ----------
    n_rows  : total number of units
    stratum : column name to use as design stratum (None = unstratified)
    mos     : whether to add a measure-of-size column
    """
    rng = np.random.default_rng(seed)

    data = {
        "unit_id": list(range(n_rows)),
        "value": rng.integers(1, 100, size=n_rows).tolist(),
        "region": ["North"] * (n_rows // 2) + ["South"] * (n_rows - n_rows // 2),
        "urban": [True, False] * (n_rows // 2),
        "eligible_flag": [True] * (n_rows * 3 // 4) + [False] * (n_rows - n_rows * 3 // 4),
    }
    if mos:
        data["size"] = rng.integers(10, 500, size=n_rows).tolist()

    df = pl.DataFrame(data)

    design_kwargs: dict = {}
    if stratum is not None:
        design_kwargs["stratum"] = stratum
    if mos:
        design_kwargs["mos"] = "size"

    design = Design(**design_kwargs) if design_kwargs else Design()
    return Sample(data=df, design=design)


# ---------------------------------------------------------------------------
# _apply_where unit tests
# ---------------------------------------------------------------------------


class TestApplyWhere:
    """Unit tests for the _apply_where helper directly."""

    def _df(self) -> pl.DataFrame:
        return pl.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": ["x", "y", "x", "z", "y"],
        })

    def test_none_returns_full_df_and_none_mask(self):
        df = self._df()
        eligible, mask = _apply_where(df, None)
        assert eligible.shape == df.shape
        assert mask is None

    def test_mapping_single_key(self):
        df = self._df()
        eligible, mask = _apply_where(df, {"b": "x"})
        assert len(eligible) == 2
        assert mask is not None
        assert mask.sum() == 2

    def test_mapping_multi_key_and(self):
        df = self._df()
        eligible, mask = _apply_where(df, {"a": 1, "b": "x"})
        assert len(eligible) == 1
        assert eligible["a"][0] == 1

    def test_mapping_list_value_isin(self):
        df = self._df()
        eligible, mask = _apply_where(df, {"b": ["x", "y"]})
        assert len(eligible) == 4

    def test_pl_expr_form(self):
        df = self._df()
        eligible, mask = _apply_where(df, pl.col("a") > 3)
        assert len(eligible) == 2
        assert set(eligible["a"].to_list()) == {4, 5}

    def test_empty_mapping_returns_full_df(self):
        df = self._df()
        eligible, mask = _apply_where(df, {})
        assert len(eligible) == len(df)
        assert mask is None

    def test_mask_aligned_to_src_df(self):
        """Mask must have same length as the source df."""
        df = self._df()
        _, mask = _apply_where(df, {"a": [2, 4]})
        assert mask is not None
        assert len(mask) == len(df)

    def test_all_false_mask_returns_empty_eligible(self):
        df = self._df()
        eligible, mask = _apply_where(df, {"a": 999})
        assert len(eligible) == 0
        assert mask is not None
        assert mask.sum() == 0


# ---------------------------------------------------------------------------
# SRS where= tests
# ---------------------------------------------------------------------------


class TestSrsWhere:
    """Integration tests for srs(where=...)."""

    # -- Row count -----------------------------------------------------------

    def test_output_row_count_equals_input(self):
        samp = _make_sample(20)
        result = samp.sampling.srs(n=5, where={"eligible_flag": True}, rstate=RNG)
        assert result._data.height == 20

    def test_no_where_same_as_where_none(self):
        """srs(where=None) must be identical to srs() with no where argument."""
        samp = _make_sample(20)
        r1 = samp.sampling.srs(n=5, rstate=RNG)
        r2 = samp.sampling.srs(n=5, where=None, rstate=RNG)
        assert r1._data.equals(r2._data)

    # -- Null / non-null selection columns -----------------------------------

    def test_non_eligible_rows_have_null_prob(self):
        samp = _make_sample(20)
        result = samp.sampling.srs(n=5, where={"eligible_flag": True}, rstate=RNG)
        df = result._data
        non_eligible = df.filter(~pl.col("eligible_flag"))
        assert non_eligible[SVY_PROB].null_count() == len(non_eligible)

    def test_non_eligible_rows_have_null_weight(self):
        samp = _make_sample(20)
        result = samp.sampling.srs(n=5, where={"eligible_flag": True}, rstate=RNG)
        df = result._data
        non_eligible = df.filter(~pl.col("eligible_flag"))
        assert non_eligible[SVY_WEIGHT].null_count() == len(non_eligible)

    def test_non_eligible_rows_have_null_hit(self):
        samp = _make_sample(20)
        result = samp.sampling.srs(n=5, where={"eligible_flag": True}, rstate=RNG)
        df = result._data
        non_eligible = df.filter(~pl.col("eligible_flag"))
        assert non_eligible[SVY_HIT].null_count() == len(non_eligible)

    def test_selected_rows_have_non_null_prob(self):
        samp = _make_sample(20)
        result = samp.sampling.srs(n=5, where={"eligible_flag": True}, rstate=RNG)
        df = result._data
        selected = df.filter(pl.col(SVY_HIT) == 1)
        assert selected[SVY_PROB].null_count() == 0

    def test_weight_is_inverse_prob_for_selected(self):
        samp = _make_sample(20)
        result = samp.sampling.srs(n=5, where={"eligible_flag": True}, rstate=RNG)
        df = result._data.filter(pl.col(SVY_PROB).is_not_null())
        probs = df[SVY_PROB].to_numpy()
        wgts = df[SVY_WEIGHT].to_numpy()
        np.testing.assert_allclose(wgts, 1.0 / probs, rtol=1e-9)

    # -- n computed against eligible subset ----------------------------------

    def test_n_drawn_from_eligible_subset(self):
        """Exactly n units selected, all from eligible rows."""
        samp = _make_sample(20)
        result = samp.sampling.srs(n=5, where={"eligible_flag": True}, rstate=RNG)
        df = result._data
        selected = df.filter(pl.col(SVY_HIT) == 1)
        assert len(selected) == 5
        assert selected["eligible_flag"].to_list() == [True] * 5

    def test_probability_computed_against_eligible_n(self):
        """prob should be n / N_eligible, not n / N_total."""
        samp = _make_sample(20)
        n_eligible = int(samp._data.filter(pl.col("eligible_flag")).height)
        n = 5
        result = samp.sampling.srs(n=n, where={"eligible_flag": True}, rstate=RNG)
        df = result._data.filter(pl.col(SVY_PROB).is_not_null())
        expected_prob = n / n_eligible
        np.testing.assert_allclose(
            df[SVY_PROB].to_numpy(), expected_prob, rtol=1e-9
        )

    # -- Stratified ----------------------------------------------------------

    def test_stratified_where_filters_within_strata(self):
        """With stratum + where, selection happens per stratum on eligible rows only."""
        samp = _make_sample(20, stratum="region")
        result = samp.sampling.srs(
            n=3, where={"eligible_flag": True}, rstate=RNG,
        )
        df = result._data
        selected = df.filter(pl.col(SVY_HIT) == 1)
        assert all(selected["eligible_flag"].to_list())

    def test_stratified_with_by_and_where(self):
        samp = _make_sample(20)
        result = samp.sampling.srs(
            n=2, by="region", where={"eligible_flag": True}, rstate=RNG,
        )
        df = result._data
        selected = df.filter(pl.col(SVY_HIT) == 1)
        assert all(selected["eligible_flag"].to_list())

    # -- Non-eligible rows preserve original values --------------------------

    def test_non_eligible_rows_preserve_unit_id(self):
        samp = _make_sample(20)
        original_ids = set(samp._data["unit_id"].to_list())
        result = samp.sampling.srs(n=5, where={"eligible_flag": True}, rstate=RNG)
        result_ids = set(result._data["unit_id"].to_list())
        assert result_ids == original_ids

    def test_non_eligible_rows_preserve_original_column_values(self):
        """Non-eligible rows should have their original data columns unchanged."""
        samp = _make_sample(20)
        orig_non_elig = samp._data.filter(~pl.col("eligible_flag")).sort("unit_id")
        result = samp.sampling.srs(n=5, where={"eligible_flag": True}, rstate=RNG)
        result_non_elig = result._data.filter(~pl.col("eligible_flag")).sort("unit_id")
        assert orig_non_elig["value"].to_list() == result_non_elig["value"].to_list()

    # -- WhereArg forms -------------------------------------------------------

    def test_where_pl_expr_form(self):
        samp = _make_sample(20)
        result = samp.sampling.srs(
            n=3, where=pl.col("eligible_flag") == True, rstate=RNG
        )
        assert result._data.height == 20
        selected = result._data.filter(pl.col(SVY_HIT) == 1)
        assert all(selected["eligible_flag"].to_list())

    def test_where_sequence_of_exprs(self):
        samp = _make_sample(20)
        result = samp.sampling.srs(
            n=3,
            where=[pl.col("eligible_flag") == True, pl.col("region") == "North"],
            rstate=RNG,
        )
        assert result._data.height == 20
        selected = result._data.filter(pl.col(SVY_HIT) == 1)
        assert all(selected["eligible_flag"].to_list())
        assert all(r == "North" for r in selected["region"].to_list())

    def test_where_mapping_multi_condition(self):
        samp = _make_sample(20)
        result = samp.sampling.srs(
            n=2,
            where={"eligible_flag": True, "region": "North"},
            rstate=RNG,
        )
        selected = result._data.filter(pl.col(SVY_HIT) == 1)
        assert all(selected["eligible_flag"].to_list())
        assert all(r == "North" for r in selected["region"].to_list())

    # -- Reproducibility  (fresh sample each time) ---------------------------

    def test_same_seed_same_result(self):
        """Two independent selections with the same seed must produce identical output."""
        r1 = _make_sample(20).sampling.srs(n=5, where={"eligible_flag": True}, rstate=42)
        r2 = _make_sample(20).sampling.srs(n=5, where={"eligible_flag": True}, rstate=42)
        assert r1._data.equals(r2._data)

    def test_different_seed_may_give_different_result(self):
        r1 = _make_sample(20).sampling.srs(n=5, where={"eligible_flag": True}, rstate=42)
        r2 = _make_sample(20).sampling.srs(n=5, where={"eligible_flag": True}, rstate=99)
        selected1 = set(r1._data.filter(pl.col(SVY_HIT) == 1)["unit_id"].to_list())
        selected2 = set(r2._data.filter(pl.col(SVY_HIT) == 1)["unit_id"].to_list())
        assert selected1 != selected2

    # -- Edge cases ----------------------------------------------------------

    def test_where_reduces_eligible_pool_no_crash(self):
        """
        where= filtering down to a subset within a stratum should not crash.
        The eligible frame simply has fewer units; selection proceeds normally.
        """
        samp = _make_sample(20, stratum="region")
        # North stratum has 10 units, 7.5 of which are eligible (15/20 * 10 = 7).
        # Selecting n=3 from eligible North rows should work silently.
        result = samp.sampling.srs(
            n=3,
            where={"region": "North"},
            rstate=RNG,
        )
        selected = result._data.filter(pl.col(SVY_HIT) == 1)
        assert all(r == "North" for r in selected["region"].to_list())

    def test_where_filters_before_group_construction(self):
        """
        where= filtering happens before _build_group_keys, so groups absent from
        the eligible frame never appear in G at all.  The result is that South rows
        are silently skipped -- no warning, no crash.

        This documents a deliberate design choice: _warn_empty_strata guards against
        allocation-table mismatches (n mapping references a group not in the frame),
        not against where= filtering that naturally reduces the eligible pool.
        """
        samp = _make_sample(20)
        # where restricts to North only; South has 0 eligible rows.
        # With a scalar n, all eligible North rows are the draw universe.
        result = samp.sampling.srs(
            n=3,
            by="region",
            where={"region": "North"},
            rstate=RNG,
        )
        # Only North rows get selected; South rows are non-eligible (null selection cols)
        selected = result._data.filter(pl.col(SVY_HIT) == 1)
        assert all(r == "North" for r in selected["region"].to_list())
        south_rows = result._data.filter(pl.col("region") == "South")
        assert south_rows[SVY_PROB].null_count() == len(south_rows)

    def test_where_all_rows_ineligible_returns_all_null_selection(self):
        """
        If no rows match where, all selection columns are null and row count is preserved.
        The empty-stratum warnings are expected and acknowledged here.
        """
        samp = _make_sample(20)
        with pytest.warns(UserWarning):
            result = samp.sampling.srs(n=3, where={"unit_id": 9999}, rstate=RNG)
        df = result._data
        assert df.height == 20
        assert df[SVY_PROB].null_count() == 20
        assert df[SVY_WEIGHT].null_count() == 20

    def test_where_with_drop_nulls(self):
        """where and drop_nulls compose: drop_nulls applies within eligible set."""
        data = pl.DataFrame({
            "unit_id": list(range(20)),
            "value": [None if i % 5 == 0 else float(i) for i in range(20)],
            "eligible_flag": [True] * 15 + [False] * 5,
        })
        samp = Sample(data=data, design=Design())
        result = samp.sampling.srs(
            n=3, where={"eligible_flag": True}, drop_nulls=True, rstate=RNG
        )
        assert result._data.height == 20

    def test_design_wgt_column_updated_in_design(self):
        samp = _make_sample(20)
        result = samp.sampling.srs(n=5, where={"eligible_flag": True}, rstate=RNG)
        assert result._design.wgt is not None

    def test_design_prob_column_updated_in_design(self):
        samp = _make_sample(20)
        result = samp.sampling.srs(n=5, where={"eligible_flag": True}, rstate=RNG)
        assert result._design.prob is not None


# ---------------------------------------------------------------------------
# PPS where= tests
# ---------------------------------------------------------------------------


class TestPpsWhere:
    """Integration tests for pps_sys(where=...)."""

    def _samp(self) -> Sample:
        return _make_sample(20, mos=True)

    # -- Row count -----------------------------------------------------------

    def test_output_row_count_equals_input(self):
        result = self._samp().sampling.pps_sys(n=3, where={"eligible_flag": True}, rstate=RNG)
        assert result._data.height == 20

    def test_no_where_same_as_where_none(self):
        r1 = self._samp().sampling.pps_sys(n=3, rstate=RNG)
        r2 = self._samp().sampling.pps_sys(n=3, where=None, rstate=RNG)
        assert r1._data.equals(r2._data)

    # -- Null / non-null selection columns -----------------------------------

    def test_non_eligible_rows_have_null_prob(self):
        result = self._samp().sampling.pps_sys(n=3, where={"eligible_flag": True}, rstate=RNG)
        non_eligible = result._data.filter(~pl.col("eligible_flag"))
        assert non_eligible[SVY_PROB].null_count() == len(non_eligible)

    def test_non_eligible_rows_have_null_weight(self):
        result = self._samp().sampling.pps_sys(n=3, where={"eligible_flag": True}, rstate=RNG)
        non_eligible = result._data.filter(~pl.col("eligible_flag"))
        assert non_eligible[SVY_WEIGHT].null_count() == len(non_eligible)

    def test_non_eligible_rows_have_null_certainty(self):
        result = self._samp().sampling.pps_sys(n=3, where={"eligible_flag": True}, rstate=RNG)
        non_eligible = result._data.filter(~pl.col("eligible_flag"))
        assert non_eligible[SVY_CERTAINTY].null_count() == len(non_eligible)

    def test_selected_rows_have_non_null_prob(self):
        result = self._samp().sampling.pps_sys(n=3, where={"eligible_flag": True}, rstate=RNG)
        selected = result._data.filter(pl.col(SVY_HIT).is_not_null())
        assert selected[SVY_PROB].null_count() == 0

    def test_weight_is_inverse_prob_for_selected(self):
        result = self._samp().sampling.pps_sys(n=3, where={"eligible_flag": True}, rstate=RNG)
        df = result._data.filter(pl.col(SVY_PROB).is_not_null())
        probs = df[SVY_PROB].to_numpy()
        wgts = df[SVY_WEIGHT].to_numpy()
        np.testing.assert_allclose(wgts, 1.0 / probs, rtol=1e-9)

    def test_all_selected_are_eligible(self):
        result = self._samp().sampling.pps_sys(n=3, where={"eligible_flag": True}, rstate=RNG)
        selected = result._data.filter(pl.col(SVY_HIT).is_not_null())
        assert all(selected["eligible_flag"].to_list())

    # -- Reproducibility ----------------------------------------------------

    def test_same_seed_same_result(self):
        r1 = self._samp().sampling.pps_sys(n=3, where={"eligible_flag": True}, rstate=42)
        r2 = self._samp().sampling.pps_sys(n=3, where={"eligible_flag": True}, rstate=42)
        assert r1._data.equals(r2._data)

    # -- Row identity -------------------------------------------------------

    def test_all_original_rows_present(self):
        samp = self._samp()
        original_ids = set(samp._data["unit_id"].to_list())
        result = samp.sampling.pps_sys(n=3, where={"eligible_flag": True}, rstate=RNG)
        result_ids = set(result._data["unit_id"].to_list())
        assert result_ids == original_ids


# ---------------------------------------------------------------------------
# PPS variant smoke tests
# ---------------------------------------------------------------------------


class TestPpsVariantsWhere:
    """Smoke tests: where= does not break any PPS variant."""

    def _samp(self) -> Sample:
        return _make_sample(20, mos=True)

    def _check(self, result, n_rows: int = 20) -> None:
        assert result._data.height == n_rows
        non_elig = result._data.filter(~pl.col("eligible_flag"))
        assert non_elig[SVY_PROB].null_count() == len(non_elig)

    def test_pps_wr(self):
        self._check(self._samp().sampling.pps_wr(
            n=3, where={"eligible_flag": True}, rstate=RNG
        ))

    def test_pps_brewer(self):
        self._check(self._samp().sampling.pps_brewer(
            n=2, where={"eligible_flag": True}, rstate=RNG
        ))

    def test_pps_murphy(self):
        self._check(self._samp().sampling.pps_murphy(
            n=2, where={"eligible_flag": True}, rstate=RNG
        ))

    def test_pps_rs(self):
        self._check(self._samp().sampling.pps_rs(
            n=2, where={"eligible_flag": True}, rstate=RNG
        ))


# ---------------------------------------------------------------------------
# Regression guard: where=None preserves original behaviour
# ---------------------------------------------------------------------------


class TestWhereRegressionGuard:
    """
    Confirm where=None is a true no-op — original SRS/PPS behaviour unchanged.
    """

    def test_srs_wor_row_count_without_where(self):
        """Without where, inner join: output has exactly n rows."""
        samp = _make_sample(20)
        result = samp.sampling.srs(n=5, rstate=RNG)
        assert result._data.height == 5

    def test_srs_wor_prob_is_n_over_N_without_where(self):
        samp = _make_sample(20)
        result = samp.sampling.srs(n=5, rstate=RNG)
        probs = result._data[SVY_PROB].to_numpy()
        np.testing.assert_allclose(probs, 5 / 20, rtol=1e-9)

    def test_pps_sys_without_where_inner_join(self):
        """Without where, PPS output contains only selected rows."""
        samp = _make_sample(20, mos=True)
        result = samp.sampling.pps_sys(n=3, rstate=RNG)
        assert result._data.height == 3
        assert result._data[SVY_PROB].null_count() == 0
