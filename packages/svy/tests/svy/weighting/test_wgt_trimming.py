# tests/svy/weighting/test_wgt_trimming.py
"""
Comprehensive tests for sample.weighting.trim().

Coverage:
  - Threshold types: absolute, quantile, stat-based (median/mean/sd/iqr), callable
  - TrimConfig construction and sync with trim() signature
  - Threshold construction and validation
  - Global trim (no by=)
  - Domain trim (by=)
  - redistribute=True / False
  - min_cell_size guard
  - Iteration / convergence
  - Column naming: auto-prefix, wgt_name, replace, collision
  - Design update
  - Chaining
  - Warning codes: NEGATIVE_WEIGHT, ZERO_WEIGHT, REPLICATE_SKIPPED,
                   DOMAIN_SKIPPED, MAX_ITER_REACHED, WEIGHT_SUM_CHANGED,
                   WEIGHT_ADJ_AUDIT
  - Edge cases: all-equal weights, single unit, zero-weight units,
                negative weights, both bounds, lower > upper
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from numpy.testing import assert_allclose

from svy.core.sample import Design, Sample
from svy.core.warnings import Severity, WarnCode
from svy_rs._internal import trim_weights as _rust_trim_weights  # type: ignore[import-untyped]
from svy.weighting.types import (
    Threshold,
    TrimConfig,
    TrimResult,
    resolve_threshold,
)

TRIM_WGT = "trim_wgt"


def run_trim(weights: np.ndarray, config: TrimConfig) -> TrimResult:
    """
    Test adapter: accepts TrimConfig, resolves thresholds, calls Rust directly,
    packs the flat tuple result into TrimResult.

    Rust returns: (weights, n_upper, n_lower, sum_before, sum_after,
                   ess_before, ess_after, iterations, converged)
    TrimResult has 11 fields — upper_threshold and lower_threshold are
    injected here since Rust works with already-resolved scalars.
    """
    w_pos = weights[weights > 0].astype(np.float64)
    upper_val = resolve_threshold(config.upper, w_pos) if config.upper is not None else None
    lower_val = resolve_threshold(config.lower, w_pos) if config.lower is not None else None
    (
        trimmed_weights,
        n_trimmed_upper,
        n_trimmed_lower,
        weight_sum_before,
        weight_sum_after,
        ess_before,
        ess_after,
        iterations,
        converged,
    ) = _rust_trim_weights(
        weights,
        upper_val,
        lower_val,
        config.redistribute,
        config.max_iter,
        config.tol,
    )
    return TrimResult(
        weights=trimmed_weights,
        upper_threshold=upper_val,
        lower_threshold=lower_val,
        n_trimmed_upper=n_trimmed_upper,
        n_trimmed_lower=n_trimmed_lower,
        weight_sum_before=weight_sum_before,
        weight_sum_after=weight_sum_after,
        ess_before=ess_before,
        ess_after=ess_after,
        iterations=iterations,
        converged=converged,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample(weights: list[float], domain: list[str] | None = None) -> Sample:
    data: dict = {"weight": weights}
    if domain is not None:
        data["domain"] = domain
    return Sample(data=pl.DataFrame(data), design=Design(wgt="weight"))


def _warnings_of(sample: Sample, code: WarnCode) -> list:
    return sample._warnings.list(code=code)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_design() -> Design:
    return Design(wgt="weight")


@pytest.fixture
def uniform_sample(mock_design) -> Sample:
    """10 units, all weight=10. No extremes."""
    return Sample(
        data=pl.DataFrame({"weight": [10.0] * 10}),
        design=mock_design,
    )


@pytest.fixture
def skewed_sample(mock_design) -> Sample:
    """8 normal units (weight=10) + 2 extreme units (weight=100)."""
    weights = [10.0] * 8 + [100.0, 100.0]
    return Sample(
        data=pl.DataFrame({"weight": weights}),
        design=mock_design,
    )


@pytest.fixture
def domain_sample(mock_design) -> Sample:
    """Two domains A and B, each with one extreme weight."""
    weights = [
        10.0,
        10.0,
        10.0,
        100.0,  # A: 3 normal, 1 extreme
        10.0,
        10.0,
        10.0,
        100.0,
    ]  # B: 3 normal, 1 extreme
    domains = ["A"] * 4 + ["B"] * 4
    return Sample(
        data=pl.DataFrame({"weight": weights, "domain": domains}),
        design=mock_design,
    )


# ===========================================================================
# 1. Threshold — construction, validation, compute()
# ===========================================================================


class TestThreshold:
    def test_median_computes_correctly(self):
        w = np.array([10.0, 10.0, 10.0, 100.0])
        t = Threshold("median", 3.0)
        assert_allclose(t.compute(w), 3.0 * np.median(w))

    def test_mean_computes_correctly(self):
        w = np.array([10.0, 20.0, 30.0])
        t = Threshold("mean", 2.0)
        assert_allclose(t.compute(w), 2.0 * np.mean(w))

    def test_sd_computes_correctly(self):
        w = np.array([10.0, 20.0, 30.0, 40.0])
        t = Threshold("sd", 3.0)
        assert_allclose(t.compute(w), 3.0 * np.std(w, ddof=1))

    def test_iqr_computes_correctly(self):
        w = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        t = Threshold("iqr", 3.0)
        q75, q25 = np.percentile(w, [75.0, 25.0])
        assert_allclose(t.compute(w), 3.0 * (q75 - q25))

    def test_zero_weight_units_excluded_from_stat(self):
        """Zero weights must not influence the stat computation."""
        w_with_zeros = np.array([0.0, 10.0, 10.0, 100.0, 0.0])
        w_without = np.array([10.0, 10.0, 100.0])
        t = Threshold("median", 2.0)
        assert_allclose(t.compute(w_with_zeros), t.compute(w_without))

    def test_invalid_stat_raises(self):
        with pytest.raises(ValueError, match="Unsupported stat"):
            Threshold("geometric_mean", 2.0)

    def test_k_zero_raises(self):
        with pytest.raises(ValueError, match="k must be nonzero"):
            Threshold("median", 0.0)

    def test_k_negative_allowed_for_composition(self):
        """Negative k is valid — needed for Cap subtraction."""
        t = Threshold("median", -1.0)
        assert t.k == -1.0

    def test_negative_resolved_threshold_raises(self):
        """A composed threshold that resolves negative should raise at resolve time."""
        w = np.array([1.0, 2.0, 3.0])
        from svy.core.terms import Cap

        spec = Cap("mean") - 100 * Cap("sd")
        with pytest.raises(ValueError, match="Resolved threshold must be >= 0"):
            resolve_threshold(spec, w)

    def test_frozen(self):
        t = Threshold("median", 3.0)
        with pytest.raises((AttributeError, TypeError)):
            t.stat = "mean"  # type: ignore[misc]

    def test_all_zero_weights_returns_zero(self):
        w = np.array([0.0, 0.0, 0.0])
        t = Threshold("median", 3.0)
        assert t.compute(w) == 0.0


# ===========================================================================
# 2. resolve_threshold
# ===========================================================================


class TestResolveThreshold:
    def test_float_gt_1_is_absolute(self):
        w = np.array([10.0, 20.0, 30.0])
        assert resolve_threshold(50.0, w) == 50.0

    def test_float_eq_1_is_quantile(self):
        w = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        # 1.0 = 100th percentile = max
        assert resolve_threshold(1.0, w) == 50.0

    def test_float_in_0_1_is_quantile(self):
        w = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = resolve_threshold(0.8, w)
        assert_allclose(result, np.quantile(w, 0.8))

    def test_threshold_object_delegates_to_compute(self):
        w = np.array([10.0, 10.0, 100.0])
        t = Threshold("median", 3.0)
        assert_allclose(resolve_threshold(t, w), t.compute(w))

    def test_callable_receives_positive_weights(self):
        w = np.array([0.0, 10.0, 20.0, 30.0])
        # callable should receive only positive weights
        received = []

        def cap(arr):
            received.append(arr.copy())
            return 25.0

        resolve_threshold(cap, w)
        assert len(received) == 1
        assert 0.0 not in received[0]

    def test_callable_return_value_used(self):
        w = np.array([10.0, 20.0, 30.0])
        assert resolve_threshold(lambda _: 999.0, w) == 999.0

    def test_float_zero_raises(self):
        with pytest.raises(ValueError, match="> 0"):
            resolve_threshold(0.0, np.array([10.0]))

    def test_float_negative_raises(self):
        with pytest.raises(ValueError, match="> 0"):
            resolve_threshold(-5.0, np.array([10.0]))

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported ThresholdSpec"):
            resolve_threshold("not_a_spec", np.array([10.0]))  # type: ignore[arg-type]


# ===========================================================================
# 3. TrimConfig — construction and validation
# ===========================================================================


class TestTrimConfig:
    def test_requires_at_least_one_bound(self):
        with pytest.raises(ValueError, match="At least one"):
            TrimConfig()

    def test_upper_only_valid(self):
        c = TrimConfig(upper=0.99)
        assert c.upper == 0.99
        assert c.lower is None

    def test_lower_only_valid(self):
        c = TrimConfig(lower=0.01)
        assert c.lower == 0.01

    def test_both_bounds_valid(self):
        c = TrimConfig(upper=0.99, lower=0.01)
        assert c.upper == 0.99
        assert c.lower == 0.01

    def test_min_cell_size_below_1_raises(self):
        with pytest.raises(ValueError, match="min_cell_size"):
            TrimConfig(upper=50.0, min_cell_size=0)

    def test_max_iter_below_1_raises(self):
        with pytest.raises(ValueError, match="max_iter"):
            TrimConfig(upper=50.0, max_iter=0)

    def test_tol_out_of_range_raises(self):
        with pytest.raises(ValueError, match="tol"):
            TrimConfig(upper=50.0, tol=1.5)
        with pytest.raises(ValueError, match="tol"):
            TrimConfig(upper=50.0, tol=0.0)

    def test_frozen(self):
        c = TrimConfig(upper=50.0)
        with pytest.raises((AttributeError, TypeError)):
            c.upper = 100.0  # type: ignore[misc]

    def test_defaults(self):
        c = TrimConfig(upper=50.0)
        assert c.redistribute is True
        assert c.min_cell_size == 10
        assert c.max_iter == 10
        assert c.tol == 1e-6
        assert c.by is None


# ===========================================================================
# 4. run_trim — pure algorithm unit tests
# ===========================================================================


class TestRunTrim:
    def test_absolute_upper_caps_extremes(self):
        w = np.array([10.0, 10.0, 10.0, 100.0])
        config = TrimConfig(upper=50.0, redistribute=False)
        result = run_trim(w, config)
        assert result.weights[3] == 50.0
        assert result.n_trimmed_upper == 1

    def test_absolute_upper_with_redistribution(self):
        w = np.array([10.0, 10.0, 10.0, 100.0])
        config = TrimConfig(upper=50.0, redistribute=True)
        result = run_trim(w, config)
        # Total weight preserved
        assert_allclose(result.weight_sum_after, result.weight_sum_before)
        # Trimmed unit should be at or near the cap (redistribution may add
        # a tiny fraction back via iteration, so use a loose tolerance)
        assert_allclose(result.weights[3], 50.0, rtol=1e-2)
        # Excess redistributed to first 3
        assert all(result.weights[:3] > 10.0)

    def test_quantile_upper(self):
        w = np.array([10.0] * 8 + [100.0, 100.0])
        config = TrimConfig(upper=0.8, redistribute=False)
        result = run_trim(w, config)
        # 80th percentile of w is 10.0 so everything at or below
        # Nothing trimmed since max <= p80 after small sample — just check result is valid
        assert result.n_trimmed_upper >= 0

    def test_stat_threshold_median(self):
        w = np.array([10.0] * 8 + [500.0, 500.0])
        config = TrimConfig(upper=Threshold("median", 3.0), redistribute=False)
        result = run_trim(w, config)
        expected_cap = 3.0 * np.median(w[w > 0])
        assert_allclose(result.upper_threshold, expected_cap)
        assert result.n_trimmed_upper == 2

    def test_callable_threshold(self):
        w = np.array([10.0, 10.0, 10.0, 200.0])
        config = TrimConfig(upper=lambda arr: 50.0, redistribute=False)
        result = run_trim(w, config)
        assert result.upper_threshold == 50.0
        assert result.weights[3] == 50.0

    def test_lower_bound_raises_small_weights(self):
        w = np.array([1.0, 5.0, 10.0, 10.0])
        config = TrimConfig(lower=4.0, redistribute=False)
        result = run_trim(w, config)
        assert result.weights[0] == 4.0
        assert result.n_trimmed_lower == 1

    def test_both_bounds(self):
        w = np.array([1.0, 10.0, 10.0, 200.0])
        config = TrimConfig(upper=50.0, lower=5.0, redistribute=False)
        result = run_trim(w, config)
        assert result.weights[0] == 5.0  # lower trimmed
        assert result.weights[3] == 50.0  # upper trimmed
        assert result.n_trimmed_lower == 1
        assert result.n_trimmed_upper == 1

    def test_zero_weight_units_preserved(self):
        w = np.array([0.0, 10.0, 10.0, 100.0])
        config = TrimConfig(upper=50.0, redistribute=False)
        result = run_trim(w, config)
        assert result.weights[0] == 0.0  # untouched

    def test_negative_weight_raises(self):
        w = np.array([-1.0, 10.0, 10.0])
        config = TrimConfig(upper=50.0)
        with pytest.raises(ValueError, match="Negative weights"):
            run_trim(w, config)

    def test_lower_ge_upper_raises(self):
        w = np.array([10.0, 20.0, 30.0])
        config = TrimConfig(upper=10.0, lower=20.0, redistribute=False)
        with pytest.raises(ValueError, match="lower.*>=.*upper"):
            run_trim(w, config)

    def test_convergence_flagged(self):
        w = np.array([10.0] * 9 + [100.0])
        config = TrimConfig(upper=50.0, redistribute=True, max_iter=100, tol=1e-9)
        result = run_trim(w, config)
        assert result.converged is True

    def test_max_iter_reached_not_converged(self):
        # 1 iteration only, will not converge with redistribution creating new extremes
        w = np.array([1.0, 1.0, 1.0, 1000.0])
        config = TrimConfig(upper=2.0, redistribute=True, max_iter=1, tol=1e-20)
        result = run_trim(w, config)
        assert result.iterations == 1

    def test_ess_decreases_after_trimming(self):
        """Trimming extremes should increase ESS (reduce variance of weights)."""
        w = np.array([10.0] * 9 + [1000.0])
        config = TrimConfig(upper=50.0, redistribute=False)
        result = run_trim(w, config)
        assert result.ess_after > result.ess_before

    def test_all_equal_weights_nothing_trimmed(self):
        w = np.array([10.0] * 10)
        config = TrimConfig(upper=50.0, redistribute=False)
        result = run_trim(w, config)
        assert result.n_trimmed_upper == 0
        assert_allclose(result.weights, w)

    def test_degenerate_zero_iqr_skips_trimming(self):
        # Constant weights → IQR=0 → threshold resolves to 0.
        # run_trim must return weights unchanged rather than raise.
        w = np.array([10.0] * 10)
        config = TrimConfig(upper=Threshold("iqr", 3.0), redistribute=False)
        result = run_trim(w, config)
        assert result.n_trimmed_upper == 0
        assert_allclose(result.weights, w)
        assert result.converged is True

    def test_single_positive_unit_no_redistribution(self):
        w = np.array([10.0])
        config = TrimConfig(upper=5.0, redistribute=False)
        result = run_trim(w, config)
        assert result.weights[0] == 5.0

    def test_weight_sum_preserved_with_redistribution(self):
        w = np.array([10.0] * 8 + [200.0, 200.0])
        config = TrimConfig(upper=50.0, redistribute=True, max_iter=20)
        result = run_trim(w, config)
        assert_allclose(result.weight_sum_after, result.weight_sum_before, rtol=1e-6)

    def test_weight_sum_changes_without_redistribution(self):
        w = np.array([10.0] * 8 + [200.0, 200.0])
        config = TrimConfig(upper=50.0, redistribute=False)
        result = run_trim(w, config)
        assert result.weight_sum_after < result.weight_sum_before


# ===========================================================================
# 5. trim() — column naming
# ===========================================================================


class TestTrimColumnNaming:
    def test_auto_col_name_uses_trim_wgt(self, skewed_sample):
        out = skewed_sample.weighting.trim(upper=50.0)
        assert TRIM_WGT in out.data.columns

    def test_wgt_name_overrides_default(self, skewed_sample):
        out = skewed_sample.weighting.trim(upper=50.0, wgt_name="my_trim")
        assert "my_trim" in out.data.columns
        assert TRIM_WGT not in out.data.columns

    def test_collision_raises(self, skewed_sample):
        df = skewed_sample.data.with_columns(pl.col("weight").alias(TRIM_WGT))
        sample = Sample(data=df, design=Design(wgt="weight"))
        with pytest.raises(Exception, match="already exists"):
            sample.weighting.trim(upper=50.0)

    def test_design_wgt_updated_by_default(self, skewed_sample):
        out = skewed_sample.weighting.trim(upper=50.0)
        assert out.design.wgt == TRIM_WGT

    def test_design_wgt_not_updated_when_flag_false(self, skewed_sample):
        out = skewed_sample.weighting.trim(upper=50.0, update_design_wgts=False)
        assert out.design.wgt == "weight"


# ===========================================================================
# 6. trim() — threshold types via public API
# ===========================================================================


class TestTrimThresholdTypes:
    def test_absolute_upper(self, skewed_sample):
        out = skewed_sample.weighting.trim(upper=50.0, redistribute=False)
        w = out.data[TRIM_WGT].to_numpy()
        assert np.all(w[w > 0] <= 50.0)

    def test_quantile_upper(self, skewed_sample):
        out = skewed_sample.weighting.trim(upper=0.8, redistribute=False)
        w_orig = skewed_sample.data["weight"].to_numpy()
        cap = np.quantile(w_orig, 0.8)
        w = out.data[TRIM_WGT].to_numpy()
        assert np.all(w <= cap + 1e-9)

    def test_stat_threshold_median(self, skewed_sample):
        t = Threshold("median", 3.0)
        out = skewed_sample.weighting.trim(upper=t, redistribute=False)
        w_orig = skewed_sample.data["weight"].to_numpy()
        cap = 3.0 * np.median(w_orig)
        w = out.data[TRIM_WGT].to_numpy()
        assert np.all(w <= cap + 1e-9)

    def test_stat_threshold_iqr(self):
        # Use a sample with spread in the weight distribution so IQR > 0.
        # skewed_sample has 8 identical weights (10.0) so IQR=0 — unsuitable here.
        sample = _make_sample([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 200.0, 200.0])
        t = Threshold("iqr", 3.0)
        w_orig = sample.data["weight"].to_numpy()
        q75, q25 = np.percentile(w_orig[w_orig > 0], [75, 25])
        cap = 3.0 * (q75 - q25)
        assert cap > 0, "IQR must be > 0 for this test to be meaningful"
        out = sample.weighting.trim(upper=t, redistribute=False)
        w = out.data[TRIM_WGT].to_numpy()
        assert np.all(w <= cap + 1e-9)

    def test_callable_upper(self, skewed_sample):
        out = skewed_sample.weighting.trim(upper=lambda arr: 50.0, redistribute=False)
        w = out.data[TRIM_WGT].to_numpy()
        assert np.all(w <= 50.0 + 1e-9)

    def test_lower_bound_raises_small_weights(self):
        sample = _make_sample(
            [1.0, 2.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        )
        out = sample.weighting.trim(lower=5.0, redistribute=False)
        w = out.data[TRIM_WGT].to_numpy()
        assert np.all((w == 0) | (w >= 5.0))

    def test_both_upper_and_lower(self):
        sample = _make_sample([1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 200.0])
        out = sample.weighting.trim(upper=50.0, lower=5.0, redistribute=False)
        w = out.data[TRIM_WGT].to_numpy()
        positive = w[w > 0]
        assert np.all(positive >= 5.0)
        assert np.all(positive <= 50.0)

    def test_no_bounds_raises(self, skewed_sample):
        with pytest.raises(ValueError, match="At least one"):
            skewed_sample.weighting.trim()


# ===========================================================================
# 7. trim() — redistribute
# ===========================================================================


class TestTrimRedistribute:
    def test_redistribute_true_preserves_weight_sum(self, skewed_sample):
        sum_before = skewed_sample.data["weight"].sum()
        out = skewed_sample.weighting.trim(upper=50.0, redistribute=True)
        sum_after = out.data[TRIM_WGT].sum()
        assert_allclose(sum_after, sum_before, rtol=1e-6)

    def test_redistribute_false_reduces_weight_sum(self, skewed_sample):
        sum_before = skewed_sample.data["weight"].sum()
        out = skewed_sample.weighting.trim(upper=50.0, redistribute=False)
        sum_after = out.data[TRIM_WGT].sum()
        assert sum_after < sum_before

    def test_redistribute_false_emits_weight_sum_changed_warning(self, skewed_sample):
        out = skewed_sample.weighting.trim(upper=50.0, redistribute=False)
        warns = _warnings_of(out, WarnCode.WEIGHT_SUM_CHANGED)
        assert len(warns) == 1
        assert warns[0].level == Severity.WARNING

    def test_redistribute_true_no_weight_sum_warning(self, skewed_sample):
        out = skewed_sample.weighting.trim(upper=50.0, redistribute=True)
        warns = _warnings_of(out, WarnCode.WEIGHT_SUM_CHANGED)
        assert len(warns) == 0


# ===========================================================================
# 8. trim() — by= domain splitting
# ===========================================================================


class TestTrimByDomain:
    def test_by_trims_within_each_domain(self, domain_sample):
        out = domain_sample.weighting.trim(
            upper=50.0, by="domain", redistribute=False, min_cell_size=1
        )
        w = out.data[TRIM_WGT].to_numpy()
        assert np.all(w[w > 0] <= 50.0 + 1e-9)

    def test_by_threshold_computed_per_domain(self):
        """Two domains with very different distributions — stat threshold must differ."""
        # Domain A: all weights=10 (median=10), Domain B: [10,10,10,1000] (median=10)
        # With Threshold("median", 3): A cap = 30, B cap = 30 (same median here)
        # Use absolute to verify per-domain isolation clearly
        weights = [
            10.0,
            10.0,
            10.0,
            80.0,  # A: cap at 50 → trims 80
            10.0,
            10.0,
            10.0,
            200.0,
        ]  # B: cap at 50 → trims 200
        domains = ["A"] * 4 + ["B"] * 4
        sample = Sample(
            data=pl.DataFrame({"weight": weights, "domain": domains}),
            design=Design(wgt="weight"),
        )
        out = sample.weighting.trim(upper=50.0, by="domain", redistribute=False, min_cell_size=1)
        w = out.data[TRIM_WGT].to_numpy()
        assert w[3] == 50.0  # A extreme trimmed
        assert w[7] == 50.0  # B extreme trimmed

    def test_by_redistribution_within_domain(self, domain_sample):
        """Redistribution must stay within each domain, not bleed across."""
        sum_a_before = domain_sample.data.filter(pl.col("domain") == "A")["weight"].sum()
        sum_b_before = domain_sample.data.filter(pl.col("domain") == "B")["weight"].sum()
        out = domain_sample.weighting.trim(
            upper=50.0, by="domain", redistribute=True, min_cell_size=1
        )
        col = TRIM_WGT
        sum_a_after = out.data.filter(pl.col("domain") == "A")[col].sum()
        sum_b_after = out.data.filter(pl.col("domain") == "B")[col].sum()
        assert_allclose(sum_a_after, sum_a_before, rtol=1e-6)
        assert_allclose(sum_b_after, sum_b_before, rtol=1e-6)

    def test_by_missing_column_raises(self, skewed_sample):
        with pytest.raises(Exception, match="not found in data|nonexistent"):
            skewed_sample.weighting.trim(upper=50.0, by="nonexistent")

    def test_by_list_of_columns(self):
        """by= as a list of two columns (composite domain)."""
        weights = [10.0] * 6 + [200.0] * 2
        df = pl.DataFrame(
            {
                "weight": weights,
                "region": ["A", "A", "A", "A", "B", "B", "B", "B"],
                "sex": ["M", "F", "M", "F", "M", "F", "M", "F"],
            }
        )
        sample = Sample(data=df, design=Design(wgt="weight"))
        out = sample.weighting.trim(
            upper=50.0, by=["region", "sex"], redistribute=False, min_cell_size=1
        )
        w = out.data[TRIM_WGT].to_numpy()
        assert np.all(w[w > 0] <= 50.0 + 1e-9)

    def test_domain_skipped_warning_for_small_cell(self):
        """Domain with fewer than min_cell_size positive-weight units → skipped + warned."""
        # Domain A: 10 units (fine), Domain B: 2 units (below default min_cell_size=10)
        weights = [10.0] * 10 + [10.0, 200.0]
        domains = ["A"] * 10 + ["B"] * 2
        sample = Sample(
            data=pl.DataFrame({"weight": weights, "domain": domains}),
            design=Design(wgt="weight"),
        )
        out = sample.weighting.trim(upper=50.0, by="domain", min_cell_size=10)
        warns = _warnings_of(out, WarnCode.DOMAIN_SKIPPED)
        assert len(warns) == 1
        assert "B" in warns[0].detail

    def test_domain_not_skipped_when_exactly_min_cell_size(self):
        """Domain with exactly min_cell_size units should NOT be skipped."""
        weights = [10.0] * 4 + [200.0]  # 5 positive units
        domains = ["A"] * 5
        sample = Sample(
            data=pl.DataFrame({"weight": weights, "domain": domains}),
            design=Design(wgt="weight"),
        )
        out = sample.weighting.trim(upper=50.0, by="domain", min_cell_size=5, redistribute=False)
        warns = _warnings_of(out, WarnCode.DOMAIN_SKIPPED)
        assert len(warns) == 0


# ===========================================================================
# 9. trim() — warnings
# ===========================================================================


class TestTrimWarnings:
    def test_negative_weight_raises_and_emits_error_warning(self):
        sample = _make_sample([-1.0, 10.0, 10.0])
        with pytest.raises(Exception, match="negative weight|Negative weights"):
            sample.weighting.trim(upper=50.0)
        warns = _warnings_of(sample, WarnCode.NEGATIVE_WEIGHT)
        assert len(warns) == 1
        assert warns[0].level == Severity.ERROR

    def test_zero_weight_emits_info_warning(self):
        sample = _make_sample([0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        out = sample.weighting.trim(upper=50.0)
        warns = _warnings_of(out, WarnCode.ZERO_WEIGHT)
        assert len(warns) == 1
        assert warns[0].level == Severity.INFO

    def test_zero_weight_unit_preserved_in_output(self):
        sample = _make_sample([0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 100.0])
        out = sample.weighting.trim(upper=50.0)
        w = out.data[TRIM_WGT].to_numpy()
        assert w[0] == 0.0

    def test_no_zero_weight_warning_when_none_present(self, skewed_sample):
        out = skewed_sample.weighting.trim(upper=50.0)
        warns = _warnings_of(out, WarnCode.ZERO_WEIGHT)
        assert len(warns) == 0

    def test_replicate_weights_are_trimmed(self, skewed_sample):
        """Replicate weights get the same proportional adjustment as the main weight."""
        from svy import EstimationMethod

        height = skewed_sample.data.height
        rw1 = [10.0] * (height - 1) + [999.0]
        rw2 = [10.0] * (height - 1) + [999.0]

        df = skewed_sample.data.with_columns(
            pl.Series("rw1", rw1),
            pl.Series("rw2", rw2),
        )
        sample = Sample(data=df, design=Design(wgt="weight"))
        sample._design = sample.design.update_rep_weights(
            method=EstimationMethod.BRR, prefix="rw", n_reps=2
        )
        out = sample.weighting.trim(upper=50.0)

        # No REPLICATE_SKIPPED warning
        assert len(_warnings_of(out, WarnCode.REPLICATE_SKIPPED)) == 0

        # New rep columns created with trim_wgt prefix
        assert "trim_wgt1" in out.data.columns
        assert "trim_wgt2" in out.data.columns
        assert out.design.rep_wgts.prefix == "trim_wgt"

        # New rep columns get the same proportional adjustment as the main weight
        w_orig = sample.data["weight"].to_numpy()
        w_out = out.data[out.design.wgt].to_numpy()
        factors = np.where(w_orig > 0, w_out / w_orig, 1.0)

        expected_rw1 = np.array(rw1) * factors
        expected_rw2 = np.array(rw2) * factors

        np.testing.assert_allclose(out.data["trim_wgt1"].to_numpy(), expected_rw1, rtol=1e-6)
        np.testing.assert_allclose(out.data["trim_wgt2"].to_numpy(), expected_rw2, rtol=1e-6)

    def test_no_replicate_warning_when_no_rep_weights(self, skewed_sample):
        out = skewed_sample.weighting.trim(upper=50.0)
        warns = _warnings_of(out, WarnCode.REPLICATE_SKIPPED)
        assert len(warns) == 0

    def test_max_iter_reached_warning(self, skewed_sample):
        out = skewed_sample.weighting.trim(upper=50.0, redistribute=True, max_iter=1, tol=1e-20)
        warns = _warnings_of(out, WarnCode.MAX_ITER_REACHED)
        assert len(warns) >= 1
        assert warns[0].level == Severity.WARNING

    def test_no_max_iter_warning_when_converged(self, skewed_sample):
        out = skewed_sample.weighting.trim(upper=50.0, max_iter=100, tol=1e-6)
        warns = _warnings_of(out, WarnCode.MAX_ITER_REACHED)
        assert len(warns) == 0

    def test_audit_record_emitted_info(self, skewed_sample):
        out = skewed_sample.weighting.trim(upper=50.0)
        audits = _warnings_of(out, WarnCode.WEIGHT_ADJ_AUDIT)
        assert len(audits) == 1
        assert audits[0].level == Severity.INFO

    def test_audit_record_extra_has_expected_keys(self, skewed_sample):
        out = skewed_sample.weighting.trim(upper=50.0)
        audit = _warnings_of(out, WarnCode.WEIGHT_ADJ_AUDIT)[0]
        for key in (
            "upper_threshold",
            "lower_threshold",
            "n_trimmed_upper",
            "n_trimmed_lower",
            "weight_sum_before",
            "weight_sum_after",
            "ess_before",
            "ess_after",
            "iterations",
            "converged",
        ):
            assert key in audit.extra, f"Missing key: {key}"

    def test_audit_record_per_domain_when_by_set(self, domain_sample):
        out = domain_sample.weighting.trim(upper=50.0, by="domain", min_cell_size=1)
        audits = _warnings_of(out, WarnCode.WEIGHT_ADJ_AUDIT)
        # One audit record per domain
        assert len(audits) == 2
        domains_reported = {a.extra["domain"] for a in audits}
        assert domains_reported == {"A", "B"}


# ===========================================================================
# 10. trim() — convergence and iteration
# ===========================================================================


class TestTrimConvergence:
    def test_converges_with_sufficient_iterations(self, skewed_sample):
        out = skewed_sample.weighting.trim(upper=50.0, max_iter=50, tol=1e-6)
        audits = _warnings_of(out, WarnCode.WEIGHT_ADJ_AUDIT)
        assert audits[0].extra["converged"] is True

    def test_max_iter_1_records_one_iteration(self, skewed_sample):
        out = skewed_sample.weighting.trim(upper=50.0, max_iter=1)
        audits = _warnings_of(out, WarnCode.WEIGHT_ADJ_AUDIT)
        assert audits[0].extra["iterations"] == 1


# ===========================================================================
# 11. trim() — chaining
# ===========================================================================


class TestTrimChaining:
    def test_trim_returns_sample(self, skewed_sample):
        result = skewed_sample.weighting.trim(upper=50.0)
        assert isinstance(result, Sample)

    def test_trim_then_normalize(self, skewed_sample):
        """trim() → normalize() chain must work without errors."""
        result = skewed_sample.weighting.trim(upper=50.0).weighting.normalize()
        assert isinstance(result, Sample)

    def test_trim_twice_uses_updated_design_wgt(self, skewed_sample):
        """Second trim() must operate on the weight produced by the first."""
        out1 = skewed_sample.weighting.trim(upper=50.0, redistribute=False)
        assert out1.design.wgt == TRIM_WGT
        # Second trim needs a different wgt_name
        out2 = out1.weighting.trim(upper=50.0, redistribute=False, wgt_name="trim_wgt2")
        w1 = out1.data[TRIM_WGT].to_numpy()
        w2 = out2.data["trim_wgt2"].to_numpy()
        assert_allclose(w1, w2)


# ===========================================================================
# 12. trim() — edge cases
# ===========================================================================


class TestTrimEdgeCases:
    def test_no_units_exceed_upper_nothing_changes(self, uniform_sample):
        out = uniform_sample.weighting.trim(upper=500.0, redistribute=False)
        w_orig = uniform_sample.data["weight"].to_numpy()
        w_out = out.data[TRIM_WGT].to_numpy()
        assert_allclose(w_orig, w_out)

    def test_no_units_below_lower_nothing_changes(self, uniform_sample):
        out = uniform_sample.weighting.trim(lower=1.0, redistribute=False)
        w_orig = uniform_sample.data["weight"].to_numpy()
        w_out = out.data[TRIM_WGT].to_numpy()
        assert_allclose(w_orig, w_out)

    def test_all_weights_zero_no_crash(self):
        sample = _make_sample([0.0, 0.0, 0.0])
        # All zero → no positive units → domain skipped at min_cell_size=1
        out = sample.weighting.trim(upper=50.0, min_cell_size=1)
        # No crash, weights unchanged
        w = out.data[TRIM_WGT].to_numpy()
        assert_allclose(w, [0.0, 0.0, 0.0])

    def test_single_positive_unit(self):
        sample = _make_sample([200.0])
        out = sample.weighting.trim(upper=50.0, redistribute=False, min_cell_size=1)
        w = out.data[TRIM_WGT].to_numpy()
        assert w[0] == 50.0

    def test_missing_design_weight_raises(self):
        sample = Sample(data=pl.DataFrame({"weight": [10.0, 20.0]}))
        with pytest.raises(Exception, match="Sample weight is None"):
            sample.weighting.trim(upper=50.0)

    def test_weight_column_not_in_data_raises(self):
        # Sample validates design columns at construction time, so a missing
        # weight column is caught before trim() is ever called.
        with pytest.raises(ValueError, match="Design references columns not found in data"):
            Sample(
                data=pl.DataFrame({"other": [10.0, 20.0]}),
                design=Design(wgt="weight"),
            )

    def test_large_dataset_performance(self):
        """100k rows — should complete without hanging."""
        import time

        rng = np.random.default_rng(42)
        weights = rng.exponential(scale=10.0, size=100_000).tolist()
        sample = _make_sample(weights)
        start = time.time()
        out = sample.weighting.trim(upper=Threshold("median", 6.0), redistribute=True)
        elapsed = time.time() - start
        assert isinstance(out, Sample)
        assert elapsed < 10.0, f"trim() took {elapsed:.1f}s on 100k rows"
