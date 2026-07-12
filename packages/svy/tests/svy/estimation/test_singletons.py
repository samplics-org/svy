# tests/svy/estimation/test_singletons.py
"""
Singleton handling at the ESTIMATION level.

Singletons (strata with a single PSU) are chosen/handled at the *sample* level
(``sample.singleton.*``), but they only become a problem when a variance is
actually computed. Estimation is therefore the decision point: when it hits an
unhandled singleton it must either adopt a chosen method or fail.

Target behavior exercised here (mirrors R's ``options(survey.lonely.psu="fail")``):

  * Default, no strategy chosen  -> estimation raises ``SingletonError``.
  * A strategy chosen at the sample level -> estimation proceeds.
  * A design with no singletons  -> estimation proceeds with no handling.

Scenarios are parametrised across the estimation methods (mean, total, prop,
ratio, median) so the policy is uniform across the API surface.
"""

from __future__ import annotations

import polars as pl
import pytest

import svy

from svy.errors.singleton_errors import SingletonError


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════
#
# Strata A and C have a single PSU each (singletons); B and D have two PSUs.
# Columns support every estimator:
#   income     -> mean / total / median (continuous)
#   fam_size   -> ratio denominator
#   low_income -> proportion (0/1)


def _rows_with_singletons():
    # (id, stratum, psu, income, fam_size)
    return [
        (0, "A", "101", 10_000.0, 2),  # stratum A: single PSU -> singleton
        (1, "A", "101", 12_000.0, 3),
        (2, "B", "201", 20_000.0, 2),
        (3, "B", "201", 22_000.0, 1),
        (4, "B", "202", 24_000.0, 4),
        (5, "B", "202", 26_000.0, 3),
        (6, "C", "301", 30_000.0, 2),  # stratum C: single PSU -> singleton
        (7, "C", "301", 15_000.0, 1),
        (8, "D", "401", 40_000.0, 3),
        (9, "D", "401", 42_000.0, 2),
        (10, "D", "402", 44_000.0, 4),
        (11, "D", "402", 46_000.0, 2),
    ]


def _to_frame(rows):
    df = pl.DataFrame(
        rows,
        schema=["id", "stratum", "psu", "income", "fam_size"],
        orient="row",
    )
    return df.with_columns(
        pl.lit(1.0).alias("weight"),
        (pl.col("income") < 25_000).cast(pl.Int64).alias("low_income"),
    )


@pytest.fixture
def singleton_sample():
    """Stratified, clustered design containing two singleton strata (A, C)."""
    df = _to_frame(_rows_with_singletons())
    design = svy.Design(row_index="id", stratum="stratum", psu="psu", wgt="weight")
    return svy.Sample(data=df, design=design)


@pytest.fixture
def clean_sample():
    """Same shape but every stratum has >= 2 PSUs (no singletons)."""
    rows = [r for r in _rows_with_singletons() if r[1] in ("B", "D")]
    df = _to_frame(rows)
    design = svy.Design(row_index="id", stratum="stratum", psu="psu", wgt="weight")
    return svy.Sample(data=df, design=design)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

ALL_METHODS = ["mean", "total", "prop", "ratio", "median"]


def estimate(sample, method):
    """Invoke an estimation method by name and return the Estimate object."""
    est = sample.estimation
    if method == "mean":
        return est.mean("income")
    if method == "total":
        return est.total("income")
    if method == "prop":
        return est.prop("low_income")
    if method == "ratio":
        return est.ratio("income", "fam_size")
    if method == "median":
        return est.median("income")
    raise ValueError(f"unknown method {method!r}")


def assert_valid_estimate(result):
    """A produced estimate must have a finite, non-negative standard error."""
    pe = result.estimates[0]
    assert pe.est is not None
    assert pe.se is not None
    assert pe.se >= 0.0


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 1: default (no strategy) FAILS on unhandled singletons
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("method", ALL_METHODS)
def test_default_unhandled_singletons_raise(singleton_sample, method):
    """With singletons present and no strategy chosen, estimation must raise."""
    assert singleton_sample.singleton.exists
    with pytest.raises(SingletonError):
        estimate(singleton_sample, method)


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 2: strategies that remove singletons from the design -> succeed
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("method", ALL_METHODS)
@pytest.mark.parametrize("strategy", ["collapse", "pool"])
def test_structural_strategies_allow_estimation(singleton_sample, method, strategy):
    """collapse/pool remap the variance design (via a config) so estimation runs.

    Note: these leave the original stratum column untouched, so ``.exists`` stays
    True; "handled" is signalled by the attached config (``last_result``), which
    is exactly the predicate the fail-by-default check keys off.
    """
    handled = getattr(singleton_sample.singleton, strategy)()
    assert handled.singleton.last_result is not None
    assert_valid_estimate(estimate(handled, method))


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 3: variance-config strategies -> succeed
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("method", ALL_METHODS)
@pytest.mark.parametrize("strategy", ["skip", "certainty", "center"])
def test_variance_config_strategies_allow_estimation(singleton_sample, method, strategy):
    """skip/certainty/center attach a config the variance engine consumes."""
    handled = getattr(singleton_sample.singleton, strategy)()
    assert_valid_estimate(estimate(handled, method))


@pytest.mark.parametrize("method", ["mean", "total"])
def test_scale_allows_estimation(singleton_sample, method):
    """scale (R's 'average') excludes singletons then inflates the variance."""
    handled = singleton_sample.singleton.scale()
    assert_valid_estimate(estimate(handled, method))


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 4: no singletons -> no handling required, no false positive
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("method", ALL_METHODS)
def test_clean_design_needs_no_handling(clean_sample, method):
    """A design without singletons estimates cleanly and never raises."""
    assert not clean_sample.singleton.exists
    assert_valid_estimate(estimate(clean_sample, method))


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 5: explicit raise_error() short-circuits before estimation
# ══════════════════════════════════════════════════════════════════════════════


def test_explicit_raise_error(singleton_sample):
    with pytest.raises(SingletonError):
        singleton_sample.singleton.raise_error()
