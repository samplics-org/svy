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
    design = svy.Design(case_id="id", stratum="stratum", psu="psu", wgt="weight")
    return svy.Sample(data=df, design=design)


@pytest.fixture
def clean_sample():
    """Same shape but every stratum has >= 2 PSUs (no singletons)."""
    rows = [r for r in _rows_with_singletons() if r[1] in ("B", "D")]
    df = _to_frame(rows)
    design = svy.Design(case_id="id", stratum="stratum", psu="psu", wgt="weight")
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


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 6: scale's full-sample pass
# ══════════════════════════════════════════════════════════════════════════════
#
# scale runs the kernel twice: variance on the singleton-excluded rows, point
# estimate on the full sample. The second pass must be prepared like the first.

SCALE_METHODS = ["mean", "total", "prop", "ratio"]


def _scale_sample(df):
    design = svy.Design(case_id="id", stratum="stratum", psu="psu", wgt="weight")
    return svy.Sample(data=df, design=design).singleton.scale()


@pytest.mark.parametrize("method", SCALE_METHODS)
@pytest.mark.parametrize("dtype", [pl.Int8, pl.Int16, pl.UInt8, pl.UInt16])
def test_scale_ignores_unrelated_small_int_columns(method, dtype):
    """An unused small-integer column must not reach the kernel or move results."""
    df = _to_frame(_rows_with_singletons()).with_columns(
        pl.lit(1).cast(dtype).alias("flag"),
        (pl.col("id") % 3).cast(dtype).alias("code"),
    )
    got = estimate(_scale_sample(df), method).estimates
    ref = estimate(
        _scale_sample(df.with_columns(pl.col("flag", "code").cast(pl.Int64))), method
    ).estimates

    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        assert g.est == pytest.approx(r.est, rel=1e-12)
        assert g.se == pytest.approx(r.se, rel=1e-12)


@pytest.mark.parametrize("method", ["mean", "total"])
def test_scale_point_estimate_respects_by_and_where(method):
    """The full-sample estimate honours by= and where= (center keeps every row)."""
    df = _to_frame(_rows_with_singletons()).with_columns(
        (pl.col("id") % 2).alias("grp"),
    )
    design = svy.Design(case_id="id", stratum="stratum", psu="psu", wgt="weight")
    sample = svy.Sample(data=df, design=design)
    scaled = sample.singleton.scale().estimation
    centered = sample.singleton.center().estimation

    by_scaled = getattr(scaled, method)("income", by="grp").estimates
    by_centered = getattr(centered, method)("income", by="grp").estimates
    assert sorted(p.est for p in by_scaled) == pytest.approx(
        sorted(p.est for p in by_centered), rel=1e-12
    )

    where = pl.col("grp") == 1
    w_scaled = getattr(scaled, method)("income", where=where).estimates[0]
    w_centered = getattr(centered, method)("income", where=where).estimates[0]
    assert w_scaled.est == pytest.approx(w_centered.est, rel=1e-12)
