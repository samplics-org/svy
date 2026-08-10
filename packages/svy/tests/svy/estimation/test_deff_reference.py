# tests/svy/estimation/test_deff_reference.py
"""Tests for the ``deff`` SRS-reference selector.

``deff`` names the simple-random-sample reference the design variance is
compared against -- ``"wor"`` (without replacement, Kish's design effect) or
``"wr"`` (with replacement, Kish's deft^2). It describes the denominator, not
the design: choosing ``"wr"`` does not assert that the sample was drawn with
replacement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from svy import Sample
from svy.core.design import Design
from svy.errors import MethodError
from svy.estimation.base import Estimation


normalize = Estimation._normalize_deff


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("wor", "wor"),
        ("wr", "wr"),
        # Case and separator folding, as _normalize_ci_method does.
        ("WOR", "wor"),
        ("WR", "wr"),
        ("  wr  ", "wr"),
        ("with_replacement", "wr"),
        ("without-replacement", "wor"),
        ("srswor", "wor"),
        ("srswr", "wr"),
        # R spells the with-replacement reference this way; redirect rather
        # than reject, since users arrive from survey().
        ("replace", "wr"),
    ],
)
def test_accepted_spellings(value, expected):
    assert normalize(value) == expected


@pytest.mark.parametrize("value", [True, False])
def test_booleans_are_rejected(value):
    """A bool cannot say which reference is wanted, so it is refused rather
    than mapped. Silently accepting it would leave two spellings for one
    meaning; silently ignoring it would drop a design effect the caller asked
    for, since Literal is not enforced at runtime."""
    with pytest.raises(MethodError) as exc:
        normalize(value)
    assert "DEFF_BOOL_REJECTED" in str(exc.value)


def test_true_is_told_which_reference_it_used_to_mean():
    """The guidance has to be actionable: True selected the without-replacement
    reference, so name it rather than just refusing."""
    with pytest.raises(MethodError) as exc:
        normalize(True)
    message = str(exc.value)
    assert "deff='wor'" in message
    assert "deff='wr'" in message


def test_false_is_told_to_drop_the_argument():
    with pytest.raises(MethodError) as exc:
        normalize(False)
    assert "Omit the argument" in str(exc.value)


@pytest.mark.parametrize("value", ["srs", "kish", "deft", "yes", ""])
def test_unknown_references_are_rejected(value):
    with pytest.raises(MethodError) as exc:
        normalize(value)
    assert "DEFF_REFERENCE_UNKNOWN" in str(exc.value)


def test_rejection_names_the_valid_options():
    """An error that does not say what to type instead is only half useful."""
    with pytest.raises(MethodError) as exc:
        normalize("nonsense")
    message = str(exc.value)
    assert "'wor'" in message and "'wr'" in message


def test_bool_check_precedes_string_handling():
    """bool subclasses int and str(True) is "True", so the bool branch has to
    run first or it would fall through to the alias lookup and be reported as
    an unknown reference rather than as a rejected boolean."""
    with pytest.raises(MethodError) as exc:
        normalize(True)
    assert "DEFF_BOOL_REJECTED" in str(exc.value)
    assert "DEFF_REFERENCE_UNKNOWN" not in str(exc.value)


# ---------------------------------------------------------------------------
# The one detectable failure: a degenerate finite-population correction
# ---------------------------------------------------------------------------


def _sample(weights) -> Sample:
    rng = np.random.default_rng(3)
    n = len(weights)
    psu = rng.integers(0, 30, n)
    return Sample(
        pl.DataFrame(
            {
                "id": range(n),
                "stratum": (psu % 5).astype(str),
                "psu": psu.astype(str),
                "w": np.asarray(weights, dtype=float),
                "y": rng.normal(100.0, 20.0, n),
            }
        ),
        Design(row_index="id", stratum="stratum", psu="psu", wgt="w"),
    )


def _honest(n=1200):
    return np.random.default_rng(11).uniform(20.0, 200.0, n)


@pytest.mark.parametrize(
    "label,make",
    [
        ("normalized to n", lambda w: w / w.sum() * len(w)),
        ("normalized to 1", lambda w: w / w.sum()),
        ("unit weights", lambda w: np.ones(len(w))),
    ],
)
def test_degenerate_correction_raises_rather_than_returning_nan(label, make):
    """sum(w) <= n makes 1 - n/N zero or negative, so no design effect exists.

    Returning a column of NaN would read as "no design effect" rather than
    "this could not be computed", which is the distinction the error draws.
    """
    est = _sample(make(_honest())).estimation
    with pytest.raises(MethodError) as exc:
        est.mean("y", deff="wor")
    assert "DEFF_NOT_COMPUTABLE" in str(exc.value)


def test_error_names_both_causes_and_the_remedy():
    """Rescaled weights and a census produce the same condition, so the message
    has to offer both readings -- and say what to do instead."""
    est = _sample(np.ones(500)).estimation
    with pytest.raises(MethodError) as exc:
        est.mean("y", deff="wor")
    message = str(exc.value)
    assert "rescaled" in message
    assert "census" in message
    assert "deff='wr'" in message


@pytest.mark.parametrize(
    "make",
    [
        lambda w: w / w.sum() * len(w),
        lambda w: np.ones(len(w)),
    ],
)
def test_wr_is_unaffected_by_the_same_weights(make):
    """The with-replacement reference has no N in it, so the condition that
    breaks 'wor' cannot touch it."""
    est = _sample(make(_honest())).estimation
    value = est.mean("y", deff="wr").to_dicts()[0]["deff"]
    assert np.isfinite(value) and value > 0.0


def test_no_deff_requested_is_unaffected():
    """The guard must not fire when the caller never asked for a design effect;
    the kernel computes the reference regardless."""
    est = _sample(np.ones(500)).estimation
    out = est.mean("y").to_polars()
    assert "deff" not in out.columns


def test_honest_weights_still_report_a_design_effect():
    est = _sample(_honest()).estimation
    value = est.mean("y", deff="wor").to_dicts()[0]["deff"]
    assert np.isfinite(value) and value > 0.0


# ---------------------------------------------------------------------------
# Large sampling fractions
#
# The regime where the reference choice stops being academic. R survey 4.5 on
# an evaluation-study shape -- 120 beneficiary schools drawn from a frame of
# 150, so f = 0.8:
#
#   des <- svydesign(id=~1, strata=~st, weights=~w, data=d)
#   svymean(~y, des, deff=TRUE)       -> deff 5.027365680015
#   svymean(~y, des, deff="replace")  -> deff 1.005473136003
# ---------------------------------------------------------------------------

R_BIGF_SE = 1.8976514768
R_BIGF_WOR = 5.027365680015
R_BIGF_WR = 1.005473136003


@pytest.fixture
def large_fraction_sample() -> Sample:
    path = Path(__file__).parents[2] / "test_data" / "deff_large_fraction.csv"
    df = pl.read_csv(path, infer_schema_length=10000).with_columns(
        pl.col("y").cast(pl.Float64), pl.col("w").cast(pl.Float64), pl.col("st").cast(pl.String)
    )
    return Sample(df, Design(row_index="id", stratum="st", wgt="w"))


@pytest.mark.parametrize(
    "ref,expected", [("wor", R_BIGF_WOR), ("wr", R_BIGF_WR)]
)
def test_large_sampling_fraction_matches_r(large_fraction_sample, ref, expected):
    row = large_fraction_sample.estimation.mean("y", deff=ref).to_dicts()[0]
    assert row["se"] == pytest.approx(R_BIGF_SE, rel=1e-9)
    assert row["deff"] == pytest.approx(expected, rel=1e-10)


def test_the_choice_is_decisive_at_a_large_fraction(large_fraction_sample):
    """At f = 0.8 the two references differ five-fold, so the documentation
    must not describe the choice as immaterial. Everything else validated here
    sits near f = 0.03, where they agree to a few percent."""
    est = large_fraction_sample.estimation
    wor = est.mean("y", deff="wor").to_dicts()[0]["deff"]
    wr = est.mean("y", deff="wr").to_dicts()[0]["deff"]
    assert wor / wr == pytest.approx(1.0 / (1.0 - 120 / 150), rel=1e-9)
    assert wor / wr == pytest.approx(5.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Batched calls
#
# `y=[...]` goes through _taylor_multi, which chooses between a shared design
# build and a per-variable loop. That is a different dispatch from the single
# variable path, so the reference has to survive it independently.
# ---------------------------------------------------------------------------


def _multi_sample() -> Sample:
    rng = np.random.default_rng(17)
    n = 900
    psu = rng.integers(0, 30, n)
    return Sample(
        pl.DataFrame(
            {
                "id": range(n),
                "stratum": (psu % 4).astype(str),
                "psu": psu.astype(str),
                "w": rng.uniform(20.0, 200.0, n),
                "y1": rng.normal(100.0, 20.0, n),
                "y2": rng.normal(50.0, 8.0, n),
            }
        ),
        Design(row_index="id", stratum="stratum", psu="psu", wgt="w"),
    )


@pytest.mark.parametrize("ref", ["wor", "wr"])
def test_batched_call_reports_a_deff_per_variable(ref):
    est = _multi_sample().estimation
    results = est.mean(["y1", "y2"], deff=ref)
    assert len(results) == 2
    for r in results:
        value = r.to_dicts()[0]["deff"]
        assert np.isfinite(value) and value > 0.0
        assert r.deff_ref == ref


@pytest.mark.parametrize("ref", ["wor", "wr"])
def test_batched_matches_the_per_variable_loop(ref):
    """The batched form shares one design build; the loop does not. They must
    still agree, or batching would quietly change the reference."""
    est = _multi_sample().estimation
    batched = [r.to_dicts()[0]["deff"] for r in est.mean(["y1", "y2"], deff=ref)]
    looped = [est.mean(v, deff=ref).to_dicts()[0]["deff"] for v in ("y1", "y2")]
    for b, single in zip(batched, looped, strict=True):
        assert b == pytest.approx(single, rel=1e-12)


def test_batched_rejects_a_boolean_before_fanning_out():
    """Normalization happens ahead of the batched dispatch, so an invalid value
    fails once rather than once per variable."""
    est = _multi_sample().estimation
    with pytest.raises(MethodError) as exc:
        est.mean(["y1", "y2"], deff=True)
    assert "DEFF_BOOL_REJECTED" in str(exc.value)
