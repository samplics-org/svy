# tests/svy/estimation/test_deff_reference.py
"""Tests for the ``deff`` SRS-reference selector.

``deff`` names the simple-random-sample reference the design variance is
compared against -- ``"wor"`` (without replacement, Kish's design effect) or
``"wr"`` (with replacement, Kish's deft^2). It describes the denominator, not
the design: choosing ``"wr"`` does not assert that the sample was drawn with
replacement.
"""

from __future__ import annotations

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
