# tests/svy/estimation/test_deff_reference.py
"""Tests for the ``deff`` SRS-reference selector.

``deff`` names the simple-random-sample reference the design variance is
compared against -- ``"wor"`` (without replacement, Kish's design effect) or
``"wr"`` (with replacement, Kish's deft^2). It describes the denominator, not
the design: choosing ``"wr"`` does not assert that the sample was drawn with
replacement.
"""

from __future__ import annotations

import pytest

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
