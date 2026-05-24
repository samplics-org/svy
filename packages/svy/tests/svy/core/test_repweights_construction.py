# tests/svy/core/test_repweights_construction.py
"""
Regression tests for RepWeights construction with string method names.

Background
----------
Prior to svy 0.17.2, ``RepWeights.__post_init__`` used ``object.__setattr__``
to coerce a string ``method`` argument to the ``EstimationMethod`` enum.
On Python 3.12 with msgspec >= 0.21, msgspec.Struct(frozen=True) intercepts
even ``object.__setattr__`` and raises::

    TypeError: can't apply this __setattr__ to RepWeights object

On Python 3.14, the same call succeeds, which is why the bug went undetected
in environments using newer Python.

The fix replaces ``object.__setattr__`` with ``msgspec.structs.force_setattr``,
the msgspec-sanctioned escape hatch for mutating frozen structs from
``__post_init__``.

These tests exercise the construction path that triggered the bug. If this
file fails on Python 3.12, the fix has regressed.
"""

from __future__ import annotations

import sys

import pytest

import svy

from svy.core.design import RepWeights, make_rep_weights
from svy.core.enumerations import EstimationMethod


# -----------------------------------------------------------------------------
# Direct constructor with string method names
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method_str, expected",
    [
        ("bootstrap", EstimationMethod.BOOTSTRAP),
        ("BOOTSTRAP", EstimationMethod.BOOTSTRAP),
        ("Bootstrap", EstimationMethod.BOOTSTRAP),
        ("  bootstrap  ", EstimationMethod.BOOTSTRAP),
        ("bs", EstimationMethod.BOOTSTRAP),
        ("brr", EstimationMethod.BRR),
        ("BRR", EstimationMethod.BRR),
        ("jackknife", EstimationMethod.JACKKNIFE),
        ("jk", EstimationMethod.JACKKNIFE),
        ("jkn", EstimationMethod.JACKKNIFE),
        ("sdr", EstimationMethod.SDR),
    ],
)
def test_string_method_normalized_in_constructor(method_str, expected):
    """RepWeights(method='<string>') must succeed and normalize to enum."""
    rw = RepWeights(method=method_str, prefix="rw", n_reps=10)
    assert rw.method is expected


def test_enum_method_accepted_unchanged():
    """Passing the enum directly must also work."""
    rw = RepWeights(method=EstimationMethod.BOOTSTRAP, prefix="rw", n_reps=10)
    assert rw.method is EstimationMethod.BOOTSTRAP


# -----------------------------------------------------------------------------
# The exact call shape from the JSS paper / Quarto comparison document
# -----------------------------------------------------------------------------


def test_paper_example_bootstrap():
    """Mirrors svy_vs_rsurvey_comparison.qmd verbatim."""
    rw = svy.RepWeights(method="bootstrap", prefix="bsrw", n_reps=1000)
    assert rw.method is EstimationMethod.BOOTSTRAP
    assert rw.prefix == "bsrw"
    assert rw.n_reps == 1000


def test_paper_example_jackknife():
    rw = svy.RepWeights(method="jackknife", prefix="jkw_", n_reps=62)
    assert rw.method is EstimationMethod.JACKKNIFE
    assert rw.n_reps == 62


def test_paper_example_brr_with_fay():
    rw = svy.RepWeights(method="brr", prefix="brr_", n_reps=32, fay_coef=0.5)
    assert rw.method is EstimationMethod.BRR
    assert rw.fay_coef == 0.5


# -----------------------------------------------------------------------------
# Validation still works (the fix must not weaken __post_init__ guards)
# -----------------------------------------------------------------------------


def test_invalid_method_string_raises_value_error():
    with pytest.raises(ValueError, match="Unknown replication method"):
        RepWeights(method="not-a-method", prefix="rw", n_reps=10)


def test_non_string_non_enum_method_raises_type_error():
    with pytest.raises(TypeError):
        RepWeights(method=42, prefix="rw", n_reps=10)


def test_taylor_method_is_rejected_as_replication_method():
    """Taylor is not a valid replicate method."""
    with pytest.raises(ValueError, match="not a valid replication method"):
        RepWeights(method=EstimationMethod.TAYLOR, prefix="rw", n_reps=10)


def test_empty_prefix_rejected():
    with pytest.raises(ValueError, match="prefix"):
        RepWeights(method="bootstrap", prefix="", n_reps=10)


def test_whitespace_prefix_rejected():
    with pytest.raises(ValueError, match="prefix"):
        RepWeights(method="bootstrap", prefix="   ", n_reps=10)


def test_n_reps_below_minimum_rejected():
    with pytest.raises(ValueError, match="n_reps"):
        RepWeights(method="bootstrap", prefix="rw", n_reps=1)


def test_negative_fay_coef_rejected():
    with pytest.raises(ValueError, match="fay_coef"):
        RepWeights(method="brr", prefix="rw", n_reps=10, fay_coef=-0.1)


def test_non_positive_df_rejected():
    with pytest.raises(ValueError, match="df"):
        RepWeights(method="bootstrap", prefix="rw", n_reps=10, df=0)


def test_negative_padding_rejected():
    with pytest.raises(ValueError, match="padding"):
        RepWeights(method="bootstrap", prefix="rw", n_reps=10, padding=-1)


# -----------------------------------------------------------------------------
# Frozen-struct contract still holds after the fix
# -----------------------------------------------------------------------------


def test_repweights_remains_frozen_after_construction():
    """Mutation after construction must still be blocked."""
    rw = RepWeights(method="bootstrap", prefix="rw", n_reps=10)
    with pytest.raises((AttributeError, TypeError)):
        rw.prefix = "different"  # type: ignore[misc]


def test_repweights_remains_frozen_for_method():
    rw = RepWeights(method="bootstrap", prefix="rw", n_reps=10)
    with pytest.raises((AttributeError, TypeError)):
        rw.method = EstimationMethod.JACKKNIFE  # type: ignore[misc]


# -----------------------------------------------------------------------------
# Factory function still works (exercises a different code path:
# the factory pre-normalizes, so __post_init__ never sees a string)
# -----------------------------------------------------------------------------


def test_make_rep_weights_factory_with_string():
    rw = make_rep_weights("bootstrap", prefix="bsrw", n_reps=1000)
    assert rw.method is EstimationMethod.BOOTSTRAP


def test_make_rep_weights_factory_with_alias():
    rw = make_rep_weights("jk", prefix="jkw_", n_reps=62)
    assert rw.method is EstimationMethod.JACKKNIFE


# -----------------------------------------------------------------------------
# Diagnostic: print versions on first run (visible with -s)
# -----------------------------------------------------------------------------


def test_emit_diagnostic_versions(capsys):
    """Not a real assertion; emits versions for failure-triage convenience."""
    import msgspec

    print(
        f"\n[diagnostic] python={sys.version.split()[0]} "
        f"svy={svy.__version__} msgspec={msgspec.__version__}"
    )
    captured = capsys.readouterr()
    assert "python=" in captured.out
