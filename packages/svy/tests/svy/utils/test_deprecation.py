# tests/svy/utils/test_deprecation.py
"""deprecated_param keeps a renamed keyword working, warns at the user's line, and refuses both."""

from __future__ import annotations

import warnings

import pytest

from svy.errors import Code, SvyError, UsageError
from svy.utils import deprecated_param


@deprecated_param("max_it", "max_iter", since="2026.1", remove_in="2027.0")
def fit(x: int, *, max_iter: int = 100) -> tuple[int, int]:
    """Fit something."""
    return x, max_iter


@deprecated_param("tolerance", "tol", since="2026.1", remove_in="2027.0")
@deprecated_param("iters", "max_iter", since="2026.2", remove_in="2027.0")
def solve(*, tol: float = 1e-6, max_iter: int = 10) -> tuple[float, int]:
    return tol, max_iter


class Model:
    @deprecated_param("max_it", "max_iter", since="2026.1", remove_in="2027.0")
    def fit(self, *, max_iter: int = 100) -> int:
        return max_iter


MESSAGE = (
    "fit(max_it=) is deprecated since svy 2026.1 and will be removed in 2027.0. "
    "Use max_iter= instead."
)


def test_old_name_warns_and_passes_its_value_on():
    with pytest.warns(DeprecationWarning) as rec:
        out = fit(1, max_it=7)
    assert out == (1, 7)
    (w,) = rec
    assert str(w.message) == MESSAGE
    assert w.filename == __file__


@pytest.mark.parametrize(
    "kwargs, expected", [({"max_iter": 7}, (1, 7)), ({}, (1, 100))], ids=["new name", "default"]
)
def test_new_name_and_default_do_not_warn(kwargs, expected):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert fit(1, **kwargs) == expected


def test_both_names_raise():
    with pytest.raises(UsageError) as err:
        fit(1, max_it=7, max_iter=8)
    e = err.value
    assert type(e) is UsageError.TypeError and isinstance(e, SvyError)
    assert e.code == Code.PARAM_GIVEN_TWICE and e.param == "max_iter"
    assert e.got == ["max_it", "max_iter"]
    assert "max_it= is the deprecated name of max_iter=" in e.detail


def test_renames_stack():
    with pytest.warns(DeprecationWarning) as rec:
        out = solve(tolerance=0.1, iters=3)
    assert out == (0.1, 3)
    assert sorted(str(r.message).split(" is ")[0] for r in rec) == [
        "solve(iters=)",
        "solve(tolerance=)",
    ]
    assert solve.__deprecated_params__ == {"iters": "max_iter", "tolerance": "tol"}


def test_a_method_is_named_by_its_class():
    with pytest.warns(DeprecationWarning, match=r"^Model\.fit\(max_it=\) is deprecated"):
        assert Model().fit(max_it=5) == 5


def test_the_docstring_says_so():
    assert fit.__doc__ == f".. deprecated:: 2026.1\n   {MESSAGE}\n\nFit something."
    assert fit.__name__ == "fit" and fit.__wrapped__.__name__ == "fit"


@pytest.mark.parametrize(
    "old, new",
    [("max_it", "iterations"), ("max_iter", "max_iter")],
    ids=["new missing", "old taken"],
)
def test_a_rename_that_does_not_fit_the_signature_fails_at_definition(old, new):
    def g(*, max_iter: int = 1) -> int:
        return max_iter

    with pytest.raises(ValueError, match="must take"):
        deprecated_param(old, new, since="2026.1", remove_in="2027.0")(g)
