# tests/svy/core/test_expr_bool.py
"""
Regression tests: Expr must refuse Python truthiness so that ``and`` / ``or``
/ ``not`` / chained comparisons fail loudly instead of silently building the
wrong filter (previously ``e1 and e2`` silently evaluated to ``e2``).
"""

import pytest

import svy

from svy.core.expr import Expr


def test_bool_raises():
    with pytest.raises(TypeError, match="truth value of an Expr is ambiguous"):
        bool(svy.col("age") > 18)


def test_and_keyword_raises():
    e1 = svy.col("age") >= 18
    e2 = svy.col("age") < 65
    with pytest.raises(TypeError, match="ambiguous"):
        e1 and e2  # noqa: B015


def test_or_keyword_raises():
    e1 = svy.col("age") >= 18
    e2 = svy.col("age") < 65
    with pytest.raises(TypeError, match="ambiguous"):
        e1 or e2  # noqa: B015


def test_not_keyword_raises():
    with pytest.raises(TypeError, match="ambiguous"):
        not (svy.col("age") >= 18)


def test_chained_comparison_raises():
    with pytest.raises(TypeError, match="ambiguous"):
        18 <= svy.col("age") < 65  # noqa: B015


def test_operator_composition_still_works():
    combined = (svy.col("age") >= 18) & (svy.col("age") < 65)
    assert isinstance(combined, Expr)
    negated = ~(svy.col("age") >= 18)
    assert isinstance(negated, Expr)
    either = (svy.col("a") > 0) | (svy.col("b") > 0)
    assert isinstance(either, Expr)


def test_eq_builds_expression_and_expr_is_unhashable():
    cmp = svy.col("age") == 30
    assert isinstance(cmp, Expr)
    # __eq__ returns an expression (not a bool), so hashing must be disabled.
    with pytest.raises(TypeError):
        hash(svy.col("age"))
