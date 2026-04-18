# src/svy/utils/where.py
"""
WhereArg compiler.
Single source of truth for converting any WhereArg form into a Polars
boolean expression.  Imported by wrangling, estimation, selection, and any
other module that needs to evaluate a user-supplied filter condition.
"""

from __future__ import annotations
from functools import reduce
from typing import Mapping
import polars as pl
from svy.core.expr import to_polars_expr
from svy.core.types import WhereArg


def _compile_where(where: WhereArg) -> pl.Expr | None:
    """
    Convert any WhereArg form to a Polars filter expression.
    Parameters
    ----------
    where : WhereArg
        One of:
        * ``None``                               → returns ``None`` (no filter)
        * ``Mapping[str, Any]``                  → AND-combined per-key predicates:
            - ``{"col": scalar}``                → ``col == scalar``
            - ``{"col": [..] | (..) | {..}}``    → ``col.is_in(list(values))``
            - ``{"col": pl.Series}``             → ``col.is_in(series.implode())``
        * ``Sequence[ExprLike]``                 → AND-combined expressions
        * ``ExprLike`` (``pl.Expr`` or wrapper)  → passed through as-is
    Returns
    -------
    pl.Expr | None
        A single boolean Polars expression, or ``None`` when ``where`` is
        ``None`` or evaluates to an empty condition.
    """
    if where is None:
        return None
    if isinstance(where, Mapping):
        preds: list[pl.Expr] = []
        for k, v in where.items():
            if isinstance(v, pl.Series):
                # Series.implode() wraps as a single-row list value — the
                # unambiguous form is_in expects.  Bare same-dtype Series
                # triggers Polars deprecation pola-rs/polars#22149 and may
                # be removed in a future release.
                preds.append(pl.col(k).is_in(v.implode()))
            elif isinstance(v, (list, tuple, set)) and not isinstance(v, (str, bytes, bytearray)):
                preds.append(pl.col(k).is_in(list(v)))
            else:
                preds.append(pl.col(k) == v)  # type: ignore[arg-type]
        if not preds:
            return None
        return reduce(lambda a, b: a & b, preds)
    if isinstance(where, (list, tuple)) and not isinstance(where, (str, bytes, bytearray)):
        if not where:
            return None
        compiled = [to_polars_expr(e) for e in where]
        return reduce(lambda a, b: a & b, compiled)
    return to_polars_expr(where)
