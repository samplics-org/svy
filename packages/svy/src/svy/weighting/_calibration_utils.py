# src/svy/weighting/_calibration_utils.py
"""
Calibration-specific helpers: term expansion and target matching.

Used by calibration.py. Kept separate from _helpers.py because these
are tightly coupled to the Cat/Cross term model and calibration matrix
construction — they are not general-purpose weighting utilities.
"""

from __future__ import annotations

import dataclasses

from typing import Any, Mapping

import numpy as np
import polars as pl

from svy.core.terms import Cat, Cross, Feature
from svy.core.types import Category
from svy.errors import DimensionError, WeightingError
from svy.weighting._keys import _NOT_FOUND, LevelIndex, _Ambiguous, match_keys, target_vector


def _expand_term(
    term: Feature, df: pl.DataFrame, where: str
) -> tuple[list[pl.Expr], list[Category]]:
    if isinstance(term, str):
        if term not in df.columns:
            raise WeightingError.missing_columns(
                where=where, param="controls", missing=[term], available=list(df.columns)
            )
        n_null = int(df.get_column(term).is_null().sum())
        if n_null > 0:
            # Silently filling with 0 would bias the calibration totals; the
            # Cat branch treats null as an explicit level needing a target,
            # so a continuous auxiliary must be complete.
            raise DimensionError(
                title="Missing values in continuous auxiliary",
                detail=f"Column '{term}' has {n_null} null value(s).",
                code="AUX_NA",
                where=where,
                param=str(term),
                hint="Impute or drop missing values (e.g. wrangling.fill_null) "
                "before calibrating on this auxiliary.",
            )
        return [pl.col(term).cast(pl.Float64)], [term]

    if isinstance(term, Cat):
        col_name = term.name
        if col_name not in df.columns:
            raise WeightingError.missing_columns(
                where=where, param="controls", missing=[col_name], available=list(df.columns)
            )

        levels = df.get_column(col_name).unique().sort().to_list()

        if term.ref is not None:
            try:
                ref = LevelIndex(levels).resolve(term.ref)
            except _Ambiguous:
                ref = _NOT_FOUND
            if ref is _NOT_FOUND:
                raise WeightingError.ref_level_unknown(
                    where=where,
                    column=col_name,
                    ref=term.ref,
                    levels=[v for v in levels if v is not None],
                )
            levels = [lbl for lbl in levels if lbl != ref]

        exprs = []
        labels = []
        for i, lvl in enumerate(levels):
            if lvl is None:
                base_expr = pl.col(col_name).is_null()
            else:
                base_expr = pl.col(col_name) == lvl

            expr = base_expr.cast(pl.Float64).fill_null(0.0).alias(f"_tmp_cat_{col_name}_{i}")
            exprs.append(expr)
            labels.append(lvl)

        return exprs, labels

    if isinstance(term, Cross):
        left_exprs, left_labs = _expand_term(term.left, df, where)
        right_exprs, right_labs = _expand_term(term.right, df, where)

        out_exprs = []
        out_labs = []

        count = 0
        for le, ll in zip(left_exprs, left_labs):
            for re, rl in zip(right_exprs, right_labs):
                out_exprs.append((le * re).alias(f"_tmp_cross_{count}"))
                count += 1

                def _to_tuple(x: Any) -> tuple:
                    return x if isinstance(x, tuple) else (x,)

                new_lab = _to_tuple(ll) + _to_tuple(rl)
                out_labs.append(new_lab)

        return out_exprs, out_labs

    raise WeightingError.term_invalid(where=where, term=term)


def _without_refs(term: Feature) -> Feature:
    if isinstance(term, Cat) and term.ref is not None:
        return dataclasses.replace(term, ref=None)
    if isinstance(term, Cross):
        return Cross(_without_refs(term.left), _without_refs(term.right))
    return term


def _match_term_targets(
    labels: list[Category],
    target_spec: Any,
    term: Feature,
    df: pl.DataFrame,
    *,
    where: str,
    param: str | None = None,
) -> list[float]:
    """Targets for one term, aligned to its expanded ``labels``.

    Keys resolve against every level of the term's columns, so a target for a
    ``Cat(ref=...)`` reference level is accepted and ignored.
    """
    param = param or f"controls[{term!r}]"
    if isinstance(target_spec, (int, float, np.integer, np.floating)) and not isinstance(
        target_spec, bool
    ):
        if len(labels) != 1:
            raise WeightingError.scalar_for_cells(
                where=where, param=param, got=target_spec, levels=labels
            )
        return target_vector(
            {labels[0]: target_spec}, labels, where=where, param=param, nonneg=False
        )

    if isinstance(target_spec, Mapping):
        known = labels
        if _without_refs(term) != term:
            _, known = _expand_term(_without_refs(term), df, where)
        width = len(labels[0]) if labels and isinstance(labels[0], tuple) else 1
        cols = _term_cols(term)
        matched = match_keys(
            target_spec,
            LevelIndex(known, width=width),
            where=where,
            param=param,
            required=labels,
            cols=cols,
        )
        return target_vector(matched, labels, where=where, param=param, nonneg=False)

    raise WeightingError.targets_type(
        where=where,
        param=param,
        got=target_spec,
        expected="a number (one column) or a dict {level: number}",
    )


def _term_cols(term: Feature) -> list[str]:
    if isinstance(term, str):
        return [term]
    if isinstance(term, Cat):
        return [term.name]
    if isinstance(term, Cross):
        return _term_cols(term.left) + _term_cols(term.right)
    return []
