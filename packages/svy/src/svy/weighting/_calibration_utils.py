# src/svy/weighting/_calibration_utils.py
"""
Calibration-specific helpers: term expansion and target matching.

Used by calibration.py. Kept separate from _helpers.py because these
are tightly coupled to the Cat/Cross term model and calibration matrix
construction — they are not general-purpose weighting utilities.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import polars as pl

from svy.core.terms import Cat, Cross, Feature
from svy.core.types import Category, Number


def _expand_term(
    term: Feature, df: pl.DataFrame, where: str
) -> tuple[list[pl.Expr], list[Category]]:
    if isinstance(term, str):
        if term not in df.columns:
            raise KeyError(f"{where}: Continuous term '{term}' not found in data.")
        return [pl.col(term).cast(pl.Float64).fill_null(0.0)], [term]

    if isinstance(term, Cat):
        col_name = term.name
        if col_name not in df.columns:
            raise KeyError(f"{where}: Cat variable '{col_name}' not found.")

        levels = df.get_column(col_name).unique().sort().to_list()

        if term.ref is not None:
            if term.ref not in levels:
                raise ValueError(
                    f"{where}: Reference level '{term.ref}' not found in '{col_name}'."
                )
            levels = [lbl for lbl in levels if lbl != term.ref]

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

    raise TypeError(f"Unsupported term type: {type(term)}")


def _match_term_targets(
    labels: list[Category],
    target_spec: Number | dict[Category, Number] | Sequence[Number],
    term_name: str,
) -> list[float]:
    if isinstance(target_spec, (int, float, np.integer, np.floating)):
        if len(labels) != 1:
            raise ValueError(
                f"Scalar target provided for term '{term_name}' which expanded "
                f"to {len(labels)} columns. Use a dict mapping labels to values."
            )
        return [float(target_spec)]

    if isinstance(target_spec, dict):
        targets = []
        missing = [lbl for lbl in labels if lbl not in target_spec]
        if missing:
            raise ValueError(f"Missing targets for term '{term_name}': {missing}")

        for lbl in labels:
            val = target_spec[lbl]  # type: ignore[index]
            if not isinstance(val, (int, float, np.integer, np.floating)):
                raise TypeError(f"Target for '{term_name}':{lbl} must be numeric.")
            targets.append(float(val))
        return targets

    raise TypeError(
        f"Unsupported target specification type for '{term_name}': {type(target_spec)}"
    )
