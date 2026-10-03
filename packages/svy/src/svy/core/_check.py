# src/svy/core/_check.py
"""Data checks on a frame: weights, record keys, PSU nesting, rake margins.

Each check takes a polars frame and column names and returns a frozen report.
They report what they find and never raise on the data's content; they raise
only on bad input (a missing column, a non-numeric weight).
"""

from __future__ import annotations

import numbers

from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl

from svy.core.check import KeyCheck, MarginCheck, NestingCheck, WeightCheck
from svy.errors import DimensionError, MethodError


__all__ = ["check_weights", "check_key", "check_nesting", "margin_totals_check"]

Columns = str | Sequence[str]

# Raking refuses margins whose totals differ by more than this, relative to the
# largest.
MARGINS_RTOL = 1e-6

_N = "__svy_check_n"
_STRATUM = "__svy_check_stratum"


def key_values(frame: pl.DataFrame, cols: list[str]) -> list[Any]:
    """Rows of ``cols`` as values for one column, as tuples for several."""
    if len(cols) == 1:
        return frame.get_column(cols[0]).to_list()
    return list(frame.select(cols).iter_rows())


def _as_cols(columns: Any, *, param: str, where: str) -> list[str]:
    cols = [columns] if isinstance(columns, str) else columns
    if not isinstance(cols, Sequence) or not cols or not all(isinstance(c, str) for c in cols):
        raise MethodError.invalid_type(
            where=where,
            param=param,
            got=columns,
            expected="a column name or a non-empty list of column names",
        )
    return list(dict.fromkeys(cols))


def _check_limit(limit: Any, *, where: str) -> int:
    if isinstance(limit, bool) or not isinstance(limit, numbers.Integral):
        raise MethodError.invalid_type(
            where=where, param="limit", got=limit, expected="a non-negative integer"
        )
    if limit < 0:
        raise MethodError.invalid_range(
            where=where, param="limit", got=limit, hint="limit is how many examples to list."
        )
    return int(limit)


def _frame(data: Any, cols: list[str], *, param: str, where: str) -> pl.DataFrame:
    """The ``cols`` of ``data``, eager."""
    if not isinstance(data, (pl.DataFrame, pl.LazyFrame)):
        raise MethodError.invalid_type(
            where=where, param="data", got=data, expected="a polars DataFrame or LazyFrame"
        )
    available = data.collect_schema().names()
    missing = [c for c in cols if c not in available]
    if missing:
        raise DimensionError.missing_columns(
            where=where, param=param, missing=missing, available=available
        )
    out = data.select(cols)
    return out.collect() if isinstance(out, pl.LazyFrame) else out


def kish_deff(w: np.ndarray) -> float:
    """Kish design effect due to weighting, ``n * sum(w^2) / sum(w)^2``."""
    mean_w = np.mean(w)
    return float(1 + np.mean(np.power(w - mean_w, 2) / mean_w**2))


def check_weights(data: pl.DataFrame | pl.LazyFrame, wgt: str) -> WeightCheck:
    """Count null, non-finite, negative and zero weights and summarize the rest.

    The summaries (min, max, max/min, sum, mean, Kish design effect, effective
    sample size) are over the positive finite weights. The sum is the
    estimated population size: one far from the known total often means a
    weight stored with implied decimals.

    The design effect is ``Sample.deff_w`` on the positive weights;
    ``Sample.deff_w`` counts every row, so zero weights raise it there.
    """
    where = "checks.check_weights"
    if not isinstance(wgt, str):
        raise MethodError.invalid_type(where=where, param="wgt", got=wgt, expected="a column name")
    frame = _frame(data, [wgt], param="wgt", where=where)
    dtype = frame.schema[wgt]
    if not (dtype.is_numeric() or dtype == pl.Null):
        raise MethodError(
            title="Weight column is not numeric",
            detail=f"{wgt!r} is {dtype}.",
            code="WEIGHT_NOT_NUMERIC",
            where=where,
            param="wgt",
            got=str(dtype),
            expected="a numeric column",
            hint=f"Cast it first, e.g. pl.col({wgt!r}).cast(pl.Float64).",
        )
    w = frame.get_column(wgt).cast(pl.Float64)
    present = w.drop_nulls()
    finite = present.filter(present.is_finite())
    pos = finite.filter(finite > 0).to_numpy()

    if pos.size:
        total = float(pos.sum())
        lo, hi = float(pos.min()), float(pos.max())
        stats: dict[str, float | None] = {
            "min": lo,
            "max": hi,
            "ratio": hi / lo,
            "sum": total,
            "mean": total / pos.size,
            "deff": kish_deff(pos),
            "ess": total**2 / float(np.sum(pos**2)),
        }
    else:
        stats = dict.fromkeys(("min", "max", "ratio", "sum", "mean", "deff", "ess"))

    return WeightCheck(
        wgt=wgt,
        n=w.len(),
        n_null=w.null_count(),
        n_nonfinite=present.len() - finite.len(),
        n_negative=int((finite < 0).sum()),
        n_zero=int((finite == 0).sum()),
        n_positive=int(pos.size),
        **stats,
    )


def check_key(data: pl.DataFrame | pl.LazyFrame, columns: Columns, *, limit: int = 10) -> KeyCheck:
    """Whether ``columns`` identify the rows: nulls and duplicated key values.

    Rows with a null in any key column are counted apart and left out of the
    duplicate count. ``examples`` lists up to ``limit`` duplicated key values,
    in key order: values for one column, tuples for several.
    """
    where = "checks.check_key"
    cols = _as_cols(columns, param="columns", where=where)
    limit = _check_limit(limit, where=where)
    frame = _frame(data, cols, param="columns", where=where)

    null = pl.any_horizontal(pl.col(c).is_null() for c in cols)
    n_null = int(frame.select(null.sum()).item())
    dup = (
        frame.filter(~null)
        .group_by(cols)
        .agg(pl.len().alias(_N))
        .filter(pl.col(_N) > 1)
        .sort(cols)
    )
    return KeyCheck(
        columns=cols,
        n=frame.height,
        n_null=n_null,
        n_duplicated=dup.height,
        n_rows_duplicated=int(dup.get_column(_N).sum()),
        examples=key_values(dup.head(limit), cols),
    )


def check_nesting(
    data: pl.DataFrame | pl.LazyFrame,
    stratum: Columns,
    psu: Columns,
    *,
    limit: int = 10,
) -> NestingCheck:
    """PSU codes that appear in more than one stratum.

    svy takes a PSU to be the (stratum, psu) pair, as R's ``nest=TRUE`` does,
    so a PSU code reused in two strata is two clusters. Data cannot tell a
    code reused by design (PSUs numbered 1, 2, ... within each stratum) from
    one PSU split across strata by mistake, so this reports and never raises.
    ``examples`` lists up to ``limit`` such codes with the strata they appear
    in: values for one column, tuples for several.
    """
    from svy.core.panel import design_varies_within_case

    where = "checks.check_nesting"
    s_cols = _as_cols(stratum, param="stratum", where=where)
    p_cols = _as_cols(psu, param="psu", where=where)
    limit = _check_limit(limit, where=where)
    frame = _frame(data, list(dict.fromkeys([*s_cols, *p_cols])), param="stratum/psu", where=where)

    pairs = frame.select(*p_cols, pl.struct(s_cols).alias(_STRATUM))
    shared = design_varies_within_case(pairs, p_cols, [_STRATUM], limit=pairs.height).get(
        _STRATUM, []
    )

    examples: list[tuple[Any, list[Any]]] = []
    if shared and limit:
        rows = [k if isinstance(k, tuple) else (k,) for k in shared[:limit]]
        keys = pl.DataFrame(rows, schema=pairs.select(p_cols).schema, orient="row")
        found = (
            pairs.join(keys, on=p_cols, how="semi", nulls_equal=True)
            .group_by(p_cols)
            .agg(pl.col(_STRATUM).unique().sort())
            .sort(p_cols)
        )
        for key, strata in zip(key_values(found, p_cols), found.get_column(_STRATUM).to_list()):
            vals = [s[s_cols[0]] if len(s_cols) == 1 else tuple(s.values()) for s in strata]
            examples.append((key, vals))

    return NestingCheck(
        stratum=s_cols,
        psu=p_cols,
        n=frame.height,
        n_strata=frame.select(s_cols).n_unique(),
        n_psus=frame.select(p_cols).n_unique(),
        n_psus_across_strata=len(shared),
        examples=examples,
    )


def margin_totals_check(totals: Mapping[str, float], *, rtol: float = MARGINS_RTOL) -> MarginCheck:
    """The margin agreement rule ``rake`` enforces, on the margins' totals."""
    totals = dict(totals)
    spread = 0.0
    if len(totals) >= 2:
        lo, hi = min(totals.values()), max(totals.values())
        if hi > 0:
            spread = (hi - lo) / hi
    return MarginCheck(totals=totals, agree=not spread > rtol, max_rel_diff=spread, rtol=rtol)
