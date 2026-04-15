# src/svy/wrangling/rows.py
"""
Row operations: filter, sort, deduplicate, add row index.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Mapping, Sequence

import polars as pl

from svy.core.expr import to_polars_expr
from svy.core.types import WhereArg
from svy.errors import MethodError, SvyError
from svy.wrangling._helpers import _resolve_target

if TYPE_CHECKING:
    from svy.core.sample import Sample


# -------------------------------------------------------------------
# Where-clause compiler (shared by filter_records)
# -------------------------------------------------------------------


def _compile_where_to_pl_expr(where: WhereArg) -> pl.Expr | None:
    """Convert WhereArg to a Polars filter expression."""
    if isinstance(where, Mapping):
        preds: list[pl.Expr] = []
        for k, v in where.items():
            if isinstance(v, (list, tuple, set)) and not isinstance(
                v, (str, bytes, bytearray)
            ):
                preds.append(pl.col(k).is_in(list(v)))
            else:
                preds.append(pl.col(k) == v)  # type: ignore[arg-type]
        if not preds:
            return None
        acc = preds[0]
        for p in preds[1:]:
            acc = acc & p
        return acc

    if isinstance(where, (list, tuple)) and not isinstance(
        where, (str, bytes, bytearray)
    ):
        if not where:
            return None
        compiled = [to_polars_expr(e) for e in where]
        acc = compiled[0]
        for p in compiled[1:]:
            acc = acc & p
        return acc

    return to_polars_expr(where)


# -------------------------------------------------------------------
# Public functions
# -------------------------------------------------------------------


def filter_records(
    sample: "Sample",
    where: WhereArg | None = None,
    *,
    negate: bool = False,
    check_singletons: bool = False,
    on_singletons: Literal["ignore", "warn", "error"] = "ignore",
    inplace: bool = False,
) -> "Sample":
    """Filter rows based on conditions."""
    if where is None:
        return sample

    try:
        pred = _compile_where_to_pl_expr(where)
        if pred is None:
            return sample
        if negate:
            pred = ~pred

        filtered_data = sample._data.filter(pred)
        target = _resolve_target(sample, filtered_data, inplace=inplace)

        if check_singletons:
            target._check_for_singletons()
            if getattr(target, "_singletons", None):
                if on_singletons == "warn":
                    target.warn(
                        code="SINGLETONS_DETECTED",
                        title="Singleton PSUs/strata detected after filtering",
                        detail=(
                            f"Found {len(target._singletons or [])} "
                            f"singleton group(s)."
                        ),
                        where="wrangling.filter_records",
                    )
                elif on_singletons == "error":
                    raise MethodError(
                        title="Singletons detected after filtering",
                        detail=(
                            f"Found {len(target._singletons or [])} "
                            f"singleton group(s)."
                        ),
                        code="SINGLETONS_AFTER_FILTER",
                        where="wrangling.filter_records",
                        hint=(
                            "Collapse strata, adjust PSUs, or handle "
                            "via the singleton utilities."
                        ),
                    )

        return target

    except SvyError:
        raise
    except Exception as ex:
        raise MethodError(
            title="Filter failed",
            detail=str(ex),
            code="FILTER_FAILED",
            where="wrangling.filter_records",
        ) from ex


def order_by(
    sample: "Sample",
    cols: str | Sequence[str],
    *,
    descending: bool | Sequence[bool] = False,
    nulls_last: bool = True,
    inplace: bool = False,
) -> "Sample":
    """Sort rows by one or more columns."""
    new_data = sample._data.sort(
        by=cols, descending=descending, nulls_last=nulls_last
    )
    return _resolve_target(sample, new_data, inplace=inplace)


def distinct(
    sample: "Sample",
    cols: str | Sequence[str] | None = None,
    *,
    keep: Literal["first", "last", "any", "none"] = "first",
    maintain_order: bool = True,
    inplace: bool = False,
) -> "Sample":
    """Remove duplicate rows."""
    subset = [cols] if isinstance(cols, str) else cols
    new_data = sample._data.unique(
        subset=subset,
        keep=keep,
        maintain_order=maintain_order,
    )
    return _resolve_target(sample, new_data, inplace=inplace)


def with_row_index(
    sample: "Sample",
    name: str = "row_index",
    offset: int = 0,
    *,
    inplace: bool = False,
) -> "Sample":
    """Add a row index column."""
    new_data = sample._data.with_row_index(name=name, offset=offset)
    return _resolve_target(sample, new_data, inplace=inplace)
