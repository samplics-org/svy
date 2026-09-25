# src/svy/weighting/_engine.py
"""
Shared target-resolution and scaling engine for the level/share methods.

``normalize``, ``poststratify``, ``standardize`` and each of ``rake``'s margin
steps answer one question: what should each cell sum to?  Every supported
argument form resolves here to an *absolute* per-cell target array, which is the
single form that crosses the FFI.

The argument grammar is uniform across all of them:

    controls=<scalar>   grand total          (requires cells=None -- one cell)
    controls={...}      absolute per cell    (requires cells=)
    shares={...}        composition, total taken from the data (requires cells=)
    controls=None       sum to n / per-cell counts

``controls`` sets the total, ``shares`` preserves it.  A scalar is one cell, a
dict is many.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import polars as pl

from svy.errors import MethodError, WeightingError
from svy.utils.where import _compile_where
from svy.weighting._keys import LevelIndex, match_keys, target_vector


try:
    from svy_rs._internal import poststratify as _rust_poststratify  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    _rust_poststratify = None

if TYPE_CHECKING:
    from svy.core.types import WhereArg


class CellSpec:
    """Dense cell coding for one adjustment.

    ``codes`` is 0..n_cells-1 for in-scope rows and -1 for rows outside the
    adjustment (failed ``where``, or a null in a cells column).  Out-of-scope
    rows are never handed to the kernel; they keep their previous weight, so the
    adjustment factor is exactly 1.

    Codes are dense and assigned in first-seen order, and ``labels`` is aligned
    to them.  That makes the Rust side's sort of the integer codes a no-op, so
    Python owns the label-to-code mapping outright -- the target vector can never
    arrive permuted relative to the cells it describes.
    """

    __slots__ = ("codes", "labels", "n_cells", "in_scope", "cols")

    def __init__(
        self,
        codes: np.ndarray,
        labels: list[Any],
        in_scope: np.ndarray | None,
        cols: list[str] | None,
    ) -> None:
        self.codes = codes
        self.labels = labels
        self.n_cells = len(labels)
        self.in_scope = in_scope
        self.cols = cols

    @property
    def has_cells(self) -> bool:
        """True when the caller named cells (as opposed to one implicit cell)."""
        return self.cols is not None


def _cells_to_cols(
    cells: str | Sequence[str] | None, *, where: str, param: str = "cells"
) -> list[str] | None:
    if cells is None:
        return None
    if isinstance(cells, str):
        return [cells]
    if isinstance(cells, Sequence) and not isinstance(cells, (bytes, bytearray)):
        cols = list(cells)
        if not cols:
            raise WeightingError.columns_empty(where=where, param=param)
        for c in cols:
            if not isinstance(c, str):
                raise MethodError.invalid_type(
                    where=where, param=param, got=c, expected="str | Sequence[str] | None"
                )
        return cols
    raise MethodError.invalid_type(
        where=where, param=param, got=cells, expected="str | Sequence[str] | None"
    )


def _where_mask(df: pl.DataFrame, where_arg: WhereArg, *, where: str) -> np.ndarray | None:
    expr = _compile_where(where_arg)
    if expr is None:
        return None
    try:
        mask = df.select(expr.alias("__svy_scope__")).get_column("__svy_scope__")
    except Exception as e:
        raise WeightingError.where_invalid(where=where, reason=str(e)) from e
    return mask.fill_null(False).to_numpy().astype(bool, copy=False)


def build_cells(
    df: pl.DataFrame,
    cells: str | Sequence[str] | None,
    where_arg: WhereArg = None,
    *,
    where: str,
) -> CellSpec:
    """Resolve ``cells`` x ``where`` into dense codes plus aligned labels.

    Labels are the raw values for a single cells column and tuples in ``cells``
    order for several, so user-supplied dict keys are compared in the form the
    user wrote them -- no internal separator to leak into error messages or to
    collide with a category value that happens to contain it.
    """
    n = df.height
    cols = _cells_to_cols(cells, where=where)
    scope = _where_mask(df, where_arg, where=where)

    if cols is None:
        codes = np.zeros(n, dtype=np.int64)
        if scope is not None:
            if not scope.any():
                raise WeightingError.no_rows_in_scope(
                    where=where, method=where.rsplit(".", 1)[-1], hint="Check `where`."
                )
            codes = np.where(scope, 0, -1).astype(np.int64)
        return CellSpec(codes, [None], scope, None)

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise WeightingError.missing_columns(
            where=where,
            param="cells",
            missing=missing,
            available=list(df.columns),
            hint="All `cells` columns must exist in the data.",
        )

    if len(cols) == 1:
        raw: list[Any] = df.get_column(cols[0]).to_list()
    else:
        series = [df.get_column(c).to_list() for c in cols]
        raw = [tuple(vals) for vals in zip(*series)]

    codes = np.full(n, -1, dtype=np.int64)
    labels: list[Any] = []
    seen: dict[Any, int] = {}
    for i, lab in enumerate(raw):
        if scope is not None and not scope[i]:
            continue
        # A null anywhere in the cells key leaves the row unassignable, so it is
        # treated exactly like a failed `where`: kept, unadjusted.
        if lab is None or (isinstance(lab, tuple) and any(v is None for v in lab)):
            continue
        code = seen.get(lab)
        if code is None:
            code = len(labels)
            seen[lab] = code
            labels.append(lab)
        codes[i] = code

    if not labels:
        raise WeightingError.no_rows_in_scope(where=where, method=where.rsplit(".", 1)[-1])

    in_scope = codes >= 0
    return CellSpec(codes, labels, in_scope if in_scope.sum() < n else None, cols)


def _is_mapping(x: Any) -> bool:
    return isinstance(x, Mapping)


def _is_scalar(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)


def cell_targets(
    supplied: Mapping[Any, Any], spec: CellSpec, *, param: str, where: str
) -> np.ndarray:
    """Per-cell targets from a mapping keyed by cell value, aligned to ``spec.labels``."""
    index = LevelIndex(spec.labels, width=len(spec.cols or [None]))
    matched = match_keys(supplied, index, where=where, param=param, cols=spec.cols)
    return np.asarray(target_vector(matched, index.levels, where=where, param=param))


def in_scope_sum(wgt_arr: np.ndarray, spec: CellSpec) -> float:
    if spec.in_scope is None:
        return float(wgt_arr.sum())
    return float(wgt_arr[spec.in_scope].sum())


def resolve_targets(
    *,
    controls: Any = None,
    shares: Any = None,
    spec: CellSpec,
    wgt_arr: np.ndarray,
    method: str,
    where: str,
    counts_when_none: bool = False,
) -> np.ndarray:
    """Resolve any supported argument form to absolute per-cell targets.

    ``counts_when_none`` enables ``controls=None`` meaning "sum to n" (per cell
    when cells are named) -- normalize's default, and not a form the constraint
    methods accept.
    """
    if controls is not None and shares is not None:
        raise WeightingError.targets_conflict(where=where, method=method)

    if controls is None and shares is None:
        if not counts_when_none:
            raise WeightingError.targets_missing(where=where, method=method)
        counts = np.bincount(spec.codes[spec.codes >= 0], minlength=spec.n_cells).astype(
            np.float64
        )
        return counts

    if shares is not None:
        if _is_scalar(shares):
            raise WeightingError.targets_type(
                where=where,
                param="shares",
                got=shares,
                expected="a dict with one entry per cell",
                hint=(
                    "`shares` must be a dict; for an equal composition build it "
                    "explicitly, e.g. dict.fromkeys(levels, 1)."
                ),
            )
        if not _is_mapping(shares):
            raise WeightingError.targets_type(
                where=where, param="shares", got=shares, expected="a dict[cell, number]"
            )
        if not spec.has_cells:
            raise WeightingError.cells_required(
                where=where,
                param="shares",
                reason="`shares` describes a composition across cells, so cells= is required.",
                hint="Name the composition axis with cells=, or use controls= for a grand total.",
            )
        vec = cell_targets(shares, spec, param="shares", where=where)
        total_share = float(vec.sum())
        if total_share <= 0:
            raise WeightingError.all_zero(where=where, param="shares")
        # Normalized internally: only composition matters, and the in-scope
        # weight total is carried through unchanged.
        return (vec / total_share) * in_scope_sum(wgt_arr, spec)

    if _is_scalar(controls):
        if spec.has_cells:
            raise WeightingError.scalar_for_cells(
                where=where, param="controls", got=controls, levels=spec.labels
            )
        val = float(controls)
        if not np.isfinite(val) or val < 0:
            raise WeightingError.value_invalid(
                where=where, param="controls", bad={"controls": controls}
            )
        return np.array([val], dtype=np.float64)

    if not _is_mapping(controls):
        raise WeightingError.targets_type(
            where=where, param="controls", got=controls, expected="a number or dict[cell, number]"
        )
    if not spec.has_cells:
        raise WeightingError.cells_required(
            where=where,
            param="controls",
            reason="`controls` is a dict but no cells were named.",
            hint="Name the cells with cells=, or pass a single number for the grand total.",
        )
    return cell_targets(controls, spec, param="controls", where=where)


def scale_to_targets(wgts: np.ndarray, spec: CellSpec, targets: np.ndarray) -> np.ndarray:
    """Scale each cell to its target. Out-of-scope rows keep their weight (g = 1).

    ``wgts`` is (n, n_cols); the same targets apply to every column, which is what
    makes each replicate hit the same absolute totals as the main weight.
    """
    assert _rust_poststratify is not None  # noqa: S101
    if spec.in_scope is None:
        return _rust_poststratify(
            np.ascontiguousarray(wgts, dtype=np.float64), spec.codes, targets
        )

    out = np.array(wgts, dtype=np.float64, copy=True)
    idx = np.flatnonzero(spec.in_scope)
    out[idx] = _rust_poststratify(
        np.ascontiguousarray(out[idx], dtype=np.float64), spec.codes[idx], targets
    )
    return out


#: Prefix for snapshotted cell columns. These are hidden from user-facing
#: output and exist so the variance sweep can reproduce the exact membership
#: the adjustment used -- a `where` expression is not reliably replayable, and
#: a cells column may legitimately be recoded afterwards.
CELLS_PREFIX = "__svy_cells_"

#: Prefix for materialized calibration auxiliary columns.
AUX_PREFIX = "__svy_aux_"


def materialize_cells(
    df: pl.DataFrame, spec: CellSpec, *, wgt_name: str, margin: int = 0
) -> tuple[pl.DataFrame, str]:
    """Snapshot a cell coding into a hidden column, null outside the scope.

    One column per margin: raking sweeps each margin separately, so a single
    concatenated A x B column would encode poststratification on the full
    cross, which is a different and stronger calibration.
    """
    name = f"{CELLS_PREFIX}{wgt_name}" if margin == 0 else f"{CELLS_PREFIX}{wgt_name}_{margin}"
    values = [None if c < 0 else int(c) for c in spec.codes]
    return df.with_columns(pl.Series(name=name, values=values, dtype=pl.Int32)), name
