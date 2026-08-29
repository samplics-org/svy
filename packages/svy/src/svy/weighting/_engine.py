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

from svy.errors import MethodError
from svy.utils.where import _compile_where


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


def _cells_to_cols(cells: str | Sequence[str] | None, *, where: str) -> list[str] | None:
    if cells is None:
        return None
    if isinstance(cells, str):
        return [cells]
    if isinstance(cells, Sequence) and not isinstance(cells, (bytes, bytearray)):
        cols = list(cells)
        if not cols:
            raise MethodError.not_applicable(
                where=where, method="weighting", reason="`cells` sequence must not be empty."
            )
        for c in cols:
            if not isinstance(c, str):
                raise MethodError.invalid_type(
                    where=where, param="cells", got=c, expected="str | Sequence[str] | None"
                )
        return cols
    raise MethodError.invalid_type(
        where=where, param="cells", got=cells, expected="str | Sequence[str] | None"
    )


def _where_mask(df: pl.DataFrame, where_arg: WhereArg, *, where: str) -> np.ndarray | None:
    expr = _compile_where(where_arg)
    if expr is None:
        return None
    try:
        mask = df.select(expr.alias("__svy_scope__")).get_column("__svy_scope__")
    except Exception as e:
        raise MethodError.not_applicable(
            where=where,
            method="weighting",
            reason=f"`where` could not be evaluated against the data: {e}",
            hint="Check that every column referenced by `where` exists.",
        ) from e
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
            codes = np.where(scope, 0, -1).astype(np.int64)
        return CellSpec(codes, [None], scope, None)

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise MethodError.invalid_choice(
            where=where,
            param="cells",
            got=missing,
            allowed=list(df.columns),
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
        raise MethodError.not_applicable(
            where=where,
            method="weighting",
            reason="No rows are in scope for this adjustment.",
            hint="Check `where` and for nulls in the `cells` columns.",
        )

    in_scope = codes >= 0
    return CellSpec(codes, labels, in_scope if in_scope.sum() < n else None, cols)


def _is_mapping(x: Any) -> bool:
    return isinstance(x, Mapping)


def _is_scalar(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)


def _scalar_with_cells_error(param: str, *, method: str, where: str) -> MethodError:
    return MethodError.not_applicable(
        where=where,
        method=method,
        reason=(
            f"`{param}` is a single number but `cells` names one or more columns. "
            "A scalar sets one cell's total, so it cannot describe several cells"
        ),
        param=param,
        hint=(
            f"Pass a dict with one entry per cell (e.g. {param}={{'a': 100, 'b': 250}}), "
            "or use shares= to pin composition, or drop cells= to set the grand total."
        ),
    )


def _validate_keys(
    supplied: Mapping[Any, Any], labels: list[Any], *, param: str, method: str, where: str
) -> None:
    data_keys = set(labels)
    given = set(supplied.keys())
    missing = data_keys - given
    extra = given - data_keys
    if missing or extra:
        raise MethodError.invalid_mapping_keys(
            where=where,
            param=param,
            missing=sorted(missing, key=str),
            extra=sorted(extra, key=str),
            hint=(
                "Keys must match the cell values exactly, one per cell. For several "
                "`cells` columns use a tuple key in cells order, e.g. ('R1', 'D1')."
            ),
        )


def _as_float_vector(
    supplied: Mapping[Any, Any], labels: list[Any], *, param: str, method: str, where: str
) -> np.ndarray:
    out = np.empty(len(labels), dtype=np.float64)
    for i, lab in enumerate(labels):
        try:
            out[i] = float(supplied[lab])
        except (TypeError, ValueError) as e:
            raise MethodError.invalid_type(
                where=where, param=f"{param}[{lab!r}]", got=supplied[lab], expected="a number"
            ) from e
    if not np.all(np.isfinite(out)):
        raise MethodError.not_applicable(
            where=where,
            method=method,
            reason=f"`{param}` values must all be finite",
            param=param,
        )
    if np.any(out < 0):
        raise MethodError.not_applicable(
            where=where,
            method=method,
            reason=f"`{param}` values must be non-negative",
            param=param,
        )
    return out


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
        raise MethodError.not_applicable(
            where=where,
            method=method,
            reason="Provide exactly one of controls= or shares=, not both",
            hint="controls= sets the total; shares= preserves it.",
        )

    if controls is None and shares is None:
        if not counts_when_none:
            raise MethodError.not_applicable(
                where=where,
                method=method,
                reason="Either controls= or shares= must be specified",
            )
        counts = np.bincount(spec.codes[spec.codes >= 0], minlength=spec.n_cells).astype(
            np.float64
        )
        return counts

    if shares is not None:
        if _is_scalar(shares):
            raise MethodError.not_applicable(
                where=where,
                method=method,
                reason="`shares` must be a dict with one entry per cell",
                param="shares",
                hint=(
                    "For an equal composition build the dict explicitly, e.g. "
                    "dict.fromkeys(levels, 1)."
                ),
            )
        if not _is_mapping(shares):
            raise MethodError.invalid_type(
                where=where, param="shares", got=shares, expected="dict[cell, number]"
            )
        if not spec.has_cells:
            raise MethodError.not_applicable(
                where=where,
                method=method,
                reason="`shares` describes a composition across cells, so cells= is required",
                param="shares",
                hint="Name the composition axis with cells=, or use controls= for a grand total.",
            )
        _validate_keys(shares, spec.labels, param="shares", method=method, where=where)
        vec = _as_float_vector(shares, spec.labels, param="shares", method=method, where=where)
        total_share = float(vec.sum())
        if total_share <= 0:
            raise MethodError.not_applicable(
                where=where,
                method=method,
                reason="`shares` must include at least one positive value",
                param="shares",
            )
        # Normalized internally: only composition matters, and the in-scope
        # weight total is carried through unchanged.
        return (vec / total_share) * in_scope_sum(wgt_arr, spec)

    if _is_scalar(controls):
        if spec.has_cells:
            raise _scalar_with_cells_error("controls", method=method, where=where)
        val = float(controls)
        if not np.isfinite(val) or val < 0:
            raise MethodError.not_applicable(
                where=where,
                method=method,
                reason="`controls` must be a finite, non-negative number",
                param="controls",
            )
        return np.array([val], dtype=np.float64)

    if not _is_mapping(controls):
        raise MethodError.invalid_type(
            where=where, param="controls", got=controls, expected="number | dict[cell, number]"
        )
    if not spec.has_cells:
        raise MethodError.not_applicable(
            where=where,
            method=method,
            reason="`controls` is a dict but no cells were named",
            param="controls",
            hint="Name the cells with cells=, or pass a single number for the grand total.",
        )
    _validate_keys(controls, spec.labels, param="controls", method=method, where=where)
    return _as_float_vector(controls, spec.labels, param="controls", method=method, where=where)


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
