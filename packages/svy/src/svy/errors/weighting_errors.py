# svy/errors/weighting_errors.py
"""Errors raised by ``Sample.weighting`` for inputs the caller can fix.

Every constructor carries a stable code, ``expected``/``got`` in the data's own
values (lists and dicts, never their printed form) and a hint written with the
caller's names.
"""

from __future__ import annotations

import datetime as _dt
import difflib

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

from .method_errors import MethodError


_SHOWN = 8


def _user_columns(cols: Iterable[str]) -> list[str]:
    """Data columns without svy's own bookkeeping columns."""
    return [
        c
        for c in cols
        if not (
            c.startswith("__svy")
            or c == "svy_row_index"
            or c.endswith("_svy_internal_cols_concatenated")
        )
    ]


def show(v: Any) -> str:
    """A value as the caller would type it (``'a'``, ``1``, ``True``, a tuple)."""
    item = getattr(v, "item", None)
    if callable(item) and not isinstance(v, (str, bytes)):
        try:
            v = item()
        except (TypeError, ValueError):
            pass
    if isinstance(v, tuple):
        inner = ", ".join(show(p) for p in v)
        return f"({inner},)" if len(v) == 1 else f"({inner})"
    if isinstance(v, (_dt.date, _dt.time)):
        return repr(v.isoformat())
    return repr(v)


def show_list(values: Iterable[Any], limit: int = _SHOWN) -> str:
    vals = list(values)
    shown = ", ".join(show(v) for v in vals[:limit])
    more = len(vals) - limit
    return f"[{shown}]" if more <= 0 else f"[{shown}, +{more} more]"


def _example(levels: Sequence[Any], value: str = "...") -> str:
    parts = [f"{show(lv)}: {value}" for lv in list(levels)[:3]]
    if len(levels) > 3:
        parts.append("...")
    return "{" + ", ".join(parts) + "}"


def _sorted(values: Iterable[Any]) -> list[Any]:
    vals = list(values)
    try:
        return sorted(vals)
    except TypeError:
        return sorted(vals, key=show)


@dataclass(eq=False)
class WeightingError(MethodError):
    """A weighting call that cannot run on the inputs given."""

    def __post_init__(self) -> None:
        if self.code == "SVY_ERROR":
            self.code = "WEIGHTING_ERROR"

    # ---- targets (controls / shares / resp_mapping) ----------------------

    @classmethod
    def keys_mismatch(
        cls,
        *,
        where: Optional[str],
        param: str,
        levels: Sequence[Any],
        missing: Sequence[Any] = (),
        extra: Sequence[Any] = (),
        cols: Sequence[str] | None = None,
    ) -> "WeightingError":
        miss, ext = list(missing), list(extra)
        on = ", ".join(repr(c) for c in cols) if cols else "the cells"
        parts = []
        if miss:
            parts.append(f"no target for {show_list(miss)}")
        if ext:
            parts.append(f"{show_list(ext)} match no level in the data")
        tuple_keys = bool(levels) and isinstance(levels[0], tuple)
        hint = f"Key {param} by the values of {on}, one entry per level: {_example(levels)}."
        if tuple_keys:
            hint += (
                " Several columns take a tuple in that column order, or its parts joined by '_&_'."
            )
        hint += " A text form of a value also works (e.g. '1' for 1, 'true' for True)."
        if ext:
            hint += (
                " A level absent from the data (or outside `where`) can only take a target of 0."
            )
        return cls(
            title="Target keys do not match the data",
            detail=f"Mapping keys mismatch in `{param}`: " + "; ".join(parts) + ".",
            code="CONTROLS_KEYS_MISMATCH",
            where=where,
            param=param,
            expected=list(levels),
            got={"missing": miss, "extra": ext},
            hint=hint,
        )

    @classmethod
    def key_ambiguous(
        cls, *, where: Optional[str], param: str, key: Any, matches: Sequence[Any]
    ) -> "WeightingError":
        m = _sorted(matches)
        return cls(
            title="Target key matches several levels",
            detail=f"The key {show(key)} in `{param}` reads as each of {show_list(m)}.",
            code="CONTROLS_KEY_AMBIGUOUS",
            where=where,
            param=param,
            expected=m,
            got=key,
            hint=f"Key it by the level's own value, e.g. {show(m[0])} instead of {show(key)}.",
        )

    @classmethod
    def key_duplicate(
        cls, *, where: Optional[str], param: str, level: Any, keys: Sequence[Any]
    ) -> "WeightingError":
        return cls(
            title="Two keys name the same level",
            detail=f"The keys {show_list(keys)} in `{param}` both name the level {show(level)}.",
            code="CONTROLS_KEY_DUPLICATE",
            where=where,
            param=param,
            expected=level,
            got=list(keys),
            hint=f"Keep one entry for {show(level)}.",
        )

    @classmethod
    def value_invalid(
        cls, *, where: Optional[str], param: str, bad: Mapping[Any, Any], nonneg: bool = True
    ) -> "WeightingError":
        kind = "finite, non-negative numbers" if nonneg else "finite numbers"
        return cls(
            title="Invalid target values",
            detail=(
                f"`{param}` values must all be {kind}; "
                f"got {', '.join(f'{show(k)}: {show(v)}' for k, v in list(bad.items())[:_SHOWN])}."
            ),
            code="CONTROLS_VALUE_INVALID",
            where=where,
            param=param,
            expected="finite number >= 0" if nonneg else "finite number",
            got=dict(bad),
            hint="Give every level a number of 0 or more."
            if nonneg
            else "Give every level a number.",
        )

    @classmethod
    def all_zero(cls, *, where: Optional[str], param: str) -> "WeightingError":
        return cls(
            title="Targets are all zero",
            detail=f"`{param}` must include at least one positive value.",
            code="CONTROLS_ALL_ZERO",
            where=where,
            param=param,
            hint="At least one level needs a positive target.",
        )

    @classmethod
    def targets_missing(cls, *, where: Optional[str], method: str) -> "WeightingError":
        return cls(
            title="No targets",
            detail=f"{method}() needs targets: Either controls= or shares= must be specified.",
            code="CONTROLS_MISSING",
            where=where,
            param="controls",
            hint="controls= sets the totals; shares= fixes the composition and keeps the total.",
        )

    @classmethod
    def targets_conflict(cls, *, where: Optional[str], method: str) -> "WeightingError":
        return cls(
            title="Both controls and shares",
            detail=f"{method}() takes exactly one of controls= or shares=, not both.",
            code="CONTROLS_CONFLICT",
            where=where,
            param="controls",
            hint="controls= sets the totals; shares= fixes the composition and keeps the total.",
        )

    @classmethod
    def targets_type(
        cls, *, where: Optional[str], param: str, got: Any, expected: str, hint: str | None = None
    ) -> "WeightingError":
        return cls(
            title="Targets have the wrong type",
            detail=f"`{param}` must be {expected}; got {type(got).__name__}.",
            code="CONTROLS_TYPE_INVALID",
            where=where,
            param=param,
            expected=expected,
            got=type(got).__name__,
            hint=hint,
        )

    @classmethod
    def scalar_for_cells(
        cls, *, where: Optional[str], param: str, got: Any, levels: Sequence[Any] | None = None
    ) -> "WeightingError":
        example = _example(levels, "100") if levels else "{'a': 100, 'b': 250}"
        return cls(
            title="One number for several cells",
            detail=(
                f"`{param}` is a single number but `cells` names one or more columns. "
                "A scalar sets one cell's total, so it cannot describe several cells."
            ),
            code="CONTROLS_SCALAR_FOR_CELLS",
            where=where,
            param=param,
            expected="dict with one entry per cell",
            got=got,
            hint=(
                f"Pass one entry per cell, e.g. {param}={example}, or use shares= to "
                "fix the composition, or drop cells= to set the grand total."
            ),
        )

    @classmethod
    def cells_required(
        cls, *, where: Optional[str], param: str, reason: str, hint: str
    ) -> "WeightingError":
        return cls(
            title="cells= is required",
            detail=reason,
            code="CELLS_REQUIRED",
            where=where,
            param=param,
            hint=hint,
        )

    @classmethod
    def margins_disagree(
        cls, *, where: Optional[str], totals: Mapping[str, float]
    ) -> "WeightingError":
        listed = ", ".join(f"{c}={t:,.6g}" for c, t in totals.items())
        first = next(iter(totals))
        return cls(
            title="Margins disagree on the total",
            detail=f"Margins disagree on the population total: {listed}.",
            code="MARGINS_DISAGREE",
            where=where,
            param="controls",
            got=dict(totals),
            hint=(
                f"Every margin must sum to the same total (e.g. scale each to "
                f"{totals[first]:,.6g}, the total of {first!r}), or pass shares= to have "
                "them normalized against one grand total."
            ),
        )

    # ---- response status (adjust) -----------------------------------------

    @classmethod
    def resp_status_unknown(
        cls,
        *,
        where: Optional[str],
        column: str,
        counts: Mapping[Any, int],
        allowed: Sequence[Any],
        mapping: Mapping[str, Any] | None,
    ) -> "WeightingError":
        unknown = list(counts)
        if mapping is None:
            example: dict[str, list[Any]] = {"rr": ["rr"], "nr": ["nr", *unknown]}
        else:
            example = {
                k: list(v) if isinstance(v, (list, tuple, set, frozenset)) else [v]
                for k, v in mapping.items()
            }
            example.setdefault("nr", [])
            example["nr"] = [*example["nr"], *unknown]
        ex = "{" + ", ".join(f"{k!r}: {show_list(v, limit=20)}" for k, v in example.items()) + "}"
        return cls(
            title="Unknown response statuses",
            detail=(
                f"Values of {column!r} match no response status: "
                + ", ".join(
                    f"{show(v)} ({n} row{'' if n == 1 else 's'})"
                    for v, n in list(counts.items())[:_SHOWN]
                )
                + ". Every row needs one, since an unmatched row would count as a respondent."
            ),
            code="RESP_STATUS_UNKNOWN",
            where=where,
            param="resp_mapping" if mapping is not None else "resp_status",
            expected=list(allowed),
            got=dict(counts),
            hint=(
                "Map each value to rr (respondent), nr (nonrespondent), in (ineligible) or "
                f"uk (unknown), e.g. resp_mapping={ex} if they are nonrespondents."
            ),
        )

    @classmethod
    def resp_mapping_key(cls, *, where: Optional[str], key: Any) -> "WeightingError":
        return cls(
            title="Unknown response status code",
            detail=f"resp_mapping key {show(key)} is not a response status code.",
            code="RESP_MAPPING_KEY_INVALID",
            where=where,
            param="resp_mapping",
            expected=["rr", "nr", "in", "uk"],
            got=key,
            hint="Key resp_mapping by rr, nr, in or uk, e.g. {'rr': 1, 'nr': [2, 3]}.",
        )

    @classmethod
    def resp_mapping_conflict(
        cls, *, where: Optional[str], conflicts: Mapping[Any, Sequence[str]]
    ) -> "WeightingError":
        listed = "; ".join(
            f"{show(v)} under {', '.join(codes)}" for v, codes in list(conflicts.items())[:_SHOWN]
        )
        return cls(
            title="A status is mapped twice",
            detail=f"resp_mapping puts a value under two statuses: {listed}.",
            code="RESP_MAPPING_CONFLICT",
            where=where,
            param="resp_mapping",
            got={v: list(c) for v, c in conflicts.items()},
            hint="List each value under one of rr, nr, in, uk only.",
        )

    # ---- columns, names and scope -----------------------------------------

    @classmethod
    def missing_columns(
        cls,
        *,
        where: Optional[str],
        param: str,
        missing: Sequence[str],
        available: Sequence[str],
        hint: str | None = None,
    ) -> "WeightingError":
        miss = list(missing)
        cols = _user_columns(available)
        close = [
            f"{m!r} -> {c[0]!r}"
            for m in miss
            if isinstance(m, str) and (c := difflib.get_close_matches(m, cols, n=1))
        ]
        guess = f" Did you mean {', '.join(close)}?" if close else ""
        return cls(
            title="Column(s) not found",
            detail=f"`{param}` names {show_list(miss)}, not found in data.",
            code="MISSING_COLUMNS",
            where=where,
            param=param,
            expected=cols,
            got=miss,
            hint=(hint or "Check spelling or pass existing column names.") + guess,
        )

    @classmethod
    def no_weight(cls, *, where: Optional[str], method: str) -> "WeightingError":
        return cls(
            title="No sample weight",
            detail=f"Sample weight is None. Set design.wgt before calling {method}().",
            code="WGT_MISSING",
            where=where,
            param="design.wgt",
            hint="sample.update_design(wgt='<weight column>').",
        )

    @classmethod
    def wgt_name_exists(
        cls, *, where: Optional[str], method: str, wgt_name: str, existing: Iterable[str]
    ) -> "WeightingError":
        taken = set(existing)
        k = 2
        while f"{wgt_name}_{k}" in taken:
            k += 1
        return cls(
            title="Weight name already used",
            detail=f"Column '{wgt_name}' already exists. Choose a different wgt_name.",
            code="WGT_NAME_EXISTS",
            where=where,
            param="wgt_name",
            got=wgt_name,
            hint=f"e.g. {method}(..., wgt_name={wgt_name + '_' + str(k)!r}).",
        )

    @classmethod
    def no_rows_in_scope(
        cls,
        *,
        where: Optional[str],
        method: str,
        hint: str | None = None,
        nulls: Mapping[str, int] | None = None,
    ) -> "WeightingError":
        if nulls:
            listed = ", ".join(f"{c!r} {n}" for c, n in nulls.items())
            return cls(
                title="Nothing to adjust",
                detail=(
                    f"No rows are in scope for this adjustment ({method}): every row "
                    f"within `where` has a null cell ({listed})."
                ),
                code="NO_ROWS_IN_SCOPE",
                where=where,
                param="cells",
                expected="at least one row with every cell present",
                got=dict(nulls),
                hint="Fill the nulls (wrangling.fill_null) or choose other cells.",
            )
        return cls(
            title="Nothing to adjust",
            detail=f"No rows are in scope for this adjustment ({method}).",
            code="NO_ROWS_IN_SCOPE",
            where=where,
            param="where",
            expected="at least one row",
            got=0,
            hint=hint or "Check `where` and for nulls in the `cells` columns.",
        )

    @classmethod
    def where_invalid(cls, *, where: Optional[str], reason: str) -> "WeightingError":
        return cls(
            title="`where` cannot be evaluated",
            detail=f"`where` could not be evaluated against the data: {reason}",
            code="WHERE_INVALID",
            where=where,
            param="where",
            hint="Check that every column referenced by `where` exists.",
        )

    @classmethod
    def columns_empty(cls, *, where: Optional[str], param: str) -> "WeightingError":
        return cls(
            title="No columns named",
            detail=f"`{param}` sequence must not be empty.",
            code="COLUMNS_EMPTY",
            where=where,
            param=param,
            expected="a column name or a non-empty list of them",
            got=[],
            hint=f"Pass {param}=None for no grouping, or name at least one column.",
        )

    @classmethod
    def param_renamed(
        cls, *, where: Optional[str], method: str, old: str, new: str, note: str = ""
    ) -> "WeightingError":
        return cls(
            title="Parameter renamed",
            detail=f"'{method}' cannot be used here: `{old}=` was renamed to `{new}=`.",
            code="PARAM_RENAMED",
            where=where,
            param=old,
            expected=new,
            got=old,
            hint=f"Replace {old}= with {new}=." + (f" {note}" if note else ""),
        )

    # ---- method-specific ---------------------------------------------------

    @classmethod
    def not_converged(
        cls,
        *,
        where: Optional[str],
        method: str,
        what: str,
        max_iter: int,
        hint: str,
        got: Mapping[str, Any] | None = None,
        expected: Mapping[str, Any] | None = None,
    ) -> "WeightingError":
        return cls(
            title="Did not converge",
            detail=(
                f"{what} did not converge after {max_iter} iterations. The sample has "
                'NOT been modified. Pass on_nonconvergence="warn" to keep the last iterate.'
            ),
            code="CONVERGENCE_FAILED",
            where=where,
            param="max_iter",
            expected=dict(expected) if expected else None,
            got=dict(got) if got else {"max_iter": max_iter},
            hint=hint,
        )

    @classmethod
    def bounds_exceeded(
        cls, *, where: Optional[str], bounds: tuple[float | None, float | None]
    ) -> "WeightingError":
        return cls(
            title="Adjustment factor out of bounds",
            detail=(
                f"The raked weights leave the factor g = new/old weight outside "
                f"bounds={show(tuple(bounds))}. Bounds are checked after raking, "
                "not enforced during it."
            ),
            code="BOUNDS_EXCEEDED",
            where=where,
            param="bounds",
            expected={"bounds": list(bounds)},
            hint="Widen the bounds, relax the margins, or collapse sparse levels.",
        )

    @classmethod
    def bounds_invalid(
        cls, *, where: Optional[str], got: Any, reason: str, as_type: bool = False
    ) -> "WeightingError":
        kind = _TypeWeightingError if as_type else _ValueWeightingError
        return kind(
            title="Invalid bounds",
            detail=f"`bounds` {reason}.",
            code="INVALID_TYPE" if as_type else "INVALID_RANGE",
            where=where,
            param="bounds",
            expected="None or a 2-tuple (lo, hi) of numbers or None, with lo <= hi",
            got=got if isinstance(got, (int, float, str, type(None))) else show(got),
            hint=(
                "bounds=(lo, hi) bounds the factor g = new/old weight; None on a side "
                "leaves it open, e.g. bounds=(0.5, None) or bounds=(0.5, 2.0)."
            ),
        )

    @classmethod
    def not_supported_yet(
        cls, *, where: Optional[str], param: str, got: Any, hint: str
    ) -> "WeightingError":
        return _NotImplementedWeightingError(
            title="Not supported yet",
            detail=f"`{param}` is not supported by this method yet.",
            code="NOT_SUPPORTED",
            where=where,
            param=param,
            expected=None,
            got=got if isinstance(got, (int, float, str, type(None))) else show(got),
            hint=hint,
        )

    @classmethod
    def calibration_not_met(
        cls,
        *,
        where: Optional[str],
        expected: Any = None,
        got: Any = None,
        max_rel_error: float | None = None,
        domains: Sequence[Any] = (),
    ) -> "WeightingError":
        doms = list(domains)
        scope = f" in domain(s) {show_list(doms)}" if doms else ""
        miss = f" (largest relative miss {max_rel_error:.3g})" if max_rel_error is not None else ""
        return cls(
            title="Controls not met",
            detail=(
                f"The calibrated weights do not reproduce the control totals{scope} within "
                f"tolerance{miss}. The system may be singular or ill-conditioned, or the "
                'controls inconsistent. Pass on_nonconvergence="warn" to keep the approximate '
                "solution."
            ),
            code="CALIBRATION_NOT_MET",
            where=where,
            param="controls",
            expected=expected,
            got=got,
            hint=(
                "Check for collinear auxiliaries (one implied by others), controls that "
                "contradict each other, or a domain too small for its auxiliaries."
            ),
            extra={"max_rel_error": max_rel_error, "tol": 1e-4, "domains": doms},
        )

    @classmethod
    def cells_by_overlap(cls, *, where: Optional[str], overlap: Sequence[str]) -> "WeightingError":
        ov = sorted(overlap)
        return cls(
            title="Same column in by= and cells=",
            detail=(
                f"{ov} appears in both by= and cells=. Domains and the composition axis "
                "must be different variables."
            ),
            code="CELLS_BY_OVERLAP",
            where=where,
            param="by",
            got=ov,
            hint=f"Drop {ov[0]!r} from one of them.",
        )

    @classmethod
    def factor_conflict(cls, *, where: Optional[str], given: Sequence[str]) -> "WeightingError":
        return cls(
            title="factor= combined with targets",
            detail=(
                "factor multiplies every weight and cannot be combined with controls, "
                f"shares, cells or where (got {', '.join(given)})."
            ),
            code="FACTOR_CONFLICT",
            where=where,
            param="factor",
            got=list(given),
            hint="Call normalize(factor=...) alone, or drop factor= and give targets.",
        )

    @classmethod
    def ref_level_unknown(
        cls, *, where: Optional[str], column: str, ref: Any, levels: Sequence[Any]
    ) -> "WeightingError":
        return cls(
            title="Reference level not found",
            detail=f"Reference level {show(ref)} not found in {column!r}.",
            code="REF_LEVEL_UNKNOWN",
            where=where,
            param=f"Cat({column!r}).ref",
            expected=list(levels),
            got=ref,
            hint=f"Use one of {show_list(levels)}.",
        )

    @classmethod
    def no_terms(cls, *, where: Optional[str]) -> "WeightingError":
        return cls(
            title="No auxiliaries",
            detail="No terms specified.",
            code="AUX_TERMS_EMPTY",
            where=where,
            param="x",
            hint="Name at least one auxiliary, e.g. x=['income', Cat('region')].",
        )

    @classmethod
    def term_invalid(cls, *, where: Optional[str], term: Any) -> "WeightingError":
        return cls(
            title="Unsupported term",
            detail=f"{type(term).__name__} is not a calibration term.",
            code="TERM_INVALID",
            where=where,
            param="controls",
            expected="str | Cat | Cross",
            got=type(term).__name__,
            hint="Name a continuous auxiliary by its column, a categorical one with Cat(...).",
        )

    @classmethod
    def trim_bounds_missing(cls, *, where: Optional[str]) -> "WeightingError":
        return _ValueWeightingError(
            title="No trimming bounds",
            detail="At least one of `upper` or `lower` must be specified.",
            code="TRIM_BOUNDS_MISSING",
            where=where,
            param="upper",
            hint="e.g. upper=Threshold.quantile(0.99), or lower=0.5.",
        )

    @classmethod
    def psu_required(
        cls, *, where: Optional[str], method: str, note: str = ""
    ) -> "WeightingError":
        return cls(
            title="PSU required",
            detail=f"{method}() resamples PSUs and requires psu (got psu=None).",
            code="PSU_REQUIRED",
            where=where,
            param="psu",
            hint="Pass psu=, or set it on the Design." + (f" {note}" if note else ""),
        )

    @classmethod
    def threshold_invalid(
        cls, *, where: Optional[str], got: Any, reason: str, as_type: bool = False
    ) -> "WeightingError":
        kind = _TypeWeightingError if as_type else _ValueWeightingError
        return kind(
            title="Invalid trimming threshold",
            detail=reason,
            code="THRESHOLD_INVALID",
            where=where,
            param="upper/lower",
            expected="a number > 0, a Threshold, or a callable",
            got=got if isinstance(got, (int, float, str)) else type(got).__name__,
            hint="A number is an absolute bound; for a quantile use svy.Threshold.quantile(p).",
        )

    @classmethod
    def trim_param_range(
        cls, *, where: Optional[str], param: str, got: Any, expected: str
    ) -> "WeightingError":
        return _ValueWeightingError(
            title="Invalid trimming setting",
            detail=f"{param} must be {expected}, got {got}.",
            code="INVALID_RANGE",
            where=where,
            param=param,
            expected=expected,
            got=got,
            hint=f"Pass {param} {expected}.",
        )


@dataclass(eq=False)
class _ValueWeightingError(WeightingError, ValueError):
    """For inputs that raised ValueError before they had a code."""


@dataclass(eq=False)
class _TypeWeightingError(WeightingError, TypeError):
    """For inputs that raised TypeError before they had a code."""


@dataclass(eq=False)
class _NotImplementedWeightingError(WeightingError, NotImplementedError):
    """For options that raised NotImplementedError before they had a code."""
