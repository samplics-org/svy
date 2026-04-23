# src/svy/core/design.py
from __future__ import annotations

import logging
import re

from typing import (
    Any,
    Literal,
    NamedTuple,
    Self,
    Sequence,
    TypeGuard,
    TypeVar,
    cast,
    overload,
)

import msgspec

from svy.core.enumerations import EstimationMethod as _EstimationMethod
from svy.ui.printing import make_panel, render_rich_to_str, resolve_width


log = logging.getLogger(__name__)


# =============================================================================
# Types & Sentinels
# =============================================================================


class _MissingType:
    pass


_MISSING = _MissingType()


def _is_MissingType(x: Any) -> TypeGuard[_MissingType]:
    return x is _MISSING


# =============================================================================
# Population Size (FPC)
# =============================================================================


class PopSize(NamedTuple):
    """
    Finite population correction (FPC) specification for multistage designs.

    Holds column names referencing population sizes at two stages:
    the PSU level and the SSU level (the second sampling stage).

    Parameters
    ----------
    psu : str
        Column name for the population size at the PSU level
        (e.g., total number of hospitals in a stratum).
    ssu : str
        Column name for the population size at the SSU level
        (e.g., total number of schools within a district, or total
        number of discharges within a hospital when there is no
        intermediate stage).

    Examples
    --------
    >>> PopSize(psu="TOTAL_HOSP", ssu="TOTAL_DISCHARGES")
    PopSize(psu='TOTAL_HOSP', ssu='TOTAL_DISCHARGES')

    Notes
    -----
    For single-stage FPC, use a plain ``str`` for the ``pop_size`` field
    in :class:`Design` instead of ``PopSize``.

    In the standard two-stage variance approximation, the variance is
    computed from PSU-level score totals. The PSU FPC corrects for not
    sampling all PSUs; the SSU FPC corrects for not sampling all SSUs
    within each PSU. Unit-level variability is absorbed into the
    PSU-level residuals.
    """

    psu: str
    ssu: str


# =============================================================================
# Replicate Weights (Strict Configuration)
# =============================================================================


class RepWeights(msgspec.Struct, frozen=True):
    """
    Strict definition of Replicate Weights.
    This object should ONLY be instantiated if a valid replicate design exists.
    """

    # Required Fields (No Defaults)
    method: _EstimationMethod | str
    prefix: str
    n_reps: int

    # Optional / defaulted fields
    fay_coef: float = 0.0
    df: int | None = None  # None = "Calculate from data", Int = "User Override"
    padding: int | None = None  # None = auto-detect, 0 = no padding, >0 = zero-pad width

    def __post_init__(self):
        """
        Enforce strict statistical validity upon initialization.
        """
        # 0. Normalize string method to enum (accept user-facing strings).
        #    Check enum first because EstimationMethod is a str enum.
        if isinstance(self.method, _EstimationMethod):
            pass  # already an enum, let step 1 validate it
        elif isinstance(self.method, str):
            object.__setattr__(self, "method", _normalize_rep_method(self.method))
        else:
            raise TypeError(
                f"'method' must be a string or EstimationMethod, got {type(self.method).__name__}."
            )

        # 1. Method Validation: Taylor is not a replicate method.
        valid_methods = (
            _EstimationMethod.BRR,
            _EstimationMethod.BOOTSTRAP,
            _EstimationMethod.JACKKNIFE,
            _EstimationMethod.SDR,
        )
        if self.method not in valid_methods:
            raise ValueError(
                f"Method '{self.method}' is not a valid replication method "
                f"(expected one of: {[m.value for m in valid_methods]})."
            )

        # 2. Prefix Validation: Must be a non-empty string.
        if not self.prefix or not self.prefix.strip():
            raise ValueError("RepWeights 'prefix' cannot be empty or whitespace.")

        # 3. Replicate Count Validation: Must be >= 2.
        if self.n_reps < 2:
            raise ValueError(f"n_reps must be >= 2. Got {self.n_reps}.")

        # 4. Fay Coefficient Validation: Sanity check (usually [0, 1]).
        if self.fay_coef < 0:
            raise ValueError(f"fay_coef cannot be negative. Got {self.fay_coef}.")

        # 5. Degrees of Freedom Validation: Must be positive if manually set.
        if self.df is not None and self.df <= 0:
            raise ValueError(f"df must be > 0. Got {self.df}.")

        # 6. Padding Validation: Must be non-negative if set.
        if self.padding is not None and self.padding < 0:
            raise ValueError(f"padding must be >= 0. Got {self.padding}.")


    def _detect_padding(self, data_columns: Sequence[str]) -> int:
        """
        Detect zero-padding width from existing column names in data.

        Matches are case-insensitive on the prefix.
        """
        pattern = re.compile(rf"^{re.escape(self.prefix)}(\d+)$", re.IGNORECASE)
        max_padding = 0
        for col in data_columns:
            match = pattern.match(col)
            if match:
                num_str = match.group(1)
                if len(num_str) > 1 and num_str[0] == "0":
                    max_padding = max(max_padding, len(num_str))
        return max_padding


    def _generate_columns(self, padding: int) -> list[str]:
        """Generate canonical column names with specified padding (exact case)."""
        if padding > 0:
            return [f"{self.prefix}{i:0{padding}d}" for i in range(1, self.n_reps + 1)]
        else:
            return [f"{self.prefix}{i}" for i in range(1, self.n_reps + 1)]


    def columns_from_data(self, data_columns: Sequence[str]) -> list[str]:
        """
        Generate column names, auto-detecting padding from actual data columns.

        Prefix matching is case-insensitive for padding detection. The returned
        names use the casing of the *first* matching column in ``data_columns``
        if any match the prefix; otherwise the casing specified at construction.

        Does NOT validate that all n_reps columns exist in data — this is a
        pure name generator. Validation happens downstream in
        ``Sample._validate_design``.
        """
        # Use explicit padding if provided
        if self.padding is not None:
            padding = self.padding
        else:
            padding = self._detect_padding(data_columns)

        # Determine the canonical casing of the prefix from actual data, if
        # any column starts with it (case-insensitive). This lets
        # Sample._validate_design find the columns when the data is in a
        # different case than the user-specified prefix.
        pattern = re.compile(rf"^{re.escape(self.prefix)}\d+$", re.IGNORECASE)
        resolved_prefix = self.prefix
        for col in data_columns:
            if pattern.match(col):
                resolved_prefix = col[: len(self.prefix)]
                break

        if padding > 0:
            return [f"{resolved_prefix}{i:0{padding}d}" for i in range(1, self.n_reps + 1)]
        else:
            return [f"{resolved_prefix}{i}" for i in range(1, self.n_reps + 1)]

    @property
    def columns(self) -> list[str]:
        """
        Dynamically generate the list of expected column names.

        WARNING: This uses explicit padding if set, otherwise assumes no padding.
        For validation against actual data, use columns_from_data() instead.

        Returns
        -------
        list[str]
            Column names with padding specified in initialization, or no padding
        """
        padding = self.padding if self.padding is not None else 0
        return self._generate_columns(padding)

    def __repr__(self) -> str:
        parts = [f"method={self.method}", f"prefix='{self.prefix}'", f"n_reps={self.n_reps}"]

        # Only show optional params if they deviate from defaults/standard
        if self.df is not None:
            parts.append(f"df={self.df}")

        if self.fay_coef != 0.0 or self.method == _EstimationMethod.BRR:
            parts.append(f"fay={self.fay_coef}")

        if self.padding is not None:
            parts.append(f"padding={self.padding}")

        return f"RepWeights({', '.join(parts)})"

    def __plain_str__(self) -> str:
        """Multi-line plain-text summary for embedding in Design output."""
        try:
            method_name = self.method.value
        except AttributeError:
            method_name = str(self.method)
        # First line is the method name (used as the header by consumers)
        lines = [
            method_name,
            f"Method   : {method_name}",
            f"Prefix   : {self.prefix}",
            f"N reps   : {self.n_reps}",
            f"DF       : {self.df if self.df is not None else 'auto'}",
        ]
        try:
            is_brr = self.method == _EstimationMethod.BRR
        except Exception:
            is_brr = False
        if self.fay_coef != 0.0 or is_brr:
            lines.append(f"Fay coef : {self.fay_coef}")
        return "\n".join(lines)


# =============================================================================
# Normalization helper + public factory for RepWeights
# =============================================================================


def _normalize_rep_method(
    method: Literal["brr", "bootstrap", "jackknife", "sdr"],
) -> _EstimationMethod:
    """
    Normalize user-facing method string to internal EstimationMethod enum.

    Accepts (case-insensitive):
      - "brr"        → EstimationMethod.BRR
      - "bootstrap"  → EstimationMethod.BOOTSTRAP
      - "jackknife", "jk" → EstimationMethod.JACKKNIFE
      - "sdr"        → EstimationMethod.SDR
    """
    _MAP = {
        "brr": _EstimationMethod.BRR,
        "bootstrap": _EstimationMethod.BOOTSTRAP,
        "bs": _EstimationMethod.BOOTSTRAP,
        "jackknife": _EstimationMethod.JACKKNIFE,
        "jk": _EstimationMethod.JACKKNIFE,
        "jkn": _EstimationMethod.JACKKNIFE,
        "sdr": _EstimationMethod.SDR,
    }
    if not isinstance(method, str):
        raise TypeError(
            f"'method' must be a string, got {type(method).__name__}. "
            f"Use 'brr', 'bootstrap', 'jackknife', or 'sdr'."
        )
    result = _MAP.get(method.strip().lower())
    if result is None:
        raise ValueError(
            f"Unknown replication method {method!r}. "
            f"Use 'brr', 'bootstrap', 'jackknife', or 'sdr'."
        )
    return result


def make_rep_weights(
    method: Literal["brr", "bootstrap", "jackknife", "sdr"],
    prefix: str,
    n_reps: int,
    *,
    fay_coef: float = 0.0,
    df: int | None = None,
    padding: int | None = None,
) -> RepWeights:
    """
    Create a RepWeights object using a plain string method name.

    Parameters
    ----------
    method : str
        Replication method: ``'brr'``, ``'bootstrap'``, ``'jackknife'`` (or ``'jk'``),
        or ``'sdr'``.
    prefix : str
        Column prefix for replicate weight columns (e.g. ``'btwt'`` for btwt1, btwt2, ...).
    n_reps : int
        Number of replicate weights (>= 2).
    fay_coef : float, default 0.0
        Fay coefficient for BRR with Fay's method.
    df : int | None, default None
        Degrees of freedom override. None = auto-calculate from data.
    padding : int | None, default None
        Zero-padding width for column names. None = auto-detect.

    Returns
    -------
    RepWeights

    Examples
    --------
    >>> rw = make_rep_weights("jackknife", prefix="jk_", n_reps=80)
    >>> rw = make_rep_weights("brr", prefix="brr_", n_reps=32, fay_coef=0.5)
    """
    return RepWeights(
        method=_normalize_rep_method(method),
        prefix=prefix,
        n_reps=n_reps,
        fay_coef=fay_coef,
        df=df,
        padding=padding,
    )


# =============================================================================
# Resolvers (Internal Helpers)
# =============================================================================

T = TypeVar("T")


@overload
def _pick(current: str, new: str | _MissingType) -> str: ...
@overload
def _pick(
    current: RepWeights | None, new: RepWeights | None | _MissingType
) -> RepWeights | None: ...
@overload
def _pick(current: bool, new: bool | _MissingType) -> bool: ...
@overload
def _pick(current: T, new: T | _MissingType) -> T: ...
def _pick(current: T, new: T | _MissingType) -> T:
    """Overwrite with `new` unless `new` is the _MissingType sentinel."""
    return current if _is_MissingType(new) else cast(T, new)


def _pick_if_none(current: T | None, new: T | _MissingType) -> T | None:
    """
    Only uses `new` when current is None; otherwise keeps current.
    (Useful for "fill defaults" semantics.)
    """
    if current is not None:
        return current
    if _is_MissingType(new):
        return None
    return cast(T, new)


def _norm_spec(
    name: str,
    value: str | Sequence[str] | None,
) -> str | tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value == "":
            raise ValueError(f"'{name}' must not be an empty string when provided")
        return value
    if isinstance(value, (bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"'{name}' must be str | Sequence[str] | None")
    items = list(value)
    if not items:
        raise ValueError(f"'{name}' sequence must not be empty")
    for i, s in enumerate(items):
        if not isinstance(s, str):
            raise TypeError(f"'{name}' items must be str; got {type(s).__name__} at index {i}")
        if s == "":
            raise ValueError(f"'{name}' items must not contain empty strings")
    return tuple(items)


def _norm_pop_size(
    value: str | PopSize | None,
) -> str | PopSize | None:
    """Normalize and validate the pop_size argument."""
    if value is None:
        return None
    if isinstance(value, PopSize):
        if not value.psu or not isinstance(value.psu, str):
            raise ValueError("PopSize.psu must be a non-empty string")
        if not value.ssu or not isinstance(value.ssu, str):
            raise ValueError("PopSize.ssu must be a non-empty string")
        return value
    if isinstance(value, str):
        if not value:
            raise ValueError("'pop_size' must not be an empty string when provided")
        return value
    raise TypeError(f"'pop_size' must be str | PopSize | None, got {type(value).__name__}")


# =============================================================================
# Design Definition
# =============================================================================

_FIELDS: tuple[str, ...] = (
    "row_index",
    "stratum",
    "wgt",
    "prob",
    "hit",
    "mos",
    "psu",
    "ssu",
    "pop_size",
    "wr",
)


class Design:
    row_index: str | None
    stratum: str | tuple[str, ...] | None
    wgt: str | None
    prob: str | None
    hit: str | None
    mos: str | None
    psu: str | tuple[str, ...] | None
    ssu: str | tuple[str, ...] | None
    pop_size: str | PopSize | None
    wr: bool
    rep_wgts: RepWeights | None
    _frozen: bool

    PRINT_WIDTH: int | None = None

    __slots__ = (*_FIELDS, "rep_wgts", "_frozen")

    def __init__(
        self,
        row_index: str | None = None,
        stratum: str | Sequence[str] | None = None,
        wgt: str | None = None,
        prob: str | None = None,
        hit: str | None = None,
        mos: str | None = None,
        psu: str | Sequence[str] | None = None,
        ssu: str | Sequence[str] | None = None,
        pop_size: str | PopSize | None = None,
        wr: bool = False,
        rep_wgts: RepWeights | None = None,
    ) -> None:
        object.__setattr__(self, "_frozen", False)

        norm_stratum = _norm_spec("stratum", stratum)
        norm_psu = _norm_spec("psu", psu)
        norm_ssu = _norm_spec("ssu", ssu)
        norm_pop_size = _norm_pop_size(pop_size)

        object.__setattr__(self, "row_index", row_index)
        object.__setattr__(self, "stratum", norm_stratum)
        object.__setattr__(self, "wgt", wgt)
        object.__setattr__(self, "prob", prob)
        object.__setattr__(self, "hit", hit)
        object.__setattr__(self, "mos", mos)
        object.__setattr__(self, "psu", norm_psu)
        object.__setattr__(self, "ssu", norm_ssu)
        object.__setattr__(self, "pop_size", norm_pop_size)
        object.__setattr__(self, "wr", wr)
        object.__setattr__(self, "rep_wgts", rep_wgts)

        # Validate simple string-or-None fields (pop_size excluded — handled by _norm_pop_size)
        for name in ("row_index", "wgt", "prob", "hit", "mos"):
            val = getattr(self, name)
            if val is not None and not isinstance(val, str):
                raise TypeError(f"{name!r} must be str | None, got {type(val).__name__}")
            if isinstance(val, str) and not val:
                raise ValueError(f"{name!r} must not be an empty string when provided")

        if not isinstance(self.wr, bool):
            raise TypeError(f"'wr' must be bool, got {type(self.wr).__name__}")
        if rep_wgts is not None and not isinstance(rep_wgts, RepWeights):
            raise TypeError("'rep_wgts' must be RepWeights | None")

        object.__setattr__(self, "_frozen", True)

    # -----------------------------
    # Immutability Guards
    # -----------------------------
    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("Design is frozen; use .update(...) to create a modified copy.")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("Design is frozen; attributes cannot be deleted.")
        object.__delattr__(self, name)

    # -----------------------------
    # Properties
    # -----------------------------
    @property
    def method(self) -> _EstimationMethod:
        """Convenience accessor for the estimation method."""
        if self.rep_wgts is None:
            return _EstimationMethod.TAYLOR
        return self.rep_wgts.method

    # -----------------------------
    # Update Methods
    # -----------------------------
    def update(
        self,
        *,
        row_index: str | None | _MissingType = _MISSING,
        stratum: str | Sequence[str] | None | _MissingType = _MISSING,
        wgt: str | None | _MissingType = _MISSING,
        prob: str | None | _MissingType = _MISSING,
        hit: str | None | _MissingType = _MISSING,
        mos: str | None | _MissingType = _MISSING,
        psu: str | Sequence[str] | None | _MissingType = _MISSING,
        ssu: str | Sequence[str] | None | _MissingType = _MISSING,
        pop_size: str | PopSize | None | _MissingType = _MISSING,
        wr: bool | _MissingType = _MISSING,
        rep_wgts: RepWeights | _MissingType | None = _MISSING,
    ) -> Self:
        return self._merge(
            only_if_none=False,
            row_index=row_index,
            stratum=stratum,
            wgt=wgt,
            prob=prob,
            hit=hit,
            mos=mos,
            psu=psu,
            ssu=ssu,
            pop_size=pop_size,
            wr=wr,
            rep_wgts=rep_wgts,
        )

    def fill_missing(
        self,
        *,
        row_index: str | None | _MissingType = _MISSING,
        stratum: str | Sequence[str] | None | _MissingType = _MISSING,
        wgt: str | None | _MissingType = _MISSING,
        prob: str | None | _MissingType = _MISSING,
        hit: str | None | _MissingType = _MISSING,
        mos: str | None | _MissingType = _MISSING,
        psu: str | Sequence[str] | None | _MissingType = _MISSING,
        ssu: str | Sequence[str] | None | _MissingType = _MISSING,
        pop_size: str | PopSize | None | _MissingType = _MISSING,
        wr: bool | _MissingType = _MISSING,
        rep_wgts: RepWeights | Sequence[str] | _MissingType | None = _MISSING,
    ) -> Self:
        # Sequence[str] is handled inside _merge (it becomes _MISSING); cast for ty.
        rep_wgts_arg: RepWeights | _MissingType | None = (
            _MISSING
            if isinstance(rep_wgts, (list, tuple)) and not isinstance(rep_wgts, str)
            else cast(RepWeights | _MissingType | None, rep_wgts)
        )
        return self._merge(
            only_if_none=True,
            row_index=row_index,
            stratum=stratum,
            wgt=wgt,
            prob=prob,
            hit=hit,
            mos=mos,
            psu=psu,
            ssu=ssu,
            pop_size=pop_size,
            wr=wr,
            rep_wgts=rep_wgts_arg,
        )

    def update_rep_weights(
        self,
        *,
        method: Literal["brr", "bootstrap", "jackknife", "sdr"] | None | _MissingType = _MISSING,
        prefix: str | _MissingType = _MISSING,
        n_reps: int | _MissingType = _MISSING,
        fay_coef: float | _MissingType = _MISSING,
        df: int | None | _MissingType = _MISSING,
        padding: int | None | _MissingType = _MISSING,
    ) -> Self:
        """
        Return a new Design with selected RepWeights fields updated.
        Ensures strict validity: if creating weights for the first time,
        mandatory fields (method, prefix, n_reps) must be provided.
        """
        # 1. Quick exit if no arguments provided
        if (
            isinstance(prefix, _MissingType)
            and isinstance(method, _MissingType)
            and isinstance(n_reps, _MissingType)
            and isinstance(fay_coef, _MissingType)
            and isinstance(df, _MissingType)
            and isinstance(padding, _MissingType)
        ):
            return self

        # 2. Explicitly handle method=None to clear weights
        if method is None:
            return self.update(rep_wgts=None)

        # 3. Get current state
        cur = self.rep_wgts

        # 4. Resolve Values

        def resolve_mandatory(arg_val: T | _MissingType, arg_name: str) -> T:
            if not isinstance(arg_val, _MissingType):
                return arg_val
            if cur is not None:
                return getattr(cur, arg_name)
            raise ValueError(
                f"When initializing RepWeights for the first time, '{arg_name}' is mandatory."
            )

        # Resolve Mandatory Fields
        resolved_method = resolve_mandatory(method, "method")
        resolved_prefix = resolve_mandatory(prefix, "prefix")
        resolved_n_reps = resolve_mandatory(n_reps, "n_reps")

        # Resolve Optional Fields
        if isinstance(fay_coef, _MissingType):
            fay_coef = cur.fay_coef if cur else 0.0

        if isinstance(df, _MissingType):
            df = cur.df if cur else None

        if isinstance(padding, _MissingType):
            padding = cur.padding if cur else None

        # 5. Create new RepWeights object
        _resolved_method = (
            _normalize_rep_method(resolved_method)
            if isinstance(resolved_method, str)
            else resolved_method
        )
        updated_rep_wgts = RepWeights(
            method=_resolved_method,
            prefix=resolved_prefix,
            n_reps=resolved_n_reps,
            fay_coef=fay_coef,
            df=df,
            padding=padding,
        )

        return self.update(rep_wgts=updated_rep_wgts)

    # -----------------------------
    # Internal Merge Logic
    # -----------------------------
    def _merge(
        self,
        *,
        only_if_none: bool,
        row_index: str | None | _MissingType = _MISSING,
        stratum: str | Sequence[str] | None | _MissingType = _MISSING,
        wgt: str | None | _MissingType = _MISSING,
        prob: str | None | _MissingType = _MISSING,
        hit: str | None | _MissingType = _MISSING,
        mos: str | None | _MissingType = _MISSING,
        psu: str | Sequence[str] | None | _MissingType = _MISSING,
        ssu: str | Sequence[str] | None | _MissingType = _MISSING,
        pop_size: str | PopSize | None | _MissingType = _MISSING,
        wr: bool | _MissingType = _MISSING,
        rep_wgts: RepWeights | _MissingType | None = _MISSING,
    ) -> Self:
        """
        Internal: merge fields either by overwriting or only filling when current is None.
        """
        # Normalize rep_wgts arg
        rep_arg: RepWeights | _MissingType | None
        if isinstance(rep_wgts, Sequence) and not isinstance(rep_wgts, (str, bytes)):
            rep_arg = _MISSING
        else:
            rep_arg = cast(RepWeights | _MissingType | None, rep_wgts)

        pick = _pick_if_none if only_if_none else _pick

        def is_missing(x: object, /) -> TypeGuard[_MissingType]:
            return x is _MISSING

        def _norm_multi_arg(
            field_name: str, val: str | Sequence[str] | None | _MissingType
        ) -> str | tuple[str, ...] | None | _MissingType:
            if is_missing(val):
                return _MISSING
            if val is None:
                return None
            if isinstance(val, str):
                if val == "":
                    raise ValueError(f"'{field_name}' must not be an empty string when provided")
                return val
            if not isinstance(val, Sequence) or isinstance(val, (bytes, bytearray)):
                raise TypeError(f"'{field_name}' must be a sequence of str")
            if len(val) == 0:
                raise ValueError(f"'{field_name}' sequence must not be empty")
            for x in val:
                if not isinstance(x, str):
                    raise TypeError(f"'{field_name}' items must be str")
                if x == "":
                    raise ValueError(f"'{field_name}' items must not be empty")
            return cast(tuple[str, ...], tuple(val))

        def _norm_pop_size_arg(
            val: str | PopSize | None | _MissingType,
        ) -> str | PopSize | None | _MissingType:
            if is_missing(val):
                return _MISSING
            return _norm_pop_size(val)

        stratum_arg = _norm_multi_arg("stratum", stratum)
        psu_arg = _norm_multi_arg("psu", psu)
        ssu_arg = _norm_multi_arg("ssu", ssu)
        pop_size_arg = _norm_pop_size_arg(pop_size)

        return type(self)(
            row_index=pick(self.row_index, row_index),
            stratum=pick(self.stratum, stratum_arg),
            wgt=pick(self.wgt, wgt),
            prob=pick(self.prob, prob),
            hit=pick(self.hit, hit),
            mos=pick(self.mos, mos),
            psu=pick(self.psu, psu_arg),
            ssu=pick(self.ssu, ssu_arg),
            pop_size=pick(self.pop_size, pop_size_arg),
            wr=_pick(self.wr, wr),
            rep_wgts=pick(self.rep_wgts, rep_arg),
        )

    # -----------------------------
    # Introspection
    # -----------------------------
    def specified_fields(
        self,
        *,
        ignore_cols: Sequence[str] | None = None,
        data_columns: Sequence[str] | None = None,
    ) -> list[str]:
        """
        Return a de-duplicated (order-preserving) list of column names referenced
        by the design (stratum/psu/ssu/etc.), including replicate weight columns.

        Parameters
        ----------
        ignore_cols : Sequence[str], optional
            Column names to ignore
        data_columns : Sequence[str], optional
            Actual data column names (used for auto-detecting padding in rep weights)

        Returns
        -------
        list[str]
            List of all column names referenced by this design
        """
        default_ignores = {"wr"}
        ignore = default_ignores | (set(ignore_cols) if ignore_cols else set())

        out: list[str] = []
        seen: set[str] = set()

        def add(name: str) -> None:
            if name and name not in ignore and name not in seen:
                out.append(name)
                seen.add(name)

        # 1. Add standard fields
        for name in _FIELDS:
            if name in ignore:
                continue

            val = getattr(self, name, None)
            if not val:
                continue

            # Handle multi-column fields
            if name in {"stratum", "psu", "ssu"}:
                if isinstance(val, str):
                    add(val)
                elif isinstance(val, (tuple, list)):
                    for s in val:
                        add(s)
                continue

            # Handle PopSize
            if name == "pop_size":
                if isinstance(val, PopSize):
                    add(val.psu)
                    add(val.ssu)
                elif isinstance(val, str):
                    add(val)
                continue

            # Handle standard string fields
            if isinstance(val, str):
                add(val)

        # 2. Add Replicate Weight columns with auto-detection
        if self.rep_wgts:
            if data_columns is not None:
                # Use auto-detection from actual data
                rep_cols = self.rep_wgts.columns_from_data(data_columns)
            else:
                # Fall back to default columns (explicit padding or no padding)
                rep_cols = self.rep_wgts.columns

            for col in rep_cols:
                add(col)

        return out

    # -----------------------------
    # Printing & Rendering
    # -----------------------------
    @staticmethod
    def _pad_and_surround(text: str, *, indent: int = 2, surround: bool = False) -> str:
        if text is None:
            return ""
        text = str(text).rstrip("\n")
        if indent > 0:
            pad = " " * indent
            text = "\n".join(pad + line if line else pad for line in text.splitlines())
        return f"\n{text}\n" if surround else text

    @staticmethod
    def _fmt_tuple_names(x) -> str:
        if x is None:
            return "None"
        if isinstance(x, (tuple, list)):
            inner = ", ".join(str(v) for v in x)
            if len(x) == 1:
                inner += ","
            return f"({inner})"
        return str(x)

    @staticmethod
    def _fmt_pop_size(x) -> str:
        if x is None:
            return "None"
        if isinstance(x, PopSize):
            return f"PopSize(psu='{x.psu}', ssu='{x.ssu}')"
        return str(x)

    def _repweights_summary(self) -> str:
        if self.rep_wgts is None:
            return "None"
        fn = getattr(self.rep_wgts, "__plain_str__", None)
        return fn() if callable(fn) else repr(self.rep_wgts)

    def __rich_console__(self, console, options):
        from rich.table import Table as RTable
        from rich.text import Text

        t = RTable(
            show_header=False,
            box=None,
            show_edge=False,
            show_lines=False,
            pad_edge=False,
            expand=False,
        )
        t.add_column("Field", justify="left", no_wrap=True)
        t.add_column("Value", justify="left", no_wrap=False, overflow="fold")

        rows: list[tuple[str, str]] = [
            ("Row index", str(self.row_index)),
            ("Stratum", self._fmt_tuple_names(self.stratum)),
            ("PSU", self._fmt_tuple_names(self.psu)),
            ("SSU", self._fmt_tuple_names(self.ssu)),
            ("Weight", str(self.wgt)),
            ("With replacement", str(bool(self.wr))),
            ("Prob", str(self.prob)),
            ("Hit", str(self.hit)),
            ("MOS", str(self.mos)),
            ("Population size", self._fmt_pop_size(self.pop_size)),
        ]
        for k, v in rows:
            t.add_row(k, v)

        # Rep weights — sub-fields as separate rows, not bold
        if self.rep_wgts is None:
            t.add_row("Replicate weights", "None")
        else:
            sub_lines = self._repweights_summary().splitlines()
            t.add_row("Replicate weights", "")
            for sub_line in sub_lines[1:]:
                t.add_row(Text(f"    {sub_line}", style="not bold"), "")

        yield make_panel([t], title="Design", obj=self, kind="estimate")

    def __plain_str__(self) -> str:
        """Plain-text fallback when rich is not installed."""
        lines: list[str] = [
            "Design",
            f"  Row index        : {self.row_index}",
            f"  Stratum          : {self._fmt_tuple_names(self.stratum)}",
            f"  PSU              : {self._fmt_tuple_names(self.psu)}",
            f"  SSU              : {self._fmt_tuple_names(self.ssu)}",
            f"  Weight           : {self.wgt}",
            f"  With replacement : {bool(self.wr)}",
            f"  Prob             : {self.prob}",
            f"  Hit              : {self.hit}",
            f"  MOS              : {self.mos}",
            f"  Population size  : {self._fmt_pop_size(self.pop_size)}",
        ]
        if self.rep_wgts is not None:
            sub_lines = self._repweights_summary().splitlines()
            lines.append("  Replicate weights")
            for sub_line in sub_lines[1:]:
                lines.append(f"      {sub_line}")
        else:
            lines.append("  Replicate weights : None")
        return "\n".join(lines)

    def __str__(self) -> str:
        result = render_rich_to_str(self, width=resolve_width(self))
        return self._pad_and_surround(result, indent=2, surround=False)

    @classmethod
    def set_default_print_width(cls, width: int | None) -> None:
        if width is None:
            cls.PRINT_WIDTH = None
            return
        try:
            w = int(width)
        except Exception as ex:
            raise TypeError(f"print width must be int or None; got {width!r}") from ex
        if w <= 20:
            raise ValueError("print width must be > 20 characters.")
        cls.PRINT_WIDTH = w

    def show(self, *, use_rich: bool = True) -> None:
        from svy.ui.printing import rich_available

        if use_rich and rich_available():
            import sys

            from rich.console import Console

            Console(
                file=sys.stdout,
                force_terminal=True,
                emoji=False,
                width=resolve_width(self),
                soft_wrap=True,
            ).print(self)
            return
        print(self.__plain_str__())

    def __repr__(self) -> str:
        parts: list[str] = []
        if self.row_index is not None:
            parts.append(f"row_index={self.row_index!r}")

        def add_nonempty(name: str, value) -> None:
            if value is None:
                return
            if isinstance(value, (tuple, list)) and not value:
                return
            parts.append(f"{name}={value!r}")

        add_nonempty("stratum", self.stratum)
        add_nonempty("psu", self.psu)
        add_nonempty("ssu", self.ssu)
        add_nonempty("wgt", self.wgt)
        add_nonempty("prob", self.prob)
        add_nonempty("hit", self.hit)
        add_nonempty("mos", self.mos)
        add_nonempty("pop_size", self.pop_size)
        if self.wr:
            parts.append("wr=True")

        if self.rep_wgts:
            rw = self.rep_wgts
            method_name = getattr(rw.method, "name", str(rw.method))
            parts.append(
                f"rep_wgts={method_name}(n_reps={rw.n_reps}, prefix='{rw.prefix}', df={rw.df})"
            )
        else:
            parts.append("rep_wgts=None")

        return f"Design({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Design):
            return False
        return all(getattr(self, f) == getattr(other, f) for f in _FIELDS) and (
            self.rep_wgts == other.rep_wgts
        )

    def __hash__(self) -> int:
        return hash((tuple(getattr(self, f) for f in _FIELDS), self.rep_wgts))
