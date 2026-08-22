# src/svy/core/design.py
from __future__ import annotations

import logging

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

from svy.core.repwgts import (
    BrrWgts,
    RepWeights,
    RepWgts,
    _RepWgtsBase,
    resolve_rep_variant,
)
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
    ssu: str | None = None


# =============================================================================
# Replicate Weights (Strict Configuration)
# =============================================================================


def make_rep_weights(
    method: Literal["brr", "bootstrap", "jackknife", "sdr"],
    prefix: str,
    n_reps: int,
    *,
    fay_coef: float = 0.0,
    df: int | None = None,
    padding: int | None = None,
) -> RepWgts:
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
        method=method,
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
def _pick(current: RepWgts | None, new: RepWgts | None | _MissingType) -> RepWgts | None: ...
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
        # ssu is optional: PopSize(psu=...) alone specifies a PSU-only FPC
        # (the type signature always allowed it; validation rejected it).
        if value.ssu is not None and (not value.ssu or not isinstance(value.ssu, str)):
            raise ValueError("PopSize.ssu must be a non-empty string or None")
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
    rep_wgts: RepWgts | None
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
        rep_wgts: RepWgts | None = None,
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
        # Every variant inherits the base, so this covers the whole union.
        if rep_wgts is not None and not isinstance(rep_wgts, _RepWgtsBase):
            raise TypeError("'rep_wgts' must be RepWgts | None")

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
    def method(self) -> str:
        """Convenience accessor for the estimation method, as a display label.

        ``"Taylor"`` when the design carries no replicate weights, otherwise
        the variant's own label. For display and reporting only.
        """
        if self.rep_wgts is None:
            return "Taylor"
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
        rep_wgts: RepWgts | _MissingType | None = _MISSING,
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
        rep_wgts: RepWgts | Sequence[str] | _MissingType | None = _MISSING,
    ) -> Self:
        # Sequence[str] is handled inside _merge (it becomes _MISSING); cast for ty.
        rep_wgts_arg: RepWgts | _MissingType | None = (
            _MISSING
            if isinstance(rep_wgts, (list, tuple)) and not isinstance(rep_wgts, str)
            else cast(RepWgts | _MissingType | None, rep_wgts)
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
        rscales: tuple[float, ...] | None | _MissingType = _MISSING,
        kind: str | None | _MissingType = _MISSING,
        paired: bool | _MissingType = _MISSING,
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
            and isinstance(rscales, _MissingType)
            and isinstance(kind, _MissingType)
            and isinstance(paired, _MissingType)
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

        if isinstance(rscales, _MissingType):
            rscales = cur.rscales if cur else None

        # 5. Create the new variant.
        _variant = resolve_rep_variant(resolved_method)

        # Variant-specific parameters carry over only within the same method:
        # a bootstrap kind means nothing on a jackknife design.
        _same = cur is not None and type(cur) is _variant
        if isinstance(kind, _MissingType):
            kind = getattr(cur, "kind", None) if _same else None
        if isinstance(paired, _MissingType):
            paired = getattr(cur, "paired", False) if _same else False

        updated_rep_wgts = RepWeights(
            method=resolved_method,
            prefix=resolved_prefix,
            n_reps=resolved_n_reps,
            fay_coef=fay_coef if _variant is BrrWgts else 0.0,
            df=df,
            padding=padding,
            rscales=rscales,
            kind=kind,
            paired=paired,
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
        rep_wgts: RepWgts | _MissingType | None = _MISSING,
    ) -> Self:
        """
        Internal: merge fields either by overwriting or only filling when current is None.
        """
        # Normalize rep_wgts arg
        rep_arg: RepWgts | _MissingType | None
        if isinstance(rep_wgts, Sequence) and not isinstance(rep_wgts, (str, bytes)):
            rep_arg = _MISSING
        else:
            rep_arg = cast(RepWgts | _MissingType | None, rep_wgts)

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
