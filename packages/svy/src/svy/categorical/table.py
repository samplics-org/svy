# src/svy/categorical/table.py
from __future__ import annotations

import logging

from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Iterable,
    Iterator,
    Self,
    Sequence,
    TypeGuard,
    TypeVar,
    cast,
)

import msgspec
import polars as pl

from msgspec import structs as ms_structs

from svy.core.containers import ChiSquare, FDist
from svy.core.enumerations import TableType
from svy.core.types import Category, Number
from svy.errors import MethodError
from svy.estimation import ParamEst

# Import central UI helpers (consistent with estimate.py and ttest.py)
from svy.ui.printing import (
    level_index_key,
    level_sort_key,
    make_panel,
    render_plain_table,
    render_rich_to_str,
    resolve_width,
)
from svy.utils.formats import _fmt_fixed


if TYPE_CHECKING:
    from svy.metadata import MetadataStore


log = logging.getLogger(__name__)


# =============================================================================
# Public structs: table-level stats & cell estimates
# =============================================================================


class TableStats(msgspec.Struct):
    """Rao-Scott chi-square test statistics for two-way tables.

    chisq : First-order Rao-Scott correction. R: statistic="Chisq".
    f     : Second-order Rao-Scott correction. R: statistic="F".
    """

    chisq: ChiSquare
    f: FDist | None


class CellEst(msgspec.Struct):
    rowvar: str
    colvar: str
    est: Number
    se: Number
    cv: Number
    lci: Number
    uci: Number

    @classmethod
    def from_param(cls, param_est: ParamEst) -> Self:
        if isinstance(param_est.y_level, str):
            parts = param_est.y_level.split("__by__")
            if len(parts) == 2:
                rowvar, colvar = parts
            elif len(parts) == 1:
                rowvar, colvar = parts[0], ""
            else:
                rowvar, colvar = "", ""
        else:
            rowvar, colvar = "", ""

        return cls(
            rowvar=rowvar,
            colvar=colvar,
            est=param_est.est,
            se=param_est.se,
            cv=param_est.cv,
            lci=param_est.lci,
            uci=param_est.uci,
        )

    def to_dict(self) -> dict[str, Category | None]:
        return {
            "rowvar": self.rowvar,
            "colvar": self.colvar,
            "est": self.est,
            "se": self.se,
            "cv": self.cv,
            "lci": self.lci,
            "uci": self.uci,
        }


# =============================================================================
# Sentinel & tiny helpers
# =============================================================================


class _Missing:
    __slots__ = ()


_MISSING: Final = _Missing()
T = TypeVar("T")


def _is_missing(x: object) -> TypeGuard[_Missing]:
    return isinstance(x, _Missing)


def _pick(current: T, new: T | _Missing) -> T:
    return current if _is_missing(new) else cast(T, new)


def _pick_if_none(current: T | None, new: T | _Missing) -> T | None:
    if current is not None:
        return current
    if _is_missing(new):
        return None
    return cast(T, new)


def _enum_label(e: object) -> str:
    if isinstance(e, Enum):
        v = e.value
        if isinstance(v, str):
            return v
        return e.name.replace("_", " ").title()
    return str(e)


def _norm_label(x: object) -> str:
    # normalize labels to stripped strings for consistent matching/printing
    return str(x).strip()


# =============================================================================
# Table (immutable)
# =============================================================================

# numeric columns we format in the print views
_DECIMAL_KEYS: tuple[str, ...] = ("est", "se", "cv", "lci", "uci")

_TBL_SLOTS: tuple[str, ...] = (
    "type",
    "rowvar",
    "colvar",
    "estimates",
    "stats",
    "alpha",
    "rowvals",
    "colvals",
    "_frozen",
    # presentation-only fields (mutable through properties)
    "_decimals",
    "_print_width",
    "_use_labels",
    # metadata reference for label resolution
    "_metadata",
)


class Table:
    __slots__ = _TBL_SLOTS

    # class-level width override (honored by resolve_width)
    PRINT_WIDTH: int | None = None
    # class-level decimals: None → module defaults; int → apply to all; dict → per column
    DECIMALS: int | dict[str, int] | None = None
    # class-level default for using labels
    USE_LABELS: bool = True

    type: TableType
    rowvar: str
    colvar: str | None
    estimates: Sequence[CellEst] | None
    stats: TableStats | None
    alpha: float
    rowvals: Sequence[Category] | None
    colvals: Sequence[Category] | None
    _frozen: bool
    _decimals: int | dict[str, int] | None
    _print_width: int | None
    _use_labels: bool | None
    _metadata: "MetadataStore | None"

    def __init__(
        self,
        *,
        type: TableType,
        rowvar: str,
        colvar: str | None = None,
        estimates: Sequence[CellEst] | None = None,
        stats: TableStats | None = None,
        rowvals: Sequence[Category] | None = None,
        colvals: Sequence[Category] | None = None,
        alpha: float = 0.05,
        metadata: "MetadataStore | None" = None,
    ) -> None:
        if not (0.0 < float(alpha) < 1.0):
            raise MethodError.invalid_range(
                where="Table.__init__",
                param="alpha",
                got=alpha,
                min_=0.0,
                max_=1.0,
                hint="Alpha is a probability; try something like 0.05.",
            )

        if type is TableType.ONE_WAY and colvar is not None:
            raise MethodError.not_applicable(
                where="Table.__init__",
                method="TableType.ONE_WAY",
                reason="ONE_WAY tables must have colvar=None",
                param="colvar",
            )
        if type is TableType.TWO_WAY and not colvar:
            raise MethodError.not_applicable(
                where="Table.__init__",
                method="TableType.TWO_WAY",
                reason="TWO_WAY tables must have a non-empty colvar",
                param="colvar",
                hint="Pass the column name used for columns in the cross-tab.",
            )

        object.__setattr__(self, "_frozen", False)
        object.__setattr__(self, "type", type)
        object.__setattr__(self, "rowvar", rowvar)
        object.__setattr__(self, "colvar", colvar)
        object.__setattr__(self, "estimates", list(estimates) if estimates is not None else [])
        object.__setattr__(self, "stats", stats)
        object.__setattr__(self, "alpha", float(alpha))
        object.__setattr__(self, "rowvals", list(rowvals) if rowvals is not None else None)
        object.__setattr__(self, "colvals", list(colvals) if colvals is not None else None)
        # presentation defaults (mutable through properties even when frozen)
        object.__setattr__(self, "_decimals", None)
        object.__setattr__(self, "_print_width", None)
        object.__setattr__(self, "_use_labels", None)
        # metadata for label resolution
        object.__setattr__(self, "_metadata", metadata)
        object.__setattr__(self, "_frozen", True)

    # =========================================================================
    # Label resolution helpers
    # =========================================================================

    def _resolve_use_labels(self) -> bool:
        """Resolve whether to use labels: instance -> class -> True."""
        if self._use_labels is not None:
            return self._use_labels
        return getattr(type(self), "USE_LABELS", True)

    def _get_var_label(self, var: str) -> str:
        """Get variable label or fall back to variable name."""
        if not self._resolve_use_labels() or self._metadata is None:
            return var
        resolved = self._metadata.resolve_labels(var)
        return resolved.var_label if resolved.has_var_label else var

    def _get_value_label(self, var: str, value: Category) -> str:
        """Get value label or fall back to string representation."""
        if not self._resolve_use_labels() or self._metadata is None:
            return str(value)
        resolved = self._metadata.resolve_labels(var)
        # Try the value as-is first
        label = resolved.display(value)
        # If we got back the string representation, try converting to int
        # (handles case where CellEst stores "1" but labels are keyed by 1)
        if label == str(value) and isinstance(value, str):
            try:
                int_value = int(value)
                label = resolved.display(int_value)
            except (ValueError, TypeError):
                pass
        return label

    def _get_row_display(self, value: Category) -> str:
        """Get display string for a row value."""
        return self._get_value_label(self.rowvar, value)

    def _get_col_display(self, value: Category) -> str:
        """Get display string for a column value."""
        if self.colvar is None:
            return str(value)
        return self._get_value_label(self.colvar, value)

    # =========================================================================
    # Label configuration
    # =========================================================================

    @property
    def use_labels(self) -> bool:
        """Whether to display labels (True) or raw codes (False)."""
        return self._resolve_use_labels()

    @use_labels.setter
    def use_labels(self, value: bool | None) -> None:
        """Set per-instance label usage preference."""
        object.__setattr__(self, "_use_labels", value)

    @classmethod
    def set_default_use_labels(cls, use: bool) -> None:
        """Set the default label usage for all Table instances."""
        cls.USE_LABELS = bool(use)

    # ----- Class Methods ---------------------------------

    @classmethod
    def one_way(
        cls,
        *,
        rowvar: str,
        estimates: list[CellEst] | None = None,
        stats: TableStats | None = None,
        rowvals: list[Category] | None = None,
        alpha: float = 0.05,
        metadata: "MetadataStore | None" = None,
    ) -> "Table":
        return cls(
            type=TableType.ONE_WAY,
            rowvar=rowvar,
            colvar=None,
            estimates=estimates,
            stats=stats,
            rowvals=rowvals,
            colvals=None,
            alpha=alpha,
            metadata=metadata,
        )

    @classmethod
    def two_way(
        cls,
        *,
        rowvar: str,
        colvar: str,
        estimates: list[CellEst] | None = None,
        stats: TableStats | None = None,
        rowvals: list[Category] | None = None,
        colvals: list[Category] | None = None,
        alpha: float = 0.05,
        metadata: "MetadataStore | None" = None,
    ) -> "Table":
        return cls(
            type=TableType.TWO_WAY,
            rowvar=rowvar,
            colvar=colvar,
            estimates=estimates,
            stats=stats,
            rowvals=rowvals,
            colvals=colvals,
            alpha=alpha,
            metadata=metadata,
        )

    # ----- Width Configuration (consistent with estimate.py and ttest.py) ----

    @classmethod
    def set_default_print_width(cls, width: int | None) -> None:
        if width is None:
            cls.PRINT_WIDTH = None
            return
        try:
            w = int(width)
        except Exception as ex:
            raise TypeError(f"class print width must be int or None; got {width!r}") from ex
        if w <= 20:
            raise ValueError("class print width must be > 20 characters.")
        cls.PRINT_WIDTH = w

    # ----- presentation properties & helpers ---------------------------------

    @staticmethod
    def _normalize_decimals(conf: int | dict[str, int] | None) -> dict[str, int] | None:
        if conf is None:
            return None
        if isinstance(conf, int):
            return {k: conf for k in _DECIMAL_KEYS}
        return {k: int(v) for k, v in conf.items() if k in _DECIMAL_KEYS}

    def _decimals_map(self, fallback_each: int) -> dict[str, int]:
        inst = self._normalize_decimals(self._decimals)
        if inst is not None:
            return {k: inst.get(k, fallback_each) for k in _DECIMAL_KEYS}
        cls = self._normalize_decimals(getattr(type(self), "DECIMALS", None))
        if cls is not None:
            return {k: cls.get(k, fallback_each) for k in _DECIMAL_KEYS}
        return {k: fallback_each for k in _DECIMAL_KEYS}

    @property
    def decimals(self) -> int | dict[str, int] | None:
        return self._decimals

    @decimals.setter
    def decimals(self, value: int | dict[str, int] | None) -> None:
        if value is None or isinstance(value, int) or isinstance(value, dict):
            object.__setattr__(self, "_decimals", value)
        else:
            raise TypeError("decimals must be int | dict[str,int] | None")

    def set_decimals(self, every: int | None = None, /, **overrides: int) -> Self:
        if every is None and not overrides:
            object.__setattr__(self, "_decimals", None)
            return self
        base = {k: int(every) for k in _DECIMAL_KEYS} if every is not None else {}
        for k, v in overrides.items():
            if k not in _DECIMAL_KEYS:
                raise ValueError(f"Unknown key {k!r}; allowed: {_DECIMAL_KEYS}")
            base[k] = int(v)
        object.__setattr__(self, "_decimals", base)
        return self

    @property
    def print_width(self) -> int | None:
        return self._print_width

    @print_width.setter
    def print_width(self, value: int | None) -> None:
        if value is None:
            object.__setattr__(self, "_print_width", None)
            return
        iv = int(value)
        if iv <= 20:
            raise ValueError("print_width must be > 20 or None")
        object.__setattr__(self, "_print_width", iv)

    def style(
        self,
        *,
        decimals: int | dict[str, int] | None = None,
        print_width: int | None = None,
        use_labels: bool | None = None,
    ) -> Self:
        if decimals is not None:
            self.decimals = decimals
        if print_width is not None:
            self.print_width = print_width
        if use_labels is not None:
            self.use_labels = use_labels
        return self

    # -------------------------------------------------------------------------

    @property
    def is_crosstab(self) -> bool:
        return self.type is TableType.TWO_WAY

    def __setattr__(self, name: str, value: object) -> None:
        if name in (
            "_decimals",
            "_print_width",
            "_use_labels",
            "decimals",
            "print_width",
            "use_labels",
        ):
            object.__setattr__(self, name, value)
            return
        if getattr(self, "_frozen", False):
            raise AttributeError("Table is frozen; use .update(...) to create a modified copy.")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("Table is frozen; attributes cannot be deleted.")
        object.__delattr__(self, name)

    # ---- Rich rendering ----

    def _build_table(self):
        from rich import box
        from rich.table import Table as RTable

        headers = _headers_for_display(self)
        if not self.is_crosstab:
            headers = [h for h in headers if h != "Col"]

        dec_map = self._decimals_map(fallback_each=4)

        t = RTable(
            show_header=True,
            header_style="bold",
            box=box.SIMPLE_HEAVY,
            show_edge=True,
            show_lines=False,
            pad_edge=False,
            highlight=False,
            expand=False,
        )

        for h in headers:
            justify = "left" if h in ("Row", "Col") else "right"
            no_wrap = h in ("Row", "Col")
            overflow = "ellipsis" if no_wrap else "fold"
            t.add_column(h, justify=justify, no_wrap=no_wrap, overflow=overflow)

        for row in _rows_for_display(self, dec_map=dec_map):
            t.add_row(*row)

        return t

    def __rich_console__(self, console: Any, options: Any) -> Iterable[Any]:
        t = self._build_table()

        row_label = self._get_var_label(self.rowvar)
        if self.is_crosstab:
            col_label = self._get_var_label(self.colvar or "")
            title = f"Table: [bold]{row_label}[/bold] × [bold]{col_label}[/bold]"
        else:
            title = f"Table: [bold]{row_label}[/bold]"

        yield make_panel([t], title=title, obj=self, kind="estimate")

    def __plain_str__(self) -> str:
        headers = _headers_for_display(self)
        rows = list(_rows_for_display(self, dec_map=self._decimals_map(fallback_each=5)))
        row_label = self._get_var_label(self.rowvar)
        if self.is_crosstab:
            col_label = self._get_var_label(self.colvar or "")
            title = f"Table: {row_label} × {col_label}"
        else:
            title = f"Table: {row_label}"
        body = render_plain_table(headers, rows)
        return f"{title}\n\n{body}"

    def __str__(self) -> str:
        return render_rich_to_str(self, width=resolve_width(self))

    def __repr__(self) -> str:
        n = len(self.estimates or [])
        r = (
            len(self.rowvals)
            if self.rowvals is not None
            else len({c.rowvar for c in (self.estimates or [])})
        )
        c = (
            0
            if self.type is TableType.ONE_WAY
            else (
                len(self.colvals)
                if self.colvals is not None
                else len({c.colvar for c in (self.estimates or [])})
            )
        )
        parts = [
            f"type={self.type.name}",
            f"rowvar={self.rowvar!r}",
        ]
        if self.is_crosstab:
            parts.append(f"colvar={self.colvar!r}")
            parts.append(f"levels={r}x{c}")
        else:
            parts.append(f"levels={r}")
        parts.append(f"n={n}")
        parts.append(f"alpha={self.alpha:g}")
        return f"Table({', '.join(parts)})"

    def _merge(
        self,
        *,
        only_if_none: bool,
        rowvar: str | _Missing = _MISSING,
        colvar: str | None | _Missing = _MISSING,
        estimates: Iterable[CellEst] | None | _Missing = _MISSING,
        stats: TableStats | None | _Missing = _MISSING,
        rowvals: Sequence[Category] | None | _Missing = _MISSING,
        colvals: Sequence[Category] | None | _Missing = _MISSING,
        alpha: float | _Missing = _MISSING,
        metadata: "MetadataStore | None | _Missing" = _MISSING,
    ) -> Self:
        pick = _pick_if_none if only_if_none else _pick

        if rowvar is _MISSING:
            rowvar_val: str = self.rowvar
        else:
            rowvar_val = cast(str, rowvar)

        if estimates is _MISSING:
            est_arg: list[CellEst] | None | _Missing = _MISSING
        elif estimates is None:
            est_arg = None
        else:
            est_arg = list(cast(Iterable[CellEst], estimates))

        if rowvals is _MISSING:
            rowvals_arg: list[Category] | None | _Missing = _MISSING
        elif rowvals is None:
            rowvals_arg = None
        else:
            rowvals_arg = list(cast(Sequence[Category], rowvals))

        if colvals is _MISSING:
            colvals_arg: list[Category] | None | _Missing = _MISSING
        elif colvals is None:
            colvals_arg = None
        else:
            colvals_arg = list(cast(Sequence[Category], colvals))

        return type(self)(
            type=self.type,
            rowvar=rowvar_val,
            colvar=pick(self.colvar, colvar),
            estimates=pick(self.estimates, est_arg),
            stats=pick(self.stats, stats),
            rowvals=pick(self.rowvals, rowvals_arg),
            colvals=pick(self.colvals, colvals_arg),
            alpha=_pick(self.alpha, alpha),
            metadata=pick(self._metadata, metadata),
        )

    def update(self, **kw) -> Self:
        return self._merge(only_if_none=False, **kw)

    def fill_missing(self, **kw) -> Self:
        return self._merge(only_if_none=True, **kw)

    def add_estimate(self, cell: CellEst) -> Self:
        ests = list(self.estimates or [])
        ests.append(cell)
        return self.update(estimates=ests)

    def extend_estimates(self, cells: Sequence[CellEst]) -> Self:
        ests = list(self.estimates or [])
        ests.extend(cells)
        return self.update(estimates=ests)

    def add_param(self, param_est: ParamEst) -> Self:
        return self.add_estimate(CellEst.from_param(param_est))

    def extend_params(self, params: Sequence[ParamEst]) -> Self:
        return self.extend_estimates([CellEst.from_param(p) for p in params])

    def set_stats(self, stats: TableStats | None) -> Self:
        return self.update(stats=stats)

    def set_levels(
        self,
        *,
        rowvals: Sequence[Category] | None = None,
        colvals: Sequence[Category] | None = None,
    ) -> Self:
        return self.update(
            rowvals=list(rowvals) if rowvals is not None else _MISSING,
            colvals=list(colvals) if colvals is not None else _MISSING,
        )

    # ------------------------------------------------------------------ #
    # Exports
    # ------------------------------------------------------------------ #

    def to_records(self, *, include_meta: bool = True) -> list[dict[str, Category | None]]:
        rows: list[dict[str, Category | None]] = []
        for c in self.estimates or []:
            rec = c.to_dict()
            if include_meta:
                rec.update({"table_type": _enum_label(self.type), "alpha": self.alpha})
            rows.append(rec)
        return rows

    def to_polars(self, *, tidy: bool = True) -> Any:
        df = pl.DataFrame(self.to_records(include_meta=True))
        drop_cols = [c for c in ("cv", "deff") if c in df.columns]
        if drop_cols:
            df = df.drop(drop_cols)
        if not tidy:
            return df
        rename = {"rowvar": self.rowvar}
        if self.colvar is not None and "colvar" in df.columns:
            rename["colvar"] = self.colvar
        elif "colvar" in df.columns:
            df = df.drop("colvar")
        return df.rename(rename)

    def to_dict(self) -> dict[str, object]:
        stats_payload: dict[str, object] | None
        if self.stats is not None:
            stats_payload = {
                "chisq": ms_structs.asdict(self.stats.chisq),
                "f": (ms_structs.asdict(self.stats.f) if self.stats.f is not None else None),
            }
        else:
            stats_payload = None

        return {
            "type": str(self.type.name if hasattr(self.type, "name") else self.type),
            "rowvar": self.rowvar,
            "colvar": self.colvar,
            "alpha": self.alpha,
            "rowvals": list(self.rowvals) if self.rowvals is not None else None,
            "colvals": list(self.colvals) if self.colvals is not None else None,
            "estimates": [c.to_dict() for c in (self.estimates or [])],
            "stats": stats_payload,
        }

    # ------------------------------------------------------------------ #
    # Show and Crosstab view
    # ------------------------------------------------------------------ #

    def show(self, *, decimals: int = 5, use_rich: bool = True) -> None:
        show_table(self, dec=decimals, use_rich=use_rich)

    def crosstab(
        self,
        stats: str | tuple[str, ...] = "est",
        *,
        by: str | None = None,
        precision: int = 3,
        fill_missing: str | float | None = None,
        cellfmt: str | None = "auto",
        sort_rows: bool = True,
        sort_cols: bool = True,
        use_labels: bool | None = None,
    ) -> pl.DataFrame:
        stats = (stats,) if isinstance(stats, str) else tuple(stats)

        recs = [c.to_dict() for c in (self.estimates or [])]
        if not recs:
            return pl.DataFrame({self.rowvar: pl.Series([], dtype=pl.Utf8)})

        df = pl.DataFrame(recs)

        resolve_labels = use_labels if use_labels is not None else self._resolve_use_labels()

        if resolve_labels and self._metadata is not None:
            row_resolved = self._metadata.resolve_labels(self.rowvar)
            if row_resolved.has_value_labels:
                _row_label_map = {str(k): str(v) for k, v in row_resolved.value_labels.items()}
                df = df.with_columns(
                    pl.col("rowvar").replace_strict(_row_label_map, default=pl.col("rowvar"))
                )

            if self.colvar is not None:
                col_resolved = self._metadata.resolve_labels(self.colvar)
                if col_resolved.has_value_labels:
                    _col_label_map = {str(k): str(v) for k, v in col_resolved.value_labels.items()}
                    df = df.with_columns(
                        pl.col("colvar").replace_strict(_col_label_map, default=pl.col("colvar"))
                    )

        row_levels = list(
            dict.fromkeys(self.rowvals or df.get_column("rowvar").unique().to_list())
        )
        if resolve_labels and self._metadata is not None:
            row_resolved = self._metadata.resolve_labels(self.rowvar)
            if row_resolved.has_value_labels:
                row_levels = [row_resolved.display(v) for v in row_levels]
            else:
                row_levels = [str(v) for v in row_levels]
        else:
            row_levels = [str(v) for v in row_levels]

        if sort_rows and (self.rowvals is None):
            row_levels = sorted([_norm_label(x) for x in row_levels], key=level_sort_key)

        # ONE-WAY
        if self.colvar is None or self.type is TableType.ONE_WAY:
            if len(stats) == 1:
                stat = stats[0]
                df = _ensure_numeric(df, stat)
                base = (
                    df.select(["rowvar", stat])
                    .group_by("rowvar")
                    .agg(pl.col(stat).last().alias(stat))
                )
                out = base.rename({"rowvar": self.rowvar})
                out = _reindex_rows_only(out, row_levels=row_levels, row_name=self.rowvar)
                if fill_missing is not None:
                    out = out.fill_null(fill_missing)
                return out

            dfm = df.select(["rowvar"] + list(stats))
            for s in stats:
                if s not in dfm.columns:
                    dfm = dfm.with_columns(pl.lit(None).alias(s))

            dfm = (
                dfm.group_by("rowvar")
                .agg([pl.col(s).last().alias(s) for s in stats])
                .sort("rowvar")
            )

            fmt_cols = [
                pl.col(s)
                .map_elements(
                    lambda x: _fmt_fixed(x, dec=precision) if x is not None else "NA",
                    return_dtype=pl.Utf8,
                )
                .alias(f"{s}__str")
                for s in stats
            ]
            dfm = dfm.with_columns(fmt_cols)
            dfm = dfm.with_columns(
                _compose_cell(dfm, stats=stats, mode=cellfmt or "auto").alias("value")
            )

            out = dfm.select(["rowvar", "value"]).rename({"rowvar": self.rowvar})
            out = _reindex_rows_only(out, row_levels=row_levels, row_name=self.rowvar)
            if fill_missing is not None:
                out = out.with_columns(pl.col("value").fill_null(fill_missing))
            return out

        # TWO-WAY
        col_levels_raw = list(
            dict.fromkeys(self.colvals or df.get_column("colvar").unique().to_list())
        )
        if resolve_labels and self._metadata is not None:
            col_resolved = self._metadata.resolve_labels(self.colvar)
            if col_resolved.has_value_labels:
                col_levels = [col_resolved.display(v) for v in col_levels_raw]
            else:
                col_levels = [str(v) for v in col_levels_raw]
        else:
            col_levels = [str(v) for v in col_levels_raw]

        if sort_cols and (self.colvals is None):
            col_levels = sorted([_norm_label(x) for x in col_levels], key=level_sort_key)

        if len(stats) == 1:
            stat = stats[0]
            df = _ensure_numeric(df, stat)
            base = df.select(["rowvar", "colvar", stat]).rename({stat: "value"})
            wide = base.pivot(
                values="value", index="rowvar", on="colvar", aggregate_function=None
            ).rename({"rowvar": self.rowvar})
            wide = _reindex_polars(
                wide, row_levels=row_levels, col_levels=col_levels, row_name=self.rowvar
            )
            if fill_missing is not None:
                non_row_cols = [c for c in wide.columns if c != self.rowvar]
                if non_row_cols:
                    wide = wide.with_columns([pl.col(c).cast(pl.Float64) for c in non_row_cols])
                wide = wide.fill_null(fill_missing)
            return wide

        dfm = df.select(["rowvar", "colvar"] + list(stats))
        for s in stats:
            if s not in dfm.columns:
                dfm = dfm.with_columns(pl.lit(None).alias(s))

        dfm = (
            dfm.group_by(["rowvar", "colvar"])
            .agg([pl.col(s).last().alias(s) for s in stats])
            .sort(["rowvar", "colvar"])
        )

        fmt_cols = [
            pl.col(s)
            .map_elements(
                lambda x: _fmt_fixed(x, dec=precision) if x is not None else "NA",
                return_dtype=pl.Utf8,
            )
            .alias(f"{s}__str")
            for s in stats
        ]
        dfm = dfm.with_columns(fmt_cols)
        dfm = dfm.with_columns(
            _compose_cell(dfm, stats=stats, mode=cellfmt or "auto").alias("__cell__")
        )

        wide = (
            dfm.select(["rowvar", "colvar", "__cell__"])
            .pivot(values="__cell__", index="rowvar", on="colvar", aggregate_function=None)
            .rename({"rowvar": self.rowvar})
        )
        wide = _reindex_polars(
            wide, row_levels=row_levels, col_levels=col_levels, row_name=self.rowvar
        )
        if fill_missing is not None:
            wide = wide.fill_null(fill_missing)
        return wide

    def to_dataframe(
        self,
        *,
        stats: str | tuple[str, ...] = ("est", "se"),
        sort_rows: bool = True,
        sort_cols: bool = True,
        fill_missing: str | None = None,
        cellfmt: str | None = None,
        precision: int = 5,
        use_labels: bool | None = None,
    ) -> pl.DataFrame:
        return self.crosstab(
            stats=stats,
            precision=precision,
            fill_missing=fill_missing,
            cellfmt=cellfmt,
            sort_rows=sort_rows,
            sort_cols=sort_cols,
            use_labels=use_labels,
        )


# =============================================================================
# Printing helpers
# =============================================================================


def _headers_for_display(tbl: "Table") -> list[str]:
    if tbl._resolve_use_labels() and tbl._metadata is not None:
        row_resolved = tbl._metadata.resolve_labels(tbl.rowvar)
        row_header = row_resolved.var_label if row_resolved.has_var_label else "Row"
    else:
        row_header = "Row"

    base = [row_header]
    if tbl.is_crosstab:
        if tbl._resolve_use_labels() and tbl._metadata is not None:
            col_resolved = tbl._metadata.resolve_labels(tbl.colvar or "")
            col_header = col_resolved.var_label if col_resolved.has_var_label else "Col"
        else:
            col_header = "Col"
        base.append(col_header)
    base += ["Estimate", "Std Err", "CV", "Lower", "Upper"]
    return base


def _rows_for_display(
    tbl: "Table",
    *,
    dec_map: dict[str, int] | None = None,
    default_dec: int = 5,
) -> Iterator[list[str]]:
    """Yield display rows, sorted on resolved display values (after label mapping).

    Sorting happens after label resolution so that labelled values like
    "1. Urban" / "2. Rural" sort on their display strings, not raw codes.
    Explicit rowvals/colvals ordering is respected via level_index_key when
    provided; otherwise level_sort_key handles numerics, intervals, and
    natural-sort strings uniformly.
    """
    include_col = tbl.colvar is not None
    ests = list(tbl.estimates or [])

    # Resolve explicit level orderings (display strings, for post-label sort)
    row_levels = (
        [tbl._get_row_display(v) for v in tbl.rowvals] if tbl.rowvals is not None else None
    )
    col_levels = (
        [tbl._get_col_display(v) for v in tbl.colvals] if tbl.colvals is not None else None
    )

    d = dec_map or {k: default_dec for k in _DECIMAL_KEYS}
    de = d.get("est", default_dec)
    ds = d.get("se", default_dec)
    dc = d.get("cv", default_dec)
    dl = d.get("lci", default_dec)
    du = d.get("uci", default_dec)

    # Build rows with resolved display values first, then sort on them.
    rows: list[list[str]] = []
    for c in ests:
        row_display = tbl._get_row_display(c.rowvar)
        row = [row_display]
        if include_col:
            col_display = tbl._get_col_display(c.colvar)
            row.append(col_display)
        row.extend(
            [
                _fmt_fixed(c.est, dec=de),
                _fmt_fixed(c.se, dec=ds),
                _fmt_fixed(c.cv, dec=dc),
                _fmt_fixed(c.lci, dec=dl),
                _fmt_fixed(c.uci, dec=du),
            ]
        )
        rows.append(row)

    # Sort on display values using level_index_key (respects explicit ordering,
    # falls back to level_sort_key for numerics / intervals / natural strings).
    if include_col:
        rows.sort(
            key=lambda r: (
                level_index_key(r[0], row_levels),
                level_index_key(r[1], col_levels),
            )
        )
    else:
        rows.sort(key=lambda r: level_index_key(r[0], row_levels))

    yield from rows


def show_table(tbl: Table, *, dec: int = 5, use_rich: bool = True) -> None:
    from svy.ui.printing import rich_available

    if use_rich and rich_available():
        from rich.console import Console

        Console(
            force_terminal=True,
            color_system="auto",
            emoji=False,
            width=resolve_width(tbl),
        ).print(tbl)
        return

    base_map = tbl._decimals_map(fallback_each=dec)
    headers = _headers_for_display(tbl)
    rows = list(_rows_for_display(tbl, dec_map=base_map, default_dec=dec))
    print(render_plain_table(headers, rows))


# =============================================================================
# Polars helpers
# =============================================================================


def _reindex_rows_only(df, *, row_levels, row_name: str):
    skel = pl.DataFrame({row_name: row_levels})
    return skel.join(df, on=row_name, how="left")


def _reindex_polars(wide, *, row_levels, col_levels, row_name: str):
    wide = pl.DataFrame({row_name: row_levels}).join(wide, on=row_name, how="left")

    current_cols = [c for c in wide.columns if c != row_name]
    desired_cols = [c for c in col_levels if c in current_cols]
    missing_cols = [c for c in col_levels if c not in current_cols]

    dtype_ref = None
    for c in desired_cols or current_cols:
        dt = wide.schema.get(c)
        if dt is not None and dt != pl.Null and dt != pl.Utf8:
            dtype_ref = dt
            break
    if dtype_ref is None:
        dtype_ref = pl.Float64

    for c in missing_cols:
        wide = wide.with_columns(pl.lit(None).cast(dtype_ref).alias(c))

    return wide.select([row_name] + desired_cols + missing_cols)


def _compose_cell(dfm, *, stats: tuple[str, ...], mode: str):
    def S(s: str) -> pl.Expr:
        return pl.col(f"{s}__str")

    mode = (mode or "auto").lower()

    if mode == "pm" and ("est" in stats and "se" in stats):
        return S("est") + pl.lit(" ± ") + S("se")
    if mode == "ci" and ("est" in stats and "lci" in stats and "uci" in stats):
        return S("est") + pl.lit(" [") + S("lci") + pl.lit(", ") + S("uci") + pl.lit("]")
    if mode == "slash" and ("est" in stats and "se" in stats):
        return S("est") + pl.lit("/") + S("se")
    if mode == "comma":
        parts = [S(s) for s in stats]
        out = parts[0]
        for p in parts[1:]:
            out = out + pl.lit(", ") + p
        return out

    if "est" in stats and "se" in stats:
        return S("est") + pl.lit(" ± ") + S("se")
    if "est" in stats and "lci" in stats and "uci" in stats:
        return S("est") + pl.lit(" [") + S("lci") + pl.lit(", ") + S("uci") + pl.lit("]")
    parts = [S(s) for s in stats]
    out = parts[0]
    for p in parts[1:]:
        out = out + pl.lit(", ") + p
    return out


def _ensure_numeric(df: pl.DataFrame, col: str) -> pl.DataFrame:
    if col not in df.columns:
        return df
    dtype = df.schema.get(col)
    if dtype in (pl.Utf8, pl.Categorical):
        return df.with_columns(pl.col(col).cast(pl.Float64))
    return df
