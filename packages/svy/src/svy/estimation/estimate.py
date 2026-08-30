# src/svy/estimation/estimate.py
from __future__ import annotations

import logging

from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

import msgspec
import numpy as np
import polars as pl

from svy.core.enumerations import PopParam, QuantileMethod
from svy.core.types import Category, Number, RandomState

# Import central UI helpers
from svy.ui.printing import (
    make_panel,
    render_plain_table,
    render_rich_to_str,
    resolve_width,
    sort_display_rows,
)


if TYPE_CHECKING:
    from svy.estimation.contrast import Contrast, ContrastExpr
    from svy.metadata import MetadataStore


log = logging.getLogger(__name__)

_DECIMAL_KEYS = ("est", "se", "lci", "uci", "cv", "deff")

# Carried by to_polars() but kept out of the printed table: df is a per-row
# value that is constant for most results, so a column would repeat one number
# down the page and widen every table. Reach for it via to_polars().
_HIDDEN_DISPLAY_COLS = ("df",)


def _display_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _HIDDEN_DISPLAY_COLS]


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


class ParamEst(msgspec.Struct, frozen=True):
    y: str
    est: Number
    se: Number
    cv: Number
    lci: Number
    uci: Number
    by: tuple[str, ...] | None = None
    by_level: tuple[Category, ...] | None = None
    y_level: Category | None = None
    x: str | None = None
    x_level: Category | None = None
    deff: Number | None = None
    df: int | None = None
    #: Target probability, set only for quantile estimates (0.5 for the median).
    prob: Number | None = None

    def to_dict(self) -> dict[str, object]:
        return {f: getattr(self, f) for f in self.__struct_fields__}


class Estimate:
    """
    Container for estimation results.
    """

    DECIMALS: int | dict[str, int] | None = None
    PRINT_WIDTH: int | None = None
    USE_LABELS: bool = True

    __slots__ = (
        "param",
        "q_method",
        "alpha",
        "estimates",
        "covariance",
        "strata",
        "singletons",
        "domains",
        "method",
        "deff_ref",
        "n_strata",
        "n_psus",
        "as_factor",
        "where_clause",
        "design_df",
        "_cov_filled",
        "_decimals",
        "_layout",
        "_print_width",
        "_use_labels",
        "_metadata",
    )

    def __init__(
        self,
        param: PopParam,
        *,
        alpha: float = 0.05,
        rstate: RandomState = None,
        metadata: "MetadataStore | None" = None,
    ):
        self.param = param
        self.alpha = alpha
        self.estimates: list[ParamEst] = []
        self.covariance: np.ndarray = np.zeros((0, 0))
        self.strata: Sequence[Category] = []
        self.singletons: Sequence[Category] = []
        self.domains: Sequence[Category] = []
        self.method: str = "Taylor"
        #: Which SRS reference the design effect was measured against, or None
        #: when no design effect was requested. Recorded because a deff is
        #: ambiguous without it: the two references differ by 1 - n/N.
        self.deff_ref: str | None = None
        self.n_strata: int = 0
        self.n_psus: int = 0
        self.as_factor: bool = False
        self.q_method: QuantileMethod = QuantileMethod.LINEAR
        self.where_clause: str | None = None
        #: Full design degrees of freedom (R's ``degf``), as opposed to the
        #: per-row domain-aware df. Cross-domain contrasts are referred to
        #: this value.
        self.design_df: int | None = None
        #: Whether the off-diagonals of ``covariance`` were actually computed.
        #: Multi-variable convenience calls estimate each variable
        #: independently and leave them zeroed; ``contrast()`` refuses those.
        self._cov_filled: bool = False

        self._decimals = None
        self._layout = "auto"
        self._print_width = None
        self._use_labels = None
        self._metadata = metadata

    # =========================================================================
    # Label resolution helpers
    # =========================================================================

    def _resolve_use_labels(self) -> bool:
        """Resolve whether to use labels: instance -> class -> True."""
        if self._use_labels is not None:
            return self._use_labels
        return getattr(type(self), "USE_LABELS", True)

    def _get_var_label(self, var: str, use_labels: bool | None = None) -> str:
        """Get variable label or fall back to variable name."""
        resolve = use_labels if use_labels is not None else self._resolve_use_labels()
        if not resolve or self._metadata is None:
            return var
        resolved = self._metadata.resolve_labels(var)
        return resolved.var_label if resolved.has_var_label else var

    def _get_value_label(self, var: str, value: Category, use_labels: bool | None = None) -> str:
        """Get value label or fall back to string representation."""
        resolve = use_labels if use_labels is not None else self._resolve_use_labels()
        if not resolve or self._metadata is None:
            return str(value)
        resolved = self._metadata.resolve_labels(var)
        # Try the value as-is first
        label = resolved.display(value)
        # If we got back the string representation, try converting to int
        if label == str(value) and isinstance(value, str):
            try:
                int_value = int(value)
                label = resolved.display(int_value)
            except (ValueError, TypeError):
                pass
        return label

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
        self._use_labels = value

    def _context(self) -> str:
        """Variance method, plus the design-effect reference when one applies.

        The reference belongs in the header rather than the frame: a deff is
        ambiguous without it, since the two references differ by 1 - n/N, but
        `to_polars` deliberately carries no provenance at all.
        """
        if self.deff_ref:
            return f"{self.method.upper()}, deff={self.deff_ref}"
        return self.method.upper()

    @classmethod
    def set_default_use_labels(cls, use: bool) -> None:
        """Set the default label usage for all Estimate instances."""
        cls.USE_LABELS = bool(use)

    @property
    def metadata(self) -> "MetadataStore | None":
        """Get the metadata store."""
        return self._metadata

    @metadata.setter
    def metadata(self, value: "MetadataStore | None") -> None:
        """Set the metadata store."""
        self._metadata = value

    # --- Configuration ---

    def set_print_width(self, width: int | None) -> "Estimate":
        if width is None:
            self._print_width = None
            return self
        try:
            w = int(width)
        except Exception as ex:
            raise TypeError(f"print width must be int or None; got {width!r}") from ex
        if w <= 20:
            raise ValueError("print width must be > 20 characters.")
        self._print_width = w
        return self

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

    # --- Properties ---

    @property
    def decimals(self) -> int | dict[str, int] | None:
        return self._decimals

    @decimals.setter
    def decimals(self, value: int | dict[str, int] | None) -> None:
        if value is None or isinstance(value, int):
            self._decimals = value
        elif isinstance(value, dict):
            if any(k not in _DECIMAL_KEYS for k in value):
                raise ValueError(f"Unknown decimals key; allowed: {_DECIMAL_KEYS}")
            self._decimals = dict(value)
        else:
            raise TypeError("decimals must be int | dict[str,int] | None")

    @property
    def print_width(self) -> int | None:
        return self._print_width

    @print_width.setter
    def print_width(self, value: int | None) -> None:
        self.set_print_width(value)

    @property
    def layout(self) -> Literal["auto", "horizontal", "vertical"]:
        return self._layout

    @layout.setter
    def layout(self, value: Literal["auto", "horizontal", "vertical"]) -> None:
        if value not in ("auto", "horizontal", "vertical"):
            raise ValueError("layout must be 'auto', 'horizontal', or 'vertical'")
        self._layout = value

    # --- Export ---

    def to_dicts(self) -> list[dict[str, Any]]:
        return [p.to_dict() for p in self.estimates] if self.estimates else []

    def to_polars(self, *, tidy: bool = True, use_labels: bool | None = None) -> pl.DataFrame:
        if not self.estimates:
            return pl.DataFrame()
        if not tidy:
            return pl.from_dicts(self.to_dicts())
        return self.to_polars_printable(use_labels=use_labels if use_labels is not None else False)

    def to_polars_printable(self, *, use_labels: bool | None = None) -> pl.DataFrame:
        """
        Convert estimates to a printable DataFrame.

        Parameters
        ----------
        use_labels : bool | None
            If True, apply value labels to by_level and y_level columns.
            If None, uses instance/class default.

        Returns
        -------
        pl.DataFrame
            DataFrame formatted for display.
        """
        if not self.estimates:
            return pl.DataFrame()

        # Resolve label usage
        resolve_labels = use_labels if use_labels is not None else self._resolve_use_labels()

        first = self.estimates[0]
        by_cols = list(first.by) if first.by else []
        y_col = first.y
        show_y_level = self.param == PopParam.PROP or self.as_factor

        # Build rows with display values first, then sort on the resolved display
        # values. Sorting must happen AFTER label resolution because raw by_level
        # codes (e.g. "Rural", "Urban") differ from their display labels
        # (e.g. "2. Rural", "1. Urban"), and natural sort on the labels is what
        # the user expects to see.
        rows = []

        show_prob = self.param == PopParam.QUANTILE

        for est in self.estimates:
            r = {}
            if show_prob and est.prob is not None:
                # Leading column, so quantiles read p → estimate left to right.
                r["prob"] = est.prob
            if by_cols:
                levels = est.by_level or (None,) * len(by_cols)
                for i, col in enumerate(by_cols):
                    raw_val = levels[i] if i < len(levels) else None
                    if raw_val is not None and resolve_labels and self._metadata is not None:
                        display_val = self._get_value_label(
                            col, raw_val, use_labels=resolve_labels
                        )
                    else:
                        display_val = str(raw_val) if raw_val is not None else None
                    col_name = self._get_var_label(col, use_labels=resolve_labels)
                    r[col_name] = display_val

            if show_y_level:
                raw_y_level = est.y_level
                if raw_y_level is not None and resolve_labels and self._metadata is not None:
                    display_y = self._get_value_label(
                        y_col, raw_y_level, use_labels=resolve_labels
                    )
                else:
                    display_y = str(raw_y_level) if raw_y_level is not None else None
                y_col_name = self._get_var_label(y_col, use_labels=resolve_labels)
                r[y_col_name] = display_y

            for key in _DECIMAL_KEYS:
                val = getattr(est, key, None)
                if val is not None:
                    r[key] = val
            if est.df is not None:
                r["df"] = est.df
            rows.append(r)

        sort_display_rows(rows, numeric_keys=set(_DECIMAL_KEYS))

        return pl.from_dicts(rows)

    # --- Contrasts & covariance ---

    def _row_key(self, p: ParamEst) -> Any:
        """The contrast key identifying one estimate row.

        Domain estimates key on their by-level (a tuple when several ``by``
        variables), categorical proportions on the y-level, and combined
        cases on the (by-levels..., y-level) tuple. A single ungrouped row
        keys on the variable name itself.
        """
        parts: list = []
        if p.by_level:
            parts.extend(p.by_level)
        if (self.param == PopParam.PROP or self.as_factor) and p.y_level is not None:
            parts.append(p.y_level)
        if not parts:
            return p.y
        return parts[0] if len(parts) == 1 else tuple(parts)

    def keys(self, *, labels: bool = False) -> list:
        """Contrast keys, one per estimate row, in row (and covariance) order.

        With ``labels=True``, each key component is replaced by its metadata
        value label where one exists — the same rendering the printed table
        uses. Both forms are accepted by :meth:`contrast`.
        """
        raw = [self._row_key(p) for p in self.estimates]
        if not labels:
            return raw
        return [self._labeled_key(p, k) for p, k in zip(self.estimates, raw)]

    def _labeled_key(self, p: ParamEst, key: Any) -> Any:
        """The key with every component swapped for its value label."""
        if self._metadata is None:
            return key
        labeled: list = []
        if p.by_level:
            for var, val in zip(p.by or (), p.by_level):
                labeled.append(self._get_value_label(var, val, use_labels=True))
        if (self.param == PopParam.PROP or self.as_factor) and p.y_level is not None:
            labeled.append(self._get_value_label(p.y, p.y_level, use_labels=True))
        if not labeled:
            return key
        return labeled[0] if len(labeled) == 1 else tuple(labeled)

    def _label_aliases(self) -> dict:
        """Label-form aliases for contrast keys (label → row index)."""
        if self._metadata is None:
            return {}
        aliases: dict = {}
        raw = [self._row_key(p) for p in self.estimates]
        for i, (p, key) in enumerate(zip(self.estimates, raw)):
            lk = self._labeled_key(p, key)
            if lk != key:
                aliases[lk] = i
        return aliases

    def contrast(
        self,
        contrasts: "Mapping[Any, Any] | ContrastExpr",
        *,
        alpha: float | None = None,
    ) -> "Contrast":
        """Estimate linear contrasts between this result's estimands.

        ``contrasts`` is a contrast expression built from :func:`svy.estd`
        references (``estd("E") - estd("H")``), a sparse ``{key: coef}``
        dict, or several named contrasts ``{"name": expression-or-dict}``.
        Keys are the row identities listed by :meth:`keys` — metadata value
        labels are accepted wherever they are unambiguous. Unmentioned
        estimands get coefficient 0; unknown keys raise.

        Inference is t-based on the full design degrees of freedom
        (:attr:`design_df`, R's ``degf`` convention), not the per-row
        domain-aware df.
        """
        from svy.errors import MethodError
        from svy.estimation.contrast import linear_contrast

        if self.param in (PopParam.QUANTILE, PopParam.MEDIAN):
            raise MethodError(
                title="Contrasts are not defined for quantiles",
                detail=(
                    "Woodruff quantile intervals do not arise from a "
                    "linearized score column, so no between-quantile "
                    "covariance exists to combine them with."
                ),
                code="CONTRAST_UNSUPPORTED_PARAM",
                where="Estimate.contrast",
                param="param",
                got=self.param.name,
            )
        if not self.estimates:
            raise MethodError(
                title="Nothing to contrast",
                detail="This result carries no estimates.",
                code="CONTRAST_EMPTY",
                where="Estimate.contrast",
            )
        k = len(self.estimates)
        if k > 1 and not self._cov_filled:
            raise MethodError(
                title="No between-estimate covariance on this result",
                detail=(
                    "Multi-variable convenience calls estimate each variable "
                    "independently, so no covariance between their rows was "
                    "computed. Re-run the single-variable form (e.g. "
                    "prop('y') instead of prop(['y', ...])) and contrast "
                    "that result."
                ),
                code="CONTRAST_NO_COVARIANCE",
                where="Estimate.contrast",
            )

        df = self.design_df
        if df is None:
            row_dfs = [p.df for p in self.estimates if p.df is not None]
            if not row_dfs:
                raise MethodError(
                    title="No degrees of freedom on this result",
                    detail="Neither a design df nor per-row df is available.",
                    code="CONTRAST_NO_DF",
                    where="Estimate.contrast",
                )
            df = max(row_dfs)

        values = np.array([p.est for p in self.estimates], dtype=float)
        return linear_contrast(
            self.keys(),
            values,
            self.covariance,
            contrasts,
            df=float(df),
            alpha=alpha if alpha is not None else self.alpha,
            method=self.method,
            aliases=self._label_aliases(),
        )

    def covariance_to_polars(self) -> pl.DataFrame:
        """Tidy lower-triangle view of the between-estimate covariance.

        One row per (key_a, key_b) pair including the diagonal; the dense
        matrix stays on :attr:`covariance` in :meth:`keys` order.
        """
        keys = [str(k) for k in self.keys()]
        cov = np.asarray(self.covariance)
        rows = [
            {"key_a": keys[i], "key_b": keys[j], "cov": float(cov[i, j])}
            for i in range(len(keys))
            for j in range(i + 1)
        ]
        return pl.from_dicts(rows) if rows else pl.DataFrame()

    # --- Formatting ---

    def _get_precision(self, col: str) -> int:
        conf = self._decimals or self.DECIMALS
        defaults = {"cv": 2, "est": 4, "se": 4, "lci": 4, "uci": 4, "deff": 4}
        if conf is None:
            return defaults.get(col, 4)
        if isinstance(conf, int):
            return conf
        return conf.get(col, defaults.get(col, 4))

    def _format_val(self, col: str, v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        if isinstance(v, (float, np.floating)):
            prec = self._get_precision(col)
            val = float(v)
            if col == "cv":
                return f"{val * 100:.{prec}f}"
            if col in ("est", "se", "lci", "uci"):
                return f"{val:,.{prec}f}"
            return f"{val:.{prec}f}"
        return str(v)

    # --- Rendering ---

    def __plain_str__(self) -> str:
        """
        Plain-text fallback used when rich is not installed.
        Called by printing.plain_text_fallback(); must NOT call str(self).
        """
        df = self.to_polars_printable()
        if df.is_empty():
            return f"Estimate: {self.param.name} ({self._context()}) — <no estimates>"

        lines = [f"Estimate: {self.param.name} ({self._context()})"]
        if self.where_clause:
            lines.append(f"  where: {self.where_clause}")
        lines.append("")

        shown = _display_columns(df)
        headers = [f"{c} (%)" if c == "cv" else c for c in shown]
        rows = [[self._format_val(c, row[c]) for c in shown] for row in df.iter_rows(named=True)]
        lines.append(render_plain_table(headers, rows))

        return "\n".join(lines)

    def __str__(self) -> str:
        """
        Return string representation for standard print().
        Delegates completely to svy.ui.printing for consistency.
        """
        try:
            # 1. Resolve width (checks instance -> class -> env -> default)
            w = resolve_width(self)
            # 2. Render to string using the centralized console config
            return render_rich_to_str(self, width=w)
        except Exception:
            return self.__repr__()

    def __rich_console__(self, console, options):
        """Integration with the Rich library for pretty printing."""
        from rich import box
        from rich.table import Table
        from rich.text import Text

        df = self.to_polars_printable()

        if df.is_empty():
            yield Text("<no estimates>", style="italic dim")
            return

        # Build content list for the panel
        content = []

        # Add where clause as first item if present
        if self.where_clause:
            where_text = Text()
            where_text.append("where: ", style="dim")
            where_text.append(self.where_clause)
            content.append(where_text)
            content.append(Text(""))  # Empty line for spacing

        # TABLE CONFIGURATION
        table = Table(
            show_header=True,
            header_style="bold",
            box=box.SIMPLE_HEAVY,
            show_edge=True,
            show_lines=False,
            pad_edge=False,
            expand=False,
        )

        shown = _display_columns(df)

        for col in shown:
            justify = "right" if col in _DECIMAL_KEYS else "left"
            header = f"{col} (%)" if col == "cv" else col
            table.add_column(header, justify=justify)

        for row in df.iter_rows(named=True):
            vals = [self._format_val(col, row[col]) for col in shown]
            table.add_row(*vals)

        content.append(table)

        title = f"Estimate: [bold]{self.param.name}[/bold] ({self._context()})"

        # PANEL CONFIGURATION
        yield make_panel(content, title=title, obj=self, kind="estimate")

    def style(
        self,
        *,
        decimals: int | dict[str, int] | None = None,
        print_width: int | None = None,
        use_labels: bool | None = None,
        layout: Literal["auto", "horizontal", "vertical"] | None = None,
    ) -> "Estimate":
        """
        Fluent convenience to set presentation options.

        Parameters
        ----------
        decimals : int | dict[str, int] | None
            Decimal places for numeric columns.
        print_width : int | None
            Print width in characters.
        use_labels : bool | None
            Whether to display labels instead of codes.
        layout : {"auto", "horizontal", "vertical"} | None
            Table layout mode.

        Returns
        -------
        Estimate
            Self for method chaining.
        """
        if decimals is not None:
            self.decimals = decimals
        if print_width is not None:
            self.print_width = print_width
        if use_labels is not None:
            self.use_labels = use_labels
        if layout is not None:
            self.layout = layout
        return self


class EstimateList(list):
    """A list of :class:`Estimate` results that prints as one table.

    Returned wherever a call estimates several things at once — a sequence of
    variables (``mean(["a", "b"])``) or a sequence of probabilities
    (``quantile("x", p=(0.25, 0.75))``).

    This is a plain ``list`` subclass, so indexing, iteration, ``len()``, and
    ``isinstance(result, list)`` all behave exactly as before; the only addition
    is rendering. Printing stacks the members into a single table with a leading
    column for whatever differs between them (``y``, ``prob``, or both), because
    a bare list would otherwise print object reprs.
    """

    __slots__ = ()

    def _members(self) -> list["Estimate"]:
        return [e for e in self if isinstance(e, Estimate) and e.estimates]

    def contrast(
        self,
        contrasts: "Mapping[Any, Any] | ContrastExpr",
        *,
        alpha: float | None = None,
    ) -> "Contrast":
        """Contrast the single member of this list; raise otherwise.

        Covariance exists only within one estimation (batched variables are
        computed independently), so a multi-member list has no joint
        covariance to contrast over — index the member instead.
        """
        members = self._members()
        if len(members) == 1:
            return members[0].contrast(contrasts, alpha=alpha)
        from svy.errors import MethodError

        raise MethodError(
            title="contrast() needs a single result",
            detail=(
                f"This list holds {len(members)} results, estimated "
                "independently — there is no covariance between them. "
                "Contrast one member, e.g. result[0].contrast(...)."
            ),
            code="CONTRAST_LIST_AMBIGUOUS",
            where="EstimateList.contrast",
        )

    def to_polars(self, *, use_labels: bool | None = None) -> pl.DataFrame:
        """Concatenate the members into one frame, one row per estimate."""
        frames = [e.to_polars_printable(use_labels=use_labels) for e in self._members()]
        frames = [f for f in frames if not f.is_empty()]
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="diagonal_relaxed")

    def _combined(self, *, use_labels: bool | None = None) -> pl.DataFrame:
        """The printable frame, with a ``y`` column added when variables differ.

        ``prob`` already arrives as a column on quantile members, so only the
        variable needs re-attaching: a single-variable result keeps ``y`` out of
        the table (it is in the title), while a multi-variable one needs it to
        stay readable.
        """
        members = self._members()
        if not members:
            return pl.DataFrame()

        ys = [m.estimates[0].y for m in members]
        show_y = len(set(ys)) > 1

        frames = []
        for m, y in zip(members, ys):
            f = m.to_polars_printable(use_labels=use_labels)
            if f.is_empty():
                continue
            if show_y and "y" not in f.columns:
                f = f.select(pl.lit(y).alias("y"), pl.all())
            frames.append(f)

        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="diagonal_relaxed")

    def _title(self) -> str:
        members = self._members()
        if not members:
            return "Estimates"
        params = {m.param.name for m in members}
        methods = {m.method.upper() for m in members}
        param = params.pop() if len(params) == 1 else "MIXED"
        method = methods.pop() if len(methods) == 1 else "MIXED"
        ys = {m.estimates[0].y for m in members}
        suffix = f": {next(iter(ys))}" if len(ys) == 1 else ""
        return f"Estimate: [bold]{param}[/bold] ({method}){suffix}"

    def __plain_str__(self) -> str:
        """Plain-text fallback used when rich is not installed."""
        members = self._members()
        if not members:
            return "Estimates — <no estimates>"

        df = self._combined()
        if df.is_empty():
            return "Estimates — <no estimates>"

        # Strip the rich markup the panel title carries.
        title = self._title().replace("[bold]", "").replace("[/bold]", "")
        lines = [title, ""]
        shown = _display_columns(df)
        headers = [f"{c} (%)" if c == "cv" else c for c in shown]
        fmt = members[0]._format_val
        rows = [[fmt(c, row[c]) for c in shown] for row in df.iter_rows(named=True)]
        lines.append(render_plain_table(headers, rows))
        return "\n".join(lines)

    def __str__(self) -> str:
        if not self._members():
            return super().__repr__()
        try:
            return render_rich_to_str(self, width=resolve_width(self))
        except Exception:
            return self.__plain_str__()

    # A list's repr is what `print([...])` and the REPL both reach for, so the
    # table has to be the repr, not only __str__.
    __repr__ = __str__

    def __rich_console__(self, console, options):
        from rich import box
        from rich.table import Table
        from rich.text import Text

        members = self._members()
        df = self._combined() if members else pl.DataFrame()
        if df.is_empty():
            yield Text("<no estimates>", style="italic dim")
            return

        table = Table(
            show_header=True,
            header_style="bold",
            box=box.SIMPLE_HEAVY,
            show_edge=True,
            show_lines=False,
            pad_edge=False,
            expand=False,
        )

        shown = _display_columns(df)
        for col in shown:
            justify = "right" if col in _DECIMAL_KEYS else "left"
            table.add_column(f"{col} (%)" if col == "cv" else col, justify=justify)

        fmt = members[0]._format_val
        for row in df.iter_rows(named=True):
            table.add_row(*[fmt(col, row[col]) for col in shown])

        content: list = []
        wheres = {m.where_clause for m in members if m.where_clause}
        if len(wheres) == 1:
            where_text = Text()
            where_text.append("where: ", style="dim")
            where_text.append(wheres.pop())
            content.append(where_text)
            content.append(Text(""))
        content.append(table)

        yield make_panel(content, title=self._title(), obj=self, kind="estimate")
