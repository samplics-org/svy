# src/svy/estimation/estimate.py
from __future__ import annotations

import logging

from typing import TYPE_CHECKING, Any, Literal, Sequence

import msgspec
import numpy as np
import polars as pl

from svy.core.enumerations import EstimationMethod, PopParam, QuantileMethod
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
    from svy.metadata import MetadataStore


log = logging.getLogger(__name__)

_DECIMAL_KEYS = ("est", "se", "lci", "uci", "cv", "deff")


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
        "n_strata",
        "n_psus",
        "degrees_freedom",
        "as_factor",
        "where_clause",
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
        self.method: EstimationMethod = EstimationMethod.TAYLOR
        self.n_strata: int = 0
        self.n_psus: int = 0
        self.degrees_freedom: int = 0
        self.as_factor: bool = False
        self.q_method: QuantileMethod = QuantileMethod.LINEAR
        self.where_clause: str | None = None

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

        for est in self.estimates:
            r = {}
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
            rows.append(r)

        sort_display_rows(rows, numeric_keys=set(_DECIMAL_KEYS))

        return pl.from_dicts(rows)

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
            return f"Estimate: {self.param.name} ({self.method.name}) — <no estimates>"

        lines = [f"Estimate: {self.param.name} ({self.method.name})"]
        if self.where_clause:
            lines.append(f"  where: {self.where_clause}")
        lines.append("")

        headers = [f"{c} (%)" if c == "cv" else c for c in df.columns]
        rows = [
            [self._format_val(c, row[c]) for c in df.columns] for row in df.iter_rows(named=True)
        ]
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

        for col in df.columns:
            justify = "right" if col in _DECIMAL_KEYS else "left"
            header = f"{col} (%)" if col == "cv" else col
            table.add_column(header, justify=justify)

        for row in df.iter_rows(named=True):
            vals = [self._format_val(col, row[col]) for col in df.columns]
            table.add_row(*vals)

        content.append(table)

        title = f"Estimate: [bold]{self.param.name}[/bold] ({self.method.name})"

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
