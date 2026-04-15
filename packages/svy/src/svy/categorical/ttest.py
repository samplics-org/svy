# src/svy/categorical/ttest.py
from __future__ import annotations

import logging

from typing import TYPE_CHECKING, Any, ClassVar, Iterator, Literal, Self, Sequence

import msgspec

from msgspec import field

from svy.core.types import Category, Number
from svy.errors import DimensionError, MethodError
from svy.estimation import ParamEst

# Import central UI helpers (consistent with estimate.py)
from svy.ui.printing import (
    format_where_clause,
    make_panel,
    natural_sort_key,
    render_plain_table,
    render_rich_to_str,
    resolve_width,
)
from svy.utils.formats import _fmt_fixed


if TYPE_CHECKING:
    import polars as pl

log = logging.getLogger(__name__)

# Keys used by the central UI for styling
_UI_KEY_RESULTS = "ttest.results"


# =============================================================================
# Data structures
# =============================================================================


class TtestEst(msgspec.Struct):
    """Estimate for a (y, group) cell in a t-test context."""

    by: str | None
    by_level: Category | None
    group: str | None
    group_level: Category | None
    y: str
    y_level: Category | None
    est: Number
    se: Number
    cv: Number
    lci: Number
    uci: Number

    @classmethod
    def from_param(cls, param_est: ParamEst) -> Self:
        return cls(
            by=None,
            by_level=None,
            group=param_est.by[0] if isinstance(param_est.by, tuple) else param_est.by,
            group_level=param_est.by_level[0]
            if isinstance(param_est.by_level, tuple)
            else param_est.by_level,
            y=param_est.y,
            y_level=param_est.y_level,
            est=param_est.est,
            se=param_est.se,
            cv=param_est.cv,
            lci=param_est.lci,
            uci=param_est.uci,
        )


class DiffEst(msgspec.Struct, frozen=True):
    """Difference estimate for t-test.

    For one-sample: diff = est - mean_h0
    For two-sample: diff = est1 - est2
    """

    y: str
    diff: Number
    se: Number
    lci: Number
    uci: Number
    by: str | None = None
    by_level: Category | None = None


class TTestStats(msgspec.Struct, frozen=True):
    """Core t-test outputs."""

    t: float  # t statistic
    df: Number  # df can be fractional for some designs
    p_value: float


class GroupLevels(msgspec.Struct, frozen=True):
    """Two-group comparison specification."""

    var: str
    levels: tuple[Category, Category]


# ------ Helper functions --------


def _validate_one_sample(tt: "TTestOneGroup") -> None:
    if not (0.0 < float(tt.alpha) < 1.0):
        raise MethodError.invalid_range(
            where="TTestOneGroup", param="alpha", got=tt.alpha, min_=0.0, max_=1.0
        )
    if not tt.estimates:
        raise DimensionError.empty_estimates(where="TTestOneGroup", param="estimates")


def _validate_two_sample(tt: "TTestTwoGroups") -> None:
    if not (0.0 < float(tt.alpha) < 1.0):
        raise MethodError.invalid_range(
            where="TTestTwoGroups", param="alpha", got=tt.alpha, min_=0.0, max_=1.0
        )
    if not tt.groups or not isinstance(tt.groups.levels, tuple) or len(tt.groups.levels) != 2:
        raise MethodError.invalid_choice(
            where="TTestTwoGroups",
            param="groups.levels",
            got=getattr(tt.groups, "levels", None),
            allowed="tuple of length 2",
            hint="Provide exactly two group levels for a two-sample t-test.",
        )
    if not tt.estimates:
        raise DimensionError.empty_estimates(where="TTestTwoGroups", param="estimates")

    gvar = tt.groups.var
    exp_levels = set(tt.groups.levels)
    got_levels = {e.group_level for e in tt.estimates if e.group == gvar}
    other_vars = {e.group for e in tt.estimates if e.group and e.group != gvar}
    if other_vars:
        raise MethodError.not_applicable(
            where="TTestTwoGroups",
            method="estimates",
            reason=f"found estimates for other group variables {sorted(other_vars)}; expected only '{gvar}'",
            param="estimates",
        )
    if not got_levels.issubset(exp_levels):
        raise DimensionError.group_levels_mismatch(
            where="TTestTwoGroups",
            var=gvar,
            expected_levels=sorted(exp_levels),
            got_levels=sorted(got_levels),
        )


# -------- Main containers --------


class TTestOneGroup(msgspec.Struct, tag="one", tag_field="kind", kw_only=True, frozen=True):
    """One-sample t-test: H0: mean(y) == mean_h0."""

    PRINT_WIDTH: ClassVar[int | None] = None

    y: str
    mean_h0: Number = 0.0
    alternative: Literal["two-sided", "less", "greater"] = "two-sided"
    diff: list[DiffEst] = field(default_factory=list)
    estimates: list[TtestEst] = field(default_factory=list)
    stats: TTestStats | None = None
    alpha: float = 0.05

    # ---------------- Width Configuration ----------------

    @classmethod
    def set_default_print_width(cls, width: int | None) -> None:
        """
        Set the default print width for all TTestOneGroup instances.

        Args:
            width: Print width in characters (must be > 20), or None to reset.

        Raises:
            TypeError: If width is not an int or None.
            ValueError: If width is <= 20.
        """
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

    # ---------------- Data Export ----------------

    def to_dict(self) -> dict[str, Any]:
        """
        Exports the entire result to a dictionary.

        Returns:
            A dictionary representation of all fields.
        """
        return msgspec.to_builtins(self)

    def to_polars(
        self,
        component: Literal["test", "estimates"] = "test",
        *,
        tidy: bool = True,
    ) -> pl.DataFrame:
        import polars as pl

        if component == "test":
            if not self.diff:
                return pl.DataFrame(
                    schema={
                        "y": pl.Utf8,
                        "diff": pl.Float64,
                        "se": pl.Float64,
                        "lci": pl.Float64,
                        "uci": pl.Float64,
                        "t": pl.Float64,
                        "df": pl.Float64,
                        "p_value": pl.Float64,
                    }
                )

            rows: list[dict[str, Any]] = []
            for d in self.diff:
                row: dict[str, Any] = {"y": d.y}
                if d.by is not None:
                    if tidy:
                        row[d.by] = d.by_level
                    else:
                        row["by"] = d.by
                        row["by_level"] = d.by_level
                row.update({"diff": d.diff, "se": d.se, "lci": d.lci, "uci": d.uci})
                if self.stats is not None:
                    row["t"] = self.stats.t
                    row["df"] = float(self.stats.df)
                    row["p_value"] = self.stats.p_value
                rows.append(row)

            df = pl.DataFrame(rows)
            if not tidy and "by" in df.columns and df["by"].is_null().all():
                df = df.drop("by", "by_level")
            return df

        elif component == "estimates":
            if not self.estimates:
                return pl.DataFrame(
                    schema={
                        "y": pl.Utf8,
                        "est": pl.Float64,
                        "se": pl.Float64,
                        "cv": pl.Float64,
                        "lci": pl.Float64,
                        "uci": pl.Float64,
                    }
                )

            if not tidy:
                df = pl.DataFrame(msgspec.to_builtins(self.estimates))
                for col_pair in [("by", "by_level"), ("group", "group_level")]:
                    if col_pair[0] in df.columns and df[col_pair[0]].is_null().all():
                        df = df.drop(*col_pair)
                if "y_level" in df.columns and df["y_level"].is_null().all():
                    df = df.drop("y_level")
                return df

            # tidy=True: promote group/by/y to named columns
            rows = []
            for e in self.estimates:
                row: dict[str, Any] = {}
                if e.by is not None:
                    row[e.by] = e.by_level
                if e.group is not None:
                    row[e.group] = e.group_level
                if e.y_level is not None:
                    row[e.y] = e.y_level
                row.update(
                    {
                        "est": e.est,
                        "se": e.se,
                        "cv": e.cv,
                        "lci": e.lci,
                        "uci": e.uci,
                    }
                )
                rows.append(row)
            return pl.DataFrame(rows)

        else:
            raise ValueError(
                f"Invalid component '{component}'. Expected one of ['test', 'estimates']."
            )

    # ---------------- Presentation ----------------

    def _build_table(self):
        """Build the Rich table for estimates."""
        from rich import box
        from rich.table import Table as RTable

        ests = self.estimates or []

        # Detect which text columns have any content
        show_group = any(e.group is not None for e in ests)
        show_level = any(e.group_level is not None for e in ests)
        show_ylev = any(e.y_level is not None for e in ests)

        # Build estimates table with only-needed columns
        headers: list[str] = []
        if show_group:
            headers.append("Group")
        if show_level:
            headers.append("Level")
        if show_ylev:
            headers.append("y_level")
        headers += ["Estimate", "Std Err", "CV", "Lower", "Upper"]

        est_tbl = RTable(
            show_header=True,
            header_style="bold",
            box=box.SIMPLE_HEAVY,
            show_edge=True,
            show_lines=False,
            pad_edge=False,
            highlight=False,
            expand=False,
        )
        # Column configs: text cols no-wrap w/ ellipsis; numeric can fold
        for h in headers:
            is_text = h in {"Group", "Level", "y_level"}
            justify = "left" if is_text else "right"
            no_wrap = True if is_text else False
            overflow = "ellipsis" if is_text else "fold"
            est_tbl.add_column(h, justify=justify, no_wrap=no_wrap, overflow=overflow)

        for e in ests:
            row = []
            if show_group:
                row.append(str(e.group or ""))
            if show_level:
                row.append(str(e.group_level or ""))
            if show_ylev:
                row.append(str(e.y_level or ""))
            row += [
                _fmt_fixed(e.est, dec=4),
                _fmt_fixed(e.se, dec=4),
                _fmt_fixed(e.cv, dec=4),
                _fmt_fixed(e.lci, dec=4),
                _fmt_fixed(e.uci, dec=4),
            ]
            est_tbl.add_row(*row)

        return est_tbl, show_group, show_level, show_ylev

    def _build_stats_table(self):
        """Build the Rich table for test statistics."""
        from rich import box
        from rich.table import Table as RTable

        if self.stats is None:
            return None

        stats_tbl = RTable(
            show_header=True,
            header_style="bold",
            box=box.SIMPLE_HEAVY,
            show_edge=True,
            show_lines=False,
            pad_edge=False,
            highlight=False,
            expand=False,
        )
        for h in ["diff", "t", "df", "p_value"]:
            stats_tbl.add_column(h, justify="right", no_wrap=False, overflow="fold")

        # Get diff value from self.diff (first entry if available)
        st = self.stats
        diff_val = self.diff[0].diff if self.diff else None

        stats_tbl.add_row(
            _fmt_fixed(diff_val, dec=4) if diff_val is not None else "",
            _fmt_fixed(st.t, dec=4),
            _fmt_fixed(st.df, dec=4),
            _fmt_p(st.p_value),
        )
        return stats_tbl

    def __rich_console__(self, console, options):
        from rich.text import Text

        # Build header info
        header_lines = [
            Text(f"Y = {self.y!r}"),
            Text(f"H₀: μ = {_fmt_fixed(self.mean_h0, dec=4)}"),
            Text(""),
        ]

        # Build tables
        est_tbl, _, _, _ = self._build_table()
        children = header_lines + [est_tbl]

        # Stats table below
        stats_tbl = self._build_stats_table()
        if stats_tbl is not None:
            children.extend([Text(""), Text("Test statistic", style="bold"), stats_tbl])

        # Build title for panel
        title = "T-Test: [bold]One-sample[/bold]"

        # Use make_panel from central UI for consistent styling with box
        yield make_panel(children, title=title, obj=self, kind="ttest")

    def __plain_str__(self) -> str:
        """Plain-text fallback when rich is not installed. Never calls str(self)."""
        return _plain_one_sample(self)

    def __str__(self) -> str:
        _validate_one_sample(self)
        return render_rich_to_str(self, width=resolve_width(self))

    def show(self, *, use_rich: bool = True) -> None:
        from svy.ui.printing import rich_available

        _validate_one_sample(self)  # or _validate_two_sample
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


class TTestTwoGroups(msgspec.Struct, tag="two", tag_field="kind", kw_only=True, frozen=True):
    """Two-sample t-test comparing two levels of a grouping variable."""

    PRINT_WIDTH: ClassVar[int | None] = None

    y: str
    groups: GroupLevels
    paired: bool = False
    alternative: Literal["two-sided", "less", "greater"] = "two-sided"
    diff: list[DiffEst] = field(default_factory=list)
    estimates: list[TtestEst] = field(default_factory=list)
    stats: TTestStats | None = None
    alpha: float = 0.05

    # ---------------- Width Configuration ----------------

    @classmethod
    def set_default_print_width(cls, width: int | None) -> None:
        """
        Set the default print width for all TTestTwoGroups instances.

        Args:
            width: Print width in characters (must be > 20), or None to reset.

        Raises:
            TypeError: If width is not an int or None.
            ValueError: If width is <= 20.
        """
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

    # ---------------- Data Export ----------------

    def to_dict(self) -> dict[str, Any]:
        """
        Exports the entire result to a dictionary.

        Returns:
            A dictionary representation of all fields.
        """
        return msgspec.to_builtins(self)

    def to_polars(
        self,
        component: Literal["test", "estimates"] = "test",
        *,
        tidy: bool = True,
    ) -> pl.DataFrame:
        import polars as pl

        if component == "test":
            if not self.diff:
                return pl.DataFrame(
                    schema={
                        "y": pl.Utf8,
                        "group_var": pl.Utf8,
                        "paired": pl.Boolean,
                        "diff": pl.Float64,
                        "se": pl.Float64,
                        "lci": pl.Float64,
                        "uci": pl.Float64,
                        "t": pl.Float64,
                        "df": pl.Float64,
                        "p_value": pl.Float64,
                    }
                )

            rows: list[dict[str, Any]] = []
            for d in self.diff:
                row: dict[str, Any] = {"y": d.y}
                if d.by is not None:
                    if tidy:
                        row[d.by] = d.by_level
                    else:
                        row["by"] = d.by
                        row["by_level"] = d.by_level
                row.update(
                    {
                        "group_var": self.groups.var,
                        "paired": self.paired,
                        "diff": d.diff,
                        "se": d.se,
                        "lci": d.lci,
                        "uci": d.uci,
                    }
                )
                if self.stats is not None:
                    row["t"] = self.stats.t
                    row["df"] = float(self.stats.df)
                    row["p_value"] = self.stats.p_value
                rows.append(row)

            df = pl.DataFrame(rows)
            if not tidy and "by" in df.columns and df["by"].is_null().all():
                df = df.drop("by", "by_level")
            return df

        elif component == "estimates":
            if not self.estimates:
                return pl.DataFrame(
                    schema={
                        "y": pl.Utf8,
                        "group": pl.Utf8,
                        "group_level": pl.Utf8,
                        "est": pl.Float64,
                        "se": pl.Float64,
                        "cv": pl.Float64,
                        "lci": pl.Float64,
                        "uci": pl.Float64,
                    }
                )

            if not tidy:
                df = pl.DataFrame(msgspec.to_builtins(self.estimates))
                if "by" in df.columns and df["by"].is_null().all():
                    df = df.drop("by", "by_level")
                if "y_level" in df.columns and df["y_level"].is_null().all():
                    df = df.drop("y_level")
                return df

            # tidy=True: promote group/by/y to named columns
            rows = []
            for e in self.estimates:
                row: dict[str, Any] = {}
                if e.by is not None:
                    row[e.by] = e.by_level
                # group is always present for two-sample
                if e.group is not None:
                    row[e.group] = e.group_level
                if e.y_level is not None:
                    row[e.y] = e.y_level
                row.update(
                    {
                        "est": e.est,
                        "se": e.se,
                        "cv": e.cv,
                        "lci": e.lci,
                        "uci": e.uci,
                    }
                )
                rows.append(row)
            return pl.DataFrame(rows)

        else:
            raise ValueError(
                f"Invalid component '{component}'. Expected one of ['test', 'estimates']."
            )

    # ---------------- Presentation ----------------

    def _build_table(self):
        """Build the Rich table for estimates."""
        from rich import box
        from rich.table import Table as RTable

        _validate_two_sample(self)

        gvar = self.groups.var
        g1, g2 = self.groups.levels

        ests_sorted = list(_iter_estimates_sorted(self.estimates, gvar, (g1, g2)))
        show_group = any(e.group is not None for e in ests_sorted)
        show_level = any(e.group_level is not None for e in ests_sorted)
        show_ylev = any(e.y_level is not None for e in ests_sorted)

        headers: list[str] = []
        if show_group:
            headers.append("Group")
        if show_level:
            headers.append("Level")
        if show_ylev:
            headers.append("y_level")
        headers += ["Estimate", "Std Err", "CV", "Lower", "Upper"]

        est_tbl = RTable(
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
            is_text = h in {"Group", "Level", "y_level"}
            justify = "left" if is_text else "right"
            no_wrap = True if is_text else False
            overflow = "ellipsis" if is_text else "fold"
            est_tbl.add_column(h, justify=justify, no_wrap=no_wrap, overflow=overflow)

        for e in ests_sorted:
            row = []
            if show_group:
                row.append(str(e.group or ""))
            if show_level:
                row.append(str(e.group_level or ""))
            if show_ylev:
                row.append(str(e.y_level or ""))
            row += [
                _fmt_fixed(e.est, dec=4),
                _fmt_fixed(e.se, dec=4),
                _fmt_fixed(e.cv, dec=4),
                _fmt_fixed(e.lci, dec=4),
                _fmt_fixed(e.uci, dec=4),
            ]
            est_tbl.add_row(*row)

        return est_tbl, show_group, show_level, show_ylev

    def _build_stats_table(self):
        """Build the Rich table for test statistics."""
        from rich import box
        from rich.table import Table as RTable

        if self.stats is None:
            return None

        stats_tbl = RTable(
            show_header=True,
            header_style="bold",
            box=box.SIMPLE_HEAVY,
            show_edge=True,
            show_lines=False,
            pad_edge=False,
            highlight=False,
            expand=False,
        )
        for h in ["diff", "t", "df", "p_value"]:
            stats_tbl.add_column(h, justify="right", no_wrap=False, overflow="fold")

        # Get diff value from self.diff (first entry if available)
        diff_val = self.diff[0].diff if self.diff else None

        stats_tbl.add_row(
            _fmt_fixed(diff_val, dec=4) if diff_val is not None else "",
            _fmt_fixed(self.stats.t, dec=4),
            _fmt_fixed(self.stats.df, dec=4),
            _fmt_p(self.stats.p_value),
        )
        return stats_tbl

    def __rich_console__(self, console, options):
        from rich.text import Text

        _validate_two_sample(self)

        gvar = self.groups.var
        g1, g2 = self.groups.levels
        pair_txt = "paired" if self.paired else "unpaired"

        # Build header info
        header_lines = [
            Text(f"Y = {self.y!r}"),
            Text(f"Groups: {gvar} = [{g1!r} vs {g2!r}]"),
            Text(""),
        ]

        # Build tables
        est_tbl, _, _, _ = self._build_table()
        children = header_lines + [est_tbl]

        # Stats table below
        stats_tbl = self._build_stats_table()
        if stats_tbl is not None:
            children.extend([Text(""), Text("Test statistic", style="bold"), stats_tbl])

        # Build title for panel
        title = f"T-Test: [bold]Two-sample ({pair_txt})[/bold]"

        # Use make_panel from central UI for consistent styling with box
        yield make_panel(children, title=title, obj=self, kind="ttest")

    def __plain_str__(self) -> str:
        """Plain-text fallback when rich is not installed. Never calls str(self)."""
        return _plain_two_sample(self)

    def __str__(self) -> str:
        _validate_two_sample(self)
        return render_rich_to_str(self, width=resolve_width(self))

    def show(self, *, use_rich: bool = True) -> None:
        from svy.ui.printing import rich_available

        _validate_two_sample(self)  # was wrongly calling _validate_one_sample
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


def _where_arg_to_str(where) -> str | None:
    """Convert a WhereArg to a display string. Delegates to printing.format_where_clause."""
    return format_where_clause(where)


# =============================================================================
# By-result container (returned when by= is set)
# =============================================================================


class TTestByResult:
    """
    Container for ttest results when ``by`` is specified.

    Wraps a list of per-level results and renders them as a single
    panel with a shared header and per-level sections.

    Supports iteration, indexing, and len() so existing code that
    does ``for r in results:`` or ``results[0]`` keeps working.
    """

    PRINT_WIDTH: ClassVar[int | None] = None

    def __init__(
        self,
        results: list[TTestOneGroup | TTestTwoGroups],
        *,
        by: str,
        y: str,
        mean_h0: Number | None = None,
        groups: GroupLevels | None = None,
        alpha: float = 0.05,
        where_clause: str | None = None,
    ) -> None:
        self.results = results
        self.by = by
        self.y = y
        self.mean_h0 = mean_h0
        self.groups = groups
        self.alpha = alpha
        self.where_clause = where_clause
        # Normalised list of by variable names for multi-by display
        self._by_list: list[str] = list(by) if isinstance(by, (list, tuple)) else [by]

    # ── Sequence protocol ────────────────────────────────────────────────

    def __iter__(self):
        return iter(self.results)

    def __len__(self):
        return len(self.results)

    def __getitem__(self, i):
        return self.results[i]

    # ── Helpers ──────────────────────────────────────────────────────────

    def _format_level(self, by_level) -> str:
        """Format a by-level value for display.

        For multi-by, levels are stored as 'v1__by__v2__by__v3'.
        Splits on '__by__' and zips with by variable names.
        """
        _BY_SEP = "__by__"
        level_str = str(by_level)
        parts = level_str.split(_BY_SEP)
        if len(parts) == len(self._by_list) and len(parts) > 1:
            return ", ".join(f"{var} = {val}" for var, val in zip(self._by_list, parts))
        # Single by or unable to split — use var = level format
        by_name = ", ".join(self._by_list)
        return f"{by_name} = {level_str}"

    # ── Width configuration ──────────────────────────────────────────────

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

    # ── Export ───────────────────────────────────────────────────────────

    def to_polars(
        self,
        component: Literal["test", "estimates"] = "test",
        *,
        tidy: bool = True,
    ) -> "pl.DataFrame":
        import polars as pl

        frames = [r.to_polars(component, tidy=tidy) for r in self.results]
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="diagonal")

    # ── Printing ─────────────────────────────────────────────────────────

    def __rich_console__(self, console, options):
        from rich.rule import Rule
        from rich.text import Text

        is_one_sample = self.mean_h0 is not None

        # Shared header
        header_lines: list = [Text(f"Y = {self.y!r}")]
        if is_one_sample:
            header_lines.append(Text(f"H₀: μ = {_fmt_fixed(self.mean_h0, dec=4)}"))
        else:
            if self.groups:
                g1, g2 = self.groups.levels
                header_lines.append(Text(f"Groups: {self.groups.var} = [{g1!r} vs {g2!r}]"))
        header_lines.append(Text(f"By: {', '.join(self._by_list)}"))
        if self.where_clause:
            where_text = Text()
            where_text.append("where: ", style="dim")
            where_text.append(self.where_clause)
            header_lines.append(where_text)
        header_lines.append(Text(""))

        children: list = list(header_lines)

        for r in self.results:
            # by-level label from first diff entry
            by_level = r.diff[0].by_level if r.diff else "?"
            children.append(Rule(title=self._format_level(by_level), style="dim"))
            children.append(Text(""))

            # estimates table
            est_tbl, _, _, _ = r._build_table()
            children.append(est_tbl)

            # stats table
            stats_tbl = r._build_stats_table()
            if stats_tbl is not None:
                children.extend([Text(""), Text("Test statistic", style="bold"), stats_tbl])
            children.append(Text(""))

        if is_one_sample:
            title = "T-Test: [bold]One-sample[/bold]"
        else:
            pair_txt = (
                "paired"
                if (self.results and getattr(self.results[0], "paired", False))
                else "unpaired"
            )
            title = f"T-Test: [bold]Two-sample ({pair_txt})[/bold]"

        yield make_panel(children, title=title, obj=self, kind="ttest")

    def __plain_str__(self) -> str:
        is_one_sample = self.mean_h0 is not None
        by_display = ", ".join(self._by_list)
        if is_one_sample:
            lines = [
                "T-Test: One-sample",
                f"  Y = {self.y!r}",
                f"  H₀: μ = {_fmt_fixed(self.mean_h0, dec=4)}",
                f"  By: {by_display}",
            ]
        else:
            if self.groups:
                g1, g2 = self.groups.levels
                pair_txt = (
                    "paired"
                    if (self.results and getattr(self.results[0], "paired", False))
                    else "unpaired"
                )
                lines = [
                    f"T-Test: Two-sample ({pair_txt})",
                    f"  Y = {self.y!r}",
                    f"  Groups: {self.groups.var} = [{g1!r} vs {g2!r}]",
                    f"  By: {by_display}",
                ]
            else:
                lines = [f"T-Test: y='{self.y}' · By: {by_display}"]
        if self.where_clause:
            lines.append(f"  where: {self.where_clause}")
        lines.append("")
        for r in self.results:
            by_level = r.diff[0].by_level if r.diff else "?"
            lines.append(f"── {self._format_level(by_level)} " + "─" * 40)
            lines.append(_plain_body_only(r))
            lines.append("")
        return "\n".join(lines)

    def __str__(self) -> str:
        return render_rich_to_str(self, width=resolve_width(self))

    def __repr__(self) -> str:
        return f"TTestByResult(y={self.y!r}, by={', '.join(self._by_list)!r}, n_levels={len(self.results)})"

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


# =============================================================================
# Convenience exports (records / Polars / Markdown)
# =============================================================================


def ttest_to_records(
    tt: TTestOneGroup | TTestTwoGroups, *, include_meta: bool = True
) -> list[dict]:
    out: list[dict] = []
    for e in tt.estimates or []:
        rec = {
            "y": e.y,
            "y_level": e.y_level,
            "group": e.group,
            "group_level": e.group_level,
            "by": e.by,
            "by_level": e.by_level,
            "est": e.est,
            "se": e.se,
            "cv": e.cv,
            "lci": e.lci,
            "uci": e.uci,
        }
        if include_meta:
            kind = "one" if isinstance(tt, TTestOneGroup) else "two"
            rec.update({"kind": kind, "alpha": tt.alpha})
            if isinstance(tt, TTestOneGroup):
                rec["mean_h0"] = tt.mean_h0
            else:
                rec["paired"] = tt.paired
                rec["group_var"] = tt.groups.var
        out.append(rec)
    return out


def ttest_to_polars(tt: TTestOneGroup | TTestTwoGroups):
    import polars as pl

    return pl.DataFrame(ttest_to_records(tt, include_meta=True))


def ttest_to_markdown(tt: TTestOneGroup | TTestTwoGroups, *, dec: int = 4) -> str:
    """Simple Markdown table of estimates; stats summarized below."""
    rows = list(_rows_for_plain(tt, dec=dec))
    headers = ["Group", "Level", "y_level", "Estimate", "Std Err", "CV", "Lower", "Upper"]
    md = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        md.append("| " + " | ".join(r) + " |")

    s = _stats_summary_line(tt)
    if s:
        md.append("")
        md.append(s)
    return "\n".join(md)


# =============================================================================
# Plain fallback rendering
# =============================================================================


def _rows_for_plain(tt: TTestOneGroup | TTestTwoGroups, *, dec: int = 4) -> Iterator[list[str]]:
    def fmt_e(e: TtestEst) -> list[str]:
        return [
            str(e.group or ""),
            str(e.group_level or ""),
            str(e.y_level or ""),
            _fmt_fixed(e.est, dec=dec),
            _fmt_fixed(e.se, dec=dec),
            _fmt_fixed(e.cv, dec=dec),
            _fmt_fixed(e.lci, dec=dec),
            _fmt_fixed(e.uci, dec=dec),
        ]

    if isinstance(tt, TTestOneGroup):
        for e in tt.estimates or []:
            yield fmt_e(e)
    else:
        gvar = tt.groups.var
        order = tt.groups.levels
        for e in _iter_estimates_sorted(tt.estimates, gvar, order):
            yield fmt_e(e)


def _plain_estimates_table(tt: TTestOneGroup | TTestTwoGroups) -> str:
    """Render just the estimates as a compact plain-text table."""
    estimates = tt.estimates or []
    is_two = isinstance(tt, TTestTwoGroups)

    headers: list[str] = []
    if is_two:
        headers.append(tt.groups.var)
    headers += ["Estimate", "Std Err", "CV", "Lower", "Upper"]

    rows: list[list[str]] = []
    for e in estimates:
        row: list[str] = []
        if is_two:
            row.append(str(e.group_level) if e.group_level is not None else "")
        row += [
            _fmt_fixed(e.est, dec=4),
            _fmt_fixed(e.se, dec=4),
            f"{e.cv:.4f}" if e.cv is not None else "",
            _fmt_fixed(e.lci, dec=4),
            _fmt_fixed(e.uci, dec=4),
        ]
        rows.append(row)

    return render_plain_table(headers, rows)


def _plain_stats_line(tt: TTestOneGroup | TTestTwoGroups) -> str | None:
    if tt.stats is None:
        return None
    st = tt.stats
    if isinstance(tt, TTestTwoGroups) and tt.diff:
        diff_val = tt.diff[0].diff if tt.diff else None
        diff_str = f"diff={_fmt_fixed(diff_val, dec=4)}, " if diff_val is not None else ""
    else:
        diff_str = ""
    return (
        f"{diff_str}t={_fmt_fixed(st.t, dec=4)}, "
        f"df={_fmt_fixed(st.df, dec=4)}, "
        f"p_value={_fmt_p(st.p_value)} ({tt.alternative})"
    )


def _plain_one_sample(tt: TTestOneGroup) -> str:
    lines = [
        "T-Test: One-sample",
        f"  Y = {tt.y!r}",
        f"  H\u2080: \u03bc = {_fmt_fixed(tt.mean_h0, dec=4)}",
        "",
    ]
    lines.append(_plain_estimates_table(tt))
    stats = _plain_stats_line(tt)
    if stats:
        lines += ["", "  Test statistic", f"  {stats}"]
    return "\n".join(lines)


def _plain_two_sample(tt: TTestTwoGroups) -> str:
    g1, g2 = tt.groups.levels
    pair_txt = "paired" if tt.paired else "unpaired"
    lines = [
        f"T-Test: Two-sample ({pair_txt})",
        f"  Y = {tt.y!r}",
        f"  Groups: {tt.groups.var} = [{g1!r} vs {g2!r}]",
        "",
    ]
    lines.append(_plain_estimates_table(tt))
    stats = _plain_stats_line(tt)
    if stats:
        lines += ["", "  Test statistic", f"  {stats}"]
    return "\n".join(lines)


def _plain_body_only(tt: TTestOneGroup | TTestTwoGroups) -> str:
    """Render just estimates + stats — no title. Used by TTestByResult.__plain_str__."""
    lines = [_plain_estimates_table(tt)]
    stats = _plain_stats_line(tt)
    if stats:
        lines += ["", "  Test statistic", f"  {stats}"]
    return "\n".join(lines)


# =============================================================================
# Small utilities
# =============================================================================


def _fmt_p(p: float, *, small: float = 1e-4) -> str:
    try:
        if p < small:
            return f"<{small:g}"
        return _fmt_fixed(p, dec=4)
    except Exception:
        return str(p)


def _iter_estimates_sorted(
    ests: Sequence[TtestEst] | None, group_var: str, order: tuple[Category, Category]
) -> Iterator[TtestEst]:
    """Yield estimates in a consistent (group level) order, then by y_level (natural sort)."""
    if not ests:
        return iter(())
    g1, g2 = order
    rest: list[TtestEst] = []
    first: list[TtestEst] = []
    second: list[TtestEst] = []
    for e in ests:
        if e.group == group_var and e.group_level == g1:
            first.append(e)
        elif e.group == group_var and e.group_level == g2:
            second.append(e)
        else:
            rest.append(e)

    def key(e):
        return (
            natural_sort_key(e.y_level),
            natural_sort_key(e.group),
            natural_sort_key(e.group_level),
        )

    for e in sorted(first, key=key):
        yield e
    for e in sorted(second, key=key):
        yield e
    for e in sorted(rest, key=key):
        yield e
