# src/svy/categorical/ranktest.py
"""
Design-based rank tests for complex survey data.

Implements the methodology of Lumley & Scott (2013):
    "Two-sample rank tests under complex sampling." Biometrika, 100(4), 831-842.

Provides survey-weighted versions of:
    - Wilcoxon rank-sum test (two-sample, auto-detected)
    - Kruskal-Wallis test (k-sample, auto-detected)
    - Van der Waerden normal-scores test
    - Mood's median test
    - Custom rank-score functions via ``score_fn``
"""

from __future__ import annotations

import logging

from typing import TYPE_CHECKING, Any, ClassVar, Literal

import msgspec

from msgspec import field

from svy.categorical.ttest import DiffEst, GroupLevels, TtestEst
from svy.core.containers import FDist, TDist
from svy.core.types import Category
from svy.ui.printing import make_panel, natural_sort_key, render_rich_to_str, resolve_width
from svy.utils.formats import _fmt_fixed


if TYPE_CHECKING:
    import polars as pl

log = logging.getLogger(__name__)


# =============================================================================
# Validation helpers
# =============================================================================


def _validate_two_sample(rt: RankTestTwoSample) -> None:
    from svy.errors import MethodError

    if not (0.0 < float(rt.alpha) < 1.0):
        raise MethodError.invalid_range(
            where="RankTestTwoSample", param="alpha", got=rt.alpha, min_=0.0, max_=1.0
        )
    if not rt.groups or not isinstance(rt.groups.levels, tuple) or len(rt.groups.levels) != 2:
        raise MethodError.invalid_choice(
            where="RankTestTwoSample",
            param="groups.levels",
            got=getattr(rt.groups, "levels", None),
            allowed="tuple of length 2",
            hint="Provide exactly two group levels for a two-sample rank test.",
        )
    if not rt.estimates:
        log.warning(
            "RankTestTwoSample has no per-group estimates; the estimates table will be empty."
        )


def _validate_k_sample(rt: RankTestKSample) -> None:
    from svy.errors import DimensionError, MethodError

    if not (0.0 < float(rt.alpha) < 1.0):
        raise MethodError.invalid_range(
            where="RankTestKSample", param="alpha", got=rt.alpha, min_=0.0, max_=1.0
        )
    if len(rt.group_levels) < 2:
        raise DimensionError.empty_estimates(where="RankTestKSample", param="group_levels")


# =============================================================================
# Result containers
# =============================================================================


class RankTestTwoSample(
    msgspec.Struct, tag="rank_two", tag_field="kind", kw_only=True, frozen=True
):
    """Two-sample design-based rank test (Wilcoxon-style).

    Returned when the grouping variable has exactly two levels.
    Uses a t-distribution reference with design degrees of freedom.
    """

    PRINT_WIDTH: ClassVar[int | None] = None

    y: str
    groups: GroupLevels
    method_name: str
    alternative: Literal["two-sided", "less", "greater"] = "two-sided"
    diff: list[DiffEst] = field(default_factory=list)
    estimates: list[TtestEst] = field(default_factory=list)
    stats: TDist | None = None
    alpha: float = 0.05

    # ---------------- Width Configuration ----------------

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

    # ---------------- Data Export ----------------

    def to_dict(self) -> dict[str, Any]:
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
                        "method": pl.Utf8,
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
                        "method": self.method_name,
                        "diff": d.diff,
                        "se": d.se,
                        "lci": d.lci,
                        "uci": d.uci,
                    }
                )
                if self.stats is not None:
                    row["t"] = self.stats.value
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
                for col_pair in [("by", "by_level"), ("group", "group_level")]:
                    if col_pair[0] in df.columns and df[col_pair[0]].is_null().all():
                        df = df.drop(*col_pair)
                if "y_level" in df.columns and df["y_level"].is_null().all():
                    df = df.drop("y_level")
                return df

            rows = []
            for e in self.estimates:
                row: dict[str, Any] = {}
                if e.by is not None:
                    row[e.by] = e.by_level
                if e.group is not None:
                    row[e.group] = e.group_level
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
        """Build the Rich table for per-group rank score estimates."""
        from rich import box
        from rich.table import Table as RTable

        ests = self.estimates or []

        show_group = any(e.group is not None for e in ests)
        show_level = any(e.group_level is not None for e in ests)

        headers: list[str] = []
        if show_group:
            headers.append("Group")
        if show_level:
            headers.append("Level")
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
            is_text = h in {"Group", "Level"}
            justify = "left" if is_text else "right"
            no_wrap = True if is_text else False
            overflow = "ellipsis" if is_text else "fold"
            est_tbl.add_column(h, justify=justify, no_wrap=no_wrap, overflow=overflow)

        sorted_ests = sorted(ests, key=lambda e: natural_sort_key(str(e.group_level or "")))
        for e in sorted_ests:
            row = []
            if show_group:
                row.append(str(e.group or ""))
            if show_level:
                row.append(str(e.group_level or ""))
            row += [
                _fmt_fixed(e.est, dec=4),
                _fmt_fixed(e.se, dec=4),
                _fmt_fixed(e.cv, dec=4),
                _fmt_fixed(e.lci, dec=4),
                _fmt_fixed(e.uci, dec=4),
            ]
            est_tbl.add_row(*row)

        return est_tbl, show_group, show_level

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

        diff_val = self.diff[0].diff if self.diff else None

        stats_tbl.add_row(
            _fmt_fixed(diff_val, dec=4) if diff_val is not None else "",
            _fmt_fixed(self.stats.value, dec=4),
            _fmt_fixed(self.stats.df, dec=4),
            _fmt_p(self.stats.p_value),
        )
        return stats_tbl

    def __rich_console__(self, console, options):
        from rich.text import Text

        _validate_two_sample(self)

        gvar = self.groups.var
        g1, g2 = self.groups.levels

        header_lines = [
            Text(f"Y = {self.y!r}"),
            Text(f"Groups: {gvar} = [{g1!r} vs {g2!r}]"),
            Text(""),
        ]

        est_tbl, _, _ = self._build_table()
        children = header_lines + [est_tbl]

        stats_tbl = self._build_stats_table()
        if stats_tbl is not None:
            children.extend([Text(""), Text("Test statistic", style="bold"), stats_tbl])

        title = f"Rank Test: [bold]Two-sample ({self.method_name})[/bold]"
        yield make_panel(children, title=title, obj=self, kind="ranktest")

    def __plain_str__(self) -> str:
        return _plain_two_sample(self)

    def __str__(self) -> str:
        _validate_two_sample(self)
        return render_rich_to_str(self, width=resolve_width(self))

    def show(self, *, use_rich: bool = True) -> None:
        from svy.ui.printing import rich_available

        _validate_two_sample(self)
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


class RankTestKSample(msgspec.Struct, tag="rank_k", tag_field="kind", kw_only=True, frozen=True):
    """K-sample design-based rank test (Kruskal-Wallis-style).

    Returned when the grouping variable has three or more levels.
    Uses an F-distribution reference with (k-1, ddf) degrees of freedom.
    """

    PRINT_WIDTH: ClassVar[int | None] = None

    y: str
    group_var: str
    group_levels: list[Category]
    method_name: str
    estimates: list[TtestEst] = field(default_factory=list)
    stats: FDist | None = None
    alpha: float = 0.05

    # ---------------- Width Configuration ----------------

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

    # ---------------- Data Export ----------------

    def to_dict(self) -> dict[str, Any]:
        return msgspec.to_builtins(self)

    def to_polars(
        self,
        component: Literal["test", "estimates"] = "test",
        *,
        tidy: bool = True,
    ) -> pl.DataFrame:
        import polars as pl

        if component == "test":
            if self.stats is None:
                return pl.DataFrame(
                    schema={
                        "y": pl.Utf8,
                        "group_var": pl.Utf8,
                        "method": pl.Utf8,
                        "ndf": pl.Int64,
                        "ddf": pl.Float64,
                        "chisq": pl.Float64,
                        "f_stat": pl.Float64,
                        "p_value": pl.Float64,
                    }
                )

            ndf = int(self.stats.df_num)
            return pl.DataFrame(
                {
                    "y": [self.y],
                    "group_var": [self.group_var],
                    "method": [self.method_name],
                    "ndf": [ndf],
                    "ddf": [float(self.stats.df_den)],
                    "chisq": [float(self.stats.value) * ndf],  # F * ndf = chisq
                    "f_stat": [float(self.stats.value)],
                    "p_value": [self.stats.p_value],
                }
            )

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
                for col_pair in [("by", "by_level"), ("group", "group_level")]:
                    if col_pair[0] in df.columns and df[col_pair[0]].is_null().all():
                        df = df.drop(*col_pair)
                if "y_level" in df.columns and df["y_level"].is_null().all():
                    df = df.drop("y_level")
                return df

            rows = []
            for e in self.estimates:
                row: dict[str, Any] = {}
                if e.by is not None:
                    row[e.by] = e.by_level
                if e.group is not None:
                    row[e.group] = e.group_level
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
        """Build the Rich table for per-group rank score estimates."""
        from rich import box
        from rich.table import Table as RTable

        ests = self.estimates or []

        show_group = any(e.group is not None for e in ests)
        show_level = any(e.group_level is not None for e in ests)

        headers: list[str] = []
        if show_group:
            headers.append("Group")
        if show_level:
            headers.append("Level")
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
            is_text = h in {"Group", "Level"}
            justify = "left" if is_text else "right"
            no_wrap = True if is_text else False
            overflow = "ellipsis" if is_text else "fold"
            est_tbl.add_column(h, justify=justify, no_wrap=no_wrap, overflow=overflow)

        sorted_ests = sorted(ests, key=lambda e: natural_sort_key(str(e.group_level or "")))
        for e in sorted_ests:
            row = []
            if show_group:
                row.append(str(e.group or ""))
            if show_level:
                row.append(str(e.group_level or ""))
            row += [
                _fmt_fixed(e.est, dec=4),
                _fmt_fixed(e.se, dec=4),
                _fmt_fixed(e.cv, dec=4),
                _fmt_fixed(e.lci, dec=4),
                _fmt_fixed(e.uci, dec=4),
            ]
            est_tbl.add_row(*row)

        return est_tbl, show_group, show_level

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
        for h in ["df", "Chisq", "F", "p_value"]:
            stats_tbl.add_column(h, justify="right", no_wrap=False, overflow="fold")

        ndf = int(self.stats.df_num)
        chisq = float(self.stats.value) * ndf

        stats_tbl.add_row(
            str(ndf),
            _fmt_fixed(chisq, dec=4),
            _fmt_fixed(self.stats.value, dec=4),
            _fmt_p(self.stats.p_value),
        )
        return stats_tbl

    def __rich_console__(self, console, options):
        from rich.text import Text

        _validate_k_sample(self)

        header_lines = [
            Text(f"Y = {self.y!r}"),
            Text(f"Groups: {self.group_var} ({len(self.group_levels)} levels)"),
            Text(""),
        ]

        est_tbl, _, _ = self._build_table()
        children = header_lines + [est_tbl]

        stats_tbl = self._build_stats_table()
        if stats_tbl is not None:
            children.extend([Text(""), Text("Test statistic", style="bold"), stats_tbl])

        title = f"Rank Test: [bold]K-sample ({self.method_name})[/bold]"
        yield make_panel(children, title=title, obj=self, kind="ranktest")

    def __plain_str__(self) -> str:
        return _plain_k_sample(self)

    def __str__(self) -> str:
        _validate_k_sample(self)
        return render_rich_to_str(self, width=resolve_width(self))

    def show(self, *, use_rich: bool = True) -> None:
        from svy.ui.printing import rich_available

        _validate_k_sample(self)
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
# By-result container (returned when by= is set)
# =============================================================================


class RankTestByResult:
    """
    Container for ranktest results when ``by`` is specified.

    Wraps a list of per-level results and renders them as a single
    panel with a shared header and per-level sections.

    Supports iteration, indexing, and len() so existing code that
    does ``for r in results:`` or ``results[0]`` keeps working.
    """

    PRINT_WIDTH: ClassVar[int | None] = None

    def __init__(
        self,
        results: list[RankTestTwoSample | RankTestKSample],
        *,
        by: str | list[str],
        y: str,
        group_var: str,
        method_name: str,
        groups: GroupLevels | None = None,  # two-sample only
        alpha: float = 0.05,
        where_clause: str | None = None,
        by_levels: list | None = None,
    ) -> None:
        self.results = results
        self.by = by
        self.y = y
        self.group_var = group_var
        self.method_name = method_name
        self.groups = groups
        self.alpha = alpha
        self.where_clause = where_clause
        self.by_levels: list = by_levels or []
        self._by_list: list[str] = list(by) if isinstance(by, (list, tuple)) else [by]

    # ── Sequence protocol ────────────────────────────────────────────────

    def __iter__(self):
        return iter(self.results)

    def __len__(self):
        return len(self.results)

    def __getitem__(self, i):
        return self.results[i]

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

    # ── Helpers ──────────────────────────────────────────────────────────

    def _format_level(self, by_level) -> str:
        """Format a by-level value for display, handling multi-by."""
        _BY_SEP = "__by__"
        level_str = str(by_level)
        parts = level_str.split(_BY_SEP)
        if len(parts) == len(self._by_list) and len(parts) > 1:
            return ", ".join(f"{var} = {val}" for var, val in zip(self._by_list, parts))
        by_name = ", ".join(self._by_list)
        return f"{by_name} = {level_str}"

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

        is_two_sample = self.groups is not None

        # Shared header
        header_lines: list = [Text(f"Y = {self.y!r}")]
        if is_two_sample:
            g1, g2 = self.groups.levels
            header_lines.append(Text(f"Groups: {self.groups.var} = [{g1!r} vs {g2!r}]"))
        else:
            header_lines.append(Text(f"Groups: {self.group_var}"))
        header_lines.append(Text(f"By: {', '.join(self._by_list)}"))
        if self.where_clause:
            where_text = Text()
            where_text.append("where: ", style="dim")
            where_text.append(self.where_clause)
            header_lines.append(where_text)
        header_lines.append(Text(""))

        children: list = list(header_lines)

        for i, r in enumerate(self.results):
            # by-level: prefer stored by_levels, fall back to diff for two-sample
            if self.by_levels and i < len(self.by_levels):
                by_level = self.by_levels[i]
            elif isinstance(r, RankTestTwoSample) and r.diff:
                by_level = r.diff[0].by_level
            else:
                by_level = "?"
            children.append(Rule(title=self._format_level(by_level), style="dim"))
            children.append(Text(""))

            est_tbl, _, _ = r._build_table()
            children.append(est_tbl)

            stats_tbl = r._build_stats_table()
            if stats_tbl is not None:
                children.extend([Text(""), Text("Test statistic", style="bold"), stats_tbl])
            children.append(Text(""))

        title = f"Rank Test: [bold]{self.method_name}[/bold]"
        yield make_panel(children, title=title, obj=self, kind="ranktest")

    def __plain_str__(self) -> str:
        by_display = ", ".join(self._by_list)
        is_two_sample = self.groups is not None
        if is_two_sample:
            g1, g2 = self.groups.levels
            pair_txt = (
                "paired"
                if (self.results and getattr(self.results[0], "paired", False))
                else "unpaired"
            )
            lines = [
                f"Rank Test: Two-sample ({self.method_name}, {pair_txt})",
                f"  Y = {self.y!r}",
                f"  Groups: {self.groups.var} = [{g1!r} vs {g2!r}]",
                f"  By: {by_display}",
            ]
        else:
            lines = [
                f"Rank Test: K-sample ({self.method_name})",
                f"  Y = {self.y!r}",
                f"  Groups: {self.group_var}",
                f"  By: {by_display}",
            ]
        if self.where_clause:
            lines.append(f"  where: {self.where_clause}")
        lines.append("")
        for i, r in enumerate(self.results):
            if self.by_levels and i < len(self.by_levels):
                by_level = self.by_levels[i]
            elif isinstance(r, RankTestTwoSample) and r.diff:
                by_level = r.diff[0].by_level
            else:
                by_level = "?"
            lines.append(f"── {self._format_level(by_level)} " + "─" * 40)
            body = (
                _plain_two_sample_body(r)
                if isinstance(r, RankTestTwoSample)
                else _plain_k_sample_body(r)
            )
            if body:
                lines += ["  Test statistic", body]
            lines.append("")
        return "\n".join(lines)

    def __str__(self) -> str:
        return render_rich_to_str(self, width=resolve_width(self))

    def __repr__(self) -> str:
        return (
            f"RankTestByResult(y={self.y!r}, by={', '.join(self._by_list)!r}, "
            f"n_levels={len(self.results)})"
        )

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
# Plain fallback rendering
# =============================================================================


def _fmt_p(p: float, *, small: float = 1e-4) -> str:
    try:
        if p < small:
            return f"<{small:g}"
        return _fmt_fixed(p, dec=4)
    except Exception:
        return str(p)


def _plain_two_sample(rt: RankTestTwoSample) -> str:
    g1, g2 = rt.groups.levels
    body_lines = [
        f"Rank Test: Two-sample ({rt.method_name})",
        f"  Y = {rt.y!r}",
        f"  Groups: {rt.groups.var} = [{g1!r} vs {g2!r}]",
        "",
    ]
    body = _plain_two_sample_body(rt)
    if body:
        body_lines += ["  Test statistic", body]
    return "\n".join(body_lines)


def _plain_k_sample(rt: RankTestKSample) -> str:
    body_lines = [
        f"Rank Test: K-sample ({rt.method_name})",
        f"  Y = {rt.y!r}",
        f"  Groups: {rt.group_var} ({len(rt.group_levels)} levels)",
        "",
    ]
    body = _plain_k_sample_body(rt)
    if body:
        body_lines += ["  Test statistic", body]
    return "\n".join(body_lines)


def _plain_two_sample_body(rt: RankTestTwoSample) -> str:
    """Stats only — no title. Used by RankTestByResult.__plain_str__."""
    if rt.stats is None:
        return ""
    diff_val = rt.diff[0].diff if rt.diff else None
    diff_str = _fmt_fixed(diff_val, dec=4) if diff_val is not None else "N/A"
    return (
        f"  diff={diff_str}, t={_fmt_fixed(rt.stats.value, dec=4)}, "
        f"df={_fmt_fixed(rt.stats.df, dec=4)}, "
        f"p_value={_fmt_p(rt.stats.p_value)} ({rt.alternative})"
    )


def _plain_k_sample_body(rt: RankTestKSample) -> str:
    """Stats only — no title. Used by RankTestByResult.__plain_str__."""
    if rt.stats is None:
        return ""
    ndf = int(rt.stats.df_num)
    chisq = float(rt.stats.value) * ndf
    return (
        f"  df={ndf}, Chisq={_fmt_fixed(chisq, dec=4)}, "
        f"F={_fmt_fixed(rt.stats.value, dec=4)}, "
        f"p_value={_fmt_p(rt.stats.p_value)}"
    )
