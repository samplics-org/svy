# src/svy/core/describe.py
from __future__ import annotations

import datetime as dt
import logging

from typing import Any, TypeAlias

import msgspec

from .enumerations import MeasurementType


log = logging.getLogger(__name__)

# ============ central UI hooks (optional) ============


def _ui_module():
    """Try to import central UI module (optional)."""
    try:
        from svy.ui import printing as _p

        return _p
    except Exception:
        return None


def _resolve_width_for(obj, default: int = 88) -> int:
    """
    Resolve preferred print width using central UI first, then a robust fallback chain.
    Mirrors Table._resolve_width_for so PRINT_WIDTH & _print_width work.
    """
    P = _ui_module()
    if P:
        for call in (
            lambda: P.resolve_width(obj=obj, default=default),
            lambda: P.resolve_width(default=default),
            lambda: P.resolve_width(),
        ):
            try:
                w = int(call())
                if w > 20:
                    return w
            except Exception:
                pass

    import builtins
    import os
    import shutil
    import sys

    def _as_int(x):
        try:
            return int(x)
        except Exception:
            return None

    w = _as_int(getattr(obj, "_print_width", None))
    if isinstance(w, int) and w > 20:
        return w

    w = _as_int(getattr(type(obj), "PRINT_WIDTH", None))
    if isinstance(w, int) and w > 20:
        return w

    try:
        mod = sys.modules.get(obj.__class__.__module__)
        if mod is not None and hasattr(mod, "SVY_PRINT_WIDTH"):
            mw = _as_int(getattr(mod, "SVY_PRINT_WIDTH"))
            if isinstance(mw, int) and mw > 20:
                return mw
    except Exception:
        pass

    main_mod = sys.modules.get("__main__")
    if main_mod is not None and hasattr(main_mod, "SVY_PRINT_WIDTH"):
        mw = _as_int(getattr(main_mod, "SVY_PRINT_WIDTH"))
        if isinstance(mw, int) and mw > 20:
            return mw

    bw = _as_int(getattr(builtins, "SVY_PRINT_WIDTH", None))
    if isinstance(bw, int) and bw > 20:
        return bw

    ew = _as_int(os.environ.get("SVY_PRINT_WIDTH"))
    if isinstance(ew, int) and ew > 20:
        return ew

    try:
        ts = shutil.get_terminal_size(fallback=(default, 24))
        if isinstance(ts.columns, int) and ts.columns > 20:
            return ts.columns
    except Exception:
        pass

    return default


# ------------- small building blocks -------------


class Quantile(msgspec.Struct, frozen=True):
    """One quantile point."""

    p: float  # in [0.0, 1.0], e.g., 0.5 for median
    value: float | None  # None if undefined (e.g., empty)


class Freq(msgspec.Struct, frozen=True):
    """One row of a frequency table; order matters (usually sorted by freq desc)."""

    level: Any
    count: float  # weighted or unweighted count
    prop: float  # proportion in [0,1]


# ------------- column-level payloads -------------


class DescribeBase(msgspec.Struct, frozen=True):
    name: str
    mtype: MeasurementType
    n: int  # total rows seen
    n_missing: int  # rows treated as missing
    weighted: bool
    drop_nulls: bool


class DescribeContinuous(DescribeBase, frozen=True):
    # numeric continuous stats
    mean: float | None = None
    std: float | None = None
    var: float | None = None
    min: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    max: float | None = None
    sum: float | None = None
    percentiles: tuple[Quantile, ...] = ()


class DescribeDiscrete(DescribeBase, frozen=True):
    # numeric discrete → same summary as continuous + (optional) top-k freq table
    mean: float | None = None
    std: float | None = None
    var: float | None = None
    min: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    max: float | None = None
    sum: float | None = None
    levels: tuple[Freq, ...] = ()


class DescribeDatetime(DescribeBase, frozen=True):
    min: dt.datetime | dt.date | dt.time | None = None
    max: dt.datetime | dt.date | dt.time | None = None
    tz: str | None = None


class DescribeNominal(DescribeBase, frozen=True):
    levels: tuple[Freq, ...]
    n_levels: int
    mode: Any | None = None
    truncated: bool = False  # true if we only return top_k


class DescribeOrdinal(DescribeBase, frozen=True):
    levels: tuple[Freq, ...]
    n_levels: int
    mode: Any | None = None
    truncated: bool = False  # true if we only return top_k


class DescribeBoolean(DescribeBase, frozen=True):
    false: Freq | None = None
    true: Freq | None = None


class DescribeString(DescribeBase, frozen=True):
    n_unique: int | None = None
    top: tuple[Freq, ...] = ()
    truncated: bool = False
    len_min: int | None = None
    len_p50: float | None = None
    len_mean: float | None = None
    len_max: int | None = None


# tagged union for one-column result
DescribeItem: TypeAlias = (
    DescribeContinuous
    | DescribeDiscrete
    | DescribeDatetime
    | DescribeNominal
    | DescribeOrdinal
    | DescribeBoolean
    | DescribeString
)


class DescribeResult(msgspec.Struct, frozen=True):
    """
    Container for a whole describe() call.
    Keep global knobs (used/returned) so repr()/str() can show them and UIs can reconstruct.
    """

    items: tuple[DescribeItem, ...]
    weighted: bool
    weight_col: str | None
    drop_nulls: bool
    top_k: int
    percentiles: tuple[float, ...]
    generated_at: dt.datetime
    notes: str | None = None

    # honor class-level width override like Table
    PRINT_WIDTH: int | None = None

    # ---------- repr ----------
    def __repr__(self) -> str:
        n = len(self.items)
        counts = {
            "cont": sum(isinstance(x, DescribeContinuous) for x in self.items),
            "disc": sum(isinstance(x, DescribeDiscrete) for x in self.items),
            "cat": sum(isinstance(x, (DescribeNominal, DescribeOrdinal)) for x in self.items),
            "bool": sum(isinstance(x, DescribeBoolean) for x in self.items),
            "str": sum(isinstance(x, DescribeString) for x in self.items),
            "dt": sum(isinstance(x, DescribeDatetime) for x in self.items),
        }
        knobs = (
            f"weighted={self.weighted}"
            f", drop_nulls={self.drop_nulls}"
            f", top_k={self.top_k}"
            f", percentiles={tuple(self.percentiles)}"
        )
        summary = ", ".join(f"{k}={v}" for k, v in counts.items() if v)
        return f"DescribeResult(n={n}, {summary}; {knobs})"

    # ---------- rich (pretty) ----------
    def __str__(self) -> str:
        try:
            import io

            from rich.console import Console

            buf = io.StringIO()
            Console(
                file=buf,
                record=True,
                emoji=False,
                soft_wrap=True,
                force_terminal=True,
                width=_resolve_width_for(self),
            ).print(self)
            return buf.getvalue().rstrip("\r\n")
        except Exception:
            # Plain-text fallback (compact summary)
            lines: list[str] = []
            lines.append("Describe")
            lines.append(f"Columns: {len(self.items)}")
            lines.append(
                f"Weighted: {self.weighted}"
                + (f" (weight_col={self.weight_col})" if self.weight_col else "")
            )
            lines.append(f"drop_nulls: {self.drop_nulls}")
            lines.append(f"percentiles: {tuple(self.percentiles)}")
            lines.append(f"generated_at: {_fmt_dt(self.generated_at)}")
            return "\n".join(lines)

    def __rich_console__(self, console, options):
        from rich import box
        from rich.console import Group
        from rich.padding import Padding
        from rich.table import Table as RTable
        from rich.text import Text

        INDENT = 2  # left indent (spaces)

        P = _ui_module()
        styles = {"header": "bold", "border": "cyan", "title": "bold"}
        if P:
            try:
                styles.update(P.styles(self, kind="estimate"))
            except Exception:
                pass

        # --- header (each item on its own line) ---
        hdr = [
            Text(f"Columns: {len(self.items)}"),
            Text(
                f"Weighted: {self.weighted}"
                + (f" (weight_col={self.weight_col})" if self.weight_col else "")
            ),
            Text(f"drop_nulls: {self.drop_nulls}"),
            Text(f"percentiles: {tuple(self.percentiles)}"),
            Text(f"generated_at: {_fmt_dt(self.generated_at)}"),
        ]

        # Partition items by type
        num_items = [
            x for x in self.items if isinstance(x, (DescribeContinuous, DescribeDiscrete))
        ]
        cat_items = [x for x in self.items if isinstance(x, (DescribeNominal, DescribeOrdinal))]
        bool_items = [x for x in self.items if isinstance(x, DescribeBoolean)]
        str_items = [x for x in self.items if isinstance(x, DescribeString)]
        dt_items = [x for x in self.items if isinstance(x, DescribeDatetime)]

        sections: list = []

        # --- Numeric ---
        if num_items:
            sections.append(Text("Numeric", style="bold"))
            sections.append(Text(""))
            t = RTable(
                show_header=True,
                header_style=styles["header"],
                box=box.HORIZONTALS,
                show_edge=False,
                show_lines=False,
                pad_edge=False,
                expand=False,
            )
            for col, just in [
                ("name", "left"),
                ("type", "left"),
                ("n", "right"),
                ("miss", "right"),
                ("mean", "right"),
                ("std", "right"),
                ("min", "right"),
                ("p25", "right"),
                ("p50", "right"),
                ("p75", "right"),
                ("max", "right"),
                ("sum", "right"),
            ]:
                t.add_column(
                    col,
                    justify=just,  # type: ignore[arg-type]
                    no_wrap=(just == "right") or (col in {"name", "type"}),
                    overflow="ellipsis" if just == "left" else "fold",
                )

            for it in num_items:
                t.add_row(
                    it.name,
                    "Continuous" if isinstance(it, DescribeContinuous) else "Discrete",
                    str(it.n),
                    str(it.n_missing),
                    _f(it.mean),
                    _f(it.std),
                    _f(it.min),
                    _f(it.p25),
                    _f(it.p50),
                    _f(it.p75),
                    _f(it.max),
                    _f(it.sum),
                )
            sections.append(Padding(t, (0, 0, 0, INDENT)))

        # --- Categorical ---
        if cat_items:
            sections.append(Text(""))
            sections.append(Text("Categorical", style="bold"))
            sections.append(Text(""))

            t = RTable(
                show_header=True,
                header_style=styles["header"],
                box=box.HORIZONTALS,
                show_edge=False,
                show_lines=False,
                pad_edge=False,
                expand=False,
            )
            for col, just in [
                ("name", "left"),
                ("type", "left"),
                ("n", "right"),
                ("miss", "right"),
                ("levels", "right"),
                ("mode", "left"),
                ("top", "left"),
            ]:
                t.add_column(
                    col,
                    justify=just,  # type: ignore[arg-type]
                    no_wrap=(col in {"name", "type", "n", "miss", "levels"}),
                    overflow="ellipsis" if just == "left" else "fold",
                )

            for it in cat_items:
                tops = list(it.levels or ())
                if not tops:
                    t.add_row(
                        it.name,
                        ("nominal" if isinstance(it, DescribeNominal) else "ordinal"),
                        str(it.n),
                        str(it.n_missing),
                        str(it.n_levels),
                        _s(it.mode),
                        "—",
                    )
                    continue

                for idx, f in enumerate(tops):
                    top_txt = f"{f.level}: {_f(f.count)} ({_perc(f.prop)})"
                    if idx == 0:
                        t.add_row(
                            it.name,
                            ("nominal" if isinstance(it, DescribeNominal) else "ordinal"),
                            str(it.n),
                            str(it.n_missing),
                            str(it.n_levels),
                            _s(it.mode),
                            top_txt + ("  …" if it.truncated else ""),
                        )
                    else:
                        t.add_row("", "", "", "", "", "", top_txt)

            sections.append(Padding(t, (0, 0, 0, INDENT)))

        # --- Boolean ---
        if bool_items:
            sections.append(Text(""))
            sections.append(Text("Boolean", style="bold"))
            sections.append(Text(""))
            t = RTable(
                show_header=True,
                header_style=styles["header"],
                box=box.HORIZONTALS,
                show_edge=False,
                show_lines=False,
                pad_edge=False,
                expand=False,
            )
            for col in ("name", "n", "miss", "False (count / prop)", "True (count / prop)"):
                t.add_column(col, justify="left" if col == "name" else "right", no_wrap=True)

            for it in bool_items:
                fpart = f"{_f(it.false.count)} / {_perc(it.false.prop)}" if it.false else "—"
                tpart = f"{_f(it.true.count)} / {_perc(it.true.prop)}" if it.true else "—"
                t.add_row(it.name, str(it.n), str(it.n_missing), fpart, tpart)

            sections.append(Padding(t, (0, 0, 0, INDENT)))

        # --- String ---
        if str_items:
            sections.append(Text(""))
            sections.append(Text("String", style="bold"))
            sections.append(Text(""))
            t = RTable(
                show_header=True,
                header_style=styles["header"],
                box=box.HORIZONTALS,
                show_edge=False,
                show_lines=False,
                pad_edge=False,
                expand=False,
            )
            for col, just in [
                ("name", "left"),
                ("n", "right"),
                ("miss", "right"),
                ("unique", "right"),
                ("shortest", "right"),
                ("longest", "right"),
            ]:
                t.add_column(
                    col,
                    justify=just,  # type: ignore[arg-type]
                    no_wrap=(just == "right") or (col == "name"),
                    overflow="ellipsis" if just == "left" else "fold",
                )

            for it in str_items:
                t.add_row(
                    it.name,
                    str(it.n),
                    str(it.n_missing),
                    _s(it.n_unique),
                    _s(it.len_min),
                    _s(it.len_max),
                )
            sections.append(Padding(t, (0, 0, 0, INDENT)))

        # --- Datetime ---
        if dt_items:
            sections.append(Text(""))
            sections.append(Text("Datetime", style="bold"))
            sections.append(Text(""))
            t = RTable(
                show_header=True,
                header_style=styles["header"],
                box=box.HORIZONTALS,
                show_edge=False,
                show_lines=False,
                pad_edge=False,
                expand=False,
            )
            for col in ("name", "n", "miss", "min", "max", "tz"):
                t.add_column(
                    col,
                    justify="left" if col == "name" else "right",
                    no_wrap=(col != "tz"),
                    overflow="ellipsis" if col == "name" else "fold",
                )
            for it in dt_items:
                t.add_row(
                    it.name,
                    str(it.n),
                    str(it.n_missing),
                    _fmt_dt(it.min),
                    _fmt_dt(it.max),
                    it.tz or "—",
                )
            sections.append(Padding(t, (0, 0, 0, INDENT)))

        content = Group(*hdr, Text(""), *sections)

        if P:
            yield P.make_panel([content], title="Describe", obj=self, kind="estimate")
        else:
            from rich.panel import Panel

            yield Panel(content, title="Describe", border_style=styles["border"])


# ------------- formatting helpers -------------


def _f(x: float | None) -> str:
    """Compact float formatting (None → '—')."""
    if x is None:
        return "—"
    try:
        return f"{float(x):.6g}"
    except Exception:
        return str(x)


def _s(x: Any) -> str:
    return "—" if x is None else str(x)


def _perc(x: float | None) -> str:
    if x is None:
        return "—"
    try:
        return f"{100.0 * float(x):.2f}%"
    except Exception:
        return "—"


def _fmt_dt(x: dt.datetime | dt.date | dt.time | None) -> str:
    if x is None:
        return "—"
    try:
        if isinstance(x, dt.datetime):
            return x.replace(microsecond=0).isoformat()
        if isinstance(x, (dt.date, dt.time)):
            return x.isoformat()
    except Exception:
        pass
    return str(x)
