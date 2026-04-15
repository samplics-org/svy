# svy/ui/printing.py
from __future__ import annotations

import io
import os
import shutil
import sys

from typing import Iterable, Mapping

from svy.ui.theme import THEME


# ---------- width & padding ---------------------------------------------------


def resolve_width(obj: object | None = None, default: int = 88) -> int:
    """Consistent width resolution (instance → class → module var → __main__ → builtins → env → terminal)."""

    def _as_int(x):
        try:
            return int(x)
        except Exception:
            return None

    # 1) per-instance override
    w = getattr(obj, "_print_width", None)
    if isinstance(w, int) and w > 20:
        return w
    # 2) class-level override
    if obj is not None:
        cw = getattr(type(obj), "PRINT_WIDTH", None)
        if isinstance(cw, int) and cw > 20:
            return cw
    # 3) module-level SVY_PRINT_WIDTH
    try:
        if obj is not None:
            mod = sys.modules.get(obj.__class__.__module__)
            if mod is not None and hasattr(mod, "SVY_PRINT_WIDTH"):
                mw = _as_int(getattr(mod, "SVY_PRINT_WIDTH"))
                if isinstance(mw, int) and mw > 20:
                    return mw
    except Exception:
        pass
    # 4) __main__.SVY_PRINT_WIDTH
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and hasattr(main_mod, "SVY_PRINT_WIDTH"):
        mw = _as_int(getattr(main_mod, "SVY_PRINT_WIDTH"))
        if isinstance(mw, int) and mw > 20:
            return mw
    # 5) builtins.SVY_PRINT_WIDTH
    try:
        import builtins

        bw = _as_int(getattr(builtins, "SVY_PRINT_WIDTH", None))
        if isinstance(bw, int) and bw > 20:
            return bw
    except Exception:
        pass
    # 6) env var
    ew = _as_int(os.environ.get("SVY_PRINT_WIDTH"))
    if isinstance(ew, int) and ew > 20:
        return ew
    # 7) terminal fallback
    try:
        ts = shutil.get_terminal_size(fallback=(default, 24))
        if isinstance(ts.columns, int) and ts.columns > 20:
            return ts.columns
    except Exception:
        pass
    return default


def pad(text: str, *, indent: int = 2, surround: bool = False) -> str:
    """Left-pad all lines; optionally add a blank line above and below."""
    if text is None:
        return ""
    text = str(text).rstrip("\n")
    if indent > 0:
        pad_ = " " * indent
        text = "\n".join(pad_ + line if line else pad_ for line in text.splitlines())
    return f"\n{text}\n" if surround else text


# ---------- theme styles ------------------------------------------------------


def styles(obj: object | None = None) -> Mapping[str, str]:
    """Resolve styling from object overrides → theme."""
    border = getattr(obj, "_PANEL_BORDER", None) or THEME.components.panel_border
    header = getattr(obj, "_HEADER_STYLE", None) or THEME.components.header_style
    title = getattr(obj, "_TITLE_STYLE", None) or THEME.components.title_style
    return {"border": border, "header": header, "title": title}


# ---------- rich helpers (optional dependency) --------------------------------


def _console(width: int):
    from rich.console import Console

    return Console(
        file=io.StringIO(),
        record=True,
        width=width,
        force_terminal=True,
        color_system="auto",
        emoji=False,
        soft_wrap=True,
    )


def render_rich_to_str(renderable, *, width: int) -> str:
    """Render any Rich renderable (or object with __rich_console__) to a string."""
    try:
        console = _console(width)
        console.print(renderable)
        return console.file.getvalue().rstrip("\r\n")
    except Exception:
        # Last-ditch: plain str()
        return str(renderable)


def make_panel(children: Iterable, *, title: str, obj: object | None = None):
    """Panel(Group(children...)) with brand border color."""
    from rich.console import Group
    from rich.panel import Panel

    s = styles(obj)
    return Panel(Group(*children), title=title, border_style=s["border"])


def make_table(
    *, header_names: Iterable[str], numeric: set[str] | None = None, obj: object | None = None
):
    """Configured Rich Table with brand header style; right-align numeric columns by name."""
    from rich import box
    from rich.table import Table

    s = styles(obj)
    t = Table(
        box=box.ROUNDED, show_header=True, header_style=s["header"], pad_edge=False, expand=False
    )
    numeric = numeric or set()
    for name in header_names:
        justify = "right" if name in numeric else "left"
        t.add_column(name, justify=justify, no_wrap=(justify == "right"), overflow="fold")
    return t
