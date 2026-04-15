# src/svy/ui/printing.py
from __future__ import annotations

import io
import logging
import os
import re
import sys

from typing import Any, Iterable, Mapping, Sequence

from svy.core.constants import SVY_DEFAULT_PRINT_WIDTH
from svy.ui.theme import THEME


log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Width resolution
# -----------------------------------------------------------------------------
def resolve_width(
    owner: object | None = None,
    default: int = SVY_DEFAULT_PRINT_WIDTH,
    **kwargs: Any,
) -> int:
    """
    Resolve print width strictly.
    Hierarchy:
      1. Instance override (obj._print_width)
      2. Class override (Class.PRINT_WIDTH)
      3. SVY_PRINT_WIDTH env var
      4. Default (usually 90-100)

    NOTE: We explicitly IGNORE os.environ["COLUMNS"] and shutil.get_terminal_size()
    to prevent modern terminals (like Ghostty) from forcing full-width output.
    """
    obj = owner or kwargs.get("obj")

    def _as_int(x):
        try:
            return int(x)
        except Exception:
            return None

    # 1. Instance-level hint
    w = getattr(obj, "_print_width", None) if obj is not None else None
    if isinstance(w, int) and w > 20:
        return w

    # 2. Class / Module hints
    if obj is not None:
        cw = getattr(type(obj), "PRINT_WIDTH", None)
        if isinstance(cw, int) and cw > 20:
            return cw
        try:
            mod = sys.modules.get(obj.__class__.__module__)
            if mod is not None and hasattr(mod, "SVY_PRINT_WIDTH"):
                mw = _as_int(getattr(mod, "SVY_PRINT_WIDTH"))
                if isinstance(mw, int) and mw > 20:
                    return mw
        except Exception:
            pass

    # 3. SVY Specific Env Var (The only way to force global width override)
    ew = _as_int(os.environ.get("SVY_PRINT_WIDTH"))
    if isinstance(ew, int) and ew > 20:
        return ew

    # 4. Fallback to constant
    return default


# -----------------------------------------------------------------------------
# Simple string padding utility
# -----------------------------------------------------------------------------
def pad(text: str, *, indent: int = 2, surround: bool = False) -> str:
    if text is None:
        return ""
    text = str(text).rstrip("\n")
    if indent > 0:
        pad_ = " " * indent
        text = "\n".join(pad_ + line if line else pad_ for line in text.splitlines())
    return f"\n{text}\n" if surround else text


# -----------------------------------------------------------------------------
# Natural sort — shared across all result classes
# -----------------------------------------------------------------------------


def natural_sort_key(value: Any) -> tuple:
    """
    Natural sort key for a single display value.

    Splits strings into alternating (text, number) chunks so that
    "1. Urban" < "2. Rural" < "10. Suburban" rather than the lexicographic
    "1..." < "10..." < "2...".  Numeric types sort directly without conversion.

    Used by sort_display_rows() and anywhere else a result class needs to sort
    levels for display.
    """
    if isinstance(value, (int, float)):
        return (value,)
    s = str(value) if value is not None else ""
    parts: list = []
    for chunk in re.split(r"(\d+)", s):
        parts.append(int(chunk) if chunk.isdigit() else chunk.lower())
    return tuple(parts)


# -----------------------------------------------------------------------------
# Table-aware sort helpers
# -----------------------------------------------------------------------------

_INTERVAL_RE = re.compile(
    r"""^\s*
        ([\(\[])\s*
        ([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*,\s*
        ([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*
        ([\)\]])\s*
        $""",
    re.VERBOSE,
)


def _interval_key_str(s: str) -> tuple[float, float] | None:
    """Return (low, high) if s is an interval label like '(0, 10]', else None."""
    m = _INTERVAL_RE.match(s.strip())
    if not m:
        return None
    return (float(m.group(2)), float(m.group(3)))


def level_sort_key(s: str) -> tuple:
    """
    Unified display-level sort key used by all result classes.

    Priority:
      0) pure numeric strings (sorted by numeric value)
      1) interval labels '(a, b]' / '[a, b)' (sorted by low bound, then high)
      2) everything else — natural sort (so "1. Urban" < "2. Rural" < "10. Sub")

    Superset of natural_sort_key: adds numeric-scalar and interval-label
    handling on top of the digit-aware string splitting.
    """
    stripped = str(s).strip() if s is not None else ""
    # 0. Pure numeric
    try:
        f = float(stripped)
        i = int(f)
        return (0, i if f == i else f, 0.0)
    except (ValueError, TypeError):
        pass
    # 1. Interval label
    iv = _interval_key_str(stripped)
    if iv is not None:
        return (1, iv[0], iv[1])
    # 2. Natural sort on remaining strings
    return (2,) + natural_sort_key(stripped)


def level_index_key(
    value: Any,
    explicit_levels: Sequence | None,
) -> tuple:
    """
    Sort key that respects an explicit level ordering when available,
    falling back to level_sort_key otherwise.

    When explicit_levels is provided and value is found in it, its position
    is used directly so user-defined order is preserved exactly.
    """
    norm = str(value).strip() if value is not None else ""
    if explicit_levels is not None:
        idx = {str(k).strip(): i for i, k in enumerate(explicit_levels)}
        if norm in idx:
            return (0, idx[norm], ())
    return (1, 0, level_sort_key(norm))


# Column names that are always right-justified in plain tables (numeric estimates).
# Matches the headers used by Table._headers_for_display and Estimate.__plain_str__.
_PLAIN_NUMERIC_HEADERS: frozenset[str] = frozenset(
    {"est", "se", "cv", "lci", "uci", "deff", "Estimate", "Std Err", "CV", "Lower", "Upper"}
)


def render_plain_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[str]],
    *,
    numeric_headers: frozenset[str] | None = None,
    indent: str = "  ",
) -> str:
    """
    Render a plain-text table with consistent styling across all result classes.

    Numeric columns (identified by ``numeric_headers``) are right-justified;
    text columns are left-justified.  The separator is a row of dashes with no
    vertical bars, matching the style used by ``Estimate.__plain_str__``.

    A blank line is inserted between the separator and the first data row so
    the table breathes consistently with the rich panel output.  All rows
    (header, separator, data) are prefixed with ``indent`` (default two spaces).

    Parameters
    ----------
    headers:
        Column header strings.
    rows:
        Iterable of row sequences (already formatted as strings).
    numeric_headers:
        Set of header names to right-justify.  Defaults to
        ``_PLAIN_NUMERIC_HEADERS`` which covers all standard estimate columns.
    indent:
        String prepended to every line.  Defaults to two spaces.
    """
    _numeric = numeric_headers if numeric_headers is not None else _PLAIN_NUMERIC_HEADERS
    rows_list = [list(map(str, r)) for r in rows]
    if not rows_list:
        return indent + "  ".join(str(h) for h in headers)
    widths = [max(len(str(h)), *(len(r[i]) for r in rows_list)) for i, h in enumerate(headers)]

    def _fmt(val: str, i: int) -> str:
        w = widths[i]
        return val.rjust(w) if headers[i] in _numeric else val.ljust(w)

    header_str = indent + "  ".join(_fmt(str(h), i) for i, h in enumerate(headers))
    separator = indent + "  ".join("-" * w for w in widths)
    data_rows = [indent + "  ".join(_fmt(v, i) for i, v in enumerate(r)) for r in rows_list]
    return "\n".join([header_str, separator] + data_rows)


def sort_display_rows(
    rows: list[dict],
    *,
    numeric_keys: set[str],
) -> list[dict]:
    """
    Sort a list of display-ready row dicts by their non-numeric columns.

    The sort is applied to all columns that are NOT in ``numeric_keys``,
    in column-order, using :func:`natural_sort_key` so that labelled levels
    like ``"1. Urban"`` / ``"2. Rural"`` sort by their leading number rather
    than alphabetically.

    Called by ``Estimate.to_polars_printable()`` and any other result class
    that builds a row-dict list before rendering, ensuring consistent ordering
    across all output types.

    Parameters
    ----------
    rows:
        List of dicts as produced by the row-building loop in each result class.
        All dicts must share the same key set.
    numeric_keys:
        Set of column names that contain numeric estimates and should be
        excluded from the sort key (e.g. ``{"est", "se", "lci", "uci", "cv",
        "deff"}``).

    Returns
    -------
    list[dict]
        The same rows, sorted in-place and returned.
    """
    if not rows:
        return rows
    display_cols = [c for c in rows[0] if c not in numeric_keys]
    if not display_cols:
        return rows
    rows.sort(key=lambda r: tuple(natural_sort_key(r.get(c)) for c in display_cols))
    return rows


# -----------------------------------------------------------------------------
# Where-clause display helper
# -----------------------------------------------------------------------------


def _clean_expr_str(expr) -> str:
    """Convert a single polars Expr (or svy.col wrapper) to a readable string."""
    if hasattr(expr, "_e"):
        expr = expr._e
    s = str(expr).strip()
    # Strip outer brackets: [...] -> ...
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    # (col("name")) or col("name") -> name
    s = re.sub(r'\(col\("([^"]+)"\)\)', r"\1", s)
    s = re.sub(r'col\("([^"]+)"\)', r"\1", s)
    # (dyn int: N) / (dyn float: N) -> N
    s = re.sub(r"\(dyn int: (-?\d+)\)", r"\1", s)
    s = re.sub(r"\(dyn float: (-?[\d.]+(?:e[+-]?\d+)?)\)", r"\1", s)
    # (String: "foo") / (Utf8: "foo") -> "foo"
    s = re.sub(r'\((?:String|Utf8): ("[^"]*")\)', r"\1", s)
    # (Boolean: true/false) -> true/false
    s = re.sub(r"\(Boolean: (true|false)\)", r"\1", s)
    # parens around bare identifiers or quoted strings
    s = re.sub(r"\(([A-Za-z_]\w*)\)", r"\1", s)
    s = re.sub(r'\(("[^"]*")\)', r"\1", s)
    # .strict_cast(...) artifacts
    s = re.sub(r"\.strict_cast\([^)]*\)", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def format_where_clause(where) -> str | None:
    """Convert a WhereArg to a human-readable display string.

    Returns None when where is None, a non-empty string otherwise.
    Suitable for use in ``Estimate.where_clause``, ``TTestByResult.where_clause``,
    and ``RankTestByResult.where_clause``.
    """
    if where is None:
        return None

    if hasattr(where, "_e") or hasattr(where, "meta"):
        # svy.col wrapper or polars Expr
        return _clean_expr_str(where) or None

    try:
        import polars as pl

        if isinstance(where, pl.Expr):
            return _clean_expr_str(where) or None
    except ImportError:
        pass

    if isinstance(where, Mapping):
        parts = []
        for k, v in where.items():
            if isinstance(v, (list, tuple, set)):
                parts.append(f"{k} in {list(v)!r}")
            else:
                parts.append(f"{k} == {v!r}")
        return " & ".join(parts) or None

    if isinstance(where, Sequence) and not isinstance(where, str):
        parts = [_clean_expr_str(w) for w in where]
        return " & ".join(p for p in parts if p) or None

    result = str(where)
    return result or None


# -----------------------------------------------------------------------------
# Central style + feature routing
# -----------------------------------------------------------------------------
def _feature_kind_from_key(key: str | None) -> str:
    if not key:
        return "panel"
    k = key.lower()
    if "error" in k:
        return "error"
    if "estimate" in k:
        return "estimate"
    if "ttest" in k:
        return "ttest"
    if "sample" in k:
        return "sample"
    if "size" in k:
        return "size"
    if "table" in k:
        return "estimate"
    return "panel"


def styles(
    obj_or_key: object | str | None = None,
    *,
    kind: str | None = None,
    title: str | None = None,
    border: str | None = None,
    header: str | None = None,
) -> Mapping[str, str]:
    # Robustly get theme components
    try:
        c = THEME.components
    except Exception:
        # Fallback if THEME is not initialized/broken
        return {
            "border": border or "cyan",
            "header": header or "bold",
            "title": title or "bold",
        }

    if isinstance(obj_or_key, str):
        eff_kind = _feature_kind_from_key(obj_or_key)
        obj = None
    else:
        eff_kind = kind or "panel"
        obj = obj_or_key

    # Theme defaults
    if eff_kind == "error":
        border_def = getattr(c, "error_border", "red")
        header_def = getattr(c, "error_header_style", "bold red")
        title_def = getattr(c, "error_title_style", "bold red")
    elif eff_kind == "ttest":
        border_def = getattr(c, "ttest_border", "green")
        header_def = getattr(c, "header_style", "bold")
        title_def = getattr(c, "title_style", "bold")
    elif eff_kind == "sample":
        border_def = getattr(c, "sample_border", "cyan")
        header_def = getattr(c, "header_style", "bold")
        title_def = getattr(c, "title_style", "bold")
    elif eff_kind == "estimate":
        border_def = getattr(c, "estimate_border", "blue")
        header_def = getattr(c, "header_style", "bold")
        title_def = getattr(c, "title_style", "bold")
    elif eff_kind == "size":
        border_def = getattr(c, "size_border", "magenta")
        header_def = getattr(c, "header_style", "bold")
        title_def = getattr(c, "title_style", "bold")
    else:
        border_def = getattr(c, "panel_border", "white")
        header_def = getattr(c, "header_style", "bold")
        title_def = getattr(c, "title_style", "bold")

    if obj is not None:
        border_def = getattr(obj, "_PANEL_BORDER", None) or border_def
        header_def = getattr(obj, "_HEADER_STYLE", None) or header_def
        title_def = getattr(obj, "_TITLE_STYLE", None) or title_def

    return {
        "border": border or border_def,
        "header": header or header_def,
        "title": title or title_def,
    }


def panel_enabled(feature_key: str | None = None) -> bool:
    v = os.environ.get("SVY_PRINT_PANEL")
    if v is not None and v.lower() in {"0", "false", "off", "no"}:
        return False
    return True


# -----------------------------------------------------------------------------
# Rich availability check
# -----------------------------------------------------------------------------
def rich_available() -> bool:
    """Return True if the rich package is installed and importable."""
    try:
        import rich  # noqa: F401

        return True
    except ImportError:
        return False


# -----------------------------------------------------------------------------
# Console + rendering helpers
# -----------------------------------------------------------------------------
def _console(width: int):
    from rich.console import Console

    return Console(
        file=io.StringIO(),
        record=True,
        width=width,
        force_terminal=True,
        color_system="auto",
        emoji=False,
        # soft_wrap=False ensures Tables calculate wrapping correctly within the width
        soft_wrap=False,
    )


def render_rich_to_str(renderable, *, width: int) -> str:
    """
    Render a Rich renderable (or any object with __rich_console__) to a plain string.

    Falls back to plain_text_fallback(renderable) if rich is not installed,
    or if rendering raises an unexpected error.

    NOTE: The fallback must NOT call str(renderable) — that would re-enter
    __str__ on the object and cause infinite recursion.
    """
    if not rich_available():
        return plain_text_fallback(renderable, width=width)
    try:
        console = _console(width)
        console.print(renderable)
        return console.file.getvalue().rstrip("\r\n")
    except Exception:
        log.debug("render_rich_to_str failed, falling back to plain text", exc_info=True)
        return plain_text_fallback(renderable, width=width)


def plain_text_fallback(renderable, *, width: int) -> str:
    """
    Produce a minimal plain-text representation when rich is not available.

    Checks for a __plain_str__ method first (opt-in per-class), then falls
    back to repr().  Never calls str() on the object to avoid recursion.
    """
    fn = getattr(renderable, "__plain_str__", None)
    if callable(fn):
        try:
            return fn()
        except Exception:
            pass
    return repr(renderable)


# -----------------------------------------------------------------------------
# Panel & Table builders (centralized)
# -----------------------------------------------------------------------------
def make_panel(
    children: Iterable,
    *,
    title: str,
    obj: object | None = None,
    kind: str = "panel",
):
    from rich.console import Group
    from rich.panel import Panel

    s = styles(obj, kind=kind)

    # Use explicit markup for title style if needed, or let Panel handle it if string
    return Panel(
        Group(*children),
        title=title,
        border_style=s["border"],
        padding=(0, 1),  # Internal padding between border and content
        expand=False,
    )


def make_table(
    *,
    header_names: Iterable[str],
    numeric: set[str] | None = None,
    obj: object | None = None,
    kind: str = "panel",
    variant: str = "rounded",
):
    from rich import box
    from rich.table import Table

    s = styles(obj, kind=kind)

    if variant == "header_only":
        chosen_box = getattr(box, "HORIZONTALS", box.SIMPLE_HEAD)
    elif variant == "minimal_head":
        chosen_box = getattr(box, "MINIMAL_HEAVY_HEAD", getattr(box, "SIMPLE_HEAVY", box.SIMPLE))
    else:
        chosen_box = box.ROUNDED

    t = Table(
        box=chosen_box,
        show_header=True,
        header_style=s["header"],
        pad_edge=False,
        expand=False,
        padding=(0, 1),
        show_edge=False if variant in {"header_only", "minimal_head"} else True,
        show_lines=False,
    )

    numeric = numeric or set()
    for name in header_names:
        justify = "right" if name in numeric else "left"
        t.add_column(name, justify=justify, no_wrap=(justify == "right"), overflow="fold")
    return t
