# src/svy/size/base.py
"""
SampleSize facade.

The SampleSize class is a thin dispatcher. Each goal method is a one-liner
that delegates to the corresponding module-level function. Core logic
(display, export, internal helpers) lives here; statistical logic does not.

Adding a new sizing goal:
  1. Implement it in svy/size/estimation_goals.py or svy/size/comparison_goals.py
     as a module-level function.
  2. Import it here and add a one-liner delegation method.
  3. If it has a pure algorithm, put that in svy/engine/size_and_power/size.py.
"""

from __future__ import annotations

import copy
import logging
import re

from collections.abc import Iterable, Mapping
from typing import Literal

from svy.core.enumerations import PopParam
from svy.core.types import DomainScalarMap, Number
from svy.size.allocation_goals import AllocationMethod
from svy.size.allocation_goals import allocate as _allocate
from svy.size.comparison_goals import compare_means as _compare_means
from svy.size.comparison_goals import compare_props as _compare_props
from svy.size.estimation_goals import estimate_mean as _estimate_mean
from svy.size.estimation_goals import estimate_prop as _estimate_prop
from svy.size.types import Allocation, Size, Target
from svy.ui.printing import make_panel, render_rich_to_str, resolve_width


log = logging.getLogger(__name__)

_CHUNK_RE = re.compile(r"(\d+)")


# =============================================================================
# SampleSize facade
# =============================================================================


def _stratum_label(stratum: object) -> str:
    """How a stratum key prints: "overall" when unstratified, "N, u" for a tuple."""
    if stratum is None:
        return "overall"
    if isinstance(stratum, tuple):
        return ", ".join(map(str, stratum))
    return str(stratum)


class SampleSize:
    """
    Compute required sample sizes for survey objectives under simple or stratified designs.

    SampleSize holds no configuration at construction time. All parameters —
    including pop_size, deff, and resp_rate — are passed directly to the goal
    methods. Stratification is inferred automatically whenever any parameter
    is provided as a dict.

    Examples
    --------
    # Unstratified
    SampleSize().estimate_prop(p=0.8, moe=0.10, deff=1.5, resp_rate=0.90)

    # Stratified — dicts declare it
    SampleSize().estimate_prop(
        p={"r1": 0.8, "r2": 0.6},
        moe=0.10,
        pop_size={"r1": 1000, "r2": 2000},
        deff=1.5,
    )
    """

    # ---------- printing/branding toggles (class-level fallbacks) ----------
    _PRINT_PANEL: bool = True
    _PANEL_TITLE: str = "Sample Size"
    _PANEL_BORDER: str = "cyan"
    PRINT_WIDTH: int | None = None

    __slots__ = ("_param", "_target", "_size", "_group_labels", "_allocation")

    def __init__(self):
        self._param = None
        self._target = None
        self._size = None
        self._group_labels = None
        self._allocation = None

    # ---------- internal helpers ----------

    @staticmethod
    def _is_size_obj(obj) -> bool:
        return all(hasattr(obj, a) for a in ("n0", "n1_deff", "n2_fpc", "n"))

    def _iter_sizes(self):
        """Return a flat list of Size-like objects from self._size."""
        sizes = getattr(self, "_size", None)
        if sizes is None:
            return []
        if self._is_size_obj(sizes):
            return [sizes]
        if isinstance(sizes, Mapping):
            out = []
            for v in sizes.values():
                if self._is_size_obj(v):
                    out.append(v)
                elif isinstance(v, Iterable) and not isinstance(v, (str, bytes)):
                    out.extend([x for x in v if self._is_size_obj(x)])
            return out
        if isinstance(sizes, Iterable) and not isinstance(sizes, (str, bytes)):
            return [v for v in sizes if self._is_size_obj(v)]
        return []

    @staticmethod
    def _fmt_num(x) -> str:
        try:
            xf = float(x)
            xi = round(xf)
            return f"{xi:.0f}" if abs(xf - xi) < 1e-9 else f"{xf:.1f}"
        except Exception:
            return str(x)

    @staticmethod
    def _nat_key(label: object):
        """Natural sort: 'region2' < 'region10'; None/'overall' first."""
        if label is None:
            return (0, [""])
        s = str(label).strip().lower()
        is_overall = 0 if s in {"", "overall", "all", "national"} else 1
        parts = _CHUNK_RE.split(s)
        key = [int(p) if p.isdigit() else p for p in parts]
        return (is_overall, key)

    # ---------------- public: Polars export -------------------

    def to_polars(
        self,
        order_by: str | None = None,
        ascending: bool = True,
        natural: bool = True,
        overall_first: bool = True,
        group_labels: list[str] | None = None,
        include_stratum: bool | None = None,
    ):
        """Normalize sizes to a Polars DataFrame."""
        try:
            import polars as pl
        except Exception as e:
            raise ImportError("to_polars() requires the 'polars' package") from e

        if self._allocation is not None:
            return self._allocation_frame()

        objs = self._iter_sizes()
        stratified = len(objs) > 1 or (len(objs) == 1 and objs[0].stratum is not None)

        if include_stratum is None:
            include_stratum = stratified

        if not objs:
            cols = (["stratum"] if include_stratum else []) + ["n0", "n1_deff", "n2_fpc", "n"]
            return pl.DataFrame({c: [] for c in cols})

        from collections.abc import Iterable as _Iterable

        def _len_if_iter(x):
            return len(x) if isinstance(x, _Iterable) and not isinstance(x, (str, bytes)) else None

        fields = ("n0", "n1_deff", "n2_fpc", "n")
        group_len = None
        for s in objs:
            lens = {f: _len_if_iter(getattr(s, f)) for f in fields}
            ls = [L for L in lens.values() if L is not None]
            if ls:
                L0 = ls[0]
                if any(L != L0 for L in ls):
                    raise ValueError(f"Inconsistent tuple lengths within a Size: {lens}")
                if group_len is None:
                    group_len = L0
                elif group_len != L0:
                    raise ValueError(
                        f"Inconsistent tuple lengths across strata: expected {group_len}, got {L0}"
                    )

        rows = []
        if group_len is None:
            for s in objs:
                label = _stratum_label(s.stratum)
                row: dict[str, object] = {
                    "n0": float(s.n0),
                    "n1_deff": float(s.n1_deff),
                    "n2_fpc": float(s.n2_fpc),
                    "n": float(s.n),
                }
                if include_stratum:
                    row["stratum"] = label
                rows.append(row)
            df = pl.DataFrame(rows)
            cols = (["stratum"] if include_stratum else []) + ["n0", "n1_deff", "n2_fpc", "n"]
            df = df.select(cols)
        else:
            if not group_labels:
                group_labels = self._group_labels or [f"group{i + 1}" for i in range(group_len)]
            if len(group_labels) != group_len:
                raise ValueError(f"group_labels length {len(group_labels)} != {group_len}")
            for s in objs:
                label = _stratum_label(s.stratum)
                n0, n1, n2, n = (getattr(s, f) for f in fields)
                for i in range(group_len):
                    row: dict[str, object] = {
                        "group": group_labels[i],
                        "n0": float(n0[i]),
                        "n1_deff": float(n1[i]),
                        "n2_fpc": float(n2[i]),
                        "n": float(n[i]),
                    }
                    if include_stratum:
                        row["stratum"] = label
                    rows.append(row)
            df = pl.DataFrame(rows)
            cols = (["stratum"] if include_stratum else []) + [
                "group",
                "n0",
                "n1_deff",
                "n2_fpc",
                "n",
            ]
            df = df.select(cols)

        if order_by is None:
            if include_stratum:
                order_by = "stratum"
            else:
                order_by = "group" if ("group" in df.columns) else "n"

        if order_by == "stratum" and include_stratum and "stratum" in df.columns:
            if natural:
                lower = df["stratum"].str.to_lowercase()
                is_overall = lower.is_in(["", "overall", "all", "national"]).cast(pl.Int8)
                overall_key = (1 - is_overall) if overall_first else is_overall
                df = (
                    df.with_columns(
                        [
                            overall_key.alias("_overall"),
                            lower.str.replace_all(r"\d+", "").alias("_prefix"),
                            lower.str.extract(r"(\d+)").cast(pl.Int64).fill_null(0).alias("_num"),
                        ]
                    )
                    .sort(["_overall", "_prefix", "_num"], descending=[not ascending] * 3)
                    .drop(["_overall", "_prefix", "_num"])
                )
            else:
                df = df.sort("stratum", descending=not ascending)
        elif order_by == "group" and "group" in df.columns:
            if natural:
                gl = df["group"].str.to_lowercase()
                df = (
                    df.with_columns(
                        [
                            gl.str.replace_all(r"\d+", "").alias("_gpre"),
                            gl.str.extract(r"(\d+)").cast(pl.Int64).fill_null(0).alias("_gnum"),
                        ]
                    )
                    .sort(["_gpre", "_gnum"], descending=[not ascending, not ascending])
                    .drop(["_gpre", "_gnum"])
                )
            else:
                df = df.sort("group", descending=not ascending)
        elif order_by in {"n", "n0", "n2_fpc", "n1_deff"}:
            df = df.sort(order_by, descending=not ascending)

        if include_stratum and "group" in df.columns and order_by in {"stratum", "group"}:
            gl = df["group"].str.to_lowercase() if "group" in df.columns else None
            if gl is not None:
                df = (
                    df.with_columns(
                        [
                            df["stratum"]
                            .str.to_lowercase()
                            .str.replace_all(r"\d+", "")
                            .alias("_spre"),
                            df["stratum"]
                            .str.to_lowercase()
                            .str.extract(r"(\d+)")
                            .cast(pl.Int64)
                            .fill_null(0)
                            .alias("_snum"),
                            gl.str.replace_all(r"\d+", "").alias("_gpre2"),
                            gl.str.extract(r"(\d+)").cast(pl.Int64).fill_null(0).alias("_gnum2"),
                        ]
                    )
                    .sort(["_spre", "_snum", "_gpre2", "_gnum2"], descending=[False] * 4)
                    .drop(["_spre", "_snum", "_gpre2", "_gnum2"])
                )

        return df

    def _format_table_ascii(self) -> str:
        """ASCII fallback — used by __plain_str__ and render_plain_table."""
        if self._allocation is not None:
            from svy.ui.printing import render_plain_table

            df = self._allocation_frame()
            rows = [
                [str(v) if i == 0 else self._fmt_num(v) for i, v in enumerate(r)]
                for r in df.rows()
            ]
            return f"Sample Size\n\n{render_plain_table(tuple(df.columns), rows)}"
        objs = self._iter_sizes()
        if not objs:
            return "<no sizes>"

        stratified = len(objs) > 1 or (len(objs) == 1 and objs[0].stratum is not None)

        from collections.abc import Iterable as _Iterable

        any_tuple = any(
            isinstance(getattr(s, "n"), _Iterable)
            and not isinstance(getattr(s, "n"), (str, bytes))
            for s in objs
        )

        if any_tuple:
            headers = (
                ("group", "n0", "n1_deff", "n2_fpc", "n")
                if not stratified
                else ("stratum", "group", "n0", "n1_deff", "n2_fpc", "n")
            )
            rows = []
            for s in objs:
                label = _stratum_label(s.stratum)
                n0, n1, n2, n = s.n0, s.n1_deff, s.n2_fpc, s.n
                m = len(n)
                _glabels = self._group_labels or [f"group{j + 1}" for j in range(m)]
                for i in range(m):
                    if stratified:
                        rows.append((label, _glabels[i], n0[i], n1[i], n2[i], n[i]))
                    else:
                        rows.append((_glabels[i], n0[i], n1[i], n2[i], n[i]))
            if stratified:
                rows.sort(key=lambda r: (self._nat_key(r[0]), self._nat_key(r[1])))
            else:
                rows.sort(key=lambda r: self._nat_key(r[0]))
        else:
            headers = (
                ("n0", "n1_deff", "n2_fpc", "n")
                if not stratified
                else ("stratum", "n0", "n1_deff", "n2_fpc", "n")
            )
            rows = []
            for s in objs:
                label = _stratum_label(s.stratum)
                if stratified:
                    rows.append((str(label), s.n0, s.n1_deff, s.n2_fpc, s.n))
                else:
                    rows.append((s.n0, s.n1_deff, s.n2_fpc, s.n))
            if stratified:
                rows.sort(key=lambda r: self._nat_key(r[0]))

        def fmt_row(r):
            if len(headers) == 6:
                return (r[0], r[1], *(self._fmt_num(v) for v in r[2:]))
            elif len(headers) == 5:
                return (r[0], *(self._fmt_num(v) for v in r[1:]))
            elif len(headers) == 4:
                return tuple(self._fmt_num(v) for v in r)
            else:
                return (r[0], *(self._fmt_num(v) for v in r[1:]))

        formatted = [tuple(map(str, fmt_row(r))) for r in rows]
        cols = list(zip(*([headers] + formatted)))
        widths = [max(len(c) for c in col) for col in cols]

        def line(cells):
            out = []
            for i, (c, w) in enumerate(zip(cells, widths)):
                left_aligned = headers[i] in {"stratum", "group"}
                out.append(c.ljust(w) if left_aligned else c.rjust(w))
            return "  ".join(out)

        out = [line(headers), line(["-" * w for w in widths])]
        out += [line(r) for r in formatted]
        return "\n".join(out)

    def __plain_str__(self) -> str:
        """Plain-text fallback when rich is not installed."""
        from svy.ui.printing import render_plain_table

        objs = self._iter_sizes()
        if not objs:
            return "Sample Size\n\n  <no sizes computed>"

        # Build headers and rows for render_plain_table
        stratified = len(objs) > 1 or (len(objs) == 1 and objs[0].stratum is not None)

        from collections.abc import Iterable as _Iterable

        any_tuple = any(
            isinstance(getattr(s, "n"), _Iterable)
            and not isinstance(getattr(s, "n"), (str, bytes))
            for s in objs
        )

        if any_tuple:
            h = (
                ["group", "n0", "n1_deff", "n2_fpc", "n"]
                if not stratified
                else ["stratum", "group", "n0", "n1_deff", "n2_fpc", "n"]
            )
            rows = []
            for s in objs:
                label = _stratum_label(s.stratum)
                n0, n1, n2, n = s.n0, s.n1_deff, s.n2_fpc, s.n
                m = len(n)
                _glabels = self._group_labels or [f"group{j + 1}" for j in range(m)]
                for i in range(m):
                    if stratified:
                        rows.append(
                            [
                                str(label),
                                _glabels[i],
                                self._fmt_num(n0[i]),
                                self._fmt_num(n1[i]),
                                self._fmt_num(n2[i]),
                                self._fmt_num(n[i]),
                            ]
                        )
                    else:
                        rows.append(
                            [
                                _glabels[i],
                                self._fmt_num(n0[i]),
                                self._fmt_num(n1[i]),
                                self._fmt_num(n2[i]),
                                self._fmt_num(n[i]),
                            ]
                        )
            if stratified:
                rows.sort(key=lambda r: (self._nat_key(r[0]), self._nat_key(r[1])))
            else:
                rows.sort(key=lambda r: self._nat_key(r[0]))
        else:
            h = (
                ["n0", "n1_deff", "n2_fpc", "n"]
                if not stratified
                else ["stratum", "n0", "n1_deff", "n2_fpc", "n"]
            )
            rows = []
            for s in objs:
                label = _stratum_label(s.stratum)
                if stratified:
                    rows.append(
                        [
                            str(label),
                            self._fmt_num(s.n0),
                            self._fmt_num(s.n1_deff),
                            self._fmt_num(s.n2_fpc),
                            self._fmt_num(s.n),
                        ]
                    )
                else:
                    rows.append(
                        [
                            self._fmt_num(s.n0),
                            self._fmt_num(s.n1_deff),
                            self._fmt_num(s.n2_fpc),
                            self._fmt_num(s.n),
                        ]
                    )
            if stratified:
                rows.sort(key=lambda r: self._nat_key(r[0]))

        return f"Sample Size\n\n{render_plain_table(h, rows)}"

    def __str__(self) -> str:
        return render_rich_to_str(self, width=resolve_width(self))

    def __repr__(self) -> str:
        if self._allocation is not None:
            return f"SampleSize(allocation, strata={len(self._allocation)}, n={self._target.n})"
        objs = self._iter_sizes()
        n = len(objs)
        stratified = n > 1 or (n == 1 and objs[0].stratum is not None)
        param = getattr(self._param, "name", None) or "uncomputed"
        return f"SampleSize(stratified={stratified}, strata={n}, param={param})"

    # ---------------- Rich integration (optional) --------------

    def to_rich_table(self, title: str | None = None):
        """Return a Rich Table (requires `rich`)."""
        try:
            from rich import box
            from rich.table import Table
        except Exception as e:
            raise ImportError("`rich` is required for to_rich_table(). `pip install rich`") from e

        cols, rows = None, None
        try:
            df = self.to_polars()
            cols, rows = df.columns, df.rows()
            start_idx = 2 if ("group" in cols) else 1
            rows = [
                tuple((v if i < start_idx else self._fmt_num(v)) for i, v in enumerate(row))
                for row in rows
            ]
        except Exception:
            from rich.text import Text

            t = Table(title=title, box=box.SIMPLE)
            t.add_column("output")
            t.add_row(Text(self._format_table_ascii()))
            return t

        table = Table(
            title=title,
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold",
            pad_edge=False,
            show_edge=True,
        )
        for i, c in enumerate(cols):
            justify = "left" if c in ("stratum", "group") else "right"
            table.add_column(c, justify=justify, overflow="fold")
        for r in rows:
            table.add_row(*[str(x) for x in r])
        return table

    def __rich_console__(self, console, options):
        try:
            tbl = self.to_rich_table()
            yield make_panel([tbl], title="Sample Size", obj=self, kind="size")
        except Exception:
            from rich.text import Text

            yield Text(self.__plain_str__())

    def _allocation_frame(self):
        import polars as pl

        rows = self._allocation or []
        neyman = any(a.sigma is not None for a in rows)
        data: dict[str, list] = {
            "stratum": [_stratum_label(a.stratum) for a in rows],
            "pop_size": [float(a.pop_size) for a in rows],
        }
        if neyman:
            data["sigma"] = [float(a.sigma) for a in rows]  # type: ignore[arg-type]
        data["n"] = [a.n for a in rows]
        return pl.DataFrame(data)

    # ---------- properties ----------

    @property
    def target(self) -> Target | None:
        return self._target

    @property
    def size(self) -> Size | list[Size] | None:
        return self._size

    @property
    def allocation(self) -> list[Allocation] | None:
        """The strata of an ``allocate()`` result, one row each; None otherwise."""
        return None if self._allocation is None else list(self._allocation)

    @property
    def n(self) -> int | float | dict[object, int | float] | None:
        """The sample size to draw: a number, or ``{stratum: n_h}`` when stratified.

        Pass it as ``n=`` to ``sample.sampling.srs()``, ``pps_sys()``, etc.
        Stratum keys are those given to the goal (tuples for several stratum
        columns), which the selectors accept as is.
        """
        if self._allocation is not None:
            return {a.stratum: a.n for a in self._allocation}
        objs = self._iter_sizes()
        if not objs:
            return None
        if len(objs) == 1 and objs[0].stratum is None:
            return objs[0].n
        return {s.stratum: s.n for s in objs}

    @property
    def param(self) -> PopParam | None:
        return copy.copy(self._param)

    # ---------- delegation methods ----------

    def estimate_prop(
        self,
        p: Number | DomainScalarMap,
        moe: Number | DomainScalarMap,
        *,
        pop_size: Number | DomainScalarMap | None = None,
        method: Literal["wald", "fleiss"] | DomainScalarMap = "wald",
        alpha: Number | DomainScalarMap = 0.05,
        deff: Number | DomainScalarMap = 1.0,
        resp_rate: Number | DomainScalarMap = 1.0,
    ) -> "SampleSize":
        self._allocation = None
        return _estimate_prop(
            self,
            p,
            moe,
            pop_size=pop_size,
            method=method,
            alpha=alpha,
            deff=deff,
            resp_rate=resp_rate,
        )

    def estimate_mean(
        self,
        sigma: Number | DomainScalarMap,
        moe: Number | DomainScalarMap,
        *,
        pop_size: Number | DomainScalarMap | None = None,
        method: Literal["wald"] = "wald",
        alpha: Number | DomainScalarMap = 0.05,
        deff: Number | DomainScalarMap = 1.0,
        resp_rate: Number | DomainScalarMap = 1.0,
    ) -> "SampleSize":
        self._allocation = None
        return _estimate_mean(
            self,
            sigma,
            moe,
            pop_size=pop_size,
            method=method,
            alpha=alpha,
            deff=deff,
            resp_rate=resp_rate,
        )

    def compare_props(
        self,
        p1: Number | DomainScalarMap,
        p2: Number | DomainScalarMap,
        *,
        pop_size: Number | DomainScalarMap | None = None,
        two_sides: bool = True,
        delta: Number | DomainScalarMap = 0.0,
        alloc_ratio: Number | DomainScalarMap = 1.0,
        method: Literal["wald"] = "wald",
        alpha: Number | DomainScalarMap = 0.05,
        power: Number | DomainScalarMap = 0.80,
        var_mode: Literal["alt-props", "pooled-prop", "max-var"] = "alt-props",
        deff: Number | DomainScalarMap = 1.0,
        resp_rate: Number | DomainScalarMap = 1.0,
        group_labels: list[str] | None = None,
    ) -> "SampleSize":
        if group_labels is not None:
            self._group_labels = group_labels
        self._allocation = None
        return _compare_props(
            self,
            p1,
            p2,
            pop_size=pop_size,
            two_sides=two_sides,
            delta=delta,
            alloc_ratio=alloc_ratio,
            method=method,
            alpha=alpha,
            power=power,
            var_mode=var_mode,
            deff=deff,
            resp_rate=resp_rate,
        )

    def compare_means(
        self,
        mu1: Number | DomainScalarMap,
        mu2: Number | DomainScalarMap,
        sigma1: Number | DomainScalarMap,
        sigma2: Number | DomainScalarMap | None = None,
        *,
        pop_size: Number | DomainScalarMap | None = None,
        two_sides: bool = True,
        delta: Number | DomainScalarMap = 0.0,
        alloc_ratio: Number | DomainScalarMap = 1.0,
        method: Literal["wald"] = "wald",
        alpha: Number | DomainScalarMap = 0.05,
        power: Number | DomainScalarMap = 0.80,
        deff: Number | DomainScalarMap = 1.0,
        resp_rate: Number | DomainScalarMap = 1.0,
        group_labels: list[str] | None = None,
    ) -> "SampleSize":
        if group_labels is not None:
            self._group_labels = group_labels
        self._allocation = None
        return _compare_means(
            self,
            mu1,
            mu2,
            sigma1,
            sigma2,
            pop_size=pop_size,
            two_sides=two_sides,
            delta=delta,
            alloc_ratio=alloc_ratio,
            method=method,
            alpha=alpha,
            power=power,
            deff=deff,
            resp_rate=resp_rate,
        )

    def allocate(
        self,
        n: int | None = None,
        *,
        pop_size: Mapping[object, Number],
        method: AllocationMethod = "proportional",
        sigma: Number | Mapping[object, Number] | None = None,
        power: Number = 1.0,
        min_n: int = 1,
        cap_at_population: bool = True,
    ) -> "SampleSize":
        """
        Split an overall sample size ``n`` across strata.

        Parameters
        ----------
        n : int | None
            Overall sample size. Omitted, it is the n of the goal this
            ``SampleSize`` holds, rounded up:
            ``SampleSize().estimate_mean(...).allocate(pop_size=...)``. That
            goal must give one overall n (scalar inputs, not a comparison).
        pop_size : dict
            ``{stratum: N_h}``: unit counts, or any size totals the split
            should follow (households per stratum for a PPS design). Keys are
            any stratum values, tuples for several stratum columns.
        method : {"proportional", "neyman", "equal"}
            ``n_h`` proportional to ``N_h``, to ``N_h * S_h``, or ``n / H``.
        sigma : number | dict | None
            ``S_h``, each stratum's standard deviation of the target variable
            (Neyman only), e.g. from a previous survey.
        power : float
            Proportional to ``N_h ** power``; 0.5 is square-root allocation.
        min_n : int
            Minimum per non-empty stratum.
        cap_at_population : bool
            Cap ``n_h`` at ``N_h``, redistributing the surplus.

        Every method rounds by largest remainder, so the strata sum to ``n``.
        ``.n`` gives ``{stratum: n_h}`` for ``srs(n=...)``, ``pps_sys(n=...)``.

        Examples
        --------
        >>> ss = svy.SampleSize().allocate(400, pop_size={"North": 52_000, "South": 31_000})
        >>> ss.n
        {'North': 251, 'South': 149}
        >>> svy.SampleSize().allocate(400, pop_size=N_h, method="neyman", sigma=S_h)
        >>> svy.SampleSize().allocate(400, pop_size=households, power=0.5)
        >>> svy.SampleSize().estimate_mean(sigma=12, moe=1.5).allocate(pop_size=N_h)
        """
        return _allocate(
            self,
            n,
            pop_size=pop_size,
            method=method,
            sigma=sigma,
            power=power,
            min_n=min_n,
            cap_at_population=cap_at_population,
        )
