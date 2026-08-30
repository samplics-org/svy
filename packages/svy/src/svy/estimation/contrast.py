# src/svy/estimation/contrast.py
"""Post-estimation linear contrasts.

One core: given named estimates with their covariance matrix, a set of sparse
linear contrasts ``L`` yields ``L θ̂`` with variance ``L V Lᵀ`` and t-based
inference on the design degrees of freedom. Both result families expose it —
``Estimate.contrast()`` over domain/level estimates and ``GLMFit.contrast()``
over model coefficients — mirroring R's ``svycontrast``, which operates on any
(coef, vcov) pair without touching the design again.
"""

from __future__ import annotations

import logging

from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping, Sequence

import msgspec
import numpy as np
import polars as pl

from svy.errors import MethodError
from svy.ui.printing import (
    make_panel,
    render_plain_table,
    render_rich_to_str,
    resolve_width,
)


log = logging.getLogger(__name__)


# =============================================================================
# Contrast expressions
# =============================================================================


def _nonlinear(op: str) -> MethodError:
    return MethodError(
        title="Nonlinear contrast not supported",
        detail=(
            f"The operation {op!r} between estimands makes the contrast "
            "nonlinear. Only linear combinations (+, -, and multiplication "
            "by a number) are currently supported; nonlinear (delta-method) "
            "contrasts are planned. The one common case has a linear recipe: "
            "estimate a ratio on the log scale, then exponentiate the bounds."
        ),
        code="CONTRAST_NONLINEAR",
        where="contrast",
    )


class ContrastExpr:
    """A symbolic linear combination of estimands.

    Built from :func:`est` references with ``+``, ``-``, and multiplication
    or division by a number. The tree is owned by svy (not delegated to any
    engine), so richer operators — and their symbolic derivatives for
    delta-method contrasts — can be added without changing this API.
    """

    __slots__ = ()

    def __add__(self, other: "ContrastExpr") -> "ContrastExpr":
        if isinstance(other, ContrastExpr):
            return _Add(self, other)
        raise (
            _nonlinear("+ constant")
            if isinstance(other, Real)
            else TypeError(f"cannot add {type(other).__name__} to a contrast expression")
        )

    def __radd__(self, other: Any) -> "ContrastExpr":
        return self.__add__(other)

    def __sub__(self, other: "ContrastExpr") -> "ContrastExpr":
        if isinstance(other, ContrastExpr):
            return _Add(self, _Scale(-1.0, other))
        raise (
            _nonlinear("- constant")
            if isinstance(other, Real)
            else TypeError(f"cannot subtract {type(other).__name__} from a contrast expression")
        )

    def __rsub__(self, other: Any) -> "ContrastExpr":
        if isinstance(other, ContrastExpr):
            return _Add(other, _Scale(-1.0, self))
        raise (
            _nonlinear("constant -")
            if isinstance(other, Real)
            else TypeError(f"cannot subtract a contrast expression from {type(other).__name__}")
        )

    def __neg__(self) -> "ContrastExpr":
        return _Scale(-1.0, self)

    def __pos__(self) -> "ContrastExpr":
        return self

    def __mul__(self, other: Any) -> "ContrastExpr":
        if isinstance(other, Real):
            return _Scale(float(other), self)
        if isinstance(other, ContrastExpr):
            raise _nonlinear("*")
        raise TypeError(f"cannot multiply a contrast expression by {type(other).__name__}")

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> "ContrastExpr":
        if isinstance(other, Real):
            return _Scale(1.0 / float(other), self)
        if isinstance(other, ContrastExpr):
            raise _nonlinear("/")
        raise TypeError(f"cannot divide a contrast expression by {type(other).__name__}")

    def __rtruediv__(self, other: Any) -> "ContrastExpr":
        raise _nonlinear("/")

    def coefs(self) -> dict[Any, float]:
        """Compile to the sparse ``{key: coefficient}`` form."""
        out: dict[Any, float] = {}
        _accumulate(self, 1.0, out)
        return out


@dataclass(frozen=True, slots=True)
class EstRef(ContrastExpr):
    """A reference to one estimand of a result, by key."""

    key: Any

    def __repr__(self) -> str:
        if isinstance(self.key, tuple):
            return f"estd{self.key!r}"
        return f"estd({self.key!r})"


@dataclass(frozen=True, slots=True)
class _Add(ContrastExpr):
    left: ContrastExpr
    right: ContrastExpr

    def __repr__(self) -> str:
        return f"({self.left!r} + {self.right!r})"


@dataclass(frozen=True, slots=True)
class _Scale(ContrastExpr):
    coef: float
    node: ContrastExpr

    def __repr__(self) -> str:
        return f"({self.coef:g} * {self.node!r})"


def _accumulate(node: ContrastExpr, weight: float, out: dict[Any, float]) -> None:
    if isinstance(node, EstRef):
        out[node.key] = out.get(node.key, 0.0) + weight
    elif isinstance(node, _Add):
        _accumulate(node.left, weight, out)
        _accumulate(node.right, weight, out)
    elif isinstance(node, _Scale):
        _accumulate(node.node, weight * node.coef, out)
    else:  # pragma: no cover — unreachable through the public operators
        raise TypeError(f"unknown contrast expression node: {type(node).__name__}")


def estd(*key: Any) -> EstRef:
    """Reference an estimand by key, for use in a contrast expression.

    The key names one row of a result: a domain level (``estd("E")``), a
    category level (``estd(1)``), a (domain, level) cell (``estd("E", 1)``),
    or a GLM coefficient (``estd("api99")``). Resolution happens inside the
    result's ``contrast()``: labels resolve through the result's metadata,
    and an unknown key fails loudly listing the valid ones — see
    ``result.keys()``.

    Examples
    --------
    >>> r.contrast(estd("E") - estd("H"))
    >>> r.contrast(
    ...     {"trend": -estd(1) + estd(3), "mid vs rest": estd(2) - 0.5 * (estd(1) + estd(3))}
    ... )
    """
    if not key:
        raise TypeError("estd() requires a key identifying an estimand")
    return EstRef(key[0] if len(key) == 1 else tuple(key))


class ContrastEst(msgspec.Struct, frozen=True):
    """One estimated linear contrast."""

    contrast: str
    est: float
    se: float
    cv: float
    lci: float
    uci: float
    t: float
    p_value: float
    df: float

    def to_dict(self) -> dict[str, Any]:
        return msgspec.to_builtins(self)


class Contrast:
    """Result of one or more linear contrasts over a set of estimates.

    Rows print like an :class:`~svy.estimation.estimate.Estimate` table with
    the added t statistic and p-value; the full contrast covariance
    ``L V Lᵀ`` is kept on :attr:`covariance` (row order = printed order).
    """

    __slots__ = ("estimates", "covariance", "alpha", "df", "method", "_print_width")

    def __init__(
        self,
        estimates: list[ContrastEst],
        covariance: np.ndarray,
        *,
        alpha: float,
        df: float,
        method: str,
    ):
        self.estimates = estimates
        self.covariance = covariance
        self.alpha = alpha
        self.df = df
        self.method = method
        self._print_width: int | None = None

    def to_dicts(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self.estimates]

    def to_polars(self) -> pl.DataFrame:
        if not self.estimates:
            return pl.DataFrame()
        return pl.from_dicts(self.to_dicts())

    # --- Rendering (mirrors Estimate's plain/rich split) ---

    @staticmethod
    def _fmt(col: str, v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            if col == "cv":
                return f"{v * 100:.2f}"
            if col == "p_value":
                return f"{v:.4g}"
            if col == "df":
                return f"{v:g}"
            return f"{v:,.4f}"
        return str(v)

    def _rows(self) -> tuple[list[str], list[list[str]]]:
        headers = ["contrast", "est", "se", "cv (%)", "t", "p_value", "lci", "uci"]
        cols = ["contrast", "est", "se", "cv", "t", "p_value", "lci", "uci"]
        rows = [[self._fmt(c, getattr(e, c)) for c in cols] for e in self.estimates]
        return headers, rows

    def _title(self) -> str:
        return f"Contrast ({self.method.upper()}, df={self.df:g})"

    def __plain_str__(self) -> str:
        if not self.estimates:
            return "Contrast — <no estimates>"
        headers, rows = self._rows()
        return "\n".join([self._title(), "", render_plain_table(headers, rows)])

    def __str__(self) -> str:
        try:
            return render_rich_to_str(self, width=resolve_width(self))
        except Exception:
            return self.__plain_str__()

    __repr__ = __str__

    def __rich_console__(self, console, options):
        from rich import box
        from rich.table import Table
        from rich.text import Text

        if not self.estimates:
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
        headers, rows = self._rows()
        for h in headers:
            table.add_column(h, justify="left" if h == "contrast" else "right")
        for r in rows:
            table.add_row(*r)

        title = f"Contrast ([bold]{self.method.upper()}[/bold], df={self.df:g})"
        yield make_panel([table], title=title, obj=self, kind="estimate")


def _normalize_contrasts(
    contrasts: Mapping[Any, Any] | ContrastExpr,
) -> dict[str, dict[Any, float]]:
    """Resolve the accepted input forms to ``{name: {key: coef}}``.

    One contrast is a :class:`ContrastExpr` or a ``{key: coef, ...}`` dict;
    several are ``{name: expr-or-dict, ...}``. The dict forms are told apart
    by the values: all-mapping/expression values mean the named form. Mixing
    coefficient values with named specs is an error.
    """
    if isinstance(contrasts, ContrastExpr):
        return {"contrast": contrasts.coefs()}
    if not isinstance(contrasts, Mapping) or not contrasts:
        raise MethodError(
            title="Invalid contrast specification",
            detail=(
                "contrast() takes a contrast expression (estd(a) - estd(b)), "
                "one contrast as {key: coef, ...}, or several as "
                "{'name': expression-or-dict, ...}."
            ),
            code="CONTRAST_INVALID_SPEC",
            where="contrast",
            param="contrasts",
            got=type(contrasts).__name__,
        )
    is_named = [isinstance(v, (Mapping, ContrastExpr)) for v in contrasts.values()]
    if all(is_named):
        return {
            str(name): spec.coefs() if isinstance(spec, ContrastExpr) else dict(spec)
            for name, spec in contrasts.items()
        }
    if any(is_named):
        raise MethodError(
            title="Mixed contrast specification",
            detail=(
                "Either every value is a contrast (an expression or a "
                "{key: coef} dict, giving several named contrasts) or every "
                "value is a coefficient (giving one contrast); mixing the "
                "two forms is ambiguous."
            ),
            code="CONTRAST_MIXED_SPEC",
            where="contrast",
            param="contrasts",
        )
    return {"contrast": dict(contrasts)}


class KeyResolver:
    """Lookup from a user-typed key to a row index.

    Resolution order: exact key, string-normalized key (``1`` finds ``"1"``),
    then any aliases the caller supplies (metadata value labels). A candidate
    that matches two different rows is dropped rather than guessed at.
    """

    __slots__ = ("keys", "_exact", "_fallback")

    def __init__(self, keys: Sequence[Any], aliases: Mapping[Any, int] | None = None):
        self.keys = list(keys)
        self._exact: dict[Any, int] = {}
        for i, k in enumerate(self.keys):
            self._exact.setdefault(k, i)

        fallback: dict[Any, int] = {}
        dropped: set = set()

        def offer(candidate: Any, idx: int) -> None:
            if candidate in self._exact or candidate in dropped:
                return
            if candidate in fallback and fallback[candidate] != idx:
                dropped.add(candidate)
                fallback.pop(candidate)
                return
            fallback[candidate] = idx

        for i, k in enumerate(self.keys):
            offer(self._norm(k), i)
        for alias, i in (aliases or {}).items():
            offer(alias, i)
            offer(self._norm(alias), i)
        self._fallback = fallback

    @staticmethod
    def _norm(k: Any) -> Any:
        if isinstance(k, tuple):
            return tuple(str(x) for x in k)
        return str(k)

    def resolve(self, k: Any) -> int | None:
        if k in self._exact:
            return self._exact[k]
        if k in self._fallback:
            return self._fallback[k]
        return self._fallback.get(self._norm(k))


def linear_contrast(
    keys: Sequence[Any],
    values: np.ndarray,
    vcov: np.ndarray,
    contrasts: Mapping[Any, Any] | ContrastExpr,
    *,
    df: float,
    alpha: float,
    method: str,
    aliases: Mapping[Any, int] | None = None,
) -> Contrast:
    """``L θ̂`` / ``L V Lᵀ`` over named estimates.

    Sparse specification: keys not mentioned in a contrast get coefficient 0;
    unknown keys fail loudly (R ``svycontrast`` parity — a typo must not
    silently drop a term). A contrast placing nonzero weight on an NA estimate
    yields an NA row; the other contrasts are unaffected.
    """
    from scipy import stats

    named = _normalize_contrasts(contrasts)
    k = len(keys)
    est_arr = np.asarray(values, dtype=float)
    vcov = np.asarray(vcov, dtype=float)
    if vcov.shape != (k, k):
        raise MethodError(
            title="Covariance shape mismatch",
            detail=f"Expected a {k}x{k} covariance matrix, got {vcov.shape}.",
            code="CONTRAST_COV_SHAPE",
            where="contrast",
        )

    resolver = KeyResolver(keys, aliases)
    m = len(named)
    L = np.zeros((m, k))
    for row, (name, spec) in enumerate(named.items()):
        unknown = []
        for key, coef in spec.items():
            idx = resolver.resolve(key)
            if idx is None:
                unknown.append(key)
            else:
                L[row, idx] = float(coef)
        if unknown:
            raise MethodError(
                title="Unknown contrast key",
                detail=(
                    f"Contrast {name!r} references {unknown!r}, which do(es) "
                    f"not identify any estimate. Valid keys: {resolver.keys!r}."
                ),
                code="CONTRAST_UNKNOWN_KEY",
                where="contrast",
                param="contrasts",
                got=unknown,
                expected=resolver.keys,
            )

    # NA propagation: any touched NA estimate (or NA variance) poisons only
    # the contrasts touching it, mirroring R's contrast() NA handling. NAs
    # are zeroed before the products (0-coefficient rows must stay clean —
    # matmul would smear 0·NaN = NaN everywhere) and re-poisoned after.
    bad = ~np.isfinite(est_arr) | ~np.isfinite(np.diag(vcov))
    touched_bad = (L[:, bad] != 0).any(axis=1) if bad.any() else np.zeros(m, dtype=bool)
    if bad.any():
        est_arr = np.where(bad, 0.0, est_arr)
        vcov = np.where(np.isfinite(vcov), vcov, 0.0)

    c_est = L @ est_arr
    c_cov = L @ vcov @ L.T
    c_var = np.diag(c_cov).copy()
    c_est[touched_bad] = np.nan
    c_var[touched_bad] = np.nan

    se = np.sqrt(np.maximum(c_var, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(c_est != 0, se / c_est, np.inf)
        t_vals = np.where(se > 0, c_est / se, np.nan)
    t_crit = float(stats.t.ppf(1 - alpha / 2, df)) if df > 0 else float("nan")
    p_vals = 2.0 * stats.t.sf(np.abs(t_vals), df)
    lci = c_est - t_crit * se
    uci = c_est + t_crit * se

    rows = [
        ContrastEst(
            contrast=name,
            est=float(c_est[i]),
            se=float(se[i]),
            cv=float(cv[i]),
            lci=float(lci[i]),
            uci=float(uci[i]),
            t=float(t_vals[i]),
            p_value=float(p_vals[i]),
            df=float(df),
        )
        for i, name in enumerate(named)
    ]
    return Contrast(rows, c_cov, alpha=alpha, df=float(df), method=method)
