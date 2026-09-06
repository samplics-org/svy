# src/svy/estimation/contrast.py
"""Post-estimation contrasts.

One core: given named estimates with their covariance matrix, a contrast
``f(θ)`` yields ``f(θ̂)`` with variance ``gᵀ V g`` (``g`` the gradient at
``θ̂``) and t-based inference on the design degrees of freedom. A linear
contrast is the special case ``g = L``, giving ``L θ̂`` and ``L V Lᵀ``
exactly; a nonlinear one (ratio, percent change, log-ratio) is the delta
method. Both result families expose it — ``Estimate.contrast()`` over
domain/level estimates and ``GLMFit.contrast()`` over model coefficients —
mirroring R's ``svycontrast``, which operates on any (coef, vcov) pair
without touching the design again.
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


class ContrastExpr:
    """A symbolic function of estimands.

    Built from :func:`estd` references with ``+``, ``-``, ``*`` and ``/``
    (between expressions or with numbers) and the :meth:`log` / :meth:`exp`
    methods. Trees holding only sums and scalings are linear and take the
    exact ``L V Lᵀ`` path; anything else is estimated by the delta method
    from the analytic gradient. The tree is owned by svy (not delegated to
    any engine), so further operators can be added without changing this
    API.
    """

    __slots__ = ()

    @staticmethod
    def _wrap(other: Any, what: str) -> "ContrastExpr":
        if isinstance(other, ContrastExpr):
            return other
        if isinstance(other, Real) and not isinstance(other, bool):
            return _Const(float(other))
        raise TypeError(f"cannot {what} a contrast expression and {type(other).__name__}")

    def __add__(self, other: Any) -> "ContrastExpr":
        return _Add(self, self._wrap(other, "add"))

    def __radd__(self, other: Any) -> "ContrastExpr":
        return _Add(self._wrap(other, "add"), self)

    def __sub__(self, other: Any) -> "ContrastExpr":
        return _Add(self, _Scale(-1.0, self._wrap(other, "subtract")))

    def __rsub__(self, other: Any) -> "ContrastExpr":
        return _Add(self._wrap(other, "subtract"), _Scale(-1.0, self))

    def __neg__(self) -> "ContrastExpr":
        return _Scale(-1.0, self)

    def __pos__(self) -> "ContrastExpr":
        return self

    def __mul__(self, other: Any) -> "ContrastExpr":
        if isinstance(other, Real) and not isinstance(other, bool):
            return _Scale(float(other), self)
        return _Mul(self, self._wrap(other, "multiply"))

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> "ContrastExpr":
        if isinstance(other, Real) and not isinstance(other, bool):
            return _Scale(1.0 / float(other), self)
        return _Div(self, self._wrap(other, "divide"))

    def __rtruediv__(self, other: Any) -> "ContrastExpr":
        return _Div(self._wrap(other, "divide"), self)

    def log(self) -> "ContrastExpr":
        """Natural log of the expression (a log-ratio is ``a.log() - b.log()``)."""
        return _Log(self)

    def exp(self) -> "ContrastExpr":
        return _Exp(self)

    def is_linear(self) -> bool:
        """True when the tree holds only references, sums and scalings."""
        return _is_linear(self)

    def coefs(self) -> dict[Any, float]:
        """Compile a linear tree to the sparse ``{key: coefficient}`` form."""
        if not self.is_linear():
            raise MethodError(
                title="Not a linear contrast",
                detail=f"{self!r} is nonlinear and has no coefficient form.",
                code="CONTRAST_NONLINEAR",
                where="contrast",
            )
        out: dict[Any, float] = {}
        _accumulate(self, 1.0, out)
        return out

    def keys(self) -> list[Any]:
        """The estimand keys the expression references, in first-seen order."""
        out: list[Any] = []
        _collect_keys(self, out)
        return list(dict.fromkeys(out))


@dataclass(frozen=True, slots=True)
class EstRef(ContrastExpr):
    """A reference to one estimand of a result, by key."""

    key: Any

    def __repr__(self) -> str:
        if isinstance(self.key, tuple):
            return f"estd{self.key!r}"
        return f"estd({self.key!r})"


@dataclass(frozen=True, slots=True)
class _Const(ContrastExpr):
    value: float

    def __repr__(self) -> str:
        return f"{self.value:g}"


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


@dataclass(frozen=True, slots=True)
class _Mul(ContrastExpr):
    left: ContrastExpr
    right: ContrastExpr

    def __repr__(self) -> str:
        return f"({self.left!r} * {self.right!r})"


@dataclass(frozen=True, slots=True)
class _Div(ContrastExpr):
    left: ContrastExpr
    right: ContrastExpr

    def __repr__(self) -> str:
        return f"({self.left!r} / {self.right!r})"


@dataclass(frozen=True, slots=True)
class _Log(ContrastExpr):
    node: ContrastExpr

    def __repr__(self) -> str:
        return f"log({self.node!r})"


@dataclass(frozen=True, slots=True)
class _Exp(ContrastExpr):
    node: ContrastExpr

    def __repr__(self) -> str:
        return f"exp({self.node!r})"


def _is_linear(node: ContrastExpr) -> bool:
    if isinstance(node, EstRef):
        return True
    if isinstance(node, _Add):
        return _is_linear(node.left) and _is_linear(node.right)
    if isinstance(node, _Scale):
        return _is_linear(node.node)
    return False


def _accumulate(node: ContrastExpr, weight: float, out: dict[Any, float]) -> None:
    if isinstance(node, EstRef):
        out[node.key] = out.get(node.key, 0.0) + weight
    elif isinstance(node, _Add):
        _accumulate(node.left, weight, out)
        _accumulate(node.right, weight, out)
    elif isinstance(node, _Scale):
        _accumulate(node.node, weight * node.coef, out)
    else:  # pragma: no cover — guarded by is_linear()
        raise TypeError(f"not a linear node: {type(node).__name__}")


def _collect_keys(node: ContrastExpr, out: list[Any]) -> None:
    if isinstance(node, EstRef):
        out.append(node.key)
    elif isinstance(node, (_Add, _Mul, _Div)):
        _collect_keys(node.left, out)
        _collect_keys(node.right, out)
    elif isinstance(node, (_Scale, _Log, _Exp)):
        _collect_keys(node.node, out)


def _evaluate(
    node: ContrastExpr, theta: np.ndarray, index: Mapping[Any, int]
) -> tuple[float, np.ndarray]:
    """Value and gradient of the tree at ``theta`` (product, quotient and
    chain rules), the delta method's two ingredients."""
    k = len(theta)
    if isinstance(node, EstRef):
        g = np.zeros(k)
        g[index[node.key]] = 1.0
        return float(theta[index[node.key]]), g
    if isinstance(node, _Const):
        return node.value, np.zeros(k)
    if isinstance(node, _Add):
        va, ga = _evaluate(node.left, theta, index)
        vb, gb = _evaluate(node.right, theta, index)
        return va + vb, ga + gb
    if isinstance(node, _Scale):
        v, g = _evaluate(node.node, theta, index)
        return node.coef * v, node.coef * g
    if isinstance(node, _Mul):
        va, ga = _evaluate(node.left, theta, index)
        vb, gb = _evaluate(node.right, theta, index)
        return va * vb, va * gb + vb * ga
    if isinstance(node, _Div):
        va, ga = _evaluate(node.left, theta, index)
        vb, gb = _evaluate(node.right, theta, index)
        if vb == 0.0:
            raise MethodError(
                title="Division by a zero estimate",
                detail=f"The denominator {node.right!r} evaluates to 0, so {node!r} is undefined.",
                code="CONTRAST_DIV_ZERO",
                where="contrast",
            )
        return va / vb, (ga * vb - va * gb) / (vb * vb)
    if isinstance(node, _Log):
        v, g = _evaluate(node.node, theta, index)
        if v <= 0.0:
            raise MethodError(
                title="Log of a non-positive estimate",
                detail=f"{node.node!r} evaluates to {v:g}, so {node!r} is undefined.",
                code="CONTRAST_LOG_DOMAIN",
                where="contrast",
            )
        return float(np.log(v)), g / v
    if isinstance(node, _Exp):
        v, g = _evaluate(node.node, theta, index)
        e = float(np.exp(v))
        return e, e * g
    raise TypeError(f"unknown contrast expression node: {type(node).__name__}")


def estd(*key: Any) -> EstRef:
    """Reference an estimand by key, for use in a contrast expression.

    The key names one row of a result: a domain level (``estd("E")``), a
    category level (``estd(1)``), a (domain, level) cell (``estd("E", 1)``),
    or a GLM coefficient (``estd("api99")``). Resolution happens inside the
    result's ``contrast()``: labels resolve through the result's metadata,
    and an unknown key fails loudly listing the valid ones — see
    ``result.keys()``.

    Linear combinations are exact; ratios, products, ``log()`` and ``exp()``
    are estimated by the delta method on the same design df.

    Examples
    --------
    >>> r.contrast(estd("E") - estd("H"))
    >>> r.contrast(
    ...     {"trend": -estd(1) + estd(3), "mid vs rest": estd(2) - 0.5 * (estd(1) + estd(3))}
    ... )
    >>> r.contrast({"ratio": estd(2) / estd(1), "pct change": estd(2) / estd(1) - 1})
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
) -> dict[str, dict[Any, float] | ContrastExpr]:
    """Resolve the accepted input forms to ``{name: spec}``.

    A spec is a :class:`ContrastExpr` or a linear ``{key: coef, ...}`` dict;
    several are ``{name: expr-or-dict, ...}``. The dict forms are told apart
    by the values: all-mapping/expression values mean the named form. Mixing
    coefficient values with named specs is an error.
    """
    if isinstance(contrasts, ContrastExpr):
        return {"contrast": contrasts}
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
            str(name): spec if isinstance(spec, ContrastExpr) else dict(spec)
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
        # by-levels arrive stringified from the kernel, so an int typed by
        # the caller only meets its row through the normalized form.
        nk = self._norm(k)
        if nk in self._exact:
            return self._exact[nk]
        return self._fallback.get(nk)


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
    """``f(θ̂)`` with variance ``gᵀ V g`` over named estimates.

    Sparse specification: keys not mentioned in a contrast get coefficient 0;
    unknown keys fail loudly (R ``svycontrast`` parity — a typo must not
    silently drop a term). Linear contrasts use their coefficient row as the
    gradient, so ``L θ̂`` / ``L V Lᵀ`` is reproduced exactly; nonlinear ones
    are delta-method estimates. A contrast touching an NA estimate yields an
    NA row; the other contrasts are unaffected.
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

    def resolve_all(name: str, spec_keys: Sequence[Any]) -> dict[Any, int]:
        index: dict[Any, int] = {}
        unknown = []
        for key in spec_keys:
            idx = resolver.resolve(key)
            if idx is None:
                unknown.append(key)
            else:
                index[key] = idx
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
        return index

    # NA propagation: any touched NA estimate (or NA variance) poisons only
    # the contrasts touching it, mirroring R's contrast() NA handling. NAs
    # are zeroed before the products (0-coefficient rows must stay clean —
    # matmul would smear 0·NaN = NaN everywhere) and re-poisoned after.
    bad = ~np.isfinite(est_arr) | ~np.isfinite(np.diag(vcov))
    theta = np.where(bad, 0.0, est_arr)
    vcov = np.where(np.isfinite(vcov), vcov, 0.0)

    m = len(named)
    G = np.zeros((m, k))
    c_est = np.zeros(m)
    touched_bad = np.zeros(m, dtype=bool)
    for row, (name, spec) in enumerate(named.items()):
        if isinstance(spec, ContrastExpr):
            index = resolve_all(name, spec.keys())
            touched_bad[row] = any(bad[i] for i in index.values())
            if touched_bad[row]:
                continue
            c_est[row], G[row] = _evaluate(spec, theta, index)
        else:
            index = resolve_all(name, list(spec))
            for key, coef in spec.items():
                G[row, index[key]] = float(coef)
            touched_bad[row] = bool((G[row, bad] != 0).any()) if bad.any() else False
            c_est[row] = G[row] @ theta

    c_cov = G @ vcov @ G.T
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
