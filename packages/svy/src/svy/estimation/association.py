"""Design-based association estimators: correlation and covariance.

Both statistics are symmetric, so unlike ``ratio`` there is no numerator or
denominator here: a call names a *set* of columns and every requested pair is
reported as its own row. The ``y``/``x`` fields on the resulting ``ParamEst``
hold the two column names positionally and carry no directional meaning.

Confidence intervals differ between the two because their ranges do. A
correlation is bounded in [-1, 1], so the default interval is built on Fisher's
z scale and transformed back — the same move ``prop`` makes with a logit, and
for the same reason: a symmetric Wald interval can otherwise spill past the
bound. A covariance is unbounded, so it takes the plain Wald interval.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import svy_rs as rs

from svy.core.enumerations import EstimationMethod, PopParam
from svy.errors import DimensionError, MethodError

from .estimate import Estimate, ParamEst
from .replication import _get_rep_params, get_rep_method_str


if TYPE_CHECKING:
    from svy.core.data_prep import PreparedData

    from .base import Estimation


#: Coefficients ``kind=`` accepts today. Widening this tuple is backward
#: compatible; ``_KNOWN_KINDS`` below is what keeps an unimplemented but
#: recognised name reporting "not supported yet" instead of "invalid".
SUPPORTED_KINDS = ("pearson",)

#: Recognised but not yet implemented — named so the error can say so.
_PLANNED_KINDS = ("spearman", "kendall")


def normalize_kind(kind: str) -> str:
    """Canonicalise ``kind``, distinguishing "not yet" from "no such thing"."""
    k = str(kind).lower().replace("_", "-").strip()
    if k in ("pearson", "corr", "correlation"):
        return "pearson"
    if k in _PLANNED_KINDS:
        raise MethodError(
            title=f"Correlation kind {k!r} is not supported yet",
            detail=(
                f"Only {', '.join(SUPPORTED_KINDS)} is implemented today. "
                f"{k!r} is planned but needs its own linearization: its ranks "
                "are themselves estimated, so the Pearson score would "
                "understate the variance."
            ),
            code="CORR_KIND_NOT_IMPLEMENTED",
            where="estimation.corr",
            param="kind",
            expected=f"one of {SUPPORTED_KINDS}",
            got=k,
            hint="Use kind='pearson', or track the issue for rank-based support.",
        )
    raise MethodError(
        title=f"Unknown correlation kind {k!r}",
        detail=f"kind must be one of {SUPPORTED_KINDS}.",
        code="CORR_KIND_UNKNOWN",
        where="estimation.corr",
        param="kind",
        expected=f"one of {SUPPORTED_KINDS}",
        got=k,
        hint="Use kind='pearson'.",
    )


def guard_pandas_method(method: object) -> None:
    """Redirect ``method='spearman'`` to ``kind=``.

    pandas spells the coefficient ``method=``; here ``method`` selects the
    variance estimator. Someone reaching for muscle memory should be told which
    argument they wanted, not handed a message about Taylor linearization.
    """
    if isinstance(method, str) and method.lower() in ("pearson", *_PLANNED_KINDS):
        raise MethodError(
            title=f"{method!r} is a correlation kind, not a variance method",
            detail=(
                "In svy, method= selects the variance estimator ('taylor' or "
                "'replication'); the coefficient is chosen with kind=. pandas "
                "spells the coefficient method=, which is the likely mix-up."
            ),
            code="CORR_METHOD_IS_KIND",
            where="estimation.corr",
            param="method",
            expected="'taylor' or 'replication'",
            got=method,
            hint=f"Use kind={method!r} instead of method={method!r}.",
        )


def parse_cols(cols: object, *, where: str) -> list[tuple[str, str]]:
    """Resolve the ``cols`` argument into an ordered list of column pairs.

    Three spellings, disambiguated by element type, and agreeing wherever they
    overlap:

    - ``("a", "b")`` or ``["a", "b"]`` — that single pair
    - ``["a", "b", "c"]`` — every unique pair, in ``i < j`` order
    - ``[("a", "b"), ("a", "c")]`` — exactly those pairs

    A flat list yields off-diagonal pairs only. Self-pairs are never implied,
    for covariance as much as correlation, so the two stay consistent; ask for
    a variance explicitly with ``cov(("a", "a"))``.
    """

    def _fail(detail: str, got: object, hint: str) -> DimensionError:
        return DimensionError(
            title="Invalid cols argument",
            detail=detail,
            code="ASSOC_COLS_INVALID",
            where=where,
            param="cols",
            expected="a pair, a list of columns, or a list of pairs",
            got=got,
            hint=hint,
        )

    if isinstance(cols, str):
        raise _fail(
            "A single column name has nothing to pair with.",
            cols,
            "Pass at least two columns, e.g. cols=('a', 'b').",
        )
    if not isinstance(cols, Sequence):
        raise _fail("cols must be a sequence.", type(cols).__name__, "Pass a tuple or list.")

    items = list(cols)
    if not items:
        raise _fail("cols is empty.", items, "Pass at least two columns.")

    if all(isinstance(c, str) for c in items):
        if len(items) < 2:
            raise _fail(
                "A single column name has nothing to pair with.",
                items,
                "Pass at least two columns.",
            )
        if len(items) == 2:
            # An explicit pair, so a self-pair is meaningful: cov(("a", "a"))
            # is the variance, and corr(("a", "a")) is 1 by construction.
            return [(items[0], items[1])]
        if len(set(items)) != len(items):
            # Beyond two columns the list expands to all unique pairs, where a
            # repeat would silently generate a degenerate self-pair.
            raise _fail(
                "cols contains duplicate column names.",
                items,
                "Remove duplicates, or request a self-pair explicitly as a tuple.",
            )
        return list(combinations(items, 2))

    pairs: list[tuple[str, str]] = []
    for item in items:
        if isinstance(item, str) or not isinstance(item, Sequence) or len(item) != 2:
            raise _fail(
                "cols mixes column names and pairs, or contains an entry that "
                "is not a pair of two column names.",
                item,
                "Use either a flat list of columns or a list of 2-tuples.",
            )
        a, b = item
        if not (isinstance(a, str) and isinstance(b, str)):
            raise _fail("Each pair must hold two column names.", item, "Use strings.")
        pairs.append((a, b))
    return pairs


def _fisher_ci(
    est: np.ndarray, se: np.ndarray, t_crit: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fisher z interval for a correlation, back-transformed to [-1, 1].

    ``z = arctanh(r)`` with ``se_z = se / (1 - r^2)`` by the delta method, then
    ``tanh(z +/- t*se_z)``. Rows at ``|r| >= 1`` have no z to work on, so the
    interval degenerates to the point estimate rather than producing infinities.
    """
    lci = est.copy()
    uci = est.copy()
    valid = np.isfinite(est) & np.isfinite(se) & (np.abs(est) < 1.0)
    if valid.any():
        r, s, t = est[valid], se[valid], t_crit[valid]
        with np.errstate(divide="ignore", invalid="ignore"):
            se_z = s / (1.0 - r * r)
        z = np.arctanh(r)
        lci[valid] = np.tanh(z - t * se_z)
        uci[valid] = np.tanh(z + t * se_z)
    return lci, uci


def result_to_param_est(
    est: Estimation,
    result_df: pl.DataFrame,
    param: PopParam,
    alpha: float,
    deff: bool,
    by_col: str | None,
    ci_method: str,
) -> list[ParamEst]:
    """Build one ``ParamEst`` per result row.

    Distinct from ``_polars_result_to_param_est`` because there each row is a
    by-group of one variable, while here every row carries its own column pair.
    """
    n_rows = result_df.height
    if n_rows == 0:
        return []

    est_arr = result_df["est"].to_numpy()
    se_arr = result_df["se"].to_numpy()
    df_arr = result_df["df"].to_numpy().astype(np.float64)
    deff_arr = result_df["deff"].to_numpy() if (deff and "deff" in result_df.columns) else None
    t_crits = est._t_crit_arr(alpha, df_arr)

    with np.errstate(divide="ignore", invalid="ignore"):
        cv_arr = np.where(est_arr != 0, se_arr / est_arr, np.inf)

    if param == PopParam.CORR and ci_method == "fisher":
        lci_arr, uci_arr = _fisher_ci(est_arr, se_arr, t_crits)
    else:
        lci_arr = est_arr - t_crits * se_arr
        uci_arr = est_arr + t_crits * se_arr

    y_names = result_df["y"].to_list()
    x_names = result_df["x"].to_list()
    by_tuple = (by_col,) if by_col else None
    by_levels: list = [None] * n_rows
    if by_col and by_col in result_df.columns:
        by_levels = [(v,) for v in result_df[by_col].to_list()]

    return [
        ParamEst(
            y=y_names[i],
            x=x_names[i],
            est=float(est_arr[i]),
            se=float(se_arr[i]),
            cv=float(cv_arr[i]),
            lci=float(lci_arr[i]),
            uci=float(uci_arr[i]),
            deff=float(deff_arr[i]) if deff_arr is not None else None,
            df=int(df_arr[i]),
            by=by_tuple,
            by_level=by_levels[i],
        )
        for i in range(n_rows)
    ]


def _kind_arg(param: PopParam) -> str:
    return "corr" if param == PopParam.CORR else "cov"


def taylor_assoc(
    est: Estimation,
    prep: PreparedData,
    pairs: list[tuple[str, str]],
    param: PopParam,
    *,
    deff_ref: str | None = None,
    alpha: float = 0.05,
    ci_method: str = "fisher",
) -> Estimate:
    """Taylor-linearized correlation or covariance over ``pairs``."""
    pop_size = getattr(est._sample._design, "pop_size", None)
    df, fpc_col, fpc_ssu_col = (
        est._compute_fpc_columns(prep.df, pop_size, prep.strata_col, prep.psu_col, prep.ssu_col)
        if pop_size is not None
        else (prep.df, None, None)
    )

    df = est._ensure_float64(df, sorted({c for p in pairs for c in p}))
    result_df = rs.taylor_assoc(
        df,
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        _kind_arg(param),
        prep.weight_col,
        strata_col=prep.strata_col,
        psu_col=prep.psu_col,
        ssu_col=prep.ssu_col,
        fpc_col=fpc_col,
        fpc_ssu_col=fpc_ssu_col,
        by_col=prep.by_col,
        singleton_method=est._get_center_method(),
        deff_ref=deff_ref,
    )

    est_list = result_to_param_est(
        est, result_df, param, alpha, deff_ref is not None, prep.by_col, ci_method
    )
    est_cov = np.diag(result_df["var"].to_numpy())
    return est._build_estimate_result_light(
        est_list,
        est_cov,
        param,
        alpha,
        prep.by_cols,
        as_factor=False,
        method=EstimationMethod.TAYLOR,
            deff_ref=deff_ref,
    )


def replicate_assoc(
    est: Estimation,
    prep: PreparedData,
    pairs: list[tuple[str, str]],
    param: PopParam,
    method: EstimationMethod,
    *,
    fay_coef: float = 0.0,
    variance_center: str = "rep_mean",
    alpha: float = 0.05,
    ci_method: str = "fisher",
) -> Estimate:
    """Replicate-weight correlation or covariance over ``pairs``."""
    rep_weight_cols, df_val, final_fay, rscales = _get_rep_params(est, fay_coef)
    data = est._ensure_float64(prep.df, [*rep_weight_cols, *sorted({c for p in pairs for c in p})])
    result_df = rs.replicate_assoc(
        data,
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        _kind_arg(param),
        weight_col=prep.weight_col,
        rep_weight_cols=rep_weight_cols,
        method=get_rep_method_str(method),
        fay_coef=final_fay,
        rscales=rscales,
        center=variance_center,
        degrees_of_freedom=df_val,
        by_col=prep.by_col,
        domain_mask_col=prep.domain_mask_col,
    )

    # The replicate kernel reports no deff (no SRS reference on that path),
    # so `deff` is never populated here.
    est_list = result_to_param_est(est, result_df, param, alpha, False, prep.by_col, ci_method)
    est_cov = np.diag(result_df["var"].to_numpy())
    return est._build_estimate_result_light(
        est_list,
        est_cov,
        param,
        alpha,
        prep.by_cols,
        as_factor=False,
        method=method,
    )
