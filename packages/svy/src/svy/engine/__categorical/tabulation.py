# src/svy/engine/categorical/tabulation.py
from __future__ import annotations

import re

from typing import Any

import numpy as np
import polars as pl

from scipy.stats import chi2, f

from svy.categorical.table import CellEst, TableStats
from svy.core.containers import ChiSquare, FDist
from svy.core.enumerations import PopParam
from svy.engine.estimation.taylor import _estimate_taylor


# Matches numbers like -12, 3, 4.5 inside any string
_NUMERIC_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$")

# Interval regex for labels like '(10, 20]'
_INTERVAL_RE = re.compile(
    r"""^\s*
        ([\(\[])\s* # open bracket/paren
        ([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*,\s* # lower bound
        ([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s* # upper bound
        ([\)\]])\s* # close bracket/paren
        $""",
    re.VERBOSE,
)


def _intervalsort_key(v: object) -> tuple[float, float] | None:
    """Return (low, high) if v looks like an interval label '(a, b]' or '[a, b)'."""
    if isinstance(v, str):
        m = _INTERVAL_RE.match(v.strip())
        if m:
            return float(m.group(2)), float(m.group(3))
    return None


def _numsort_key(v: object) -> tuple[int, float | str, float | None]:
    """
    Sort priority:
      0) pure numeric scalars by numeric value
      1) interval labels by (low, high)
      2) everything else lexicographically
    """
    if isinstance(v, (int, float, np.integer, np.floating)):
        return (0, float(v), None)
    if isinstance(v, str) and _NUMERIC_RE.fullmatch(v.strip()):
        return (0, float(v), None)

    ii = _intervalsort_key(v)
    if ii is not None:
        return (1, ii[0], ii[1])

    return (2, str(v), None)


def _norm_label(v: object | None) -> str | None:
    """Normalize labels to consistent strings."""
    if v is None:
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        x = float(v)
        return str(int(x)) if x.is_integer() else str(x)
    if isinstance(v, str) and _NUMERIC_RE.fullmatch(v.strip()):
        try:
            x = float(v.strip())
            return str(int(x)) if x.is_integer() else str(x)
        except ValueError:
            pass
    return str(v).strip()


def _build_design_matrix_vectorized(
    observed_pairs: list[tuple[Any, Any]], unique_r: list[Any], unique_c: list[Any]
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Vectorized construction of the design matrix for ~ row + col + row:col.
    Uses reference cell coding (dropping first level).
    """
    nr = len(unique_r)
    nc = len(unique_c)
    n_obs = len(observed_pairs)

    # Map levels to indices
    r_map = {val: i for i, val in enumerate(unique_r)}
    c_map = {val: i for i, val in enumerate(unique_c)}

    # Convert pairs to index arrays
    r_indices = np.empty(n_obs, dtype=int)
    c_indices = np.empty(n_obs, dtype=int)

    valid_mask = np.ones(n_obs, dtype=bool)

    for i, (r_val, c_val) in enumerate(observed_pairs):
        if r_val in r_map and c_val in c_map:
            r_indices[i] = r_map[r_val]
            c_indices[i] = c_map[c_val]
        else:
            valid_mask[i] = False

    if not np.all(valid_mask):
        r_indices = r_indices[valid_mask]
        c_indices = c_indices[valid_mask]
        n_obs = len(r_indices)

    # 1. Main Effects (X1)
    # Drop first column (reference level 0)
    # np.eye(nr)[r_indices] creates one-hot encoding of shape (n_obs, nr)
    D_r = np.eye(nr)[r_indices][:, 1:]
    D_c = np.eye(nc)[c_indices][:, 1:]

    # Intercept + Row Dummies + Col Dummies
    X1 = np.hstack([np.ones((n_obs, 1)), D_r, D_c])

    # 2. Interactions (X2)
    # Row-wise Kronecker product (Khatri-Rao) of the dummy matrices
    # D_r is (N, R-1), D_c is (N, C-1) -> Result is (N, (R-1)*(C-1))
    # We use broadcasting: (N, R-1, 1) * (N, 1, C-1) -> (N, R-1, C-1) -> flatten last two dims
    X2 = (D_r[:, :, None] * D_c[:, None, :]).reshape(n_obs, -1)

    return X1, X2, nr, nc


def _rao_scott_adjustments(
    cell_rows: list[CellEst],
    tbl_est_cov_prop: np.ndarray,
    vars_str: pl.DataFrame,
    varnames: list[str],
    n_obs: int,
    stratum: np.ndarray | None,
    psu: np.ndarray,
) -> TableStats:
    """Calculate Pearson and Likelihood Ratio stats with Rao-Scott corrections."""

    #
    # We project the interaction effects (X2) onto the space orthogonal to main effects (X1).

    # 1. Prepare unique levels for design matrix
    # We need all unique levels from data to define the full dummy space
    unique_df = vars_str.select(varnames).unique().sort(varnames)

    u_rows_raw = unique_df[varnames[0]].to_list()
    u_cols_raw = unique_df[varnames[1]].to_list()

    unique_rows = sorted(
        list({_norm_label(x) for x in u_rows_raw if x is not None}), key=_numsort_key
    )
    unique_cols = sorted(
        list({_norm_label(x) for x in u_cols_raw if x is not None}), key=_numsort_key
    )

    observed_pairs = [(r.rowvar, r.colvar) for r in cell_rows]
    x1, x2, nr, nc = _build_design_matrix_vectorized(observed_pairs, unique_rows, unique_cols)

    # 2. Contrast matrix: OLS residual of X2 projected off X1
    # Matches R: Cmat <- qr.resid(qr(X1), X12[,-(1:(nr+nc-1))])
    q, _ = np.linalg.qr(x1, mode="reduced")
    cmat = x2 - q @ (q.T @ x2)

    # 3. iDmat = diag(1/p) with 0 for zero proportions (matches R)
    p_vec = np.array([r.est for r in cell_rows])
    idmat = np.diag(np.where(p_vec == 0, 0.0, 1.0 / p_vec))

    # 4. Delta computation (matches R lines 100-102):
    #    denom = Cmat' * (iDmat / N) * Cmat
    #    numr  = Cmat' * iDmat * V * iDmat * Cmat
    #    Delta = solve(denom, numr)
    denom = cmat.T @ (idmat / float(n_obs)) @ cmat
    numr = cmat.T @ idmat @ tbl_est_cov_prop @ idmat @ cmat
    denom_inv = np.linalg.pinv(denom, rcond=1e-12, hermitian=True)
    delta_est = denom_inv @ numr

    # 5. Chi-Square Statistics (using Polars for expected values)
    # Expected (Null) = RowMarginal * ColMarginal (since sum(p)=1)
    # We use window functions (.over) to compute marginals efficiently
    est_df = pl.DataFrame(
        {"row": [r.rowvar for r in cell_rows], "col": [r.colvar for r in cell_rows], "est": p_vec}
    )

    est_df = est_df.with_columns(
        [
            pl.col("est").sum().over("row").alias("row_margin"),
            pl.col("est").sum().over("col").alias("col_margin"),
        ]
    ).with_columns((pl.col("row_margin") * pl.col("col_margin")).alias("expected"))

    # Pearson X2 = n * sum((obs - exp)^2 / exp)
    # R only computes Pearson X², using it for both "Chisq" and "F" statistics
    chisq_p = float(n_obs * ((est_df["est"] - est_df["expected"]) ** 2 / est_df["expected"]).sum())

    # 6. Degrees of Freedom & Corrections
    if stratum is not None:
        n_strata = int(np.unique(stratum).size)
        # Fast tuple-like unique for PSU count
        n_psus = int(len(set(zip(stratum, psu))))
    else:
        n_strata = 1
        n_psus = int(np.unique(psu).size)

    df_base = (nr - 1) * (nc - 1)

    trace_delta = float(np.trace(delta_est))

    # First-order (unadj) Rao-Scott: X²/mean(δ) ~ χ²(df_base)
    mean_delta = trace_delta / df_base if df_base > 0 and trace_delta > 1e-9 else 1.0
    p_value_unadj = float(1 - chi2.cdf(chisq_p / mean_delta, df_base))

    # Second-order (adj) F correction: F = X²/trace(δ) with Satterthwaite df
    if trace_delta > 1e-9:
        f_p = float(chisq_p / trace_delta)

        # Satterthwaite DF approximation
        sum_sq_delta = np.trace(delta_est @ delta_est)
        df_num = float((trace_delta**2) / sum_sq_delta)
        df_den = float((n_psus - n_strata) * df_num)
    else:
        f_p = df_num = df_den = 0.0

    # 7. Construct Result
    return TableStats(
        chisq=ChiSquare(df=df_base, value=chisq_p, p_value=p_value_unadj),
        f=FDist(
            df_num=df_num, df_den=df_den, value=f_p, p_value=float(1 - f.cdf(f_p, df_num, df_den))
        ),
    )


def _rows_from_est(est_results: list[Any], is_two_way: bool) -> list[CellEst]:
    """Convert estimation results into sorted CellEst objects."""
    rows = []
    for p in est_results:
        rv, cv = "", ""
        if is_two_way and isinstance(p.y_level, str):
            parts = p.y_level.split("__by__")
            if len(parts) == 2:
                rv, cv = parts
            elif len(parts) == 1:
                rv = parts[0]
        else:
            rv = str(p.y_level) if p.y_level is not None else ""

        rows.append(
            CellEst(
                rowvar=_norm_label(rv) or "",
                colvar=(_norm_label(cv) or "") if is_two_way else "",
                est=float(p.est),
                se=float(p.se),
                cv=float(p.cv),
                lci=float(p.lci),
                uci=float(p.uci),
            )
        )

    if is_two_way:
        rows.sort(key=lambda r: (_numsort_key(r.rowvar), _numsort_key(r.colvar)))
    else:
        rows.sort(key=lambda r: _numsort_key(r.rowvar))

    return rows


def _tabulate(
    *,
    vars: pl.DataFrame,
    wgt: np.ndarray,
    stratum: np.ndarray | None,
    psu: np.ndarray,
    ssu: np.ndarray | None,
    fpc: dict | float = 1,
    alpha: float = 0.05,
) -> tuple[list[CellEst], TableStats | None]:
    """
    Compute categorical table estimates and optional Rao-Scott tests.
    """
    varnames = vars.columns
    is_two_way = len(varnames) == 2

    # Prepare keys for Taylor estimation
    # Casting to Utf8 ensures consistent categorical handling
    if is_two_way:
        y_name = f"{varnames[0]}__by__{varnames[1]}"
        # Efficiently concat columns in Polars
        y_keys = (
            vars.lazy()
            .select(
                pl.concat_str(
                    [
                        pl.col(varnames[0]).cast(pl.Utf8),
                        pl.lit("__by__"),
                        pl.col(varnames[1]).cast(pl.Utf8),
                    ]
                )
            )
            .collect()
            .to_series()  # type: ignore[union-attr]
            .to_numpy()
        )
    else:
        y_name = varnames[0]
        y_keys = vars[varnames[0]].cast(pl.Utf8).to_numpy()

    # 1. Estimate Proportions (Required for Rao-Scott Covariance)
    tbl_est_prop, tbl_est_cov_prop = _estimate_taylor(
        param=PopParam.PROP,
        y=y_keys,
        y_name=y_name,
        wgt=wgt,
        stratum=stratum,
        psu=psu,
        ssu=ssu,
        fpc=fpc,
        as_factor=True,
        alpha=alpha,
    )

    cell_rows = _rows_from_est(tbl_est_prop, is_two_way)
    n_obs = vars.height

    # 2. Compute Statistics (Two-way only)
    tbl_stats = None
    if is_two_way:
        # Cast vars to string once for unique level extraction
        vars_str = vars.select([pl.col(c).cast(pl.Utf8) for c in varnames])

        tbl_stats = _rao_scott_adjustments(
            cell_rows=cell_rows,
            tbl_est_cov_prop=tbl_est_cov_prop,
            vars_str=vars_str,
            varnames=varnames,
            n_obs=n_obs,
            stratum=stratum,
            psu=psu,
        )

    # 3. Final Output: Switch to Totals if weights are not normalized
    # If sum(w) != 1, the user likely wants Count estimates, not Proportions.
    # We re-run estimation for TOTAL to get correct SEs for counts.
    if abs(wgt.sum() - 1.0) > 1e-6:
        tbl_est_tot, _ = _estimate_taylor(
            param=PopParam.TOTAL,
            y=y_keys,
            y_name=y_name,
            wgt=wgt,
            stratum=stratum,
            psu=psu,
            ssu=ssu,
            fpc=fpc,
            as_factor=True,
            alpha=alpha,
        )
        cell_rows = _rows_from_est(tbl_est_tot, is_two_way)

        # R's confint(svytotal(...)) uses z-based CIs (qnorm), not t-based.
        # _estimate_taylor returns t-based CIs, so we override them here.
        from scipy.stats import norm as norm_dist

        z_crit = float(norm_dist.ppf(1 - alpha / 2))
        cell_rows = [
            CellEst(
                rowvar=c.rowvar,
                colvar=c.colvar,
                est=c.est,
                se=c.se,
                cv=c.cv,
                lci=c.est - z_crit * c.se,
                uci=c.est + z_crit * c.se,
            )
            for c in cell_rows
        ]

    return cell_rows, tbl_stats
