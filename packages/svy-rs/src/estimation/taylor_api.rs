//! PyO3 wrappers for Taylor linearisation; the heavy math lives in `taylor`.

use crate::errors::to_py_err;
use crate::estimation::error::{EResult, EstimationError};
use numpy::PyReadonlyArray1;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use rayon::prelude::*;

use crate::estimation::association::{
    AssocKind, point_estimate_assoc, scores_assoc, srs_variance_assoc_of,
};
use crate::estimation::calib_sweep::{CalibSweep, make_calib};
use crate::estimation::into_contiguous;
use crate::estimation::taylor::{
    SrsRef, SvyQuantileMethod, TaylorDesign, active_count, build_taylor_design, degrees_of_freedom,
    degrees_of_freedom_from_design, point_estimate_mean, point_estimate_mean_domain,
    point_estimate_ratio, point_estimate_ratio_domain, point_estimate_total,
    point_estimate_total_domain, quantiles_woodruff, rows_column, scores_mean, scores_mean_arr,
    scores_mean_domain, scores_ratio, scores_ratio_domain, scores_total, scores_total_arr,
    scores_total_domain, srs_variance_mean, srs_variance_mean_domain, srs_variance_ratio,
    srs_variance_ratio_domain, srs_variance_total, srs_variance_total_domain,
    taylor_covariance_apply, taylor_covariance_apply_in, taylor_variance_apply_in,
    weighted_quantile,
};

/// One by-group: label, estimate, SE, variance, n, deff and scores.
type GroupRow<'a> = (&'a str, f64, f64, f64, u32, f64, Vec<f64>);

/// The SRS reference for the design effect: wor with N = sum of weights unless told otherwise.
/// `deff_pop_total`, the declared population size, keeps it right under rescaled weights.
fn parse_srs_ref(deff_ref: Option<&str>, deff_pop_total: Option<f64>) -> EResult<SrsRef> {
    match deff_ref.unwrap_or("wor") {
        "wr" => Ok(SrsRef::WithReplacement),
        "wor" => Ok(SrsRef::WithoutReplacement {
            pop_total: deff_pop_total,
        }),
        other => Err(EstimationError::InvalidChoice {
            what: "deff reference",
            got: other.to_string(),
            expected: "'wor' or 'wr'",
        }),
    }
}

// ============================================================================
// Mean
// ============================================================================

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, by_col=None, singleton_method=None, domain_rows_col=None, deff_ref=None, deff_pop_total=None, calib_kind=None, calib_cells=None, calib_aux=None, calib_prev_wgt=None, calib_pins_total=None, calib_new_wgt=None))]
#[allow(clippy::too_many_arguments)]
pub fn taylor_mean(
    _py: Python,
    data: PyDataFrame,
    value_col: String,
    weight_col: String,
    strata_col: Option<String>,
    psu_col: Option<String>,
    ssu_col: Option<String>,
    fpc_col: Option<String>,
    fpc_ssu_col: Option<String>,
    by_col: Option<String>,
    singleton_method: Option<String>,
    domain_rows_col: Option<String>,
    deff_ref: Option<String>,
    deff_pop_total: Option<f64>,
    calib_kind: Option<String>,
    calib_cells: Option<Vec<String>>,
    calib_aux: Option<Vec<String>>,
    calib_prev_wgt: Option<String>,
    calib_pins_total: Option<bool>,
    calib_new_wgt: Option<String>,
) -> PyResult<(PyDataFrame, Option<Vec<f64>>)> {
    let df = into_contiguous(data);
    let calib = make_calib(
        &df,
        &weight_col,
        calib_kind,
        calib_cells,
        calib_aux,
        calib_prev_wgt,
        calib_pins_total,
        calib_new_wgt,
    )
    .map_err(|e| to_py_err(&e))?;
    let srs = parse_srs_ref(deff_ref.as_deref(), deff_pop_total).map_err(|e| to_py_err(&e))?;

    let Some(by) = by_col else {
        // Detach here too: the GIL would serialise `compute_mean_ungrouped`'s `rayon::join`.
        let result = _py
            .detach(|| {
                compute_mean_ungrouped(
                    &df,
                    &value_col,
                    &weight_col,
                    strata_col.as_deref(),
                    psu_col.as_deref(),
                    ssu_col.as_deref(),
                    fpc_col.as_deref(),
                    fpc_ssu_col.as_deref(),
                    singleton_method.as_deref(),
                    domain_rows_col.as_deref(),
                    srs,
                    calib,
                )
            })
            .map_err(|e| to_py_err(&e))?;
        return Ok((PyDataFrame(result), None));
    };
    let (result, cov) = _py
        .detach(|| {
            compute_mean_grouped(
                &df,
                &value_col,
                &weight_col,
                strata_col.as_deref(),
                psu_col.as_deref(),
                ssu_col.as_deref(),
                fpc_col.as_deref(),
                fpc_ssu_col.as_deref(),
                &by,
                singleton_method.as_deref(),
                domain_rows_col.as_deref(),
                srs,
                calib,
            )
        })
        .map_err(|e| to_py_err(&e))?;
    Ok((PyDataFrame(result), Some(cov)))
}

/// Batched ungrouped mean over many variables sharing one design; one row per variable, in order.
#[pyfunction]
#[pyo3(signature = (data, value_cols, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, singleton_method=None, domain_rows_col=None, deff_ref=None, deff_pop_total=None))]
#[allow(clippy::too_many_arguments)]
pub fn taylor_mean_multi(
    _py: Python,
    data: PyDataFrame,
    value_cols: Vec<String>,
    weight_col: String,
    strata_col: Option<String>,
    psu_col: Option<String>,
    ssu_col: Option<String>,
    fpc_col: Option<String>,
    fpc_ssu_col: Option<String>,
    singleton_method: Option<String>,
    domain_rows_col: Option<String>,
    deff_ref: Option<String>,
    deff_pop_total: Option<f64>,
) -> PyResult<PyDataFrame> {
    let df = into_contiguous(data);
    let srs = parse_srs_ref(deff_ref.as_deref(), deff_pop_total).map_err(|e| to_py_err(&e))?;
    let result = _py
        .detach(|| {
            compute_mean_multi(
                &df,
                &value_cols,
                &weight_col,
                strata_col.as_deref(),
                psu_col.as_deref(),
                ssu_col.as_deref(),
                fpc_col.as_deref(),
                fpc_ssu_col.as_deref(),
                singleton_method.as_deref(),
                domain_rows_col.as_deref(),
                srs,
            )
        })
        .map_err(|e| to_py_err(&e))?;
    Ok(PyDataFrame(result))
}

/// Ungrouped single-estimate skeleton: design build overlaps the score pass, variance overlaps df.
/// Each half still sums in row order, so output is bit-identical at any thread count.
#[allow(clippy::too_many_arguments)]
fn ungrouped_estimate<T: Send>(
    weights: &Float64Chunked,
    strata: Option<&Column>,
    psu: Option<&Column>,
    ssu: Option<&Column>,
    fpc: Option<&Float64Chunked>,
    fpc_ssu: Option<&Float64Chunked>,
    singleton_method: Option<&str>,
    rows: Option<&BooleanChunked>,
    calib: Option<CalibSweep>,
    value_work: impl FnOnce() -> EResult<(Vec<f64>, T)> + Send,
) -> EResult<(f64, u32, T)> {
    let (design_res, value_res) = rayon::join(
        || build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method),
        value_work,
    );
    let design = design_res?.with_calib(calib);
    let (scores, extra) = value_res?;

    let active = design.domain_rows(weights.len(), rows, None);
    let (variance, df_val) = rayon::join(
        || taylor_variance_apply_in(&scores, &design, active.as_deref()),
        || degrees_of_freedom_from_design(weights, &design, None),
    );
    Ok((variance, df_val, extra))
}

/// Scores with nulls as 0.0, as `taylor_variance` treats them.
fn scores_to_arr(scores: &Float64Chunked) -> Vec<f64> {
    scores.iter().map(|s| s.unwrap_or(0.0)).collect()
}

/// Row-major flat k×k covariance for the FFI boundary.
fn flatten_cov(cov: Vec<Vec<f64>>) -> Vec<f64> {
    cov.into_iter().flatten().collect()
}

// A zero denominator (weightless domain, or ratio with zero weighted x) gives a NaN row, as R's
// svymean(subset()), instead of failing every domain; its zero scores need the variance masked.

/// Variance and SE of an estimate, both NaN when the estimate is undefined.
fn var_se(estimate: f64, variance: f64) -> (f64, f64) {
    if estimate.is_nan() {
        (f64::NAN, f64::NAN)
    } else {
        (variance, variance.max(0.0).sqrt())
    }
}

/// Flat covariance with NaN in the rows and columns of undefined estimates.
fn flat_cov(score_cols: &[Vec<f64>], design: &TaylorDesign, estimates: &[f64]) -> Vec<f64> {
    flat_cov_in(score_cols, design, estimates, None)
}

/// [`flat_cov`] for estimates of one domain (`active`).
fn flat_cov_in(
    score_cols: &[Vec<f64>],
    design: &TaylorDesign,
    estimates: &[f64],
    active: Option<&[bool]>,
) -> Vec<f64> {
    let mut cov = taylor_covariance_apply_in(score_cols, design, active);
    for (i, est) in estimates.iter().enumerate() {
        if est.is_nan() {
            cov[i].fill(f64::NAN);
            for row in cov.iter_mut() {
                row[i] = f64::NAN;
            }
        }
    }
    flatten_cov(cov)
}

#[allow(clippy::too_many_arguments)]
fn compute_mean_ungrouped(
    df: &DataFrame,
    value_col: &str,
    weight_col: &str,
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>,
    domain_rows_col: Option<&str>,
    srs: SrsRef,
    calib: Option<CalibSweep>,
) -> EResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let rows = rows_column(df, domain_rows_col)?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc = fpc_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;
    let fpc_ssu = fpc_ssu_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;

    let (variance, df_val, (estimate, srs_var)) = ungrouped_estimate(
        weights,
        strata,
        psu,
        ssu,
        fpc,
        fpc_ssu,
        singleton_method,
        rows.as_ref(),
        calib,
        || {
            let Ok(estimate) = point_estimate_mean(y, weights) else {
                return Ok((vec![0.0; y.len()], (f64::NAN, f64::NAN)));
            };
            let scores = scores_mean_arr(y, weights)?;
            let srs_var = srs_variance_mean(y, weights, srs)?;
            Ok((scores, (estimate, srs_var)))
        },
    )?;

    let (variance, se) = var_se(estimate, variance);
    let n = active_count(weights, None);
    let deff = if srs_var > 0.0 {
        variance / srs_var
    } else {
        f64::NAN
    };

    Ok(
        df!["y" => vec![value_col], "est" => vec![estimate], "se" => vec![se],
        "var" => vec![variance], "df" => vec![df_val], "n" => vec![n], "deff" => vec![deff]]?,
    )
}

/// Batched ungrouped means: one design build amortised over the variables, which run in parallel.
/// Each row is bit-identical to the corresponding `compute_mean_ungrouped` call.
#[allow(clippy::too_many_arguments)]
fn compute_mean_multi(
    df: &DataFrame,
    value_cols: &[String],
    weight_col: &str,
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>,
    domain_rows_col: Option<&str>,
    srs: SrsRef,
) -> EResult<DataFrame> {
    let weights = df.column(weight_col)?.f64()?;
    let rows = rows_column(df, domain_rows_col)?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc = fpc_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;
    let fpc_ssu = fpc_ssu_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;

    // Take df off the design's codes: indexing twice would dominate this serial head.
    let design = build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
    let df_val = degrees_of_freedom_from_design(weights, &design, None);
    let active = design.domain_rows(weights.len(), rows.as_ref(), None);

    // Resolve columns before fanning out: `df.column()` mutates the frame's schema cache,
    // so calling it from the parallel closures would race.
    let y_cols: Vec<&Float64Chunked> = value_cols
        .iter()
        .map(|vc| df.column(vc).and_then(|c| c.f64()))
        .collect::<PolarsResult<Vec<_>>>()?;

    let rows = (0..value_cols.len())
        .into_par_iter()
        .map(|i| -> EResult<(String, f64, f64, f64, u32, f64)> {
            let y = y_cols[i];
            let n = active_count(weights, None);
            let Ok(estimate) = point_estimate_mean(y, weights) else {
                let nan = f64::NAN;
                return Ok((value_cols[i].clone(), nan, nan, nan, n, nan));
            };
            let scores_arr = scores_mean_arr(y, weights)?;
            let variance = taylor_variance_apply_in(&scores_arr, &design, active.as_deref());
            let se = variance.max(0.0).sqrt();
            let srs_var = srs_variance_mean(y, weights, srs)?;
            let deff = if srs_var > 0.0 {
                variance / srs_var
            } else {
                f64::NAN
            };
            Ok((value_cols[i].clone(), estimate, se, variance, n, deff))
        })
        .collect::<EResult<Vec<_>>>()?;

    let nv = rows.len();
    let mut ys: Vec<String> = Vec::with_capacity(nv);
    let mut estimates: Vec<f64> = Vec::with_capacity(nv);
    let mut ses: Vec<f64> = Vec::with_capacity(nv);
    let mut variances: Vec<f64> = Vec::with_capacity(nv);
    let mut ns: Vec<u32> = Vec::with_capacity(nv);
    let mut deffs: Vec<f64> = Vec::with_capacity(nv);
    for (y, est, se, var, n, deff) in rows {
        ys.push(y);
        estimates.push(est);
        ses.push(se);
        variances.push(var);
        ns.push(n);
        deffs.push(deff);
    }
    let dfs = vec![df_val; nv];
    Ok(
        df!["y" => ys, "est" => estimates, "se" => ses, "var" => variances,
        "df" => dfs, "n" => ns, "deff" => deffs]?,
    )
}

#[allow(clippy::too_many_arguments)]
fn compute_mean_grouped(
    df: &DataFrame,
    value_col: &str,
    weight_col: &str,
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_ssu_col: Option<&str>,
    by_col: &str,
    singleton_method: Option<&str>,
    domain_rows_col: Option<&str>,
    srs: SrsRef,
    calib: Option<CalibSweep>,
) -> EResult<(DataFrame, Vec<f64>)> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let rows = rows_column(df, domain_rows_col)?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc = fpc_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;
    let fpc_ssu = fpc_ssu_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;
    let by_str = df.column(by_col)?.str()?;
    // Group discovery and the design build are independent full passes.
    let (unique_groups, design) = rayon::join(
        || crate::estimation::sorted_groups(by_str),
        || build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method),
    );
    let unique_groups = unique_groups?;

    // One design for every group; only the domain-masked scores change.
    let design = design?.with_calib(calib);

    // Groups fan out over rayon and are collected in group order, independent of thread count.
    let groups: Vec<&str> = unique_groups.iter().flatten().collect();
    // A by-group is a domain: its df counts its own active PSUs/strata, not the frame's.
    let masks: Vec<BooleanChunked> = groups.par_iter().map(|&g| by_str.equal(g)).collect();
    let (group_dfs, rows) = rayon::join(
        || {
            masks
                .par_iter()
                .map(|m| degrees_of_freedom_from_design(weights, &design, Some(m)))
                .collect::<Vec<u32>>()
        },
        || {
            groups
                .par_iter()
                .zip(masks.par_iter())
                .map(|(&group, mask)| -> EResult<GroupRow<'_>> {
                    let domain_mask = mask.clone();
                    let active =
                        design.domain_rows(weights.len(), rows.as_ref(), Some(&domain_mask));
                    let n_domain = active_count(weights, Some(&domain_mask));
                    let Ok(estimate) = point_estimate_mean_domain(y, weights, &domain_mask) else {
                        let nan = f64::NAN;
                        return Ok((group, nan, nan, nan, n_domain, nan, vec![0.0; y.len()]));
                    };
                    let scores = scores_mean_domain(y, weights, &domain_mask)?;
                    let scores_arr: Vec<f64> = scores.iter().map(|s| s.unwrap_or(0.0)).collect();
                    let variance =
                        taylor_variance_apply_in(&scores_arr, &design, active.as_deref());
                    let se = variance.max(0.0).sqrt();
                    let srs_var = srs_variance_mean_domain(y, weights, &domain_mask, srs)?;
                    let deff = if srs_var > 0.0 {
                        variance / srs_var
                    } else {
                        f64::NAN
                    };
                    Ok((group, estimate, se, variance, n_domain, deff, scores_arr))
                })
                .collect::<EResult<Vec<_>>>()
        },
    );
    let rows = rows?;

    let n_groups = rows.len();
    let mut by_vals: Vec<&str> = Vec::with_capacity(n_groups);
    let mut estimates: Vec<f64> = Vec::with_capacity(n_groups);
    let mut ses: Vec<f64> = Vec::with_capacity(n_groups);
    let mut variances: Vec<f64> = Vec::with_capacity(n_groups);
    let mut ns: Vec<u32> = Vec::with_capacity(n_groups);
    let mut deffs: Vec<f64> = Vec::with_capacity(n_groups);
    let mut score_cols: Vec<Vec<f64>> = Vec::with_capacity(n_groups);
    for (g, est, se, var, n, deff, scores_arr) in rows {
        by_vals.push(g);
        estimates.push(est);
        ses.push(se);
        variances.push(var);
        ns.push(n);
        deffs.push(deff);
        score_cols.push(scores_arr);
    }
    let cov = flat_cov(&score_cols, &design, &estimates);
    let dfs = group_dfs;
    let out = df![by_col => by_vals, "y" => vec![value_col; n_groups], "est" => estimates,
        "se" => ses, "var" => variances, "df" => dfs, "n" => ns, "deff" => deffs]?;
    Ok((out, cov))
}

// ============================================================================
// Total
// ============================================================================

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, by_col=None, singleton_method=None, domain_rows_col=None, deff_ref=None, deff_pop_total=None, calib_kind=None, calib_cells=None, calib_aux=None, calib_prev_wgt=None, calib_pins_total=None, calib_new_wgt=None))]
#[allow(clippy::too_many_arguments)]
pub fn taylor_total(
    _py: Python,
    data: PyDataFrame,
    value_col: String,
    weight_col: String,
    strata_col: Option<String>,
    psu_col: Option<String>,
    ssu_col: Option<String>,
    fpc_col: Option<String>,
    fpc_ssu_col: Option<String>,
    by_col: Option<String>,
    singleton_method: Option<String>,
    domain_rows_col: Option<String>,
    deff_ref: Option<String>,
    deff_pop_total: Option<f64>,
    calib_kind: Option<String>,
    calib_cells: Option<Vec<String>>,
    calib_aux: Option<Vec<String>>,
    calib_prev_wgt: Option<String>,
    calib_pins_total: Option<bool>,
    calib_new_wgt: Option<String>,
) -> PyResult<(PyDataFrame, Option<Vec<f64>>)> {
    let df = into_contiguous(data);
    let calib = make_calib(
        &df,
        &weight_col,
        calib_kind,
        calib_cells,
        calib_aux,
        calib_prev_wgt,
        calib_pins_total,
        calib_new_wgt,
    )
    .map_err(|e| to_py_err(&e))?;
    let srs = parse_srs_ref(deff_ref.as_deref(), deff_pop_total).map_err(|e| to_py_err(&e))?;
    let Some(by) = by_col else {
        let result = compute_total_ungrouped(
            &df,
            &value_col,
            &weight_col,
            strata_col.as_deref(),
            psu_col.as_deref(),
            ssu_col.as_deref(),
            fpc_col.as_deref(),
            fpc_ssu_col.as_deref(),
            singleton_method.as_deref(),
            domain_rows_col.as_deref(),
            srs,
            calib,
        )
        .map_err(|e| to_py_err(&e))?;
        return Ok((PyDataFrame(result), None));
    };
    let (result, cov) = _py
        .detach(|| {
            compute_total_grouped(
                &df,
                &value_col,
                &weight_col,
                strata_col.as_deref(),
                psu_col.as_deref(),
                ssu_col.as_deref(),
                fpc_col.as_deref(),
                fpc_ssu_col.as_deref(),
                &by,
                singleton_method.as_deref(),
                domain_rows_col.as_deref(),
                srs,
                calib,
            )
        })
        .map_err(|e| to_py_err(&e))?;
    Ok((PyDataFrame(result), Some(cov)))
}

/// Batched ungrouped total over many variables sharing one design build.
#[pyfunction]
#[pyo3(signature = (data, value_cols, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, singleton_method=None, domain_rows_col=None, deff_ref=None, deff_pop_total=None))]
#[allow(clippy::too_many_arguments)]
pub fn taylor_total_multi(
    _py: Python,
    data: PyDataFrame,
    value_cols: Vec<String>,
    weight_col: String,
    strata_col: Option<String>,
    psu_col: Option<String>,
    ssu_col: Option<String>,
    fpc_col: Option<String>,
    fpc_ssu_col: Option<String>,
    singleton_method: Option<String>,
    domain_rows_col: Option<String>,
    deff_ref: Option<String>,
    deff_pop_total: Option<f64>,
) -> PyResult<PyDataFrame> {
    let df = into_contiguous(data);
    let srs = parse_srs_ref(deff_ref.as_deref(), deff_pop_total).map_err(|e| to_py_err(&e))?;
    let result = _py
        .detach(|| {
            compute_total_multi(
                &df,
                &value_cols,
                &weight_col,
                strata_col.as_deref(),
                psu_col.as_deref(),
                ssu_col.as_deref(),
                fpc_col.as_deref(),
                fpc_ssu_col.as_deref(),
                singleton_method.as_deref(),
                domain_rows_col.as_deref(),
                srs,
            )
        })
        .map_err(|e| to_py_err(&e))?;
    Ok(PyDataFrame(result))
}

#[allow(clippy::too_many_arguments)]
fn compute_total_ungrouped(
    df: &DataFrame,
    value_col: &str,
    weight_col: &str,
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>,
    domain_rows_col: Option<&str>,
    srs: SrsRef,
    calib: Option<CalibSweep>,
) -> EResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let rows = rows_column(df, domain_rows_col)?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc = fpc_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;
    let fpc_ssu = fpc_ssu_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;

    let (variance, df_val, (estimate, srs_var)) = ungrouped_estimate(
        weights,
        strata,
        psu,
        ssu,
        fpc,
        fpc_ssu,
        singleton_method,
        rows.as_ref(),
        calib,
        || {
            let estimate = point_estimate_total(y, weights)?;
            let scores = scores_total_arr(y, weights)?;
            let srs_var = srs_variance_total(y, weights, srs)?;
            Ok((scores, (estimate, srs_var)))
        },
    )?;

    let se = variance.max(0.0).sqrt();
    let n = active_count(weights, None);
    let deff = if srs_var > 0.0 {
        variance / srs_var
    } else {
        f64::NAN
    };

    Ok(
        df!["y" => vec![value_col], "est" => vec![estimate], "se" => vec![se],
        "var" => vec![variance], "df" => vec![df_val], "n" => vec![n], "deff" => vec![deff]]?,
    )
}

/// Batched ungrouped totals, as `compute_mean_multi`; rows match `compute_total_ungrouped` exactly.
#[allow(clippy::too_many_arguments)]
fn compute_total_multi(
    df: &DataFrame,
    value_cols: &[String],
    weight_col: &str,
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>,
    domain_rows_col: Option<&str>,
    srs: SrsRef,
) -> EResult<DataFrame> {
    let weights = df.column(weight_col)?.f64()?;
    let rows = rows_column(df, domain_rows_col)?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc = fpc_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;
    let fpc_ssu = fpc_ssu_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;

    // Design indexed once, df taken off its codes (see `compute_mean_multi`).
    let design = build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
    let df_val = degrees_of_freedom_from_design(weights, &design, None);
    let active = design.domain_rows(weights.len(), rows.as_ref(), None);

    // Hoist column resolution out of the parallel region (see compute_mean_multi).
    let y_cols: Vec<&Float64Chunked> = value_cols
        .iter()
        .map(|vc| df.column(vc).and_then(|c| c.f64()))
        .collect::<PolarsResult<Vec<_>>>()?;

    let rows = (0..value_cols.len())
        .into_par_iter()
        .map(|i| -> EResult<(String, f64, f64, f64, u32, f64)> {
            let y = y_cols[i];
            let estimate = point_estimate_total(y, weights)?;
            let scores_arr = scores_total_arr(y, weights)?;
            let variance = taylor_variance_apply_in(&scores_arr, &design, active.as_deref());
            let se = variance.max(0.0).sqrt();
            let n = active_count(weights, None);
            let srs_var = srs_variance_total(y, weights, srs)?;
            let deff = if srs_var > 0.0 {
                variance / srs_var
            } else {
                f64::NAN
            };
            Ok((value_cols[i].clone(), estimate, se, variance, n, deff))
        })
        .collect::<EResult<Vec<_>>>()?;

    let nv = rows.len();
    let mut ys: Vec<String> = Vec::with_capacity(nv);
    let mut estimates: Vec<f64> = Vec::with_capacity(nv);
    let mut ses: Vec<f64> = Vec::with_capacity(nv);
    let mut variances: Vec<f64> = Vec::with_capacity(nv);
    let mut ns: Vec<u32> = Vec::with_capacity(nv);
    let mut deffs: Vec<f64> = Vec::with_capacity(nv);
    for (y, est, se, var, n, deff) in rows {
        ys.push(y);
        estimates.push(est);
        ses.push(se);
        variances.push(var);
        ns.push(n);
        deffs.push(deff);
    }
    let dfs = vec![df_val; nv];
    Ok(
        df!["y" => ys, "est" => estimates, "se" => ses, "var" => variances,
        "df" => dfs, "n" => ns, "deff" => deffs]?,
    )
}

#[allow(clippy::too_many_arguments)]
fn compute_total_grouped(
    df: &DataFrame,
    value_col: &str,
    weight_col: &str,
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_ssu_col: Option<&str>,
    by_col: &str,
    singleton_method: Option<&str>,
    domain_rows_col: Option<&str>,
    srs: SrsRef,
    calib: Option<CalibSweep>,
) -> EResult<(DataFrame, Vec<f64>)> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let rows = rows_column(df, domain_rows_col)?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc = fpc_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;
    let fpc_ssu = fpc_ssu_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;
    let by_str = df.column(by_col)?.str()?;
    // Group discovery and the design build are independent full passes.
    let (unique_groups, design) = rayon::join(
        || crate::estimation::sorted_groups(by_str),
        || build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method),
    );
    let unique_groups = unique_groups?;
    let design = design?.with_calib(calib);

    let groups: Vec<&str> = unique_groups.iter().flatten().collect();
    // A by-group is a domain: its df counts its own active PSUs/strata, not the frame's.
    let masks: Vec<BooleanChunked> = groups.par_iter().map(|&g| by_str.equal(g)).collect();
    let (group_dfs, rows) = rayon::join(
        || {
            masks
                .par_iter()
                .map(|m| degrees_of_freedom_from_design(weights, &design, Some(m)))
                .collect::<Vec<u32>>()
        },
        || {
            groups
                .par_iter()
                .zip(masks.par_iter())
                .map(|(&group, mask)| -> EResult<GroupRow<'_>> {
                    let domain_mask = mask.clone();
                    let active =
                        design.domain_rows(weights.len(), rows.as_ref(), Some(&domain_mask));
                    let n_domain = active_count(weights, Some(&domain_mask));
                    let estimate = point_estimate_total_domain(y, weights, &domain_mask)?;
                    let scores = scores_total_domain(y, weights, &domain_mask)?;
                    let scores_arr: Vec<f64> = scores.iter().map(|s| s.unwrap_or(0.0)).collect();
                    let variance =
                        taylor_variance_apply_in(&scores_arr, &design, active.as_deref());
                    let se = variance.max(0.0).sqrt();
                    let srs_var = srs_variance_total_domain(y, weights, &domain_mask, srs)?;
                    let deff = if srs_var > 0.0 {
                        variance / srs_var
                    } else {
                        f64::NAN
                    };
                    Ok((group, estimate, se, variance, n_domain, deff, scores_arr))
                })
                .collect::<EResult<Vec<_>>>()
        },
    );
    let rows = rows?;

    let n_groups = rows.len();
    let mut by_vals: Vec<&str> = Vec::with_capacity(n_groups);
    let mut estimates: Vec<f64> = Vec::with_capacity(n_groups);
    let mut ses: Vec<f64> = Vec::with_capacity(n_groups);
    let mut variances: Vec<f64> = Vec::with_capacity(n_groups);
    let mut ns: Vec<u32> = Vec::with_capacity(n_groups);
    let mut deffs: Vec<f64> = Vec::with_capacity(n_groups);
    let mut score_cols: Vec<Vec<f64>> = Vec::with_capacity(n_groups);
    for (g, est, se, var, n, deff, scores_arr) in rows {
        by_vals.push(g);
        estimates.push(est);
        ses.push(se);
        variances.push(var);
        ns.push(n);
        deffs.push(deff);
        score_cols.push(scores_arr);
    }
    let cov = flatten_cov(taylor_covariance_apply(&score_cols, &design));
    let dfs = group_dfs;
    let out = df![by_col => by_vals, "y" => vec![value_col; n_groups], "est" => estimates,
        "se" => ses, "var" => variances, "df" => dfs, "n" => ns, "deff" => deffs]?;
    Ok((out, cov))
}

// ============================================================================
// Ratio
// ============================================================================

#[pyfunction]
#[pyo3(signature = (data, numerator_col, denominator_col, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, by_col=None, singleton_method=None, domain_rows_col=None, deff_ref=None, deff_pop_total=None, calib_kind=None, calib_cells=None, calib_aux=None, calib_prev_wgt=None, calib_pins_total=None, calib_new_wgt=None))]
#[allow(clippy::too_many_arguments)]
pub fn taylor_ratio(
    _py: Python,
    data: PyDataFrame,
    numerator_col: String,
    denominator_col: String,
    weight_col: String,
    strata_col: Option<String>,
    psu_col: Option<String>,
    ssu_col: Option<String>,
    fpc_col: Option<String>,
    fpc_ssu_col: Option<String>,
    by_col: Option<String>,
    singleton_method: Option<String>,
    domain_rows_col: Option<String>,
    deff_ref: Option<String>,
    deff_pop_total: Option<f64>,
    calib_kind: Option<String>,
    calib_cells: Option<Vec<String>>,
    calib_aux: Option<Vec<String>>,
    calib_prev_wgt: Option<String>,
    calib_pins_total: Option<bool>,
    calib_new_wgt: Option<String>,
) -> PyResult<(PyDataFrame, Option<Vec<f64>>)> {
    let df = into_contiguous(data);
    let calib = make_calib(
        &df,
        &weight_col,
        calib_kind,
        calib_cells,
        calib_aux,
        calib_prev_wgt,
        calib_pins_total,
        calib_new_wgt,
    )
    .map_err(|e| to_py_err(&e))?;
    let srs = parse_srs_ref(deff_ref.as_deref(), deff_pop_total).map_err(|e| to_py_err(&e))?;
    let Some(by) = by_col else {
        let result = compute_ratio_ungrouped(
            &df,
            &numerator_col,
            &denominator_col,
            &weight_col,
            strata_col.as_deref(),
            psu_col.as_deref(),
            ssu_col.as_deref(),
            fpc_col.as_deref(),
            fpc_ssu_col.as_deref(),
            singleton_method.as_deref(),
            domain_rows_col.as_deref(),
            srs,
            calib,
        )
        .map_err(|e| to_py_err(&e))?;
        return Ok((PyDataFrame(result), None));
    };
    let (result, cov) = _py
        .detach(|| {
            compute_ratio_grouped(
                &df,
                &numerator_col,
                &denominator_col,
                &weight_col,
                strata_col.as_deref(),
                psu_col.as_deref(),
                ssu_col.as_deref(),
                fpc_col.as_deref(),
                fpc_ssu_col.as_deref(),
                &by,
                singleton_method.as_deref(),
                domain_rows_col.as_deref(),
                srs,
                calib,
            )
        })
        .map_err(|e| to_py_err(&e))?;
    Ok((PyDataFrame(result), Some(cov)))
}

/// Batched ungrouped ratios over equal-length numerator/denominator column lists, one design build.
#[pyfunction]
#[pyo3(signature = (data, numerator_cols, denominator_cols, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, singleton_method=None, domain_rows_col=None, deff_ref=None, deff_pop_total=None))]
#[allow(clippy::too_many_arguments)]
pub fn taylor_ratio_multi(
    _py: Python,
    data: PyDataFrame,
    numerator_cols: Vec<String>,
    denominator_cols: Vec<String>,
    weight_col: String,
    strata_col: Option<String>,
    psu_col: Option<String>,
    ssu_col: Option<String>,
    fpc_col: Option<String>,
    fpc_ssu_col: Option<String>,
    singleton_method: Option<String>,
    domain_rows_col: Option<String>,
    deff_ref: Option<String>,
    deff_pop_total: Option<f64>,
) -> PyResult<PyDataFrame> {
    let df = into_contiguous(data);
    let srs = parse_srs_ref(deff_ref.as_deref(), deff_pop_total).map_err(|e| to_py_err(&e))?;
    let result = _py
        .detach(|| {
            compute_ratio_multi(
                &df,
                &numerator_cols,
                &denominator_cols,
                &weight_col,
                strata_col.as_deref(),
                psu_col.as_deref(),
                ssu_col.as_deref(),
                fpc_col.as_deref(),
                fpc_ssu_col.as_deref(),
                singleton_method.as_deref(),
                domain_rows_col.as_deref(),
                srs,
            )
        })
        .map_err(|e| to_py_err(&e))?;
    Ok(PyDataFrame(result))
}

#[allow(clippy::too_many_arguments)]
fn compute_ratio_ungrouped(
    df: &DataFrame,
    numerator_col: &str,
    denominator_col: &str,
    weight_col: &str,
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>,
    domain_rows_col: Option<&str>,
    srs: SrsRef,
    calib: Option<CalibSweep>,
) -> EResult<DataFrame> {
    let y = df.column(numerator_col)?.f64()?;
    let x = df.column(denominator_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let rows = rows_column(df, domain_rows_col)?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc = fpc_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;
    let fpc_ssu = fpc_ssu_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;

    let (variance, df_val, (estimate, srs_var)) = ungrouped_estimate(
        weights,
        strata,
        psu,
        ssu,
        fpc,
        fpc_ssu,
        singleton_method,
        rows.as_ref(),
        calib,
        || {
            let Ok(estimate) = point_estimate_ratio(y, x, weights) else {
                return Ok((vec![0.0; y.len()], (f64::NAN, f64::NAN)));
            };
            let scores = scores_to_arr(&scores_ratio(y, x, weights)?);
            let srs_var = srs_variance_ratio(y, x, weights, srs)?;
            Ok((scores, (estimate, srs_var)))
        },
    )?;

    let (variance, se) = var_se(estimate, variance);
    let n = active_count(weights, None);
    let deff = if srs_var > 0.0 {
        variance / srs_var
    } else {
        f64::NAN
    };

    Ok(
        df!["y" => vec![numerator_col], "x" => vec![denominator_col], "est" => vec![estimate],
        "se" => vec![se], "var" => vec![variance], "df" => vec![df_val], "n" => vec![n], "deff" => vec![deff]]?,
    )
}

/// Batched ungrouped ratios, as `compute_mean_multi`: one row per (numerator, denominator) pair.
#[allow(clippy::too_many_arguments)]
fn compute_ratio_multi(
    df: &DataFrame,
    numerator_cols: &[String],
    denominator_cols: &[String],
    weight_col: &str,
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>,
    domain_rows_col: Option<&str>,
    srs: SrsRef,
) -> EResult<DataFrame> {
    let weights = df.column(weight_col)?.f64()?;
    let rows = rows_column(df, domain_rows_col)?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc = fpc_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;
    let fpc_ssu = fpc_ssu_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;

    // Design indexed once, df taken off its codes (see `compute_mean_multi`).
    let design = build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
    let df_val = degrees_of_freedom_from_design(weights, &design, None);
    let active = design.domain_rows(weights.len(), rows.as_ref(), None);

    // Hoist column resolution out of the parallel region (see compute_mean_multi).
    let y_cols: Vec<&Float64Chunked> = numerator_cols
        .iter()
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .collect::<PolarsResult<Vec<_>>>()?;
    let x_cols: Vec<&Float64Chunked> = denominator_cols
        .iter()
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .collect::<PolarsResult<Vec<_>>>()?;

    let rows = (0..numerator_cols.len())
        .into_par_iter()
        .map(|i| -> EResult<(String, String, f64, f64, f64, u32, f64)> {
            let y = y_cols[i];
            let x = x_cols[i];
            let n = active_count(weights, None);
            let Ok(estimate) = point_estimate_ratio(y, x, weights) else {
                let nan = f64::NAN;
                return Ok((
                    numerator_cols[i].clone(),
                    denominator_cols[i].clone(),
                    nan,
                    nan,
                    nan,
                    n,
                    nan,
                ));
            };
            let scores = scores_ratio(y, x, weights)?;
            let scores_arr: Vec<f64> = scores.iter().map(|s| s.unwrap_or(0.0)).collect();
            let variance = taylor_variance_apply_in(&scores_arr, &design, active.as_deref());
            let se = variance.max(0.0).sqrt();
            let srs_var = srs_variance_ratio(y, x, weights, srs)?;
            let deff = if srs_var > 0.0 {
                variance / srs_var
            } else {
                f64::NAN
            };
            Ok((
                numerator_cols[i].clone(),
                denominator_cols[i].clone(),
                estimate,
                se,
                variance,
                n,
                deff,
            ))
        })
        .collect::<EResult<Vec<_>>>()?;

    let nv = rows.len();
    let mut ys: Vec<String> = Vec::with_capacity(nv);
    let mut xs: Vec<String> = Vec::with_capacity(nv);
    let mut estimates: Vec<f64> = Vec::with_capacity(nv);
    let mut ses: Vec<f64> = Vec::with_capacity(nv);
    let mut variances: Vec<f64> = Vec::with_capacity(nv);
    let mut ns: Vec<u32> = Vec::with_capacity(nv);
    let mut deffs: Vec<f64> = Vec::with_capacity(nv);
    for (y, x, est, se, var, n, deff) in rows {
        ys.push(y);
        xs.push(x);
        estimates.push(est);
        ses.push(se);
        variances.push(var);
        ns.push(n);
        deffs.push(deff);
    }
    let dfs = vec![df_val; nv];
    Ok(
        df!["y" => ys, "x" => xs, "est" => estimates, "se" => ses, "var" => variances,
        "df" => dfs, "n" => ns, "deff" => deffs]?,
    )
}

#[allow(clippy::too_many_arguments)]
fn compute_ratio_grouped(
    df: &DataFrame,
    numerator_col: &str,
    denominator_col: &str,
    weight_col: &str,
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_ssu_col: Option<&str>,
    by_col: &str,
    singleton_method: Option<&str>,
    domain_rows_col: Option<&str>,
    srs: SrsRef,
    calib: Option<CalibSweep>,
) -> EResult<(DataFrame, Vec<f64>)> {
    let y = df.column(numerator_col)?.f64()?;
    let x = df.column(denominator_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let rows = rows_column(df, domain_rows_col)?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc = fpc_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;
    let fpc_ssu = fpc_ssu_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;
    let by_str = df.column(by_col)?.str()?;
    // Group discovery and the design build are independent full passes.
    let (unique_groups, design) = rayon::join(
        || crate::estimation::sorted_groups(by_str),
        || build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method),
    );
    let unique_groups = unique_groups?;
    let design = design?.with_calib(calib);

    let groups: Vec<&str> = unique_groups.iter().flatten().collect();
    // A by-group is a domain: its df counts its own active PSUs/strata, not the frame's.
    let masks: Vec<BooleanChunked> = groups.par_iter().map(|&g| by_str.equal(g)).collect();
    let (group_dfs, rows) = rayon::join(
        || {
            masks
                .par_iter()
                .map(|m| degrees_of_freedom_from_design(weights, &design, Some(m)))
                .collect::<Vec<u32>>()
        },
        || {
            groups
                .par_iter()
                .zip(masks.par_iter())
                .map(|(&group, mask)| -> EResult<GroupRow<'_>> {
                    let domain_mask = mask.clone();
                    let active =
                        design.domain_rows(weights.len(), rows.as_ref(), Some(&domain_mask));
                    let n_domain = active_count(weights, Some(&domain_mask));
                    let Ok(estimate) = point_estimate_ratio_domain(y, x, weights, &domain_mask)
                    else {
                        let nan = f64::NAN;
                        return Ok((group, nan, nan, nan, n_domain, nan, vec![0.0; y.len()]));
                    };
                    let scores = scores_ratio_domain(y, x, weights, &domain_mask)?;
                    let scores_arr: Vec<f64> = scores.iter().map(|s| s.unwrap_or(0.0)).collect();
                    let variance =
                        taylor_variance_apply_in(&scores_arr, &design, active.as_deref());
                    let se = variance.max(0.0).sqrt();
                    let srs_var = srs_variance_ratio_domain(y, x, weights, &domain_mask, srs)?;
                    let deff = if srs_var > 0.0 {
                        variance / srs_var
                    } else {
                        f64::NAN
                    };
                    Ok((group, estimate, se, variance, n_domain, deff, scores_arr))
                })
                .collect::<EResult<Vec<_>>>()
        },
    );
    let rows = rows?;

    let n_groups = rows.len();
    let mut by_vals: Vec<&str> = Vec::with_capacity(n_groups);
    let mut estimates: Vec<f64> = Vec::with_capacity(n_groups);
    let mut ses: Vec<f64> = Vec::with_capacity(n_groups);
    let mut variances: Vec<f64> = Vec::with_capacity(n_groups);
    let mut ns: Vec<u32> = Vec::with_capacity(n_groups);
    let mut deffs: Vec<f64> = Vec::with_capacity(n_groups);
    let mut score_cols: Vec<Vec<f64>> = Vec::with_capacity(n_groups);
    for (g, est, se, var, n, deff, scores_arr) in rows {
        by_vals.push(g);
        estimates.push(est);
        ses.push(se);
        variances.push(var);
        ns.push(n);
        deffs.push(deff);
        score_cols.push(scores_arr);
    }
    let cov = flat_cov(&score_cols, &design, &estimates);
    let dfs = group_dfs;
    let out = df![by_col => by_vals, "y" => vec![numerator_col; n_groups], "x" => vec![denominator_col; n_groups],
        "est" => estimates, "se" => ses, "var" => variances, "df" => dfs, "n" => ns, "deff" => deffs]?;
    Ok((out, cov))
}

// ============================================================================
// Association (covariance / correlation)
// ============================================================================

/// Covariance or correlation over column pairs sharing one design, ungrouped or by group.
/// The pair columns `y`/`x` are positional only: both statistics are symmetric.
#[pyfunction]
#[pyo3(signature = (data, y_cols, x_cols, kind, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, by_col=None, singleton_method=None, domain_rows_col=None, deff_ref=None, deff_pop_total=None))]
#[allow(clippy::too_many_arguments)]
pub fn taylor_assoc(
    _py: Python,
    data: PyDataFrame,
    y_cols: Vec<String>,
    x_cols: Vec<String>,
    kind: String,
    weight_col: String,
    strata_col: Option<String>,
    psu_col: Option<String>,
    ssu_col: Option<String>,
    fpc_col: Option<String>,
    fpc_ssu_col: Option<String>,
    by_col: Option<String>,
    singleton_method: Option<String>,
    domain_rows_col: Option<String>,
    deff_ref: Option<String>,
    deff_pop_total: Option<f64>,
) -> PyResult<PyDataFrame> {
    let df = into_contiguous(data);
    let srs = parse_srs_ref(deff_ref.as_deref(), deff_pop_total).map_err(|e| to_py_err(&e))?;
    let result = _py
        .detach(|| {
            let kind = AssocKind::from_name(&kind)?;
            if y_cols.len() != x_cols.len() {
                return Err(EstimationError::PairLengths {
                    y: y_cols.len(),
                    x: x_cols.len(),
                });
            }
            if y_cols.is_empty() {
                return Err(EstimationError::Empty(
                    "at least one column pair is required",
                ));
            }
            compute_assoc(
                &df,
                &y_cols,
                &x_cols,
                kind,
                &weight_col,
                strata_col.as_deref(),
                psu_col.as_deref(),
                ssu_col.as_deref(),
                fpc_col.as_deref(),
                fpc_ssu_col.as_deref(),
                by_col.as_deref(),
                singleton_method.as_deref(),
                domain_rows_col.as_deref(),
                srs,
            )
        })
        .map_err(|e| to_py_err(&e))?;
    Ok(PyDataFrame(result))
}

#[allow(clippy::too_many_arguments)]
fn compute_assoc(
    df: &DataFrame,
    y_cols: &[String],
    x_cols: &[String],
    kind: AssocKind,
    weight_col: &str,
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_ssu_col: Option<&str>,
    by_col: Option<&str>,
    singleton_method: Option<&str>,
    domain_rows_col: Option<&str>,
    srs: SrsRef,
) -> EResult<DataFrame> {
    let weights = df.column(weight_col)?.f64()?;
    let rows = rows_column(df, domain_rows_col)?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc = fpc_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;
    let fpc_ssu = fpc_ssu_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;

    // Resolve columns outside the parallel region, as the other multi kernels do.
    let ys: Vec<&Float64Chunked> = y_cols
        .iter()
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .collect::<PolarsResult<Vec<_>>>()?;
    let xs: Vec<&Float64Chunked> = x_cols
        .iter()
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .collect::<PolarsResult<Vec<_>>>()?;

    let design = build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;

    // Ungrouped runs as a single unmasked group, so one fan-out covers both shapes.
    let by_str = by_col
        .map(|c| df.column(c).and_then(|s| s.str()))
        .transpose()?;
    let unique_groups = by_str.map(crate::estimation::sorted_groups).transpose()?;
    let groups: Vec<Option<&str>> = match unique_groups.as_ref() {
        Some(u) => u.iter().flatten().map(Some).collect(),
        None => vec![None],
    };

    // A by-group is a domain: its df counts its own active PSUs/strata, not the frame's.
    let group_dfs: Vec<u32> = groups
        .par_iter()
        .map(|g| match (g, by_str) {
            (Some(gv), Some(bs)) => {
                degrees_of_freedom_from_design(weights, &design, Some(&bs.equal(*gv)))
            }
            _ => degrees_of_freedom_from_design(weights, &design, None),
        })
        .collect();

    let n_pairs = ys.len();
    let combos: Vec<(usize, usize)> = (0..groups.len())
        .flat_map(|gi| (0..n_pairs).map(move |pi| (gi, pi)))
        .collect();

    let rows = combos
        .par_iter()
        .map(
            |&(gi, pi)| -> EResult<(usize, usize, f64, f64, f64, u32, f64)> {
                let mask = match (groups[gi], by_str) {
                    (Some(g), Some(bs)) => Some(bs.equal(g)),
                    _ => None,
                };
                let (y, x) = (ys[pi], xs[pi]);
                let active = design.domain_rows(weights.len(), rows.as_ref(), mask.as_ref());
                let estimate = point_estimate_assoc(kind, y, x, weights, mask.as_ref())?;
                let n = active_count(weights, mask.as_ref());
                // An undefined estimate (a constant column's correlation) has no variance either.
                if estimate.is_nan() {
                    return Ok((gi, pi, estimate, f64::NAN, f64::NAN, n, f64::NAN));
                }
                let scores = scores_assoc(kind, y, x, weights, mask.as_ref())?;
                let scores_arr: Vec<f64> = scores.iter().map(|s| s.unwrap_or(0.0)).collect();
                let variance = taylor_variance_apply_in(&scores_arr, &design, active.as_deref());
                let se = variance.max(0.0).sqrt();
                let srs_var = srs_variance_assoc_of(kind, y, x, weights, mask.as_ref(), srs)?;
                let deff = if srs_var > 0.0 {
                    variance / srs_var
                } else {
                    f64::NAN
                };
                Ok((gi, pi, estimate, se, variance, n, deff))
            },
        )
        .collect::<EResult<Vec<_>>>()?;

    let nv = rows.len();
    let mut by_vals: Vec<&str> = Vec::with_capacity(nv);
    let mut y_names: Vec<&str> = Vec::with_capacity(nv);
    let mut x_names: Vec<&str> = Vec::with_capacity(nv);
    let mut estimates: Vec<f64> = Vec::with_capacity(nv);
    let mut ses: Vec<f64> = Vec::with_capacity(nv);
    let mut variances: Vec<f64> = Vec::with_capacity(nv);
    let mut dfs: Vec<u32> = Vec::with_capacity(nv);
    let mut ns: Vec<u32> = Vec::with_capacity(nv);
    let mut deffs: Vec<f64> = Vec::with_capacity(nv);
    for (gi, pi, est, se, var, n, deff) in rows {
        if let Some(g) = groups[gi] {
            by_vals.push(g);
        }
        y_names.push(y_cols[pi].as_str());
        x_names.push(x_cols[pi].as_str());
        estimates.push(est);
        ses.push(se);
        variances.push(var);
        dfs.push(group_dfs[gi]);
        ns.push(n);
        deffs.push(deff);
    }

    let kinds = vec![kind.as_str(); nv];
    let mut out = df![
        "y" => y_names, "x" => x_names, "kind" => kinds, "est" => estimates,
        "se" => ses, "var" => variances, "df" => dfs, "n" => ns, "deff" => deffs
    ]?;
    if let Some(name) = by_col {
        out.insert_column(0, Column::new(name.into(), by_vals))?;
    }
    Ok(out)
}

// ============================================================================
// Proportion
// ============================================================================

/// A categorical level's share of the (domain) weight, or its estimated count.
#[derive(Clone, Copy)]
enum LevelScale {
    Share,
    Count,
}

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, by_col=None, singleton_method=None, domain_rows_col=None, deff_ref=None, deff_pop_total=None, calib_kind=None, calib_cells=None, calib_aux=None, calib_prev_wgt=None, calib_pins_total=None, calib_new_wgt=None))]
#[allow(clippy::too_many_arguments)]
pub fn taylor_prop(
    _py: Python,
    data: PyDataFrame,
    value_col: String,
    weight_col: String,
    strata_col: Option<String>,
    psu_col: Option<String>,
    ssu_col: Option<String>,
    fpc_col: Option<String>,
    fpc_ssu_col: Option<String>,
    by_col: Option<String>,
    singleton_method: Option<String>,
    domain_rows_col: Option<String>,
    deff_ref: Option<String>,
    deff_pop_total: Option<f64>,
    calib_kind: Option<String>,
    calib_cells: Option<Vec<String>>,
    calib_aux: Option<Vec<String>>,
    calib_prev_wgt: Option<String>,
    calib_pins_total: Option<bool>,
    calib_new_wgt: Option<String>,
) -> PyResult<(PyDataFrame, Option<Vec<f64>>)> {
    taylor_levels(
        _py,
        data,
        value_col,
        weight_col,
        strata_col,
        psu_col,
        ssu_col,
        fpc_col,
        fpc_ssu_col,
        by_col,
        singleton_method,
        domain_rows_col,
        deff_ref,
        deff_pop_total,
        calib_kind,
        calib_cells,
        calib_aux,
        calib_prev_wgt,
        calib_pins_total,
        calib_new_wgt,
        LevelScale::Share,
    )
}

/// Count per level (R `svytotal(~factor(y))`) with the joint covariance across levels and groups.
#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, by_col=None, singleton_method=None, domain_rows_col=None, deff_ref=None, deff_pop_total=None, calib_kind=None, calib_cells=None, calib_aux=None, calib_prev_wgt=None, calib_pins_total=None, calib_new_wgt=None))]
#[allow(clippy::too_many_arguments)]
pub fn taylor_factor_total(
    _py: Python,
    data: PyDataFrame,
    value_col: String,
    weight_col: String,
    strata_col: Option<String>,
    psu_col: Option<String>,
    ssu_col: Option<String>,
    fpc_col: Option<String>,
    fpc_ssu_col: Option<String>,
    by_col: Option<String>,
    singleton_method: Option<String>,
    domain_rows_col: Option<String>,
    deff_ref: Option<String>,
    deff_pop_total: Option<f64>,
    calib_kind: Option<String>,
    calib_cells: Option<Vec<String>>,
    calib_aux: Option<Vec<String>>,
    calib_prev_wgt: Option<String>,
    calib_pins_total: Option<bool>,
    calib_new_wgt: Option<String>,
) -> PyResult<(PyDataFrame, Option<Vec<f64>>)> {
    taylor_levels(
        _py,
        data,
        value_col,
        weight_col,
        strata_col,
        psu_col,
        ssu_col,
        fpc_col,
        fpc_ssu_col,
        by_col,
        singleton_method,
        domain_rows_col,
        deff_ref,
        deff_pop_total,
        calib_kind,
        calib_cells,
        calib_aux,
        calib_prev_wgt,
        calib_pins_total,
        calib_new_wgt,
        LevelScale::Count,
    )
}

#[allow(clippy::too_many_arguments)]
fn taylor_levels(
    _py: Python,
    data: PyDataFrame,
    value_col: String,
    weight_col: String,
    strata_col: Option<String>,
    psu_col: Option<String>,
    ssu_col: Option<String>,
    fpc_col: Option<String>,
    fpc_ssu_col: Option<String>,
    by_col: Option<String>,
    singleton_method: Option<String>,
    domain_rows_col: Option<String>,
    deff_ref: Option<String>,
    deff_pop_total: Option<f64>,
    calib_kind: Option<String>,
    calib_cells: Option<Vec<String>>,
    calib_aux: Option<Vec<String>>,
    calib_prev_wgt: Option<String>,
    calib_pins_total: Option<bool>,
    calib_new_wgt: Option<String>,
    scale: LevelScale,
) -> PyResult<(PyDataFrame, Option<Vec<f64>>)> {
    let df = into_contiguous(data);
    let calib = make_calib(
        &df,
        &weight_col,
        calib_kind,
        calib_cells,
        calib_aux,
        calib_prev_wgt,
        calib_pins_total,
        calib_new_wgt,
    )
    .map_err(|e| to_py_err(&e))?;
    let srs = parse_srs_ref(deff_ref.as_deref(), deff_pop_total).map_err(|e| to_py_err(&e))?;
    let Some(by) = by_col else {
        let (result, cov) = compute_levels_ungrouped(
            &df,
            &value_col,
            &weight_col,
            strata_col.as_deref(),
            psu_col.as_deref(),
            ssu_col.as_deref(),
            fpc_col.as_deref(),
            fpc_ssu_col.as_deref(),
            singleton_method.as_deref(),
            domain_rows_col.as_deref(),
            srs,
            calib,
            scale,
        )
        .map_err(|e| to_py_err(&e))?;
        return Ok((PyDataFrame(result), Some(cov)));
    };
    let (result, cov) = _py
        .detach(|| {
            compute_levels_grouped(
                &df,
                &value_col,
                &weight_col,
                strata_col.as_deref(),
                psu_col.as_deref(),
                ssu_col.as_deref(),
                fpc_col.as_deref(),
                fpc_ssu_col.as_deref(),
                &by,
                singleton_method.as_deref(),
                domain_rows_col.as_deref(),
                srs,
                calib,
                scale,
            )
        })
        .map_err(|e| to_py_err(&e))?;
    Ok((PyDataFrame(result), Some(cov)))
}

/// Batched ungrouped proportions over many category columns; rows (variable, level) in input order.
#[pyfunction]
#[pyo3(signature = (data, value_cols, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, singleton_method=None, domain_rows_col=None, deff_ref=None, deff_pop_total=None))]
#[allow(clippy::too_many_arguments)]
pub fn taylor_prop_multi(
    _py: Python,
    data: PyDataFrame,
    value_cols: Vec<String>,
    weight_col: String,
    strata_col: Option<String>,
    psu_col: Option<String>,
    ssu_col: Option<String>,
    fpc_col: Option<String>,
    fpc_ssu_col: Option<String>,
    singleton_method: Option<String>,
    domain_rows_col: Option<String>,
    deff_ref: Option<String>,
    deff_pop_total: Option<f64>,
) -> PyResult<PyDataFrame> {
    let df = into_contiguous(data);
    let srs = parse_srs_ref(deff_ref.as_deref(), deff_pop_total).map_err(|e| to_py_err(&e))?;
    let result = _py
        .detach(|| {
            compute_prop_multi(
                &df,
                &value_cols,
                &weight_col,
                strata_col.as_deref(),
                psu_col.as_deref(),
                ssu_col.as_deref(),
                fpc_col.as_deref(),
                fpc_ssu_col.as_deref(),
                singleton_method.as_deref(),
                domain_rows_col.as_deref(),
                srs,
            )
        })
        .map_err(|e| to_py_err(&e))?;
    Ok(PyDataFrame(result))
}

#[allow(clippy::too_many_arguments)]
fn compute_levels_ungrouped(
    df: &DataFrame,
    value_col: &str,
    weight_col: &str,
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>,
    domain_rows_col: Option<&str>,
    srs: SrsRef,
    calib: Option<CalibSweep>,
    scale: LevelScale,
) -> EResult<(DataFrame, Vec<f64>)> {
    let weights = df.column(weight_col)?.f64()?;
    let rows = rows_column(df, domain_rows_col)?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc = fpc_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;
    let fpc_ssu = fpc_ssu_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;

    let value_series = df.column(value_col)?;
    let value_str = value_series.cast(&DataType::String)?;
    let value_str = value_str.str()?;
    let mut levels: Vec<String> = value_str
        .unique()?
        .iter()
        .filter_map(|v| v.map(|s| s.to_string()))
        .collect();
    levels.sort();

    let mut level_vals: Vec<String> = Vec::new();
    let mut estimates: Vec<f64> = Vec::new();
    let mut ses: Vec<f64> = Vec::new();
    let mut variances: Vec<f64> = Vec::new();
    let mut dfs_vec: Vec<u32> = Vec::new();
    let mut ns: Vec<u32> = Vec::new();
    let mut deffs: Vec<f64> = Vec::new();
    let mut score_cols: Vec<Vec<f64>> = Vec::new();
    let n = active_count(weights, None);
    // One design for every level; df comes off its codes rather than a second densification.
    let design =
        build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method)?.with_calib(calib);
    let df_val = degrees_of_freedom_from_design(weights, &design, None);
    let active = design.domain_rows(weights.len(), rows.as_ref(), None);

    for lvl in &levels {
        let indicator: Vec<Option<f64>> = value_str
            .iter()
            .map(|v| match v {
                Some(val) if val == lvl => Some(1.0),
                Some(_) => Some(0.0),
                None => None,
            })
            .collect();
        let indicator_ca = Float64Chunked::from_slice_options("indicator".into(), &indicator);
        let (estimate, scores, srs_var) = match scale {
            LevelScale::Share => match point_estimate_mean(&indicator_ca, weights) {
                Ok(estimate) => (
                    estimate,
                    scores_mean(&indicator_ca, weights)?,
                    srs_variance_mean(&indicator_ca, weights, srs)?,
                ),
                Err(_) => (
                    f64::NAN,
                    Float64Chunked::full("scores".into(), 0.0, indicator_ca.len()),
                    f64::NAN,
                ),
            },
            LevelScale::Count => (
                point_estimate_total(&indicator_ca, weights)?,
                scores_total(&indicator_ca, weights)?,
                srs_variance_total(&indicator_ca, weights, srs)?,
            ),
        };
        let scores_arr: Vec<f64> = scores.iter().map(|s| s.unwrap_or(0.0)).collect();
        let (variance, se) = var_se(
            estimate,
            taylor_variance_apply_in(&scores_arr, &design, active.as_deref()),
        );
        let deff = if srs_var > 0.0 {
            variance / srs_var
        } else {
            f64::NAN
        };

        level_vals.push(lvl.clone());
        estimates.push(estimate);
        ses.push(se);
        variances.push(variance);
        dfs_vec.push(df_val);
        ns.push(n);
        deffs.push(deff);
        score_cols.push(scores_arr);
    }
    let cov = flat_cov_in(&score_cols, &design, &estimates, active.as_deref());
    let n_levels = level_vals.len();
    let out = df!["y" => vec![value_col; n_levels], "level" => level_vals, "est" => estimates,
        "se" => ses, "var" => variances, "df" => dfs_vec, "n" => ns, "deff" => deffs]?;
    Ok((out, cov))
}

/// Batched ungrouped proportions, as `compute_mean_multi`; rows match `compute_levels_ungrouped`.
#[allow(clippy::too_many_arguments)]
fn compute_prop_multi(
    df: &DataFrame,
    value_cols: &[String],
    weight_col: &str,
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>,
    domain_rows_col: Option<&str>,
    srs: SrsRef,
) -> EResult<DataFrame> {
    let weights = df.column(weight_col)?.f64()?;
    let rows = rows_column(df, domain_rows_col)?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc = fpc_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;
    let fpc_ssu = fpc_ssu_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;

    let n = active_count(weights, None);
    // Design indexed once; df taken off its codes (see `compute_mean_multi`).
    let design = build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
    let df_val = degrees_of_freedom_from_design(weights, &design, None);
    let active = design.domain_rows(weights.len(), rows.as_ref(), None);

    // Cast to String and enumerate sorted levels before the parallel region; closures only read.
    let value_series_str: Vec<Column> = value_cols
        .iter()
        .map(|vc| df.column(vc)?.cast(&DataType::String))
        .collect::<PolarsResult<Vec<_>>>()?;
    let value_strs: Vec<&StringChunked> = value_series_str
        .iter()
        .map(|s| s.str())
        .collect::<PolarsResult<Vec<_>>>()?;
    let levels_per_var: Vec<Vec<String>> = value_strs
        .iter()
        .map(|vs| -> EResult<Vec<String>> {
            let mut lv: Vec<String> = vs
                .unique()?
                .iter()
                .filter_map(|v| v.map(|s| s.to_string()))
                .collect();
            lv.sort();
            Ok(lv)
        })
        .collect::<EResult<Vec<_>>>()?;

    type PropRow = (String, String, f64, f64, f64, u32, f64); // y, level, est, se, var, n, deff
    let per_var = (0..value_cols.len())
        .into_par_iter()
        .map(|i| -> EResult<Vec<PropRow>> {
            let value_str = value_strs[i];
            let levels = &levels_per_var[i];
            let mut out: Vec<PropRow> = Vec::with_capacity(levels.len());
            for lvl in levels {
                let indicator: Vec<Option<f64>> = value_str
                    .iter()
                    .map(|v| match v {
                        Some(val) if val == lvl => Some(1.0),
                        Some(_) => Some(0.0),
                        None => None,
                    })
                    .collect();
                let indicator_ca =
                    Float64Chunked::from_slice_options("indicator".into(), &indicator);
                let Ok(estimate) = point_estimate_mean(&indicator_ca, weights) else {
                    let nan = f64::NAN;
                    out.push((value_cols[i].clone(), lvl.clone(), nan, nan, nan, n, nan));
                    continue;
                };
                let scores = scores_mean(&indicator_ca, weights)?;
                let scores_arr: Vec<f64> = scores.iter().map(|s| s.unwrap_or(0.0)).collect();
                let variance = taylor_variance_apply_in(&scores_arr, &design, active.as_deref());
                let se = variance.max(0.0).sqrt();
                let srs_var = srs_variance_mean(&indicator_ca, weights, srs)?;
                let deff = if srs_var > 0.0 {
                    variance / srs_var
                } else {
                    f64::NAN
                };
                out.push((
                    value_cols[i].clone(),
                    lvl.clone(),
                    estimate,
                    se,
                    variance,
                    n,
                    deff,
                ));
            }
            Ok(out)
        })
        .collect::<EResult<Vec<_>>>()?;

    let mut ys: Vec<String> = Vec::new();
    let mut level_vals: Vec<String> = Vec::new();
    let mut estimates: Vec<f64> = Vec::new();
    let mut ses: Vec<f64> = Vec::new();
    let mut variances: Vec<f64> = Vec::new();
    let mut ns: Vec<u32> = Vec::new();
    let mut deffs: Vec<f64> = Vec::new();
    for var_rows in per_var {
        for (y, lvl, est, se, var, cnt, deff) in var_rows {
            ys.push(y);
            level_vals.push(lvl);
            estimates.push(est);
            ses.push(se);
            variances.push(var);
            ns.push(cnt);
            deffs.push(deff);
        }
    }
    let n_rows = ys.len();
    let dfs = vec![df_val; n_rows];
    Ok(
        df!["y" => ys, "level" => level_vals, "est" => estimates, "se" => ses,
        "var" => variances, "df" => dfs, "n" => ns, "deff" => deffs]?,
    )
}

#[allow(clippy::too_many_arguments)]
fn compute_levels_grouped(
    df: &DataFrame,
    value_col: &str,
    weight_col: &str,
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_ssu_col: Option<&str>,
    by_col: &str,
    singleton_method: Option<&str>,
    domain_rows_col: Option<&str>,
    srs: SrsRef,
    calib: Option<CalibSweep>,
    scale: LevelScale,
) -> EResult<(DataFrame, Vec<f64>)> {
    let weights = df.column(weight_col)?.f64()?;
    let rows = rows_column(df, domain_rows_col)?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc = fpc_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;
    let fpc_ssu = fpc_ssu_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;

    let value_series = df.column(value_col)?;
    let value_str = value_series.cast(&DataType::String)?;
    let value_str = value_str.str()?;
    let mut levels: Vec<String> = value_str
        .unique()?
        .iter()
        .filter_map(|v| v.map(|s| s.to_string()))
        .collect();
    levels.sort();

    let by_str = df.column(by_col)?.str()?;
    // Group discovery and the design build are independent full passes.
    let (unique_groups, design) = rayon::join(
        || crate::estimation::sorted_groups(by_str),
        || build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method),
    );
    let unique_groups = unique_groups?;

    // Design is identical across all (group, level) cells; index it once.
    let design = design?.with_calib(calib);

    // Groups emit level rows in `levels` order, flattened in group order.
    type PropRow = (String, String, f64, f64, f64, u32, f64, Vec<f64>);
    let groups: Vec<&str> = unique_groups.iter().flatten().collect();
    // A by-group is a domain: its df counts its own active PSUs/strata, not the frame's.
    let masks: Vec<BooleanChunked> = groups.par_iter().map(|&g| by_str.equal(g)).collect();
    let group_dfs: Vec<u32> = masks
        .par_iter()
        .map(|m| degrees_of_freedom_from_design(weights, &design, Some(m)))
        .collect();
    let indicators: Vec<Float64Chunked> = levels
        .par_iter()
        .map(|lvl| {
            let indicator: Vec<Option<f64>> = value_str
                .iter()
                .map(|v| match v {
                    Some(val) if val == lvl => Some(1.0),
                    Some(_) => Some(0.0),
                    None => None,
                })
                .collect();
            Float64Chunked::from_slice_options("indicator".into(), &indicator)
        })
        .collect();
    let per_group = groups
        .par_iter()
        .zip(masks.par_iter())
        .map(|(&group, mask)| -> EResult<Vec<PropRow>> {
            let domain_mask = mask.clone();
            let active = design.domain_rows(weights.len(), rows.as_ref(), Some(&domain_mask));
            let n_domain = active_count(weights, Some(&domain_mask));
            let mut out: Vec<PropRow> = Vec::with_capacity(levels.len());
            for (lvl, indicator_ca) in levels.iter().zip(&indicators) {
                let (estimate, scores, srs_var) = match scale {
                    LevelScale::Share => {
                        match point_estimate_mean_domain(indicator_ca, weights, &domain_mask) {
                            Ok(estimate) => (
                                estimate,
                                scores_mean_domain(indicator_ca, weights, &domain_mask)?,
                                srs_variance_mean_domain(indicator_ca, weights, &domain_mask, srs)?,
                            ),
                            Err(_) => (
                                f64::NAN,
                                Float64Chunked::full("scores".into(), 0.0, indicator_ca.len()),
                                f64::NAN,
                            ),
                        }
                    }
                    LevelScale::Count => (
                        point_estimate_total_domain(indicator_ca, weights, &domain_mask)?,
                        scores_total_domain(indicator_ca, weights, &domain_mask)?,
                        srs_variance_total_domain(indicator_ca, weights, &domain_mask, srs)?,
                    ),
                };
                let scores_arr: Vec<f64> = scores.iter().map(|s| s.unwrap_or(0.0)).collect();
                let (variance, se) = var_se(
                    estimate,
                    taylor_variance_apply_in(&scores_arr, &design, active.as_deref()),
                );
                let deff = if srs_var > 0.0 {
                    variance / srs_var
                } else {
                    f64::NAN
                };
                out.push((
                    group.to_string(),
                    lvl.clone(),
                    estimate,
                    se,
                    variance,
                    n_domain,
                    deff,
                    scores_arr,
                ));
            }
            Ok(out)
        })
        .collect::<EResult<Vec<_>>>()?;

    let mut by_vals: Vec<String> = Vec::new();
    let mut level_vals: Vec<String> = Vec::new();
    let mut estimates: Vec<f64> = Vec::new();
    let mut ses: Vec<f64> = Vec::new();
    let mut variances: Vec<f64> = Vec::new();
    let mut ns: Vec<u32> = Vec::new();
    let mut deffs: Vec<f64> = Vec::new();
    let mut score_cols: Vec<Vec<f64>> = Vec::new();
    for group_rows in per_group {
        for (g, lvl, est, se, var, n, deff, scores_arr) in group_rows {
            by_vals.push(g);
            level_vals.push(lvl);
            estimates.push(est);
            ses.push(se);
            variances.push(var);
            ns.push(n);
            deffs.push(deff);
            score_cols.push(scores_arr);
        }
    }
    let cov = flat_cov(&score_cols, &design, &estimates);
    let n_rows = by_vals.len();
    let dfs_vec: Vec<u32> = group_dfs
        .iter()
        .flat_map(|d| std::iter::repeat_n(*d, levels.len()))
        .collect();
    debug_assert_eq!(dfs_vec.len(), n_rows);
    let out = df![by_col => by_vals, "y" => vec![value_col; n_rows], "level" => level_vals,
        "est" => estimates, "se" => ses, "var" => variances, "df" => dfs_vec, "n" => ns, "deff" => deffs]?;
    Ok((out, cov))
}

// ============================================================================
// Quantiles (median is the p = 0.5 case)
// ============================================================================

/// Woodruff quantiles for one variable: one row per probability (and domain), with a `prob` column.
#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, probs, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, by_col=None, singleton_method=None, domain_rows_col=None, quantile_method=None, calib_kind=None, calib_cells=None, calib_aux=None, calib_prev_wgt=None, calib_pins_total=None, calib_new_wgt=None))]
#[allow(clippy::too_many_arguments)]
pub fn taylor_quantile(
    _py: Python,
    data: PyDataFrame,
    value_col: String,
    weight_col: String,
    probs: Vec<f64>,
    strata_col: Option<String>,
    psu_col: Option<String>,
    ssu_col: Option<String>,
    fpc_col: Option<String>,
    fpc_ssu_col: Option<String>,
    by_col: Option<String>,
    singleton_method: Option<String>,
    domain_rows_col: Option<String>,
    quantile_method: Option<String>,
    calib_kind: Option<String>,
    calib_cells: Option<Vec<String>>,
    calib_aux: Option<Vec<String>>,
    calib_prev_wgt: Option<String>,
    calib_pins_total: Option<bool>,
    calib_new_wgt: Option<String>,
) -> PyResult<PyDataFrame> {
    let df = into_contiguous(data);
    let calib = make_calib(
        &df,
        &weight_col,
        calib_kind,
        calib_cells,
        calib_aux,
        calib_prev_wgt,
        calib_pins_total,
        calib_new_wgt,
    )
    .map_err(|e| to_py_err(&e))?;
    let q_method = quantile_method
        .as_deref()
        .map(SvyQuantileMethod::parse)
        .transpose()
        .map_err(|e| to_py_err(&e))?
        .unwrap_or(SvyQuantileMethod::Higher);

    let result = match by_col.as_deref() {
        None => compute_quantile_ungrouped(
            &df,
            &value_col,
            &weight_col,
            strata_col.as_deref(),
            psu_col.as_deref(),
            ssu_col.as_deref(),
            fpc_col.as_deref(),
            fpc_ssu_col.as_deref(),
            singleton_method.as_deref(),
            domain_rows_col.as_deref(),
            calib,
            &probs,
            q_method,
        ),
        Some(by) => compute_quantile_grouped(
            &df,
            &value_col,
            &weight_col,
            strata_col.as_deref(),
            psu_col.as_deref(),
            ssu_col.as_deref(),
            fpc_col.as_deref(),
            fpc_ssu_col.as_deref(),
            by,
            singleton_method.as_deref(),
            domain_rows_col.as_deref(),
            calib,
            &probs,
            q_method,
        ),
    }
    .map_err(|e| to_py_err(&e))?;
    Ok(PyDataFrame(result))
}

/// Batched ungrouped quantiles in parallel; rows variable-major, then by probability.
#[pyfunction]
#[pyo3(signature = (data, value_cols, weight_col, probs, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, singleton_method=None, domain_rows_col=None, quantile_method=None, calib_kind=None, calib_cells=None, calib_aux=None, calib_prev_wgt=None, calib_pins_total=None, calib_new_wgt=None))]
#[allow(clippy::too_many_arguments)]
pub fn taylor_quantile_multi(
    _py: Python,
    data: PyDataFrame,
    value_cols: Vec<String>,
    weight_col: String,
    probs: Vec<f64>,
    strata_col: Option<String>,
    psu_col: Option<String>,
    ssu_col: Option<String>,
    fpc_col: Option<String>,
    fpc_ssu_col: Option<String>,
    singleton_method: Option<String>,
    domain_rows_col: Option<String>,
    quantile_method: Option<String>,
    calib_kind: Option<String>,
    calib_cells: Option<Vec<String>>,
    calib_aux: Option<Vec<String>>,
    calib_prev_wgt: Option<String>,
    calib_pins_total: Option<bool>,
    calib_new_wgt: Option<String>,
) -> PyResult<PyDataFrame> {
    let df = into_contiguous(data);
    let calib = make_calib(
        &df,
        &weight_col,
        calib_kind,
        calib_cells,
        calib_aux,
        calib_prev_wgt,
        calib_pins_total,
        calib_new_wgt,
    )
    .map_err(|e| to_py_err(&e))?;
    let q_method = quantile_method
        .as_deref()
        .map(SvyQuantileMethod::parse)
        .transpose()
        .map_err(|e| to_py_err(&e))?
        .unwrap_or(SvyQuantileMethod::Higher);
    let result = _py
        .detach(|| {
            compute_quantile_multi(
                &df,
                &value_cols,
                &weight_col,
                strata_col.as_deref(),
                psu_col.as_deref(),
                ssu_col.as_deref(),
                fpc_col.as_deref(),
                fpc_ssu_col.as_deref(),
                singleton_method.as_deref(),
                domain_rows_col.as_deref(),
                calib,
                &probs,
                q_method,
            )
        })
        .map_err(|e| to_py_err(&e))?;
    Ok(PyDataFrame(result))
}

/// The weighted quantile rule at arbitrary probabilities, over a sorted variable and its CDF.
/// Python's Woodruff limits reuse this rule, as R uses one `method`/`f` for estimate and endpoints.
#[pyfunction]
#[pyo3(signature = (values_sorted, cdf, probs, quantile_method=None))]
pub fn weighted_quantile_at(
    values_sorted: PyReadonlyArray1<f64>,
    cdf: PyReadonlyArray1<f64>,
    probs: Vec<f64>,
    quantile_method: Option<String>,
) -> PyResult<Vec<f64>> {
    let y = values_sorted.as_slice()?;
    let c = cdf.as_slice()?;
    if y.len() != c.len() {
        return Err(to_py_err(&EstimationError::CdfLength {
            values: y.len(),
            cdf: c.len(),
        }));
    }
    let q_method = quantile_method
        .as_deref()
        .map(SvyQuantileMethod::parse)
        .transpose()
        .map_err(|e| to_py_err(&e))?
        .unwrap_or(SvyQuantileMethod::Higher);
    Ok(probs
        .iter()
        .map(|&p| weighted_quantile(y, c, p, q_method))
        .collect())
}

/// Drop the `prob` column: the median entry points report no probability.
fn without_prob(mut df: DataFrame) -> EResult<DataFrame> {
    let _ = df.drop_in_place("prob")?;
    Ok(df)
}

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, by_col=None, singleton_method=None, domain_rows_col=None, quantile_method=None, calib_kind=None, calib_cells=None, calib_aux=None, calib_prev_wgt=None, calib_pins_total=None, calib_new_wgt=None))]
#[allow(clippy::too_many_arguments)]
pub fn taylor_median(
    _py: Python,
    data: PyDataFrame,
    value_col: String,
    weight_col: String,
    strata_col: Option<String>,
    psu_col: Option<String>,
    ssu_col: Option<String>,
    fpc_col: Option<String>,
    fpc_ssu_col: Option<String>,
    by_col: Option<String>,
    singleton_method: Option<String>,
    domain_rows_col: Option<String>,
    quantile_method: Option<String>,
    calib_kind: Option<String>,
    calib_cells: Option<Vec<String>>,
    calib_aux: Option<Vec<String>>,
    calib_prev_wgt: Option<String>,
    calib_pins_total: Option<bool>,
    calib_new_wgt: Option<String>,
) -> PyResult<PyDataFrame> {
    let out = taylor_quantile(
        _py,
        data,
        value_col,
        weight_col,
        vec![0.5],
        strata_col,
        psu_col,
        ssu_col,
        fpc_col,
        fpc_ssu_col,
        by_col,
        singleton_method,
        domain_rows_col,
        quantile_method,
        calib_kind,
        calib_cells,
        calib_aux,
        calib_prev_wgt,
        calib_pins_total,
        calib_new_wgt,
    )?;
    let result = without_prob(out.0).map_err(|e| to_py_err(&e))?;
    Ok(PyDataFrame(result))
}

/// Batched ungrouped median over many variables; one row per variable, in input order.
#[pyfunction]
#[pyo3(signature = (data, value_cols, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, singleton_method=None, domain_rows_col=None, quantile_method=None, calib_kind=None, calib_cells=None, calib_aux=None, calib_prev_wgt=None, calib_pins_total=None, calib_new_wgt=None))]
#[allow(clippy::too_many_arguments)]
pub fn taylor_median_multi(
    _py: Python,
    data: PyDataFrame,
    value_cols: Vec<String>,
    weight_col: String,
    strata_col: Option<String>,
    psu_col: Option<String>,
    ssu_col: Option<String>,
    fpc_col: Option<String>,
    fpc_ssu_col: Option<String>,
    singleton_method: Option<String>,
    domain_rows_col: Option<String>,
    quantile_method: Option<String>,
    calib_kind: Option<String>,
    calib_cells: Option<Vec<String>>,
    calib_aux: Option<Vec<String>>,
    calib_prev_wgt: Option<String>,
    calib_pins_total: Option<bool>,
    calib_new_wgt: Option<String>,
) -> PyResult<PyDataFrame> {
    let out = taylor_quantile_multi(
        _py,
        data,
        value_cols,
        weight_col,
        vec![0.5],
        strata_col,
        psu_col,
        ssu_col,
        fpc_col,
        fpc_ssu_col,
        singleton_method,
        domain_rows_col,
        quantile_method,
        calib_kind,
        calib_cells,
        calib_aux,
        calib_prev_wgt,
        calib_pins_total,
        calib_new_wgt,
    )?;
    let result = without_prob(out.0).map_err(|e| to_py_err(&e))?;
    Ok(PyDataFrame(result))
}

/// The design columns every quantile path needs, resolved once.
struct QuantileCols<'a> {
    weights: &'a Float64Chunked,
    rows: Option<BooleanChunked>,
    strata: Option<&'a Column>,
    psu: Option<&'a Column>,
    design: TaylorDesign,
}

#[allow(clippy::too_many_arguments)]
fn resolve_quantile_cols<'a>(
    df: &'a DataFrame,
    weight_col: &str,
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>,
    domain_rows_col: Option<&str>,
    calib: Option<CalibSweep>,
) -> EResult<QuantileCols<'a>> {
    let weights = df.column(weight_col)?.f64()?;
    let rows = rows_column(df, domain_rows_col)?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc = fpc_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;
    let fpc_ssu = fpc_ssu_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;

    // One design build serves every (variable, prob) pair.
    let design =
        build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method)?.with_calib(calib);
    Ok(QuantileCols {
        weights,
        rows,
        strata,
        psu,
        design,
    })
}

#[allow(clippy::too_many_arguments)]
fn compute_quantile_ungrouped(
    df: &DataFrame,
    value_col: &str,
    weight_col: &str,
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>,
    domain_rows_col: Option<&str>,
    calib: Option<CalibSweep>,
    probs: &[f64],
    q_method: SvyQuantileMethod,
) -> EResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let cols = resolve_quantile_cols(
        df,
        weight_col,
        strata_col,
        psu_col,
        ssu_col,
        fpc_col,
        fpc_ssu_col,
        singleton_method,
        domain_rows_col,
        calib,
    )?;

    let rows = quantiles_woodruff(
        y,
        cols.weights,
        None,
        cols.rows.as_ref(),
        &cols.design,
        probs,
        q_method,
    )?;
    let df_val = degrees_of_freedom(cols.weights, cols.strata, cols.psu)?;
    let n = active_count(cols.weights, None);
    let k = rows.len();

    let (estimates, ses, variances) = unzip_woodruff(rows);
    Ok(
        df!["y" => vec![value_col; k], "prob" => probs.to_vec(), "est" => estimates,
        "se" => ses, "var" => variances, "df" => vec![df_val; k], "n" => vec![n; k]]?,
    )
}

/// Batched ungrouped quantiles: sort-bound, so the win is parallel variables, not design reuse.
#[allow(clippy::too_many_arguments)]
fn compute_quantile_multi(
    df: &DataFrame,
    value_cols: &[String],
    weight_col: &str,
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>,
    domain_rows_col: Option<&str>,
    calib: Option<CalibSweep>,
    probs: &[f64],
    q_method: SvyQuantileMethod,
) -> EResult<DataFrame> {
    let cols = resolve_quantile_cols(
        df,
        weight_col,
        strata_col,
        psu_col,
        ssu_col,
        fpc_col,
        fpc_ssu_col,
        singleton_method,
        domain_rows_col,
        calib,
    )?;

    // df is design-only, identical across variables — compute once.
    let df_val = degrees_of_freedom(cols.weights, cols.strata, cols.psu)?;

    let y_cols: Vec<&Float64Chunked> = value_cols
        .iter()
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .collect::<PolarsResult<Vec<_>>>()?;

    let per_var = (0..value_cols.len())
        .into_par_iter()
        .map(|i| {
            quantiles_woodruff(
                y_cols[i],
                cols.weights,
                None,
                cols.rows.as_ref(),
                &cols.design,
                probs,
                q_method,
            )
        })
        .collect::<EResult<Vec<_>>>()?;

    let k = probs.len();
    let n_rows = per_var.len() * k;
    let mut ys: Vec<String> = Vec::with_capacity(n_rows);
    let mut ns: Vec<u32> = Vec::with_capacity(n_rows);
    let mut flat: Vec<(f64, f64, f64)> = Vec::with_capacity(n_rows);
    for (i, rows) in per_var.into_iter().enumerate() {
        ys.extend(std::iter::repeat_n(value_cols[i].clone(), k));
        ns.extend(std::iter::repeat_n(active_count(cols.weights, None), k));
        flat.extend(rows);
    }

    let (estimates, ses, variances) = unzip_woodruff(flat);
    let probs_rep: Vec<f64> = std::iter::repeat_n(probs, value_cols.len())
        .flatten()
        .copied()
        .collect();
    Ok(
        df!["y" => ys, "prob" => probs_rep, "est" => estimates, "se" => ses,
        "var" => variances, "df" => vec![df_val; n_rows], "n" => ns]?,
    )
}

#[allow(clippy::too_many_arguments)]
fn compute_quantile_grouped(
    df: &DataFrame,
    value_col: &str,
    weight_col: &str,
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_ssu_col: Option<&str>,
    by_col: &str,
    singleton_method: Option<&str>,
    domain_rows_col: Option<&str>,
    calib: Option<CalibSweep>,
    probs: &[f64],
    q_method: SvyQuantileMethod,
) -> EResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let cols = resolve_quantile_cols(
        df,
        weight_col,
        strata_col,
        psu_col,
        ssu_col,
        fpc_col,
        fpc_ssu_col,
        singleton_method,
        domain_rows_col,
        calib,
    )?;
    let by_str = df.column(by_col)?.str()?;
    let unique_groups = crate::estimation::sorted_groups(by_str)?;

    let k = probs.len();
    let mut by_vals: Vec<&str> = Vec::new();
    let mut dfs: Vec<u32> = Vec::new();
    let mut ns: Vec<u32> = Vec::new();
    let mut flat: Vec<(f64, f64, f64)> = Vec::new();

    for group in unique_groups.iter().flatten() {
        let domain_mask = by_str.equal(group);
        let n_domain = active_count(cols.weights, Some(&domain_mask));
        let rows = quantiles_woodruff(
            y,
            cols.weights,
            Some(&domain_mask),
            cols.rows.as_ref(),
            &cols.design,
            probs,
            q_method,
        )?;
        // Per-group df: see the note in compute_mean_grouped.
        let df_val = degrees_of_freedom_from_design(cols.weights, &cols.design, Some(&domain_mask));

        by_vals.extend(std::iter::repeat_n(group, k));
        dfs.extend(std::iter::repeat_n(df_val, k));
        ns.extend(std::iter::repeat_n(n_domain, k));
        flat.extend(rows);
    }

    let n_rows = by_vals.len();
    let probs_rep: Vec<f64> = std::iter::repeat_n(probs, n_rows / k.max(1))
        .flatten()
        .copied()
        .collect();
    let (estimates, ses, variances) = unzip_woodruff(flat);
    Ok(
        df![by_col => by_vals, "y" => vec![value_col; n_rows], "prob" => probs_rep,
        "est" => estimates, "se" => ses, "var" => variances, "df" => dfs, "n" => ns]?,
    )
}

fn unzip_woodruff(rows: Vec<(f64, f64, f64)>) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut estimates = Vec::with_capacity(rows.len());
    let mut ses = Vec::with_capacity(rows.len());
    let mut variances = Vec::with_capacity(rows.len());
    for (est, var, se) in rows {
        estimates.push(est);
        ses.push(se);
        variances.push(var);
    }
    (estimates, ses, variances)
}
