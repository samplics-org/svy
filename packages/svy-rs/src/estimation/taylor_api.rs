// src/estimation/taylor_api.rs
//
// PyO3-facing wrappers and `compute_*` helpers for Taylor linearization.
// The heavy math lives in taylor.rs; this file only handles DataFrame I/O,
// argument parsing, and looping over by-groups.

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use rayon::prelude::*;

use crate::estimation::taylor::{
    SvyQuantileMethod,
    build_taylor_design,
    degrees_of_freedom,
    median_variance_woodruff,
    median_variance_woodruff_domain,
    point_estimate_mean, point_estimate_mean_domain,
    point_estimate_ratio, point_estimate_ratio_domain,
    point_estimate_total, point_estimate_total_domain,
    scores_mean, scores_mean_domain,
    scores_ratio, scores_ratio_domain,
    scores_total, scores_total_domain,
    srs_variance_mean, srs_variance_mean_domain,
    srs_variance_ratio, srs_variance_ratio_domain,
    srs_variance_total, srs_variance_total_domain,
    taylor_variance, taylor_variance_apply,
    weighted_median, weighted_median_domain,
};

/// Convert the incoming Python DataFrame and ensure one chunk per column.
///
/// After `prepare_data` the frame is usually already single-chunk, but scaled or
/// concatenated inputs can arrive fragmented. A single rechunk here (one copy)
/// lets every downstream kernel take its contiguous `cont_slice` fast path
/// instead of the per-element chunked-iterator fallback.
fn into_contiguous(data: PyDataFrame) -> DataFrame {
    let mut df: DataFrame = data.into();
    if df.first_col_n_chunks() > 1 {
        df.as_single_chunk_par();
    }
    df
}

// ============================================================================
// Mean
// ============================================================================

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, by_col=None, singleton_method=None))]
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
) -> PyResult<PyDataFrame> {
    let df = into_contiguous(data);

    if by_col.is_none() {
        let result = compute_mean_ungrouped(
            &df, &value_col, &weight_col,
            strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(),
            fpc_col.as_deref(), fpc_ssu_col.as_deref(), singleton_method.as_deref(),
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        return Ok(PyDataFrame(result));
    }

    let by = by_col.unwrap();
    let result = _py
        .detach(|| {
            compute_mean_grouped(
                &df, &value_col, &weight_col,
                strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(),
                fpc_col.as_deref(), fpc_ssu_col.as_deref(),
                &by, singleton_method.as_deref(),
            )
        })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(PyDataFrame(result))
}

/// Batched ungrouped mean over many variables sharing one design (see
/// `compute_mean_multi`). Returns one row per variable, in input order.
#[pyfunction]
#[pyo3(signature = (data, value_cols, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, singleton_method=None))]
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
) -> PyResult<PyDataFrame> {
    let df = into_contiguous(data);
    let result = _py
        .detach(|| {
            compute_mean_multi(
                &df, &value_cols, &weight_col,
                strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(),
                fpc_col.as_deref(), fpc_ssu_col.as_deref(), singleton_method.as_deref(),
            )
        })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(PyDataFrame(result))
}

fn compute_mean_ungrouped(
    df: &DataFrame,
    value_col: &str, weight_col: &str,
    strata_col: Option<&str>, psu_col: Option<&str>, ssu_col: Option<&str>,
    fpc_col: Option<&str>, fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>,
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu    = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu    = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc    = fpc_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;
    let fpc_ssu = fpc_ssu_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;

    let estimate = point_estimate_mean(y, weights)?;
    let scores   = scores_mean(y, weights)?;
    let variance = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
    let se       = variance.max(0.0).sqrt();
    let df_val   = degrees_of_freedom(weights, strata, psu)?;
    let n        = y.len() as u32;
    let srs_var  = srs_variance_mean(y, weights)?;
    let deff     = if srs_var > 0.0 { variance / srs_var } else { f64::NAN };

    df!["y" => vec![value_col], "est" => vec![estimate], "se" => vec![se],
        "var" => vec![variance], "df" => vec![df_val], "n" => vec![n], "deff" => vec![deff]]
}

/// Batched ungrouped means: build the design ONCE and estimate every variable
/// against it, in parallel. The design build (index strata/PSU, per-stratum
/// maps, FPC) is ~half the cost of a single call and identical across variables,
/// so amortising it over N variables is the whole win. Each variable is still
/// computed with the same kernels as `compute_mean_ungrouped`, so every row is
/// bit-identical to the corresponding single-variable call. Output is one row
/// per variable, in input order.
fn compute_mean_multi(
    df: &DataFrame,
    value_cols: &[String], weight_col: &str,
    strata_col: Option<&str>, psu_col: Option<&str>, ssu_col: Option<&str>,
    fpc_col: Option<&str>, fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>,
) -> PolarsResult<DataFrame> {
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu    = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu    = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc    = fpc_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;
    let fpc_ssu = fpc_ssu_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;

    let df_val = degrees_of_freedom(weights, strata, psu)?;
    let design = build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;

    // Resolve every response column to its typed slice BEFORE fanning out.
    // `df.column()` mutates the frame's internal schema cache, so calling it
    // from parallel closures is a data race; hoisting it makes each worker read
    // only already-borrowed, immutable `&Float64Chunked`s (deterministic).
    let y_cols: Vec<&Float64Chunked> = value_cols
        .iter()
        .map(|vc| df.column(vc).and_then(|c| c.f64()))
        .collect::<PolarsResult<Vec<_>>>()?;

    let rows = (0..value_cols.len())
        .into_par_iter()
        .map(|i| -> PolarsResult<(String, f64, f64, f64, u32, f64)> {
            let y        = y_cols[i];
            let estimate = point_estimate_mean(y, weights)?;
            let scores   = scores_mean(y, weights)?;
            let scores_arr: Vec<f64> = scores.iter().map(|s| s.unwrap_or(0.0)).collect();
            let variance = taylor_variance_apply(&scores_arr, &design);
            let se       = variance.max(0.0).sqrt();
            let n        = y.len() as u32;
            let srs_var  = srs_variance_mean(y, weights)?;
            let deff     = if srs_var > 0.0 { variance / srs_var } else { f64::NAN };
            Ok((value_cols[i].clone(), estimate, se, variance, n, deff))
        })
        .collect::<PolarsResult<Vec<_>>>()?;

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
    df!["y" => ys, "est" => estimates, "se" => ses, "var" => variances,
        "df" => dfs, "n" => ns, "deff" => deffs]
}

fn compute_mean_grouped(
    df: &DataFrame,
    value_col: &str, weight_col: &str,
    strata_col: Option<&str>, psu_col: Option<&str>, ssu_col: Option<&str>,
    fpc_col: Option<&str>, fpc_ssu_col: Option<&str>,
    by_col: &str, singleton_method: Option<&str>,
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu    = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu    = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc    = fpc_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;
    let fpc_ssu = fpc_ssu_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;
    let by_str = df.column(by_col)?.str()?;
    let unique_groups = by_str.unique()?;

    let df_val = degrees_of_freedom(weights, strata, psu)?;
    // Index the design once — it is identical across by-groups; only the
    // domain-masked scores change per group.
    let design = build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;

    // Groups are independent; fan the per-group work out over the rayon pool
    // and collect in group order (deterministic, thread-count-independent).
    let groups: Vec<&str> = unique_groups.iter().flatten().collect();
    let rows = groups
        .par_iter()
        .map(|&group| -> PolarsResult<(&str, f64, f64, f64, u32, f64)> {
            let domain_mask = by_str.equal(group);
            let n_domain    = domain_mask.sum().unwrap_or(0) as u32;
            let estimate    = point_estimate_mean_domain(y, weights, &domain_mask)?;
            let scores      = scores_mean_domain(y, weights, &domain_mask)?;
            let scores_arr: Vec<f64> = scores.iter().map(|s| s.unwrap_or(0.0)).collect();
            let variance    = taylor_variance_apply(&scores_arr, &design);
            let se          = variance.max(0.0).sqrt();
            let srs_var     = srs_variance_mean_domain(y, weights, &domain_mask)?;
            let deff        = if srs_var > 0.0 { variance / srs_var } else { f64::NAN };
            Ok((group, estimate, se, variance, n_domain, deff))
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    let n_groups = rows.len();
    let mut by_vals: Vec<&str> = Vec::with_capacity(n_groups);
    let mut estimates: Vec<f64> = Vec::with_capacity(n_groups);
    let mut ses: Vec<f64> = Vec::with_capacity(n_groups);
    let mut variances: Vec<f64> = Vec::with_capacity(n_groups);
    let mut ns: Vec<u32> = Vec::with_capacity(n_groups);
    let mut deffs: Vec<f64> = Vec::with_capacity(n_groups);
    for (g, est, se, var, n, deff) in rows {
        by_vals.push(g);
        estimates.push(est);
        ses.push(se);
        variances.push(var);
        ns.push(n);
        deffs.push(deff);
    }
    let dfs = vec![df_val; n_groups];
    df![by_col => by_vals, "y" => vec![value_col; n_groups], "est" => estimates,
        "se" => ses, "var" => variances, "df" => dfs, "n" => ns, "deff" => deffs]
}

// ============================================================================
// Total
// ============================================================================

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, by_col=None, singleton_method=None))]
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
) -> PyResult<PyDataFrame> {
    let df = into_contiguous(data);
    if by_col.is_none() {
        let result = compute_total_ungrouped(
            &df, &value_col, &weight_col,
            strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(),
            fpc_col.as_deref(), fpc_ssu_col.as_deref(), singleton_method.as_deref(),
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        return Ok(PyDataFrame(result));
    }
    let by = by_col.unwrap();
    let result = _py
        .detach(|| {
            compute_total_grouped(
                &df, &value_col, &weight_col,
                strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(),
                fpc_col.as_deref(), fpc_ssu_col.as_deref(),
                &by, singleton_method.as_deref(),
            )
        })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(PyDataFrame(result))
}

/// Batched ungrouped total over many variables sharing one design build.
#[pyfunction]
#[pyo3(signature = (data, value_cols, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, singleton_method=None))]
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
) -> PyResult<PyDataFrame> {
    let df = into_contiguous(data);
    let result = _py
        .detach(|| {
            compute_total_multi(
                &df, &value_cols, &weight_col,
                strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(),
                fpc_col.as_deref(), fpc_ssu_col.as_deref(), singleton_method.as_deref(),
            )
        })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(PyDataFrame(result))
}

fn compute_total_ungrouped(
    df: &DataFrame,
    value_col: &str, weight_col: &str,
    strata_col: Option<&str>, psu_col: Option<&str>, ssu_col: Option<&str>,
    fpc_col: Option<&str>, fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>,
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu    = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu    = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc    = fpc_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;
    let fpc_ssu = fpc_ssu_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;

    let estimate = point_estimate_total(y, weights)?;
    let scores   = scores_total(y, weights)?;
    let variance = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
    let se       = variance.max(0.0).sqrt();
    let df_val   = degrees_of_freedom(weights, strata, psu)?;
    let n        = y.len() as u32;
    let srs_var  = srs_variance_total(y, weights)?;
    let deff     = if srs_var > 0.0 { variance / srs_var } else { f64::NAN };

    df!["y" => vec![value_col], "est" => vec![estimate], "se" => vec![se],
        "var" => vec![variance], "df" => vec![df_val], "n" => vec![n], "deff" => vec![deff]]
}

/// Batched ungrouped totals: design built once, variables in parallel. See
/// `compute_mean_multi`. Each row is bit-identical to `compute_total_ungrouped`.
fn compute_total_multi(
    df: &DataFrame,
    value_cols: &[String], weight_col: &str,
    strata_col: Option<&str>, psu_col: Option<&str>, ssu_col: Option<&str>,
    fpc_col: Option<&str>, fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>,
) -> PolarsResult<DataFrame> {
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu    = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu    = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc    = fpc_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;
    let fpc_ssu = fpc_ssu_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;

    let df_val = degrees_of_freedom(weights, strata, psu)?;
    let design = build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;

    // Hoist column resolution out of the parallel region (see compute_mean_multi).
    let y_cols: Vec<&Float64Chunked> = value_cols
        .iter()
        .map(|vc| df.column(vc).and_then(|c| c.f64()))
        .collect::<PolarsResult<Vec<_>>>()?;

    let rows = (0..value_cols.len())
        .into_par_iter()
        .map(|i| -> PolarsResult<(String, f64, f64, f64, u32, f64)> {
            let y        = y_cols[i];
            let estimate = point_estimate_total(y, weights)?;
            let scores   = scores_total(y, weights)?;
            let scores_arr: Vec<f64> = scores.iter().map(|s| s.unwrap_or(0.0)).collect();
            let variance = taylor_variance_apply(&scores_arr, &design);
            let se       = variance.max(0.0).sqrt();
            let n        = y.len() as u32;
            let srs_var  = srs_variance_total(y, weights)?;
            let deff     = if srs_var > 0.0 { variance / srs_var } else { f64::NAN };
            Ok((value_cols[i].clone(), estimate, se, variance, n, deff))
        })
        .collect::<PolarsResult<Vec<_>>>()?;

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
    df!["y" => ys, "est" => estimates, "se" => ses, "var" => variances,
        "df" => dfs, "n" => ns, "deff" => deffs]
}

fn compute_total_grouped(
    df: &DataFrame,
    value_col: &str, weight_col: &str,
    strata_col: Option<&str>, psu_col: Option<&str>, ssu_col: Option<&str>,
    fpc_col: Option<&str>, fpc_ssu_col: Option<&str>,
    by_col: &str, singleton_method: Option<&str>,
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu    = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu    = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc    = fpc_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;
    let fpc_ssu = fpc_ssu_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;
    let by_str = df.column(by_col)?.str()?;
    let unique_groups = by_str.unique()?;

    let df_val = degrees_of_freedom(weights, strata, psu)?;
    let design = build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;

    let groups: Vec<&str> = unique_groups.iter().flatten().collect();
    let rows = groups
        .par_iter()
        .map(|&group| -> PolarsResult<(&str, f64, f64, f64, u32, f64)> {
            let domain_mask = by_str.equal(group);
            let n_domain    = domain_mask.sum().unwrap_or(0) as u32;
            let estimate    = point_estimate_total_domain(y, weights, &domain_mask)?;
            let scores      = scores_total_domain(y, weights, &domain_mask)?;
            let scores_arr: Vec<f64> = scores.iter().map(|s| s.unwrap_or(0.0)).collect();
            let variance    = taylor_variance_apply(&scores_arr, &design);
            let se          = variance.max(0.0).sqrt();
            let srs_var     = srs_variance_total_domain(y, weights, &domain_mask)?;
            let deff        = if srs_var > 0.0 { variance / srs_var } else { f64::NAN };
            Ok((group, estimate, se, variance, n_domain, deff))
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    let n_groups = rows.len();
    let mut by_vals: Vec<&str> = Vec::with_capacity(n_groups);
    let mut estimates: Vec<f64> = Vec::with_capacity(n_groups);
    let mut ses: Vec<f64> = Vec::with_capacity(n_groups);
    let mut variances: Vec<f64> = Vec::with_capacity(n_groups);
    let mut ns: Vec<u32> = Vec::with_capacity(n_groups);
    let mut deffs: Vec<f64> = Vec::with_capacity(n_groups);
    for (g, est, se, var, n, deff) in rows {
        by_vals.push(g);
        estimates.push(est);
        ses.push(se);
        variances.push(var);
        ns.push(n);
        deffs.push(deff);
    }
    let dfs = vec![df_val; n_groups];
    df![by_col => by_vals, "y" => vec![value_col; n_groups], "est" => estimates,
        "se" => ses, "var" => variances, "df" => dfs, "n" => ns, "deff" => deffs]
}

// ============================================================================
// Ratio
// ============================================================================

#[pyfunction]
#[pyo3(signature = (data, numerator_col, denominator_col, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, by_col=None, singleton_method=None))]
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
) -> PyResult<PyDataFrame> {
    let df = into_contiguous(data);
    if by_col.is_none() {
        let result = compute_ratio_ungrouped(
            &df, &numerator_col, &denominator_col, &weight_col,
            strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(),
            fpc_col.as_deref(), fpc_ssu_col.as_deref(), singleton_method.as_deref(),
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        return Ok(PyDataFrame(result));
    }
    let by = by_col.unwrap();
    let result = _py
        .detach(|| {
            compute_ratio_grouped(
                &df, &numerator_col, &denominator_col, &weight_col,
                strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(),
                fpc_col.as_deref(), fpc_ssu_col.as_deref(),
                &by, singleton_method.as_deref(),
            )
        })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(PyDataFrame(result))
}

/// Batched ungrouped ratio over paired numerator/denominator columns sharing one
/// design build. `numerator_cols` and `denominator_cols` must be equal length.
#[pyfunction]
#[pyo3(signature = (data, numerator_cols, denominator_cols, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, singleton_method=None))]
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
) -> PyResult<PyDataFrame> {
    let df = into_contiguous(data);
    let result = _py
        .detach(|| {
            compute_ratio_multi(
                &df, &numerator_cols, &denominator_cols, &weight_col,
                strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(),
                fpc_col.as_deref(), fpc_ssu_col.as_deref(), singleton_method.as_deref(),
            )
        })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(PyDataFrame(result))
}

fn compute_ratio_ungrouped(
    df: &DataFrame,
    numerator_col: &str, denominator_col: &str, weight_col: &str,
    strata_col: Option<&str>, psu_col: Option<&str>, ssu_col: Option<&str>,
    fpc_col: Option<&str>, fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>,
) -> PolarsResult<DataFrame> {
    let y = df.column(numerator_col)?.f64()?;
    let x = df.column(denominator_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu    = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu    = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc    = fpc_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;
    let fpc_ssu = fpc_ssu_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;

    let estimate = point_estimate_ratio(y, x, weights)?;
    let scores   = scores_ratio(y, x, weights)?;
    let variance = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
    let se       = variance.max(0.0).sqrt();
    let df_val   = degrees_of_freedom(weights, strata, psu)?;
    let n        = y.len() as u32;
    let srs_var  = srs_variance_ratio(y, x, weights)?;
    let deff     = if srs_var > 0.0 { variance / srs_var } else { f64::NAN };

    df!["y" => vec![numerator_col], "x" => vec![denominator_col], "est" => vec![estimate],
        "se" => vec![se], "var" => vec![variance], "df" => vec![df_val], "n" => vec![n], "deff" => vec![deff]]
}

/// Batched ungrouped ratios: design built once, (numerator, denominator) pairs
/// estimated in parallel. See `compute_mean_multi`. One row per pair, in order.
fn compute_ratio_multi(
    df: &DataFrame,
    numerator_cols: &[String], denominator_cols: &[String], weight_col: &str,
    strata_col: Option<&str>, psu_col: Option<&str>, ssu_col: Option<&str>,
    fpc_col: Option<&str>, fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>,
) -> PolarsResult<DataFrame> {
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu    = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu    = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc    = fpc_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;
    let fpc_ssu = fpc_ssu_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;

    let df_val = degrees_of_freedom(weights, strata, psu)?;
    let design = build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;

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
        .map(|i| -> PolarsResult<(String, String, f64, f64, f64, u32, f64)> {
            let y = y_cols[i];
            let x = x_cols[i];
            let estimate = point_estimate_ratio(y, x, weights)?;
            let scores   = scores_ratio(y, x, weights)?;
            let scores_arr: Vec<f64> = scores.iter().map(|s| s.unwrap_or(0.0)).collect();
            let variance = taylor_variance_apply(&scores_arr, &design);
            let se       = variance.max(0.0).sqrt();
            let n        = y.len() as u32;
            let srs_var  = srs_variance_ratio(y, x, weights)?;
            let deff     = if srs_var > 0.0 { variance / srs_var } else { f64::NAN };
            Ok((
                numerator_cols[i].clone(),
                denominator_cols[i].clone(),
                estimate, se, variance, n, deff,
            ))
        })
        .collect::<PolarsResult<Vec<_>>>()?;

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
    df!["y" => ys, "x" => xs, "est" => estimates, "se" => ses, "var" => variances,
        "df" => dfs, "n" => ns, "deff" => deffs]
}

fn compute_ratio_grouped(
    df: &DataFrame,
    numerator_col: &str, denominator_col: &str, weight_col: &str,
    strata_col: Option<&str>, psu_col: Option<&str>, ssu_col: Option<&str>,
    fpc_col: Option<&str>, fpc_ssu_col: Option<&str>,
    by_col: &str, singleton_method: Option<&str>,
) -> PolarsResult<DataFrame> {
    let y = df.column(numerator_col)?.f64()?;
    let x = df.column(denominator_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu    = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu    = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc    = fpc_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;
    let fpc_ssu = fpc_ssu_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;
    let by_str = df.column(by_col)?.str()?;
    let unique_groups = by_str.unique()?;

    let df_val = degrees_of_freedom(weights, strata, psu)?;
    let design = build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;

    let groups: Vec<&str> = unique_groups.iter().flatten().collect();
    let rows = groups
        .par_iter()
        .map(|&group| -> PolarsResult<(&str, f64, f64, f64, u32, f64)> {
            let domain_mask = by_str.equal(group);
            let n_domain    = domain_mask.sum().unwrap_or(0) as u32;
            let estimate    = point_estimate_ratio_domain(y, x, weights, &domain_mask)?;
            let scores      = scores_ratio_domain(y, x, weights, &domain_mask)?;
            let scores_arr: Vec<f64> = scores.iter().map(|s| s.unwrap_or(0.0)).collect();
            let variance    = taylor_variance_apply(&scores_arr, &design);
            let se          = variance.max(0.0).sqrt();
            let srs_var     = srs_variance_ratio_domain(y, x, weights, &domain_mask)?;
            let deff        = if srs_var > 0.0 { variance / srs_var } else { f64::NAN };
            Ok((group, estimate, se, variance, n_domain, deff))
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    let n_groups = rows.len();
    let mut by_vals: Vec<&str> = Vec::with_capacity(n_groups);
    let mut estimates: Vec<f64> = Vec::with_capacity(n_groups);
    let mut ses: Vec<f64> = Vec::with_capacity(n_groups);
    let mut variances: Vec<f64> = Vec::with_capacity(n_groups);
    let mut ns: Vec<u32> = Vec::with_capacity(n_groups);
    let mut deffs: Vec<f64> = Vec::with_capacity(n_groups);
    for (g, est, se, var, n, deff) in rows {
        by_vals.push(g);
        estimates.push(est);
        ses.push(se);
        variances.push(var);
        ns.push(n);
        deffs.push(deff);
    }
    let dfs = vec![df_val; n_groups];
    df![by_col => by_vals, "y" => vec![numerator_col; n_groups], "x" => vec![denominator_col; n_groups],
        "est" => estimates, "se" => ses, "var" => variances, "df" => dfs, "n" => ns, "deff" => deffs]
}

// ============================================================================
// Proportion
// ============================================================================

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, by_col=None, singleton_method=None))]
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
) -> PyResult<PyDataFrame> {
    let df = into_contiguous(data);
    if by_col.is_none() {
        let result = compute_prop_ungrouped(
            &df, &value_col, &weight_col,
            strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(),
            fpc_col.as_deref(), fpc_ssu_col.as_deref(), singleton_method.as_deref(),
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        return Ok(PyDataFrame(result));
    }
    let by = by_col.unwrap();
    let result = _py
        .detach(|| {
            compute_prop_grouped(
                &df, &value_col, &weight_col,
                strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(),
                fpc_col.as_deref(), fpc_ssu_col.as_deref(),
                &by, singleton_method.as_deref(),
            )
        })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(PyDataFrame(result))
}

/// Batched ungrouped proportions over many category columns sharing one design
/// build. Rows are (variable, level), grouped by variable in input order.
#[pyfunction]
#[pyo3(signature = (data, value_cols, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, singleton_method=None))]
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
) -> PyResult<PyDataFrame> {
    let df = into_contiguous(data);
    let result = _py
        .detach(|| {
            compute_prop_multi(
                &df, &value_cols, &weight_col,
                strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(),
                fpc_col.as_deref(), fpc_ssu_col.as_deref(), singleton_method.as_deref(),
            )
        })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(PyDataFrame(result))
}

fn compute_prop_ungrouped(
    df: &DataFrame,
    value_col: &str, weight_col: &str,
    strata_col: Option<&str>, psu_col: Option<&str>, ssu_col: Option<&str>,
    fpc_col: Option<&str>, fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>,
) -> PolarsResult<DataFrame> {
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu    = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu    = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc    = fpc_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;
    let fpc_ssu = fpc_ssu_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;

    let value_series = df.column(value_col)?;
    let value_str    = value_series.cast(&DataType::String)?;
    let value_str    = value_str.str()?;
    let mut levels: Vec<String> = value_str.unique()?.iter()
        .filter_map(|v| v.map(|s| s.to_string())).collect();
    levels.sort();

    let mut level_vals: Vec<String> = Vec::new();
    let mut estimates: Vec<f64> = Vec::new();
    let mut ses: Vec<f64> = Vec::new();
    let mut variances: Vec<f64> = Vec::new();
    let mut dfs_vec: Vec<u32> = Vec::new();
    let mut ns: Vec<u32> = Vec::new();
    let mut deffs: Vec<f64> = Vec::new();
    let df_val = degrees_of_freedom(weights, strata, psu)?;
    let n = weights.len() as u32;
    // Design is identical across levels; index it once.
    let design = build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;

    for lvl in &levels {
        let indicator: Vec<Option<f64>> = value_str.iter()
            .map(|v| match v {
                Some(val) if val == lvl => Some(1.0),
                Some(_) => Some(0.0),
                None => None,
            })
            .collect();
        let indicator_ca = Float64Chunked::from_slice_options("indicator".into(), &indicator);
        let estimate = point_estimate_mean(&indicator_ca, weights)?;
        let scores   = scores_mean(&indicator_ca, weights)?;
        let scores_arr: Vec<f64> = scores.iter().map(|s| s.unwrap_or(0.0)).collect();
        let variance = taylor_variance_apply(&scores_arr, &design);
        let se       = variance.max(0.0).sqrt();
        let srs_var  = srs_variance_mean(&indicator_ca, weights)?;
        let deff     = if srs_var > 0.0 { variance / srs_var } else { f64::NAN };

        level_vals.push(lvl.clone());
        estimates.push(estimate);
        ses.push(se);
        variances.push(variance);
        dfs_vec.push(df_val);
        ns.push(n);
        deffs.push(deff);
    }
    let n_levels = level_vals.len();
    df!["y" => vec![value_col; n_levels], "level" => level_vals, "est" => estimates,
        "se" => ses, "var" => variances, "df" => dfs_vec, "n" => ns, "deff" => deffs]
}

/// Batched ungrouped proportions: design built once, variables estimated in
/// parallel (each variable loops its own category levels). See
/// `compute_mean_multi`. Rows are (variable, level), grouped by variable in
/// input order, levels sorted — identical to per-variable `compute_prop_ungrouped`.
fn compute_prop_multi(
    df: &DataFrame,
    value_cols: &[String], weight_col: &str,
    strata_col: Option<&str>, psu_col: Option<&str>, ssu_col: Option<&str>,
    fpc_col: Option<&str>, fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>,
) -> PolarsResult<DataFrame> {
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu    = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu    = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc    = fpc_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;
    let fpc_ssu = fpc_ssu_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;

    let df_val = degrees_of_freedom(weights, strata, psu)?;
    let n = weights.len() as u32;
    let design = build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;

    // Hoist String-cast + level enumeration out of the parallel region: keep the
    // owned casted columns alive, borrow their StringChunked, and precompute each
    // variable's sorted levels. Closures then only read immutable data.
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
        .map(|vs| -> PolarsResult<Vec<String>> {
            let mut lv: Vec<String> = vs
                .unique()?
                .iter()
                .filter_map(|v| v.map(|s| s.to_string()))
                .collect();
            lv.sort();
            Ok(lv)
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    type PropRow = (String, String, f64, f64, f64, u32, f64); // y, level, est, se, var, n, deff
    let per_var = (0..value_cols.len())
        .into_par_iter()
        .map(|i| -> PolarsResult<Vec<PropRow>> {
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
                let estimate = point_estimate_mean(&indicator_ca, weights)?;
                let scores   = scores_mean(&indicator_ca, weights)?;
                let scores_arr: Vec<f64> = scores.iter().map(|s| s.unwrap_or(0.0)).collect();
                let variance = taylor_variance_apply(&scores_arr, &design);
                let se       = variance.max(0.0).sqrt();
                let srs_var  = srs_variance_mean(&indicator_ca, weights)?;
                let deff     = if srs_var > 0.0 { variance / srs_var } else { f64::NAN };
                out.push((value_cols[i].clone(), lvl.clone(), estimate, se, variance, n, deff));
            }
            Ok(out)
        })
        .collect::<PolarsResult<Vec<_>>>()?;

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
    df!["y" => ys, "level" => level_vals, "est" => estimates, "se" => ses,
        "var" => variances, "df" => dfs, "n" => ns, "deff" => deffs]
}

fn compute_prop_grouped(
    df: &DataFrame,
    value_col: &str, weight_col: &str,
    strata_col: Option<&str>, psu_col: Option<&str>, ssu_col: Option<&str>,
    fpc_col: Option<&str>, fpc_ssu_col: Option<&str>,
    by_col: &str, singleton_method: Option<&str>,
) -> PolarsResult<DataFrame> {
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu    = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu    = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc    = fpc_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;
    let fpc_ssu = fpc_ssu_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;

    let value_series = df.column(value_col)?;
    let value_str    = value_series.cast(&DataType::String)?;
    let value_str    = value_str.str()?;
    let mut levels: Vec<String> = value_str.unique()?.iter()
        .filter_map(|v| v.map(|s| s.to_string())).collect();
    levels.sort();

    let by_str = df.column(by_col)?.str()?;
    let unique_groups = by_str.unique()?;

    let df_val = degrees_of_freedom(weights, strata, psu)?;
    // Design is identical across all (group, level) cells; index it once.
    let design = build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;

    // Fan out over groups; each group emits its level rows in `levels` order,
    // then flatten in group order for a deterministic layout.
    type PropRow = (String, String, f64, f64, f64, u32, f64);
    let groups: Vec<&str> = unique_groups.iter().flatten().collect();
    let per_group = groups
        .par_iter()
        .map(|&group| -> PolarsResult<Vec<PropRow>> {
            let domain_mask = by_str.equal(group);
            let n_domain    = domain_mask.sum().unwrap_or(0) as u32;
            let mut out: Vec<PropRow> = Vec::with_capacity(levels.len());
            for lvl in &levels {
                let indicator: Vec<Option<f64>> = value_str.iter()
                    .map(|v| match v {
                        Some(val) if val == lvl => Some(1.0),
                        Some(_) => Some(0.0),
                        None => None,
                    })
                    .collect();
                let indicator_ca = Float64Chunked::from_slice_options("indicator".into(), &indicator);
                let estimate = point_estimate_mean_domain(&indicator_ca, weights, &domain_mask)?;
                let scores   = scores_mean_domain(&indicator_ca, weights, &domain_mask)?;
                let scores_arr: Vec<f64> = scores.iter().map(|s| s.unwrap_or(0.0)).collect();
                let variance = taylor_variance_apply(&scores_arr, &design);
                let se       = variance.max(0.0).sqrt();
                let srs_var  = srs_variance_mean_domain(&indicator_ca, weights, &domain_mask)?;
                let deff     = if srs_var > 0.0 { variance / srs_var } else { f64::NAN };
                out.push((group.to_string(), lvl.clone(), estimate, se, variance, n_domain, deff));
            }
            Ok(out)
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    let mut by_vals: Vec<String> = Vec::new();
    let mut level_vals: Vec<String> = Vec::new();
    let mut estimates: Vec<f64> = Vec::new();
    let mut ses: Vec<f64> = Vec::new();
    let mut variances: Vec<f64> = Vec::new();
    let mut ns: Vec<u32> = Vec::new();
    let mut deffs: Vec<f64> = Vec::new();
    for group_rows in per_group {
        for (g, lvl, est, se, var, n, deff) in group_rows {
            by_vals.push(g);
            level_vals.push(lvl);
            estimates.push(est);
            ses.push(se);
            variances.push(var);
            ns.push(n);
            deffs.push(deff);
        }
    }
    let n_rows = by_vals.len();
    let dfs_vec = vec![df_val; n_rows];
    df![by_col => by_vals, "y" => vec![value_col; n_rows], "level" => level_vals,
        "est" => estimates, "se" => ses, "var" => variances, "df" => dfs_vec, "n" => ns, "deff" => deffs]
}

// ============================================================================
// Median
// ============================================================================

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, by_col=None, singleton_method=None, quantile_method=None))]
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
    quantile_method: Option<String>,
) -> PyResult<PyDataFrame> {
    let df = into_contiguous(data);
    let q_method = quantile_method
        .as_deref()
        .map(SvyQuantileMethod::from_str)
        .unwrap_or(SvyQuantileMethod::Higher);

    if by_col.is_none() {
        let result = compute_median_ungrouped(
            &df, &value_col, &weight_col,
            strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(),
            fpc_col.as_deref(), fpc_ssu_col.as_deref(), singleton_method.as_deref(),
            q_method,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        return Ok(PyDataFrame(result));
    }

    let result = compute_median_grouped(
        &df, &value_col, &weight_col,
        strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(),
        fpc_col.as_deref(), fpc_ssu_col.as_deref(),
        &by_col.unwrap(), singleton_method.as_deref(), q_method,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(PyDataFrame(result))
}

/// Batched ungrouped median over many variables (run in parallel; see
/// `compute_median_multi`). One row per variable, in input order.
#[pyfunction]
#[pyo3(signature = (data, value_cols, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, singleton_method=None, quantile_method=None))]
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
    quantile_method: Option<String>,
) -> PyResult<PyDataFrame> {
    let df = into_contiguous(data);
    let q_method = quantile_method
        .as_deref()
        .map(SvyQuantileMethod::from_str)
        .unwrap_or(SvyQuantileMethod::Higher);
    let result = _py
        .detach(|| {
            compute_median_multi(
                &df, &value_cols, &weight_col,
                strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(),
                fpc_col.as_deref(), fpc_ssu_col.as_deref(), singleton_method.as_deref(),
                q_method,
            )
        })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(PyDataFrame(result))
}

fn compute_median_ungrouped(
    df: &DataFrame,
    value_col: &str, weight_col: &str,
    strata_col: Option<&str>, psu_col: Option<&str>, ssu_col: Option<&str>,
    fpc_col: Option<&str>, fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>, q_method: SvyQuantileMethod,
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu    = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu    = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc    = fpc_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;
    let fpc_ssu = fpc_ssu_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;

    let estimate = weighted_median(y, weights, q_method)?;
    let (var_p, se_p) = median_variance_woodruff(
        y, weights, strata, psu, ssu, fpc, fpc_ssu, singleton_method, q_method,
    )?;
    let df_val = degrees_of_freedom(weights, strata, psu)?;
    let n = y.len() as u32;

    df!["y" => vec![value_col], "est" => vec![estimate], "se" => vec![se_p],
        "var" => vec![var_p], "df" => vec![df_val], "n" => vec![n]]
}

/// Batched ungrouped medians: variables fanned out over rayon. Median is
/// sort-bound and the Woodruff variance rebuilds its own design per variable, so
/// unlike mean/total this amortises nothing — the win is running independent
/// medians in parallel. Each row is identical to `compute_median_ungrouped`.
fn compute_median_multi(
    df: &DataFrame,
    value_cols: &[String], weight_col: &str,
    strata_col: Option<&str>, psu_col: Option<&str>, ssu_col: Option<&str>,
    fpc_col: Option<&str>, fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>, q_method: SvyQuantileMethod,
) -> PolarsResult<DataFrame> {
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu    = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu    = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc    = fpc_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;
    let fpc_ssu = fpc_ssu_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;

    // df is design-only, identical across variables — compute once.
    let df_val = degrees_of_freedom(weights, strata, psu)?;

    let y_cols: Vec<&Float64Chunked> = value_cols
        .iter()
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .collect::<PolarsResult<Vec<_>>>()?;

    let rows = (0..value_cols.len())
        .into_par_iter()
        .map(|i| -> PolarsResult<(String, f64, f64, f64, u32)> {
            let y = y_cols[i];
            let estimate = weighted_median(y, weights, q_method)?;
            let (var_p, se_p) = median_variance_woodruff(
                y, weights, strata, psu, ssu, fpc, fpc_ssu, singleton_method, q_method,
            )?;
            let n = y.len() as u32;
            Ok((value_cols[i].clone(), estimate, se_p, var_p, n))
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    let nv = rows.len();
    let mut ys: Vec<String> = Vec::with_capacity(nv);
    let mut estimates: Vec<f64> = Vec::with_capacity(nv);
    let mut ses: Vec<f64> = Vec::with_capacity(nv);
    let mut variances: Vec<f64> = Vec::with_capacity(nv);
    let mut ns: Vec<u32> = Vec::with_capacity(nv);
    for (y, est, se, var, n) in rows {
        ys.push(y);
        estimates.push(est);
        ses.push(se);
        variances.push(var);
        ns.push(n);
    }
    let dfs = vec![df_val; nv];
    df!["y" => ys, "est" => estimates, "se" => ses, "var" => variances,
        "df" => dfs, "n" => ns]
}

fn compute_median_grouped(
    df: &DataFrame,
    value_col: &str, weight_col: &str,
    strata_col: Option<&str>, psu_col: Option<&str>, ssu_col: Option<&str>,
    fpc_col: Option<&str>, fpc_ssu_col: Option<&str>,
    by_col: &str, singleton_method: Option<&str>, q_method: SvyQuantileMethod,
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu    = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu    = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc    = fpc_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;
    let fpc_ssu = fpc_ssu_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;
    let by_str = df.column(by_col)?.str()?;
    let unique_groups = by_str.unique()?;

    let mut by_vals: Vec<&str> = Vec::new();
    let mut estimates: Vec<f64> = Vec::new();
    let mut ses: Vec<f64> = Vec::new();
    let mut variances: Vec<f64> = Vec::new();
    let mut dfs: Vec<u32> = Vec::new();
    let mut ns: Vec<u32> = Vec::new();
    let df_val = degrees_of_freedom(weights, strata, psu)?;

    for group_val in unique_groups.iter() {
        if let Some(group) = group_val {
            let domain_mask = by_str.equal(group);
            let n_domain    = domain_mask.sum().unwrap_or(0) as u32;
            let estimate    = weighted_median_domain(y, weights, &domain_mask, q_method)?;
            let (var_p, se_p) = median_variance_woodruff_domain(
                y, weights, &domain_mask, strata, psu, ssu, fpc, fpc_ssu,
                singleton_method, q_method,
            )?;

            by_vals.push(group);
            estimates.push(estimate);
            ses.push(se_p);
            variances.push(var_p);
            dfs.push(df_val);
            ns.push(n_domain);
        }
    }
    let n_groups = by_vals.len();
    df![by_col => by_vals, "y" => vec![value_col; n_groups], "est" => estimates,
        "se" => ses, "var" => variances, "df" => dfs, "n" => ns]
}
