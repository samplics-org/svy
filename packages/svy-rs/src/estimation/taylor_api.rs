// src/estimation/taylor_api.rs
//
// PyO3-facing wrappers and `compute_*` helpers for Taylor linearization.
// The heavy math lives in taylor.rs; this file only handles DataFrame I/O,
// argument parsing, and looping over by-groups.

use polars::prelude::*;
use pyo3::prelude::*;
use numpy::PyReadonlyArray1;
use pyo3_polars::PyDataFrame;
use rayon::prelude::*;

use crate::estimation::taylor::{
    SvyQuantileMethod,
    TaylorDesign,
    build_taylor_design,
    degrees_of_freedom,
    degrees_of_freedom_from_design,
    degrees_of_freedom_in_domain,
    point_estimate_mean, point_estimate_mean_domain,
    point_estimate_ratio, point_estimate_ratio_domain,
    point_estimate_total, point_estimate_total_domain,
    quantiles_woodruff,
    weighted_quantile,
    scores_mean, scores_mean_arr, scores_mean_domain,
    scores_ratio, scores_ratio_domain,
    scores_total, scores_total_domain,
    srs_variance_mean, srs_variance_mean_domain,
    srs_variance_ratio, srs_variance_ratio_domain,
    srs_variance_total, srs_variance_total_domain,
    taylor_variance_apply,
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
        df.rechunk_mut_par();
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
        // Detach for the ungrouped path too: `compute_mean_ungrouped` overlaps
        // the design build with the score pass via `rayon::join`, and holding
        // the GIL would keep the rayon worker from ever picking up the second
        // half (see the policy note in estimation/mod.rs).
        let result = _py
            .detach(|| {
                compute_mean_ungrouped(
                    &df, &value_col, &weight_col,
                    strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(),
                    fpc_col.as_deref(), fpc_ssu_col.as_deref(), singleton_method.as_deref(),
                )
            })
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

/// Shared skeleton for the ungrouped single-estimate Taylor kernels.
///
/// Two overlaps, both free of any effect on the numbers:
///
/// 1. Indexing the design (densify strata, nest PSU codes, per-stratum maps)
///    reads none of the response columns, and the point estimate / scores / SRS
///    variance read none of the design columns. They run concurrently.
/// 2. The variance and the df both read the finished design but neither feeds
///    the other, so they run concurrently too — and taking df from the design's
///    codes drops the second densification pass over the same columns, which at
///    1M rows was on its own about a third of the kernel.
///
/// Every half is internally unchanged and still accumulates in row order, so the
/// output is bit-identical and independent of `RAYON_NUM_THREADS`, as
/// `estimation/mod.rs` requires.
///
/// `value_work` returns the score vector plus whatever else the caller needs out
/// of the response columns.
fn ungrouped_estimate<T: Send>(
    weights: &Float64Chunked,
    strata: Option<&Column>,
    psu: Option<&Column>,
    ssu: Option<&Column>,
    fpc: Option<&Float64Chunked>,
    fpc_ssu: Option<&Float64Chunked>,
    singleton_method: Option<&str>,
    value_work: impl FnOnce() -> PolarsResult<(Vec<f64>, T)> + Send,
) -> PolarsResult<(f64, u32, T)> {
    let (design_res, value_res) = rayon::join(
        || build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method),
        value_work,
    );
    let design = design_res?;
    let (scores, extra) = value_res?;

    let (variance, df_val) = rayon::join(
        || taylor_variance_apply(&scores, &design),
        || degrees_of_freedom_from_design(weights, &design, None),
    );
    Ok((variance, df_val, extra))
}

/// Flatten a score `Float64Chunked` to the array the variance consumes, mapping
/// nulls to 0.0 exactly as `taylor_variance` did.
fn scores_to_arr(scores: &Float64Chunked) -> Vec<f64> {
    scores.iter().map(|s| s.unwrap_or(0.0)).collect()
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

    let (variance, df_val, (estimate, srs_var)) = ungrouped_estimate(
        weights, strata, psu, ssu, fpc, fpc_ssu, singleton_method,
        || {
            let estimate = point_estimate_mean(y, weights)?;
            let scores   = scores_mean_arr(y, weights)?;
            let srs_var  = srs_variance_mean(y, weights)?;
            Ok((scores, (estimate, srs_var)))
        },
    )?;

    let se   = variance.max(0.0).sqrt();
    let n    = y.len() as u32;
    let deff = if srs_var > 0.0 { variance / srs_var } else { f64::NAN };

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

    // Serial head, previously ~22 ms of a ~35 ms kernel at 1M rows and the whole
    // reason the batched path stalled well under its available width: the design
    // was indexed once here and *again* inside `degrees_of_freedom`. Building it
    // once and deriving df from the same codes roughly halves the head, which is
    // the term Amdahl's law was multiplying.
    let design = build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
    let df_val = degrees_of_freedom_from_design(weights, &design, None);

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
            let scores_arr = scores_mean_arr(y, weights)?;
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

    // Index the design once — it is identical across by-groups; only the
    // domain-masked scores change per group.
    let design = build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;

    // Groups are independent; fan the per-group work out over the rayon pool
    // and collect in group order (deterministic, thread-count-independent).
    let groups: Vec<&str> = unique_groups.iter().flatten().collect();
    // A by-group is a domain, so its df must be counted on its own active
    // PSUs/strata. Broadcasting one frame-level df here would hand every group
    // the df of the surrounding analysis mask instead (issue #3).
    let group_dfs: Vec<u32> = groups
        .par_iter()
        .map(|&g| degrees_of_freedom_in_domain(weights, strata, psu, Some(&by_str.equal(g))))
        .collect::<PolarsResult<Vec<_>>>()?;
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
    let dfs = group_dfs;
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

    let (variance, df_val, (estimate, srs_var)) = ungrouped_estimate(
        weights, strata, psu, ssu, fpc, fpc_ssu, singleton_method,
        || {
            let estimate = point_estimate_total(y, weights)?;
            let scores   = scores_to_arr(&scores_total(y, weights)?);
            let srs_var  = srs_variance_total(y, weights)?;
            Ok((scores, (estimate, srs_var)))
        },
    )?;

    let se   = variance.max(0.0).sqrt();
    let n    = y.len() as u32;
    let deff = if srs_var > 0.0 { variance / srs_var } else { f64::NAN };

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

    // Design indexed once; df taken off its codes rather than densifying the
    // same strata/PSU columns a second time (see `compute_mean_multi`).
    let design = build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
    let df_val = degrees_of_freedom_from_design(weights, &design, None);

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

    let design = build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;

    let groups: Vec<&str> = unique_groups.iter().flatten().collect();
    // A by-group is a domain, so its df must be counted on its own active
    // PSUs/strata. Broadcasting one frame-level df here would hand every group
    // the df of the surrounding analysis mask instead (issue #3).
    let group_dfs: Vec<u32> = groups
        .par_iter()
        .map(|&g| degrees_of_freedom_in_domain(weights, strata, psu, Some(&by_str.equal(g))))
        .collect::<PolarsResult<Vec<_>>>()?;
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
    let dfs = group_dfs;
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

    let (variance, df_val, (estimate, srs_var)) = ungrouped_estimate(
        weights, strata, psu, ssu, fpc, fpc_ssu, singleton_method,
        || {
            let estimate = point_estimate_ratio(y, x, weights)?;
            let scores   = scores_to_arr(&scores_ratio(y, x, weights)?);
            let srs_var  = srs_variance_ratio(y, x, weights)?;
            Ok((scores, (estimate, srs_var)))
        },
    )?;

    let se   = variance.max(0.0).sqrt();
    let n    = y.len() as u32;
    let deff = if srs_var > 0.0 { variance / srs_var } else { f64::NAN };

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

    // Design indexed once; df taken off its codes rather than densifying the
    // same strata/PSU columns a second time (see `compute_mean_multi`).
    let design = build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
    let df_val = degrees_of_freedom_from_design(weights, &design, None);

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

    let design = build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;

    let groups: Vec<&str> = unique_groups.iter().flatten().collect();
    // A by-group is a domain, so its df must be counted on its own active
    // PSUs/strata. Broadcasting one frame-level df here would hand every group
    // the df of the surrounding analysis mask instead (issue #3).
    let group_dfs: Vec<u32> = groups
        .par_iter()
        .map(|&g| degrees_of_freedom_in_domain(weights, strata, psu, Some(&by_str.equal(g))))
        .collect::<PolarsResult<Vec<_>>>()?;
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
    let dfs = group_dfs;
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
    let n = weights.len() as u32;
    // Design is identical across levels; index it once — and take the df off its
    // codes rather than densifying the same columns a second time.
    let design = build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
    let df_val = degrees_of_freedom_from_design(weights, &design, None);

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

    let n = weights.len() as u32;
    // Design indexed once; df taken off its codes (see `compute_mean_multi`).
    let design = build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
    let df_val = degrees_of_freedom_from_design(weights, &design, None);

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

    // Design is identical across all (group, level) cells; index it once.
    let design = build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;

    // Fan out over groups; each group emits its level rows in `levels` order,
    // then flatten in group order for a deterministic layout.
    type PropRow = (String, String, f64, f64, f64, u32, f64);
    let groups: Vec<&str> = unique_groups.iter().flatten().collect();
    // A by-group is a domain, so its df must be counted on its own active
    // PSUs/strata. Broadcasting one frame-level df here would hand every group
    // the df of the surrounding analysis mask instead (issue #3).
    let group_dfs: Vec<u32> = groups
        .par_iter()
        .map(|&g| degrees_of_freedom_in_domain(weights, strata, psu, Some(&by_str.equal(g))))
        .collect::<PolarsResult<Vec<_>>>()?;
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
    let dfs_vec: Vec<u32> = group_dfs
        .iter()
        .flat_map(|d| std::iter::repeat(*d).take(levels.len()))
        .collect();
    debug_assert_eq!(dfs_vec.len(), n_rows);
    df![by_col => by_vals, "y" => vec![value_col; n_rows], "level" => level_vals,
        "est" => estimates, "se" => ses, "var" => variances, "df" => dfs_vec, "n" => ns, "deff" => deffs]
}

// ============================================================================
// Quantiles (median is the p = 0.5 case)
// ============================================================================

/// Woodruff quantiles for one variable. One row per probability (and per
/// domain when `by_col` is set), carrying the probability in a `prob` column.
#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, probs, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, by_col=None, singleton_method=None, quantile_method=None))]
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
    quantile_method: Option<String>,
) -> PyResult<PyDataFrame> {
    let df = into_contiguous(data);
    let q_method = quantile_method
        .as_deref()
        .map(SvyQuantileMethod::from_str)
        .unwrap_or(SvyQuantileMethod::Higher);

    let result = match by_col.as_deref() {
        None => compute_quantile_ungrouped(
            &df, &value_col, &weight_col,
            strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(),
            fpc_col.as_deref(), fpc_ssu_col.as_deref(), singleton_method.as_deref(),
            &probs, q_method,
        ),
        Some(by) => compute_quantile_grouped(
            &df, &value_col, &weight_col,
            strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(),
            fpc_col.as_deref(), fpc_ssu_col.as_deref(),
            by, singleton_method.as_deref(), &probs, q_method,
        ),
    }
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(PyDataFrame(result))
}

/// Batched ungrouped quantiles over many variables (run in parallel; see
/// `compute_quantile_multi`). Rows are ordered variable-major, then by
/// probability, matching the input order of both.
#[pyfunction]
#[pyo3(signature = (data, value_cols, weight_col, probs, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, singleton_method=None, quantile_method=None))]
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
    quantile_method: Option<String>,
) -> PyResult<PyDataFrame> {
    let df = into_contiguous(data);
    let q_method = quantile_method
        .as_deref()
        .map(SvyQuantileMethod::from_str)
        .unwrap_or(SvyQuantileMethod::Higher);
    let result = _py
        .detach(|| {
            compute_quantile_multi(
                &df, &value_cols, &weight_col,
                strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(),
                fpc_col.as_deref(), fpc_ssu_col.as_deref(), singleton_method.as_deref(),
                &probs, q_method,
            )
        })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(PyDataFrame(result))
}

/// Evaluate the weighted quantile rule at arbitrary probabilities, given a
/// pre-sorted variable and its normalised weighted CDF.
///
/// Exposed so the Python side can invert the CDF for Woodruff confidence
/// limits using the *same* interpolation rule as the point estimate, the way
/// R's `oldsvyquantile` passes one `method`/`f` pair to both its point
/// `approxfun` and its endpoint `approx`. Keeping the rule in one place stops
/// the two from drifting.
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
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "values_sorted has {} entries but cdf has {}",
            y.len(),
            c.len()
        )));
    }
    let q_method = quantile_method
        .as_deref()
        .map(SvyQuantileMethod::from_str)
        .unwrap_or(SvyQuantileMethod::Higher);
    Ok(probs
        .iter()
        .map(|&p| weighted_quantile(y, c, p, q_method))
        .collect())
}

/// Drop the `prob` column so the legacy median entry points keep their exact
/// pre-quantile schema.
fn without_prob(mut df: DataFrame) -> PolarsResult<DataFrame> {
    let _ = df.drop_in_place("prob")?;
    Ok(df)
}

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, by_col=None, singleton_method=None, quantile_method=None))]
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
    quantile_method: Option<String>,
) -> PyResult<PyDataFrame> {
    let out = taylor_quantile(
        _py, data, value_col, weight_col, vec![0.5],
        strata_col, psu_col, ssu_col, fpc_col, fpc_ssu_col, by_col,
        singleton_method, quantile_method,
    )?;
    let result = without_prob(out.0)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(PyDataFrame(result))
}

/// Batched ungrouped median over many variables. One row per variable, in
/// input order.
#[pyfunction]
#[pyo3(signature = (data, value_cols, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, singleton_method=None, quantile_method=None))]
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
    quantile_method: Option<String>,
) -> PyResult<PyDataFrame> {
    let out = taylor_quantile_multi(
        _py, data, value_cols, weight_col, vec![0.5],
        strata_col, psu_col, ssu_col, fpc_col, fpc_ssu_col,
        singleton_method, quantile_method,
    )?;
    let result = without_prob(out.0)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(PyDataFrame(result))
}

/// The design columns every quantile path needs, resolved once.
struct QuantileCols<'a> {
    weights: &'a Float64Chunked,
    strata: Option<&'a Column>,
    psu: Option<&'a Column>,
    design: TaylorDesign,
}

fn resolve_quantile_cols<'a>(
    df: &'a DataFrame,
    weight_col: &str,
    strata_col: Option<&str>, psu_col: Option<&str>, ssu_col: Option<&str>,
    fpc_col: Option<&str>, fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>,
) -> PolarsResult<QuantileCols<'a>> {
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu    = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu    = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc    = fpc_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;
    let fpc_ssu = fpc_ssu_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;

    // The design is independent of both the variable and the probability, so
    // indexing it once serves every (variable, prob) pair below.
    let design = build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
    Ok(QuantileCols { weights, strata, psu, design })
}

fn compute_quantile_ungrouped(
    df: &DataFrame,
    value_col: &str, weight_col: &str,
    strata_col: Option<&str>, psu_col: Option<&str>, ssu_col: Option<&str>,
    fpc_col: Option<&str>, fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>, probs: &[f64], q_method: SvyQuantileMethod,
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let cols = resolve_quantile_cols(
        df, weight_col, strata_col, psu_col, ssu_col, fpc_col, fpc_ssu_col, singleton_method,
    )?;

    let rows = quantiles_woodruff(y, cols.weights, None, &cols.design, probs, q_method)?;
    let df_val = degrees_of_freedom(cols.weights, cols.strata, cols.psu)?;
    let n = y.len() as u32;
    let k = rows.len();

    let (estimates, ses, variances) = unzip_woodruff(rows);
    df!["y" => vec![value_col; k], "prob" => probs.to_vec(), "est" => estimates,
        "se" => ses, "var" => variances, "df" => vec![df_val; k], "n" => vec![n; k]]
}

/// Batched ungrouped quantiles: variables fanned out over rayon. Quantiles are
/// sort-bound, so unlike mean/total this amortises no design work beyond the
/// shared `TaylorDesign` — the win is running independent variables in
/// parallel. Each row is identical to `compute_quantile_ungrouped`.
fn compute_quantile_multi(
    df: &DataFrame,
    value_cols: &[String], weight_col: &str,
    strata_col: Option<&str>, psu_col: Option<&str>, ssu_col: Option<&str>,
    fpc_col: Option<&str>, fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>, probs: &[f64], q_method: SvyQuantileMethod,
) -> PolarsResult<DataFrame> {
    let cols = resolve_quantile_cols(
        df, weight_col, strata_col, psu_col, ssu_col, fpc_col, fpc_ssu_col, singleton_method,
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
            quantiles_woodruff(y_cols[i], cols.weights, None, &cols.design, probs, q_method)
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    let k = probs.len();
    let n_rows = per_var.len() * k;
    let mut ys: Vec<String> = Vec::with_capacity(n_rows);
    let mut ns: Vec<u32> = Vec::with_capacity(n_rows);
    let mut flat: Vec<(f64, f64, f64)> = Vec::with_capacity(n_rows);
    for (i, rows) in per_var.into_iter().enumerate() {
        ys.extend(std::iter::repeat_n(value_cols[i].clone(), k));
        ns.extend(std::iter::repeat_n(y_cols[i].len() as u32, k));
        flat.extend(rows);
    }

    let (estimates, ses, variances) = unzip_woodruff(flat);
    let probs_rep: Vec<f64> = std::iter::repeat_n(probs, value_cols.len())
        .flatten()
        .copied()
        .collect();
    df!["y" => ys, "prob" => probs_rep, "est" => estimates, "se" => ses,
        "var" => variances, "df" => vec![df_val; n_rows], "n" => ns]
}

fn compute_quantile_grouped(
    df: &DataFrame,
    value_col: &str, weight_col: &str,
    strata_col: Option<&str>, psu_col: Option<&str>, ssu_col: Option<&str>,
    fpc_col: Option<&str>, fpc_ssu_col: Option<&str>,
    by_col: &str, singleton_method: Option<&str>, probs: &[f64],
    q_method: SvyQuantileMethod,
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let cols = resolve_quantile_cols(
        df, weight_col, strata_col, psu_col, ssu_col, fpc_col, fpc_ssu_col, singleton_method,
    )?;
    let by_str = df.column(by_col)?.str()?;
    let unique_groups = by_str.unique()?;

    let k = probs.len();
    let mut by_vals: Vec<&str> = Vec::new();
    let mut dfs: Vec<u32> = Vec::new();
    let mut ns: Vec<u32> = Vec::new();
    let mut flat: Vec<(f64, f64, f64)> = Vec::new();

    for group_val in unique_groups.iter() {
        if let Some(group) = group_val {
            let domain_mask = by_str.equal(group);
            let n_domain    = domain_mask.sum().unwrap_or(0) as u32;
            let rows = quantiles_woodruff(
                y, cols.weights, Some(&domain_mask), &cols.design, probs, q_method,
            )?;
            // Per-group df: see the note in compute_mean_grouped.
            let df_val = degrees_of_freedom_in_domain(
                cols.weights, cols.strata, cols.psu, Some(&domain_mask),
            )?;

            by_vals.extend(std::iter::repeat_n(group, k));
            dfs.extend(std::iter::repeat_n(df_val, k));
            ns.extend(std::iter::repeat_n(n_domain, k));
            flat.extend(rows);
        }
    }

    let n_rows = by_vals.len();
    let probs_rep: Vec<f64> = std::iter::repeat_n(probs, n_rows / k.max(1))
        .flatten()
        .copied()
        .collect();
    let (estimates, ses, variances) = unzip_woodruff(flat);
    df![by_col => by_vals, "y" => vec![value_col; n_rows], "prob" => probs_rep,
        "est" => estimates, "se" => ses, "var" => variances, "df" => dfs, "n" => ns]
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
