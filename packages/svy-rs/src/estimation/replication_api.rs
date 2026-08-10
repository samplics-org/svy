// src/estimation/replication_api.rs
//
// PyO3-facing wrappers and `compute_replicate_*` helpers for replication-based
// variance estimation.  The matrix math lives in replication.rs; this file
// only handles DataFrame I/O, argument parsing, and looping over by-groups.

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::estimation::taylor::SvyQuantileMethod;
use crate::estimation::association::{
    AssocKind, PairProducts, replicate_association,
};
use crate::estimation::replication::{
    RepMethod, VarianceCenter,
    extract_rep_weights_matrix,
    index_domains,
    matrix_mean_by_domain, matrix_mean_by_domain_cols, matrix_mean_estimates,
    matrix_mean_estimates_cols,
    matrix_quantile_by_domain, matrix_quantile_estimates,
    matrix_prop_by_domain, matrix_prop_estimates, matrix_prop_estimates_cols,
    matrix_prop_by_domain_str, matrix_prop_estimates_str, matrix_prop_estimates_str_cols,
    matrix_ratio_by_domain, matrix_ratio_by_domain_cols, matrix_ratio_estimates,
    matrix_ratio_estimates_cols,
    matrix_total_by_domain, matrix_total_by_domain_cols, matrix_total_estimates,
    matrix_total_estimates_cols,
    replicate_coefficients,
    variance_from_replicates,
};

// ============================================================================
// Shared helpers
// ============================================================================

/// Borrow every replicate-weight column as a contiguous null-free slice, or
/// `None` if any column is chunked/nullable (→ caller uses the flat-matrix
/// fallback). Lets the ungrouped estimators accumulate straight from the
/// columns with no n×R matrix materialised.
fn get_cont_rep_cols<'a>(
    df: &'a DataFrame,
    rep_weight_cols: &[String],
) -> PolarsResult<Option<Vec<&'a [f64]>>> {
    let mut cols = Vec::with_capacity(rep_weight_cols.len());
    for name in rep_weight_cols {
        match df.column(name)?.f64()?.cont_slice() {
            Ok(s) => cols.push(s),
            Err(_) => return Ok(None),
        }
    }
    Ok(Some(cols))
}

/// Materialise the `where=` domain mask as a dense `Vec<f64>` (1.0 inside the
/// domain, 0.0 outside), or `None` when no domain is set. Passed to the
/// ungrouped replicate kernels so they zero out-of-domain replicate weights on
/// the fly — replacing the R materialised zeroed columns the Python side would
/// otherwise build per `where=` call.
fn get_domain_mask(df: &DataFrame, mask_col: Option<&str>) -> PolarsResult<Option<Vec<f64>>> {
    match mask_col {
        None => Ok(None),
        Some(name) => {
            let ca = df.column(name)?.f64()?;
            Ok(Some(ca.iter().map(|o| o.unwrap_or(0.0)).collect()))
        }
    }
}

fn parse_rep_method(method: &str) -> PyResult<RepMethod> {
    RepMethod::from_str(method).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown method: {}. Use 'BRR', 'Bootstrap', 'Jackknife', or 'SDR'",
            method
        ))
    })
}

fn parse_variance_center(center: &str) -> PyResult<VarianceCenter> {
    VarianceCenter::from_str(center).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown center: {}. Use 'replicate_mean' or 'full_sample'",
            center
        ))
    })
}

// ============================================================================
// Mean
// ============================================================================

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, rep_weight_cols, method, fay_coef=0.0, rscales=None, center="rep_mean", degrees_of_freedom=None, by_col=None, domain_mask_col=None))]
pub fn replicate_mean(
    _py: Python,
    data: PyDataFrame,
    value_col: String,
    weight_col: String,
    rep_weight_cols: Vec<String>,
    method: String,
    fay_coef: f64,
    rscales: Option<Vec<f64>>,
    center: &str,
    degrees_of_freedom: Option<u32>,
    by_col: Option<String>,
    domain_mask_col: Option<String>,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let n_reps = rep_weight_cols.len();
    let rep_method       = parse_rep_method(&method)?;
    let variance_center  = parse_variance_center(center)?;
    if let Some(r) = &rscales {
        if r.len() != n_reps {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "rscales has {} entries but there are {} replicate weight columns",
                r.len(),
                n_reps
            )));
        }
    }
    let df_val = degrees_of_freedom.unwrap_or(n_reps.saturating_sub(1) as u32);

    let result = _py.detach(|| {
        if by_col.is_none() {
            compute_replicate_mean_ungrouped(&df, &value_col, &weight_col, &rep_weight_cols,
                rep_method, fay_coef, rscales.as_deref(), variance_center, df_val, domain_mask_col.as_deref())
        } else {
            compute_replicate_mean_grouped(&df, &value_col, &weight_col, &rep_weight_cols,
                rep_method, fay_coef, rscales.as_deref(), variance_center, df_val, by_col.as_ref().unwrap())
        }
    });
    result.map(PyDataFrame).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

fn compute_replicate_mean_ungrouped(
    df: &DataFrame,
    value_col: &str, weight_col: &str, rep_weight_cols: &[String],
    method: RepMethod, fay_coef: f64, rscales: Option<&[f64]>,
    center: VarianceCenter, df_val: u32,
    domain_mask_col: Option<&str>,
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let n = y.len();
    let n_reps = rep_weight_cols.len();
    let y_arr: Vec<f64> = y.iter().map(|v| v.unwrap_or(0.0)).collect();
    let w_arr: Vec<f64> = weights.iter().map(|v| v.unwrap_or(0.0)).collect();
    let domain_mask = get_domain_mask(df, domain_mask_col)?;
    let mask = domain_mask.as_deref();

    // No-materialisation path: accumulate from the contiguous replicate columns
    // (parallel over replicates); flat-matrix fallback for chunked/nullable.
    let (theta_full, theta_reps) = match get_cont_rep_cols(df, rep_weight_cols)? {
        Some(cols) => matrix_mean_estimates_cols(&y_arr, &w_arr, &cols, n, mask),
        None => {
            let (rep_w_matrix, _, _) = extract_rep_weights_matrix(df, rep_weight_cols)?;
            matrix_mean_estimates(&y_arr, &w_arr, &rep_w_matrix, n, n_reps, mask)
        }
    };
    let rep_coefs = rscales.map(<[f64]>::to_vec)
        .unwrap_or_else(|| replicate_coefficients(method, n_reps, fay_coef));
    let variance  = variance_from_replicates(method, theta_full, &theta_reps, &rep_coefs, center);
    let se = variance.sqrt();

    df!["y" => vec![value_col], "est" => vec![theta_full], "se" => vec![se],
        "var" => vec![variance], "df" => vec![df_val], "n" => vec![n as u32]]
}

fn compute_replicate_mean_grouped(
    df: &DataFrame,
    value_col: &str, weight_col: &str, rep_weight_cols: &[String],
    method: RepMethod, fay_coef: f64, rscales: Option<&[f64]>,
    center: VarianceCenter, df_val: u32,
    by_col: &str,
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let by_str = df.column(by_col)?.str()?;
    let n = y.len();
    let n_reps = rep_weight_cols.len();
    let y_arr: Vec<f64> = y.iter().map(|v| v.unwrap_or(0.0)).collect();
    let w_arr: Vec<f64> = weights.iter().map(|v| v.unwrap_or(0.0)).collect();

    let (domain_ids, domain_names, n_domains) = index_domains(by_str);
    let (theta_full_vec, theta_reps_vec, counts) = match get_cont_rep_cols(df, rep_weight_cols)? {
        Some(cols) => {
            matrix_mean_by_domain_cols(&y_arr, &w_arr, &cols, &domain_ids, n_domains, n)
        }
        None => {
            let (rep_w_matrix, _, _) = extract_rep_weights_matrix(df, rep_weight_cols)?;
            matrix_mean_by_domain(&y_arr, &w_arr, &rep_w_matrix, &domain_ids, n_domains, n, n_reps)
        }
    };
    let rep_coefs = rscales.map(<[f64]>::to_vec)
        .unwrap_or_else(|| replicate_coefficients(method, n_reps, fay_coef));

    let mut by_vals: Vec<String> = Vec::with_capacity(n_domains);
    let mut estimates: Vec<f64> = Vec::with_capacity(n_domains);
    let mut ses: Vec<f64> = Vec::with_capacity(n_domains);
    let mut variances: Vec<f64> = Vec::with_capacity(n_domains);
    let mut dfs: Vec<u32> = Vec::with_capacity(n_domains);
    let mut ns: Vec<u32> = Vec::with_capacity(n_domains);

    for (k, domain_name) in domain_names.iter().enumerate() {
        let variance = variance_from_replicates(method, theta_full_vec[k], &theta_reps_vec[k], &rep_coefs, center);
        by_vals.push(domain_name.clone());
        estimates.push(theta_full_vec[k]);
        ses.push(variance.sqrt());
        variances.push(variance);
        dfs.push(df_val);
        ns.push(counts[k]);
    }
    df![by_col => by_vals, "y" => vec![value_col; n_domains], "est" => estimates,
        "se" => ses, "var" => variances, "df" => dfs, "n" => ns]
}

// ============================================================================
// Total
// ============================================================================

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, rep_weight_cols, method, fay_coef=0.0, rscales=None, center="rep_mean", degrees_of_freedom=None, by_col=None, domain_mask_col=None))]
pub fn replicate_total(
    _py: Python,
    data: PyDataFrame,
    value_col: String,
    weight_col: String,
    rep_weight_cols: Vec<String>,
    method: String,
    fay_coef: f64,
    rscales: Option<Vec<f64>>,
    center: &str,
    degrees_of_freedom: Option<u32>,
    by_col: Option<String>,
    domain_mask_col: Option<String>,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let n_reps = rep_weight_cols.len();
    let rep_method      = parse_rep_method(&method)?;
    let variance_center = parse_variance_center(center)?;
    if let Some(r) = &rscales {
        if r.len() != n_reps {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "rscales has {} entries but there are {} replicate weight columns",
                r.len(),
                n_reps
            )));
        }
    }
    let df_val = degrees_of_freedom.unwrap_or(n_reps.saturating_sub(1) as u32);

    let result = _py.detach(|| {
        if by_col.is_none() {
            compute_replicate_total_ungrouped(&df, &value_col, &weight_col, &rep_weight_cols,
                rep_method, fay_coef, rscales.as_deref(), variance_center, df_val, domain_mask_col.as_deref())
        } else {
            compute_replicate_total_grouped(&df, &value_col, &weight_col, &rep_weight_cols,
                rep_method, fay_coef, rscales.as_deref(), variance_center, df_val, by_col.as_ref().unwrap())
        }
    });
    result.map(PyDataFrame).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

fn compute_replicate_total_ungrouped(
    df: &DataFrame,
    value_col: &str, weight_col: &str, rep_weight_cols: &[String],
    method: RepMethod, fay_coef: f64, rscales: Option<&[f64]>,
    center: VarianceCenter, df_val: u32,
    domain_mask_col: Option<&str>,
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let n = y.len();
    let n_reps = rep_weight_cols.len();
    let y_arr: Vec<f64> = y.iter().map(|v| v.unwrap_or(0.0)).collect();
    let w_arr: Vec<f64> = weights.iter().map(|v| v.unwrap_or(0.0)).collect();
    let domain_mask = get_domain_mask(df, domain_mask_col)?;
    let mask = domain_mask.as_deref();

    let (theta_full, theta_reps) = match get_cont_rep_cols(df, rep_weight_cols)? {
        Some(cols) => matrix_total_estimates_cols(&y_arr, &w_arr, &cols, n, mask),
        None => {
            let (rep_w_matrix, _, _) = extract_rep_weights_matrix(df, rep_weight_cols)?;
            matrix_total_estimates(&y_arr, &w_arr, &rep_w_matrix, n, n_reps, mask)
        }
    };
    let rep_coefs = rscales.map(<[f64]>::to_vec)
        .unwrap_or_else(|| replicate_coefficients(method, n_reps, fay_coef));
    let variance  = variance_from_replicates(method, theta_full, &theta_reps, &rep_coefs, center);
    let se = variance.sqrt();

    df!["y" => vec![value_col], "est" => vec![theta_full], "se" => vec![se],
        "var" => vec![variance], "df" => vec![df_val], "n" => vec![n as u32]]
}

fn compute_replicate_total_grouped(
    df: &DataFrame,
    value_col: &str, weight_col: &str, rep_weight_cols: &[String],
    method: RepMethod, fay_coef: f64, rscales: Option<&[f64]>,
    center: VarianceCenter, df_val: u32,
    by_col: &str,
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let by_str = df.column(by_col)?.str()?;
    let n = y.len();
    let n_reps = rep_weight_cols.len();
    let y_arr: Vec<f64> = y.iter().map(|v| v.unwrap_or(0.0)).collect();
    let w_arr: Vec<f64> = weights.iter().map(|v| v.unwrap_or(0.0)).collect();

    let (domain_ids, domain_names, n_domains) = index_domains(by_str);
    let (theta_full_vec, theta_reps_vec, counts) = match get_cont_rep_cols(df, rep_weight_cols)? {
        Some(cols) => {
            matrix_total_by_domain_cols(&y_arr, &w_arr, &cols, &domain_ids, n_domains, n)
        }
        None => {
            let (rep_w_matrix, _, _) = extract_rep_weights_matrix(df, rep_weight_cols)?;
            matrix_total_by_domain(&y_arr, &w_arr, &rep_w_matrix, &domain_ids, n_domains, n, n_reps)
        }
    };
    let rep_coefs = rscales.map(<[f64]>::to_vec)
        .unwrap_or_else(|| replicate_coefficients(method, n_reps, fay_coef));

    let mut by_vals: Vec<String> = Vec::with_capacity(n_domains);
    let mut estimates: Vec<f64> = Vec::with_capacity(n_domains);
    let mut ses: Vec<f64> = Vec::with_capacity(n_domains);
    let mut variances: Vec<f64> = Vec::with_capacity(n_domains);
    let mut dfs: Vec<u32> = Vec::with_capacity(n_domains);
    let mut ns: Vec<u32> = Vec::with_capacity(n_domains);

    for (k, domain_name) in domain_names.iter().enumerate() {
        let variance = variance_from_replicates(method, theta_full_vec[k], &theta_reps_vec[k], &rep_coefs, center);
        by_vals.push(domain_name.clone());
        estimates.push(theta_full_vec[k]);
        ses.push(variance.sqrt());
        variances.push(variance);
        dfs.push(df_val);
        ns.push(counts[k]);
    }
    df![by_col => by_vals, "y" => vec![value_col; n_domains], "est" => estimates,
        "se" => ses, "var" => variances, "df" => dfs, "n" => ns]
}

// ============================================================================
// Association (covariance / correlation)
// ============================================================================

/// Replicate-weight covariance or correlation over one or more column pairs.
///
/// Every replicate re-centers on its own weighted means -- that is what R's
/// `svrepvar` does inside its `v(w)` closure -- while `n` for the covariance's
/// Kish factor stays fixed at the full-sample value, as R fixes it outside that
/// closure. Both fall out of `PairProducts`, which forms the weight-independent
/// products once so each replicate costs six dot products rather than a full
/// recomputation.
#[pyfunction]
#[pyo3(signature = (data, y_cols, x_cols, kind, weight_col, rep_weight_cols, method, fay_coef=0.0, rscales=None, center="rep_mean", degrees_of_freedom=None, by_col=None, domain_mask_col=None))]
#[allow(clippy::too_many_arguments)]
pub fn replicate_assoc(
    _py: Python,
    data: PyDataFrame,
    y_cols: Vec<String>,
    x_cols: Vec<String>,
    kind: String,
    weight_col: String,
    rep_weight_cols: Vec<String>,
    method: String,
    fay_coef: f64,
    rscales: Option<Vec<f64>>,
    center: &str,
    degrees_of_freedom: Option<u32>,
    by_col: Option<String>,
    domain_mask_col: Option<String>,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let n_reps = rep_weight_cols.len();
    let rep_method = parse_rep_method(&method)?;
    let variance_center = parse_variance_center(center)?;
    if let Some(r) = &rscales
        && r.len() != n_reps
    {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "rscales has {} entries but there are {} replicate weight columns",
            r.len(),
            n_reps
        )));
    }
    if y_cols.len() != x_cols.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "pair columns must be equal length, got {} and {}",
            y_cols.len(),
            x_cols.len()
        )));
    }
    let df_val = degrees_of_freedom.unwrap_or(n_reps.saturating_sub(1) as u32);

    let result = _py.detach(|| {
        let assoc_kind = AssocKind::from_name(&kind)?;
        compute_replicate_assoc(
            &df,
            &y_cols,
            &x_cols,
            assoc_kind,
            &weight_col,
            &rep_weight_cols,
            rep_method,
            fay_coef,
            rscales.as_deref(),
            variance_center,
            df_val,
            by_col.as_deref(),
            domain_mask_col.as_deref(),
        )
    });
    result
        .map(PyDataFrame)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

#[allow(clippy::too_many_arguments)]
fn compute_replicate_assoc(
    df: &DataFrame,
    y_cols: &[String],
    x_cols: &[String],
    kind: AssocKind,
    weight_col: &str,
    rep_weight_cols: &[String],
    method: RepMethod,
    fay_coef: f64,
    rscales: Option<&[f64]>,
    center: VarianceCenter,
    df_val: u32,
    by_col: Option<&str>,
    domain_mask_col: Option<&str>,
) -> PolarsResult<DataFrame> {
    let weights = df.column(weight_col)?.f64()?;
    let n = weights.len();
    let n_reps = rep_weight_cols.len();

    let ys: Vec<&Float64Chunked> = y_cols
        .iter()
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .collect::<PolarsResult<Vec<_>>>()?;
    let xs: Vec<&Float64Chunked> = x_cols
        .iter()
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .collect::<PolarsResult<Vec<_>>>()?;

    // Replicate weights as slices; materialize only when they are not already
    // contiguous, so the common case borrows straight from the frame.
    let cont = get_cont_rep_cols(df, rep_weight_cols)?;
    let owned: Option<Vec<Vec<f64>>> = if cont.is_none() {
        Some(
            rep_weight_cols
                .iter()
                .map(|c| {
                    df.column(c)
                        .and_then(|s| s.f64())
                        .map(|ca| ca.iter().map(|v| v.unwrap_or(0.0)).collect::<Vec<f64>>())
                })
                .collect::<PolarsResult<Vec<_>>>()?,
        )
    } else {
        None
    };
    let rep_cols: Vec<&[f64]> = match (&cont, &owned) {
        (Some(c), _) => c.clone(),
        (None, Some(o)) => o.iter().map(|v| v.as_slice()).collect(),
        (None, None) => Vec::new(),
    };

    // Ungrouped runs as a single unnamed domain, so one path covers both shapes.
    let by_str = by_col.map(|c| df.column(c).and_then(|s| s.str())).transpose()?;
    let unique_groups = by_str.map(|s| s.unique()).transpose()?;
    let group_names: Vec<Option<&str>> = match unique_groups.as_ref() {
        Some(u) => u.iter().flatten().map(Some).collect(),
        None => vec![None],
    };
    let explicit_mask = get_domain_mask(df, domain_mask_col)?;
    let group_masks: Vec<Option<Vec<f64>>> = group_names
        .iter()
        .map(|g| match (g, by_str) {
            (Some(gv), Some(bs)) => Some(
                bs.equal(*gv)
                    .iter()
                    .map(|v| if v == Some(true) { 1.0 } else { 0.0 })
                    .collect(),
            ),
            _ => explicit_mask.clone(),
        })
        .collect();

    let rep_coefs = rscales
        .map(<[f64]>::to_vec)
        .unwrap_or_else(|| replicate_coefficients(method, n_reps, fay_coef));

    let n_pairs = ys.len();
    let combos: Vec<(usize, usize)> = (0..group_names.len())
        .flat_map(|gi| (0..n_pairs).map(move |pi| (gi, pi)))
        .collect();

    // Sequential over (group, pair): `replicate_association` already fans out
    // across replicates, which is the wide axis. Nesting a second rayon level
    // here would oversubscribe without adding usable parallelism.
    let w_full: Vec<f64> = weights.iter().map(|v| v.unwrap_or(0.0)).collect();
    let rows = combos
        .iter()
        .map(|&(gi, pi)| -> PolarsResult<(usize, usize, f64, f64, f64, u32)> {
            let mask = group_masks[gi].as_deref();
            let products = PairProducts::new(ys[pi], xs[pi], weights, mask)?;
            let theta_full = products.estimate(&w_full, kind);
            let theta_reps = replicate_association(&products, &rep_cols, kind);
            let variance =
                variance_from_replicates(method, theta_full, &theta_reps, &rep_coefs, center);
            let n_rows = match mask {
                Some(m) => m.iter().filter(|v| **v != 0.0).count() as u32,
                None => n as u32,
            };
            Ok((gi, pi, theta_full, variance.sqrt(), variance, n_rows))
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    let nv = rows.len();
    let mut by_vals: Vec<&str> = Vec::with_capacity(nv);
    let mut y_names: Vec<&str> = Vec::with_capacity(nv);
    let mut x_names: Vec<&str> = Vec::with_capacity(nv);
    let mut estimates: Vec<f64> = Vec::with_capacity(nv);
    let mut ses: Vec<f64> = Vec::with_capacity(nv);
    let mut variances: Vec<f64> = Vec::with_capacity(nv);
    let mut ns: Vec<u32> = Vec::with_capacity(nv);
    for (gi, pi, est, se, var, n_rows) in rows {
        if let Some(g) = group_names[gi] {
            by_vals.push(g);
        }
        y_names.push(y_cols[pi].as_str());
        x_names.push(x_cols[pi].as_str());
        estimates.push(est);
        ses.push(se);
        variances.push(var);
        ns.push(n_rows);
    }

    let mut out = df![
        "y" => y_names, "x" => x_names, "kind" => vec![kind.as_str(); nv],
        "est" => estimates, "se" => ses, "var" => variances,
        "df" => vec![df_val; nv], "n" => ns
    ]?;
    if let Some(name) = by_col {
        out.insert_column(0, Column::new(name.into(), by_vals))?;
    }
    Ok(out)
}

// ============================================================================
// Ratio
// ============================================================================

#[pyfunction]
#[pyo3(signature = (data, numerator_col, denominator_col, weight_col, rep_weight_cols, method, fay_coef=0.0, rscales=None, center="rep_mean", degrees_of_freedom=None, by_col=None, domain_mask_col=None))]
pub fn replicate_ratio(
    _py: Python,
    data: PyDataFrame,
    numerator_col: String,
    denominator_col: String,
    weight_col: String,
    rep_weight_cols: Vec<String>,
    method: String,
    fay_coef: f64,
    rscales: Option<Vec<f64>>,
    center: &str,
    degrees_of_freedom: Option<u32>,
    by_col: Option<String>,
    domain_mask_col: Option<String>,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let n_reps = rep_weight_cols.len();
    let rep_method      = parse_rep_method(&method)?;
    let variance_center = parse_variance_center(center)?;
    if let Some(r) = &rscales {
        if r.len() != n_reps {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "rscales has {} entries but there are {} replicate weight columns",
                r.len(),
                n_reps
            )));
        }
    }
    let df_val = degrees_of_freedom.unwrap_or(n_reps.saturating_sub(1) as u32);

    let result = _py.detach(|| {
        if by_col.is_none() {
            compute_replicate_ratio_ungrouped(&df, &numerator_col, &denominator_col, &weight_col,
                &rep_weight_cols, rep_method, fay_coef, rscales.as_deref(), variance_center, df_val,
                domain_mask_col.as_deref())
        } else {
            compute_replicate_ratio_grouped(&df, &numerator_col, &denominator_col, &weight_col,
                &rep_weight_cols, rep_method, fay_coef, rscales.as_deref(), variance_center, df_val,
                by_col.as_ref().unwrap())
        }
    });
    result.map(PyDataFrame).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

fn compute_replicate_ratio_ungrouped(
    df: &DataFrame,
    numerator_col: &str, denominator_col: &str, weight_col: &str,
    rep_weight_cols: &[String],
    method: RepMethod, fay_coef: f64, rscales: Option<&[f64]>,
    center: VarianceCenter, df_val: u32,
    domain_mask_col: Option<&str>,
) -> PolarsResult<DataFrame> {
    let y = df.column(numerator_col)?.f64()?;
    let x = df.column(denominator_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let n = y.len();
    let n_reps = rep_weight_cols.len();
    let y_arr: Vec<f64> = y.iter().map(|v| v.unwrap_or(0.0)).collect();
    let x_arr: Vec<f64> = x.iter().map(|v| v.unwrap_or(0.0)).collect();
    let w_arr: Vec<f64> = weights.iter().map(|v| v.unwrap_or(0.0)).collect();
    let domain_mask = get_domain_mask(df, domain_mask_col)?;
    let mask = domain_mask.as_deref();

    let (theta_full, theta_reps) = match get_cont_rep_cols(df, rep_weight_cols)? {
        Some(cols) => matrix_ratio_estimates_cols(&y_arr, &x_arr, &w_arr, &cols, n, mask),
        None => {
            let (rep_w_matrix, _, _) = extract_rep_weights_matrix(df, rep_weight_cols)?;
            matrix_ratio_estimates(&y_arr, &x_arr, &w_arr, &rep_w_matrix, n, n_reps, mask)
        }
    };
    let rep_coefs = rscales.map(<[f64]>::to_vec)
        .unwrap_or_else(|| replicate_coefficients(method, n_reps, fay_coef));
    let variance  = variance_from_replicates(method, theta_full, &theta_reps, &rep_coefs, center);
    let se = variance.sqrt();

    df!["y" => vec![numerator_col], "x" => vec![denominator_col], "est" => vec![theta_full],
        "se" => vec![se], "var" => vec![variance], "df" => vec![df_val], "n" => vec![n as u32]]
}

fn compute_replicate_ratio_grouped(
    df: &DataFrame,
    numerator_col: &str, denominator_col: &str, weight_col: &str,
    rep_weight_cols: &[String],
    method: RepMethod, fay_coef: f64, rscales: Option<&[f64]>,
    center: VarianceCenter, df_val: u32,
    by_col: &str,
) -> PolarsResult<DataFrame> {
    let y = df.column(numerator_col)?.f64()?;
    let x = df.column(denominator_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let by_str = df.column(by_col)?.str()?;
    let n = y.len();
    let n_reps = rep_weight_cols.len();
    let y_arr: Vec<f64> = y.iter().map(|v| v.unwrap_or(0.0)).collect();
    let x_arr: Vec<f64> = x.iter().map(|v| v.unwrap_or(0.0)).collect();
    let w_arr: Vec<f64> = weights.iter().map(|v| v.unwrap_or(0.0)).collect();

    let (domain_ids, domain_names, n_domains) = index_domains(by_str);
    let (theta_full_vec, theta_reps_vec, counts) = match get_cont_rep_cols(df, rep_weight_cols)? {
        Some(cols) => {
            matrix_ratio_by_domain_cols(&y_arr, &x_arr, &w_arr, &cols, &domain_ids, n_domains, n)
        }
        None => {
            let (rep_w_matrix, _, _) = extract_rep_weights_matrix(df, rep_weight_cols)?;
            matrix_ratio_by_domain(&y_arr, &x_arr, &w_arr, &rep_w_matrix, &domain_ids, n_domains, n, n_reps)
        }
    };
    let rep_coefs = rscales.map(<[f64]>::to_vec)
        .unwrap_or_else(|| replicate_coefficients(method, n_reps, fay_coef));

    let mut by_vals: Vec<String> = Vec::with_capacity(n_domains);
    let mut estimates: Vec<f64> = Vec::with_capacity(n_domains);
    let mut ses: Vec<f64> = Vec::with_capacity(n_domains);
    let mut variances: Vec<f64> = Vec::with_capacity(n_domains);
    let mut dfs: Vec<u32> = Vec::with_capacity(n_domains);
    let mut ns: Vec<u32> = Vec::with_capacity(n_domains);

    for (k, domain_name) in domain_names.iter().enumerate() {
        let variance = variance_from_replicates(method, theta_full_vec[k], &theta_reps_vec[k], &rep_coefs, center);
        by_vals.push(domain_name.clone());
        estimates.push(theta_full_vec[k]);
        ses.push(variance.sqrt());
        variances.push(variance);
        dfs.push(df_val);
        ns.push(counts[k]);
    }
    df![by_col => by_vals, "y" => vec![numerator_col; n_domains], "x" => vec![denominator_col; n_domains],
        "est" => estimates, "se" => ses, "var" => variances, "df" => dfs, "n" => ns]
}

// ============================================================================
// Proportion
// ============================================================================

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, rep_weight_cols, method, fay_coef=0.0, rscales=None, center="rep_mean", degrees_of_freedom=None, by_col=None, domain_mask_col=None))]
pub fn replicate_prop(
    _py: Python,
    data: PyDataFrame,
    value_col: String,
    weight_col: String,
    rep_weight_cols: Vec<String>,
    method: String,
    fay_coef: f64,
    rscales: Option<Vec<f64>>,
    center: &str,
    degrees_of_freedom: Option<u32>,
    by_col: Option<String>,
    domain_mask_col: Option<String>,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let n_reps = rep_weight_cols.len();
    let rep_method      = parse_rep_method(&method)?;
    let variance_center = parse_variance_center(center)?;
    if let Some(r) = &rscales {
        if r.len() != n_reps {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "rscales has {} entries but there are {} replicate weight columns",
                r.len(),
                n_reps
            )));
        }
    }
    let df_val = degrees_of_freedom.unwrap_or(n_reps.saturating_sub(1) as u32);

    let result = _py.detach(|| {
        if by_col.is_none() {
            compute_replicate_prop_ungrouped(&df, &value_col, &weight_col, &rep_weight_cols,
                rep_method, fay_coef, rscales.as_deref(), variance_center, df_val, domain_mask_col.as_deref())
        } else {
            compute_replicate_prop_grouped(&df, &value_col, &weight_col, &rep_weight_cols,
                rep_method, fay_coef, rscales.as_deref(), variance_center, df_val, by_col.as_ref().unwrap())
        }
    });
    result.map(PyDataFrame).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

fn compute_replicate_prop_ungrouped(
    df: &DataFrame,
    value_col: &str, weight_col: &str, rep_weight_cols: &[String],
    method: RepMethod, fay_coef: f64, rscales: Option<&[f64]>,
    center: VarianceCenter, df_val: u32,
    domain_mask_col: Option<&str>,
) -> PolarsResult<DataFrame> {
    let y_series = df.column(value_col)?;
    let weights  = df.column(weight_col)?.f64()?;
    let n = y_series.len();
    let n_reps = rep_weight_cols.len();
    let w_arr: Vec<f64> = weights.iter().map(|v| v.unwrap_or(0.0)).collect();
    let cont_cols = get_cont_rep_cols(df, rep_weight_cols)?;
    let domain_mask = get_domain_mask(df, domain_mask_col)?;
    let mask = domain_mask.as_deref();
    let rep_coefs = rscales.map(<[f64]>::to_vec)
        .unwrap_or_else(|| replicate_coefficients(method, n_reps, fay_coef));

    // String/Categorical: use string-keyed level functions so level labels are
    // preserved as-is without any numeric cast.
    let is_string = matches!(y_series.dtype(), DataType::String | DataType::Categorical(_, _));

    let (n_levels, level_strs, estimates, variances) = if is_string {
        let y_cast = y_series.cast(&DataType::String)?;
        let y_arr: Vec<String> = y_cast.str()?.iter()
            .map(|v| v.unwrap_or("").to_string())
            .collect();
        let (levels, theta_full, theta_reps) = match &cont_cols {
            Some(cols) => matrix_prop_estimates_str_cols(&y_arr, &w_arr, cols, n, mask),
            None => {
                let (m, _, _) = extract_rep_weights_matrix(df, rep_weight_cols)?;
                matrix_prop_estimates_str(&y_arr, &w_arr, &m, n, n_reps, mask)
            }
        };
        let n_l = levels.len();
        let vars: Vec<f64> = (0..n_l)
            .map(|l| variance_from_replicates(method, theta_full[l], &theta_reps[l], &rep_coefs, center))
            .collect();
        (n_l, levels, theta_full, vars)
    } else {
        // Integer or boolean: convert to i64 category codes.
        let y_arr: Vec<i64> = if y_series.dtype().is_integer() {
            y_series.i64()?.iter().map(|v| v.unwrap_or(0)).collect()
        } else if y_series.dtype() == &DataType::Boolean {
            y_series.bool()?.iter()
                .map(|v| if v.unwrap_or(false) { 1 } else { 0 })
                .collect()
        } else {
            return Err(PolarsError::InvalidOperation(
                format!(
                    "prop() does not support dtype {:?} for column '{}'. \
                     Use a String, Categorical, Boolean, or integer column.",
                    y_series.dtype(), value_col
                ).into()
            ));
        };
        let (levels, theta_full, theta_reps) = match &cont_cols {
            Some(cols) => matrix_prop_estimates_cols(&y_arr, &w_arr, cols, n, mask),
            None => {
                let (m, _, _) = extract_rep_weights_matrix(df, rep_weight_cols)?;
                matrix_prop_estimates(&y_arr, &w_arr, &m, n, n_reps, mask)
            }
        };
        let n_l = levels.len();
        let vars: Vec<f64> = (0..n_l)
            .map(|l| variance_from_replicates(method, theta_full[l], &theta_reps[l], &rep_coefs, center))
            .collect();
        let str_levels: Vec<String> = levels.iter().map(|l| l.to_string()).collect();
        (n_l, str_levels, theta_full, vars)
    };

    let ses: Vec<f64> = variances.iter().map(|v| v.sqrt()).collect();
    let ns:  Vec<u32> = vec![n as u32; n_levels];
    let dfs: Vec<u32> = vec![df_val; n_levels];
    df!["y" => vec![value_col; n_levels], "level" => level_strs, "est" => estimates,
        "se" => ses, "var" => variances, "df" => dfs, "n" => ns]
}

fn compute_replicate_prop_grouped(
    df: &DataFrame,
    value_col: &str, weight_col: &str, rep_weight_cols: &[String],
    method: RepMethod, fay_coef: f64, rscales: Option<&[f64]>,
    center: VarianceCenter, df_val: u32,
    by_col: &str,
) -> PolarsResult<DataFrame> {
    let y_series = df.column(value_col)?;
    let weights  = df.column(weight_col)?.f64()?;
    let by_str   = df.column(by_col)?.str()?;
    let n = y_series.len();
    let n_reps = rep_weight_cols.len();
    let w_arr: Vec<f64> = weights.iter().map(|v| v.unwrap_or(0.0)).collect();

    let (domain_ids, domain_names, n_domains) = index_domains(by_str);
    let (rep_w_matrix, _, _) = extract_rep_weights_matrix(df, rep_weight_cols)?;
    let rep_coefs = rscales.map(<[f64]>::to_vec)
        .unwrap_or_else(|| replicate_coefficients(method, n_reps, fay_coef));

    let is_string = matches!(y_series.dtype(), DataType::String | DataType::Categorical(_, _));

    let (levels_str, theta_full_mat, theta_reps_mat, counts) = if is_string {
        let y_cast = y_series.cast(&DataType::String)?;
        let y_arr: Vec<String> = y_cast.str()?.iter()
            .map(|v| v.unwrap_or("").to_string())
            .collect();
        let (levels, tf, tr, counts) =
            matrix_prop_by_domain_str(&y_arr, &w_arr, &rep_w_matrix, &domain_ids, n_domains, n, n_reps);
        (levels, tf, tr, counts)
    } else {
        let y_arr: Vec<i64> = if y_series.dtype().is_integer() {
            y_series.i64()?.iter().map(|v| v.unwrap_or(0)).collect()
        } else if y_series.dtype() == &DataType::Boolean {
            y_series.bool()?.iter()
                .map(|v| if v.unwrap_or(false) { 1 } else { 0 })
                .collect()
        } else {
            return Err(PolarsError::InvalidOperation(
                format!(
                    "prop() does not support dtype {:?} for column '{}'. \
                     Use a String, Categorical, Boolean, or integer column.",
                    y_series.dtype(), value_col
                ).into()
            ));
        };
        let (levels, tf, tr, counts) =
            matrix_prop_by_domain(&y_arr, &w_arr, &rep_w_matrix, &domain_ids, n_domains, n, n_reps);
        let str_levels: Vec<String> = levels.iter().map(|l| l.to_string()).collect();
        (str_levels, tf, tr, counts)
    };

    let n_levels = levels_str.len();
    let mut by_vals:    Vec<String> = Vec::new();
    let mut level_strs: Vec<String> = Vec::new();
    let mut estimates:  Vec<f64>    = Vec::new();
    let mut ses:        Vec<f64>    = Vec::new();
    let mut variances:  Vec<f64>    = Vec::new();
    let mut dfs:        Vec<u32>    = Vec::new();
    let mut ns:         Vec<u32>    = Vec::new();

    for (d, domain_name) in domain_names.iter().enumerate() {
        for (l, level) in levels_str.iter().enumerate() {
            let theta_full = theta_full_mat[d][l];
            let variance   = variance_from_replicates(method, theta_full, &theta_reps_mat[d][l], &rep_coefs, center);
            by_vals.push(domain_name.clone());
            level_strs.push(level.clone());
            estimates.push(theta_full);
            ses.push(variance.sqrt());
            variances.push(variance);
            dfs.push(df_val);
            ns.push(counts[d]);
        }
    }
    let n_rows = by_vals.len();
    let _ = n_levels; // used implicitly via levels_str iteration above
    df![by_col => by_vals, "y" => vec![value_col; n_rows], "level" => level_strs,
        "est" => estimates, "se" => ses, "var" => variances, "df" => dfs, "n" => ns]
}

// ============================================================================
// Quantiles (median is the p = 0.5 case)
// ============================================================================

/// Replicate-weight quantiles. One row per probability (and per domain when
/// `by_col` is set), carrying the probability in a `prob` column.
#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, rep_weight_cols, method, probs, fay_coef=0.0, rscales=None, center="rep_mean", degrees_of_freedom=None, by_col=None, quantile_method=None))]
#[allow(clippy::too_many_arguments)]
pub fn replicate_quantile(
    _py: Python,
    data: PyDataFrame,
    value_col: String,
    weight_col: String,
    rep_weight_cols: Vec<String>,
    method: String,
    probs: Vec<f64>,
    fay_coef: f64,
    rscales: Option<Vec<f64>>,
    center: &str,
    degrees_of_freedom: Option<u32>,
    by_col: Option<String>,
    quantile_method: Option<String>,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let n_reps = rep_weight_cols.len();
    let rep_method      = parse_rep_method(&method)?;
    let variance_center = parse_variance_center(center)?;
    if let Some(r) = &rscales {
        if r.len() != n_reps {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "rscales has {} entries but there are {} replicate weight columns",
                r.len(),
                n_reps
            )));
        }
    }
    let q_method = quantile_method
        .as_deref()
        .map(SvyQuantileMethod::from_str)
        .unwrap_or(SvyQuantileMethod::Higher);
    let df_val = degrees_of_freedom.unwrap_or(n_reps.saturating_sub(1) as u32);

    let result = _py.detach(|| {
        match by_col.as_deref() {
            None => compute_replicate_quantile_ungrouped(&df, &value_col, &weight_col, &rep_weight_cols,
                rep_method, fay_coef, rscales.as_deref(), variance_center, df_val, &probs, q_method),
            Some(by) => compute_replicate_quantile_grouped(&df, &value_col, &weight_col, &rep_weight_cols,
                rep_method, fay_coef, rscales.as_deref(), variance_center, df_val, by, &probs, q_method),
        }
    });
    result.map(PyDataFrame).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, rep_weight_cols, method, fay_coef=0.0, rscales=None, center="rep_mean", degrees_of_freedom=None, by_col=None, quantile_method=None))]
#[allow(clippy::too_many_arguments)]
pub fn replicate_median(
    _py: Python,
    data: PyDataFrame,
    value_col: String,
    weight_col: String,
    rep_weight_cols: Vec<String>,
    method: String,
    fay_coef: f64,
    rscales: Option<Vec<f64>>,
    center: &str,
    degrees_of_freedom: Option<u32>,
    by_col: Option<String>,
    quantile_method: Option<String>,
) -> PyResult<PyDataFrame> {
    let out = replicate_quantile(
        _py, data, value_col, weight_col, rep_weight_cols, method, vec![0.5],
        fay_coef, rscales, center, degrees_of_freedom, by_col, quantile_method,
    )?;
    // Keep the legacy median schema: drop the probability column.
    let mut result = out.0;
    result
        .drop_in_place("prob")
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(PyDataFrame(result))
}

fn compute_replicate_quantile_ungrouped(
    df: &DataFrame,
    value_col: &str, weight_col: &str, rep_weight_cols: &[String],
    method: RepMethod, fay_coef: f64, rscales: Option<&[f64]>,
    center: VarianceCenter, df_val: u32, probs: &[f64],
    q_method: SvyQuantileMethod,
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let n = y.len();
    let n_reps = rep_weight_cols.len();
    let y_arr: Vec<f64> = y.iter().map(|v| v.unwrap_or(f64::NAN)).collect();
    let w_arr: Vec<f64> = weights.iter().map(|v| v.unwrap_or(0.0)).collect();

    let (rep_w_matrix, _, _) = extract_rep_weights_matrix(df, rep_weight_cols)?;
    let (theta_full, theta_reps) =
        matrix_quantile_estimates(&y_arr, &w_arr, &rep_w_matrix, n, n_reps, probs, q_method);
    let rep_coefs = rscales.map(<[f64]>::to_vec)
        .unwrap_or_else(|| replicate_coefficients(method, n_reps, fay_coef));

    let variances: Vec<f64> = (0..probs.len())
        .map(|j| variance_from_replicates(method, theta_full[j], &theta_reps[j], &rep_coefs, center))
        .collect();
    let ses: Vec<f64> = variances.iter().map(|v| v.sqrt()).collect();
    let k = probs.len();

    df!["y" => vec![value_col; k], "prob" => probs.to_vec(), "est" => theta_full,
        "se" => ses, "var" => variances, "df" => vec![df_val; k], "n" => vec![n as u32; k]]
}

fn compute_replicate_quantile_grouped(
    df: &DataFrame,
    value_col: &str, weight_col: &str, rep_weight_cols: &[String],
    method: RepMethod, fay_coef: f64, rscales: Option<&[f64]>,
    center: VarianceCenter, df_val: u32,
    by_col: &str, probs: &[f64], q_method: SvyQuantileMethod,
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let by_str  = df.column(by_col)?.str()?;
    let n = y.len();
    let n_reps = rep_weight_cols.len();
    let y_arr: Vec<f64> = y.iter().map(|v| v.unwrap_or(f64::NAN)).collect();
    let w_arr: Vec<f64> = weights.iter().map(|v| v.unwrap_or(0.0)).collect();

    let (domain_ids, domain_names, n_domains) = index_domains(by_str);
    let (rep_w_matrix, _, _) = extract_rep_weights_matrix(df, rep_weight_cols)?;
    let (theta_full_vec, theta_reps_vec, counts) = matrix_quantile_by_domain(
        &y_arr, &w_arr, &rep_w_matrix, &domain_ids, n_domains, n, n_reps, probs, q_method);
    let rep_coefs = rscales.map(<[f64]>::to_vec)
        .unwrap_or_else(|| replicate_coefficients(method, n_reps, fay_coef));

    let k = probs.len();
    let n_rows = n_domains * k;
    let mut by_vals:   Vec<String> = Vec::with_capacity(n_rows);
    let mut estimates: Vec<f64>    = Vec::with_capacity(n_rows);
    let mut ses:       Vec<f64>    = Vec::with_capacity(n_rows);
    let mut variances: Vec<f64>    = Vec::with_capacity(n_rows);
    let mut ns:        Vec<u32>    = Vec::with_capacity(n_rows);

    for (d, domain_name) in domain_names.iter().enumerate() {
        for j in 0..k {
            let variance = variance_from_replicates(
                method, theta_full_vec[d][j], &theta_reps_vec[d][j], &rep_coefs, center);
            by_vals.push(domain_name.clone());
            estimates.push(theta_full_vec[d][j]);
            ses.push(variance.sqrt());
            variances.push(variance);
            ns.push(counts[d]);
        }
    }
    let probs_rep: Vec<f64> = std::iter::repeat_n(probs, n_domains).flatten().copied().collect();
    df![by_col => by_vals, "y" => vec![value_col; n_rows], "prob" => probs_rep,
        "est" => estimates, "se" => ses, "var" => variances,
        "df" => vec![df_val; n_rows], "n" => ns]
}
