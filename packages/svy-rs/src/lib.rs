// src/lib.rs
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use polars::prelude::*;

mod estimation;
mod regression;
mod weighting;  // NEW: Add weighting module

// Imports from taylor (via mod.rs re-exports)
use estimation::{
    point_estimate_mean, point_estimate_total, point_estimate_ratio,
    point_estimate_mean_domain, point_estimate_total_domain, point_estimate_ratio_domain,
    scores_mean, scores_total, scores_ratio,
    scores_mean_domain, scores_total_domain, scores_ratio_domain,
    taylor_variance, degrees_of_freedom,
    srs_variance_mean, srs_variance_mean_domain,
    srs_variance_total, srs_variance_total_domain,
    srs_variance_ratio, srs_variance_ratio_domain,
};

// Imports from replication (direct from module)
use crate::estimation::replication::{
    RepMethod, VarianceCenter,
    replicate_coefficients, variance_from_replicates,
    extract_rep_weights_matrix, index_domains,
    matrix_mean_estimates, matrix_total_estimates, matrix_ratio_estimates,
    matrix_mean_by_domain, matrix_total_by_domain, matrix_ratio_by_domain,
    matrix_prop_estimates, matrix_prop_by_domain,
};

// Imports from regression
use regression::glm::fit_glm;

// NEW: Import NumPy for weighting functions
use ndarray::{Array1, ArrayView1};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};

// ============================================================================
// Taylor Linearization Functions
// ============================================================================

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_stage2_col=None, by_col=None, singleton_method=None))]
fn taylor_mean(
    _py: Python,
    data: PyDataFrame,
    value_col: String,
    weight_col: String,
    strata_col: Option<String>,
    psu_col: Option<String>,
    ssu_col: Option<String>,
    fpc_col: Option<String>,
    fpc_stage2_col: Option<String>,
    by_col: Option<String>,
    singleton_method: Option<String>,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();

    if by_col.is_none() {
        let result = compute_mean_ungrouped(
            &df, &value_col, &weight_col, strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(),
            fpc_col.as_deref(), fpc_stage2_col.as_deref(), singleton_method.as_deref()
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        return Ok(PyDataFrame(result));
    }

    let by_col_name = by_col.unwrap();
    let result = compute_mean_grouped(
        &df, &value_col, &weight_col, strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(),
        fpc_col.as_deref(), fpc_stage2_col.as_deref(), &by_col_name, singleton_method.as_deref()
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(PyDataFrame(result))
}

fn compute_mean_ungrouped(
    df: &DataFrame, value_col: &str, weight_col: &str, strata_col: Option<&str>, psu_col: Option<&str>,
    ssu_col: Option<&str>, fpc_col: Option<&str>, fpc_stage2_col: Option<&str>, singleton_method: Option<&str>
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col.map(|col| df.column(col).and_then(|s| s.str())).transpose()?;
    let psu = psu_col.map(|col| df.column(col).and_then(|s| s.str())).transpose()?;
    let ssu = ssu_col.map(|col| df.column(col).and_then(|s| s.str())).transpose()?;
    let fpc = fpc_col.map(|col| df.column(col).and_then(|s| s.f64())).transpose()?;
    let fpc_stage2 = fpc_stage2_col.map(|col| df.column(col).and_then(|s| s.f64())).transpose()?;

    let estimate = point_estimate_mean(y, weights)?;
    let scores = scores_mean(y, weights)?;
    let variance = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_stage2, singleton_method)?;
    let se = variance.sqrt();
    let df_val = degrees_of_freedom(weights, strata, psu)?;
    let n = y.len() as u32;
    let srs_var = srs_variance_mean(y, weights)?;
    let deff = if srs_var > 0.0 { variance / srs_var } else { f64::NAN };

    df!["y" => vec![value_col], "est" => vec![estimate], "se" => vec![se], "var" => vec![variance], "df" => vec![df_val], "n" => vec![n], "deff" => vec![deff]]
}

fn compute_mean_grouped(
    df: &DataFrame, value_col: &str, weight_col: &str, strata_col: Option<&str>, psu_col: Option<&str>,
    ssu_col: Option<&str>, fpc_col: Option<&str>, fpc_stage2_col: Option<&str>, by_col: &str, singleton_method: Option<&str>
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col.map(|col| df.column(col).and_then(|s| s.str())).transpose()?;
    let psu = psu_col.map(|col| df.column(col).and_then(|s| s.str())).transpose()?;
    let ssu = ssu_col.map(|col| df.column(col).and_then(|s| s.str())).transpose()?;
    let fpc = fpc_col.map(|col| df.column(col).and_then(|s| s.f64())).transpose()?;
    let fpc_stage2 = fpc_stage2_col.map(|col| df.column(col).and_then(|s| s.f64())).transpose()?;
    let by_series = df.column(by_col)?;
    let by_str = by_series.str()?;
    let unique_groups = by_str.unique()?;

    let mut by_vals: Vec<&str> = Vec::new();
    let mut estimates: Vec<f64> = Vec::new();
    let mut ses: Vec<f64> = Vec::new();
    let mut variances: Vec<f64> = Vec::new();
    let mut dfs: Vec<u32> = Vec::new();
    let mut ns: Vec<u32> = Vec::new();
    let mut deffs: Vec<f64> = Vec::new();
    let df_val = degrees_of_freedom(weights, strata, psu)?;

    for group_val in unique_groups.iter() {
        if let Some(group) = group_val {
            let domain_mask = by_str.equal(group);
            let n_domain = domain_mask.sum().unwrap_or(0) as u32;
            let estimate = point_estimate_mean_domain(y, weights, &domain_mask)?;
            let scores = scores_mean_domain(y, weights, &domain_mask)?;
            let variance = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_stage2, singleton_method)?;
            let se = variance.sqrt();
            let srs_var = srs_variance_mean_domain(y, weights, &domain_mask)?;
            let deff = if srs_var > 0.0 { variance / srs_var } else { f64::NAN };

            by_vals.push(group);
            estimates.push(estimate);
            ses.push(se);
            variances.push(variance);
            dfs.push(df_val);
            ns.push(n_domain);
            deffs.push(deff);
        }
    }
    let n_groups = by_vals.len();
    df![by_col => by_vals, "y" => vec![value_col; n_groups], "est" => estimates, "se" => ses, "var" => variances, "df" => dfs, "n" => ns, "deff" => deffs]
}

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_stage2_col=None, by_col=None, singleton_method=None))]
fn taylor_total(
    _py: Python, data: PyDataFrame, value_col: String, weight_col: String, strata_col: Option<String>, psu_col: Option<String>,
    ssu_col: Option<String>, fpc_col: Option<String>, fpc_stage2_col: Option<String>, by_col: Option<String>, singleton_method: Option<String>
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    if by_col.is_none() {
        let result = compute_total_ungrouped(&df, &value_col, &weight_col, strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(), fpc_col.as_deref(), fpc_stage2_col.as_deref(), singleton_method.as_deref())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        return Ok(PyDataFrame(result));
    }
    let by_col_name = by_col.unwrap();
    let result = compute_total_grouped(&df, &value_col, &weight_col, strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(), fpc_col.as_deref(), fpc_stage2_col.as_deref(), &by_col_name, singleton_method.as_deref())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(PyDataFrame(result))
}

fn compute_total_ungrouped(
    df: &DataFrame, value_col: &str, weight_col: &str, strata_col: Option<&str>, psu_col: Option<&str>,
    ssu_col: Option<&str>, fpc_col: Option<&str>, fpc_stage2_col: Option<&str>, singleton_method: Option<&str>
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col.map(|col| df.column(col).and_then(|s| s.str())).transpose()?;
    let psu = psu_col.map(|col| df.column(col).and_then(|s| s.str())).transpose()?;
    let ssu = ssu_col.map(|col| df.column(col).and_then(|s| s.str())).transpose()?;
    let fpc = fpc_col.map(|col| df.column(col).and_then(|s| s.f64())).transpose()?;
    let fpc_stage2 = fpc_stage2_col.map(|col| df.column(col).and_then(|s| s.f64())).transpose()?;

    let estimate = point_estimate_total(y, weights)?;
    let scores = scores_total(y, weights)?;
    let variance = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_stage2, singleton_method)?;
    let se = variance.sqrt();
    let df_val = degrees_of_freedom(weights, strata, psu)?;
    let n = y.len() as u32;
    let srs_var = srs_variance_total(y, weights)?;
    let deff = if srs_var > 0.0 { variance / srs_var } else { f64::NAN };

    df!["y" => vec![value_col], "est" => vec![estimate], "se" => vec![se], "var" => vec![variance], "df" => vec![df_val], "n" => vec![n], "deff" => vec![deff]]
}

fn compute_total_grouped(
    df: &DataFrame, value_col: &str, weight_col: &str, strata_col: Option<&str>, psu_col: Option<&str>,
    ssu_col: Option<&str>, fpc_col: Option<&str>, fpc_stage2_col: Option<&str>, by_col: &str, singleton_method: Option<&str>
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col.map(|col| df.column(col).and_then(|s| s.str())).transpose()?;
    let psu = psu_col.map(|col| df.column(col).and_then(|s| s.str())).transpose()?;
    let ssu = ssu_col.map(|col| df.column(col).and_then(|s| s.str())).transpose()?;
    let fpc = fpc_col.map(|col| df.column(col).and_then(|s| s.f64())).transpose()?;
    let fpc_stage2 = fpc_stage2_col.map(|col| df.column(col).and_then(|s| s.f64())).transpose()?;
    let by_series = df.column(by_col)?;
    let by_str = by_series.str()?;
    let unique_groups = by_str.unique()?;

    let mut by_vals: Vec<&str> = Vec::new();
    let mut estimates: Vec<f64> = Vec::new();
    let mut ses: Vec<f64> = Vec::new();
    let mut variances: Vec<f64> = Vec::new();
    let mut dfs: Vec<u32> = Vec::new();
    let mut ns: Vec<u32> = Vec::new();
    let mut deffs: Vec<f64> = Vec::new();
    let df_val = degrees_of_freedom(weights, strata, psu)?;

    for group_val in unique_groups.iter() {
        if let Some(group) = group_val {
            let domain_mask = by_str.equal(group);
            let n_domain = domain_mask.sum().unwrap_or(0) as u32;
            let estimate = point_estimate_total_domain(y, weights, &domain_mask)?;
            let scores = scores_total_domain(y, weights, &domain_mask)?;
            let variance = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_stage2, singleton_method)?;
            let se = variance.sqrt();
            let srs_var = srs_variance_total_domain(y, weights, &domain_mask)?;
            let deff = if srs_var > 0.0 { variance / srs_var } else { f64::NAN };

            by_vals.push(group);
            estimates.push(estimate);
            ses.push(se);
            variances.push(variance);
            dfs.push(df_val);
            ns.push(n_domain);
            deffs.push(deff);
        }
    }
    let n_groups = by_vals.len();
    df![by_col => by_vals, "y" => vec![value_col; n_groups], "est" => estimates, "se" => ses, "var" => variances, "df" => dfs, "n" => ns, "deff" => deffs]
}

#[pyfunction]
#[pyo3(signature = (data, numerator_col, denominator_col, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_stage2_col=None, by_col=None, singleton_method=None))]
fn taylor_ratio(
    _py: Python, data: PyDataFrame, numerator_col: String, denominator_col: String, weight_col: String,
    strata_col: Option<String>, psu_col: Option<String>, ssu_col: Option<String>, fpc_col: Option<String>,
    fpc_stage2_col: Option<String>, by_col: Option<String>, singleton_method: Option<String>
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    if by_col.is_none() {
        let result = compute_ratio_ungrouped(&df, &numerator_col, &denominator_col, &weight_col, strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(), fpc_col.as_deref(), fpc_stage2_col.as_deref(), singleton_method.as_deref())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        return Ok(PyDataFrame(result));
    }
    let by_col_name = by_col.unwrap();
    let result = compute_ratio_grouped(&df, &numerator_col, &denominator_col, &weight_col, strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(), fpc_col.as_deref(), fpc_stage2_col.as_deref(), &by_col_name, singleton_method.as_deref())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(PyDataFrame(result))
}

fn compute_ratio_ungrouped(
    df: &DataFrame, numerator_col: &str, denominator_col: &str, weight_col: &str, strata_col: Option<&str>,
    psu_col: Option<&str>, ssu_col: Option<&str>, fpc_col: Option<&str>, fpc_stage2_col: Option<&str>, singleton_method: Option<&str>
) -> PolarsResult<DataFrame> {
    let y = df.column(numerator_col)?.f64()?;
    let x = df.column(denominator_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col.map(|col| df.column(col).and_then(|s| s.str())).transpose()?;
    let psu = psu_col.map(|col| df.column(col).and_then(|s| s.str())).transpose()?;
    let ssu = ssu_col.map(|col| df.column(col).and_then(|s| s.str())).transpose()?;
    let fpc = fpc_col.map(|col| df.column(col).and_then(|s| s.f64())).transpose()?;
    let fpc_stage2 = fpc_stage2_col.map(|col| df.column(col).and_then(|s| s.f64())).transpose()?;

    let estimate = point_estimate_ratio(y, x, weights)?;
    let scores = scores_ratio(y, x, weights)?;
    let variance = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_stage2, singleton_method)?;
    let se = variance.sqrt();
    let df_val = degrees_of_freedom(weights, strata, psu)?;
    let n = y.len() as u32;
    let srs_var = srs_variance_ratio(y, x, weights)?;
    let deff = if srs_var > 0.0 { variance / srs_var } else { f64::NAN };

    df!["y" => vec![numerator_col], "x" => vec![denominator_col], "est" => vec![estimate], "se" => vec![se], "var" => vec![variance], "df" => vec![df_val], "n" => vec![n], "deff" => vec![deff]]
}

fn compute_ratio_grouped(
    df: &DataFrame, numerator_col: &str, denominator_col: &str, weight_col: &str, strata_col: Option<&str>,
    psu_col: Option<&str>, ssu_col: Option<&str>, fpc_col: Option<&str>, fpc_stage2_col: Option<&str>, by_col: &str, singleton_method: Option<&str>
) -> PolarsResult<DataFrame> {
    let y = df.column(numerator_col)?.f64()?;
    let x = df.column(denominator_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col.map(|col| df.column(col).and_then(|s| s.str())).transpose()?;
    let psu = psu_col.map(|col| df.column(col).and_then(|s| s.str())).transpose()?;
    let ssu = ssu_col.map(|col| df.column(col).and_then(|s| s.str())).transpose()?;
    let fpc = fpc_col.map(|col| df.column(col).and_then(|s| s.f64())).transpose()?;
    let fpc_stage2 = fpc_stage2_col.map(|col| df.column(col).and_then(|s| s.f64())).transpose()?;
    let by_series = df.column(by_col)?;
    let by_str = by_series.str()?;
    let unique_groups = by_str.unique()?;

    let mut by_vals: Vec<&str> = Vec::new();
    let mut estimates: Vec<f64> = Vec::new();
    let mut ses: Vec<f64> = Vec::new();
    let mut variances: Vec<f64> = Vec::new();
    let mut dfs: Vec<u32> = Vec::new();
    let mut ns: Vec<u32> = Vec::new();
    let mut deffs: Vec<f64> = Vec::new();
    let df_val = degrees_of_freedom(weights, strata, psu)?;

    for group_val in unique_groups.iter() {
        if let Some(group) = group_val {
            let domain_mask = by_str.equal(group);
            let n_domain = domain_mask.sum().unwrap_or(0) as u32;
            let estimate = point_estimate_ratio_domain(y, x, weights, &domain_mask)?;
            let scores = scores_ratio_domain(y, x, weights, &domain_mask)?;
            let variance = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_stage2, singleton_method)?;
            let se = variance.sqrt();
            let srs_var = srs_variance_ratio_domain(y, x, weights, &domain_mask)?;
            let deff = if srs_var > 0.0 { variance / srs_var } else { f64::NAN };

            by_vals.push(group);
            estimates.push(estimate);
            ses.push(se);
            variances.push(variance);
            dfs.push(df_val);
            ns.push(n_domain);
            deffs.push(deff);
        }
    }
    let n_groups = by_vals.len();
    df![by_col => by_vals, "y" => vec![numerator_col; n_groups], "x" => vec![denominator_col; n_groups], "est" => estimates, "se" => ses, "var" => variances, "df" => dfs, "n" => ns, "deff" => deffs]
}

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_stage2_col=None, by_col=None, singleton_method=None))]
fn taylor_prop(
    _py: Python, data: PyDataFrame, value_col: String, weight_col: String, strata_col: Option<String>, psu_col: Option<String>,
    ssu_col: Option<String>, fpc_col: Option<String>, fpc_stage2_col: Option<String>, by_col: Option<String>, singleton_method: Option<String>
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    if by_col.is_none() {
        let result = compute_prop_ungrouped(&df, &value_col, &weight_col, strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(), fpc_col.as_deref(), fpc_stage2_col.as_deref(), singleton_method.as_deref())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        return Ok(PyDataFrame(result));
    }
    let by_col_name = by_col.unwrap();
    let result = compute_prop_grouped(&df, &value_col, &weight_col, strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(), fpc_col.as_deref(), fpc_stage2_col.as_deref(), &by_col_name, singleton_method.as_deref())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(PyDataFrame(result))
}

fn compute_prop_ungrouped(
    df: &DataFrame, value_col: &str, weight_col: &str, strata_col: Option<&str>, psu_col: Option<&str>,
    ssu_col: Option<&str>, fpc_col: Option<&str>, fpc_stage2_col: Option<&str>, singleton_method: Option<&str>
) -> PolarsResult<DataFrame> {
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col.map(|col| df.column(col).and_then(|s| s.str())).transpose()?;
    let psu = psu_col.map(|col| df.column(col).and_then(|s| s.str())).transpose()?;
    let ssu = ssu_col.map(|col| df.column(col).and_then(|s| s.str())).transpose()?;
    let fpc = fpc_col.map(|col| df.column(col).and_then(|s| s.f64())).transpose()?;
    let fpc_stage2 = fpc_stage2_col.map(|col| df.column(col).and_then(|s| s.f64())).transpose()?;

    let value_series = df.column(value_col)?;
    let value_str = value_series.cast(&DataType::String)?;
    let value_str = value_str.str()?;
    let unique_levels = value_str.unique()?;
    let mut levels: Vec<String> = unique_levels.iter().filter_map(|v| v.map(|s| s.to_string())).collect();
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

    for lvl in &levels {
        let indicator: Vec<Option<f64>> = value_str.iter().map(|v| match v { Some(val) if val == lvl => Some(1.0), Some(_) => Some(0.0), None => None }).collect();
        let indicator_ca = Float64Chunked::from_slice_options("indicator".into(), &indicator);
        let estimate = point_estimate_mean(&indicator_ca, weights)?;
        let scores = scores_mean(&indicator_ca, weights)?;
        let variance = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_stage2, singleton_method)?;
        let se = variance.sqrt();
        let srs_var = srs_variance_mean(&indicator_ca, weights)?;
        let deff = if srs_var > 0.0 { variance / srs_var } else { f64::NAN };

        level_vals.push(lvl.clone());
        estimates.push(estimate);
        ses.push(se);
        variances.push(variance);
        dfs_vec.push(df_val);
        ns.push(n);
        deffs.push(deff);
    }
    let n_levels = level_vals.len();
    df!["y" => vec![value_col; n_levels], "level" => level_vals, "est" => estimates, "se" => ses, "var" => variances, "df" => dfs_vec, "n" => ns, "deff" => deffs]
}

fn compute_prop_grouped(
    df: &DataFrame, value_col: &str, weight_col: &str, strata_col: Option<&str>, psu_col: Option<&str>,
    ssu_col: Option<&str>, fpc_col: Option<&str>, fpc_stage2_col: Option<&str>, by_col: &str, singleton_method: Option<&str>
) -> PolarsResult<DataFrame> {
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col.map(|col| df.column(col).and_then(|s| s.str())).transpose()?;
    let psu = psu_col.map(|col| df.column(col).and_then(|s| s.str())).transpose()?;
    let ssu = ssu_col.map(|col| df.column(col).and_then(|s| s.str())).transpose()?;
    let fpc = fpc_col.map(|col| df.column(col).and_then(|s| s.f64())).transpose()?;
    let fpc_stage2 = fpc_stage2_col.map(|col| df.column(col).and_then(|s| s.f64())).transpose()?;

    let value_series = df.column(value_col)?;
    let value_str = value_series.cast(&DataType::String)?;
    let value_str = value_str.str()?;
    let unique_levels = value_str.unique()?;
    let mut levels: Vec<String> = unique_levels.iter().filter_map(|v| v.map(|s| s.to_string())).collect();
    levels.sort();

    let by_series = df.column(by_col)?;
    let by_str = by_series.str()?;
    let unique_groups = by_str.unique()?;

    let mut by_vals: Vec<String> = Vec::new();
    let mut level_vals: Vec<String> = Vec::new();
    let mut estimates: Vec<f64> = Vec::new();
    let mut ses: Vec<f64> = Vec::new();
    let mut variances: Vec<f64> = Vec::new();
    let mut dfs_vec: Vec<u32> = Vec::new();
    let mut ns: Vec<u32> = Vec::new();
    let mut deffs: Vec<f64> = Vec::new();
    let df_val = degrees_of_freedom(weights, strata, psu)?;

    for group_val in unique_groups.iter() {
        if let Some(group) = group_val {
            let domain_mask = by_str.equal(group);
            let n_domain = domain_mask.sum().unwrap_or(0) as u32;
            for lvl in &levels {
                let indicator: Vec<Option<f64>> = value_str.iter().map(|v| match v { Some(val) if val == lvl => Some(1.0), Some(_) => Some(0.0), None => None }).collect();
                let indicator_ca = Float64Chunked::from_slice_options("indicator".into(), &indicator);
                let estimate = point_estimate_mean_domain(&indicator_ca, weights, &domain_mask)?;
                let scores = scores_mean_domain(&indicator_ca, weights, &domain_mask)?;
                let variance = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_stage2, singleton_method)?;
                let se = variance.sqrt();
                let srs_var = srs_variance_mean_domain(&indicator_ca, weights, &domain_mask)?;
                let deff = if srs_var > 0.0 { variance / srs_var } else { f64::NAN };

                by_vals.push(group.to_string());
                level_vals.push(lvl.clone());
                estimates.push(estimate);
                ses.push(se);
                variances.push(variance);
                dfs_vec.push(df_val);
                ns.push(n_domain);
                deffs.push(deff);
            }
        }
    }
    let n_rows = by_vals.len();
    df![by_col => by_vals, "y" => vec![value_col; n_rows], "level" => level_vals, "est" => estimates, "se" => ses, "var" => variances, "df" => dfs_vec, "n" => ns, "deff" => deffs]
}

// ============================================================================
// Replication-Based Estimation Functions
// ============================================================================

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, rep_weight_cols, method, fay_coef=0.0, center="rep_mean", degrees_of_freedom=None, by_col=None))]
fn replicate_mean(
    _py: Python,
    data: PyDataFrame,
    value_col: String,
    weight_col: String,
    rep_weight_cols: Vec<String>,
    method: String,
    fay_coef: f64,
    center: &str,
    degrees_of_freedom: Option<u32>,
    by_col: Option<String>,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let n_reps = rep_weight_cols.len();

    let rep_method = RepMethod::from_str(&method)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unknown method: {}. Use 'BRR', 'Bootstrap', 'Jackknife', or 'SDR'", method)
        ))?;

    let variance_center = VarianceCenter::from_str(&center)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unknown center: {}. Use 'replicate_mean' or 'full_sample'", center)
        ))?;

    let df_val = degrees_of_freedom.unwrap_or(n_reps.saturating_sub(1) as u32);

    let result = if by_col.is_none() {
        compute_replicate_mean_ungrouped(&df, &value_col, &weight_col, &rep_weight_cols, rep_method, fay_coef, variance_center, df_val)
    } else {
        compute_replicate_mean_grouped(&df, &value_col, &weight_col, &rep_weight_cols, rep_method, fay_coef, variance_center, df_val, by_col.as_ref().unwrap())
    };

    result.map(PyDataFrame)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

fn compute_replicate_mean_ungrouped(
    df: &DataFrame,
    value_col: &str,
    weight_col: &str,
    rep_weight_cols: &[String],
    method: RepMethod,
    fay_coef: f64,
    center: VarianceCenter,
    df_val: u32,
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let n = y.len();
    let n_reps = rep_weight_cols.len();

    let y_arr: Vec<f64> = y.into_iter().map(|v| v.unwrap_or(0.0)).collect();
    let w_arr: Vec<f64> = weights.into_iter().map(|v| v.unwrap_or(0.0)).collect();

    let (rep_w_matrix, _, _) = extract_rep_weights_matrix(df, rep_weight_cols)?;
    let (theta_full, theta_reps) = matrix_mean_estimates(&y_arr, &w_arr, &rep_w_matrix, n, n_reps);

    let rep_coefs = replicate_coefficients(method, n_reps, fay_coef);
    let variance = variance_from_replicates(method, theta_full, &theta_reps, &rep_coefs, center);
    let se = variance.sqrt();

    df![
        "y" => vec![value_col],
        "est" => vec![theta_full],
        "se" => vec![se],
        "var" => vec![variance],
        "df" => vec![df_val],
        "n" => vec![n as u32],
    ]
}

fn compute_replicate_mean_grouped(
    df: &DataFrame,
    value_col: &str,
    weight_col: &str,
    rep_weight_cols: &[String],
    method: RepMethod,
    fay_coef: f64,
    center: VarianceCenter,
    df_val: u32,
    by_col: &str,
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let by_series = df.column(by_col)?;
    let by_str = by_series.str()?;
    let n = y.len();
    let n_reps = rep_weight_cols.len();

    let y_arr: Vec<f64> = y.into_iter().map(|v| v.unwrap_or(0.0)).collect();
    let w_arr: Vec<f64> = weights.into_iter().map(|v| v.unwrap_or(0.0)).collect();

    let (domain_ids, domain_names, n_domains) = index_domains(by_str);
    let (rep_w_matrix, _, _) = extract_rep_weights_matrix(df, rep_weight_cols)?;

    let (theta_full_vec, theta_reps_vec, counts) = matrix_mean_by_domain(
        &y_arr, &w_arr, &rep_w_matrix, &domain_ids, n_domains, n, n_reps
    );

    let rep_coefs = replicate_coefficients(method, n_reps, fay_coef);

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

    df![
        by_col => by_vals,
        "y" => vec![value_col; n_domains],
        "est" => estimates,
        "se" => ses,
        "var" => variances,
        "df" => dfs,
        "n" => ns,
    ]
}

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, rep_weight_cols, method, fay_coef=0.0, center="rep_mean", degrees_of_freedom=None, by_col=None))]
fn replicate_total(
    _py: Python,
    data: PyDataFrame,
    value_col: String,
    weight_col: String,
    rep_weight_cols: Vec<String>,
    method: String,
    fay_coef: f64,
    center: &str,
    degrees_of_freedom: Option<u32>,
    by_col: Option<String>,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let n_reps = rep_weight_cols.len();

    let rep_method = RepMethod::from_str(&method)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unknown method: {}. Use 'BRR', 'Bootstrap', 'Jackknife', or 'SDR'", method)
        ))?;

    let variance_center = VarianceCenter::from_str(&center)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unknown center: {}. Use 'replicate_mean' or 'full_sample'", center)
        ))?;

    let df_val = degrees_of_freedom.unwrap_or(n_reps.saturating_sub(1) as u32);

    let result = if by_col.is_none() {
        compute_replicate_total_ungrouped(&df, &value_col, &weight_col, &rep_weight_cols, rep_method, fay_coef, variance_center, df_val)
    } else {
        compute_replicate_total_grouped(&df, &value_col, &weight_col, &rep_weight_cols, rep_method, fay_coef, variance_center, df_val, by_col.as_ref().unwrap())
    };

    result.map(PyDataFrame)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

fn compute_replicate_total_ungrouped(
    df: &DataFrame,
    value_col: &str,
    weight_col: &str,
    rep_weight_cols: &[String],
    method: RepMethod,
    fay_coef: f64,
    center: VarianceCenter,
    df_val: u32,
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let n = y.len();
    let n_reps = rep_weight_cols.len();

    let y_arr: Vec<f64> = y.into_iter().map(|v| v.unwrap_or(0.0)).collect();
    let w_arr: Vec<f64> = weights.into_iter().map(|v| v.unwrap_or(0.0)).collect();

    let (rep_w_matrix, _, _) = extract_rep_weights_matrix(df, rep_weight_cols)?;
    let (theta_full, theta_reps) = matrix_total_estimates(&y_arr, &w_arr, &rep_w_matrix, n, n_reps);

    let rep_coefs = replicate_coefficients(method, n_reps, fay_coef);
    let variance = variance_from_replicates(method, theta_full, &theta_reps, &rep_coefs, center);
    let se = variance.sqrt();

    df![
        "y" => vec![value_col],
        "est" => vec![theta_full],
        "se" => vec![se],
        "var" => vec![variance],
        "df" => vec![df_val],
        "n" => vec![n as u32],
    ]
}

fn compute_replicate_total_grouped(
    df: &DataFrame,
    value_col: &str,
    weight_col: &str,
    rep_weight_cols: &[String],
    method: RepMethod,
    fay_coef: f64,
    center: VarianceCenter,
    df_val: u32,
    by_col: &str,
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let by_series = df.column(by_col)?;
    let by_str = by_series.str()?;
    let n = y.len();
    let n_reps = rep_weight_cols.len();

    let y_arr: Vec<f64> = y.into_iter().map(|v| v.unwrap_or(0.0)).collect();
    let w_arr: Vec<f64> = weights.into_iter().map(|v| v.unwrap_or(0.0)).collect();

    let (domain_ids, domain_names, n_domains) = index_domains(by_str);
    let (rep_w_matrix, _, _) = extract_rep_weights_matrix(df, rep_weight_cols)?;

    let (theta_full_vec, theta_reps_vec, counts) = matrix_total_by_domain(
        &y_arr, &w_arr, &rep_w_matrix, &domain_ids, n_domains, n, n_reps
    );

    let rep_coefs = replicate_coefficients(method, n_reps, fay_coef);

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

    df![
        by_col => by_vals,
        "y" => vec![value_col; n_domains],
        "est" => estimates,
        "se" => ses,
        "var" => variances,
        "df" => dfs,
        "n" => ns,
    ]
}

#[pyfunction]
#[pyo3(signature = (data, numerator_col, denominator_col, weight_col, rep_weight_cols, method, fay_coef=0.0, center="rep_mean", degrees_of_freedom=None, by_col=None))]
fn replicate_ratio(
    _py: Python,
    data: PyDataFrame,
    numerator_col: String,
    denominator_col: String,
    weight_col: String,
    rep_weight_cols: Vec<String>,
    method: String,
    fay_coef: f64,
    center: &str,
    degrees_of_freedom: Option<u32>,
    by_col: Option<String>,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let n_reps = rep_weight_cols.len();

    let rep_method = RepMethod::from_str(&method)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unknown method: {}. Use 'BRR', 'Bootstrap', 'Jackknife', or 'SDR'", method)
        ))?;

    let variance_center = VarianceCenter::from_str(&center)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unknown center: {}. Use 'replicate_mean' or 'full_sample'", center)
        ))?;

    let df_val = degrees_of_freedom.unwrap_or(n_reps.saturating_sub(1) as u32);

    let result = if by_col.is_none() {
        compute_replicate_ratio_ungrouped(&df, &numerator_col, &denominator_col, &weight_col, &rep_weight_cols, rep_method, fay_coef, variance_center, df_val)
    } else {
        compute_replicate_ratio_grouped(&df, &numerator_col, &denominator_col, &weight_col, &rep_weight_cols, rep_method, fay_coef, variance_center, df_val, by_col.as_ref().unwrap())
    };

    result.map(PyDataFrame)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

fn compute_replicate_ratio_ungrouped(
    df: &DataFrame,
    numerator_col: &str,
    denominator_col: &str,
    weight_col: &str,
    rep_weight_cols: &[String],
    method: RepMethod,
    fay_coef: f64,
    center: VarianceCenter,
    df_val: u32,
) -> PolarsResult<DataFrame> {
    let y = df.column(numerator_col)?.f64()?;
    let x = df.column(denominator_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let n = y.len();
    let n_reps = rep_weight_cols.len();

    let y_arr: Vec<f64> = y.into_iter().map(|v| v.unwrap_or(0.0)).collect();
    let x_arr: Vec<f64> = x.into_iter().map(|v| v.unwrap_or(0.0)).collect();
    let w_arr: Vec<f64> = weights.into_iter().map(|v| v.unwrap_or(0.0)).collect();

    let (rep_w_matrix, _, _) = extract_rep_weights_matrix(df, rep_weight_cols)?;
    let (theta_full, theta_reps) = matrix_ratio_estimates(&y_arr, &x_arr, &w_arr, &rep_w_matrix, n, n_reps);

    let rep_coefs = replicate_coefficients(method, n_reps, fay_coef);
    let variance = variance_from_replicates(method, theta_full, &theta_reps, &rep_coefs, center);
    let se = variance.sqrt();

    df![
        "y" => vec![numerator_col],
        "x" => vec![denominator_col],
        "est" => vec![theta_full],
        "se" => vec![se],
        "var" => vec![variance],
        "df" => vec![df_val],
        "n" => vec![n as u32],
    ]
}

fn compute_replicate_ratio_grouped(
    df: &DataFrame,
    numerator_col: &str,
    denominator_col: &str,
    weight_col: &str,
    rep_weight_cols: &[String],
    method: RepMethod,
    fay_coef: f64,
    center: VarianceCenter,
    df_val: u32,
    by_col: &str,
) -> PolarsResult<DataFrame> {
    let y = df.column(numerator_col)?.f64()?;
    let x = df.column(denominator_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let by_series = df.column(by_col)?;
    let by_str = by_series.str()?;
    let n = y.len();
    let n_reps = rep_weight_cols.len();

    let y_arr: Vec<f64> = y.into_iter().map(|v| v.unwrap_or(0.0)).collect();
    let x_arr: Vec<f64> = x.into_iter().map(|v| v.unwrap_or(0.0)).collect();
    let w_arr: Vec<f64> = weights.into_iter().map(|v| v.unwrap_or(0.0)).collect();

    let (domain_ids, domain_names, n_domains) = index_domains(by_str);
    let (rep_w_matrix, _, _) = extract_rep_weights_matrix(df, rep_weight_cols)?;

    let (theta_full_vec, theta_reps_vec, counts) = matrix_ratio_by_domain(
        &y_arr, &x_arr, &w_arr, &rep_w_matrix, &domain_ids, n_domains, n, n_reps
    );

    let rep_coefs = replicate_coefficients(method, n_reps, fay_coef);

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

    df![
        by_col => by_vals,
        "y" => vec![numerator_col; n_domains],
        "x" => vec![denominator_col; n_domains],
        "est" => estimates,
        "se" => ses,
        "var" => variances,
        "df" => dfs,
        "n" => ns,
    ]
}

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, rep_weight_cols, method, fay_coef=0.0, center="rep_mean", degrees_of_freedom=None, by_col=None))]
fn replicate_prop(
    _py: Python,
    data: PyDataFrame,
    value_col: String,
    weight_col: String,
    rep_weight_cols: Vec<String>,
    method: String,
    fay_coef: f64,
    center: &str,
    degrees_of_freedom: Option<u32>,
    by_col: Option<String>,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let n_reps = rep_weight_cols.len();

    let rep_method = RepMethod::from_str(&method)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unknown method: {}. Use 'BRR', 'Bootstrap', 'Jackknife', or 'SDR'", method)
        ))?;

    let variance_center = VarianceCenter::from_str(&center)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unknown center: {}. Use 'replicate_mean' or 'full_sample'", center)
        ))?;

    let df_val = degrees_of_freedom.unwrap_or(n_reps.saturating_sub(1) as u32);

    let result = if by_col.is_none() {
        compute_replicate_prop_ungrouped(&df, &value_col, &weight_col, &rep_weight_cols, rep_method, fay_coef, variance_center, df_val)
    } else {
        compute_replicate_prop_grouped(&df, &value_col, &weight_col, &rep_weight_cols, rep_method, fay_coef, variance_center, df_val, by_col.as_ref().unwrap())
    };

    result.map(PyDataFrame)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

fn compute_replicate_prop_ungrouped(
    df: &DataFrame,
    value_col: &str,
    weight_col: &str,
    rep_weight_cols: &[String],
    method: RepMethod,
    fay_coef: f64,
    center: VarianceCenter,
    df_val: u32,
) -> PolarsResult<DataFrame> {
    let y_series = df.column(value_col)?;
    let weights = df.column(weight_col)?.f64()?;
    let n = y_series.len();
    let n_reps = rep_weight_cols.len();

    // Convert y to i64 for categorical handling
    let y_arr: Vec<i64> = if y_series.dtype().is_integer() {
        y_series.i64()?.into_iter().map(|v| v.unwrap_or(0)).collect()
    } else if y_series.dtype() == &DataType::Boolean {
        y_series.bool()?.into_iter().map(|v| if v.unwrap_or(false) { 1 } else { 0 }).collect()
    } else {
        // Try to cast to i64
        y_series.cast(&DataType::Int64)?.i64()?.into_iter().map(|v| v.unwrap_or(0)).collect()
    };

    let w_arr: Vec<f64> = weights.into_iter().map(|v| v.unwrap_or(0.0)).collect();

    let (rep_w_matrix, _, _) = extract_rep_weights_matrix(df, rep_weight_cols)?;
    let (levels, theta_full_vec, theta_reps_vec) = matrix_prop_estimates(&y_arr, &w_arr, &rep_w_matrix, n, n_reps);

    let rep_coefs = replicate_coefficients(method, n_reps, fay_coef);
    let n_levels = levels.len();

    let mut level_strs: Vec<String> = Vec::with_capacity(n_levels);
    let mut estimates: Vec<f64> = Vec::with_capacity(n_levels);
    let mut ses: Vec<f64> = Vec::with_capacity(n_levels);
    let mut variances: Vec<f64> = Vec::with_capacity(n_levels);
    let mut dfs: Vec<u32> = Vec::with_capacity(n_levels);
    let mut ns: Vec<u32> = Vec::with_capacity(n_levels);

    for (l, &level) in levels.iter().enumerate() {
        let variance = variance_from_replicates(method, theta_full_vec[l], &theta_reps_vec[l], &rep_coefs, center);

        level_strs.push(level.to_string());
        estimates.push(theta_full_vec[l]);
        ses.push(variance.sqrt());
        variances.push(variance);
        dfs.push(df_val);
        ns.push(n as u32);
    }

    df![
        "y" => vec![value_col; n_levels],
        "level" => level_strs,
        "est" => estimates,
        "se" => ses,
        "var" => variances,
        "df" => dfs,
        "n" => ns,
    ]
}

fn compute_replicate_prop_grouped(
    df: &DataFrame,
    value_col: &str,
    weight_col: &str,
    rep_weight_cols: &[String],
    method: RepMethod,
    fay_coef: f64,
    center: VarianceCenter,
    df_val: u32,
    by_col: &str,
) -> PolarsResult<DataFrame> {
    let y_series = df.column(value_col)?;
    let weights = df.column(weight_col)?.f64()?;
    let by_series = df.column(by_col)?;
    let by_str = by_series.str()?;
    let n = y_series.len();
    let n_reps = rep_weight_cols.len();

    let y_arr: Vec<i64> = if y_series.dtype().is_integer() {
        y_series.i64()?.into_iter().map(|v| v.unwrap_or(0)).collect()
    } else if y_series.dtype() == &DataType::Boolean {
        y_series.bool()?.into_iter().map(|v| if v.unwrap_or(false) { 1 } else { 0 }).collect()
    } else {
        y_series.cast(&DataType::Int64)?.i64()?.into_iter().map(|v| v.unwrap_or(0)).collect()
    };

    let w_arr: Vec<f64> = weights.into_iter().map(|v| v.unwrap_or(0.0)).collect();

    let (domain_ids, domain_names, n_domains) = index_domains(by_str);
    let (rep_w_matrix, _, _) = extract_rep_weights_matrix(df, rep_weight_cols)?;

    let (levels, theta_full_mat, theta_reps_mat, counts) = matrix_prop_by_domain(
        &y_arr, &w_arr, &rep_w_matrix, &domain_ids, n_domains, n, n_reps
    );

    let rep_coefs = replicate_coefficients(method, n_reps, fay_coef);
    let _n_levels = levels.len();

    // Output: one row per (domain, level)
    let mut by_vals: Vec<String> = Vec::new();
    let mut level_strs: Vec<String> = Vec::new();
    let mut estimates: Vec<f64> = Vec::new();
    let mut ses: Vec<f64> = Vec::new();
    let mut variances: Vec<f64> = Vec::new();
    let mut dfs: Vec<u32> = Vec::new();
    let mut ns: Vec<u32> = Vec::new();

    for (d, domain_name) in domain_names.iter().enumerate() {
        for (l, &level) in levels.iter().enumerate() {
            let theta_full = theta_full_mat[d][l];
            let theta_reps = &theta_reps_mat[d][l];
            let variance = variance_from_replicates(method, theta_full, theta_reps, &rep_coefs, center);

            by_vals.push(domain_name.clone());
            level_strs.push(level.to_string());
            estimates.push(theta_full);
            ses.push(variance.sqrt());
            variances.push(variance);
            dfs.push(df_val);
            ns.push(counts[d]);
        }
    }

    let n_rows = by_vals.len();
    df![
        by_col => by_vals,
        "y" => vec![value_col; n_rows],
        "level" => level_strs,
        "est" => estimates,
        "se" => ses,
        "var" => variances,
        "df" => dfs,
        "n" => ns,
    ]
}

// ============================================================================
// GLM Regression Function
// ============================================================================

#[pyfunction]
fn fit_glm_rs(
    _py: Python,
    y_name: String,
    x_names: Vec<String>,
    weight_name: String,
    stratum_name: Option<String>,
    psu_name: Option<String>,
    family: String,
    link: String,
    tol: f64,
    max_iter: usize,
    data: PyDataFrame,
) -> PyResult<(Vec<f64>, Vec<f64>, f64, f64, f64, f64, u32)> {
    let df: DataFrame = data.into();

    // Fix: Convert &Column to Series
    let y = df.column(&y_name)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
        .as_materialized_series()
        .clone();

    let mut x_cols = Vec::with_capacity(x_names.len());
    for name in &x_names {
        let s = df.column(name)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
            .as_materialized_series()
            .clone();
        x_cols.push(s);
    }

    let weights = df.column(&weight_name)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
        .as_materialized_series()
        .clone();

    let stratum = if let Some(s) = stratum_name {
        Some(df.column(&s)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
            .as_materialized_series()
            .clone())
    } else {
        None
    };

    let psu = if let Some(p) = psu_name {
        Some(df.column(&p)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
            .as_materialized_series()
            .clone())
    } else {
        None
    };

    let result = fit_glm(&y, x_cols, &weights, stratum.as_ref(), psu.as_ref(), &family, &link, tol, max_iter)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok((
        result.params,
        result.cov_params,
        result.scale,
        result.df_resid,
        result.deviance,
        result.null_deviance,
        result.iterations
    ))
}

// ============================================================================
// NEW: Weighting Adjustment Functions
// ============================================================================

/// Raking (Iterative Proportional Fitting) for weight adjustment
#[pyfunction]
#[pyo3(signature = (wgt, margin_indices, margin_targets, ll_bound=None, up_bound=None, tol=1e-6, max_iter=100))]
fn rake(
    py: Python<'_>,
    wgt: PyReadonlyArray2<f64>,
    margin_indices: Vec<PyReadonlyArray1<i64>>,
    margin_targets: Vec<PyReadonlyArray1<f64>>,
    ll_bound: Option<f64>,
    up_bound: Option<f64>,
    tol: f64,
    max_iter: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let wgt_arr = wgt.as_array();

    let indices: Vec<_> = margin_indices.iter()
        .map(|x| x.as_array().to_owned())
        .collect();

    let targets: Vec<_> = margin_targets.iter()
        .map(|x| x.as_array().to_owned())
        .collect();

    let result = weighting::raking::rake_impl(
        wgt_arr,
        &indices,
        &targets,
        ll_bound,
        up_bound,
        tol,
        max_iter,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(result.into_pyarray(py).to_owned().into())
}

/// Non-response adjustment
#[pyfunction]
#[pyo3(signature = (wgts, adj_class, resp_status, unknown_to_inelig=true))]
fn adjust_nr(
    py: Python<'_>,
    wgts: PyReadonlyArray2<f64>,
    adj_class: PyReadonlyArray1<i64>,
    resp_status: PyReadonlyArray1<i64>,
    unknown_to_inelig: bool,
) -> PyResult<Py<PyArray2<f64>>> {
    let wgts_arr = wgts.as_array();
    let class_arr = adj_class.as_array();
    let status_arr = resp_status.as_array();

    let result = weighting::nonresponse::adjust_nr_impl(
        wgts_arr,
        class_arr,
        status_arr,
        unknown_to_inelig,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(result.into_pyarray(py).to_owned().into())
}

/// Weight normalization
#[pyfunction]
#[pyo3(signature = (wgt, by_arr=None, control=None))]
fn normalize(
    py: Python<'_>,
    wgt: PyReadonlyArray2<f64>,
    by_arr: Option<PyReadonlyArray1<i64>>,
    control: Option<PyReadonlyArray1<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let wgt_arr = wgt.as_array();
    let by = by_arr.map(|x| x.as_array().to_owned());
    let ctrl = control.map(|x| x.as_array().to_owned());

    let result = weighting::normalization::normalize_impl(
        wgt_arr,
        by.as_ref(),
        ctrl.as_ref(),
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(result.into_pyarray(py).to_owned().into())
}

/// Post-stratification adjustment
#[pyfunction]
fn poststratify(
    py: Python<'_>,
    wgt: PyReadonlyArray2<f64>,
    by_arr: PyReadonlyArray1<i64>,
    control: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let wgt_arr = wgt.as_array();
    let by = by_arr.as_array();
    let ctrl = control.as_array().to_owned();

    let result = weighting::poststratification::poststratify_impl(
        wgt_arr,
        by,
        &ctrl,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(result.into_pyarray(py).to_owned().into())
}

/// Post-stratification with factors
#[pyfunction]
fn poststratify_factor(
    py: Python<'_>,
    wgt: PyReadonlyArray2<f64>,
    by_arr: PyReadonlyArray1<i64>,
    factor: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let wgt_arr = wgt.as_array();
    let by = by_arr.as_array();
    let fct = factor.as_array().to_owned();

    let result = weighting::poststratification::poststratify_factor(
        wgt_arr,
        by,
        &fct,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(result.into_pyarray(py).to_owned().into())
}

// ============================================================================
// NEW: Calibration Functions
// ============================================================================

/// Linear calibration (Deville-Särndal method)
///
/// Calibrates sample weights to match known population totals using linear calibration.
/// Minimizes chi-squared distance: Σᵢ (gᵢ - 1)² / sᵢ
/// Subject to: Σᵢ wᵢ gᵢ xᵢⱼ = Tⱼ for all j
///
/// # Arguments
/// * `wgt` - Initial weights matrix (n_obs, n_reps)
/// * `x_matrix` - Auxiliary variables matrix (n_obs, n_aux)
/// * `totals` - Known population totals (n_aux,)
/// * `scale` - Optional scale factors for distance function (n_obs,)
/// * `additive` - If true, return g-factors instead of calibrated weights
///
/// # Returns
/// Calibrated weights or g-factors (n_obs, n_reps)
#[pyfunction]
#[pyo3(signature = (wgt, x_matrix, totals, scale=None, additive=false))]
fn calibrate(
    py: Python<'_>,
    wgt: PyReadonlyArray2<f64>,
    x_matrix: PyReadonlyArray2<f64>,
    totals: PyReadonlyArray1<f64>,
    scale: Option<PyReadonlyArray1<f64>>,
    additive: bool,
) -> PyResult<Py<PyArray2<f64>>> {
    let wgt_arr = wgt.as_array();
    let x_arr = x_matrix.as_array();
    let totals_arr = totals.as_array();
    let scale_owned: Option<Array1<f64>> = scale.map(|s| Array1::from(s.as_array().to_vec()));
    let scale_view: Option<ArrayView1<f64>> = scale_owned.as_ref().map(|a| a.view());

    let result = weighting::calibration::calibrate_linear(
        wgt_arr,
        x_arr,
        totals_arr,
        scale_view,
        additive,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(result.into_pyarray(py).to_owned().into())
}

/// Calibration by domain (group-specific calibration)
///
/// Calibrates weights separately within each domain/group.
/// More efficient than global calibration when different domains have different totals.
///
/// # Arguments
/// * `wgt` - Initial weights matrix (n_obs, n_reps)
/// * `x_matrix` - Auxiliary variables matrix (n_obs, n_aux)
/// * `domain` - Domain identifiers (n_obs,)
/// * `controls_dict` - Map from domain ID to control totals array
/// * `scale` - Optional scale factors (n_obs,)
/// * `additive` - If true, return g-factors
///
/// # Returns
/// Calibrated weights or g-factors (n_obs, n_reps)
#[pyfunction]
#[pyo3(signature = (wgt, x_matrix, domain, controls_dict, scale=None, additive=false))]
fn calibrate_by_domain(
    py: Python<'_>,
    wgt: PyReadonlyArray2<f64>,
    x_matrix: PyReadonlyArray2<f64>,
    domain: PyReadonlyArray1<i64>,
    controls_dict: std::collections::HashMap<i64, Vec<f64>>,
    scale: Option<PyReadonlyArray1<f64>>,
    additive: bool,
) -> PyResult<Py<PyArray2<f64>>> {
    use ndarray::Array1;

    let wgt_arr = wgt.as_array();
    let x_arr = x_matrix.as_array();
    let domain_arr = domain.as_array();
    let scale_owned: Option<Array1<f64>> = scale.map(|s| Array1::from(s.as_array().to_vec()));
    let scale_view: Option<ArrayView1<f64>> = scale_owned.as_ref().map(|a| a.view());

    // Convert HashMap<i64, Vec<f64>> to HashMap<i64, Array1<f64>>
    let controls: std::collections::HashMap<i64, Array1<f64>> = controls_dict
        .into_iter()
        .map(|(k, v)| (k, Array1::from(v)))
        .collect();

    let result = weighting::calibration::calibrate_by_domain(
        wgt_arr,
        x_arr,
        domain_arr,
        &controls,
        scale_view,
        additive,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;


    Ok(result.into_pyarray(py).to_owned().into())
}

/// Parallel calibration (optimized for many replicates)
///
/// Processes each replicate weight in parallel using multiple CPU cores.
/// Recommended for 10+ replicates (e.g., jackknife, bootstrap).
///
/// # Arguments
/// * `wgt` - Initial weights matrix (n_obs, n_reps)
/// * `x_matrix` - Auxiliary variables matrix (n_obs, n_aux)
/// * `totals` - Known population totals (n_aux,)
/// * `scale` - Optional scale factors (n_obs,)
///
/// # Returns
/// Calibrated weights (n_obs, n_reps)
#[pyfunction]
#[pyo3(signature = (wgt, x_matrix, totals, scale=None))]
fn calibrate_parallel(
    py: Python<'_>,
    wgt: PyReadonlyArray2<f64>,
    x_matrix: PyReadonlyArray2<f64>,
    totals: PyReadonlyArray1<f64>,
    scale: Option<PyReadonlyArray1<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let wgt_arr = wgt.as_array();
    let x_arr = x_matrix.as_array();
    let totals_arr = totals.as_array();
    let scale_owned: Option<Array1<f64>> = scale.map(|s| Array1::from(s.as_array().to_vec()));
    let scale_view: Option<ArrayView1<f64>> = scale_owned.as_ref().map(|a| a.view());

    let result = weighting::calibration::calibrate_parallel(
        wgt_arr,
        x_arr,
        totals_arr,
        scale_view,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;


    Ok(result.into_pyarray(py).to_owned().into())
}


// ============================================================================
// Replicate Weight Creation Functions (add to lib.rs)
// ============================================================================

// Add to imports at top of lib.rs:
// use crate::weighting::replication::{
//     create_brr_weights, create_jkn_weights, create_bootstrap_weights, create_sdr_weights
// };

/// Create BRR (Balanced Repeated Replication) weights
///
/// # Arguments
/// * `wgt` - Base weights (n_obs,)
/// * `stratum` - Stratum identifiers (n_obs,)
/// * `psu` - PSU identifiers (n_obs,)
/// * `n_reps` - Number of replicates (optional, defaults to min needed)
/// * `fay_coef` - Fay adjustment (0.0 = standard BRR)
///
/// # Returns
/// Tuple of (replicate_weights, degrees_of_freedom)
#[pyfunction]
#[pyo3(signature = (wgt, stratum, psu, n_reps=None, fay_coef=0.0))]
fn create_brr_wgts(
    py: Python<'_>,
    wgt: PyReadonlyArray1<f64>,
    stratum: PyReadonlyArray1<i64>,
    psu: PyReadonlyArray1<i64>,
    n_reps: Option<usize>,
    fay_coef: f64,
) -> PyResult<(Py<PyArray2<f64>>, f64)> {
    let wgt_arr = wgt.as_array();
    let stratum_arr = stratum.as_array();
    let psu_arr = psu.as_array();

    let (result, df) = weighting::replication::create_brr_weights(
        wgt_arr,
        stratum_arr,
        psu_arr,
        n_reps,
        fay_coef,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok((result.into_pyarray(py).to_owned().into(), df))
}

/// Create Jackknife (JKn) replicate weights
///
/// # Arguments
/// * `wgt` - Base weights (n_obs,)
/// * `stratum` - Stratum identifiers (n_obs,), optional
/// * `psu` - PSU identifiers (n_obs,)
///
/// # Returns
/// Tuple of (replicate_weights, degrees_of_freedom)
#[pyfunction]
#[pyo3(signature = (wgt, psu, stratum=None))]
fn create_jkn_wgts(
    py: Python<'_>,
    wgt: PyReadonlyArray1<f64>,
    psu: PyReadonlyArray1<i64>,
    stratum: Option<PyReadonlyArray1<i64>>,
) -> PyResult<(Py<PyArray2<f64>>, f64)> {
    let wgt_arr = wgt.as_array();
    let psu_arr = psu.as_array();
    let stratum_view = stratum.as_ref().map(|s| s.as_array());

    let (result, df) = weighting::replication::create_jkn_weights(
        wgt_arr,
        stratum_view,
        psu_arr,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok((result.into_pyarray(py).to_owned().into(), df))
}

/// Create Bootstrap replicate weights
///
/// # Arguments
/// * `wgt` - Base weights (n_obs,)
/// * `psu` - PSU identifiers (n_obs,)
/// * `n_reps` - Number of bootstrap replicates
/// * `stratum` - Stratum identifiers (n_obs,), optional
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// Tuple of (replicate_weights, degrees_of_freedom)
#[pyfunction]
#[pyo3(signature = (wgt, psu, n_reps, stratum=None, seed=None))]
fn create_bootstrap_wgts(
    py: Python<'_>,
    wgt: PyReadonlyArray1<f64>,
    psu: PyReadonlyArray1<i64>,
    n_reps: usize,
    stratum: Option<PyReadonlyArray1<i64>>,
    seed: Option<u64>,
) -> PyResult<(Py<PyArray2<f64>>, f64)> {
    let wgt_arr = wgt.as_array();
    let psu_arr = psu.as_array();
    let stratum_view = stratum.as_ref().map(|s| s.as_array());

    // Use provided seed or generate from system time
    let seed = seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42)
    });

    let (result, df) = weighting::replication::create_bootstrap_weights(
        wgt_arr,
        stratum_view,
        psu_arr,
        n_reps,
        seed,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok((result.into_pyarray(py).to_owned().into(), df))
}

/// Create SDR (Successive Difference Replication) weights
///
/// # Arguments
/// * `wgt` - Base weights (n_obs,)
/// * `n_reps` - Number of replicates
/// * `stratum` - Stratum identifiers (n_obs,), optional
/// * `order` - Sort order within strata (n_obs,), optional
///
/// # Returns
/// Tuple of (replicate_weights, degrees_of_freedom)
#[pyfunction]
#[pyo3(signature = (wgt, n_reps, stratum=None, order=None))]
fn create_sdr_wgts(
    py: Python<'_>,
    wgt: PyReadonlyArray1<f64>,
    n_reps: usize,
    stratum: Option<PyReadonlyArray1<i64>>,
    order: Option<PyReadonlyArray1<i64>>,
) -> PyResult<(Py<PyArray2<f64>>, f64)> {
    let wgt_arr = wgt.as_array();
    let stratum_view = stratum.as_ref().map(|s| s.as_array());
    let order_view = order.as_ref().map(|o| o.as_array());

    let (result, df) = weighting::replication::create_sdr_weights(
        wgt_arr,
        stratum_view,
        order_view,
        n_reps,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok((result.into_pyarray(py).to_owned().into(), df))
}


#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Taylor functions
    m.add_function(wrap_pyfunction!(taylor_mean, m)?)?;
    m.add_function(wrap_pyfunction!(taylor_total, m)?)?;
    m.add_function(wrap_pyfunction!(taylor_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(taylor_prop, m)?)?;

    // Replication functions
    m.add_function(wrap_pyfunction!(replicate_mean, m)?)?;
    m.add_function(wrap_pyfunction!(replicate_total, m)?)?;
    m.add_function(wrap_pyfunction!(replicate_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(replicate_prop, m)?)?;

    // GLM Regression
    m.add_function(wrap_pyfunction!(fit_glm_rs, m)?)?;

    // Weighting functions
    m.add_function(wrap_pyfunction!(rake, m)?)?;
    m.add_function(wrap_pyfunction!(adjust_nr, m)?)?;
    m.add_function(wrap_pyfunction!(normalize, m)?)?;
    m.add_function(wrap_pyfunction!(poststratify, m)?)?;
    m.add_function(wrap_pyfunction!(poststratify_factor, m)?)?;

    // Calibration functions (NEW)
    m.add_function(wrap_pyfunction!(calibrate, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate_by_domain, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate_parallel, m)?)?;

    // Replicate weight creation functions
    m.add_function(wrap_pyfunction!(create_brr_wgts, m)?)?;
    m.add_function(wrap_pyfunction!(create_jkn_wgts, m)?)?;
    m.add_function(wrap_pyfunction!(create_bootstrap_wgts, m)?)?;
    m.add_function(wrap_pyfunction!(create_sdr_wgts, m)?)?;
    Ok(())
}
