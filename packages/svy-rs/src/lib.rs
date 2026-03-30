// src/lib.rs
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use std::collections::HashMap;

mod categorical;
mod estimation;
mod regression;
mod weighting;

// Imports from taylor (via mod.rs re-exports)
use estimation::{
    SvyQuantileMethod, degrees_of_freedom, median_variance_woodruff,
    median_variance_woodruff_domain, point_estimate_mean, point_estimate_mean_domain,
    point_estimate_ratio, point_estimate_ratio_domain, point_estimate_total,
    point_estimate_total_domain, scores_mean, scores_mean_domain, scores_ratio,
    scores_ratio_domain, scores_total, scores_total_domain, srs_variance_mean,
    srs_variance_mean_domain, srs_variance_ratio, srs_variance_ratio_domain, srs_variance_total,
    srs_variance_total_domain, taylor_variance, weighted_median, weighted_median_domain,
};

// Imports from replication (direct from module)
use crate::estimation::replication::{
    RepMethod,
    VarianceCenter,
    extract_rep_weights_matrix,
    index_domains,
    matrix_mean_by_domain,
    matrix_mean_estimates,
    matrix_median_by_domain,
    // NEW: Median replication imports
    matrix_median_estimates,
    matrix_prop_by_domain,
    matrix_prop_estimates,
    matrix_ratio_by_domain,
    matrix_ratio_estimates,
    matrix_total_by_domain,
    matrix_total_estimates,
    replicate_coefficients,
    variance_from_replicates,
};

// Imports from regression
use regression::glm::fit_glm;

// Imports from categorical
use categorical::ranktest::{RankScoreMethod, ranktest_k_sample, ranktest_two_sample};
use categorical::tabulation::{
    count_strata_psus, estimate_proportions, estimate_totals, rao_scott, sort_levels,
};
use categorical::ttest::{ttest_one_sample, ttest_one_sample_domain, ttest_two_sample};

// Import NumPy for weighting functions
use ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};

// ============================================================================
// Taylor Linearization Functions
// ============================================================================

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, by_col=None, singleton_method=None))]
fn taylor_mean(
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
    let df: DataFrame = data.into();

    if by_col.is_none() {
        let result = compute_mean_ungrouped(
            &df,
            &value_col,
            &weight_col,
            strata_col.as_deref(),
            psu_col.as_deref(),
            ssu_col.as_deref(),
            fpc_col.as_deref(),
            fpc_ssu_col.as_deref(),
            singleton_method.as_deref(),
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        return Ok(PyDataFrame(result));
    }

    let by_col_name = by_col.unwrap();
    let result = compute_mean_grouped(
        &df,
        &value_col,
        &weight_col,
        strata_col.as_deref(),
        psu_col.as_deref(),
        ssu_col.as_deref(),
        fpc_col.as_deref(),
        fpc_ssu_col.as_deref(),
        &by_col_name,
        singleton_method.as_deref(),
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(PyDataFrame(result))
}

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
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col
        .map(|col| df.column(col).and_then(|s| s.str()))
        .transpose()?;
    let psu = psu_col
        .map(|col| df.column(col).and_then(|s| s.str()))
        .transpose()?;
    let ssu = ssu_col
        .map(|col| df.column(col).and_then(|s| s.str()))
        .transpose()?;
    let fpc = fpc_col
        .map(|col| df.column(col).and_then(|s| s.f64()))
        .transpose()?;
    let fpc_ssu = fpc_ssu_col
        .map(|col| df.column(col).and_then(|s| s.f64()))
        .transpose()?;

    let estimate = point_estimate_mean(y, weights)?;
    let scores = scores_mean(y, weights)?;
    let variance = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
    let se = variance.sqrt();
    let df_val = degrees_of_freedom(weights, strata, psu)?;
    let n = y.len() as u32;
    let srs_var = srs_variance_mean(y, weights)?;
    let deff = if srs_var > 0.0 {
        variance / srs_var
    } else {
        f64::NAN
    };

    df!["y" => vec![value_col], "est" => vec![estimate], "se" => vec![se], "var" => vec![variance], "df" => vec![df_val], "n" => vec![n], "deff" => vec![deff]]
}

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
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col
        .map(|col| df.column(col).and_then(|s| s.str()))
        .transpose()?;
    let psu = psu_col
        .map(|col| df.column(col).and_then(|s| s.str()))
        .transpose()?;
    let ssu = ssu_col
        .map(|col| df.column(col).and_then(|s| s.str()))
        .transpose()?;
    let fpc = fpc_col
        .map(|col| df.column(col).and_then(|s| s.f64()))
        .transpose()?;
    let fpc_ssu = fpc_ssu_col
        .map(|col| df.column(col).and_then(|s| s.f64()))
        .transpose()?;
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
            let variance =
                taylor_variance(&scores, strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
            let se = variance.sqrt();
            let srs_var = srs_variance_mean_domain(y, weights, &domain_mask)?;
            let deff = if srs_var > 0.0 {
                variance / srs_var
            } else {
                f64::NAN
            };

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
#[pyo3(signature = (data, value_col, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, by_col=None, singleton_method=None))]
fn taylor_total(
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
    let df: DataFrame = data.into();
    if by_col.is_none() {
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
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        return Ok(PyDataFrame(result));
    }
    let by_col_name = by_col.unwrap();
    let result = compute_total_grouped(
        &df,
        &value_col,
        &weight_col,
        strata_col.as_deref(),
        psu_col.as_deref(),
        ssu_col.as_deref(),
        fpc_col.as_deref(),
        fpc_ssu_col.as_deref(),
        &by_col_name,
        singleton_method.as_deref(),
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(PyDataFrame(result))
}

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
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col
        .map(|col| df.column(col).and_then(|s| s.str()))
        .transpose()?;
    let psu = psu_col
        .map(|col| df.column(col).and_then(|s| s.str()))
        .transpose()?;
    let ssu = ssu_col
        .map(|col| df.column(col).and_then(|s| s.str()))
        .transpose()?;
    let fpc = fpc_col
        .map(|col| df.column(col).and_then(|s| s.f64()))
        .transpose()?;
    let fpc_ssu = fpc_ssu_col
        .map(|col| df.column(col).and_then(|s| s.f64()))
        .transpose()?;

    let estimate = point_estimate_total(y, weights)?;
    let scores = scores_total(y, weights)?;
    let variance = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
    let se = variance.sqrt();
    let df_val = degrees_of_freedom(weights, strata, psu)?;
    let n = y.len() as u32;
    let srs_var = srs_variance_total(y, weights)?;
    let deff = if srs_var > 0.0 {
        variance / srs_var
    } else {
        f64::NAN
    };

    df!["y" => vec![value_col], "est" => vec![estimate], "se" => vec![se], "var" => vec![variance], "df" => vec![df_val], "n" => vec![n], "deff" => vec![deff]]
}

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
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col
        .map(|col| df.column(col).and_then(|s| s.str()))
        .transpose()?;
    let psu = psu_col
        .map(|col| df.column(col).and_then(|s| s.str()))
        .transpose()?;
    let ssu = ssu_col
        .map(|col| df.column(col).and_then(|s| s.str()))
        .transpose()?;
    let fpc = fpc_col
        .map(|col| df.column(col).and_then(|s| s.f64()))
        .transpose()?;
    let fpc_ssu = fpc_ssu_col
        .map(|col| df.column(col).and_then(|s| s.f64()))
        .transpose()?;
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
            let variance =
                taylor_variance(&scores, strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
            let se = variance.sqrt();
            let srs_var = srs_variance_total_domain(y, weights, &domain_mask)?;
            let deff = if srs_var > 0.0 {
                variance / srs_var
            } else {
                f64::NAN
            };

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
#[pyo3(signature = (data, numerator_col, denominator_col, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, by_col=None, singleton_method=None))]
fn taylor_ratio(
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
    let df: DataFrame = data.into();
    if by_col.is_none() {
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
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        return Ok(PyDataFrame(result));
    }
    let by_col_name = by_col.unwrap();
    let result = compute_ratio_grouped(
        &df,
        &numerator_col,
        &denominator_col,
        &weight_col,
        strata_col.as_deref(),
        psu_col.as_deref(),
        ssu_col.as_deref(),
        fpc_col.as_deref(),
        fpc_ssu_col.as_deref(),
        &by_col_name,
        singleton_method.as_deref(),
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(PyDataFrame(result))
}

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
) -> PolarsResult<DataFrame> {
    let y = df.column(numerator_col)?.f64()?;
    let x = df.column(denominator_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col
        .map(|col| df.column(col).and_then(|s| s.str()))
        .transpose()?;
    let psu = psu_col
        .map(|col| df.column(col).and_then(|s| s.str()))
        .transpose()?;
    let ssu = ssu_col
        .map(|col| df.column(col).and_then(|s| s.str()))
        .transpose()?;
    let fpc = fpc_col
        .map(|col| df.column(col).and_then(|s| s.f64()))
        .transpose()?;
    let fpc_ssu = fpc_ssu_col
        .map(|col| df.column(col).and_then(|s| s.f64()))
        .transpose()?;

    let estimate = point_estimate_ratio(y, x, weights)?;
    let scores = scores_ratio(y, x, weights)?;
    let variance = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
    let se = variance.sqrt();
    let df_val = degrees_of_freedom(weights, strata, psu)?;
    let n = y.len() as u32;
    let srs_var = srs_variance_ratio(y, x, weights)?;
    let deff = if srs_var > 0.0 {
        variance / srs_var
    } else {
        f64::NAN
    };

    df!["y" => vec![numerator_col], "x" => vec![denominator_col], "est" => vec![estimate], "se" => vec![se], "var" => vec![variance], "df" => vec![df_val], "n" => vec![n], "deff" => vec![deff]]
}

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
) -> PolarsResult<DataFrame> {
    let y = df.column(numerator_col)?.f64()?;
    let x = df.column(denominator_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col
        .map(|col| df.column(col).and_then(|s| s.str()))
        .transpose()?;
    let psu = psu_col
        .map(|col| df.column(col).and_then(|s| s.str()))
        .transpose()?;
    let ssu = ssu_col
        .map(|col| df.column(col).and_then(|s| s.str()))
        .transpose()?;
    let fpc = fpc_col
        .map(|col| df.column(col).and_then(|s| s.f64()))
        .transpose()?;
    let fpc_ssu = fpc_ssu_col
        .map(|col| df.column(col).and_then(|s| s.f64()))
        .transpose()?;
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
            let variance =
                taylor_variance(&scores, strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
            let se = variance.sqrt();
            let srs_var = srs_variance_ratio_domain(y, x, weights, &domain_mask)?;
            let deff = if srs_var > 0.0 {
                variance / srs_var
            } else {
                f64::NAN
            };

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
#[pyo3(signature = (data, value_col, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, by_col=None, singleton_method=None))]
fn taylor_prop(
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
    let df: DataFrame = data.into();
    if by_col.is_none() {
        let result = compute_prop_ungrouped(
            &df,
            &value_col,
            &weight_col,
            strata_col.as_deref(),
            psu_col.as_deref(),
            ssu_col.as_deref(),
            fpc_col.as_deref(),
            fpc_ssu_col.as_deref(),
            singleton_method.as_deref(),
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        return Ok(PyDataFrame(result));
    }
    let by_col_name = by_col.unwrap();
    let result = compute_prop_grouped(
        &df,
        &value_col,
        &weight_col,
        strata_col.as_deref(),
        psu_col.as_deref(),
        ssu_col.as_deref(),
        fpc_col.as_deref(),
        fpc_ssu_col.as_deref(),
        &by_col_name,
        singleton_method.as_deref(),
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(PyDataFrame(result))
}

fn compute_prop_ungrouped(
    df: &DataFrame,
    value_col: &str,
    weight_col: &str,
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>,
) -> PolarsResult<DataFrame> {
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col
        .map(|col| df.column(col).and_then(|s| s.str()))
        .transpose()?;
    let psu = psu_col
        .map(|col| df.column(col).and_then(|s| s.str()))
        .transpose()?;
    let ssu = ssu_col
        .map(|col| df.column(col).and_then(|s| s.str()))
        .transpose()?;
    let fpc = fpc_col
        .map(|col| df.column(col).and_then(|s| s.f64()))
        .transpose()?;
    let fpc_ssu = fpc_ssu_col
        .map(|col| df.column(col).and_then(|s| s.f64()))
        .transpose()?;

    let value_series = df.column(value_col)?;
    let value_str = value_series.cast(&DataType::String)?;
    let value_str = value_str.str()?;
    let unique_levels = value_str.unique()?;
    let mut levels: Vec<String> = unique_levels
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
    let df_val = degrees_of_freedom(weights, strata, psu)?;
    let n = weights.len() as u32;

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
        let estimate = point_estimate_mean(&indicator_ca, weights)?;
        let scores = scores_mean(&indicator_ca, weights)?;
        let variance = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
        let se = variance.sqrt();
        let srs_var = srs_variance_mean(&indicator_ca, weights)?;
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
    }
    let n_levels = level_vals.len();
    df!["y" => vec![value_col; n_levels], "level" => level_vals, "est" => estimates, "se" => ses, "var" => variances, "df" => dfs_vec, "n" => ns, "deff" => deffs]
}

fn compute_prop_grouped(
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
) -> PolarsResult<DataFrame> {
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col
        .map(|col| df.column(col).and_then(|s| s.str()))
        .transpose()?;
    let psu = psu_col
        .map(|col| df.column(col).and_then(|s| s.str()))
        .transpose()?;
    let ssu = ssu_col
        .map(|col| df.column(col).and_then(|s| s.str()))
        .transpose()?;
    let fpc = fpc_col
        .map(|col| df.column(col).and_then(|s| s.f64()))
        .transpose()?;
    let fpc_ssu = fpc_ssu_col
        .map(|col| df.column(col).and_then(|s| s.f64()))
        .transpose()?;

    let value_series = df.column(value_col)?;
    let value_str = value_series.cast(&DataType::String)?;
    let value_str = value_str.str()?;
    let unique_levels = value_str.unique()?;
    let mut levels: Vec<String> = unique_levels
        .iter()
        .filter_map(|v| v.map(|s| s.to_string()))
        .collect();
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
                let estimate = point_estimate_mean_domain(&indicator_ca, weights, &domain_mask)?;
                let scores = scores_mean_domain(&indicator_ca, weights, &domain_mask)?;
                let variance =
                    taylor_variance(&scores, strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
                let se = variance.sqrt();
                let srs_var = srs_variance_mean_domain(&indicator_ca, weights, &domain_mask)?;
                let deff = if srs_var > 0.0 {
                    variance / srs_var
                } else {
                    f64::NAN
                };

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

    let rep_method = RepMethod::from_str(&method).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown method: {}. Use 'BRR', 'Bootstrap', 'Jackknife', or 'SDR'",
            method
        ))
    })?;

    let variance_center = VarianceCenter::from_str(&center).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown center: {}. Use 'replicate_mean' or 'full_sample'",
            center
        ))
    })?;

    let df_val = degrees_of_freedom.unwrap_or(n_reps.saturating_sub(1) as u32);

    let result = if by_col.is_none() {
        compute_replicate_mean_ungrouped(
            &df,
            &value_col,
            &weight_col,
            &rep_weight_cols,
            rep_method,
            fay_coef,
            variance_center,
            df_val,
        )
    } else {
        compute_replicate_mean_grouped(
            &df,
            &value_col,
            &weight_col,
            &rep_weight_cols,
            rep_method,
            fay_coef,
            variance_center,
            df_val,
            by_col.as_ref().unwrap(),
        )
    };

    result
        .map(PyDataFrame)
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
        &y_arr,
        &w_arr,
        &rep_w_matrix,
        &domain_ids,
        n_domains,
        n,
        n_reps,
    );

    let rep_coefs = replicate_coefficients(method, n_reps, fay_coef);

    let mut by_vals: Vec<String> = Vec::with_capacity(n_domains);
    let mut estimates: Vec<f64> = Vec::with_capacity(n_domains);
    let mut ses: Vec<f64> = Vec::with_capacity(n_domains);
    let mut variances: Vec<f64> = Vec::with_capacity(n_domains);
    let mut dfs: Vec<u32> = Vec::with_capacity(n_domains);
    let mut ns: Vec<u32> = Vec::with_capacity(n_domains);

    for (k, domain_name) in domain_names.iter().enumerate() {
        let variance = variance_from_replicates(
            method,
            theta_full_vec[k],
            &theta_reps_vec[k],
            &rep_coefs,
            center,
        );

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

    let rep_method = RepMethod::from_str(&method).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown method: {}. Use 'BRR', 'Bootstrap', 'Jackknife', or 'SDR'",
            method
        ))
    })?;

    let variance_center = VarianceCenter::from_str(&center).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown center: {}. Use 'replicate_mean' or 'full_sample'",
            center
        ))
    })?;

    let df_val = degrees_of_freedom.unwrap_or(n_reps.saturating_sub(1) as u32);

    let result = if by_col.is_none() {
        compute_replicate_total_ungrouped(
            &df,
            &value_col,
            &weight_col,
            &rep_weight_cols,
            rep_method,
            fay_coef,
            variance_center,
            df_val,
        )
    } else {
        compute_replicate_total_grouped(
            &df,
            &value_col,
            &weight_col,
            &rep_weight_cols,
            rep_method,
            fay_coef,
            variance_center,
            df_val,
            by_col.as_ref().unwrap(),
        )
    };

    result
        .map(PyDataFrame)
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
        &y_arr,
        &w_arr,
        &rep_w_matrix,
        &domain_ids,
        n_domains,
        n,
        n_reps,
    );

    let rep_coefs = replicate_coefficients(method, n_reps, fay_coef);

    let mut by_vals: Vec<String> = Vec::with_capacity(n_domains);
    let mut estimates: Vec<f64> = Vec::with_capacity(n_domains);
    let mut ses: Vec<f64> = Vec::with_capacity(n_domains);
    let mut variances: Vec<f64> = Vec::with_capacity(n_domains);
    let mut dfs: Vec<u32> = Vec::with_capacity(n_domains);
    let mut ns: Vec<u32> = Vec::with_capacity(n_domains);

    for (k, domain_name) in domain_names.iter().enumerate() {
        let variance = variance_from_replicates(
            method,
            theta_full_vec[k],
            &theta_reps_vec[k],
            &rep_coefs,
            center,
        );

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

    let rep_method = RepMethod::from_str(&method).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown method: {}. Use 'BRR', 'Bootstrap', 'Jackknife', or 'SDR'",
            method
        ))
    })?;

    let variance_center = VarianceCenter::from_str(&center).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown center: {}. Use 'replicate_mean' or 'full_sample'",
            center
        ))
    })?;

    let df_val = degrees_of_freedom.unwrap_or(n_reps.saturating_sub(1) as u32);

    let result = if by_col.is_none() {
        compute_replicate_ratio_ungrouped(
            &df,
            &numerator_col,
            &denominator_col,
            &weight_col,
            &rep_weight_cols,
            rep_method,
            fay_coef,
            variance_center,
            df_val,
        )
    } else {
        compute_replicate_ratio_grouped(
            &df,
            &numerator_col,
            &denominator_col,
            &weight_col,
            &rep_weight_cols,
            rep_method,
            fay_coef,
            variance_center,
            df_val,
            by_col.as_ref().unwrap(),
        )
    };

    result
        .map(PyDataFrame)
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
    let (theta_full, theta_reps) =
        matrix_ratio_estimates(&y_arr, &x_arr, &w_arr, &rep_w_matrix, n, n_reps);

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
        &y_arr,
        &x_arr,
        &w_arr,
        &rep_w_matrix,
        &domain_ids,
        n_domains,
        n,
        n_reps,
    );

    let rep_coefs = replicate_coefficients(method, n_reps, fay_coef);

    let mut by_vals: Vec<String> = Vec::with_capacity(n_domains);
    let mut estimates: Vec<f64> = Vec::with_capacity(n_domains);
    let mut ses: Vec<f64> = Vec::with_capacity(n_domains);
    let mut variances: Vec<f64> = Vec::with_capacity(n_domains);
    let mut dfs: Vec<u32> = Vec::with_capacity(n_domains);
    let mut ns: Vec<u32> = Vec::with_capacity(n_domains);

    for (k, domain_name) in domain_names.iter().enumerate() {
        let variance = variance_from_replicates(
            method,
            theta_full_vec[k],
            &theta_reps_vec[k],
            &rep_coefs,
            center,
        );

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

    let rep_method = RepMethod::from_str(&method).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown method: {}. Use 'BRR', 'Bootstrap', 'Jackknife', or 'SDR'",
            method
        ))
    })?;

    let variance_center = VarianceCenter::from_str(&center).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown center: {}. Use 'replicate_mean' or 'full_sample'",
            center
        ))
    })?;

    let df_val = degrees_of_freedom.unwrap_or(n_reps.saturating_sub(1) as u32);

    let result = if by_col.is_none() {
        compute_replicate_prop_ungrouped(
            &df,
            &value_col,
            &weight_col,
            &rep_weight_cols,
            rep_method,
            fay_coef,
            variance_center,
            df_val,
        )
    } else {
        compute_replicate_prop_grouped(
            &df,
            &value_col,
            &weight_col,
            &rep_weight_cols,
            rep_method,
            fay_coef,
            variance_center,
            df_val,
            by_col.as_ref().unwrap(),
        )
    };

    result
        .map(PyDataFrame)
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
        y_series
            .i64()?
            .into_iter()
            .map(|v| v.unwrap_or(0))
            .collect()
    } else if y_series.dtype() == &DataType::Boolean {
        y_series
            .bool()?
            .into_iter()
            .map(|v| if v.unwrap_or(false) { 1 } else { 0 })
            .collect()
    } else {
        // Try to cast to i64
        y_series
            .cast(&DataType::Int64)?
            .i64()?
            .into_iter()
            .map(|v| v.unwrap_or(0))
            .collect()
    };

    let w_arr: Vec<f64> = weights.into_iter().map(|v| v.unwrap_or(0.0)).collect();

    let (rep_w_matrix, _, _) = extract_rep_weights_matrix(df, rep_weight_cols)?;
    let (levels, theta_full_vec, theta_reps_vec) =
        matrix_prop_estimates(&y_arr, &w_arr, &rep_w_matrix, n, n_reps);

    let rep_coefs = replicate_coefficients(method, n_reps, fay_coef);
    let n_levels = levels.len();

    let mut level_strs: Vec<String> = Vec::with_capacity(n_levels);
    let mut estimates: Vec<f64> = Vec::with_capacity(n_levels);
    let mut ses: Vec<f64> = Vec::with_capacity(n_levels);
    let mut variances: Vec<f64> = Vec::with_capacity(n_levels);
    let mut dfs: Vec<u32> = Vec::with_capacity(n_levels);
    let mut ns: Vec<u32> = Vec::with_capacity(n_levels);

    for (l, &level) in levels.iter().enumerate() {
        let variance = variance_from_replicates(
            method,
            theta_full_vec[l],
            &theta_reps_vec[l],
            &rep_coefs,
            center,
        );

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
        y_series
            .i64()?
            .into_iter()
            .map(|v| v.unwrap_or(0))
            .collect()
    } else if y_series.dtype() == &DataType::Boolean {
        y_series
            .bool()?
            .into_iter()
            .map(|v| if v.unwrap_or(false) { 1 } else { 0 })
            .collect()
    } else {
        y_series
            .cast(&DataType::Int64)?
            .i64()?
            .into_iter()
            .map(|v| v.unwrap_or(0))
            .collect()
    };

    let w_arr: Vec<f64> = weights.into_iter().map(|v| v.unwrap_or(0.0)).collect();

    let (domain_ids, domain_names, n_domains) = index_domains(by_str);
    let (rep_w_matrix, _, _) = extract_rep_weights_matrix(df, rep_weight_cols)?;

    let (levels, theta_full_mat, theta_reps_mat, counts) = matrix_prop_by_domain(
        &y_arr,
        &w_arr,
        &rep_w_matrix,
        &domain_ids,
        n_domains,
        n,
        n_reps,
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
            let variance =
                variance_from_replicates(method, theta_full, theta_reps, &rep_coefs, center);

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
    let y = df
        .column(&y_name)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
        .as_materialized_series()
        .clone();

    let mut x_cols = Vec::with_capacity(x_names.len());
    for name in &x_names {
        let s = df
            .column(name)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
            .as_materialized_series()
            .clone();
        x_cols.push(s);
    }

    let weights = df
        .column(&weight_name)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
        .as_materialized_series()
        .clone();

    let stratum = if let Some(s) = stratum_name {
        Some(
            df.column(&s)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
                .as_materialized_series()
                .clone(),
        )
    } else {
        None
    };

    let psu = if let Some(p) = psu_name {
        Some(
            df.column(&p)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
                .as_materialized_series()
                .clone(),
        )
    } else {
        None
    };

    let result = fit_glm(
        &y,
        x_cols,
        &weights,
        stratum.as_ref(),
        psu.as_ref(),
        &family,
        &link,
        tol,
        max_iter,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok((
        result.params,
        result.cov_params,
        result.scale,
        result.df_resid,
        result.deviance,
        result.null_deviance,
        result.iterations,
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

    let indices: Vec<_> = margin_indices
        .iter()
        .map(|x| x.as_array().to_owned())
        .collect();

    let targets: Vec<_> = margin_targets
        .iter()
        .map(|x| x.as_array().to_owned())
        .collect();

    let result = weighting::raking::rake_impl(
        wgt_arr, &indices, &targets, ll_bound, up_bound, tol, max_iter,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

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

    let result =
        weighting::nonresponse::adjust_nr_impl(wgts_arr, class_arr, status_arr, unknown_to_inelig)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

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

    let result = weighting::normalization::normalize_impl(wgt_arr, by.as_ref(), ctrl.as_ref())
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

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

    let result = weighting::poststratification::poststratify_impl(wgt_arr, by, &ctrl)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

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

    let result = weighting::poststratification::poststratify_factor(wgt_arr, by, &fct)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(result.into_pyarray(py).to_owned().into())
}

// ============================================================================
// Calibration Functions
// ============================================================================

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

    let result =
        weighting::calibration::calibrate_linear(wgt_arr, x_arr, totals_arr, scale_view, additive)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(result.into_pyarray(py).to_owned().into())
}

/// Calibration by domain (group-specific calibration)
///

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
        wgt_arr, x_arr, domain_arr, &controls, scale_view, additive,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

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

    let result = weighting::calibration::calibrate_parallel(wgt_arr, x_arr, totals_arr, scale_view)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok(result.into_pyarray(py).to_owned().into())
}

// ============================================================================
// Replicate Weight Creation Functions (add to lib.rs)
// ============================================================================

/// Create BRR (Balanced Repeated Replication) weights
///
/// # Arguments
/// * `wgt` - Base weights (n_obs,)
/// * `stratum` - Stratum identifiers (n_obs,)
/// * `psu` - PSU identifiers (n_obs,)
/// * `n_reps` - Number of replicates (optional, defaults to min needed)
/// * `fay_coef` - Fay adjustment (0.0 = standard BRR)
/// * `seed` - Random seed for PSU ordering within strata (optional)
///
/// # Returns
/// Tuple of (replicate_weights, degrees_of_freedom)
#[pyfunction]
#[pyo3(signature = (wgt, stratum, psu, n_reps=None, fay_coef=0.0, seed=None))]
fn create_brr_wgts(
    py: Python<'_>,
    wgt: PyReadonlyArray1<f64>,
    stratum: PyReadonlyArray1<i64>,
    psu: PyReadonlyArray1<i64>,
    n_reps: Option<usize>,
    fay_coef: f64,
    seed: Option<u64>,
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
        seed,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok((result.into_pyarray(py).to_owned().into(), df))
}

/// Create Jackknife replicate weights (unified interface)
///
/// # Arguments
/// * `wgt` - Base weights (n_obs,)
/// * `psu` - PSU identifiers (n_obs,)
/// * `stratum` - Stratum identifiers (n_obs,), optional
/// * `paired` - If true, use JK2 (paired); if false, use JK1/JKn
/// * `seed` - Random seed for JK2 PSU selection (optional, ignored for JK1/JKn)
///
/// # Returns
/// Tuple of (replicate_weights, degrees_of_freedom)
#[pyfunction]
#[pyo3(signature = (wgt, psu, stratum=None, paired=false, seed=None))]
fn create_jk_wgts(
    py: Python<'_>,
    wgt: PyReadonlyArray1<f64>,
    psu: PyReadonlyArray1<i64>,
    stratum: Option<PyReadonlyArray1<i64>>,
    paired: bool,
    seed: Option<u64>,
) -> PyResult<(Py<PyArray2<f64>>, f64)> {
    let wgt_arr = wgt.as_array();
    let psu_arr = psu.as_array();
    let stratum_view = stratum.as_ref().map(|s| s.as_array());

    let (result, df) =
        weighting::replication::create_jk_weights(wgt_arr, stratum_view, psu_arr, paired, seed)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

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
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

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

    let (result, df) =
        weighting::replication::create_sdr_weights(wgt_arr, stratum_view, order_view, n_reps)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    Ok((result.into_pyarray(py).to_owned().into(), df))
}

// ============================================================================
// Median Estimation Functions (FIXED VERSION)
// ============================================================================

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, by_col=None, singleton_method=None, quantile_method=None))]
fn taylor_median(
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
    let df: DataFrame = data.into();

    // Use SvyQuantileMethod (our own type, not polars::QuantileMethod)
    let q_method = quantile_method
        .as_deref()
        .map(SvyQuantileMethod::from_str)
        .unwrap_or(SvyQuantileMethod::Higher);

    if by_col.is_none() {
        let result = compute_median_ungrouped(
            &df,
            &value_col,
            &weight_col,
            strata_col.as_deref(),
            psu_col.as_deref(),
            ssu_col.as_deref(),
            fpc_col.as_deref(),
            fpc_ssu_col.as_deref(),
            singleton_method.as_deref(),
            q_method,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        return Ok(PyDataFrame(result));
    }

    let by_col_name = by_col.unwrap();
    let result = compute_median_grouped(
        &df,
        &value_col,
        &weight_col,
        strata_col.as_deref(),
        psu_col.as_deref(),
        ssu_col.as_deref(),
        fpc_col.as_deref(),
        fpc_ssu_col.as_deref(),
        &by_col_name,
        singleton_method.as_deref(),
        q_method,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(PyDataFrame(result))
}

fn compute_median_ungrouped(
    df: &DataFrame,
    value_col: &str,
    weight_col: &str,
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>,
    q_method: SvyQuantileMethod,
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col
        .map(|col| df.column(col).and_then(|s| s.str()))
        .transpose()?;
    let psu = psu_col
        .map(|col| df.column(col).and_then(|s| s.str()))
        .transpose()?;
    let ssu = ssu_col
        .map(|col| df.column(col).and_then(|s| s.str()))
        .transpose()?;
    let fpc = fpc_col
        .map(|col| df.column(col).and_then(|s| s.f64()))
        .transpose()?;
    let fpc_ssu = fpc_ssu_col
        .map(|col| df.column(col).and_then(|s| s.f64()))
        .transpose()?;

    // Compute weighted median
    let estimate = weighted_median(y, weights, q_method)?;

    // Compute variance using Woodruff method
    let (var_p, se_p) = median_variance_woodruff(
        y,
        weights,
        strata,
        psu,
        ssu,
        fpc,
        fpc_ssu,
        singleton_method,
        q_method,
    )?;

    let df_val = degrees_of_freedom(weights, strata, psu)?;
    let n = y.len() as u32;

    // Return DataFrame with variance on proportion scale (for CI calculation in Python)
    df![
        "y" => vec![value_col],
        "est" => vec![estimate],
        "se" => vec![se_p],      // SE of proportion (for Woodruff CI)
        "var" => vec![var_p],    // Variance of proportion
        "df" => vec![df_val],
        "n" => vec![n]
    ]
}

fn compute_median_grouped(
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
    q_method: SvyQuantileMethod,
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col
        .map(|col| df.column(col).and_then(|s| s.str()))
        .transpose()?;
    let psu = psu_col
        .map(|col| df.column(col).and_then(|s| s.str()))
        .transpose()?;
    let ssu = ssu_col
        .map(|col| df.column(col).and_then(|s| s.str()))
        .transpose()?;
    let fpc = fpc_col
        .map(|col| df.column(col).and_then(|s| s.f64()))
        .transpose()?;
    let fpc_ssu = fpc_ssu_col
        .map(|col| df.column(col).and_then(|s| s.f64()))
        .transpose()?;
    let by_series = df.column(by_col)?;
    let by_str = by_series.str()?;
    let unique_groups = by_str.unique()?;

    let mut by_vals: Vec<&str> = Vec::new();
    let mut estimates: Vec<f64> = Vec::new();
    let mut ses: Vec<f64> = Vec::new();
    let mut variances: Vec<f64> = Vec::new();
    let mut dfs: Vec<u32> = Vec::new();
    let mut ns: Vec<u32> = Vec::new();

    // Degrees of freedom is the same for all domains (proper domain estimation)
    let df_val = degrees_of_freedom(weights, strata, psu)?;

    for group_val in unique_groups.iter() {
        if let Some(group) = group_val {
            let domain_mask = by_str.equal(group);
            let n_domain = domain_mask.sum().unwrap_or(0) as u32;

            // Compute weighted median for this domain
            let estimate = weighted_median_domain(y, weights, &domain_mask, q_method)?;

            // Compute variance using Woodruff method with domain scores
            let (var_p, se_p) = median_variance_woodruff_domain(
                y,
                weights,
                &domain_mask,
                strata,
                psu,
                ssu,
                fpc,
                fpc_ssu,
                singleton_method,
                q_method,
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
    df![
        by_col => by_vals,
        "y" => vec![value_col; n_groups],
        "est" => estimates,
        "se" => ses,
        "var" => variances,
        "df" => dfs,
        "n" => ns
    ]
}

// ============================================================================
// Replicate-Based Median Estimation (FIXED VERSION)
// ============================================================================

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, rep_weight_cols, method, fay_coef=0.0, center="rep_mean", degrees_of_freedom=None, by_col=None, quantile_method=None))]
fn replicate_median(
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
    quantile_method: Option<String>,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let n_reps = rep_weight_cols.len();

    let rep_method = RepMethod::from_str(&method).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown method: {}. Use 'BRR', 'Bootstrap', 'Jackknife', or 'SDR'",
            method
        ))
    })?;

    let variance_center = VarianceCenter::from_str(&center).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown center: {}. Use 'rep_mean' or 'full_sample'",
            center
        ))
    })?;

    // Use SvyQuantileMethod (our own type)
    let q_method = quantile_method
        .as_deref()
        .map(SvyQuantileMethod::from_str)
        .unwrap_or(SvyQuantileMethod::Higher);

    let df_val = degrees_of_freedom.unwrap_or(n_reps.saturating_sub(1) as u32);

    let result = if by_col.is_none() {
        compute_replicate_median_ungrouped(
            &df,
            &value_col,
            &weight_col,
            &rep_weight_cols,
            rep_method,
            fay_coef,
            variance_center,
            df_val,
            q_method,
        )
    } else {
        compute_replicate_median_grouped(
            &df,
            &value_col,
            &weight_col,
            &rep_weight_cols,
            rep_method,
            fay_coef,
            variance_center,
            df_val,
            by_col.as_ref().unwrap(),
            q_method,
        )
    };

    result
        .map(PyDataFrame)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

fn compute_replicate_median_ungrouped(
    df: &DataFrame,
    value_col: &str,
    weight_col: &str,
    rep_weight_cols: &[String],
    method: RepMethod,
    fay_coef: f64,
    center: VarianceCenter,
    df_val: u32,
    q_method: SvyQuantileMethod,
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let n = y.len();
    let n_reps = rep_weight_cols.len();

    let y_arr: Vec<f64> = y.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect();
    let w_arr: Vec<f64> = weights.into_iter().map(|v| v.unwrap_or(0.0)).collect();

    let (rep_w_matrix, _, _) = extract_rep_weights_matrix(df, rep_weight_cols)?;
    let (theta_full, theta_reps) =
        matrix_median_estimates(&y_arr, &w_arr, &rep_w_matrix, n, n_reps, q_method);

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

fn compute_replicate_median_grouped(
    df: &DataFrame,
    value_col: &str,
    weight_col: &str,
    rep_weight_cols: &[String],
    method: RepMethod,
    fay_coef: f64,
    center: VarianceCenter,
    df_val: u32,
    by_col: &str,
    q_method: SvyQuantileMethod,
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let by_series = df.column(by_col)?;
    let by_str = by_series.str()?;
    let n = y.len();
    let n_reps = rep_weight_cols.len();

    let y_arr: Vec<f64> = y.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect();
    let w_arr: Vec<f64> = weights.into_iter().map(|v| v.unwrap_or(0.0)).collect();

    let (domain_ids, domain_names, n_domains) = index_domains(by_str);
    let (rep_w_matrix, _, _) = extract_rep_weights_matrix(df, rep_weight_cols)?;

    let (theta_full_vec, theta_reps_vec, counts) = matrix_median_by_domain(
        &y_arr,
        &w_arr,
        &rep_w_matrix,
        &domain_ids,
        n_domains,
        n,
        n_reps,
        q_method,
    );

    let rep_coefs = replicate_coefficients(method, n_reps, fay_coef);

    let mut by_vals: Vec<String> = Vec::with_capacity(n_domains);
    let mut estimates: Vec<f64> = Vec::with_capacity(n_domains);
    let mut ses: Vec<f64> = Vec::with_capacity(n_domains);
    let mut variances: Vec<f64> = Vec::with_capacity(n_domains);
    let mut dfs: Vec<u32> = Vec::with_capacity(n_domains);
    let mut ns: Vec<u32> = Vec::with_capacity(n_domains);

    for (k, domain_name) in domain_names.iter().enumerate() {
        let variance = variance_from_replicates(
            method,
            theta_full_vec[k],
            &theta_reps_vec[k],
            &rep_coefs,
            center,
        );

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

// ============================================================================
// Categorical Test Helpers
// ============================================================================

/// Helper: extract optional StringChunked from DataFrame column.
fn get_opt_str<'a>(
    df: &'a DataFrame,
    col: Option<&str>,
) -> PolarsResult<Option<&'a StringChunked>> {
    match col {
        Some(c) => Ok(Some(df.column(c)?.as_materialized_series().str()?)),
        None => Ok(None),
    }
}

/// Helper: extract optional Float64Chunked from DataFrame column.
fn get_opt_f64<'a>(
    df: &'a DataFrame,
    col: Option<&str>,
) -> PolarsResult<Option<&'a Float64Chunked>> {
    match col {
        Some(c) => Ok(Some(df.column(c)?.as_materialized_series().f64()?)),
        None => Ok(None),
    }
}

/// Prepare data arrays for two-sample / k-sample tests.
///
/// For domain estimation: zeros weights for non-domain obs but keeps all rows
/// so strata/PSU structure is preserved for correct variance estimation.
fn prepare_two_sample_data(
    df: &DataFrame,
    y_col: &str,
    w_col: &str,
    g_col: &str,
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    domain_col: Option<&str>,
    domain_val: Option<&str>,
) -> PolarsResult<(
    Vec<f64>,    // y
    Vec<f64>,    // w (zeroed for non-domain/NA)
    Vec<u32>,    // g (0-based group indices)
    Vec<String>, // sorted unique group labels
    usize,       // n
)> {
    let n = df.height();
    let y_ca = df.column(y_col)?.as_materialized_series().f64()?;
    let w_ca = df.column(w_col)?.as_materialized_series().f64()?;
    let g_series = df.column(g_col)?.as_materialized_series();
    let g_str = g_series.cast(&DataType::String)?;
    let g_ca = g_str.str()?;

    // Build domain mask
    let domain_mask: Option<Vec<bool>> = match (domain_col, domain_val) {
        (Some(d_col), Some(d_val)) => {
            let d_series = df.column(d_col)?.as_materialized_series();
            let d_str = d_series.cast(&DataType::String)?;
            let d_ca = d_str.str()?;
            Some(
                d_ca.iter()
                    .map(|opt| opt.map_or(false, |v| v == d_val))
                    .collect(),
            )
        }
        _ => None,
    };

    // Collect unique group labels (sorted), considering domain
    let mut level_set: Vec<String> = Vec::new();
    {
        let mut seen = std::collections::HashSet::new();
        for i in 0..n {
            let in_domain = domain_mask.as_ref().map_or(true, |m| m[i]);
            let w_val = w_ca.get(i).unwrap_or(0.0);
            if in_domain && w_val > 0.0 {
                if let Some(label) = g_ca.get(i) {
                    if seen.insert(label.to_string()) {
                        level_set.push(label.to_string());
                    }
                }
            }
        }
    }
    level_set.sort();

    let mut label_to_idx: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    for (i, label) in level_set.iter().enumerate() {
        label_to_idx.insert(label.clone(), i as u32);
    }

    // Build arrays: zero weights for non-domain/NA, keep all rows
    let mut y_arr = Vec::with_capacity(n);
    let mut w_arr = Vec::with_capacity(n);
    let mut g_arr = Vec::with_capacity(n);

    for i in 0..n {
        let yi = y_ca.get(i).unwrap_or(f64::NAN);
        let wi = w_ca.get(i).unwrap_or(0.0);
        let in_domain = domain_mask.as_ref().map_or(true, |m| m[i]);
        let label = g_ca.get(i).unwrap_or("__NULL__");
        let gi = label_to_idx.get(label).copied().unwrap_or(0);

        if yi.is_nan() || !in_domain {
            y_arr.push(0.0);
            w_arr.push(0.0);
            g_arr.push(0);
        } else {
            y_arr.push(yi);
            w_arr.push(wi);
            g_arr.push(gi);
        }
    }

    Ok((y_arr, w_arr, g_arr, level_set, n))
}

// ============================================================================
// T-Test Functions
// ============================================================================

#[pyfunction]
#[pyo3(signature = (
    data, y_col, weight_col, group_col=None,
    strata_col=None, psu_col=None, ssu_col=None,
    fpc_col=None, fpc_ssu_col=None, singleton_method=None,
    null_value=0.0, domain_col=None, domain_val=None,
))]
fn ttest_rs(
    _py: Python,
    data: PyDataFrame,
    y_col: String,
    weight_col: String,
    group_col: Option<String>,
    strata_col: Option<String>,
    psu_col: Option<String>,
    ssu_col: Option<String>,
    fpc_col: Option<String>,
    fpc_ssu_col: Option<String>,
    singleton_method: Option<String>,
    null_value: f64,
    domain_col: Option<String>,
    domain_val: Option<String>,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let result = compute_svyttest(
        &df,
        &y_col,
        &weight_col,
        group_col.as_deref(),
        strata_col.as_deref(),
        psu_col.as_deref(),
        ssu_col.as_deref(),
        fpc_col.as_deref(),
        fpc_ssu_col.as_deref(),
        singleton_method.as_deref(),
        null_value,
        domain_col.as_deref(),
        domain_val.as_deref(),
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(PyDataFrame(result))
}

fn compute_svyttest(
    df: &DataFrame,
    y_col: &str,
    weight_col: &str,
    group_col: Option<&str>,
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>,
    null_value: f64,
    domain_col: Option<&str>,
    domain_val: Option<&str>,
) -> PolarsResult<DataFrame> {
    let strata = get_opt_str(df, strata_col)?;
    let psu = get_opt_str(df, psu_col)?;
    let ssu = get_opt_str(df, ssu_col)?;
    let fpc = get_opt_f64(df, fpc_col)?;
    let fpc_ssu = get_opt_f64(df, fpc_ssu_col)?;

    match group_col {
        None => {
            // One-sample test
            let y = df.column(y_col)?.as_materialized_series().f64()?;
            let weights = df.column(weight_col)?.as_materialized_series().f64()?;

            let res = if let (Some(d_col), Some(d_val)) = (domain_col, domain_val) {
                let d_series = df.column(d_col)?.as_materialized_series();
                let d_str = d_series.cast(&DataType::String)?;
                let d_ca = d_str.str()?;
                let mask = d_ca.equal(d_val);
                ttest_one_sample_domain(
                    y,
                    weights,
                    &mask,
                    strata,
                    psu,
                    ssu,
                    fpc,
                    fpc_ssu,
                    singleton_method,
                    null_value,
                )?
            } else {
                ttest_one_sample(
                    y,
                    weights,
                    strata,
                    psu,
                    ssu,
                    fpc,
                    fpc_ssu,
                    singleton_method,
                    null_value,
                )?
            };

            df![
                "type" => vec!["one-sample"],
                "y" => vec![y_col],
                "estimate" => vec![res.estimate],
                "diff" => vec![res.diff],
                "se" => vec![res.se_diff],
                "t" => vec![res.t_stat],
                "df" => vec![res.df],
                "p_value" => vec![res.p_value],
                "n" => vec![res.n_obs as u32],
            ]
        }
        Some(g_col) => {
            // Two-sample test
            let (y_arr, w_arr, g_arr, levels, n) = prepare_two_sample_data(
                df, y_col, weight_col, g_col, strata_col, psu_col, domain_col, domain_val,
            )?;

            if levels.len() != 2 {
                return Err(PolarsError::ComputeError(
                    format!(
                        "Two-sample t-test requires exactly 2 groups, found {}",
                        levels.len()
                    )
                    .into(),
                ));
            }

            let res = ttest_two_sample(
                &y_arr,
                &g_arr,
                &w_arr,
                n,
                strata,
                psu,
                ssu,
                fpc,
                fpc_ssu,
                singleton_method,
                levels.clone(),
                null_value,
            )?;

            df![
                "type" => vec!["two-sample"],
                "y" => vec![y_col],
                "group" => vec![g_col],
                "level_0" => vec![levels[0].as_str()],
                "level_1" => vec![levels[1].as_str()],
                "diff" => vec![res.diff],
                "se" => vec![res.se_diff],
                "t" => vec![res.t_stat],
                "df" => vec![res.df],
                "p_value" => vec![res.p_value],
                "mean_0" => vec![res.group_means[0]],
                "mean_1" => vec![res.group_means[1]],
                "se_0" => vec![res.group_ses[0]],
                "se_1" => vec![res.group_ses[1]],
                "n" => vec![res.n_obs as u32],
            ]
        }
    }
}

// ============================================================================
// Rank Test Functions
// ============================================================================

#[pyfunction]
#[pyo3(signature = (
    data, y_col, group_col, weight_col,
    strata_col=None, psu_col=None, ssu_col=None,
    fpc_col=None, fpc_ssu_col=None, score_method=None,
    singleton_method=None, domain_col=None, domain_val=None,
))]
fn ranktest_rs(
    _py: Python,
    data: PyDataFrame,
    y_col: String,
    group_col: String,
    weight_col: String,
    strata_col: Option<String>,
    psu_col: Option<String>,
    ssu_col: Option<String>,
    fpc_col: Option<String>,
    fpc_ssu_col: Option<String>,
    score_method: Option<String>,
    singleton_method: Option<String>,
    domain_col: Option<String>,
    domain_val: Option<String>,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let result = compute_svyranktest(
        &df,
        &y_col,
        &group_col,
        &weight_col,
        strata_col.as_deref(),
        psu_col.as_deref(),
        ssu_col.as_deref(),
        fpc_col.as_deref(),
        fpc_ssu_col.as_deref(),
        score_method.as_deref(),
        singleton_method.as_deref(),
        domain_col.as_deref(),
        domain_val.as_deref(),
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(PyDataFrame(result))
}

fn compute_svyranktest(
    df: &DataFrame,
    y_col: &str,
    group_col: &str,
    weight_col: &str,
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_ssu_col: Option<&str>,
    score_method: Option<&str>,
    singleton_method: Option<&str>,
    domain_col: Option<&str>,
    domain_val: Option<&str>,
) -> PolarsResult<DataFrame> {
    let method = score_method
        .and_then(RankScoreMethod::from_str)
        .unwrap_or(RankScoreMethod::Wilcoxon);

    let method_name = match method {
        RankScoreMethod::Wilcoxon | RankScoreMethod::KruskalWallis => "KruskalWallis",
        RankScoreMethod::VanDerWaerden => "vanderWaerden",
        RankScoreMethod::Median => "median",
    };

    let strata = get_opt_str(df, strata_col)?;
    let psu = get_opt_str(df, psu_col)?;
    let ssu = get_opt_str(df, ssu_col)?;
    let fpc = get_opt_f64(df, fpc_col)?;
    let fpc_ssu = get_opt_f64(df, fpc_ssu_col)?;

    let (y_arr, w_arr, g_arr, levels, n) = prepare_two_sample_data(
        df, y_col, weight_col, group_col, strata_col, psu_col, domain_col, domain_val,
    )?;

    let n_groups = levels.len();

    if n_groups == 2 {
        let res = ranktest_two_sample(
            &y_arr,
            &g_arr,
            &w_arr,
            n,
            strata,
            psu,
            ssu,
            fpc,
            fpc_ssu,
            method,
            singleton_method,
            levels,
        )?;

        df![
            "type" => vec!["two-sample"],
            "method" => vec![method_name],
            "y" => vec![y_col],
            "group" => vec![group_col],
            "delta" => vec![res.delta],
            "se" => vec![res.se],
            "t" => vec![res.t_stat],
            "df" => vec![res.df],
            "p_value" => vec![res.p_value],
            "n" => vec![res.n_obs as u32],
        ]
    } else {
        let res = ranktest_k_sample(
            &y_arr,
            &g_arr,
            &w_arr,
            n,
            n_groups,
            strata,
            psu,
            ssu,
            fpc,
            fpc_ssu,
            method,
            singleton_method,
            levels,
        )?;

        df![
            "type" => vec!["k-sample"],
            "method" => vec![method_name],
            "y" => vec![y_col],
            "group" => vec![group_col],
            "ndf" => vec![res.ndf as u32],
            "ddf" => vec![res.ddf],
            "chisq" => vec![res.chisq],
            "f_stat" => vec![res.f_stat],
            "p_value" => vec![res.p_value],
            "n" => vec![res.n_obs as u32],
        ]
    }
}

// ============================================================================
// Tabulation Function — add this to lib.rs
// ============================================================================

#[pyfunction]
#[pyo3(signature = (
    data, rowvar_col, weight_col,
    colvar_col=None, strata_col=None, psu_col=None, ssu_col=None,
    fpc_col=None, fpc_ssu_col=None, singleton_method=None,
    compute_totals=false,
))]
fn tabulate_rs(
    _py: Python,
    data: PyDataFrame,
    rowvar_col: String,
    weight_col: String,
    colvar_col: Option<String>,
    strata_col: Option<String>,
    psu_col: Option<String>,
    ssu_col: Option<String>,
    fpc_col: Option<String>,
    fpc_ssu_col: Option<String>,
    singleton_method: Option<String>,
    compute_totals: bool,
) -> PyResult<(PyDataFrame, PyDataFrame)> {
    let df: DataFrame = data.into();
    let result = compute_tabulate(
        &df,
        &rowvar_col,
        &weight_col,
        colvar_col.as_deref(),
        strata_col.as_deref(),
        psu_col.as_deref(),
        ssu_col.as_deref(),
        fpc_col.as_deref(),
        fpc_ssu_col.as_deref(),
        singleton_method.as_deref(),
        compute_totals,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok((PyDataFrame(result.0), PyDataFrame(result.1)))
}

fn compute_tabulate(
    df: &DataFrame,
    rowvar_col: &str,
    weight_col: &str,
    colvar_col: Option<&str>,
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>,
    compute_totals: bool,
) -> PolarsResult<(DataFrame, DataFrame)> {
    let weights = df.column(weight_col)?.as_materialized_series().f64()?;
    let strata = strata_col
        .map(|c| {
            df.column(c)
                .and_then(|s| Ok(s.as_materialized_series().str()?.clone()))
        })
        .transpose()?;
    let psu = psu_col
        .map(|c| {
            df.column(c)
                .and_then(|s| Ok(s.as_materialized_series().str()?.clone()))
        })
        .transpose()?;
    let ssu = ssu_col
        .map(|c| {
            df.column(c)
                .and_then(|s| Ok(s.as_materialized_series().str()?.clone()))
        })
        .transpose()?;
    let fpc = fpc_col
        .map(|c| {
            df.column(c)
                .and_then(|s| Ok(s.as_materialized_series().f64()?.clone()))
        })
        .transpose()?;
    let fpc_ssu = fpc_ssu_col
        .map(|c| {
            df.column(c)
                .and_then(|s| Ok(s.as_materialized_series().f64()?.clone()))
        })
        .transpose()?;

    let is_two_way = colvar_col.is_some();
    let n_obs = df.height();

    // Build combined key for two-way tables
    let rowvar_series = df.column(rowvar_col)?.as_materialized_series();
    let rowvar_str = rowvar_series.cast(&DataType::String)?;
    let rowvar_ca = rowvar_str.str()?;

    let combined_key: StringChunked;
    let y_effective: &StringChunked;

    if let Some(cv_col) = colvar_col {
        let colvar_series = df.column(cv_col)?.as_materialized_series();
        let colvar_str = colvar_series.cast(&DataType::String)?;
        let colvar_ca = colvar_str.str()?;

        let keys: Vec<Option<String>> = rowvar_ca
            .iter()
            .zip(colvar_ca.iter())
            .map(|(a, b)| match (a, b) {
                (Some(va), Some(vb)) => Some(format!("{}__by__{}", va, vb)),
                _ => None,
            })
            .collect();
        combined_key = StringChunked::from_iter(keys.into_iter());
        y_effective = &combined_key;
    } else {
        combined_key = rowvar_ca.clone();
        y_effective = &combined_key;
    }

    // Estimate proportions + covariance
    let (levels, proportions, ses, cov_matrix, deff_vec, df_val) =
        categorical::tabulation::estimate_proportions(
            y_effective,
            weights,
            strata.as_ref(),
            psu.as_ref(),
            ssu.as_ref(),
            fpc.as_ref(),
            fpc_ssu.as_ref(),
            singleton_method,
        )?;

    // Parse levels into row/col parts
    let k = levels.len();
    let mut rowvars: Vec<String> = Vec::with_capacity(k);
    let mut colvars: Vec<String> = Vec::with_capacity(k);
    for lvl in &levels {
        if is_two_way {
            match lvl.split_once("__by__") {
                Some((r, c)) => {
                    rowvars.push(r.to_string());
                    colvars.push(c.to_string());
                }
                None => {
                    rowvars.push(lvl.clone());
                    colvars.push(String::new());
                }
            }
        } else {
            rowvars.push(lvl.clone());
            colvars.push(String::new());
        }
    }

    // Build estimates: use totals if requested, otherwise proportions
    let (est_vals, se_vals) = if compute_totals {
        categorical::tabulation::estimate_totals(
            y_effective,
            weights,
            strata.as_ref(),
            psu.as_ref(),
            ssu.as_ref(),
            fpc.as_ref(),
            fpc_ssu.as_ref(),
            singleton_method,
            &levels,
        )?
    } else {
        (proportions.clone(), ses.clone())
    };

    let cv_vals: Vec<f64> = est_vals
        .iter()
        .zip(se_vals.iter())
        .map(|(&e, &s)| {
            if e.abs() > 0.0 {
                (s / e).abs()
            } else {
                f64::NAN
            }
        })
        .collect();

    // Cell estimates DataFrame
    let cells_df = df![
        "rowvar" => rowvars,
        "colvar" => colvars,
        "est" => &est_vals,
        "se" => &se_vals,
        "cv" => &cv_vals,
        "deff" => &deff_vec,
        "df" => vec![df_val; k],
        "n" => vec![n_obs as u32; k],
    ]?;

    // Rao-Scott statistics (two-way only)
    let stats_df = if is_two_way {
        let cv_col = colvar_col.unwrap();
        let colvar_series2 = df.column(cv_col)?.as_materialized_series();
        let colvar_str2 = colvar_series2.cast(&DataType::String)?;
        let colvar_ca2 = colvar_str2.str()?;

        let mut row_levels: Vec<String> = rowvar_ca
            .unique()?
            .iter()
            .filter_map(|v| v.map(|s| s.to_string()))
            .collect();
        sort_levels(&mut row_levels);
        let mut col_levels: Vec<String> = colvar_ca2
            .unique()?
            .iter()
            .filter_map(|v| v.map(|s| s.to_string()))
            .collect();
        sort_levels(&mut col_levels);

        let nr = row_levels.len();
        let nc = col_levels.len();

        // Reorder proportions and covariance to match row-major (sorted row, sorted col)
        let mut prop_ordered = vec![0.0; nr * nc];
        let mut cov_ordered = vec![vec![0.0; nr * nc]; nr * nc];

        // Build mapping from level string to ordered index
        let mut level_to_ordered: HashMap<String, usize> = HashMap::new();
        for (ri, rl) in row_levels.iter().enumerate() {
            for (ci, cl) in col_levels.iter().enumerate() {
                let key = format!("{}__by__{}", rl, cl);
                level_to_ordered.insert(key, ri * nc + ci);
            }
        }

        // Map original level order to row-major order
        for (orig_idx, lvl) in levels.iter().enumerate() {
            if let Some(&ordered_idx) = level_to_ordered.get(lvl) {
                prop_ordered[ordered_idx] = proportions[orig_idx];
                for (orig_idx2, lvl2) in levels.iter().enumerate() {
                    if let Some(&ordered_idx2) = level_to_ordered.get(lvl2) {
                        cov_ordered[ordered_idx][ordered_idx2] = cov_matrix[orig_idx][orig_idx2];
                    }
                }
            }
        }

        let (n_strata_count, n_psu_count) =
            categorical::tabulation::count_strata_psus(strata.as_ref(), psu.as_ref(), n_obs);

        let (
            p_chisq,
            p_df,
            p_p,
            p_adj_f,
            p_adj_ndf,
            p_adj_ddf,
            p_adj_p,
            lr_chisq,
            lr_df,
            lr_p,
            lr_adj_f,
            lr_adj_ndf,
            lr_adj_ddf,
            lr_adj_p,
        ) = categorical::tabulation::rao_scott(
            &prop_ordered,
            &cov_ordered,
            nr,
            nc,
            n_obs,
            n_strata_count,
            n_psu_count,
        );

        df![
            "stat" => vec!["chisq", "f"],
            "value" => vec![p_chisq, p_adj_f],
            "df" => vec![p_df, p_adj_ndf],
            "df2" => vec![f64::NAN, p_adj_ddf],
            "p_value" => vec![p_p, p_adj_p],
        ]?
    } else {
        // Empty stats DataFrame for one-way tables
        df![
            "stat" => Vec::<String>::new(),
            "value" => Vec::<f64>::new(),
            "df" => Vec::<f64>::new(),
            "df2" => Vec::<f64>::new(),
            "p_value" => Vec::<f64>::new(),
        ]?
    };

    Ok((cells_df, stats_df))
}

// ============================================================================
// Module Registration
// ============================================================================

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Taylor functions
    m.add_function(wrap_pyfunction!(taylor_mean, m)?)?;
    m.add_function(wrap_pyfunction!(taylor_total, m)?)?;
    m.add_function(wrap_pyfunction!(taylor_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(taylor_prop, m)?)?;
    m.add_function(wrap_pyfunction!(taylor_median, m)?)?;
    // Replication functions
    m.add_function(wrap_pyfunction!(replicate_mean, m)?)?;
    m.add_function(wrap_pyfunction!(replicate_total, m)?)?;
    m.add_function(wrap_pyfunction!(replicate_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(replicate_prop, m)?)?;
    m.add_function(wrap_pyfunction!(replicate_median, m)?)?;
    // GLM Regression
    m.add_function(wrap_pyfunction!(fit_glm_rs, m)?)?;
    // Categorical tests
    m.add_function(wrap_pyfunction!(ttest_rs, m)?)?;
    m.add_function(wrap_pyfunction!(ranktest_rs, m)?)?;
    m.add_function(wrap_pyfunction!(tabulate_rs, m)?)?;
    // Weighting functions
    m.add_function(wrap_pyfunction!(rake, m)?)?;
    m.add_function(wrap_pyfunction!(adjust_nr, m)?)?;
    m.add_function(wrap_pyfunction!(normalize, m)?)?;
    m.add_function(wrap_pyfunction!(poststratify, m)?)?;
    m.add_function(wrap_pyfunction!(poststratify_factor, m)?)?;
    // Calibration functions
    m.add_function(wrap_pyfunction!(calibrate, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate_by_domain, m)?)?;
    m.add_function(wrap_pyfunction!(calibrate_parallel, m)?)?;
    // Replicate weight creation functions
    m.add_function(wrap_pyfunction!(create_brr_wgts, m)?)?;
    m.add_function(wrap_pyfunction!(create_jk_wgts, m)?)?;
    m.add_function(wrap_pyfunction!(create_bootstrap_wgts, m)?)?;
    m.add_function(wrap_pyfunction!(create_sdr_wgts, m)?)?;
    Ok(())
}
