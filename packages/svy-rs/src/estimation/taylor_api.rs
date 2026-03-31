// src/estimation/taylor_api.rs
//
// PyO3-facing wrappers and `compute_*` helpers for Taylor linearization.
// The heavy math lives in taylor.rs; this file only handles DataFrame I/O,
// argument parsing, and looping over by-groups.

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::estimation::taylor::{
    SvyQuantileMethod,
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
    taylor_variance,
    weighted_median, weighted_median_domain,
};

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
    let df: DataFrame = data.into();

    if by_col.is_none() {
        let result = compute_mean_ungrouped(
            &df, &value_col, &weight_col,
            strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(),
            fpc_col.as_deref(), fpc_ssu_col.as_deref(), singleton_method.as_deref(),
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        return Ok(PyDataFrame(result));
    }

    let result = compute_mean_grouped(
        &df, &value_col, &weight_col,
        strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(),
        fpc_col.as_deref(), fpc_ssu_col.as_deref(),
        &by_col.unwrap(), singleton_method.as_deref(),
    )
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
    let strata = strata_col.map(|c| df.column(c).and_then(|s| s.str())).transpose()?;
    let psu    = psu_col.map(|c| df.column(c).and_then(|s| s.str())).transpose()?;
    let ssu    = ssu_col.map(|c| df.column(c).and_then(|s| s.str())).transpose()?;
    let fpc    = fpc_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;
    let fpc_ssu = fpc_ssu_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;

    let estimate = point_estimate_mean(y, weights)?;
    let scores   = scores_mean(y, weights)?;
    let variance = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
    let se       = variance.sqrt();
    let df_val   = degrees_of_freedom(weights, strata, psu)?;
    let n        = y.len() as u32;
    let srs_var  = srs_variance_mean(y, weights)?;
    let deff     = if srs_var > 0.0 { variance / srs_var } else { f64::NAN };

    df!["y" => vec![value_col], "est" => vec![estimate], "se" => vec![se],
        "var" => vec![variance], "df" => vec![df_val], "n" => vec![n], "deff" => vec![deff]]
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
    let strata = strata_col.map(|c| df.column(c).and_then(|s| s.str())).transpose()?;
    let psu    = psu_col.map(|c| df.column(c).and_then(|s| s.str())).transpose()?;
    let ssu    = ssu_col.map(|c| df.column(c).and_then(|s| s.str())).transpose()?;
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
    let mut deffs: Vec<f64> = Vec::new();
    let df_val = degrees_of_freedom(weights, strata, psu)?;

    for group_val in unique_groups.iter() {
        if let Some(group) = group_val {
            let domain_mask = by_str.equal(group);
            let n_domain    = domain_mask.sum().unwrap_or(0) as u32;
            let estimate    = point_estimate_mean_domain(y, weights, &domain_mask)?;
            let scores      = scores_mean_domain(y, weights, &domain_mask)?;
            let variance    = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
            let se          = variance.sqrt();
            let srs_var     = srs_variance_mean_domain(y, weights, &domain_mask)?;
            let deff        = if srs_var > 0.0 { variance / srs_var } else { f64::NAN };

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
    let df: DataFrame = data.into();
    if by_col.is_none() {
        let result = compute_total_ungrouped(
            &df, &value_col, &weight_col,
            strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(),
            fpc_col.as_deref(), fpc_ssu_col.as_deref(), singleton_method.as_deref(),
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        return Ok(PyDataFrame(result));
    }
    let result = compute_total_grouped(
        &df, &value_col, &weight_col,
        strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(),
        fpc_col.as_deref(), fpc_ssu_col.as_deref(),
        &by_col.unwrap(), singleton_method.as_deref(),
    )
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
    let strata = strata_col.map(|c| df.column(c).and_then(|s| s.str())).transpose()?;
    let psu    = psu_col.map(|c| df.column(c).and_then(|s| s.str())).transpose()?;
    let ssu    = ssu_col.map(|c| df.column(c).and_then(|s| s.str())).transpose()?;
    let fpc    = fpc_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;
    let fpc_ssu = fpc_ssu_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;

    let estimate = point_estimate_total(y, weights)?;
    let scores   = scores_total(y, weights)?;
    let variance = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
    let se       = variance.sqrt();
    let df_val   = degrees_of_freedom(weights, strata, psu)?;
    let n        = y.len() as u32;
    let srs_var  = srs_variance_total(y, weights)?;
    let deff     = if srs_var > 0.0 { variance / srs_var } else { f64::NAN };

    df!["y" => vec![value_col], "est" => vec![estimate], "se" => vec![se],
        "var" => vec![variance], "df" => vec![df_val], "n" => vec![n], "deff" => vec![deff]]
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
    let strata = strata_col.map(|c| df.column(c).and_then(|s| s.str())).transpose()?;
    let psu    = psu_col.map(|c| df.column(c).and_then(|s| s.str())).transpose()?;
    let ssu    = ssu_col.map(|c| df.column(c).and_then(|s| s.str())).transpose()?;
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
    let mut deffs: Vec<f64> = Vec::new();
    let df_val = degrees_of_freedom(weights, strata, psu)?;

    for group_val in unique_groups.iter() {
        if let Some(group) = group_val {
            let domain_mask = by_str.equal(group);
            let n_domain    = domain_mask.sum().unwrap_or(0) as u32;
            let estimate    = point_estimate_total_domain(y, weights, &domain_mask)?;
            let scores      = scores_total_domain(y, weights, &domain_mask)?;
            let variance    = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
            let se          = variance.sqrt();
            let srs_var     = srs_variance_total_domain(y, weights, &domain_mask)?;
            let deff        = if srs_var > 0.0 { variance / srs_var } else { f64::NAN };

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
    let df: DataFrame = data.into();
    if by_col.is_none() {
        let result = compute_ratio_ungrouped(
            &df, &numerator_col, &denominator_col, &weight_col,
            strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(),
            fpc_col.as_deref(), fpc_ssu_col.as_deref(), singleton_method.as_deref(),
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        return Ok(PyDataFrame(result));
    }
    let result = compute_ratio_grouped(
        &df, &numerator_col, &denominator_col, &weight_col,
        strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(),
        fpc_col.as_deref(), fpc_ssu_col.as_deref(),
        &by_col.unwrap(), singleton_method.as_deref(),
    )
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
    let strata = strata_col.map(|c| df.column(c).and_then(|s| s.str())).transpose()?;
    let psu    = psu_col.map(|c| df.column(c).and_then(|s| s.str())).transpose()?;
    let ssu    = ssu_col.map(|c| df.column(c).and_then(|s| s.str())).transpose()?;
    let fpc    = fpc_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;
    let fpc_ssu = fpc_ssu_col.map(|c| df.column(c).and_then(|s| s.f64())).transpose()?;

    let estimate = point_estimate_ratio(y, x, weights)?;
    let scores   = scores_ratio(y, x, weights)?;
    let variance = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
    let se       = variance.sqrt();
    let df_val   = degrees_of_freedom(weights, strata, psu)?;
    let n        = y.len() as u32;
    let srs_var  = srs_variance_ratio(y, x, weights)?;
    let deff     = if srs_var > 0.0 { variance / srs_var } else { f64::NAN };

    df!["y" => vec![numerator_col], "x" => vec![denominator_col], "est" => vec![estimate],
        "se" => vec![se], "var" => vec![variance], "df" => vec![df_val], "n" => vec![n], "deff" => vec![deff]]
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
    let strata = strata_col.map(|c| df.column(c).and_then(|s| s.str())).transpose()?;
    let psu    = psu_col.map(|c| df.column(c).and_then(|s| s.str())).transpose()?;
    let ssu    = ssu_col.map(|c| df.column(c).and_then(|s| s.str())).transpose()?;
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
    let mut deffs: Vec<f64> = Vec::new();
    let df_val = degrees_of_freedom(weights, strata, psu)?;

    for group_val in unique_groups.iter() {
        if let Some(group) = group_val {
            let domain_mask = by_str.equal(group);
            let n_domain    = domain_mask.sum().unwrap_or(0) as u32;
            let estimate    = point_estimate_ratio_domain(y, x, weights, &domain_mask)?;
            let scores      = scores_ratio_domain(y, x, weights, &domain_mask)?;
            let variance    = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
            let se          = variance.sqrt();
            let srs_var     = srs_variance_ratio_domain(y, x, weights, &domain_mask)?;
            let deff        = if srs_var > 0.0 { variance / srs_var } else { f64::NAN };

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
    let df: DataFrame = data.into();
    if by_col.is_none() {
        let result = compute_prop_ungrouped(
            &df, &value_col, &weight_col,
            strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(),
            fpc_col.as_deref(), fpc_ssu_col.as_deref(), singleton_method.as_deref(),
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        return Ok(PyDataFrame(result));
    }
    let result = compute_prop_grouped(
        &df, &value_col, &weight_col,
        strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(),
        fpc_col.as_deref(), fpc_ssu_col.as_deref(),
        &by_col.unwrap(), singleton_method.as_deref(),
    )
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
    let strata = strata_col.map(|c| df.column(c).and_then(|s| s.str())).transpose()?;
    let psu    = psu_col.map(|c| df.column(c).and_then(|s| s.str())).transpose()?;
    let ssu    = ssu_col.map(|c| df.column(c).and_then(|s| s.str())).transpose()?;
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
        let variance = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
        let se       = variance.sqrt();
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

fn compute_prop_grouped(
    df: &DataFrame,
    value_col: &str, weight_col: &str,
    strata_col: Option<&str>, psu_col: Option<&str>, ssu_col: Option<&str>,
    fpc_col: Option<&str>, fpc_ssu_col: Option<&str>,
    by_col: &str, singleton_method: Option<&str>,
) -> PolarsResult<DataFrame> {
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col.map(|c| df.column(c).and_then(|s| s.str())).transpose()?;
    let psu    = psu_col.map(|c| df.column(c).and_then(|s| s.str())).transpose()?;
    let ssu    = ssu_col.map(|c| df.column(c).and_then(|s| s.str())).transpose()?;
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
            let n_domain    = domain_mask.sum().unwrap_or(0) as u32;
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
                let variance = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
                let se       = variance.sqrt();
                let srs_var  = srs_variance_mean_domain(&indicator_ca, weights, &domain_mask)?;
                let deff     = if srs_var > 0.0 { variance / srs_var } else { f64::NAN };

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
    let df: DataFrame = data.into();
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

fn compute_median_ungrouped(
    df: &DataFrame,
    value_col: &str, weight_col: &str,
    strata_col: Option<&str>, psu_col: Option<&str>, ssu_col: Option<&str>,
    fpc_col: Option<&str>, fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>, q_method: SvyQuantileMethod,
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col.map(|c| df.column(c).and_then(|s| s.str())).transpose()?;
    let psu    = psu_col.map(|c| df.column(c).and_then(|s| s.str())).transpose()?;
    let ssu    = ssu_col.map(|c| df.column(c).and_then(|s| s.str())).transpose()?;
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

fn compute_median_grouped(
    df: &DataFrame,
    value_col: &str, weight_col: &str,
    strata_col: Option<&str>, psu_col: Option<&str>, ssu_col: Option<&str>,
    fpc_col: Option<&str>, fpc_ssu_col: Option<&str>,
    by_col: &str, singleton_method: Option<&str>, q_method: SvyQuantileMethod,
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let strata = strata_col.map(|c| df.column(c).and_then(|s| s.str())).transpose()?;
    let psu    = psu_col.map(|c| df.column(c).and_then(|s| s.str())).transpose()?;
    let ssu    = ssu_col.map(|c| df.column(c).and_then(|s| s.str())).transpose()?;
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
