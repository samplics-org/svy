// src/lib.rs
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use polars::prelude::*;

mod estimation;
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

use crate::estimation::replication::{
    RepMethod, replicate_coefficients, variance_from_replicates,
    extract_rep_weights_matrix, index_domains,
    matrix_mean_estimates, matrix_total_estimates, matrix_ratio_estimates,
    matrix_mean_by_domain, matrix_total_by_domain, matrix_ratio_by_domain,
    matrix_prop_estimates, matrix_prop_by_domain,
};


// ============================================================================
// Taylor Linearization Functions
// ============================================================================

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_stage2_col=None, by_col=None))]
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
            fpc_stage2_col.as_deref(),
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
        fpc_stage2_col.as_deref(),
        &by_col_name,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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
    fpc_stage2_col: Option<&str>,
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
    let variance = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_stage2)?;
    let se = variance.sqrt();
    let df_val = degrees_of_freedom(weights, strata, psu)?;
    let n = y.len() as u32;

    // Calculate DEFF
    let srs_var = srs_variance_mean(y, weights)?;
    let deff = if srs_var > 0.0 { variance / srs_var } else { f64::NAN };

    df![
        "y" => vec![value_col],
        "est" => vec![estimate],
        "se" => vec![se],
        "var" => vec![variance],
        "df" => vec![df_val],
        "n" => vec![n],
        "deff" => vec![deff],
    ]
}

fn compute_mean_grouped(
    df: &DataFrame,
    value_col: &str,
    weight_col: &str,
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_stage2_col: Option<&str>,
    by_col: &str,
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
            let variance = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_stage2)?;
            let se = variance.sqrt();

            // Calculate DEFF for domain
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
    df![
        by_col => by_vals,
        "y" => vec![value_col; n_groups],
        "est" => estimates,
        "se" => ses,
        "var" => variances,
        "df" => dfs,
        "n" => ns,
        "deff" => deffs,
    ]
}

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_stage2_col=None, by_col=None))]
fn taylor_total(
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
            fpc_stage2_col.as_deref(),
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
        fpc_stage2_col.as_deref(),
        &by_col_name,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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
    fpc_stage2_col: Option<&str>,
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
    let variance = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_stage2)?;
    let se = variance.sqrt();
    let df_val = degrees_of_freedom(weights, strata, psu)?;
    let n = y.len() as u32;

    // Calculate DEFF
    let srs_var = srs_variance_total(y, weights)?;
    let deff = if srs_var > 0.0 { variance / srs_var } else { f64::NAN };

    df![
        "y" => vec![value_col],
        "est" => vec![estimate],
        "se" => vec![se],
        "var" => vec![variance],
        "df" => vec![df_val],
        "n" => vec![n],
        "deff" => vec![deff],
    ]
}

fn compute_total_grouped(
    df: &DataFrame,
    value_col: &str,
    weight_col: &str,
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_stage2_col: Option<&str>,
    by_col: &str,
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
            let variance = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_stage2)?;
            let se = variance.sqrt();

            // Calculate DEFF for domain
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
    df![
        by_col => by_vals,
        "y" => vec![value_col; n_groups],
        "est" => estimates,
        "se" => ses,
        "var" => variances,
        "df" => dfs,
        "n" => ns,
        "deff" => deffs,
    ]
}

#[pyfunction]
#[pyo3(signature = (data, numerator_col, denominator_col, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_stage2_col=None, by_col=None))]
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
    fpc_stage2_col: Option<String>,
    by_col: Option<String>,
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
            fpc_stage2_col.as_deref(),
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
        fpc_stage2_col.as_deref(),
        &by_col_name,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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
    fpc_stage2_col: Option<&str>,
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
    let variance = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_stage2)?;
    let se = variance.sqrt();
    let df_val = degrees_of_freedom(weights, strata, psu)?;
    let n = y.len() as u32;

    // Calculate DEFF
    let srs_var = srs_variance_ratio(y, x, weights)?;
    let deff = if srs_var > 0.0 { variance / srs_var } else { f64::NAN };

    df![
        "y" => vec![numerator_col],
        "x" => vec![denominator_col],
        "est" => vec![estimate],
        "se" => vec![se],
        "var" => vec![variance],
        "df" => vec![df_val],
        "n" => vec![n],
        "deff" => vec![deff],
    ]
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
    fpc_stage2_col: Option<&str>,
    by_col: &str,
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
            let variance = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_stage2)?;
            let se = variance.sqrt();

            // Calculate DEFF for domain
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
    df![
        by_col => by_vals,
        "y" => vec![numerator_col; n_groups],
        "x" => vec![denominator_col; n_groups],
        "est" => estimates,
        "se" => ses,
        "var" => variances,
        "df" => dfs,
        "n" => ns,
        "deff" => deffs,
    ]
}

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_stage2_col=None, by_col=None))]
fn taylor_prop(
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
            fpc_stage2_col.as_deref(),
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
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
        fpc_stage2_col.as_deref(),
        &by_col_name,
    ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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
    fpc_stage2_col: Option<&str>,
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

    let mut levels: Vec<String> = unique_levels.iter()
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
        let indicator: Vec<Option<f64>> = value_str.iter()
            .map(|v| {
                match v {
                    Some(val) if val == lvl => Some(1.0),
                    Some(_) => Some(0.0),
                    None => None,
                }
            })
            .collect();
        let indicator_ca = Float64Chunked::from_slice_options("indicator".into(), &indicator);

        let estimate = point_estimate_mean(&indicator_ca, weights)?;
        let scores = scores_mean(&indicator_ca, weights)?;
        let variance = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_stage2)?;
        let se = variance.sqrt();

        // Calculate DEFF for proportion
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
    df![
        "y" => vec![value_col; n_levels],
        "level" => level_vals,
        "est" => estimates,
        "se" => ses,
        "var" => variances,
        "df" => dfs_vec,
        "n" => ns,
        "deff" => deffs,
    ]
}

fn compute_prop_grouped(
    df: &DataFrame,
    value_col: &str,
    weight_col: &str,
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_stage2_col: Option<&str>,
    by_col: &str,
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

    let mut levels: Vec<String> = unique_levels.iter()
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
                let indicator: Vec<Option<f64>> = value_str.iter()
                    .map(|v| {
                        match v {
                            Some(val) if val == lvl => Some(1.0),
                            Some(_) => Some(0.0),
                            None => None,
                        }
                    })
                    .collect();
                let indicator_ca = Float64Chunked::from_slice_options("indicator".into(), &indicator);

                let estimate = point_estimate_mean_domain(&indicator_ca, weights, &domain_mask)?;
                let scores = scores_mean_domain(&indicator_ca, weights, &domain_mask)?;
                let variance = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_stage2)?;
                let se = variance.sqrt();

                // Calculate DEFF for proportion within domain
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
    df![
        by_col => by_vals,
        "y" => vec![value_col; n_rows],
        "level" => level_vals,
        "est" => estimates,
        "se" => ses,
        "var" => variances,
        "df" => dfs_vec,
        "n" => ns,
        "deff" => deffs,
    ]
}


// ============================================================================
// Replication-Based Estimation Functions
// ============================================================================

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, rep_weight_cols, method, fay_coef=0.0, degrees_of_freedom=None, by_col=None))]
fn replicate_mean(
    _py: Python,
    data: PyDataFrame,
    value_col: String,
    weight_col: String,
    rep_weight_cols: Vec<String>,
    method: String,
    fay_coef: f64,
    degrees_of_freedom: Option<u32>,
    by_col: Option<String>,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let n_reps = rep_weight_cols.len();

    let rep_method = RepMethod::from_str(&method)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unknown method: {}. Use 'BRR', 'Bootstrap', 'Jackknife', or 'SDR'", method)
        ))?;

    let df_val = degrees_of_freedom.unwrap_or(n_reps.saturating_sub(1) as u32);

    let result = if by_col.is_none() {
        compute_replicate_mean_ungrouped(&df, &value_col, &weight_col, &rep_weight_cols, rep_method, fay_coef, df_val)
    } else {
        compute_replicate_mean_grouped(&df, &value_col, &weight_col, &rep_weight_cols, rep_method, fay_coef, df_val, by_col.as_ref().unwrap())
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
    let variance = variance_from_replicates(method, theta_full, &theta_reps, &rep_coefs);
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
        let variance = variance_from_replicates(method, theta_full_vec[k], &theta_reps_vec[k], &rep_coefs);

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
#[pyo3(signature = (data, value_col, weight_col, rep_weight_cols, method, fay_coef=0.0, degrees_of_freedom=None, by_col=None))]
fn replicate_total(
    _py: Python,
    data: PyDataFrame,
    value_col: String,
    weight_col: String,
    rep_weight_cols: Vec<String>,
    method: String,
    fay_coef: f64,
    degrees_of_freedom: Option<u32>,
    by_col: Option<String>,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let n_reps = rep_weight_cols.len();

    let rep_method = RepMethod::from_str(&method)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unknown method: {}. Use 'BRR', 'Bootstrap', 'Jackknife', or 'SDR'", method)
        ))?;

    let df_val = degrees_of_freedom.unwrap_or(n_reps.saturating_sub(1) as u32);

    let result = if by_col.is_none() {
        compute_replicate_total_ungrouped(&df, &value_col, &weight_col, &rep_weight_cols, rep_method, fay_coef, df_val)
    } else {
        compute_replicate_total_grouped(&df, &value_col, &weight_col, &rep_weight_cols, rep_method, fay_coef, df_val, by_col.as_ref().unwrap())
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
    let variance = variance_from_replicates(method, theta_full, &theta_reps, &rep_coefs);
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
        let variance = variance_from_replicates(method, theta_full_vec[k], &theta_reps_vec[k], &rep_coefs);

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
#[pyo3(signature = (data, numerator_col, denominator_col, weight_col, rep_weight_cols, method, fay_coef=0.0, degrees_of_freedom=None, by_col=None))]
fn replicate_ratio(
    _py: Python,
    data: PyDataFrame,
    numerator_col: String,
    denominator_col: String,
    weight_col: String,
    rep_weight_cols: Vec<String>,
    method: String,
    fay_coef: f64,
    degrees_of_freedom: Option<u32>,
    by_col: Option<String>,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let n_reps = rep_weight_cols.len();

    let rep_method = RepMethod::from_str(&method)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unknown method: {}. Use 'BRR', 'Bootstrap', 'Jackknife', or 'SDR'", method)
        ))?;

    let df_val = degrees_of_freedom.unwrap_or(n_reps.saturating_sub(1) as u32);

    let result = if by_col.is_none() {
        compute_replicate_ratio_ungrouped(&df, &numerator_col, &denominator_col, &weight_col, &rep_weight_cols, rep_method, fay_coef, df_val)
    } else {
        compute_replicate_ratio_grouped(&df, &numerator_col, &denominator_col, &weight_col, &rep_weight_cols, rep_method, fay_coef, df_val, by_col.as_ref().unwrap())
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
    let variance = variance_from_replicates(method, theta_full, &theta_reps, &rep_coefs);
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
        let variance = variance_from_replicates(method, theta_full_vec[k], &theta_reps_vec[k], &rep_coefs);

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
#[pyo3(signature = (data, value_col, weight_col, rep_weight_cols, method, fay_coef=0.0, degrees_of_freedom=None, by_col=None))]
fn replicate_prop(
    _py: Python,
    data: PyDataFrame,
    value_col: String,
    weight_col: String,
    rep_weight_cols: Vec<String>,
    method: String,
    fay_coef: f64,
    degrees_of_freedom: Option<u32>,
    by_col: Option<String>,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let n_reps = rep_weight_cols.len();

    let rep_method = RepMethod::from_str(&method)
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unknown method: {}. Use 'BRR', 'Bootstrap', 'Jackknife', or 'SDR'", method)
        ))?;

    let df_val = degrees_of_freedom.unwrap_or(n_reps.saturating_sub(1) as u32);

    let result = if by_col.is_none() {
        compute_replicate_prop_ungrouped(&df, &value_col, &weight_col, &rep_weight_cols, rep_method, fay_coef, df_val)
    } else {
        compute_replicate_prop_grouped(&df, &value_col, &weight_col, &rep_weight_cols, rep_method, fay_coef, df_val, by_col.as_ref().unwrap())
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
        let variance = variance_from_replicates(method, theta_full_vec[l], &theta_reps_vec[l], &rep_coefs);

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
    let n_levels = levels.len();

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
            let variance = variance_from_replicates(method, theta_full, theta_reps, &rep_coefs);

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


#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Taylor linearization functions
    m.add_function(wrap_pyfunction!(taylor_mean, m)?)?;
    m.add_function(wrap_pyfunction!(taylor_total, m)?)?;
    m.add_function(wrap_pyfunction!(taylor_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(taylor_prop, m)?)?;

    // Replication functions
    m.add_function(wrap_pyfunction!(replicate_mean, m)?)?;
    m.add_function(wrap_pyfunction!(replicate_total, m)?)?;
    m.add_function(wrap_pyfunction!(replicate_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(replicate_prop, m)?)?;

    Ok(())
}
