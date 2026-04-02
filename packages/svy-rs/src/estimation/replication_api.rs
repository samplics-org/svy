// src/estimation/replication_api.rs
//
// PyO3-facing wrappers and `compute_replicate_*` helpers for replication-based
// variance estimation.  The matrix math lives in replication.rs; this file
// only handles DataFrame I/O, argument parsing, and looping over by-groups.

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::estimation::taylor::SvyQuantileMethod;
use crate::estimation::replication::{
    RepMethod, VarianceCenter,
    extract_rep_weights_matrix,
    index_domains,
    matrix_mean_by_domain, matrix_mean_estimates,
    matrix_median_by_domain, matrix_median_estimates,
    matrix_prop_by_domain, matrix_prop_estimates,
    matrix_prop_by_domain_str, matrix_prop_estimates_str,
    matrix_ratio_by_domain, matrix_ratio_estimates,
    matrix_total_by_domain, matrix_total_estimates,
    replicate_coefficients,
    variance_from_replicates,
};

// ============================================================================
// Shared helpers
// ============================================================================

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
#[pyo3(signature = (data, value_col, weight_col, rep_weight_cols, method, fay_coef=0.0, center="rep_mean", degrees_of_freedom=None, by_col=None))]
pub fn replicate_mean(
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
    let rep_method       = parse_rep_method(&method)?;
    let variance_center  = parse_variance_center(center)?;
    let df_val = degrees_of_freedom.unwrap_or(n_reps.saturating_sub(1) as u32);

    let result = if by_col.is_none() {
        compute_replicate_mean_ungrouped(&df, &value_col, &weight_col, &rep_weight_cols,
            rep_method, fay_coef, variance_center, df_val)
    } else {
        compute_replicate_mean_grouped(&df, &value_col, &weight_col, &rep_weight_cols,
            rep_method, fay_coef, variance_center, df_val, by_col.as_ref().unwrap())
    };
    result.map(PyDataFrame).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

fn compute_replicate_mean_ungrouped(
    df: &DataFrame,
    value_col: &str, weight_col: &str, rep_weight_cols: &[String],
    method: RepMethod, fay_coef: f64, center: VarianceCenter, df_val: u32,
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
    let variance  = variance_from_replicates(method, theta_full, &theta_reps, &rep_coefs, center);
    let se = variance.sqrt();

    df!["y" => vec![value_col], "est" => vec![theta_full], "se" => vec![se],
        "var" => vec![variance], "df" => vec![df_val], "n" => vec![n as u32]]
}

fn compute_replicate_mean_grouped(
    df: &DataFrame,
    value_col: &str, weight_col: &str, rep_weight_cols: &[String],
    method: RepMethod, fay_coef: f64, center: VarianceCenter, df_val: u32,
    by_col: &str,
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let by_str = df.column(by_col)?.str()?;
    let n = y.len();
    let n_reps = rep_weight_cols.len();
    let y_arr: Vec<f64> = y.into_iter().map(|v| v.unwrap_or(0.0)).collect();
    let w_arr: Vec<f64> = weights.into_iter().map(|v| v.unwrap_or(0.0)).collect();

    let (domain_ids, domain_names, n_domains) = index_domains(by_str);
    let (rep_w_matrix, _, _) = extract_rep_weights_matrix(df, rep_weight_cols)?;
    let (theta_full_vec, theta_reps_vec, counts) =
        matrix_mean_by_domain(&y_arr, &w_arr, &rep_w_matrix, &domain_ids, n_domains, n, n_reps);
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
    df![by_col => by_vals, "y" => vec![value_col; n_domains], "est" => estimates,
        "se" => ses, "var" => variances, "df" => dfs, "n" => ns]
}

// ============================================================================
// Total
// ============================================================================

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, rep_weight_cols, method, fay_coef=0.0, center="rep_mean", degrees_of_freedom=None, by_col=None))]
pub fn replicate_total(
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
    let rep_method      = parse_rep_method(&method)?;
    let variance_center = parse_variance_center(center)?;
    let df_val = degrees_of_freedom.unwrap_or(n_reps.saturating_sub(1) as u32);

    let result = if by_col.is_none() {
        compute_replicate_total_ungrouped(&df, &value_col, &weight_col, &rep_weight_cols,
            rep_method, fay_coef, variance_center, df_val)
    } else {
        compute_replicate_total_grouped(&df, &value_col, &weight_col, &rep_weight_cols,
            rep_method, fay_coef, variance_center, df_val, by_col.as_ref().unwrap())
    };
    result.map(PyDataFrame).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

fn compute_replicate_total_ungrouped(
    df: &DataFrame,
    value_col: &str, weight_col: &str, rep_weight_cols: &[String],
    method: RepMethod, fay_coef: f64, center: VarianceCenter, df_val: u32,
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
    let variance  = variance_from_replicates(method, theta_full, &theta_reps, &rep_coefs, center);
    let se = variance.sqrt();

    df!["y" => vec![value_col], "est" => vec![theta_full], "se" => vec![se],
        "var" => vec![variance], "df" => vec![df_val], "n" => vec![n as u32]]
}

fn compute_replicate_total_grouped(
    df: &DataFrame,
    value_col: &str, weight_col: &str, rep_weight_cols: &[String],
    method: RepMethod, fay_coef: f64, center: VarianceCenter, df_val: u32,
    by_col: &str,
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let by_str = df.column(by_col)?.str()?;
    let n = y.len();
    let n_reps = rep_weight_cols.len();
    let y_arr: Vec<f64> = y.into_iter().map(|v| v.unwrap_or(0.0)).collect();
    let w_arr: Vec<f64> = weights.into_iter().map(|v| v.unwrap_or(0.0)).collect();

    let (domain_ids, domain_names, n_domains) = index_domains(by_str);
    let (rep_w_matrix, _, _) = extract_rep_weights_matrix(df, rep_weight_cols)?;
    let (theta_full_vec, theta_reps_vec, counts) =
        matrix_total_by_domain(&y_arr, &w_arr, &rep_w_matrix, &domain_ids, n_domains, n, n_reps);
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
    df![by_col => by_vals, "y" => vec![value_col; n_domains], "est" => estimates,
        "se" => ses, "var" => variances, "df" => dfs, "n" => ns]
}

// ============================================================================
// Ratio
// ============================================================================

#[pyfunction]
#[pyo3(signature = (data, numerator_col, denominator_col, weight_col, rep_weight_cols, method, fay_coef=0.0, center="rep_mean", degrees_of_freedom=None, by_col=None))]
pub fn replicate_ratio(
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
    let rep_method      = parse_rep_method(&method)?;
    let variance_center = parse_variance_center(center)?;
    let df_val = degrees_of_freedom.unwrap_or(n_reps.saturating_sub(1) as u32);

    let result = if by_col.is_none() {
        compute_replicate_ratio_ungrouped(&df, &numerator_col, &denominator_col, &weight_col,
            &rep_weight_cols, rep_method, fay_coef, variance_center, df_val)
    } else {
        compute_replicate_ratio_grouped(&df, &numerator_col, &denominator_col, &weight_col,
            &rep_weight_cols, rep_method, fay_coef, variance_center, df_val,
            by_col.as_ref().unwrap())
    };
    result.map(PyDataFrame).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

fn compute_replicate_ratio_ungrouped(
    df: &DataFrame,
    numerator_col: &str, denominator_col: &str, weight_col: &str,
    rep_weight_cols: &[String],
    method: RepMethod, fay_coef: f64, center: VarianceCenter, df_val: u32,
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
    let variance  = variance_from_replicates(method, theta_full, &theta_reps, &rep_coefs, center);
    let se = variance.sqrt();

    df!["y" => vec![numerator_col], "x" => vec![denominator_col], "est" => vec![theta_full],
        "se" => vec![se], "var" => vec![variance], "df" => vec![df_val], "n" => vec![n as u32]]
}

fn compute_replicate_ratio_grouped(
    df: &DataFrame,
    numerator_col: &str, denominator_col: &str, weight_col: &str,
    rep_weight_cols: &[String],
    method: RepMethod, fay_coef: f64, center: VarianceCenter, df_val: u32,
    by_col: &str,
) -> PolarsResult<DataFrame> {
    let y = df.column(numerator_col)?.f64()?;
    let x = df.column(denominator_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let by_str = df.column(by_col)?.str()?;
    let n = y.len();
    let n_reps = rep_weight_cols.len();
    let y_arr: Vec<f64> = y.into_iter().map(|v| v.unwrap_or(0.0)).collect();
    let x_arr: Vec<f64> = x.into_iter().map(|v| v.unwrap_or(0.0)).collect();
    let w_arr: Vec<f64> = weights.into_iter().map(|v| v.unwrap_or(0.0)).collect();

    let (domain_ids, domain_names, n_domains) = index_domains(by_str);
    let (rep_w_matrix, _, _) = extract_rep_weights_matrix(df, rep_weight_cols)?;
    let (theta_full_vec, theta_reps_vec, counts) =
        matrix_ratio_by_domain(&y_arr, &x_arr, &w_arr, &rep_w_matrix, &domain_ids, n_domains, n, n_reps);
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
    df![by_col => by_vals, "y" => vec![numerator_col; n_domains], "x" => vec![denominator_col; n_domains],
        "est" => estimates, "se" => ses, "var" => variances, "df" => dfs, "n" => ns]
}

// ============================================================================
// Proportion
// ============================================================================

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, rep_weight_cols, method, fay_coef=0.0, center="rep_mean", degrees_of_freedom=None, by_col=None))]
pub fn replicate_prop(
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
    let rep_method      = parse_rep_method(&method)?;
    let variance_center = parse_variance_center(center)?;
    let df_val = degrees_of_freedom.unwrap_or(n_reps.saturating_sub(1) as u32);

    let result = if by_col.is_none() {
        compute_replicate_prop_ungrouped(&df, &value_col, &weight_col, &rep_weight_cols,
            rep_method, fay_coef, variance_center, df_val)
    } else {
        compute_replicate_prop_grouped(&df, &value_col, &weight_col, &rep_weight_cols,
            rep_method, fay_coef, variance_center, df_val, by_col.as_ref().unwrap())
    };
    result.map(PyDataFrame).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

fn compute_replicate_prop_ungrouped(
    df: &DataFrame,
    value_col: &str, weight_col: &str, rep_weight_cols: &[String],
    method: RepMethod, fay_coef: f64, center: VarianceCenter, df_val: u32,
) -> PolarsResult<DataFrame> {
    let y_series = df.column(value_col)?;
    let weights  = df.column(weight_col)?.f64()?;
    let n = y_series.len();
    let n_reps = rep_weight_cols.len();
    let w_arr: Vec<f64> = weights.into_iter().map(|v| v.unwrap_or(0.0)).collect();
    let (rep_w_matrix, _, _) = extract_rep_weights_matrix(df, rep_weight_cols)?;
    let rep_coefs = replicate_coefficients(method, n_reps, fay_coef);

    // String/Categorical: use string-keyed level functions so level labels are
    // preserved as-is without any numeric cast.
    let is_string = matches!(y_series.dtype(), DataType::String | DataType::Categorical(_, _));

    let (n_levels, level_strs, estimates, variances) = if is_string {
        let y_cast = y_series.cast(&DataType::String)?;
        let y_arr: Vec<String> = y_cast.str()?.into_iter()
            .map(|v| v.unwrap_or("").to_string())
            .collect();
        let (levels, theta_full, theta_reps) =
            matrix_prop_estimates_str(&y_arr, &w_arr, &rep_w_matrix, n, n_reps);
        let n_l = levels.len();
        let vars: Vec<f64> = (0..n_l)
            .map(|l| variance_from_replicates(method, theta_full[l], &theta_reps[l], &rep_coefs, center))
            .collect();
        (n_l, levels, theta_full, vars)
    } else {
        // Integer or boolean: convert to i64 category codes.
        let y_arr: Vec<i64> = if y_series.dtype().is_integer() {
            y_series.i64()?.into_iter().map(|v| v.unwrap_or(0)).collect()
        } else if y_series.dtype() == &DataType::Boolean {
            y_series.bool()?.into_iter()
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
        let (levels, theta_full, theta_reps) =
            matrix_prop_estimates(&y_arr, &w_arr, &rep_w_matrix, n, n_reps);
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
    method: RepMethod, fay_coef: f64, center: VarianceCenter, df_val: u32,
    by_col: &str,
) -> PolarsResult<DataFrame> {
    let y_series = df.column(value_col)?;
    let weights  = df.column(weight_col)?.f64()?;
    let by_str   = df.column(by_col)?.str()?;
    let n = y_series.len();
    let n_reps = rep_weight_cols.len();
    let w_arr: Vec<f64> = weights.into_iter().map(|v| v.unwrap_or(0.0)).collect();

    let (domain_ids, domain_names, n_domains) = index_domains(by_str);
    let (rep_w_matrix, _, _) = extract_rep_weights_matrix(df, rep_weight_cols)?;
    let rep_coefs = replicate_coefficients(method, n_reps, fay_coef);

    let is_string = matches!(y_series.dtype(), DataType::String | DataType::Categorical(_, _));

    let (levels_str, theta_full_mat, theta_reps_mat, counts) = if is_string {
        let y_cast = y_series.cast(&DataType::String)?;
        let y_arr: Vec<String> = y_cast.str()?.into_iter()
            .map(|v| v.unwrap_or("").to_string())
            .collect();
        let (levels, tf, tr, counts) =
            matrix_prop_by_domain_str(&y_arr, &w_arr, &rep_w_matrix, &domain_ids, n_domains, n, n_reps);
        (levels, tf, tr, counts)
    } else {
        let y_arr: Vec<i64> = if y_series.dtype().is_integer() {
            y_series.i64()?.into_iter().map(|v| v.unwrap_or(0)).collect()
        } else if y_series.dtype() == &DataType::Boolean {
            y_series.bool()?.into_iter()
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
// Median
// ============================================================================

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, rep_weight_cols, method, fay_coef=0.0, center="rep_mean", degrees_of_freedom=None, by_col=None, quantile_method=None))]
pub fn replicate_median(
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
    let rep_method      = parse_rep_method(&method)?;
    let variance_center = parse_variance_center(center)?;
    let q_method = quantile_method
        .as_deref()
        .map(SvyQuantileMethod::from_str)
        .unwrap_or(SvyQuantileMethod::Higher);
    let df_val = degrees_of_freedom.unwrap_or(n_reps.saturating_sub(1) as u32);

    let result = if by_col.is_none() {
        compute_replicate_median_ungrouped(&df, &value_col, &weight_col, &rep_weight_cols,
            rep_method, fay_coef, variance_center, df_val, q_method)
    } else {
        compute_replicate_median_grouped(&df, &value_col, &weight_col, &rep_weight_cols,
            rep_method, fay_coef, variance_center, df_val, by_col.as_ref().unwrap(), q_method)
    };
    result.map(PyDataFrame).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

fn compute_replicate_median_ungrouped(
    df: &DataFrame,
    value_col: &str, weight_col: &str, rep_weight_cols: &[String],
    method: RepMethod, fay_coef: f64, center: VarianceCenter, df_val: u32,
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
    let variance  = variance_from_replicates(method, theta_full, &theta_reps, &rep_coefs, center);
    let se = variance.sqrt();

    df!["y" => vec![value_col], "est" => vec![theta_full], "se" => vec![se],
        "var" => vec![variance], "df" => vec![df_val], "n" => vec![n as u32]]
}

fn compute_replicate_median_grouped(
    df: &DataFrame,
    value_col: &str, weight_col: &str, rep_weight_cols: &[String],
    method: RepMethod, fay_coef: f64, center: VarianceCenter, df_val: u32,
    by_col: &str, q_method: SvyQuantileMethod,
) -> PolarsResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
    let weights = df.column(weight_col)?.f64()?;
    let by_str  = df.column(by_col)?.str()?;
    let n = y.len();
    let n_reps = rep_weight_cols.len();
    let y_arr: Vec<f64> = y.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect();
    let w_arr: Vec<f64> = weights.into_iter().map(|v| v.unwrap_or(0.0)).collect();

    let (domain_ids, domain_names, n_domains) = index_domains(by_str);
    let (rep_w_matrix, _, _) = extract_rep_weights_matrix(df, rep_weight_cols)?;
    let (theta_full_vec, theta_reps_vec, counts) =
        matrix_median_by_domain(&y_arr, &w_arr, &rep_w_matrix, &domain_ids, n_domains, n, n_reps, q_method);
    let rep_coefs = replicate_coefficients(method, n_reps, fay_coef);

    let mut by_vals:   Vec<String> = Vec::with_capacity(n_domains);
    let mut estimates: Vec<f64>    = Vec::with_capacity(n_domains);
    let mut ses:       Vec<f64>    = Vec::with_capacity(n_domains);
    let mut variances: Vec<f64>    = Vec::with_capacity(n_domains);
    let mut dfs:       Vec<u32>    = Vec::with_capacity(n_domains);
    let mut ns:        Vec<u32>    = Vec::with_capacity(n_domains);

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
