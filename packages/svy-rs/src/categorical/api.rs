// src/categorical/api.rs
//
// PyO3-facing wrappers and compute helpers for categorical tests and
// tabulation.  The statistical implementations live in categorical/ttest.rs,
// categorical/ranktest.rs, and categorical/tabulation.rs; this file only
// handles DataFrame I/O, argument parsing, domain/by-group orchestration,
// and building the result DataFrames.

use std::collections::HashMap;

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::categorical::ranktest::{RankScoreMethod, ranktest_k_sample, ranktest_two_sample};
use crate::categorical::tabulation::{
    count_strata_psus, estimate_proportions, estimate_totals, rao_scott, sort_levels,
};
use crate::categorical::ttest::{ttest_one_sample, ttest_one_sample_domain, ttest_two_sample};

// ============================================================================
// Shared column-extraction helpers
// ============================================================================

pub(crate) fn get_opt_str<'a>(
    df: &'a DataFrame,
    col: Option<&str>,
) -> PolarsResult<Option<&'a StringChunked>> {
    match col {
        Some(c) => Ok(Some(df.column(c)?.as_materialized_series().str()?)),
        None => Ok(None),
    }
}

pub(crate) fn get_opt_f64<'a>(
    df: &'a DataFrame,
    col: Option<&str>,
) -> PolarsResult<Option<&'a Float64Chunked>> {
    match col {
        Some(c) => Ok(Some(df.column(c)?.as_materialized_series().f64()?)),
        None => Ok(None),
    }
}

// ============================================================================
// Two-sample / k-sample data preparation helper
// ============================================================================

/// Prepare data arrays for two-sample / k-sample tests.
///
/// For domain estimation: zeros weights for non-domain obs but keeps all rows
/// so strata/PSU structure is preserved for correct variance estimation.
fn prepare_two_sample_data(
    df: &DataFrame,
    y_col: &str,
    w_col: &str,
    g_col: &str,
    _strata_col: Option<&str>,
    _psu_col: Option<&str>,
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
    let g_str    = g_series.cast(&DataType::String)?;
    let g_ca     = g_str.str()?;

    let domain_mask: Option<Vec<bool>> = match (domain_col, domain_val) {
        (Some(d_col), Some(d_val)) => {
            let d_series = df.column(d_col)?.as_materialized_series();
            let d_str    = d_series.cast(&DataType::String)?;
            let d_ca     = d_str.str()?;
            Some(d_ca.iter().map(|opt| opt.map_or(false, |v| v == d_val)).collect())
        }
        _ => None,
    };

    // Collect unique group labels (sorted), considering domain
    let mut level_set: Vec<String> = Vec::new();
    {
        let mut seen = std::collections::HashSet::new();
        for i in 0..n {
            let in_domain = domain_mask.as_ref().map_or(true, |m| m[i]);
            let w_val     = w_ca.get(i).unwrap_or(0.0);
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

    let mut label_to_idx: HashMap<String, u32> = HashMap::new();
    for (i, label) in level_set.iter().enumerate() {
        label_to_idx.insert(label.clone(), i as u32);
    }

    let mut y_arr = Vec::with_capacity(n);
    let mut w_arr = Vec::with_capacity(n);
    let mut g_arr = Vec::with_capacity(n);

    for i in 0..n {
        let yi        = y_ca.get(i).unwrap_or(f64::NAN);
        let wi        = w_ca.get(i).unwrap_or(0.0);
        let in_domain = domain_mask.as_ref().map_or(true, |m| m[i]);
        let label     = g_ca.get(i).unwrap_or("__NULL__");
        let gi        = label_to_idx.get(label).copied().unwrap_or(0);

        if yi.is_nan() {
            // NaN y: zero everything
            y_arr.push(0.0);
            w_arr.push(0.0);
            g_arr.push(0);
        } else {
            // Keep original y for non-domain rows — rank must be computed on
            // full dataset (matching R's subset() behavior), only weight is zeroed
            y_arr.push(yi);
            w_arr.push(if in_domain { wi } else { 0.0 });
            g_arr.push(gi);
        }
    }

    Ok((y_arr, w_arr, g_arr, level_set, n))
}

// ============================================================================
// Generic by-group helper
// ============================================================================

/// Run a single-test function for each level of `by_col`, combining with an
/// existing domain mask if present.  Returns a vertically stacked DataFrame
/// with a `by_col` column appended.
fn compute_by_groups<F>(
    df: &DataFrame,
    by_col: &str,
    domain_col: Option<&str>,
    domain_val: Option<&str>,
    run_single: F,
) -> PolarsResult<DataFrame>
where
    F: Fn(&DataFrame, Option<&str>, Option<&str>) -> PolarsResult<DataFrame>,
{
    let by_series = df.column(by_col)?.as_materialized_series();
    let by_str    = by_series.cast(&DataType::String)?;
    let by_ca     = by_str.str()?;
    let mut by_levels: Vec<String> = by_ca
        .unique()?
        .iter()
        .filter_map(|v| v.map(|s| s.to_string()))
        .collect();
    by_levels.sort();

    let mut result_dfs: Vec<DataFrame> = Vec::new();

    for level in &by_levels {
        let by_mask: BooleanChunked = by_ca.equal(level.as_str());

        let combined_mask: BooleanChunked =
            if let (Some(d_col), Some(d_val)) = (domain_col, domain_val) {
                let d_series = df.column(d_col)?.as_materialized_series();
                let d_str    = d_series.cast(&DataType::String)?;
                let d_ca     = d_str.str()?;
                let d_mask   = d_ca.equal(d_val);
                (&by_mask) & (&d_mask)
            } else {
                by_mask
            };

        // Build the domain mask column without cloning the full DataFrame.
        // DataFrame::with_column requires ownership so we use hstack on a
        // single-column DataFrame — avoids copying all existing columns.
        let mask_col: Series = combined_mask
            .iter()
            .map(|v| if v.unwrap_or(false) { "true" } else { "false" })
            .collect::<StringChunked>()
            .with_name("__svy_by_domain__".into())
            .into_series();

        // hstack adds columns without touching existing data; clone is shallow (Arc)
        let temp_df = df.hstack(&[mask_col.into()])?;

        let mut single = run_single(&temp_df, Some("__svy_by_domain__"), Some("true"))?;
        let with_by    = single
            .with_column(Series::new(by_col.into(), vec![level.as_str()]))?
            .clone();

        result_dfs.push(with_by);
    }

    if result_dfs.is_empty() {
        run_single(df, domain_col, domain_val)
    } else {
        let mut acc = result_dfs.remove(0);
        for r in result_dfs {
            acc = acc.vstack(&r)?;
        }
        Ok(acc)
    }
}

// ============================================================================
// T-test
// ============================================================================

#[pyfunction]
#[pyo3(signature = (
    data, y_col, weight_col, group_col=None,
    strata_col=None, psu_col=None, ssu_col=None,
    fpc_col=None, fpc_ssu_col=None, singleton_method=None,
    null_value=0.0, domain_col=None, domain_val=None, by_col=None,
))]
pub fn ttest_rs(
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
    by_col: Option<String>,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let result = compute_svyttest(
        &df,
        &y_col, &weight_col,
        group_col.as_deref(),
        strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(),
        fpc_col.as_deref(), fpc_ssu_col.as_deref(),
        singleton_method.as_deref(),
        null_value,
        domain_col.as_deref(), domain_val.as_deref(),
        by_col.as_deref(),
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(PyDataFrame(result))
}

fn compute_svyttest(
    df: &DataFrame,
    y_col: &str, weight_col: &str,
    group_col: Option<&str>,
    strata_col: Option<&str>, psu_col: Option<&str>, ssu_col: Option<&str>,
    fpc_col: Option<&str>, fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>,
    null_value: f64,
    domain_col: Option<&str>, domain_val: Option<&str>,
    by_col: Option<&str>,
) -> PolarsResult<DataFrame> {
    match by_col {
        None => compute_svyttest_single(
            df, y_col, weight_col, group_col, strata_col, psu_col, ssu_col,
            fpc_col, fpc_ssu_col, singleton_method, null_value, domain_col, domain_val,
        ),
        Some(by) => compute_by_groups(df, by, domain_col, domain_val, |temp_df, d_col, d_val| {
            compute_svyttest_single(
                temp_df, y_col, weight_col, group_col, strata_col, psu_col, ssu_col,
                fpc_col, fpc_ssu_col, singleton_method, null_value, d_col, d_val,
            )
        }),
    }
}

fn compute_svyttest_single(
    df: &DataFrame,
    y_col: &str, weight_col: &str,
    group_col: Option<&str>,
    strata_col: Option<&str>, psu_col: Option<&str>, ssu_col: Option<&str>,
    fpc_col: Option<&str>, fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>,
    null_value: f64,
    domain_col: Option<&str>, domain_val: Option<&str>,
) -> PolarsResult<DataFrame> {
    let strata  = get_opt_str(df, strata_col)?;
    let psu     = get_opt_str(df, psu_col)?;
    let ssu     = get_opt_str(df, ssu_col)?;
    let fpc     = get_opt_f64(df, fpc_col)?;
    let fpc_ssu = get_opt_f64(df, fpc_ssu_col)?;

    match group_col {
        None => {
            // One-sample test
            let y       = df.column(y_col)?.as_materialized_series().f64()?;
            let weights = df.column(weight_col)?.as_materialized_series().f64()?;

            let res = if let (Some(d_col), Some(d_val)) = (domain_col, domain_val) {
                let d_series = df.column(d_col)?.as_materialized_series();
                let d_str    = d_series.cast(&DataType::String)?;
                let d_ca     = d_str.str()?;
                let mask     = d_ca.equal(d_val);
                ttest_one_sample_domain(
                    y, weights, &mask, strata, psu, ssu, fpc, fpc_ssu,
                    singleton_method, null_value,
                )?
            } else {
                ttest_one_sample(
                    y, weights, strata, psu, ssu, fpc, fpc_ssu,
                    singleton_method, null_value,
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
                &y_arr, &g_arr, &w_arr, n, strata, psu, ssu, fpc, fpc_ssu,
                singleton_method, levels.clone(), null_value,
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
// Rank test
// ============================================================================

#[pyfunction]
#[pyo3(signature = (
    data, y_col, group_col, weight_col,
    strata_col=None, psu_col=None, ssu_col=None,
    fpc_col=None, fpc_ssu_col=None, score_method=None,
    singleton_method=None, domain_col=None, domain_val=None, by_col=None,
))]
pub fn ranktest_rs(
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
    by_col: Option<String>,
) -> PyResult<PyDataFrame> {
    let df: DataFrame = data.into();
    let result = compute_svyranktest(
        &df,
        &y_col, &group_col, &weight_col,
        strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(),
        fpc_col.as_deref(), fpc_ssu_col.as_deref(),
        score_method.as_deref(), singleton_method.as_deref(),
        domain_col.as_deref(), domain_val.as_deref(),
        by_col.as_deref(),
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(PyDataFrame(result))
}

fn compute_svyranktest(
    df: &DataFrame,
    y_col: &str, group_col: &str, weight_col: &str,
    strata_col: Option<&str>, psu_col: Option<&str>, ssu_col: Option<&str>,
    fpc_col: Option<&str>, fpc_ssu_col: Option<&str>,
    score_method: Option<&str>, singleton_method: Option<&str>,
    domain_col: Option<&str>, domain_val: Option<&str>,
    by_col: Option<&str>,
) -> PolarsResult<DataFrame> {
    match by_col {
        None => compute_svyranktest_single(
            df, y_col, group_col, weight_col, strata_col, psu_col, ssu_col,
            fpc_col, fpc_ssu_col, score_method, singleton_method, domain_col, domain_val,
        ),
        Some(by) => compute_by_groups(df, by, domain_col, domain_val, |temp_df, d_col, d_val| {
            compute_svyranktest_single(
                temp_df, y_col, group_col, weight_col, strata_col, psu_col, ssu_col,
                fpc_col, fpc_ssu_col, score_method, singleton_method, d_col, d_val,
            )
        }),
    }
}

fn compute_svyranktest_single(
    df: &DataFrame,
    y_col: &str, group_col: &str, weight_col: &str,
    strata_col: Option<&str>, psu_col: Option<&str>, ssu_col: Option<&str>,
    fpc_col: Option<&str>, fpc_ssu_col: Option<&str>,
    score_method: Option<&str>, singleton_method: Option<&str>,
    domain_col: Option<&str>, domain_val: Option<&str>,
) -> PolarsResult<DataFrame> {
    let method = score_method
        .and_then(RankScoreMethod::from_str)
        .unwrap_or(RankScoreMethod::Wilcoxon);

    let method_name = match method {
        RankScoreMethod::Wilcoxon | RankScoreMethod::KruskalWallis => "KruskalWallis",
        RankScoreMethod::VanDerWaerden => "vanderWaerden",
        RankScoreMethod::Median => "median",
    };

    let strata  = get_opt_str(df, strata_col)?;
    let psu     = get_opt_str(df, psu_col)?;
    let ssu     = get_opt_str(df, ssu_col)?;
    let fpc     = get_opt_f64(df, fpc_col)?;
    let fpc_ssu = get_opt_f64(df, fpc_ssu_col)?;

    let (y_arr, w_arr, g_arr, levels, n) = prepare_two_sample_data(
        df, y_col, weight_col, group_col, strata_col, psu_col, domain_col, domain_val,
    )?;
    let n_groups = levels.len();

    if n_groups == 2 {
        let res = ranktest_two_sample(
            &y_arr, &g_arr, &w_arr, n, strata, psu, ssu, fpc, fpc_ssu,
            method, singleton_method, levels,
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
            &y_arr, &g_arr, &w_arr, n, n_groups, strata, psu, ssu, fpc,
            fpc_ssu, method, singleton_method, levels,
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
// Tabulation
// ============================================================================

#[pyfunction]
#[pyo3(signature = (
    data, rowvar_col, weight_col,
    colvar_col=None, strata_col=None, psu_col=None, ssu_col=None,
    fpc_col=None, fpc_ssu_col=None, singleton_method=None,
    compute_totals=false,
))]
pub fn tabulate_rs(
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
        &rowvar_col, &weight_col,
        colvar_col.as_deref(),
        strata_col.as_deref(), psu_col.as_deref(), ssu_col.as_deref(),
        fpc_col.as_deref(), fpc_ssu_col.as_deref(),
        singleton_method.as_deref(),
        compute_totals,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok((PyDataFrame(result.0), PyDataFrame(result.1)))
}

fn compute_tabulate(
    df: &DataFrame,
    rowvar_col: &str, weight_col: &str,
    colvar_col: Option<&str>,
    strata_col: Option<&str>, psu_col: Option<&str>, ssu_col: Option<&str>,
    fpc_col: Option<&str>, fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>,
    compute_totals: bool,
) -> PolarsResult<(DataFrame, DataFrame)> {
    let weights = df.column(weight_col)?.as_materialized_series().f64()?;
    let strata = strata_col
        .map(|c| df.column(c).and_then(|s| Ok(s.as_materialized_series().str()?.clone())))
        .transpose()?;
    let psu = psu_col
        .map(|c| df.column(c).and_then(|s| Ok(s.as_materialized_series().str()?.clone())))
        .transpose()?;
    let ssu = ssu_col
        .map(|c| df.column(c).and_then(|s| Ok(s.as_materialized_series().str()?.clone())))
        .transpose()?;
    let fpc = fpc_col
        .map(|c| df.column(c).and_then(|s| Ok(s.as_materialized_series().f64()?.clone())))
        .transpose()?;
    let fpc_ssu = fpc_ssu_col
        .map(|c| df.column(c).and_then(|s| Ok(s.as_materialized_series().f64()?.clone())))
        .transpose()?;

    let is_two_way = colvar_col.is_some();
    let n_obs = df.height();

    let rowvar_series = df.column(rowvar_col)?.as_materialized_series();
    let rowvar_str    = rowvar_series.cast(&DataType::String)?;
    let rowvar_ca     = rowvar_str.str()?;

    let combined_key: StringChunked;
    let y_effective: &StringChunked;

    if let Some(cv_col) = colvar_col {
        let colvar_series = df.column(cv_col)?.as_materialized_series();
        let colvar_str    = colvar_series.cast(&DataType::String)?;
        let colvar_ca     = colvar_str.str()?;

        let keys: Vec<Option<String>> = rowvar_ca
            .iter()
            .zip(colvar_ca.iter())
            .map(|(a, b)| match (a, b) {
                (Some(va), Some(vb)) => Some(format!("{}__by__{}", va, vb)),
                _ => None,
            })
            .collect();
        combined_key = StringChunked::from_iter(keys.into_iter());
        y_effective  = &combined_key;
    } else {
        combined_key = rowvar_ca.clone();
        y_effective  = &combined_key;
    }

    let (levels, proportions, ses, cov_matrix, deff_vec, df_val) = estimate_proportions(
        y_effective, weights,
        strata.as_ref(), psu.as_ref(), ssu.as_ref(), fpc.as_ref(), fpc_ssu.as_ref(),
        singleton_method,
    )?;

    let k = levels.len();
    let mut rowvars: Vec<String> = Vec::with_capacity(k);
    let mut colvars: Vec<String> = Vec::with_capacity(k);
    for lvl in &levels {
        if is_two_way {
            match lvl.split_once("__by__") {
                Some((r, c)) => { rowvars.push(r.to_string()); colvars.push(c.to_string()); }
                None         => { rowvars.push(lvl.clone());   colvars.push(String::new()); }
            }
        } else {
            rowvars.push(lvl.clone());
            colvars.push(String::new());
        }
    }

    let (est_vals, se_vals) = if compute_totals {
        estimate_totals(
            y_effective, weights,
            strata.as_ref(), psu.as_ref(), ssu.as_ref(), fpc.as_ref(), fpc_ssu.as_ref(),
            singleton_method, &levels,
        )?
    } else {
        (proportions.clone(), ses.clone())
    };

    let cv_vals: Vec<f64> = est_vals.iter().zip(se_vals.iter())
        .map(|(&e, &s)| if e.abs() > 0.0 { (s / e).abs() } else { f64::NAN })
        .collect();

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

    let stats_df = if is_two_way {
        let cv_col        = colvar_col.unwrap();
        let colvar_series = df.column(cv_col)?.as_materialized_series();
        let colvar_str2   = colvar_series.cast(&DataType::String)?;
        let colvar_ca2    = colvar_str2.str()?;

        let mut row_levels: Vec<String> = rowvar_ca.unique()?.iter()
            .filter_map(|v| v.map(|s| s.to_string())).collect();
        sort_levels(&mut row_levels);
        let mut col_levels: Vec<String> = colvar_ca2.unique()?.iter()
            .filter_map(|v| v.map(|s| s.to_string())).collect();
        sort_levels(&mut col_levels);

        let nr = row_levels.len();
        let nc = col_levels.len();

        let mut prop_ordered = vec![0.0; nr * nc];
        let mut cov_ordered  = vec![vec![0.0; nr * nc]; nr * nc];

        let mut level_to_ordered: HashMap<String, usize> = HashMap::new();
        for (ri, rl) in row_levels.iter().enumerate() {
            for (ci, cl) in col_levels.iter().enumerate() {
                level_to_ordered.insert(format!("{}__by__{}", rl, cl), ri * nc + ci);
            }
        }

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
            count_strata_psus(strata.as_ref(), psu.as_ref(), n_obs);

        let (
            p_chisq, p_df, p_p, p_adj_f, p_adj_ndf, p_adj_ddf, p_adj_p,
        ) = rao_scott(&prop_ordered, &cov_ordered, nr, nc, n_obs, n_strata_count, n_psu_count);

        df![
            "stat" => vec!["chisq", "f"],
            "value" => vec![p_chisq, p_adj_f],
            "df" => vec![p_df, p_adj_ndf],
            "df2" => vec![f64::NAN, p_adj_ddf],
            "p_value" => vec![p_p, p_adj_p],
        ]?
    } else {
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
