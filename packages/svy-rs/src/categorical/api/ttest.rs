use super::*;

#[pyfunction]
#[pyo3(signature = (
    data, y_col, weight_col, group_col=None,
    strata_col=None, psu_col=None, ssu_col=None,
    fpc_col=None, fpc_ssu_col=None, singleton_method=None,
    null_value=0.0, domain_col=None, domain_val=None, by_col=None,
    calib_kind=None, calib_cells=None, calib_aux=None, calib_prev_wgt=None,
    calib_pins_total=None, calib_new_wgt=None,
    rep_weight_cols=None, rep_coefs=None, rep_df=None, variance_center="rep_mean"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ttest_rs(
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
    calib_kind: Option<String>,
    calib_cells: Option<Vec<String>>,
    calib_aux: Option<Vec<String>>,
    calib_prev_wgt: Option<String>,
    calib_pins_total: Option<bool>,
    calib_new_wgt: Option<String>,
    rep_weight_cols: Option<Vec<String>>,
    rep_coefs: Option<Vec<f64>>,
    rep_df: Option<f64>,
    variance_center: &str,
) -> PyResult<PyDataFrame> {
    let df = into_contiguous(data);
    let rep = rep_spec(&df, rep_weight_cols, rep_coefs, rep_df, variance_center)?;
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
        by_col.as_deref(),
        calib.as_ref(),
        rep.as_ref(),
    )
    .map_err(|e| to_py_err(&e))?;
    Ok(PyDataFrame(result))
}

#[allow(clippy::too_many_arguments)]
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
    by_col: Option<&str>,
    calib: Option<&CalibSweep>,
    rep: Option<&RepSpec>,
) -> CResult<DataFrame> {
    match by_col {
        None => compute_svyttest_single(
            df,
            y_col,
            weight_col,
            group_col,
            strata_col,
            psu_col,
            ssu_col,
            fpc_col,
            fpc_ssu_col,
            singleton_method,
            null_value,
            domain_col,
            domain_val,
            calib,
            rep,
            None,
        ),
        Some(by) => {
            let frame_levels = match group_col {
                Some(g_col) => {
                    prepare_two_sample_data(
                        df, y_col, weight_col, g_col, strata_col, psu_col, domain_col, domain_val,
                    )?
                    .3
                }
                None => Vec::new(),
            };
            compute_by_groups(df, by, domain_col, domain_val, |temp_df, d_col, d_val| {
                compute_svyttest_single(
                    temp_df,
                    y_col,
                    weight_col,
                    group_col,
                    strata_col,
                    psu_col,
                    ssu_col,
                    fpc_col,
                    fpc_ssu_col,
                    singleton_method,
                    null_value,
                    d_col,
                    d_val,
                    calib,
                    rep,
                    Some(&frame_levels),
                )
            })
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_svyttest_single(
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
    calib: Option<&CalibSweep>,
    rep: Option<&RepSpec>,
    // By-group only: the frame's levels, so a group lacking weight or a level yields a NaN row, not an error.
    frame_levels: Option<&[String]>,
) -> CResult<DataFrame> {
    let strata = get_opt_col(df, strata_col)?;
    let psu = get_opt_col(df, psu_col)?;
    let ssu = get_opt_col(df, ssu_col)?;
    let fpc = get_opt_f64(df, fpc_col)?;
    let fpc_ssu = get_opt_f64(df, fpc_ssu_col)?;
    let nan = f64::NAN;

    match group_col {
        None => {
            let y = df.column(y_col)?.as_materialized_series().f64()?;
            let weights = df.column(weight_col)?.as_materialized_series().f64()?;
            let mask = match (domain_col, domain_val) {
                (Some(d_col), Some(d_val)) => Some(
                    df.column(d_col)?
                        .as_materialized_series()
                        .cast(&DataType::String)?
                        .str()?
                        .equal(d_val),
                ),
                _ => None,
            };

            if frame_levels.is_some() && !has_positive_weight(y, weights, mask.as_ref()) {
                return Ok(df![
                    "type" => vec!["one-sample"],
                    "y" => vec![y_col],
                    "estimate" => vec![nan],
                    "diff" => vec![nan],
                    "se" => vec![nan],
                    "t" => vec![nan],
                    "df" => vec![nan],
                    "p_value" => vec![nan],
                    "n" => vec![0u32],
                ]?);
            }

            let res = if let Some(spec) = rep {
                ttest_one_sample_rep(y, weights, mask.as_ref(), null_value, spec)?
            } else if let Some(mask) = mask.as_ref() {
                ttest_one_sample_domain(
                    y,
                    weights,
                    mask,
                    strata,
                    psu,
                    ssu,
                    fpc,
                    fpc_ssu,
                    singleton_method,
                    calib,
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
                    calib,
                    null_value,
                )?
            };

            Ok(df![
                "type" => vec!["one-sample"],
                "y" => vec![y_col],
                "estimate" => vec![res.estimate],
                "diff" => vec![res.diff],
                "se" => vec![res.se_diff],
                "t" => vec![res.t_stat],
                "df" => vec![res.df],
                "p_value" => vec![res.p_value],
                "n" => vec![res.n_obs as u32],
            ]?)
        }
        Some(g_col) => {
            let (y_arr, w_arr, g_arr, levels, n, rows) = prepare_two_sample_data(
                df, y_col, weight_col, g_col, strata_col, psu_col, domain_col, domain_val,
            )?;

            if let Some(frame) = frame_levels
                && levels.len() != 2
                && frame.len() == 2
            {
                let counts = level_counts(&w_arr, &g_arr, &levels, frame);
                return Ok(df![
                    "type" => vec!["two-sample"],
                    "y" => vec![y_col],
                    "group" => vec![g_col],
                    "level_0" => vec![frame[0].as_str()],
                    "level_1" => vec![frame[1].as_str()],
                    "diff" => vec![nan],
                    "se" => vec![nan],
                    "t" => vec![nan],
                    "df" => vec![nan],
                    "p_value" => vec![nan],
                    "mean_0" => vec![nan],
                    "mean_1" => vec![nan],
                    "se_0" => vec![nan],
                    "se_1" => vec![nan],
                    "n" => vec![counts.iter().sum::<u32>()],
                    "n_0" => vec![counts[0]],
                    "n_1" => vec![counts[1]],
                ]?);
            }

            if levels.len() != 2 {
                return Err(CategoricalError::TwoGroups(levels.len()));
            }

            let res = match rep {
                Some(spec) => ttest_two_sample_rep(
                    &y_arr,
                    &g_arr,
                    &w_arr,
                    levels.clone(),
                    null_value,
                    &rows,
                    spec,
                )?,
                None => ttest_two_sample(
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
                    calib,
                    levels.clone(),
                    null_value,
                    &rows,
                )?,
            };

            Ok(df![
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
                "n_0" => vec![res.group_ns[0] as u32],
                "n_1" => vec![res.group_ns[1] as u32],
            ]?)
        }
    }
}
