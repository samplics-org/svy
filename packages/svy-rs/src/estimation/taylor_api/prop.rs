use super::*;

/// A categorical level's share of the (domain) weight, or its estimated count.
#[derive(Clone, Copy)]
enum LevelScale {
    Share,
    Count,
}

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, by_col=None, singleton_method=None, domain_rows_col=None, deff_ref=None, deff_pop_total=None, calib_kind=None, calib_cells=None, calib_aux=None, calib_prev_wgt=None, calib_pins_total=None, calib_new_wgt=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn taylor_prop(
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
    domain_rows_col: Option<String>,
    deff_ref: Option<String>,
    deff_pop_total: Option<f64>,
    calib_kind: Option<String>,
    calib_cells: Option<Vec<String>>,
    calib_aux: Option<Vec<String>>,
    calib_prev_wgt: Option<String>,
    calib_pins_total: Option<bool>,
    calib_new_wgt: Option<String>,
) -> PyResult<(PyDataFrame, Option<Vec<f64>>)> {
    taylor_levels(
        _py,
        data,
        value_col,
        weight_col,
        strata_col,
        psu_col,
        ssu_col,
        fpc_col,
        fpc_ssu_col,
        by_col,
        singleton_method,
        domain_rows_col,
        deff_ref,
        deff_pop_total,
        calib_kind,
        calib_cells,
        calib_aux,
        calib_prev_wgt,
        calib_pins_total,
        calib_new_wgt,
        LevelScale::Share,
    )
}

/// Count per level (R `svytotal(~factor(y))`) with the joint covariance across levels and groups.
#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, by_col=None, singleton_method=None, domain_rows_col=None, deff_ref=None, deff_pop_total=None, calib_kind=None, calib_cells=None, calib_aux=None, calib_prev_wgt=None, calib_pins_total=None, calib_new_wgt=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn taylor_factor_total(
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
    domain_rows_col: Option<String>,
    deff_ref: Option<String>,
    deff_pop_total: Option<f64>,
    calib_kind: Option<String>,
    calib_cells: Option<Vec<String>>,
    calib_aux: Option<Vec<String>>,
    calib_prev_wgt: Option<String>,
    calib_pins_total: Option<bool>,
    calib_new_wgt: Option<String>,
) -> PyResult<(PyDataFrame, Option<Vec<f64>>)> {
    taylor_levels(
        _py,
        data,
        value_col,
        weight_col,
        strata_col,
        psu_col,
        ssu_col,
        fpc_col,
        fpc_ssu_col,
        by_col,
        singleton_method,
        domain_rows_col,
        deff_ref,
        deff_pop_total,
        calib_kind,
        calib_cells,
        calib_aux,
        calib_prev_wgt,
        calib_pins_total,
        calib_new_wgt,
        LevelScale::Count,
    )
}

#[allow(clippy::too_many_arguments)]
fn taylor_levels(
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
    domain_rows_col: Option<String>,
    deff_ref: Option<String>,
    deff_pop_total: Option<f64>,
    calib_kind: Option<String>,
    calib_cells: Option<Vec<String>>,
    calib_aux: Option<Vec<String>>,
    calib_prev_wgt: Option<String>,
    calib_pins_total: Option<bool>,
    calib_new_wgt: Option<String>,
    scale: LevelScale,
) -> PyResult<(PyDataFrame, Option<Vec<f64>>)> {
    let df = into_contiguous(data);
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
    let srs = parse_srs_ref(deff_ref.as_deref(), deff_pop_total).map_err(|e| to_py_err(&e))?;
    let Some(by) = by_col else {
        let (result, cov) = compute_levels_ungrouped(
            &df,
            &value_col,
            &weight_col,
            strata_col.as_deref(),
            psu_col.as_deref(),
            ssu_col.as_deref(),
            fpc_col.as_deref(),
            fpc_ssu_col.as_deref(),
            singleton_method.as_deref(),
            domain_rows_col.as_deref(),
            srs,
            calib,
            scale,
        )
        .map_err(|e| to_py_err(&e))?;
        return Ok((PyDataFrame(result), Some(cov)));
    };
    let (result, cov) = _py
        .detach(|| {
            compute_levels_grouped(
                &df,
                &value_col,
                &weight_col,
                strata_col.as_deref(),
                psu_col.as_deref(),
                ssu_col.as_deref(),
                fpc_col.as_deref(),
                fpc_ssu_col.as_deref(),
                &by,
                singleton_method.as_deref(),
                domain_rows_col.as_deref(),
                srs,
                calib,
                scale,
            )
        })
        .map_err(|e| to_py_err(&e))?;
    Ok((PyDataFrame(result), Some(cov)))
}

/// Batched ungrouped proportions over many category columns; rows (variable, level) in input order.
#[pyfunction]
#[pyo3(signature = (data, value_cols, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, singleton_method=None, domain_rows_col=None, deff_ref=None, deff_pop_total=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn taylor_prop_multi(
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
    domain_rows_col: Option<String>,
    deff_ref: Option<String>,
    deff_pop_total: Option<f64>,
) -> PyResult<PyDataFrame> {
    let df = into_contiguous(data);
    let srs = parse_srs_ref(deff_ref.as_deref(), deff_pop_total).map_err(|e| to_py_err(&e))?;
    let result = _py
        .detach(|| {
            compute_prop_multi(
                &df,
                &value_cols,
                &weight_col,
                strata_col.as_deref(),
                psu_col.as_deref(),
                ssu_col.as_deref(),
                fpc_col.as_deref(),
                fpc_ssu_col.as_deref(),
                singleton_method.as_deref(),
                domain_rows_col.as_deref(),
                srs,
            )
        })
        .map_err(|e| to_py_err(&e))?;
    Ok(PyDataFrame(result))
}

#[allow(clippy::too_many_arguments)]
fn compute_levels_ungrouped(
    df: &DataFrame,
    value_col: &str,
    weight_col: &str,
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>,
    domain_rows_col: Option<&str>,
    srs: SrsRef,
    calib: Option<CalibSweep>,
    scale: LevelScale,
) -> EResult<(DataFrame, Vec<f64>)> {
    let weights = df.column(weight_col)?.f64()?;
    let rows = rows_column(df, domain_rows_col)?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc = fpc_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;
    let fpc_ssu = fpc_ssu_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;

    let value_series = df.column(value_col)?;
    let value_str = value_series.cast(&DataType::String)?;
    let value_str = value_str.str()?;
    let mut levels: Vec<String> = value_str
        .unique()?
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
    let mut score_cols: Vec<Vec<f64>> = Vec::new();
    let n = active_count(weights, None);
    // One design for every level; df comes off its codes rather than a second densification.
    let design =
        build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method)?.with_calib(calib);
    let df_val = degrees_of_freedom_from_design(weights, &design, None);
    let active = design.domain_rows(weights.len(), rows.as_ref(), None);

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
        let (estimate, scores, srs_var) = match scale {
            LevelScale::Share => match point_estimate_mean(&indicator_ca, weights) {
                Ok(estimate) => (
                    estimate,
                    scores_mean(&indicator_ca, weights)?,
                    srs_variance_mean(&indicator_ca, weights, srs)?,
                ),
                Err(_) => (
                    f64::NAN,
                    Float64Chunked::full("scores".into(), 0.0, indicator_ca.len()),
                    f64::NAN,
                ),
            },
            LevelScale::Count => (
                point_estimate_total(&indicator_ca, weights)?,
                scores_total(&indicator_ca, weights)?,
                srs_variance_total(&indicator_ca, weights, srs)?,
            ),
        };
        let scores_arr: Vec<f64> = scores.iter().map(|s| s.unwrap_or(0.0)).collect();
        let (variance, se) = var_se(
            estimate,
            taylor_variance_apply_in(&scores_arr, &design, active.as_deref()),
        );
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
        score_cols.push(scores_arr);
    }
    let cov = flat_cov_in(&score_cols, &design, &estimates, active.as_deref());
    let n_levels = level_vals.len();
    let out = df!["y" => vec![value_col; n_levels], "level" => level_vals, "est" => estimates,
        "se" => ses, "var" => variances, "df" => dfs_vec, "n" => ns, "deff" => deffs]?;
    Ok((out, cov))
}

/// Batched ungrouped proportions, as `compute_mean_multi`; rows match `compute_levels_ungrouped`.
#[allow(clippy::too_many_arguments)]
fn compute_prop_multi(
    df: &DataFrame,
    value_cols: &[String],
    weight_col: &str,
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>,
    domain_rows_col: Option<&str>,
    srs: SrsRef,
) -> EResult<DataFrame> {
    let weights = df.column(weight_col)?.f64()?;
    let rows = rows_column(df, domain_rows_col)?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc = fpc_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;
    let fpc_ssu = fpc_ssu_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;

    let n = active_count(weights, None);
    // Design indexed once; df taken off its codes (see `compute_mean_multi`).
    let design = build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
    let df_val = degrees_of_freedom_from_design(weights, &design, None);
    let active = design.domain_rows(weights.len(), rows.as_ref(), None);

    // Cast to String and enumerate sorted levels before the parallel region; closures only read.
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
        .map(|vs| -> EResult<Vec<String>> {
            let mut lv: Vec<String> = vs
                .unique()?
                .iter()
                .filter_map(|v| v.map(|s| s.to_string()))
                .collect();
            lv.sort();
            Ok(lv)
        })
        .collect::<EResult<Vec<_>>>()?;

    type PropRow = (String, String, f64, f64, f64, u32, f64); // y, level, est, se, var, n, deff
    let per_var = (0..value_cols.len())
        .into_par_iter()
        .map(|i| -> EResult<Vec<PropRow>> {
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
                let Ok(estimate) = point_estimate_mean(&indicator_ca, weights) else {
                    let nan = f64::NAN;
                    out.push((value_cols[i].clone(), lvl.clone(), nan, nan, nan, n, nan));
                    continue;
                };
                let scores = scores_mean(&indicator_ca, weights)?;
                let scores_arr: Vec<f64> = scores.iter().map(|s| s.unwrap_or(0.0)).collect();
                let variance = taylor_variance_apply_in(&scores_arr, &design, active.as_deref());
                let se = variance.max(0.0).sqrt();
                let srs_var = srs_variance_mean(&indicator_ca, weights, srs)?;
                let deff = if srs_var > 0.0 {
                    variance / srs_var
                } else {
                    f64::NAN
                };
                out.push((
                    value_cols[i].clone(),
                    lvl.clone(),
                    estimate,
                    se,
                    variance,
                    n,
                    deff,
                ));
            }
            Ok(out)
        })
        .collect::<EResult<Vec<_>>>()?;

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
    Ok(
        df!["y" => ys, "level" => level_vals, "est" => estimates, "se" => ses,
        "var" => variances, "df" => dfs, "n" => ns, "deff" => deffs]?,
    )
}

#[allow(clippy::too_many_arguments)]
fn compute_levels_grouped(
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
    domain_rows_col: Option<&str>,
    srs: SrsRef,
    calib: Option<CalibSweep>,
    scale: LevelScale,
) -> EResult<(DataFrame, Vec<f64>)> {
    let weights = df.column(weight_col)?.f64()?;
    let rows = rows_column(df, domain_rows_col)?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc = fpc_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;
    let fpc_ssu = fpc_ssu_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;

    let value_series = df.column(value_col)?;
    let value_str = value_series.cast(&DataType::String)?;
    let value_str = value_str.str()?;
    let mut levels: Vec<String> = value_str
        .unique()?
        .iter()
        .filter_map(|v| v.map(|s| s.to_string()))
        .collect();
    levels.sort();

    let by_str = df.column(by_col)?.str()?;
    // Group discovery and the design build are independent full passes.
    let (unique_groups, design) = rayon::join(
        || crate::estimation::sorted_groups(by_str),
        || build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method),
    );
    let unique_groups = unique_groups?;

    // Design is identical across all (group, level) cells; index it once.
    let design = design?.with_calib(calib);

    // Groups emit level rows in `levels` order, flattened in group order.
    type PropRow = (String, String, f64, f64, f64, u32, f64, Vec<f64>);
    let groups: Vec<&str> = unique_groups.iter().flatten().collect();
    // A by-group is a domain: its df counts its own active PSUs/strata, not the frame's.
    let masks: Vec<BooleanChunked> = groups.par_iter().map(|&g| by_str.equal(g)).collect();
    let group_dfs: Vec<u32> = masks
        .par_iter()
        .map(|m| degrees_of_freedom_from_design(weights, &design, Some(m)))
        .collect();
    let indicators: Vec<Float64Chunked> = levels
        .par_iter()
        .map(|lvl| {
            let indicator: Vec<Option<f64>> = value_str
                .iter()
                .map(|v| match v {
                    Some(val) if val == lvl => Some(1.0),
                    Some(_) => Some(0.0),
                    None => None,
                })
                .collect();
            Float64Chunked::from_slice_options("indicator".into(), &indicator)
        })
        .collect();
    let per_group = groups
        .par_iter()
        .zip(masks.par_iter())
        .map(|(&group, mask)| -> EResult<Vec<PropRow>> {
            let domain_mask = mask.clone();
            let active = design.domain_rows(weights.len(), rows.as_ref(), Some(&domain_mask));
            let n_domain = active_count(weights, Some(&domain_mask));
            let mut out: Vec<PropRow> = Vec::with_capacity(levels.len());
            for (lvl, indicator_ca) in levels.iter().zip(&indicators) {
                let (estimate, scores, srs_var) = match scale {
                    LevelScale::Share => {
                        match point_estimate_mean_domain(indicator_ca, weights, &domain_mask) {
                            Ok(estimate) => (
                                estimate,
                                scores_mean_domain(indicator_ca, weights, &domain_mask)?,
                                srs_variance_mean_domain(indicator_ca, weights, &domain_mask, srs)?,
                            ),
                            Err(_) => (
                                f64::NAN,
                                Float64Chunked::full("scores".into(), 0.0, indicator_ca.len()),
                                f64::NAN,
                            ),
                        }
                    }
                    LevelScale::Count => (
                        point_estimate_total_domain(indicator_ca, weights, &domain_mask)?,
                        scores_total_domain(indicator_ca, weights, &domain_mask)?,
                        srs_variance_total_domain(indicator_ca, weights, &domain_mask, srs)?,
                    ),
                };
                let scores_arr: Vec<f64> = scores.iter().map(|s| s.unwrap_or(0.0)).collect();
                let (variance, se) = var_se(
                    estimate,
                    taylor_variance_apply_in(&scores_arr, &design, active.as_deref()),
                );
                let deff = if srs_var > 0.0 {
                    variance / srs_var
                } else {
                    f64::NAN
                };
                out.push((
                    group.to_string(),
                    lvl.clone(),
                    estimate,
                    se,
                    variance,
                    n_domain,
                    deff,
                    scores_arr,
                ));
            }
            Ok(out)
        })
        .collect::<EResult<Vec<_>>>()?;

    let mut by_vals: Vec<String> = Vec::new();
    let mut level_vals: Vec<String> = Vec::new();
    let mut estimates: Vec<f64> = Vec::new();
    let mut ses: Vec<f64> = Vec::new();
    let mut variances: Vec<f64> = Vec::new();
    let mut ns: Vec<u32> = Vec::new();
    let mut deffs: Vec<f64> = Vec::new();
    let mut score_cols: Vec<Vec<f64>> = Vec::new();
    for group_rows in per_group {
        for (g, lvl, est, se, var, n, deff, scores_arr) in group_rows {
            by_vals.push(g);
            level_vals.push(lvl);
            estimates.push(est);
            ses.push(se);
            variances.push(var);
            ns.push(n);
            deffs.push(deff);
            score_cols.push(scores_arr);
        }
    }
    let cov = flat_cov(&score_cols, &design, &estimates);
    let n_rows = by_vals.len();
    let dfs_vec: Vec<u32> = group_dfs
        .iter()
        .flat_map(|d| std::iter::repeat_n(*d, levels.len()))
        .collect();
    debug_assert_eq!(dfs_vec.len(), n_rows);
    let out = df![by_col => by_vals, "y" => vec![value_col; n_rows], "level" => level_vals,
        "est" => estimates, "se" => ses, "var" => variances, "df" => dfs_vec, "n" => ns, "deff" => deffs]?;
    Ok((out, cov))
}
