use super::*;

#[allow(clippy::too_many_arguments)]
pub fn compute_ratio_ungrouped(
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
    domain_rows_col: Option<&str>,
    srs: SrsRef,
    calib: Option<CalibSweep>,
) -> EResult<DataFrame> {
    let y = df.column(numerator_col)?.f64()?;
    let x = df.column(denominator_col)?.f64()?;
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

    let (variance, df_val, (estimate, srs_var)) = ungrouped_estimate(
        weights,
        strata,
        psu,
        ssu,
        fpc,
        fpc_ssu,
        singleton_method,
        rows.as_ref(),
        calib,
        || {
            let Ok(estimate) = point_estimate_ratio(y, x, weights) else {
                return Ok((vec![0.0; y.len()], (f64::NAN, f64::NAN)));
            };
            let scores = scores_to_arr(&scores_ratio(y, x, weights)?);
            let srs_var = srs_variance_ratio(y, x, weights, srs)?;
            Ok((scores, (estimate, srs_var)))
        },
    )?;

    let (variance, se) = var_se(estimate, variance);
    let n = active_count(weights, None);
    let deff = if srs_var > 0.0 {
        variance / srs_var
    } else {
        f64::NAN
    };

    Ok(
        df!["y" => vec![numerator_col], "x" => vec![denominator_col], "est" => vec![estimate],
        "se" => vec![se], "var" => vec![variance], "df" => vec![df_val], "n" => vec![n], "deff" => vec![deff]]?,
    )
}

/// Batched ungrouped ratios, as `compute_mean_multi`: one row per (numerator, denominator) pair.
#[allow(clippy::too_many_arguments)]
pub fn compute_ratio_multi(
    df: &DataFrame,
    numerator_cols: &[String],
    denominator_cols: &[String],
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

    // Design indexed once, df taken off its codes (see `compute_mean_multi`).
    let design = build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
    let df_val = degrees_of_freedom_from_design(weights, &design, None);
    let active = design.domain_rows(weights.len(), rows.as_ref(), None);

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
        .map(|i| -> EResult<(String, String, f64, f64, f64, u32, f64)> {
            let y = y_cols[i];
            let x = x_cols[i];
            let n = active_count(weights, None);
            let Ok(estimate) = point_estimate_ratio(y, x, weights) else {
                let nan = f64::NAN;
                return Ok((
                    numerator_cols[i].clone(),
                    denominator_cols[i].clone(),
                    nan,
                    nan,
                    nan,
                    n,
                    nan,
                ));
            };
            let scores = scores_ratio(y, x, weights)?;
            let scores_arr: Vec<f64> = scores.iter().map(|s| s.unwrap_or(0.0)).collect();
            let variance = taylor_variance_apply_in(&scores_arr, &design, active.as_deref());
            let se = variance.max(0.0).sqrt();
            let srs_var = srs_variance_ratio(y, x, weights, srs)?;
            let deff = if srs_var > 0.0 {
                variance / srs_var
            } else {
                f64::NAN
            };
            Ok((
                numerator_cols[i].clone(),
                denominator_cols[i].clone(),
                estimate,
                se,
                variance,
                n,
                deff,
            ))
        })
        .collect::<EResult<Vec<_>>>()?;

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
    Ok(
        df!["y" => ys, "x" => xs, "est" => estimates, "se" => ses, "var" => variances,
        "df" => dfs, "n" => ns, "deff" => deffs]?,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn compute_ratio_grouped(
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
    domain_rows_col: Option<&str>,
    srs: SrsRef,
    calib: Option<CalibSweep>,
) -> EResult<(DataFrame, Vec<f64>)> {
    let y = df.column(numerator_col)?.f64()?;
    let x = df.column(denominator_col)?.f64()?;
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
    let by_str = df.column(by_col)?.str()?;
    // Group discovery and the design build are independent full passes.
    let (unique_groups, design) = rayon::join(
        || crate::estimation::sorted_groups(by_str),
        || build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method),
    );
    let unique_groups = unique_groups?;
    let design = design?.with_calib(calib);

    let groups: Vec<&str> = unique_groups.iter().flatten().collect();
    // A by-group is a domain: its df counts its own active PSUs/strata, not the frame's.
    let masks: Vec<BooleanChunked> = groups.par_iter().map(|&g| by_str.equal(g)).collect();
    let (group_dfs, rows) = rayon::join(
        || {
            masks
                .par_iter()
                .map(|m| degrees_of_freedom_from_design(weights, &design, Some(m)))
                .collect::<Vec<u32>>()
        },
        || {
            groups
                .par_iter()
                .zip(masks.par_iter())
                .map(|(&group, mask)| -> EResult<GroupRow<'_>> {
                    let domain_mask = mask.clone();
                    let active =
                        design.domain_rows(weights.len(), rows.as_ref(), Some(&domain_mask));
                    let n_domain = active_count(weights, Some(&domain_mask));
                    let Ok(estimate) = point_estimate_ratio_domain(y, x, weights, &domain_mask)
                    else {
                        let nan = f64::NAN;
                        return Ok((group, nan, nan, nan, n_domain, nan, vec![0.0; y.len()]));
                    };
                    let scores = scores_ratio_domain(y, x, weights, &domain_mask)?;
                    let scores_arr: Vec<f64> = scores.iter().map(|s| s.unwrap_or(0.0)).collect();
                    let variance =
                        taylor_variance_apply_in(&scores_arr, &design, active.as_deref());
                    let se = variance.max(0.0).sqrt();
                    let srs_var = srs_variance_ratio_domain(y, x, weights, &domain_mask, srs)?;
                    let deff = if srs_var > 0.0 {
                        variance / srs_var
                    } else {
                        f64::NAN
                    };
                    Ok((group, estimate, se, variance, n_domain, deff, scores_arr))
                })
                .collect::<EResult<Vec<_>>>()
        },
    );
    let rows = rows?;

    let n_groups = rows.len();
    let mut by_vals: Vec<&str> = Vec::with_capacity(n_groups);
    let mut estimates: Vec<f64> = Vec::with_capacity(n_groups);
    let mut ses: Vec<f64> = Vec::with_capacity(n_groups);
    let mut variances: Vec<f64> = Vec::with_capacity(n_groups);
    let mut ns: Vec<u32> = Vec::with_capacity(n_groups);
    let mut deffs: Vec<f64> = Vec::with_capacity(n_groups);
    let mut score_cols: Vec<Vec<f64>> = Vec::with_capacity(n_groups);
    for (g, est, se, var, n, deff, scores_arr) in rows {
        by_vals.push(g);
        estimates.push(est);
        ses.push(se);
        variances.push(var);
        ns.push(n);
        deffs.push(deff);
        score_cols.push(scores_arr);
    }
    let cov = flat_cov(&score_cols, &design, &estimates);
    let dfs = group_dfs;
    let out = df![by_col => by_vals, "y" => vec![numerator_col; n_groups], "x" => vec![denominator_col; n_groups],
        "est" => estimates, "se" => ses, "var" => variances, "df" => dfs, "n" => ns, "deff" => deffs]?;
    Ok((out, cov))
}
