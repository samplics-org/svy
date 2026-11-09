use super::*;

/// Ungrouped single-estimate skeleton: design build overlaps the score pass, variance overlaps df.
/// Each half still sums in row order, so output is bit-identical at any thread count.
#[allow(clippy::too_many_arguments)]
pub(super) fn ungrouped_estimate<T: Send>(
    weights: &Float64Chunked,
    strata: Option<&Column>,
    psu: Option<&Column>,
    ssu: Option<&Column>,
    fpc: Option<&Float64Chunked>,
    fpc_ssu: Option<&Float64Chunked>,
    singleton_method: Option<&str>,
    rows: Option<&BooleanChunked>,
    calib: Option<CalibSweep>,
    value_work: impl FnOnce() -> EResult<(Vec<f64>, T)> + Send,
) -> EResult<(f64, u32, T)> {
    let (design_res, value_res) = rayon::join(
        || build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method),
        value_work,
    );
    let design = design_res?.with_calib(calib);
    let (scores, extra) = value_res?;

    let active = design.domain_rows(weights.len(), rows, None);
    let (variance, df_val) = rayon::join(
        || taylor_variance_apply_in(&scores, &design, active.as_deref()),
        || degrees_of_freedom_from_design(weights, &design, None),
    );
    Ok((variance, df_val, extra))
}

/// Scores with nulls as 0.0, as `taylor_variance` treats them.
pub(super) fn scores_to_arr(scores: &Float64Chunked) -> Vec<f64> {
    scores.iter().map(|s| s.unwrap_or(0.0)).collect()
}

/// Row-major flat k×k covariance for the FFI boundary.
pub(super) fn flatten_cov(cov: Vec<Vec<f64>>) -> Vec<f64> {
    cov.into_iter().flatten().collect()
}

// A zero denominator (weightless domain, or ratio with zero weighted x) gives a NaN row, as R's
// svymean(subset()), instead of failing every domain; its zero scores need the variance masked.

/// Variance and SE of an estimate, both NaN when the estimate is undefined.
pub(super) fn var_se(estimate: f64, variance: f64) -> (f64, f64) {
    if estimate.is_nan() {
        (f64::NAN, f64::NAN)
    } else {
        (variance, variance.max(0.0).sqrt())
    }
}

/// Flat covariance with NaN in the rows and columns of undefined estimates.
pub(super) fn flat_cov(
    score_cols: &[Vec<f64>],
    design: &TaylorDesign,
    estimates: &[f64],
) -> Vec<f64> {
    flat_cov_in(score_cols, design, estimates, None)
}

/// [`flat_cov`] for estimates of one domain (`active`).
pub(super) fn flat_cov_in(
    score_cols: &[Vec<f64>],
    design: &TaylorDesign,
    estimates: &[f64],
    active: Option<&[bool]>,
) -> Vec<f64> {
    let mut cov = taylor_covariance_apply_in(score_cols, design, active);
    for (i, est) in estimates.iter().enumerate() {
        if est.is_nan() {
            cov[i].fill(f64::NAN);
            for row in cov.iter_mut() {
                row[i] = f64::NAN;
            }
        }
    }
    flatten_cov(cov)
}

#[allow(clippy::too_many_arguments)]
pub fn compute_mean_ungrouped(
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
) -> EResult<DataFrame> {
    let y = df.column(value_col)?.f64()?;
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
            let Ok(estimate) = point_estimate_mean(y, weights) else {
                return Ok((vec![0.0; y.len()], (f64::NAN, f64::NAN)));
            };
            let scores = scores_mean_arr(y, weights)?;
            let srs_var = srs_variance_mean(y, weights, srs)?;
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
        df!["y" => vec![value_col], "est" => vec![estimate], "se" => vec![se],
        "var" => vec![variance], "df" => vec![df_val], "n" => vec![n], "deff" => vec![deff]]?,
    )
}

/// Batched ungrouped means: one design build amortised over the variables, which run in parallel.
/// Each row is bit-identical to the corresponding `compute_mean_ungrouped` call.
#[allow(clippy::too_many_arguments)]
pub fn compute_mean_multi(
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

    // Take df off the design's codes: indexing twice would dominate this serial head.
    let design = build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
    let df_val = degrees_of_freedom_from_design(weights, &design, None);
    let active = design.domain_rows(weights.len(), rows.as_ref(), None);

    // Resolve columns before fanning out: `df.column()` mutates the frame's schema cache,
    // so calling it from the parallel closures would race.
    let y_cols: Vec<&Float64Chunked> = value_cols
        .iter()
        .map(|vc| df.column(vc).and_then(|c| c.f64()))
        .collect::<PolarsResult<Vec<_>>>()?;

    let rows = (0..value_cols.len())
        .into_par_iter()
        .map(|i| -> EResult<(String, f64, f64, f64, u32, f64)> {
            let y = y_cols[i];
            let n = active_count(weights, None);
            let Ok(estimate) = point_estimate_mean(y, weights) else {
                let nan = f64::NAN;
                return Ok((value_cols[i].clone(), nan, nan, nan, n, nan));
            };
            let scores_arr = scores_mean_arr(y, weights)?;
            let variance = taylor_variance_apply_in(&scores_arr, &design, active.as_deref());
            let se = variance.max(0.0).sqrt();
            let srs_var = srs_variance_mean(y, weights, srs)?;
            let deff = if srs_var > 0.0 {
                variance / srs_var
            } else {
                f64::NAN
            };
            Ok((value_cols[i].clone(), estimate, se, variance, n, deff))
        })
        .collect::<EResult<Vec<_>>>()?;

    let nv = rows.len();
    let mut ys: Vec<String> = Vec::with_capacity(nv);
    let mut estimates: Vec<f64> = Vec::with_capacity(nv);
    let mut ses: Vec<f64> = Vec::with_capacity(nv);
    let mut variances: Vec<f64> = Vec::with_capacity(nv);
    let mut ns: Vec<u32> = Vec::with_capacity(nv);
    let mut deffs: Vec<f64> = Vec::with_capacity(nv);
    for (y, est, se, var, n, deff) in rows {
        ys.push(y);
        estimates.push(est);
        ses.push(se);
        variances.push(var);
        ns.push(n);
        deffs.push(deff);
    }
    let dfs = vec![df_val; nv];
    Ok(
        df!["y" => ys, "est" => estimates, "se" => ses, "var" => variances,
        "df" => dfs, "n" => ns, "deff" => deffs]?,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn compute_mean_grouped(
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
) -> EResult<(DataFrame, Vec<f64>)> {
    let y = df.column(value_col)?.f64()?;
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

    // One design for every group; only the domain-masked scores change.
    let design = design?.with_calib(calib);

    // Groups fan out over rayon and are collected in group order, independent of thread count.
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
                    let Some((estimate, scores_arr, srs_var)) =
                        mean_domain_parts(y, weights, &domain_mask, srs)?
                    else {
                        let nan = f64::NAN;
                        return Ok((group, nan, nan, nan, n_domain, nan, vec![0.0; y.len()]));
                    };
                    let variance =
                        taylor_variance_apply_in(&scores_arr, &design, active.as_deref());
                    let se = variance.max(0.0).sqrt();
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
    let out = df![by_col => by_vals, "y" => vec![value_col; n_groups], "est" => estimates,
        "se" => ses, "var" => variances, "df" => dfs, "n" => ns, "deff" => deffs]?;
    Ok((out, cov))
}
