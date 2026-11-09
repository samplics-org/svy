use super::*;
use crate::estimation::taylor::WeightedCdf;

/// Drop the `prob` column: the median entry points report no probability.
pub fn without_prob(mut df: DataFrame) -> EResult<DataFrame> {
    let _ = df.drop_in_place("prob")?;
    Ok(df)
}

/// Each by-group's label and the CDF its quantiles came from.
pub type GroupCdfs = Vec<(String, Option<WeightedCdf>)>;

/// The design columns every quantile path needs, resolved once.
struct QuantileCols<'a> {
    weights: &'a Float64Chunked,
    rows: Option<BooleanChunked>,
    strata: Option<&'a Column>,
    psu: Option<&'a Column>,
    design: TaylorDesign,
}

#[allow(clippy::too_many_arguments)]
fn resolve_quantile_cols<'a>(
    df: &'a DataFrame,
    weight_col: &str,
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>,
    domain_rows_col: Option<&str>,
    calib: Option<CalibSweep>,
) -> EResult<QuantileCols<'a>> {
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

    // One design build serves every (variable, prob) pair.
    let design =
        build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method)?.with_calib(calib);
    Ok(QuantileCols {
        weights,
        rows,
        strata,
        psu,
        design,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn compute_quantile_ungrouped(
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
    calib: Option<CalibSweep>,
    probs: &[f64],
    q_method: SvyQuantileMethod,
) -> EResult<(DataFrame, Option<WeightedCdf>)> {
    let y = df.column(value_col)?.f64()?;
    let cols = resolve_quantile_cols(
        df,
        weight_col,
        strata_col,
        psu_col,
        ssu_col,
        fpc_col,
        fpc_ssu_col,
        singleton_method,
        domain_rows_col,
        calib,
    )?;

    let (rows, cdf) = quantiles_woodruff(
        y,
        cols.weights,
        None,
        cols.rows.as_ref(),
        &cols.design,
        probs,
        q_method,
    )?;
    let df_val = degrees_of_freedom(cols.weights, cols.strata, cols.psu)?;
    let n = active_count(cols.weights, None);
    let k = rows.len();

    let (estimates, ses, variances) = unzip_woodruff(rows);
    let out = df!["y" => vec![value_col; k], "prob" => probs.to_vec(), "est" => estimates,
        "se" => ses, "var" => variances, "df" => vec![df_val; k], "n" => vec![n; k]]?;
    Ok((out, cdf))
}

/// Batched ungrouped quantiles: sort-bound, so the win is parallel variables, not design reuse.
/// The CDFs come back one per variable, in `value_cols` order.
#[allow(clippy::too_many_arguments)]
pub fn compute_quantile_multi(
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
    calib: Option<CalibSweep>,
    probs: &[f64],
    q_method: SvyQuantileMethod,
) -> EResult<(DataFrame, Vec<Option<WeightedCdf>>)> {
    let cols = resolve_quantile_cols(
        df,
        weight_col,
        strata_col,
        psu_col,
        ssu_col,
        fpc_col,
        fpc_ssu_col,
        singleton_method,
        domain_rows_col,
        calib,
    )?;

    // df is design-only, identical across variables — compute once.
    let df_val = degrees_of_freedom(cols.weights, cols.strata, cols.psu)?;

    let y_cols: Vec<&Float64Chunked> = value_cols
        .iter()
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .collect::<PolarsResult<Vec<_>>>()?;

    let per_var = (0..value_cols.len())
        .into_par_iter()
        .map(|i| {
            quantiles_woodruff(
                y_cols[i],
                cols.weights,
                None,
                cols.rows.as_ref(),
                &cols.design,
                probs,
                q_method,
            )
        })
        .collect::<EResult<Vec<_>>>()?;

    let k = probs.len();
    let n_rows = per_var.len() * k;
    let mut ys: Vec<String> = Vec::with_capacity(n_rows);
    let mut ns: Vec<u32> = Vec::with_capacity(n_rows);
    let mut flat: Vec<(f64, f64, f64)> = Vec::with_capacity(n_rows);
    let mut cdfs: Vec<Option<WeightedCdf>> = Vec::with_capacity(value_cols.len());
    for (i, (rows, cdf)) in per_var.into_iter().enumerate() {
        ys.extend(std::iter::repeat_n(value_cols[i].clone(), k));
        ns.extend(std::iter::repeat_n(active_count(cols.weights, None), k));
        flat.extend(rows);
        cdfs.push(cdf);
    }

    let (estimates, ses, variances) = unzip_woodruff(flat);
    let probs_rep: Vec<f64> = std::iter::repeat_n(probs, value_cols.len())
        .flatten()
        .copied()
        .collect();
    let out = df!["y" => ys, "prob" => probs_rep, "est" => estimates, "se" => ses,
        "var" => variances, "df" => vec![df_val; n_rows], "n" => ns]?;
    Ok((out, cdfs))
}

#[allow(clippy::too_many_arguments)]
pub fn compute_quantile_grouped(
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
    calib: Option<CalibSweep>,
    probs: &[f64],
    q_method: SvyQuantileMethod,
) -> EResult<(DataFrame, GroupCdfs)> {
    let y = df.column(value_col)?.f64()?;
    let cols = resolve_quantile_cols(
        df,
        weight_col,
        strata_col,
        psu_col,
        ssu_col,
        fpc_col,
        fpc_ssu_col,
        singleton_method,
        domain_rows_col,
        calib,
    )?;
    let by_str = df.column(by_col)?.str()?;
    let unique_groups = crate::estimation::sorted_groups(by_str)?;

    let k = probs.len();
    let mut by_vals: Vec<&str> = Vec::new();
    let mut dfs: Vec<u32> = Vec::new();
    let mut ns: Vec<u32> = Vec::new();
    let mut flat: Vec<(f64, f64, f64)> = Vec::new();
    let mut cdfs: GroupCdfs = Vec::new();

    for group in unique_groups.iter().flatten() {
        let domain_mask = by_str.equal(group);
        let n_domain = active_count(cols.weights, Some(&domain_mask));
        let (rows, cdf) = quantiles_woodruff(
            y,
            cols.weights,
            Some(&domain_mask),
            cols.rows.as_ref(),
            &cols.design,
            probs,
            q_method,
        )?;
        // Per-group df: see the note in compute_mean_grouped.
        let df_val = degrees_of_freedom_from_design(cols.weights, &cols.design, Some(&domain_mask));

        by_vals.extend(std::iter::repeat_n(group, k));
        dfs.extend(std::iter::repeat_n(df_val, k));
        ns.extend(std::iter::repeat_n(n_domain, k));
        flat.extend(rows);
        cdfs.push((group.to_string(), cdf));
    }

    let n_rows = by_vals.len();
    let probs_rep: Vec<f64> = std::iter::repeat_n(probs, n_rows / k.max(1))
        .flatten()
        .copied()
        .collect();
    let (estimates, ses, variances) = unzip_woodruff(flat);
    let out = df![by_col => by_vals, "y" => vec![value_col; n_rows], "prob" => probs_rep,
        "est" => estimates, "se" => ses, "var" => variances, "df" => dfs, "n" => ns]?;
    Ok((out, cdfs))
}

fn unzip_woodruff(rows: Vec<(f64, f64, f64)>) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut estimates = Vec::with_capacity(rows.len());
    let mut ses = Vec::with_capacity(rows.len());
    let mut variances = Vec::with_capacity(rows.len());
    for (est, var, se) in rows {
        estimates.push(est);
        ses.push(se);
        variances.push(var);
    }
    (estimates, ses, variances)
}
