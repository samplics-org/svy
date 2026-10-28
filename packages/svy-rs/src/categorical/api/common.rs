use super::*;

/// What `prepare_two_sample_data` hands the two-sample and rank tests.
pub(super) type TwoSampleData = (
    Vec<f64>,    // y
    Vec<f64>,    // w, zeroed out of domain and where y is missing
    Vec<u32>,    // g
    Vec<String>, // sorted group labels
    usize,       // n
    Vec<bool>,   // the domain's rows (in the domain, y present), whatever the weight
);

/// An optional design column, any dtype: the kernels hash strings and densify integer codes.
pub(crate) fn get_opt_col<'a>(df: &'a DataFrame, col: Option<&str>) -> CResult<Option<&'a Column>> {
    match col {
        Some(c) => Ok(Some(df.column(c)?)),
        None => Ok(None),
    }
}

pub(crate) fn get_opt_f64<'a>(
    df: &'a DataFrame,
    col: Option<&str>,
) -> CResult<Option<&'a Float64Chunked>> {
    match col {
        Some(c) => Ok(Some(df.column(c)?.as_materialized_series().f64()?)),
        None => Ok(None),
    }
}

/// The replicate weights, when the caller asked for a replication variance.
pub(super) fn rep_spec(
    df: &DataFrame,
    rep_weight_cols: Option<Vec<String>>,
    rep_coefs: Option<Vec<f64>>,
    rep_df: Option<f64>,
    variance_center: &str,
) -> PyResult<Option<RepSpec>> {
    match rep_weight_cols {
        None => Ok(None),
        Some(cols) => {
            let coefs = rep_coefs.ok_or_else(|| {
                to_py_err(&CategoricalError::MissingInput(
                    "rep_coefs is required with rep_weight_cols",
                ))
            })?;
            RepSpec::from_frame(df, &cols, coefs, variance_center, rep_df)
                .map(Some)
                .map_err(|e| to_py_err(&e))
        }
    }
}

/// Numeric source columns sort by value, so 2 comes before 10; others lexically.
pub(super) fn sort_labels(labels: &mut [String], dtype: &DataType) {
    if dtype.is_primitive_numeric() {
        let key = |s: &String| s.parse::<f64>().unwrap_or(f64::NAN);
        labels.sort_by(|a, b| key(a).total_cmp(&key(b)));
    } else {
        labels.sort();
    }
}

/// Arrays for the two- and k-sample tests; out-of-domain rows get weight 0 but stay, keeping the PSU structure.
#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_two_sample_data(
    df: &DataFrame,
    y_col: &str,
    w_col: &str,
    g_col: &str,
    _strata_col: Option<&str>,
    _psu_col: Option<&str>,
    domain_col: Option<&str>,
    domain_val: Option<&str>,
) -> CResult<TwoSampleData> {
    let n = df.height();
    let y_ca = df.column(y_col)?.as_materialized_series().f64()?;
    let w_ca = df.column(w_col)?.as_materialized_series().f64()?;
    let g_series = df.column(g_col)?.as_materialized_series();
    let g_str = g_series.cast(&DataType::String)?;
    let g_ca = g_str.str()?;

    let domain_mask: Option<Vec<bool>> = match (domain_col, domain_val) {
        (Some(d_col), Some(d_val)) => {
            let d_series = df.column(d_col)?.as_materialized_series();
            let d_str = d_series.cast(&DataType::String)?;
            let d_ca = d_str.str()?;
            Some(d_ca.iter().map(|opt| opt == Some(d_val)).collect())
        }
        _ => None,
    };

    let mut level_set: Vec<String> = Vec::new();
    {
        let mut seen = std::collections::HashSet::new();
        for i in 0..n {
            let in_domain = domain_mask.as_ref().is_none_or(|m| m[i]);
            let w_val = w_ca.get(i).unwrap_or(0.0);
            if in_domain
                && w_val > 0.0
                && let Some(label) = g_ca.get(i)
                && seen.insert(label.to_string())
            {
                level_set.push(label.to_string());
            }
        }
    }
    sort_labels(&mut level_set, g_series.dtype());

    let mut label_to_idx: HashMap<String, u32> = HashMap::new();
    for (i, label) in level_set.iter().enumerate() {
        label_to_idx.insert(label.clone(), i as u32);
    }

    let mut y_arr = Vec::with_capacity(n);
    let mut w_arr = Vec::with_capacity(n);
    let mut g_arr = Vec::with_capacity(n);
    let mut rows = Vec::with_capacity(n);

    for i in 0..n {
        let yi = y_ca.get(i).unwrap_or(f64::NAN);
        let wi = w_ca.get(i).unwrap_or(0.0);
        let in_domain = domain_mask.as_ref().is_none_or(|m| m[i]);
        let label = g_ca.get(i).unwrap_or("__NULL__");
        let gi = label_to_idx.get(label).copied().unwrap_or(0);

        rows.push(in_domain && !yi.is_nan());
        if yi.is_nan() {
            y_arr.push(0.0);
            w_arr.push(0.0);
            g_arr.push(0);
        } else {
            y_arr.push(yi);
            w_arr.push(if in_domain { wi } else { 0.0 });
            g_arr.push(gi);
        }
    }

    Ok((y_arr, w_arr, g_arr, level_set, n, rows))
}

/// Whether any row in the domain has `y` and a positive weight.
pub(super) fn has_positive_weight(
    y: &Float64Chunked,
    weights: &Float64Chunked,
    mask: Option<&BooleanChunked>,
) -> bool {
    (0..y.len()).any(|i| {
        let in_domain = mask.is_none_or(|m| m.get(i).unwrap_or(false));
        in_domain && y.get(i).is_some_and(|v| !v.is_nan()) && weights.get(i).unwrap_or(0.0) > 0.0
    })
}

/// Positive-weight rows per level of `frame_levels`, from `prepare_two_sample_data`'s output.
pub(super) fn level_counts(
    w: &[f64],
    g: &[u32],
    levels: &[String],
    frame_levels: &[String],
) -> Vec<u32> {
    let mut counts = vec![0u32; frame_levels.len()];
    for (&wi, &gi) in w.iter().zip(g) {
        if wi <= 0.0 {
            continue;
        }
        let label = &levels[gi as usize];
        if let Some(k) = frame_levels.iter().position(|l| l == label) {
            counts[k] += 1;
        }
    }
    counts
}

/// Run `run_single` per level of `by_col`, within any domain, stacking the results with a `by_col` column.
pub(super) fn compute_by_groups<F>(
    df: &DataFrame,
    by_col: &str,
    domain_col: Option<&str>,
    domain_val: Option<&str>,
    run_single: F,
) -> CResult<DataFrame>
where
    F: Fn(&DataFrame, Option<&str>, Option<&str>) -> CResult<DataFrame>,
{
    let by_series = df.column(by_col)?.as_materialized_series();
    let by_str = by_series.cast(&DataType::String)?;
    let by_ca = by_str.str()?;
    let mut by_levels: Vec<String> = by_ca
        .unique()?
        .iter()
        .filter_map(|v| v.map(|s| s.to_string()))
        .collect();
    sort_labels(&mut by_levels, by_series.dtype());

    let mut result_dfs: Vec<DataFrame> = Vec::new();

    for level in &by_levels {
        let by_mask: BooleanChunked = by_ca.equal(level.as_str());

        let combined_mask: BooleanChunked =
            if let (Some(d_col), Some(d_val)) = (domain_col, domain_val) {
                let d_series = df.column(d_col)?.as_materialized_series();
                let d_str = d_series.cast(&DataType::String)?;
                let d_ca = d_str.str()?;
                let d_mask = d_ca.equal(d_val);
                (&by_mask) & (&d_mask)
            } else {
                by_mask
            };

        // hstack, not with_column: the latter needs an owned frame, i.e. a full clone.
        let mask_col: Series = combined_mask
            .iter()
            .map(|v| if v.unwrap_or(false) { "true" } else { "false" })
            .collect::<StringChunked>()
            .with_name("__svy_by_domain__".into())
            .into_series();

        let temp_df = df.hstack(&[mask_col.into()])?;

        let mut single = run_single(&temp_df, Some("__svy_by_domain__"), Some("true"))?;
        let with_by = single
            .with_column(Column::new(by_col.into(), vec![level.as_str()]))?
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
