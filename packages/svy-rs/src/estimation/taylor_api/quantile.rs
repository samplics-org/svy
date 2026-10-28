use super::*;

/// Woodruff quantiles for one variable: one row per probability (and domain), with a `prob` column.
#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, probs, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, by_col=None, singleton_method=None, domain_rows_col=None, quantile_method=None, calib_kind=None, calib_cells=None, calib_aux=None, calib_prev_wgt=None, calib_pins_total=None, calib_new_wgt=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn taylor_quantile(
    _py: Python,
    data: PyDataFrame,
    value_col: String,
    weight_col: String,
    probs: Vec<f64>,
    strata_col: Option<String>,
    psu_col: Option<String>,
    ssu_col: Option<String>,
    fpc_col: Option<String>,
    fpc_ssu_col: Option<String>,
    by_col: Option<String>,
    singleton_method: Option<String>,
    domain_rows_col: Option<String>,
    quantile_method: Option<String>,
    calib_kind: Option<String>,
    calib_cells: Option<Vec<String>>,
    calib_aux: Option<Vec<String>>,
    calib_prev_wgt: Option<String>,
    calib_pins_total: Option<bool>,
    calib_new_wgt: Option<String>,
) -> PyResult<PyDataFrame> {
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
    let q_method = quantile_method
        .as_deref()
        .map(SvyQuantileMethod::parse)
        .transpose()
        .map_err(|e| to_py_err(&e))?
        .unwrap_or(SvyQuantileMethod::Higher);

    let result = match by_col.as_deref() {
        None => compute_quantile_ungrouped(
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
            calib,
            &probs,
            q_method,
        ),
        Some(by) => compute_quantile_grouped(
            &df,
            &value_col,
            &weight_col,
            strata_col.as_deref(),
            psu_col.as_deref(),
            ssu_col.as_deref(),
            fpc_col.as_deref(),
            fpc_ssu_col.as_deref(),
            by,
            singleton_method.as_deref(),
            domain_rows_col.as_deref(),
            calib,
            &probs,
            q_method,
        ),
    }
    .map_err(|e| to_py_err(&e))?;
    Ok(PyDataFrame(result))
}

/// Batched ungrouped quantiles in parallel; rows variable-major, then by probability.
#[pyfunction]
#[pyo3(signature = (data, value_cols, weight_col, probs, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, singleton_method=None, domain_rows_col=None, quantile_method=None, calib_kind=None, calib_cells=None, calib_aux=None, calib_prev_wgt=None, calib_pins_total=None, calib_new_wgt=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn taylor_quantile_multi(
    _py: Python,
    data: PyDataFrame,
    value_cols: Vec<String>,
    weight_col: String,
    probs: Vec<f64>,
    strata_col: Option<String>,
    psu_col: Option<String>,
    ssu_col: Option<String>,
    fpc_col: Option<String>,
    fpc_ssu_col: Option<String>,
    singleton_method: Option<String>,
    domain_rows_col: Option<String>,
    quantile_method: Option<String>,
    calib_kind: Option<String>,
    calib_cells: Option<Vec<String>>,
    calib_aux: Option<Vec<String>>,
    calib_prev_wgt: Option<String>,
    calib_pins_total: Option<bool>,
    calib_new_wgt: Option<String>,
) -> PyResult<PyDataFrame> {
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
    let q_method = quantile_method
        .as_deref()
        .map(SvyQuantileMethod::parse)
        .transpose()
        .map_err(|e| to_py_err(&e))?
        .unwrap_or(SvyQuantileMethod::Higher);
    let result = _py
        .detach(|| {
            compute_quantile_multi(
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
                calib,
                &probs,
                q_method,
            )
        })
        .map_err(|e| to_py_err(&e))?;
    Ok(PyDataFrame(result))
}

/// The weighted quantile rule at arbitrary probabilities, over a sorted variable and its CDF.
/// Python's Woodruff limits reuse this rule, as R uses one `method`/`f` for estimate and endpoints.
#[pyfunction]
#[pyo3(signature = (values_sorted, cdf, probs, quantile_method=None))]
pub(crate) fn weighted_quantile_at(
    values_sorted: PyReadonlyArray1<f64>,
    cdf: PyReadonlyArray1<f64>,
    probs: Vec<f64>,
    quantile_method: Option<String>,
) -> PyResult<Vec<f64>> {
    let y = values_sorted.as_slice()?;
    let c = cdf.as_slice()?;
    if y.len() != c.len() {
        return Err(to_py_err(&EstimationError::CdfLength {
            values: y.len(),
            cdf: c.len(),
        }));
    }
    let q_method = quantile_method
        .as_deref()
        .map(SvyQuantileMethod::parse)
        .transpose()
        .map_err(|e| to_py_err(&e))?
        .unwrap_or(SvyQuantileMethod::Higher);
    Ok(probs
        .iter()
        .map(|&p| weighted_quantile(y, c, p, q_method))
        .collect())
}

/// Drop the `prob` column: the median entry points report no probability.
fn without_prob(mut df: DataFrame) -> EResult<DataFrame> {
    let _ = df.drop_in_place("prob")?;
    Ok(df)
}

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, by_col=None, singleton_method=None, domain_rows_col=None, quantile_method=None, calib_kind=None, calib_cells=None, calib_aux=None, calib_prev_wgt=None, calib_pins_total=None, calib_new_wgt=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn taylor_median(
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
    quantile_method: Option<String>,
    calib_kind: Option<String>,
    calib_cells: Option<Vec<String>>,
    calib_aux: Option<Vec<String>>,
    calib_prev_wgt: Option<String>,
    calib_pins_total: Option<bool>,
    calib_new_wgt: Option<String>,
) -> PyResult<PyDataFrame> {
    let out = taylor_quantile(
        _py,
        data,
        value_col,
        weight_col,
        vec![0.5],
        strata_col,
        psu_col,
        ssu_col,
        fpc_col,
        fpc_ssu_col,
        by_col,
        singleton_method,
        domain_rows_col,
        quantile_method,
        calib_kind,
        calib_cells,
        calib_aux,
        calib_prev_wgt,
        calib_pins_total,
        calib_new_wgt,
    )?;
    let result = without_prob(out.0).map_err(|e| to_py_err(&e))?;
    Ok(PyDataFrame(result))
}

/// Batched ungrouped median over many variables; one row per variable, in input order.
#[pyfunction]
#[pyo3(signature = (data, value_cols, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, singleton_method=None, domain_rows_col=None, quantile_method=None, calib_kind=None, calib_cells=None, calib_aux=None, calib_prev_wgt=None, calib_pins_total=None, calib_new_wgt=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn taylor_median_multi(
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
    quantile_method: Option<String>,
    calib_kind: Option<String>,
    calib_cells: Option<Vec<String>>,
    calib_aux: Option<Vec<String>>,
    calib_prev_wgt: Option<String>,
    calib_pins_total: Option<bool>,
    calib_new_wgt: Option<String>,
) -> PyResult<PyDataFrame> {
    let out = taylor_quantile_multi(
        _py,
        data,
        value_cols,
        weight_col,
        vec![0.5],
        strata_col,
        psu_col,
        ssu_col,
        fpc_col,
        fpc_ssu_col,
        singleton_method,
        domain_rows_col,
        quantile_method,
        calib_kind,
        calib_cells,
        calib_aux,
        calib_prev_wgt,
        calib_pins_total,
        calib_new_wgt,
    )?;
    let result = without_prob(out.0).map_err(|e| to_py_err(&e))?;
    Ok(PyDataFrame(result))
}

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
fn compute_quantile_ungrouped(
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
) -> EResult<DataFrame> {
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

    let rows = quantiles_woodruff(
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
    Ok(
        df!["y" => vec![value_col; k], "prob" => probs.to_vec(), "est" => estimates,
        "se" => ses, "var" => variances, "df" => vec![df_val; k], "n" => vec![n; k]]?,
    )
}

/// Batched ungrouped quantiles: sort-bound, so the win is parallel variables, not design reuse.
#[allow(clippy::too_many_arguments)]
fn compute_quantile_multi(
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
) -> EResult<DataFrame> {
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
    for (i, rows) in per_var.into_iter().enumerate() {
        ys.extend(std::iter::repeat_n(value_cols[i].clone(), k));
        ns.extend(std::iter::repeat_n(active_count(cols.weights, None), k));
        flat.extend(rows);
    }

    let (estimates, ses, variances) = unzip_woodruff(flat);
    let probs_rep: Vec<f64> = std::iter::repeat_n(probs, value_cols.len())
        .flatten()
        .copied()
        .collect();
    Ok(
        df!["y" => ys, "prob" => probs_rep, "est" => estimates, "se" => ses,
        "var" => variances, "df" => vec![df_val; n_rows], "n" => ns]?,
    )
}

#[allow(clippy::too_many_arguments)]
fn compute_quantile_grouped(
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
) -> EResult<DataFrame> {
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

    for group in unique_groups.iter().flatten() {
        let domain_mask = by_str.equal(group);
        let n_domain = active_count(cols.weights, Some(&domain_mask));
        let rows = quantiles_woodruff(
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
    }

    let n_rows = by_vals.len();
    let probs_rep: Vec<f64> = std::iter::repeat_n(probs, n_rows / k.max(1))
        .flatten()
        .copied()
        .collect();
    let (estimates, ses, variances) = unzip_woodruff(flat);
    Ok(
        df![by_col => by_vals, "y" => vec![value_col; n_rows], "prob" => probs_rep,
        "est" => estimates, "se" => ses, "var" => variances, "df" => dfs, "n" => ns]?,
    )
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
