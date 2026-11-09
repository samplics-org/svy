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
