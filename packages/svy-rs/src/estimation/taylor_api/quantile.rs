use super::*;
use numpy::IntoPyArray;
use polars::prelude::DataFrame;
use pyo3::types::PyList;
use svy_core::estimation::taylor::WeightedCdf;

/// Each result domain's label (`None` when ungrouped) and the CDF its quantiles came from.
type DomainCdfs = Vec<(Option<String>, Option<WeightedCdf>)>;

/// The result frame, plus each domain's sorted values and CDF (numpy, moved) when `return_cdf`.
fn frame_with_cdfs(
    py: Python<'_>,
    frame: DataFrame,
    cdfs: DomainCdfs,
    return_cdf: bool,
) -> PyResult<Py<PyAny>> {
    let frame = PyDataFrame(frame).into_pyobject(py)?;
    if !return_cdf {
        return Ok(frame.into_any().unbind());
    }
    let list = PyList::empty(py);
    for (label, cdf) in cdfs {
        let arrays = cdf.map(|(y, c)| (y.into_pyarray(py), c.into_pyarray(py)));
        list.append((label, arrays))?;
    }
    Ok((frame, list).into_pyobject(py)?.into_any().unbind())
}

fn parse_quantile_method(quantile_method: Option<&str>) -> PyResult<SvyQuantileMethod> {
    Ok(quantile_method
        .map(SvyQuantileMethod::parse)
        .transpose()
        .map_err(|e| to_py_err(&e))?
        .unwrap_or(SvyQuantileMethod::Higher))
}

#[allow(clippy::too_many_arguments)]
fn quantile_frame(
    data: PyDataFrame,
    value_col: &str,
    weight_col: &str,
    probs: &[f64],
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_ssu_col: Option<&str>,
    by_col: Option<&str>,
    singleton_method: Option<&str>,
    domain_rows_col: Option<&str>,
    quantile_method: Option<&str>,
    calib_kind: Option<String>,
    calib_cells: Option<Vec<String>>,
    calib_aux: Option<Vec<String>>,
    calib_prev_wgt: Option<String>,
    calib_pins_total: Option<bool>,
    calib_new_wgt: Option<String>,
) -> PyResult<(DataFrame, DomainCdfs)> {
    let df = into_contiguous(data);
    let calib = make_calib(
        &df,
        weight_col,
        calib_kind,
        calib_cells,
        calib_aux,
        calib_prev_wgt,
        calib_pins_total,
        calib_new_wgt,
    )
    .map_err(|e| to_py_err(&e))?;
    let q_method = parse_quantile_method(quantile_method)?;

    match by_col {
        None => compute_quantile_ungrouped(
            &df,
            value_col,
            weight_col,
            strata_col,
            psu_col,
            ssu_col,
            fpc_col,
            fpc_ssu_col,
            singleton_method,
            domain_rows_col,
            calib,
            probs,
            q_method,
        )
        .map(|(frame, cdf)| (frame, vec![(None, cdf)])),
        Some(by) => compute_quantile_grouped(
            &df,
            value_col,
            weight_col,
            strata_col,
            psu_col,
            ssu_col,
            fpc_col,
            fpc_ssu_col,
            by,
            singleton_method,
            domain_rows_col,
            calib,
            probs,
            q_method,
        )
        .map(|(frame, cdfs)| {
            let labelled = cdfs.into_iter().map(|(g, c)| (Some(g), c)).collect();
            (frame, labelled)
        }),
    }
    .map_err(|e| to_py_err(&e))
}

#[allow(clippy::too_many_arguments)]
fn quantile_multi_frame(
    py: Python<'_>,
    data: PyDataFrame,
    value_cols: &[String],
    weight_col: &str,
    probs: &[f64],
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>,
    domain_rows_col: Option<&str>,
    quantile_method: Option<&str>,
    calib_kind: Option<String>,
    calib_cells: Option<Vec<String>>,
    calib_aux: Option<Vec<String>>,
    calib_prev_wgt: Option<String>,
    calib_pins_total: Option<bool>,
    calib_new_wgt: Option<String>,
) -> PyResult<(DataFrame, DomainCdfs)> {
    let df = into_contiguous(data);
    let calib = make_calib(
        &df,
        weight_col,
        calib_kind,
        calib_cells,
        calib_aux,
        calib_prev_wgt,
        calib_pins_total,
        calib_new_wgt,
    )
    .map_err(|e| to_py_err(&e))?;
    let q_method = parse_quantile_method(quantile_method)?;
    let (frame, cdfs) = py
        .detach(|| {
            compute_quantile_multi(
                &df,
                value_cols,
                weight_col,
                strata_col,
                psu_col,
                ssu_col,
                fpc_col,
                fpc_ssu_col,
                singleton_method,
                domain_rows_col,
                calib,
                probs,
                q_method,
            )
        })
        .map_err(|e| to_py_err(&e))?;
    Ok((frame, cdfs.into_iter().map(|c| (None, c)).collect()))
}

/// Woodruff quantiles for one variable: one row per probability (and domain), with a `prob` column.
/// With `return_cdf`, also each domain's `(label, (values_sorted, cdf) | None)` for the limits.
#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, probs, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, by_col=None, singleton_method=None, domain_rows_col=None, quantile_method=None, calib_kind=None, calib_cells=None, calib_aux=None, calib_prev_wgt=None, calib_pins_total=None, calib_new_wgt=None, return_cdf=false))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn taylor_quantile(
    py: Python,
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
    return_cdf: bool,
) -> PyResult<Py<PyAny>> {
    let (frame, cdfs) = quantile_frame(
        data,
        &value_col,
        &weight_col,
        &probs,
        strata_col.as_deref(),
        psu_col.as_deref(),
        ssu_col.as_deref(),
        fpc_col.as_deref(),
        fpc_ssu_col.as_deref(),
        by_col.as_deref(),
        singleton_method.as_deref(),
        domain_rows_col.as_deref(),
        quantile_method.as_deref(),
        calib_kind,
        calib_cells,
        calib_aux,
        calib_prev_wgt,
        calib_pins_total,
        calib_new_wgt,
    )?;
    frame_with_cdfs(py, frame, cdfs, return_cdf)
}

/// Batched ungrouped quantiles in parallel; rows variable-major, then by probability.
#[pyfunction]
#[pyo3(signature = (data, value_cols, weight_col, probs, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, singleton_method=None, domain_rows_col=None, quantile_method=None, calib_kind=None, calib_cells=None, calib_aux=None, calib_prev_wgt=None, calib_pins_total=None, calib_new_wgt=None, return_cdf=false))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn taylor_quantile_multi(
    py: Python,
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
    return_cdf: bool,
) -> PyResult<Py<PyAny>> {
    let (frame, cdfs) = quantile_multi_frame(
        py,
        data,
        &value_cols,
        &weight_col,
        &probs,
        strata_col.as_deref(),
        psu_col.as_deref(),
        ssu_col.as_deref(),
        fpc_col.as_deref(),
        fpc_ssu_col.as_deref(),
        singleton_method.as_deref(),
        domain_rows_col.as_deref(),
        quantile_method.as_deref(),
        calib_kind,
        calib_cells,
        calib_aux,
        calib_prev_wgt,
        calib_pins_total,
        calib_new_wgt,
    )?;
    frame_with_cdfs(py, frame, cdfs, return_cdf)
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
    let q_method = parse_quantile_method(quantile_method.as_deref())?;
    Ok(probs
        .iter()
        .map(|&p| weighted_quantile(y, c, p, q_method))
        .collect())
}

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, by_col=None, singleton_method=None, domain_rows_col=None, quantile_method=None, calib_kind=None, calib_cells=None, calib_aux=None, calib_prev_wgt=None, calib_pins_total=None, calib_new_wgt=None, return_cdf=false))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn taylor_median(
    py: Python,
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
    return_cdf: bool,
) -> PyResult<Py<PyAny>> {
    let (frame, cdfs) = quantile_frame(
        data,
        &value_col,
        &weight_col,
        &[0.5],
        strata_col.as_deref(),
        psu_col.as_deref(),
        ssu_col.as_deref(),
        fpc_col.as_deref(),
        fpc_ssu_col.as_deref(),
        by_col.as_deref(),
        singleton_method.as_deref(),
        domain_rows_col.as_deref(),
        quantile_method.as_deref(),
        calib_kind,
        calib_cells,
        calib_aux,
        calib_prev_wgt,
        calib_pins_total,
        calib_new_wgt,
    )?;
    let frame = without_prob(frame).map_err(|e| to_py_err(&e))?;
    frame_with_cdfs(py, frame, cdfs, return_cdf)
}

/// Batched ungrouped median over many variables; one row per variable, in input order.
#[pyfunction]
#[pyo3(signature = (data, value_cols, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, singleton_method=None, domain_rows_col=None, quantile_method=None, calib_kind=None, calib_cells=None, calib_aux=None, calib_prev_wgt=None, calib_pins_total=None, calib_new_wgt=None, return_cdf=false))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn taylor_median_multi(
    py: Python,
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
    return_cdf: bool,
) -> PyResult<Py<PyAny>> {
    let (frame, cdfs) = quantile_multi_frame(
        py,
        data,
        &value_cols,
        &weight_col,
        &[0.5],
        strata_col.as_deref(),
        psu_col.as_deref(),
        ssu_col.as_deref(),
        fpc_col.as_deref(),
        fpc_ssu_col.as_deref(),
        singleton_method.as_deref(),
        domain_rows_col.as_deref(),
        quantile_method.as_deref(),
        calib_kind,
        calib_cells,
        calib_aux,
        calib_prev_wgt,
        calib_pins_total,
        calib_new_wgt,
    )?;
    let frame = without_prob(frame).map_err(|e| to_py_err(&e))?;
    frame_with_cdfs(py, frame, cdfs, return_cdf)
}
