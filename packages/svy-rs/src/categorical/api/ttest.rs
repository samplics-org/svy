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
