use super::*;

#[pyfunction]
#[pyo3(signature = (
    data, rowvar_col, weight_col,
    colvar_col=None, strata_col=None, psu_col=None, ssu_col=None,
    fpc_col=None, fpc_ssu_col=None, singleton_method=None,
    compute_totals=false,
    calib_kind=None, calib_cells=None, calib_aux=None, calib_prev_wgt=None,
    calib_pins_total=None, calib_new_wgt=None, domain_col=None,
    rep_weight_cols=None, rep_coefs=None, rep_df=None, variance_center="rep_mean",
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn tabulate_rs(
    _py: Python,
    data: PyDataFrame,
    rowvar_col: String,
    weight_col: String,
    colvar_col: Option<String>,
    strata_col: Option<String>,
    psu_col: Option<String>,
    ssu_col: Option<String>,
    fpc_col: Option<String>,
    fpc_ssu_col: Option<String>,
    singleton_method: Option<String>,
    compute_totals: bool,
    calib_kind: Option<String>,
    calib_cells: Option<Vec<String>>,
    calib_aux: Option<Vec<String>>,
    calib_prev_wgt: Option<String>,
    calib_pins_total: Option<bool>,
    calib_new_wgt: Option<String>,
    domain_col: Option<String>,
    rep_weight_cols: Option<Vec<String>>,
    rep_coefs: Option<Vec<f64>>,
    rep_df: Option<f64>,
    variance_center: &str,
) -> PyResult<(PyDataFrame, PyDataFrame)> {
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
    let result = compute_tabulate(
        &df,
        &rowvar_col,
        &weight_col,
        colvar_col.as_deref(),
        strata_col.as_deref(),
        psu_col.as_deref(),
        ssu_col.as_deref(),
        fpc_col.as_deref(),
        fpc_ssu_col.as_deref(),
        singleton_method.as_deref(),
        compute_totals,
        calib.as_ref(),
        domain_col.as_deref(),
        rep.as_ref(),
    )
    .map_err(|e| to_py_err(&e))?;
    Ok((PyDataFrame(result.0), PyDataFrame(result.1)))
}
