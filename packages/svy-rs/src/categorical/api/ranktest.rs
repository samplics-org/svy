use super::*;

#[pyfunction]
#[pyo3(signature = (
    data, y_col, group_col, weight_col,
    strata_col=None, psu_col=None, ssu_col=None,
    fpc_col=None, fpc_ssu_col=None, score_method=None,
    singleton_method=None, domain_col=None, domain_val=None, by_col=None,
    calib_kind=None, calib_cells=None, calib_aux=None, calib_prev_wgt=None,
    calib_pins_total=None, calib_new_wgt=None,
    rep_weight_cols=None, rep_coefs=None, rep_df=None, variance_center="rep_mean",
    scores_given=false,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ranktest_rs(
    _py: Python,
    data: PyDataFrame,
    y_col: String,
    group_col: String,
    weight_col: String,
    strata_col: Option<String>,
    psu_col: Option<String>,
    ssu_col: Option<String>,
    fpc_col: Option<String>,
    fpc_ssu_col: Option<String>,
    score_method: Option<String>,
    singleton_method: Option<String>,
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
    // `y_col` holds caller-supplied scores; replication only, as Taylor routes them to ttest_rs.
    scores_given: bool,
) -> PyResult<PyDataFrame> {
    let df = into_contiguous(data);
    let rep = rep_spec(&df, rep_weight_cols, rep_coefs, rep_df, variance_center)?;
    if scores_given && rep.is_none() {
        return Err(to_py_err(&CategoricalError::MissingInput(
            "scores_given requires rep_weight_cols",
        )));
    }
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
    let result = compute_svyranktest(
        &df,
        &y_col,
        &group_col,
        &weight_col,
        strata_col.as_deref(),
        psu_col.as_deref(),
        ssu_col.as_deref(),
        fpc_col.as_deref(),
        fpc_ssu_col.as_deref(),
        score_method.as_deref(),
        singleton_method.as_deref(),
        domain_col.as_deref(),
        domain_val.as_deref(),
        by_col.as_deref(),
        calib.as_ref(),
        rep.as_ref(),
        scores_given,
    )
    .map_err(|e| to_py_err(&e))?;
    Ok(PyDataFrame(result))
}
