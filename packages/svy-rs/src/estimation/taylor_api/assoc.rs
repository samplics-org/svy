use super::*;

/// Covariance or correlation over column pairs sharing one design, ungrouped or by group.
/// The pair columns `y`/`x` are positional only: both statistics are symmetric.
#[pyfunction]
#[pyo3(signature = (data, y_cols, x_cols, kind, weight_col, strata_col=None, psu_col=None, ssu_col=None, fpc_col=None, fpc_ssu_col=None, by_col=None, singleton_method=None, domain_rows_col=None, deff_ref=None, deff_pop_total=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn taylor_assoc(
    _py: Python,
    data: PyDataFrame,
    y_cols: Vec<String>,
    x_cols: Vec<String>,
    kind: String,
    weight_col: String,
    strata_col: Option<String>,
    psu_col: Option<String>,
    ssu_col: Option<String>,
    fpc_col: Option<String>,
    fpc_ssu_col: Option<String>,
    by_col: Option<String>,
    singleton_method: Option<String>,
    domain_rows_col: Option<String>,
    deff_ref: Option<String>,
    deff_pop_total: Option<f64>,
) -> PyResult<PyDataFrame> {
    let df = into_contiguous(data);
    let srs = parse_srs_ref(deff_ref.as_deref(), deff_pop_total).map_err(|e| to_py_err(&e))?;
    let result = _py
        .detach(|| {
            let kind = AssocKind::from_name(&kind)?;
            if y_cols.len() != x_cols.len() {
                return Err(EstimationError::PairLengths {
                    y: y_cols.len(),
                    x: x_cols.len(),
                });
            }
            if y_cols.is_empty() {
                return Err(EstimationError::Empty(
                    "at least one column pair is required",
                ));
            }
            compute_assoc(
                &df,
                &y_cols,
                &x_cols,
                kind,
                &weight_col,
                strata_col.as_deref(),
                psu_col.as_deref(),
                ssu_col.as_deref(),
                fpc_col.as_deref(),
                fpc_ssu_col.as_deref(),
                by_col.as_deref(),
                singleton_method.as_deref(),
                domain_rows_col.as_deref(),
                srs,
            )
        })
        .map_err(|e| to_py_err(&e))?;
    Ok(PyDataFrame(result))
}
