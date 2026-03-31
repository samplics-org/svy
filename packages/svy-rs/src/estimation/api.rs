// src/regression/api.rs
//
// PyO3-facing wrapper for the GLM regression function.
// The actual fitting logic lives in regression/glm.rs.

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::regression::glm::fit_glm;

#[pyfunction]
pub fn fit_glm_rs(
    _py: Python,
    y_name: String,
    x_names: Vec<String>,
    weight_name: String,
    stratum_name: Option<String>,
    psu_name: Option<String>,
    family: String,
    link: String,
    tol: f64,
    max_iter: usize,
    data: PyDataFrame,
) -> PyResult<(Vec<f64>, Vec<f64>, f64, f64, f64, f64, u32)> {
    let df: DataFrame = data.into();

    let y = df
        .column(&y_name)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
        .as_materialized_series()
        .clone();

    let mut x_cols = Vec::with_capacity(x_names.len());
    for name in &x_names {
        let s = df
            .column(name)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
            .as_materialized_series()
            .clone();
        x_cols.push(s);
    }

    let weights = df
        .column(&weight_name)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
        .as_materialized_series()
        .clone();

    let stratum = if let Some(s) = stratum_name {
        Some(
            df.column(&s)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
                .as_materialized_series()
                .clone(),
        )
    } else {
        None
    };

    let psu = if let Some(p) = psu_name {
        Some(
            df.column(&p)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?
                .as_materialized_series()
                .clone(),
        )
    } else {
        None
    };

    let result = fit_glm(
        &y, x_cols, &weights,
        stratum.as_ref(), psu.as_ref(),
        &family, &link, tol, max_iter,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok((
        result.params,
        result.cov_params,
        result.scale,
        result.df_resid,
        result.deviance,
        result.null_deviance,
        result.iterations,
    ))
}
