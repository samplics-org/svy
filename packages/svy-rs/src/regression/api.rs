// src/regression/api.rs
//
// PyO3-facing wrapper for the GLM regression function.
// The actual fitting logic lives in regression/glm.rs.
//
// Return shape: Vec<(level, params, cov_params, scale, df_resid, deviance,
//                    null_deviance, iterations, n_obs)>.
// When by_col is None, a single-element vec with level="" is returned, so the
// Python side can treat both cases uniformly.

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::regression::glm::{fit_glm, fit_glm_by};

type GlmTuple = (String, Vec<f64>, Vec<f64>, f64, f64, f64, f64, u32, usize);

fn column_to_series(df: &DataFrame, name: &str) -> PyResult<Series> {
    df.column(name)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
        .map(|c| c.as_materialized_series().clone())
}

fn optional_column_to_series(df: &DataFrame, name: &Option<String>) -> PyResult<Option<Series>> {
    match name {
        Some(n) => column_to_series(df, n).map(Some),
        None => Ok(None),
    }
}

#[pyfunction]
#[pyo3(signature = (
    y_name,
    x_names,
    weight_name,
    stratum_name=None,
    psu_name=None,
    by_col=None,
    family="gaussian".to_string(),
    link="identity".to_string(),
    tol=1e-8,
    max_iter=100,
    data=None,
))]
pub fn fit_glm_rs(
    _py: Python,
    y_name: String,
    x_names: Vec<String>,
    weight_name: String,
    stratum_name: Option<String>,
    psu_name: Option<String>,
    by_col: Option<String>,
    family: String,
    link: String,
    tol: f64,
    max_iter: usize,
    data: Option<PyDataFrame>,
) -> PyResult<Vec<GlmTuple>> {
    let df: DataFrame = data
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("`data` is required"))?
        .into();

    let y = column_to_series(&df, &y_name)?;
    let weights = column_to_series(&df, &weight_name)?;

    let mut x_cols = Vec::with_capacity(x_names.len());
    for name in &x_names {
        x_cols.push(column_to_series(&df, name)?);
    }

    let stratum = optional_column_to_series(&df, &stratum_name)?;
    let psu = optional_column_to_series(&df, &psu_name)?;

    // No by_col: single fit, wrap in one-element vec for API uniformity.
    if by_col.is_none() {
        // Release the GIL for the (iterative, CPU-bound) IRLS solve.
        let result = _py
            .detach(|| {
                fit_glm(
                    &y,
                    x_cols,
                    &weights,
                    stratum.as_ref(),
                    psu.as_ref(),
                    &family,
                    &link,
                    tol,
                    max_iter,
                )
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        return Ok(vec![(
            String::new(),
            result.params,
            result.cov_params,
            result.scale,
            result.df_resid,
            result.deviance,
            result.null_deviance,
            result.iterations,
            result.n_obs,
        )]);
    }

    // by_col supplied: one fit per domain level (fanned out in parallel, GIL
    // released — the domain fits are independent).
    let by_series = column_to_series(&df, &by_col.unwrap())?;

    let results = _py
        .detach(|| {
            fit_glm_by(
                &y,
                x_cols,
                &weights,
                stratum.as_ref(),
                psu.as_ref(),
                &by_series,
                &family,
                &link,
                tol,
                max_iter,
            )
        })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(results
        .into_iter()
        .map(|(level, r)| {
            (
                level,
                r.params,
                r.cov_params,
                r.scale,
                r.df_resid,
                r.deviance,
                r.null_deviance,
                r.iterations,
                r.n_obs,
            )
        })
        .collect())
}
