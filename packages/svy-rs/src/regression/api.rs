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

use crate::estimation::calib_sweep::{CalibSpec, CalibSweep, build_calib_sweep};
use crate::regression::glm::{fit_glm, fit_glm_by};

type GlmTuple = (
    String,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    f64,
    f64,
    f64,
    f64,
    u32,
    usize,
);

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
    fpc_name=None,
    offset_name=None,
    by_col=None,
    family="gaussian".to_string(),
    link="identity".to_string(),
    tol=1e-8,
    max_iter=100,
    data=None,
    calib_kind=None,
    calib_cells=None,
    calib_aux=None,
    calib_prev_wgt=None,
    calib_pins_total=None,
    calib_new_wgt=None,
))]
pub fn fit_glm_rs(
    _py: Python,
    y_name: String,
    x_names: Vec<String>,
    weight_name: String,
    stratum_name: Option<String>,
    psu_name: Option<String>,
    fpc_name: Option<String>,
    offset_name: Option<String>,
    by_col: Option<String>,
    family: String,
    link: String,
    tol: f64,
    max_iter: usize,
    data: Option<PyDataFrame>,
    calib_kind: Option<String>,
    calib_cells: Option<Vec<String>>,
    calib_aux: Option<Vec<String>>,
    calib_prev_wgt: Option<String>,
    calib_pins_total: Option<bool>,
    calib_new_wgt: Option<String>,
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
    let fpc = optional_column_to_series(&df, &fpc_name)?;
    let offset = optional_column_to_series(&df, &offset_name)?;

    // The weight-adjustment record, when Python judged it still valid. `new_wgt`
    // is normally the active weight; a subpopulation filter zeroes that column in
    // place, so Python snapshots the full-sample calibrated weights and names the
    // snapshot here.
    let calib: Option<CalibSweep> = calib_kind.and_then(|kind| {
        let spec = CalibSpec {
            kind,
            cells_cols: calib_cells.unwrap_or_default(),
            aux_cols: calib_aux.unwrap_or_default(),
            prev_wgt_col: calib_prev_wgt?,
            new_wgt_col: calib_new_wgt.unwrap_or_else(|| weight_name.clone()),
            pins_total: calib_pins_total.unwrap_or(true),
        };
        build_calib_sweep(&df, &spec)
    });
    let calib = calib.as_ref();

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
                    fpc.as_ref(),
                    offset.as_ref(),
                    &family,
                    &link,
                    tol,
                    max_iter,
                    calib,
                )
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        return Ok(vec![(
            String::new(),
            result.params,
            result.cov_params,
            result.naive_cov,
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
                fpc.as_ref(),
                offset.as_ref(),
                &by_series,
                &family,
                &link,
                tol,
                max_iter,
                calib,
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
                r.naive_cov,
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
