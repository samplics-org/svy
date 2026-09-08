// src/regression/api.rs
//
// PyO3-facing wrapper for the GLM regression function.
// The actual fitting logic lives in regression/glm.rs.
//
// Return shape: Vec<(level, params, cov_params, naive_cov, scale, df_resid,
//                    deviance, null_deviance, iterations, n_obs, converged,
//                    (theta, theta_se) | None)>.
// When neither by_col nor where_col is given, a single-element vec with
// level="" is returned, so the Python side can treat every case uniformly.

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::estimation::calib_sweep::{CalibSpec, CalibSweep, build_calib_sweep};
use crate::regression::glm::{
    design_codes, design_vcov_of_totals, fit_glm, fit_glm_by, fit_glm_where,
};

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
    bool,
    // (theta, theta_se) when the family has a dispersion parameter; theta_se
    // is None when theta was supplied rather than estimated. One element
    // rather than two because pyo3 only converts tuples up to twelve long.
    Option<(f64, Option<f64>)>,
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
    where_col=None,
    family="gaussian".to_string(),
    link="identity".to_string(),
    theta=None,
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
    where_col: Option<String>,
    family: String,
    link: String,
    theta: Option<f64>,
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

    // A `where=` domain is one fit. It used to be routed through `fit_glm_by`
    // on a "true"/"false" string column, which fitted the complement as well
    // and discarded it.
    if let Some(name) = where_col {
        let mask = column_to_series(&df, &name)?;
        let result = _py
            .detach(|| {
                fit_glm_where(
                    &y,
                    x_cols,
                    &weights,
                    stratum.as_ref(),
                    psu.as_ref(),
                    fpc.as_ref(),
                    offset.as_ref(),
                    &mask,
                    &family,
                    &link,
                    theta,
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
            result.converged,
            result.theta.map(|t| (t, result.theta_se)),
        )]);
    }

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
                    theta,
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
            result.converged,
            result.theta.map(|t| (t, result.theta_se)),
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
                theta,
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
                r.converged,
                r.theta.map(|t| (t, r.theta_se)),
            )
        })
        .collect())
}

/// Design-based variance-covariance of the totals of several already-weighted
/// columns — R survey's `svyrecvar`.
///
/// Returns the full symmetric `p x p` matrix, row-major. The columns are
/// influence functions, so the weights are already folded into them; this
/// computes only the design part (PSU totals centred within stratum, the
/// with-replacement factor, the stratum FPC).
///
/// The GLM sandwich uses the same code internally. It is exposed because the
/// negative binomial's joint (theta, beta) variance is assembled on the Python
/// side, where digamma and trigamma live — svy-rs carries no math dependency —
/// and only this part belongs in the kernel.
#[pyfunction]
#[pyo3(signature = (data, value_cols, strata_col=None, psu_col=None, fpc_col=None))]
pub fn design_vcov_rs(
    _py: Python,
    data: PyDataFrame,
    value_cols: Vec<String>,
    strata_col: Option<String>,
    psu_col: Option<String>,
    fpc_col: Option<String>,
) -> PyResult<Vec<f64>> {
    let df: DataFrame = data.into();
    let n = df.height();
    let p = value_cols.len();
    if p == 0 {
        return Ok(Vec::new());
    }

    let mut cols = vec![0.0f64; n * p];
    for (j, name) in value_cols.iter().enumerate() {
        let s = column_to_series(&df, name)?;
        let cast = s
            .cast(&DataType::Float64)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        let ca = cast
            .f64()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        for (i, v) in ca.iter().enumerate() {
            cols[j * n + i] = v.unwrap_or(0.0);
        }
    }

    let strata = optional_column_to_series(&df, &strata_col)?;
    let psu = optional_column_to_series(&df, &psu_col)?;
    let fpc = optional_column_to_series(&df, &fpc_col)?;

    _py.detach(|| {
        let (strata_idx, n_strata) = match strata {
            Some(ref s) => design_codes(Some(s), None, n)?,
            None => (vec![0usize; n], 1usize),
        };
        let (psu_idx, n_psu_levels) = match psu {
            Some(ref pcol) => design_codes(strata.as_ref(), Some(pcol), n)?,
            None => ((0..n).collect::<Vec<_>>(), n),
        };

        let mut strata_obs: Vec<Vec<usize>> = vec![Vec::new(); n_strata];
        for i in 0..n {
            strata_obs[strata_idx[i]].push(i);
        }

        let fpc_rows: Option<Vec<f64>> = match fpc {
            Some(ref s) => {
                let cast = s.cast(&DataType::Float64)?;
                let ca = cast.f64()?;
                Some(ca.iter().map(|v| v.unwrap_or(1.0)).collect())
            }
            None => None,
        };

        let psu_opt = if psu.is_some() {
            Some(psu_idx.as_slice())
        } else {
            None
        };

        Ok(design_vcov_of_totals(
            &cols,
            n,
            p,
            &strata_idx,
            &strata_obs,
            psu_opt,
            n_psu_levels,
            fpc_rows.as_deref(),
        ))
    })
    .map_err(|e: PolarsError| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}
