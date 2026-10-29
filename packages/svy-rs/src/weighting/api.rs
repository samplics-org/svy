//! PyO3 glue for the weighting kernels; the algorithms live in the sibling modules.

use ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::errors::to_py_err;
use crate::weighting::replication::ReplicationError;

/// What `trim_weights` returns to Python.
type TrimOutput = (
    Py<numpy::PyArray1<f64>>, // trimmed weights
    usize,                    // n_trimmed_upper
    usize,                    // n_trimmed_lower
    f64,                      // weight_sum_before
    f64,                      // weight_sum_after
    f64,                      // ess_before
    f64,                      // ess_after
    usize,                    // iterations
    bool,                     // converged
);

// ============================================================================
// Raking
// ============================================================================

#[pyfunction]
#[pyo3(signature = (wgt, margin_indices, margin_targets, tol=1e-6, max_iter=100))]
pub(crate) fn rake(
    py: Python<'_>,
    wgt: PyReadonlyArray2<f64>,
    margin_indices: Vec<PyReadonlyArray1<i64>>,
    margin_targets: Vec<PyReadonlyArray1<f64>>,
    tol: f64,
    max_iter: usize,
) -> PyResult<Py<PyArray2<f64>>> {
    let wgt_arr = wgt.as_array();
    let indices: Vec<_> = margin_indices
        .iter()
        .map(|x| x.as_array().to_owned())
        .collect();
    let targets: Vec<_> = margin_targets
        .iter()
        .map(|x| x.as_array().to_owned())
        .collect();

    let result = crate::weighting::raking::rake_impl(wgt_arr, &indices, &targets, tol, max_iter)
        .map_err(|e| to_py_err(&e))?;

    Ok(result.into_pyarray(py).to_owned().into())
}

// ============================================================================
// Non-response adjustment
// ============================================================================

#[pyfunction]
#[pyo3(signature = (wgts, adj_class, resp_status, unknown_to_inelig=true))]
pub(crate) fn adjust_nr(
    py: Python<'_>,
    wgts: PyReadonlyArray2<f64>,
    adj_class: PyReadonlyArray1<i64>,
    resp_status: PyReadonlyArray1<i64>,
    unknown_to_inelig: bool,
) -> PyResult<Py<PyArray2<f64>>> {
    let result = crate::weighting::nonresponse::adjust_nr_impl(
        wgts.as_array(),
        adj_class.as_array(),
        resp_status.as_array(),
        unknown_to_inelig,
    )
    .map_err(|e| to_py_err(&e))?;

    Ok(result.into_pyarray(py).to_owned().into())
}

// ============================================================================
// Normalization
// ============================================================================

#[pyfunction]
#[pyo3(signature = (wgt, by_arr=None, control=None))]
pub(crate) fn normalize(
    py: Python<'_>,
    wgt: PyReadonlyArray2<f64>,
    by_arr: Option<PyReadonlyArray1<i64>>,
    control: Option<PyReadonlyArray1<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let by = by_arr.map(|x| x.as_array().to_owned());
    let ctrl = control.map(|x| x.as_array().to_owned());

    let result =
        crate::weighting::normalization::normalize_impl(wgt.as_array(), by.as_ref(), ctrl.as_ref())
            .map_err(|e| to_py_err(&e))?;

    Ok(result.into_pyarray(py).to_owned().into())
}

// ============================================================================
// Post-stratification
// ============================================================================

#[pyfunction]
pub(crate) fn poststratify(
    py: Python<'_>,
    wgt: PyReadonlyArray2<f64>,
    by_arr: PyReadonlyArray1<i64>,
    control: PyReadonlyArray1<f64>,
) -> PyResult<Py<PyArray2<f64>>> {
    let ctrl = control.as_array().to_owned();
    let result = crate::weighting::poststratification::poststratify_impl(
        wgt.as_array(),
        by_arr.as_array(),
        &ctrl,
    )
    .map_err(|e| to_py_err(&e))?;

    Ok(result.into_pyarray(py).to_owned().into())
}

// ============================================================================
// Calibration
// ============================================================================

#[pyfunction]
#[pyo3(signature = (wgt, x_matrix, totals, scale=None, additive=false))]
pub(crate) fn calibrate(
    py: Python<'_>,
    wgt: PyReadonlyArray2<f64>,
    x_matrix: PyReadonlyArray2<f64>,
    totals: PyReadonlyArray1<f64>,
    scale: Option<PyReadonlyArray1<f64>>,
    additive: bool,
) -> PyResult<Py<PyArray2<f64>>> {
    let scale_owned: Option<Array1<f64>> = scale.map(|s| s.as_array().to_owned());
    let scale_view: Option<ArrayView1<f64>> = scale_owned.as_ref().map(|a| a.view());

    let result = crate::weighting::calibration::calibrate_linear(
        wgt.as_array(),
        x_matrix.as_array(),
        totals.as_array(),
        scale_view,
        additive,
    )
    .map_err(|e| to_py_err(&e))?;

    Ok(result.into_pyarray(py).to_owned().into())
}

#[pyfunction]
#[pyo3(signature = (wgt, x_matrix, domain, controls_dict, scale=None, additive=false))]
pub(crate) fn calibrate_by_domain(
    py: Python<'_>,
    wgt: PyReadonlyArray2<f64>,
    x_matrix: PyReadonlyArray2<f64>,
    domain: PyReadonlyArray1<i64>,
    controls_dict: std::collections::HashMap<i64, Vec<f64>>,
    scale: Option<PyReadonlyArray1<f64>>,
    additive: bool,
) -> PyResult<Py<PyArray2<f64>>> {
    let scale_owned: Option<Array1<f64>> = scale.map(|s| s.as_array().to_owned());
    let scale_view: Option<ArrayView1<f64>> = scale_owned.as_ref().map(|a| a.view());

    let controls: std::collections::HashMap<i64, Array1<f64>> = controls_dict
        .into_iter()
        .map(|(k, v)| (k, Array1::from(v)))
        .collect();

    let result = crate::weighting::calibration::calibrate_by_domain(
        wgt.as_array(),
        x_matrix.as_array(),
        domain.as_array(),
        &controls,
        scale_view,
        additive,
    )
    .map_err(|e| to_py_err(&e))?;

    Ok(result.into_pyarray(py).to_owned().into())
}

#[pyfunction]
#[pyo3(signature = (wgt, x_matrix, totals, scale=None))]
pub(crate) fn calibrate_parallel(
    py: Python<'_>,
    wgt: PyReadonlyArray2<f64>,
    x_matrix: PyReadonlyArray2<f64>,
    totals: PyReadonlyArray1<f64>,
    scale: Option<PyReadonlyArray1<f64>>,
) -> PyResult<Py<PyArray2<f64>>> {
    let scale_owned: Option<Array1<f64>> = scale.map(|s| s.as_array().to_owned());
    let scale_view: Option<ArrayView1<f64>> = scale_owned.as_ref().map(|a| a.view());

    let result = crate::weighting::calibration::calibrate_parallel(
        wgt.as_array(),
        x_matrix.as_array(),
        totals.as_array(),
        scale_view,
    )
    .map_err(|e| to_py_err(&e))?;

    Ok(result.into_pyarray(py).to_owned().into())
}

/// What the bounded calibrations return: weights, then per column the largest
/// `|misfit| / (1 + |t|)` and the Newton steps taken.
type BoundedOutput = (
    Py<PyArray2<f64>>,
    Py<numpy::PyArray1<f64>>,
    Py<numpy::PyArray1<usize>>,
);

#[pyfunction]
#[pyo3(signature = (wgt, x_matrix, totals, lower, upper, tol, max_iter, scale=None, distance="linear"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn calibrate_bounded(
    py: Python<'_>,
    wgt: PyReadonlyArray2<f64>,
    x_matrix: PyReadonlyArray2<f64>,
    totals: PyReadonlyArray1<f64>,
    lower: f64,
    upper: f64,
    tol: f64,
    max_iter: usize,
    scale: Option<PyReadonlyArray1<f64>>,
    distance: &str,
) -> PyResult<BoundedOutput> {
    let distance =
        crate::weighting::calibration::Distance::parse(distance).map_err(|e| to_py_err(&e))?;
    let scale_owned: Option<Array1<f64>> = scale.map(|s| s.as_array().to_owned());
    let (w, fits) = crate::weighting::calibration::calibrate_bounded(
        wgt.as_array(),
        x_matrix.as_array(),
        totals.as_array(),
        scale_owned.as_ref().map(|a| a.view()),
        crate::weighting::calibration::BoundedSolve {
            lower,
            upper,
            tol,
            max_iter,
            distance,
        },
    )
    .map_err(|e| to_py_err(&e))?;
    let achieved: Vec<f64> = fits.iter().map(|f| f.achieved).collect();
    let steps: Vec<usize> = fits.iter().map(|f| f.steps).collect();
    Ok((
        w.into_pyarray(py).to_owned().into(),
        Array1::from(achieved).into_pyarray(py).to_owned().into(),
        Array1::from(steps).into_pyarray(py).to_owned().into(),
    ))
}

/// `calibrate_bounded` per domain: weights, then per domain (rows, ascending domain code) and
/// column the largest `|misfit| / (1 + |t|)` and the Newton steps taken.
type BoundedByDomainOutput = (
    Py<PyArray2<f64>>,
    Vec<i64>,
    Py<PyArray2<f64>>,
    Py<PyArray2<usize>>,
);

#[pyfunction]
#[pyo3(signature = (wgt, x_matrix, domain, controls_dict, lower, upper, tol, max_iter, scale=None, distance="linear"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn calibrate_bounded_by_domain(
    py: Python<'_>,
    wgt: PyReadonlyArray2<f64>,
    x_matrix: PyReadonlyArray2<f64>,
    domain: PyReadonlyArray1<i64>,
    controls_dict: std::collections::HashMap<i64, Vec<f64>>,
    lower: f64,
    upper: f64,
    tol: f64,
    max_iter: usize,
    scale: Option<PyReadonlyArray1<f64>>,
    distance: &str,
) -> PyResult<BoundedByDomainOutput> {
    let distance =
        crate::weighting::calibration::Distance::parse(distance).map_err(|e| to_py_err(&e))?;
    let scale_owned: Option<Array1<f64>> = scale.map(|s| s.as_array().to_owned());
    let controls: std::collections::HashMap<i64, Array1<f64>> = controls_dict
        .into_iter()
        .map(|(k, v)| (k, Array1::from(v)))
        .collect();
    let (w, domains, fits) = crate::weighting::calibration::calibrate_bounded_by_domain(
        wgt.as_array(),
        x_matrix.as_array(),
        domain.as_array(),
        &controls,
        scale_owned.as_ref().map(|a| a.view()),
        crate::weighting::calibration::BoundedSolve {
            lower,
            upper,
            tol,
            max_iter,
            distance,
        },
    )
    .map_err(|e| to_py_err(&e))?;
    let n_cols = w.ncols();
    let achieved =
        ndarray::Array2::from_shape_fn((fits.len(), n_cols), |(d, r)| fits[d][r].achieved);
    let steps = ndarray::Array2::from_shape_fn((fits.len(), n_cols), |(d, r)| fits[d][r].steps);
    Ok((
        w.into_pyarray(py).to_owned().into(),
        domains,
        achieved.into_pyarray(py).to_owned().into(),
        steps.into_pyarray(py).to_owned().into(),
    ))
}

// ============================================================================
// Replicate weight creation
// ============================================================================

/// BRR Hadamard order (max distinct replicates) for `n_strata`, so Python validates n_reps first.
#[pyfunction]
pub(crate) fn brr_hadamard_size(n_strata: usize) -> PyResult<usize> {
    let (_h, size) =
        crate::weighting::replication::get_hadamard_for_brr(n_strata).map_err(|e| to_py_err(&e))?;
    Ok(size)
}

#[pyfunction]
#[pyo3(signature = (wgt, stratum, psu, n_reps=None, fay_coef=0.0, seed=None))]
pub(crate) fn create_brr_wgts(
    py: Python<'_>,
    wgt: PyReadonlyArray1<f64>,
    stratum: PyReadonlyArray1<i64>,
    psu: PyReadonlyArray1<i64>,
    n_reps: Option<usize>,
    fay_coef: f64,
    seed: Option<u64>,
) -> PyResult<(Py<PyArray2<f64>>, f64)> {
    let (result, df) = crate::weighting::replication::create_brr_weights(
        wgt.as_array(),
        stratum.as_array(),
        psu.as_array(),
        n_reps,
        fay_coef,
        seed,
    )
    .map_err(|e| to_py_err(&e))?;

    Ok((result.into_pyarray(py).to_owned().into(), df))
}

#[pyfunction]
#[pyo3(signature = (wgt, psu, stratum=None, paired=false, seed=None))]
pub(crate) fn create_jk_wgts(
    py: Python<'_>,
    wgt: PyReadonlyArray1<f64>,
    psu: PyReadonlyArray1<i64>,
    stratum: Option<PyReadonlyArray1<i64>>,
    paired: bool,
    seed: Option<u64>,
) -> PyResult<(Py<PyArray2<f64>>, f64, Vec<f64>)> {
    let stratum_view = stratum.as_ref().map(|s| s.as_array());
    let (result, df, rscales) = crate::weighting::replication::create_jk_weights(
        wgt.as_array(),
        stratum_view,
        psu.as_array(),
        paired,
        seed,
    )
    .map_err(|e| to_py_err(&e))?;

    Ok((result.into_pyarray(py).to_owned().into(), df, rscales))
}

#[pyfunction]
#[pyo3(signature = (wgt, psu, n_reps, stratum=None, seed=None))]
pub(crate) fn create_bootstrap_wgts(
    py: Python<'_>,
    wgt: PyReadonlyArray1<f64>,
    psu: PyReadonlyArray1<i64>,
    n_reps: usize,
    stratum: Option<PyReadonlyArray1<i64>>,
    seed: Option<u64>,
) -> PyResult<(Py<PyArray2<f64>>, f64)> {
    let stratum_view = stratum.as_ref().map(|s| s.as_array());
    let seed = seed.ok_or_else(|| to_py_err(&ReplicationError::MissingSeed))?;

    let (result, df) = crate::weighting::replication::create_bootstrap_weights(
        wgt.as_array(),
        stratum_view,
        psu.as_array(),
        n_reps,
        seed,
    )
    .map_err(|e| to_py_err(&e))?;

    Ok((result.into_pyarray(py).to_owned().into(), df))
}

#[pyfunction]
#[pyo3(signature = (wgt, n_reps, seed=None))]
pub(crate) fn create_poisson_bootstrap_wgts(
    py: Python<'_>,
    wgt: PyReadonlyArray1<f64>,
    n_reps: usize,
    seed: Option<u64>,
) -> PyResult<(Py<PyArray2<f64>>, f64)> {
    let seed = seed.ok_or_else(|| to_py_err(&ReplicationError::MissingSeed))?;

    let (result, df) = crate::weighting::replication::create_poisson_bootstrap_weights(
        wgt.as_array(),
        n_reps,
        seed,
    )
    .map_err(|e| to_py_err(&e))?;

    Ok((result.into_pyarray(py).to_owned().into(), df))
}

#[pyfunction]
#[pyo3(signature = (wgt, n_reps, stratum=None, order=None))]
pub(crate) fn create_sdr_wgts(
    py: Python<'_>,
    wgt: PyReadonlyArray1<f64>,
    n_reps: usize,
    stratum: Option<PyReadonlyArray1<i64>>,
    order: Option<PyReadonlyArray1<i64>>,
) -> PyResult<(Py<PyArray2<f64>>, f64)> {
    let stratum_view = stratum.as_ref().map(|s| s.as_array());
    let order_view = order.as_ref().map(|o| o.as_array());

    let (result, df) = crate::weighting::replication::create_sdr_weights(
        wgt.as_array(),
        stratum_view,
        order_view,
        n_reps,
    )
    .map_err(|e| to_py_err(&e))?;

    Ok((result.into_pyarray(py).to_owned().into(), df))
}

// ============================================================================
// Trimming
// ============================================================================

#[pyfunction]
#[pyo3(signature = (weights, upper=None, lower=None, redistribute=true, max_iter=10, tol=1e-6))]
pub(crate) fn trim_weights(
    py: Python<'_>,
    weights: PyReadonlyArray1<f64>,
    upper: Option<f64>,
    lower: Option<f64>,
    redistribute: bool,
    max_iter: usize,
    tol: f64,
) -> PyResult<TrimOutput> {
    let out = crate::weighting::trimming::trim_impl(
        weights.as_array(),
        upper,
        lower,
        redistribute,
        max_iter,
        tol,
    )
    .map_err(|e| to_py_err(&e))?;

    Ok((
        out.weights.into_pyarray(py).to_owned().into(),
        out.n_trimmed_upper,
        out.n_trimmed_lower,
        out.weight_sum_before,
        out.weight_sum_after,
        out.ess_before,
        out.ess_after,
        out.iterations,
        out.converged,
    ))
}

#[pyfunction]
#[pyo3(signature = (weights, upper=None, lower=None, redistribute=true, max_iter=10, tol=1e-6))]
pub(crate) fn trim_weights_matrix(
    py: Python<'_>,
    weights: PyReadonlyArray2<f64>,
    upper: Option<f64>,
    lower: Option<f64>,
    redistribute: bool,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    let out = crate::weighting::trimming::trim_matrix_impl(
        weights.as_array(),
        upper,
        lower,
        redistribute,
        max_iter,
        tol,
    )
    .map_err(|e| to_py_err(&e))?;

    Ok(out.into_pyarray(py).to_owned().into())
}
