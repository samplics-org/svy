// src/lib.rs
use pyo3::prelude::*;

mod categorical;
mod estimation;
mod regression;
mod weighting;

// ============================================================================
// Module Registration
// ============================================================================

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Taylor linearization functions
    m.add_function(wrap_pyfunction!(estimation::taylor_api::taylor_mean, m)?)?;
    m.add_function(wrap_pyfunction!(estimation::taylor_api::taylor_total, m)?)?;
    m.add_function(wrap_pyfunction!(estimation::taylor_api::taylor_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(estimation::taylor_api::taylor_prop, m)?)?;
    m.add_function(wrap_pyfunction!(estimation::taylor_api::taylor_median, m)?)?;
    // Replication-based estimation functions
    m.add_function(wrap_pyfunction!(estimation::replication_api::replicate_mean, m)?)?;
    m.add_function(wrap_pyfunction!(estimation::replication_api::replicate_total, m)?)?;
    m.add_function(wrap_pyfunction!(estimation::replication_api::replicate_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(estimation::replication_api::replicate_prop, m)?)?;
    m.add_function(wrap_pyfunction!(estimation::replication_api::replicate_median, m)?)?;
    // GLM regression
    m.add_function(wrap_pyfunction!(regression::api::fit_glm_rs, m)?)?;
    // Categorical tests
    m.add_function(wrap_pyfunction!(categorical::api::ttest_rs, m)?)?;
    m.add_function(wrap_pyfunction!(categorical::api::ranktest_rs, m)?)?;
    m.add_function(wrap_pyfunction!(categorical::api::tabulate_rs, m)?)?;
    // Weighting and calibration
    m.add_function(wrap_pyfunction!(weighting::api::rake, m)?)?;
    m.add_function(wrap_pyfunction!(weighting::api::adjust_nr, m)?)?;
    m.add_function(wrap_pyfunction!(weighting::api::normalize, m)?)?;
    m.add_function(wrap_pyfunction!(weighting::api::poststratify, m)?)?;
    m.add_function(wrap_pyfunction!(weighting::api::poststratify_factor, m)?)?;
    m.add_function(wrap_pyfunction!(weighting::api::calibrate, m)?)?;
    m.add_function(wrap_pyfunction!(weighting::api::calibrate_by_domain, m)?)?;
    m.add_function(wrap_pyfunction!(weighting::api::calibrate_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(weighting::api::trim_weights, m)?)?;
    m.add_function(wrap_pyfunction!(weighting::api::trim_weights_matrix, m)?)?;
    // Replicate weight creation
    m.add_function(wrap_pyfunction!(weighting::api::create_brr_wgts, m)?)?;
    m.add_function(wrap_pyfunction!(weighting::api::create_jk_wgts, m)?)?;
    m.add_function(wrap_pyfunction!(weighting::api::create_bootstrap_wgts, m)?)?;
    m.add_function(wrap_pyfunction!(weighting::api::create_sdr_wgts, m)?)?;
    Ok(())
}
