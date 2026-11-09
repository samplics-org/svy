//! Rust survey kernels exposed to Python as the `_internal` module.
#![warn(unreachable_pub)]
use pyo3::prelude::*;

mod categorical;
mod errors;
mod estimation;
mod regression;
mod sampling;
mod weighting;

// ============================================================================
// Module Registration
// ============================================================================

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add(
        "KernelValueError",
        py.get_type::<errors::KernelValueError>(),
    )?;
    m.add(
        "KernelRuntimeError",
        py.get_type::<errors::KernelRuntimeError>(),
    )?;

    m.add_function(wrap_pyfunction!(estimation::taylor_api::taylor_mean, m)?)?;
    m.add_function(wrap_pyfunction!(
        estimation::taylor_api::taylor_mean_multi,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(estimation::taylor_api::taylor_total, m)?)?;
    m.add_function(wrap_pyfunction!(
        estimation::taylor_api::taylor_total_multi,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(estimation::taylor_api::taylor_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(
        estimation::taylor_api::taylor_ratio_multi,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(estimation::taylor_api::taylor_assoc, m)?)?;
    m.add_function(wrap_pyfunction!(estimation::taylor_api::taylor_prop, m)?)?;
    m.add_function(wrap_pyfunction!(
        estimation::taylor_api::taylor_factor_total,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        estimation::taylor_api::taylor_prop_multi,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(estimation::taylor_api::taylor_median, m)?)?;
    m.add_function(wrap_pyfunction!(
        estimation::taylor_api::taylor_median_multi,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        estimation::taylor_api::taylor_quantile,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        estimation::taylor_api::taylor_quantile_multi,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        estimation::taylor_api::weighted_quantile_at,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        estimation::replication_api::replicate_mean,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        estimation::replication_api::replicate_total,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        estimation::replication_api::replicate_factor_total,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        estimation::replication_api::replicate_ratio,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        estimation::replication_api::replicate_assoc,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        estimation::replication_api::replicate_prop,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        estimation::replication_api::replicate_median,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        estimation::replication_api::replicate_quantile,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(regression::api::fit_glm_rs, m)?)?;
    m.add_function(wrap_pyfunction!(regression::api::fit_glm_reps_rs, m)?)?;
    m.add_function(wrap_pyfunction!(regression::api::design_vcov_rs, m)?)?;
    m.add_function(wrap_pyfunction!(categorical::api::ttest_rs, m)?)?;
    m.add_function(wrap_pyfunction!(categorical::api::ranktest_rs, m)?)?;
    m.add_function(wrap_pyfunction!(categorical::api::tabulate_rs, m)?)?;
    m.add_function(wrap_pyfunction!(weighting::rake, m)?)?;
    m.add_function(wrap_pyfunction!(weighting::adjust_nr, m)?)?;
    m.add_function(wrap_pyfunction!(weighting::normalize, m)?)?;
    m.add_function(wrap_pyfunction!(weighting::poststratify, m)?)?;
    m.add_function(wrap_pyfunction!(weighting::calibrate, m)?)?;
    m.add_function(wrap_pyfunction!(weighting::calibrate_by_domain, m)?)?;
    m.add_function(wrap_pyfunction!(weighting::calibrate_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(weighting::calibrate_bounded, m)?)?;
    m.add_function(wrap_pyfunction!(weighting::calibrate_bounded_by_domain, m)?)?;
    m.add_function(wrap_pyfunction!(weighting::trim_weights, m)?)?;
    m.add_function(wrap_pyfunction!(weighting::trim_weights_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(weighting::brr_hadamard_size, m)?)?;
    m.add_function(wrap_pyfunction!(weighting::create_brr_wgts, m)?)?;
    m.add_function(wrap_pyfunction!(weighting::create_jk_wgts, m)?)?;
    m.add_function(wrap_pyfunction!(weighting::create_bootstrap_wgts, m)?)?;
    m.add_function(wrap_pyfunction!(
        weighting::create_poisson_bootstrap_wgts,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(weighting::create_sdr_wgts, m)?)?;
    m.add_function(wrap_pyfunction!(sampling::select_srs_rs, m)?)?;
    m.add_function(wrap_pyfunction!(sampling::select_pps_rs, m)?)?;
    m.add_function(wrap_pyfunction!(sampling::select_srs_batch_rs, m)?)?;
    m.add_function(wrap_pyfunction!(sampling::select_pps_batch_rs, m)?)?;
    Ok(())
}
