// src/estimation/mod.rs
mod taylor;
pub mod replication;
pub use taylor::{
    // Point estimates
    point_estimate_mean, point_estimate_total, point_estimate_ratio,
    point_estimate_mean_domain, point_estimate_total_domain, point_estimate_ratio_domain,
    // Linearization scores
    scores_mean, scores_total, scores_ratio,
    scores_mean_domain, scores_total_domain, scores_ratio_domain,
    // Taylor variance (now with two-stage support)
    taylor_variance, degrees_of_freedom,
    // SRS variance (for DEFF)
    srs_variance_mean, srs_variance_mean_domain,
    srs_variance_total, srs_variance_total_domain,
    srs_variance_ratio, srs_variance_ratio_domain,
};
pub use replication::{
    RepMethod,
    VarianceCenter,  // ← Add this
    replicate_coefficients, variance_from_replicates,
    extract_rep_weights_matrix, index_domains,
    matrix_mean_estimates, matrix_total_estimates, matrix_ratio_estimates,
    matrix_mean_by_domain, matrix_total_by_domain, matrix_ratio_by_domain,
    matrix_prop_estimates, matrix_prop_by_domain,
};
