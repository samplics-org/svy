// src/estimation/mod.rs
pub mod taylor;  // <-- Change from `mod taylor;` to `pub mod taylor;`
pub mod replication;

// Re-export Taylor items (used by lib.rs via `use estimation::...`)
pub use taylor::{
    SingletonMethod,
    // Point estimates
    point_estimate_mean, point_estimate_total, point_estimate_ratio,
    point_estimate_mean_domain, point_estimate_total_domain, point_estimate_ratio_domain,
    // Linearization scores
    scores_mean, scores_total, scores_ratio,
    scores_mean_domain, scores_total_domain, scores_ratio_domain,
    // Taylor variance
    taylor_variance,  // <-- Remove taylor_covariance for now
    degrees_of_freedom,
    // SRS variance
    srs_variance_mean, srs_variance_mean_domain,
    srs_variance_total, srs_variance_total_domain,
    srs_variance_ratio, srs_variance_ratio_domain,
};
