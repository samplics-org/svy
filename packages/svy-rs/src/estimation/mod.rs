// src/estimation/mod.rs
pub mod replication;
pub mod taylor;

// Re-export Taylor items (used by lib.rs via `use estimation::...`)
pub use taylor::{
    SingletonMethod,
    SvyQuantileMethod, // Renamed to avoid collision with polars::prelude::QuantileMethod
    degrees_of_freedom,
    median_variance_woodruff,
    median_variance_woodruff_domain,
    // Point estimates
    point_estimate_mean,
    point_estimate_mean_domain,
    point_estimate_ratio,
    point_estimate_ratio_domain,
    point_estimate_total,
    point_estimate_total_domain,
    // Linearization scores
    scores_mean,
    scores_mean_domain,
    // Median scores
    scores_median,
    scores_median_domain,
    scores_ratio,
    scores_ratio_domain,
    scores_total,
    scores_total_domain,
    // SRS variance
    srs_variance_mean,
    srs_variance_mean_domain,
    srs_variance_ratio,
    srs_variance_ratio_domain,
    srs_variance_total,
    srs_variance_total_domain,
    // Taylor variance
    taylor_variance,
    taylor_variance_matrix,
    weighted_median,
    weighted_median_domain,
    // Median/Quantile functions
    weighted_quantile,
    weighted_quantile_chunked,
};
