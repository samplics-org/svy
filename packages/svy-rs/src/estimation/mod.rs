// src/estimation/mod.rs
pub mod replication;
pub mod replication_api;
pub mod taylor;
pub mod taylor_api;

// Re-export only what categorical/tabulation.rs imports via `crate::estimation::*`.
// All other callers (taylor_api, replication_api) import directly from
// crate::estimation::taylor::* to avoid unused-import warnings here.
pub use taylor::{
    degrees_of_freedom,
    point_estimate_mean,
    point_estimate_total,
    scores_mean,
    scores_total,
    srs_variance_mean,
    taylor_variance,
    taylor_variance_matrix,
};
