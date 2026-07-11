// src/estimation/mod.rs
//
// # Parallelism & determinism policy
//
// Estimation kernels parallelise over *independent* dimensions only — by-group
// cells (`taylor_api`) and replicate weights (`replication`) — never over the
// summation of a single estimate. Each group/replicate is still accumulated in
// row order, and results are collected back in the original (group, replicate)
// order. Consequently **output is bit-for-bit identical regardless of the rayon
// thread count** (`RAYON_NUM_THREADS`) and stable run-to-run for a given input.
// The golden suite is the contract; changing thread count must not change a bit.
//
// The PyO3 entry points release the GIL (`Python::detach`) around this compute.
// This is mandatory, not cosmetic: without it the interpreter lock keeps the
// rayon worker threads from ever running the closures, so the `par_iter` /
// per-replicate parallelism silently executes serially. Release-then-parallelise
// is what turns the fan-out into real wall-clock speedup (measured 3–6×).
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
