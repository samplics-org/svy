// src/weighting/mod.rs

pub mod utils;
pub mod raking;
pub mod nonresponse;
pub mod normalization;
pub mod poststratification;
pub mod calibration;
pub mod replication;
mod hadamard_tables;

// Re-export main implementations for easier access
pub use raking::rake_impl;
pub use nonresponse::adjust_nr_impl;
pub use normalization::normalize_impl;
pub use poststratification::poststratify_impl;
pub use calibration::{calibrate_linear, calibrate_by_domain, calibrate_parallel, CalibrationMethod};
pub use replication::{create_brr_weights, create_jkn_weights, create_bootstrap_weights, create_sdr_weights};
