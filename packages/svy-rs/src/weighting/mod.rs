// src/weighting/mod.rs

pub mod calibration;
mod hadamard_tables;
pub mod nonresponse;
pub mod normalization;
pub mod poststratification;
pub mod raking;
pub mod replication;
pub mod utils;

// Re-export main implementations for easier access
pub use calibration::{
    CalibrationMethod, calibrate_by_domain, calibrate_linear, calibrate_parallel,
};
pub use nonresponse::adjust_nr_impl;
pub use normalization::normalize_impl;
pub use poststratification::poststratify_impl;
pub use raking::rake_impl;
pub use replication::{
    create_bootstrap_weights, create_brr_weights, create_jkn_weights, create_sdr_weights,
};
