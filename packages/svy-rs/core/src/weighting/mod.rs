//! Weight adjustment, calibration, trimming and replicate-weight kernels.
pub mod calibration;
pub(crate) mod hadamard_tables;
pub mod nonresponse;
pub mod normalization;
pub mod poststratification;
pub mod raking;
pub mod replication;
pub mod trimming;
pub mod utils;
