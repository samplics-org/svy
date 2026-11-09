//! Survey kernels in plain Rust; `svy_rs` binds them to Python.
#![warn(unreachable_pub)]

pub mod errors;
pub(crate) mod rng;
pub mod sampling;
pub mod weighting;
