//! Survey kernels in plain Rust; `svy_rs` binds them to Python.
#![warn(unreachable_pub)]

pub mod categorical;
pub mod errors;
pub mod estimation;
pub mod regression;
pub(crate) mod rng;
pub mod sampling;
#[cfg(test)]
mod test_fixtures;
pub mod weighting;
