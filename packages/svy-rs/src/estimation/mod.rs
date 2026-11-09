//! Python bindings for the estimation kernels in `svy_core::estimation`.
//!
//! - Entry points release the GIL (`Python::detach`); without it rayon silently runs serially.

pub(crate) mod replication_api;
pub(crate) mod taylor_api;

use polars::prelude::DataFrame;
use pyo3_polars::PyDataFrame;

/// The frame with one chunk per column: kernels fall back to a per-element copy on chunked ones.
pub(crate) fn into_contiguous(data: PyDataFrame) -> DataFrame {
    let mut df: DataFrame = data.into();
    if df.columns().iter().any(|c| c.n_chunks() > 1) {
        df.rechunk_mut_par();
    }
    df
}
