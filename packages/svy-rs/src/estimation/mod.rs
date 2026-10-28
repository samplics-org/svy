//! Survey estimation kernels: Taylor linearisation, replication and association.
//!
//! - Parallel over by-groups and replicates only, each summed in row order: bit-identical output.
//! - Entry points release the GIL (`Python::detach`); without it rayon silently runs serially.

pub mod calib_sweep;
pub mod error;

pub mod association;
pub mod replication;
pub mod replication_api;
pub mod taylor;
pub mod taylor_api;

// Only what categorical/tabulation.rs uses; anything more warns as an unused import.
pub use taylor::{
    degrees_of_freedom, point_estimate_mean, point_estimate_total, scores_mean, scores_total,
    singleton_domain_rows, srs_variance_mean, taylor_variance, taylor_variance_in,
    taylor_variance_matrix, taylor_variance_matrix_in,
};

use crate::estimation::error::EResult;
use polars::prelude::{ChunkSort, ChunkUnique, DataFrame, StringChunked};
use pyo3_polars::PyDataFrame;

/// The frame with one chunk per column: kernels fall back to a per-element copy on chunked ones.
pub(crate) fn into_contiguous(data: PyDataFrame) -> DataFrame {
    let mut df: DataFrame = data.into();
    if df.columns().iter().any(|c| c.n_chunks() > 1) {
        df.rechunk_mut_par();
    }
    df
}

/// Unique by-group levels, sorted: `unique()` is hash-ordered and varies by process.
pub(crate) fn sorted_groups(by: &StringChunked) -> EResult<StringChunked> {
    Ok(by.unique()?.sort(false))
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::NewChunkedArray;

    #[test]
    fn sorted_groups_ignores_input_order() {
        let by = StringChunked::from_slice("g".into(), &["c", "a", "b", "a", "c"]);
        let groups = sorted_groups(&by).unwrap();
        let got: Vec<&str> = groups.iter().flatten().collect();
        assert_eq!(got, ["a", "b", "c"]);
    }
}
