//! Survey estimation kernels: Taylor linearisation, replication and association.
//!
//! - Parallel over by-groups and replicates only, each summed in row order: bit-identical output.

pub mod calib_sweep;
pub mod error;

pub mod association;
pub mod replication;
pub mod replication_frame;
pub mod taylor;
pub mod taylor_frame;

// Only what categorical/tabulation.rs uses; anything more warns as an unused import.
pub(crate) use taylor::{
    point_estimate_mean, point_estimate_total, scores_mean, scores_total, singleton_domain_rows,
    srs_variance_mean,
};

use crate::estimation::error::EResult;
use polars::prelude::{ChunkSort, ChunkUnique, StringChunked};

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
