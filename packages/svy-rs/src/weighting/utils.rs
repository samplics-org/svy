// src/weighting/utils.rs

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum WeightingError {
    #[error("Convergence failed after {0} iterations")]
    ConvergenceFailed(usize),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("All weights are zero in group {0}")]
    ZeroWeights(usize),
}

pub type Result<T> = std::result::Result<T, WeightingError>;

/// Sum weights by group for a 1D array
pub fn sum_by_group_1d(
    weights: ArrayView1<f64>,
    indices: &[usize],
    n_groups: usize,
) -> Array1<f64> {
    let mut sums = Array1::zeros(n_groups);

    for (i, &group_id) in indices.iter().enumerate() {
        sums[group_id] += weights[i];
    }

    sums
}

/// Sum weights by group for a 2D array (each column separately)
pub fn sum_by_group_2d(
    weights: ArrayView2<f64>,
    indices: &[usize],
    n_groups: usize,
) -> Array2<f64> {
    let n_cols = weights.ncols();
    let mut sums = Array2::zeros((n_groups, n_cols));

    for (row_idx, &group_id) in indices.iter().enumerate() {
        for col_idx in 0..n_cols {
            sums[[group_id, col_idx]] += weights[[row_idx, col_idx]];
        }
    }

    sums
}

/// Check convergence by comparing maximum relative difference
pub fn check_convergence(
    current: ArrayView2<f64>,
    previous: ArrayView2<f64>,
    tol: f64,
) -> (bool, f64) {
    let mut max_rel_diff = 0.0;

    for (curr, prev) in current.iter().zip(previous.iter()) {
        if prev.abs() > 1e-10 {
            let rel_diff = (curr - prev).abs() / prev.abs();
            if rel_diff > max_rel_diff {
                max_rel_diff = rel_diff;
            }
        }
    }

    (max_rel_diff < tol, max_rel_diff)
}

/// Check if weights are within bounds
pub fn check_bounds(
    raked: ArrayView2<f64>,
    initial: ArrayView2<f64>,
    ll_bound: Option<f64>,
    up_bound: Option<f64>,
) -> bool {
    if ll_bound.is_none() && up_bound.is_none() {
        return true;
    }

    for (&raked_w, &init_w) in raked.iter().zip(initial.iter()) {
        if init_w > 1e-10 {
            let ratio = raked_w / init_w;

            if let Some(ll) = ll_bound {
                if ratio < ll {
                    return false;
                }
            }

            if let Some(up) = up_bound {
                if ratio > up {
                    return false;
                }
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_sum_by_group_1d() {
        let weights = array![1.0, 2.0, 3.0, 4.0];
        let indices = vec![0, 0, 1, 1];
        let sums = sum_by_group_1d(weights.view(), &indices, 2);

        assert_eq!(sums[0], 3.0);
        assert_eq!(sums[1], 7.0);
    }

    #[test]
    fn test_sum_by_group_2d() {
        let weights = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let indices = vec![0, 1, 0];
        let sums = sum_by_group_2d(weights.view(), &indices, 2);

        assert_eq!(sums[[0, 0]], 6.0);  // 1 + 5
        assert_eq!(sums[[0, 1]], 8.0);  // 2 + 6
        assert_eq!(sums[[1, 0]], 3.0);
        assert_eq!(sums[[1, 1]], 4.0);
    }

    #[test]
    fn test_check_convergence() {
        let current = array![[1.0, 2.0], [3.0, 4.0]];
        let previous = array![[1.01, 2.01], [3.01, 4.01]];

        let (converged, max_diff) = check_convergence(
            current.view(),
            previous.view(),
            0.02,
        );

        assert!(converged);
        assert!(max_diff < 0.02);
    }
}
