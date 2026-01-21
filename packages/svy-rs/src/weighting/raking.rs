// src/weighting/raking.rs

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use super::utils::{sum_by_group_2d, check_convergence, check_bounds, Result, WeightingError};

/// Margin structure for efficient raking
struct MarginSpec {
    indices: Vec<usize>,
    targets: Array1<f64>,
    n_groups: usize,
}

impl MarginSpec {
    fn new(indices: Array1<i64>, targets: Array1<f64>) -> Self {
        // Convert i64 indices to usize and find n_groups
        let indices_usize: Vec<usize> = indices.iter().map(|&x| x as usize).collect();
        let n_groups = *indices_usize.iter().max().unwrap_or(&0) + 1;

        Self {
            indices: indices_usize,
            targets,
            n_groups,
        }
    }
}

/// Core raking algorithm (Iterative Proportional Fitting)
///
/// # Arguments
/// * `wgt` - Weight matrix (n_obs, n_reps) where each column is a replicate
/// * `margin_indices` - Vector of index arrays, one per margin
/// * `margin_targets` - Vector of target arrays, one per margin
/// * `ll_bound` - Optional lower bound for weight ratios
/// * `up_bound` - Optional upper bound for weight ratios
/// * `tol` - Convergence tolerance
/// * `max_iter` - Maximum iterations
///
/// # Returns
/// Raked weight matrix of same shape as input
pub fn rake_impl(
    wgt: ArrayView2<f64>,
    margin_indices: &[Array1<i64>],
    margin_targets: &[Array1<f64>],
    ll_bound: Option<f64>,
    up_bound: Option<f64>,
    tol: f64,
    max_iter: usize,
) -> Result<Array2<f64>> {
    let (n_obs, n_reps) = wgt.dim();

    // Validate inputs
    if margin_indices.is_empty() {
        return Err(WeightingError::InvalidInput("No margins provided".to_string()));
    }

    if margin_indices.len() != margin_targets.len() {
        return Err(WeightingError::InvalidInput(
            format!("Margin count mismatch: {} indices vs {} targets",
                    margin_indices.len(), margin_targets.len())
        ));
    }

    for (idx, indices) in margin_indices.iter().enumerate() {
        if indices.len() != n_obs {
            return Err(WeightingError::DimensionMismatch {
                expected: n_obs,
                got: indices.len(),
            });
        }
    }

    // Pre-compute margin specifications
    let margins: Vec<MarginSpec> = margin_indices
        .iter()
        .zip(margin_targets.iter())
        .map(|(idx, tgt)| MarginSpec::new(idx.clone(), tgt.clone()))
        .collect();

    // Initialize with copy of input weights
    let mut raked_weights = wgt.to_owned();
    let mut converged = false;

    // Iteration loop
    for iteration in 0..max_iter {
        let weights_start = raked_weights.clone();

        // Cycle through margins (IPF steps)
        for margin in &margins {
            // Sum current weights by group
            let current_sums = sum_by_group_2d(
                raked_weights.view(),
                &margin.indices,
                margin.n_groups,
            );

            // Calculate adjustment factors for each group
            let mut factors = Array2::zeros((margin.n_groups, n_reps));

            for g in 0..margin.n_groups {
                for r in 0..n_reps {
                    let current_sum = current_sums[[g, r]];
                    let target = margin.targets[g];

                    if current_sum > 1e-10 {
                        factors[[g, r]] = target / current_sum;
                    } else if target > 1e-10 {
                        // Target is non-zero but current sum is zero - can't solve
                        return Err(WeightingError::ZeroWeights(g));
                    } else {
                        // Both zero - no adjustment needed
                        factors[[g, r]] = 1.0;
                    }
                }
            }

            // Apply factors: multiply each observation's weight by its group's factor
            for (row_idx, &group_id) in margin.indices.iter().enumerate() {
                for col_idx in 0..n_reps {
                    raked_weights[[row_idx, col_idx]] *= factors[[group_id, col_idx]];
                }
            }
        }

        // Check convergence
        let (conv, max_diff) = check_convergence(
            raked_weights.view(),
            weights_start.view(),
            tol,
        );

        converged = conv;

        if converged {
            // Check bounds if specified
            if !check_bounds(raked_weights.view(), wgt, ll_bound, up_bound) {
                return Err(WeightingError::InvalidInput(
                    "Raking exceeded weight bounds".to_string()
                ));
            }
            break;
        }
    }

    if !converged {
        return Err(WeightingError::ConvergenceFailed(max_iter));
    }

    Ok(raked_weights)
}

/// Parallel raking - rake each replicate independently in parallel
///
/// This is useful when you have many replicates and each might converge
/// at different rates. Uses rayon for parallel processing.
pub fn rake_parallel(
    wgt: ArrayView2<f64>,
    margin_indices: &[Array1<i64>],
    margin_targets: &[Array1<f64>],
    ll_bound: Option<f64>,
    up_bound: Option<f64>,
    tol: f64,
    max_iter: usize,
) -> Result<Array2<f64>> {
    let (n_obs, n_reps) = wgt.dim();

    // Pre-compute margin specifications (shared across threads)
    let margins: Vec<MarginSpec> = margin_indices
        .iter()
        .zip(margin_targets.iter())
        .map(|(idx, tgt)| MarginSpec::new(idx.clone(), tgt.clone()))
        .collect();

    // Rake each replicate in parallel
    let results: std::result::Result<Vec<Array1<f64>>, WeightingError> = (0..n_reps)
        .into_par_iter()
        .map(|rep_idx| {
            // Extract single replicate column
            let wgt_col = wgt.column(rep_idx).to_owned();

            // Rake this replicate
            rake_single_replicate(
                wgt_col.view(),
                &margins,
                ll_bound,
                up_bound,
                tol,
                max_iter,
            )
        })
        .collect();

    let raked_cols = results?;

    // Stack columns into matrix
    let mut result = Array2::zeros((n_obs, n_reps));
    for (col_idx, col) in raked_cols.iter().enumerate() {
        result.column_mut(col_idx).assign(col);
    }

    Ok(result)
}

/// Rake a single replicate (1D weight array)
fn rake_single_replicate(
    wgt: ArrayView1<f64>,
    margins: &[MarginSpec],
    ll_bound: Option<f64>,
    up_bound: Option<f64>,
    tol: f64,
    max_iter: usize,
) -> Result<Array1<f64>> {
    let n_obs = wgt.len();
    let mut raked = wgt.to_owned();
    let mut converged = false;

    for _iteration in 0..max_iter {
        let weights_start = raked.clone();

        // Cycle through margins
        for margin in margins {
            // Sum by group
            let mut current_sums: Array1<f64> = Array1::zeros(margin.n_groups);
            for (obs_idx, &group_id) in margin.indices.iter().enumerate() {
                current_sums[group_id] += raked[obs_idx];
            }

            // Calculate factors
            let mut factors: Array1<f64> = Array1::zeros(margin.n_groups);
            for g in 0..margin.n_groups {
                if current_sums[g] > 1e-10 {
                    factors[g] = margin.targets[g] / current_sums[g];
                } else if margin.targets[g] > 1e-10 {
                    return Err(WeightingError::ZeroWeights(g));
                } else {
                    factors[g] = 1.0;
                }
            }

            // Apply factors
            for (obs_idx, &group_id) in margin.indices.iter().enumerate() {
                raked[obs_idx] *= factors[group_id];
            }
        }

        // Check convergence
        let mut max_rel_diff = 0.0;
        for (&curr, &prev) in raked.iter().zip(weights_start.iter()) {
            if prev.abs() > 1e-10 {
                let rel_diff = (curr - prev).abs() / prev.abs();
                if rel_diff > max_rel_diff {
                    max_rel_diff = rel_diff;
                }
            }
        }

        if max_rel_diff < tol {
            converged = true;
            break;
        }
    }

    if !converged {
        return Err(WeightingError::ConvergenceFailed(max_iter));
    }

    Ok(raked)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_relative_eq;

    #[test]
    fn test_simple_raking() {
        // 4 observations, 2 replicates
        let wgt = array![[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]];

        // One margin: 2 groups
        let indices = array![0, 0, 1, 1];
        let targets = array![6.0, 2.0];  // Group 0 should sum to 6, group 1 to 2

        let result = rake_impl(
            wgt.view(),
            &[indices],
            &[targets],
            None,
            None,
            1e-6,
            100,
        ).unwrap();

        // Check group sums
        let sum_g0 = result[[0, 0]] + result[[1, 0]];
        let sum_g1 = result[[2, 0]] + result[[3, 0]];

        assert_relative_eq!(sum_g0, 6.0, epsilon = 1e-6);
        assert_relative_eq!(sum_g1, 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_two_margin_raking() {
        // Classic 2x2 cross-classification
        let wgt = array![[1.0], [1.0], [1.0], [1.0]];

        // Margin 1: rows
        let row_indices = array![0, 0, 1, 1];
        let row_targets = array![10.0, 20.0];

        // Margin 2: columns
        let col_indices = array![0, 1, 0, 1];
        let col_targets = array![12.0, 18.0];

        let result = rake_impl(
            wgt.view(),
            &[row_indices, col_indices],
            &[row_targets, col_targets],
            None,
            None,
            1e-6,
            100,
        ).unwrap();

        // Total should be preserved
        let total: f64 = result.sum();
        assert_relative_eq!(total, 30.0, epsilon = 1e-6);
    }
}
