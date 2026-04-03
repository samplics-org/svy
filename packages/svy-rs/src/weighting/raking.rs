// src/weighting/raking.rs

use super::utils::{Result, WeightingError, check_bounds, check_convergence, sum_by_group_2d};
use ndarray::{Array1, Array2, ArrayView2};

struct MarginSpec {
    indices: Vec<usize>,
    targets: Array1<f64>,
    n_groups: usize,
}

impl MarginSpec {
    fn new(indices: Array1<i64>, targets: Array1<f64>) -> Self {
        let indices_usize: Vec<usize> = indices.iter().map(|&x| x as usize).collect();
        let n_groups = *indices_usize.iter().max().unwrap_or(&0) + 1;
        Self { indices: indices_usize, targets, n_groups }
    }
}

/// Core raking algorithm (Iterative Proportional Fitting)
///
/// Non-convergence is NOT an error — the best weights reached after
/// max_iter are returned.  Bounds violations ARE errors.
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

    if margin_indices.is_empty() {
        return Err(WeightingError::InvalidInput("No margins provided".to_string()));
    }
    if margin_indices.len() != margin_targets.len() {
        return Err(WeightingError::InvalidInput(format!(
            "Margin count mismatch: {} indices vs {} targets",
            margin_indices.len(), margin_targets.len()
        )));
    }
    for indices in margin_indices.iter() {
        if indices.len() != n_obs {
            return Err(WeightingError::DimensionMismatch { expected: n_obs, got: indices.len() });
        }
    }

    let margins: Vec<MarginSpec> = margin_indices
        .iter()
        .zip(margin_targets.iter())
        .map(|(idx, tgt)| MarginSpec::new(idx.clone(), tgt.clone()))
        .collect();

    let mut raked_weights = wgt.to_owned();

    for _ in 0..max_iter {
        let weights_start = raked_weights.clone();

        for margin in &margins {
            let current_sums =
                sum_by_group_2d(raked_weights.view(), &margin.indices, margin.n_groups);
            let mut factors = Array2::zeros((margin.n_groups, n_reps));

            for g in 0..margin.n_groups {
                for r in 0..n_reps {
                    let current_sum = current_sums[[g, r]];
                    let target = margin.targets[g];
                    if current_sum > 1e-10 {
                        factors[[g, r]] = target / current_sum;
                    } else if target > 1e-10 {
                        return Err(WeightingError::ZeroWeights(g));
                    } else {
                        factors[[g, r]] = 1.0;
                    }
                }
            }

            for (row_idx, &group_id) in margin.indices.iter().enumerate() {
                for col_idx in 0..n_reps {
                    raked_weights[[row_idx, col_idx]] *= factors[[group_id, col_idx]];
                }
            }
        }

        let (converged, _) = check_convergence(raked_weights.view(), weights_start.view(), tol);
        if converged {
            // Bounds check only on actual convergence — explicit constraint violation
            if !check_bounds(raked_weights.view(), wgt, ll_bound, up_bound) {
                return Err(WeightingError::InvalidInput(
                    "Raking exceeded weight bounds".to_string(),
                ));
            }
            break;
        }
    }

    // Non-convergence: return best weights reached, no error
    Ok(raked_weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_simple_raking() {
        let wgt = array![[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]];
        let indices = array![0, 0, 1, 1];
        let targets = array![6.0, 2.0];
        let result = rake_impl(wgt.view(), &[indices], &[targets], None, None, 1e-6, 100).unwrap();
        assert_relative_eq!(result[[0, 0]] + result[[1, 0]], 6.0, epsilon = 1e-6);
        assert_relative_eq!(result[[2, 0]] + result[[3, 0]], 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_two_margin_raking() {
        let wgt = array![[1.0], [1.0], [1.0], [1.0]];
        let result = rake_impl(
            wgt.view(),
            &[array![0, 0, 1, 1], array![0, 1, 0, 1]],
            &[array![10.0, 20.0], array![12.0, 18.0]],
            None, None, 1e-6, 100,
        ).unwrap();
        assert_relative_eq!(result.sum(), 30.0, epsilon = 1e-6);
    }

    #[test]
    fn test_non_convergence_returns_partial_result() {
        let wgt = array![[1.0], [1.0], [1.0], [1.0]];
        let result = rake_impl(
            wgt.view(), &[array![0, 0, 1, 1]], &[array![6.0, 2.0]],
            None, None, 1e-20, 1,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_bounds_exceeded_is_error() {
        let wgt = array![[1.0], [1.0], [1.0], [1.0]];
        let result = rake_impl(
            wgt.view(), &[array![0, 0, 1, 1]], &[array![6.0, 2.0]],
            None, Some(1.1), 1e-6, 100,
        );
        assert!(result.is_err());
    }
}
