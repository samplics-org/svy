//! Weight normalization to target totals, globally or by group.

use super::utils::{Result, WeightingError, sum_by_group_2d};
use ndarray::{Array1, Array2, ArrayView2, Axis};

/// Scales weights to `control` totals (default: row counts), globally or per sorted `by_arr` group.
pub(crate) fn normalize_impl(
    wgt: ArrayView2<f64>,
    by_arr: Option<&Array1<i64>>,
    control: Option<&Array1<f64>>,
) -> Result<Array2<f64>> {
    let (n_obs, _) = wgt.dim();

    match by_arr {
        None => normalize_global(wgt, control, n_obs),
        Some(groups) => normalize_by_group(wgt, groups, control),
    }
}

fn normalize_global(
    wgt: ArrayView2<f64>,
    control: Option<&Array1<f64>>,
    n_obs: usize,
) -> Result<Array2<f64>> {
    let (_, n_reps) = wgt.dim();

    let targets: Vec<f64> = match control {
        None => vec![n_obs as f64; n_reps],
        Some(ctrl) => {
            if ctrl.len() == 1 {
                vec![ctrl[0]; n_reps]
            } else if ctrl.len() == n_reps {
                ctrl.to_vec()
            } else {
                return Err(WeightingError::DimensionMismatch {
                    expected: n_reps,
                    got: ctrl.len(),
                });
            }
        }
    };

    let current_sums = wgt.sum_axis(Axis(0));

    let mut factors = Array1::zeros(n_reps);
    for r in 0..n_reps {
        if current_sums[r] > 0.0 {
            factors[r] = targets[r] / current_sums[r];
        } else if targets[r] > 0.0 {
            return Err(WeightingError::InvalidInput(format!(
                "Cannot normalize: current sum is 0 but target is {}",
                targets[r]
            )));
        } else {
            factors[r] = 1.0;
        }
    }

    let mut normalized = wgt.to_owned();
    for r in 0..n_reps {
        normalized.column_mut(r).mapv_inplace(|x| x * factors[r]);
    }

    Ok(normalized)
}

fn normalize_by_group(
    wgt: ArrayView2<f64>,
    by_arr: &Array1<i64>,
    control: Option<&Array1<f64>>,
) -> Result<Array2<f64>> {
    let (n_obs, n_reps) = wgt.dim();

    if by_arr.len() != n_obs {
        return Err(WeightingError::DimensionMismatch {
            expected: n_obs,
            got: by_arr.len(),
        });
    }

    let (group_indices, n_groups) = create_group_mapping(by_arr);

    let current_sums = sum_by_group_2d(wgt, &group_indices, n_groups);

    let group_targets: Vec<f64> = match control {
        None => {
            let mut counts = vec![0usize; n_groups];
            for &g in &group_indices {
                counts[g] += 1;
            }
            counts.iter().map(|&c| c as f64).collect()
        }
        Some(ctrl) => {
            if ctrl.len() != n_groups {
                return Err(WeightingError::DimensionMismatch {
                    expected: n_groups,
                    got: ctrl.len(),
                });
            }
            ctrl.to_vec()
        }
    };

    let mut factors = Array2::zeros((n_groups, n_reps));
    for g in 0..n_groups {
        let tgt = group_targets[g];
        for r in 0..n_reps {
            if current_sums[[g, r]] > 0.0 {
                factors[[g, r]] = tgt / current_sums[[g, r]];
            } else if tgt > 0.0 {
                return Err(WeightingError::InvalidInput(format!(
                    "Cannot normalize group {}: current sum is 0 but target is {}",
                    g, tgt
                )));
            } else {
                factors[[g, r]] = 1.0;
            }
        }
    }

    let mut normalized = Array2::zeros((n_obs, n_reps));
    for i in 0..n_obs {
        let g = group_indices[i];
        for r in 0..n_reps {
            normalized[[i, r]] = wgt[[i, r]] * factors[[g, r]];
        }
    }

    Ok(normalized)
}

/// Group ids in sorted value order, the order controls are given in (as `poststratify` validates).
fn create_group_mapping(groups: &Array1<i64>) -> (Vec<usize>, usize) {
    use std::collections::HashMap;

    let mut sorted_vals: Vec<i64> = groups.iter().copied().collect();
    sorted_vals.sort_unstable();
    sorted_vals.dedup();

    let group_map: HashMap<i64, usize> = sorted_vals
        .iter()
        .enumerate()
        .map(|(id, &v)| (v, id))
        .collect();

    let indices = groups.iter().map(|v| group_map[v]).collect();
    (indices, sorted_vals.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_global_normalization() {
        let wgt = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

        // Normalize to count (3.0)
        let result = normalize_impl(wgt.view(), None, None).unwrap();

        let sum_col0: f64 = result.column(0).sum();
        let sum_col1: f64 = result.column(1).sum();

        assert_relative_eq!(sum_col0, 3.0, epsilon = 1e-12);
        assert_relative_eq!(sum_col1, 3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_group_normalization() {
        let wgt = array![[1.0], [2.0], [3.0], [4.0]];
        let groups = array![0, 0, 1, 1];

        // Normalize each group to its count (2.0)
        let result = normalize_impl(wgt.view(), Some(&groups), None).unwrap();

        let sum_g0 = result[[0, 0]] + result[[1, 0]];
        assert_relative_eq!(sum_g0, 2.0, epsilon = 1e-12);

        let sum_g1 = result[[2, 0]] + result[[3, 0]];
        assert_relative_eq!(sum_g1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_custom_targets() {
        let wgt = array![[1.0], [1.0], [1.0]];
        let groups = array![0, 0, 1];
        let targets = array![10.0, 5.0];

        let result = normalize_impl(wgt.view(), Some(&groups), Some(&targets)).unwrap();

        let sum_g0 = result[[0, 0]] + result[[1, 0]];
        assert_relative_eq!(sum_g0, 10.0, epsilon = 1e-12);

        assert_relative_eq!(result[[2, 0]], 5.0, epsilon = 1e-12);
    }
}
