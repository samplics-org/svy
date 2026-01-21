// src/weighting/normalization.rs

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use super::utils::{sum_by_group_2d, Result, WeightingError};

/// Normalize weights to sum to a target
///
/// # Arguments
/// * `wgt` - Weight matrix (n_obs, n_reps)
/// * `by_arr` - Optional grouping variable (n_obs,). If None, normalize globally.
/// * `control` - Optional target sums. If None, target = count of observations.
///               Can be:
///               - None: target = n (or counts by group)
///               - Single value: same target for all groups
///               - Array: one target per group
///
/// # Returns
/// Normalized weight matrix
pub fn normalize_impl(
    wgt: ArrayView2<f64>,
    by_arr: Option<&Array1<i64>>,
    control: Option<&Array1<f64>>,
) -> Result<Array2<f64>> {
    let (n_obs, n_reps) = wgt.dim();
    
    match by_arr {
        None => normalize_global(wgt, control, n_obs),
        Some(groups) => normalize_by_group(wgt, groups, control),
    }
}

/// Global normalization (no grouping)
fn normalize_global(
    wgt: ArrayView2<f64>,
    control: Option<&Array1<f64>>,
    n_obs: usize,
) -> Result<Array2<f64>> {
    let (_, n_reps) = wgt.dim();
    
    // Determine target for each replicate
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
    
    // Calculate current sums
    let current_sums = wgt.sum_axis(Axis(0));
    
    // Calculate factors
    let mut factors = Array1::zeros(n_reps);
    for r in 0..n_reps {
        if current_sums[r] > 1e-10 {
            factors[r] = targets[r] / current_sums[r];
        } else if targets[r] > 1e-10 {
            return Err(WeightingError::InvalidInput(
                format!("Cannot normalize: current sum is 0 but target is {}", targets[r])
            ));
        } else {
            factors[r] = 1.0;
        }
    }
    
    // Apply factors (broadcast multiply)
    let mut normalized = wgt.to_owned();
    for r in 0..n_reps {
        normalized.column_mut(r).mapv_inplace(|x| x * factors[r]);
    }
    
    Ok(normalized)
}

/// Normalization by groups
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
    
    // Map groups to indices
    let (group_indices, n_groups) = create_group_mapping(by_arr);
    
    // Sum weights by group
    let current_sums = sum_by_group_2d(wgt, &group_indices, n_groups);
    
    // Determine targets for each group
    let targets = match control {
        None => {
            // Target = count of observations in each group
            let mut counts = vec![0; n_groups];
            for &g in &group_indices {
                counts[g] += 1;
            }
            
            let mut target_matrix = Array2::zeros((n_groups, n_reps));
            for g in 0..n_groups {
                for r in 0..n_reps {
                    target_matrix[[g, r]] = counts[g] as f64;
                }
            }
            target_matrix
        }
        Some(ctrl) => {
            if ctrl.len() != n_groups {
                return Err(WeightingError::DimensionMismatch {
                    expected: n_groups,
                    got: ctrl.len(),
                });
            }
            
            // Broadcast targets to all replicates
            let mut target_matrix = Array2::zeros((n_groups, n_reps));
            for g in 0..n_groups {
                for r in 0..n_reps {
                    target_matrix[[g, r]] = ctrl[g];
                }
            }
            target_matrix
        }
    };
    
    // Calculate factors for each group and replicate
    let mut factors = Array2::zeros((n_groups, n_reps));
    for g in 0..n_groups {
        for r in 0..n_reps {
            if current_sums[[g, r]] > 1e-10 {
                factors[[g, r]] = targets[[g, r]] / current_sums[[g, r]];
            } else if targets[[g, r]] > 1e-10 {
                return Err(WeightingError::InvalidInput(
                    format!("Cannot normalize group {}: current sum is 0 but target is {}", 
                           g, targets[[g, r]])
                ));
            } else {
                factors[[g, r]] = 1.0;
            }
        }
    }
    
    // Apply factors
    let mut normalized = Array2::zeros((n_obs, n_reps));
    for i in 0..n_obs {
        let g = group_indices[i];
        for r in 0..n_reps {
            normalized[[i, r]] = wgt[[i, r]] * factors[[g, r]];
        }
    }
    
    Ok(normalized)
}

/// Create integer mapping for groups
fn create_group_mapping(groups: &Array1<i64>) -> (Vec<usize>, usize) {
    use std::collections::HashMap;
    
    let mut group_map: HashMap<i64, usize> = HashMap::new();
    let mut indices = Vec::with_capacity(groups.len());
    let mut next_id = 0;
    
    for &group_val in groups.iter() {
        let id = *group_map.entry(group_val).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        indices.push(id);
    }
    
    (indices, next_id)
}

/// Parallel normalization - process each replicate independently
pub fn normalize_parallel(
    wgt: ArrayView2<f64>,
    by_arr: Option<&Array1<i64>>,
    control: Option<&Array1<f64>>,
) -> Result<Array2<f64>> {
    let (n_obs, n_reps) = wgt.dim();
    
    // Process each replicate in parallel
    let results: std::result::Result<Vec<_>, WeightingError> = (0..n_reps)
        .into_par_iter()
        .map(|r| {
            let wgt_col = wgt.column(r);
            let ctrl_val = control.map(|c| c[r.min(c.len() - 1)]);
            normalize_single(wgt_col, by_arr, ctrl_val)
        })
        .collect();
    
    let norm_cols = results?;
    
    // Stack columns
    let mut result = Array2::zeros((n_obs, n_reps));
    for (col_idx, col) in norm_cols.iter().enumerate() {
        result.column_mut(col_idx).assign(col);
    }
    
    Ok(result)
}

/// Normalize a single replicate
fn normalize_single(
    wgt: ArrayView1<f64>,
    by_arr: Option<&Array1<i64>>,
    control: Option<f64>,
) -> Result<ndarray::Array1<f64>> {
    let n_obs = wgt.len();
    
    match by_arr {
        None => {
            // Global normalization
            let current_sum: f64 = wgt.sum();
            let target = control.unwrap_or(n_obs as f64);
            
            let factor = if current_sum > 1e-10 {
                target / current_sum
            } else if target > 1e-10 {
                return Err(WeightingError::InvalidInput(
                    "Cannot normalize: weights sum to zero".to_string()
                ));
            } else {
                1.0
            };
            
            Ok(wgt.mapv(|x| x * factor))
        }
        Some(groups) => {
            // Group normalization
            let (group_indices, n_groups) = create_group_mapping(groups);
            
            // Sum by group
            let mut group_sums = vec![0.0; n_groups];
            for (i, &g) in group_indices.iter().enumerate() {
                group_sums[g] += wgt[i];
            }
            
            // Calculate targets
            let group_targets: Vec<f64> = if let Some(ctrl) = control {
                vec![ctrl; n_groups]
            } else {
                let mut counts = vec![0; n_groups];
                for &g in &group_indices {
                    counts[g] += 1;
                }
                counts.into_iter().map(|c| c as f64).collect()
            };
            
            // Calculate factors
            let mut factors = vec![1.0; n_groups];
            for g in 0..n_groups {
                if group_sums[g] > 1e-10 {
                    factors[g] = group_targets[g] / group_sums[g];
                } else if group_targets[g] > 1e-10 {
                    return Err(WeightingError::InvalidInput(
                        format!("Group {} has zero weights but non-zero target", g)
                    ));
                }
            }
            
            // Apply factors
            let mut normalized = ndarray::Array1::zeros(n_obs);
            for (i, &g) in group_indices.iter().enumerate() {
                normalized[i] = wgt[i] * factors[g];
            }
            
            Ok(normalized)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_global_normalization() {
        let wgt = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        
        // Normalize to count (3.0)
        let result = normalize_impl(wgt.view(), None, None).unwrap();
        
        let sum_col0: f64 = result.column(0).sum();
        let sum_col1: f64 = result.column(1).sum();
        
        assert_relative_eq!(sum_col0, 3.0, epsilon = 1e-6);
        assert_relative_eq!(sum_col1, 3.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_group_normalization() {
        let wgt = array![[1.0], [2.0], [3.0], [4.0]];
        let groups = array![0, 0, 1, 1];
        
        // Normalize each group to its count (2.0)
        let result = normalize_impl(wgt.view(), Some(&groups), None).unwrap();
        
        // Group 0: sum should be 2.0
        let sum_g0 = result[[0, 0]] + result[[1, 0]];
        assert_relative_eq!(sum_g0, 2.0, epsilon = 1e-6);
        
        // Group 1: sum should be 2.0
        let sum_g1 = result[[2, 0]] + result[[3, 0]];
        assert_relative_eq!(sum_g1, 2.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_custom_targets() {
        let wgt = array![[1.0], [1.0], [1.0]];
        let groups = array![0, 0, 1];
        let targets = array![10.0, 5.0];
        
        let result = normalize_impl(wgt.view(), Some(&groups), Some(&targets)).unwrap();
        
        // Group 0: sum should be 10.0
        let sum_g0 = result[[0, 0]] + result[[1, 0]];
        assert_relative_eq!(sum_g0, 10.0, epsilon = 1e-6);
        
        // Group 1: sum should be 5.0
        assert_relative_eq!(result[[2, 0]], 5.0, epsilon = 1e-6);
    }
}
