// src/weighting/poststratification.rs

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use super::normalization::normalize_impl;
use super::utils::{Result, WeightingError};

/// Post-stratification adjustment
///
/// Post-stratification is a special case of normalization where:
/// 1. Each stratum must have a control total
/// 2. The controls must match the data strata exactly (bijective mapping)
///
/// # Arguments
/// * `wgt` - Weight matrix (n_obs, n_reps)
/// * `by_arr` - Post-stratification variable (n_obs,)
/// * `control` - Target totals for each stratum (one per unique value in by_arr)
///
/// # Returns
/// Post-stratified weight matrix
///
/// # Notes
/// This is essentially normalization with strict validation that all
/// strata in the data have corresponding controls and vice versa.
pub fn poststratify_impl(
    wgt: ArrayView2<f64>,
    by_arr: ArrayView1<i64>,
    control: &Array1<f64>,
) -> Result<Array2<f64>> {
    let n_obs = wgt.nrows();
    
    if by_arr.len() != n_obs {
        return Err(WeightingError::DimensionMismatch {
            expected: n_obs,
            got: by_arr.len(),
        });
    }
    
    // Validate that control has exactly one entry per unique stratum
    let unique_strata = get_unique_sorted(&by_arr);
    let n_strata = unique_strata.len();
    
    if control.len() != n_strata {
        return Err(WeightingError::InvalidInput(
            format!(
                "Post-stratification requires exact match: {} strata in data but {} controls provided",
                n_strata,
                control.len()
            )
        ));
    }
    
    // Validate controls are non-negative
    for (i, &ctrl) in control.iter().enumerate() {
        if ctrl < 0.0 {
            return Err(WeightingError::InvalidInput(
                format!("Control total for stratum {} is negative: {}", i, ctrl)
            ));
        }
    }
    
    // Post-stratification is just normalization with these validated controls
    normalize_impl(wgt, Some(&by_arr.to_owned()), Some(control))
}

/// Post-stratification with factor
///
/// Instead of providing target totals, provide factors (proportions).
/// The factors are converted to totals based on current weight sums.
///
/// # Arguments
/// * `wgt` - Weight matrix (n_obs, n_reps)
/// * `by_arr` - Post-stratification variable
/// * `factor` - Target proportions for each stratum (must sum to 1.0)
///
/// # Returns
/// Post-stratified weight matrix
pub fn poststratify_factor(
    wgt: ArrayView2<f64>,
    by_arr: ArrayView1<i64>,
    factor: &Array1<f64>,
) -> Result<Array2<f64>> {
    let n_obs = wgt.nrows();
    let n_reps = wgt.ncols();
    
    // Validate factors
    let factor_sum: f64 = factor.sum();
    if (factor_sum - 1.0).abs() > 1e-6 {
        return Err(WeightingError::InvalidInput(
            format!("Factors must sum to 1.0, got {}", factor_sum)
        ));
    }
    
    for (i, &f) in factor.iter().enumerate() {
        if f < 0.0 {
            return Err(WeightingError::InvalidInput(
                format!("Factor for stratum {} is negative: {}", i, f)
            ));
        }
    }
    
    // Calculate total weight for each replicate
    let total_weights = wgt.sum_axis(ndarray::Axis(0));
    
    // Convert factors to targets for each replicate
    // control[stratum, replicate] = factor[stratum] * total_weight[replicate]
    let n_strata = factor.len();
    let mut controls = Array2::zeros((n_strata, n_reps));
    
    for s in 0..n_strata {
        for r in 0..n_reps {
            controls[[s, r]] = factor[s] * total_weights[r];
        }
    }
    
    // Apply post-stratification to each replicate
    // This is a bit tricky since normalize_impl expects 1D control per group
    // We need to handle each replicate separately with its specific targets
    
    let mut result = Array2::zeros((n_obs, n_reps));
    
    for r in 0..n_reps {
        let wgt_col = wgt.column(r).to_owned();
        let ctrl_col = controls.column(r).to_owned();
        
        // Reshape to 2D for normalize_impl
        let wgt_2d = wgt_col.insert_axis(ndarray::Axis(1));
        
        let normalized = normalize_impl(
            wgt_2d.view(),
            Some(&by_arr.to_owned()),
            Some(&ctrl_col),
        )?;
        
        result.column_mut(r).assign(&normalized.column(0));
    }
    
    Ok(result)
}

/// Get unique values from array, sorted
fn get_unique_sorted(arr: &ArrayView1<i64>) -> Vec<i64> {
    use std::collections::BTreeSet;
    
    let unique: BTreeSet<i64> = arr.iter().copied().collect();
    unique.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_poststratify() {
        let wgt = array![[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]];
        let strata = array![0, 0, 1, 1];
        let controls = array![10.0, 20.0];
        
        let result = poststratify_impl(wgt.view(), strata.view(), &controls).unwrap();
        
        // Stratum 0 should sum to 10.0
        let sum_s0_r0 = result[[0, 0]] + result[[1, 0]];
        assert_relative_eq!(sum_s0_r0, 10.0, epsilon = 1e-6);
        
        // Stratum 1 should sum to 20.0
        let sum_s1_r0 = result[[2, 0]] + result[[3, 0]];
        assert_relative_eq!(sum_s1_r0, 20.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_poststratify_with_factor() {
        let wgt = array![[10.0], [10.0], [10.0], [10.0]];
        let strata = array![0, 0, 1, 1];
        let factors = array![0.25, 0.75];  // Stratum 0 gets 25%, stratum 1 gets 75%
        
        let result = poststratify_factor(wgt.view(), strata.view(), &factors).unwrap();
        
        // Total should be preserved (40.0)
        let total: f64 = result.sum();
        assert_relative_eq!(total, 40.0, epsilon = 1e-6);
        
        // Stratum 0 should have 25% of total (10.0)
        let sum_s0 = result[[0, 0]] + result[[1, 0]];
        assert_relative_eq!(sum_s0, 10.0, epsilon = 1e-6);
        
        // Stratum 1 should have 75% of total (30.0)
        let sum_s1 = result[[2, 0]] + result[[3, 0]];
        assert_relative_eq!(sum_s1, 30.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_poststratify_validation() {
        let wgt = array![[1.0], [1.0], [1.0]];
        let strata = array![0, 0, 1];
        let controls = array![10.0];  // Wrong: need 2 controls for 2 strata
        
        let result = poststratify_impl(wgt.view(), strata.view(), &controls);
        assert!(result.is_err());
    }
}
