// src/weighting/nonresponse.rs

use ndarray::{Array2, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use super::utils::{sum_by_group_2d, Result, WeightingError};

/// Response status codes (numeric encoding)
const RR: i64 = 0;  // Respondent
const NR: i64 = 1;  // Non-respondent
const IN: i64 = 2;  // Ineligible
const UK: i64 = 3;  // Unknown

/// Non-response adjustment
///
/// Adjusts weights for non-response by redistributing NR/UK weights
/// to respondents within adjustment classes.
///
/// # Arguments
/// * `wgts` - Weight matrix (n_obs, n_reps)
/// * `adj_class` - Adjustment class for each observation (n_obs,)
/// * `resp_status` - Response status: 0=RR, 1=NR, 2=IN, 3=UK (n_obs,)
/// * `unknown_to_inelig` - If true, distribute UK to eligible (IN+RR+NR);
///                         If false, UK joins NR pool (distributed to RR only)
///
/// # Returns
/// Adjusted weight matrix. Non-respondents and unknowns get weight=0.
pub fn adjust_nr_impl(
    wgts: ArrayView2<f64>,
    adj_class: ArrayView1<i64>,
    resp_status: ArrayView1<i64>,
    unknown_to_inelig: bool,
) -> Result<Array2<f64>> {
    let (n_obs, n_reps) = wgts.dim();
    
    // Validate inputs
    if adj_class.len() != n_obs {
        return Err(WeightingError::DimensionMismatch {
            expected: n_obs,
            got: adj_class.len(),
        });
    }
    
    if resp_status.len() != n_obs {
        return Err(WeightingError::DimensionMismatch {
            expected: n_obs,
            got: resp_status.len(),
        });
    }
    
    // Create masks for each status
    let is_rr: Vec<bool> = resp_status.iter().map(|&s| s == RR).collect();
    let is_nr: Vec<bool> = resp_status.iter().map(|&s| s == NR).collect();
    let is_in: Vec<bool> = resp_status.iter().map(|&s| s == IN).collect();
    let is_uk: Vec<bool> = resp_status.iter().map(|&s| s == UK).collect();
    
    // Map adjustment classes to integer indices
    let (class_indices, n_classes) = create_class_mapping(&adj_class);
    
    // Sum weights by class and status
    let sum_rr = sum_by_class_and_status(wgts, &class_indices, &is_rr, n_classes);
    let sum_nr = sum_by_class_and_status(wgts, &class_indices, &is_nr, n_classes);
    let sum_in = sum_by_class_and_status(wgts, &class_indices, &is_in, n_classes);
    let sum_uk = sum_by_class_and_status(wgts, &class_indices, &is_uk, n_classes);
    
    // Calculate adjustment factors for each class
    let mut factor_rr = Array2::ones((n_classes, n_reps));
    let mut factor_in = Array2::ones((n_classes, n_reps));
    
    for c in 0..n_classes {
        for r in 0..n_reps {
            if unknown_to_inelig {
                // Unknowns distributed over eligible (IN + RR + NR)
                let denom_uk = sum_in[[c, r]] + sum_rr[[c, r]] + sum_nr[[c, r]];
                let adj_uk = if denom_uk > 1e-10 {
                    (denom_uk + sum_uk[[c, r]]) / denom_uk
                } else {
                    1.0
                };
                
                // NR distributed over RR
                let denom_rr = sum_rr[[c, r]];
                let adj_rr = if denom_rr > 1e-10 {
                    (sum_rr[[c, r]] + sum_nr[[c, r]]) / denom_rr
                } else {
                    1.0
                };
                
                factor_rr[[c, r]] = adj_rr * adj_uk;
                factor_in[[c, r]] = adj_uk;
            } else {
                // Unknowns join NR pool -> distributed to RR
                let denom_rr = sum_rr[[c, r]];
                let adj_rr = if denom_rr > 1e-10 {
                    (sum_rr[[c, r]] + sum_nr[[c, r]] + sum_uk[[c, r]]) / denom_rr
                } else {
                    1.0
                };
                
                factor_rr[[c, r]] = adj_rr;
                factor_in[[c, r]] = 1.0;
            }
        }
    }
    
    // Apply factors to weights
    let mut adj_weights = Array2::zeros((n_obs, n_reps));
    
    for i in 0..n_obs {
        let class_id = class_indices[i];
        
        if is_rr[i] {
            for r in 0..n_reps {
                adj_weights[[i, r]] = wgts[[i, r]] * factor_rr[[class_id, r]];
            }
        } else if is_in[i] {
            for r in 0..n_reps {
                adj_weights[[i, r]] = wgts[[i, r]] * factor_in[[class_id, r]];
            }
        }
        // NR and UK get weight = 0 (already initialized to zero)
    }
    
    Ok(adj_weights)
}

/// Create integer mapping for adjustment classes
fn create_class_mapping(adj_class: &ArrayView1<i64>) -> (Vec<usize>, usize) {
    use std::collections::HashMap;
    
    let mut class_map: HashMap<i64, usize> = HashMap::new();
    let mut indices = Vec::with_capacity(adj_class.len());
    let mut next_id = 0;
    
    for &class_val in adj_class.iter() {
        let id = *class_map.entry(class_val).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        indices.push(id);
    }
    
    (indices, next_id)
}

/// Sum weights by class for observations with a given status
fn sum_by_class_and_status(
    wgts: ArrayView2<f64>,
    class_indices: &[usize],
    status_mask: &[bool],
    n_classes: usize,
) -> Array2<f64> {
    let (n_obs, n_reps) = wgts.dim();
    let mut sums = Array2::zeros((n_classes, n_reps));
    
    for i in 0..n_obs {
        if status_mask[i] {
            let class_id = class_indices[i];
            for r in 0..n_reps {
                sums[[class_id, r]] += wgts[[i, r]];
            }
        }
    }
    
    sums
}

/// Parallel version - process each replicate independently
pub fn adjust_nr_parallel(
    wgts: ArrayView2<f64>,
    adj_class: ArrayView1<i64>,
    resp_status: ArrayView1<i64>,
    unknown_to_inelig: bool,
) -> Result<Array2<f64>> {
    let (n_obs, n_reps) = wgts.dim();
    
    // Process each replicate in parallel
    let results: std::result::Result<Vec<_>, WeightingError> = (0..n_reps)
        .into_par_iter()
        .map(|r| {
            let wgt_col = wgts.column(r);
            adjust_nr_single(wgt_col, adj_class, resp_status, unknown_to_inelig)
        })
        .collect();
    
    let adj_cols = results?;
    
    // Stack columns into matrix
    let mut result = Array2::zeros((n_obs, n_reps));
    for (col_idx, col) in adj_cols.iter().enumerate() {
        result.column_mut(col_idx).assign(col);
    }
    
    Ok(result)
}

/// Adjust a single replicate
fn adjust_nr_single(
    wgt: ArrayView1<f64>,
    adj_class: ArrayView1<i64>,
    resp_status: ArrayView1<i64>,
    unknown_to_inelig: bool,
) -> Result<ndarray::Array1<f64>> {
    let n_obs = wgt.len();
    
    // Create masks
    let is_rr: Vec<bool> = resp_status.iter().map(|&s| s == RR).collect();
    let is_nr: Vec<bool> = resp_status.iter().map(|&s| s == NR).collect();
    let is_in: Vec<bool> = resp_status.iter().map(|&s| s == IN).collect();
    let is_uk: Vec<bool> = resp_status.iter().map(|&s| s == UK).collect();
    
    // Map classes
    let (class_indices, n_classes) = create_class_mapping(&adj_class);
    
    // Sum by class and status
    let mut sum_rr = vec![0.0; n_classes];
    let mut sum_nr = vec![0.0; n_classes];
    let mut sum_in = vec![0.0; n_classes];
    let mut sum_uk = vec![0.0; n_classes];
    
    for i in 0..n_obs {
        let c = class_indices[i];
        let w = wgt[i];
        
        if is_rr[i] { sum_rr[c] += w; }
        if is_nr[i] { sum_nr[c] += w; }
        if is_in[i] { sum_in[c] += w; }
        if is_uk[i] { sum_uk[c] += w; }
    }
    
    // Calculate factors
    let mut factor_rr = vec![1.0; n_classes];
    let mut factor_in = vec![1.0; n_classes];
    
    for c in 0..n_classes {
        if unknown_to_inelig {
            let denom_uk = sum_in[c] + sum_rr[c] + sum_nr[c];
            let adj_uk = if denom_uk > 1e-10 {
                (denom_uk + sum_uk[c]) / denom_uk
            } else {
                1.0
            };
            
            let denom_rr = sum_rr[c];
            let adj_rr = if denom_rr > 1e-10 {
                (sum_rr[c] + sum_nr[c]) / denom_rr
            } else {
                1.0
            };
            
            factor_rr[c] = adj_rr * adj_uk;
            factor_in[c] = adj_uk;
        } else {
            let denom_rr = sum_rr[c];
            let adj_rr = if denom_rr > 1e-10 {
                (sum_rr[c] + sum_nr[c] + sum_uk[c]) / denom_rr
            } else {
                1.0
            };
            
            factor_rr[c] = adj_rr;
            factor_in[c] = 1.0;
        }
    }
    
    // Apply factors
    let mut adj_wgt = ndarray::Array1::zeros(n_obs);
    
    for i in 0..n_obs {
        let c = class_indices[i];
        
        if is_rr[i] {
            adj_wgt[i] = wgt[i] * factor_rr[c];
        } else if is_in[i] {
            adj_wgt[i] = wgt[i] * factor_in[c];
        }
    }
    
    Ok(adj_wgt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_simple_nr_adjustment() {
        // 4 observations: 2 RR, 1 NR, 1 IN
        // All in same class, equal weights
        let wgts = array![[1.0], [1.0], [1.0], [1.0]];
        let adj_class = array![0, 0, 0, 0];
        let resp_status = array![RR, RR, NR, IN];
        
        let result = adjust_nr_impl(
            wgts.view(),
            adj_class.view(),
            resp_status.view(),
            false,  // UK joins NR
        ).unwrap();
        
        // RR should get weight * (RR+NR)/RR = 1.0 * 3/2 = 1.5
        assert_relative_eq!(result[[0, 0]], 1.5, epsilon = 1e-6);
        assert_relative_eq!(result[[1, 0]], 1.5, epsilon = 1e-6);
        
        // NR gets 0
        assert_relative_eq!(result[[2, 0]], 0.0, epsilon = 1e-6);
        
        // IN keeps weight
        assert_relative_eq!(result[[3, 0]], 1.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_multiple_classes() {
        // 6 observations: 2 classes
        let wgts = array![[1.0], [1.0], [1.0], [2.0], [2.0], [2.0]];
        let adj_class = array![0, 0, 0, 1, 1, 1];
        let resp_status = array![RR, RR, NR, RR, NR, NR];
        
        let result = adjust_nr_impl(
            wgts.view(),
            adj_class.view(),
            resp_status.view(),
            false,
        ).unwrap();
        
        // Class 0: RR gets (2+1)/2 = 1.5x
        assert_relative_eq!(result[[0, 0]], 1.5, epsilon = 1e-6);
        assert_relative_eq!(result[[1, 0]], 1.5, epsilon = 1e-6);
        
        // Class 1: RR gets (2+4)/2 = 3.0x
        assert_relative_eq!(result[[3, 0]], 6.0, epsilon = 1e-6);
    }
}
