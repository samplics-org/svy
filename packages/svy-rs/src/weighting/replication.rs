// src/weighting/replication.rs
//
// Replicate weight creation for variance estimation methods:
// - BRR (Balanced Repeated Replication)
// - Jackknife (JKn - delete-one-group)
// - Bootstrap (stratified)
// - SDR (Successive Difference Replication)

use ndarray::{Array1, Array2, ArrayView1};
use rayon::prelude::*;
use std::collections::HashMap;

/// Result type for replication functions
pub type Result<T> = std::result::Result<T, ReplicationError>;

/// Errors that can occur during replicate weight creation
#[derive(Debug, Clone)]
pub enum ReplicationError {
    /// Dimension mismatch between arrays
    DimensionMismatch { expected: usize, got: usize },
    /// Invalid input parameters
    InvalidInput(String),
    /// BRR requires exactly 2 PSUs per stratum
    BrrPsuCount { stratum: i64, count: usize },
    /// Not enough PSUs for the method
    InsufficientPsus { required: usize, got: usize },
}

impl std::fmt::Display for ReplicationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReplicationError::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got)
            }
            ReplicationError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            ReplicationError::BrrPsuCount { stratum, count } => {
                write!(f, "BRR requires 2 PSUs per stratum, stratum {} has {}", stratum, count)
            }
            ReplicationError::InsufficientPsus { required, got } => {
                write!(f, "Insufficient PSUs: required {}, got {}", required, got)
            }
        }
    }
}

impl std::error::Error for ReplicationError {}

// ============================================================================
// BRR (Balanced Repeated Replication)
// ============================================================================

/// Create BRR replicate weights
///
/// Balanced Repeated Replication requires exactly 2 PSUs per stratum.
/// Uses Hadamard matrices for balanced half-samples.
///
/// # Arguments
/// * `wgt` - Base weights (n_obs,)
/// * `stratum` - Stratum identifiers (n_obs,)
/// * `psu` - PSU identifiers (n_obs,)
/// * `n_reps` - Number of replicates (if None, uses minimum = n_strata)
/// * `fay_coef` - Fay adjustment factor (0.0 = standard BRR, 0.5 = Fay's method)
///
/// # Returns
/// * Replicate weight matrix (n_obs, n_reps)
/// * Degrees of freedom for variance estimation
pub fn create_brr_weights(
    wgt: ArrayView1<f64>,
    stratum: ArrayView1<i64>,
    psu: ArrayView1<i64>,
    n_reps: Option<usize>,
    fay_coef: f64,
) -> Result<(Array2<f64>, f64)> {
    let n_obs = wgt.len();

    if stratum.len() != n_obs || psu.len() != n_obs {
        return Err(ReplicationError::DimensionMismatch {
            expected: n_obs,
            got: stratum.len().min(psu.len()),
        });
    }

    if fay_coef < 0.0 || fay_coef >= 1.0 {
        return Err(ReplicationError::InvalidInput(
            "fay_coef must be in [0, 1)".to_string()
        ));
    }

    // Build stratum -> PSUs mapping and validate 2 PSUs per stratum
    let (stratum_psu_map, psu_to_idx) = build_stratum_psu_map(&stratum, &psu)?;
    let n_strata = stratum_psu_map.len();

    // Determine number of replicates
    let n_reps = match n_reps {
        Some(r) => {
            // Must be at least n_strata and a power of 2 for Hadamard
            let min_reps = next_power_of_2(n_strata);
            if r < min_reps {
                min_reps
            } else {
                next_power_of_2(r)
            }
        }
        None => next_power_of_2(n_strata),
    };

    // Generate Hadamard matrix
    let hadamard = generate_hadamard(n_reps);

    // BRR adjustment factors
    let k_plus = 2.0 - fay_coef;  // Weight multiplier when PSU is "in"
    let k_minus = fay_coef;       // Weight multiplier when PSU is "out"

    // Create replicate weights in parallel
    let rep_weights: Vec<Array1<f64>> = (0..n_reps)
        .into_par_iter()
        .map(|r| {
            let mut rep_wgt = Array1::zeros(n_obs);

            for i in 0..n_obs {
                let s = stratum[i];
                let p = psu[i];

                // Get stratum index (0 to n_strata-1)
                let stratum_idx = stratum_psu_map.get(&s).unwrap().0 as usize;

                // Get PSU index within stratum (0 or 1)
                let psu_idx = psu_to_idx[&(s, p)];

                // Hadamard value determines if PSU is "in" or "out"
                // Use stratum_idx mod n_reps to handle more strata than reps
                let h_val = hadamard[[stratum_idx % n_reps, r]];

                // If h_val == 1, first PSU is "in"; if h_val == -1, second PSU is "in"
                let multiplier = if (h_val > 0.0 && psu_idx == 0) || (h_val < 0.0 && psu_idx == 1) {
                    k_plus
                } else {
                    k_minus
                };

                rep_wgt[i] = wgt[i] * multiplier;
            }

            rep_wgt
        })
        .collect();

    // Assemble result matrix
    let mut result = Array2::zeros((n_obs, n_reps));
    for (r, col) in rep_weights.into_iter().enumerate() {
        result.column_mut(r).assign(&col);
    }

    // Degrees of freedom
    let df = n_strata as f64;

    Ok((result, df))
}

// ============================================================================
// Jackknife (JKn - Delete-one-group)
// ============================================================================

/// Create Jackknife replicate weights (JKn method)
///
/// Delete-one-group jackknife: each replicate removes one PSU and adjusts others.
///
/// # Arguments
/// * `wgt` - Base weights (n_obs,)
/// * `stratum` - Stratum identifiers (n_obs,), or None for unstratified
/// * `psu` - PSU identifiers (n_obs,)
///
/// # Returns
/// * Replicate weight matrix (n_obs, n_reps) where n_reps = total PSUs
/// * Degrees of freedom
pub fn create_jkn_weights(
    wgt: ArrayView1<f64>,
    stratum: Option<ArrayView1<i64>>,
    psu: ArrayView1<i64>,
) -> Result<(Array2<f64>, f64)> {
    let n_obs = wgt.len();

    if psu.len() != n_obs {
        return Err(ReplicationError::DimensionMismatch {
            expected: n_obs,
            got: psu.len(),
        });
    }

    // Handle stratification
    let stratum_vec: Array1<i64> = match stratum {
        Some(s) => {
            if s.len() != n_obs {
                return Err(ReplicationError::DimensionMismatch {
                    expected: n_obs,
                    got: s.len(),
                });
            }
            s.to_owned()
        }
        None => Array1::ones(n_obs), // Single stratum
    };

    // Build stratum -> PSUs mapping
    let stratum_psus = build_stratum_psu_list(&stratum_vec, &psu);

    // Count total PSUs (= number of replicates)
    let n_reps: usize = stratum_psus.values().map(|v| v.len()).sum();

    if n_reps < 2 {
        return Err(ReplicationError::InsufficientPsus { required: 2, got: n_reps });
    }

    // Map (stratum, psu) -> replicate index
    let mut psu_to_rep: HashMap<(i64, i64), usize> = HashMap::new();
    let mut rep_idx = 0;
    for (&s, psus) in &stratum_psus {
        for &p in psus {
            psu_to_rep.insert((s, p), rep_idx);
            rep_idx += 1;
        }
    }

    // Compute adjustment factors per stratum
    // When PSU j is deleted in stratum h with n_h PSUs:
    // - Deleted PSU: weight = 0
    // - Other PSUs in same stratum: weight *= n_h / (n_h - 1)
    // - PSUs in other strata: weight unchanged
    let stratum_nh: HashMap<i64, f64> = stratum_psus
        .iter()
        .map(|(&s, psus)| (s, psus.len() as f64))
        .collect();

    // Create replicate weights in parallel
    let rep_weights: Vec<Array1<f64>> = (0..n_reps)
        .into_par_iter()
        .map(|r| {
            let mut rep_wgt = wgt.to_owned();

            for i in 0..n_obs {
                let s = stratum_vec[i];
                let p = psu[i];
                let this_rep = psu_to_rep[&(s, p)];

                if this_rep == r {
                    // This PSU is deleted in this replicate
                    rep_wgt[i] = 0.0;
                } else {
                    // Find which stratum the deleted PSU belongs to
                    let deleted_stratum = find_deleted_stratum(&psu_to_rep, &stratum_psus, r);

                    if deleted_stratum == Some(s) {
                        // Same stratum: adjust weight
                        let nh = stratum_nh[&s];
                        rep_wgt[i] *= nh / (nh - 1.0);
                    }
                    // Different stratum: keep original weight
                }
            }

            rep_wgt
        })
        .collect();

    // Assemble result matrix
    let mut result = Array2::zeros((n_obs, n_reps));
    for (r, col) in rep_weights.into_iter().enumerate() {
        result.column_mut(r).assign(&col);
    }

    // Degrees of freedom = n_reps (number of PSUs)
    let df = n_reps as f64;

    Ok((result, df))
}

/// Helper to find which stratum the deleted PSU belongs to
fn find_deleted_stratum(
    psu_to_rep: &HashMap<(i64, i64), usize>,
    stratum_psus: &HashMap<i64, Vec<i64>>,
    rep_idx: usize,
) -> Option<i64> {
    for (&s, psus) in stratum_psus {
        for &p in psus {
            if psu_to_rep.get(&(s, p)) == Some(&rep_idx) {
                return Some(s);
            }
        }
    }
    None
}

// ============================================================================
// Bootstrap (Stratified)
// ============================================================================

/// Create Bootstrap replicate weights
///
/// Stratified bootstrap: within each stratum, resample PSUs with replacement.
///
/// # Arguments
/// * `wgt` - Base weights (n_obs,)
/// * `stratum` - Stratum identifiers (n_obs,), or None for unstratified
/// * `psu` - PSU identifiers (n_obs,)
/// * `n_reps` - Number of bootstrap replicates
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// * Replicate weight matrix (n_obs, n_reps)
/// * Degrees of freedom
pub fn create_bootstrap_weights(
    wgt: ArrayView1<f64>,
    stratum: Option<ArrayView1<i64>>,
    psu: ArrayView1<i64>,
    n_reps: usize,
    seed: u64,
) -> Result<(Array2<f64>, f64)> {
    let n_obs = wgt.len();

    if psu.len() != n_obs {
        return Err(ReplicationError::DimensionMismatch {
            expected: n_obs,
            got: psu.len(),
        });
    }

    // Handle stratification
    let stratum_vec: Array1<i64> = match stratum {
        Some(s) => {
            if s.len() != n_obs {
                return Err(ReplicationError::DimensionMismatch {
                    expected: n_obs,
                    got: s.len(),
                });
            }
            s.to_owned()
        }
        None => Array1::ones(n_obs),
    };

    // Build stratum -> PSUs mapping
    let stratum_psus = build_stratum_psu_list(&stratum_vec, &psu);

    // Create replicate weights in parallel
    // Use different seed for each replicate
    let rep_weights: Vec<Array1<f64>> = (0..n_reps)
        .into_par_iter()
        .map(|r| {
            // Simple LCG random number generator (thread-safe)
            let mut rng_state = seed.wrapping_add(r as u64);

            let mut rep_wgt = Array1::zeros(n_obs);

            for (&stratum_id, psus) in &stratum_psus {
                let n_psu = psus.len();
                if n_psu == 0 {
                    continue;
                }

                // Count how many times each PSU is selected
                let mut psu_counts: HashMap<i64, usize> = HashMap::new();
                for _ in 0..n_psu {
                    // Random PSU selection
                    rng_state = lcg_next(rng_state);
                    let selected_idx = (rng_state % n_psu as u64) as usize;
                    let selected_psu = psus[selected_idx];
                    *psu_counts.entry(selected_psu).or_insert(0) += 1;
                }

                // Apply counts to weights
                for i in 0..n_obs {
                    if stratum_vec[i] == stratum_id {
                        let p = psu[i];
                        let count = *psu_counts.get(&p).unwrap_or(&0);
                        rep_wgt[i] = wgt[i] * count as f64;
                    }
                }
            }

            rep_wgt
        })
        .collect();

    // Assemble result matrix
    let mut result = Array2::zeros((n_obs, n_reps));
    for (r, col) in rep_weights.into_iter().enumerate() {
        result.column_mut(r).assign(&col);
    }

    // Degrees of freedom = n_reps - 1
    let df = (n_reps - 1) as f64;

    Ok((result, df))
}

/// Simple LCG random number generator
fn lcg_next(state: u64) -> u64 {
    state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
}

// ============================================================================
// SDR (Successive Difference Replication)
// ============================================================================

/// Create SDR (Successive Difference Replication) weights
///
/// SDR is designed for systematic samples where units are ordered.
/// It creates replicates based on successive differences between adjacent units.
///
/// # Arguments
/// * `wgt` - Base weights (n_obs,)
/// * `stratum` - Stratum identifiers (n_obs,), or None for unstratified
/// * `order` - Sort order within strata (n_obs,) - units should be in systematic order
/// * `n_reps` - Number of replicates (typically 4 or more)
///
/// # Returns
/// * Replicate weight matrix (n_obs, n_reps)
/// * Degrees of freedom
///
/// # Reference
/// Fay & Train (1995): "Aspects of Survey and Model-Based Postcensal Estimation of Income and Poverty"
pub fn create_sdr_weights(
    wgt: ArrayView1<f64>,
    stratum: Option<ArrayView1<i64>>,
    order: Option<ArrayView1<i64>>,
    n_reps: usize,
) -> Result<(Array2<f64>, f64)> {
    let n_obs = wgt.len();

    if n_reps < 2 {
        return Err(ReplicationError::InvalidInput(
            "SDR requires at least 2 replicates".to_string()
        ));
    }

    // Handle stratification
    let stratum_vec: Array1<i64> = match stratum {
        Some(s) => {
            if s.len() != n_obs {
                return Err(ReplicationError::DimensionMismatch {
                    expected: n_obs,
                    got: s.len(),
                });
            }
            s.to_owned()
        }
        None => Array1::ones(n_obs),
    };

    // Handle ordering (default to row order)
    let order_vec: Array1<i64> = match order {
        Some(o) => {
            if o.len() != n_obs {
                return Err(ReplicationError::DimensionMismatch {
                    expected: n_obs,
                    got: o.len(),
                });
            }
            o.to_owned()
        }
        None => Array1::from_iter(0..n_obs as i64),
    };

    // Build stratum -> sorted indices mapping
    let stratum_sorted_indices = build_stratum_sorted_indices(&stratum_vec, &order_vec);

    // Generate Hadamard-like matrix for SDR
    let hadamard = generate_hadamard(next_power_of_2(n_reps));

    // Create replicate weights in parallel
    let rep_weights: Vec<Array1<f64>> = (0..n_reps)
        .into_par_iter()
        .map(|r| {
            let mut rep_wgt = wgt.to_owned();

            for (&_stratum_id, sorted_indices) in &stratum_sorted_indices {
                let n_h = sorted_indices.len();
                if n_h < 2 {
                    continue;
                }

                // SDR adjustment factors based on successive differences
                // For each pair of adjacent units, apply Hadamard-based adjustment
                for k in 0..(n_h - 1) {
                    let i1 = sorted_indices[k];
                    let i2 = sorted_indices[k + 1];

                    // Use Hadamard value for this pair and replicate
                    let h_idx = k % hadamard.nrows();
                    let h_val = hadamard[[h_idx, r % hadamard.ncols()]];

                    // SDR adjustment: sqrt(2) * h_val for successive difference
                    let adj = std::f64::consts::SQRT_2 * h_val / 2.0;

                    // Apply symmetric adjustment
                    rep_wgt[i1] *= 1.0 + adj;
                    rep_wgt[i2] *= 1.0 - adj;
                }
            }

            rep_wgt
        })
        .collect();

    // Assemble result matrix
    let mut result = Array2::zeros((n_obs, n_reps));
    for (r, col) in rep_weights.into_iter().enumerate() {
        result.column_mut(r).assign(&col);
    }

    // Degrees of freedom = n_reps
    let df = n_reps as f64;

    Ok((result, df))
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Build mapping of stratum -> (stratum_index, [psu1, psu2]) for BRR
fn build_stratum_psu_map(
    stratum: &ArrayView1<i64>,
    psu: &ArrayView1<i64>,
) -> Result<(HashMap<i64, (i64, Vec<i64>)>, HashMap<(i64, i64), usize>)> {
    let n = stratum.len();

    // Collect unique PSUs per stratum
    let mut stratum_psus: HashMap<i64, Vec<i64>> = HashMap::new();
    for i in 0..n {
        let s = stratum[i];
        let p = psu[i];
        let psus = stratum_psus.entry(s).or_insert_with(Vec::new);
        if !psus.contains(&p) {
            psus.push(p);
        }
    }

    // Validate: exactly 2 PSUs per stratum for BRR
    for (&s, psus) in &stratum_psus {
        if psus.len() != 2 {
            return Err(ReplicationError::BrrPsuCount {
                stratum: s,
                count: psus.len(),
            });
        }
    }

    // Build stratum index map
    let mut stratum_map: HashMap<i64, (i64, Vec<i64>)> = HashMap::new();
    let mut psu_to_idx: HashMap<(i64, i64), usize> = HashMap::new();

    for (idx, (&s, psus)) in stratum_psus.iter().enumerate() {
        stratum_map.insert(s, (idx as i64, psus.clone()));
        for (psu_idx, &p) in psus.iter().enumerate() {
            psu_to_idx.insert((s, p), psu_idx);
        }
    }

    Ok((stratum_map, psu_to_idx))
}

/// Build mapping of stratum -> [psu1, psu2, ...] for general methods
fn build_stratum_psu_list(
    stratum: &Array1<i64>,
    psu: &ArrayView1<i64>,
) -> HashMap<i64, Vec<i64>> {
    let n = stratum.len();
    let mut stratum_psus: HashMap<i64, Vec<i64>> = HashMap::new();

    for i in 0..n {
        let s = stratum[i];
        let p = psu[i];
        let psus = stratum_psus.entry(s).or_insert_with(Vec::new);
        if !psus.contains(&p) {
            psus.push(p);
        }
    }

    stratum_psus
}

/// Build mapping of stratum -> sorted observation indices for SDR
fn build_stratum_sorted_indices(
    stratum: &Array1<i64>,
    order: &Array1<i64>,
) -> HashMap<i64, Vec<usize>> {
    let n = stratum.len();

    // Group indices by stratum
    let mut stratum_indices: HashMap<i64, Vec<(i64, usize)>> = HashMap::new();
    for i in 0..n {
        let s = stratum[i];
        let o = order[i];
        stratum_indices.entry(s).or_insert_with(Vec::new).push((o, i));
    }

    // Sort each stratum's indices by order
    let mut result: HashMap<i64, Vec<usize>> = HashMap::new();
    for (s, mut indices) in stratum_indices {
        indices.sort_by_key(|(o, _)| *o);
        result.insert(s, indices.into_iter().map(|(_, i)| i).collect());
    }

    result
}

/// Generate Hadamard matrix of given size (must be power of 2)
fn generate_hadamard(n: usize) -> Array2<f64> {
    assert!(n.is_power_of_two(), "Hadamard size must be power of 2");

    let mut h = Array2::from_elem((1, 1), 1.0);

    while h.nrows() < n {
        let size = h.nrows();
        let mut new_h = Array2::zeros((size * 2, size * 2));

        // [H  H ]
        // [H -H]
        for i in 0..size {
            for j in 0..size {
                new_h[[i, j]] = h[[i, j]];
                new_h[[i, j + size]] = h[[i, j]];
                new_h[[i + size, j]] = h[[i, j]];
                new_h[[i + size, j + size]] = -h[[i, j]];
            }
        }

        h = new_h;
    }

    h
}

/// Round up to next power of 2
fn next_power_of_2(n: usize) -> usize {
    if n.is_power_of_two() {
        n
    } else {
        1 << (64 - (n - 1).leading_zeros())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_hadamard_generation() {
        let h4 = generate_hadamard(4);
        assert_eq!(h4.dim(), (4, 4));

        // Verify orthogonality: H * H' = n * I
        let hht = h4.dot(&h4.t());
        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 4.0 } else { 0.0 };
                assert!((hht[[i, j]] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_brr_basic() {
        let wgt = array![1.0, 1.0, 1.0, 1.0];
        let stratum = array![1, 1, 2, 2];
        let psu = array![1, 2, 1, 2];

        let (rep_wgt, df) = create_brr_weights(
            wgt.view(),
            stratum.view(),
            psu.view(),
            Some(4),
            0.0,
        ).unwrap();

        assert_eq!(rep_wgt.dim(), (4, 4));
        assert_eq!(df, 2.0);

        // Check that each replicate sums to total (no Fay adjustment)
        for r in 0..4 {
            let col_sum: f64 = rep_wgt.column(r).sum();
            assert!((col_sum - 4.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_jkn_basic() {
        let wgt = array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let stratum = array![1, 1, 1, 2, 2, 2];
        let psu = array![1, 2, 3, 1, 2, 3];

        let (rep_wgt, df) = create_jkn_weights(
            wgt.view(),
            Some(stratum.view()),
            psu.view(),
        ).unwrap();

        // 6 PSUs = 6 replicates
        assert_eq!(rep_wgt.dim(), (6, 6));
        assert_eq!(df, 6.0);
    }

    #[test]
    fn test_bootstrap_basic() {
        let wgt = array![1.0, 1.0, 1.0, 1.0];
        let stratum = array![1, 1, 2, 2];
        let psu = array![1, 2, 1, 2];

        let (rep_wgt, df) = create_bootstrap_weights(
            wgt.view(),
            Some(stratum.view()),
            psu.view(),
            10,
            42,
        ).unwrap();

        assert_eq!(rep_wgt.dim(), (4, 10));
        assert_eq!(df, 9.0);
    }

    #[test]
    fn test_sdr_basic() {
        let wgt = array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let order = array![0, 1, 2, 3, 4, 5];

        let (rep_wgt, df) = create_sdr_weights(
            wgt.view(),
            None,
            Some(order.view()),
            4,
        ).unwrap();

        assert_eq!(rep_wgt.dim(), (6, 4));
        assert_eq!(df, 4.0);
    }
}
