// src/estimation/replication.rs
//! Replication-based variance estimation (BRR, Bootstrap, Jackknife)

use polars::prelude::*;
use std::collections::HashMap;

/// Replication method
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RepMethod {
    BRR,
    Bootstrap,
    Jackknife,
    SDR,
}

impl RepMethod {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "brr" => Some(RepMethod::BRR),
            "bootstrap" | "bs" => Some(RepMethod::Bootstrap),
            "jackknife" | "jk" | "jk1" => Some(RepMethod::Jackknife),
            "sdr" | "acs" | "successive-difference" => Some(RepMethod::SDR),
            _ => None,
        }
    }
}

/// Compute replicate coefficients based on method
pub fn replicate_coefficients(method: RepMethod, n_reps: usize, fay_coef: f64) -> Vec<f64> {
    match method {
        RepMethod::Bootstrap => vec![1.0 / n_reps as f64; n_reps],
        RepMethod::BRR => {
            let scale = 1.0 / (n_reps as f64 * (1.0 - fay_coef).powi(2));
            vec![scale; n_reps]
        }
        RepMethod::Jackknife => {
            let coef = (n_reps as f64 - 1.0) / n_reps as f64;
            vec![coef; n_reps]
        }
        RepMethod::SDR => {
            // SDR (e.g. ACS): coefficient = 4/R for each replicate
            vec![4.0 / n_reps as f64; n_reps]
        }
    }
}

/// Compute variance from replicate estimates
///
/// # Arguments
/// * `method` - Replication method
/// * `theta_full` - Full sample estimate
/// * `theta_reps` - Vector of replicate estimates
/// * `rep_coefs` - Replicate coefficients
pub fn variance_from_replicates(
    method: RepMethod,
    theta_full: f64,
    theta_reps: &[f64],
    rep_coefs: &[f64],
) -> f64 {
    let n_reps = theta_reps.len();
    if n_reps == 0 {
        return 0.0;
    }

    match method {
        RepMethod::Jackknife => {
            // Jackknife pseudo-value approach
            // For JK1 with coef = (n-1)/n:
            // factor = 1 / (1 - coef) = n
            // pseudo = factor * theta_full - (factor - 1) * theta_rep

            let factors: Vec<f64> = rep_coefs.iter()
                .map(|&c| if c < 1.0 { 1.0 / (1.0 - c) } else { f64::INFINITY })
                .collect();

            // Compute pseudo values
            let pseudo: Vec<f64> = theta_reps.iter()
                .zip(factors.iter())
                .map(|(&rep, &f)| f * theta_full - (f - 1.0) * rep)
                .collect();

            // Center (mean of pseudo values)
            let center: f64 = pseudo.iter().sum::<f64>() / n_reps as f64;

            // Variance = sum(c * ((pseudo - center) / (factor - 1))^2)
            let var: f64 = pseudo.iter()
                .zip(rep_coefs.iter())
                .zip(factors.iter())
                .map(|((&p, &c), &f)| {
                    let denom = f - 1.0;
                    if denom > 0.0 {
                        let diff = (p - center) / denom;
                        c * diff * diff
                    } else {
                        0.0
                    }
                })
                .sum();

            var
        }
        RepMethod::SDR => {
            // SDR always uses MSE formula (center on full sample estimate)
            // Var = sum(coef * (theta_rep - theta_full)^2)
            let var: f64 = theta_reps.iter()
                .zip(rep_coefs.iter())
                .map(|(&rep, &c)| {
                    let diff = rep - theta_full;
                    c * diff * diff
                })
                .sum();

            var
        }
        RepMethod::BRR | RepMethod::Bootstrap => {
            // Simple squared difference from mean of replicates
            let center: f64 = theta_reps.iter().sum::<f64>() / n_reps as f64;

            let var: f64 = theta_reps.iter()
                .zip(rep_coefs.iter())
                .map(|(&rep, &c)| {
                    let diff = rep - center;
                    c * diff * diff
                })
                .sum();

            var
        }
    }
}

// ============================================================================
// Matrix-based computation for efficiency
// ============================================================================

/// Extract replicate weights matrix from DataFrame columns
/// Returns flattened row-major matrix (n × R)
pub fn extract_rep_weights_matrix(
    df: &DataFrame,
    rep_weight_cols: &[String],
) -> PolarsResult<(Vec<f64>, usize, usize)> {
    let n = df.height();
    let n_reps = rep_weight_cols.len();
    let mut matrix = vec![0.0; n * n_reps];

    for (r, col_name) in rep_weight_cols.iter().enumerate() {
        let col = df.column(col_name)?.f64()?;
        for (i, v) in col.into_iter().enumerate() {
            matrix[i * n_reps + r] = v.unwrap_or(0.0);
        }
    }

    Ok((matrix, n, n_reps))
}

/// Compute mean estimates for all replicates simultaneously
/// Returns (full_estimate, vec of replicate estimates)
pub fn matrix_mean_estimates(
    y: &[f64],
    full_weights: &[f64],
    rep_weights: &[f64],  // Flattened (n × R) row-major
    n: usize,
    n_reps: usize,
) -> (f64, Vec<f64>) {
    // Full sample estimate
    let sum_wy: f64 = y.iter().zip(full_weights.iter()).map(|(yi, wi)| yi * wi).sum();
    let sum_w: f64 = full_weights.iter().sum();
    let theta_full = if sum_w > 0.0 { sum_wy / sum_w } else { f64::NAN };

    // Replicate estimates - single pass
    let mut rep_sum_wy = vec![0.0; n_reps];
    let mut rep_sum_w = vec![0.0; n_reps];

    for i in 0..n {
        let yi = y[i];
        let base_idx = i * n_reps;
        for r in 0..n_reps {
            let w_ir = rep_weights[base_idx + r];
            rep_sum_wy[r] += yi * w_ir;
            rep_sum_w[r] += w_ir;
        }
    }

    // Compute means
    let theta_reps: Vec<f64> = rep_sum_wy.iter()
        .zip(rep_sum_w.iter())
        .map(|(&wy, &w)| if w > 0.0 { wy / w } else { f64::NAN })
        .collect();

    (theta_full, theta_reps)
}

/// Compute total estimates for all replicates
pub fn matrix_total_estimates(
    y: &[f64],
    full_weights: &[f64],
    rep_weights: &[f64],
    n: usize,
    n_reps: usize,
) -> (f64, Vec<f64>) {
    let theta_full: f64 = y.iter().zip(full_weights.iter()).map(|(yi, wi)| yi * wi).sum();

    let mut theta_reps = vec![0.0; n_reps];

    for i in 0..n {
        let yi = y[i];
        let base_idx = i * n_reps;
        for r in 0..n_reps {
            theta_reps[r] += yi * rep_weights[base_idx + r];
        }
    }

    (theta_full, theta_reps)
}

/// Compute ratio estimates for all replicates
pub fn matrix_ratio_estimates(
    y: &[f64],
    x: &[f64],
    full_weights: &[f64],
    rep_weights: &[f64],
    n: usize,
    n_reps: usize,
) -> (f64, Vec<f64>) {
    let sum_wy: f64 = y.iter().zip(full_weights.iter()).map(|(yi, wi)| yi * wi).sum();
    let sum_wx: f64 = x.iter().zip(full_weights.iter()).map(|(xi, wi)| xi * wi).sum();
    let theta_full = if sum_wx > 0.0 { sum_wy / sum_wx } else { f64::NAN };

    let mut rep_sum_wy = vec![0.0; n_reps];
    let mut rep_sum_wx = vec![0.0; n_reps];

    for i in 0..n {
        let yi = y[i];
        let xi = x[i];
        let base_idx = i * n_reps;
        for r in 0..n_reps {
            let w_ir = rep_weights[base_idx + r];
            rep_sum_wy[r] += yi * w_ir;
            rep_sum_wx[r] += xi * w_ir;
        }
    }

    let theta_reps: Vec<f64> = rep_sum_wy.iter()
        .zip(rep_sum_wx.iter())
        .map(|(&wy, &wx)| if wx > 0.0 { wy / wx } else { f64::NAN })
        .collect();

    (theta_full, theta_reps)
}

// ============================================================================
// Domain-level computation
// ============================================================================

/// Index categorical column to integers 0..K-1
/// Returns (indices, domain_names, n_domains)
pub fn index_domains(by: &StringChunked) -> (Vec<u32>, Vec<String>, usize) {
    let mut map: HashMap<&str, u32> = HashMap::new();
    let mut names: Vec<String> = Vec::new();
    let mut next_idx = 0u32;

    let indices: Vec<u32> = by.into_iter()
        .map(|opt| {
            match opt {
                Some(s) => {
                    *map.entry(s).or_insert_with(|| {
                        let idx = next_idx;
                        names.push(s.to_string());
                        next_idx += 1;
                        idx
                    })
                }
                None => u32::MAX
            }
        })
        .collect();

    (indices, names, next_idx as usize)
}

/// Compute mean estimates by domain for all replicates
/// Returns (full_estimates[K], replicate_estimates[K][R], domain_counts[K])
pub fn matrix_mean_by_domain(
    y: &[f64],
    full_weights: &[f64],
    rep_weights: &[f64],
    domain_ids: &[u32],
    n_domains: usize,
    n: usize,
    n_reps: usize,
) -> (Vec<f64>, Vec<Vec<f64>>, Vec<u32>) {
    let mut sum_wy = vec![0.0; n_domains];
    let mut sum_w = vec![0.0; n_domains];
    let mut rep_sum_wy = vec![vec![0.0; n_reps]; n_domains];
    let mut rep_sum_w = vec![vec![0.0; n_reps]; n_domains];
    let mut counts = vec![0u32; n_domains];

    for i in 0..n {
        let d = domain_ids[i] as usize;
        if d >= n_domains {
            continue;
        }

        let yi = y[i];
        let wi = full_weights[i];

        sum_wy[d] += yi * wi;
        sum_w[d] += wi;
        counts[d] += 1;

        let base_idx = i * n_reps;
        for r in 0..n_reps {
            let w_ir = rep_weights[base_idx + r];
            rep_sum_wy[d][r] += yi * w_ir;
            rep_sum_w[d][r] += w_ir;
        }
    }

    let theta_full: Vec<f64> = sum_wy.iter()
        .zip(sum_w.iter())
        .map(|(&wy, &w)| if w > 0.0 { wy / w } else { f64::NAN })
        .collect();

    let theta_reps: Vec<Vec<f64>> = rep_sum_wy.iter()
        .zip(rep_sum_w.iter())
        .map(|(wy_vec, w_vec)| {
            wy_vec.iter()
                .zip(w_vec.iter())
                .map(|(&wy, &w)| if w > 0.0 { wy / w } else { f64::NAN })
                .collect()
        })
        .collect();

    (theta_full, theta_reps, counts)
}

/// Compute total estimates by domain
pub fn matrix_total_by_domain(
    y: &[f64],
    full_weights: &[f64],
    rep_weights: &[f64],
    domain_ids: &[u32],
    n_domains: usize,
    n: usize,
    n_reps: usize,
) -> (Vec<f64>, Vec<Vec<f64>>, Vec<u32>) {
    let mut sum_wy = vec![0.0; n_domains];
    let mut rep_sum_wy = vec![vec![0.0; n_reps]; n_domains];
    let mut counts = vec![0u32; n_domains];

    for i in 0..n {
        let d = domain_ids[i] as usize;
        if d >= n_domains {
            continue;
        }

        let yi = y[i];
        let wi = full_weights[i];

        sum_wy[d] += yi * wi;
        counts[d] += 1;

        let base_idx = i * n_reps;
        for r in 0..n_reps {
            rep_sum_wy[d][r] += yi * rep_weights[base_idx + r];
        }
    }

    let theta_reps: Vec<Vec<f64>> = rep_sum_wy.iter()
        .map(|wy_vec| wy_vec.clone())
        .collect();

    (sum_wy, theta_reps, counts)
}

/// Compute ratio estimates by domain
pub fn matrix_ratio_by_domain(
    y: &[f64],
    x: &[f64],
    full_weights: &[f64],
    rep_weights: &[f64],
    domain_ids: &[u32],
    n_domains: usize,
    n: usize,
    n_reps: usize,
) -> (Vec<f64>, Vec<Vec<f64>>, Vec<u32>) {
    let mut sum_wy = vec![0.0; n_domains];
    let mut sum_wx = vec![0.0; n_domains];
    let mut rep_sum_wy = vec![vec![0.0; n_reps]; n_domains];
    let mut rep_sum_wx = vec![vec![0.0; n_reps]; n_domains];
    let mut counts = vec![0u32; n_domains];

    for i in 0..n {
        let d = domain_ids[i] as usize;
        if d >= n_domains {
            continue;
        }

        let yi = y[i];
        let xi = x[i];
        let wi = full_weights[i];

        sum_wy[d] += yi * wi;
        sum_wx[d] += xi * wi;
        counts[d] += 1;

        let base_idx = i * n_reps;
        for r in 0..n_reps {
            let w_ir = rep_weights[base_idx + r];
            rep_sum_wy[d][r] += yi * w_ir;
            rep_sum_wx[d][r] += xi * w_ir;
        }
    }

    let theta_full: Vec<f64> = sum_wy.iter()
        .zip(sum_wx.iter())
        .map(|(&wy, &wx)| if wx > 0.0 { wy / wx } else { f64::NAN })
        .collect();

    let theta_reps: Vec<Vec<f64>> = rep_sum_wy.iter()
        .zip(rep_sum_wx.iter())
        .map(|(wy_vec, wx_vec)| {
            wy_vec.iter()
                .zip(wx_vec.iter())
                .map(|(&wy, &wx)| if wx > 0.0 { wy / wx } else { f64::NAN })
                .collect()
        })
        .collect();

    (theta_full, theta_reps, counts)
}

// ============================================================================
// Proportion estimation (multi-category)
// ============================================================================

/// Compute proportion estimates by level for all replicates
/// Returns (levels, full_estimates[L], replicate_estimates[L][R])
pub fn matrix_prop_estimates(
    y: &[i64],  // Category values
    full_weights: &[f64],
    rep_weights: &[f64],
    n: usize,
    n_reps: usize,
) -> (Vec<i64>, Vec<f64>, Vec<Vec<f64>>) {
    // Find unique levels
    let mut level_map: HashMap<i64, usize> = HashMap::new();
    let mut levels: Vec<i64> = Vec::new();

    for &val in y {
        if !level_map.contains_key(&val) {
            level_map.insert(val, levels.len());
            levels.push(val);
        }
    }
    levels.sort();

    // Re-index after sorting
    for (idx, &lev) in levels.iter().enumerate() {
        level_map.insert(lev, idx);
    }

    let n_levels = levels.len();

    // Accumulate weighted counts per level
    let mut sum_w_level = vec![0.0; n_levels];
    let mut sum_w_total = 0.0;
    let mut rep_sum_w_level = vec![vec![0.0; n_reps]; n_levels];
    let mut rep_sum_w_total = vec![0.0; n_reps];

    for i in 0..n {
        let lev_idx = level_map[&y[i]];
        let wi = full_weights[i];

        sum_w_level[lev_idx] += wi;
        sum_w_total += wi;

        let base_idx = i * n_reps;
        for r in 0..n_reps {
            let w_ir = rep_weights[base_idx + r];
            rep_sum_w_level[lev_idx][r] += w_ir;
            rep_sum_w_total[r] += w_ir;
        }
    }

    // Compute proportions
    let theta_full: Vec<f64> = sum_w_level.iter()
        .map(|&w_l| if sum_w_total > 0.0 { w_l / sum_w_total } else { f64::NAN })
        .collect();

    let theta_reps: Vec<Vec<f64>> = rep_sum_w_level.iter()
        .map(|w_l_vec| {
            w_l_vec.iter()
                .zip(rep_sum_w_total.iter())
                .map(|(&w_l, &w_t)| if w_t > 0.0 { w_l / w_t } else { f64::NAN })
                .collect()
        })
        .collect();

    (levels, theta_full, theta_reps)
}

/// Compute proportion estimates by level and domain
pub fn matrix_prop_by_domain(
    y: &[i64],
    full_weights: &[f64],
    rep_weights: &[f64],
    domain_ids: &[u32],
    n_domains: usize,
    n: usize,
    n_reps: usize,
) -> (Vec<i64>, Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>, Vec<u32>) {
    // Find unique levels
    let mut level_map: HashMap<i64, usize> = HashMap::new();
    let mut levels: Vec<i64> = Vec::new();

    for &val in y {
        if !level_map.contains_key(&val) {
            level_map.insert(val, levels.len());
            levels.push(val);
        }
    }
    levels.sort();

    for (idx, &lev) in levels.iter().enumerate() {
        level_map.insert(lev, idx);
    }

    let n_levels = levels.len();

    // [domain][level]
    let mut sum_w_level = vec![vec![0.0; n_levels]; n_domains];
    let mut sum_w_total = vec![0.0; n_domains];
    // [domain][level][rep]
    let mut rep_sum_w_level = vec![vec![vec![0.0; n_reps]; n_levels]; n_domains];
    let mut rep_sum_w_total = vec![vec![0.0; n_reps]; n_domains];
    let mut counts = vec![0u32; n_domains];

    for i in 0..n {
        let d = domain_ids[i] as usize;
        if d >= n_domains {
            continue;
        }

        let lev_idx = level_map[&y[i]];
        let wi = full_weights[i];

        sum_w_level[d][lev_idx] += wi;
        sum_w_total[d] += wi;
        counts[d] += 1;

        let base_idx = i * n_reps;
        for r in 0..n_reps {
            let w_ir = rep_weights[base_idx + r];
            rep_sum_w_level[d][lev_idx][r] += w_ir;
            rep_sum_w_total[d][r] += w_ir;
        }
    }

    // Compute proportions [domain][level]
    let theta_full: Vec<Vec<f64>> = sum_w_level.iter()
        .zip(sum_w_total.iter())
        .map(|(w_l_vec, &w_t)| {
            w_l_vec.iter()
                .map(|&w_l| if w_t > 0.0 { w_l / w_t } else { f64::NAN })
                .collect()
        })
        .collect();

    // [domain][level][rep]
    let theta_reps: Vec<Vec<Vec<f64>>> = rep_sum_w_level.iter()
        .zip(rep_sum_w_total.iter())
        .map(|(dom_levels, dom_totals)| {
            dom_levels.iter()
                .map(|w_l_reps| {
                    w_l_reps.iter()
                        .zip(dom_totals.iter())
                        .map(|(&w_l, &w_t)| if w_t > 0.0 { w_l / w_t } else { f64::NAN })
                        .collect()
                })
                .collect()
        })
        .collect();

    (levels, theta_full, theta_reps, counts)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdr_coefficients() {
        let coefs = replicate_coefficients(RepMethod::SDR, 80, 0.0);
        assert_eq!(coefs.len(), 80);
        assert!((coefs[0] - 4.0 / 80.0).abs() < 1e-10);
    }

    #[test]
    fn test_sdr_variance() {
        // Simple test: 4 replicates
        let theta_full = 100.0;
        let theta_reps = vec![98.0, 102.0, 99.0, 101.0];
        let rep_coefs = replicate_coefficients(RepMethod::SDR, 4, 0.0);

        // Var = (4/4) * sum((rep - 100)^2)
        //     = 1.0 * (4 + 4 + 1 + 1) = 10.0
        let var = variance_from_replicates(RepMethod::SDR, theta_full, &theta_reps, &rep_coefs);
        assert!((var - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_sdr_from_str() {
        assert_eq!(RepMethod::from_str("sdr"), Some(RepMethod::SDR));
        assert_eq!(RepMethod::from_str("SDR"), Some(RepMethod::SDR));
        assert_eq!(RepMethod::from_str("acs"), Some(RepMethod::SDR));
        assert_eq!(RepMethod::from_str("ACS"), Some(RepMethod::SDR));
        assert_eq!(RepMethod::from_str("successive-difference"), Some(RepMethod::SDR));
    }
}
