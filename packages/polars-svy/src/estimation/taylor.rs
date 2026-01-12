// src/estimation/taylor.rs

use polars::prelude::*;
use std::collections::HashMap;

// ============================================================================
// Linearization Scores
// ============================================================================

/// Calculate linearization scores for mean estimation
/// Score: z_i = (w_i / N) * (y_i - ȳ)
pub fn scores_mean(y: &Float64Chunked, weights: &Float64Chunked) -> PolarsResult<Float64Chunked> {
    let sum_wy: f64 = y.iter()
        .zip(weights.iter())
        .filter_map(|(yi, wi)| Some(yi? * wi?))
        .sum();

    let sum_w: f64 = weights.iter().filter_map(|w| w).sum();

    if sum_w == 0.0 {
        return Err(PolarsError::ComputeError(
            "Sum of weights is zero".into()
        ));
    }

    let mean = sum_wy / sum_w;

    let scores: Vec<Option<f64>> = y.iter()
        .zip(weights.iter())
        .map(|(yi, wi)| {
            match (yi, wi) {
                (Some(y_val), Some(w_val)) => {
                    Some((w_val / sum_w) * (y_val - mean))
                }
                _ => None
            }
        })
        .collect();

    Ok(Float64Chunked::from_slice_options("scores".into(), &scores))
}

/// Calculate linearization scores for mean estimation within a domain
pub fn scores_mean_domain(
    y: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
) -> PolarsResult<Float64Chunked> {
    let sum_wy: f64 = y.iter()
        .zip(weights.iter())
        .zip(domain_mask.iter())
        .filter_map(|((yi, wi), in_domain)| {
            if in_domain? { Some(yi? * wi?) } else { None }
        })
        .sum();

    let sum_w: f64 = weights.iter()
        .zip(domain_mask.iter())
        .filter_map(|(w, in_domain)| {
            if in_domain? { w } else { None }
        })
        .sum();

    if sum_w == 0.0 {
        let zeros: Vec<Option<f64>> = vec![Some(0.0); y.len()];
        return Ok(Float64Chunked::from_slice_options("scores".into(), &zeros));
    }

    let mean = sum_wy / sum_w;

    let scores: Vec<Option<f64>> = y.iter()
        .zip(weights.iter())
        .zip(domain_mask.iter())
        .map(|((yi, wi), in_domain)| {
            match (yi, wi, in_domain) {
                (Some(y_val), Some(w_val), Some(true)) => {
                    Some((w_val / sum_w) * (y_val - mean))
                }
                _ => Some(0.0)
            }
        })
        .collect();

    Ok(Float64Chunked::from_slice_options("scores".into(), &scores))
}

/// Calculate linearization scores for total estimation
pub fn scores_total(y: &Float64Chunked, weights: &Float64Chunked) -> PolarsResult<Float64Chunked> {
    let scores: Vec<Option<f64>> = y.iter()
        .zip(weights.iter())
        .map(|(yi, wi)| {
            match (yi, wi) {
                (Some(y_val), Some(w_val)) => Some(w_val * y_val),
                _ => None
            }
        })
        .collect();

    Ok(Float64Chunked::from_slice_options("scores".into(), &scores))
}

/// Calculate linearization scores for total estimation within a domain
pub fn scores_total_domain(
    y: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
) -> PolarsResult<Float64Chunked> {
    let scores: Vec<Option<f64>> = y.iter()
        .zip(weights.iter())
        .zip(domain_mask.iter())
        .map(|((yi, wi), in_domain)| {
            match (yi, wi, in_domain) {
                (Some(y_val), Some(w_val), Some(true)) => Some(w_val * y_val),
                _ => Some(0.0)
            }
        })
        .collect();

    Ok(Float64Chunked::from_slice_options("scores".into(), &scores))
}

/// Calculate linearization scores for ratio estimation
pub fn scores_ratio(
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked
) -> PolarsResult<Float64Chunked> {
    let sum_wy: f64 = y.iter()
        .zip(weights.iter())
        .filter_map(|(yi, wi)| Some(yi? * wi?))
        .sum();

    let sum_wx: f64 = x.iter()
        .zip(weights.iter())
        .filter_map(|(xi, wi)| Some(xi? * wi?))
        .sum();

    if sum_wx == 0.0 {
        return Err(PolarsError::ComputeError(
            "Weighted sum of x is zero in ratio estimation".into()
        ));
    }

    let ratio = sum_wy / sum_wx;

    let scores: Vec<Option<f64>> = y.iter()
        .zip(x.iter())
        .zip(weights.iter())
        .map(|((yi, xi), wi)| {
            match (yi, xi, wi) {
                (Some(y_val), Some(x_val), Some(w_val)) => {
                    Some((w_val / sum_wx) * (y_val - ratio * x_val))
                }
                _ => None
            }
        })
        .collect();

    Ok(Float64Chunked::from_slice_options("scores".into(), &scores))
}

/// Calculate linearization scores for ratio estimation within a domain
pub fn scores_ratio_domain(
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
) -> PolarsResult<Float64Chunked> {
    let sum_wy: f64 = y.iter()
        .zip(weights.iter())
        .zip(domain_mask.iter())
        .filter_map(|((yi, wi), in_domain)| {
            if in_domain? { Some(yi? * wi?) } else { None }
        })
        .sum();

    let sum_wx: f64 = x.iter()
        .zip(weights.iter())
        .zip(domain_mask.iter())
        .filter_map(|((xi, wi), in_domain)| {
            if in_domain? { Some(xi? * wi?) } else { None }
        })
        .sum();

    if sum_wx == 0.0 {
        let zeros: Vec<Option<f64>> = vec![Some(0.0); y.len()];
        return Ok(Float64Chunked::from_slice_options("scores".into(), &zeros));
    }

    let ratio = sum_wy / sum_wx;

    let scores: Vec<Option<f64>> = y.iter()
        .zip(x.iter())
        .zip(weights.iter())
        .zip(domain_mask.iter())
        .map(|(((yi, xi), wi), in_domain)| {
            match (yi, xi, wi, in_domain) {
                (Some(y_val), Some(x_val), Some(w_val), Some(true)) => {
                    Some((w_val / sum_wx) * (y_val - ratio * x_val))
                }
                _ => Some(0.0)
            }
        })
        .collect();

    Ok(Float64Chunked::from_slice_options("scores".into(), &scores))
}

// ============================================================================
// OPTIMIZED Taylor Variance - With Two-Stage Support
// ============================================================================

/// Pre-index categorical columns to integer indices
/// Returns (indices, n_unique)
fn index_categorical(col: &StringChunked) -> (Vec<u32>, u32) {
    let mut map: HashMap<&str, u32> = HashMap::new();
    let mut next_idx = 0u32;

    let indices: Vec<u32> = col.iter()
        .map(|opt| {
            match opt {
                Some(s) => {
                    *map.entry(s).or_insert_with(|| {
                        let idx = next_idx;
                        next_idx += 1;
                        idx
                    })
                }
                None => u32::MAX // Sentinel for null
            }
        })
        .collect();

    (indices, next_idx)
}

/// Re-index values within a subset to contiguous integers 0..n-1
fn reindex_within_subset(raw: &[u32]) -> (Vec<u32>, u32) {
    let mut map: HashMap<u32, u32> = HashMap::new();
    let mut next_idx = 0u32;

    let indices: Vec<u32> = raw.iter()
        .map(|&val| {
            if val == u32::MAX {
                u32::MAX
            } else {
                *map.entry(val).or_insert_with(|| {
                    let idx = next_idx;
                    next_idx += 1;
                    idx
                })
            }
        })
        .collect();

    (indices, next_idx)
}

/// Compute variance within a single PSU (between SSUs) for second stage
fn variance_within_psu(
    scores: &[f64],
    ssu_indices: &[u32],
    n_ssus: u32,
) -> f64 {
    if n_ssus <= 1 {
        return 0.0;
    }

    // Accumulate SSU totals
    let mut ssu_totals = vec![0.0; n_ssus as usize];
    for (score, &ssu) in scores.iter().zip(ssu_indices.iter()) {
        if ssu != u32::MAX {
            ssu_totals[ssu as usize] += score;
        }
    }

    // Variance of SSU totals
    let ssu_mean = ssu_totals.iter().sum::<f64>() / (n_ssus as f64);
    let sum_sq_diff: f64 = ssu_totals.iter()
        .map(|&t| (t - ssu_mean).powi(2))
        .sum();

    (n_ssus as f64 / (n_ssus as f64 - 1.0)) * sum_sq_diff
}

/// Optimized variance for unstratified design (single pass)
fn variance_unstratified_optimized(
    scores: &[f64],
    psu_indices: Option<&[u32]>,
    n_psus: u32,
) -> f64 {
    let n = scores.len();
    if n == 0 {
        return 0.0;
    }

    match psu_indices {
        Some(psu_idx) => {
            // Clustered: accumulate PSU totals in single pass
            if n_psus <= 1 {
                return 0.0;
            }

            let mut psu_totals = vec![0.0; n_psus as usize];
            for (score, &psu) in scores.iter().zip(psu_idx.iter()) {
                if psu != u32::MAX {
                    psu_totals[psu as usize] += score;
                }
            }

            let psu_mean = psu_totals.iter().sum::<f64>() / (n_psus as f64);
            let sum_sq_diff: f64 = psu_totals.iter()
                .map(|&t| (t - psu_mean).powi(2))
                .sum();

            (n_psus as f64 / (n_psus as f64 - 1.0)) * sum_sq_diff
        }
        None => {
            // SRS: simple variance of scores
            if n <= 1 {
                return 0.0;
            }

            let mean: f64 = scores.iter().sum::<f64>() / (n as f64);
            let sum_sq_diff: f64 = scores.iter()
                .map(|&s| (s - mean).powi(2))
                .sum();

            (n as f64 / (n as f64 - 1.0)) * sum_sq_diff
        }
    }
}

/// Optimized variance for stratified design (single pass accumulation)
fn variance_stratified_optimized(
    scores: &[f64],
    strata_indices: &[u32],
    n_strata: u32,
    psu_indices: Option<&[u32]>,
    psu_per_stratum: Option<&[Vec<u32>]>,
    n_psus_per_stratum: Option<&[u32]>,
) -> f64 {
    let n = scores.len();
    if n == 0 || n_strata == 0 {
        return 0.0;
    }

    match (psu_indices, psu_per_stratum, n_psus_per_stratum) {
        (Some(psu_idx), Some(psu_map), Some(n_psus)) => {
            // Stratified clustered design
            let max_psu = psu_idx.iter().filter(|&&p| p != u32::MAX).max().copied().unwrap_or(0);
            let mut psu_totals = vec![0.0; (max_psu + 1) as usize];

            // Single pass: accumulate scores by PSU
            for (score, &psu) in scores.iter().zip(psu_idx.iter()) {
                if psu != u32::MAX {
                    psu_totals[psu as usize] += score;
                }
            }

            // Calculate variance per stratum
            let mut total_var = 0.0;
            for h in 0..n_strata as usize {
                let n_psus_h = n_psus[h];
                if n_psus_h <= 1 {
                    continue;
                }

                let psu_indices_h = &psu_map[h];
                let psu_totals_h: Vec<f64> = psu_indices_h.iter()
                    .map(|&p| psu_totals[p as usize])
                    .collect();

                let psu_mean_h = psu_totals_h.iter().sum::<f64>() / (n_psus_h as f64);
                let sum_sq_diff: f64 = psu_totals_h.iter()
                    .map(|&t| (t - psu_mean_h).powi(2))
                    .sum();

                let var_h = (n_psus_h as f64 / (n_psus_h as f64 - 1.0)) * sum_sq_diff;
                total_var += var_h;
            }

            total_var
        }
        _ => {
            // Stratified without clustering
            let mut stratum_sums = vec![0.0; n_strata as usize];
            let mut stratum_sum_sq = vec![0.0; n_strata as usize];
            let mut stratum_counts = vec![0u32; n_strata as usize];

            for (&score, &stratum) in scores.iter().zip(strata_indices.iter()) {
                if stratum != u32::MAX {
                    let h = stratum as usize;
                    stratum_sums[h] += score;
                    stratum_sum_sq[h] += score * score;
                    stratum_counts[h] += 1;
                }
            }

            let mut total_var = 0.0;
            for h in 0..n_strata as usize {
                let n_h = stratum_counts[h];
                if n_h <= 1 {
                    continue;
                }
                let var_h = (stratum_sum_sq[h] - stratum_sums[h] * stratum_sums[h] / (n_h as f64))
                            / (n_h as f64 - 1.0) * (n_h as f64);
                total_var += var_h;
            }

            total_var
        }
    }
}

/// Build stratum -> PSU mapping for stratified clustered designs
fn build_stratum_psu_map(
    strata_indices: &[u32],
    n_strata: u32,
    psu_indices: &[u32],
) -> (Vec<Vec<u32>>, Vec<u32>) {
    let mut stratum_psus: Vec<HashMap<u32, ()>> = vec![HashMap::new(); n_strata as usize];

    for (&stratum, &psu) in strata_indices.iter().zip(psu_indices.iter()) {
        if stratum != u32::MAX && psu != u32::MAX {
            stratum_psus[stratum as usize].insert(psu, ());
        }
    }

    let psu_per_stratum: Vec<Vec<u32>> = stratum_psus.iter()
        .map(|m| m.keys().copied().collect())
        .collect();

    let n_psus_per_stratum: Vec<u32> = psu_per_stratum.iter()
        .map(|v| v.len() as u32)
        .collect();

    (psu_per_stratum, n_psus_per_stratum)
}

/// Compute second-stage variance contribution
///
/// For each PSU, compute within-PSU variance (between SSUs) and scale by
/// the first-stage sampling fraction.
///
/// Formula: V_stage2 = Σ (1 - fpc1) * fpc2 * V_within_psu
///
/// With default fpc=1.0, the contribution is zero (matching R's ultimate cluster approach)
fn compute_stage2_variance(
    scores: &[f64],
    psu_indices: &[u32],
    ssu_indices: &[u32],
    _strata_indices: Option<&[u32]>,
    fpc: f64,
    fpc_stage2: f64,
) -> f64 {
    let n = scores.len();
    if n == 0 {
        return 0.0;
    }

    // Early exit if stage1 sampling fraction is 0 (fpc = 1.0 means no second stage contribution)
    let stage1_sampling_fraction = 1.0 - fpc;
    if stage1_sampling_fraction <= 0.0 {
        return 0.0;
    }

    // Group observations by PSU
    let max_psu = psu_indices.iter().filter(|&&p| p != u32::MAX).max().copied().unwrap_or(0);
    let n_psus = (max_psu + 1) as usize;

    // Build PSU -> observations mapping
    let mut psu_obs: Vec<Vec<usize>> = vec![Vec::new(); n_psus];
    for (i, &psu) in psu_indices.iter().enumerate() {
        if psu != u32::MAX {
            psu_obs[psu as usize].push(i);
        }
    }

    let mut total_stage2_var = 0.0;

    for psu in 0..n_psus {
        let obs_indices = &psu_obs[psu];
        if obs_indices.is_empty() {
            continue;
        }

        // Extract scores and SSU indices for this PSU
        let psu_scores: Vec<f64> = obs_indices.iter().map(|&i| scores[i]).collect();
        let psu_ssu_raw: Vec<u32> = obs_indices.iter().map(|&i| ssu_indices[i]).collect();

        // Re-index SSUs within this PSU to 0..n_ssus-1
        let (psu_ssu_indices, n_ssus) = reindex_within_subset(&psu_ssu_raw);

        if n_ssus <= 1 {
            // Only one SSU in this PSU, no within-PSU variance contribution
            continue;
        }

        // Compute within-PSU variance (between SSUs)
        let var_within_psu = variance_within_psu(&psu_scores, &psu_ssu_indices, n_ssus);

        // Scale by sampling fractions
        // V_total = V_1 + (n_1/N_1) * V_2
        // where n_1/N_1 = 1 - fpc1 (first-stage sampling fraction)
        total_stage2_var += stage1_sampling_fraction * fpc_stage2 * var_within_psu;
    }

    total_stage2_var
}

/// Main Taylor variance calculation with two-stage support
///
/// When ssu is None: one-stage variance (between PSUs within strata)
/// When ssu is provided: two-stage variance using:
///     V(θ̂) = V₁ + Σ (1 - fpc1) * fpc2 * V₂
///
/// With default fpc=1.0 and fpc_stage2=1.0, the second stage contribution
/// is zero (since 1 - fpc = 0), matching R's ultimate cluster approach.
pub fn taylor_variance(
    scores: &Float64Chunked,
    strata: Option<&StringChunked>,
    psu: Option<&StringChunked>,
    ssu: Option<&StringChunked>,
    fpc: Option<&Float64Chunked>,
    fpc_stage2: Option<&Float64Chunked>,
) -> PolarsResult<f64> {
    let n = scores.len();
    if n == 0 {
        return Ok(0.0);
    }

    // Extract scores to contiguous array
    let scores_arr: Vec<f64> = scores.iter()
        .map(|s| s.unwrap_or(0.0))
        .collect();

    // Get FPC values (use first value, assume constant within design)
    let fpc_val = fpc.and_then(|f| f.get(0)).unwrap_or(1.0);
    let fpc_stage2_val = fpc_stage2.and_then(|f| f.get(0)).unwrap_or(1.0);

    // =========================================================================
    // STAGE 1: Between-PSU variance
    // =========================================================================

    let (var_stage1, psu_indices_opt, strata_indices_opt) = if strata.is_none() {
        // Case 1: No stratification
        let (psu_indices, n_psus) = match psu {
            Some(psu_col) => {
                let (idx, n) = index_categorical(psu_col);
                (Some(idx), n)
            }
            None => (None, 0)
        };

        let var = variance_unstratified_optimized(
            &scores_arr,
            psu_indices.as_deref(),
            n_psus,
        );

        (fpc_val * var, psu_indices, None)
    } else {
        // Case 2: Stratified design
        let strata_col = strata.unwrap();
        let (strata_indices, n_strata) = index_categorical(strata_col);

        match psu {
            Some(psu_col) => {
                // Stratified clustered
                let (psu_indices, _) = index_categorical(psu_col);
                let (psu_per_stratum, n_psus_per_stratum) = build_stratum_psu_map(
                    &strata_indices,
                    n_strata,
                    &psu_indices,
                );

                let var = variance_stratified_optimized(
                    &scores_arr,
                    &strata_indices,
                    n_strata,
                    Some(&psu_indices),
                    Some(&psu_per_stratum),
                    Some(&n_psus_per_stratum),
                );

                (fpc_val * var, Some(psu_indices), Some(strata_indices))
            }
            None => {
                // Stratified only (no clustering)
                let var = variance_stratified_optimized(
                    &scores_arr,
                    &strata_indices,
                    n_strata,
                    None,
                    None,
                    None,
                );

                (fpc_val * var, None, Some(strata_indices))
            }
        }
    };

    // =========================================================================
    // STAGE 2: Within-PSU variance (if ssu provided)
    // =========================================================================

    if ssu.is_none() || psu.is_none() {
        return Ok(var_stage1);
    }

    let ssu_col = ssu.unwrap();
    let (ssu_indices, _) = index_categorical(ssu_col);

    // Use previously computed PSU indices, or compute if needed
    let psu_indices = match psu_indices_opt {
        Some(idx) => idx,
        None => {
            let psu_col = psu.unwrap();
            index_categorical(psu_col).0
        }
    };

    let var_stage2 = compute_stage2_variance(
        &scores_arr,
        &psu_indices,
        &ssu_indices,
        strata_indices_opt.as_deref(),
        fpc_val,
        fpc_stage2_val,
    );

    Ok(var_stage1 + var_stage2)
}

/// Calculate degrees of freedom
pub fn degrees_of_freedom(
    weights: &Float64Chunked,
    strata: Option<&StringChunked>,
    psu: Option<&StringChunked>,
) -> PolarsResult<u32> {
    let n = weights.len();

    if n == 0 {
        return Ok(0);
    }

    if strata.is_none() {
        if psu.is_none() {
            return Ok(n.saturating_sub(1) as u32);
        }
        let (_, n_psus) = index_categorical(psu.unwrap());
        return Ok(n_psus.saturating_sub(1));
    }

    let strata_col = strata.unwrap();
    let (strata_indices, n_strata) = index_categorical(strata_col);

    match psu {
        Some(psu_col) => {
            let (psu_indices, _) = index_categorical(psu_col);
            let (_, n_psus_per_stratum) = build_stratum_psu_map(
                &strata_indices,
                n_strata,
                &psu_indices,
            );

            let total_df: u32 = n_psus_per_stratum.iter()
                .map(|&n| n.saturating_sub(1))
                .sum();

            Ok(total_df)
        }
        None => {
            let mut stratum_counts = vec![0u32; n_strata as usize];
            for &stratum in &strata_indices {
                if stratum != u32::MAX {
                    stratum_counts[stratum as usize] += 1;
                }
            }

            let total_df: u32 = stratum_counts.iter()
                .map(|&n| n.saturating_sub(1))
                .sum();

            Ok(total_df)
        }
    }
}

// ============================================================================
// Point Estimates
// ============================================================================

pub fn point_estimate_mean(y: &Float64Chunked, weights: &Float64Chunked) -> PolarsResult<f64> {
    let sum_wy: f64 = y.iter()
        .zip(weights.iter())
        .filter_map(|(yi, wi)| Some(yi? * wi?))
        .sum();

    let sum_w: f64 = weights.iter().filter_map(|w| w).sum();

    if sum_w == 0.0 {
        return Err(PolarsError::ComputeError("Sum of weights is zero".into()));
    }

    Ok(sum_wy / sum_w)
}

pub fn point_estimate_mean_domain(
    y: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
) -> PolarsResult<f64> {
    let sum_wy: f64 = y.iter()
        .zip(weights.iter())
        .zip(domain_mask.iter())
        .filter_map(|((yi, wi), in_domain)| {
            if in_domain? { Some(yi? * wi?) } else { None }
        })
        .sum();

    let sum_w: f64 = weights.iter()
        .zip(domain_mask.iter())
        .filter_map(|(w, in_domain)| {
            if in_domain? { w } else { None }
        })
        .sum();

    if sum_w == 0.0 {
        return Err(PolarsError::ComputeError("Sum of weights is zero".into()));
    }

    Ok(sum_wy / sum_w)
}

pub fn point_estimate_total(y: &Float64Chunked, weights: &Float64Chunked) -> PolarsResult<f64> {
    let total: f64 = y.iter()
        .zip(weights.iter())
        .filter_map(|(yi, wi)| Some(yi? * wi?))
        .sum();

    Ok(total)
}

pub fn point_estimate_total_domain(
    y: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
) -> PolarsResult<f64> {
    let total: f64 = y.iter()
        .zip(weights.iter())
        .zip(domain_mask.iter())
        .filter_map(|((yi, wi), in_domain)| {
            if in_domain? { Some(yi? * wi?) } else { None }
        })
        .sum();

    Ok(total)
}

pub fn point_estimate_ratio(
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked
) -> PolarsResult<f64> {
    let sum_wy: f64 = y.iter()
        .zip(weights.iter())
        .filter_map(|(yi, wi)| Some(yi? * wi?))
        .sum();

    let sum_wx: f64 = x.iter()
        .zip(weights.iter())
        .filter_map(|(xi, wi)| Some(xi? * wi?))
        .sum();

    if sum_wx == 0.0 {
        return Err(PolarsError::ComputeError(
            "Weighted sum of x is zero in ratio estimation".into()
        ));
    }

    Ok(sum_wy / sum_wx)
}

pub fn point_estimate_ratio_domain(
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
) -> PolarsResult<f64> {
    let sum_wy: f64 = y.iter()
        .zip(weights.iter())
        .zip(domain_mask.iter())
        .filter_map(|((yi, wi), in_domain)| {
            if in_domain? { Some(yi? * wi?) } else { None }
        })
        .sum();

    let sum_wx: f64 = x.iter()
        .zip(weights.iter())
        .zip(domain_mask.iter())
        .filter_map(|((xi, wi), in_domain)| {
            if in_domain? { Some(xi? * wi?) } else { None }
        })
        .sum();

    if sum_wx == 0.0 {
        return Err(PolarsError::ComputeError(
            "Weighted sum of x is zero in ratio estimation".into()
        ));
    }

    Ok(sum_wy / sum_wx)
}

// ============================================================================
// SRS Variance Functions (for DEFF calculation)
// ============================================================================

fn weighted_s2(y: &[f64], wn: &[f64]) -> f64 {
    let n = y.len() as f64;
    if n <= 1.0 {
        return f64::NAN;
    }

    let mu: f64 = y.iter().zip(wn.iter()).map(|(yi, wi)| wi * yi).sum();
    let ss: f64 = y.iter()
        .zip(wn.iter())
        .map(|(yi, wi)| wi * (yi - mu).powi(2))
        .sum();

    (n / (n - 1.0)) * ss
}

pub fn srs_variance_mean(y: &Float64Chunked, weights: &Float64Chunked) -> PolarsResult<f64> {
    let n = y.len() as f64;
    if n < 2.0 {
        return Ok(f64::NAN);
    }

    let sum_w: f64 = weights.into_iter().filter_map(|v| v).sum();
    if sum_w <= 0.0 {
        return Ok(f64::NAN);
    }

    let wn: Vec<f64> = weights.into_iter()
        .map(|v| v.unwrap_or(0.0) / sum_w)
        .collect();
    let yv: Vec<f64> = y.into_iter().map(|v| v.unwrap_or(0.0)).collect();

    let s2_y = weighted_s2(&yv, &wn);
    let vsrs = s2_y / n;
    let fpc = 1.0 - (n / sum_w);

    Ok(vsrs * fpc)
}

pub fn srs_variance_mean_domain(
    y: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
) -> PolarsResult<f64> {
    let mut yv = Vec::new();
    let mut wv = Vec::new();

    for ((yi, wi), mi) in y.into_iter().zip(weights.into_iter()).zip(domain_mask.into_iter()) {
        if let (Some(y_val), Some(w_val), Some(true)) = (yi, wi, mi) {
            yv.push(y_val);
            wv.push(w_val);
        }
    }

    let n = yv.len() as f64;
    if n < 2.0 {
        return Ok(f64::NAN);
    }

    let sum_w: f64 = wv.iter().sum();
    if sum_w <= 0.0 {
        return Ok(f64::NAN);
    }

    let wn: Vec<f64> = wv.iter().map(|w| w / sum_w).collect();
    let s2_y = weighted_s2(&yv, &wn);
    let vsrs = s2_y / n;
    let fpc = 1.0 - (n / sum_w);

    Ok(vsrs * fpc)
}

pub fn srs_variance_total(y: &Float64Chunked, weights: &Float64Chunked) -> PolarsResult<f64> {
    let n = y.len() as f64;
    if n < 2.0 {
        return Ok(f64::NAN);
    }

    let sum_w: f64 = weights.into_iter().filter_map(|v| v).sum();
    if sum_w <= 0.0 {
        return Ok(f64::NAN);
    }

    let wn: Vec<f64> = weights.into_iter()
        .map(|v| v.unwrap_or(0.0) / sum_w)
        .collect();
    let yv: Vec<f64> = y.into_iter().map(|v| v.unwrap_or(0.0)).collect();

    let s2_y = weighted_s2(&yv, &wn);
    let vsrs = (sum_w.powi(2) / n) * s2_y;
    let fpc = 1.0 - (n / sum_w);

    Ok(vsrs * fpc)
}

pub fn srs_variance_total_domain(
    y: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
) -> PolarsResult<f64> {
    let mut yv = Vec::new();
    let mut wv = Vec::new();

    for ((yi, wi), mi) in y.into_iter().zip(weights.into_iter()).zip(domain_mask.into_iter()) {
        if let (Some(y_val), Some(w_val), Some(true)) = (yi, wi, mi) {
            yv.push(y_val);
            wv.push(w_val);
        }
    }

    let n = yv.len() as f64;
    if n < 2.0 {
        return Ok(f64::NAN);
    }

    let sum_w: f64 = wv.iter().sum();
    if sum_w <= 0.0 {
        return Ok(f64::NAN);
    }

    let wn: Vec<f64> = wv.iter().map(|w| w / sum_w).collect();
    let s2_y = weighted_s2(&yv, &wn);
    let vsrs = (sum_w.powi(2) / n) * s2_y;
    let fpc = 1.0 - (n / sum_w);

    Ok(vsrs * fpc)
}

pub fn srs_variance_ratio(
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked,
) -> PolarsResult<f64> {
    let n = y.len() as f64;
    if n < 2.0 {
        return Ok(f64::NAN);
    }

    let sum_w: f64 = weights.into_iter().filter_map(|v| v).sum();
    if sum_w <= 0.0 {
        return Ok(f64::NAN);
    }

    let wn: Vec<f64> = weights.into_iter()
        .map(|v| v.unwrap_or(0.0) / sum_w)
        .collect();
    let yv: Vec<f64> = y.into_iter().map(|v| v.unwrap_or(0.0)).collect();
    let xv: Vec<f64> = x.into_iter().map(|v| v.unwrap_or(0.0)).collect();

    let ybar: f64 = yv.iter().zip(wn.iter()).map(|(yi, wi)| wi * yi).sum();
    let xbar: f64 = xv.iter().zip(wn.iter()).map(|(xi, wi)| wi * xi).sum();

    if xbar == 0.0 {
        return Ok(f64::NAN);
    }

    let rhat = ybar / xbar;
    let ev: Vec<f64> = yv.iter().zip(xv.iter()).map(|(yi, xi)| yi - rhat * xi).collect();

    let s2_e = weighted_s2(&ev, &wn);
    let vsrs = s2_e / (n * xbar.powi(2));
    let fpc = 1.0 - (n / sum_w);

    Ok(vsrs * fpc)
}

pub fn srs_variance_ratio_domain(
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
) -> PolarsResult<f64> {
    let mut yv = Vec::new();
    let mut xv = Vec::new();
    let mut wv = Vec::new();

    for (((yi, xi), wi), mi) in y.into_iter().zip(x.into_iter()).zip(weights.into_iter()).zip(domain_mask.into_iter()) {
        if let (Some(y_val), Some(x_val), Some(w_val), Some(true)) = (yi, xi, wi, mi) {
            yv.push(y_val);
            xv.push(x_val);
            wv.push(w_val);
        }
    }

    let n = yv.len() as f64;
    if n < 2.0 {
        return Ok(f64::NAN);
    }

    let sum_w: f64 = wv.iter().sum();
    if sum_w <= 0.0 {
        return Ok(f64::NAN);
    }

    let wn: Vec<f64> = wv.iter().map(|w| w / sum_w).collect();

    let ybar: f64 = yv.iter().zip(wn.iter()).map(|(yi, wi)| wi * yi).sum();
    let xbar: f64 = xv.iter().zip(wn.iter()).map(|(xi, wi)| wi * xi).sum();

    if xbar == 0.0 {
        return Ok(f64::NAN);
    }

    let rhat = ybar / xbar;
    let ev: Vec<f64> = yv.iter().zip(xv.iter()).map(|(yi, xi)| yi - rhat * xi).collect();

    let s2_e = weighted_s2(&ev, &wn);
    let vsrs = s2_e / (n * xbar.powi(2));
    let fpc = 1.0 - (n / sum_w);

    Ok(vsrs * fpc)
}

pub fn srs_variance_prop(p: f64, n: f64, sum_w: f64) -> f64 {
    if n < 2.0 || sum_w <= 0.0 {
        return f64::NAN;
    }
    let vsrs = p * (1.0 - p) * (n / (n - 1.0)) / n;
    let fpc = 1.0 - (n / sum_w);
    vsrs * fpc
}
