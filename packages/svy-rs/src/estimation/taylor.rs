// src/estimation/taylor.rs

use polars::prelude::*;
use std::collections::HashMap;

// ============================================================================
// Enums & Config
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SingletonMethod {
    None,       // Default: Treat n=1 as 0 variance contribution
    Center,     // Grand mean centering: (z_i - z_bar)^2
}

/// Quantile interpolation method (matches R's approxfun and Python's QuantileMethod)
/// Named SvyQuantileMethod to avoid collision with polars::prelude::QuantileMethod
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SvyQuantileMethod {
    /// Return lower value at discontinuity (R: method="constant", f=0)
    Lower,
    /// Return higher value at discontinuity (R: method="constant", f=1)
    #[default]
    Higher,
    /// Return midpoint of lower and higher
    Middle,
    /// Return nearest value
    Nearest,
    /// Linear interpolation (R: method="linear", default)
    Linear,
}

impl SvyQuantileMethod {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "lower" => SvyQuantileMethod::Lower,
            "higher" => SvyQuantileMethod::Higher,
            "middle" => SvyQuantileMethod::Middle,
            "nearest" => SvyQuantileMethod::Nearest,
            "linear" => SvyQuantileMethod::Linear,
            _ => SvyQuantileMethod::Higher, // default
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

pub fn point_estimate_mean_domain(y: &Float64Chunked, weights: &Float64Chunked, domain_mask: &BooleanChunked) -> PolarsResult<f64> {
    let sum_wy: f64 = y.iter().zip(weights.iter()).zip(domain_mask.iter())
        .filter_map(|((yi, wi), m)| if m? { Some(yi? * wi?) } else { None })
        .sum();
    let sum_w: f64 = weights.iter().zip(domain_mask.iter())
        .filter_map(|(w, m)| if m? { w } else { None })
        .sum();

    if sum_w == 0.0 {
        return Err(PolarsError::ComputeError("Sum of weights is zero in domain".into()));
    }
    Ok(sum_wy / sum_w)
}

pub fn point_estimate_total(y: &Float64Chunked, weights: &Float64Chunked) -> PolarsResult<f64> {
    Ok(y.iter().zip(weights.iter()).filter_map(|(yi, wi)| Some(yi? * wi?)).sum())
}

pub fn point_estimate_total_domain(y: &Float64Chunked, weights: &Float64Chunked, domain_mask: &BooleanChunked) -> PolarsResult<f64> {
    Ok(y.iter().zip(weights.iter()).zip(domain_mask.iter())
        .filter_map(|((yi, wi), m)| if m? { Some(yi? * wi?) } else { None })
        .sum())
}

pub fn point_estimate_ratio(y: &Float64Chunked, x: &Float64Chunked, weights: &Float64Chunked) -> PolarsResult<f64> {
    let sum_wy: f64 = y.iter().zip(weights.iter()).filter_map(|(yi, wi)| Some(yi? * wi?)).sum();
    let sum_wx: f64 = x.iter().zip(weights.iter()).filter_map(|(xi, wi)| Some(xi? * wi?)).sum();

    if sum_wx == 0.0 {
        return Err(PolarsError::ComputeError("Weighted sum of denominator (x) is zero".into()));
    }
    Ok(sum_wy / sum_wx)
}

pub fn point_estimate_ratio_domain(y: &Float64Chunked, x: &Float64Chunked, weights: &Float64Chunked, domain_mask: &BooleanChunked) -> PolarsResult<f64> {
    let sum_wy: f64 = y.iter().zip(weights.iter()).zip(domain_mask.iter())
        .filter_map(|((yi, wi), m)| if m? { Some(yi? * wi?) } else { None }).sum();
    let sum_wx: f64 = x.iter().zip(weights.iter()).zip(domain_mask.iter())
        .filter_map(|((xi, wi), m)| if m? { Some(xi? * wi?) } else { None }).sum();

    if sum_wx == 0.0 {
        return Err(PolarsError::ComputeError("Weighted sum of denominator (x) is zero in domain".into()));
    }
    Ok(sum_wy / sum_wx)
}

// ============================================================================
// Weighted Quantile Functions
// ============================================================================

/// Compute weighted quantile using the specified interpolation method.
///
/// This matches R's svyquantile behavior:
/// - Lower: method="constant", f=0
/// - Higher: method="constant", f=1
/// - Linear: method="linear" (default in R)
///
/// # Arguments
/// * `y_sorted` - Y values sorted in ascending order
/// * `cdf` - Cumulative distribution function values (cumsum(weights) / sum(weights))
/// * `p` - Target quantile (0.0 to 1.0, e.g., 0.5 for median)
/// * `method` - Interpolation method
pub fn weighted_quantile(
    y_sorted: &[f64],
    cdf: &[f64],
    p: f64,
    method: SvyQuantileMethod,
) -> f64 {
    let n = y_sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return y_sorted[0];
    }
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }

    // Find bracketing indices
    let (left, right) = if p <= cdf[0] {
        (0, 1.min(n - 1))
    } else if p >= cdf[n - 1] {
        ((n - 2).max(0), n - 1)
    } else {
        // Binary search for the right position
        let idx = cdf.partition_point(|&x| x < p);
        ((idx.saturating_sub(1)), idx.min(n - 1))
    };

    match method {
        SvyQuantileMethod::Lower => y_sorted[left],
        SvyQuantileMethod::Higher => y_sorted[right],
        SvyQuantileMethod::Middle => (y_sorted[left] + y_sorted[right]) / 2.0,
        SvyQuantileMethod::Nearest => {
            let dl = (p - cdf[left]).abs();
            let dr = (cdf[right] - p).abs();
            if dl <= dr { y_sorted[left] } else { y_sorted[right] }
        }
        SvyQuantileMethod::Linear => {
            let denom = cdf[right] - cdf[left];
            if denom <= 0.0 {
                return y_sorted[left];
            }
            // Special case: if p exactly equals cdf[left] and we have two distinct points
            if (p - cdf[left]).abs() < 1e-12 && right != left {
                return (y_sorted[left] + y_sorted[right]) / 2.0;
            }
            let w = (p - cdf[left]) / denom;
            (1.0 - w) * y_sorted[left] + w * y_sorted[right]
        }
    }
}

/// Compute weighted median (p=0.5 quantile)
pub fn weighted_median(
    y: &Float64Chunked,
    weights: &Float64Chunked,
    method: SvyQuantileMethod,
) -> PolarsResult<f64> {
    weighted_quantile_chunked(y, weights, 0.5, method)
}

/// Compute weighted median for a domain subset
pub fn weighted_median_domain(
    y: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
    method: SvyQuantileMethod,
) -> PolarsResult<f64> {
    // Collect values in domain
    let mut pairs: Vec<(f64, f64)> = y.iter()
        .zip(weights.iter())
        .zip(domain_mask.iter())
        .filter_map(|((yi, wi), m)| {
            if m? && yi.is_some() && wi.is_some() {
                Some((yi.unwrap(), wi.unwrap()))
            } else {
                None
            }
        })
        .collect();

    if pairs.is_empty() {
        return Ok(f64::NAN);
    }

    // Sort by y value
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let y_sorted: Vec<f64> = pairs.iter().map(|(y, _)| *y).collect();
    let w_sorted: Vec<f64> = pairs.iter().map(|(_, w)| *w).collect();

    // Compute CDF
    let total_w: f64 = w_sorted.iter().sum();
    if total_w <= 0.0 {
        return Ok(f64::NAN);
    }

    let mut cdf = Vec::with_capacity(w_sorted.len());
    let mut cumsum = 0.0;
    for w in &w_sorted {
        cumsum += w;
        cdf.push(cumsum / total_w);
    }

    Ok(weighted_quantile(&y_sorted, &cdf, 0.5, method))
}

/// Compute weighted quantile from Polars chunked arrays
pub fn weighted_quantile_chunked(
    y: &Float64Chunked,
    weights: &Float64Chunked,
    p: f64,
    method: SvyQuantileMethod,
) -> PolarsResult<f64> {
    // Collect non-null pairs
    let mut pairs: Vec<(f64, f64)> = y.iter()
        .zip(weights.iter())
        .filter_map(|(yi, wi)| Some((yi?, wi?)))
        .collect();

    if pairs.is_empty() {
        return Ok(f64::NAN);
    }

    // Sort by y value
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let y_sorted: Vec<f64> = pairs.iter().map(|(y, _)| *y).collect();
    let w_sorted: Vec<f64> = pairs.iter().map(|(_, w)| *w).collect();

    // Compute CDF
    let total_w: f64 = w_sorted.iter().sum();
    if total_w <= 0.0 {
        return Ok(f64::NAN);
    }

    let mut cdf = Vec::with_capacity(w_sorted.len());
    let mut cumsum = 0.0;
    for w in &w_sorted {
        cumsum += w;
        cdf.push(cumsum / total_w);
    }

    Ok(weighted_quantile(&y_sorted, &cdf, p, method))
}

// ============================================================================
// Linearization Scores
// ============================================================================

pub fn scores_mean(y: &Float64Chunked, weights: &Float64Chunked) -> PolarsResult<Float64Chunked> {
    let est = point_estimate_mean(y, weights)?;
    let sum_w: f64 = weights.iter().filter_map(|w| w).sum();

    let scores: Vec<Option<f64>> = y.iter().zip(weights.iter())
        .map(|(yi, wi)| match (yi, wi) {
            (Some(y_val), Some(w_val)) => Some((w_val / sum_w) * (y_val - est)),
            _ => None,
        }).collect();

    Ok(Float64Chunked::from_slice_options("scores".into(), &scores))
}

pub fn scores_mean_domain(y: &Float64Chunked, weights: &Float64Chunked, domain_mask: &BooleanChunked) -> PolarsResult<Float64Chunked> {
    // If domain is empty or sum_w is 0, return 0 scores safely
    let sum_w: f64 = weights.iter().zip(domain_mask.iter())
        .filter_map(|(w, m)| if m? { w } else { None }).sum();

    if sum_w == 0.0 {
        let zeros = vec![Some(0.0); y.len()];
        return Ok(Float64Chunked::from_slice_options("scores".into(), &zeros));
    }

    let est = point_estimate_mean_domain(y, weights, domain_mask)?;

    let scores: Vec<Option<f64>> = y.iter().zip(weights.iter()).zip(domain_mask.iter())
        .map(|((yi, wi), m)| match (yi, wi, m) {
            (Some(y_val), Some(w_val), Some(true)) => Some((w_val / sum_w) * (y_val - est)),
            _ => Some(0.0), // Zero score outside domain
        }).collect();

    Ok(Float64Chunked::from_slice_options("scores".into(), &scores))
}

pub fn scores_total(y: &Float64Chunked, weights: &Float64Chunked) -> PolarsResult<Float64Chunked> {
    let scores: Vec<Option<f64>> = y.iter().zip(weights.iter())
        .map(|(yi, wi)| match (yi, wi) {
            (Some(y_val), Some(w_val)) => Some(w_val * y_val),
            _ => None,
        }).collect();
    Ok(Float64Chunked::from_slice_options("scores".into(), &scores))
}

pub fn scores_total_domain(y: &Float64Chunked, weights: &Float64Chunked, domain_mask: &BooleanChunked) -> PolarsResult<Float64Chunked> {
    let scores: Vec<Option<f64>> = y.iter().zip(weights.iter()).zip(domain_mask.iter())
        .map(|((yi, wi), m)| match (yi, wi, m) {
            (Some(y_val), Some(w_val), Some(true)) => Some(w_val * y_val),
            _ => Some(0.0),
        }).collect();
    Ok(Float64Chunked::from_slice_options("scores".into(), &scores))
}

pub fn scores_ratio(y: &Float64Chunked, x: &Float64Chunked, weights: &Float64Chunked) -> PolarsResult<Float64Chunked> {
    let r_hat = point_estimate_ratio(y, x, weights)?;
    let sum_wx: f64 = x.iter().zip(weights.iter()).filter_map(|(xi, wi)| Some(xi? * wi?)).sum();

    let scores: Vec<Option<f64>> = y.iter().zip(x.iter()).zip(weights.iter())
        .map(|((yi, xi), wi)| match (yi, xi, wi) {
            (Some(y_val), Some(x_val), Some(w_val)) => Some((w_val / sum_wx) * (y_val - r_hat * x_val)),
            _ => None,
        }).collect();
    Ok(Float64Chunked::from_slice_options("scores".into(), &scores))
}

pub fn scores_ratio_domain(y: &Float64Chunked, x: &Float64Chunked, weights: &Float64Chunked, domain_mask: &BooleanChunked) -> PolarsResult<Float64Chunked> {
    let sum_wx: f64 = x.iter().zip(weights.iter()).zip(domain_mask.iter())
        .filter_map(|((xi, wi), m)| if m? { Some(xi? * wi?) } else { None }).sum();

    if sum_wx == 0.0 {
        let zeros = vec![Some(0.0); y.len()];
        return Ok(Float64Chunked::from_slice_options("scores".into(), &zeros));
    }

    let r_hat = point_estimate_ratio_domain(y, x, weights, domain_mask)?;

    let scores: Vec<Option<f64>> = y.iter().zip(x.iter()).zip(weights.iter()).zip(domain_mask.iter())
        .map(|(((yi, xi), wi), m)| match (yi, xi, wi, m) {
            (Some(y_val), Some(x_val), Some(w_val), Some(true)) =>
                Some((w_val / sum_wx) * (y_val - r_hat * x_val)),
            _ => Some(0.0),
        }).collect();
    Ok(Float64Chunked::from_slice_options("scores".into(), &scores))
}

/// Compute influence function scores for median (quantile) estimation.
///
/// For median, the influence function is based on the indicator I(y <= q) - p
/// where q is the quantile and p is the target probability (0.5 for median).
///
/// Score_i = (w_i / sum_w) * (I(y_i > q) - (1 - p))
///
/// This follows the approach used in R's survey package.
pub fn scores_median(
    y: &Float64Chunked,
    weights: &Float64Chunked,
    method: SvyQuantileMethod,
) -> PolarsResult<Float64Chunked> {
    let _p = 0.5; // median
    let q = weighted_median(y, weights, method)?;

    if q.is_nan() {
        let nans = vec![Some(f64::NAN); y.len()];
        return Ok(Float64Chunked::from_slice_options("scores".into(), &nans));
    }

    let sum_w: f64 = weights.iter().filter_map(|w| w).sum();
    if sum_w <= 0.0 {
        let zeros = vec![Some(0.0); y.len()];
        return Ok(Float64Chunked::from_slice_options("scores".into(), &zeros));
    }

    let scores: Vec<Option<f64>> = y.iter().zip(weights.iter())
        .map(|(yi, wi)| match (yi, wi) {
            (Some(y_val), Some(w_val)) => {
                // u = I(y > q) - (1 - p) = I(y > q) - 0.5
                let u = if y_val > q { 1.0 } else { 0.0 } - 0.5;
                Some((w_val / sum_w) * u)
            },
            _ => None,
        }).collect();

    Ok(Float64Chunked::from_slice_options("scores".into(), &scores))
}

/// Compute influence function scores for median within a domain.
pub fn scores_median_domain(
    y: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
    method: SvyQuantileMethod,
) -> PolarsResult<Float64Chunked> {
    let q = weighted_median_domain(y, weights, domain_mask, method)?;

    if q.is_nan() {
        let zeros = vec![Some(0.0); y.len()];
        return Ok(Float64Chunked::from_slice_options("scores".into(), &zeros));
    }

    let sum_w: f64 = weights.iter().zip(domain_mask.iter())
        .filter_map(|(w, m)| if m? { w } else { None }).sum();

    if sum_w <= 0.0 {
        let zeros = vec![Some(0.0); y.len()];
        return Ok(Float64Chunked::from_slice_options("scores".into(), &zeros));
    }

    let scores: Vec<Option<f64>> = y.iter().zip(weights.iter()).zip(domain_mask.iter())
        .map(|((yi, wi), m)| match (yi, wi, m) {
            (Some(y_val), Some(w_val), Some(true)) => {
                let u = if y_val > q { 1.0 } else { 0.0 } - 0.5;
                Some((w_val / sum_w) * u)
            },
            _ => Some(0.0), // Zero score outside domain
        }).collect();

    Ok(Float64Chunked::from_slice_options("scores".into(), &scores))
}

// ============================================================================
// Variance Helpers (Indexing & Math)
// ============================================================================

fn index_categorical(col: &StringChunked) -> (Vec<u32>, u32) {
    let mut map: HashMap<&str, u32> = HashMap::new();
    let mut next_idx = 0u32;
    let indices: Vec<u32> = col.iter().map(|opt| match opt {
        Some(s) => *map.entry(s).or_insert_with(|| { let i = next_idx; next_idx += 1; i }),
        None => u32::MAX
    }).collect();
    (indices, next_idx)
}

fn reindex_within_subset(raw: &[u32]) -> (Vec<u32>, u32) {
    let mut map: HashMap<u32, u32> = HashMap::new();
    let mut next_idx = 0u32;
    let indices: Vec<u32> = raw.iter().map(|&val| {
        if val == u32::MAX { u32::MAX }
        else { *map.entry(val).or_insert_with(|| { let i = next_idx; next_idx += 1; i }) }
    }).collect();
    (indices, next_idx)
}

fn build_stratum_psu_map(strata_indices: &[u32], n_strata: u32, psu_indices: &[u32]) -> (Vec<Vec<u32>>, Vec<u32>) {
    let mut stratum_psus: Vec<HashMap<u32, ()>> = vec![HashMap::new(); n_strata as usize];
    for (&stratum, &psu) in strata_indices.iter().zip(psu_indices.iter()) {
        if stratum != u32::MAX && psu != u32::MAX {
            stratum_psus[stratum as usize].insert(psu, ());
        }
    }
    let psu_per_stratum: Vec<Vec<u32>> = stratum_psus.iter().map(|m| m.keys().copied().collect()).collect();
    let n_psus_per_stratum: Vec<u32> = psu_per_stratum.iter().map(|v| v.len() as u32).collect();
    (psu_per_stratum, n_psus_per_stratum)
}

fn variance_unstratified_optimized(scores: &[f64], psu_indices: Option<&[u32]>, n_psus: u32) -> f64 {
    let n = scores.len();
    if n == 0 { return 0.0; }

    match psu_indices {
        Some(psu_idx) => {
            if n_psus <= 1 { return 0.0; }
            let mut psu_totals = vec![0.0; n_psus as usize];
            for (score, &psu) in scores.iter().zip(psu_idx.iter()) {
                if psu != u32::MAX { psu_totals[psu as usize] += score; }
            }
            let psu_mean = psu_totals.iter().sum::<f64>() / (n_psus as f64);
            let sum_sq_diff: f64 = psu_totals.iter().map(|&t| (t - psu_mean).powi(2)).sum();
            (n_psus as f64 / (n_psus as f64 - 1.0)) * sum_sq_diff
        }
        None => {
            if n <= 1 { return 0.0; }
            let mean: f64 = scores.iter().sum::<f64>() / (n as f64);
            let sum_sq_diff: f64 = scores.iter().map(|&s| (s - mean).powi(2)).sum();
            (n as f64 / (n as f64 - 1.0)) * sum_sq_diff
        }
    }
}

fn variance_stratified_optimized(
    scores: &[f64],
    strata_indices: &[u32],
    n_strata: u32,
    psu_indices: Option<&[u32]>,
    psu_per_stratum: Option<&[Vec<u32>]>,
    n_psus_per_stratum: Option<&[u32]>,
    singleton_method: SingletonMethod,
) -> f64 {
    let n = scores.len();
    if n == 0 || n_strata == 0 { return 0.0; }

    match (psu_indices, psu_per_stratum, n_psus_per_stratum) {
        (Some(psu_idx), Some(psu_map), Some(n_psus)) => {
            let max_psu = psu_idx.iter().filter(|&&p| p != u32::MAX).max().copied().unwrap_or(0);
            let mut psu_totals = vec![0.0; (max_psu + 1) as usize];
            let mut psu_exists = vec![false; (max_psu + 1) as usize];

            for (score, &psu) in scores.iter().zip(psu_idx.iter()) {
                if psu != u32::MAX {
                    psu_totals[psu as usize] += score;
                    psu_exists[psu as usize] = true;
                }
            }

            let mut grand_mean = 0.0;
            if singleton_method == SingletonMethod::Center {
                let total_score: f64 = psu_totals.iter().sum();
                let total_count = psu_exists.iter().filter(|&&e| e).count();
                if total_count > 0 { grand_mean = total_score / (total_count as f64); }
            }

            let mut total_var = 0.0;
            for h in 0..n_strata as usize {
                let n_psus_h = n_psus[h];
                if n_psus_h == 0 { continue; }
                if n_psus_h == 1 {
                    if singleton_method == SingletonMethod::Center {
                        if let Some(&p) = psu_map[h].first() {
                            total_var += (psu_totals[p as usize] - grand_mean).powi(2);
                        }
                    }
                    continue;
                }

                let psu_indices_h = &psu_map[h];
                let psu_totals_h: Vec<f64> = psu_indices_h.iter().map(|&p| psu_totals[p as usize]).collect();
                let psu_mean_h = psu_totals_h.iter().sum::<f64>() / (n_psus_h as f64);
                let sum_sq_diff: f64 = psu_totals_h.iter().map(|&t| (t - psu_mean_h).powi(2)).sum();
                total_var += (n_psus_h as f64 / (n_psus_h as f64 - 1.0)) * sum_sq_diff;
            }
            total_var
        }
        _ => {
            // Stratified element sampling
            let mut stratum_sums = vec![0.0; n_strata as usize];
            let mut stratum_sum_sq = vec![0.0; n_strata as usize];
            let mut stratum_counts = vec![0u32; n_strata as usize];
            let mut grand_total = 0.0;
            let mut total_n = 0;

            for (&score, &stratum) in scores.iter().zip(strata_indices.iter()) {
                if stratum != u32::MAX {
                    let h = stratum as usize;
                    stratum_sums[h] += score;
                    stratum_sum_sq[h] += score * score;
                    stratum_counts[h] += 1;
                    if singleton_method == SingletonMethod::Center {
                        grand_total += score;
                        total_n += 1;
                    }
                }
            }

            let mut grand_mean = 0.0;
            if singleton_method == SingletonMethod::Center && total_n > 0 {
                grand_mean = grand_total / (total_n as f64);
            }

            let mut total_var = 0.0;
            for h in 0..n_strata as usize {
                let n_h = stratum_counts[h];
                if n_h == 0 { continue; }
                if n_h == 1 {
                    if singleton_method == SingletonMethod::Center {
                        total_var += (stratum_sums[h] - grand_mean).powi(2);
                    }
                    continue;
                }
                let var_h = (stratum_sum_sq[h] - stratum_sums[h].powi(2) / (n_h as f64)) / (n_h as f64 - 1.0) * (n_h as f64);
                total_var += var_h;
            }
            total_var
        }
    }
}

fn compute_stage2_variance(
    scores: &[f64], psu_indices: &[u32], ssu_indices: &[u32], _strata_indices: Option<&[u32]>,
    fpc: f64, fpc_stage2: f64
) -> f64 {
    let n = scores.len();
    if n == 0 { return 0.0; }

    let stage1_sampling_fraction = 1.0 - fpc;
    if stage1_sampling_fraction <= 0.0 { return 0.0; }

    let max_psu = psu_indices.iter().filter(|&&p| p != u32::MAX).max().copied().unwrap_or(0);
    let n_psus = (max_psu + 1) as usize;
    let mut psu_obs: Vec<Vec<usize>> = vec![Vec::new(); n_psus];

    for (i, &psu) in psu_indices.iter().enumerate() {
        if psu != u32::MAX { psu_obs[psu as usize].push(i); }
    }

    let mut total_stage2_var = 0.0;
    for psu in 0..n_psus {
        let obs_indices = &psu_obs[psu];
        if obs_indices.is_empty() { continue; }

        let psu_scores: Vec<f64> = obs_indices.iter().map(|&i| scores[i]).collect();
        let psu_ssu_raw: Vec<u32> = obs_indices.iter().map(|&i| ssu_indices[i]).collect();
        let (psu_ssu_indices, n_ssus) = reindex_within_subset(&psu_ssu_raw);

        if n_ssus <= 1 { continue; }

        let mut ssu_totals = vec![0.0; n_ssus as usize];
        for (score, &ssu) in psu_scores.iter().zip(psu_ssu_indices.iter()) {
            if ssu != u32::MAX { ssu_totals[ssu as usize] += score; }
        }
        let ssu_mean = ssu_totals.iter().sum::<f64>() / (n_ssus as f64);
        let sum_sq_diff: f64 = ssu_totals.iter().map(|&t| (t - ssu_mean).powi(2)).sum();
        let var_within_psu = (n_ssus as f64 / (n_ssus as f64 - 1.0)) * sum_sq_diff;

        total_stage2_var += stage1_sampling_fraction * fpc_stage2 * var_within_psu;
    }
    total_stage2_var
}

// ============================================================================
// Main Public Variance Function
// ============================================================================

pub fn taylor_variance(
    scores: &Float64Chunked,
    strata: Option<&StringChunked>,
    psu: Option<&StringChunked>,
    ssu: Option<&StringChunked>,
    fpc: Option<&Float64Chunked>,
    fpc_stage2: Option<&Float64Chunked>,
    singleton_method: Option<&str>,
) -> PolarsResult<f64> {
    let n = scores.len();
    if n == 0 { return Ok(0.0); }

    let sm_enum = match singleton_method {
        Some(s) if s.eq_ignore_ascii_case("center") || s.eq_ignore_ascii_case("adjust") => SingletonMethod::Center,
        _ => SingletonMethod::None,
    };

    let scores_arr: Vec<f64> = scores.iter().map(|s| s.unwrap_or(0.0)).collect();
    let fpc_val = fpc.and_then(|f| f.get(0)).unwrap_or(1.0);
    let fpc_stage2_val = fpc_stage2.and_then(|f| f.get(0)).unwrap_or(1.0);

    // --- STAGE 1 VARIANCE ---
    let (var_stage1, psu_indices_opt, strata_indices_opt) = if strata.is_none() {
        let (psu_indices, n_psus) = match psu {
            Some(psu_col) => {
                let (idx, n) = index_categorical(psu_col);
                (Some(idx), n)
            }
            None => (None, 0)
        };
        let var = variance_unstratified_optimized(&scores_arr, psu_indices.as_deref(), n_psus);
        (fpc_val * var, psu_indices, None)
    } else {
        let strata_col = strata.unwrap();
        let (strata_indices, n_strata) = index_categorical(strata_col);

        match psu {
            Some(psu_col) => {
                let (psu_indices, _) = index_categorical(psu_col);
                let (psu_per_stratum, n_psus_per_stratum) = build_stratum_psu_map(&strata_indices, n_strata, &psu_indices);
                let var = variance_stratified_optimized(
                    &scores_arr, &strata_indices, n_strata, Some(&psu_indices),
                    Some(&psu_per_stratum), Some(&n_psus_per_stratum), sm_enum
                );
                (fpc_val * var, Some(psu_indices), Some(strata_indices))
            }
            None => {
                let var = variance_stratified_optimized(
                    &scores_arr, &strata_indices, n_strata, None, None, None, sm_enum
                );
                (fpc_val * var, None, Some(strata_indices))
            }
        }
    };

    // --- STAGE 2 VARIANCE ---
    if ssu.is_none() || psu.is_none() {
        return Ok(var_stage1);
    }

    let ssu_col = ssu.unwrap();
    let (ssu_indices, _) = index_categorical(ssu_col);

    // We need PSUs for stage 2. If design is unstratified element sampling (psu=None),
    // then rows are PSUs.
    let psu_indices = match psu_indices_opt {
        Some(idx) => idx,
        None => {
            // Implicit PSUs = rows.
            if let Some(p) = psu { index_categorical(p).0 } else { (0..n as u32).collect() }
        }
    };

    let var_stage2 = compute_stage2_variance(
        &scores_arr, &psu_indices, &ssu_indices, strata_indices_opt.as_deref(), fpc_val, fpc_stage2_val
    );

    Ok(var_stage1 + var_stage2)
}

pub fn degrees_of_freedom(
    weights: &Float64Chunked,
    strata: Option<&StringChunked>,
    psu: Option<&StringChunked>,
) -> PolarsResult<u32> {
    let n = weights.len();
    if n == 0 { return Ok(0); }

    if strata.is_none() {
        if psu.is_none() { return Ok(n.saturating_sub(1) as u32); }
        let (_, n_psus) = index_categorical(psu.unwrap());
        return Ok(n_psus.saturating_sub(1));
    }

    let strata_col = strata.unwrap();
    let (strata_indices, n_strata) = index_categorical(strata_col);

    match psu {
        Some(psu_col) => {
            let (psu_indices, _) = index_categorical(psu_col);
            let (_, n_psus_per_stratum) = build_stratum_psu_map(&strata_indices, n_strata, &psu_indices);
            let total_df: u32 = n_psus_per_stratum.iter().map(|&n| n.saturating_sub(1)).sum();
            Ok(total_df)
        }
        None => {
            let mut stratum_counts = vec![0u32; n_strata as usize];
            for &stratum in &strata_indices {
                if stratum != u32::MAX { stratum_counts[stratum as usize] += 1; }
            }
            let total_df: u32 = stratum_counts.iter().map(|&n| n.saturating_sub(1)).sum();
            Ok(total_df)
        }
    }
}

// ============================================================================
// Median Variance using Woodruff Method
// ============================================================================

/// Compute variance of the median using the Woodruff (1952) method.
///
/// The Woodruff method computes variance for quantiles by:
/// 1. Computing the variance of the proportion P(Y <= q) using Taylor linearization
/// 2. Converting this to variance on the quantile scale using the inverse CDF
///
/// This matches R's svyquantile with interval.type="Wald" (default).
///
/// Returns: (variance, se_proportion) where se_proportion is needed for CI calculation
pub fn median_variance_woodruff(
    y: &Float64Chunked,
    weights: &Float64Chunked,
    strata: Option<&StringChunked>,
    psu: Option<&StringChunked>,
    ssu: Option<&StringChunked>,
    fpc: Option<&Float64Chunked>,
    fpc_stage2: Option<&Float64Chunked>,
    singleton_method: Option<&str>,
    quantile_method: SvyQuantileMethod,
) -> PolarsResult<(f64, f64)> {
    let q = weighted_median(y, weights, quantile_method)?;

    if q.is_nan() {
        return Ok((f64::NAN, f64::NAN));
    }

    // Compute scores for the proportion P(Y <= q)
    let scores = scores_median(y, weights, quantile_method)?;

    // Compute variance of the proportion using Taylor linearization
    let var_p = taylor_variance(
        &scores,
        strata,
        psu,
        ssu,
        fpc,
        fpc_stage2,
        singleton_method,
    )?;

    let se_p = var_p.max(0.0).sqrt();

    // For the Woodruff method, we need to convert variance on probability scale
    // to variance on quantile scale. This is done during CI calculation using
    // the inverse CDF. For now, we return both the variance of the proportion
    // and se_p so the caller can compute proper CIs.

    // The variance of the quantile itself requires density estimation at the quantile,
    // which is complex. Instead, we follow R's approach of computing CIs on the
    // probability scale and inverting them.

    // For a simple SE approximation, we can use the CI width approach:
    // SE_q ≈ (Q(p + z*se_p) - Q(p - z*se_p)) / (2*z)
    // But this requires the full data for inverse CDF lookup.

    // Return var_p and se_p - the caller can use these for CI calculation
    Ok((var_p, se_p))
}

/// Compute median variance for a domain using Woodruff method
pub fn median_variance_woodruff_domain(
    y: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
    strata: Option<&StringChunked>,
    psu: Option<&StringChunked>,
    ssu: Option<&StringChunked>,
    fpc: Option<&Float64Chunked>,
    fpc_stage2: Option<&Float64Chunked>,
    singleton_method: Option<&str>,
    quantile_method: SvyQuantileMethod,
) -> PolarsResult<(f64, f64)> {
    let q = weighted_median_domain(y, weights, domain_mask, quantile_method)?;

    if q.is_nan() {
        return Ok((f64::NAN, f64::NAN));
    }

    let scores = scores_median_domain(y, weights, domain_mask, quantile_method)?;

    let var_p = taylor_variance(
        &scores,
        strata,
        psu,
        ssu,
        fpc,
        fpc_stage2,
        singleton_method,
    )?;

    let se_p = var_p.max(0.0).sqrt();

    Ok((var_p, se_p))
}

// ============================================================================
// SRS Variance (Simple Random Sampling)
// ============================================================================

fn weighted_s2(y: &[f64], wn: &[f64]) -> f64 {
    let n = y.len() as f64;
    if n <= 1.0 { return f64::NAN; }
    let mu: f64 = y.iter().zip(wn.iter()).map(|(yi, wi)| wi * yi).sum();
    let ss: f64 = y.iter().zip(wn.iter()).map(|(yi, wi)| wi * (yi - mu).powi(2)).sum();
    (n / (n - 1.0)) * ss
}

pub fn srs_variance_mean(y: &Float64Chunked, weights: &Float64Chunked) -> PolarsResult<f64> {
    let n = y.len() as f64;
    if n < 2.0 { return Ok(f64::NAN); }
    let sum_w: f64 = weights.into_iter().filter_map(|v| v).sum();
    if sum_w <= 0.0 { return Ok(f64::NAN); }
    let wn: Vec<f64> = weights.into_iter().map(|v| v.unwrap_or(0.0) / sum_w).collect();
    let yv: Vec<f64> = y.into_iter().map(|v| v.unwrap_or(0.0)).collect();
    let s2_y = weighted_s2(&yv, &wn);
    Ok((s2_y / n) * (1.0 - (n / sum_w)))
}

pub fn srs_variance_mean_domain(y: &Float64Chunked, weights: &Float64Chunked, domain_mask: &BooleanChunked) -> PolarsResult<f64> {
    let mut yv = Vec::new();
    let mut wv = Vec::new();
    for ((yi, wi), mi) in y.into_iter().zip(weights.into_iter()).zip(domain_mask.into_iter()) {
        if let (Some(y_val), Some(w_val), Some(true)) = (yi, wi, mi) {
            yv.push(y_val);
            wv.push(w_val);
        }
    }
    let n = yv.len() as f64;
    if n < 2.0 { return Ok(f64::NAN); }
    let sum_w: f64 = wv.iter().sum();
    if sum_w <= 0.0 { return Ok(f64::NAN); }
    let wn: Vec<f64> = wv.iter().map(|w| w / sum_w).collect();
    let s2_y = weighted_s2(&yv, &wn);
    Ok((s2_y / n) * (1.0 - (n / sum_w)))
}

pub fn srs_variance_total(y: &Float64Chunked, weights: &Float64Chunked) -> PolarsResult<f64> {
    let n = y.len() as f64;
    if n < 2.0 { return Ok(f64::NAN); }
    let sum_w: f64 = weights.into_iter().filter_map(|v| v).sum();
    if sum_w <= 0.0 { return Ok(f64::NAN); }
    let wn: Vec<f64> = weights.into_iter().map(|v| v.unwrap_or(0.0) / sum_w).collect();
    let yv: Vec<f64> = y.into_iter().map(|v| v.unwrap_or(0.0)).collect();
    let s2_y = weighted_s2(&yv, &wn);
    Ok(((sum_w.powi(2) / n) * s2_y) * (1.0 - (n / sum_w)))
}

pub fn srs_variance_total_domain(y: &Float64Chunked, weights: &Float64Chunked, domain_mask: &BooleanChunked) -> PolarsResult<f64> {
    let mut yv = Vec::new();
    let mut wv = Vec::new();
    for ((yi, wi), mi) in y.into_iter().zip(weights.into_iter()).zip(domain_mask.into_iter()) {
        if let (Some(y_val), Some(w_val), Some(true)) = (yi, wi, mi) {
            yv.push(y_val);
            wv.push(w_val);
        }
    }
    let n = yv.len() as f64;
    if n < 2.0 { return Ok(f64::NAN); }
    let sum_w: f64 = wv.iter().sum();
    if sum_w <= 0.0 { return Ok(f64::NAN); }
    let wn: Vec<f64> = wv.iter().map(|w| w / sum_w).collect();
    let s2_y = weighted_s2(&yv, &wn);
    Ok(((sum_w.powi(2) / n) * s2_y) * (1.0 - (n / sum_w)))
}

pub fn srs_variance_ratio(y: &Float64Chunked, x: &Float64Chunked, weights: &Float64Chunked) -> PolarsResult<f64> {
    let n = y.len() as f64;
    if n < 2.0 { return Ok(f64::NAN); }
    let sum_w: f64 = weights.into_iter().filter_map(|v| v).sum();
    let wn: Vec<f64> = weights.into_iter().map(|v| v.unwrap_or(0.0) / sum_w).collect();
    let yv: Vec<f64> = y.into_iter().map(|v| v.unwrap_or(0.0)).collect();
    let xv: Vec<f64> = x.into_iter().map(|v| v.unwrap_or(0.0)).collect();
    let ybar: f64 = yv.iter().zip(wn.iter()).map(|(yi, wi)| wi * yi).sum();
    let xbar: f64 = xv.iter().zip(wn.iter()).map(|(xi, wi)| wi * xi).sum();
    if xbar == 0.0 { return Ok(f64::NAN); }
    let rhat = ybar / xbar;
    let ev: Vec<f64> = yv.iter().zip(xv.iter()).map(|(yi, xi)| yi - rhat * xi).collect();
    let s2_e = weighted_s2(&ev, &wn);
    Ok((s2_e / (n * xbar.powi(2))) * (1.0 - (n / sum_w)))
}

pub fn srs_variance_ratio_domain(y: &Float64Chunked, x: &Float64Chunked, weights: &Float64Chunked, domain_mask: &BooleanChunked) -> PolarsResult<f64> {
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
    if n < 2.0 { return Ok(f64::NAN); }
    let sum_w: f64 = wv.iter().sum();
    let wn: Vec<f64> = wv.iter().map(|w| w / sum_w).collect();
    let ybar: f64 = yv.iter().zip(wn.iter()).map(|(yi, wi)| wi * yi).sum();
    let xbar: f64 = xv.iter().zip(wn.iter()).map(|(xi, wi)| wi * xi).sum();
    if xbar == 0.0 { return Ok(f64::NAN); }
    let rhat = ybar / xbar;
    let ev: Vec<f64> = yv.iter().zip(xv.iter()).map(|(yi, xi)| yi - rhat * xi).collect();
    let s2_e = weighted_s2(&ev, &wn);
    Ok((s2_e / (n * xbar.powi(2))) * (1.0 - (n / sum_w)))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weighted_quantile_uniform_weights() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let w = vec![1.0, 1.0, 1.0, 1.0, 1.0];

        // CDF = [0.2, 0.4, 0.6, 0.8, 1.0]
        let total: f64 = w.iter().sum();
        let cdf: Vec<f64> = w.iter()
            .scan(0.0, |acc, &x| { *acc += x; Some(*acc / total) })
            .collect();

        let median_lower = weighted_quantile(&y, &cdf, 0.5, SvyQuantileMethod::Lower);
        let median_higher = weighted_quantile(&y, &cdf, 0.5, SvyQuantileMethod::Higher);
        let median_linear = weighted_quantile(&y, &cdf, 0.5, SvyQuantileMethod::Linear);

        assert!(median_lower <= 3.0);
        assert!(median_higher >= 3.0);
        // Linear should interpolate
        assert!(median_linear >= 2.0 && median_linear <= 3.0);
    }

    #[test]
    fn test_weighted_quantile_nonuniform_weights() {
        let y = vec![1.0, 2.0, 3.0];
        let w = vec![1.0, 2.0, 1.0]; // value 2 has more weight

        // CDF = [0.25, 0.75, 1.0]
        let total: f64 = w.iter().sum();
        let cdf: Vec<f64> = w.iter()
            .scan(0.0, |acc, &x| { *acc += x; Some(*acc / total) })
            .collect();

        let median_linear = weighted_quantile(&y, &cdf, 0.5, SvyQuantileMethod::Linear);

        // p=0.5 falls between 0.25 and 0.75
        // Should interpolate between y[0]=1 and y[1]=2
        assert!(median_linear >= 1.0 && median_linear <= 2.0);
    }

    #[test]
    fn test_quantile_method_from_str() {
        assert_eq!(SvyQuantileMethod::from_str("lower"), SvyQuantileMethod::Lower);
        assert_eq!(SvyQuantileMethod::from_str("HIGHER"), SvyQuantileMethod::Higher);
        assert_eq!(SvyQuantileMethod::from_str("Linear"), SvyQuantileMethod::Linear);
        assert_eq!(SvyQuantileMethod::from_str("unknown"), SvyQuantileMethod::Higher); // default
    }
}
