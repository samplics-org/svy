// src/estimation/taylor.rs

use polars::prelude::*;
use std::collections::HashMap;

// ============================================================================
// Enums & Config
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SingletonMethod {
    None,   // Default: Treat n=1 as 0 variance contribution
    Center, // Grand mean centering: (z_i - z_bar)^2
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
    let sum_wy: f64 = y
        .iter()
        .zip(weights.iter())
        .filter_map(|(yi, wi)| Some(yi? * wi?))
        .sum();
    let sum_w: f64 = y
        .iter()
        .zip(weights.iter())
        .filter_map(|(yi, wi)| {
            yi?;
            wi
        })
        .sum();

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
    let sum_wy: f64 = y
        .iter()
        .zip(weights.iter())
        .zip(domain_mask.iter())
        .filter_map(|((yi, wi), m)| if m? { Some(yi? * wi?) } else { None })
        .sum();
    let sum_w: f64 = weights
        .iter()
        .zip(domain_mask.iter())
        .filter_map(|(w, m)| if m? { w } else { None })
        .sum();

    if sum_w == 0.0 {
        return Err(PolarsError::ComputeError(
            "Sum of weights is zero in domain".into(),
        ));
    }
    Ok(sum_wy / sum_w)
}

pub fn point_estimate_total(y: &Float64Chunked, weights: &Float64Chunked) -> PolarsResult<f64> {
    Ok(y.iter()
        .zip(weights.iter())
        .filter_map(|(yi, wi)| Some(yi? * wi?))
        .sum())
}

pub fn point_estimate_total_domain(
    y: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
) -> PolarsResult<f64> {
    Ok(y.iter()
        .zip(weights.iter())
        .zip(domain_mask.iter())
        .filter_map(|((yi, wi), m)| if m? { Some(yi? * wi?) } else { None })
        .sum())
}

pub fn point_estimate_ratio(
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked,
) -> PolarsResult<f64> {
    let sum_wy: f64 = y
        .iter()
        .zip(weights.iter())
        .filter_map(|(yi, wi)| Some(yi? * wi?))
        .sum();
    let sum_wx: f64 = x
        .iter()
        .zip(weights.iter())
        .filter_map(|(xi, wi)| Some(xi? * wi?))
        .sum();

    if sum_wx == 0.0 {
        return Err(PolarsError::ComputeError(
            "Weighted sum of denominator (x) is zero".into(),
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
    let sum_wy: f64 = y
        .iter()
        .zip(weights.iter())
        .zip(domain_mask.iter())
        .filter_map(|((yi, wi), m)| if m? { Some(yi? * wi?) } else { None })
        .sum();
    let sum_wx: f64 = x
        .iter()
        .zip(weights.iter())
        .zip(domain_mask.iter())
        .filter_map(|((xi, wi), m)| if m? { Some(xi? * wi?) } else { None })
        .sum();

    if sum_wx == 0.0 {
        return Err(PolarsError::ComputeError(
            "Weighted sum of denominator (x) is zero in domain".into(),
        ));
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
pub fn weighted_quantile(y_sorted: &[f64], cdf: &[f64], p: f64, method: SvyQuantileMethod) -> f64 {
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
            if dl <= dr {
                y_sorted[left]
            } else {
                y_sorted[right]
            }
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
    let mut pairs: Vec<(f64, f64)> = y
        .iter()
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

    let (y_sorted, w_sorted): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();

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
    let mut pairs: Vec<(f64, f64)> = y
        .iter()
        .zip(weights.iter())
        .filter_map(|(yi, wi)| Some((yi?, wi?)))
        .collect();

    if pairs.is_empty() {
        return Ok(f64::NAN);
    }

    // Sort by y value
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // unzip consumes pairs in one allocation instead of two map+collect passes
    let (y_sorted, w_sorted): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();

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
    // Single pass: accumulate sum_wy, sum_w and collect (y,w) pairs simultaneously.
    // Avoids the 3 separate iterations that the previous point_estimate_mean + sum_w + score
    // build pattern required.
    let n = y.len();
    let mut sum_wy = 0.0f64;
    let mut sum_w  = 0.0f64;
    let mut pairs: Vec<Option<(f64, f64)>> = Vec::with_capacity(n);

    for (yi, wi) in y.iter().zip(weights.iter()) {
        match (yi, wi) {
            (Some(yv), Some(wv)) => {
                sum_wy += yv * wv;
                sum_w  += wv;
                pairs.push(Some((yv, wv)));
            }
            _ => pairs.push(None),
        }
    }

    if sum_w == 0.0 {
        return Err(PolarsError::ComputeError("Sum of weights is zero".into()));
    }
    let est = sum_wy / sum_w;

    let scores: Vec<Option<f64>> = pairs
        .iter()
        .map(|p| p.map(|(yv, wv)| (wv / sum_w) * (yv - est)))
        .collect();

    Ok(Float64Chunked::from_slice_options("scores".into(), &scores))
}

pub fn scores_mean_domain(
    y: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
) -> PolarsResult<Float64Chunked> {
    // Single pass: accumulate sum_wy / sum_w and collect triples.
    let n = y.len();
    let mut sum_wy = 0.0f64;
    let mut sum_w  = 0.0f64;
    let mut triples: Vec<Option<(f64, f64)>> = Vec::with_capacity(n); // (yv,wv) only in-domain

    for ((yi, wi), mi) in y.iter().zip(weights.iter()).zip(domain_mask.iter()) {
        match (yi, wi, mi) {
            (Some(yv), Some(wv), Some(true)) => {
                sum_wy += yv * wv;
                sum_w  += wv;
                triples.push(Some((yv, wv)));
            }
            _ => triples.push(None),
        }
    }

    if sum_w == 0.0 {
        let zeros = vec![Some(0.0); n];
        return Ok(Float64Chunked::from_slice_options("scores".into(), &zeros));
    }
    let est = sum_wy / sum_w;

    let scores: Vec<Option<f64>> = triples
        .iter()
        .map(|p| Some(match p {
            Some((yv, wv)) => (wv / sum_w) * (yv - est),
            None => 0.0,
        }))
        .collect();

    Ok(Float64Chunked::from_slice_options("scores".into(), &scores))
}

pub fn scores_total(y: &Float64Chunked, weights: &Float64Chunked) -> PolarsResult<Float64Chunked> {
    let scores: Vec<Option<f64>> = y
        .iter()
        .zip(weights.iter())
        .map(|(yi, wi)| match (yi, wi) {
            (Some(y_val), Some(w_val)) => Some(w_val * y_val),
            _ => None,
        })
        .collect();
    Ok(Float64Chunked::from_slice_options("scores".into(), &scores))
}

pub fn scores_total_domain(
    y: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
) -> PolarsResult<Float64Chunked> {
    let scores: Vec<Option<f64>> = y
        .iter()
        .zip(weights.iter())
        .zip(domain_mask.iter())
        .map(|((yi, wi), m)| match (yi, wi, m) {
            (Some(y_val), Some(w_val), Some(true)) => Some(w_val * y_val),
            _ => Some(0.0),
        })
        .collect();
    Ok(Float64Chunked::from_slice_options("scores".into(), &scores))
}

pub fn scores_ratio(
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked,
) -> PolarsResult<Float64Chunked> {
    // Single pass: accumulate sum_wy, sum_wx and collect (y,x,w) triples.
    let n = y.len();
    let mut sum_wy = 0.0f64;
    let mut sum_wx = 0.0f64;
    let mut triples: Vec<Option<(f64, f64, f64)>> = Vec::with_capacity(n);

    for ((yi, xi), wi) in y.iter().zip(x.iter()).zip(weights.iter()) {
        match (yi, xi, wi) {
            (Some(yv), Some(xv), Some(wv)) => {
                sum_wy += yv * wv;
                sum_wx += xv * wv;
                triples.push(Some((yv, xv, wv)));
            }
            _ => triples.push(None),
        }
    }

    if sum_wx == 0.0 {
        return Err(PolarsError::ComputeError(
            "Weighted sum of denominator (x) is zero".into(),
        ));
    }
    let r_hat = sum_wy / sum_wx;

    let scores: Vec<Option<f64>> = triples
        .iter()
        .map(|t| t.map(|(yv, xv, wv)| (wv / sum_wx) * (yv - r_hat * xv)))
        .collect();
    Ok(Float64Chunked::from_slice_options("scores".into(), &scores))
}

pub fn scores_ratio_domain(
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
) -> PolarsResult<Float64Chunked> {
    // Single pass: accumulate sum_wy, sum_wx (in-domain) and collect quads.
    let n = y.len();
    let mut sum_wy = 0.0f64;
    let mut sum_wx = 0.0f64;
    let mut quads: Vec<Option<(f64, f64, f64)>> = Vec::with_capacity(n); // (yv,xv,wv)

    for (((yi, xi), wi), mi) in y.iter().zip(x.iter()).zip(weights.iter()).zip(domain_mask.iter()) {
        match (yi, xi, wi, mi) {
            (Some(yv), Some(xv), Some(wv), Some(true)) => {
                sum_wy += yv * wv;
                sum_wx += xv * wv;
                quads.push(Some((yv, xv, wv)));
            }
            _ => quads.push(None),
        }
    }

    if sum_wx == 0.0 {
        let zeros = vec![Some(0.0); n];
        return Ok(Float64Chunked::from_slice_options("scores".into(), &zeros));
    }
    let r_hat = sum_wy / sum_wx;

    let scores: Vec<Option<f64>> = quads
        .iter()
        .map(|q| Some(match q {
            Some((yv, xv, wv)) => (wv / sum_wx) * (yv - r_hat * xv),
            None => 0.0,
        }))
        .collect();
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

    let scores: Vec<Option<f64>> = y
        .iter()
        .zip(weights.iter())
        .map(|(yi, wi)| match (yi, wi) {
            (Some(y_val), Some(w_val)) => {
                // u = I(y > q) - (1 - p) = I(y > q) - 0.5
                let u = if y_val > q { 1.0 } else { 0.0 } - 0.5;
                Some((w_val / sum_w) * u)
            }
            _ => None,
        })
        .collect();

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

    let sum_w: f64 = weights
        .iter()
        .zip(domain_mask.iter())
        .filter_map(|(w, m)| if m? { w } else { None })
        .sum();

    if sum_w <= 0.0 {
        let zeros = vec![Some(0.0); y.len()];
        return Ok(Float64Chunked::from_slice_options("scores".into(), &zeros));
    }

    let scores: Vec<Option<f64>> = y
        .iter()
        .zip(weights.iter())
        .zip(domain_mask.iter())
        .map(|((yi, wi), m)| match (yi, wi, m) {
            (Some(y_val), Some(w_val), Some(true)) => {
                let u = if y_val > q { 1.0 } else { 0.0 } - 0.5;
                Some((w_val / sum_w) * u)
            }
            _ => Some(0.0), // Zero score outside domain
        })
        .collect();

    Ok(Float64Chunked::from_slice_options("scores".into(), &scores))
}

// ============================================================================
// Variance Helpers (Indexing & Math)
// ============================================================================

fn index_categorical(col: &StringChunked) -> (Vec<u32>, u32) {
    let mut map: HashMap<&str, u32> = HashMap::new();
    let mut next_idx = 0u32;
    let indices: Vec<u32> = col
        .iter()
        .map(|opt| match opt {
            Some(s) => *map.entry(s).or_insert_with(|| {
                let i = next_idx;
                next_idx += 1;
                i
            }),
            None => u32::MAX,
        })
        .collect();
    (indices, next_idx)
}

fn reindex_within_subset(raw: &[u32]) -> (Vec<u32>, u32) {
    let mut map: HashMap<u32, u32> = HashMap::new();
    let mut next_idx = 0u32;
    let indices: Vec<u32> = raw
        .iter()
        .map(|&val| {
            if val == u32::MAX {
                u32::MAX
            } else {
                *map.entry(val).or_insert_with(|| {
                    let i = next_idx;
                    next_idx += 1;
                    i
                })
            }
        })
        .collect();
    (indices, next_idx)
}

fn build_stratum_psu_map(
    strata_indices: &[u32],
    n_strata: u32,
    psu_indices: &[u32],
) -> (Vec<Vec<u32>>, Vec<u32>) {
    use std::collections::HashSet;
    let mut stratum_psus: Vec<HashSet<u32>> = vec![HashSet::new(); n_strata as usize];
    for (&stratum, &psu) in strata_indices.iter().zip(psu_indices.iter()) {
        if stratum != u32::MAX && psu != u32::MAX {
            stratum_psus[stratum as usize].insert(psu);
        }
    }
    let psu_per_stratum: Vec<Vec<u32>> = stratum_psus
        .iter()
        .map(|s| s.iter().copied().collect())
        .collect();
    let n_psus_per_stratum: Vec<u32> = psu_per_stratum.iter().map(|v| v.len() as u32).collect();
    (psu_per_stratum, n_psus_per_stratum)
}

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
            let sum_sq_diff: f64 = psu_totals.iter().map(|&t| (t - psu_mean).powi(2)).sum();
            (n_psus as f64 / (n_psus as f64 - 1.0)) * sum_sq_diff
        }
        None => {
            if n <= 1 {
                return 0.0;
            }
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
    fpc_per_stratum: Option<&[f64]>,
    singleton_method: SingletonMethod,
) -> f64 {
    let n = scores.len();
    if n == 0 || n_strata == 0 {
        return 0.0;
    }

    match (psu_indices, psu_per_stratum, n_psus_per_stratum) {
        (Some(psu_idx), Some(psu_map), Some(n_psus)) => {
            let max_psu = psu_idx
                .iter()
                .filter(|&&p| p != u32::MAX)
                .max()
                .copied()
                .unwrap_or(0);
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
                if total_count > 0 {
                    grand_mean = total_score / (total_count as f64);
                }
            }

            let mut total_var = 0.0;
            for h in 0..n_strata as usize {
                let n_psus_h = n_psus[h];
                if n_psus_h == 0 {
                    continue;
                }

                let fpc_h = fpc_per_stratum.map(|f| f[h]).unwrap_or(1.0);

                if n_psus_h == 1 {
                    if singleton_method == SingletonMethod::Center {
                        if let Some(&p) = psu_map[h].first() {
                            total_var += fpc_h * (psu_totals[p as usize] - grand_mean).powi(2);
                        }
                    }
                    continue;
                }

                let psu_indices_h = &psu_map[h];
                // Compute mean directly from global psu_totals — no per-stratum Vec alloc
                let psu_mean_h: f64 = psu_indices_h.iter()
                    .map(|&p| psu_totals[p as usize])
                    .sum::<f64>() / (n_psus_h as f64);
                let sum_sq_diff: f64 = psu_indices_h.iter()
                    .map(|&p| (psu_totals[p as usize] - psu_mean_h).powi(2))
                    .sum();
                total_var += fpc_h * (n_psus_h as f64 / (n_psus_h as f64 - 1.0)) * sum_sq_diff;
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
                if n_h == 0 {
                    continue;
                }

                let fpc_h = fpc_per_stratum.map(|f| f[h]).unwrap_or(1.0);

                if n_h == 1 {
                    if singleton_method == SingletonMethod::Center {
                        total_var += fpc_h * (stratum_sums[h] - grand_mean).powi(2);
                    }
                    continue;
                }
                let var_h = (stratum_sum_sq[h] - stratum_sums[h].powi(2) / (n_h as f64))
                    / (n_h as f64 - 1.0)
                    * (n_h as f64);
                total_var += fpc_h * var_h;
            }
            total_var
        }
    }
}

fn compute_stage2_variance(
    scores: &[f64],
    psu_indices: &[u32],
    ssu_indices: &[u32],
    _strata_indices: Option<&[u32]>,
    fpc_per_stratum: Option<&[f64]>,
    fpc_ssu_arr: Option<&[f64]>,
    strata_for_psu: Option<&[u32]>,
) -> f64 {
    let n = scores.len();
    if n == 0 {
        return 0.0;
    }

    let max_psu = psu_indices
        .iter()
        .filter(|&&p| p != u32::MAX)
        .max()
        .copied()
        .unwrap_or(0);
    let n_psus = (max_psu + 1) as usize;
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

        // Get the FPC for this PSU's stratum: (N_h - n_h) / N_h
        // stage1_sampling_fraction = n_h / N_h = 1 - fpc_h
        let fpc_psu_val = match (strata_for_psu, fpc_per_stratum) {
            (Some(s_for_p), Some(fpc_s)) => {
                let stratum_idx = s_for_p[psu] as usize;
                if stratum_idx < fpc_s.len() {
                    fpc_s[stratum_idx]
                } else {
                    1.0
                }
            }
            _ => fpc_per_stratum
                .and_then(|f| f.first().copied())
                .unwrap_or(1.0),
        };

        let stage1_sampling_fraction = 1.0 - fpc_psu_val;
        if stage1_sampling_fraction <= 0.0 {
            continue;
        }

        // Get the unit-level FPC for this PSU
        // Use the first observation in this PSU to look up the per-row fpc_ssu value
        let fpc_ssu_val = fpc_ssu_arr.map(|f| f[obs_indices[0]]).unwrap_or(1.0);

        let psu_scores: Vec<f64> = obs_indices.iter().map(|&i| scores[i]).collect();
        let psu_ssu_raw: Vec<u32> = obs_indices.iter().map(|&i| ssu_indices[i]).collect();
        let (psu_ssu_indices, n_ssus) = reindex_within_subset(&psu_ssu_raw);

        if n_ssus <= 1 {
            continue;
        }

        let mut ssu_totals = vec![0.0; n_ssus as usize];
        for (score, &ssu) in psu_scores.iter().zip(psu_ssu_indices.iter()) {
            if ssu != u32::MAX {
                ssu_totals[ssu as usize] += score;
            }
        }
        let ssu_mean = ssu_totals.iter().sum::<f64>() / (n_ssus as f64);
        let sum_sq_diff: f64 = ssu_totals.iter().map(|&t| (t - ssu_mean).powi(2)).sum();
        let var_within_psu = (n_ssus as f64 / (n_ssus as f64 - 1.0)) * sum_sq_diff;

        total_stage2_var += stage1_sampling_fraction * fpc_ssu_val * var_within_psu;
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
    fpc_ssu: Option<&Float64Chunked>,
    singleton_method: Option<&str>,
) -> PolarsResult<f64> {
    let n = scores.len();
    if n == 0 {
        return Ok(0.0);
    }

    let sm_enum = match singleton_method {
        Some(s) if s.eq_ignore_ascii_case("center") || s.eq_ignore_ascii_case("adjust") => {
            SingletonMethod::Center
        }
        _ => SingletonMethod::None,
    };

    let scores_arr: Vec<f64> = scores.iter().map(|s| s.unwrap_or(0.0)).collect();
    let fpc_arr: Option<Vec<f64>> = fpc.map(|f| f.iter().map(|v| v.unwrap_or(1.0)).collect());
    let fpc_ssu_arr: Option<Vec<f64>> =
        fpc_ssu.map(|f| f.iter().map(|v| v.unwrap_or(1.0)).collect());

    // --- STAGE 1 VARIANCE ---
    let (
        var_stage1,
        psu_indices_opt,
        strata_indices_opt,
        _n_strata_val,
        fpc_per_stratum,
        psu_stratum_map,
    ) = if strata.is_none() {
        // Unstratified: single FPC value (take first element or 1.0)
        let fpc_val = fpc_arr
            .as_ref()
            .and_then(|f| f.first().copied())
            .unwrap_or(1.0);

        let (psu_indices, n_psus) = match psu {
            Some(psu_col) => {
                let (idx, n) = index_categorical(psu_col);
                (Some(idx), n)
            }
            None => (None, 0),
        };
        let var = variance_unstratified_optimized(&scores_arr, psu_indices.as_deref(), n_psus);
        (fpc_val * var, psu_indices, None, 0u32, None, None)
    } else {
        let strata_col = strata.unwrap();
        let (strata_indices, n_strata) = index_categorical(strata_col);

        // Build per-stratum FPC: for each stratum index h, take the FPC value
        // from the first row belonging to that stratum.
        let fpc_by_stratum: Option<Vec<f64>> = fpc_arr.as_ref().map(|fpc_vals| {
            let mut per_stratum = vec![1.0; n_strata as usize];
            let mut seen = vec![false; n_strata as usize];
            for (i, &s) in strata_indices.iter().enumerate() {
                if s != u32::MAX && !seen[s as usize] {
                    per_stratum[s as usize] = fpc_vals[i];
                    seen[s as usize] = true;
                }
            }
            per_stratum
        });

        match psu {
            Some(psu_col) => {
                let (psu_indices, _) = index_categorical(psu_col);
                let (psu_per_stratum, n_psus_per_stratum) =
                    build_stratum_psu_map(&strata_indices, n_strata, &psu_indices);

                // Build PSU -> stratum mapping for stage 2
                let max_psu = psu_indices
                    .iter()
                    .filter(|&&p| p != u32::MAX)
                    .max()
                    .copied()
                    .unwrap_or(0);
                let mut psu_to_stratum = vec![0u32; (max_psu + 1) as usize];
                for (&psu_idx, &str_idx) in psu_indices.iter().zip(strata_indices.iter()) {
                    if psu_idx != u32::MAX && str_idx != u32::MAX {
                        psu_to_stratum[psu_idx as usize] = str_idx;
                    }
                }

                let var = variance_stratified_optimized(
                    &scores_arr,
                    &strata_indices,
                    n_strata,
                    Some(&psu_indices),
                    Some(&psu_per_stratum),
                    Some(&n_psus_per_stratum),
                    fpc_by_stratum.as_deref(),
                    sm_enum,
                );
                (
                    var,
                    Some(psu_indices),
                    Some(strata_indices),
                    n_strata,
                    fpc_by_stratum,
                    Some(psu_to_stratum),
                )
            }
            None => {
                let var = variance_stratified_optimized(
                    &scores_arr,
                    &strata_indices,
                    n_strata,
                    None,
                    None,
                    None,
                    fpc_by_stratum.as_deref(),
                    sm_enum,
                );
                (
                    var,
                    None,
                    Some(strata_indices),
                    n_strata,
                    fpc_by_stratum,
                    None,
                )
            }
        }
    };

    // --- STAGE 2 VARIANCE ---
    if ssu.is_none() || psu.is_none() {
        return Ok(var_stage1);
    }

    let ssu_col = ssu.unwrap();
    let (ssu_indices, _) = index_categorical(ssu_col);

    let psu_indices = match psu_indices_opt {
        Some(idx) => idx,
        None => {
            if let Some(p) = psu {
                index_categorical(p).0
            } else {
                (0..n as u32).collect()
            }
        }
    };

    let var_stage2 = compute_stage2_variance(
        &scores_arr,
        &psu_indices,
        &ssu_indices,
        strata_indices_opt.as_deref(),
        fpc_per_stratum.as_deref(),
        fpc_ssu_arr.as_deref(),
        psu_stratum_map.as_deref(),
    );

    Ok(var_stage1 + var_stage2)
}

pub fn degrees_of_freedom(
    weights: &Float64Chunked,
    strata: Option<&StringChunked>,
    psu: Option<&StringChunked>,
) -> PolarsResult<u32> {
    let n = weights.len();
    if n == 0 {
        return Ok(0);
    }

    // Build mask of active (positive weight) observations.
    // This ensures domain estimation (where non-domain obs have w=0)
    // gives the correct df, matching R's degf() on subsetted designs.
    let active: Vec<bool> = weights
        .iter()
        .map(|w| w.map_or(false, |v| v > 0.0))
        .collect();

    if strata.is_none() {
        if psu.is_none() {
            // No strata, no PSU: each obs is its own PSU, df = n_active - 1
            let n_active = active.iter().filter(|&&a| a).count();
            return Ok(n_active.saturating_sub(1) as u32);
        }
        // No strata, with PSU: df = n_unique_active_psus - 1
        let psu_col = psu.unwrap();
        let mut seen = std::collections::HashSet::new();
        for (opt_p, &act) in psu_col.iter().zip(active.iter()) {
            if act {
                if let Some(p) = opt_p {
                    seen.insert(p.to_string());
                }
            }
        }
        return Ok(seen.len().saturating_sub(1) as u32);
    }

    let strata_col = strata.unwrap();

    match psu {
        Some(psu_col) => {
            // Stratified + clustered: df = sum_h(n_active_psus_h - 1)
            // Count unique active PSUs per stratum
            let mut stratum_psus: HashMap<String, std::collections::HashSet<String>> =
                HashMap::new();
            for ((opt_s, opt_p), &act) in strata_col.iter().zip(psu_col.iter()).zip(active.iter()) {
                if act {
                    if let (Some(s), Some(p)) = (opt_s, opt_p) {
                        stratum_psus
                            .entry(s.to_string())
                            .or_default()
                            .insert(p.to_string());
                    }
                }
            }
            let total_df: u32 = stratum_psus
                .values()
                .map(|psus| psus.len().saturating_sub(1) as u32)
                .sum();
            Ok(total_df)
        }
        None => {
            // Stratified, no PSU: df = sum_h(n_active_h - 1)
            let mut stratum_counts: HashMap<String, u32> = HashMap::new();
            for (opt_s, &act) in strata_col.iter().zip(active.iter()) {
                if act {
                    if let Some(s) = opt_s {
                        *stratum_counts.entry(s.to_string()).or_insert(0) += 1;
                    }
                }
            }
            let total_df: u32 = stratum_counts
                .values()
                .map(|&count| count.saturating_sub(1))
                .sum();
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
    fpc_ssu: Option<&Float64Chunked>,
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
    let var_p = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;

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
    fpc_ssu: Option<&Float64Chunked>,
    singleton_method: Option<&str>,
    quantile_method: SvyQuantileMethod,
) -> PolarsResult<(f64, f64)> {
    let q = weighted_median_domain(y, weights, domain_mask, quantile_method)?;

    if q.is_nan() {
        return Ok((f64::NAN, f64::NAN));
    }

    let scores = scores_median_domain(y, weights, domain_mask, quantile_method)?;

    let var_p = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;

    let se_p = var_p.max(0.0).sqrt();

    Ok((var_p, se_p))
}

// ============================================================================
// SRS Variance (Simple Random Sampling)
// ============================================================================

fn weighted_s2(y: &[f64], wn: &[f64]) -> f64 {
    let n = y.len() as f64;
    if n <= 1.0 {
        return f64::NAN;
    }
    let mu: f64 = y.iter().zip(wn.iter()).map(|(yi, wi)| wi * yi).sum();
    let ss: f64 = y
        .iter()
        .zip(wn.iter())
        .map(|(yi, wi)| wi * (yi - mu).powi(2))
        .sum();
    (n / (n - 1.0)) * ss
}

pub fn srs_variance_mean(y: &Float64Chunked, weights: &Float64Chunked) -> PolarsResult<f64> {
    // Only use observations where both y and w are non-null
    let mut yv = Vec::new();
    let mut wv = Vec::new();
    for (yi, wi) in y.into_iter().zip(weights.into_iter()) {
        if let (Some(y_val), Some(w_val)) = (yi, wi) {
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
    Ok((s2_y / n) * (1.0 - (n / sum_w)))
}

pub fn srs_variance_mean_domain(
    y: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
) -> PolarsResult<f64> {
    let mut yv = Vec::new();
    let mut wv = Vec::new();
    for ((yi, wi), mi) in y
        .into_iter()
        .zip(weights.into_iter())
        .zip(domain_mask.into_iter())
    {
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
    Ok((s2_y / n) * (1.0 - (n / sum_w)))
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
    let wn: Vec<f64> = weights
        .into_iter()
        .map(|v| v.unwrap_or(0.0) / sum_w)
        .collect();
    let yv: Vec<f64> = y.into_iter().map(|v| v.unwrap_or(0.0)).collect();
    let s2_y = weighted_s2(&yv, &wn);
    Ok(((sum_w.powi(2) / n) * s2_y) * (1.0 - (n / sum_w)))
}

pub fn srs_variance_total_domain(
    y: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
) -> PolarsResult<f64> {
    let mut yv = Vec::new();
    let mut wv = Vec::new();
    for ((yi, wi), mi) in y
        .into_iter()
        .zip(weights.into_iter())
        .zip(domain_mask.into_iter())
    {
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
    Ok(((sum_w.powi(2) / n) * s2_y) * (1.0 - (n / sum_w)))
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
    let wn: Vec<f64> = weights
        .into_iter()
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
    let ev: Vec<f64> = yv
        .iter()
        .zip(xv.iter())
        .map(|(yi, xi)| yi - rhat * xi)
        .collect();
    let s2_e = weighted_s2(&ev, &wn);
    Ok((s2_e / (n * xbar.powi(2))) * (1.0 - (n / sum_w)))
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
    for (((yi, xi), wi), mi) in y
        .into_iter()
        .zip(x.into_iter())
        .zip(weights.into_iter())
        .zip(domain_mask.into_iter())
    {
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
    let wn: Vec<f64> = wv.iter().map(|w| w / sum_w).collect();
    let ybar: f64 = yv.iter().zip(wn.iter()).map(|(yi, wi)| wi * yi).sum();
    let xbar: f64 = xv.iter().zip(wn.iter()).map(|(xi, wi)| wi * xi).sum();
    if xbar == 0.0 {
        return Ok(f64::NAN);
    }
    let rhat = ybar / xbar;
    let ev: Vec<f64> = yv
        .iter()
        .zip(xv.iter())
        .map(|(yi, xi)| yi - rhat * xi)
        .collect();
    let s2_e = weighted_s2(&ev, &wn);
    Ok((s2_e / (n * xbar.powi(2))) * (1.0 - (n / sum_w)))
}

// ============================================================================
// Multivariate Taylor Variance (k score columns -> k x k covariance matrix)
// ============================================================================
/// Compute the full k×k covariance matrix from k score columns.
///
/// This is the multivariate extension of `taylor_variance`. For each stratum,
/// it sums scores by PSU to get k-dimensional PSU totals, then computes
/// the between-PSU covariance matrix using the outer product formula:
///   Cov_h = fpc_h * (m_h / (m_h - 1)) * Σ_a (T_ha - T̄_h)(T_ha - T̄_h)'
///
/// This matches R's `vcov(svymean(...))` which computes the full covariance
/// in a single pass over PSU totals.
pub fn taylor_variance_matrix(
    score_columns: &[Float64Chunked], // k score columns, each length n
    strata: Option<&StringChunked>,
    psu: Option<&StringChunked>,
    fpc: Option<&Float64Chunked>,
    singleton_method: Option<&str>,
) -> PolarsResult<Vec<Vec<f64>>> {
    let k = score_columns.len();
    if k == 0 {
        return Ok(vec![]);
    }
    let n = score_columns[0].len();
    if n == 0 {
        return Ok(vec![vec![0.0; k]; k]);
    }

    let sm_enum = match singleton_method {
        Some(s) if s.eq_ignore_ascii_case("center") || s.eq_ignore_ascii_case("adjust") => {
            SingletonMethod::Center
        }
        _ => SingletonMethod::None,
    };

    // Extract all scores into a Vec<Vec<f64>> (k columns of n values)
    let scores: Vec<Vec<f64>> = score_columns
        .iter()
        .map(|col| col.iter().map(|s| s.unwrap_or(0.0)).collect())
        .collect();

    let fpc_arr: Option<Vec<f64>> = fpc.map(|f| f.iter().map(|v| v.unwrap_or(1.0)).collect());

    let mut cov = vec![vec![0.0; k]; k];

    if strata.is_none() {
        // ── Unstratified ──
        let fpc_val = fpc_arr
            .as_ref()
            .and_then(|f| f.first().copied())
            .unwrap_or(1.0);

        let (psu_indices, n_psus) = match psu {
            Some(psu_col) => index_categorical(psu_col),
            None => {
                // No PSU: each observation is its own PSU
                ((0..n as u32).collect(), n as u32)
            }
        };

        if n_psus <= 1 {
            return Ok(cov);
        }

        // Sum scores by PSU: n_psus x k
        let mut psu_totals = vec![vec![0.0; k]; n_psus as usize];
        for i in 0..n {
            let p = psu_indices[i];
            if p != u32::MAX {
                for j in 0..k {
                    psu_totals[p as usize][j] += scores[j][i];
                }
            }
        }

        // PSU means
        let mf = n_psus as f64;
        let mut psu_mean = vec![0.0; k];
        for t in &psu_totals {
            for j in 0..k {
                psu_mean[j] += t[j];
            }
        }
        for j in 0..k {
            psu_mean[j] /= mf;
        }

        // Covariance: fpc * (m / (m-1)) * Σ (T_a - T̄)(T_a - T̄)'
        let scale = fpc_val * mf / (mf - 1.0);
        for t in &psu_totals {
            for j in 0..k {
                let dj = t[j] - psu_mean[j];
                for l in j..k {
                    let dl = t[l] - psu_mean[l];
                    let v = scale * dj * dl;
                    cov[j][l] += v;
                    if l != j {
                        cov[l][j] += v;
                    }
                }
            }
        }
    } else {
        // ── Stratified ──
        let strata_col = strata.unwrap();
        let (strata_indices, n_strata) = index_categorical(strata_col);

        let (psu_indices, _n_psus_global) = match psu {
            Some(psu_col) => index_categorical(psu_col),
            None => ((0..n as u32).collect(), n as u32),
        };

        let (psu_per_stratum, n_psus_per_stratum) =
            build_stratum_psu_map(&strata_indices, n_strata, &psu_indices);

        // Per-stratum FPC
        let fpc_per_stratum: Vec<f64> = if let Some(ref fpc_vals) = fpc_arr {
            let mut per_s = vec![1.0; n_strata as usize];
            let mut seen = vec![false; n_strata as usize];
            for (i, &s) in strata_indices.iter().enumerate() {
                if s != u32::MAX && !seen[s as usize] {
                    per_s[s as usize] = fpc_vals[i];
                    seen[s as usize] = true;
                }
            }
            per_s
        } else {
            vec![1.0; n_strata as usize]
        };

        // Build global PSU totals (k-dimensional)
        let max_psu = psu_indices
            .iter()
            .filter(|&&p| p != u32::MAX)
            .max()
            .copied()
            .unwrap_or(0);
        let mut psu_totals = vec![vec![0.0; k]; (max_psu + 1) as usize];

        for i in 0..n {
            let p = psu_indices[i];
            if p != u32::MAX {
                for j in 0..k {
                    psu_totals[p as usize][j] += scores[j][i];
                }
            }
        }

        // Grand mean for singleton handling
        let mut grand_mean = vec![0.0; k];
        if sm_enum == SingletonMethod::Center {
            let mut total_count = 0usize;
            for h in 0..n_strata as usize {
                for &p in &psu_per_stratum[h] {
                    for j in 0..k {
                        grand_mean[j] += psu_totals[p as usize][j];
                    }
                    total_count += 1;
                }
            }
            if total_count > 0 {
                for j in 0..k {
                    grand_mean[j] /= total_count as f64;
                }
            }
        }

        // Per-stratum covariance accumulation
        for h in 0..n_strata as usize {
            let m_h = n_psus_per_stratum[h];
            if m_h == 0 {
                continue;
            }

            let fpc_h = fpc_per_stratum[h];

            if m_h == 1 {
                if sm_enum == SingletonMethod::Center {
                    if let Some(&p) = psu_per_stratum[h].first() {
                        for j in 0..k {
                            let dj = psu_totals[p as usize][j] - grand_mean[j];
                            for l in j..k {
                                let dl = psu_totals[p as usize][l] - grand_mean[l];
                                let v = fpc_h * dj * dl;
                                cov[j][l] += v;
                                if l != j {
                                    cov[l][j] += v;
                                }
                            }
                        }
                    }
                }
                continue;
            }

            // PSU means within stratum
            let mf = m_h as f64;
            let mut psu_mean_h = vec![0.0; k];
            for &p in &psu_per_stratum[h] {
                for j in 0..k {
                    psu_mean_h[j] += psu_totals[p as usize][j];
                }
            }
            for j in 0..k {
                psu_mean_h[j] /= mf;
            }

            // Accumulate covariance
            let scale = fpc_h * mf / (mf - 1.0);
            // Use psu_totals directly indexed — no per-stratum copy
            for &p in &psu_per_stratum[h] {
                let pt = &psu_totals[p as usize];
                for j in 0..k {
                    let dj = pt[j] - psu_mean_h[j];
                    for l in j..k {
                        let v = scale * dj * (pt[l] - psu_mean_h[l]);
                        cov[j][l] += v;
                        if l != j {
                            cov[l][j] += v;
                        }
                    }
                }
            }
        }
    }

    Ok(cov)
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
        let cdf: Vec<f64> = w
            .iter()
            .scan(0.0, |acc, &x| {
                *acc += x;
                Some(*acc / total)
            })
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
        let cdf: Vec<f64> = w
            .iter()
            .scan(0.0, |acc, &x| {
                *acc += x;
                Some(*acc / total)
            })
            .collect();

        let median_linear = weighted_quantile(&y, &cdf, 0.5, SvyQuantileMethod::Linear);

        // p=0.5 falls between 0.25 and 0.75
        // Should interpolate between y[0]=1 and y[1]=2
        assert!(median_linear >= 1.0 && median_linear <= 2.0);
    }

    #[test]
    fn test_quantile_method_from_str() {
        assert_eq!(
            SvyQuantileMethod::from_str("lower"),
            SvyQuantileMethod::Lower
        );
        assert_eq!(
            SvyQuantileMethod::from_str("HIGHER"),
            SvyQuantileMethod::Higher
        );
        assert_eq!(
            SvyQuantileMethod::from_str("Linear"),
            SvyQuantileMethod::Linear
        );
        assert_eq!(
            SvyQuantileMethod::from_str("unknown"),
            SvyQuantileMethod::Higher
        ); // default
    }
}
