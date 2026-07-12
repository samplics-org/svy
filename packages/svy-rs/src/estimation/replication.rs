// src/estimation/replication.rs
//! Replication-based variance estimation (BRR, Bootstrap, Jackknife)

use polars::prelude::*;
use std::collections::HashMap;

// Import from taylor module for median functions
use super::taylor::{SvyQuantileMethod, weighted_quantile};

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

/// Centering method for variance estimation
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VarianceCenter {
    /// Center on mean of replicate estimates (default, matches Stata and R survey default)
    ReplicateMean,
    /// Center on full sample estimate (matches R survey mse=TRUE)
    FullSample,
}

impl Default for VarianceCenter {
    fn default() -> Self {
        VarianceCenter::ReplicateMean
    }
}

impl VarianceCenter {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "rep_mean" | "repmean" | "mean" | "average" | "replicates" => {
                Some(VarianceCenter::ReplicateMean)
            }
            "full_sample" | "fullsample" | "full" | "estimate" | "mse" => {
                Some(VarianceCenter::FullSample)
            }
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
/// * `center` - Centering method (ReplicateMean or FullSample)
pub fn variance_from_replicates(
    method: RepMethod,
    theta_full: f64,
    theta_reps: &[f64],
    rep_coefs: &[f64],
    center: VarianceCenter,
) -> f64 {
    let n_reps = theta_reps.len();
    if n_reps == 0 {
        return 0.0;
    }

    match method {
        RepMethod::Jackknife => {
            // Jackknife pseudo-value approach (unchanged)
            // For JK1 with coef = (n-1)/n:
            // factor = 1 / (1 - coef) = n
            // pseudo = factor * theta_full - (factor - 1) * theta_rep

            let factors: Vec<f64> = rep_coefs
                .iter()
                .map(|&c| {
                    if c < 1.0 {
                        1.0 / (1.0 - c)
                    } else {
                        f64::INFINITY
                    }
                })
                .collect();

            // Compute pseudo values
            let pseudo: Vec<f64> = theta_reps
                .iter()
                .zip(factors.iter())
                .map(|(&rep, &f)| f * theta_full - (f - 1.0) * rep)
                .collect();

            // Center (mean of pseudo values)
            let center: f64 = pseudo.iter().sum::<f64>() / n_reps as f64;

            // Variance = sum(c * ((pseudo - center) / (factor - 1))^2)
            let var: f64 = pseudo
                .iter()
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
        RepMethod::SDR | RepMethod::BRR | RepMethod::Bootstrap => {
            // Choose centering point based on parameter
            let center_value = match center {
                VarianceCenter::ReplicateMean => theta_reps.iter().sum::<f64>() / n_reps as f64,
                VarianceCenter::FullSample => theta_full,
            };

            let var: f64 = theta_reps
                .iter()
                .zip(rep_coefs.iter())
                .map(|(&rep, &c)| {
                    let diff = rep - center_value;
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
///
/// The naive build (column-outer loop writing `matrix[i*n_reps + r]`) is a
/// strided write — a cache miss on essentially every store — and dominated the
/// whole replication call (~530 ms at 1M rows × 40 reps). Instead, when the
/// replicate columns are contiguous and null-free (the common post-`prepare_data`
/// case) this does a **cache-blocked parallel transpose**: each row block's
/// slice of the row-major matrix (BLOCK × R floats) fits in L2, so its writes
/// stay hot, and blocks are filled in parallel. The output matrix is byte-for-
/// byte identical to the naive build, so downstream results are unchanged.
pub fn extract_rep_weights_matrix(
    df: &DataFrame,
    rep_weight_cols: &[String],
) -> PolarsResult<(Vec<f64>, usize, usize)> {
    let n = df.height();
    let n_reps = rep_weight_cols.len();
    let mut matrix = vec![0.0; n * n_reps];
    if n == 0 || n_reps == 0 {
        return Ok((matrix, n, n_reps));
    }

    let f64_cols: Vec<&Float64Chunked> = rep_weight_cols
        .iter()
        .map(|c| df.column(c)?.f64())
        .collect::<PolarsResult<_>>()?;

    // Borrow each column as one contiguous null-free slice, if possible.
    let contiguous: Option<Vec<&[f64]>> = f64_cols
        .iter()
        .map(|c| c.cont_slice().ok())
        .collect();

    match contiguous {
        Some(cols) => {
            use rayon::prelude::*;
            const BLOCK: usize = 256; // BLOCK * n_reps * 8 bytes stays in L2
            matrix
                .par_chunks_mut(BLOCK * n_reps)
                .enumerate()
                .for_each(|(b, chunk)| {
                    let start = b * BLOCK;
                    let rows = chunk.len() / n_reps;
                    for local_i in 0..rows {
                        let i = start + local_i;
                        let dst = &mut chunk[local_i * n_reps..local_i * n_reps + n_reps];
                        for (r, col) in cols.iter().enumerate() {
                            dst[r] = col[i];
                        }
                    }
                });
        }
        None => {
            // Fallback for chunked/nullable columns — identical semantics.
            for (r, col) in f64_cols.iter().enumerate() {
                for (i, v) in col.into_iter().enumerate() {
                    matrix[i * n_reps + r] = v.unwrap_or(0.0);
                }
            }
        }
    }

    Ok((matrix, n, n_reps))
}

/// Compute mean estimates for all replicates simultaneously
/// Returns (full_estimate, vec of replicate estimates)
pub fn matrix_mean_estimates(
    y: &[f64],
    full_weights: &[f64],
    rep_weights: &[f64], // Flattened (n × R) row-major
    n: usize,
    n_reps: usize,
    domain_mask: Option<&[f64]>,
) -> (f64, Vec<f64>) {
    // Full sample estimate
    let sum_wy: f64 = y
        .iter()
        .zip(full_weights.iter())
        .map(|(yi, wi)| yi * wi)
        .sum();
    let sum_w: f64 = full_weights.iter().sum();
    let theta_full = if sum_w > 0.0 {
        sum_wy / sum_w
    } else {
        f64::NAN
    };

    // Replicate estimates - single pass. `domain_mask` (1.0 inside the domain,
    // 0.0 outside) zeroes the replicate weight on the fly for `where=` domains,
    // avoiding materialising R zeroed columns upstream. mask=None ⇒ mi=1.0.
    let mut rep_sum_wy = vec![0.0; n_reps];
    let mut rep_sum_w = vec![0.0; n_reps];

    for i in 0..n {
        let yi = y[i];
        let mi = domain_mask.map_or(1.0, |m| m[i]);
        let base_idx = i * n_reps;
        for r in 0..n_reps {
            let w_ir = rep_weights[base_idx + r] * mi;
            rep_sum_wy[r] += yi * w_ir;
            rep_sum_w[r] += w_ir;
        }
    }

    // Compute means
    let theta_reps: Vec<f64> = rep_sum_wy
        .iter()
        .zip(rep_sum_w.iter())
        .map(|(&wy, &w)| if w > 0.0 { wy / w } else { f64::NAN })
        .collect();

    (theta_full, theta_reps)
}

/// Replicate mean estimates computed directly from the (contiguous) replicate
/// weight columns — no n×R matrix is materialised.
///
/// Each replicate is summed over rows in order, in parallel across replicates,
/// so the result is bit-identical to `matrix_mean_estimates`. Reading each
/// column once, contiguously, avoids both the 3× memory traffic (zero + write +
/// read) of building the flat matrix and its strided gather, which dominate at
/// large replicate counts.
pub fn matrix_mean_estimates_cols(
    y: &[f64],
    full_weights: &[f64],
    rep_cols: &[&[f64]],
    n: usize,
    domain_mask: Option<&[f64]>,
) -> (f64, Vec<f64>) {
    let sum_wy: f64 = y.iter().zip(full_weights.iter()).map(|(a, b)| a * b).sum();
    let sum_w: f64 = full_weights.iter().sum();
    let theta_full = if sum_w > 0.0 { sum_wy / sum_w } else { f64::NAN };

    use rayon::prelude::*;
    // `domain_mask` zeroes the replicate weight for `where=` domain rows in the
    // accumulation loop, so no R zeroed columns are materialised upstream. The
    // mask=None path stays multiply-free (the hot, non-domain case).
    let theta_reps: Vec<f64> = rep_cols
        .par_iter()
        .map(|col| {
            let mut swy = 0.0f64;
            let mut sw = 0.0f64;
            match domain_mask {
                None => {
                    for i in 0..n {
                        let w = col[i];
                        swy += y[i] * w;
                        sw += w;
                    }
                }
                Some(m) => {
                    for i in 0..n {
                        let w = col[i] * m[i];
                        swy += y[i] * w;
                        sw += w;
                    }
                }
            }
            if sw > 0.0 { swy / sw } else { f64::NAN }
        })
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
    domain_mask: Option<&[f64]>,
) -> (f64, Vec<f64>) {
    let theta_full: f64 = y
        .iter()
        .zip(full_weights.iter())
        .map(|(yi, wi)| yi * wi)
        .sum();

    let mut theta_reps = vec![0.0; n_reps];

    for i in 0..n {
        let yi = y[i];
        let mi = domain_mask.map_or(1.0, |m| m[i]);
        let base_idx = i * n_reps;
        for r in 0..n_reps {
            theta_reps[r] += yi * rep_weights[base_idx + r] * mi;
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
    domain_mask: Option<&[f64]>,
) -> (f64, Vec<f64>) {
    let sum_wy: f64 = y
        .iter()
        .zip(full_weights.iter())
        .map(|(yi, wi)| yi * wi)
        .sum();
    let sum_wx: f64 = x
        .iter()
        .zip(full_weights.iter())
        .map(|(xi, wi)| xi * wi)
        .sum();
    let theta_full = if sum_wx > 0.0 {
        sum_wy / sum_wx
    } else {
        f64::NAN
    };

    let mut rep_sum_wy = vec![0.0; n_reps];
    let mut rep_sum_wx = vec![0.0; n_reps];

    for i in 0..n {
        let yi = y[i];
        let xi = x[i];
        let mi = domain_mask.map_or(1.0, |m| m[i]);
        let base_idx = i * n_reps;
        for r in 0..n_reps {
            let w_ir = rep_weights[base_idx + r] * mi;
            rep_sum_wy[r] += yi * w_ir;
            rep_sum_wx[r] += xi * w_ir;
        }
    }

    let theta_reps: Vec<f64> = rep_sum_wy
        .iter()
        .zip(rep_sum_wx.iter())
        .map(|(&wy, &wx)| if wx > 0.0 { wy / wx } else { f64::NAN })
        .collect();

    (theta_full, theta_reps)
}

/// Replicate total estimates from the contiguous replicate columns (no matrix).
/// Bit-identical to `matrix_total_estimates`; see `matrix_mean_estimates_cols`.
pub fn matrix_total_estimates_cols(
    y: &[f64],
    full_weights: &[f64],
    rep_cols: &[&[f64]],
    n: usize,
    domain_mask: Option<&[f64]>,
) -> (f64, Vec<f64>) {
    let theta_full: f64 = y.iter().zip(full_weights.iter()).map(|(a, b)| a * b).sum();

    use rayon::prelude::*;
    let theta_reps: Vec<f64> = rep_cols
        .par_iter()
        .map(|col| {
            let mut swy = 0.0f64;
            match domain_mask {
                None => {
                    for i in 0..n {
                        swy += y[i] * col[i];
                    }
                }
                Some(m) => {
                    for i in 0..n {
                        swy += y[i] * col[i] * m[i];
                    }
                }
            }
            swy
        })
        .collect();

    (theta_full, theta_reps)
}

/// Replicate ratio estimates from the contiguous replicate columns (no matrix).
/// Bit-identical to `matrix_ratio_estimates`; see `matrix_mean_estimates_cols`.
pub fn matrix_ratio_estimates_cols(
    y: &[f64],
    x: &[f64],
    full_weights: &[f64],
    rep_cols: &[&[f64]],
    n: usize,
    domain_mask: Option<&[f64]>,
) -> (f64, Vec<f64>) {
    let sum_wy: f64 = y.iter().zip(full_weights.iter()).map(|(a, b)| a * b).sum();
    let sum_wx: f64 = x.iter().zip(full_weights.iter()).map(|(a, b)| a * b).sum();
    let theta_full = if sum_wx > 0.0 { sum_wy / sum_wx } else { f64::NAN };

    use rayon::prelude::*;
    let theta_reps: Vec<f64> = rep_cols
        .par_iter()
        .map(|col| {
            let mut swy = 0.0f64;
            let mut swx = 0.0f64;
            match domain_mask {
                None => {
                    for i in 0..n {
                        let w = col[i];
                        swy += y[i] * w;
                        swx += x[i] * w;
                    }
                }
                Some(m) => {
                    for i in 0..n {
                        let w = col[i] * m[i];
                        swy += y[i] * w;
                        swx += x[i] * w;
                    }
                }
            }
            if swx > 0.0 { swy / swx } else { f64::NAN }
        })
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

    let indices: Vec<u32> = by
        .into_iter()
        .map(|opt| match opt {
            Some(s) => *map.entry(s).or_insert_with(|| {
                let idx = next_idx;
                names.push(s.to_string());
                next_idx += 1;
                idx
            }),
            None => u32::MAX,
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

    let theta_full: Vec<f64> = sum_wy
        .iter()
        .zip(sum_w.iter())
        .map(|(&wy, &w)| if w > 0.0 { wy / w } else { f64::NAN })
        .collect();

    let theta_reps: Vec<Vec<f64>> = rep_sum_wy
        .iter()
        .zip(rep_sum_w.iter())
        .map(|(wy_vec, w_vec)| {
            wy_vec
                .iter()
                .zip(w_vec.iter())
                .map(|(&wy, &w)| if w > 0.0 { wy / w } else { f64::NAN })
                .collect()
        })
        .collect();

    (theta_full, theta_reps, counts)
}

/// By-domain replicate mean from the contiguous replicate columns (no matrix).
/// Each replicate accumulates per-domain sums over rows in order, in parallel
/// across replicates; the per-replicate results are reshaped to the
/// `[domain][rep]` layout `matrix_mean_by_domain` returns. Bit-identical.
pub fn matrix_mean_by_domain_cols(
    y: &[f64],
    full_weights: &[f64],
    rep_cols: &[&[f64]],
    domain_ids: &[u32],
    n_domains: usize,
    n: usize,
) -> (Vec<f64>, Vec<Vec<f64>>, Vec<u32>) {
    let mut sum_wy = vec![0.0; n_domains];
    let mut sum_w = vec![0.0; n_domains];
    let mut counts = vec![0u32; n_domains];
    for i in 0..n {
        let d = domain_ids[i] as usize;
        if d >= n_domains {
            continue;
        }
        sum_wy[d] += y[i] * full_weights[i];
        sum_w[d] += full_weights[i];
        counts[d] += 1;
    }
    let theta_full: Vec<f64> = sum_wy
        .iter()
        .zip(sum_w.iter())
        .map(|(&wy, &w)| if w > 0.0 { wy / w } else { f64::NAN })
        .collect();

    use rayon::prelude::*;
    let per_rep: Vec<Vec<f64>> = rep_cols
        .par_iter()
        .map(|col| {
            let mut swy = vec![0.0; n_domains];
            let mut sw = vec![0.0; n_domains];
            for i in 0..n {
                let d = domain_ids[i] as usize;
                if d < n_domains {
                    let w = col[i];
                    swy[d] += y[i] * w;
                    sw[d] += w;
                }
            }
            (0..n_domains)
                .map(|d| if sw[d] > 0.0 { swy[d] / sw[d] } else { f64::NAN })
                .collect()
        })
        .collect();

    // Reshape per_rep[rep][domain] → theta_reps[domain][rep].
    let n_reps = rep_cols.len();
    let theta_reps: Vec<Vec<f64>> = (0..n_domains)
        .map(|d| (0..n_reps).map(|r| per_rep[r][d]).collect())
        .collect();

    (theta_full, theta_reps, counts)
}

/// By-domain replicate total from the contiguous replicate columns (no matrix).
/// Bit-identical to `matrix_total_by_domain`.
pub fn matrix_total_by_domain_cols(
    y: &[f64],
    full_weights: &[f64],
    rep_cols: &[&[f64]],
    domain_ids: &[u32],
    n_domains: usize,
    n: usize,
) -> (Vec<f64>, Vec<Vec<f64>>, Vec<u32>) {
    let mut sum_wy = vec![0.0; n_domains];
    let mut counts = vec![0u32; n_domains];
    for i in 0..n {
        let d = domain_ids[i] as usize;
        if d >= n_domains {
            continue;
        }
        sum_wy[d] += y[i] * full_weights[i];
        counts[d] += 1;
    }

    use rayon::prelude::*;
    let per_rep: Vec<Vec<f64>> = rep_cols
        .par_iter()
        .map(|col| {
            let mut swy = vec![0.0; n_domains];
            for i in 0..n {
                let d = domain_ids[i] as usize;
                if d < n_domains {
                    swy[d] += y[i] * col[i];
                }
            }
            swy
        })
        .collect();

    let n_reps = rep_cols.len();
    let theta_reps: Vec<Vec<f64>> = (0..n_domains)
        .map(|d| (0..n_reps).map(|r| per_rep[r][d]).collect())
        .collect();

    (sum_wy, theta_reps, counts)
}

/// By-domain replicate ratio from the contiguous replicate columns (no matrix).
/// Bit-identical to `matrix_ratio_by_domain`.
pub fn matrix_ratio_by_domain_cols(
    y: &[f64],
    x: &[f64],
    full_weights: &[f64],
    rep_cols: &[&[f64]],
    domain_ids: &[u32],
    n_domains: usize,
    n: usize,
) -> (Vec<f64>, Vec<Vec<f64>>, Vec<u32>) {
    let mut sum_wy = vec![0.0; n_domains];
    let mut sum_wx = vec![0.0; n_domains];
    let mut counts = vec![0u32; n_domains];
    for i in 0..n {
        let d = domain_ids[i] as usize;
        if d >= n_domains {
            continue;
        }
        let w = full_weights[i];
        sum_wy[d] += y[i] * w;
        sum_wx[d] += x[i] * w;
        counts[d] += 1;
    }
    let theta_full: Vec<f64> = (0..n_domains)
        .map(|d| if sum_wx[d] > 0.0 { sum_wy[d] / sum_wx[d] } else { f64::NAN })
        .collect();

    use rayon::prelude::*;
    let per_rep: Vec<Vec<f64>> = rep_cols
        .par_iter()
        .map(|col| {
            let mut swy = vec![0.0; n_domains];
            let mut swx = vec![0.0; n_domains];
            for i in 0..n {
                let d = domain_ids[i] as usize;
                if d < n_domains {
                    let w = col[i];
                    swy[d] += y[i] * w;
                    swx[d] += x[i] * w;
                }
            }
            (0..n_domains)
                .map(|d| if swx[d] > 0.0 { swy[d] / swx[d] } else { f64::NAN })
                .collect()
        })
        .collect();

    let n_reps = rep_cols.len();
    let theta_reps: Vec<Vec<f64>> = (0..n_domains)
        .map(|d| (0..n_reps).map(|r| per_rep[r][d]).collect())
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

    let theta_reps: Vec<Vec<f64>> = rep_sum_wy.iter().map(|wy_vec| wy_vec.clone()).collect();

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

    let theta_full: Vec<f64> = sum_wy
        .iter()
        .zip(sum_wx.iter())
        .map(|(&wy, &wx)| if wx > 0.0 { wy / wx } else { f64::NAN })
        .collect();

    let theta_reps: Vec<Vec<f64>> = rep_sum_wy
        .iter()
        .zip(rep_sum_wx.iter())
        .map(|(wy_vec, wx_vec)| {
            wy_vec
                .iter()
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

/// Compute proportion estimates by level for all replicates.
/// Category values are integers (Boolean or numeric columns).
/// Returns (levels, full_estimates[L], replicate_estimates[L][R])
pub fn matrix_prop_estimates(
    y: &[i64], // Category values
    full_weights: &[f64],
    rep_weights: &[f64],
    n: usize,
    n_reps: usize,
    domain_mask: Option<&[f64]>,
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

        let mi = domain_mask.map_or(1.0, |m| m[i]);
        let base_idx = i * n_reps;
        for r in 0..n_reps {
            let w_ir = rep_weights[base_idx + r] * mi;
            rep_sum_w_level[lev_idx][r] += w_ir;
            rep_sum_w_total[r] += w_ir;
        }
    }

    // Compute proportions
    let theta_full: Vec<f64> = sum_w_level
        .iter()
        .map(|&w_l| {
            if sum_w_total > 0.0 {
                w_l / sum_w_total
            } else {
                f64::NAN
            }
        })
        .collect();

    let theta_reps: Vec<Vec<f64>> = rep_sum_w_level
        .iter()
        .map(|w_l_vec| {
            w_l_vec
                .iter()
                .zip(rep_sum_w_total.iter())
                .map(|(&w_l, &w_t)| if w_t > 0.0 { w_l / w_t } else { f64::NAN })
                .collect()
        })
        .collect();

    (levels, theta_full, theta_reps)
}

/// Sorted category levels and the per-row level index for integer category data.
fn prop_level_index(y: &[i64]) -> (Vec<i64>, Vec<usize>) {
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
    let lev_idx: Vec<usize> = y.iter().map(|v| level_map[v]).collect();
    (levels, lev_idx)
}

/// `matrix_prop_estimates` from the contiguous replicate columns (no matrix).
/// Bit-identical: each replicate accumulates per-level weight in row order,
/// parallel over replicates, reshaped to `[level][rep]`.
pub fn matrix_prop_estimates_cols(
    y: &[i64],
    full_weights: &[f64],
    rep_cols: &[&[f64]],
    n: usize,
    domain_mask: Option<&[f64]>,
) -> (Vec<i64>, Vec<f64>, Vec<Vec<f64>>) {
    let (levels, lev_idx) = prop_level_index(y);
    let n_levels = levels.len();

    let mut sum_w_level = vec![0.0; n_levels];
    let mut sum_w_total = 0.0;
    for i in 0..n {
        sum_w_level[lev_idx[i]] += full_weights[i];
        sum_w_total += full_weights[i];
    }
    let theta_full: Vec<f64> = sum_w_level
        .iter()
        .map(|&w_l| if sum_w_total > 0.0 { w_l / sum_w_total } else { f64::NAN })
        .collect();

    use rayon::prelude::*;
    let per_rep: Vec<Vec<f64>> = rep_cols
        .par_iter()
        .map(|col| {
            let mut swl = vec![0.0; n_levels];
            let mut swt = 0.0;
            match domain_mask {
                None => {
                    for i in 0..n {
                        let w = col[i];
                        swl[lev_idx[i]] += w;
                        swt += w;
                    }
                }
                Some(m) => {
                    for i in 0..n {
                        let w = col[i] * m[i];
                        swl[lev_idx[i]] += w;
                        swt += w;
                    }
                }
            }
            (0..n_levels)
                .map(|l| if swt > 0.0 { swl[l] / swt } else { f64::NAN })
                .collect()
        })
        .collect();

    let n_reps = rep_cols.len();
    let theta_reps: Vec<Vec<f64>> = (0..n_levels)
        .map(|l| (0..n_reps).map(|r| per_rep[r][l]).collect())
        .collect();

    (levels, theta_full, theta_reps)
}

/// `matrix_prop_estimates_str` from the contiguous replicate columns. Same as
/// the integer version but with string category levels. Bit-identical.
pub fn matrix_prop_estimates_str_cols(
    y: &[String],
    full_weights: &[f64],
    rep_cols: &[&[f64]],
    n: usize,
    domain_mask: Option<&[f64]>,
) -> (Vec<String>, Vec<f64>, Vec<Vec<f64>>) {
    let mut level_map: HashMap<&str, usize> = HashMap::new();
    let mut levels: Vec<String> = Vec::new();
    for val in y {
        if !level_map.contains_key(val.as_str()) {
            level_map.insert(val.as_str(), levels.len());
            levels.push(val.clone());
        }
    }
    levels.sort();
    for (idx, lev) in levels.iter().enumerate() {
        level_map.insert(lev.as_str(), idx);
    }
    let n_levels = levels.len();
    let lev_idx: Vec<usize> = y.iter().map(|v| level_map[v.as_str()]).collect();

    let mut sum_w_level = vec![0.0; n_levels];
    let mut sum_w_total = 0.0;
    for i in 0..n {
        sum_w_level[lev_idx[i]] += full_weights[i];
        sum_w_total += full_weights[i];
    }
    let theta_full: Vec<f64> = sum_w_level
        .iter()
        .map(|&w_l| if sum_w_total > 0.0 { w_l / sum_w_total } else { f64::NAN })
        .collect();

    use rayon::prelude::*;
    let per_rep: Vec<Vec<f64>> = rep_cols
        .par_iter()
        .map(|col| {
            let mut swl = vec![0.0; n_levels];
            let mut swt = 0.0;
            match domain_mask {
                None => {
                    for i in 0..n {
                        let w = col[i];
                        swl[lev_idx[i]] += w;
                        swt += w;
                    }
                }
                Some(m) => {
                    for i in 0..n {
                        let w = col[i] * m[i];
                        swl[lev_idx[i]] += w;
                        swt += w;
                    }
                }
            }
            (0..n_levels)
                .map(|l| if swt > 0.0 { swl[l] / swt } else { f64::NAN })
                .collect()
        })
        .collect();

    let n_reps = rep_cols.len();
    let theta_reps: Vec<Vec<f64>> = (0..n_levels)
        .map(|l| (0..n_reps).map(|r| per_rep[r][l]).collect())
        .collect();

    (levels, theta_full, theta_reps)
}

/// Compute proportion estimates by level and domain.
/// Category values are integers (Boolean or numeric columns).
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
    let theta_full: Vec<Vec<f64>> = sum_w_level
        .iter()
        .zip(sum_w_total.iter())
        .map(|(w_l_vec, &w_t)| {
            w_l_vec
                .iter()
                .map(|&w_l| if w_t > 0.0 { w_l / w_t } else { f64::NAN })
                .collect()
        })
        .collect();

    // [domain][level][rep]
    let theta_reps: Vec<Vec<Vec<f64>>> = rep_sum_w_level
        .iter()
        .zip(rep_sum_w_total.iter())
        .map(|(dom_levels, dom_totals)| {
            dom_levels
                .iter()
                .map(|w_l_reps| {
                    w_l_reps
                        .iter()
                        .zip(dom_totals.iter())
                        .map(|(&w_l, &w_t)| if w_t > 0.0 { w_l / w_t } else { f64::NAN })
                        .collect()
                })
                .collect()
        })
        .collect();

    (levels, theta_full, theta_reps, counts)
}

/// Compute proportion estimates by level for all replicates.
/// Category values are strings (String or Categorical columns).
/// Levels are sorted lexicographically and preserved as-is in the output.
/// Returns (levels, full_estimates[L], replicate_estimates[L][R])
pub fn matrix_prop_estimates_str(
    y: &[String],
    full_weights: &[f64],
    rep_weights: &[f64],
    n: usize,
    n_reps: usize,
    domain_mask: Option<&[f64]>,
) -> (Vec<String>, Vec<f64>, Vec<Vec<f64>>) {
    // Find unique levels
    let mut level_map: HashMap<&str, usize> = HashMap::new();
    let mut levels: Vec<String> = Vec::new();

    for val in y {
        if !level_map.contains_key(val.as_str()) {
            level_map.insert(val.as_str(), levels.len());
            levels.push(val.clone());
        }
    }
    levels.sort();

    // Re-index after sorting
    for (idx, lev) in levels.iter().enumerate() {
        level_map.insert(lev.as_str(), idx);
    }

    let n_levels = levels.len();

    // Accumulate weighted counts per level
    let mut sum_w_level = vec![0.0f64; n_levels];
    let mut sum_w_total = 0.0f64;
    let mut rep_sum_w_level = vec![vec![0.0f64; n_reps]; n_levels];
    let mut rep_sum_w_total = vec![0.0f64; n_reps];

    for i in 0..n {
        let lev_idx = level_map[y[i].as_str()];
        let wi = full_weights[i];

        sum_w_level[lev_idx] += wi;
        sum_w_total += wi;

        let mi = domain_mask.map_or(1.0, |m| m[i]);
        let base_idx = i * n_reps;
        for r in 0..n_reps {
            let w_ir = rep_weights[base_idx + r] * mi;
            rep_sum_w_level[lev_idx][r] += w_ir;
            rep_sum_w_total[r] += w_ir;
        }
    }

    // Compute proportions
    let theta_full: Vec<f64> = sum_w_level
        .iter()
        .map(|&w_l| if sum_w_total > 0.0 { w_l / sum_w_total } else { f64::NAN })
        .collect();

    let theta_reps: Vec<Vec<f64>> = rep_sum_w_level
        .iter()
        .map(|w_l_vec| {
            w_l_vec
                .iter()
                .zip(rep_sum_w_total.iter())
                .map(|(&w_l, &w_t)| if w_t > 0.0 { w_l / w_t } else { f64::NAN })
                .collect()
        })
        .collect();

    (levels, theta_full, theta_reps)
}

/// Compute proportion estimates by level and domain.
/// Category values are strings (String or Categorical columns).
/// Returns (levels, full_estimates[D][L], replicate_estimates[D][L][R], counts[D])
pub fn matrix_prop_by_domain_str(
    y: &[String],
    full_weights: &[f64],
    rep_weights: &[f64],
    domain_ids: &[u32],
    n_domains: usize,
    n: usize,
    n_reps: usize,
) -> (Vec<String>, Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>, Vec<u32>) {
    // Find unique levels
    let mut level_map: HashMap<&str, usize> = HashMap::new();
    let mut levels: Vec<String> = Vec::new();

    for val in y {
        if !level_map.contains_key(val.as_str()) {
            level_map.insert(val.as_str(), levels.len());
            levels.push(val.clone());
        }
    }
    levels.sort();

    // Re-index after sorting
    for (idx, lev) in levels.iter().enumerate() {
        level_map.insert(lev.as_str(), idx);
    }

    let n_levels = levels.len();

    // [domain][level]
    let mut sum_w_level = vec![vec![0.0f64; n_levels]; n_domains];
    let mut sum_w_total = vec![0.0f64; n_domains];
    // [domain][level][rep]
    let mut rep_sum_w_level = vec![vec![vec![0.0f64; n_reps]; n_levels]; n_domains];
    let mut rep_sum_w_total = vec![vec![0.0f64; n_reps]; n_domains];
    let mut counts = vec![0u32; n_domains];

    for i in 0..n {
        let d = domain_ids[i] as usize;
        if d >= n_domains {
            continue;
        }

        let lev_idx = level_map[y[i].as_str()];
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
    let theta_full: Vec<Vec<f64>> = sum_w_level
        .iter()
        .zip(sum_w_total.iter())
        .map(|(w_l_vec, &w_t)| {
            w_l_vec
                .iter()
                .map(|&w_l| if w_t > 0.0 { w_l / w_t } else { f64::NAN })
                .collect()
        })
        .collect();

    // [domain][level][rep]
    let theta_reps: Vec<Vec<Vec<f64>>> = rep_sum_w_level
        .iter()
        .zip(rep_sum_w_total.iter())
        .map(|(dom_levels, dom_totals)| {
            dom_levels
                .iter()
                .map(|w_l_reps| {
                    w_l_reps
                        .iter()
                        .zip(dom_totals.iter())
                        .map(|(&w_l, &w_t)| if w_t > 0.0 { w_l / w_t } else { f64::NAN })
                        .collect()
                })
                .collect()
        })
        .collect();

    (levels, theta_full, theta_reps, counts)
}

// ============================================================================
// Median Estimation Functions
// ============================================================================

/// Compute weighted median for a single set of weights (used by replication)
fn weighted_median_vec(y: &[f64], weights: &[f64], n: usize, q_method: SvyQuantileMethod) -> f64 {
    // Collect (y, w) pairs, filtering zeros/NaN
    let mut pairs: Vec<(f64, f64)> = Vec::with_capacity(n);
    for i in 0..n {
        let yi = y[i];
        let wi = weights[i];
        if wi > 0.0 && yi.is_finite() {
            pairs.push((yi, wi));
        }
    }

    if pairs.is_empty() {
        return f64::NAN;
    }

    // Sort by y value
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let y_sorted: Vec<f64> = pairs.iter().map(|(y, _)| *y).collect();
    let w_sorted: Vec<f64> = pairs.iter().map(|(_, w)| *w).collect();

    // Compute CDF
    let total_w: f64 = w_sorted.iter().sum();
    if total_w <= 0.0 {
        return f64::NAN;
    }

    let mut cdf = Vec::with_capacity(w_sorted.len());
    let mut cumsum = 0.0;
    for w in &w_sorted {
        cumsum += w;
        cdf.push(cumsum / total_w);
    }

    weighted_quantile(&y_sorted, &cdf, 0.5, q_method)
}

/// Compute median estimates for all replicates simultaneously
/// Returns (full_estimate, vec of replicate estimates)
pub fn matrix_median_estimates(
    y: &[f64],
    full_weights: &[f64],
    rep_weights: &[f64], // Flattened (n × R) row-major
    n: usize,
    n_reps: usize,
    q_method: SvyQuantileMethod,
) -> (f64, Vec<f64>) {
    // Full sample estimate
    let theta_full = weighted_median_vec(y, full_weights, n, q_method);

    // Replicate estimates
    let mut theta_reps = Vec::with_capacity(n_reps);

    // Extract each replicate's weights and compute median
    for r in 0..n_reps {
        let rep_w: Vec<f64> = (0..n).map(|i| rep_weights[i * n_reps + r]).collect();

        let theta_r = weighted_median_vec(y, &rep_w, n, q_method);
        theta_reps.push(theta_r);
    }

    (theta_full, theta_reps)
}

/// Compute median estimates by domain for all replicates
/// Returns (full_estimates[K], replicate_estimates[K][R], domain_counts[K])
pub fn matrix_median_by_domain(
    y: &[f64],
    full_weights: &[f64],
    rep_weights: &[f64],
    domain_ids: &[u32],
    n_domains: usize,
    n: usize,
    n_reps: usize,
    q_method: SvyQuantileMethod,
) -> (Vec<f64>, Vec<Vec<f64>>, Vec<u32>) {
    // Organize data by domain
    let mut domain_data: Vec<Vec<(f64, f64, usize)>> = vec![Vec::new(); n_domains];
    let mut counts = vec![0u32; n_domains];

    for i in 0..n {
        let d = domain_ids[i] as usize;
        if d >= n_domains {
            continue;
        }
        domain_data[d].push((y[i], full_weights[i], i));
        counts[d] += 1;
    }

    // Compute full sample estimates for each domain
    let mut theta_full = Vec::with_capacity(n_domains);
    for d in 0..n_domains {
        let y_d: Vec<f64> = domain_data[d].iter().map(|(yi, _, _)| *yi).collect();
        let w_d: Vec<f64> = domain_data[d].iter().map(|(_, wi, _)| *wi).collect();
        let n_d = y_d.len();

        let est = weighted_median_vec(&y_d, &w_d, n_d, q_method);
        theta_full.push(est);
    }

    // Compute replicate estimates for each domain
    let mut theta_reps: Vec<Vec<f64>> = vec![Vec::with_capacity(n_reps); n_domains];

    for r in 0..n_reps {
        for d in 0..n_domains {
            let y_d: Vec<f64> = domain_data[d].iter().map(|(yi, _, _)| *yi).collect();
            let w_d: Vec<f64> = domain_data[d]
                .iter()
                .map(|(_, _, i)| rep_weights[i * n_reps + r])
                .collect();
            let n_d = y_d.len();

            let est = weighted_median_vec(&y_d, &w_d, n_d, q_method);
            theta_reps[d].push(est);
        }
    }

    (theta_full, theta_reps, counts)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variance_center_from_str() {
        assert_eq!(
            VarianceCenter::from_str("rep_mean"),
            Some(VarianceCenter::ReplicateMean)
        );
        assert_eq!(
            VarianceCenter::from_str("mean"),
            Some(VarianceCenter::ReplicateMean)
        );
        assert_eq!(
            VarianceCenter::from_str("full_sample"),
            Some(VarianceCenter::FullSample)
        );
        assert_eq!(
            VarianceCenter::from_str("estimate"),
            Some(VarianceCenter::FullSample)
        );
        assert_eq!(
            VarianceCenter::from_str("mse"),
            Some(VarianceCenter::FullSample)
        );
    }

    #[test]
    fn test_sdr_coefficients() {
        let coefs = replicate_coefficients(RepMethod::SDR, 80, 0.0);
        assert_eq!(coefs.len(), 80);
        assert!((coefs[0] - 4.0 / 80.0).abs() < 1e-10);
    }

    #[test]
    fn test_sdr_variance_replicate_mean() {
        // Test with default centering (replicate mean)
        let theta_full = 100.0;
        let theta_reps = vec![98.0, 102.0, 99.0, 101.0];
        let rep_coefs = replicate_coefficients(RepMethod::SDR, 4, 0.0);

        // Mean of replicates = 100.0
        // Var = (4/4) * sum((rep - 100)^2) = 1.0 * (4 + 4 + 1 + 1) = 10.0
        let var = variance_from_replicates(
            RepMethod::SDR,
            theta_full,
            &theta_reps,
            &rep_coefs,
            VarianceCenter::ReplicateMean,
        );
        assert!((var - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_sdr_variance_full_sample() {
        // Test with mse=TRUE centering (full sample estimate)
        let theta_full = 105.0; // Different from mean of replicates
        let theta_reps = vec![98.0, 102.0, 99.0, 101.0];
        let rep_coefs = replicate_coefficients(RepMethod::SDR, 4, 0.0);

        // Var = (4/4) * sum((rep - 105)^2)
        //     = 1.0 * (49 + 9 + 36 + 16) = 110.0
        let var = variance_from_replicates(
            RepMethod::SDR,
            theta_full,
            &theta_reps,
            &rep_coefs,
            VarianceCenter::FullSample,
        );
        assert!((var - 110.0).abs() < 1e-10);
    }

    #[test]
    fn test_sdr_from_str() {
        assert_eq!(RepMethod::from_str("sdr"), Some(RepMethod::SDR));
        assert_eq!(RepMethod::from_str("SDR"), Some(RepMethod::SDR));
        assert_eq!(RepMethod::from_str("acs"), Some(RepMethod::SDR));
        assert_eq!(RepMethod::from_str("ACS"), Some(RepMethod::SDR));
        assert_eq!(
            RepMethod::from_str("successive-difference"),
            Some(RepMethod::SDR)
        );
    }

    #[test]
    fn test_weighted_median_vec_uniform() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let w = vec![1.0, 1.0, 1.0, 1.0, 1.0];

        let median = weighted_median_vec(&y, &w, 5, SvyQuantileMethod::Linear);
        assert!(
            median >= 2.5 && median <= 3.5,
            "Median {} should be around 3",
            median
        );
    }

    #[test]
    fn test_weighted_median_vec_nonuniform() {
        // Heavy weight on 5.0
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let w = vec![1.0, 1.0, 1.0, 1.0, 10.0];

        let median = weighted_median_vec(&y, &w, 5, SvyQuantileMethod::Linear);
        // With weight 10 on 5.0, median should be pulled towards 5
        assert!(
            median >= 4.0,
            "Median {} should be >= 4 with heavy weight on 5",
            median
        );
    }

    #[test]
    fn test_matrix_median_estimates() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let full_w = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        // 2 replicates, flattened row-major
        let rep_w = vec![
            1.0, 1.0, // obs 0: rep0=1, rep1=1
            2.0, 0.5, // obs 1: rep0=2, rep1=0.5
            1.0, 1.0, // obs 2
            0.5, 2.0, // obs 3
            1.0, 1.0, // obs 4
        ];

        let (theta_full, theta_reps) =
            matrix_median_estimates(&y, &full_w, &rep_w, 5, 2, SvyQuantileMethod::Linear);

        assert!(theta_full.is_finite());
        assert_eq!(theta_reps.len(), 2);
        assert!(theta_reps[0].is_finite());
        assert!(theta_reps[1].is_finite());
    }

    #[test]
    fn test_matrix_prop_estimates_str_two_levels() {
        // 6 obs: 4 "yes", 2 "no", equal weights → yes=2/3, no=1/3
        let y = vec![
            "yes".to_string(), "yes".to_string(), "no".to_string(),
            "yes".to_string(), "no".to_string(), "yes".to_string(),
        ];
        let w = vec![1.0; 6];
        // 2 replicates, all weight = 1
        let rep_w = vec![1.0f64; 6 * 2];

        let (levels, theta_full, theta_reps) =
            matrix_prop_estimates_str(&y, &w, &rep_w, 6, 2, None);

        assert_eq!(levels, vec!["no".to_string(), "yes".to_string()]);
        let no_idx = 0;
        let yes_idx = 1;
        assert!((theta_full[no_idx]  - 2.0 / 6.0).abs() < 1e-10);
        assert!((theta_full[yes_idx] - 4.0 / 6.0).abs() < 1e-10);
        assert!((theta_full[no_idx] + theta_full[yes_idx] - 1.0).abs() < 1e-10);
        assert_eq!(theta_reps[no_idx].len(), 2);
        assert_eq!(theta_reps[yes_idx].len(), 2);
    }

    #[test]
    fn test_matrix_prop_by_domain_str_basic() {
        // 4 obs: domain A (obs 0,1), domain B (obs 2,3)
        // y: A→["yes","no"], B→["no","no"]
        let y = vec![
            "yes".to_string(), "no".to_string(),
            "no".to_string(),  "no".to_string(),
        ];
        let w    = vec![1.0; 4];
        let rep_w = vec![1.0f64; 4 * 2]; // 2 reps
        let domain_ids = vec![0u32, 0, 1, 1];

        let (levels, theta_full, _theta_reps, counts) =
            matrix_prop_by_domain_str(&y, &w, &rep_w, &domain_ids, 2, 4, 2);

        assert_eq!(levels, vec!["no".to_string(), "yes".to_string()]);
        // Domain A: no=0.5, yes=0.5
        assert!((theta_full[0][0] - 0.5).abs() < 1e-10, "A no");
        assert!((theta_full[0][1] - 0.5).abs() < 1e-10, "A yes");
        // Domain B: no=1.0, yes=0.0
        assert!((theta_full[1][0] - 1.0).abs() < 1e-10, "B no");
        assert!((theta_full[1][1] - 0.0).abs() < 1e-10, "B yes");
        assert_eq!(counts, vec![2, 2]);
    }
}
