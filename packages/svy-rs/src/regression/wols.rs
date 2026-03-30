// src/regression/wols.rs
//
// Weighted Ordinary Least Squares with influence function computation.
//
// This provides a closed-form WLS solver (no iteration) for Gaussian
// identity-link models. Used by ttest and ranktest engines which need:
//   1. Regression coefficients: beta = (X'WX)^{-1} X'Wy
//   2. Influence functions: h_i = (x_i * e_i) * (X'WX)^{-1}
//      where e_i = y_i - x_i'beta (residuals)
//   3. Design-based variance via taylor_variance on influence functions
//
// Matches R's pattern:
//   m <- lm(y ~ g, weights = w)
//   infn <- (model.matrix(m) * residuals(m)) %*% summary(m)$cov.unscaled
//   tot.infn <- svytotal(infn, design)
//   SE(tot.infn)  # design-based SE for each coefficient

use polars::prelude::*;
use std::collections::HashMap;

use crate::estimation::taylor::taylor_variance;

// ============================================================================
// Core WLS Result
// ============================================================================

/// Result of a weighted OLS fit with influence functions.
pub struct WolsResult {
    /// Regression coefficients, length k
    pub beta: Vec<f64>,
    /// Fitted values, length n
    pub fitted: Vec<f64>,
    /// Residuals (y - fitted), length n
    pub residuals: Vec<f64>,
    /// (X'WX)^{-1}, the unscaled covariance, k x k flattened row-major
    pub cov_unscaled: Vec<f64>,
    /// Influence functions, n x k flattened row-major
    /// infn[i * k + j] = influence of obs i on coefficient j
    pub influence: Vec<f64>,
    /// Number of observations
    pub n: usize,
    /// Number of parameters
    pub k: usize,
}

// ============================================================================
// Matrix helpers (lightweight, no external dep needed for small k)
// ============================================================================

/// Solve A x = b for small symmetric positive definite A (k x k).
/// Uses Cholesky with LU fallback. A is k x k row-major, b is k x 1.
pub fn solve_kxk(a: &[f64], b: &[f64], k: usize) -> Option<Vec<f64>> {
    // Try Cholesky first
    if let Some(x) = solve_cholesky(a, b, k) {
        return Some(x);
    }
    // LU fallback via Gaussian elimination with partial pivoting
    solve_gauss(a, b, k)
}

fn solve_cholesky(a: &[f64], b: &[f64], k: usize) -> Option<Vec<f64>> {
    // Cholesky decomposition: A = L L'
    let mut l = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..=i {
            let mut sum = 0.0;
            for p in 0..j {
                sum += l[i * k + p] * l[j * k + p];
            }
            if i == j {
                let diag = a[i * k + i] - sum;
                if diag <= 0.0 {
                    return None; // Not positive definite
                }
                l[i * k + j] = diag.sqrt();
            } else {
                let denom = l[j * k + j];
                if denom.abs() < 1e-30 {
                    return None;
                }
                l[i * k + j] = (a[i * k + j] - sum) / denom;
            }
        }
    }

    // Forward substitution: L y = b
    let mut y = vec![0.0; k];
    for i in 0..k {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[i * k + j] * y[j];
        }
        y[i] = (b[i] - sum) / l[i * k + i];
    }

    // Back substitution: L' x = y
    let mut x = vec![0.0; k];
    for i in (0..k).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..k {
            sum += l[j * k + i] * x[j]; // L' element is l[j][i]
        }
        x[i] = (y[i] - sum) / l[i * k + i];
    }

    Some(x)
}

fn solve_gauss(a: &[f64], b: &[f64], k: usize) -> Option<Vec<f64>> {
    // Augmented matrix [A | b]
    let mut aug = vec![0.0; k * (k + 1)];
    for i in 0..k {
        for j in 0..k {
            aug[i * (k + 1) + j] = a[i * k + j];
        }
        aug[i * (k + 1) + k] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..k {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col * (k + 1) + col].abs();
        for row in (col + 1)..k {
            let v = aug[row * (k + 1) + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-30 {
            return None; // Singular
        }

        // Swap rows
        if max_row != col {
            for j in 0..=(k) {
                let tmp = aug[col * (k + 1) + j];
                aug[col * (k + 1) + j] = aug[max_row * (k + 1) + j];
                aug[max_row * (k + 1) + j] = tmp;
            }
        }

        // Eliminate
        let pivot = aug[col * (k + 1) + col];
        for row in (col + 1)..k {
            let factor = aug[row * (k + 1) + col] / pivot;
            for j in col..=(k) {
                aug[row * (k + 1) + j] -= factor * aug[col * (k + 1) + j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; k];
    for i in (0..k).rev() {
        let mut sum = aug[i * (k + 1) + k];
        for j in (i + 1)..k {
            sum -= aug[i * (k + 1) + j] * x[j];
        }
        x[i] = sum / aug[i * (k + 1) + i];
    }

    Some(x)
}

/// Invert a k x k symmetric matrix. Returns None if singular.
fn invert_kxk(a: &[f64], k: usize) -> Option<Vec<f64>> {
    let mut inv = vec![0.0; k * k];
    for j in 0..k {
        let mut e = vec![0.0; k];
        e[j] = 1.0;
        let col = solve_kxk(a, &e, k)?;
        for i in 0..k {
            inv[i * k + j] = col[i];
        }
    }
    Some(inv)
}

// ============================================================================
// Core WLS Fit
// ============================================================================

/// Fit weighted OLS: y ~ X with sampling weights w.
///
/// Computes:
///   beta = (X'WX)^{-1} X'Wy
///   residuals = y - X beta
///   cov_unscaled = (X'WX)^{-1}    (R's summary(lm)$cov.unscaled)
///   influence[i] = (x_i * resid_i) %*% cov_unscaled   (n x k)
///
/// # Arguments
/// * `y` - Response vector, length n
/// * `x` - Design matrix, n x k, row-major
/// * `w` - Sampling weights, length n (must be positive)
/// * `n` - Number of observations
/// * `k` - Number of predictors (columns of X)
pub fn fit_wols(y: &[f64], x: &[f64], w: &[f64], n: usize, k: usize) -> Result<WolsResult, String> {
    if n == 0 || k == 0 {
        return Err("Empty data or no predictors".to_string());
    }

    // Build X'WX (k x k) and X'Wy (k x 1)
    let mut xtwx = vec![0.0; k * k];
    let mut xtwy = vec![0.0; k];

    for i in 0..n {
        let wi = w[i];
        if wi <= 0.0 {
            continue;
        }
        let yi = y[i];
        for r in 0..k {
            let xir = x[i * k + r];
            xtwy[r] += wi * xir * yi;
            for c in r..k {
                let xic = x[i * k + c];
                let v = wi * xir * xic;
                xtwx[r * k + c] += v;
                if c != r {
                    xtwx[c * k + r] += v;
                }
            }
        }
    }

    // Solve for beta
    let beta = solve_kxk(&xtwx, &xtwy, k).ok_or_else(|| "Singular X'WX matrix".to_string())?;

    // Compute fitted values and residuals
    let mut fitted = vec![0.0; n];
    let mut residuals = vec![0.0; n];
    for i in 0..n {
        let mut yhat = 0.0;
        for j in 0..k {
            yhat += x[i * k + j] * beta[j];
        }
        fitted[i] = yhat;
        residuals[i] = y[i] - yhat;
    }

    // cov_unscaled = (X'WX)^{-1}
    let cov_unscaled = invert_kxk(&xtwx, k).ok_or_else(|| "Cannot invert X'WX".to_string())?;

    // Influence functions: infn[i, j] = sum_p (x[i,p] * resid[i]) * cov_unscaled[p, j]
    // This matches R's: (xmat * (rankscore - fitted(m))) %*% summary(m)$cov.unscaled
    let mut influence = vec![0.0; n * k];
    for i in 0..n {
        let ei = residuals[i];
        for j in 0..k {
            let mut val = 0.0;
            for p in 0..k {
                val += x[i * k + p] * ei * cov_unscaled[p * k + j];
            }
            influence[i * k + j] = val;
        }
    }

    Ok(WolsResult {
        beta,
        fitted,
        residuals,
        cov_unscaled,
        influence,
        n,
        k,
    })
}

// ============================================================================
// Design-based variance of influence function totals
// ============================================================================

/// Compute design-based SE for each WLS coefficient using influence functions.
///
/// This is the equivalent of R's:
///   tot.infn <- svytotal(infn, design)
///   SE(tot.infn)
///
/// R's svytotal(x, design) computes total = sum(x_i * w_i) and its variance
/// from scores z_i = x_i * w_i. So we must multiply influence functions by
/// sampling weights before passing to taylor_variance.
///
/// # Returns
/// Vector of length k: the design-based SE for each coefficient.
pub fn influence_se(
    influence: &[f64],
    w: &[f64],
    n: usize,
    k: usize,
    strata: Option<&StringChunked>,
    psu: Option<&StringChunked>,
    ssu: Option<&StringChunked>,
    fpc: Option<&Float64Chunked>,
    fpc_ssu: Option<&Float64Chunked>,
    singleton_method: Option<&str>,
) -> PolarsResult<Vec<f64>> {
    let mut ses = Vec::with_capacity(k);

    for j in 0..k {
        // Extract column j of influence matrix, scaled by weights:
        // score_i = infn_i * w_i  (matches R's svytotal internals)
        let col_vals: Vec<f64> = (0..n).map(|i| influence[i * k + j] * w[i]).collect();
        let scores = Float64Chunked::from_vec("infn".into(), col_vals);

        let var = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;

        ses.push(var.max(0.0).sqrt());
    }

    Ok(ses)
}

/// Compute the full k x k design-based covariance matrix of WLS coefficients.
///
/// This computes the variance-covariance using PSU totals of weight-scaled
/// influence functions, matching R's svyrecvar applied to svytotal influence.
///
/// # Returns
/// Flattened k x k covariance matrix (row-major).
pub fn influence_covariance(
    influence: &[f64],
    w: &[f64],
    n: usize,
    k: usize,
    strata: Option<&StringChunked>,
    psu: Option<&StringChunked>,
    singleton_method: Option<&str>,
) -> PolarsResult<Vec<f64>> {
    // Index strata and PSUs
    let (strata_idx, n_strata) = match strata {
        Some(s) => {
            let mut map: HashMap<&str, u32> = HashMap::new();
            let mut next = 0u32;
            let indices: Vec<u32> = s
                .iter()
                .map(|opt| {
                    let key = opt.unwrap_or("__NULL__");
                    *map.entry(key).or_insert_with(|| {
                        let i = next;
                        next += 1;
                        i
                    })
                })
                .collect();
            (indices, next)
        }
        None => (vec![0u32; n], 1u32),
    };

    let psu_idx: Vec<u32> = match psu {
        Some(p) => {
            let mut map: HashMap<&str, u32> = HashMap::new();
            let mut next = 0u32;
            p.iter()
                .map(|opt| {
                    let key = opt.unwrap_or("__NULL__");
                    *map.entry(key).or_insert_with(|| {
                        let i = next;
                        next += 1;
                        i
                    })
                })
                .collect()
        }
        None => (0..n as u32).collect(),
    };

    let sm = match singleton_method {
        Some(s) if s.eq_ignore_ascii_case("center") || s.eq_ignore_ascii_case("adjust") => true,
        _ => false,
    };

    let mut cov = vec![0.0; k * k];

    for h in 0..n_strata {
        // Collect PSU totals of influence functions within this stratum
        let mut psu_map: HashMap<u32, usize> = HashMap::new();
        let mut psu_totals: Vec<Vec<f64>> = Vec::new();

        for i in 0..n {
            if strata_idx[i] != h {
                continue;
            }
            let pid = psu_idx[i];
            let li = *psu_map.entry(pid).or_insert_with(|| {
                let idx = psu_totals.len();
                psu_totals.push(vec![0.0; k]);
                idx
            });
            for j in 0..k {
                // Weight-scaled influence: infn_i * w_i (matches R svytotal)
                psu_totals[li][j] += influence[i * k + j] * w[i];
            }
        }

        let m = psu_totals.len();
        if m <= 1 {
            if m == 1 && sm {
                // Singleton stratum with centering: would need grand mean
                // For now, skip (matches R's default lonely.psu="fail" -> 0 contrib)
            }
            continue;
        }

        // Compute mean of PSU totals
        let mut mean = vec![0.0; k];
        for t in &psu_totals {
            for j in 0..k {
                mean[j] += t[j];
            }
        }
        for j in 0..k {
            mean[j] /= m as f64;
        }

        // Accumulate variance: m/(m-1) * sum((t_h - tbar_h)(t_h - tbar_h)')
        let scale = m as f64 / (m as f64 - 1.0);
        for a in 0..k {
            for b in 0..k {
                let mut ss = 0.0;
                for t in &psu_totals {
                    ss += (t[a] - mean[a]) * (t[b] - mean[b]);
                }
                cov[a * k + b] += scale * ss;
            }
        }
    }

    Ok(cov)
}
