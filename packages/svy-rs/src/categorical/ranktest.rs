// src/categorical/ranktest.rs
//
// Design-based rank tests for complex survey data.
//
// Implements Lumley & Scott (2013), Biometrika 100(4), 831-842.
//
// Algorithm (aligned with R survey::svyranktest):
//   1. Compute estimated population mid-ranks using survey weights.
//   2. Apply rank-score transformation (Wilcoxon, van der Waerden, median).
//   3. Fit weighted OLS: rankscore ~ group (via wols).
//   4. Compute influence functions from the OLS fit.
//   5. Design-based variance via taylor_variance on influence functions.
//   6. Two-sample: t = coef[1] / SE(infn)[1], df = degf(design) - 1.
//      K-sample: Wald F-test on contrast covariance.

use polars::prelude::*;

use crate::estimation::taylor::degrees_of_freedom;
use crate::regression::wols::{fit_wols, influence_covariance, influence_se};

// ============================================================================
// Rank Score Methods
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RankScoreMethod {
    Wilcoxon,      // g(r) = r / N (same as KruskalWallis for 2 groups)
    VanDerWaerden, // g(r) = Φ⁻¹(r / N)
    Median,        // g(r) = I(r > N/2)
    KruskalWallis, // synonym for Wilcoxon
}

impl RankScoreMethod {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "wilcoxon" => Some(RankScoreMethod::Wilcoxon),
            "vanderwaerden" | "vander_waerden" => Some(RankScoreMethod::VanDerWaerden),
            "median" => Some(RankScoreMethod::Median),
            "kruskalwallis" | "kruskal_wallis" => Some(RankScoreMethod::KruskalWallis),
            _ => None,
        }
    }
}

// ============================================================================
// Rank computation
// ============================================================================

/// Compute estimated population mid-ranks.
///
/// Matches R's:
///   ii <- order(y)
///   rankhat[ii] <- ave(cumsum(w[ii]) - w[ii]/2, factor(y[ii]))
fn compute_midranks(y: &[f64], w: &[f64], n: usize) -> Vec<f64> {
    // Sort by y
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| y[a].partial_cmp(&y[b]).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumw = vec![0.0; n];
    let mut running = 0.0;
    for (pos, &idx) in indices.iter().enumerate() {
        running += w[idx];
        cumw[pos] = running - w[idx] / 2.0;
    }

    // Average over ties (same y value)
    let mut midrank_sorted = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let y_val = y[indices[i]];
        let mut j = i + 1;
        while j < n && y[indices[j]] == y_val {
            j += 1;
        }
        // Average midranks for positions i..j
        let avg: f64 = cumw[i..j].iter().sum::<f64>() / (j - i) as f64;
        for pos in i..j {
            midrank_sorted[pos] = avg;
        }
        i = j;
    }

    // Map back to original order
    let mut rankhat = vec![0.0; n];
    for (pos, &idx) in indices.iter().enumerate() {
        rankhat[idx] = midrank_sorted[pos];
    }
    rankhat
}

/// Apply rank-score transformation.
fn apply_score(rankhat: &[f64], n_hat: f64, method: RankScoreMethod) -> Vec<f64> {
    match method {
        RankScoreMethod::Wilcoxon | RankScoreMethod::KruskalWallis => {
            rankhat.iter().map(|&r| r / n_hat).collect()
        }
        RankScoreMethod::VanDerWaerden => {
            rankhat
                .iter()
                .map(|&r| {
                    let u = (r / n_hat).clamp(1e-10, 1.0 - 1e-10);
                    // Probit (inverse normal CDF) - rational approximation
                    probit(u)
                })
                .collect()
        }
        RankScoreMethod::Median => rankhat
            .iter()
            .map(|&r| if r > n_hat / 2.0 { 1.0 } else { 0.0 })
            .collect(),
    }
}

/// High-precision inverse standard normal CDF (probit function).
/// Uses Peter Acklam's rational approximation, accurate to ~1.15e-9.
/// Reference: https://web.archive.org/web/20151030215612/http://home.online.no/~pjacklam/notes/invnorm/
fn probit(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Coefficients for rational approximation
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        // Lower tail
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        // Central region
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        // Upper tail
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

// ============================================================================
// Result structures
// ============================================================================

/// Result for a two-sample rank test.
pub struct RankTestTwoResult {
    /// Difference in mean rank scores (group1 effect)
    pub delta: f64,
    /// Design-based SE of delta
    pub se: f64,
    /// t-statistic: delta / se
    pub t_stat: f64,
    /// Degrees of freedom: degf(design) - 1
    pub df: f64,
    /// Two-sided p-value
    pub p_value: f64,
    /// Group labels (sorted)
    pub levels: Vec<String>,
    /// Per-group mean rank scores
    pub group_means: Vec<f64>,
    /// Per-group SEs
    pub group_ses: Vec<f64>,
    /// Number of observations
    pub n_obs: usize,
}

/// Result for a k-sample rank test.
pub struct RankTestKResult {
    /// Numerator df (k - 1)
    pub ndf: usize,
    /// Denominator df: degf(design) - ndf
    pub ddf: f64,
    /// Wald chi-square statistic
    pub chisq: f64,
    /// F-statistic: chisq / ndf
    pub f_stat: f64,
    /// p-value from F distribution
    pub p_value: f64,
    /// Group labels (sorted)
    pub levels: Vec<String>,
    /// Per-group mean rank scores
    pub group_means: Vec<f64>,
    /// Per-group SEs
    pub group_ses: Vec<f64>,
    /// Number of observations
    pub n_obs: usize,
}

// ============================================================================
// Two-sample rank test
// ============================================================================

/// Two-sample design-based rank test.
///
/// Matches R's svyranktest for 2 groups:
///   1. rankhat[ii] <- ave(cumsum(w[ii]) - w[ii]/2, factor(y[ii]))
///   2. rankscore <- testf(rankhat, N)
///   3. m <- lm(rankscore ~ g, weights = w)
///   4. infn <- (xmat * (rankscore - fitted(m))) %*% summary(m)$cov.unscaled
///   5. tot.infn <- svytotal(infn, design)
///   6. t = coef(m)[2] / SE(tot.infn)[2]
///   7. df = degf(design) - 1
pub fn ranktest_two_sample(
    y: &[f64],
    g: &[u32], // group indices: 0 or 1
    w: &[f64],
    n: usize,
    strata: Option<&StringChunked>,
    psu: Option<&StringChunked>,
    ssu: Option<&StringChunked>,
    fpc: Option<&Float64Chunked>,
    fpc_ssu: Option<&Float64Chunked>,
    score_method: RankScoreMethod,
    singleton_method: Option<&str>,
    levels: Vec<String>,
) -> PolarsResult<RankTestTwoResult> {
    // 1 & 2. Compute rank scores
    let n_hat: f64 = w.iter().sum();
    let rankhat = compute_midranks(y, w, n);
    let rankscore = apply_score(&rankhat, n_hat, score_method);

    // 3. Build design matrix: [1, indicator(g == 1)]
    let k = 2;
    let mut xmat = vec![0.0; n * k];
    for i in 0..n {
        xmat[i * k] = 1.0; // intercept
        xmat[i * k + 1] = g[i] as f64; // group indicator
    }

    // Fit weighted OLS
    let wols =
        fit_wols(&rankscore, &xmat, w, n, k).map_err(|e| PolarsError::ComputeError(e.into()))?;

    // 4 & 5. Design-based SE via influence functions + taylor_variance
    let ses = influence_se(
        &wols.influence,
        w,
        n,
        k,
        strata,
        psu,
        ssu,
        fpc,
        fpc_ssu,
        singleton_method,
    )?;

    // 6. Test statistic
    let delta = wols.beta[1];
    let se_delta = ses[1];
    let t_stat = if se_delta > 0.0 {
        delta / se_delta
    } else {
        f64::NAN
    };

    // 7. Degrees of freedom
    let weights_chunked = Float64Chunked::from_vec("w".into(), w.to_vec());
    let design_df = degrees_of_freedom(&weights_chunked, strata, psu)? as f64;
    let df = (design_df - 1.0).max(1.0);

    // p-value (two-sided, t-distribution)
    let p_value = two_sided_t_pvalue(t_stat, df);

    // Per-group means: beta[0] = mean of group 0, beta[0] + beta[1] = mean of group 1
    let group_means = vec![wols.beta[0], wols.beta[0] + wols.beta[1]];
    let group_ses = vec![ses[0], ses[1]]; // approximate

    Ok(RankTestTwoResult {
        delta,
        se: se_delta,
        t_stat,
        df,
        p_value,
        levels,
        group_means,
        group_ses,
        n_obs: n,
    })
}

// ============================================================================
// K-sample rank test
// ============================================================================

/// K-sample design-based rank test (Kruskal-Wallis style).
///
/// Matches R's multiranktest:
///   1. Compute rank scores
///   2. m <- glm(rankscore ~ factor(g), weights = w)
///   3. V <- svy.varcoef(m, design)
///   4. Wald test: beta[-1]' V[-1,-1]^{-1} beta[-1]
///   5. F = chisq / ndf, p from F(ndf, ddf)
pub fn ranktest_k_sample(
    y: &[f64],
    g: &[u32], // group indices: 0..k-1
    w: &[f64],
    n: usize,
    n_groups: usize,
    strata: Option<&StringChunked>,
    psu: Option<&StringChunked>,
    ssu: Option<&StringChunked>,
    fpc: Option<&Float64Chunked>,
    fpc_ssu: Option<&Float64Chunked>,
    score_method: RankScoreMethod,
    singleton_method: Option<&str>,
    levels: Vec<String>,
) -> PolarsResult<RankTestKResult> {
    let ndf = n_groups - 1;

    // 1 & 2. Compute rank scores
    let n_hat: f64 = w.iter().sum();
    let rankhat = compute_midranks(y, w, n);
    let rankscore = apply_score(&rankhat, n_hat, score_method);

    // 3. Build design matrix: intercept + (k-1) dummy indicators
    // Reference group = 0
    let k = 1 + ndf; // intercept + (n_groups - 1) dummies
    let mut xmat = vec![0.0; n * k];
    for i in 0..n {
        xmat[i * k] = 1.0; // intercept
        let gi = g[i] as usize;
        if gi > 0 && gi < n_groups {
            xmat[i * k + gi] = 1.0; // dummy for group gi
        }
    }

    // Fit weighted OLS
    let wols =
        fit_wols(&rankscore, &xmat, w, n, k).map_err(|e| PolarsError::ComputeError(e.into()))?;

    // 4. Design-based covariance of coefficients
    let cov_flat = influence_covariance(&wols.influence, w, n, k, strata, psu, singleton_method)?;

    // 5. Wald test on non-intercept coefficients
    // Extract beta[-1] and V[-1,-1]
    let beta_test: Vec<f64> = wols.beta[1..].to_vec();
    let mut v_test = vec![0.0; ndf * ndf];
    for a in 0..ndf {
        for b in 0..ndf {
            v_test[a * ndf + b] = cov_flat[(a + 1) * k + (b + 1)];
        }
    }

    // chisq = beta' V^{-1} beta
    let chisq = match crate::regression::wols::solve_kxk(&v_test, &beta_test, ndf) {
        Some(v_inv_beta) => beta_test
            .iter()
            .zip(v_inv_beta.iter())
            .map(|(&b, &vb)| b * vb)
            .sum::<f64>(),
        None => f64::NAN,
    };

    // Degrees of freedom
    let weights_chunked = Float64Chunked::from_vec("w".into(), w.to_vec());
    let design_df = degrees_of_freedom(&weights_chunked, strata, psu)? as f64;
    let ddf = (design_df - ndf as f64).max(1.0);

    // F-statistic and p-value
    let f_stat = chisq / ndf as f64;
    let p_value = f_survival(f_stat, ndf as f64, ddf);

    // Per-group means and SEs
    let ses = influence_se(
        &wols.influence,
        w,
        n,
        k,
        strata,
        psu,
        ssu,
        fpc,
        fpc_ssu,
        singleton_method,
    )?;

    let mut group_means = vec![wols.beta[0]]; // reference group mean
    let mut group_ses = vec![ses[0]];
    for j in 1..k {
        group_means.push(wols.beta[0] + wols.beta[j]);
        group_ses.push(ses[j]); // approximate
    }

    Ok(RankTestKResult {
        ndf,
        ddf,
        chisq,
        f_stat,
        p_value,
        levels,
        group_means,
        group_ses,
        n_obs: n,
    })
}

// ============================================================================
// Distribution helpers
// ============================================================================

/// Two-sided p-value from t-distribution using the regularized incomplete beta function.
pub fn two_sided_t_pvalue(t: f64, df: f64) -> f64 {
    if t.is_nan() || df <= 0.0 {
        return f64::NAN;
    }
    // P(|T| > |t|) = 2 * P(T > |t|) for symmetric t-distribution
    // Using the relationship: P(T > t) = 0.5 * I_{df/(df + t^2)}(df/2, 1/2)
    let x = df / (df + t * t);
    let p = regularized_incomplete_beta(x, df / 2.0, 0.5);
    p // This gives 2 * P(T > |t|) directly
}

/// Survival function of F distribution: P(F > x) where F ~ F(d1, d2).
pub fn f_survival(x: f64, d1: f64, d2: f64) -> f64 {
    if x.is_nan() || x < 0.0 || d1 <= 0.0 || d2 <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    // P(F > x) = I_{d2/(d2 + d1*x)}(d2/2, d1/2)
    let z = d2 / (d2 + d1 * x);
    regularized_incomplete_beta(z, d2 / 2.0, d1 / 2.0)
}

/// Regularized incomplete beta function I_x(a, b) using continued fraction.
/// Uses Lentz's algorithm for the continued fraction expansion.
fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Use symmetry: I_x(a,b) = 1 - I_{1-x}(b,a) when x > (a+1)/(a+b+2)
    let threshold = (a + 1.0) / (a + b + 2.0);
    if x > threshold {
        return 1.0 - regularized_incomplete_beta(1.0 - x, b, a);
    }

    // Log of the prefactor: x^a * (1-x)^b / (a * B(a,b))
    let log_prefix = a * x.ln() + b * (1.0 - x).ln() - (a.ln() + ln_beta(a, b));
    let prefix = log_prefix.exp();

    // Continued fraction (Lentz's method)
    let max_iter = 200;
    let eps = 1e-14;
    let tiny = 1e-30;

    let mut f: f64;
    let mut c = 1.0_f64;
    let mut d;

    // a_1 / (b_1 + a_2 / (b_2 + ...))
    // All b_i = 1 for the standard expansion
    d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    f = d;

    for m in 1..=max_iter {
        let mf = m as f64;

        // Even step: a_{2m}
        let a_even = mf * (b - mf) * x / ((a + 2.0 * mf - 1.0) * (a + 2.0 * mf));
        d = 1.0 + a_even * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + a_even / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        f *= c * d;

        // Odd step: a_{2m+1}
        let a_odd = -((a + mf) * (a + b + mf) * x) / ((a + 2.0 * mf) * (a + 2.0 * mf + 1.0));
        d = 1.0 + a_odd * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + a_odd / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;

        if (delta - 1.0).abs() < eps {
            break;
        }
    }

    prefix * f
}

/// Log of the beta function: ln(B(a, b)) = ln(Gamma(a)) + ln(Gamma(b)) - ln(Gamma(a+b))
fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Lanczos approximation to ln(Gamma(x)) for x > 0.
fn ln_gamma(x: f64) -> f64 {
    let coeffs = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.001208650973866179,
        -0.000005395239384953,
    ];
    let y = x;
    let mut tmp = x + 5.5;
    tmp -= (x + 0.5) * tmp.ln();
    let mut ser = 1.000000000190015;
    for (j, &c) in coeffs.iter().enumerate() {
        ser += c / (y + 1.0 + j as f64);
    }
    -tmp + (2.5066282746310005 * ser / x).ln()
}
