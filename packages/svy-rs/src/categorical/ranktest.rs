//! Design-based rank tests (Lumley & Scott 2013, Biometrika 100(4)), matching R's `svyranktest`.
//!
//! - Weighted mid-ranks are scored, regressed on the groups by WLS, and tested with the design variance of the influence.

use crate::categorical::error::{CResult, CategoricalError};
use crate::estimation::error::{EResult, EstimationError};
use polars::prelude::*;

use crate::categorical::replicate::RepSpec;
use crate::estimation::taylor::{degrees_of_freedom, singleton_rows};
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
    pub fn from_str(s: &str) -> EResult<Self> {
        match s.to_lowercase().as_str() {
            "wilcoxon" => Ok(RankScoreMethod::Wilcoxon),
            "vanderwaerden" | "vander_waerden" => Ok(RankScoreMethod::VanDerWaerden),
            "median" => Ok(RankScoreMethod::Median),
            "kruskalwallis" | "kruskal_wallis" => Ok(RankScoreMethod::KruskalWallis),
            _ => Err(EstimationError::InvalidChoice {
                what: "rank score method",
                got: s.to_string(),
                expected: "'wilcoxon', 'vanderwaerden', 'median' or 'kruskalwallis'",
            }),
        }
    }
}

// ============================================================================
// Rank computation
// ============================================================================

/// Estimated population mid-ranks: R's `rankhat[ii] <- ave(cumsum(w[ii]) - w[ii]/2, factor(y[ii]))`.
fn compute_midranks(y: &[f64], w: &[f64], n: usize) -> Vec<f64> {
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| y[a].partial_cmp(&y[b]).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumw = vec![0.0; n];
    let mut running = 0.0;
    for (pos, &idx) in indices.iter().enumerate() {
        running += w[idx];
        cumw[pos] = running - w[idx] / 2.0;
    }

    let mut midrank_sorted = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let y_val = y[indices[i]];
        let mut j = i + 1;
        while j < n && y[indices[j]] == y_val {
            j += 1;
        }
        // Zero-weight rows are not in R's subset design, so they must not join the tie average.
        let (sum, count) = (i..j)
            .filter(|&pos| w[indices[pos]] > 0.0)
            .fold((0.0, 0usize), |(s, c), pos| (s + cumw[pos], c + 1));
        let avg = if count > 0 {
            sum / count as f64
        } else {
            cumw[i..j].iter().sum::<f64>() / (j - i) as f64
        };
        midrank_sorted[i..j].fill(avg);
        i = j;
    }

    let mut rankhat = vec![0.0; n];
    for (pos, &idx) in indices.iter().enumerate() {
        rankhat[idx] = midrank_sorted[pos];
    }
    rankhat
}

fn apply_score(rankhat: &[f64], n_hat: f64, method: RankScoreMethod) -> Vec<f64> {
    match method {
        RankScoreMethod::Wilcoxon | RankScoreMethod::KruskalWallis => {
            rankhat.iter().map(|&r| r / n_hat).collect()
        }
        RankScoreMethod::VanDerWaerden => {
            rankhat
                .iter()
                .map(|&r| {
                    // Clamped so the extreme ranks keep a finite score.
                    let u = (r / n_hat).clamp(1e-10, 1.0 - 1e-10);
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

/// Inverse normal CDF by Acklam's rational approximation (~1.15e-9); the GLM probit link only seeds IRLS with it.
pub(crate) fn probit(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    #[allow(clippy::excessive_precision)] // as published
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
#[allow(dead_code)]
pub struct RankTestTwoResult {
    /// Group 1 minus group 0 mean rank score.
    pub delta: f64,
    pub se: f64,
    pub t_stat: f64,
    /// degf(design) - 1
    pub df: f64,
    pub p_value: f64,
    /// Sorted.
    pub levels: Vec<String>,
    pub group_means: Vec<f64>,
    pub group_ses: Vec<f64>,
    pub n_obs: usize,
}

/// Result for a k-sample rank test.
#[allow(dead_code)]
pub struct RankTestKResult {
    /// k - 1
    pub ndf: usize,
    /// degf(design) - ndf
    pub ddf: f64,
    pub chisq: f64,
    /// chisq / ndf
    pub f_stat: f64,
    pub p_value: f64,
    /// Sorted.
    pub levels: Vec<String>,
    pub group_means: Vec<f64>,
    pub group_ses: Vec<f64>,
    pub n_obs: usize,
}

// ============================================================================
// Two-sample rank test
// ============================================================================

/// Two-sample rank test as R svyranktest: t = coef(m)[2] / SE(svytotal(infn))[2], df = degf(design) - 1.
#[allow(clippy::too_many_arguments)]
pub fn ranktest_two_sample(
    y: &[f64],
    g: &[u32], // group indices: 0 or 1
    w: &[f64],
    n: usize,
    strata: Option<&Column>,
    psu: Option<&Column>,
    ssu: Option<&Column>,
    fpc: Option<&Float64Chunked>,
    fpc_ssu: Option<&Float64Chunked>,
    score_method: RankScoreMethod,
    singleton_method: Option<&str>,
    levels: Vec<String>,
    // The domain's rows for the singleton rules, whatever their weight.
    rows: &[bool],
) -> CResult<RankTestTwoResult> {
    let n_hat: f64 = w.iter().sum();
    let rankhat = compute_midranks(y, w, n);
    let rankscore = apply_score(&rankhat, n_hat, score_method);

    let k = 2;
    let mut xmat = vec![0.0; n * k];
    for i in 0..n {
        xmat[i * k] = 1.0;
        xmat[i * k + 1] = g[i] as f64;
    }

    let wols = fit_wols(&rankscore, &xmat, w, n, k)?;

    let active = singleton_rows(singleton_method, false, strata.is_some(), rows)?;
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
        None,
        active.as_deref(),
    )?;

    let delta = wols.beta[1];
    let se_delta = ses[1];
    let t_stat = if se_delta > 0.0 {
        delta / se_delta
    } else {
        f64::NAN
    };

    let weights_chunked = Float64Chunked::from_slice("w".into(), w);
    let design_df = degrees_of_freedom(&weights_chunked, strata, psu)? as f64;
    let df = (design_df - 1.0).max(1.0);

    let p_value = two_sided_t_pvalue(t_stat, df);

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

/// K-sample rank test as R multiranktest: Wald beta[-1]' V[-1,-1]^{-1} beta[-1], F = chisq / ndf.
#[allow(clippy::too_many_arguments)]
pub fn ranktest_k_sample(
    y: &[f64],
    g: &[u32], // group indices: 0..k-1
    w: &[f64],
    n: usize,
    n_groups: usize,
    strata: Option<&Column>,
    psu: Option<&Column>,
    ssu: Option<&Column>,
    fpc: Option<&Float64Chunked>,
    fpc_ssu: Option<&Float64Chunked>,
    score_method: RankScoreMethod,
    singleton_method: Option<&str>,
    levels: Vec<String>,
    // The domain's rows for the singleton rules, whatever their weight.
    rows: &[bool],
) -> CResult<RankTestKResult> {
    if n_groups < 2 {
        return Err(CategoricalError::TooFewGroups(n_groups));
    }
    let ndf = n_groups - 1;

    let n_hat: f64 = w.iter().sum();
    let rankhat = compute_midranks(y, w, n);
    let rankscore = apply_score(&rankhat, n_hat, score_method);

    let k = 1 + ndf; // group 0 is the reference
    let mut xmat = vec![0.0; n * k];
    for i in 0..n {
        xmat[i * k] = 1.0;
        let gi = g[i] as usize;
        if gi > 0 && gi < n_groups {
            xmat[i * k + gi] = 1.0;
        }
    }

    let wols = fit_wols(&rankscore, &xmat, w, n, k)?;

    let active = singleton_rows(singleton_method, false, strata.is_some(), rows)?;
    let cov_flat = influence_covariance(
        &wols.influence,
        w,
        n,
        k,
        strata,
        psu,
        singleton_method,
        None,
        active.as_deref(),
    )?;

    let beta_test: Vec<f64> = wols.beta[1..].to_vec();
    let mut v_test = vec![0.0; ndf * ndf];
    for a in 0..ndf {
        for b in 0..ndf {
            v_test[a * ndf + b] = cov_flat[(a + 1) * k + (b + 1)];
        }
    }

    let chisq = match crate::regression::wols::solve_kxk(&v_test, &beta_test, ndf) {
        Some(v_inv_beta) => beta_test
            .iter()
            .zip(v_inv_beta.iter())
            .map(|(&b, &vb)| b * vb)
            .sum::<f64>(),
        None => f64::NAN,
    };

    let weights_chunked = Float64Chunked::from_slice("w".into(), w);
    let design_df = degrees_of_freedom(&weights_chunked, strata, psu)? as f64;
    let ddf = (design_df - ndf as f64).max(1.0);

    let f_stat = chisq / ndf as f64;
    let p_value = f_survival(f_stat, ndf as f64, ddf);

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
        None,
        active.as_deref(),
    )?;

    let mut group_means = vec![wols.beta[0]]; // reference group mean
    let mut group_ses = vec![ses[0]];
    for (&b, &se) in wols.beta[1..k].iter().zip(&ses[1..k]) {
        group_means.push(wols.beta[0] + b);
        group_ses.push(se); // approximate
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
// Replication variance
// ============================================================================

/// Rank-score WLS with R's replicate `svytotal(infn)` covariance: re-totals the influence, never re-ranks.
///
/// `score_method = None` means `y` already holds caller-supplied scores.
#[allow(clippy::too_many_arguments)]
fn rep_fit(
    y: &[f64],
    g: &[u32],
    w: &[f64],
    n: usize,
    n_groups: usize,
    score_method: Option<RankScoreMethod>,
    rows: &[bool],
    spec: &RepSpec,
) -> CResult<(Vec<f64>, Vec<Vec<f64>>)> {
    let rankscore = match score_method {
        Some(m) => {
            let n_hat: f64 = w.iter().sum();
            apply_score(&compute_midranks(y, w, n), n_hat, m)
        }
        None => y.to_vec(),
    };
    let k = n_groups;
    let mut xmat = vec![0.0; n * k];
    for i in 0..n {
        xmat[i * k] = 1.0;
        let gi = g[i] as usize;
        if gi > 0 && gi < n_groups {
            xmat[i * k + gi] = 1.0;
        }
    }
    let wols = fit_wols(&rankscore, &xmat, w, n, k)?;
    // Rows outside the domain, or zero-weighted, are not in R's subset design.
    let mask: Vec<bool> = rows.iter().zip(w).map(|(&r, &wi)| r && wi != 0.0).collect();
    let full: Vec<f64> = (0..k)
        .map(|j| {
            (0..n)
                .filter(|&i| mask[i])
                .map(|i| w[i] * wols.influence[i * k + j])
                .sum()
        })
        .collect();
    let cov = spec.cov(&full, &spec.totals(&wols.influence, k, &mask));
    Ok((wols.beta, cov))
}

/// Two-sample rank test on a replicate design (R's svyranktest on svyrep.design); df = replicate df - 1.
#[allow(clippy::too_many_arguments)]
pub fn ranktest_two_sample_rep(
    y: &[f64],
    g: &[u32],
    w: &[f64],
    n: usize,
    score_method: Option<RankScoreMethod>,
    levels: Vec<String>,
    rows: &[bool],
    spec: &RepSpec,
) -> CResult<RankTestTwoResult> {
    let (beta, cov) = rep_fit(y, g, w, n, 2, score_method, rows, spec)?;
    let ses: Vec<f64> = (0..2).map(|j| cov[j][j].max(0.0).sqrt()).collect();
    let delta = beta[1];
    let t_stat = if ses[1] > 0.0 {
        delta / ses[1]
    } else {
        f64::NAN
    };
    let df = (spec.df - 1.0).max(1.0);
    Ok(RankTestTwoResult {
        delta,
        se: ses[1],
        t_stat,
        df,
        p_value: two_sided_t_pvalue(t_stat, df),
        levels,
        group_means: vec![beta[0], beta[0] + beta[1]],
        group_ses: ses,
        n_obs: n,
    })
}

/// K-sample rank test on a replicate design (R's multiranktest); ddf = replicate df - (k - 1).
#[allow(clippy::too_many_arguments)]
pub fn ranktest_k_sample_rep(
    y: &[f64],
    g: &[u32],
    w: &[f64],
    n: usize,
    n_groups: usize,
    score_method: Option<RankScoreMethod>,
    levels: Vec<String>,
    rows: &[bool],
    spec: &RepSpec,
) -> CResult<RankTestKResult> {
    if n_groups < 2 {
        return Err(CategoricalError::TooFewGroups(n_groups));
    }
    let ndf = n_groups - 1;
    let (beta, cov) = rep_fit(y, g, w, n, n_groups, score_method, rows, spec)?;
    let beta_test: Vec<f64> = beta[1..].to_vec();
    let mut v_test = vec![0.0; ndf * ndf];
    for a in 0..ndf {
        for b in 0..ndf {
            v_test[a * ndf + b] = cov[a + 1][b + 1];
        }
    }
    let chisq = match crate::regression::wols::solve_kxk(&v_test, &beta_test, ndf) {
        Some(v_inv_beta) => beta_test
            .iter()
            .zip(v_inv_beta.iter())
            .map(|(&b, &vb)| b * vb)
            .sum(),
        None => f64::NAN,
    };
    let ddf = (spec.df - ndf as f64).max(1.0);
    let f_stat = chisq / ndf as f64;
    let mut group_means = vec![beta[0]];
    let mut group_ses = vec![cov[0][0].max(0.0).sqrt()];
    for j in 1..n_groups {
        group_means.push(beta[0] + beta[j]);
        group_ses.push(cov[j][j].max(0.0).sqrt());
    }
    Ok(RankTestKResult {
        ndf,
        ddf,
        chisq,
        f_stat,
        p_value: f_survival(f_stat, ndf as f64, ddf),
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
    // P(T > t) = 0.5 * I_{df/(df + t^2)}(df/2, 1/2)
    let x = df / (df + t * t);

    regularized_incomplete_beta(x, df / 2.0, 0.5) // This gives 2 * P(T > |t|) directly
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

/// Regularized incomplete beta I_x(a, b) by Lentz's continued fraction.
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

    let max_iter = 200;
    let eps = 1e-14;
    let tiny = 1e-30;

    let mut f: f64;
    let mut c = 1.0_f64;
    let mut d;

    // All b_i = 1 in the standard expansion.
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
