// src/categorical/ttest.rs
//
// Design-based t-tests for complex survey data.
//
// Two-sample test aligned with R survey::svyttest:
//   R uses svyglm(y ~ group, design, family=gaussian()),
//   which is equivalent to weighted OLS + sandwich variance.
//
// One-sample test uses the existing taylor_mean infrastructure
// (scores_mean + taylor_variance), since R's svyttest for one-sample
// just calls svymean.
//
// For two-sample:
//   1. Fit weighted OLS: y ~ group_indicator
//   2. Compute influence functions
//   3. Design-based SE via taylor_variance on influence functions
//   4. t = coef[1] / SE[1], df = degf(design) - 1

use polars::prelude::*;

use crate::estimation::taylor::{
    degrees_of_freedom, point_estimate_mean, point_estimate_mean_domain, scores_mean,
    scores_mean_domain, taylor_variance,
};
use crate::regression::wols::{fit_wols, influence_se};

// ============================================================================
// Result structures
// ============================================================================

/// Result for a one-sample t-test.
pub struct TTestOneResult {
    /// Estimated mean
    pub estimate: f64,
    /// Design-based SE of the mean
    pub se: f64,
    /// Difference: estimate - null_value
    pub diff: f64,
    /// SE of the difference (same as se for one-sample)
    pub se_diff: f64,
    /// t-statistic: diff / se_diff
    pub t_stat: f64,
    /// Degrees of freedom: degf(design) - 1
    pub df: f64,
    /// p-value
    pub p_value: f64,
    /// Number of observations
    pub n_obs: usize,
}

/// Result for a two-sample t-test.
pub struct TTestTwoResult {
    /// Difference in means (group1 - group0)
    pub diff: f64,
    /// Design-based SE of the difference
    pub se_diff: f64,
    /// t-statistic: diff / se_diff
    pub t_stat: f64,
    /// Degrees of freedom: degf(design) - 1
    pub df: f64,
    /// p-value
    pub p_value: f64,
    /// Group labels
    pub levels: Vec<String>,
    /// Per-group estimates [mean0, mean1]
    pub group_means: Vec<f64>,
    /// Per-group SEs [se0, se1]
    pub group_ses: Vec<f64>,
    /// Number of observations
    pub n_obs: usize,
}

// ============================================================================
// One-sample t-test
// ============================================================================

/// One-sample design-based t-test.
///
/// Matches R's svyttest(y ~ 0, design) or svyttest(y ~ 1, design):
///   tt <- svymean(~y, design)
///   t = coef(tt) / SE(tt)
///   df = degf(design) - 1
pub fn ttest_one_sample(
    y: &Float64Chunked,
    weights: &Float64Chunked,
    strata: Option<&StringChunked>,
    psu: Option<&StringChunked>,
    ssu: Option<&StringChunked>,
    fpc: Option<&Float64Chunked>,
    fpc_ssu: Option<&Float64Chunked>,
    singleton_method: Option<&str>,
    null_value: f64,
) -> PolarsResult<TTestOneResult> {
    let n = y.len();

    // Estimate mean using existing infrastructure
    let estimate = point_estimate_mean(y, weights)?;
    let scores = scores_mean(y, weights)?;
    let variance = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
    let se = variance.max(0.0).sqrt();

    // df = degf(design) - 1
    let design_df = degrees_of_freedom(weights, strata, psu)? as f64;
    let df = (design_df - 1.0).max(1.0);

    // Test
    let diff = estimate - null_value;
    let t_stat = if se > 0.0 { diff / se } else { f64::NAN };
    let p_value = crate::categorical::ranktest::two_sided_t_pvalue(t_stat, df);

    Ok(TTestOneResult {
        estimate,
        se,
        diff,
        se_diff: se,
        t_stat,
        df,
        p_value,
        n_obs: n,
    })
}

/// One-sample t-test within a domain.
pub fn ttest_one_sample_domain(
    y: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
    strata: Option<&StringChunked>,
    psu: Option<&StringChunked>,
    ssu: Option<&StringChunked>,
    fpc: Option<&Float64Chunked>,
    fpc_ssu: Option<&Float64Chunked>,
    singleton_method: Option<&str>,
    null_value: f64,
) -> PolarsResult<TTestOneResult> {
    // Domain-aware mean estimation
    let estimate = point_estimate_mean_domain(y, weights, domain_mask)?;
    let scores = scores_mean_domain(y, weights, domain_mask)?;
    let variance = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
    let se = variance.max(0.0).sqrt();

    // df from FULL design (not domain subset)
    let design_df = degrees_of_freedom(weights, strata, psu)? as f64;
    let df = (design_df - 1.0).max(1.0);

    let diff = estimate - null_value;
    let t_stat = if se > 0.0 { diff / se } else { f64::NAN };
    let p_value = crate::categorical::ranktest::two_sided_t_pvalue(t_stat, df);

    // Count domain observations
    let n_domain = domain_mask.iter().filter(|m| m.unwrap_or(false)).count();

    Ok(TTestOneResult {
        estimate,
        se,
        diff,
        se_diff: se,
        t_stat,
        df,
        p_value,
        n_obs: n_domain,
    })
}

// ============================================================================
// Two-sample t-test
// ============================================================================

/// Two-sample design-based t-test.
///
/// Matches R's svyttest(y ~ group, design):
///   m <- svyglm(y ~ group, design, family=gaussian())
///   t = coef(m)[2] / SE(m)[2]
///   df = m$df.resid
///
/// We implement this via weighted OLS + influence functions + taylor_variance,
/// which is mathematically identical to the sandwich variance from svyglm
/// for Gaussian identity-link models.
pub fn ttest_two_sample(
    y: &[f64],
    g: &[u32], // group indices: 0 or 1
    w: &[f64],
    n: usize,
    strata: Option<&StringChunked>,
    psu: Option<&StringChunked>,
    ssu: Option<&StringChunked>,
    fpc: Option<&Float64Chunked>,
    fpc_ssu: Option<&Float64Chunked>,
    singleton_method: Option<&str>,
    levels: Vec<String>,
    null_value: f64,
) -> PolarsResult<TTestTwoResult> {
    // Build design matrix: [intercept, group_indicator]
    let k = 2;
    let mut xmat = vec![0.0; n * k];
    for i in 0..n {
        xmat[i * k] = 1.0;
        xmat[i * k + 1] = g[i] as f64;
    }

    // Fit weighted OLS
    let wols = fit_wols(y, &xmat, w, n, k).map_err(|e| PolarsError::ComputeError(e.into()))?;

    // Design-based SE via influence functions
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

    // coef[1] = difference in means, SE[1] = design-based SE
    let diff = wols.beta[1] - null_value;
    let se_diff = ses[1];
    let t_stat = if se_diff > 0.0 {
        diff / se_diff
    } else {
        f64::NAN
    };

    // df = degf(design) - 1 (matches R's m$df.resid for svyglm)
    let weights_chunked = Float64Chunked::from_vec("w".into(), w.to_vec());
    let design_df = degrees_of_freedom(&weights_chunked, strata, psu)? as f64;
    let df = (design_df - 1.0).max(1.0);

    let p_value = crate::categorical::ranktest::two_sided_t_pvalue(t_stat, df);

    // Per-group means: beta[0] = mean of group 0, beta[0] + beta[1] = mean of group 1
    let group_means = vec![wols.beta[0], wols.beta[0] + wols.beta[1]];

    // Per-group SEs from domain estimation
    let group_ses =
        compute_per_group_ses(y, g, w, n, strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;

    Ok(TTestTwoResult {
        diff: wols.beta[1], // raw difference (before subtracting null)
        se_diff,
        t_stat,
        df,
        p_value,
        levels,
        group_means,
        group_ses,
        n_obs: n,
    })
}

/// Compute per-group SEs using domain estimation.
/// This gives the correct marginal SE for each group mean.
fn compute_per_group_ses(
    y: &[f64],
    g: &[u32],
    w: &[f64],
    _n: usize,
    strata: Option<&StringChunked>,
    psu: Option<&StringChunked>,
    ssu: Option<&StringChunked>,
    fpc: Option<&Float64Chunked>,
    fpc_ssu: Option<&Float64Chunked>,
    singleton_method: Option<&str>,
) -> PolarsResult<Vec<f64>> {
    let y_chunked = Float64Chunked::from_vec("y".into(), y.to_vec());
    let w_chunked = Float64Chunked::from_vec("w".into(), w.to_vec());

    let mut ses = Vec::new();
    for group_val in 0..2u32 {
        let mask_vec: Vec<bool> = g.iter().map(|&gi| gi == group_val).collect();
        let mask = BooleanChunked::from_slice("mask".into(), &mask_vec);

        let scores = scores_mean_domain(&y_chunked, &w_chunked, &mask)?;
        let var = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
        ses.push(var.max(0.0).sqrt());
    }

    Ok(ses)
}
