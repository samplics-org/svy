// src/categorical/tabulation.rs
//!
//! Design-based categorical tabulation with Rao-Scott corrections.
//!
//! Computes one-way or two-way frequency tables with:
//!   - Proportion estimates and Taylor-linearized SEs
//!   - Optional total estimates (when weights are not normalized)
//!   - Rao-Scott adjusted Pearson chi-square test (two-way)
//!
//! Matches R's `survey::svytable` + `survey::svychisq`.

use faer::Mat;
use faer::prelude::Reborrow;
use polars::prelude::*;

use crate::estimation::{
    degrees_of_freedom, point_estimate_mean, point_estimate_total, scores_mean, scores_total,
    srs_variance_mean, taylor_variance,
};

// ============================================================================
// Numeric-aware sorting (matches Python's _numsort_key)
// ============================================================================

fn numsort_key(s: &str) -> (u8, f64, String) {
    let trimmed = s.trim();
    if let Ok(v) = trimmed.parse::<f64>() {
        (0, v, String::new())
    } else {
        (2, 0.0, s.to_string())
    }
}

pub fn sort_levels(levels: &mut Vec<String>) {
    levels.sort_by(|a, b| {
        let ka = numsort_key(a);
        let kb = numsort_key(b);
        ka.0.cmp(&kb.0)
            .then(ka.1.partial_cmp(&kb.1).unwrap_or(std::cmp::Ordering::Equal))
            .then(ka.2.cmp(&kb.2))
    });
}

// ============================================================================
// Distribution functions (chi-square and F survival)
// ============================================================================

/// Chi-square survival function: P(X > x) where X ~ chi2(df).
fn chi2_survival(x: f64, df: f64) -> f64 {
    if x <= 0.0 || df <= 0.0 {
        return 1.0;
    }
    1.0 - regularized_lower_gamma(df / 2.0, x / 2.0)
}

/// F-distribution survival function: P(X > x) where X ~ F(d1, d2).
fn f_survival(x: f64, d1: f64, d2: f64) -> f64 {
    if x <= 0.0 || d1 <= 0.0 || d2 <= 0.0 {
        return 1.0;
    }
    let z = d1 * x / (d1 * x + d2);
    1.0 - regularized_beta(z, d1 / 2.0, d2 / 2.0)
}

fn regularized_lower_gamma(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        gamma_series(a, x)
    } else {
        1.0 - gamma_cf(a, x)
    }
}

fn gamma_series(a: f64, x: f64) -> f64 {
    let mut sum = 1.0 / a;
    let mut term = 1.0 / a;
    for n in 1..200 {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < sum.abs() * 1e-14 {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

fn gamma_cf(a: f64, x: f64) -> f64 {
    let mut f = 1.0 / (x + 1.0 - a).max(1e-30);
    let mut c = 1e30_f64;
    let mut d = f;
    for n in 1..200 {
        let an = -(n as f64) * (n as f64 - a);
        let bn = x + 2.0 * n as f64 + 1.0 - a;
        d = bn + an * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        d = 1.0 / d;
        c = bn + an / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        let delta = d * c;
        f *= delta;
        if (delta - 1.0).abs() < 1e-14 {
            break;
        }
    }
    f * (-x + a * x.ln() - ln_gamma(a)).exp()
}

fn ln_gamma(x: f64) -> f64 {
    let coeffs = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.001208650973866179,
        -0.000005395239384953,
    ];
    let mut y = x;
    let mut tmp = x + 5.5;
    tmp -= (x + 0.5) * tmp.ln();
    let mut ser = 1.000000000190015;
    for &c in &coeffs {
        y += 1.0;
        ser += c / y;
    }
    -tmp + (2.5066282746310005 * ser / x).ln()
}

fn regularized_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - ln_beta).exp() / a;
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_cf(x, a, b)
    } else {
        let front2 = (b * (1.0 - x).ln() + a * x.ln() - ln_beta).exp() / b;
        1.0 - front2 * beta_cf(1.0 - x, b, a)
    }
}

fn beta_cf(x: f64, a: f64, b: f64) -> f64 {
    let mut c = 1.0_f64;
    let d0 = 1.0 - (a + b) * x / (a + 1.0);
    let mut d = if d0.abs() < 1e-30 { 1e30 } else { 1.0 / d0 };
    let mut h = d;
    for m in 1..200 {
        let mf = m as f64;
        let num = mf * (b - mf) * x / ((a + 2.0 * mf - 1.0) * (a + 2.0 * mf));
        d = 1.0 + num * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        d = 1.0 / d;
        c = 1.0 + num / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        h *= d * c;

        let num2 = -(a + mf) * (a + b + mf) * x / ((a + 2.0 * mf) * (a + 2.0 * mf + 1.0));
        d = 1.0 + num2 * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        d = 1.0 / d;
        c = 1.0 + num2 / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 1e-14 {
            break;
        }
    }
    h
}

// ============================================================================
// Proportion estimation with covariance matrix
// ============================================================================

/// Compute proportion estimates and full covariance matrix for all levels.
///
/// Uses a single multivariate Taylor variance computation (matching R's approach)
/// rather than separate per-level calls + polarization identity.
///
/// Returns (levels, proportions, SEs, covariance_matrix, deff_vec, df).
pub fn estimate_proportions(
    y: &StringChunked,
    weights: &Float64Chunked,
    strata: Option<&StringChunked>,
    psu: Option<&StringChunked>,
    ssu: Option<&StringChunked>,
    fpc: Option<&Float64Chunked>,
    fpc_ssu: Option<&Float64Chunked>,
    singleton_method: Option<&str>,
) -> PolarsResult<(
    Vec<String>,
    Vec<f64>,
    Vec<f64>,
    Vec<Vec<f64>>,
    Vec<f64>,
    u32,
)> {
    let mut levels: Vec<String> = y
        .unique()?
        .iter()
        .filter_map(|v| v.map(|s| s.to_string()))
        .collect();
    sort_levels(&mut levels);

    let k = levels.len();
    let df_val = degrees_of_freedom(weights, strata, psu)?;

    let sm = singleton_method;

    // Build all k indicator columns in a SINGLE pass over y.
    // Previously: k separate passes (one per level). Now: one pass, O(N) regardless of k.
    let n_rows = y.len();
    // level_idx[i] = index of level that row i belongs to, or u32::MAX for null
    let level_map: std::collections::HashMap<&str, usize> = levels
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();

    // indicators[j][i] = value for level j at row i (0.0, 1.0, or NaN-sentinel None)
    let mut indicators: Vec<Vec<Option<f64>>> = vec![vec![Some(0.0); n_rows]; k];
    for (i, opt_val) in y.iter().enumerate() {
        match opt_val {
            None => {
                for j in 0..k {
                    indicators[j][i] = None;
                }
            }
            Some(val) => {
                if let Some(&j) = level_map.get(val) {
                    indicators[j][i] = Some(1.0);
                }
                // rows with a value not in levels stay 0.0 (already initialised)
            }
        }
    }

    let mut proportions = Vec::with_capacity(k);
    let mut score_columns: Vec<Float64Chunked> = Vec::with_capacity(k);
    let mut deff_vec = Vec::with_capacity(k);

    for (j, _lvl) in levels.iter().enumerate() {
        let ind_ca = Float64Chunked::from_slice_options("ind".into(), &indicators[j]);

        let est = point_estimate_mean(&ind_ca, weights)?;
        let scores = scores_mean(&ind_ca, weights)?;

        let var_scalar = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_ssu, sm)?;
        let srs_var = srs_variance_mean(&ind_ca, weights)?;
        let deff = if srs_var > 0.0 {
            var_scalar / srs_var
        } else {
            f64::NAN
        };

        proportions.push(est);
        score_columns.push(scores);
        deff_vec.push(deff);
    }

    // Compute full k×k covariance matrix using taylor_variance_matrix
    // This uses the SAME PSU indexing and variance formula as taylor_variance
    use crate::estimation::taylor_variance_matrix;
    let cov = taylor_variance_matrix(&score_columns, strata, psu, fpc, sm)?;

    // Extract SEs from diagonal
    let ses: Vec<f64> = (0..k).map(|j| cov[j][j].max(0.0).sqrt()).collect();

    Ok((levels, proportions, ses, cov, deff_vec, df_val))
}

// ============================================================================
// Total estimation
// ============================================================================

/// Compute weighted totals and SEs for each level.
pub fn estimate_totals(
    y: &StringChunked,
    weights: &Float64Chunked,
    strata: Option<&StringChunked>,
    psu: Option<&StringChunked>,
    ssu: Option<&StringChunked>,
    fpc: Option<&Float64Chunked>,
    fpc_ssu: Option<&Float64Chunked>,
    singleton_method: Option<&str>,
    levels: &[String],
) -> PolarsResult<(Vec<f64>, Vec<f64>)> {
    let mut totals = Vec::with_capacity(levels.len());
    let mut total_ses = Vec::with_capacity(levels.len());

    // Single pass over y to build all indicator columns at once
    let n_rows = y.len();
    let level_map: std::collections::HashMap<&str, usize> = levels
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    let kk = levels.len();
    let mut indicators: Vec<Vec<Option<f64>>> = vec![vec![Some(0.0); n_rows]; kk];
    for (i, opt_val) in y.iter().enumerate() {
        match opt_val {
            None => { for j in 0..kk { indicators[j][i] = None; } }
            Some(val) => {
                if let Some(&j) = level_map.get(val) {
                    indicators[j][i] = Some(1.0);
                }
            }
        }
    }

    for j in 0..kk {
        let ind_ca = Float64Chunked::from_slice_options("ind".into(), &indicators[j]);
        let est = point_estimate_total(&ind_ca, weights)?;
        let scores = scores_total(&ind_ca, weights)?;
        let var = taylor_variance(&scores, strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
        totals.push(est);
        total_ses.push(var.max(0.0).sqrt());
    }

    Ok((totals, total_ses))
}

// ============================================================================
// Rao-Scott chi-square adjustments
// ============================================================================

/// Compute Rao-Scott adjusted chi-square statistics for a two-way table.
/// Returns the 7 Pearson-based statistics: (chisq, df, p, adj_f, adj_ndf, adj_ddf, adj_p).
pub fn rao_scott(
    proportions: &[f64],
    cov_survey: &[Vec<f64>],
    nr: usize,
    nc: usize,
    n_obs: usize,
    n_strata: usize,
    n_psus: usize,
) -> (f64, f64, f64, f64, f64, f64, f64) {
    let k = nr * nc;
    let n_f = n_obs as f64;
    let p = proportions;

    // 1. Design matrices: X1 (main effects), X2 (interactions)
    let x1_cols = 1 + (nr - 1) + (nc - 1);
    let x2_cols = (nr - 1) * (nc - 1);

    let mut x1 = Mat::<f64>::zeros(k, x1_cols);
    let mut x2 = Mat::<f64>::zeros(k, x2_cols);

    for r in 0..nr {
        for c in 0..nc {
            let i = r * nc + c;
            x1[(i, 0)] = 1.0;
            if r > 0 {
                x1[(i, r)] = 1.0;
            }
            if c > 0 {
                x1[(i, nr - 1 + c)] = 1.0;
            }
            if r > 0 && c > 0 {
                x2[(i, (r - 1) * (nc - 1) + (c - 1))] = 1.0;
            }
        }
    }

    // 2. Survey covariance V (from design-based estimation)
    let mut v_surv = Mat::<f64>::zeros(k, k);
    for i in 0..k {
        for j in 0..k {
            v_surv[(i, j)] = cov_survey[i][j];
        }
    }

    // 3. Build full interaction model matrix X12 = [X1 | X2]
    //    Then extract Cmat = QR residuals of X2 columns projected off X1
    //    This matches R: Cmat <- qr.resid(qr(X1), X12[,-(1:(nr+nc-1))])
    let x1_qr = x1.rb().thin_svd().unwrap();
    let x1_proj = {
        // Q * Q' projection matrix (via thin SVD of X1)
        let u = x1_qr.U();
        &u * u.transpose()
    };
    // Cmat = (I - X1 (X1'X1)^{-1} X1') X2 = X2 - proj(X1) X2
    let cmat = &x2 - &x1_proj * &x2;

    // 4. Build iDmat = diag(1/p), with 0 for zero proportions (matches R)
    let mut idmat = Mat::<f64>::zeros(k, k);
    for i in 0..k {
        if p[i] != 0.0 {
            idmat[(i, i)] = 1.0 / p[i];
        }
    }

    // 5. Delta computation (matches R lines 100-102):
    //    denom = Cmat' * (iDmat / N) * Cmat
    //    numr  = Cmat' * iDmat * V * iDmat * Cmat
    //    Delta = solve(denom, numr)
    let idmat_over_n = {
        let mut m = idmat.clone();
        for i in 0..k {
            m[(i, i)] = m[(i, i)] / n_f;
        }
        m
    };
    let cmat_t = cmat.transpose();
    let denom = &cmat_t * &idmat_over_n * &cmat;
    let numr = &cmat_t * &idmat * &v_surv * &idmat * &cmat;
    let denom_inv = denom.rb().thin_svd().unwrap().pseudoinverse();
    let delta = &denom_inv * &numr;

    // 6. Chi-square statistics
    let mut row_margin = vec![0.0; nr];
    let mut col_margin = vec![0.0; nc];
    for r in 0..nr {
        for c in 0..nc {
            row_margin[r] += p[r * nc + c];
            col_margin[c] += p[r * nc + c];
        }
    }

    let mut chisq_p = 0.0;
    for r in 0..nr {
        for c in 0..nc {
            let i = r * nc + c;
            let expected = row_margin[r] * col_margin[c];
            if expected > 0.0 {
                chisq_p += (p[i] - expected).powi(2) / expected;
            }
        }
    }
    chisq_p *= n_f;

    // 7. Rao-Scott corrections
    let df_base = ((nr - 1) * (nc - 1)) as f64;
    let dd = delta.nrows().min(delta.ncols());
    let trace_d: f64 = (0..dd).map(|i| delta[(i, i)]).sum();
    let d2 = &delta * &delta;
    let trace_d2: f64 = (0..dd).map(|i| d2[(i, i)]).sum();

    // First-order Rao-Scott correction (matches R's svychisq statistic="Chisq"):
    //   R reports the RAW X² as the displayed "X-squared" value, but computes
    //   the p-value from X²/mean(delta) evaluated against chi2(df_base).
    let mean_delta = if df_base > 0.0 && trace_d > 1e-9 {
        trace_d / df_base
    } else {
        1.0
    };
    let chisq_p_for_p = chisq_p / mean_delta;

    // Second-order (F) correction: F = X² / trace(delta), with Satterthwaite df
    // R only computes F from the Pearson X² (see surveychisq.R line 111).
    let (f_p, ndf, ddf) = if trace_d > 1e-9 {
        let fp = chisq_p / trace_d;
        let nd = trace_d.powi(2) / trace_d2;
        let dd_val = (n_psus - n_strata) as f64 * nd;
        (fp, nd, dd_val)
    } else {
        (0.0, 0.0, 0.0)
    };

    (
        chisq_p,
        df_base,
        chi2_survival(chisq_p_for_p, df_base),
        f_p,
        ndf,
        ddf,
        f_survival(f_p, ndf, ddf),
    )
}

/// Count unique strata and PSUs for Rao-Scott df calculation.
pub fn count_strata_psus(
    strata: Option<&StringChunked>,
    psu: Option<&StringChunked>,
    n: usize,
) -> (usize, usize) {
    match (strata, psu) {
        (Some(s), Some(p)) => {
            let n_strata = s.unique().map(|u| u.len()).unwrap_or(1);
            let mut pairs = std::collections::HashSet::new();
            for (si, pi) in s.iter().zip(p.iter()) {
                if let (Some(sv), Some(pv)) = (si, pi) {
                    pairs.insert((sv.to_string(), pv.to_string()));
                }
            }
            (n_strata, pairs.len())
        }
        (None, Some(p)) => (1, p.unique().map(|u| u.len()).unwrap_or(n)),
        (Some(s), None) => (s.unique().map(|u| u.len()).unwrap_or(1), n),
        (None, None) => (1, n),
    }
}
