//! Design-based one- and two-way tabulation with the Rao-Scott chi-square test.
//!
//! - Matches R's `survey::svytable` + `survey::svychisq`.

use crate::categorical::error::CResult;
use faer::Mat;
use faer::prelude::Reborrow;
use polars::prelude::*;

use crate::estimation::calib_sweep::{CalibSweep, sweep_scores};
use crate::estimation::{
    degrees_of_freedom, point_estimate_mean, point_estimate_total, scores_mean, scores_total,
    singleton_domain_rows, srs_variance_mean, taylor::SrsRef, taylor_variance_in,
    taylor_variance_matrix_in,
};

/// Levels, proportions, SEs, covariance matrix, deffs and df.
type PropTable = (
    Vec<String>,
    Vec<f64>,
    Vec<f64>,
    Vec<Vec<f64>>,
    Vec<f64>,
    u32,
);

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

pub fn sort_levels(levels: &mut [String]) {
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

/// Proportions of every level with one joint Taylor covariance, as R does (no per-level polarization).
#[allow(clippy::too_many_arguments)]
pub fn estimate_proportions(
    y: &StringChunked,
    weights: &Float64Chunked,
    strata: Option<&Column>,
    psu: Option<&Column>,
    ssu: Option<&Column>,
    fpc: Option<&Float64Chunked>,
    fpc_ssu: Option<&Float64Chunked>,
    singleton_method: Option<&str>,
    calib: Option<&CalibSweep>,
    domain: Option<&BooleanChunked>,
) -> CResult<PropTable> {
    let levels = domain_levels(y, domain)?;

    let k = levels.len();
    let df_val = degrees_of_freedom(weights, strata, psu)?;

    let sm = singleton_method;
    let active =
        singleton_domain_rows(sm, calib.is_some(), strata.is_some(), weights.len(), domain)?;

    let n_rows = y.len();
    let level_map: std::collections::HashMap<&str, usize> = levels
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();

    // indicators[j][i]: None marks a missing key.
    let mut indicators: Vec<Vec<Option<f64>>> = vec![vec![Some(0.0); n_rows]; k];
    for (i, opt_val) in y.iter().enumerate() {
        // Out-of-domain rows stay at 0 to keep the PSU structure; a null key there is not missing data.
        if !in_domain(domain, i) {
            continue;
        }
        match opt_val {
            None => {
                for ind in indicators.iter_mut().take(k) {
                    ind[i] = None;
                }
            }
            Some(val) => {
                if let Some(&j) = level_map.get(val) {
                    indicators[j][i] = Some(1.0);
                }
            }
        }
    }

    let mut proportions = Vec::with_capacity(k);
    let mut score_columns: Vec<Float64Chunked> = Vec::with_capacity(k);
    let mut deff_vec = Vec::with_capacity(k);

    for (j, _lvl) in levels.iter().enumerate() {
        let ind_ca = Float64Chunked::from_slice_options("ind".into(), &indicators[j]);

        let est = point_estimate_mean(&ind_ca, weights)?;
        let scores = sweep_scores(&scores_mean(&ind_ca, weights)?, calib);

        let var_scalar = taylor_variance_in(
            &scores,
            strata,
            psu,
            ssu,
            fpc,
            fpc_ssu,
            sm,
            active.as_deref(),
        )?;
        let srs_var = srs_variance_mean(
            &ind_ca,
            weights,
            SrsRef::WithoutReplacement { pop_total: None },
        )?;
        let deff = if srs_var > 0.0 {
            var_scalar / srs_var
        } else {
            f64::NAN
        };

        proportions.push(est);
        score_columns.push(scores);
        deff_vec.push(deff);
    }

    let cov = taylor_variance_matrix_in(&score_columns, strata, psu, fpc, sm, active.as_deref())?;

    let ses: Vec<f64> = (0..k).map(|j| cell_se(proportions[j], cov[j][j])).collect();

    Ok((levels, proportions, ses, cov, deff_vec, df_val))
}

/// SE of a cell share, NaN when the share is: `f64::max` turns a NaN variance into 0.
pub fn cell_se(estimate: f64, variance: f64) -> f64 {
    if estimate.is_nan() {
        f64::NAN
    } else {
        variance.max(0.0).sqrt()
    }
}

/// Row `i` is in the tabulation domain (every row when there is no mask).
#[inline]
pub fn in_domain(domain: Option<&BooleanChunked>, i: usize) -> bool {
    domain.is_none_or(|m| m.get(i).unwrap_or(false))
}

/// Sorted distinct in-domain keys: an out-of-domain row's category never forms a cell.
pub fn domain_levels(y: &StringChunked, domain: Option<&BooleanChunked>) -> CResult<Vec<String>> {
    let mut levels: Vec<String> = match domain {
        None => y
            .unique()?
            .iter()
            .filter_map(|v| v.map(|s| s.to_string()))
            .collect(),
        Some(m) => y
            .filter(m)?
            .unique()?
            .iter()
            .filter_map(|v| v.map(|s| s.to_string()))
            .collect(),
    };
    sort_levels(&mut levels);
    Ok(levels)
}

// ============================================================================
// Total estimation
// ============================================================================

/// Compute weighted totals and SEs for each level.
#[allow(clippy::too_many_arguments)]
pub fn estimate_totals(
    y: &StringChunked,
    weights: &Float64Chunked,
    strata: Option<&Column>,
    psu: Option<&Column>,
    ssu: Option<&Column>,
    fpc: Option<&Float64Chunked>,
    fpc_ssu: Option<&Float64Chunked>,
    singleton_method: Option<&str>,
    calib: Option<&CalibSweep>,
    levels: &[String],
    domain: Option<&BooleanChunked>,
) -> CResult<(Vec<f64>, Vec<f64>)> {
    let mut totals = Vec::with_capacity(levels.len());
    let mut total_ses = Vec::with_capacity(levels.len());
    let active = singleton_domain_rows(
        singleton_method,
        calib.is_some(),
        strata.is_some(),
        weights.len(),
        domain,
    )?;

    let n_rows = y.len();
    let level_map: std::collections::HashMap<&str, usize> = levels
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    let kk = levels.len();
    let mut indicators: Vec<Vec<Option<f64>>> = vec![vec![Some(0.0); n_rows]; kk];
    for (i, opt_val) in y.iter().enumerate() {
        if !in_domain(domain, i) {
            continue;
        }
        match opt_val {
            None => {
                for ind in indicators.iter_mut().take(kk) {
                    ind[i] = None;
                }
            }
            Some(val) => {
                if let Some(&j) = level_map.get(val) {
                    indicators[j][i] = Some(1.0);
                }
            }
        }
    }

    for ind in indicators.iter().take(kk) {
        let ind_ca = Float64Chunked::from_slice_options("ind".into(), ind);
        let est = point_estimate_total(&ind_ca, weights)?;
        let scores = sweep_scores(&scores_total(&ind_ca, weights)?, calib);
        let var = taylor_variance_in(
            &scores,
            strata,
            psu,
            ssu,
            fpc,
            fpc_ssu,
            singleton_method,
            active.as_deref(),
        )?;
        totals.push(est);
        total_ses.push(var.max(0.0).sqrt());
    }

    Ok((totals, total_ses))
}

// ============================================================================
// Rao-Scott chi-square adjustments
// ============================================================================

/// Rao-Scott Pearson statistics: (chisq, df, p, adj_f, adj_ndf, adj_ddf, adj_p).
pub fn rao_scott(
    proportions: &[f64],
    cov_survey: &[Vec<f64>],
    nr: usize,
    nc: usize,
    n_obs: usize,
    // Design df: #PSUs - #strata for Taylor, the replicate df for replication.
    nu: f64,
) -> (f64, f64, f64, f64, f64, f64, f64) {
    let k = nr * nc;
    let n_f = n_obs as f64;
    let p = proportions;
    let df_base = (nr.saturating_sub(1) * nc.saturating_sub(1)) as f64;
    let nan = f64::NAN;

    // One row or column level, or a weightless domain (NaN proportions): nothing to test.
    if df_base == 0.0 || p.iter().any(|v| !v.is_finite()) {
        return (nan, df_base, nan, nan, nan, nan, nan);
    }

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

    // 3. R: Cmat <- qr.resid(qr(X1), X12[,-(1:(nr+nc-1))])
    let Ok(x1_qr) = x1.rb().thin_svd() else {
        return (nan, df_base, nan, nan, nan, nan, nan);
    };
    let x1_proj = {
        // Q * Q' projection matrix (via thin SVD of X1)
        let u = x1_qr.U();
        u * u.transpose()
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

    // 5. As R: Delta = solve(Cmat' (iDmat/N) Cmat, Cmat' iDmat V iDmat Cmat)
    let idmat_over_n = {
        let mut m = idmat.clone();
        for i in 0..k {
            m[(i, i)] /= n_f;
        }
        m
    };
    let cmat_t = cmat.transpose();
    let denom = cmat_t * &idmat_over_n * &cmat;
    let numr = cmat_t * &idmat * &v_surv * &idmat * &cmat;
    let Ok(denom_svd) = denom.rb().thin_svd() else {
        return (nan, df_base, nan, nan, nan, nan, nan);
    };
    let denom_inv = denom_svd.pseudoinverse();
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
    for (r, &rm) in row_margin.iter().enumerate() {
        for (c, &cm) in col_margin.iter().enumerate() {
            let i = r * nc + c;
            let expected = rm * cm;
            if expected > 0.0 {
                chisq_p += (p[i] - expected).powi(2) / expected;
            }
        }
    }
    chisq_p *= n_f;

    // 7. Rao-Scott corrections
    let dd = delta.nrows().min(delta.ncols());
    let trace_d: f64 = (0..dd).map(|i| delta[(i, i)]).sum();
    let d2 = &delta * &delta;
    let trace_d2: f64 = (0..dd).map(|i| d2[(i, i)]).sum();

    // No design variance to correct by: X² is kept, the corrected p and F are undefined.
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // NaN takes this branch on purpose
    if !(trace_d > 1e-9) {
        return (chisq_p, df_base, nan, nan, nan, nan, nan);
    }

    // As R svychisq "Chisq": report the raw X², take p from X²/mean(delta) on chi2(df_base).
    let mean_delta = trace_d / df_base;
    let chisq_p_for_p = chisq_p / mean_delta;

    // F = X² / trace(delta) with Satterthwaite df; R computes F from the Pearson X² only.
    let f_p = chisq_p / trace_d;
    let ndf = trace_d.powi(2) / trace_d2;
    let ddf = nu * ndf;

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
    strata: Option<&Column>,
    psu: Option<&Column>,
    n: usize,
    domain: Option<&BooleanChunked>,
) -> (usize, usize) {
    // R's degf() on a subset design counts the units with an in-domain row.
    let filtered: (Option<Column>, Option<Column>) = match domain {
        Some(m) => (
            strata.and_then(|c| c.filter(m).ok()),
            psu.and_then(|c| c.filter(m).ok()),
        ),
        None => (None, None),
    };
    let strata = if domain.is_some() {
        filtered.0.as_ref()
    } else {
        strata
    };
    let psu = if domain.is_some() {
        filtered.1.as_ref()
    } else {
        psu
    };
    match (strata, psu) {
        (Some(s), Some(p)) => {
            let n_strata = s.unique().map(|u| u.len()).unwrap_or(1);
            // Tabulate always passes String design columns; borrow &str pairs.
            let n_pairs = match (s.str(), p.str()) {
                (Ok(s), Ok(p)) => {
                    let mut pairs = std::collections::HashSet::new();
                    for (si, pi) in s.iter().zip(p.iter()) {
                        if let (Some(sv), Some(pv)) = (si, pi) {
                            pairs.insert((sv, pv));
                        }
                    }
                    pairs.len()
                }
                _ => 0,
            };
            (n_strata, n_pairs)
        }
        (None, Some(p)) => (1, p.unique().map(|u| u.len()).unwrap_or(n)),
        (Some(s), None) => (s.unique().map(|u| u.len()).unwrap_or(1), n),
        (None, None) => (1, n),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn zero_cov(k: usize) -> Vec<Vec<f64>> {
        vec![vec![0.0; k]; k]
    }

    fn all_nan(stats: &[f64]) -> bool {
        stats.iter().all(|v| v.is_nan())
    }

    #[test]
    fn nan_proportions_give_nan_statistics() {
        let p = [f64::NAN; 4];
        let (chisq, df, p_chisq, f, ndf, ddf, p_f) = rao_scott(&p, &zero_cov(4), 2, 2, 10, 3.0);
        assert_eq!(df, 1.0);
        assert!(all_nan(&[chisq, p_chisq, f, ndf, ddf, p_f]));
    }

    #[test]
    fn one_row_level_has_no_test() {
        let p = [0.5, 0.5];
        let (chisq, df, p_chisq, f, ndf, ddf, p_f) = rao_scott(&p, &zero_cov(2), 1, 2, 10, 3.0);
        assert_eq!(df, 0.0);
        assert!(all_nan(&[chisq, p_chisq, f, ndf, ddf, p_f]));
    }

    #[test]
    fn no_design_variance_keeps_x2_and_leaves_the_corrections_undefined() {
        // Expected cells 0.25: X2 = 100 * 4 * (0.4 - 0.25)^2 / 0.25 = 36.
        let p = [0.4, 0.1, 0.1, 0.4];
        let (chisq, df, p_chisq, f, ndf, ddf, p_f) = rao_scott(&p, &zero_cov(4), 2, 2, 100, 3.0);
        assert!((chisq - 36.0).abs() < 1e-12);
        assert_eq!(df, 1.0);
        assert!(all_nan(&[p_chisq, f, ndf, ddf, p_f]));
    }

    // R values: tests/golden/kernel_unit_r.R. The cells of the 2 x 3 table, row-major,
    // with their svymean covariance, and svychisq's Chisq and F statistics.
    #[test]
    #[allow(clippy::excessive_precision)] // as R prints them
    fn rao_scott_matches_r_svychisq() {
        let p = [
            0.10869565217391304,
            0.23913043478260870,
            0.13043478260869565,
            0.13043478260869565,
            0.21739130434782608,
            0.17391304347826086,
        ];
        let cov: Vec<Vec<f64>> = [
            [
                0.00646304508631687438,
                -0.00661223694883880404,
                0.00050653764101757804,
                0.00617761514574347633,
                -0.00868975596856786693,
                0.00215479504432874308,
            ],
            [
                -0.00661223694883880404,
                0.01973844075743011808,
                -0.01390164414792686293,
                -0.00823056664320096117,
                0.01332274398676391337,
                -0.00431673700422740073,
            ],
            [
                0.00050653764101757804,
                -0.01390164414792686293,
                0.01776901883569598564,
                0.00642686382624418906,
                -0.01131535407606462131,
                0.00051457792103373008,
            ],
            [
                0.00617761514574347633,
                -0.00823056664320096117,
                0.00642686382624418906,
                0.01020758216272812068,
                -0.01509607241254855292,
                0.00051457792103372976,
            ],
            [
                -0.00868975596856786693,
                0.01332274398676391337,
                -0.01131535407606462131,
                -0.01509607241254855292,
                0.02256459918310754764,
                -0.00078616071269042040,
            ],
            [
                0.00215479504432874308,
                -0.00431673700422740073,
                0.00051457792103373008,
                0.00051457792103372976,
                -0.00078616071269042040,
                0.00191894683052161734,
            ],
        ]
        .iter()
        .map(|r| r.to_vec())
        .collect();
        let (chisq, df, p_chisq, f, ndf, ddf, p_f) = rao_scott(&p, &cov, 2, 3, 12, 3.0);
        assert_relative_eq!(chisq, 0.088154269972451765, max_relative = 1e-12);
        assert_eq!(df, 2.0);
        assert_relative_eq!(p_chisq, 0.93441396503424778, max_relative = 1e-12);
        assert_relative_eq!(f, 0.067835721563683851, max_relative = 1e-12);
        assert_relative_eq!(ndf, 1.676372467122929, max_relative = 1e-12);
        assert_relative_eq!(ddf, 5.0291174013687865, max_relative = 1e-12);
        assert_relative_eq!(p_f, 0.90958351450705999, max_relative = 1e-12);
    }
}
