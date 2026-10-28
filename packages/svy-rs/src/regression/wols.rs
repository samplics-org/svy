//! Closed-form weighted least squares with influence functions, for the t-tests and rank tests.
//!
//! - Matches R: `infn <- (model.matrix(m) * residuals(m)) %*% summary(m)$cov.unscaled`, then `svytotal(infn, design)`.

use crate::regression::error::{RResult, RegressionError};
use polars::prelude::*;

use crate::estimation::calib_sweep::CalibSweep;

use crate::estimation::taylor::{
    build_taylor_design, taylor_covariance_apply_in, taylor_variance_in,
};

// ============================================================================
// Core WLS Result
// ============================================================================

/// Result of a weighted OLS fit with influence functions.
#[allow(dead_code)]
pub struct WolsResult {
    pub beta: Vec<f64>,
    pub fitted: Vec<f64>,
    pub residuals: Vec<f64>,
    /// (X'WX)^{-1}, row-major k x k
    pub cov_unscaled: Vec<f64>,
    /// Row-major n x k: influence of obs i on coefficient j at i * k + j
    pub influence: Vec<f64>,
    pub n: usize,
    pub k: usize,
}

// ============================================================================
// Matrix helpers (lightweight, no external dep needed for small k)
// ============================================================================

/// Solve A x = b for a small row-major SPD A by Cholesky, falling back to LU.
pub fn solve_kxk(a: &[f64], b: &[f64], k: usize) -> Option<Vec<f64>> {
    if let Some(x) = solve_cholesky(a, b, k) {
        return Some(x);
    }
    solve_gauss(a, b, k)
}

fn solve_cholesky(a: &[f64], b: &[f64], k: usize) -> Option<Vec<f64>> {
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

    let mut y = vec![0.0; k];
    for i in 0..k {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[i * k + j] * y[j];
        }
        y[i] = (b[i] - sum) / l[i * k + i];
    }

    let mut x = vec![0.0; k];
    for i in (0..k).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..k {
            sum += l[j * k + i] * x[j]; // L'[i][j] = l[j][i]
        }
        x[i] = (y[i] - sum) / l[i * k + i];
    }

    Some(x)
}

fn solve_gauss(a: &[f64], b: &[f64], k: usize) -> Option<Vec<f64>> {
    let mut aug = vec![0.0; k * (k + 1)];
    for i in 0..k {
        for j in 0..k {
            aug[i * (k + 1) + j] = a[i * k + j];
        }
        aug[i * (k + 1) + k] = b[i];
    }

    for col in 0..k {
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

        if max_row != col {
            for j in 0..=(k) {
                aug.swap(col * (k + 1) + j, max_row * (k + 1) + j);
            }
        }

        let pivot = aug[col * (k + 1) + col];
        for row in (col + 1)..k {
            let factor = aug[row * (k + 1) + col] / pivot;
            for j in col..=(k) {
                aug[row * (k + 1) + j] -= factor * aug[col * (k + 1) + j];
            }
        }
    }

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

/// Weighted OLS of `y` on row-major n x k `x`, with influence `(x_i * resid_i) %*% (X'WX)^{-1}`.
pub fn fit_wols(y: &[f64], x: &[f64], w: &[f64], n: usize, k: usize) -> RResult<WolsResult> {
    if n == 0 || k == 0 {
        return Err(RegressionError::WolsEmpty);
    }
    for (what, expected, got) in [("y", n, y.len()), ("w", n, w.len()), ("x", n * k, x.len())] {
        if got != expected {
            return Err(RegressionError::DimensionMismatch {
                what,
                expected,
                got,
            });
        }
    }

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

    let beta = solve_kxk(&xtwx, &xtwy, k).ok_or(RegressionError::WolsSingular)?;

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

    let cov_unscaled = invert_kxk(&xtwx, k).ok_or(RegressionError::WolsNotInvertible)?;

    // R: (xmat * (rankscore - fitted(m))) %*% summary(m)$cov.unscaled
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

/// Design SE per coefficient, R's `SE(svytotal(infn, design))`: hence the influence is scaled by w first.
#[allow(clippy::too_many_arguments)]
pub fn influence_se(
    influence: &[f64],
    w: &[f64],
    n: usize,
    k: usize,
    strata: Option<&Column>,
    psu: Option<&Column>,
    ssu: Option<&Column>,
    fpc: Option<&Float64Chunked>,
    fpc_ssu: Option<&Float64Chunked>,
    singleton_method: Option<&str>,
    // Swept on `influence * w`, which is R svy.varcoef's `estfun %*% Ainv`; the sweep commutes with the bread.
    calib: Option<&CalibSweep>,
    active: Option<&[bool]>,
) -> RResult<Vec<f64>> {
    let mut ses = Vec::with_capacity(k);

    for j in 0..k {
        let mut col_vals: Vec<f64> = (0..n).map(|i| influence[i * k + j] * w[i]).collect();
        if let Some(c) = calib {
            c.apply(&mut col_vals);
        }
        let scores = Float64Chunked::from_vec("infn".into(), col_vals);

        let var = taylor_variance_in(
            &scores,
            strata,
            psu,
            ssu,
            fpc,
            fpc_ssu,
            singleton_method,
            active,
        )?;

        ses.push(var.max(0.0).sqrt());
    }

    Ok(ses)
}

/// Row-major k x k design covariance of the coefficients: R's svyrecvar on the weight-scaled influence.
#[allow(clippy::too_many_arguments)]
pub fn influence_covariance(
    influence: &[f64],
    w: &[f64],
    n: usize,
    k: usize,
    strata: Option<&Column>,
    psu: Option<&Column>,
    ssu: Option<&Column>,
    fpc: Option<&Float64Chunked>,
    fpc_ssu: Option<&Float64Chunked>,
    singleton_method: Option<&str>,
    // Swept on `influence * w`, which is R svy.varcoef's `estfun %*% Ainv`; the sweep commutes with the bread.
    calib: Option<&CalibSweep>,
    active: Option<&[bool]>,
) -> RResult<Vec<f64>> {
    let cols: Vec<Vec<f64>> = (0..k)
        .map(|j| {
            let mut col: Vec<f64> = (0..n).map(|i| influence[i * k + j] * w[i]).collect();
            if let Some(c) = calib {
                c.apply(&mut col);
            }
            col
        })
        .collect();

    // A missing stratum or PSU label is a group of its own here.
    let labels = |c: &Column| -> RResult<Column> {
        let s = c.cast(&DataType::String)?;
        let filled: StringChunked = s
            .str()?
            .iter()
            .map(|v| Some(v.unwrap_or("__NULL__")))
            .collect();
        Ok(filled.with_name(c.name().clone()).into_column())
    };
    let strata = strata.map(labels).transpose()?;
    let psu = psu.map(labels).transpose()?;
    let ssu = ssu.map(labels).transpose()?;
    let design = build_taylor_design(
        strata.as_ref(),
        psu.as_ref(),
        ssu.as_ref(),
        fpc,
        fpc_ssu,
        singleton_method,
    )?;
    Ok(taylor_covariance_apply_in(&cols, &design, active)
        .into_iter()
        .flatten()
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::{G2, N, W, X2, Y, psu, strata};
    use approx::assert_relative_eq;

    #[test]
    fn slices_that_disagree_with_n_and_k_raise_dimension_mismatch() {
        let (y, w, x) = ([1.0, 2.0, 3.0], [1.0; 3], [1.0, 1.0, 1.0, 0.0, 1.0, 2.0]);
        let cases = [
            (&y[..2], &w[..], &x[..], "y", 3, 2),
            (&y[..], &w[..2], &x[..], "w", 3, 2),
            (&y[..], &w[..], &x[..5], "x", 6, 5),
        ];
        for (y, w, x, what, expected, got) in cases {
            let err = fit_wols(y, x, w, 3, 2).err().expect("mismatch must fail");
            assert_eq!(
                err.to_string(),
                format!("{what} has length {got}, expected {expected}")
            );
        }
    }

    // R values: tests/golden/kernel_unit_r.R, lm(y ~ g2 + x2, weights = w) and
    // the SEs of svyglm(y ~ g2 + x2) on the stratified cluster design.
    #[test]
    #[allow(clippy::excessive_precision)] // as R prints them
    fn coefficients_and_design_ses_match_r() {
        let k = 3;
        let x: Vec<f64> = (0..N).flat_map(|i| [1.0, G2[i] as f64, X2[i]]).collect();
        let fit = fit_wols(&Y, &x, &W, N, k).unwrap();
        let beta = [
            1.89629588451279929,
            0.13714752408103748,
            1.00834193469162869,
        ];
        for (got, want) in fit.beta.iter().zip(beta) {
            assert_relative_eq!(*got, want, max_relative = 1e-12);
        }

        let (st, ps) = (strata(), psu());
        let ses = influence_se(
            &fit.influence,
            &W,
            N,
            k,
            Some(&st),
            Some(&ps),
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let r_ses = [
            0.298175333094242601,
            0.090228917595754188,
            0.098383047947286223,
        ];
        for (got, want) in ses.iter().zip(r_ses) {
            assert_relative_eq!(*got, want, max_relative = 1e-12);
        }
    }
}
