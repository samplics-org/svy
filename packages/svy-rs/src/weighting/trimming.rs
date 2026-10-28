//! Iterative weight trimming, optionally redistributing the trimmed mass.

use super::utils::{Result, WeightingError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

// ---------------------------------------------------------------------------
// Output type
// ---------------------------------------------------------------------------

pub(crate) struct TrimOutput {
    pub weights: Array1<f64>,
    pub n_trimmed_upper: usize,
    pub n_trimmed_lower: usize,
    pub weight_sum_before: f64,
    pub weight_sum_after: f64,
    pub ess_before: f64,
    pub ess_after: f64,
    pub iterations: usize,
    pub converged: bool,
}

// ---------------------------------------------------------------------------
// ESS helper
// ---------------------------------------------------------------------------

#[inline]
fn ess(weights: &[f64]) -> f64 {
    let mut s = 0.0_f64;
    let mut s2 = 0.0_f64;
    for &w in weights {
        s += w;
        s2 += w * w;
    }
    if s2 < 1e-300 { 0.0 } else { (s * s) / s2 }
}

// ---------------------------------------------------------------------------
// Single iteration
// ---------------------------------------------------------------------------

/// One in-place trimming pass; returns (n_upper, n_lower, n_changed).
#[inline]
fn trim_once(
    w: &mut [f64],
    zero_mask: &[bool],
    upper: Option<f64>,
    lower: Option<f64>,
    redistribute: bool,
) -> (usize, usize, usize) {
    let n = w.len();
    let mut n_upper: usize = 0;
    let mut n_lower: usize = 0;
    let mut n_changed: usize = 0;

    // ── Upper cap ────────────────────────────────────────────────────────────
    if let Some(cap) = upper {
        if redistribute {
            let mut excess = 0.0_f64;
            for i in 0..n {
                if !zero_mask[i] && w[i] > cap {
                    excess += w[i] - cap;
                    w[i] = cap;
                    n_upper += 1;
                    n_changed += 1;
                }
            }
            if excess > 0.0 {
                // Strictly below the cap: just-capped units would overshoot and re-trim.
                let mut denom = 0.0_f64;
                for i in 0..n {
                    if !zero_mask[i] && w[i] < cap {
                        denom += w[i];
                    }
                }
                if denom > 1e-300 {
                    let factor = excess / denom;
                    for i in 0..n {
                        if !zero_mask[i] && w[i] < cap {
                            w[i] += w[i] * factor;
                        }
                    }
                }
            }
        } else {
            for i in 0..n {
                if !zero_mask[i] && w[i] > cap {
                    w[i] = cap;
                    n_upper += 1;
                    n_changed += 1;
                }
            }
        }
    }

    // ── Lower floor ──────────────────────────────────────────────────────────
    if let Some(floor) = lower {
        if redistribute {
            let mut deficit = 0.0_f64;
            for i in 0..n {
                if !zero_mask[i] && w[i] > 0.0 && w[i] < floor {
                    deficit += floor - w[i];
                    w[i] = floor;
                    n_lower += 1;
                    n_changed += 1;
                }
            }
            if deficit > 0.0 {
                let mut denom = 0.0_f64;
                for i in 0..n {
                    if !zero_mask[i] && w[i] > floor {
                        denom += w[i];
                    }
                }
                if denom > deficit {
                    let factor = deficit / denom;
                    for i in 0..n {
                        if !zero_mask[i] && w[i] > floor {
                            w[i] -= w[i] * factor;
                        }
                    }
                }
            }
        } else {
            for i in 0..n {
                if !zero_mask[i] && w[i] > 0.0 && w[i] < floor {
                    w[i] = floor;
                    n_lower += 1;
                    n_changed += 1;
                }
            }
        }
    }

    (n_upper, n_lower, n_changed)
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Trims to caller-resolved bounds; converged once at most `tol` of nonzero weights change.
pub(crate) fn trim_impl(
    weights: ArrayView1<f64>,
    upper: Option<f64>,
    lower: Option<f64>,
    redistribute: bool,
    max_iter: usize,
    tol: f64,
) -> Result<TrimOutput> {
    let raw: Vec<f64> = weights.iter().copied().collect();

    // NaN would pass every comparison below and silently propagate.
    for &w in &raw {
        if w < 0.0 {
            return Err(WeightingError::InvalidInput(
                "Negative weights found. Trimming cannot proceed.".to_string(),
            ));
        }
        if w.is_nan() {
            return Err(WeightingError::InvalidInput(
                "NaN weights found. Trimming cannot proceed.".to_string(),
            ));
        }
    }

    // Zero weights are never trimmed nor given redistributed mass.
    let zero_mask: Vec<bool> = raw.iter().map(|&w| w == 0.0).collect();
    let n_active = zero_mask.iter().filter(|&&z| !z).count();

    let weight_sum_before: f64 = raw.iter().sum();
    let ess_before = ess(&raw);

    // A cap <= 0 is degenerate, not an error: the weights come back unchanged.
    if let Some(cap) = upper
        && cap <= 0.0
    {
        return Ok(TrimOutput {
            weights: Array1::from(raw),
            n_trimmed_upper: 0,
            n_trimmed_lower: 0,
            weight_sum_before,
            weight_sum_after: weight_sum_before,
            ess_before,
            ess_after: ess_before,
            iterations: 0,
            converged: true,
        });
    }
    let lower = match lower {
        Some(f) if f <= 0.0 => None, // degenerate lower — ignore
        other => other,
    };

    if let (Some(cap), Some(floor)) = (upper, lower)
        && floor >= cap
    {
        return Err(WeightingError::InvalidInput(format!(
            "lower ({floor:.6}) >= upper ({cap:.6}). Check threshold specifications."
        )));
    }

    let mut w: Vec<f64> = raw;

    let mut n_upper_total: usize = 0;
    let mut n_lower_total: usize = 0;
    let mut iterations: usize = 0;
    let mut converged = false;

    for _ in 0..max_iter {
        iterations += 1;
        let (n_up, n_lo, n_changed) = trim_once(&mut w, &zero_mask, upper, lower, redistribute);

        n_upper_total += n_up;
        n_lower_total += n_lo;

        let frac = if n_active > 0 {
            n_changed as f64 / n_active as f64
        } else {
            0.0
        };

        if frac <= tol {
            converged = true;
            break;
        }
    }

    let weight_sum_after: f64 = w.iter().sum();
    let ess_after = ess(&w);

    Ok(TrimOutput {
        weights: Array1::from(w),
        n_trimmed_upper: n_upper_total,
        n_trimmed_lower: n_lower_total,
        weight_sum_before,
        weight_sum_after,
        ess_before,
        ess_after,
        iterations,
        converged,
    })
}

// ---------------------------------------------------------------------------
// Matrix entry point (replicate weights)
// ---------------------------------------------------------------------------

/// Trims each replicate with the full-sample cutoffs, so variance reflects weighting, not cutoffs.
pub(crate) fn trim_matrix_impl(
    weights: ArrayView2<f64>,
    upper: Option<f64>,
    lower: Option<f64>,
    redistribute: bool,
    max_iter: usize,
    tol: f64,
) -> Result<Array2<f64>> {
    let (n_rows, n_cols) = weights.dim();
    let mut out = Array2::zeros((n_rows, n_cols));

    for col in 0..n_cols {
        let col_view = weights.column(col);
        let result = trim_impl(col_view, upper, lower, redistribute, max_iter, tol)?;
        out.column_mut(col).assign(&result.weights);
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn trim(w: &[f64], upper: Option<f64>, lower: Option<f64>, redistribute: bool) -> TrimOutput {
        trim_impl(
            ndarray::ArrayView1::from(w),
            upper,
            lower,
            redistribute,
            10,
            1e-6,
        )
        .unwrap()
    }

    #[test]
    fn test_absolute_upper_no_redistribute() {
        let w = [10.0, 10.0, 10.0, 100.0];
        let out = trim(&w, Some(50.0), None, false);
        assert_eq!(out.weights[3], 50.0);
        assert_eq!(out.n_trimmed_upper, 1);
        assert!(out.weight_sum_after < out.weight_sum_before);
        assert!(out.converged);
    }

    #[test]
    fn test_absolute_upper_with_redistribute() {
        let w = [10.0, 10.0, 10.0, 100.0];
        let out = trim(&w, Some(50.0), None, true);
        assert_eq!(out.weights[3], 50.0);
        assert_relative_eq!(out.weight_sum_after, out.weight_sum_before, epsilon = 1e-9);
        assert!(out.weights[0] > 10.0);
    }

    #[test]
    fn test_lower_floor() {
        let w = [1.0, 5.0, 10.0, 10.0];
        let out = trim(&w, None, Some(4.0), false);
        assert_eq!(out.weights[0], 4.0);
        assert_eq!(out.n_trimmed_lower, 1);
    }

    #[test]
    fn test_both_bounds() {
        let w = [1.0, 10.0, 10.0, 200.0];
        let out = trim(&w, Some(50.0), Some(5.0), false);
        assert_eq!(out.weights[0], 5.0);
        assert_eq!(out.weights[3], 50.0);
        assert_eq!(out.n_trimmed_lower, 1);
        assert_eq!(out.n_trimmed_upper, 1);
    }

    #[test]
    fn test_zero_weight_preserved() {
        let w = [0.0, 10.0, 10.0, 100.0];
        let out = trim(&w, Some(50.0), None, false);
        assert_eq!(out.weights[0], 0.0);
    }

    #[test]
    fn test_negative_weight_error() {
        let result = trim_impl(
            ndarray::ArrayView1::from(&[-1.0_f64, 10.0, 10.0][..]),
            Some(50.0),
            None,
            false,
            10,
            1e-6,
        );
        assert!(
            matches!(&result, Err(WeightingError::InvalidInput(m)) if m.contains("Negative weights")),
            "got {:?}",
            result.as_ref().err()
        );
    }

    #[test]
    fn test_nan_weight_error() {
        let result = trim_impl(
            ndarray::ArrayView1::from(&[f64::NAN, 10.0, 10.0][..]),
            Some(50.0),
            None,
            false,
            10,
            1e-6,
        );
        assert!(
            matches!(&result, Err(WeightingError::InvalidInput(m)) if m.contains("NaN weights")),
            "got {:?}",
            result.as_ref().err()
        );
    }

    #[test]
    fn test_lower_ge_upper_error() {
        let result = trim_impl(
            ndarray::ArrayView1::from(&[10.0_f64, 20.0, 30.0][..]),
            Some(10.0),
            Some(20.0),
            false,
            10,
            1e-6,
        );
        assert!(
            matches!(&result, Err(WeightingError::InvalidInput(m)) if m.contains("lower (20.000000) >= upper (10.000000)")),
            "got {:?}",
            result.as_ref().err()
        );
    }

    #[test]
    fn test_all_equal_weights_nothing_trimmed() {
        let w = [10.0; 10];
        let out = trim(&w, Some(50.0), None, false);
        assert_eq!(out.n_trimmed_upper, 0);
        for &v in out.weights.iter() {
            assert_relative_eq!(v, 10.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_degenerate_zero_upper_skips_trimming() {
        let w = [10.0; 10]; // IQR=0 → upper resolves to 0 in Python
        let out = trim_impl(
            ndarray::ArrayView1::from(&w[..]),
            Some(0.0),
            None,
            false,
            10,
            1e-6,
        )
        .unwrap();
        assert_eq!(out.n_trimmed_upper, 0);
        assert!(out.converged);
    }

    #[test]
    fn test_weight_sum_preserved_with_redistribution() {
        let w = vec![10.0_f64; 8]
            .into_iter()
            .chain([200.0, 200.0])
            .collect::<Vec<_>>();
        let out = trim_impl(
            ndarray::ArrayView1::from(&w[..]),
            Some(50.0),
            None,
            true,
            20,
            1e-6,
        )
        .unwrap();
        assert_relative_eq!(out.weight_sum_after, out.weight_sum_before, epsilon = 1e-12);
    }

    #[test]
    fn test_ess_increases_after_trimming_extremes() {
        let mut w = vec![10.0_f64; 9];
        w.push(1000.0);
        let out = trim(&w, Some(50.0), None, false);
        assert!(out.ess_after > out.ess_before);
    }

    #[test]
    fn test_convergence_flag() {
        let mut w = vec![10.0_f64; 9];
        w.push(100.0);
        let out = trim_impl(
            ndarray::ArrayView1::from(&w[..]),
            Some(50.0),
            None,
            true,
            100,
            1e-9,
        )
        .unwrap();
        assert!(out.converged);
    }

    #[test]
    fn test_max_iter_not_converged() {
        let w = [1.0, 1.0, 1.0, 1000.0];
        let out = trim_impl(
            ndarray::ArrayView1::from(&w[..]),
            Some(2.0),
            None,
            true,
            1,
            1e-20, // impossible tolerance
        )
        .unwrap();
        assert_eq!(out.iterations, 1);
        assert!(!out.converged);
    }
}
