// src/weighting/trimming.rs
//
// Pure trimming algorithm — no PyO3, no Python types.
// Called from api.rs which handles the Python boundary.
//
// Performance notes:
// - Single allocation for the output array (no clone per iteration)
// - Zero-mask computed once upfront; zero units never touched
// - Convergence check uses a running count of changed weights (integer)
//   rather than floating-point max-diff scan over the full array
// - Redistribution uses a single pass with precomputed sums
// - All hot paths are branch-minimized and cache-friendly (sequential access)

use super::utils::{Result, WeightingError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

// ---------------------------------------------------------------------------
// Output type
// ---------------------------------------------------------------------------

/// Output of trim_impl — flat primitives, maps directly to the Python tuple.
pub struct TrimOutput {
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
    if s2 < 1e-300 {
        0.0
    } else {
        (s * s) / s2
    }
}

// ---------------------------------------------------------------------------
// Single iteration
// ---------------------------------------------------------------------------

/// One trimming pass over the weight slice.
///
/// Returns (n_upper_trimmed, n_lower_trimmed, n_changed).
/// Mutates `w` in place.
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
            // Two-pass: identify trimmed units and accumulate excess, then
            // redistribute in a single pass over non-trimmed units.
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
                // Sum of non-trimmed active weights (post-cap)
                let mut denom = 0.0_f64;
                for i in 0..n {
                    if !zero_mask[i] && w[i] <= cap {
                        denom += w[i];
                    }
                }
                if denom > 1e-300 {
                    let factor = excess / denom;
                    for i in 0..n {
                        if !zero_mask[i] && w[i] <= cap {
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

/// Iterative weight trimming on a 1-D weight array.
///
/// All thresholds must already be resolved to scalars by the caller
/// (Python: `trim.py` / `types.py`). The engine receives flat primitives only.
///
/// # Arguments
/// * `weights`      - Input float64 array, shape (n,). Must have no negatives.
/// * `upper`        - Resolved upper cap, or None.
/// * `lower`        - Resolved lower floor, or None.
/// * `redistribute` - Redistribute trimmed mass to non-trimmed units.
/// * `max_iter`     - Maximum iterations.
/// * `tol`          - Convergence: fraction of active weights that changed.
///
/// # Returns
/// `TrimOutput` holding the trimmed array and all diagnostics.
pub fn trim_impl(
    weights: ArrayView1<f64>,
    upper: Option<f64>,
    lower: Option<f64>,
    redistribute: bool,
    max_iter: usize,
    tol: f64,
) -> Result<TrimOutput> {
    // Collect into owned Vec once — avoids repeated bounds checks in hot paths
    // and ensures we own the data regardless of input memory layout.
    let raw: Vec<f64> = weights.iter().copied().collect();

    // Guard: caller should have caught negatives, but assert at boundary
    for &w in &raw {
        if w < 0.0 {
            return Err(WeightingError::InvalidInput(
                "Negative weights found. Trimming cannot proceed.".to_string(),
            ));
        }
    }

    // Build zero mask once — zero units are never touched
    let zero_mask: Vec<bool> = raw.iter().map(|&w| w == 0.0).collect();
    let n_active = zero_mask.iter().filter(|&&z| !z).count();

    // Pre-trim diagnostics
    let weight_sum_before: f64 = raw.iter().sum();
    let ess_before = ess(&raw);

    // Degenerate threshold: return unchanged
    if let Some(cap) = upper {
        if cap <= 0.0 {
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
    }
    let lower = match lower {
        Some(f) if f <= 0.0 => None, // degenerate lower — ignore
        other => other,
    };

    if let (Some(cap), Some(floor)) = (upper, lower) {
        if floor >= cap {
            return Err(WeightingError::InvalidInput(format!(
                "lower ({floor:.6}) >= upper ({cap:.6}). Check threshold specifications."
            )));
        }
    }

    // Working copy — single allocation (raw consumed here)
    let mut w: Vec<f64> = raw;

    let mut n_upper_total: usize = 0;
    let mut n_lower_total: usize = 0;
    let mut iterations: usize = 0;
    let mut converged = false;

    for _ in 0..max_iter {
        iterations += 1;
        let (n_up, n_lo, n_changed) =
            trim_once(&mut w, &zero_mask, upper, lower, redistribute);

        n_upper_total += n_up;
        n_lower_total += n_lo;

        // Convergence: fraction of active weights that changed
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

/// Trim a weight matrix column-by-column using pre-resolved scalar thresholds.
///
/// Designed for replicate weight trimming. Thresholds are fixed scalars derived
/// from the main weight distribution by the Python caller — replicates all use
/// the same cutoffs so variance estimates reflect weighting variability only,
/// not threshold variability.
///
/// No diagnostics are returned: audit stats (ESS, n_trimmed, …) are only
/// meaningful for the main weight and are computed by `trim_impl` / `trim_weights`.
///
/// # Arguments
/// * `weights`      - Input matrix, shape (n, n_reps). No negative values.
/// * `upper`        - Pre-resolved upper cap, or None.
/// * `lower`        - Pre-resolved lower floor, or None.
/// * `redistribute` - Redistribute trimmed mass within each column.
/// * `max_iter`     - Maximum iterations per column.
/// * `tol`          - Convergence tolerance per column.
///
/// # Returns
/// Trimmed weight matrix, same shape as input.
pub fn trim_matrix_impl(
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
        // trim_impl owns its input — cheap copy of one weight column
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
    use ndarray::array;

    fn trim(
        w: &[f64],
        upper: Option<f64>,
        lower: Option<f64>,
        redistribute: bool,
    ) -> TrimOutput {
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
        // Excess redistributed to first 3
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
        assert!(result.is_err());
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
        assert!(result.is_err());
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
            Some(0.0), // degenerate
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
        assert_relative_eq!(out.weight_sum_after, out.weight_sum_before, epsilon = 1e-6);
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
            1,    // force 1 iteration
            1e-20, // impossible tolerance
        )
        .unwrap();
        assert_eq!(out.iterations, 1);
        assert!(!out.converged);
    }
}
