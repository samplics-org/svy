// src/weighting/calibration.rs

use super::utils::{Result, WeightingError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use rayon::prelude::*;
use std::collections::HashMap;

/// Linear calibration using Deville-Särndal method
///
/// Calibrates weights to match auxiliary information using linear calibration.
/// This solves for Lagrange multipliers λ and computes: g = 1 + (X'λ)/s
/// where the calibrated weights are: w_cal = w * g
///
/// # Arguments
/// * `wgt` - Initial weights matrix (n_obs, n_reps)
/// * `x_matrix` - Auxiliary variables matrix (n_obs, n_aux)
/// * `totals` - Known population totals for auxiliary variables (n_aux,)
/// * `scale` - Scale factors for distance function (n_obs,). Default is 1.0
/// * `additive` - If true, return g-factors instead of calibrated weights
///
/// # Returns
/// Calibrated weight matrix (or g-factors if additive=true)
///
/// # Algorithm
/// Solves the system: (X' diag(w/s) X) λ = (tx - X' w)
/// Then computes: g = 1 + (X λ) / s
/// Calibrated weights: w_cal = w * g
pub fn calibrate_linear(
    wgt: ArrayView2<f64>,
    x_matrix: ArrayView2<f64>,
    totals: ArrayView1<f64>,
    scale: Option<ArrayView1<f64>>,
    additive: bool,
) -> Result<Array2<f64>> {
    let (n_obs, n_reps) = wgt.dim();
    let (n_obs_x, n_aux) = x_matrix.dim();

    // Validate inputs
    if n_obs_x != n_obs {
        return Err(WeightingError::DimensionMismatch {
            expected: n_obs,
            got: n_obs_x,
        });
    }

    if totals.len() != n_aux {
        return Err(WeightingError::DimensionMismatch {
            expected: n_aux,
            got: totals.len(),
        });
    }

    // Handle scale factors
    let s = match scale {
        Some(s_arr) => {
            if s_arr.len() != n_obs {
                return Err(WeightingError::DimensionMismatch {
                    expected: n_obs,
                    got: s_arr.len(),
                });
            }
            s_arr.to_owned()
        }
        None => Array1::ones(n_obs),
    };

    if let Some((row, &value)) = s
        .iter()
        .enumerate()
        .find(|(_, x)| !(x.is_finite() && **x > 0.0))
    {
        return Err(WeightingError::ScaleNotPositive { row, value });
    }

    // Calibrate each replicate IN PARALLEL using rayon
    let results: std::result::Result<Vec<_>, WeightingError> = (0..n_reps)
        .into_par_iter()
        .map(|r| {
            let wgt_col = wgt.column(r);
            calibrate_linear_single(wgt_col, x_matrix, totals, s.view(), additive)
        })
        .collect();

    let cal_cols = results?;

    // Assemble result matrix
    let mut result = Array2::zeros((n_obs, n_reps));
    for (col_idx, col) in cal_cols.into_iter().enumerate() {
        result.column_mut(col_idx).assign(&col);
    }

    Ok(result)
}

/// Calibrate a single replicate using linear method
fn calibrate_linear_single(
    wgt: ArrayView1<f64>,
    x_matrix: ArrayView2<f64>,
    totals: ArrayView1<f64>,
    scale: ArrayView1<f64>,
    additive: bool,
) -> Result<Array1<f64>> {
    let n_obs = wgt.len();
    let n_aux = totals.len();

    // 1. Compute current weighted totals: x_w = X' w
    let mut x_w: Array1<f64> = Array1::zeros(n_aux);
    for j in 0..n_aux {
        for i in 0..n_obs {
            x_w[j] += x_matrix[[i, j]] * wgt[i];
        }
    }

    // 2. Construct system matrix: A = X' diag(w/s) X
    let mut a_matrix = Array2::zeros((n_aux, n_aux));
    for j1 in 0..n_aux {
        for j2 in 0..n_aux {
            let mut sum = 0.0;
            for i in 0..n_obs {
                sum += x_matrix[[i, j1]] * (wgt[i] / scale[i]) * x_matrix[[i, j2]];
            }
            a_matrix[[j1, j2]] = sum;
        }
    }

    // 3. Construct RHS: b = totals - x_w
    let b = &totals - &x_w;

    // 4. Solve for Lagrange multipliers: A λ = b
    let lambda = solve_linear_system(&a_matrix, &b)?;

    // 5. Compute g-factors: g = 1 + (X λ) / s
    let mut g_factors = Array1::ones(n_obs);
    for i in 0..n_obs {
        let mut x_lambda_i = 0.0;
        for j in 0..n_aux {
            x_lambda_i += x_matrix[[i, j]] * lambda[j];
        }
        g_factors[i] = 1.0 + x_lambda_i / scale[i];
    }

    // 6. Return result
    if additive {
        Ok(g_factors)
    } else {
        Ok(&wgt * &g_factors)
    }
}

/// Solve linear system Ax = b using Gaussian elimination with partial pivoting
fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    let n = a.nrows();

    if a.ncols() != n || b.len() != n {
        return Err(WeightingError::InvalidInput(
            "Matrix dimensions incompatible for solving".to_string(),
        ));
    }

    // Create augmented matrix [A|b]
    let mut aug = Array2::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Forward elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_val = aug[[k, k]].abs();
        let mut max_row = k;

        for i in (k + 1)..n {
            let val = aug[[i, k]].abs();
            if val > max_val {
                max_val = val;
                max_row = i;
            }
        }

        // Singular, e.g. controls that imply one another: the minimum-norm solution.
        if max_val < 1e-10 {
            return solve_pseudoinverse(a, b);
        }

        // Swap rows
        if max_row != k {
            for j in 0..=n {
                let temp = aug[[k, j]];
                aug[[k, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Eliminate
        for i in (k + 1)..n {
            let factor = aug[[i, k]] / aug[[k, k]];
            for j in k..=n {
                aug[[i, j]] -= factor * aug[[k, j]];
            }
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum -= aug[[i, j]] * x[j];
        }
        x[i] = sum / aug[[i, i]];
    }

    Ok(x)
}

/// `A⁺ b` through the SVD, dropping singular values at or below 256·ε times the largest,
/// as R's `MASS::ginv(tol = 256 * .Machine$double.eps)` in `grake`.
fn solve_pseudoinverse(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    let n = a.nrows();
    let m = faer::Mat::<f64>::from_fn(n, n, |i, j| a[[i, j]]);
    let svd = m.thin_svd().map_err(|_| WeightingError::SvdNotConverged)?;
    let (u, v) = (svd.U(), svd.V());
    let s = svd.S().column_vector();
    let s_max = (0..n).map(|k| s[k]).fold(0.0, f64::max);

    let mut x = Array1::zeros(n);
    for k in (0..n).filter(|&k| s[k] > 256.0 * f64::EPSILON * s_max) {
        let coef = (0..n).map(|i| u[(i, k)] * b[i]).sum::<f64>() / s[k];
        for j in 0..n {
            x[j] += v[(j, k)] * coef;
        }
    }
    Ok(x)
}

/// Domain-specific calibration
///
/// Calibrates weights separately within each domain (group).
/// Processes domains in parallel using rayon for better performance.
///
/// # Arguments
/// * `wgt` - Initial weights matrix (n_obs, n_reps)
/// * `x_matrix` - Auxiliary variables matrix (n_obs, n_aux)
/// * `domain` - Domain identifiers for each observation (n_obs,)
/// * `controls` - Map from domain ID to control totals array
/// * `scale` - Optional scale factors
/// * `additive` - If true, return g-factors
///
/// # Returns
/// Calibrated weight matrix
pub fn calibrate_by_domain(
    wgt: ArrayView2<f64>,
    x_matrix: ArrayView2<f64>,
    domain: ArrayView1<i64>,
    controls: &HashMap<i64, Array1<f64>>,
    scale: Option<ArrayView1<f64>>,
    additive: bool,
) -> Result<Array2<f64>> {
    let (n_obs, n_reps) = wgt.dim();

    if domain.len() != n_obs {
        return Err(WeightingError::DimensionMismatch {
            expected: n_obs,
            got: domain.len(),
        });
    }
    if x_matrix.nrows() != n_obs {
        return Err(WeightingError::DimensionMismatch {
            expected: n_obs,
            got: x_matrix.nrows(),
        });
    }
    if let Some(s_arr) = &scale
        && s_arr.len() != n_obs
    {
        return Err(WeightingError::DimensionMismatch {
            expected: n_obs,
            got: s_arr.len(),
        });
    }

    // Handle scale
    let s = match scale {
        Some(s_arr) => s_arr.to_owned(),
        None => Array1::ones(n_obs),
    };

    // Sort data by domain for efficient processing
    let (domain_sorted, sort_idx, unsort_idx) = sort_by_domain(&domain);

    // Apply sort
    let wgt_sorted = reorder_rows(wgt, &sort_idx);
    let x_sorted = reorder_rows_2d(x_matrix, &sort_idx);
    let s_sorted = reorder_array(&s.view(), &sort_idx);

    // Find domain boundaries
    let boundaries = find_domain_boundaries(&domain_sorted);

    // Collect domain info for parallel processing
    if let Some(&dom_val) = boundaries
        .keys()
        .filter(|d| !controls.contains_key(d))
        .min()
    {
        return Err(WeightingError::MissingControls(dom_val));
    }
    let domain_tasks: Vec<_> = boundaries
        .iter()
        .map(|(&dom_val, &(start, end))| (dom_val, start, end, controls[&dom_val].clone()))
        .collect();

    // Process each domain IN PARALLEL
    let domain_results: std::result::Result<Vec<_>, WeightingError> = domain_tasks
        .par_iter()
        .map(|(_, start, end, totals)| {
            let wgt_slice = wgt_sorted.slice(s![*start..*end, ..]);
            let x_slice = x_sorted.slice(s![*start..*end, ..]);
            let s_slice = s_sorted.slice(s![*start..*end]);

            let cal_slice =
                calibrate_linear(wgt_slice, x_slice, totals.view(), Some(s_slice), additive)?;

            Ok((*start, *end, cal_slice))
        })
        .collect();

    let calibrated_domains = domain_results?;

    // Assemble result matrix
    let mut result_sorted = if additive {
        Array2::ones((n_obs, n_reps))
    } else {
        wgt_sorted.clone()
    };

    for (start, end, cal_slice) in calibrated_domains {
        result_sorted
            .slice_mut(s![start..end, ..])
            .assign(&cal_slice);
    }

    // Restore original order
    let result = reorder_rows(result_sorted.view(), &unsort_idx);

    Ok(result)
}

// ==================== Helper Functions ====================

fn sort_by_domain(domain: &ArrayView1<i64>) -> (Array1<i64>, Vec<usize>, Vec<usize>) {
    let n = domain.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by_key(|&i| domain[i]);

    let domain_sorted = Array1::from_shape_fn(n, |i| domain[indices[i]]);

    let mut unsort_idx = vec![0; n];
    for (new_pos, &old_pos) in indices.iter().enumerate() {
        unsort_idx[old_pos] = new_pos;
    }

    (domain_sorted, indices, unsort_idx)
}

fn reorder_rows(arr: ArrayView2<f64>, indices: &[usize]) -> Array2<f64> {
    let (n, m) = arr.dim();
    let mut result = Array2::zeros((n, m));

    for (new_i, &old_i) in indices.iter().enumerate() {
        result.row_mut(new_i).assign(&arr.row(old_i));
    }

    result
}

fn reorder_rows_2d(arr: ArrayView2<f64>, indices: &[usize]) -> Array2<f64> {
    reorder_rows(arr, indices)
}

fn reorder_array(arr: &ArrayView1<f64>, indices: &[usize]) -> Array1<f64> {
    Array1::from_shape_fn(arr.len(), |i| arr[indices[i]])
}

fn find_domain_boundaries(domain_sorted: &Array1<i64>) -> HashMap<i64, (usize, usize)> {
    let mut boundaries = HashMap::new();

    if domain_sorted.is_empty() {
        return boundaries;
    }

    let mut current_domain = domain_sorted[0];
    let mut start = 0;

    for i in 1..domain_sorted.len() {
        if domain_sorted[i] != current_domain {
            boundaries.insert(current_domain, (start, i));
            current_domain = domain_sorted[i];
            start = i;
        }
    }

    boundaries.insert(current_domain, (start, domain_sorted.len()));
    boundaries
}

/// Parallel calibration
///
/// Identical computation to `calibrate_linear` with `additive=false`;
/// delegates so all input validation (dimension checks, scale sanitization)
/// lives in one place.
pub fn calibrate_parallel(
    wgt: ArrayView2<f64>,
    x_matrix: ArrayView2<f64>,
    totals: ArrayView1<f64>,
    scale: Option<ArrayView1<f64>>,
) -> Result<Array2<f64>> {
    calibrate_linear(wgt, x_matrix, totals, scale, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn singular_system_gets_the_minimum_norm_solution() {
        // pinv([[1, 1], [1, 1]]) = [[1, 1], [1, 1]] / 4, so x = [4, 4] / 4.
        let a = array![[1.0, 1.0], [1.0, 1.0]];
        let x = solve_pseudoinverse(&a, &array![2.0, 2.0]).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-14 && (x[1] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn redundant_controls_calibrate_like_the_reduced_set() {
        // Intercept + female + male spans what intercept + female spans; linear
        // calibration weights depend only on that span.
        let wgt = array![[1.0], [2.0], [1.5], [1.0], [2.5], [1.0]];
        let female = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let full = Array2::from_shape_fn((6, 3), |(i, j)| match j {
            0 => 1.0,
            1 => female[i],
            _ => 1.0 - female[i],
        });
        let reduced = full.slice(s![.., 0..2]).to_owned();

        let w_full = calibrate_linear(
            wgt.view(),
            full.view(),
            array![12.0, 7.0, 5.0].view(),
            None,
            false,
        )
        .unwrap();
        let w_red = calibrate_linear(
            wgt.view(),
            reduced.view(),
            array![12.0, 7.0].view(),
            None,
            false,
        )
        .unwrap();

        for i in 0..6 {
            assert!((w_full[[i, 0]] - w_red[[i, 0]]).abs() < 1e-12);
        }
        let females: f64 = (0..6).map(|i| female[i] * w_full[[i, 0]]).sum();
        assert!((w_full.column(0).sum() - 12.0).abs() < 1e-12 && (females - 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_linear_calibration_simple() {
        let wgt = array![[1.0], [1.0], [1.0], [1.0]];
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let totals = array![15.0];

        let result = calibrate_linear(wgt.view(), x.view(), totals.view(), None, false).unwrap();

        // Check calibrated totals
        let mut x_cal = 0.0;
        for i in 0..4 {
            x_cal += x[[i, 0]] * result[[i, 0]];
        }
        assert!((x_cal - 15.0).abs() < 1e-6);
    }

    // Regression: these used to panic (index out of bounds) instead of
    // returning clean errors.
    #[test]
    fn test_calibrate_parallel_bad_dims_are_errors() {
        let wgt = array![[1.0], [1.0], [1.0]];
        let x = array![[1.0], [2.0], [3.0]];
        let totals = array![10.0];

        // scale shorter than n_obs
        let short_scale = array![1.0];
        let res = calibrate_parallel(
            wgt.view(),
            x.view(),
            totals.view(),
            Some(short_scale.view()),
        );
        assert!(res.is_err());

        // totals longer than n_aux
        let long_totals = array![10.0, 20.0];
        let res = calibrate_parallel(wgt.view(), x.view(), long_totals.view(), None);
        assert!(res.is_err());
    }

    #[test]
    fn test_calibrate_by_domain_bad_dims_are_errors() {
        let wgt = array![[1.0], [1.0], [1.0]];
        let x_short = array![[1.0], [2.0]]; // fewer rows than wgt
        let domain = array![0, 0, 1];
        let mut controls = HashMap::new();
        controls.insert(0i64, Array1::from(vec![10.0]));
        controls.insert(1i64, Array1::from(vec![5.0]));

        let res = calibrate_by_domain(
            wgt.view(),
            x_short.view(),
            domain.view(),
            &controls,
            None,
            false,
        );
        assert!(res.is_err());

        let x = array![[1.0], [2.0], [3.0]];
        let short_scale = array![1.0];
        let res = calibrate_by_domain(
            wgt.view(),
            x.view(),
            domain.view(),
            &controls,
            Some(short_scale.view()),
            false,
        );
        assert!(res.is_err());
    }
}
