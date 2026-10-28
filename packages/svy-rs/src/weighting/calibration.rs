//! Linear (Deville-Särndal) calibration, overall and by domain.

use super::utils::{Result, WeightingError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use rayon::prelude::*;
use std::collections::HashMap;

/// Solves (X' diag(w/s) X) λ = t - X'w per column and returns w * g, with g = 1 + Xλ / s.
/// With `additive`, returns g itself.
pub(crate) fn calibrate_linear(
    wgt: ArrayView2<f64>,
    x_matrix: ArrayView2<f64>,
    totals: ArrayView1<f64>,
    scale: Option<ArrayView1<f64>>,
    additive: bool,
) -> Result<Array2<f64>> {
    let (n_obs, n_reps) = wgt.dim();
    let (n_obs_x, n_aux) = x_matrix.dim();

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

    let results: std::result::Result<Vec<_>, WeightingError> = (0..n_reps)
        .into_par_iter()
        .map(|r| {
            let wgt_col = wgt.column(r);
            calibrate_linear_single(wgt_col, x_matrix, totals, s.view(), additive)
        })
        .collect();

    let cal_cols = results?;

    let mut result = Array2::zeros((n_obs, n_reps));
    for (col_idx, col) in cal_cols.into_iter().enumerate() {
        result.column_mut(col_idx).assign(&col);
    }

    Ok(result)
}

fn calibrate_linear_single(
    wgt: ArrayView1<f64>,
    x_matrix: ArrayView2<f64>,
    totals: ArrayView1<f64>,
    scale: ArrayView1<f64>,
    additive: bool,
) -> Result<Array1<f64>> {
    let n_obs = wgt.len();
    let n_aux = totals.len();

    let mut x_w: Array1<f64> = Array1::zeros(n_aux);
    for j in 0..n_aux {
        for i in 0..n_obs {
            x_w[j] += x_matrix[[i, j]] * wgt[i];
        }
    }

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

    let b = &totals - &x_w;

    let lambda = solve_linear_system(&a_matrix, &b)?;

    let mut g_factors = Array1::ones(n_obs);
    for i in 0..n_obs {
        let mut x_lambda_i = 0.0;
        for j in 0..n_aux {
            x_lambda_i += x_matrix[[i, j]] * lambda[j];
        }
        g_factors[i] = 1.0 + x_lambda_i / scale[i];
    }

    if additive {
        Ok(g_factors)
    } else {
        Ok(&wgt * &g_factors)
    }
}

/// Gaussian elimination with partial pivoting; a singular system gets the minimum-norm solution.
fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    let n = a.nrows();

    if a.ncols() != n || b.len() != n {
        return Err(WeightingError::InvalidInput(
            "Matrix dimensions incompatible for solving".to_string(),
        ));
    }

    // Solve D A D, D = diag(A)^(-1/2): an auxiliary's units then decide neither singularity nor
    // which directions the pseudo-inverse keeps. A zero diagonal (no weight) stays unscaled.
    let d: Array1<f64> = (0..n)
        .map(|i| {
            let v = a[[i, i]];
            if v > 0.0 { 1.0 / v.sqrt() } else { 1.0 }
        })
        .collect();
    let scaled_a = Array2::from_shape_fn((n, n), |(i, j)| a[[i, j]] * d[i] * d[j]);
    let scaled_b = b * &d;

    let mut aug = Array2::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = scaled_a[[i, j]];
        }
        aug[[i, n]] = scaled_b[i];
    }

    // Relative to the scaled system, whose nonzero diagonal entries are 1.
    let pivot_tol = 256.0 * f64::EPSILON;

    for k in 0..n {
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
        if max_val <= pivot_tol {
            return Ok(solve_pseudoinverse(&scaled_a, &scaled_b)? * &d);
        }

        if max_row != k {
            for j in 0..=n {
                let temp = aug[[k, j]];
                aug[[k, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        for i in (k + 1)..n {
            let factor = aug[[i, k]] / aug[[k, k]];
            for j in k..=n {
                aug[[i, j]] -= factor * aug[[k, j]];
            }
        }
    }

    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum -= aug[[i, j]] * x[j];
        }
        x[i] = sum / aug[[i, i]];
    }
    x *= &d;

    Ok(x)
}

/// `A⁺ b` by SVD, dropping singular values <= 256·ε·max, as `MASS::ginv` in R's `grake`.
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

/// `calibrate_linear` separately within each domain, against that domain's control totals.
pub(crate) fn calibrate_by_domain(
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

    let s = match scale {
        Some(s_arr) => s_arr.to_owned(),
        None => Array1::ones(n_obs),
    };

    let (domain_sorted, sort_idx, unsort_idx) = sort_by_domain(&domain);

    let wgt_sorted = reorder_rows(wgt, &sort_idx);
    let x_sorted = reorder_rows_2d(x_matrix, &sort_idx);
    let s_sorted = reorder_array(&s.view(), &sort_idx);

    let boundaries = find_domain_boundaries(&domain_sorted);

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

/// `calibrate_linear` with `additive = false`.
pub(crate) fn calibrate_parallel(
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
    use approx::assert_abs_diff_eq;
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
        // Linear calibration depends only on the span of X, which adding male leaves unchanged.
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

        let mut x_cal = 0.0;
        for i in 0..4 {
            x_cal += x[[i, 0]] * result[[i, 0]];
        }
        assert_abs_diff_eq!(x_cal, 15.0, epsilon = 1e-12);
    }

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
        assert!(
            matches!(
                &res,
                Err(WeightingError::DimensionMismatch {
                    expected: 3,
                    got: 1
                })
            ),
            "got {res:?}"
        );

        // totals longer than n_aux
        let long_totals = array![10.0, 20.0];
        let res = calibrate_parallel(wgt.view(), x.view(), long_totals.view(), None);
        assert!(
            matches!(
                &res,
                Err(WeightingError::DimensionMismatch {
                    expected: 1,
                    got: 2
                })
            ),
            "got {res:?}"
        );
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
        assert!(
            matches!(
                &res,
                Err(WeightingError::DimensionMismatch {
                    expected: 3,
                    got: 2
                })
            ),
            "got {res:?}"
        );

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
        assert!(
            matches!(
                &res,
                Err(WeightingError::DimensionMismatch {
                    expected: 3,
                    got: 1
                })
            ),
            "got {res:?}"
        );
    }
}
