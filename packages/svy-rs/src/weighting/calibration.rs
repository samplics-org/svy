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

/// Where a bounded solve stopped: the largest `|misfit| / (1 + |t|)` and the Newton steps taken.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BoundedFit {
    pub(crate) achieved: f64,
    pub(crate) steps: usize,
}

/// Bounds `[lower, upper]` on g = new / old weight, with the Newton loop's stopping rule.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BoundedSolve {
    pub(crate) lower: f64,
    pub(crate) upper: f64,
    pub(crate) tol: f64,
    pub(crate) max_iter: usize,
}

/// Linear calibration with g held in `[lower, upper]`, per column: R survey's `grake` with
/// `cal.linear`. Returns the weights and, per column, where the solve stopped; a column that
/// did not converge holds the last iterate.
pub(crate) fn calibrate_linear_bounded(
    wgt: ArrayView2<f64>,
    x_matrix: ArrayView2<f64>,
    totals: ArrayView1<f64>,
    scale: Option<ArrayView1<f64>>,
    solve: BoundedSolve,
) -> Result<(Array2<f64>, Vec<BoundedFit>)> {
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
        Some(s_arr) if s_arr.len() != n_obs => {
            return Err(WeightingError::DimensionMismatch {
                expected: n_obs,
                got: s_arr.len(),
            });
        }
        Some(s_arr) => s_arr.to_owned(),
        None => Array1::ones(n_obs),
    };
    if let Some((row, &value)) = s
        .iter()
        .enumerate()
        .find(|(_, x)| !(x.is_finite() && **x > 0.0))
    {
        return Err(WeightingError::ScaleNotPositive { row, value });
    }

    let cols: std::result::Result<Vec<_>, WeightingError> = (0..n_reps)
        .into_par_iter()
        .map(|r| grake_linear(wgt.column(r), x_matrix, totals, s.view(), solve))
        .collect();

    let mut result = Array2::zeros((n_obs, n_reps));
    let mut fits = Vec::with_capacity(n_reps);
    for (r, (col, fit)) in cols?.into_iter().enumerate() {
        result.column_mut(r).assign(&col);
        fits.push(fit);
    }
    Ok((result, fits))
}

/// One column of `calibrate_linear_bounded`, a literal port of `grake` so results match R.
fn grake_linear(
    wgt: ArrayView1<f64>,
    x: ArrayView2<f64>,
    totals: ArrayView1<f64>,
    scale: ArrayView1<f64>,
    solve: BoundedSolve,
) -> Result<(Array1<f64>, BoundedFit)> {
    let BoundedSolve {
        lower,
        upper,
        tol,
        max_iter,
    } = solve;
    let (n_obs, n_aux) = x.dim();
    // NaN must propagate as in R's pmin/pmax (f64::max drops it), so step halving sees it.
    let fm1 = |u: f64| {
        if u.is_nan() {
            f64::NAN
        } else {
            (u + 1.0).max(lower).min(upper) - 1.0
        }
    };
    // Strict, as in R: a row at a bound drops out of the Jacobian and stops moving.
    let deriv = |u: f64| {
        if u < upper - 1.0 && u > lower - 1.0 {
            1.0
        } else {
            0.0
        }
    };

    let mut w = wgt.to_owned();
    let mut sample_total = x.t().dot(&w);

    // R rescales weights far below the totals (e.g. unit weights) and reads the bounds after it.
    let ratios = &totals / &sample_total;
    let mut prescale = None;
    if ratios.iter().all(|r| r.is_finite()) && ratios.fold(f64::INFINITY, |m, &r| m.min(r)) > 20.0 {
        let c = ratios.mean().unwrap_or(1.0);
        w *= c;
        sample_total *= c;
        prescale = Some(c);
    }

    let x_eta = |eta: &Array1<f64>| -> Array1<f64> { x.dot(eta) / scale };
    let misfit_of = |u: &Array1<f64>| -> Array1<f64> {
        let shift: Array1<f64> = Array1::from_shape_fn(n_obs, |i| w[i] * fm1(u[i]));
        &totals - &sample_total - &x.t().dot(&shift)
    };
    let achieved_of = |misfit: &Array1<f64>| -> f64 {
        misfit
            .iter()
            .zip(totals.iter())
            .map(|(m, t)| m.abs() / (1.0 + t.abs()))
            .fold(0.0, |acc: f64, v| {
                if v.is_nan() || acc.is_nan() {
                    f64::NAN
                } else {
                    acc.max(v)
                }
            })
    };

    let mut eta: Array1<f64> = Array1::zeros(n_aux);
    let mut u = x_eta(&eta);
    let mut iter = 1usize;
    let mut steps = 0usize;
    let achieved = loop {
        steps += 1;
        let mut t_mat = Array2::zeros((n_aux, n_aux));
        for i in 0..n_obs {
            let c = w[i] * deriv(u[i]) / scale[i];
            if c == 0.0 {
                continue;
            }
            for j1 in 0..n_aux {
                let a = x[[i, j1]] * c;
                for j2 in 0..n_aux {
                    t_mat[[j1, j2]] += a * x[[i, j2]];
                }
            }
        }
        let mut deta = solve_pseudoinverse(&t_mat, &misfit_of(&u))?;
        eta += &deta;
        u = x_eta(&eta);
        while iter < max_iter && u.iter().any(|v| !fm1(*v).is_finite()) {
            iter += 1;
            deta /= 2.0;
            eta -= &deta;
            u = x_eta(&eta);
        }
        let achieved = achieved_of(&misfit_of(&u));
        if achieved < tol {
            break achieved;
        }
        iter += 1;
        if iter > max_iter {
            break achieved;
        }
    };

    let c = prescale.unwrap_or(1.0);
    let new_w = Array1::from_shape_fn(n_obs, |i| wgt[i] * (1.0 + fm1(u[i])) * c);
    Ok((new_w, BoundedFit { achieved, steps }))
}

/// Weights, the domains in ascending order, and each domain's fit per column.
type BoundedByDomain = (Array2<f64>, Vec<i64>, Vec<Vec<BoundedFit>>);

/// `calibrate_linear_bounded` separately within each domain, against that domain's totals.
pub(crate) fn calibrate_bounded_by_domain(
    wgt: ArrayView2<f64>,
    x_matrix: ArrayView2<f64>,
    domain: ArrayView1<i64>,
    controls: &HashMap<i64, Array1<f64>>,
    scale: Option<ArrayView1<f64>>,
    solve: BoundedSolve,
) -> Result<BoundedByDomain> {
    let (n_obs, _) = wgt.dim();
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
    let s = match scale {
        Some(s_arr) if s_arr.len() != n_obs => {
            return Err(WeightingError::DimensionMismatch {
                expected: n_obs,
                got: s_arr.len(),
            });
        }
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
    let mut tasks: Vec<_> = boundaries
        .iter()
        .map(|(&d, &(start, end))| (d, start, end))
        .collect();
    tasks.sort_by_key(|&(d, _, _)| d);

    let solved: std::result::Result<Vec<_>, WeightingError> = tasks
        .par_iter()
        .map(|&(d, start, end)| {
            let (w, fits) = calibrate_linear_bounded(
                wgt_sorted.slice(s![start..end, ..]),
                x_sorted.slice(s![start..end, ..]),
                controls[&d].view(),
                Some(s_sorted.slice(s![start..end])),
                solve,
            )?;
            Ok((start, end, w, fits))
        })
        .collect();

    let mut result_sorted = wgt_sorted.clone();
    let mut fits_by_domain = Vec::with_capacity(tasks.len());
    for (start, end, w, fits) in solved? {
        result_sorted.slice_mut(s![start..end, ..]).assign(&w);
        fits_by_domain.push(fits);
    }
    let domains = tasks.iter().map(|&(d, _, _)| d).collect();
    Ok((
        reorder_rows(result_sorted.view(), &unsort_idx),
        domains,
        fits_by_domain,
    ))
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

    fn solve(lower: f64, upper: f64, max_iter: usize) -> BoundedSolve {
        BoundedSolve {
            lower,
            upper,
            tol: 1e-12,
            max_iter,
        }
    }

    #[test]
    fn bounds_that_never_bind_reproduce_the_closed_form() {
        let wgt = array![[1.0, 2.0], [2.0, 1.0], [1.5, 1.5], [1.0, 3.0]];
        let x = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]];
        let totals = array![7.0, 18.0];
        let open = calibrate_linear(wgt.view(), x.view(), totals.view(), None, false).unwrap();
        let (bounded, fits) = calibrate_linear_bounded(
            wgt.view(),
            x.view(),
            totals.view(),
            None,
            solve(0.0, 10.0, 50),
        )
        .unwrap();
        for (a, b) in open.iter().zip(bounded.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-12);
        }
        assert!(fits.iter().all(|f| f.achieved < 1e-12 && f.steps == 1));
    }

    #[test]
    fn a_row_at_the_bound_stops_moving_and_the_rest_meet_the_total() {
        // x = 1..4, unit weights, total 15: unbounded g = 1 + x/6 puts row 4 at 5/3. Held at
        // 1.6, rows 1-3 solve 6 + 14λ = 15 - 6.4, so λ = 2.6/14.
        let wgt = array![[1.0], [1.0], [1.0], [1.0]];
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let (w, fits) = calibrate_linear_bounded(
            wgt.view(),
            x.view(),
            array![15.0].view(),
            None,
            solve(0.5, 1.6, 50),
        )
        .unwrap();
        let lam = 2.6 / 14.0;
        let expected = [1.0 + lam, 1.0 + 2.0 * lam, 1.0 + 3.0 * lam, 1.6];
        for i in 0..4 {
            assert_abs_diff_eq!(w[[i, 0]], expected[i], epsilon = 1e-14);
        }
        assert_eq!(fits[0].steps, 2);
        assert!(fits[0].achieved < 1e-12);
    }

    #[test]
    fn unreachable_totals_keep_the_last_iterate_and_report_the_miss() {
        // At most 1.4 * (1 + 2 + 3 + 4) = 14 < 15: every row ends at the bound, missing by 1/16.
        let wgt = array![[1.0], [1.0], [1.0], [1.0]];
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let (w, fits) = calibrate_linear_bounded(
            wgt.view(),
            x.view(),
            array![15.0].view(),
            None,
            solve(0.5, 1.4, 5),
        )
        .unwrap();
        assert!(w.iter().all(|&v| (v - 1.4).abs() < 1e-14));
        assert_eq!(fits[0].steps, 5);
        assert_abs_diff_eq!(fits[0].achieved, 1.0 / 16.0, epsilon = 1e-14);
    }

    #[test]
    fn weights_far_below_the_totals_are_rescaled_before_the_bounds_apply() {
        // Totals 50x the weighted sums: R rescales by 50 first, so g = 1 fits within (0.9, 1.1).
        let wgt = array![[1.0], [1.0]];
        let x = array![[1.0], [1.0]];
        let (w, fits) = calibrate_linear_bounded(
            wgt.view(),
            x.view(),
            array![100.0].view(),
            None,
            solve(0.9, 1.1, 50),
        )
        .unwrap();
        assert_abs_diff_eq!(w[[0, 0]], 50.0, epsilon = 1e-12);
        assert_abs_diff_eq!(w[[1, 0]], 50.0, epsilon = 1e-12);
        assert!(fits[0].achieved < 1e-12);
    }

    #[test]
    fn bounded_domains_solve_apart_and_report_in_domain_order() {
        let wgt = array![[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]];
        let x = array![[1.0], [2.0], [3.0], [4.0], [1.0], [2.0], [3.0], [4.0]];
        let domain = array![7, 7, 7, 7, 3, 3, 3, 3];
        let mut controls = HashMap::new();
        controls.insert(7i64, array![15.0]);
        controls.insert(3i64, array![10.0]);
        let (w, domains, fits) = calibrate_bounded_by_domain(
            wgt.view(),
            x.view(),
            domain.view(),
            &controls,
            None,
            solve(0.5, 1.6, 50),
        )
        .unwrap();
        assert_eq!(domains, vec![3, 7]);
        // Domain 3 already meets its total; domain 7 is the case above.
        let lam = 2.6 / 14.0;
        let expected = [
            1.0 + lam,
            1.0 + 2.0 * lam,
            1.0 + 3.0 * lam,
            1.6,
            1.0,
            1.0,
            1.0,
            1.0,
        ];
        for i in 0..8 {
            assert_abs_diff_eq!(w[[i, 0]], expected[i], epsilon = 1e-14);
        }
        assert_eq!((fits[0][0].steps, fits[1][0].steps), (1, 2));
    }

    #[test]
    fn bounded_bad_dims_are_errors() {
        let wgt = array![[1.0], [1.0], [1.0]];
        let x = array![[1.0], [2.0], [3.0]];
        let res = calibrate_linear_bounded(
            wgt.view(),
            x.view(),
            array![10.0, 20.0].view(),
            None,
            solve(0.5, 2.0, 50),
        );
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
        let res = calibrate_linear_bounded(
            wgt.view(),
            x.view(),
            array![10.0].view(),
            Some(array![1.0, 0.0, 1.0].view()),
            solve(0.5, 2.0, 50),
        );
        assert!(
            matches!(&res, Err(WeightingError::ScaleNotPositive { row: 1, .. })),
            "got {res:?}"
        );
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
