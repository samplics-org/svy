//! Post-stratification.

use super::normalization::normalize_impl;
use super::utils::{Result, WeightingError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Normalization requiring exactly one non-negative control per stratum, in sorted stratum order.
pub fn poststratify_impl(
    wgt: ArrayView2<f64>,
    by_arr: ArrayView1<i64>,
    control: &Array1<f64>,
) -> Result<Array2<f64>> {
    let n_obs = wgt.nrows();

    if by_arr.len() != n_obs {
        return Err(WeightingError::DimensionMismatch {
            expected: n_obs,
            got: by_arr.len(),
        });
    }

    let unique_strata = get_unique_sorted(&by_arr);
    let n_strata = unique_strata.len();

    if control.len() != n_strata {
        return Err(WeightingError::InvalidInput(format!(
            "Post-stratification requires exact match: {} strata in data but {} controls provided",
            n_strata,
            control.len()
        )));
    }

    for (i, &ctrl) in control.iter().enumerate() {
        if ctrl < 0.0 {
            return Err(WeightingError::InvalidInput(format!(
                "Control total for stratum {} is negative: {}",
                i, ctrl
            )));
        }
    }

    normalize_impl(wgt, Some(&by_arr.to_owned()), Some(control))
}

fn get_unique_sorted(arr: &ArrayView1<i64>) -> Vec<i64> {
    use std::collections::BTreeSet;

    let unique: BTreeSet<i64> = arr.iter().copied().collect();
    unique.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_poststratify() {
        let wgt = array![[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]];
        let strata = array![0, 0, 1, 1];
        let controls = array![10.0, 20.0];

        let result = poststratify_impl(wgt.view(), strata.view(), &controls).unwrap();

        let sum_s0_r0 = result[[0, 0]] + result[[1, 0]];
        assert_relative_eq!(sum_s0_r0, 10.0, epsilon = 1e-12);

        let sum_s1_r0 = result[[2, 0]] + result[[3, 0]];
        assert_relative_eq!(sum_s1_r0, 20.0, epsilon = 1e-12);
    }

    #[test]
    fn test_poststratify_validation() {
        let wgt = array![[1.0], [1.0], [1.0]];
        let strata = array![0, 0, 1];
        let controls = array![10.0];

        let result = poststratify_impl(wgt.view(), strata.view(), &controls);
        assert!(
            matches!(&result, Err(WeightingError::InvalidInput(m)) if m.contains("2 strata in data but 1 controls")),
            "got {result:?}"
        );
    }
}
