//! PyO3 wrappers for replicate-weight variance; the frame-level work lives in `svy_core::estimation::replication_frame`.

use crate::errors::to_py_err;
use crate::estimation::into_contiguous;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use svy_core::estimation::association::AssocKind;
use svy_core::estimation::error::EstimationError;
use svy_core::estimation::replication::VarianceCenter;
use svy_core::estimation::replication_frame::*;
use svy_core::estimation::taylor::{SrsRef, SvyQuantileMethod};
use svy_core::estimation::taylor_frame::parse_srs_ref;

fn parse_variance_center(center: &str) -> PyResult<VarianceCenter> {
    VarianceCenter::parse(center)
        .ok_or_else(|| to_py_err(&EstimationError::UnknownCenter(center.to_string())))
}

/// The SRS reference when a design effect is asked for; `None` leaves the `deff` column out.
fn parse_deff_ref(deff_ref: Option<&str>) -> PyResult<Option<SrsRef>> {
    deff_ref
        .map(|r| parse_srs_ref(Some(r), None))
        .transpose()
        .map_err(|e| to_py_err(&e))
}

// ============================================================================
// Mean
// ============================================================================

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, rep_weight_cols, rep_coefs, center="rep_mean", degrees_of_freedom=None, by_col=None, domain_mask_col=None, deff_ref=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn replicate_mean(
    _py: Python,
    data: PyDataFrame,
    value_col: String,
    weight_col: String,
    rep_weight_cols: Vec<String>,
    rep_coefs: Vec<f64>,
    center: &str,
    degrees_of_freedom: Option<f64>,
    by_col: Option<String>,
    domain_mask_col: Option<String>,
    deff_ref: Option<String>,
) -> PyResult<(PyDataFrame, Option<Vec<f64>>)> {
    let df = into_contiguous(data);
    let srs = parse_deff_ref(deff_ref.as_deref())?;
    let n_reps = rep_weight_cols.len();
    let variance_center = parse_variance_center(center)?;
    // Coefficients come from the RepWeights variant on the Python side.
    if rep_coefs.len() != n_reps {
        return Err(to_py_err(&EstimationError::RepCoefsLength {
            got: rep_coefs.len(),
            expected: n_reps,
        }));
    }
    let df_val = degrees_of_freedom.unwrap_or(n_reps.saturating_sub(1) as f64);

    let result = _py.detach(|| match by_col.as_ref() {
        None => compute_replicate_mean_ungrouped(
            &df,
            &value_col,
            &weight_col,
            &rep_weight_cols,
            &rep_coefs,
            variance_center,
            df_val,
            domain_mask_col.as_deref(),
            srs,
        )
        .map(|out| (out, None)),
        Some(by) => compute_replicate_mean_grouped(
            &df,
            &value_col,
            &weight_col,
            &rep_weight_cols,
            &rep_coefs,
            variance_center,
            df_val,
            by,
            srs,
        )
        .map(|(out, cov)| (out, Some(cov))),
    });
    result
        .map(|(out, cov)| (PyDataFrame(out), cov))
        .map_err(|e| to_py_err(&e))
}

// ============================================================================
// Total
// ============================================================================

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, rep_weight_cols, rep_coefs, center="rep_mean", degrees_of_freedom=None, by_col=None, domain_mask_col=None, deff_ref=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn replicate_total(
    _py: Python,
    data: PyDataFrame,
    value_col: String,
    weight_col: String,
    rep_weight_cols: Vec<String>,
    rep_coefs: Vec<f64>,
    center: &str,
    degrees_of_freedom: Option<f64>,
    by_col: Option<String>,
    domain_mask_col: Option<String>,
    deff_ref: Option<String>,
) -> PyResult<(PyDataFrame, Option<Vec<f64>>)> {
    let df = into_contiguous(data);
    let srs = parse_deff_ref(deff_ref.as_deref())?;
    let n_reps = rep_weight_cols.len();
    let variance_center = parse_variance_center(center)?;
    // Coefficients come from the RepWeights variant on the Python side.
    if rep_coefs.len() != n_reps {
        return Err(to_py_err(&EstimationError::RepCoefsLength {
            got: rep_coefs.len(),
            expected: n_reps,
        }));
    }
    let df_val = degrees_of_freedom.unwrap_or(n_reps.saturating_sub(1) as f64);

    let result = _py.detach(|| match by_col.as_ref() {
        None => compute_replicate_total_ungrouped(
            &df,
            &value_col,
            &weight_col,
            &rep_weight_cols,
            &rep_coefs,
            variance_center,
            df_val,
            domain_mask_col.as_deref(),
            srs,
        )
        .map(|out| (out, None)),
        Some(by) => compute_replicate_total_grouped(
            &df,
            &value_col,
            &weight_col,
            &rep_weight_cols,
            &rep_coefs,
            variance_center,
            df_val,
            by,
            srs,
        )
        .map(|(out, cov)| (out, Some(cov))),
    });
    result
        .map(|(out, cov)| (PyDataFrame(out), cov))
        .map_err(|e| to_py_err(&e))
}

// ============================================================================
// Association (covariance / correlation)
// ============================================================================

/// Replicate-weight covariance or correlation over one or more column pairs.
/// Replicates re-centre on their own means but keep the full-sample Kish `n`, like R's `svrepvar`.
#[pyfunction]
#[pyo3(signature = (data, y_cols, x_cols, kind, weight_col, rep_weight_cols, rep_coefs, center="rep_mean", degrees_of_freedom=None, by_col=None, domain_mask_col=None, deff_ref=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn replicate_assoc(
    _py: Python,
    data: PyDataFrame,
    y_cols: Vec<String>,
    x_cols: Vec<String>,
    kind: String,
    weight_col: String,
    rep_weight_cols: Vec<String>,
    rep_coefs: Vec<f64>,
    center: &str,
    degrees_of_freedom: Option<f64>,
    by_col: Option<String>,
    domain_mask_col: Option<String>,
    deff_ref: Option<String>,
) -> PyResult<PyDataFrame> {
    let df = into_contiguous(data);
    let srs = parse_deff_ref(deff_ref.as_deref())?;
    let n_reps = rep_weight_cols.len();
    let variance_center = parse_variance_center(center)?;
    // Coefficients come from the RepWeights variant on the Python side.
    if rep_coefs.len() != n_reps {
        return Err(to_py_err(&EstimationError::RepCoefsLength {
            got: rep_coefs.len(),
            expected: n_reps,
        }));
    }
    if y_cols.len() != x_cols.len() {
        return Err(to_py_err(&EstimationError::PairLengths {
            y: y_cols.len(),
            x: x_cols.len(),
        }));
    }
    let df_val = degrees_of_freedom.unwrap_or(n_reps.saturating_sub(1) as f64);

    let result = _py.detach(|| {
        let assoc_kind = AssocKind::from_name(&kind)?;
        compute_replicate_assoc(
            &df,
            &y_cols,
            &x_cols,
            assoc_kind,
            &weight_col,
            &rep_weight_cols,
            &rep_coefs,
            variance_center,
            df_val,
            by_col.as_deref(),
            domain_mask_col.as_deref(),
            srs,
        )
    });
    result.map(PyDataFrame).map_err(|e| to_py_err(&e))
}

// ============================================================================
// Ratio
// ============================================================================

#[pyfunction]
#[pyo3(signature = (data, numerator_col, denominator_col, weight_col, rep_weight_cols, rep_coefs, center="rep_mean", degrees_of_freedom=None, by_col=None, domain_mask_col=None, deff_ref=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn replicate_ratio(
    _py: Python,
    data: PyDataFrame,
    numerator_col: String,
    denominator_col: String,
    weight_col: String,
    rep_weight_cols: Vec<String>,
    rep_coefs: Vec<f64>,
    center: &str,
    degrees_of_freedom: Option<f64>,
    by_col: Option<String>,
    domain_mask_col: Option<String>,
    deff_ref: Option<String>,
) -> PyResult<(PyDataFrame, Option<Vec<f64>>)> {
    let df = into_contiguous(data);
    let srs = parse_deff_ref(deff_ref.as_deref())?;
    let n_reps = rep_weight_cols.len();
    let variance_center = parse_variance_center(center)?;
    // Coefficients come from the RepWeights variant on the Python side.
    if rep_coefs.len() != n_reps {
        return Err(to_py_err(&EstimationError::RepCoefsLength {
            got: rep_coefs.len(),
            expected: n_reps,
        }));
    }
    let df_val = degrees_of_freedom.unwrap_or(n_reps.saturating_sub(1) as f64);

    let result = _py.detach(|| match by_col.as_ref() {
        None => compute_replicate_ratio_ungrouped(
            &df,
            &numerator_col,
            &denominator_col,
            &weight_col,
            &rep_weight_cols,
            &rep_coefs,
            variance_center,
            df_val,
            domain_mask_col.as_deref(),
            srs,
        )
        .map(|out| (out, None)),
        Some(by) => compute_replicate_ratio_grouped(
            &df,
            &numerator_col,
            &denominator_col,
            &weight_col,
            &rep_weight_cols,
            &rep_coefs,
            variance_center,
            df_val,
            by,
            srs,
        )
        .map(|(out, cov)| (out, Some(cov))),
    });
    result
        .map(|(out, cov)| (PyDataFrame(out), cov))
        .map_err(|e| to_py_err(&e))
}

// ============================================================================
// Proportion
// ============================================================================

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, rep_weight_cols, rep_coefs, center="rep_mean", degrees_of_freedom=None, by_col=None, domain_mask_col=None, deff_ref=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn replicate_prop(
    _py: Python,
    data: PyDataFrame,
    value_col: String,
    weight_col: String,
    rep_weight_cols: Vec<String>,
    rep_coefs: Vec<f64>,
    center: &str,
    degrees_of_freedom: Option<f64>,
    by_col: Option<String>,
    domain_mask_col: Option<String>,
    deff_ref: Option<String>,
) -> PyResult<(PyDataFrame, Option<Vec<f64>>)> {
    replicate_levels(
        _py,
        data,
        value_col,
        weight_col,
        rep_weight_cols,
        rep_coefs,
        center,
        degrees_of_freedom,
        by_col,
        domain_mask_col,
        deff_ref,
        true,
    )
}

/// Count per level (R `svytotal(~factor(y))`) with the joint covariance across levels and groups.
#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, rep_weight_cols, rep_coefs, center="rep_mean", degrees_of_freedom=None, by_col=None, domain_mask_col=None, deff_ref=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn replicate_factor_total(
    _py: Python,
    data: PyDataFrame,
    value_col: String,
    weight_col: String,
    rep_weight_cols: Vec<String>,
    rep_coefs: Vec<f64>,
    center: &str,
    degrees_of_freedom: Option<f64>,
    by_col: Option<String>,
    domain_mask_col: Option<String>,
    deff_ref: Option<String>,
) -> PyResult<(PyDataFrame, Option<Vec<f64>>)> {
    replicate_levels(
        _py,
        data,
        value_col,
        weight_col,
        rep_weight_cols,
        rep_coefs,
        center,
        degrees_of_freedom,
        by_col,
        domain_mask_col,
        deff_ref,
        false,
    )
}

/// `normalize` gives each level's share of the (domain) weight instead of its weighted count.
#[allow(clippy::too_many_arguments)]
fn replicate_levels(
    _py: Python,
    data: PyDataFrame,
    value_col: String,
    weight_col: String,
    rep_weight_cols: Vec<String>,
    rep_coefs: Vec<f64>,
    center: &str,
    degrees_of_freedom: Option<f64>,
    by_col: Option<String>,
    domain_mask_col: Option<String>,
    deff_ref: Option<String>,
    normalize: bool,
) -> PyResult<(PyDataFrame, Option<Vec<f64>>)> {
    let df = into_contiguous(data);
    let srs = parse_deff_ref(deff_ref.as_deref())?;
    let n_reps = rep_weight_cols.len();
    let variance_center = parse_variance_center(center)?;
    // Coefficients come from the RepWeights variant on the Python side.
    if rep_coefs.len() != n_reps {
        return Err(to_py_err(&EstimationError::RepCoefsLength {
            got: rep_coefs.len(),
            expected: n_reps,
        }));
    }
    let df_val = degrees_of_freedom.unwrap_or(n_reps.saturating_sub(1) as f64);

    let result = _py.detach(|| match by_col.as_ref() {
        None => compute_replicate_prop_ungrouped(
            &df,
            &value_col,
            &weight_col,
            &rep_weight_cols,
            &rep_coefs,
            variance_center,
            df_val,
            domain_mask_col.as_deref(),
            normalize,
            srs,
        ),
        Some(by) => compute_replicate_prop_grouped(
            &df,
            &value_col,
            &weight_col,
            &rep_weight_cols,
            &rep_coefs,
            variance_center,
            df_val,
            by,
            normalize,
            srs,
        ),
    });
    result
        .map(|(out, cov)| (PyDataFrame(out), Some(cov)))
        .map_err(|e| to_py_err(&e))
}

// ============================================================================
// Quantiles (median is the p = 0.5 case)
// ============================================================================

/// Replicate-weight quantiles: one row per probability (and domain), with a `prob` column.
#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, rep_weight_cols, rep_coefs, probs, center="rep_mean", degrees_of_freedom=None, by_col=None, quantile_method=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn replicate_quantile(
    _py: Python,
    data: PyDataFrame,
    value_col: String,
    weight_col: String,
    rep_weight_cols: Vec<String>,
    rep_coefs: Vec<f64>,
    probs: Vec<f64>,
    center: &str,
    degrees_of_freedom: Option<f64>,
    by_col: Option<String>,
    quantile_method: Option<String>,
) -> PyResult<PyDataFrame> {
    let df = into_contiguous(data);
    let n_reps = rep_weight_cols.len();
    let variance_center = parse_variance_center(center)?;
    // Coefficients come from the RepWeights variant on the Python side.
    if rep_coefs.len() != n_reps {
        return Err(to_py_err(&EstimationError::RepCoefsLength {
            got: rep_coefs.len(),
            expected: n_reps,
        }));
    }
    let q_method = quantile_method
        .as_deref()
        .map(SvyQuantileMethod::parse)
        .transpose()
        .map_err(|e| to_py_err(&e))?
        .unwrap_or(SvyQuantileMethod::Higher);
    let df_val = degrees_of_freedom.unwrap_or(n_reps.saturating_sub(1) as f64);

    let result = _py.detach(|| match by_col.as_deref() {
        None => compute_replicate_quantile_ungrouped(
            &df,
            &value_col,
            &weight_col,
            &rep_weight_cols,
            &rep_coefs,
            variance_center,
            df_val,
            &probs,
            q_method,
        ),
        Some(by) => compute_replicate_quantile_grouped(
            &df,
            &value_col,
            &weight_col,
            &rep_weight_cols,
            &rep_coefs,
            variance_center,
            df_val,
            by,
            &probs,
            q_method,
        ),
    });
    result.map(PyDataFrame).map_err(|e| to_py_err(&e))
}

#[pyfunction]
#[pyo3(signature = (data, value_col, weight_col, rep_weight_cols, rep_coefs, center="rep_mean", degrees_of_freedom=None, by_col=None, quantile_method=None))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn replicate_median(
    _py: Python,
    data: PyDataFrame,
    value_col: String,
    weight_col: String,
    rep_weight_cols: Vec<String>,
    rep_coefs: Vec<f64>,
    center: &str,
    degrees_of_freedom: Option<f64>,
    by_col: Option<String>,
    quantile_method: Option<String>,
) -> PyResult<PyDataFrame> {
    let out = replicate_quantile(
        _py,
        data,
        value_col,
        weight_col,
        rep_weight_cols,
        rep_coefs,
        vec![0.5],
        center,
        degrees_of_freedom,
        by_col,
        quantile_method,
    )?;
    // The median schema has no probability column.
    let mut result = out.0;
    result
        .drop_in_place("prob")
        .map_err(|e| to_py_err(&EstimationError::from(e)))?;
    Ok(PyDataFrame(result))
}
