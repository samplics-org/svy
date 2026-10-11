// src/sampling/api.rs
//
// PyO3-facing wrappers for SRS and PPS selection.
// Arrays arrive as numpy arrays (read in place) or sequences (copied).
// Returned as tuples of Vec<T> which PyO3 converts to Python lists.

use numpy::{Element, PyReadonlyArray1};
use pyo3::prelude::*;
use std::borrow::Cow;
use std::collections::HashMap;

use crate::sampling::pps::{PpsMethod, PpsN, select_pps};
use crate::sampling::srs::{SrsN, select_srs};

/// A 1-D input: a numpy array of exactly `T`, or any sequence (older callers pass lists).
#[derive(FromPyObject)]
pub enum Array1<'py, T: Element> {
    Array(PyReadonlyArray1<'py, T>),
    Seq(Vec<T>),
}

impl<T: Element + Clone> Array1<'_, T> {
    fn values(&self) -> Cow<'_, [T]> {
        match self {
            Array1::Array(a) => match a.as_slice() {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(a.as_array().to_vec()),
            },
            Array1::Seq(v) => Cow::Borrowed(v),
        }
    }
}

// ---------------------------------------------------------------------------
// SRS
// ---------------------------------------------------------------------------

/// Simple Random Sampling.
///
/// Parameters
/// ----------
/// frame    : row index values (svy_row_index column as i64 array)
/// n_scalar : sample size when unstratified or scalar broadcast
/// n_map    : per-stratum {stratum_label: n} — mutually exclusive with n_scalar
/// stratum  : stratum label per row (None for unstratified)
/// wr       : sampling with replacement
/// seed     : RNG seed (None → 0)
///
/// Returns
/// -------
/// (selected_indices, hits, probs) as Python lists
#[pyfunction]
#[pyo3(signature = (frame, n_scalar=None, n_map=None, stratum=None, wr=false, seed=None))]
pub fn select_srs_rs(
    frame: Array1<'_, i64>,
    n_scalar: Option<usize>,
    n_map: Option<HashMap<i64, usize>>,
    stratum: Option<Array1<'_, i64>>,
    wr: bool,
    seed: Option<u64>,
) -> PyResult<(Vec<i64>, Vec<i64>, Vec<f64>)> {
    let n = match (n_scalar, n_map) {
        (Some(v), None) => SrsN::Scalar(v),
        (None, Some(m)) => SrsN::PerStratum(m),
        (Some(_), Some(_)) => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Provide either n_scalar or n_map, not both",
            ));
        }
        (None, None) => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Provide either n_scalar or n_map",
            ));
        }
    };

    let stratum = stratum.as_ref().map(Array1::values);

    select_srs(&frame.values(), n, stratum.as_deref(), wr, seed)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}

// ---------------------------------------------------------------------------
// PPS
// ---------------------------------------------------------------------------

/// Probability Proportional to Size sampling.
///
/// Parameters
/// ----------
/// frame               : row index values (i64)
/// mos                 : measure of size per row (f64)
/// n_scalar            : sample size (scalar or broadcast)
/// n_map               : per-stratum {stratum_label: n}
/// stratum             : stratum label per row (None for unstratified)
/// method              : "sys" | "wr" | "brewer" | "murphy" | "rs"
/// certainty_threshold : units with pi >= threshold selected with certainty
/// seed                : RNG seed
///
/// Returns
/// -------
/// (selected_indices, hits, probs, certainty_flags)
#[pyfunction]
#[pyo3(signature = (
    frame,
    mos,
    n_scalar=None,
    n_map=None,
    stratum=None,
    method="sys",
    certainty_threshold=1.0,
    seed=None
))]
pub fn select_pps_rs(
    frame: Array1<'_, i64>,
    mos: Array1<'_, f64>,
    n_scalar: Option<usize>,
    n_map: Option<HashMap<i64, usize>>,
    stratum: Option<Array1<'_, i64>>,
    method: &str,
    certainty_threshold: f64,
    seed: Option<u64>,
) -> PyResult<(Vec<i64>, Vec<i64>, Vec<f64>, Vec<bool>)> {
    let pps_method = PpsMethod::from_str(method).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown PPS method {method:?}. Use: sys, wr, brewer, murphy, rs"
        ))
    })?;

    let n = match (n_scalar, n_map) {
        (Some(v), None) => PpsN::Scalar(v),
        (None, Some(m)) => PpsN::PerStratum(m),
        (Some(_), Some(_)) => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Provide either n_scalar or n_map, not both",
            ));
        }
        (None, None) => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Provide either n_scalar or n_map",
            ));
        }
    };

    let stratum = stratum.as_ref().map(Array1::values);

    select_pps(
        &frame.values(),
        n,
        &mos.values(),
        stratum.as_deref(),
        pps_method,
        certainty_threshold,
        seed,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}
