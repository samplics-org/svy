//! PyO3 wrappers for SRS and PPS selection.

use numpy::{Element, PyReadonlyArray1};
use pyo3::prelude::*;
use std::borrow::Cow;
use std::collections::HashMap;

use crate::errors::to_py_err;
use crate::sampling::pps::{PpsDraw, PpsMethod, PpsN, select_pps};
use crate::sampling::srs::{SamplingError, SrsDraw, SrsN, select_srs};

/// A 1-D input: a numpy array of exactly `T`, or any sequence (older callers pass lists).
#[derive(FromPyObject)]
pub(crate) enum Array1<'py, T: Element> {
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

/// The sample size as a scalar or per-stratum map; exactly one must be given.
fn parse_n<N>(
    n_scalar: Option<usize>,
    n_map: Option<HashMap<i64, usize>>,
    scalar: impl FnOnce(usize) -> N,
    per_stratum: impl FnOnce(HashMap<i64, usize>) -> N,
) -> PyResult<N> {
    match (n_scalar, n_map) {
        (Some(v), None) => Ok(scalar(v)),
        (None, Some(m)) => Ok(per_stratum(m)),
        (Some(_), Some(_)) => Err(to_py_err(&SamplingError::InvalidInput(
            "Provide either n_scalar or n_map, not both".into(),
        ))),
        (None, None) => Err(to_py_err(&SamplingError::InvalidInput(
            "Provide either n_scalar or n_map".into(),
        ))),
    }
}

// ---------------------------------------------------------------------------
// SRS
// ---------------------------------------------------------------------------

/// SRS of the row indices in `frame`: (selected, hits, probs); pass `n_scalar` or `n_map`.
#[pyfunction]
#[pyo3(signature = (frame, n_scalar=None, n_map=None, stratum=None, wr=false, seed=None))]
pub(crate) fn select_srs_rs(
    frame: Array1<'_, i64>,
    n_scalar: Option<usize>,
    n_map: Option<HashMap<i64, usize>>,
    stratum: Option<Array1<'_, i64>>,
    wr: bool,
    seed: Option<u64>,
) -> PyResult<SrsDraw> {
    let n = parse_n(n_scalar, n_map, SrsN::Scalar, SrsN::PerStratum)?;

    let stratum = stratum.as_ref().map(Array1::values);

    let seed = seed.ok_or_else(|| to_py_err(&SamplingError::MissingSeed))?;
    select_srs(&frame.values(), n, stratum.as_deref(), wr, seed).map_err(|e| to_py_err(&e))
}

// ---------------------------------------------------------------------------
// PPS
// ---------------------------------------------------------------------------

/// PPS of the row indices in `frame` by `mos`: (selected, hits, probs, certainty flags).
/// Units with pi >= `certainty_threshold` are taken with certainty.
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn select_pps_rs(
    frame: Array1<'_, i64>,
    mos: Array1<'_, f64>,
    n_scalar: Option<usize>,
    n_map: Option<HashMap<i64, usize>>,
    stratum: Option<Array1<'_, i64>>,
    method: &str,
    certainty_threshold: f64,
    seed: Option<u64>,
) -> PyResult<PpsDraw> {
    let pps_method = PpsMethod::from_str(method)
        .ok_or_else(|| to_py_err(&SamplingError::UnknownMethod(method.to_string())))?;

    let n = parse_n(n_scalar, n_map, PpsN::Scalar, PpsN::PerStratum)?;

    let stratum = stratum.as_ref().map(Array1::values);

    let seed = seed.ok_or_else(|| to_py_err(&SamplingError::MissingSeed))?;
    select_pps(
        &frame.values(),
        n,
        &mos.values(),
        stratum.as_deref(),
        pps_method,
        certainty_threshold,
        seed,
    )
    .map_err(|e| to_py_err(&e))
}

// ---------------------------------------------------------------------------
// Many draws
// ---------------------------------------------------------------------------

/// `select_srs_rs` once per seed, in parallel; draw r is that call with `seeds[r]`.
#[pyfunction]
#[pyo3(signature = (frame, seeds, n_scalar=None, n_map=None, stratum=None, wr=false))]
pub(crate) fn select_srs_batch_rs(
    frame: Array1<'_, i64>,
    seeds: Vec<u64>,
    n_scalar: Option<usize>,
    n_map: Option<HashMap<i64, usize>>,
    stratum: Option<Array1<'_, i64>>,
    wr: bool,
) -> PyResult<Vec<SrsDraw>> {
    use rayon::prelude::*;
    let n = parse_n(n_scalar, n_map, SrsN::Scalar, SrsN::PerStratum)?;
    let frame = frame.values();
    let stratum = stratum.as_ref().map(Array1::values);
    seeds
        .par_iter()
        .map(|&seed| select_srs(&frame, n.clone(), stratum.as_deref(), wr, seed))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| to_py_err(&e))
}

/// `select_pps_rs` once per seed, in parallel; draw r is that call with `seeds[r]`.
#[pyfunction]
#[pyo3(signature = (
    frame,
    mos,
    seeds,
    n_scalar=None,
    n_map=None,
    stratum=None,
    method="sys",
    certainty_threshold=1.0
))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn select_pps_batch_rs(
    frame: Array1<'_, i64>,
    mos: Array1<'_, f64>,
    seeds: Vec<u64>,
    n_scalar: Option<usize>,
    n_map: Option<HashMap<i64, usize>>,
    stratum: Option<Array1<'_, i64>>,
    method: &str,
    certainty_threshold: f64,
) -> PyResult<Vec<PpsDraw>> {
    use rayon::prelude::*;
    let pps_method = PpsMethod::from_str(method)
        .ok_or_else(|| to_py_err(&SamplingError::UnknownMethod(method.to_string())))?;
    let n = parse_n(n_scalar, n_map, PpsN::Scalar, PpsN::PerStratum)?;
    let frame = frame.values();
    let mos = mos.values();
    let stratum = stratum.as_ref().map(Array1::values);
    seeds
        .par_iter()
        .map(|&seed| {
            select_pps(
                &frame,
                n.clone(),
                &mos,
                stratum.as_deref(),
                pps_method,
                certainty_threshold,
                seed,
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| to_py_err(&e))
}
