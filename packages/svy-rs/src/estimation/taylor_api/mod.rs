//! PyO3 wrappers for Taylor linearisation; the heavy math lives in `taylor`.

use crate::errors::to_py_err;
use crate::estimation::error::{EResult, EstimationError};
use numpy::PyReadonlyArray1;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use rayon::prelude::*;

use crate::estimation::association::{
    AssocKind, point_estimate_assoc, scores_assoc, srs_variance_assoc_of,
};
use crate::estimation::calib_sweep::{CalibSweep, make_calib};
use crate::estimation::into_contiguous;
use crate::estimation::taylor::{
    SrsRef, SvyQuantileMethod, TaylorDesign, active_count, build_taylor_design, degrees_of_freedom,
    degrees_of_freedom_from_design, point_estimate_mean, point_estimate_mean_domain,
    point_estimate_ratio, point_estimate_ratio_domain, point_estimate_total,
    point_estimate_total_domain, quantiles_woodruff, rows_column, scores_mean, scores_mean_arr,
    scores_mean_domain, scores_ratio, scores_ratio_domain, scores_total, scores_total_arr,
    scores_total_domain, srs_variance_mean, srs_variance_mean_domain, srs_variance_ratio,
    srs_variance_ratio_domain, srs_variance_total, srs_variance_total_domain,
    taylor_covariance_apply, taylor_covariance_apply_in, taylor_variance_apply_in,
    weighted_quantile,
};

mod assoc;
mod common;
mod mean;
mod prop;
mod quantile;
mod ratio;
mod total;

pub(crate) use assoc::*;
use common::*;
pub(crate) use mean::*;
pub(crate) use prop::*;
pub(crate) use quantile::*;
pub(crate) use ratio::*;
pub(crate) use total::*;
