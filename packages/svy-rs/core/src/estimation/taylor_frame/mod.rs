//! Taylor linearisation over a frame: column lookup, by-groups and result frames over `taylor`.

use crate::estimation::error::{EResult, EstimationError};
use polars::prelude::*;
use rayon::prelude::*;

use crate::estimation::association::{
    AssocKind, point_estimate_assoc, scores_assoc, srs_variance_assoc_of,
};
use crate::estimation::calib_sweep::CalibSweep;
use crate::estimation::taylor::{
    SrsRef, SvyQuantileMethod, TaylorDesign, active_count, build_taylor_design, degrees_of_freedom,
    degrees_of_freedom_from_design, point_estimate_mean, point_estimate_mean_domain,
    point_estimate_ratio, point_estimate_ratio_domain, point_estimate_total,
    point_estimate_total_domain, quantiles_woodruff, rows_column, scores_mean, scores_mean_arr,
    scores_mean_domain, scores_ratio, scores_ratio_domain, scores_total, scores_total_arr,
    scores_total_domain, srs_variance_mean, srs_variance_mean_domain, srs_variance_ratio,
    srs_variance_ratio_domain, srs_variance_total, srs_variance_total_domain,
    taylor_covariance_apply, taylor_covariance_apply_in, taylor_variance_apply_in,
};

mod assoc;
mod common;
mod mean;
mod prop;
mod quantile;
mod ratio;
mod total;

pub use assoc::*;
pub use common::parse_srs_ref;
use common::*;
pub use mean::*;
pub use prop::*;
pub use quantile::*;
pub use ratio::*;
pub use total::*;
