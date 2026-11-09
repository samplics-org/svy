//! PyO3 wrappers for Taylor linearisation; the frame-level work lives in `svy_core::estimation::taylor_frame`.

use crate::errors::to_py_err;
use crate::estimation::into_contiguous;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use svy_core::estimation::association::AssocKind;
use svy_core::estimation::calib_sweep::make_calib;
use svy_core::estimation::error::EstimationError;
use svy_core::estimation::taylor::{SvyQuantileMethod, weighted_quantile};
use svy_core::estimation::taylor_frame::*;

mod assoc;
mod mean;
mod prop;
mod quantile;
mod ratio;
mod total;

pub(crate) use assoc::*;
pub(crate) use mean::*;
pub(crate) use prop::*;
pub(crate) use quantile::*;
pub(crate) use ratio::*;
pub(crate) use total::*;
