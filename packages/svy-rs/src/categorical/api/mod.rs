//! PyO3 wrappers for the t-tests, rank tests and tabulation; the frame-level work lives in `svy_core::categorical::frame`.

use crate::errors::to_py_err;
use crate::estimation::into_contiguous;
use polars::prelude::DataFrame;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use svy_core::categorical::error::CategoricalError;
use svy_core::categorical::frame::*;
use svy_core::categorical::replicate::RepSpec;
use svy_core::estimation::calib_sweep::make_calib;

mod common;
use common::*;
mod ranktest;
mod tabulate;
mod ttest;

pub(crate) use ranktest::*;
pub(crate) use tabulate::*;
pub(crate) use ttest::*;
