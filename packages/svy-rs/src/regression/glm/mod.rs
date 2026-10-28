//! Survey GLM by IRLS with a bread-meat-bread sandwich, matching R `svyglm` numerically.
//!
//! - Bread from the converged Fisher information; meat from PSU score totals centred within stratum, scaled m/(m-1).
//! - Out-of-domain rows contribute 0, but strata and PSUs are the full design's, as R's subset().

#![allow(non_snake_case)]

use crate::regression::error::{RResult, RegressionError};
use faer::prelude::Solve;
use faer::{Mat, MatRef, Side};

use polars::prelude::*;

use crate::categorical::ranktest::probit;
use crate::estimation::calib_sweep::CalibSweep;
use crate::estimation::taylor::{
    SingletonMethod, SingletonRule, design_col_codes, design_pair_codes,
};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

mod config;
mod helpers;
mod irls;
mod numerics;
mod result;
#[cfg(test)]
mod tests;

pub(crate) use config::*;
pub(crate) use helpers::*;
pub(crate) use irls::*;
use numerics::*;
pub(crate) use result::*;
