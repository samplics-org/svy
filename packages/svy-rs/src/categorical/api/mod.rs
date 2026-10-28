//! PyO3 wrappers for the t-tests, rank tests and tabulation: frame I/O, domains and by-groups only.

use crate::categorical::error::{CResult, CategoricalError};
use crate::errors::to_py_err;
use std::collections::HashMap;

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;

use crate::estimation::calib_sweep::{CalibSweep, make_calib};
use crate::estimation::into_contiguous;
use crate::estimation::taylor::active_count;

use crate::categorical::ranktest::{
    RankScoreMethod, ranktest_k_sample, ranktest_k_sample_rep, ranktest_two_sample,
    ranktest_two_sample_rep,
};
use crate::categorical::replicate::{RepSpec, replicate_cells};
use crate::categorical::tabulation::{
    cell_se, count_strata_psus, domain_levels, estimate_proportions, estimate_totals, rao_scott,
};
use crate::categorical::ttest::{
    ttest_one_sample, ttest_one_sample_domain, ttest_one_sample_rep, ttest_two_sample,
    ttest_two_sample_rep,
};

mod common;
mod ranktest;
mod tabulate;
#[cfg(test)]
mod tests;
mod ttest;

pub(crate) use common::*;
pub(crate) use ranktest::*;
pub(crate) use tabulate::*;
pub(crate) use ttest::*;
