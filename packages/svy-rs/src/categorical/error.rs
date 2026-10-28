use polars::prelude::PolarsError;
use thiserror::Error;

use crate::errors::{Coded, KERNEL_INTERNAL};
use crate::estimation::error::EstimationError;

#[derive(Error, Debug)]
pub enum CategoricalError {
    #[error("Two-sample t-test requires exactly 2 groups, found {0}")]
    TwoGroups(usize),

    #[error("Rank test requires at least 2 groups with positive weight, found {0}")]
    TooFewGroups(usize),

    #[error("{0}")]
    MissingInput(&'static str),

    #[error("{0}")]
    Wols(String),

    #[error(transparent)]
    Estimation(#[from] EstimationError),

    #[error(transparent)]
    Polars(#[from] PolarsError),
}

pub type CResult<T> = Result<T, CategoricalError>;

impl Coded for CategoricalError {
    fn code(&self) -> &'static str {
        match self {
            Self::TwoGroups(_) | Self::TooFewGroups(_) => "TEST_GROUP_COUNT",
            Self::MissingInput(_) => "INVALID_VALUE",
            Self::Wols(_) => "RANK_DEFICIENT",
            Self::Estimation(e) => e.code(),
            Self::Polars(_) => KERNEL_INTERNAL,
        }
    }

    fn invalid_input(&self) -> bool {
        match self {
            Self::TwoGroups(_) | Self::TooFewGroups(_) | Self::MissingInput(_) => true,
            Self::Wols(_) | Self::Polars(_) => false,
            Self::Estimation(e) => e.invalid_input(),
        }
    }
}
