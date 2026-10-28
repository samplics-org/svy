use polars::prelude::PolarsError;
use thiserror::Error;

use crate::errors::{Coded, KERNEL_INTERNAL};

#[derive(Error, Debug)]
pub enum EstimationError {
    #[error("unknown {what} '{got}'; expected {expected}")]
    InvalidChoice {
        what: &'static str,
        got: String,
        expected: &'static str,
    },

    #[error("pair columns must be equal length, got {y} and {x}")]
    PairLengths { y: usize, x: usize },

    #[error("column length {got} does not match weights length {expected}")]
    ColumnLength { got: usize, expected: usize },

    #[error("rep_coefs has {got} entries but there are {expected} replicate weight columns")]
    RepCoefsLength { got: usize, expected: usize },

    #[error("values_sorted has {values} entries but cdf has {cdf}")]
    CdfLength { values: usize, cdf: usize },

    #[error("Unknown center: {0}. Use 'rep_mean' or 'estimate'")]
    UnknownCenter(String),

    #[error("{0}")]
    Empty(&'static str),

    #[error("{0}")]
    UnsupportedDtype(String),

    #[error("{col} column must be String or integer, got {dtype}")]
    DesignDtype { col: &'static str, dtype: String },

    #[error("Sum of weights is zero{}", if *.in_domain { " in domain" } else { "" })]
    ZeroWeight { in_domain: bool },

    #[error("Weighted sum of denominator (x) is zero{}", if *.in_domain { " in domain" } else { "" })]
    ZeroDenominator { in_domain: bool },

    #[error(transparent)]
    Polars(#[from] PolarsError),
}

pub(crate) type EResult<T> = Result<T, EstimationError>;

impl Coded for EstimationError {
    fn code(&self) -> &'static str {
        match self {
            Self::InvalidChoice { .. } | Self::UnknownCenter(_) => "INVALID_CHOICE",
            Self::PairLengths { .. } | Self::ColumnLength { .. } | Self::CdfLength { .. } => {
                "SHAPE_MISMATCH"
            }
            Self::RepCoefsLength { .. } => "REP_COEFS_LENGTH_MISMATCH",
            Self::Empty(_) => "INVALID_VALUE",
            Self::DesignDtype { .. } | Self::UnsupportedDtype(_) => "INVALID_TYPE",
            Self::ZeroWeight { .. } => "ZERO_WEIGHT",
            Self::ZeroDenominator { .. } => "ZERO_DENOMINATOR",
            Self::Polars(_) => KERNEL_INTERNAL,
        }
    }

    fn invalid_input(&self) -> bool {
        matches!(
            self,
            Self::InvalidChoice { .. }
                | Self::PairLengths { .. }
                | Self::ColumnLength { .. }
                | Self::RepCoefsLength { .. }
                | Self::CdfLength { .. }
                | Self::UnknownCenter(_)
                | Self::Empty(_)
                | Self::DesignDtype { .. }
                | Self::UnsupportedDtype(_)
        )
    }
}
