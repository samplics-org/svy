use polars::prelude::PolarsError;
use thiserror::Error;

use crate::errors::{Coded, KERNEL_INTERNAL};
use crate::estimation::error::EstimationError;

#[derive(Error, Debug)]
pub enum RegressionError {
    #[error("Unsupported family: {0}")]
    UnsupportedFamily(String),

    #[error("Unsupported link: {0}")]
    UnsupportedLink(String),

    #[error("negative binomial needs its dispersion: pass theta")]
    ThetaMissing,

    #[error("negative binomial theta must be finite and positive, got {0}")]
    ThetaInvalid(f64),

    #[error(
        "negative binomial: the Pearson statistic is zero, so theta has no \
         moment estimate — the response does not vary around the fitted mean"
    )]
    PearsonZero,

    #[error(
        "negative binomial: the profile likelihood in theta reached {0}. \
         The counts are probably not overdispersed — fit family='poisson', \
         or pass theta to fit at a known value."
    )]
    ThetaDiverged(f64),

    #[error(
        "GLM design matrix is rank deficient: {0} is collinear with the columns \
         before it. Drop it, or one of the columns it duplicates."
    )]
    RankDeficient(String),

    #[error("GLM information matrix could not be inverted (degenerate or non-finite fit)")]
    InfoNotInvertible,

    #[error("GLM did not produce finite coefficients (degenerate or non-convergent fit)")]
    NonFiniteCoefficients,

    #[error("GLM {what} column '{col}' contains null values")]
    NullValues { what: &'static str, col: String },

    #[error("GLM {what} contains a non-finite value at row {row}")]
    NonFinite { what: String, row: usize },

    #[error("{what} has length {got}, expected {expected}")]
    DimensionMismatch {
        what: &'static str,
        expected: usize,
        got: usize,
    },

    #[error("GLM needs more observations than parameters: n_obs={n_obs}, k={k}")]
    TooFewObservations { n_obs: usize, k: usize },

    #[error("GLM failed on {n_failed} of {total} levels of '{by}' ({levels}): {source}")]
    LevelsFailed {
        n_failed: usize,
        total: usize,
        by: String,
        levels: String,
        source: Box<RegressionError>,
    },

    #[error("Empty data or no predictors")]
    WolsEmpty,

    #[error("Singular X'WX matrix")]
    WolsSingular,

    #[error("Cannot invert X'WX")]
    WolsNotInvertible,

    #[error("`data` is required")]
    MissingData,

    // A column the wrapper could not read or cast: the caller's input.
    #[error("{0}")]
    InputColumn(PolarsError),

    #[error(transparent)]
    Estimation(#[from] EstimationError),

    #[error(transparent)]
    Polars(#[from] PolarsError),
}

pub type RResult<T> = Result<T, RegressionError>;

impl Coded for RegressionError {
    fn code(&self) -> &'static str {
        match self {
            Self::UnsupportedFamily(_) | Self::UnsupportedLink(_) => "INVALID_CHOICE",
            Self::ThetaMissing | Self::WolsEmpty | Self::MissingData | Self::InputColumn(_) => {
                "INVALID_VALUE"
            }
            Self::ThetaInvalid(_) => "NB_THETA_INVALID",
            Self::PearsonZero => "NB_THETA_UNDEFINED",
            Self::ThetaDiverged(_) => "NB_THETA_DIVERGED",
            Self::RankDeficient(_) | Self::WolsSingular | Self::WolsNotInvertible => {
                "RANK_DEFICIENT"
            }
            Self::InfoNotInvertible | Self::NonFiniteCoefficients => "GLM_DEGENERATE",
            Self::NullValues { .. } => "MISSING_VALUES",
            Self::NonFinite { .. } => "NON_FINITE_INPUT",
            Self::TooFewObservations { .. } => "INSUFFICIENT_OBSERVATIONS",
            Self::DimensionMismatch { .. } => "SHAPE_MISMATCH",
            Self::LevelsFailed { source, .. } => source.code(),
            Self::Estimation(e) => e.code(),
            Self::Polars(_) => KERNEL_INTERNAL,
        }
    }

    fn invalid_input(&self) -> bool {
        match self {
            Self::UnsupportedFamily(_)
            | Self::UnsupportedLink(_)
            | Self::ThetaMissing
            | Self::ThetaInvalid(_)
            | Self::RankDeficient(_)
            | Self::NullValues { .. }
            | Self::NonFinite { .. }
            | Self::TooFewObservations { .. }
            | Self::DimensionMismatch { .. }
            | Self::WolsEmpty
            | Self::MissingData
            | Self::InputColumn(_) => true,
            Self::PearsonZero
            | Self::ThetaDiverged(_)
            | Self::InfoNotInvertible
            | Self::NonFiniteCoefficients
            | Self::WolsSingular
            | Self::WolsNotInvertible
            | Self::Polars(_) => false,
            Self::LevelsFailed { source, .. } => source.invalid_input(),
            Self::Estimation(e) => e.invalid_input(),
        }
    }
}
