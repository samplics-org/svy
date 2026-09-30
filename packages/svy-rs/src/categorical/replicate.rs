// src/categorical/replicate.rs
//
// Replication variance for tabulate, ttest and ranktest: R's svyrep.design
// methods. Cell proportions, totals and means are re-estimated with every
// replicate weight (svymean/svytotal/svrepglm); rank tests re-total the
// full-sample influence values instead (svyranktest calls svytotal(infn)).
//
// The stratum, PSU, FPC, singleton and calibration inputs of the Taylor path
// play no part here: the replicate weights carry the design.

use polars::prelude::*;
use rayon::prelude::*;

use crate::estimation::replication::{VarianceCenter, covariance_from_replicates};

pub struct RepSpec {
    /// One weight vector per replicate.
    pub weights: Vec<Vec<f64>>,
    pub coefs: Vec<f64>,
    pub center: VarianceCenter,
    /// Design df: the recorded one, or n_reps - 1. It does not shrink on a
    /// domain.
    pub df: f64,
}

impl RepSpec {
    pub fn from_frame(
        df: &DataFrame,
        cols: &[String],
        coefs: Vec<f64>,
        center: &str,
        rep_df: Option<f64>,
    ) -> PolarsResult<Self> {
        if coefs.len() != cols.len() {
            return Err(PolarsError::ComputeError(
                format!(
                    "rep_coefs has {} entries but there are {} replicate weight columns",
                    coefs.len(),
                    cols.len()
                )
                .into(),
            ));
        }
        let center = VarianceCenter::from_str(center).ok_or_else(|| {
            PolarsError::ComputeError(
                format!("Unknown center: {center}. Use 'rep_mean' or 'estimate'").into(),
            )
        })?;
        let weights = cols
            .iter()
            .map(|c| {
                let s = df
                    .column(c)?
                    .as_materialized_series()
                    .cast(&DataType::Float64)?;
                Ok(s.f64()?.iter().map(|v| v.unwrap_or(0.0)).collect())
            })
            .collect::<PolarsResult<Vec<Vec<f64>>>>()?;
        let df_val = rep_df.unwrap_or(cols.len().saturating_sub(1) as f64);
        Ok(Self {
            weights,
            coefs,
            center,
            df: df_val,
        })
    }

    /// Covariance of k estimates from their replicate values (k × n_reps).
    pub fn cov(&self, full: &[f64], reps: &[Vec<f64>]) -> Vec<Vec<f64>> {
        covariance_from_replicates(full, reps, &self.coefs, self.center)
    }

    /// Weighted mean of `y` over `mask`, with the full-sample weights and with
    /// each replicate's.
    pub fn means(&self, y: &[f64], w: &[f64], mask: &[bool]) -> (f64, Vec<f64>) {
        let mean = |wt: &[f64]| {
            let (mut num, mut den) = (0.0, 0.0);
            for i in 0..y.len() {
                if mask[i] {
                    num += wt[i] * y[i];
                    den += wt[i];
                }
            }
            if den != 0.0 { num / den } else { f64::NAN }
        };
        (
            mean(w),
            self.weights.par_iter().map(|wr| mean(wr)).collect(),
        )
    }

    /// Replicate totals of the columns of an n × k row-major matrix over
    /// `mask`, as k × n_reps.
    pub fn totals(&self, x: &[f64], k: usize, mask: &[bool]) -> Vec<Vec<f64>> {
        let per_rep: Vec<Vec<f64>> = self
            .weights
            .par_iter()
            .map(|wr| {
                let mut t = vec![0.0; k];
                for (i, &wi) in wr.iter().enumerate() {
                    if mask[i] && wi != 0.0 {
                        for j in 0..k {
                            t[j] += wi * x[i * k + j];
                        }
                    }
                }
                t
            })
            .collect();
        (0..k)
            .map(|j| per_rep.iter().map(|t| t[j]).collect())
            .collect()
    }
}

/// Cells of a tabulation with a replication covariance.
pub struct RepCells {
    pub proportions: Vec<f64>,
    pub prop_cov: Vec<Vec<f64>>,
    pub totals: Vec<f64>,
    pub total_cov: Vec<Vec<f64>>,
}

/// Cell proportions (svymean of the cell indicators) and totals (svytotal),
/// re-estimated with every replicate. A row counts when it is in the domain
/// and its key is one of `levels`.
pub fn replicate_cells(
    y: &StringChunked,
    weights: &Float64Chunked,
    domain: Option<&BooleanChunked>,
    levels: &[String],
    spec: &RepSpec,
) -> RepCells {
    let k = levels.len();
    let level_map: std::collections::HashMap<&str, usize> = levels
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    let cell: Vec<Option<usize>> = y
        .iter()
        .enumerate()
        .map(|(i, v)| {
            if !crate::categorical::tabulation::in_domain(domain, i) {
                return None;
            }
            v.and_then(|s| level_map.get(s).copied())
        })
        .collect();

    let tally = |wt: &dyn Fn(usize) -> f64| {
        let mut t = vec![0.0; k];
        for (i, c) in cell.iter().enumerate() {
            if let Some(j) = c {
                t[*j] += wt(i);
            }
        }
        t
    };
    let share = |t: &[f64]| {
        let s: f64 = t.iter().sum();
        t.iter()
            .map(|v| if s != 0.0 { v / s } else { f64::NAN })
            .collect::<Vec<f64>>()
    };

    let totals = tally(&|i| weights.get(i).unwrap_or(0.0));
    let proportions = share(&totals);
    let rep_totals: Vec<Vec<f64>> = spec
        .weights
        .par_iter()
        .map(|wr| tally(&|i| wr[i]))
        .collect();
    let rep_props: Vec<Vec<f64>> = rep_totals.iter().map(|t| share(t)).collect();

    let by_cell = |m: &[Vec<f64>]| -> Vec<Vec<f64>> {
        (0..k).map(|j| m.iter().map(|r| r[j]).collect()).collect()
    };
    RepCells {
        prop_cov: spec.cov(&proportions, &by_cell(&rep_props)),
        total_cov: spec.cov(&totals, &by_cell(&rep_totals)),
        proportions,
        totals,
    }
}
