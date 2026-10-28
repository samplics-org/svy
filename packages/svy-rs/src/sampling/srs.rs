// src/sampling/srs.rs

use crate::errors::Coded;
use crate::rng::Rng;
use std::collections::HashMap;
use thiserror::Error;

/// Selected rows, their hits and inclusion probabilities.
pub type SrsDraw = (Vec<i64>, Vec<i64>, Vec<f64>);

#[derive(Error, Debug, Clone)]
pub enum SamplingError {
    #[error("Cannot sample from an empty frame")]
    EmptyFrame,

    #[error("Cannot draw n={n} without replacement from population of {pop}")]
    OversamplingWor { n: usize, pop: usize },

    #[error("a seed is required: the draw would otherwise repeat on every call")]
    MissingSeed,

    #[error("n_map has no entry for stratum {0}")]
    NMapMissingStratum(i64),

    #[error("n_map names stratum {0}, which is not in the frame")]
    NMapUnknownStratum(i64),

    #[error("Unknown PPS method {0:?}. Use: sys, wr, brewer, murphy, rs")]
    UnknownMethod(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

impl Coded for SamplingError {
    fn code(&self) -> &'static str {
        match self {
            Self::EmptyFrame
            | Self::MissingSeed
            | Self::NMapMissingStratum(_)
            | Self::InvalidInput(_) => "INVALID_VALUE",
            Self::OversamplingWor { .. } => "SELECTION_N_EXCEEDS_GROUP",
            Self::NMapUnknownStratum(_) | Self::UnknownMethod(_) => "INVALID_CHOICE",
        }
    }

    fn invalid_input(&self) -> bool {
        true
    }
}

pub type Result<T> = std::result::Result<T, SamplingError>;

/// Every stratum needs its n, and every n its stratum.
pub(crate) fn check_n_map<V>(map: &HashMap<i64, usize>, strata: &HashMap<i64, V>) -> Result<()> {
    if let Some(&s) = map.keys().filter(|s| !strata.contains_key(s)).min() {
        return Err(SamplingError::NMapUnknownStratum(s));
    }
    if let Some(&s) = strata.keys().filter(|s| !map.contains_key(s)).min() {
        return Err(SamplingError::NMapMissingStratum(s));
    }
    Ok(())
}

#[derive(Clone)]
pub enum SrsN {
    Scalar(usize),
    PerStratum(HashMap<i64, usize>),
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub fn select_srs(
    frame: &[i64],
    n: SrsN,
    stratum: Option<&[i64]>,
    wr: bool,
    seed: u64,
) -> Result<SrsDraw> {
    match stratum {
        None => {
            let n_scalar = match n {
                SrsN::Scalar(v) => v,
                SrsN::PerStratum(_) => {
                    return Err(SamplingError::InvalidInput(
                        "per-stratum n requires stratum array".into(),
                    ));
                }
            };
            srs_unstratified_indexed(
                frame,
                &(0..frame.len()).collect::<Vec<_>>(),
                n_scalar,
                wr,
                seed,
            )
        }
        Some(strat) => {
            if strat.len() != frame.len() {
                return Err(SamplingError::InvalidInput(format!(
                    "stratum length ({}) must match frame length ({})",
                    strat.len(),
                    frame.len()
                )));
            }
            select_srs_stratified(frame, n, strat, wr, seed)
        }
    }
}

// ---------------------------------------------------------------------------
// Core: operates on positions into the original frame (no copy)
// ---------------------------------------------------------------------------

fn srs_unstratified_indexed(
    frame: &[i64],
    positions: &[usize],
    n: usize,
    wr: bool,
    seed: u64,
) -> Result<SrsDraw> {
    let cap = positions.len();

    if cap == 0 {
        return Err(SamplingError::EmptyFrame);
    }
    if !wr && n > cap {
        return Err(SamplingError::OversamplingWor { n, pop: cap });
    }
    if n == 0 {
        return Ok((vec![], vec![], vec![]));
    }

    let mut rng = Rng::new(seed);
    // hits[i] counts how many times positions[i] was selected
    let mut hits = vec![0i64; cap];

    if wr {
        for _ in 0..n {
            hits[rng.next_index(cap)] += 1;
        }
    } else {
        // Fisher-Yates partial shuffle over local indices [0..cap)
        // We never copy frame data — just shuffle an index permutation
        let mut perm: Vec<usize> = (0..cap).collect();
        for i in 0..n {
            let j = i + rng.next_index(cap - i);
            perm.swap(i, j);
        }
        for &i in &perm[..n] {
            hits[i] = 1;
        }
    }

    let pi = if wr {
        let n_f = n as f64;
        let cap_f = cap as f64;
        1.0 - (1.0 - 1.0 / cap_f).powf(n_f)
    } else {
        n as f64 / cap as f64
    };

    let mut selected = Vec::with_capacity(n);
    let mut out_hits = Vec::with_capacity(n);
    let mut out_probs = Vec::with_capacity(n);

    for (local_i, &h) in hits.iter().enumerate() {
        if h > 0 {
            selected.push(frame[positions[local_i]]);
            out_hits.push(h);
            out_probs.push(pi);
        }
    }

    Ok((selected, out_hits, out_probs))
}

// ---------------------------------------------------------------------------
// Stratified SRS — strata drawn in parallel via rayon
// ---------------------------------------------------------------------------

fn select_srs_stratified(
    frame: &[i64],
    n: SrsN,
    stratum: &[i64],
    wr: bool,
    seed: u64,
) -> Result<SrsDraw> {
    // Group positions by stratum — one pass, no per-stratum alloc of frame values
    let mut strat_map: HashMap<i64, Vec<usize>> = HashMap::new();
    for (pos, &s) in stratum.iter().enumerate() {
        strat_map.entry(s).or_default().push(pos);
    }

    let mut strata: Vec<i64> = strat_map.keys().copied().collect();
    strata.sort_unstable();

    if let SrsN::PerStratum(map) = &n {
        check_n_map(map, &strat_map)?;
    }

    // Build (stratum_label, positions, n_s, child_seed) tuples
    let tasks: Vec<(i64, &Vec<usize>, usize, u64)> = strata
        .iter()
        .enumerate()
        .filter_map(|(strat_idx, &s)| {
            let positions = &strat_map[&s];
            let n_s = match &n {
                SrsN::Scalar(v) => *v,
                SrsN::PerStratum(map) => map[&s],
            };
            if n_s == 0 {
                return None;
            }
            let child_seed = seed.wrapping_add((strat_idx as u64).wrapping_mul(0x9e3779b97f4a7c15));
            Some((s, positions, n_s, child_seed))
        })
        .collect();

    // Parallel execution when there are enough strata to justify the overhead
    use rayon::prelude::*;
    let results: Vec<Result<SrsDraw>> = if tasks.len() >= 4 {
        tasks
            .par_iter()
            .map(|(_, positions, n_s, child_seed)| {
                srs_unstratified_indexed(frame, positions, *n_s, wr, *child_seed)
            })
            .collect()
    } else {
        tasks
            .iter()
            .map(|(_, positions, n_s, child_seed)| {
                srs_unstratified_indexed(frame, positions, *n_s, wr, *child_seed)
            })
            .collect()
    };

    // Merge — pre-size output from n totals
    let total_n: usize = tasks.iter().map(|(_, _, n_s, _)| n_s).sum();
    let mut selected = Vec::with_capacity(total_n);
    let mut out_hits = Vec::with_capacity(total_n);
    let mut out_probs = Vec::with_capacity(total_n);

    for res in results {
        let (sel, hits, probs) = res?;
        selected.extend_from_slice(&sel);
        out_hits.extend_from_slice(&hits);
        out_probs.extend_from_slice(&probs);
    }

    Ok((selected, out_hits, out_probs))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Regression: stratum length mismatch used to panic (longer) or silently
    // truncate the frame (shorter); both must now be clean errors.
    #[test]
    fn test_stratum_length_mismatch_rejected() {
        let frame = vec![1, 2];
        let stratum_long = vec![0, 0, 0];
        assert!(select_srs(&frame, SrsN::Scalar(1), Some(&stratum_long), false, 0).is_err());

        let frame = vec![1, 2, 3];
        let stratum_short = vec![0, 0];
        assert!(select_srs(&frame, SrsN::Scalar(1), Some(&stratum_short), false, 0).is_err());
    }
}
