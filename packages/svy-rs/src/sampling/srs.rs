// src/sampling/srs.rs

use std::collections::HashMap;
use crate::rng::Rng;

#[derive(Debug, Clone)]
pub enum SamplingError {
    EmptyFrame,
    OversamplingWor { n: usize, pop: usize },
    InvalidInput(String),
}

impl std::fmt::Display for SamplingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyFrame => write!(f, "Cannot sample from an empty frame"),
            Self::OversamplingWor { n, pop } => write!(
                f, "Cannot draw n={n} without replacement from population of {pop}"
            ),
            Self::InvalidInput(msg) => write!(f, "Invalid input: {msg}"),
        }
    }
}

impl std::error::Error for SamplingError {}

pub type Result<T> = std::result::Result<T, SamplingError>;

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
    seed: Option<u64>,
) -> Result<(Vec<i64>, Vec<i64>, Vec<f64>)> {
    match stratum {
        None => {
            let n_scalar = match n {
                SrsN::Scalar(v) => v,
                SrsN::PerStratum(_) => return Err(SamplingError::InvalidInput(
                    "per-stratum n requires stratum array".into(),
                )),
            };
            srs_unstratified_indexed(frame, &(0..frame.len()).collect::<Vec<_>>(), n_scalar, wr, seed)
        }
        Some(strat) => select_srs_stratified(frame, n, strat, wr, seed),
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
    seed: Option<u64>,
) -> Result<(Vec<i64>, Vec<i64>, Vec<f64>)> {
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

    let mut rng = Rng::new(seed.unwrap_or(0));
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
    seed: Option<u64>,
) -> Result<(Vec<i64>, Vec<i64>, Vec<f64>)> {
    // Group positions by stratum — one pass, no per-stratum alloc of frame values
    let mut strat_map: HashMap<i64, Vec<usize>> = HashMap::new();
    for (pos, &s) in stratum.iter().enumerate() {
        strat_map.entry(s).or_default().push(pos);
    }

    let mut strata: Vec<i64> = strat_map.keys().copied().collect();
    strata.sort_unstable();

    let base_seed = seed.unwrap_or(0);

    // Build (stratum_label, positions, n_s, child_seed) tuples
    let tasks: Vec<(i64, &Vec<usize>, usize, u64)> = strata
        .iter()
        .enumerate()
        .filter_map(|(strat_idx, &s)| {
            let positions = &strat_map[&s];
            let n_s = match &n {
                SrsN::Scalar(v) => *v,
                SrsN::PerStratum(map) => *map.get(&s).unwrap_or(&0),
            };
            if n_s == 0 {
                return None;
            }
            let child_seed = base_seed
                .wrapping_add((strat_idx as u64).wrapping_mul(0x9e3779b97f4a7c15));
            Some((s, positions, n_s, child_seed))
        })
        .collect();

    // Parallel execution when there are enough strata to justify the overhead
    use rayon::prelude::*;
    let results: Vec<Result<(Vec<i64>, Vec<i64>, Vec<f64>)>> = if tasks.len() >= 4 {
        tasks
            .par_iter()
            .map(|(_, positions, n_s, child_seed)| {
                srs_unstratified_indexed(frame, positions, *n_s, wr, Some(*child_seed))
            })
            .collect()
    } else {
        tasks
            .iter()
            .map(|(_, positions, n_s, child_seed)| {
                srs_unstratified_indexed(frame, positions, *n_s, wr, Some(*child_seed))
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
