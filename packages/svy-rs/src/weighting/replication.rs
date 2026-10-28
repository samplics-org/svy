//! Replicate weights: BRR, jackknife (JKn, JK2), Rao-Wu and Poisson bootstrap, and SDR.

use super::hadamard_tables::get_hardcoded_hadamard;
use crate::errors::Coded;
use crate::rng::Rng;
use ndarray::{Array1, Array2, ArrayView1, ShapeBuilder};
use rayon::prelude::*;
use std::collections::HashMap;
use thiserror::Error;

/// Each stratum's index and PSUs, and each (stratum, PSU)'s position within its stratum.
type BrrStratumMap = (HashMap<i64, (i64, Vec<i64>)>, HashMap<(i64, i64), usize>);

pub(crate) type Result<T> = std::result::Result<T, ReplicationError>;

#[derive(Error, Debug, Clone)]
pub(crate) enum ReplicationError {
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("BRR requires 2 PSUs per stratum, stratum {stratum} has {count}")]
    BrrPsuCount { stratum: i64, count: usize },

    #[error("Paired JK requires ≥2 PSUs per stratum, stratum {stratum} has {count}")]
    PairedJkPsuCount { stratum: i64, count: usize },

    #[error("Insufficient PSUs: required {required}, got {got}")]
    InsufficientPsus { required: usize, got: usize },

    #[error("a seed is required: the replicates would otherwise repeat on every call")]
    MissingSeed,
}

impl Coded for ReplicationError {
    fn code(&self) -> &'static str {
        match self {
            Self::DimensionMismatch { .. } => "SHAPE_MISMATCH",
            _ => "INVALID_VALUE",
        }
    }

    fn invalid_input(&self) -> bool {
        true
    }
}

// ============================================================================
// Hadamard Matrix Generation
// ============================================================================

/// BRR Hadamard matrix and its order (the replicate count), which must exceed `n_strata`:
/// strata take rows 1.., since one on the all-ones row 0 would lose a PSU in every replicate.
pub(crate) fn get_hadamard_for_brr(n_strata: usize) -> Result<(Array2<f64>, usize)> {
    let need = n_strata + 1;
    if let Some(h) = try_get_hadamard(need) {
        return Ok((h, need));
    }
    for size in (need + 1)..=(need * 2) {
        if let Some(h) = try_get_hadamard(size) {
            return Ok((h, size));
        }
    }
    let pow2 = next_power_of_2(need);
    Ok((generate_hadamard_sylvester(pow2), pow2))
}

/// Hadamard matrix of the smallest available order >= `n_strata`, and that order.
pub(crate) fn get_hadamard_for_strata(n_strata: usize) -> Result<(Array2<f64>, usize)> {
    if let Some(h) = try_get_hadamard(n_strata) {
        return Ok((h, n_strata));
    }
    for size in (n_strata + 1)..=(n_strata * 2) {
        if let Some(h) = try_get_hadamard(size) {
            return Ok((h, size));
        }
    }
    let pow2 = next_power_of_2(n_strata);
    Ok((generate_hadamard_sylvester(pow2), pow2))
}

fn try_get_hadamard(n: usize) -> Option<Array2<f64>> {
    if n < 2 {
        return None;
    }
    if n == 2 {
        return Some(generate_hadamard_sylvester(2));
    }
    if n.is_power_of_two() {
        return Some(generate_hadamard_sylvester(n));
    }
    if !n.is_multiple_of(4) {
        return None;
    }
    if is_prime(n - 1) && (n - 1) % 4 == 3 {
        return Some(generate_hadamard_paley(n));
    }
    get_hardcoded_hadamard(n)
}

fn generate_hadamard_sylvester(n: usize) -> Array2<f64> {
    let mut h = Array2::from_elem((1, 1), 1.0);
    while h.nrows() < n {
        let size = h.nrows();
        let mut new_h = Array2::zeros((size * 2, size * 2));
        for i in 0..size {
            for j in 0..size {
                new_h[[i, j]] = h[[i, j]];
                new_h[[i, j + size]] = h[[i, j]];
                new_h[[i + size, j]] = h[[i, j]];
                new_h[[i + size, j + size]] = -h[[i, j]];
            }
        }
        h = new_h;
    }
    h
}

fn generate_hadamard_paley(n: usize) -> Array2<f64> {
    let p = n - 1;
    let mut is_residue = vec![false; p];
    for i in 1..p {
        is_residue[(i * i) % p] = true;
    }

    let mut h = Array2::zeros((n, n));
    for j in 0..n {
        h[[0, j]] = 1.0;
    }
    for i in 0..n {
        h[[i, 0]] = 1.0;
    }
    // Lower-right block: Q - I
    for i in 0..p {
        for j in 0..p {
            let q_val = if i == j {
                0.0
            } else {
                if is_residue[(i + p - j) % p] {
                    1.0
                } else {
                    -1.0
                }
            };
            h[[i + 1, j + 1]] = q_val - if i == j { 1.0 } else { 0.0 };
        }
    }
    h
}

fn is_prime(n: usize) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 {
        return true;
    }
    if n.is_multiple_of(2) {
        return false;
    }
    let sqrt_n = (n as f64).sqrt() as usize + 1;
    (3..=sqrt_n).step_by(2).all(|i| !n.is_multiple_of(i))
}

fn next_power_of_2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    if n.is_power_of_two() {
        n
    } else {
        1 << (64 - (n - 1).leading_zeros())
    }
}

// ============================================================================
// BRR (Balanced Repeated Replication)
// ============================================================================

/// BRR replicate weights with Fay's coefficient; `seed` shuffles which PSU of each pair is first.
pub(crate) fn create_brr_weights(
    wgt: ArrayView1<f64>,
    stratum: ArrayView1<i64>,
    psu: ArrayView1<i64>,
    n_reps: Option<usize>,
    fay_coef: f64,
    seed: Option<u64>,
) -> Result<(Array2<f64>, f64)> {
    let n_obs = wgt.len();
    if stratum.len() != n_obs || psu.len() != n_obs {
        return Err(ReplicationError::DimensionMismatch {
            expected: n_obs,
            got: stratum.len().min(psu.len()),
        });
    }
    if !(0.0..1.0).contains(&fay_coef) {
        return Err(ReplicationError::InvalidInput(
            "fay_coef must be in [0, 1)".into(),
        ));
    }

    let (stratum_map, psu_to_idx) = build_brr_stratum_map(&stratum, &psu, seed)?;
    let n_strata = stratum_map.len();
    let (hadamard, h_size) = get_hadamard_for_brr(n_strata)?;
    // Balance needs the full Hadamard set, so n_reps rounds up to it; more would duplicate columns.
    if let Some(r) = n_reps
        && r > h_size
    {
        return Err(ReplicationError::InvalidInput(format!(
            "n_reps={r} exceeds the Hadamard matrix order {h_size} for {n_strata} strata; \
                 BRR supports at most {h_size} distinct replicates here (n_reps <= {h_size}, \
                 or omit n_reps to use {h_size})"
        )));
    }
    let n_reps = n_reps.map(|r| r.max(h_size)).unwrap_or(h_size);

    let k_plus = 2.0 - fay_coef;
    let k_minus = fay_coef;

    let rep_weights: Vec<Array1<f64>> = (0..n_reps)
        .into_par_iter()
        .map(|r| {
            let mut rep_wgt = Array1::zeros(n_obs);
            for i in 0..n_obs {
                let s = stratum[i];
                let p = psu[i];
                let stratum_idx = stratum_map[&s].0 as usize;
                let psu_idx = psu_to_idx[&(s, p)];
                // Row 0 (all +1s) is never assigned; see get_hadamard_for_brr.
                let h_val = hadamard[[(stratum_idx + 1) % h_size, r % h_size]];
                let mult = if (h_val > 0.0 && psu_idx == 0) || (h_val < 0.0 && psu_idx == 1) {
                    k_plus
                } else {
                    k_minus
                };
                rep_wgt[i] = wgt[i] * mult;
            }
            rep_wgt
        })
        .collect();

    let mut result = Array2::zeros((n_obs, n_reps));
    for (r, col) in rep_weights.into_iter().enumerate() {
        result.column_mut(r).assign(&col);
    }
    Ok((result, n_strata as f64))
}

fn build_brr_stratum_map(
    stratum: &ArrayView1<i64>,
    psu: &ArrayView1<i64>,
    seed: Option<u64>,
) -> Result<BrrStratumMap> {
    use std::collections::HashSet as HSet;
    let mut stratum_set: HashMap<i64, HSet<i64>> = HashMap::new();
    for i in 0..stratum.len() {
        stratum_set.entry(stratum[i]).or_default().insert(psu[i]);
    }
    let mut stratum_psus: HashMap<i64, Vec<i64>> = stratum_set
        .into_iter()
        .map(|(s, set)| {
            let mut v: Vec<i64> = set.into_iter().collect();
            v.sort_unstable();
            (s, v)
        })
        .collect();

    for (&s, psus) in &stratum_psus {
        if psus.len() != 2 {
            return Err(ReplicationError::BrrPsuCount {
                stratum: s,
                count: psus.len(),
            });
        }
    }

    if let Some(seed) = seed {
        let mut rng = Rng::new(seed);
        // Sorted keys: HashMap order would make the seeded shuffle differ between runs.
        let mut keys: Vec<i64> = stratum_psus.keys().copied().collect();
        keys.sort_unstable();
        for k in &keys {
            if let Some(psus) = stratum_psus.get_mut(k) {
                rng.shuffle(psus);
            }
        }
    }

    let mut strata: Vec<i64> = stratum_psus.keys().copied().collect();
    strata.sort();

    let mut stratum_map = HashMap::new();
    let mut psu_to_idx = HashMap::new();
    for (idx, &s) in strata.iter().enumerate() {
        let psus = stratum_psus[&s].clone();
        for (pi, &p) in psus.iter().enumerate() {
            psu_to_idx.insert((s, p), pi);
        }
        stratum_map.insert(s, (idx as i64, psus));
    }
    Ok((stratum_map, psu_to_idx))
}

// ============================================================================
// Jackknife (JK1/JKn and JK2)
// ============================================================================

/// Jackknife weights: delete-one-PSU JKn, or JK2 when `paired`.
pub(crate) fn create_jk_weights(
    wgt: ArrayView1<f64>,
    stratum: Option<ArrayView1<i64>>,
    psu: ArrayView1<i64>,
    paired: bool,
    seed: Option<u64>,
) -> Result<(Array2<f64>, f64, Vec<f64>)> {
    if paired {
        create_jk2_weights(wgt, stratum, psu, seed)
    } else {
        create_jkn_weights(wgt, stratum, psu)
    }
}

/// Delete-one-PSU jackknife; returns (weights, df = #PSUs - #strata, rscales).
/// rscales[r] = (n_h - 1) / n_h of the stratum r deletes from, as R `svrepdesign(type = "JKn")`.
pub(crate) fn create_jkn_weights(
    wgt: ArrayView1<f64>,
    stratum: Option<ArrayView1<i64>>,
    psu: ArrayView1<i64>,
) -> Result<(Array2<f64>, f64, Vec<f64>)> {
    let n_obs = wgt.len();
    if psu.len() != n_obs {
        return Err(ReplicationError::DimensionMismatch {
            expected: n_obs,
            got: psu.len(),
        });
    }
    if let Some(s) = &stratum
        && s.len() != n_obs
    {
        return Err(ReplicationError::DimensionMismatch {
            expected: n_obs,
            got: s.len(),
        });
    }

    let stratum_vec = stratum
        .map(|s| s.to_owned())
        .unwrap_or_else(|| Array1::ones(n_obs));
    let stratum_psus = build_stratum_psu_list(&stratum_vec, &psu);
    let n_reps: usize = stratum_psus.values().map(|v| v.len()).sum();

    if n_reps < 2 {
        return Err(ReplicationError::InsufficientPsus {
            required: 2,
            got: n_reps,
        });
    }

    let mut psu_to_rep: HashMap<(i64, i64), usize> = HashMap::new();
    let mut rep_to_stratum: Vec<i64> = Vec::new();
    let mut strata: Vec<i64> = stratum_psus.keys().copied().collect();
    strata.sort();

    for &s in &strata {
        for &p in &stratum_psus[&s] {
            psu_to_rep.insert((s, p), rep_to_stratum.len());
            rep_to_stratum.push(s);
        }
    }

    let stratum_nh: HashMap<i64, f64> = stratum_psus
        .iter()
        .map(|(&s, psus)| (s, psus.len() as f64))
        .collect();

    let rep_weights: Vec<Array1<f64>> = (0..n_reps)
        .into_par_iter()
        .map(|r| {
            let mut rep_wgt = wgt.to_owned();
            let del_stratum = rep_to_stratum[r];
            for i in 0..n_obs {
                let s = stratum_vec[i];
                let p = psu[i];
                if psu_to_rep[&(s, p)] == r {
                    rep_wgt[i] = 0.0;
                } else if s == del_stratum {
                    let nh = stratum_nh[&s];
                    rep_wgt[i] *= nh / (nh - 1.0);
                }
            }
            rep_wgt
        })
        .collect();

    let mut result = Array2::zeros((n_obs, n_reps));
    for (r, col) in rep_weights.into_iter().enumerate() {
        result.column_mut(r).assign(&col);
    }

    let n_strata = stratum_psus.len();
    let df = (n_reps - n_strata) as f64;
    let rscales: Vec<f64> = rep_to_stratum
        .iter()
        .map(|s| {
            let nh = stratum_nh[s];
            (nh - 1.0) / nh
        })
        .collect();
    Ok((result, df, rscales))
}

/// JK2: per stratum, one replicate drops a seeded random PSU and scales the rest by n_h/(n_h - 1).
/// Each rscale is 1.0; a global (R-1)/R would understate the variance.
fn create_jk2_weights(
    wgt: ArrayView1<f64>,
    stratum: Option<ArrayView1<i64>>,
    psu: ArrayView1<i64>,
    seed: Option<u64>,
) -> Result<(Array2<f64>, f64, Vec<f64>)> {
    let n_obs = wgt.len();
    if psu.len() != n_obs {
        return Err(ReplicationError::DimensionMismatch {
            expected: n_obs,
            got: psu.len(),
        });
    }

    let stratum_vec =
        stratum.ok_or_else(|| ReplicationError::InvalidInput("JK2 requires stratum".into()))?;
    if stratum_vec.len() != n_obs {
        return Err(ReplicationError::DimensionMismatch {
            expected: n_obs,
            got: stratum_vec.len(),
        });
    }
    let stratum_vec = stratum_vec.to_owned();
    let stratum_psus = build_stratum_psu_list(&stratum_vec, &psu);

    for (&s, psus) in &stratum_psus {
        if psus.len() < 2 {
            return Err(ReplicationError::PairedJkPsuCount {
                stratum: s,
                count: psus.len(),
            });
        }
    }

    let mut strata: Vec<i64> = stratum_psus.keys().copied().collect();
    strata.sort();
    let n_reps = strata.len();

    let mut rng = Rng::new(seed.ok_or(ReplicationError::MissingSeed)?);
    let deleted_psu_idx: HashMap<i64, usize> = strata
        .iter()
        .map(|&s| (s, rng.next_index(stratum_psus[&s].len())))
        .collect();

    let mut psu_to_idx: HashMap<(i64, i64), usize> = HashMap::new();
    for (&s, psus) in &stratum_psus {
        for (idx, &p) in psus.iter().enumerate() {
            psu_to_idx.insert((s, p), idx);
        }
    }

    let stratum_nh: HashMap<i64, usize> = stratum_psus
        .iter()
        .map(|(&s, psus)| (s, psus.len()))
        .collect();

    let rep_weights: Vec<Array1<f64>> = (0..n_reps)
        .into_par_iter()
        .map(|r| {
            let mut rep_wgt = wgt.to_owned();
            let target = strata[r];
            let nh = stratum_nh[&target];
            let del_idx = deleted_psu_idx[&target];
            let adj = nh as f64 / (nh as f64 - 1.0);

            for i in 0..n_obs {
                let s = stratum_vec[i];
                if s == target {
                    let psu_idx = psu_to_idx[&(s, psu[i])];
                    if psu_idx == del_idx {
                        rep_wgt[i] = 0.0;
                    } else {
                        rep_wgt[i] *= adj;
                    }
                }
            }
            rep_wgt
        })
        .collect();

    let mut result = Array2::zeros((n_obs, n_reps));
    for (r, col) in rep_weights.into_iter().enumerate() {
        result.column_mut(r).assign(&col);
    }
    Ok((result, n_reps as f64, vec![1.0; n_reps]))
}

// ============================================================================
// Bootstrap
// ============================================================================

pub(crate) fn create_bootstrap_weights(
    wgt: ArrayView1<f64>,
    stratum: Option<ArrayView1<i64>>,
    psu: ArrayView1<i64>,
    n_reps: usize,
    seed: u64,
) -> Result<(Array2<f64>, f64)> {
    let n_obs = wgt.len();
    if psu.len() != n_obs {
        return Err(ReplicationError::DimensionMismatch {
            expected: n_obs,
            got: psu.len(),
        });
    }
    if let Some(s) = &stratum
        && s.len() != n_obs
    {
        return Err(ReplicationError::DimensionMismatch {
            expected: n_obs,
            got: s.len(),
        });
    }
    if n_reps < 1 {
        return Err(ReplicationError::InvalidInput(
            "Bootstrap requires n_reps >= 1".into(),
        ));
    }

    let stratum_vec = stratum
        .map(|s| s.to_owned())
        .unwrap_or_else(|| Array1::ones(n_obs));
    let stratum_psus = build_stratum_psu_list(&stratum_vec, &psu);

    // Sorted strata: HashMap order would consume each replicate's RNG differently per call.
    let mut strata_sorted: Vec<(i64, &Vec<i64>)> =
        stratum_psus.iter().map(|(&s, v)| (s, v)).collect();
    strata_sorted.sort_unstable_by_key(|(s, _)| *s);

    let mut psu_obs: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for i in 0..n_obs {
        psu_obs.entry((stratum_vec[i], psu[i])).or_default().push(i);
    }

    let rep_weights: Vec<Array1<f64>> = (0..n_reps)
        .into_par_iter()
        .map(|r| {
            let mut rng = Rng::new(seed.wrapping_add(r as u64));
            let mut rep_wgt = Array1::zeros(n_obs);

            for &(s, psus) in &strata_sorted {
                let n_psu = psus.len();
                // Rao-Wu: n_h - 1 draws scaled by n_h / (n_h - 1); n_h unscaled draws bias low.
                if n_psu == 1 {
                    // A single-PSU stratum has nothing to resample.
                    if let Some(obs) = psu_obs.get(&(s, psus[0])) {
                        for &i in obs {
                            rep_wgt[i] = wgt[i];
                        }
                    }
                    continue;
                }
                let m = n_psu - 1;
                let adj = n_psu as f64 / m as f64;
                let mut counts = vec![0usize; n_psu];
                for _ in 0..m {
                    counts[rng.next_index(n_psu)] += 1;
                }
                for (pi, &p) in psus.iter().enumerate() {
                    let c = counts[pi] as f64;
                    if c > 0.0
                        && let Some(obs) = psu_obs.get(&(s, p))
                    {
                        for &i in obs {
                            rep_wgt[i] = wgt[i] * c * adj;
                        }
                    }
                }
            }
            rep_wgt
        })
        .collect();

    let mut result = Array2::zeros((n_obs, n_reps));
    for (r, col) in rep_weights.into_iter().enumerate() {
        result.column_mut(r).assign(&col);
    }
    Ok((result, (n_reps - 1) as f64))
}

// ============================================================================
// Poisson bootstrap (Beaumont & Patak 2012)
// ============================================================================

/// bsw = w ± w·sqrt((w - 1) / w), signs w.p. 1/2 per unit: no stratum or PSU structure to disclose.
/// The optional calibration of these replicates is left to the general weighting adjustments.
pub(crate) fn create_poisson_bootstrap_weights(
    wgt: ArrayView1<f64>,
    n_reps: usize,
    seed: u64,
) -> Result<(Array2<f64>, f64)> {
    let n_obs = wgt.len();
    if n_reps < 1 {
        return Err(ReplicationError::InvalidInput(
            "Poisson bootstrap requires n_reps >= 1".into(),
        ));
    }
    if n_obs == 0 {
        return Err(ReplicationError::InvalidInput(
            "Poisson bootstrap requires at least one observation".into(),
        ));
    }

    // sqrt((w - 1) / w) is not real below 1; `!(w >= 1.0)` also rejects NaN.
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // NaN takes this branch on purpose
    if let Some(bad) = wgt.iter().position(|&w| !(w >= 1.0)) {
        return Err(ReplicationError::InvalidInput(format!(
            "Poisson bootstrap requires weights >= 1 (the adjustment factor is \
             sqrt((w - 1) / w)); observation {} has weight {}",
            bad, wgt[bad]
        )));
    }

    // bsw = w ± adj stays positive: sqrt((w - 1) / w) < 1 for finite w >= 1.
    let adj: Vec<f64> = wgt.iter().map(|&w| w * ((w - 1.0) / w).sqrt()).collect();

    // One column-major buffer: `Vec<Array1>` then a copy would hold two full matrices at once.
    let mut buf = vec![0.0f64; n_obs * n_reps];
    buf.par_chunks_mut(n_obs).enumerate().for_each(|(r, col)| {
        let mut rng = Rng::new(seed.wrapping_add(r as u64));

        // One u64 supplies 64 sign bits, not one RNG call per unit.
        let mut bits = 0u64;
        let mut n_bits = 0u32;
        for i in 0..n_obs {
            if n_bits == 0 {
                bits = rng.next_u64();
                n_bits = 64;
            }
            let sign = if bits & 1 == 1 { 1.0 } else { -1.0 };
            bits >>= 1;
            n_bits -= 1;
            col[i] = wgt[i] + sign * adj[i];
        }
    });

    let result = Array2::from_shape_vec((n_obs, n_reps).f(), buf)
        .map_err(|e| ReplicationError::InvalidInput(e.to_string()))?;
    Ok((result, (n_reps - 1) as f64))
}

// ============================================================================
// SDR
// ============================================================================

pub(crate) fn create_sdr_weights(
    wgt: ArrayView1<f64>,
    stratum: Option<ArrayView1<i64>>,
    order: Option<ArrayView1<i64>>,
    n_reps: usize,
) -> Result<(Array2<f64>, f64)> {
    let n_obs = wgt.len();
    if n_reps < 2 {
        return Err(ReplicationError::InvalidInput(
            "SDR requires ≥2 replicates".into(),
        ));
    }
    if let Some(s) = &stratum
        && s.len() != n_obs
    {
        return Err(ReplicationError::DimensionMismatch {
            expected: n_obs,
            got: s.len(),
        });
    }
    if let Some(o) = &order
        && o.len() != n_obs
    {
        return Err(ReplicationError::DimensionMismatch {
            expected: n_obs,
            got: o.len(),
        });
    }

    let stratum_vec = stratum
        .map(|s| s.to_owned())
        .unwrap_or_else(|| Array1::ones(n_obs));
    let order_vec = order
        .map(|o| o.to_owned())
        .unwrap_or_else(|| Array1::from_iter(0..n_obs as i64));

    let sorted_indices = build_stratum_sorted_indices(&stratum_vec, &order_vec);
    let (hadamard, h_size) = get_hadamard_for_strata(n_reps)?;

    // Fay-Train: unit k (stratum order) gets f = 1 + 2^{-3/2} (h[row k, r] - h[row k+1, r]), rows
    // cycling over 1..h_size-1 (all-ones row 0 adds no variance); with 4/R this is the SDR estimator.
    let n_rows = h_size - 1;
    let sqrt2_over_4 = std::f64::consts::SQRT_2 / 4.0; // 2^{-3/2}

    let rep_weights: Vec<Array1<f64>> = (0..n_reps)
        .into_par_iter()
        .map(|r| {
            let mut rep_wgt = wgt.to_owned();
            let col = r % h_size;
            for indices in sorted_indices.values() {
                let n_h = indices.len();
                if n_h < 2 {
                    continue;
                }
                for (k, &i) in indices.iter().enumerate() {
                    let row_a = 1 + (k % n_rows);
                    let row_b = 1 + ((k + 1) % n_rows);
                    let diff = hadamard[[row_a, col]] - hadamard[[row_b, col]];
                    rep_wgt[i] *= 1.0 + sqrt2_over_4 * diff;
                }
            }
            rep_wgt
        })
        .collect();

    let mut result = Array2::zeros((n_obs, n_reps));
    for (r, col) in rep_weights.into_iter().enumerate() {
        result.column_mut(r).assign(&col);
    }
    Ok((result, n_reps as f64))
}

// ============================================================================
// Helpers
// ============================================================================

fn build_stratum_psu_list(stratum: &Array1<i64>, psu: &ArrayView1<i64>) -> HashMap<i64, Vec<i64>> {
    use std::collections::HashSet;
    let mut set_map: HashMap<i64, HashSet<i64>> = HashMap::new();
    for i in 0..stratum.len() {
        set_map.entry(stratum[i]).or_default().insert(psu[i]);
    }
    set_map
        .into_iter()
        .map(|(s, set)| {
            let mut v: Vec<i64> = set.into_iter().collect();
            v.sort_unstable(); // deterministic ordering
            (s, v)
        })
        .collect()
}

fn build_stratum_sorted_indices(
    stratum: &Array1<i64>,
    order: &Array1<i64>,
) -> HashMap<i64, Vec<usize>> {
    let mut groups: HashMap<i64, Vec<(i64, usize)>> = HashMap::new();
    for i in 0..stratum.len() {
        groups.entry(stratum[i]).or_default().push((order[i], i));
    }
    groups
        .into_iter()
        .map(|(s, mut v)| {
            v.sort_by_key(|(o, _)| *o);
            (s, v.into_iter().map(|(_, i)| i).collect())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_brr_basic() {
        // 2 strata -> Sylvester H4; strata take rows 1 = (+-+-) and 2 = (++--).
        // Fay 0.5: the PSU picked by the row's sign gets 1.5w, the other 0.5w.
        let wgt = array![1.0, 2.0, 3.0, 4.0];
        let stratum = array![1, 1, 2, 2];
        let psu = array![1, 2, 1, 2];
        let (rep_wgt, df) =
            create_brr_weights(wgt.view(), stratum.view(), psu.view(), Some(4), 0.5, None).unwrap();
        let expected = array![
            [1.5, 0.5, 1.5, 0.5],
            [1.0, 3.0, 1.0, 3.0],
            [4.5, 4.5, 1.5, 1.5],
            [2.0, 2.0, 6.0, 6.0],
        ];
        assert_eq!(rep_wgt, expected);
        assert_eq!(df, 2.0);
    }

    #[test]
    fn test_brr_n_reps_beyond_hadamard_is_error() {
        let wgt = array![1.0, 1.0, 1.0, 1.0];
        let stratum = array![1, 1, 2, 2];
        let psu = array![1, 2, 1, 2];
        let res = create_brr_weights(wgt.view(), stratum.view(), psu.view(), Some(8), 0.0, None);
        assert!(
            matches!(&res, Err(ReplicationError::InvalidInput(m)) if m.contains("exceeds the Hadamard matrix order 4")),
            "got {res:?}"
        );
    }

    #[test]
    fn test_brr_is_balanced_every_psu_participates() {
        for n_strata in 2..=6usize {
            let n_obs = n_strata * 2;
            let wgt = Array1::from_elem(n_obs, 1.0);
            let stratum = Array1::from_iter((0..n_obs).map(|i| (i / 2) as i64 + 1));
            let psu = Array1::from_iter((0..n_obs).map(|i| (i % 2) as i64 + 1));
            let (rep_wgt, _df) =
                create_brr_weights(wgt.view(), stratum.view(), psu.view(), None, 0.0, None)
                    .unwrap();
            let n_reps = rep_wgt.ncols();
            for i in 0..n_obs {
                let row_sum: f64 = rep_wgt.row(i).sum();
                let mean = row_sum / n_reps as f64;
                assert!(
                    (mean - 1.0).abs() < 1e-12,
                    "n_strata={n_strata}, obs {i}: mean replicate weight {mean} != base 1.0"
                );
                let n_nonzero = rep_wgt.row(i).iter().filter(|&&w| w > 0.0).count();
                assert_eq!(
                    n_nonzero,
                    n_reps / 2,
                    "n_strata={n_strata}, obs {i}: PSU active in {n_nonzero}/{n_reps} replicates"
                );
            }
        }
    }

    #[test]
    fn test_jkn_basic() {
        // Same matrix and rscales as R as.svrepdesign(type = "JKn").
        let wgt = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let stratum = array![1, 1, 1, 2, 2, 2];
        let psu = array![1, 2, 3, 1, 2, 3];
        let (rep_wgt, df, rscales) =
            create_jkn_weights(wgt.view(), Some(stratum.view()), psu.view()).unwrap();
        let expected = array![
            [0.0, 1.5, 1.5, 1.0, 1.0, 1.0],
            [3.0, 0.0, 3.0, 2.0, 2.0, 2.0],
            [4.5, 4.5, 0.0, 3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0, 0.0, 6.0, 6.0],
            [5.0, 5.0, 5.0, 7.5, 0.0, 7.5],
            [6.0, 6.0, 6.0, 9.0, 9.0, 0.0],
        ];
        assert_eq!(rep_wgt, expected);
        assert_eq!(df, 4.0);
        assert_eq!(rscales, vec![2.0 / 3.0; 6]);
    }

    #[test]
    fn test_jk2_paired() {
        // Replicate r touches stratum r only: one seeded PSU drops to 0, the other doubles.
        let wgt = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let stratum = array![1, 1, 1, 2, 2, 2];
        let psu = array![1, 1, 2, 3, 4, 4];
        let (rep_wgt, df, rscales) =
            create_jk_weights(wgt.view(), Some(stratum.view()), psu.view(), true, Some(42))
                .unwrap();
        assert_eq!(rep_wgt.ncols(), 2);
        for (r, rows) in [[0, 1, 2], [3, 4, 5]].iter().enumerate() {
            let other: Vec<usize> = (0..6).filter(|i| !rows.contains(i)).collect();
            for &i in &other {
                assert_eq!(rep_wgt[[i, r]], wgt[i]);
            }
            let dropped_first = rep_wgt[[rows[0], r]] == 0.0;
            for &i in rows {
                let same_psu = psu[i] == psu[rows[0]];
                let expected = if same_psu == dropped_first {
                    0.0
                } else {
                    2.0 * wgt[i]
                };
                assert_eq!(rep_wgt[[i, r]], expected, "obs {i}, replicate {r}");
            }
        }
        assert_eq!(df, 2.0);
        assert_eq!(rscales, vec![1.0; 2]);
    }

    #[test]
    fn test_bootstrap_basic() {
        // A PSU's rows get w * c * n_h / (n_h - 1), with integer draw counts c summing to n_h - 1.
        let wgt = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let stratum = array![1, 1, 1, 2, 2, 2, 2];
        let psu = array![1, 1, 2, 3, 4, 5, 5];
        let strata: [(f64, &[&[usize]]); 2] =
            [(2.0, &[&[0, 1], &[2]]), (3.0, &[&[3], &[4], &[5, 6]])];
        let (rep_wgt, df) =
            create_bootstrap_weights(wgt.view(), Some(stratum.view()), psu.view(), 10, 42).unwrap();
        assert_eq!(rep_wgt.ncols(), 10);
        for r in 0..10 {
            for (n_h, psus) in strata {
                let adj = n_h / (n_h - 1.0);
                let mut total_draws = 0.0;
                for rows in psus {
                    let c = rep_wgt[[rows[0], r]] / (wgt[rows[0]] * adj);
                    assert_eq!(c, c.round(), "replicate {r}: non-integer count {c}");
                    for &i in *rows {
                        assert_eq!(rep_wgt[[i, r]], wgt[i] * c * adj);
                    }
                    total_draws += c;
                }
                assert_eq!(total_draws, n_h - 1.0, "replicate {r}");
            }
        }
        assert_eq!(df, 9.0);
    }

    #[test]
    fn test_sdr_basic() {
        // H4 rows pair cyclically as (1,2), (2,3), (3,1), ... with f = 1 + 2^{-3/2} (h_a - h_b).
        let wgt = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let order = array![0, 1, 2, 3, 4, 5];
        let (rep_wgt, df) = create_sdr_weights(wgt.view(), None, Some(order.view()), 4).unwrap();
        let hi = 1.0 + std::f64::consts::FRAC_1_SQRT_2;
        let lo = 1.0 - std::f64::consts::FRAC_1_SQRT_2;
        let factors = [[1.0, lo, hi, 1.0], [1.0, hi, 1.0, lo], [1.0, 1.0, lo, hi]];
        for i in 0..6 {
            for r in 0..4 {
                // a few ulps from computing 2^{-3/2} * 2 instead of 2^{-1/2}
                assert_abs_diff_eq!(rep_wgt[[i, r]], wgt[i] * factors[i % 3][r], epsilon = 1e-14);
            }
        }
        assert_eq!(df, 4.0);
    }

    #[test]
    fn test_bootstrap_zero_reps_is_error() {
        let wgt = array![1.0, 1.0];
        let psu = array![1, 2];
        let res = create_bootstrap_weights(wgt.view(), None, psu.view(), 0, 42);
        assert!(
            matches!(&res, Err(ReplicationError::InvalidInput(m)) if m.contains("n_reps >= 1")),
            "got {res:?}"
        );
    }

    #[test]
    fn test_bootstrap_stratum_length_mismatch_is_error() {
        let wgt = array![1.0, 1.0];
        let psu = array![1, 2];
        let stratum = array![1, 1, 2];
        let res = create_bootstrap_weights(wgt.view(), Some(stratum.view()), psu.view(), 5, 42);
        assert!(
            matches!(
                &res,
                Err(ReplicationError::DimensionMismatch {
                    expected: 2,
                    got: 3
                })
            ),
            "got {res:?}"
        );
    }

    #[test]
    fn test_jkn_stratum_length_mismatch_is_error() {
        let wgt = array![1.0, 1.0];
        let psu = array![1, 2];
        let stratum = array![1, 1, 2];
        let res = create_jkn_weights(wgt.view(), Some(stratum.view()), psu.view());
        assert!(
            matches!(
                &res,
                Err(ReplicationError::DimensionMismatch {
                    expected: 2,
                    got: 3
                })
            ),
            "got {res:?}"
        );
    }

    #[test]
    fn test_sdr_factors_are_fay_train() {
        // At n_reps = Hadamard order the factors average to 1: every non-first row sums to 0.
        let wgt = array![2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
        let order = array![0, 1, 2, 3, 4, 5];
        let (rep, _) = create_sdr_weights(wgt.view(), None, Some(order.view()), 8).unwrap();
        let f_hi = 1.0 + std::f64::consts::FRAC_1_SQRT_2;
        let f_lo = 1.0 - std::f64::consts::FRAC_1_SQRT_2;
        for i in 0..6 {
            let mut mean = 0.0;
            for r in 0..8 {
                let f = rep[[i, r]] / 2.0;
                assert!(
                    (f - 1.0).abs() < 1e-12 || (f - f_hi).abs() < 1e-12 || (f - f_lo).abs() < 1e-12,
                    "factor {f} not a Fay-Train factor"
                );
                mean += f;
            }
            assert!((mean / 8.0 - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_sdr_length_mismatch_is_error() {
        let wgt = array![1.0, 1.0, 1.0, 1.0];
        let stratum_long = array![0, 0, 0, 0, 0, 0];
        let res = create_sdr_weights(wgt.view(), Some(stratum_long.view()), None, 4);
        assert!(
            matches!(
                &res,
                Err(ReplicationError::DimensionMismatch {
                    expected: 4,
                    got: 6
                })
            ),
            "got {res:?}"
        );
        let order_short = array![0, 1];
        let res = create_sdr_weights(wgt.view(), None, Some(order_short.view()), 4);
        assert!(
            matches!(
                &res,
                Err(ReplicationError::DimensionMismatch {
                    expected: 4,
                    got: 2
                })
            ),
            "got {res:?}"
        );
    }

    // PSU totals (2, 3) and (2, 5): the paired estimator sum_h (t_h1 - t_h2)^2 gives 1 + 9 = 10.
    #[test]
    fn test_bootstrap_variance_matches_paired_estimator() {
        let wgt = array![1.0, 1.0, 3.0, 2.0, 5.0];
        let stratum = array![1, 1, 1, 2, 2];
        let psu = array![1, 1, 2, 1, 2];
        let n_reps = 2000usize;
        let (rep_wgt, _df) =
            create_bootstrap_weights(wgt.view(), Some(stratum.view()), psu.view(), n_reps, 42)
                .unwrap();

        let t_full: f64 = wgt.sum(); // 12
        let mut v = 0.0f64;
        for r in 0..n_reps {
            let t_r: f64 = rep_wgt.column(r).sum();
            v += (t_r - t_full).powi(2);
        }
        v /= n_reps as f64; // bootstrap coefficient 1/R with full-sample centering

        assert!(
            (v - 10.0).abs() < 1.0,
            "bootstrap variance {v} should be near 10 (paired estimator)"
        );
    }
}
