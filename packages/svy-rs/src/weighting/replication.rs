// src/weighting/replication.rs
//
// Replicate weight creation for variance estimation methods:
// - BRR (Balanced Repeated Replication)
// - Jackknife (JK1/JKn and JK2 paired)
// - Bootstrap (stratified)
// - SDR (Successive Difference Replication)

use super::hadamard_tables::get_hardcoded_hadamard;
use crate::rng::Rng;
use ndarray::{Array1, Array2, ArrayView1};
use rayon::prelude::*;
use std::collections::HashMap;

/// Result type for replication functions
pub type Result<T> = std::result::Result<T, ReplicationError>;

/// Errors that can occur during replicate weight creation
#[derive(Debug, Clone)]
pub enum ReplicationError {
    DimensionMismatch { expected: usize, got: usize },
    InvalidInput(String),
    BrrPsuCount { stratum: i64, count: usize },
    PairedJkPsuCount { stratum: i64, count: usize },
    InsufficientPsus { required: usize, got: usize },
}

impl std::fmt::Display for ReplicationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got)
            }
            Self::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            Self::BrrPsuCount { stratum, count } => write!(
                f,
                "BRR requires 2 PSUs per stratum, stratum {} has {}",
                stratum, count
            ),
            Self::PairedJkPsuCount { stratum, count } => write!(
                f,
                "Paired JK requires ≥2 PSUs per stratum, stratum {} has {}",
                stratum, count
            ),
            Self::InsufficientPsus { required, got } => {
                write!(f, "Insufficient PSUs: required {}, got {}", required, got)
            }
        }
    }
}

impl std::error::Error for ReplicationError {}

// ============================================================================
// Hadamard Matrix Generation
// ============================================================================

/// Get a Hadamard matrix for BRR with `n_strata` strata. Returns
/// (matrix, order); the order is the number of replicates.
///
/// The order must EXCEED n_strata because strata are assigned to rows
/// 1..order-1, skipping row 0: in a normalized Hadamard matrix row 0 is
/// all +1s, and a stratum assigned to it would keep the same PSU doubled
/// in every replicate — its partner PSU would carry zero weight in all
/// replicates and silently vanish from variance estimation (the previous
/// behavior for the stratum mapped to row 0).
pub fn get_hadamard_for_brr(n_strata: usize) -> Result<(Array2<f64>, usize)> {
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

/// Get Hadamard matrix for n strata. Returns (matrix, actual_size).
/// Used by SDR, which assigns rows 1..size-1 cyclically (row 0 of a
/// normalized Hadamard matrix is all ones and is never assigned).
pub fn get_hadamard_for_strata(n_strata: usize) -> Result<(Array2<f64>, usize)> {
    // Try exact size
    if let Some(h) = try_get_hadamard(n_strata) {
        return Ok((h, n_strata));
    }
    // Try nearby sizes
    for size in (n_strata + 1)..=(n_strata * 2) {
        if let Some(h) = try_get_hadamard(size) {
            return Ok((h, size));
        }
    }
    // Fall back to next power of 2
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
    if n % 4 != 0 {
        return None;
    }
    // Try Paley Type I (n-1 prime, ≡ 3 mod 4)
    if is_prime(n - 1) && (n - 1) % 4 == 3 {
        return Some(generate_hadamard_paley(n));
    }
    // Try hardcoded
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
    // Compute quadratic residues
    let mut is_residue = vec![false; p];
    for i in 1..p {
        is_residue[(i * i) % p] = true;
    }

    // Build Paley matrix
    let mut h = Array2::zeros((n, n));
    // First row and column: all 1s
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
    if n % 2 == 0 {
        return false;
    }
    let sqrt_n = (n as f64).sqrt() as usize + 1;
    (3..=sqrt_n).step_by(2).all(|i| n % i != 0)
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

/// Create BRR replicate weights with optional random seed for PSU ordering.
pub fn create_brr_weights(
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
    // n_reps below the Hadamard order is rounded UP to it (balance requires
    // a full Hadamard set). Requesting MORE would silently duplicate columns
    // via `r % h_size` — duplicated replicates add no information while
    // looking like more — so that is an error.
    if let Some(r) = n_reps {
        if r > h_size {
            return Err(ReplicationError::InvalidInput(format!(
                "n_reps={r} exceeds the Hadamard matrix order {h_size} for {n_strata} strata; \
                 BRR supports at most {h_size} distinct replicates here (n_reps <= {h_size}, \
                 or omit n_reps to use {h_size})"
            )));
        }
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
                // Rows 1..h_size: row 0 (all +1s) is never assigned — see
                // get_hadamard_for_brr.
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
) -> Result<(HashMap<i64, (i64, Vec<i64>)>, HashMap<(i64, i64), usize>)> {
    // Use HashSet for O(1) PSU deduplication, then convert to sorted Vec
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
        // Sort keys first so RNG is consumed in a deterministic order
        // across runs (HashMap iteration order is otherwise nondeterministic).
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

/// Create Jackknife weights. paired=false for JK1/JKn, paired=true for JK2.
pub fn create_jk_weights(
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

/// JKn: delete-one-PSU-at-a-time.
///
/// Returns (replicate weights, design df, per-replicate rscales).
/// df is the stratified-jackknife convention #PSUs - #strata (n - 1 for
/// unstratified JK1); the previous total-PSU df was anti-conservative.
/// rscales[r] = (n_h - 1)/n_h for the stratum whose PSU replicate r
/// deletes — R's svrepdesign(type="JKn") convention. The estimation layer
/// previously applied a global (R-1)/R to every replicate, which is only
/// correct for unstratified JK1.
pub fn create_jkn_weights(
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
    if let Some(s) = &stratum {
        if s.len() != n_obs {
            return Err(ReplicationError::DimensionMismatch {
                expected: n_obs,
                got: s.len(),
            });
        }
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

/// JK2: paired jackknife (one replicate per stratum)
/// Returns (replicate weights, design df, per-replicate rscales).
/// One random delete-one replicate per stratum with the surviving PSUs
/// reweighted by n_h/(n_h-1): the single-replicate-per-stratum jackknife,
/// whose variance coefficient is 1.0 per replicate (a global (R-1)/R
/// would understate the variance by that factor).
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

    let mut rng = Rng::new(seed.unwrap_or(0));
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

pub fn create_bootstrap_weights(
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
    if let Some(s) = &stratum {
        if s.len() != n_obs {
            return Err(ReplicationError::DimensionMismatch {
                expected: n_obs,
                got: s.len(),
            });
        }
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

    // Materialize a sorted (stratum_key, &psus) list so the parallel loop
    // iterates strata in a deterministic order across runs.  HashMap
    // iteration order is otherwise nondeterministic, which would cause the
    // per-replicate RNG to be consumed in a different order on each call
    // and produce different replicate weights despite using the same seed.
    let mut strata_sorted: Vec<(i64, &Vec<i64>)> =
        stratum_psus.iter().map(|(&s, v)| (s, v)).collect();
    strata_sorted.sort_unstable_by_key(|(s, _)| *s);

    // Pre-build PSU → observation indices map once, shared across replicates.
    // Avoids O(N) full-scan per stratum per replicate.
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
                // Rao-Wu-Yue rescaling bootstrap: draw m_h = n_h - 1 PSUs
                // with replacement and rescale by n_h / (n_h - 1).  The naive
                // n_h-draw scheme without rescaling understates variance by a
                // factor of (n_h - 1) / n_h per stratum (50% for 2-PSU strata).
                if n_psu == 1 {
                    // Single-PSU stratum: no resampling variability; keep the
                    // original weights.
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
                // Apply counts via pre-built index — no O(N) scan
                for (pi, &p) in psus.iter().enumerate() {
                    let c = counts[pi] as f64;
                    if c > 0.0 {
                        if let Some(obs) = psu_obs.get(&(s, p)) {
                            for &i in obs {
                                rep_wgt[i] = wgt[i] * c * adj;
                            }
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
// SDR
// ============================================================================

pub fn create_sdr_weights(
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
    if let Some(s) = &stratum {
        if s.len() != n_obs {
            return Err(ReplicationError::DimensionMismatch {
                expected: n_obs,
                got: s.len(),
            });
        }
    }
    if let Some(o) = &order {
        if o.len() != n_obs {
            return Err(ReplicationError::DimensionMismatch {
                expected: n_obs,
                got: o.len(),
            });
        }
    }

    let stratum_vec = stratum
        .map(|s| s.to_owned())
        .unwrap_or_else(|| Array1::ones(n_obs));
    let order_vec = order
        .map(|o| o.to_owned())
        .unwrap_or_else(|| Array1::from_iter(0..n_obs as i64));

    let sorted_indices = build_stratum_sorted_indices(&stratum_vec, &order_vec);
    let (hadamard, h_size) = get_hadamard_for_strata(n_reps)?;

    // Fay–Train successive difference replication: unit k (in systematic
    // order within its stratum) is assigned the cyclic Hadamard-row pair
    // (row(k), row(k+1)) over rows 1..h_size-1 — row 0 of a normalized
    // Hadamard matrix is all ones and would give a constant adjustment
    // with zero between-replicate variance. Each unit gets ONE additive
    // factor per replicate:
    //     f = 1 + 2^{-3/2} (h[row(k), r] - h[row(k+1), r])
    // so f is in {1, 1 +- 2^{-1/2}}. Successive units share a row, which
    // is what makes the resulting variance (with the 4/R coefficient) the
    // successive-difference estimator. The previous construction instead
    // composed pair adjustments multiplicatively (introducing a spurious
    // +-1/2 cross-term on interior units) and used rows from 0.
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
    // Two-pass: first collect unique PSUs per stratum using HashSet (O(1) lookup),
    // then convert to sorted Vec for deterministic replicate assignment.
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
    use ndarray::array;

    #[test]
    fn test_brr_basic() {
        let wgt = array![1.0, 1.0, 1.0, 1.0];
        let stratum = array![1, 1, 2, 2];
        let psu = array![1, 2, 1, 2];
        // 2 strata -> Hadamard order 4 (rows 1..3 usable; row 0 skipped).
        let (rep_wgt, df) =
            create_brr_weights(wgt.view(), stratum.view(), psu.view(), Some(4), 0.0, None).unwrap();
        assert_eq!(rep_wgt.dim(), (4, 4));
        assert_eq!(df, 2.0);
    }

    #[test]
    fn test_brr_n_reps_beyond_hadamard_is_error() {
        // Requesting more replicates than the Hadamard order used to silently
        // duplicate columns (r % h_size); it is now an error.
        let wgt = array![1.0, 1.0, 1.0, 1.0];
        let stratum = array![1, 1, 2, 2];
        let psu = array![1, 2, 1, 2];
        let err = create_brr_weights(wgt.view(), stratum.view(), psu.view(), Some(8), 0.0, None)
            .unwrap_err();
        assert!(err.to_string().contains("Hadamard"), "got: {err}");
    }

    #[test]
    fn test_brr_is_balanced_every_psu_participates() {
        // Balance: every PSU must carry weight in exactly half the
        // replicates, so the mean replicate weight equals the base weight.
        // Previously the stratum assigned to the all-ones Hadamard row had
        // one PSU zeroed in EVERY replicate.
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
        let wgt = array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let stratum = array![1, 1, 1, 2, 2, 2];
        let psu = array![1, 2, 3, 1, 2, 3];
        let (rep_wgt, df, rscales) =
            create_jkn_weights(wgt.view(), Some(stratum.view()), psu.view()).unwrap();
        assert_eq!(rep_wgt.dim(), (6, 6));
        // Stratified jackknife df = #PSUs - #strata (was: total PSUs).
        assert_eq!(df, 4.0);
        // R's JKn rscales: (n_h - 1)/n_h per replicate; 3 PSUs per stratum.
        assert_eq!(rscales, vec![2.0 / 3.0; 6]);
    }

    #[test]
    fn test_jk2_paired() {
        let wgt = array![1.0, 1.0, 1.0, 1.0];
        let stratum = array![1, 1, 2, 2];
        let psu = array![1, 2, 1, 2];
        let (rep_wgt, df, rscales) =
            create_jk_weights(wgt.view(), Some(stratum.view()), psu.view(), true, Some(42))
                .unwrap();
        assert_eq!(rep_wgt.dim(), (4, 2)); // 2 strata = 2 replicates
        assert_eq!(df, 2.0);
        assert_eq!(rscales, vec![1.0; 2]);
    }

    #[test]
    fn test_bootstrap_basic() {
        let wgt = array![1.0, 1.0, 1.0, 1.0];
        let stratum = array![1, 1, 2, 2];
        let psu = array![1, 2, 1, 2];
        let (rep_wgt, df) =
            create_bootstrap_weights(wgt.view(), Some(stratum.view()), psu.view(), 10, 42).unwrap();
        assert_eq!(rep_wgt.dim(), (4, 10));
        assert_eq!(df, 9.0);
    }

    #[test]
    fn test_sdr_basic() {
        let wgt = array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let order = array![0, 1, 2, 3, 4, 5];
        let (rep_wgt, df) = create_sdr_weights(wgt.view(), None, Some(order.view()), 4).unwrap();
        assert_eq!(rep_wgt.dim(), (6, 4));
        assert_eq!(df, 4.0);
    }

    // Regression: mismatched stratum/order lengths used to panic with an
    // index out of bounds; n_reps=0 underflowed the df computation.
    #[test]
    fn test_bootstrap_zero_reps_is_error() {
        let wgt = array![1.0, 1.0];
        let psu = array![1, 2];
        assert!(create_bootstrap_weights(wgt.view(), None, psu.view(), 0, 42).is_err());
    }

    #[test]
    fn test_bootstrap_stratum_length_mismatch_is_error() {
        let wgt = array![1.0, 1.0];
        let psu = array![1, 2];
        let stratum = array![1, 1, 2];
        assert!(
            create_bootstrap_weights(wgt.view(), Some(stratum.view()), psu.view(), 5, 42).is_err()
        );
    }

    #[test]
    fn test_jkn_stratum_length_mismatch_is_error() {
        let wgt = array![1.0, 1.0];
        let psu = array![1, 2];
        let stratum = array![1, 1, 2];
        assert!(create_jkn_weights(wgt.view(), Some(stratum.view()), psu.view()).is_err());
    }

    #[test]
    fn test_sdr_factors_are_fay_train() {
        // Every replicate factor must be one of {1, 1 +- 2^{-1/2}}, and with
        // n_reps equal to the Hadamard order the mean replicate weight over
        // all replicates equals the full-sample weight exactly (each
        // non-first Hadamard row sums to zero).
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
                    (f - 1.0).abs() < 1e-12
                        || (f - f_hi).abs() < 1e-12
                        || (f - f_lo).abs() < 1e-12,
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
        assert!(create_sdr_weights(wgt.view(), Some(stratum_long.view()), None, 4).is_err());
        let order_short = array![0, 1];
        assert!(create_sdr_weights(wgt.view(), None, Some(order_short.view()), 4).is_err());
    }

    // Rao-Wu rescaled bootstrap: for a stratified design with 2 PSUs per
    // stratum, the bootstrap variance of the estimated total must match the
    // standard paired estimator sum_h (t_h1 - t_h2)^2.  Here the PSU totals
    // are (2, 3) and (2, 5), so the target variance is 1 + 9 = 10.  The old
    // unrescaled scheme converged to 5 (biased low by (n_h-1)/n_h = 1/2).
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
