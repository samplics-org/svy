// src/weighting/replication.rs
//
// Replicate weight creation for variance estimation methods:
// - BRR (Balanced Repeated Replication)
// - Jackknife (JK1/JKn and JK2 paired)
// - Bootstrap (stratified)
// - SDR (Successive Difference Replication)

use ndarray::{Array1, Array2, ArrayView1};
use rayon::prelude::*;
use std::collections::HashMap;

use super::hadamard_tables::get_hardcoded_hadamard;

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
    NoHadamardMatrix { size: usize },
}

impl std::fmt::Display for ReplicationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } =>
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got),
            Self::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            Self::BrrPsuCount { stratum, count } =>
                write!(f, "BRR requires 2 PSUs per stratum, stratum {} has {}", stratum, count),
            Self::PairedJkPsuCount { stratum, count } =>
                write!(f, "Paired JK requires ≥2 PSUs per stratum, stratum {} has {}", stratum, count),
            Self::InsufficientPsus { required, got } =>
                write!(f, "Insufficient PSUs: required {}, got {}", required, got),
            Self::NoHadamardMatrix { size } =>
                write!(f, "No Hadamard matrix available for size {}", size),
        }
    }
}

impl std::error::Error for ReplicationError {}

// ============================================================================
// Random Number Generator (xoshiro256**)
// ============================================================================

#[derive(Clone)]
struct Rng {
    state: [u64; 4],
}

impl Rng {
    fn new(seed: u64) -> Self {
        let mut sm_state = seed;
        let mut state = [0u64; 4];
        for s in &mut state {
            sm_state = sm_state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = sm_state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *s = z ^ (z >> 31);
        }
        Rng { state }
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    fn next_index(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.next_index(i + 1);
            slice.swap(i, j);
        }
    }
}

// ============================================================================
// Hadamard Matrix Generation
// ============================================================================

/// Get Hadamard matrix for n strata. Returns (matrix, actual_size).
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
    if n < 2 { return None; }
    if n == 2 { return Some(generate_hadamard_sylvester(2)); }
    if n.is_power_of_two() { return Some(generate_hadamard_sylvester(n)); }
    if n % 4 != 0 { return None; }
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
    for i in 1..p { is_residue[(i * i) % p] = true; }

    // Build Paley matrix
    let mut h = Array2::zeros((n, n));
    // First row and column: all 1s
    for j in 0..n { h[[0, j]] = 1.0; }
    for i in 0..n { h[[i, 0]] = 1.0; }
    // Lower-right block: Q - I
    for i in 0..p {
        for j in 0..p {
            let q_val = if i == j { 0.0 }
                else { if is_residue[(i + p - j) % p] { 1.0 } else { -1.0 } };
            h[[i + 1, j + 1]] = q_val - if i == j { 1.0 } else { 0.0 };
        }
    }
    h
}

fn is_prime(n: usize) -> bool {
    if n < 2 { return false; }
    if n == 2 { return true; }
    if n % 2 == 0 { return false; }
    let sqrt_n = (n as f64).sqrt() as usize + 1;
    (3..=sqrt_n).step_by(2).all(|i| n % i != 0)
}

fn next_power_of_2(n: usize) -> usize {
    if n == 0 { return 1; }
    if n.is_power_of_two() { n }
    else { 1 << (64 - (n - 1).leading_zeros()) }
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
        return Err(ReplicationError::DimensionMismatch { expected: n_obs, got: stratum.len().min(psu.len()) });
    }
    if !(0.0..1.0).contains(&fay_coef) {
        return Err(ReplicationError::InvalidInput("fay_coef must be in [0, 1)".into()));
    }

    let (stratum_map, psu_to_idx) = build_brr_stratum_map(&stratum, &psu, seed)?;
    let n_strata = stratum_map.len();
    let (hadamard, h_size) = get_hadamard_for_strata(n_strata)?;
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
                let h_val = hadamard[[stratum_idx % h_size, r % h_size]];
                let mult = if (h_val > 0.0 && psu_idx == 0) || (h_val < 0.0 && psu_idx == 1)
                    { k_plus } else { k_minus };
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
    let mut stratum_psus: HashMap<i64, Vec<i64>> = HashMap::new();
    for i in 0..stratum.len() {
        let s = stratum[i];
        let p = psu[i];
        let psus = stratum_psus.entry(s).or_default();
        if !psus.contains(&p) { psus.push(p); }
    }

    for (&s, psus) in &stratum_psus {
        if psus.len() != 2 {
            return Err(ReplicationError::BrrPsuCount { stratum: s, count: psus.len() });
        }
    }

    if let Some(seed) = seed {
        let mut rng = Rng::new(seed);
        for psus in stratum_psus.values_mut() { rng.shuffle(psus); }
    }

    let mut strata: Vec<i64> = stratum_psus.keys().copied().collect();
    strata.sort();

    let mut stratum_map = HashMap::new();
    let mut psu_to_idx = HashMap::new();
    for (idx, &s) in strata.iter().enumerate() {
        let psus = stratum_psus[&s].clone();
        for (pi, &p) in psus.iter().enumerate() { psu_to_idx.insert((s, p), pi); }
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
) -> Result<(Array2<f64>, f64)> {
    if paired { create_jk2_weights(wgt, stratum, psu, seed) }
    else { create_jkn_weights(wgt, stratum, psu) }
}

/// JKn: delete-one-PSU-at-a-time
pub fn create_jkn_weights(
    wgt: ArrayView1<f64>,
    stratum: Option<ArrayView1<i64>>,
    psu: ArrayView1<i64>,
) -> Result<(Array2<f64>, f64)> {
    let n_obs = wgt.len();
    if psu.len() != n_obs {
        return Err(ReplicationError::DimensionMismatch { expected: n_obs, got: psu.len() });
    }

    let stratum_vec = stratum.map(|s| s.to_owned()).unwrap_or_else(|| Array1::ones(n_obs));
    let stratum_psus = build_stratum_psu_list(&stratum_vec, &psu);
    let n_reps: usize = stratum_psus.values().map(|v| v.len()).sum();

    if n_reps < 2 {
        return Err(ReplicationError::InsufficientPsus { required: 2, got: n_reps });
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

    let stratum_nh: HashMap<i64, f64> = stratum_psus.iter()
        .map(|(&s, psus)| (s, psus.len() as f64)).collect();

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
    Ok((result, n_reps as f64))
}

/// JK2: paired jackknife (one replicate per stratum)
fn create_jk2_weights(
    wgt: ArrayView1<f64>,
    stratum: Option<ArrayView1<i64>>,
    psu: ArrayView1<i64>,
    seed: Option<u64>,
) -> Result<(Array2<f64>, f64)> {
    let n_obs = wgt.len();
    if psu.len() != n_obs {
        return Err(ReplicationError::DimensionMismatch { expected: n_obs, got: psu.len() });
    }

    let stratum_vec = stratum.ok_or_else(||
        ReplicationError::InvalidInput("JK2 requires stratum".into()))?;
    if stratum_vec.len() != n_obs {
        return Err(ReplicationError::DimensionMismatch { expected: n_obs, got: stratum_vec.len() });
    }
    let stratum_vec = stratum_vec.to_owned();
    let stratum_psus = build_stratum_psu_list(&stratum_vec, &psu);

    for (&s, psus) in &stratum_psus {
        if psus.len() < 2 {
            return Err(ReplicationError::PairedJkPsuCount { stratum: s, count: psus.len() });
        }
    }

    let mut strata: Vec<i64> = stratum_psus.keys().copied().collect();
    strata.sort();
    let n_reps = strata.len();

    let mut rng = Rng::new(seed.unwrap_or(0));
    let deleted_psu_idx: HashMap<i64, usize> = strata.iter()
        .map(|&s| (s, rng.next_index(stratum_psus[&s].len()))).collect();

    let mut psu_to_idx: HashMap<(i64, i64), usize> = HashMap::new();
    for (&s, psus) in &stratum_psus {
        for (idx, &p) in psus.iter().enumerate() { psu_to_idx.insert((s, p), idx); }
    }

    let stratum_nh: HashMap<i64, usize> = stratum_psus.iter()
        .map(|(&s, psus)| (s, psus.len())).collect();

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
                    if psu_idx == del_idx { rep_wgt[i] = 0.0; }
                    else { rep_wgt[i] *= adj; }
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
        return Err(ReplicationError::DimensionMismatch { expected: n_obs, got: psu.len() });
    }

    let stratum_vec = stratum.map(|s| s.to_owned()).unwrap_or_else(|| Array1::ones(n_obs));
    let stratum_psus = build_stratum_psu_list(&stratum_vec, &psu);

    let rep_weights: Vec<Array1<f64>> = (0..n_reps)
        .into_par_iter()
        .map(|r| {
            let mut rng = Rng::new(seed.wrapping_add(r as u64));
            let mut rep_wgt = Array1::zeros(n_obs);

            for (&s, psus) in &stratum_psus {
                let n_psu = psus.len();
                let mut counts: HashMap<i64, usize> = HashMap::new();
                for _ in 0..n_psu {
                    let sel = psus[rng.next_index(n_psu)];
                    *counts.entry(sel).or_insert(0) += 1;
                }
                for i in 0..n_obs {
                    if stratum_vec[i] == s {
                        rep_wgt[i] = wgt[i] * *counts.get(&psu[i]).unwrap_or(&0) as f64;
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
        return Err(ReplicationError::InvalidInput("SDR requires ≥2 replicates".into()));
    }

    let stratum_vec = stratum.map(|s| s.to_owned()).unwrap_or_else(|| Array1::ones(n_obs));
    let order_vec = order.map(|o| o.to_owned())
        .unwrap_or_else(|| Array1::from_iter(0..n_obs as i64));

    let sorted_indices = build_stratum_sorted_indices(&stratum_vec, &order_vec);
    let (hadamard, h_size) = get_hadamard_for_strata(n_reps)?;

    let rep_weights: Vec<Array1<f64>> = (0..n_reps)
        .into_par_iter()
        .map(|r| {
            let mut rep_wgt = wgt.to_owned();
            for indices in sorted_indices.values() {
                let n_h = indices.len();
                if n_h < 2 { continue; }
                for k in 0..(n_h - 1) {
                    let i1 = indices[k];
                    let i2 = indices[k + 1];
                    let h_val = hadamard[[k % h_size, r % h_size]];
                    let adj = std::f64::consts::SQRT_2 * h_val / 2.0;
                    rep_wgt[i1] *= 1.0 + adj;
                    rep_wgt[i2] *= 1.0 - adj;
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
    let mut map: HashMap<i64, Vec<i64>> = HashMap::new();
    for i in 0..stratum.len() {
        let psus = map.entry(stratum[i]).or_default();
        if !psus.contains(&psu[i]) { psus.push(psu[i]); }
    }
    map
}

fn build_stratum_sorted_indices(stratum: &Array1<i64>, order: &Array1<i64>) -> HashMap<i64, Vec<usize>> {
    let mut groups: HashMap<i64, Vec<(i64, usize)>> = HashMap::new();
    for i in 0..stratum.len() {
        groups.entry(stratum[i]).or_default().push((order[i], i));
    }
    groups.into_iter().map(|(s, mut v)| {
        v.sort_by_key(|(o, _)| *o);
        (s, v.into_iter().map(|(_, i)| i).collect())
    }).collect()
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
        let (rep_wgt, df) = create_brr_weights(wgt.view(), stratum.view(), psu.view(), Some(4), 0.0, None).unwrap();
        assert_eq!(rep_wgt.dim(), (4, 4));
        assert_eq!(df, 2.0);
    }

    #[test]
    fn test_jkn_basic() {
        let wgt = array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let stratum = array![1, 1, 1, 2, 2, 2];
        let psu = array![1, 2, 3, 1, 2, 3];
        let (rep_wgt, df) = create_jkn_weights(wgt.view(), Some(stratum.view()), psu.view()).unwrap();
        assert_eq!(rep_wgt.dim(), (6, 6));
        assert_eq!(df, 6.0);
    }

    #[test]
    fn test_jk2_paired() {
        let wgt = array![1.0, 1.0, 1.0, 1.0];
        let stratum = array![1, 1, 2, 2];
        let psu = array![1, 2, 1, 2];
        let (rep_wgt, df) = create_jk_weights(wgt.view(), Some(stratum.view()), psu.view(), true, Some(42)).unwrap();
        assert_eq!(rep_wgt.dim(), (4, 2)); // 2 strata = 2 replicates
        assert_eq!(df, 2.0);
    }

    #[test]
    fn test_bootstrap_basic() {
        let wgt = array![1.0, 1.0, 1.0, 1.0];
        let stratum = array![1, 1, 2, 2];
        let psu = array![1, 2, 1, 2];
        let (rep_wgt, df) = create_bootstrap_weights(wgt.view(), Some(stratum.view()), psu.view(), 10, 42).unwrap();
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
}
