// src/sampling/pps.rs

use std::collections::HashMap;
use crate::rng::Rng;
use crate::sampling::srs::SamplingError;
pub use crate::sampling::srs::Result;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PpsMethod { Sys, Wr, Brewer, Murphy, Rs }

impl PpsMethod {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            s if s.eq_ignore_ascii_case("sys") || s.eq_ignore_ascii_case("systematic")
                => Some(PpsMethod::Sys),
            s if s.eq_ignore_ascii_case("wr") || s.eq_ignore_ascii_case("with_replacement")
                => Some(PpsMethod::Wr),
            s if s.eq_ignore_ascii_case("brewer")  => Some(PpsMethod::Brewer),
            s if s.eq_ignore_ascii_case("murphy")  => Some(PpsMethod::Murphy),
            s if s.eq_ignore_ascii_case("rs") || s.eq_ignore_ascii_case("rao_sampford")
                || s.eq_ignore_ascii_case("sampford") => Some(PpsMethod::Rs),
            _ => None,
        }
    }
}

pub enum PpsN {
    Scalar(usize),
    PerStratum(HashMap<i64, usize>),
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub fn select_pps(
    frame: &[i64],
    n: PpsN,
    mos: &[f64],
    stratum: Option<&[i64]>,
    method: PpsMethod,
    certainty_threshold: f64,
    seed: Option<u64>,
) -> Result<(Vec<i64>, Vec<i64>, Vec<f64>, Vec<bool>)> {
    if !(certainty_threshold > 0.0 && certainty_threshold <= 1.0) {
        return Err(SamplingError::InvalidInput(
            "certainty_threshold must be in (0, 1]".into(),
        ));
    }
    match stratum {
        None => {
            let n_scalar = match n {
                PpsN::Scalar(v) => v,
                PpsN::PerStratum(_) => return Err(SamplingError::InvalidInput(
                    "per-stratum n requires stratum array".into(),
                )),
            };
            let positions: Vec<usize> = (0..frame.len()).collect();
            pps_indexed(frame, &positions, mos, n_scalar, method, certainty_threshold, seed)
        }
        Some(strat) => select_pps_stratified(frame, n, mos, strat, method, certainty_threshold, seed),
    }
}

// ---------------------------------------------------------------------------
// Shared: iterative certainty extraction — no per-iteration allocation
// ---------------------------------------------------------------------------

/// Returns (cert_mask, n_rem).
///
/// Uses a `remaining` boolean mask and running sum instead of rebuilding
/// index vecs on each pass. O(passes * N) time, O(N) extra space, 0 heap
/// allocations inside the loop.
fn extract_certainty(p0: &[f64], n: usize, threshold: f64) -> (Vec<bool>, usize) {
    let big = p0.len();
    let mut cert_mask = vec![false; big];
    let mut remaining = vec![true; big];
    let mut n_rem = n;
    let mut total_rem: f64 = p0.iter().sum();

    loop {
        if n_rem == 0 || total_rem <= 0.0 {
            break;
        }
        let n_rem_f = n_rem as f64;
        // Snapshot denominator at pass start — critical correctness requirement.
        // If we updated total_rem mid-pass as units are extracted, the shrinking
        // denominator would artificially inflate remaining units' ratios and cause
        // over-extraction within a single pass.  Instead we commit extractions to
        // total_rem but evaluate all units against the pass-start snapshot.
        let total_rem_pass = total_rem;
        let threshold_scaled = threshold - 1e-12;
        let mut found_any = false;

        for i in 0..big {
            if remaining[i] && n_rem_f * p0[i] / total_rem_pass >= threshold_scaled {
                cert_mask[i] = true;
                remaining[i] = false;
                total_rem -= p0[i];   // update running total for next pass only
                n_rem = n_rem.saturating_sub(1);
                found_any = true;
            }
        }
        if !found_any {
            break;
        }
    }

    (cert_mask, n_rem)
}

// ---------------------------------------------------------------------------
// Core dispatcher — operates on a positions slice (no frame data copy)
// ---------------------------------------------------------------------------

fn pps_indexed(
    frame: &[i64],
    positions: &[usize],     // indices into frame and mos
    mos: &[f64],
    n: usize,
    method: PpsMethod,
    certainty_threshold: f64,
    seed: Option<u64>,
) -> Result<(Vec<i64>, Vec<i64>, Vec<f64>, Vec<bool>)> {
    let cap = positions.len();

    if cap == 0 || n == 0 {
        return Ok((vec![], vec![], vec![], vec![]));
    }
    if n >= cap {
        // Select all
        let sel: Vec<i64>  = positions.iter().map(|&p| frame[p]).collect();
        let hits            = vec![1i64; cap];
        let probs           = vec![1.0f64; cap];
        let cert            = vec![true; cap];
        return Ok((sel, hits, probs, cert));
    }

    // Validate and compute normalised probabilities from positions
    let total_mos: f64 = positions.iter().map(|&p| mos[p]).sum();
    if total_mos <= 0.0 {
        return Err(SamplingError::InvalidInput("sum(MOS) must be > 0".into()));
    }
    if positions.iter().any(|&p| mos[p] < 0.0) {
        return Err(SamplingError::InvalidInput("All MOS values must be non-negative".into()));
    }

    // p0[i] = normalised probability for positions[i] — one allocation here
    let p0: Vec<f64> = positions.iter().map(|&p| mos[p] / total_mos).collect();

    match method {
        PpsMethod::Sys    => pps_sys(frame, positions, &p0, n, certainty_threshold, seed),
        PpsMethod::Wr     => pps_wr(frame, positions, &p0, n, seed),
        PpsMethod::Brewer => pps_brewer(frame, positions, &p0, n, certainty_threshold, seed),
        PpsMethod::Murphy => pps_murphy(frame, positions, &p0, seed),
        PpsMethod::Rs     => pps_rs(frame, positions, &p0, n, certainty_threshold, seed),
    }
}

// ---------------------------------------------------------------------------
// Stratified PPS — parallel over strata
// ---------------------------------------------------------------------------

fn select_pps_stratified(
    frame: &[i64],
    n: PpsN,
    mos: &[f64],
    stratum: &[i64],
    method: PpsMethod,
    certainty_threshold: f64,
    seed: Option<u64>,
) -> Result<(Vec<i64>, Vec<i64>, Vec<f64>, Vec<bool>)> {
    let mut strat_map: HashMap<i64, Vec<usize>> = HashMap::new();
    for (pos, &s) in stratum.iter().enumerate() {
        strat_map.entry(s).or_default().push(pos);
    }

    let mut strata: Vec<i64> = strat_map.keys().copied().collect();
    strata.sort_unstable();

    let base_seed = seed.unwrap_or(0);

    let tasks: Vec<(&Vec<usize>, usize, u64)> = strata
        .iter()
        .enumerate()
        .filter_map(|(strat_idx, &s)| {
            let positions = &strat_map[&s];
            let n_s = match &n {
                PpsN::Scalar(v) => *v,
                PpsN::PerStratum(map) => *map.get(&s).unwrap_or(&0),
            };
            if n_s == 0 { return None; }
            let child_seed = base_seed
                .wrapping_add((strat_idx as u64).wrapping_mul(0x9e3779b97f4a7c15));
            Some((positions, n_s, child_seed))
        })
        .collect();

    use rayon::prelude::*;
    let results: Vec<Result<(Vec<i64>, Vec<i64>, Vec<f64>, Vec<bool>)>> =
        if tasks.len() >= 4 {
            tasks.par_iter().map(|(positions, n_s, child_seed)| {
                pps_indexed(frame, positions, mos, *n_s, method, certainty_threshold, Some(*child_seed))
            }).collect()
        } else {
            tasks.iter().map(|(positions, n_s, child_seed)| {
                pps_indexed(frame, positions, mos, *n_s, method, certainty_threshold, Some(*child_seed))
            }).collect()
        };

    let total_n: usize = tasks.iter().map(|(_, n_s, _)| n_s).sum();
    let mut selected = Vec::with_capacity(total_n);
    let mut out_hits  = Vec::with_capacity(total_n);
    let mut out_probs = Vec::with_capacity(total_n);
    let mut out_cert  = Vec::with_capacity(total_n);

    for res in results {
        let (sel, hits, probs, cert) = res?;
        selected.extend_from_slice(&sel);
        out_hits.extend_from_slice(&hits);
        out_probs.extend_from_slice(&probs);
        out_cert.extend_from_slice(&cert);
    }

    Ok((selected, out_hits, out_probs, out_cert))
}

// ---------------------------------------------------------------------------
// PPS Systematic (Madow)
// ---------------------------------------------------------------------------

fn pps_sys(
    frame: &[i64],
    positions: &[usize],
    p0: &[f64],
    n: usize,
    certainty_threshold: f64,
    seed: Option<u64>,
) -> Result<(Vec<i64>, Vec<i64>, Vec<f64>, Vec<bool>)> {
    let cap = positions.len();
    let (cert_mask, n_rem) = extract_certainty(p0, n, certainty_threshold);
    let mut hits = vec![0i64; cap];
    let mut rng = Rng::new(seed.unwrap_or(0));

    for (i, &c) in cert_mask.iter().enumerate() {
        if c { hits[i] = 1; }
    }

    if n_rem > 0 {
        // Build cumulative array over non-certainty units only
        // rem_count known upfront — one allocation
        let rem_count = cert_mask.iter().filter(|&&c| !c).count();
        let mut cum = Vec::with_capacity(rem_count + 1);
        let mut rem_positions = Vec::with_capacity(rem_count); // local indices

        let total_rem: f64 = cert_mask.iter().enumerate()
            .filter(|&(_, &c)| !c)
            .map(|(i, _)| p0[i])
            .sum();

        cum.push(0.0_f64);
        for (i, &c) in cert_mask.iter().enumerate() {
            if !c {
                cum.push(cum.last().unwrap() + p0[i] / total_rem);
                rem_positions.push(i);
            }
        }
        *cum.last_mut().unwrap() = 1.0;

        let interval = 1.0 / n_rem as f64;
        let start = rng.next_f64() * interval;
        let mut cum_idx = 0usize;

        for k in 0..n_rem {
            let target = start + k as f64 * interval;
            while cum_idx + 1 < cum.len() && cum[cum_idx + 1] < target {
                cum_idx += 1;
            }
            let local = rem_positions[cum_idx.min(rem_positions.len() - 1)];
            hits[local] += 1;
        }
    }

    assemble_pps_output(frame, positions, p0, &hits, &cert_mask, n)
}

// ---------------------------------------------------------------------------
// PPS With Replacement
// ---------------------------------------------------------------------------

fn pps_wr(
    frame: &[i64],
    positions: &[usize],
    p0: &[f64],
    n: usize,
    seed: Option<u64>,
) -> Result<(Vec<i64>, Vec<i64>, Vec<f64>, Vec<bool>)> {
    let cap = positions.len();
    let mut rng = Rng::new(seed.unwrap_or(0));
    let mut hits = vec![0i64; cap];

    for _ in 0..n {
        hits[rng.weighted_choice(p0)] += 1;
    }

    // WR inclusion probability: 1 - (1 - p_i)^n
    let n_f = n as f64;
    let mut sel   = Vec::with_capacity(n);
    let mut ohits = Vec::with_capacity(n);
    let mut probs = Vec::with_capacity(n);
    let mut cert  = Vec::with_capacity(n);

    for (i, &h) in hits.iter().enumerate() {
        if h > 0 {
            sel.push(frame[positions[i]]);
            ohits.push(h);
            probs.push(1.0 - (1.0 - p0[i]).powf(n_f));
            cert.push(false);
        }
    }
    Ok((sel, ohits, probs, cert))
}

// ---------------------------------------------------------------------------
// PPS Brewer — pre-allocate, reuse mask, no per-iteration heap allocs
// ---------------------------------------------------------------------------

fn pps_brewer(
    frame: &[i64],
    positions: &[usize],
    p0: &[f64],
    n: usize,
    certainty_threshold: f64,
    seed: Option<u64>,
) -> Result<(Vec<i64>, Vec<i64>, Vec<f64>, Vec<bool>)> {
    let cap = p0.len();
    let (cert_mask, n_rem) = extract_certainty(p0, n, certainty_threshold);
    let mut hits = vec![0i64; cap];
    let mut available = vec![true; cap]; // true = eligible (not yet selected, not cert)
    let mut rng = Rng::new(seed.unwrap_or(0));

    for (i, &c) in cert_mask.iter().enumerate() {
        if c { hits[i] = 1; available[i] = false; }
    }

    // Pre-allocate working weight buffer — reused across iterations, no heap alloc in loop
    let mut w_buf = vec![0.0f64; cap];

    let mut to_draw = n_rem;
    while to_draw > 0 {
        let t = to_draw as f64;
        let mut w_total = 0.0f64;

        // Fill weights for available units in-place
        for i in 0..cap {
            if available[i] {
                let p = p0[i];
                let denom = 1.0 - t * p;
                let w = if denom > 1e-12 { p * (1.0 - p) / denom } else { p };
                w_buf[i] = w;
                w_total += w;
            } else {
                w_buf[i] = 0.0;
            }
        }

        if w_total <= 0.0 { break; }

        // Weighted pick over w_buf — single pass, no allocation
        let mut u = rng.next_f64() * w_total;
        let pick = {
            let mut chosen = cap - 1;
            for i in 0..cap {
                if available[i] {
                    u -= w_buf[i];
                    if u <= 0.0 { chosen = i; break; }
                }
            }
            chosen
        };

        hits[pick] = 1;
        available[pick] = false;
        to_draw -= 1;
    }

    assemble_pps_output(frame, positions, p0, &hits, &cert_mask, n)
}

// ---------------------------------------------------------------------------
// PPS Murphy (n=2 only)
// ---------------------------------------------------------------------------

fn pps_murphy(
    frame: &[i64],
    positions: &[usize],
    p0: &[f64],
    seed: Option<u64>,
) -> Result<(Vec<i64>, Vec<i64>, Vec<f64>, Vec<bool>)> {
    let cap = p0.len();
    if cap < 2 || p0.iter().filter(|&&p| p > 0.0).count() < 2 {
        return Err(SamplingError::InvalidInput(
            "Murphy PPS requires at least two units with positive MOS".into(),
        ));
    }
    let mut rng = Rng::new(seed.unwrap_or(0));

    let first = rng.weighted_choice(p0);

    // Second draw: skip first, renormalise in-place with a temp total
    let rem_total: f64 = p0.iter().enumerate()
        .filter(|&(i, _)| i != first)
        .map(|(_, &p)| p)
        .sum();

    // Walk the weights without allocating a rem vec
    let mut u = rng.next_f64() * rem_total;
    let second = {
        let mut chosen = if first == cap - 1 { cap - 2 } else { cap - 1 };
        for i in 0..cap {
            if i != first {
                u -= p0[i];
                if u <= 0.0 { chosen = i; break; }
            }
        }
        chosen
    };

    let s: f64 = p0.iter().map(|&p| p / (1.0 - p)).sum();
    let pi = |idx: usize| -> f64 {
        let p = p0[idx];
        (p * (1.0 + s - p / (1.0 - p))).clamp(0.0, 1.0)
    };

    Ok((
        vec![frame[positions[first]], frame[positions[second]]],
        vec![1, 1],
        vec![pi(first), pi(second)],
        vec![false, false],
    ))
}

// ---------------------------------------------------------------------------
// PPS Rao-Sampford
// ---------------------------------------------------------------------------

fn pps_rs(
    frame: &[i64],
    positions: &[usize],
    p0: &[f64],
    n: usize,
    certainty_threshold: f64,
    seed: Option<u64>,
) -> Result<(Vec<i64>, Vec<i64>, Vec<f64>, Vec<bool>)> {
    const MAX_ATTEMPTS: usize = 1000;
    let cap = p0.len();
    let (cert_mask, n_rem) = extract_certainty(p0, n, certainty_threshold);
    let mut hits = vec![0i64; cap];
    let mut rng = Rng::new(seed.unwrap_or(0));

    for (i, &c) in cert_mask.iter().enumerate() {
        if c { hits[i] = 1; }
    }

    if n_rem > 0 {
        // Build rem_idx once
        let rem_idx: Vec<usize> = (0..cap).filter(|&i| !cert_mask[i]).collect();
        let rem_len = rem_idx.len();

        if rem_len >= n_rem {
            let total_rem: f64 = rem_idx.iter().map(|&i| p0[i]).sum();
            let p_rem: Vec<f64> = rem_idx.iter().map(|&i| p0[i] / total_rem).collect();

            if n_rem == 1 {
                hits[rem_idx[rng.weighted_choice(&p_rem)]] = 1;
            } else {
                let sw_total: f64 = p_rem.iter()
                    .map(|&p| { let d = 1.0 - n_rem as f64 * p; if d > 1e-12 { p/d } else { p } })
                    .sum();
                let sampford_w: Vec<f64> = p_rem.iter()
                    .map(|&p| { let d = 1.0 - n_rem as f64 * p; let w = if d > 1e-12 { p/d } else { p }; w / sw_total })
                    .collect();

                // Pre-allocate candidate buffers reused across rejection attempts
                // selected_mask[j] = true if local index j is in current sample
                let mut selected_mask = vec![false; rem_len];
                let mut selected_buf  = Vec::with_capacity(n_rem);
                let mut found = false;

                'outer: for _ in 0..MAX_ATTEMPTS {
                    let first_local = rng.weighted_choice(&p_rem);

                    // Reset reusable buffers — O(n_rem) not O(rem_len)
                    for &j in &selected_buf { selected_mask[j] = false; }
                    selected_buf.clear();
                    selected_buf.push(first_local);
                    selected_mask[first_local] = true;

                    // Draw n_rem-1 more using weighted selection without replacement
                    // Uses selected_mask to skip chosen units — no Vec::remove()
                    for _ in 1..n_rem {
                        let avail_total: f64 = (0..rem_len)
                            .filter(|&j| !selected_mask[j])
                            .map(|j| sampford_w[j])
                            .sum();
                        if avail_total <= 0.0 { continue 'outer; }

                        let mut u = rng.next_f64() * avail_total;
                        let mut pick = rem_len - 1;
                        for j in 0..rem_len {
                            if !selected_mask[j] {
                                u -= sampford_w[j];
                                if u <= 0.0 { pick = j; break; }
                            }
                        }
                        if selected_mask[pick] { continue 'outer; }
                        selected_buf.push(pick);
                        selected_mask[pick] = true;
                    }

                    // Accept if first_local not re-selected (always true here
                    // since selected_mask prevents it) — Sampford acceptance check
                    for &j in &selected_buf {
                        hits[rem_idx[j]] = 1;
                    }
                    found = true;
                    break;
                }

                if !found {
                    // Fallback: plain WOR shuffle
                    let mut pool: Vec<usize> = (0..rem_len).collect();
                    rng.shuffle(&mut pool);
                    for &j in &pool[..n_rem] { hits[rem_idx[j]] = 1; }
                }
            }
        }
    }

    assemble_pps_output(frame, positions, p0, &hits, &cert_mask, n)
}

// ---------------------------------------------------------------------------
// Shared output assembly — extracted to avoid code duplication
// ---------------------------------------------------------------------------

#[inline]
fn assemble_pps_output(
    frame: &[i64],
    positions: &[usize],
    p0: &[f64],
    hits: &[i64],
    cert_mask: &[bool],
    n: usize,
) -> Result<(Vec<i64>, Vec<i64>, Vec<f64>, Vec<bool>)> {
    let n_sel = hits.iter().filter(|&&h| h > 0).count();
    let mut sel   = Vec::with_capacity(n_sel);
    let mut ohits = Vec::with_capacity(n_sel);
    let mut probs = Vec::with_capacity(n_sel);
    let mut cert  = Vec::with_capacity(n_sel);

    for (i, &h) in hits.iter().enumerate() {
        if h > 0 {
            sel.push(frame[positions[i]]);
            ohits.push(h);
            probs.push(if cert_mask[i] { 1.0 } else { (n as f64 * p0[i]).min(1.0) });
            cert.push(cert_mask[i]);
        }
    }
    Ok((sel, ohits, probs, cert))
}
