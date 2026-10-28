//! PPS selection: systematic, with replacement, Brewer, Murphy and Rao-Sampford.

use crate::rng::Rng;
pub(crate) use crate::sampling::srs::Result;
use crate::sampling::srs::{SamplingError, check_n_map};
use std::collections::HashMap;

/// Selected rows, their hits, inclusion probabilities and certainty flags.
pub(crate) type PpsDraw = (Vec<i64>, Vec<i64>, Vec<f64>, Vec<bool>);

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum PpsMethod {
    Sys,
    Wr,
    Brewer,
    Murphy,
    Rs,
}

impl PpsMethod {
    pub(crate) fn from_str(s: &str) -> Option<Self> {
        match s {
            s if s.eq_ignore_ascii_case("sys") || s.eq_ignore_ascii_case("systematic") => {
                Some(PpsMethod::Sys)
            }
            s if s.eq_ignore_ascii_case("wr") || s.eq_ignore_ascii_case("with_replacement") => {
                Some(PpsMethod::Wr)
            }
            s if s.eq_ignore_ascii_case("brewer") => Some(PpsMethod::Brewer),
            s if s.eq_ignore_ascii_case("murphy") => Some(PpsMethod::Murphy),
            s if s.eq_ignore_ascii_case("rs")
                || s.eq_ignore_ascii_case("rao_sampford")
                || s.eq_ignore_ascii_case("sampford") =>
            {
                Some(PpsMethod::Rs)
            }
            _ => None,
        }
    }
}

#[derive(Clone)]
pub(crate) enum PpsN {
    Scalar(usize),
    PerStratum(HashMap<i64, usize>),
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub(crate) fn select_pps(
    frame: &[i64],
    n: PpsN,
    mos: &[f64],
    stratum: Option<&[i64]>,
    method: PpsMethod,
    certainty_threshold: f64,
    seed: u64,
) -> Result<PpsDraw> {
    if !(certainty_threshold > 0.0 && certainty_threshold <= 1.0) {
        return Err(SamplingError::InvalidInput(
            "certainty_threshold must be in (0, 1]".into(),
        ));
    }
    if mos.len() != frame.len() {
        return Err(SamplingError::InvalidInput(format!(
            "mos length ({}) must match frame length ({})",
            mos.len(),
            frame.len()
        )));
    }
    if let Some(strat) = stratum
        && strat.len() != frame.len()
    {
        return Err(SamplingError::InvalidInput(format!(
            "stratum length ({}) must match frame length ({})",
            strat.len(),
            frame.len()
        )));
    }
    match stratum {
        None => {
            let n_scalar = match n {
                PpsN::Scalar(v) => v,
                PpsN::PerStratum(_) => {
                    return Err(SamplingError::InvalidInput(
                        "per-stratum n requires stratum array".into(),
                    ));
                }
            };
            let positions: Vec<usize> = (0..frame.len()).collect();
            pps_indexed(
                frame,
                &positions,
                mos,
                n_scalar,
                method,
                certainty_threshold,
                seed,
            )
        }
        Some(strat) => {
            select_pps_stratified(frame, n, mos, strat, method, certainty_threshold, seed)
        }
    }
}

// ---------------------------------------------------------------------------
// Shared: iterative certainty extraction — no per-iteration allocation
// ---------------------------------------------------------------------------

/// Repeatedly marks units with n * p >= threshold as certainties; returns (cert_mask, n_rem).
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
        // Judge a whole pass against its starting total: shrinking it mid-pass over-extracts.
        let total_rem_pass = total_rem;
        let threshold_scaled = threshold - 1e-12;
        let mut found_any = false;

        for i in 0..big {
            // Never mark more than n units certain, even mid-pass.
            if n_rem == 0 {
                break;
            }
            if remaining[i] && n_rem_f * p0[i] / total_rem_pass >= threshold_scaled {
                cert_mask[i] = true;
                remaining[i] = false;
                total_rem -= p0[i]; // update running total for next pass only
                n_rem -= 1;
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
    positions: &[usize], // indices into frame and mos
    mos: &[f64],
    n: usize,
    method: PpsMethod,
    certainty_threshold: f64,
    seed: u64,
) -> Result<PpsDraw> {
    let cap = positions.len();

    if cap == 0 || n == 0 {
        return Ok((vec![], vec![], vec![], vec![]));
    }
    if matches!(method, PpsMethod::Murphy) && n != 2 {
        return Err(SamplingError::InvalidInput(format!(
            "Murphy PPS selects exactly 2 units; got n={n}. Use method='wr' or \
             'rs' for other sample sizes."
        )));
    }
    // WR may legitimately draw n >= N, so the take-all shortcut is for WOR methods only.
    if n >= cap && !matches!(method, PpsMethod::Wr) {
        let sel: Vec<i64> = positions.iter().map(|&p| frame[p]).collect();
        let hits = vec![1i64; cap];
        let probs = vec![1.0f64; cap];
        let cert = vec![true; cap];
        return Ok((sel, hits, probs, cert));
    }

    if positions
        .iter()
        .any(|&p| !mos[p].is_finite() || mos[p] < 0.0)
    {
        return Err(SamplingError::InvalidInput(
            "All MOS values must be finite and non-negative".into(),
        ));
    }
    let total_mos: f64 = positions.iter().map(|&p| mos[p]).sum();
    if total_mos <= 0.0 {
        return Err(SamplingError::InvalidInput("sum(MOS) must be > 0".into()));
    }

    let p0: Vec<f64> = positions.iter().map(|&p| mos[p] / total_mos).collect();

    match method {
        PpsMethod::Sys => pps_sys(frame, positions, &p0, n, certainty_threshold, seed),
        PpsMethod::Wr => pps_wr(frame, positions, &p0, n, seed),
        PpsMethod::Brewer => pps_brewer(frame, positions, &p0, n, certainty_threshold, seed),
        PpsMethod::Murphy => pps_murphy(frame, positions, &p0, seed),
        PpsMethod::Rs => pps_rs(frame, positions, &p0, n, certainty_threshold, seed),
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
    seed: u64,
) -> Result<PpsDraw> {
    let mut strat_map: HashMap<i64, Vec<usize>> = HashMap::new();
    for (pos, &s) in stratum.iter().enumerate() {
        strat_map.entry(s).or_default().push(pos);
    }

    let mut strata: Vec<i64> = strat_map.keys().copied().collect();
    strata.sort_unstable();

    if let PpsN::PerStratum(map) = &n {
        check_n_map(map, &strat_map)?;
    }

    let tasks: Vec<(&Vec<usize>, usize, u64)> = strata
        .iter()
        .enumerate()
        .filter_map(|(strat_idx, &s)| {
            let positions = &strat_map[&s];
            let n_s = match &n {
                PpsN::Scalar(v) => *v,
                PpsN::PerStratum(map) => map[&s],
            };
            if n_s == 0 {
                return None;
            }
            let child_seed = seed.wrapping_add((strat_idx as u64).wrapping_mul(0x9e3779b97f4a7c15));
            Some((positions, n_s, child_seed))
        })
        .collect();

    use rayon::prelude::*;
    let results: Vec<Result<PpsDraw>> = if tasks.len() >= 4 {
        tasks
            .par_iter()
            .map(|(positions, n_s, child_seed)| {
                pps_indexed(
                    frame,
                    positions,
                    mos,
                    *n_s,
                    method,
                    certainty_threshold,
                    *child_seed,
                )
            })
            .collect()
    } else {
        tasks
            .iter()
            .map(|(positions, n_s, child_seed)| {
                pps_indexed(
                    frame,
                    positions,
                    mos,
                    *n_s,
                    method,
                    certainty_threshold,
                    *child_seed,
                )
            })
            .collect()
    };

    let total_n: usize = tasks.iter().map(|(_, n_s, _)| n_s).sum();
    let mut selected = Vec::with_capacity(total_n);
    let mut out_hits = Vec::with_capacity(total_n);
    let mut out_probs = Vec::with_capacity(total_n);
    let mut out_cert = Vec::with_capacity(total_n);

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
    seed: u64,
) -> Result<PpsDraw> {
    let cap = positions.len();
    let (cert_mask, n_rem) = extract_certainty(p0, n, certainty_threshold);
    let mut hits = vec![0i64; cap];
    let mut rng = Rng::new(seed);

    for (i, &c) in cert_mask.iter().enumerate() {
        if c {
            hits[i] = 1;
        }
    }

    if n_rem > 0 {
        let rem_count = cert_mask.iter().filter(|&&c| !c).count();
        let mut cum = Vec::with_capacity(rem_count + 1);
        let mut rem_positions = Vec::with_capacity(rem_count); // local indices

        let total_rem: f64 = cert_mask
            .iter()
            .enumerate()
            .filter(|&(_, &c)| !c)
            .map(|(i, _)| p0[i])
            .sum();
        if total_rem <= 0.0 {
            return Err(SamplingError::InvalidInput(
                "sample size exceeds number of units with positive measure of size".into(),
            ));
        }

        let mut acc = 0.0_f64;
        cum.push(acc);
        for (i, &c) in cert_mask.iter().enumerate() {
            if !c {
                acc += p0[i] / total_rem;
                cum.push(acc);
                rem_positions.push(i);
            }
        }
        if let Some(last) = cum.last_mut() {
            *last = 1.0;
        }

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

    assemble_pps_output(frame, positions, p0, &hits, &cert_mask, n_rem)
}

// ---------------------------------------------------------------------------
// PPS With Replacement
// ---------------------------------------------------------------------------

fn draw_one(rng: &mut Rng, weights: &[f64]) -> Result<usize> {
    rng.weighted_choice(weights)
        .ok_or_else(|| SamplingError::InvalidInput("sum(MOS) must be > 0".into()))
}

fn pps_wr(frame: &[i64], positions: &[usize], p0: &[f64], n: usize, seed: u64) -> Result<PpsDraw> {
    let cap = positions.len();
    let mut rng = Rng::new(seed);
    let mut hits = vec![0i64; cap];

    for _ in 0..n {
        hits[draw_one(&mut rng, p0)?] += 1;
    }

    let n_f = n as f64;
    let mut sel = Vec::with_capacity(n);
    let mut ohits = Vec::with_capacity(n);
    let mut probs = Vec::with_capacity(n);
    let mut cert = Vec::with_capacity(n);

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
// PPS Brewer (Brewer 1975; Brewer and Hanif 1983, procedure 2)
// ---------------------------------------------------------------------------

/// Brewer's draw-r weights over the available units; drawn units carry their pi into A.
fn brewer_weights(pi: &[f64], available: &[bool], n: usize, r: usize, out: &mut [f64]) -> f64 {
    let a: f64 = (0..pi.len())
        .filter(|&i| !available[i])
        .map(|i| pi[i])
        .sum();
    let left = n as f64 - a;
    let draws_left = (n - r + 1) as f64;
    let mut total = 0.0f64;
    for i in 0..pi.len() {
        out[i] = if available[i] {
            pi[i] * (left - pi[i]) / (left - pi[i] * draws_left)
        } else {
            0.0
        };
        total += out[i];
    }
    total
}

fn pps_brewer(
    frame: &[i64],
    positions: &[usize],
    p0: &[f64],
    n: usize,
    certainty_threshold: f64,
    seed: u64,
) -> Result<PpsDraw> {
    let cap = p0.len();
    let (cert_mask, n_rem) = extract_certainty(p0, n, certainty_threshold);
    let mut hits = vec![0i64; cap];
    let mut available = vec![true; cap]; // true = eligible (not yet selected, not cert)
    let mut rng = Rng::new(seed);

    for (i, &c) in cert_mask.iter().enumerate() {
        if c {
            hits[i] = 1;
            available[i] = false;
        }
    }

    let rem_mass: f64 = (0..cap).filter(|&i| available[i]).map(|i| p0[i]).sum();
    if n_rem > 0 && rem_mass <= 0.0 {
        return Err(SamplingError::InvalidInput(
            "sample size exceeds number of units with positive measure of size".into(),
        ));
    }
    // Certainties get pi 0 so they stay out of A: the draws only spend n_rem.
    let pi: Vec<f64> = (0..cap)
        .map(|i| {
            if available[i] && rem_mass > 0.0 {
                n_rem as f64 * p0[i] / rem_mass
            } else {
                0.0
            }
        })
        .collect();

    let mut w_buf = vec![0.0f64; cap];

    for r in 1..=n_rem {
        let w_total = brewer_weights(&pi, &available, n_rem, r, &mut w_buf);
        if w_total <= 0.0 {
            break;
        }

        let mut u = rng.next_f64() * w_total;
        let pick = {
            let mut chosen = cap - 1;
            for i in 0..cap {
                if available[i] {
                    u -= w_buf[i];
                    if u <= 0.0 {
                        chosen = i;
                        break;
                    }
                }
            }
            chosen
        };

        hits[pick] = 1;
        available[pick] = false;
    }

    assemble_pps_output(frame, positions, p0, &hits, &cert_mask, n_rem)
}

// ---------------------------------------------------------------------------
// PPS Murphy (n=2 only)
// ---------------------------------------------------------------------------

fn pps_murphy(frame: &[i64], positions: &[usize], p0: &[f64], seed: u64) -> Result<PpsDraw> {
    let cap = p0.len();
    if cap < 2 || p0.iter().filter(|&&p| p > 0.0).count() < 2 {
        return Err(SamplingError::InvalidInput(
            "Murphy PPS requires at least two units with positive MOS".into(),
        ));
    }
    let mut rng = Rng::new(seed);

    let first = draw_one(&mut rng, p0)?;

    let rem_total: f64 = p0
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != first)
        .map(|(_, &p)| p)
        .sum();

    let mut u = rng.next_f64() * rem_total;
    let second = {
        let mut chosen = if first == cap - 1 { cap - 2 } else { cap - 1 };
        for (i, &p) in p0.iter().enumerate().take(cap) {
            if i != first {
                u -= p;
                if u <= 0.0 {
                    chosen = i;
                    break;
                }
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
    seed: u64,
) -> Result<PpsDraw> {
    const MAX_ATTEMPTS: usize = 1000;
    let cap = p0.len();
    let (cert_mask, n_rem) = extract_certainty(p0, n, certainty_threshold);
    let mut hits = vec![0i64; cap];
    let mut rng = Rng::new(seed);

    for (i, &c) in cert_mask.iter().enumerate() {
        if c {
            hits[i] = 1;
        }
    }

    if n_rem > 0 {
        let rem_idx: Vec<usize> = (0..cap).filter(|&i| !cert_mask[i]).collect();
        let rem_len = rem_idx.len();

        if rem_len >= n_rem {
            let total_rem: f64 = rem_idx.iter().map(|&i| p0[i]).sum();
            if total_rem <= 0.0 {
                return Err(SamplingError::InvalidInput(
                    "sample size exceeds number of units with positive measure of size".into(),
                ));
            }
            let p_rem: Vec<f64> = rem_idx.iter().map(|&i| p0[i] / total_rem).collect();

            if n_rem == 1 {
                hits[rem_idx[draw_one(&mut rng, &p_rem)?]] = 1;
            } else {
                // Sampford needs n*p_i < 1; larger units should already be certainties.
                if let Some(bad) = p_rem.iter().position(|&p| n_rem as f64 * p >= 1.0) {
                    return Err(SamplingError::InvalidInput(format!(
                        "Rao-Sampford requires n*p < 1 for all non-certainty units; \
                         unit at frame position {} has n*p = {:.6}. Lower the \
                         certainty threshold or reduce n.",
                        rem_idx[bad],
                        n_rem as f64 * p_rem[bad]
                    )));
                }

                // Sampford weights for the n-1 WR draws: lambda_i ∝ p_i / (1 - n*p_i).
                let lam_raw: Vec<f64> = p_rem
                    .iter()
                    .map(|&p| p / (1.0 - n_rem as f64 * p))
                    .collect();
                let lam_total: f64 = lam_raw.iter().sum();
                let lambda: Vec<f64> = lam_raw.iter().map(|&w| w / lam_total).collect();

                // Rejective: first draw by p_i, the rest WR by lambda_i, until all are distinct.
                let mut selected_mask = vec![false; rem_len];
                let mut selected_buf: Vec<usize> = Vec::with_capacity(n_rem);
                let mut found = false;

                'outer: for _ in 0..MAX_ATTEMPTS {
                    for &j in &selected_buf {
                        selected_mask[j] = false;
                    }
                    selected_buf.clear();

                    let first_local = draw_one(&mut rng, &p_rem)?;
                    selected_buf.push(first_local);
                    selected_mask[first_local] = true;

                    for _ in 1..n_rem {
                        let pick = draw_one(&mut rng, &lambda)?;
                        if selected_mask[pick] {
                            // Duplicate anywhere -> reject the WHOLE sample.
                            continue 'outer;
                        }
                        selected_buf.push(pick);
                        selected_mask[pick] = true;
                    }

                    for &j in &selected_buf {
                        hits[rem_idx[j]] = 1;
                    }
                    found = true;
                    break;
                }

                if !found {
                    // Erroring beats an equal-probability fallback with wrong weights.
                    return Err(SamplingError::InvalidInput(format!(
                        "Rao-Sampford rejective sampling did not accept a sample \
                         within {MAX_ATTEMPTS} attempts; the largest n*p is likely \
                         too close to 1. Lower the certainty threshold or use \
                         method='sys' or 'brewer'."
                    )));
                }
            }
        }
    }

    assemble_pps_output(frame, positions, p0, &hits, &cert_mask, n_rem)
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
    n_rem: usize,
) -> Result<PpsDraw> {
    // Non-certainties come from n_rem draws over the rest: pi_i = n_rem * p0_i / total_rem.
    let total_rem: f64 = p0
        .iter()
        .zip(cert_mask.iter())
        .filter(|&(_, &c)| !c)
        .map(|(&p, _)| p)
        .sum();

    let n_sel = hits.iter().filter(|&&h| h > 0).count();
    let mut sel = Vec::with_capacity(n_sel);
    let mut ohits = Vec::with_capacity(n_sel);
    let mut probs = Vec::with_capacity(n_sel);
    let mut cert = Vec::with_capacity(n_sel);

    for (i, &h) in hits.iter().enumerate() {
        if h > 0 {
            sel.push(frame[positions[i]]);
            ohits.push(h);
            probs.push(if cert_mask[i] {
                1.0
            } else if total_rem > 0.0 {
                (n_rem as f64 * p0[i] / total_rem).min(1.0)
            } else {
                0.0
            });
            cert.push(cert_mask[i]);
        }
    }
    Ok((sel, ohits, probs, cert))
}

#[cfg(test)]
mod tests {
    use super::*;

    // MOS [100, 1, 1], n=2: unit 0 is certain; the others have pi = 1 * (1/102) / (2/102) = 0.5.
    #[test]
    fn test_pi_renormalised_after_certainty_extraction() {
        for method in [PpsMethod::Sys, PpsMethod::Brewer, PpsMethod::Rs] {
            let frame = vec![10, 20, 30];
            let mos = vec![100.0, 1.0, 1.0];
            let (sel, _hits, probs, cert) =
                select_pps(&frame, PpsN::Scalar(2), &mos, None, method, 1.0, 7).unwrap();
            assert_eq!(sel.len(), 2, "{method:?}: expected 2 selected units");
            for (i, &s) in sel.iter().enumerate() {
                if s == 10 {
                    assert!(cert[i], "{method:?}: unit 10 should be a certainty");
                    assert!((probs[i] - 1.0).abs() < 1e-12);
                } else {
                    assert!(!cert[i]);
                    assert!(
                        (probs[i] - 0.5).abs() < 1e-9,
                        "{method:?}: expected pi=0.5 for non-certainty unit, got {}",
                        probs[i]
                    );
                }
            }
        }
    }

    #[test]
    fn test_certainty_extraction_capped_at_n_equal_units() {
        let p0 = vec![0.1; 10];
        let (mask, n_rem) = extract_certainty(&p0, 1, 0.1);
        let n_cert = mask.iter().filter(|&&c| c).count();
        assert!(n_cert <= 1, "extracted {n_cert} certainties for n=1");
        assert_eq!(n_cert + n_rem, 1);
    }

    #[test]
    fn test_certainty_extraction_capped_at_n_two_units() {
        let p0 = vec![0.5, 0.5];
        let (mask, n_rem) = extract_certainty(&p0, 1, 0.5);
        let n_cert = mask.iter().filter(|&&c| c).count();
        assert_eq!(n_cert, 1, "extracted {n_cert} certainties for n=1");
        assert_eq!(n_rem, 0);
    }

    #[test]
    fn test_n_exceeding_positive_mos_units_errors() {
        for method in [PpsMethod::Sys, PpsMethod::Brewer, PpsMethod::Rs] {
            let frame = vec![1, 2, 3, 4];
            let mos = vec![5.0, 5.0, 0.0, 0.0];
            let res = select_pps(&frame, PpsN::Scalar(3), &mos, None, method, 1.0, 1);
            assert!(
                matches!(&res, Err(SamplingError::InvalidInput(m)) if m.contains("positive measure of size")),
                "{method:?}: got {res:?}"
            );
        }
    }

    #[test]
    fn test_mos_nan_rejected() {
        let frame = vec![1, 2, 3];
        let mos = vec![1.0, f64::NAN, 1.0];
        let res = select_pps(&frame, PpsN::Scalar(2), &mos, None, PpsMethod::Sys, 1.0, 0);
        assert!(
            matches!(&res, Err(SamplingError::InvalidInput(m)) if m.contains("must be finite")),
            "got {res:?}"
        );
    }

    #[test]
    fn test_mos_length_mismatch_rejected() {
        let frame = vec![1, 2, 3];
        let mos = vec![1.0];
        let res = select_pps(&frame, PpsN::Scalar(2), &mos, None, PpsMethod::Sys, 1.0, 0);
        assert!(
            matches!(&res, Err(SamplingError::InvalidInput(m)) if m.contains("mos length (1)")),
            "got {res:?}"
        );
    }

    #[test]
    fn test_stratum_length_mismatch_rejected() {
        let frame = vec![1, 2, 3];
        let mos = vec![1.0, 1.0, 1.0];
        let stratum = vec![0, 0, 1, 1];
        let res = select_pps(
            &frame,
            PpsN::Scalar(1),
            &mos,
            Some(&stratum),
            PpsMethod::Sys,
            1.0,
            0,
        );
        assert!(
            matches!(&res, Err(SamplingError::InvalidInput(m)) if m.contains("stratum length (4)")),
            "got {res:?}"
        );
    }
}

#[cfg(test)]
mod sampford_tests {
    use super::*;

    #[test]
    fn murphy_requires_n_2() {
        let frame = vec![1, 2, 3, 4];
        let mos = vec![1.0, 2.0, 3.0, 4.0];
        let err = select_pps(
            &frame,
            PpsN::Scalar(3),
            &mos,
            None,
            PpsMethod::Murphy,
            1.0,
            1,
        )
        .unwrap_err();
        assert!(
            matches!(&err, SamplingError::InvalidInput(m) if m.contains("exactly 2")),
            "got {err:?}"
        );
        let (sel, ..) = select_pps(
            &frame,
            PpsN::Scalar(2),
            &mos,
            None,
            PpsMethod::Murphy,
            1.0,
            1,
        )
        .unwrap();
        assert_eq!(sel.len(), 2);
    }

    #[test]
    fn wr_draws_n_even_beyond_population() {
        let frame = vec![1, 2, 3];
        let mos = vec![1.0, 1.0, 2.0];
        let (_sel, hits, _probs, _cert) =
            select_pps(&frame, PpsN::Scalar(7), &mos, None, PpsMethod::Wr, 1.0, 3).unwrap();
        assert_eq!(
            hits.iter().sum::<i64>(),
            7,
            "WR must perform exactly n draws"
        );
        assert!(hits.iter().any(|&h| h > 1), "n > N forces repeats");
    }

    #[test]
    fn sampford_degenerate_np_close_to_1_errors() {
        // n*p = 1 - 1e-9 for unit 0 leaves ~1e-8 odds of distinct lambda draws per attempt.
        let frame = vec![1, 2, 3, 4, 5];
        let small = 0.5 + 0.25e-9;
        let mos = vec![1.0 - 1e-9, small, small, small, small];
        for seed in 0..5 {
            let res = select_pps(
                &frame,
                PpsN::Scalar(3),
                &mos,
                None,
                PpsMethod::Rs,
                1.0,
                seed,
            );
            assert!(
                matches!(&res, Err(SamplingError::InvalidInput(m)) if m.contains("did not accept")),
                "seed {seed}: got {res:?}"
            );
        }
    }

    /// Monte Carlo inclusion frequencies match pi_i = n * p_i.
    #[test]
    fn sampford_inclusion_probabilities_quick() {
        run_sampford_mc(20_000, 0.03);
    }

    /// Tight version; run with `cargo test -- --ignored`.
    #[test]
    #[ignore]
    fn sampford_inclusion_probabilities_full() {
        run_sampford_mc(200_000, 0.01);
    }

    fn run_sampford_mc(reps: u64, tol: f64) {
        // Largest n*p is 3 * 15/60 = 0.75: skewed, yet no certainties.
        let frame: Vec<i64> = (0..8).collect();
        let mos = vec![15.0, 11.0, 9.0, 8.0, 6.0, 5.0, 4.0, 2.0];
        let total: f64 = mos.iter().sum();
        let n = 3usize;
        let mut counts = vec![0u64; frame.len()];

        for seed in 0..reps {
            let (sel, _hits, _probs, _cert) = select_pps(
                &frame,
                PpsN::Scalar(n),
                &mos,
                None,
                PpsMethod::Rs,
                1.0,
                seed,
            )
            .unwrap();
            // Frame ids equal positions here.
            for &s in &sel {
                counts[s as usize] += 1;
            }
        }

        for (i, &c) in counts.iter().enumerate() {
            let expected = n as f64 * mos[i] / total;
            let observed = c as f64 / reps as f64;
            assert!(
                (observed - expected).abs() < tol,
                "unit {i}: observed inclusion {observed:.4} vs n*p {expected:.4} (tol {tol})"
            );
        }
    }
}

#[cfg(test)]
mod brewer_tests {
    use super::*;

    // Total 118; the largest n*p at n = 4 is 4 * 25 / 118 = 0.85.
    const MOS: [f64; 10] = [2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0];

    /// Exact inclusion probabilities: sums the probability of every ordered draw sequence.
    fn enumerate_inclusion(pi: &[f64], available: &[bool], n: usize) -> Vec<f64> {
        fn walk(
            pi: &[f64],
            available: &mut [bool],
            n: usize,
            r: usize,
            path_prob: f64,
            incl: &mut [f64],
            leaf_total: &mut f64,
        ) {
            if r > n {
                *leaf_total += path_prob;
                return;
            }
            let mut w = vec![0.0; pi.len()];
            let total = brewer_weights(pi, available, n, r, &mut w);
            for k in 0..pi.len() {
                if !available[k] || pi[k] == 0.0 {
                    continue;
                }
                assert!(w[k] > 0.0, "draw {r}: unit {k} weight {}", w[k]);
                let p = path_prob * w[k] / total;
                incl[k] += p;
                available[k] = false;
                walk(pi, available, n, r + 1, p, incl, leaf_total);
                available[k] = true;
            }
        }
        let mut incl = vec![0.0; pi.len()];
        let mut leaf_total = 0.0;
        walk(
            pi,
            &mut available.to_vec(),
            n,
            1,
            1.0,
            &mut incl,
            &mut leaf_total,
        );
        assert!(
            (leaf_total - 1.0).abs() < 1e-12,
            "sequences sum to {leaf_total}"
        );
        incl
    }

    #[test]
    fn brewer_inclusion_equals_n_p_exactly() {
        let total: f64 = MOS.iter().sum();
        for n in [2usize, 3, 4] {
            let pi: Vec<f64> = MOS.iter().map(|&m| n as f64 * m / total).collect();
            let incl = enumerate_inclusion(&pi, &[true; 10], n);
            for k in 0..MOS.len() {
                assert!(
                    (incl[k] - pi[k]).abs() < 1e-12,
                    "n={n} unit {k}: inclusion {} vs n*p {}",
                    incl[k],
                    pi[k]
                );
            }
        }
    }

    // R sampling 2.11: UPbrewer's draw expression on inclusionprobabilities(MOS, 3), after
    // drawing nothing, unit 10, then units 10 and 1 (R's 1-based labels).
    #[test]
    fn brewer_draw_probabilities_match_r_upbrewer() {
        let r_draws: [(&[usize], [f64; 10]); 3] = [
            (
                &[],
                [
                    0.011195647477404159,
                    0.017106921011028817,
                    0.029647663724896849,
                    0.050598083977116383,
                    0.066331579098099863,
                    0.083840239813966891,
                    0.11438916151924987,
                    0.15200986876647885,
                    0.18264480783589185,
                    0.2922360267758663,
                ],
            ),
            (
                &[9],
                [
                    0.017853472870139334,
                    0.027094793044679058,
                    0.046282404183414276,
                    0.07710091180967836,
                    0.099265094153032724,
                    0.12298679973323885,
                    0.16213868218800007,
                    0.20677605218307696,
                    0.24050178983474027,
                    0.0,
                ],
            ),
            (
                &[9, 0],
                [
                    0.0,
                    0.032967032967032968,
                    0.054945054945054944,
                    0.087912087912087919,
                    0.10989010989010989,
                    0.13186813186813187,
                    0.16483516483516483,
                    0.19780219780219779,
                    0.21978021978021978,
                    0.0,
                ],
            ),
        ];
        let total: f64 = MOS.iter().sum();
        let pi: Vec<f64> = MOS.iter().map(|&m| 3.0 * m / total).collect();
        for (drawn, expected) in r_draws {
            let mut available = [true; 10];
            for &k in drawn {
                available[k] = false;
            }
            let mut w = vec![0.0; 10];
            let w_total = brewer_weights(&pi, &available, 3, drawn.len() + 1, &mut w);
            for k in 0..10 {
                assert!(
                    (w[k] / w_total - expected[k]).abs() < 1e-14,
                    "drawn {drawn:?} unit {k}: {} vs R {}",
                    w[k] / w_total,
                    expected[k]
                );
            }
        }
    }

    // Adding MOS 100 (total 218) at n = 4 makes it certain (4 * 100 / 218 > 1); 3 draws remain.
    #[test]
    fn brewer_inclusion_exact_after_certainty() {
        let rest: f64 = MOS.iter().sum();
        let mut pi: Vec<f64> = MOS.iter().map(|&m| 3.0 * m / rest).collect();
        pi.push(0.0);
        let mut available = [true; 11];
        available[10] = false;
        let incl = enumerate_inclusion(&pi, &available, 3);
        for k in 0..MOS.len() {
            assert!(
                (incl[k] - pi[k]).abs() < 1e-12,
                "unit {k}: inclusion {} vs pi {}",
                incl[k],
                pi[k]
            );
        }

        let frame: Vec<i64> = (0..11).collect();
        let mut mos = MOS.to_vec();
        mos.push(100.0);
        for seed in 0..20 {
            let (sel, _hits, probs, cert) = select_pps(
                &frame,
                PpsN::Scalar(4),
                &mos,
                None,
                PpsMethod::Brewer,
                1.0,
                seed,
            )
            .unwrap();
            assert_eq!(sel.len(), 4);
            for (j, &s) in sel.iter().enumerate() {
                let expected = if s == 10 { 1.0 } else { pi[s as usize] };
                assert_eq!(cert[j], s == 10, "seed {seed}: unit {s}");
                assert!((probs[j] - expected).abs() < 1e-12, "seed {seed}: unit {s}");
            }
        }
    }
}
