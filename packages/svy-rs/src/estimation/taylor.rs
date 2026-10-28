//! Taylor-linearisation kernels: point estimates, scores, design indexing and variance.

use crate::estimation::error::{EResult, EstimationError};
use polars::prelude::*;

use crate::estimation::calib_sweep::CalibSweep;
use rustc_hash::{FxHashMap, FxHashSet};

// ============================================================================
// Enums & Config
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum SingletonMethod {
    None,   // Default: Treat n=1 as 0 variance contribution
    Center, // Grand mean centering: (z_i - z_bar)^2
    Scale,  // n=1 contributes 0; the total is scaled by nstrat/nokstrat
}

impl SingletonMethod {
    /// Parse `"<method>[:domains]"`; `adjust`/`average` are R's names for `center`/`scale`, and
    /// `:domains` is R's `survey.adjust.domain.lonely`. `"none:domains"` drops without scaling.
    pub(crate) fn parse_rule(s: Option<&str>) -> EResult<(Self, bool)> {
        let Some(s) = s else {
            return Ok((SingletonMethod::None, false));
        };
        let invalid = || EstimationError::InvalidChoice {
            what: "singleton method",
            got: s.to_string(),
            expected: "'none', 'center' or 'scale', optionally followed by ':domains'",
        };
        let (name, domains) = match s.split_once(':') {
            Some((name, flag)) if flag.eq_ignore_ascii_case("domains") => (name, true),
            Some(_) => return Err(invalid()),
            None => (s, false),
        };
        let method = match name.to_ascii_lowercase().as_str() {
            "none" => SingletonMethod::None,
            "center" | "adjust" => SingletonMethod::Center,
            "scale" | "average" => SingletonMethod::Scale,
            _ => return Err(invalid()),
        };
        Ok((method, domains))
    }
}

/// A singleton method and whether it also applies to strata with one PSU in the domain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SingletonRule {
    pub method: SingletonMethod,
    pub domains: bool,
}

impl SingletonRule {
    pub(crate) const NONE: Self = Self {
        method: SingletonMethod::None,
        domains: false,
    };

    pub(crate) fn parse(s: Option<&str>) -> EResult<Self> {
        let (method, domains) = SingletonMethod::parse_rule(s)?;
        Ok(Self { method, domains })
    }
}

/// Quantile interpolation, as R's `approxfun`; named to avoid polars' `QuantileMethod`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) enum SvyQuantileMethod {
    /// Return lower value at discontinuity (R: method="constant", f=0)
    Lower,
    /// Return higher value at discontinuity (R: method="constant", f=1)
    #[default]
    Higher,
    /// Return midpoint of lower and higher
    Middle,
    Nearest,
    /// Linear interpolation (R: method="linear", default)
    Linear,
}

impl SvyQuantileMethod {
    pub(crate) fn parse(s: &str) -> EResult<Self> {
        match s.to_lowercase().as_str() {
            "lower" => Ok(SvyQuantileMethod::Lower),
            "higher" => Ok(SvyQuantileMethod::Higher),
            "middle" => Ok(SvyQuantileMethod::Middle),
            "nearest" => Ok(SvyQuantileMethod::Nearest),
            "linear" => Ok(SvyQuantileMethod::Linear),
            _ => Err(EstimationError::InvalidChoice {
                what: "quantile method",
                got: s.to_string(),
                expected: "'lower', 'higher', 'middle', 'nearest' or 'linear'",
            }),
        }
    }
}

// ============================================================================
// Point Estimates
// ============================================================================

/// A mask as in/out flags (null = out) read from its bits; `iter()` costs several times more.
pub(crate) fn mask_flags(mask: &BooleanChunked) -> Vec<bool> {
    let mut out = Vec::with_capacity(mask.len());
    for arr in mask.downcast_iter() {
        match arr.validity() {
            Some(valid) if valid.unset_bits() > 0 => {
                out.extend(arr.values().iter().zip(valid.iter()).map(|(v, ok)| v && ok))
            }
            _ => out.extend(arr.values().iter()),
        }
    }
    out
}

/// Contiguous, null-free slices of two columns, or `None` for the `Option` fallback.
#[inline]
fn cont_pair<'a>(a: &'a Float64Chunked, b: &'a Float64Chunked) -> Option<(&'a [f64], &'a [f64])> {
    if a.null_count() == 0
        && b.null_count() == 0
        && let (Ok(sa), Ok(sb)) = (a.cont_slice(), b.cont_slice())
    {
        return Some((sa, sb));
    }
    None
}

pub fn point_estimate_mean(y: &Float64Chunked, weights: &Float64Chunked) -> EResult<f64> {
    // Sequential summation, as in the fallback, so the fast path is bit-identical.
    let (sum_wy, sum_w) = if let Some((ys, ws)) = cont_pair(y, weights) {
        let mut sum_wy = 0.0f64;
        let mut sum_w = 0.0f64;
        for i in 0..ys.len() {
            sum_wy += ys[i] * ws[i];
            sum_w += ws[i];
        }
        (sum_wy, sum_w)
    } else {
        let sum_wy: f64 = y
            .iter()
            .zip(weights.iter())
            .filter_map(|(yi, wi)| Some(yi? * wi?))
            .sum();
        let sum_w: f64 = y
            .iter()
            .zip(weights.iter())
            .filter_map(|(yi, wi)| {
                yi?;
                wi
            })
            .sum();
        (sum_wy, sum_w)
    };

    if sum_w == 0.0 {
        return Err(EstimationError::ZeroWeight { in_domain: false });
    }
    Ok(sum_wy / sum_w)
}

pub(crate) fn point_estimate_mean_domain(
    y: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
) -> EResult<f64> {
    // Fast path: slices plus one pass over the mask; same rows and order, so bit-identical.
    let (sum_wy, sum_w) = if let Some((ys, ws)) = cont_pair(y, weights) {
        let mut sum_wy = 0.0f64;
        let mut sum_w = 0.0f64;
        for (i, m) in mask_flags(domain_mask).into_iter().enumerate() {
            if m {
                sum_wy += ys[i] * ws[i];
                sum_w += ws[i];
            }
        }
        (sum_wy, sum_w)
    } else {
        let sum_wy: f64 = y
            .iter()
            .zip(weights.iter())
            .zip(domain_mask.iter())
            .filter_map(|((yi, wi), m)| if m? { Some(yi? * wi?) } else { None })
            .sum();
        let sum_w: f64 = weights
            .iter()
            .zip(domain_mask.iter())
            .filter_map(|(w, m)| if m? { w } else { None })
            .sum();
        (sum_wy, sum_w)
    };

    if sum_w == 0.0 {
        return Err(EstimationError::ZeroWeight { in_domain: true });
    }
    Ok(sum_wy / sum_w)
}

pub(crate) fn point_estimate_total(y: &Float64Chunked, weights: &Float64Chunked) -> EResult<f64> {
    if let Some((ys, ws)) = cont_pair(y, weights) {
        let mut sum_wy = 0.0f64;
        for i in 0..ys.len() {
            sum_wy += ys[i] * ws[i];
        }
        return Ok(sum_wy);
    }
    Ok(y.iter()
        .zip(weights.iter())
        .filter_map(|(yi, wi)| Some(yi? * wi?))
        .sum())
}

pub(crate) fn point_estimate_total_domain(
    y: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
) -> EResult<f64> {
    if let Some((ys, ws)) = cont_pair(y, weights) {
        let mut sum_wy = 0.0f64;
        for (i, m) in mask_flags(domain_mask).into_iter().enumerate() {
            if m {
                sum_wy += ys[i] * ws[i];
            }
        }
        return Ok(sum_wy);
    }
    Ok(y.iter()
        .zip(weights.iter())
        .zip(domain_mask.iter())
        .filter_map(|((yi, wi), m)| if m? { Some(yi? * wi?) } else { None })
        .sum())
}

pub(crate) fn point_estimate_ratio(
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked,
) -> EResult<f64> {
    // Fast path only when y, x and w are all contiguous and null-free.
    let fast = match (cont_pair(y, weights), x.null_count() == 0) {
        (Some((ys, ws)), true) => x.cont_slice().ok().map(|xs| (ys, ws, xs)),
        _ => None,
    };
    let (sum_wy, sum_wx) = if let Some((ys, ws, xs)) = fast {
        let mut sum_wy = 0.0f64;
        let mut sum_wx = 0.0f64;
        for i in 0..ys.len() {
            sum_wy += ys[i] * ws[i];
            sum_wx += xs[i] * ws[i];
        }
        (sum_wy, sum_wx)
    } else {
        let sum_wy: f64 = y
            .iter()
            .zip(weights.iter())
            .filter_map(|(yi, wi)| Some(yi? * wi?))
            .sum();
        let sum_wx: f64 = x
            .iter()
            .zip(weights.iter())
            .filter_map(|(xi, wi)| Some(xi? * wi?))
            .sum();
        (sum_wy, sum_wx)
    };

    if sum_wx == 0.0 {
        return Err(EstimationError::ZeroDenominator { in_domain: false });
    }
    Ok(sum_wy / sum_wx)
}

pub(crate) fn point_estimate_ratio_domain(
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
) -> EResult<f64> {
    let fast = match (cont_pair(y, weights), x.null_count() == 0) {
        (Some((ys, ws)), true) => x.cont_slice().ok().map(|xs| (ys, ws, xs)),
        _ => None,
    };
    let (sum_wy, sum_wx) = if let Some((ys, ws, xs)) = fast {
        let mut sum_wy = 0.0f64;
        let mut sum_wx = 0.0f64;
        for (i, m) in mask_flags(domain_mask).into_iter().enumerate() {
            if m {
                sum_wy += ys[i] * ws[i];
                sum_wx += xs[i] * ws[i];
            }
        }
        (sum_wy, sum_wx)
    } else {
        let sum_wy: f64 = y
            .iter()
            .zip(weights.iter())
            .zip(domain_mask.iter())
            .filter_map(|((yi, wi), m)| if m? { Some(yi? * wi?) } else { None })
            .sum();
        let sum_wx: f64 = x
            .iter()
            .zip(weights.iter())
            .zip(domain_mask.iter())
            .filter_map(|((xi, wi), m)| if m? { Some(xi? * wi?) } else { None })
            .sum();
        (sum_wy, sum_wx)
    };

    if sum_wx == 0.0 {
        return Err(EstimationError::ZeroDenominator { in_domain: true });
    }
    Ok(sum_wy / sum_wx)
}

// ============================================================================
// Weighted Quantile Functions
// ============================================================================

/// Weighted quantile of sorted `y` at `p`, with `cdf` = cumsum(w)/sum(w). Lower/Higher are R's
/// `method="constant"` with f=0/1; Linear is `method="linear"`.
pub(crate) fn weighted_quantile(
    y_sorted: &[f64],
    cdf: &[f64],
    p: f64,
    method: SvyQuantileMethod,
) -> f64 {
    let n = y_sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return y_sorted[0];
    }
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }

    // At either boundary the quantile is that value: a two-point bracket would break Higher/Middle
    // and let Linear extrapolate past the data.
    let (left, right) = if p <= cdf[0] {
        (0, 0)
    } else if p >= cdf[n - 1] {
        (n - 1, n - 1)
    } else {
        let idx = cdf.partition_point(|&x| x < p);
        ((idx.saturating_sub(1)), idx.min(n - 1))
    };

    match method {
        SvyQuantileMethod::Lower => y_sorted[left],
        SvyQuantileMethod::Higher => y_sorted[right],
        SvyQuantileMethod::Middle => (y_sorted[left] + y_sorted[right]) / 2.0,
        SvyQuantileMethod::Nearest => {
            let dl = (p - cdf[left]).abs();
            let dr = (cdf[right] - p).abs();
            if dl <= dr {
                y_sorted[left]
            } else {
                y_sorted[right]
            }
        }
        SvyQuantileMethod::Linear => {
            let denom = cdf[right] - cdf[left];
            if denom <= 0.0 {
                return y_sorted[left];
            }
            if (p - cdf[left]).abs() < 1e-12 && right != left {
                return (y_sorted[left] + y_sorted[right]) / 2.0;
            }
            let w = (p - cdf[left]) / denom;
            (1.0 - w) * y_sorted[left] + w * y_sorted[right]
        }
    }
}

/// Sort pairs by value with the normalised CDF; `None` without positive weight.
fn sorted_weighted_cdf(mut pairs: Vec<(f64, f64)>) -> Option<(Vec<f64>, Vec<f64>, f64)> {
    if pairs.is_empty() {
        return None;
    }

    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let (y_sorted, w_sorted): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();

    let total_w: f64 = w_sorted.iter().sum();
    if total_w <= 0.0 {
        return None;
    }

    let mut cdf = Vec::with_capacity(w_sorted.len());
    let mut cumsum = 0.0;
    for w in &w_sorted {
        cumsum += w;
        cdf.push(cumsum / total_w);
    }

    Some((y_sorted, cdf, total_w))
}

/// Collect the non-null `(value, weight)` pairs, optionally restricted to a domain.
fn quantile_pairs(
    y: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: Option<&BooleanChunked>,
) -> Vec<(f64, f64)> {
    match domain_mask {
        None => y
            .iter()
            .zip(weights.iter())
            .filter_map(|(yi, wi)| Some((yi?, wi?)))
            .collect(),
        Some(mask) => y
            .iter()
            .zip(weights.iter())
            .zip(mask.iter())
            .filter_map(|((yi, wi), m)| if m? { Some((yi?, wi?)) } else { None })
            .collect(),
    }
}

// ============================================================================
// Linearization Scores
// ============================================================================

pub fn scores_mean(y: &Float64Chunked, weights: &Float64Chunked) -> EResult<Float64Chunked> {
    // Fast path skips the fallback's `Vec<Option<_>>`; same summation order, so bit-identical.
    if let Some((ys, ws)) = cont_pair(y, weights) {
        let mut sum_wy = 0.0f64;
        let mut sum_w = 0.0f64;
        for i in 0..ys.len() {
            sum_wy += ys[i] * ws[i];
            sum_w += ws[i];
        }
        if sum_w == 0.0 {
            return Err(EstimationError::ZeroWeight { in_domain: false });
        }
        let est = sum_wy / sum_w;
        let scores: Vec<f64> = (0..ys.len())
            .map(|i| (ws[i] / sum_w) * (ys[i] - est))
            .collect();
        return Ok(Float64Chunked::from_slice("scores".into(), &scores));
    }

    // Fallback keeps nulls in place in the output.
    let n = y.len();
    let mut sum_wy = 0.0f64;
    let mut sum_w = 0.0f64;
    let mut pairs: Vec<Option<(f64, f64)>> = Vec::with_capacity(n);

    for (yi, wi) in y.iter().zip(weights.iter()) {
        match (yi, wi) {
            (Some(yv), Some(wv)) => {
                sum_wy += yv * wv;
                sum_w += wv;
                pairs.push(Some((yv, wv)));
            }
            _ => pairs.push(None),
        }
    }

    if sum_w == 0.0 {
        return Err(EstimationError::ZeroWeight { in_domain: false });
    }
    let est = sum_wy / sum_w;

    let scores: Vec<Option<f64>> = pairs
        .iter()
        .map(|p| p.map(|(yv, wv)| (wv / sum_w) * (yv - est)))
        .collect();

    Ok(Float64Chunked::from_slice_options("scores".into(), &scores))
}

/// `scores_mean` as the variance's array, nulls as 0.0, without the `Float64Chunked` round-trip.
pub(crate) fn scores_mean_arr(y: &Float64Chunked, weights: &Float64Chunked) -> EResult<Vec<f64>> {
    if let Some((ys, ws)) = cont_pair(y, weights) {
        let mut sum_wy = 0.0f64;
        let mut sum_w = 0.0f64;
        for i in 0..ys.len() {
            sum_wy += ys[i] * ws[i];
            sum_w += ws[i];
        }
        if sum_w == 0.0 {
            return Err(EstimationError::ZeroWeight { in_domain: false });
        }
        let est = sum_wy / sum_w;
        return Ok((0..ys.len())
            .map(|i| (ws[i] / sum_w) * (ys[i] - est))
            .collect());
    }
    // Nulls or chunking: the reference implementation, flattened as `taylor_variance` would.
    Ok(scores_mean(y, weights)?
        .iter()
        .map(|s| s.unwrap_or(0.0))
        .collect())
}

pub(crate) fn scores_mean_domain(
    y: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
) -> EResult<Float64Chunked> {
    // Fast path: an in-domain flag vector, then two branch-free slice passes; bit-identical.
    if let Some((ys, ws)) = cont_pair(y, weights) {
        let n = ys.len();
        let in_domain = mask_flags(domain_mask);
        let mut sum_wy = 0.0f64;
        let mut sum_w = 0.0f64;
        for i in 0..n {
            if in_domain[i] {
                sum_wy += ys[i] * ws[i];
                sum_w += ws[i];
            }
        }
        if sum_w == 0.0 {
            return Ok(Float64Chunked::from_slice("scores".into(), &vec![0.0; n]));
        }
        let est = sum_wy / sum_w;
        let scores: Vec<f64> = (0..n)
            .map(|i| {
                if in_domain[i] {
                    (ws[i] / sum_w) * (ys[i] - est)
                } else {
                    0.0
                }
            })
            .collect();
        return Ok(Float64Chunked::from_slice("scores".into(), &scores));
    }

    let n = y.len();
    let mut sum_wy = 0.0f64;
    let mut sum_w = 0.0f64;
    let mut triples: Vec<Option<(f64, f64)>> = Vec::with_capacity(n); // (yv,wv) only in-domain

    for ((yi, wi), mi) in y.iter().zip(weights.iter()).zip(domain_mask.iter()) {
        match (yi, wi, mi) {
            (Some(yv), Some(wv), Some(true)) => {
                sum_wy += yv * wv;
                sum_w += wv;
                triples.push(Some((yv, wv)));
            }
            _ => triples.push(None),
        }
    }

    if sum_w == 0.0 {
        let zeros = vec![Some(0.0); n];
        return Ok(Float64Chunked::from_slice_options("scores".into(), &zeros));
    }
    let est = sum_wy / sum_w;

    let scores: Vec<Option<f64>> = triples
        .iter()
        .map(|p| {
            Some(match p {
                Some((yv, wv)) => (wv / sum_w) * (yv - est),
                None => 0.0,
            })
        })
        .collect();

    Ok(Float64Chunked::from_slice_options("scores".into(), &scores))
}

/// [`scores_total`] for the variance, nulls as 0.0, without the `Float64Chunked` round-trip.
pub(crate) fn scores_total_arr(y: &Float64Chunked, weights: &Float64Chunked) -> EResult<Vec<f64>> {
    if let Some((ys, ws)) = cont_pair(y, weights) {
        return Ok(ys.iter().zip(ws).map(|(y, w)| w * y).collect());
    }
    Ok(scores_total(y, weights)?
        .iter()
        .map(|s| s.unwrap_or(0.0))
        .collect())
}

pub(crate) fn scores_total(
    y: &Float64Chunked,
    weights: &Float64Chunked,
) -> EResult<Float64Chunked> {
    let scores: Vec<Option<f64>> = y
        .iter()
        .zip(weights.iter())
        .map(|(yi, wi)| match (yi, wi) {
            (Some(y_val), Some(w_val)) => Some(w_val * y_val),
            _ => None,
        })
        .collect();
    Ok(Float64Chunked::from_slice_options("scores".into(), &scores))
}

pub(crate) fn scores_total_domain(
    y: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
) -> EResult<Float64Chunked> {
    if let Some((ys, ws)) = cont_pair(y, weights) {
        let scores: Vec<f64> = mask_flags(domain_mask)
            .into_iter()
            .enumerate()
            .map(|(i, m)| if m { ws[i] * ys[i] } else { 0.0 })
            .collect();
        return Ok(Float64Chunked::from_slice("scores".into(), &scores));
    }
    let scores: Vec<Option<f64>> = y
        .iter()
        .zip(weights.iter())
        .zip(domain_mask.iter())
        .map(|((yi, wi), m)| match (yi, wi, m) {
            (Some(y_val), Some(w_val), Some(true)) => Some(w_val * y_val),
            _ => Some(0.0),
        })
        .collect();
    Ok(Float64Chunked::from_slice_options("scores".into(), &scores))
}

pub(crate) fn scores_ratio(
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked,
) -> EResult<Float64Chunked> {
    let n = y.len();
    let mut sum_wy = 0.0f64;
    let mut sum_wx = 0.0f64;
    let mut triples: Vec<Option<(f64, f64, f64)>> = Vec::with_capacity(n);

    for ((yi, xi), wi) in y.iter().zip(x.iter()).zip(weights.iter()) {
        match (yi, xi, wi) {
            (Some(yv), Some(xv), Some(wv)) => {
                sum_wy += yv * wv;
                sum_wx += xv * wv;
                triples.push(Some((yv, xv, wv)));
            }
            _ => triples.push(None),
        }
    }

    if sum_wx == 0.0 {
        return Err(EstimationError::ZeroDenominator { in_domain: false });
    }
    let r_hat = sum_wy / sum_wx;

    let scores: Vec<Option<f64>> = triples
        .iter()
        .map(|t| t.map(|(yv, xv, wv)| (wv / sum_wx) * (yv - r_hat * xv)))
        .collect();
    Ok(Float64Chunked::from_slice_options("scores".into(), &scores))
}

pub(crate) fn scores_ratio_domain(
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
) -> EResult<Float64Chunked> {
    let fast = match (cont_pair(y, weights), x.null_count() == 0) {
        (Some((ys, ws)), true) => x.cont_slice().ok().map(|xs| (ys, ws, xs)),
        _ => None,
    };
    if let Some((ys, ws, xs)) = fast {
        let n = ys.len();
        let in_domain = mask_flags(domain_mask);
        let mut sum_wy = 0.0f64;
        let mut sum_wx = 0.0f64;
        for i in 0..n {
            if in_domain[i] {
                sum_wy += ys[i] * ws[i];
                sum_wx += xs[i] * ws[i];
            }
        }
        if sum_wx == 0.0 {
            return Ok(Float64Chunked::from_slice("scores".into(), &vec![0.0; n]));
        }
        let r_hat = sum_wy / sum_wx;
        let scores: Vec<f64> = (0..n)
            .map(|i| {
                if in_domain[i] {
                    (ws[i] / sum_wx) * (ys[i] - r_hat * xs[i])
                } else {
                    0.0
                }
            })
            .collect();
        return Ok(Float64Chunked::from_slice("scores".into(), &scores));
    }

    let n = y.len();
    let mut sum_wy = 0.0f64;
    let mut sum_wx = 0.0f64;
    let mut quads: Vec<Option<(f64, f64, f64)>> = Vec::with_capacity(n); // (yv,xv,wv)

    for (((yi, xi), wi), mi) in y
        .iter()
        .zip(x.iter())
        .zip(weights.iter())
        .zip(domain_mask.iter())
    {
        match (yi, xi, wi, mi) {
            (Some(yv), Some(xv), Some(wv), Some(true)) => {
                sum_wy += yv * wv;
                sum_wx += xv * wv;
                quads.push(Some((yv, xv, wv)));
            }
            _ => quads.push(None),
        }
    }

    if sum_wx == 0.0 {
        let zeros = vec![Some(0.0); n];
        return Ok(Float64Chunked::from_slice_options("scores".into(), &zeros));
    }
    let r_hat = sum_wy / sum_wx;

    let scores: Vec<Option<f64>> = quads
        .iter()
        .map(|q| {
            Some(match q {
                Some((yv, xv, wv)) => (wv / sum_wx) * (yv - r_hat * xv),
                None => 0.0,
            })
        })
        .collect();
    Ok(Float64Chunked::from_slice_options("scores".into(), &scores))
}

/// Quantile scores at `q`: `(w_i/sum_w)(u_i - u_bar)`, `u_i = I(y_i > q) - (1 - p)` in R's sign.
/// Centring on `u_bar` matters under Linear (F(q) != p exactly), as R's `svymean(U)` centres.
fn scores_quantile_arr(
    y: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: Option<&BooleanChunked>,
    q: f64,
    p: f64,
    sum_w: f64,
) -> Vec<f64> {
    let complement = 1.0 - p;

    // Weights are folded in after centring on the weighted mean of u.
    let residuals: Vec<(f64, f64)> = match domain_mask {
        None => y
            .iter()
            .zip(weights.iter())
            .map(|(yi, wi)| match (yi, wi) {
                (Some(y_val), Some(w_val)) => {
                    (w_val, if y_val > q { 1.0 } else { 0.0 } - complement)
                }
                _ => (0.0, 0.0),
            })
            .collect(),
        Some(mask) => y
            .iter()
            .zip(weights.iter())
            .zip(mask.iter())
            .map(|((yi, wi), m)| match (yi, wi, m) {
                (Some(y_val), Some(w_val), Some(true)) => {
                    (w_val, if y_val > q { 1.0 } else { 0.0 } - complement)
                }
                _ => (0.0, 0.0), // Out of scope: contributes nothing, scores zero
            })
            .collect(),
    };

    let u_bar: f64 = residuals.iter().map(|(w, u)| w * u).sum::<f64>() / sum_w;

    residuals
        .iter()
        .map(|&(w, u)| {
            if w == 0.0 {
                0.0
            } else {
                (w / sum_w) * (u - u_bar)
            }
        })
        .collect()
}

// ============================================================================
// Variance Helpers (Indexing & Math)
// ============================================================================

#[doc(hidden)] // internal; exposed only so criterion benches can measure it
pub fn index_categorical(col: &StringChunked) -> (Vec<u32>, u32) {
    // FxHashMap is safe: codes follow first appearance (`next_idx`), not the map's order.
    let mut map: FxHashMap<&str, u32> = FxHashMap::default();
    let mut next_idx = 0u32;
    let indices: Vec<u32> = col
        .iter()
        .map(|opt| match opt {
            Some(s) => *map.entry(s).or_insert_with(|| {
                let i = next_idx;
                next_idx += 1;
                i
            }),
            None => u32::MAX,
        })
        .collect();
    (indices, next_idx)
}

/// Index PSUs on the (stratum, psu) pair, so labels reused across strata stay distinct, as R does.
#[doc(hidden)] // internal; exposed only so criterion benches can measure it
pub fn index_categorical_pair(a: &StringChunked, b: &StringChunked) -> (Vec<u32>, u32) {
    // FxHashMap: codes are first-appearance order, independent of the hash.
    let mut map: FxHashMap<(&str, &str), u32> = FxHashMap::default();
    let mut next_idx = 0u32;
    let indices: Vec<u32> = a
        .iter()
        .zip(b.iter())
        .map(|(oa, ob)| match (oa, ob) {
            (Some(sa), Some(sb)) => *map.entry((sa, sb)).or_insert_with(|| {
                let i = next_idx;
                next_idx += 1;
                i
            }),
            _ => u32::MAX,
        })
        .collect();
    (indices, next_idx)
}

/// Densify integer codes to 0-based FIRST-APPEARANCE codes (nulls → `u32::MAX`): exactly what
/// `index_categorical` gives the original labels, at ~10× less hashing cost.
fn densify_int_codes(col: &Column) -> EResult<(Vec<u32>, u32)> {
    if let Ok(ca) = col.u32()
        && let Some(codes) = densify_small_u32(ca)
    {
        return Ok(codes);
    }
    let s = col.cast(&DataType::Int64)?;
    let ca = s.i64()?;
    let mut map: FxHashMap<i64, u32> = FxHashMap::default();
    let mut next_idx = 0u32;
    let indices: Vec<u32> = ca
        .iter()
        .map(|opt| match opt {
            Some(v) => *map.entry(v).or_insert_with(|| {
                let i = next_idx;
                next_idx += 1;
                i
            }),
            None => u32::MAX,
        })
        .collect();
    Ok((indices, next_idx))
}

/// [`densify_int_codes`] by lookup table for small UInt32 codes; `None` when too sparse for one.
fn densify_small_u32(ca: &UInt32Chunked) -> Option<(Vec<u32>, u32)> {
    let max = ca.max()? as usize;
    if max > 4 * ca.len() + 1024 {
        return None;
    }
    let mut table = vec![u32::MAX; max + 1];
    let mut next_idx = 0u32;
    let mut code = |v: u32| {
        let t = &mut table[v as usize];
        if *t == u32::MAX {
            *t = next_idx;
            next_idx += 1;
        }
        *t
    };
    let indices: Vec<u32> = match ca.cont_slice() {
        Ok(vals) => vals.iter().map(|&v| code(v)).collect(),
        Err(_) => ca
            .iter()
            .map(|opt| opt.map_or(u32::MAX, &mut code))
            .collect(),
    };
    Some((indices, next_idx))
}

/// Dense first-appearance codes for one design column: strings are hashed, integers densified.
pub(crate) fn design_col_codes(col: &Column) -> EResult<(Vec<u32>, u32)> {
    let dt = col.dtype();
    if matches!(dt, DataType::String) {
        Ok(index_categorical(col.str()?))
    } else if dt.is_integer() {
        densify_int_codes(col)
    } else {
        Err(EstimationError::DesignDtype {
            col: "design",
            dtype: format!("{dt:?}"),
        })
    }
}

/// Nested PSU codes: strings key on the `(stratum, psu)` pair; integer PSU codes arrive
/// pair-nested from Python (`__svy_psu_code__`) and are densified directly.
pub(crate) fn design_pair_codes(strata: &Column, psu: &Column) -> EResult<(Vec<u32>, u32)> {
    let dt = psu.dtype();
    if matches!(dt, DataType::String) {
        Ok(index_categorical_pair(strata.str()?, psu.str()?))
    } else if dt.is_integer() {
        densify_int_codes(psu)
    } else {
        Err(EstimationError::DesignDtype {
            col: "psu",
            dtype: format!("{dt:?}"),
        })
    }
}

fn reindex_within_subset(raw: &[u32]) -> (Vec<u32>, u32) {
    // FxHashMap: codes assigned in first-appearance order, hash-independent.
    let mut map: FxHashMap<u32, u32> = FxHashMap::default();
    let mut next_idx = 0u32;
    let indices: Vec<u32> = raw
        .iter()
        .map(|&val| {
            if val == u32::MAX {
                u32::MAX
            } else {
                *map.entry(val).or_insert_with(|| {
                    let i = next_idx;
                    next_idx += 1;
                    i
                })
            }
        })
        .collect();
    (indices, next_idx)
}

fn build_stratum_psu_map(
    strata_indices: &[u32],
    n_strata: u32,
    psu_indices: &[u32],
) -> (Vec<Vec<u32>>, Vec<u32>) {
    // PSUs sorted per stratum: that order is the variance's summation order, so SEs are stable.
    // Dense codes allow an owner table; a code shared across strata (un-nested) falls back.
    let n_psus = psu_indices
        .iter()
        .filter(|&&p| p != u32::MAX)
        .max()
        .map_or(0, |&p| p as usize + 1);
    let mut owner = vec![u32::MAX; n_psus];
    let mut psu_per_stratum: Vec<Vec<u32>> = vec![Vec::new(); n_strata as usize];
    let mut shared = false;
    for (&stratum, &psu) in strata_indices.iter().zip(psu_indices.iter()) {
        if stratum != u32::MAX && psu != u32::MAX {
            let o = &mut owner[psu as usize];
            if *o == u32::MAX {
                *o = stratum;
                psu_per_stratum[stratum as usize].push(psu);
            } else if *o != stratum {
                shared = true;
                break;
            }
        }
    }
    if !shared {
        for v in &mut psu_per_stratum {
            v.sort_unstable();
        }
        let n_psus_per_stratum = psu_per_stratum.iter().map(|v| v.len() as u32).collect();
        return (psu_per_stratum, n_psus_per_stratum);
    }
    let mut stratum_psus: Vec<FxHashSet<u32>> = vec![FxHashSet::default(); n_strata as usize];
    for (&stratum, &psu) in strata_indices.iter().zip(psu_indices.iter()) {
        if stratum != u32::MAX && psu != u32::MAX {
            stratum_psus[stratum as usize].insert(psu);
        }
    }
    let psu_per_stratum: Vec<Vec<u32>> = stratum_psus
        .iter()
        .map(|s| {
            let mut v: Vec<u32> = s.iter().copied().collect();
            v.sort_unstable();
            v
        })
        .collect();
    let n_psus_per_stratum: Vec<u32> = psu_per_stratum.iter().map(|v| v.len() as u32).collect();
    (psu_per_stratum, n_psus_per_stratum)
}

fn variance_unstratified_optimized(
    scores: &[f64],
    psu_indices: Option<&[u32]>,
    n_psus: u32,
) -> f64 {
    let n = scores.len();
    if n == 0 {
        return 0.0;
    }

    match psu_indices {
        Some(psu_idx) => {
            if n_psus <= 1 {
                return 0.0;
            }
            let mut psu_totals = vec![0.0; n_psus as usize];
            for (score, &psu) in scores.iter().zip(psu_idx.iter()) {
                if psu != u32::MAX {
                    psu_totals[psu as usize] += score;
                }
            }
            let psu_mean = psu_totals.iter().sum::<f64>() / (n_psus as f64);
            let sum_sq_diff: f64 = psu_totals.iter().map(|&t| (t - psu_mean).powi(2)).sum();
            (n_psus as f64 / (n_psus as f64 - 1.0)) * sum_sq_diff
        }
        None => {
            if n <= 1 {
                return 0.0;
            }
            let mean: f64 = scores.iter().sum::<f64>() / (n as f64);
            let sum_sq_diff: f64 = scores.iter().map(|&s| (s - mean).powi(2)).sum();
            (n as f64 / (n as f64 - 1.0)) * sum_sq_diff
        }
    }
}

/// Strata a domain reaches, with their units and singletons: R's singleton rules see only strata
/// holding domain rows (`survey:::onestage` on the subset). `active` None takes every stratum.
struct CenterScope {
    present: Vec<bool>,
    lonely: Vec<bool>, // domain rule: one unit holds domain rows (R's `nsubset == 1 && nPSU > 1`)
    n_units: usize,
    n_strata: usize,
    n_singletons: usize,
}

impl CenterScope {
    /// R's `nstrat/nokstrat`: NaN when every stratum is a singleton.
    fn scale_factor(&self) -> f64 {
        if self.n_strata > self.n_singletons {
            self.n_strata as f64 / (self.n_strata - self.n_singletons) as f64
        } else {
            f64::NAN
        }
    }
}

/// Units of each stratum holding an `active` row; `unit_of_row` is the PSU
/// codes, or `None` when every row is a unit.
fn active_units(
    strata_indices: &[u32],
    n_strata: usize,
    unit_of_row: Option<&[u32]>,
    active: &[bool],
) -> Vec<u32> {
    let mut counts = vec![0u32; n_strata];
    match unit_of_row {
        Some(units) => {
            let n_codes = units
                .iter()
                .filter(|&&u| u != u32::MAX)
                .max()
                .map_or(0, |&u| u + 1);
            let mut seen = vec![false; n_codes as usize];
            for ((&act, &h), &u) in active.iter().zip(strata_indices).zip(units) {
                if act && h != u32::MAX && u != u32::MAX && !seen[u as usize] {
                    seen[u as usize] = true;
                    counts[h as usize] += 1;
                }
            }
        }
        None => {
            for (&act, &h) in active.iter().zip(strata_indices) {
                if act && h != u32::MAX {
                    counts[h as usize] += 1;
                }
            }
        }
    }
    counts
}

fn center_scope(
    strata_indices: &[u32],
    units_per_stratum: &[u32],
    unit_of_row: Option<&[u32]>,
    active: Option<&[bool]>,
    domain_rule: bool,
) -> CenterScope {
    let reached: Vec<u32> = match active {
        None => units_per_stratum.to_vec(),
        Some(a) => active_units(strata_indices, units_per_stratum.len(), unit_of_row, a),
    };
    let present: Vec<bool> = reached.iter().map(|&m| m > 0).collect();
    let lonely: Vec<bool> = reached
        .iter()
        .zip(units_per_stratum)
        .map(|(&r, &m)| domain_rule && r == 1 && m > 1)
        .collect();
    let n_units = present
        .iter()
        .zip(units_per_stratum.iter())
        .filter(|(p, _)| **p)
        .map(|(_, &m)| m as usize)
        .sum();
    let (mut n_strata, mut n_singletons) = (0, 0);
    for ((&p, &m), &l) in present.iter().zip(units_per_stratum.iter()).zip(&lonely) {
        if p && m > 0 {
            n_strata += 1;
            n_singletons += usize::from(m == 1 || l);
        }
    }
    CenterScope {
        present,
        lonely,
        n_units,
        n_strata,
        n_singletons,
    }
}

/// `singleton.scale()`: the stage-1 matrix times `nstrat/nokstrat`.
fn scale_stage1(cov: &mut [Vec<f64>], scope: Option<&CenterScope>, sm: SingletonMethod) {
    if let Some(s) = scope.filter(|_| sm == SingletonMethod::Scale) {
        let f = s.scale_factor();
        cov.iter_mut().flatten().for_each(|v| *v *= f);
    }
}

#[allow(clippy::too_many_arguments)]
fn variance_stratified_optimized(
    scores: &[f64],
    strata_indices: &[u32],
    n_strata: u32,
    psu_indices: Option<&[u32]>,
    psu_per_stratum: Option<&[Vec<u32>]>,
    n_psus_per_stratum: Option<&[u32]>,
    fpc_per_stratum: Option<&[f64]>,
    singleton_method: SingletonMethod,
    domain_rule: bool,
    active: Option<&[bool]>,
) -> f64 {
    let n = scores.len();
    if n == 0 || n_strata == 0 {
        return 0.0;
    }
    let scoped = singleton_method != SingletonMethod::None || domain_rule;

    match (psu_indices, psu_per_stratum, n_psus_per_stratum) {
        (Some(psu_idx), Some(psu_map), Some(n_psus)) => {
            let max_psu = psu_idx
                .iter()
                .filter(|&&p| p != u32::MAX)
                .max()
                .copied()
                .unwrap_or(0);
            let mut psu_totals = vec![0.0; (max_psu + 1) as usize];

            for (score, &psu) in scores.iter().zip(psu_idx.iter()) {
                if psu != u32::MAX {
                    psu_totals[psu as usize] += score;
                }
            }

            let center = singleton_method == SingletonMethod::Center;
            let scope = scoped
                .then(|| center_scope(strata_indices, n_psus, Some(psu_idx), active, domain_rule));
            let mut grand_mean = 0.0;
            if let Some(s) = scope.as_ref().filter(|_| center) {
                let total_score: f64 = psu_totals.iter().sum();
                if s.n_units > 0 {
                    grand_mean = total_score / (s.n_units as f64);
                }
            }

            let mut total_var = 0.0;
            for h in 0..n_strata as usize {
                let n_psus_h = n_psus[h];
                if n_psus_h == 0 {
                    continue;
                }

                let fpc_h = fpc_per_stratum.map(|f| f[h]).unwrap_or(1.0);

                if n_psus_h == 1 {
                    if center
                        && scope.as_ref().is_some_and(|s| s.present[h])
                        && let Some(&p) = psu_map[h].first()
                    {
                        total_var += fpc_h * (psu_totals[p as usize] - grand_mean).powi(2);
                    }
                    continue;
                }

                let lonely = scope.as_ref().is_some_and(|s| s.lonely[h]);
                if lonely && !center {
                    continue;
                }
                let psu_indices_h = &psu_map[h];
                let psu_mean_h: f64 = if lonely {
                    grand_mean
                } else {
                    psu_indices_h
                        .iter()
                        .map(|&p| psu_totals[p as usize])
                        .sum::<f64>()
                        / (n_psus_h as f64)
                };
                let sum_sq_diff: f64 = psu_indices_h
                    .iter()
                    .map(|&p| (psu_totals[p as usize] - psu_mean_h).powi(2))
                    .sum();
                total_var += fpc_h * (n_psus_h as f64 / (n_psus_h as f64 - 1.0)) * sum_sq_diff;
            }
            match scope.filter(|_| singleton_method == SingletonMethod::Scale) {
                Some(s) => total_var * s.scale_factor(),
                None => total_var,
            }
        }
        _ => {
            // Stratified element sampling
            let mut stratum_sums = vec![0.0; n_strata as usize];
            let mut stratum_sum_sq = vec![0.0; n_strata as usize];
            let mut stratum_counts = vec![0u32; n_strata as usize];
            let mut grand_total = 0.0;

            for (&score, &stratum) in scores.iter().zip(strata_indices.iter()) {
                if stratum != u32::MAX {
                    let h = stratum as usize;
                    stratum_sums[h] += score;
                    stratum_sum_sq[h] += score * score;
                    stratum_counts[h] += 1;
                    grand_total += score;
                }
            }

            let center = singleton_method == SingletonMethod::Center;
            let scope = scoped
                .then(|| center_scope(strata_indices, &stratum_counts, None, active, domain_rule));
            let mut grand_mean = 0.0;
            if let Some(s) = scope.as_ref().filter(|s| center && s.n_units > 0) {
                grand_mean = grand_total / (s.n_units as f64);
            }

            let mut total_var = 0.0;
            for h in 0..n_strata as usize {
                let n_h = stratum_counts[h];
                if n_h == 0 {
                    continue;
                }

                let fpc_h = fpc_per_stratum.map(|f| f[h]).unwrap_or(1.0);

                if n_h == 1 {
                    if center && scope.as_ref().is_some_and(|s| s.present[h]) {
                        total_var += fpc_h * (stratum_sums[h] - grand_mean).powi(2);
                    }
                    continue;
                }
                let nf = n_h as f64;
                let ss_h = if scope.as_ref().is_some_and(|s| s.lonely[h]) {
                    if !center {
                        continue;
                    }
                    // Centred at the grand mean: sum (s - c)^2 = ss - 2c sum + n c^2.
                    stratum_sum_sq[h] - 2.0 * grand_mean * stratum_sums[h]
                        + nf * grand_mean * grand_mean
                } else {
                    stratum_sum_sq[h] - stratum_sums[h].powi(2) / nf
                };
                total_var += fpc_h * ss_h / (nf - 1.0) * nf;
            }
            match scope.filter(|_| singleton_method == SingletonMethod::Scale) {
                Some(s) => total_var * s.scale_factor(),
                None => total_var,
            }
        }
    }
}

fn compute_stage2_variance(
    scores: &[f64],
    psu_indices: &[u32],
    ssu_indices: &[u32],
    _strata_indices: Option<&[u32]>,
    fpc_per_stratum: Option<&[f64]>,
    fpc_ssu_arr: Option<&[f64]>,
    strata_for_psu: Option<&[u32]>,
) -> f64 {
    let n = scores.len();
    if n == 0 {
        return 0.0;
    }

    let max_psu = psu_indices
        .iter()
        .filter(|&&p| p != u32::MAX)
        .max()
        .copied()
        .unwrap_or(0);
    let n_psus = (max_psu + 1) as usize;
    let mut psu_obs: Vec<Vec<usize>> = vec![Vec::new(); n_psus];

    for (i, &psu) in psu_indices.iter().enumerate() {
        if psu != u32::MAX {
            psu_obs[psu as usize].push(i);
        }
    }

    let mut total_stage2_var = 0.0;
    for psu in 0..n_psus {
        let obs_indices = &psu_obs[psu];
        if obs_indices.is_empty() {
            continue;
        }

        // FPC of this PSU's stratum, (N_h - n_h) / N_h, so the sampling fraction is 1 - fpc_h.
        let fpc_psu_val = match (strata_for_psu, fpc_per_stratum) {
            (Some(s_for_p), Some(fpc_s)) => {
                let stratum_idx = s_for_p[psu] as usize;
                if stratum_idx < fpc_s.len() {
                    fpc_s[stratum_idx]
                } else {
                    1.0
                }
            }
            _ => fpc_per_stratum
                .and_then(|f| f.first().copied())
                .unwrap_or(1.0),
        };

        let stage1_sampling_fraction = 1.0 - fpc_psu_val;
        if stage1_sampling_fraction <= 0.0 {
            continue;
        }

        // Unit-level FPC of this PSU, read from its first row.
        let fpc_ssu_val = fpc_ssu_arr.map(|f| f[obs_indices[0]]).unwrap_or(1.0);

        let psu_scores: Vec<f64> = obs_indices.iter().map(|&i| scores[i]).collect();
        let psu_ssu_raw: Vec<u32> = obs_indices.iter().map(|&i| ssu_indices[i]).collect();
        let (psu_ssu_indices, n_ssus) = reindex_within_subset(&psu_ssu_raw);

        if n_ssus <= 1 {
            continue;
        }

        let mut ssu_totals = vec![0.0; n_ssus as usize];
        for (score, &ssu) in psu_scores.iter().zip(psu_ssu_indices.iter()) {
            if ssu != u32::MAX {
                ssu_totals[ssu as usize] += score;
            }
        }
        let ssu_mean = ssu_totals.iter().sum::<f64>() / (n_ssus as f64);
        let sum_sq_diff: f64 = ssu_totals.iter().map(|&t| (t - ssu_mean).powi(2)).sum();
        let var_within_psu = (n_ssus as f64 / (n_ssus as f64 - 1.0)) * sum_sq_diff;

        total_stage2_var += stage1_sampling_fraction * fpc_ssu_val * var_within_psu;
    }
    total_stage2_var
}

// ============================================================================
// Main Public Variance Function
// ============================================================================

/// Design-only Taylor variance intermediates, built once and reused across domains and by-groups.
pub(crate) struct TaylorDesign {
    sm_enum: SingletonMethod,
    /// The method also applies to strata with one PSU in the domain.
    domain_rule: bool,
    // Stage 1 — one of {unstratified, stratified}
    strata_indices: Option<Vec<u32>>, // None ⇒ unstratified
    n_strata: u32,
    psu_indices: Option<Vec<u32>>, // stage-1 PSU codes (nested when stratified)
    n_psus: u32,                   // unstratified PSU count
    psu_per_stratum: Option<Vec<Vec<u32>>>,
    n_psus_per_stratum: Option<Vec<u32>>,
    fpc_val: f64,                      // unstratified single FPC
    fpc_per_stratum: Option<Vec<f64>>, // stratified per-stratum FPC (or [fpc_val] unstrat)
    stage2: Option<Stage2>,            // None ⇒ single stage
    /// Score centring for a calibrating adjustment, so every variance caller applies it.
    calib: Option<CalibSweep>,
}

/// Exists only when both PSU and SSU are set, so stage 2 never runs without them.
struct Stage2 {
    psu_indices: Vec<u32>, // the stage-1 PSU codes
    ssu_indices: Vec<u32>,
    fpc_ssu_arr: Option<Vec<f64>>,
    psu_to_stratum: Option<Vec<u32>>, // None when unstratified
}

impl TaylorDesign {
    pub(crate) fn with_calib(mut self, calib: Option<CalibSweep>) -> Self {
        self.calib = calib;
        self
    }

    /// The domain's rows ([`row_mask`]) when a stratified singleton rule depends on which strata it
    /// reaches; `None` otherwise, and when calibrated (nonzero scores outside the domain, as in R).
    pub(crate) fn domain_rows(
        &self,
        n: usize,
        rows: Option<&BooleanChunked>,
        domain: Option<&BooleanChunked>,
    ) -> Option<Vec<bool>> {
        let needed = (self.sm_enum != SingletonMethod::None || self.domain_rule)
            && self.strata_indices.is_some()
            && self.calib.is_none();
        needed.then(|| row_mask(n, rows, domain))
    }
}

pub(crate) fn build_taylor_design(
    strata: Option<&Column>,
    psu: Option<&Column>,
    ssu: Option<&Column>,
    fpc: Option<&Float64Chunked>,
    fpc_ssu: Option<&Float64Chunked>,
    singleton_method: Option<&str>,
) -> EResult<TaylorDesign> {
    let (sm_enum, domain_rule) = SingletonMethod::parse_rule(singleton_method)?;

    let fpc_arr: Option<Vec<f64>> = fpc.map(|f| f.iter().map(|v| v.unwrap_or(1.0)).collect());
    let fpc_ssu_arr: Option<Vec<f64>> =
        fpc_ssu.map(|f| f.iter().map(|v| v.unwrap_or(1.0)).collect());

    let Some(strata_col) = strata else {
        let fpc_val = fpc_arr
            .as_ref()
            .and_then(|f| f.first().copied())
            .unwrap_or(1.0);
        let (psu_indices, n_psus) = match psu {
            Some(psu_col) => {
                let (idx, cnt) = design_col_codes(psu_col)?;
                (Some(idx), cnt)
            }
            None => (None, 0),
        };
        let stage2 = build_stage2(ssu, psu_indices.as_deref(), None, fpc_ssu_arr)?;
        return Ok(TaylorDesign {
            sm_enum,
            domain_rule,
            strata_indices: None,
            n_strata: 0,
            psu_indices,
            n_psus,
            psu_per_stratum: None,
            n_psus_per_stratum: None,
            fpc_val,
            // Stage 2 derives the stage-1 sampling fraction from this FPC.
            fpc_per_stratum: Some(vec![fpc_val]),
            stage2,
            calib: None,
        });
    };

    let (strata_indices, n_strata) = design_col_codes(strata_col)?;

    // Per-stratum FPC: for each stratum h, the FPC of its first row.
    let fpc_by_stratum: Option<Vec<f64>> = fpc_arr.as_ref().map(|fpc_vals| {
        let mut per_stratum = vec![1.0; n_strata as usize];
        let mut seen = vec![false; n_strata as usize];
        for (i, &s) in strata_indices.iter().enumerate() {
            if s != u32::MAX && !seen[s as usize] {
                per_stratum[s as usize] = fpc_vals[i];
                seen[s as usize] = true;
            }
        }
        per_stratum
    });

    match psu {
        Some(psu_col) => {
            // Nest PSU in stratum so reused labels stay distinct PSUs, as R survey and glm.rs do.
            let (psu_indices, _) = design_pair_codes(strata_col, psu_col)?;
            let (psu_per_stratum, n_psus_per_stratum) =
                build_stratum_psu_map(&strata_indices, n_strata, &psu_indices);
            let stage2 = build_stage2(ssu, Some(&psu_indices), Some(&strata_indices), fpc_ssu_arr)?;

            Ok(TaylorDesign {
                sm_enum,
                domain_rule,
                strata_indices: Some(strata_indices),
                n_strata,
                psu_indices: Some(psu_indices),
                n_psus: 0,
                psu_per_stratum: Some(psu_per_stratum),
                n_psus_per_stratum: Some(n_psus_per_stratum),
                fpc_val: 1.0,
                fpc_per_stratum: fpc_by_stratum,
                stage2,
                calib: None,
            })
        }
        None => Ok(TaylorDesign {
            sm_enum,
            domain_rule,
            strata_indices: Some(strata_indices),
            n_strata,
            psu_indices: None,
            n_psus: 0,
            psu_per_stratum: None,
            n_psus_per_stratum: None,
            fpc_val: 1.0,
            fpc_per_stratum: fpc_by_stratum,
            stage2: None,
            calib: None,
        }),
    }
}

fn build_stage2(
    ssu: Option<&Column>,
    psu_indices: Option<&[u32]>,
    strata_indices: Option<&[u32]>,
    fpc_ssu_arr: Option<Vec<f64>>,
) -> EResult<Option<Stage2>> {
    let (Some(ssu_col), Some(psu_indices)) = (ssu, psu_indices) else {
        return Ok(None);
    };
    let psu_to_stratum = strata_indices.map(|strata_indices| {
        let max_psu = psu_indices
            .iter()
            .filter(|&&p| p != u32::MAX)
            .max()
            .copied()
            .unwrap_or(0);
        let mut psu_to_stratum = vec![0u32; (max_psu + 1) as usize];
        for (&psu_idx, &str_idx) in psu_indices.iter().zip(strata_indices.iter()) {
            if psu_idx != u32::MAX && str_idx != u32::MAX {
                psu_to_stratum[psu_idx as usize] = str_idx;
            }
        }
        psu_to_stratum
    });
    Ok(Some(Stage2 {
        psu_indices: psu_indices.to_vec(),
        ssu_indices: design_col_codes(ssu_col)?.0,
        fpc_ssu_arr,
        psu_to_stratum,
    }))
}

/// Apply a prebuilt [`TaylorDesign`] to scores, the only per-group part of the variance.
#[cfg(test)]
pub(crate) fn taylor_variance_apply(scores_arr: &[f64], d: &TaylorDesign) -> f64 {
    taylor_variance_apply_in(scores_arr, d, None)
}

/// [`taylor_variance_apply`] for a domain: `active` is [`TaylorDesign::domain_rows`].
pub(crate) fn taylor_variance_apply_in(
    scores_arr: &[f64],
    d: &TaylorDesign,
    active: Option<&[bool]>,
) -> f64 {
    // Centre first, as R applies postStrata at the top of svyrecvar.
    let swept;
    let scores_arr: &[f64] = match &d.calib {
        Some(c) => {
            let mut v = scores_arr.to_vec();
            c.apply(&mut v);
            swept = v;
            &swept
        }
        None => scores_arr,
    };

    // --- STAGE 1 ---
    let var_stage1 = match d.strata_indices.as_deref() {
        None => {
            d.fpc_val
                * variance_unstratified_optimized(scores_arr, d.psu_indices.as_deref(), d.n_psus)
        }
        Some(strata_indices) => variance_stratified_optimized(
            scores_arr,
            strata_indices,
            d.n_strata,
            d.psu_indices.as_deref(),
            d.psu_per_stratum.as_deref(),
            d.n_psus_per_stratum.as_deref(),
            d.fpc_per_stratum.as_deref(),
            d.sm_enum,
            d.domain_rule,
            active,
        ),
    };

    // --- STAGE 2 ---
    let Some(s2) = &d.stage2 else {
        return var_stage1;
    };

    let var_stage2 = compute_stage2_variance(
        scores_arr,
        &s2.psu_indices,
        &s2.ssu_indices,
        d.strata_indices.as_deref(),
        d.fpc_per_stratum.as_deref(),
        s2.fpc_ssu_arr.as_deref(),
        s2.psu_to_stratum.as_deref(),
    );

    var_stage1 + var_stage2
}

pub fn taylor_variance(
    scores: &Float64Chunked,
    strata: Option<&Column>,
    psu: Option<&Column>,
    ssu: Option<&Column>,
    fpc: Option<&Float64Chunked>,
    fpc_ssu: Option<&Float64Chunked>,
    singleton_method: Option<&str>,
) -> EResult<f64> {
    taylor_variance_in(
        scores,
        strata,
        psu,
        ssu,
        fpc,
        fpc_ssu,
        singleton_method,
        None,
    )
}

/// [`taylor_variance`] for a domain: `active` is [`singleton_domain_rows`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn taylor_variance_in(
    scores: &Float64Chunked,
    strata: Option<&Column>,
    psu: Option<&Column>,
    ssu: Option<&Column>,
    fpc: Option<&Float64Chunked>,
    fpc_ssu: Option<&Float64Chunked>,
    singleton_method: Option<&str>,
    active: Option<&[bool]>,
) -> EResult<f64> {
    if scores.is_empty() {
        return Ok(0.0);
    }
    let design = build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method)?;
    let scores_arr: Vec<f64> = scores.iter().map(|s| s.unwrap_or(0.0)).collect();
    Ok(taylor_variance_apply_in(&scores_arr, &design, active))
}

/// A domain's rows as the singleton rules see them: in `rows` (`where=`) and `domain` (`by`).
/// Weight plays no part: R's `subset()` keeps zero-weight rows, so their PSUs still count.
pub(crate) fn row_mask(
    n: usize,
    rows: Option<&BooleanChunked>,
    domain: Option<&BooleanChunked>,
) -> Vec<bool> {
    let get = |m: Option<&BooleanChunked>, i: usize| m.is_none_or(|m| m.get(i).unwrap_or(false));
    (0..n).map(|i| get(rows, i) && get(domain, i)).collect()
}

fn singleton_rows_needed(
    singleton_method: Option<&str>,
    calibrated: bool,
    stratified: bool,
) -> EResult<bool> {
    let (method, domain_rule) = SingletonMethod::parse_rule(singleton_method)?;
    Ok((method != SingletonMethod::None || domain_rule) && stratified && !calibrated)
}

/// [`TaylorDesign::domain_rows`] for kernels that sweep calibration into the scores themselves.
pub(crate) fn singleton_domain_rows(
    singleton_method: Option<&str>,
    calibrated: bool,
    stratified: bool,
    n: usize,
    domain: Option<&BooleanChunked>,
) -> EResult<Option<Vec<bool>>> {
    Ok(
        singleton_rows_needed(singleton_method, calibrated, stratified)?
            .then(|| row_mask(n, None, domain)),
    )
}

/// [`singleton_domain_rows`] with the domain's rows already resolved.
pub(crate) fn singleton_rows(
    singleton_method: Option<&str>,
    calibrated: bool,
    stratified: bool,
    rows: &[bool],
) -> EResult<Option<Vec<bool>>> {
    Ok(singleton_rows_needed(singleton_method, calibrated, stratified)?.then(|| rows.to_vec()))
}

/// The boolean domain-row column `col` of `df` ([`row_mask`]'s `rows`).
pub(crate) fn rows_column(df: &DataFrame, col: Option<&str>) -> EResult<Option<BooleanChunked>> {
    col.map(|c| {
        let s = df.column(c)?.as_materialized_series();
        let b = match s.dtype() {
            DataType::Boolean => s.bool()?.clone(),
            _ => s.cast(&DataType::String)?.str()?.equal("true"),
        };
        Ok(b)
    })
    .transpose()
}

/// k×k covariance of k estimates from their score columns, under `taylor_variance_apply`'s rules.
/// The diagonal matches it only up to summation order: callers needing exact agreement keep that.
pub(crate) fn taylor_covariance_apply(score_cols: &[Vec<f64>], d: &TaylorDesign) -> Vec<Vec<f64>> {
    taylor_covariance_apply_in(score_cols, d, None)
}

/// Columns with a nonzero value among `values(j)`, or all `k` when any is non-finite: zero columns
/// add only ±0 terms, leaving sums bit-identical unless a ±∞/NaN partner makes 0·∞ = NaN.
fn present_cols<I: Iterator<Item = f64>>(k: usize, values: impl Fn(usize) -> I) -> Vec<usize> {
    let mut present = Vec::new();
    for j in 0..k {
        let mut nonzero = false;
        for v in values(j) {
            if !v.is_finite() {
                return (0..k).collect();
            }
            nonzero |= v != 0.0;
        }
        if nonzero {
            present.push(j);
        }
    }
    present
}

/// [`taylor_covariance_apply`] for estimates of one domain. Different domains keep the whole frame,
/// as R's `svyby(covmat=TRUE)` does.
pub(crate) fn taylor_covariance_apply_in(
    score_cols: &[Vec<f64>],
    d: &TaylorDesign,
    active: Option<&[bool]>,
) -> Vec<Vec<f64>> {
    let k = score_cols.len();
    if k == 0 {
        return vec![];
    }
    let n = score_cols[0].len();
    let mut cov = vec![vec![0.0; k]; k];
    if n == 0 {
        return cov;
    }

    let swept: Vec<Vec<f64>>;
    let cols: Vec<&[f64]> = match &d.calib {
        Some(c) => {
            swept = score_cols
                .iter()
                .map(|s| {
                    let mut v = s.clone();
                    c.apply(&mut v);
                    v
                })
                .collect();
            swept.iter().map(|v| v.as_slice()).collect()
        }
        None => score_cols.iter().map(|v| v.as_slice()).collect(),
    };

    let add = |cov: &mut Vec<Vec<f64>>, j: usize, l: usize, v: f64| {
        cov[j][l] += v;
        if l != j {
            cov[l][j] += v;
        }
    };

    // --- STAGE 1 ---
    match d.strata_indices.as_deref() {
        None => {
            // Unstratified: PSU totals (or the units themselves when unclustered).
            let (totals, m): (Vec<Vec<f64>>, usize) = match d.psu_indices.as_deref() {
                Some(psu_idx) => {
                    if d.n_psus <= 1 {
                        return cov;
                    }
                    let mut t = vec![vec![0.0; d.n_psus as usize]; k];
                    for j in 0..k {
                        for (s, &p) in cols[j].iter().zip(psu_idx.iter()) {
                            if p != u32::MAX {
                                t[j][p as usize] += s;
                            }
                        }
                    }
                    (t, d.n_psus as usize)
                }
                None => {
                    if n <= 1 {
                        return cov;
                    }
                    (cols.iter().map(|c| c.to_vec()).collect(), n)
                }
            };
            let mf = m as f64;
            let means: Vec<f64> = totals.iter().map(|t| t.iter().sum::<f64>() / mf).collect();
            let scale = d.fpc_val * mf / (mf - 1.0);
            for j in 0..k {
                for l in j..k {
                    let s: f64 = (0..m)
                        .map(|a| (totals[j][a] - means[j]) * (totals[l][a] - means[l]))
                        .sum();
                    add(&mut cov, j, l, scale * s);
                }
            }
        }
        Some(strata_idx) => {
            let n_strata = d.n_strata as usize;
            match (
                d.psu_indices.as_deref(),
                d.psu_per_stratum.as_deref(),
                d.n_psus_per_stratum.as_deref(),
            ) {
                (Some(psu_idx), Some(psu_map), Some(n_psus)) => {
                    let max_psu = psu_idx
                        .iter()
                        .filter(|&&p| p != u32::MAX)
                        .max()
                        .copied()
                        .unwrap_or(0);
                    let mut psu_totals = vec![vec![0.0; (max_psu + 1) as usize]; k];
                    for j in 0..k {
                        for (s, &p) in cols[j].iter().zip(psu_idx.iter()) {
                            if p != u32::MAX {
                                psu_totals[j][p as usize] += s;
                            }
                        }
                    }

                    let center = d.sm_enum == SingletonMethod::Center;
                    let scope = (d.sm_enum != SingletonMethod::None || d.domain_rule).then(|| {
                        center_scope(strata_idx, n_psus, Some(psu_idx), active, d.domain_rule)
                    });
                    let mut grand_mean = vec![0.0; k];
                    if let Some(s) = scope.as_ref().filter(|s| center && s.n_units > 0) {
                        for j in 0..k {
                            grand_mean[j] = psu_totals[j].iter().sum::<f64>() / (s.n_units as f64);
                        }
                    }

                    for h in 0..n_strata {
                        let m_h = n_psus[h];
                        if m_h == 0 {
                            continue;
                        }
                        let fpc_h = d.fpc_per_stratum.as_deref().map(|f| f[h]).unwrap_or(1.0);
                        if m_h == 1 {
                            if center
                                && scope.as_ref().is_some_and(|s| s.present[h])
                                && let Some(&p) = psu_map[h].first()
                            {
                                for j in 0..k {
                                    let dj = psu_totals[j][p as usize] - grand_mean[j];
                                    for l in j..k {
                                        let dl = psu_totals[l][p as usize] - grand_mean[l];
                                        add(&mut cov, j, l, fpc_h * dj * dl);
                                    }
                                }
                            }
                            continue;
                        }
                        let lonely = scope.as_ref().is_some_and(|s| s.lonely[h]);
                        if lonely && !center {
                            continue;
                        }
                        let mf = m_h as f64;
                        let means_h: Vec<f64> = if lonely {
                            grand_mean.clone()
                        } else {
                            (0..k)
                                .map(|j| {
                                    psu_map[h]
                                        .iter()
                                        .map(|&p| psu_totals[j][p as usize])
                                        .sum::<f64>()
                                        / mf
                                })
                                .collect()
                        };
                        let scale = fpc_h * mf / (mf - 1.0);
                        // A lonely stratum centres at the grand mean, so every column counts.
                        let present = if lonely {
                            (0..k).collect()
                        } else {
                            {
                                let pt = &psu_totals;
                                present_cols(k, |j| {
                                    psu_map[h].iter().map(move |&p| pt[j][p as usize])
                                })
                            }
                        };
                        for &p in &psu_map[h] {
                            for (a, &j) in present.iter().enumerate() {
                                let dj = psu_totals[j][p as usize] - means_h[j];
                                for &l in &present[a..] {
                                    let dl = psu_totals[l][p as usize] - means_h[l];
                                    add(&mut cov, j, l, scale * dj * dl);
                                }
                            }
                        }
                    }
                    scale_stage1(&mut cov, scope.as_ref(), d.sm_enum);
                }
                _ => {
                    // Stratified element sampling: per-stratum sums and counts in one pass.
                    let mut sums = vec![vec![0.0; n_strata]; k];
                    let mut counts = vec![0u32; n_strata];
                    let mut rows_of: Vec<Vec<usize>> = vec![Vec::new(); n_strata];
                    for (i, &st) in strata_idx.iter().enumerate() {
                        if st == u32::MAX {
                            continue;
                        }
                        let h = st as usize;
                        counts[h] += 1;
                        rows_of[h].push(i);
                        for j in 0..k {
                            sums[j][h] += cols[j][i];
                        }
                    }

                    let center = d.sm_enum == SingletonMethod::Center;
                    let scope = (d.sm_enum != SingletonMethod::None || d.domain_rule)
                        .then(|| center_scope(strata_idx, &counts, None, active, d.domain_rule));
                    let mut grand_mean = vec![0.0; k];
                    if let Some(s) = scope.as_ref().filter(|s| center && s.n_units > 0) {
                        for j in 0..k {
                            grand_mean[j] = sums[j].iter().sum::<f64>() / (s.n_units as f64);
                        }
                    }

                    for h in 0..n_strata {
                        let n_h = counts[h];
                        if n_h == 0 {
                            continue;
                        }
                        let fpc_h = d.fpc_per_stratum.as_deref().map(|f| f[h]).unwrap_or(1.0);
                        if n_h == 1 {
                            if center && scope.as_ref().is_some_and(|s| s.present[h]) {
                                for j in 0..k {
                                    let dj = sums[j][h] - grand_mean[j];
                                    for l in j..k {
                                        let dl = sums[l][h] - grand_mean[l];
                                        add(&mut cov, j, l, fpc_h * dj * dl);
                                    }
                                }
                            }
                            continue;
                        }
                        let lonely = scope.as_ref().is_some_and(|s| s.lonely[h]);
                        if lonely && !center {
                            continue;
                        }
                        let nf = n_h as f64;
                        let scale = fpc_h * nf / (nf - 1.0);
                        let rows_h = &rows_of[h];
                        // A lonely stratum centres at the grand mean, so every column counts.
                        let present = if lonely {
                            (0..k).collect()
                        } else {
                            {
                                let cs = &cols;
                                present_cols(k, |j| rows_h.iter().map(move |&i| cs[j][i]))
                            }
                        };
                        let np = present.len();
                        let mut cross = vec![0.0; np * np];
                        for &i in rows_h {
                            for (a, &j) in present.iter().enumerate() {
                                let sj = cols[j][i];
                                for (b, &l) in present.iter().enumerate().skip(a) {
                                    cross[a * np + b] += sj * cols[l][i];
                                }
                            }
                        }
                        for (a, &j) in present.iter().enumerate() {
                            for (b, &l) in present.iter().enumerate().skip(a) {
                                let cross_h = cross[a * np + b];
                                let c_h = if lonely {
                                    let (gj, gl) = (grand_mean[j], grand_mean[l]);
                                    cross_h - gj * sums[l][h] - gl * sums[j][h] + nf * gj * gl
                                } else {
                                    cross_h - sums[j][h] * sums[l][h] / nf
                                };
                                add(&mut cov, j, l, scale * c_h);
                            }
                        }
                    }
                    scale_stage1(&mut cov, scope.as_ref(), d.sm_enum);
                }
            }
        }
    }

    // --- STAGE 2 ---
    let Some(s2) = &d.stage2 else {
        return cov;
    };
    let psu_idx = &s2.psu_indices;
    let ssu_idx = &s2.ssu_indices;

    let max_psu = psu_idx
        .iter()
        .filter(|&&p| p != u32::MAX)
        .max()
        .copied()
        .unwrap_or(0);
    let n_psus = (max_psu + 1) as usize;
    let mut psu_obs: Vec<Vec<usize>> = vec![Vec::new(); n_psus];
    for (i, &p) in psu_idx.iter().enumerate() {
        if p != u32::MAX {
            psu_obs[p as usize].push(i);
        }
    }

    for psu in 0..n_psus {
        let obs = &psu_obs[psu];
        if obs.is_empty() {
            continue;
        }
        let fpc_psu_val = match (s2.psu_to_stratum.as_deref(), d.fpc_per_stratum.as_deref()) {
            (Some(s_for_p), Some(fpc_s)) => {
                let h = s_for_p[psu] as usize;
                if h < fpc_s.len() { fpc_s[h] } else { 1.0 }
            }
            _ => d
                .fpc_per_stratum
                .as_deref()
                .and_then(|f| f.first().copied())
                .unwrap_or(1.0),
        };
        let stage1_sampling_fraction = 1.0 - fpc_psu_val;
        if stage1_sampling_fraction <= 0.0 {
            continue;
        }
        let fpc_ssu_val = s2.fpc_ssu_arr.as_deref().map(|f| f[obs[0]]).unwrap_or(1.0);

        let psu_ssu_raw: Vec<u32> = obs.iter().map(|&i| ssu_idx[i]).collect();
        let (local_ssu, n_ssus) = reindex_within_subset(&psu_ssu_raw);
        if n_ssus <= 1 {
            continue;
        }

        let mut ssu_totals = vec![vec![0.0; n_ssus as usize]; k];
        for j in 0..k {
            for (&i, &s) in obs.iter().zip(local_ssu.iter()) {
                if s != u32::MAX {
                    ssu_totals[j][s as usize] += cols[j][i];
                }
            }
        }
        let nf = n_ssus as f64;
        let means: Vec<f64> = ssu_totals
            .iter()
            .map(|t| t.iter().sum::<f64>() / nf)
            .collect();
        let scale = stage1_sampling_fraction * fpc_ssu_val * nf / (nf - 1.0);
        let present = present_cols(k, |j| ssu_totals[j].iter().copied());
        for (a, &j) in present.iter().enumerate() {
            for &l in &present[a..] {
                let s: f64 = (0..n_ssus as usize)
                    .map(|a| (ssu_totals[j][a] - means[j]) * (ssu_totals[l][a] - means[l]))
                    .sum();
                add(&mut cov, j, l, scale * s);
            }
        }
    }

    cov
}

pub(crate) fn degrees_of_freedom(
    weights: &Float64Chunked,
    strata: Option<&Column>,
    psu: Option<&Column>,
) -> EResult<u32> {
    degrees_of_freedom_in_domain(weights, strata, psu, None)
}

/// Design df restricted to `domain`: by-groups arrive only as a mask, `where=` as zero weights.
pub(crate) fn degrees_of_freedom_in_domain(
    weights: &Float64Chunked,
    strata: Option<&Column>,
    psu: Option<&Column>,
    domain: Option<&BooleanChunked>,
) -> EResult<u32> {
    let n = weights.len();
    if n == 0 {
        return Ok(0);
    }

    // Active rows (nonzero weight, in-domain), as R's degf() on a subsetted design.
    let active = active_mask(weights, domain);

    // Unique counts ignore code order, so either dtype path gives the same df.
    let str_idx = strata.map(design_col_codes).transpose()?;
    let psu_idx = match (strata, psu) {
        // Pair-nested PSUs when stratified, so reused labels stay distinct.
        (Some(s), Some(p)) => Some(design_pair_codes(s, p)?.0),
        (None, Some(p)) => Some(design_col_codes(p)?.0),
        _ => None,
    };

    Ok(df_from_codes(
        &active,
        str_idx.as_ref().map(|(v, n)| (v.as_slice(), *n)),
        psu_idx.as_deref(),
    ))
}

/// Design df from a [`TaylorDesign`]'s codes, sparing a second densification of the same columns.
pub(crate) fn degrees_of_freedom_from_design(
    weights: &Float64Chunked,
    design: &TaylorDesign,
    domain: Option<&BooleanChunked>,
) -> u32 {
    if weights.is_empty() {
        return 0;
    }
    let active = active_mask(weights, domain);
    df_from_codes(
        &active,
        design
            .strata_indices
            .as_ref()
            .map(|v| (v.as_slice(), design.n_strata)),
        design.psu_indices.as_deref(),
    )
}

/// Active means nonzero weight: zero-weight rows stand for no units; negative calibrated ones do.
#[inline]
fn is_active(w: Option<f64>) -> bool {
    w.is_some_and(|v| v != 0.0 && !v.is_nan())
}

#[inline]
fn weights_slice(weights: &Float64Chunked) -> Option<&[f64]> {
    if weights.null_count() == 0 {
        weights.cont_slice().ok()
    } else {
        None
    }
}

/// Active (nonzero-weight, in-domain) rows, shared so both df entry points agree.
fn active_mask(weights: &Float64Chunked, domain: Option<&BooleanChunked>) -> Vec<bool> {
    match domain {
        None => match weights_slice(weights) {
            Some(ws) => ws.iter().map(|&w| is_active(Some(w))).collect(),
            None => weights.iter().map(is_active).collect(),
        },
        Some(mask) => {
            let flags = mask_flags(mask);
            match weights_slice(weights) {
                Some(ws) => ws
                    .iter()
                    .zip(flags)
                    .map(|(&w, d)| d && is_active(Some(w)))
                    .collect(),
                None => weights
                    .iter()
                    .zip(flags)
                    .map(|(w, d)| d && is_active(w))
                    .collect(),
            }
        }
    }
}

/// Domain respondents (nonzero weight, in `domain`): the Korn–Graubard `n` of the proportion CIs.
/// `where=` arrives as zeroed weights, so a row count would include out-of-domain rows.
pub(crate) fn active_count(weights: &Float64Chunked, domain: Option<&BooleanChunked>) -> u32 {
    match domain {
        None => match weights_slice(weights) {
            Some(ws) => ws.iter().filter(|&&w| is_active(Some(w))).count() as u32,
            None => weights.iter().filter(|w| is_active(*w)).count() as u32,
        },
        Some(mask) => {
            let flags = mask_flags(mask);
            match weights_slice(weights) {
                Some(ws) => ws
                    .iter()
                    .zip(flags)
                    .filter(|&(&w, d)| d && is_active(Some(w)))
                    .count() as u32,
                None => weights
                    .iter()
                    .zip(flags)
                    .filter(|(w, d)| *d && is_active(*w))
                    .count() as u32,
            }
        }
    }
}

/// The df arithmetic itself, over resolved design codes.
fn df_from_codes(active: &[bool], strata: Option<(&[u32], u32)>, psu: Option<&[u32]>) -> u32 {
    match (strata, psu) {
        (None, None) => {
            // No strata, no PSU: each obs is its own PSU, df = n_active - 1
            let n_active = active.iter().filter(|&&a| a).count();
            n_active.saturating_sub(1) as u32
        }
        (None, Some(psu_idx)) => {
            // No strata, with PSU: df = n_unique_active_psus - 1
            let n_psus = psu_idx
                .iter()
                .filter(|&&p| p != u32::MAX)
                .max()
                .map_or(0, |&p| p as usize + 1);
            let mut seen = vec![false; n_psus];
            let mut n_seen = 0u32;
            for (&p, &act) in psu_idx.iter().zip(active.iter()) {
                if act && p != u32::MAX && !seen[p as usize] {
                    seen[p as usize] = true;
                    n_seen += 1;
                }
            }
            n_seen.saturating_sub(1)
        }
        (Some((str_idx, n_strata)), None) => {
            // Stratified, no PSU: df = sum_h(n_active_h - 1)
            let mut counts = vec![0u32; n_strata as usize];
            for (&s, &act) in str_idx.iter().zip(active.iter()) {
                if act && s != u32::MAX {
                    counts[s as usize] += 1;
                }
            }
            counts.iter().map(|&c| c.saturating_sub(1)).sum()
        }
        (Some((str_idx, n_strata)), Some(psu_idx)) => {
            // Stratified + clustered: df = sum_h(n_active_psus_h - 1), PSUs pair-nested.
            // An owner table replaces hash sets; un-nested codes shared across strata fall back.
            let n_psus = psu_idx
                .iter()
                .filter(|&&p| p != u32::MAX)
                .max()
                .map_or(0, |&p| p as usize + 1);
            let mut owner = vec![u32::MAX; n_psus];
            let mut counts = vec![0u32; n_strata as usize];
            for ((&s, &p), &act) in str_idx.iter().zip(psu_idx.iter()).zip(active.iter()) {
                if act && s != u32::MAX && p != u32::MAX {
                    let o = &mut owner[p as usize];
                    if *o == u32::MAX {
                        *o = s;
                        counts[s as usize] += 1;
                    } else if *o != s {
                        return df_shared_psu_codes(active, str_idx, n_strata, psu_idx);
                    }
                }
            }
            counts.iter().map(|&c| c.saturating_sub(1)).sum()
        }
    }
}

fn df_shared_psu_codes(active: &[bool], str_idx: &[u32], n_strata: u32, psu_idx: &[u32]) -> u32 {
    let mut stratum_psus: Vec<FxHashSet<u32>> = vec![FxHashSet::default(); n_strata as usize];
    for ((&s, &p), &act) in str_idx.iter().zip(psu_idx.iter()).zip(active.iter()) {
        if act && s != u32::MAX && p != u32::MAX {
            stratum_psus[s as usize].insert(p);
        }
    }
    stratum_psus
        .iter()
        .map(|psus| psus.len().saturating_sub(1) as u32)
        .sum()
}

// ============================================================================
// Quantile Variance using Woodruff Method
// ============================================================================

/// Woodruff `(estimate, var_p, se_p)` per prob, sorting once; all-NaN where undefined.
/// With `domain_mask`, scores outside are zero, so the variance still spans the full design.
pub(crate) fn quantiles_woodruff(
    y: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: Option<&BooleanChunked>,
    rows: Option<&BooleanChunked>,
    design: &TaylorDesign,
    probs: &[f64],
    quantile_method: SvyQuantileMethod,
) -> EResult<Vec<(f64, f64, f64)>> {
    let nan_rows = || vec![(f64::NAN, f64::NAN, f64::NAN); probs.len()];

    let Some((y_sorted, cdf, sum_w)) = sorted_weighted_cdf(quantile_pairs(y, weights, domain_mask))
    else {
        return Ok(nan_rows());
    };
    let active = design.domain_rows(y.len(), rows, domain_mask);

    Ok(probs
        .iter()
        .map(|&p| {
            let q = weighted_quantile(&y_sorted, &cdf, p, quantile_method);
            if q.is_nan() {
                return (f64::NAN, f64::NAN, f64::NAN);
            }
            // Variance of P(Y <= q); the caller inverts the CDF for the interval.
            let scores = scores_quantile_arr(y, weights, domain_mask, q, p, sum_w);
            let var_p = taylor_variance_apply_in(&scores, design, active.as_deref());
            (q, var_p, var_p.max(0.0).sqrt())
        })
        .collect())
}

// ============================================================================
// SRS Variance (Simple Random Sampling)
// ============================================================================

/// The SRS reference for deff; `WithoutReplacement` (Kish) and `WithReplacement` differ by the FPC.
/// Without the FPC, `WithReplacement` ignores the weight scale, so it suits rescaled weights.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SrsRef {
    /// `S^2/n` -- no finite-population correction.
    WithReplacement,
    /// `S^2/n * (1 - n/N)`; without `pop_total` the weight sum stands in for N, which is valid
    /// only while weights are inverse selection probabilities.
    WithoutReplacement { pop_total: Option<f64> },
}

impl SrsRef {
    /// The FPC, NaN when degenerate: it runs on every call, so Python raises only if deff is asked.
    /// The 1e-9 floor absorbs weights normalized to `n` (~1e-12 off); near-census FPCs are ~1e-6.
    #[inline]
    pub fn fpc(&self, n: f64, sum_w: f64) -> f64 {
        match self {
            SrsRef::WithReplacement => 1.0,
            SrsRef::WithoutReplacement { pop_total } => {
                let n_pop = pop_total.unwrap_or(sum_w);
                let factor = 1.0 - (n / n_pop);
                if factor > 1e-9 { factor } else { f64::NAN }
            }
        }
    }
}

pub(crate) fn weighted_s2(y: &[f64], wn: &[f64]) -> f64 {
    let n = y.len() as f64;
    if n <= 1.0 {
        return f64::NAN;
    }
    let mu: f64 = y.iter().zip(wn.iter()).map(|(yi, wi)| wi * yi).sum();
    let ss: f64 = y
        .iter()
        .zip(wn.iter())
        .map(|(yi, wi)| wi * (yi - mu).powi(2))
        .sum();
    (n / (n - 1.0)) * ss
}

fn srs_mean_from(yv: &[f64], wv: &[f64], srs: SrsRef) -> EResult<f64> {
    let n = yv.len() as f64;
    if n < 2.0 {
        return Ok(f64::NAN);
    }
    let sum_w: f64 = wv.iter().sum();
    if sum_w <= 0.0 {
        return Ok(f64::NAN);
    }
    let wn: Vec<f64> = wv.iter().map(|w| w / sum_w).collect();
    let s2_y = weighted_s2(yv, &wn);
    Ok((s2_y / n) * srs.fpc(n, sum_w))
}

pub fn srs_variance_mean(
    y: &Float64Chunked,
    weights: &Float64Chunked,
    srs: SrsRef,
) -> EResult<f64> {
    // Zero-weight rows must not inflate n, as in R's deff on subset() designs.
    // Fast path for all-positive contiguous slices; same accumulation order, so bit-identical.
    if let Some((ys, ws)) = cont_pair(y, weights) {
        if ws.iter().all(|&w| w > 0.0) {
            return srs_mean_from(ys, ws, srs);
        }
        let mut yv = Vec::new();
        let mut wv = Vec::new();
        for (yi, wi) in ys.iter().zip(ws.iter()) {
            if *wi > 0.0 {
                yv.push(*yi);
                wv.push(*wi);
            }
        }
        return srs_mean_from(&yv, &wv, srs);
    }

    // Fallback: only observations where y and w are non-null and w > 0.
    let mut yv = Vec::new();
    let mut wv = Vec::new();
    for (yi, wi) in y.iter().zip(weights.iter()) {
        if let (Some(y_val), Some(w_val)) = (yi, wi)
            && w_val > 0.0
        {
            yv.push(y_val);
            wv.push(w_val);
        }
    }
    srs_mean_from(&yv, &wv, srs)
}

pub(crate) fn srs_variance_mean_domain(
    y: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
    srs: SrsRef,
) -> EResult<f64> {
    let (yv, wv) = if let Some((ys, ws)) = cont_pair(y, weights) {
        let mut yv = Vec::new();
        let mut wv = Vec::new();
        for (i, m) in mask_flags(domain_mask).into_iter().enumerate() {
            // w > 0 as in srs_variance_mean: counting zero-weight rows would inflate n and deff.
            if m && ws[i] > 0.0 {
                yv.push(ys[i]);
                wv.push(ws[i]);
            }
        }
        (yv, wv)
    } else {
        let mut yv = Vec::new();
        let mut wv = Vec::new();
        for ((yi, wi), mi) in y.iter().zip(weights.iter()).zip(domain_mask.iter()) {
            if let (Some(y_val), Some(w_val), Some(true)) = (yi, wi, mi)
                && w_val > 0.0
            {
                yv.push(y_val);
                wv.push(w_val);
            }
        }
        (yv, wv)
    };
    let n = yv.len() as f64;
    if n < 2.0 {
        return Ok(f64::NAN);
    }
    let sum_w: f64 = wv.iter().sum();
    if sum_w <= 0.0 {
        return Ok(f64::NAN);
    }
    let wn: Vec<f64> = wv.iter().map(|w| w / sum_w).collect();
    let s2_y = weighted_s2(&yv, &wn);
    Ok((s2_y / n) * srs.fpc(n, sum_w))
}

pub(crate) fn srs_variance_total(
    y: &Float64Chunked,
    weights: &Float64Chunked,
    srs: SrsRef,
) -> EResult<f64> {
    // Zero-weight rows are excluded from n (see srs_variance_mean).
    if let Some((ys, ws)) = cont_pair(y, weights)
        && ws.iter().all(|&w| w > 0.0)
    {
        return srs_total_from(ys, ws, srs);
    }
    let mut yv: Vec<f64> = Vec::new();
    let mut wv: Vec<f64> = Vec::new();
    for (yi, wi) in y.iter().zip(weights.iter()) {
        let w_val = wi.unwrap_or(0.0);
        if w_val > 0.0 {
            yv.push(yi.unwrap_or(0.0));
            wv.push(w_val);
        }
    }
    srs_total_from(&yv, &wv, srs)
}

fn srs_total_from(yv: &[f64], wv: &[f64], srs: SrsRef) -> EResult<f64> {
    let n = yv.len() as f64;
    if n < 2.0 {
        return Ok(f64::NAN);
    }
    let sum_w: f64 = wv.iter().sum();
    if sum_w <= 0.0 {
        return Ok(f64::NAN);
    }
    let wn: Vec<f64> = wv.iter().map(|w| w / sum_w).collect();
    let s2_y = weighted_s2(yv, &wn);
    Ok(((sum_w.powi(2) / n) * s2_y) * srs.fpc(n, sum_w))
}

pub(crate) fn srs_variance_total_domain(
    y: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
    srs: SrsRef,
) -> EResult<f64> {
    let (yv, wv) = if let Some((ys, ws)) = cont_pair(y, weights) {
        let mut yv = Vec::new();
        let mut wv = Vec::new();
        for (i, m) in mask_flags(domain_mask).into_iter().enumerate() {
            // w > 0 as in srs_variance_mean: counting zero-weight rows would inflate n and deff.
            if m && ws[i] > 0.0 {
                yv.push(ys[i]);
                wv.push(ws[i]);
            }
        }
        (yv, wv)
    } else {
        let mut yv = Vec::new();
        let mut wv = Vec::new();
        for ((yi, wi), mi) in y.iter().zip(weights.iter()).zip(domain_mask.iter()) {
            if let (Some(y_val), Some(w_val), Some(true)) = (yi, wi, mi)
                && w_val > 0.0
            {
                yv.push(y_val);
                wv.push(w_val);
            }
        }
        (yv, wv)
    };
    srs_total_from(&yv, &wv, srs)
}

pub(crate) fn srs_variance_ratio(
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked,
    srs: SrsRef,
) -> EResult<f64> {
    // Zero-weight rows are excluded from n (see srs_variance_mean).
    let mut yv: Vec<f64> = Vec::new();
    let mut xv: Vec<f64> = Vec::new();
    let mut wv: Vec<f64> = Vec::new();
    for ((yi, xi), wi) in y.iter().zip(x.iter()).zip(weights.iter()) {
        let w_val = wi.unwrap_or(0.0);
        if w_val > 0.0 {
            yv.push(yi.unwrap_or(0.0));
            xv.push(xi.unwrap_or(0.0));
            wv.push(w_val);
        }
    }
    let n = yv.len() as f64;
    if n < 2.0 {
        return Ok(f64::NAN);
    }
    let sum_w: f64 = wv.iter().sum();
    if sum_w <= 0.0 {
        return Ok(f64::NAN);
    }
    let wn: Vec<f64> = wv.iter().map(|w| w / sum_w).collect();
    let ybar: f64 = yv.iter().zip(wn.iter()).map(|(yi, wi)| wi * yi).sum();
    let xbar: f64 = xv.iter().zip(wn.iter()).map(|(xi, wi)| wi * xi).sum();
    if xbar == 0.0 {
        return Ok(f64::NAN);
    }
    let rhat = ybar / xbar;
    let ev: Vec<f64> = yv
        .iter()
        .zip(xv.iter())
        .map(|(yi, xi)| yi - rhat * xi)
        .collect();
    let s2_e = weighted_s2(&ev, &wn);
    Ok((s2_e / (n * xbar.powi(2))) * srs.fpc(n, sum_w))
}

pub(crate) fn srs_variance_ratio_domain(
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
    srs: SrsRef,
) -> EResult<f64> {
    let fast = match (cont_pair(y, weights), x.null_count() == 0) {
        (Some((ys, ws)), true) => x.cont_slice().ok().map(|xs| (ys, ws, xs)),
        _ => None,
    };
    let (yv, xv, wv) = if let Some((ys, ws, xs)) = fast {
        let mut yv = Vec::new();
        let mut xv = Vec::new();
        let mut wv = Vec::new();
        for (i, m) in mask_flags(domain_mask).into_iter().enumerate() {
            // See srs_variance_mean_domain: zero-weight rows must not inflate n.
            if m && ws[i] > 0.0 {
                yv.push(ys[i]);
                xv.push(xs[i]);
                wv.push(ws[i]);
            }
        }
        (yv, xv, wv)
    } else {
        let mut yv = Vec::new();
        let mut xv = Vec::new();
        let mut wv = Vec::new();
        for (((yi, xi), wi), mi) in y
            .iter()
            .zip(x.iter())
            .zip(weights.iter())
            .zip(domain_mask.iter())
        {
            if let (Some(y_val), Some(x_val), Some(w_val), Some(true)) = (yi, xi, wi, mi)
                && w_val > 0.0
            {
                yv.push(y_val);
                xv.push(x_val);
                wv.push(w_val);
            }
        }
        (yv, xv, wv)
    };
    let n = yv.len() as f64;
    if n < 2.0 {
        return Ok(f64::NAN);
    }
    let sum_w: f64 = wv.iter().sum();
    if sum_w <= 0.0 {
        return Ok(f64::NAN);
    }
    let wn: Vec<f64> = wv.iter().map(|w| w / sum_w).collect();
    let ybar: f64 = yv.iter().zip(wn.iter()).map(|(yi, wi)| wi * yi).sum();
    let xbar: f64 = xv.iter().zip(wn.iter()).map(|(xi, wi)| wi * xi).sum();
    if xbar == 0.0 {
        return Ok(f64::NAN);
    }
    let rhat = ybar / xbar;
    let ev: Vec<f64> = yv
        .iter()
        .zip(xv.iter())
        .map(|(yi, xi)| yi - rhat * xi)
        .collect();
    let s2_e = weighted_s2(&ev, &wn);
    Ok((s2_e / (n * xbar.powi(2))) * srs.fpc(n, sum_w))
}

// ============================================================================
// Multivariate Taylor Variance (k score columns -> k x k covariance matrix)
// ============================================================================
/// k×k covariance of k score columns, as R's `vcov(svymean(...))`; per stratum
/// `fpc_h * m_h/(m_h-1) * Σ_a (T_ha - T̄_h)(T_ha - T̄_h)'`.
#[cfg(test)]
pub(crate) fn taylor_variance_matrix(
    score_columns: &[Float64Chunked], // k score columns, each length n
    strata: Option<&Column>,
    psu: Option<&Column>,
    fpc: Option<&Float64Chunked>,
    singleton_method: Option<&str>,
) -> EResult<Vec<Vec<f64>>> {
    taylor_variance_matrix_in(score_columns, strata, psu, fpc, singleton_method, None)
}

/// [`taylor_variance_matrix`] for estimates of one domain: `active` marks its
/// rows (see [`TaylorDesign::domain_rows`]). Stage 1 only.
pub(crate) fn taylor_variance_matrix_in(
    score_columns: &[Float64Chunked],
    strata: Option<&Column>,
    psu: Option<&Column>,
    fpc: Option<&Float64Chunked>,
    singleton_method: Option<&str>,
    active: Option<&[bool]>,
) -> EResult<Vec<Vec<f64>>> {
    let k = score_columns.len();
    if k == 0 {
        return Ok(vec![]);
    }
    if score_columns[0].is_empty() {
        return Ok(vec![vec![0.0; k]; k]);
    }
    let design = build_taylor_design(strata, psu, None, fpc, None, singleton_method)?;
    let cols: Vec<Vec<f64>> = score_columns
        .iter()
        .map(|c| c.iter().map(|s| s.unwrap_or(0.0)).collect())
        .collect();
    Ok(taylor_covariance_apply_in(&cols, &design, active))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::{abs_diff_eq, assert_abs_diff_eq, assert_relative_eq, relative_eq};

    /// A String design column for the variance kernels.
    fn scol(name: PlSmallStr, vals: &[&str]) -> Column {
        Column::from(StringChunked::from_slice(name, vals).into_series())
    }

    /// An integer design code column.
    fn icol(name: PlSmallStr, vals: &[i64]) -> Column {
        Column::from(Int64Chunked::from_slice(name, vals).into_series())
    }

    #[test]
    fn test_active_count_is_nonzero_weight_rows() {
        let w = Float64Chunked::from_slice_options(
            "w".into(),
            &[
                Some(1.0),
                Some(0.0),
                Some(-2.0),
                None,
                Some(f64::NAN),
                Some(3.0),
            ],
        );
        assert_eq!(active_count(&w, None), 3);
        let dom = BooleanChunked::from_slice("d".into(), &[true, true, true, true, true, false]);
        assert_eq!(active_count(&w, Some(&dom)), 2);
    }

    #[test]
    fn test_df_counts_psus_with_negative_weights() {
        // PSU "2" in stratum A has only a negative weight; it still represents
        // population units, so it counts toward df. PSU "2" in B is all zero.
        let strata = scol("s".into(), &["A", "A", "B", "B", "B"]);
        let psu = scol("p".into(), &["1", "2", "1", "2", "3"]);
        let w = Float64Chunked::from_slice("w".into(), &[1.0, -0.5, 2.0, 0.0, 1.0]);
        // Active PSUs: A{1,2}, B{1,3} = 4; strata 2; df = 2.
        assert_eq!(
            degrees_of_freedom(&w, Some(&strata), Some(&psu)).unwrap(),
            2
        );
    }

    /// Sparse, arbitrarily valued integer codes re-densify to exactly the string path's codes.
    #[test]
    fn test_integer_codes_match_string_path() {
        let y =
            Float64Chunked::from_slice("y".into(), &[1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 14.0, 16.0]);
        let w = Float64Chunked::from_slice("w".into(), &[1.0; 8]);
        let scores = scores_mean(&y, &w).unwrap();

        // String design: PSU labels "1"/"2" are reused across strata A and B.
        let strata_s = scol("s".into(), &["A", "A", "A", "A", "B", "B", "B", "B"]);
        let psu_s = scol("p".into(), &["1", "1", "2", "2", "1", "1", "2", "2"]);
        // SPARSE codes out of first-appearance order; psu codes encode the (stratum, psu) pairs.
        let strata_i = icol("s".into(), &[10, 10, 10, 10, 99, 99, 99, 99]);
        let psu_i = icol("p".into(), &[20, 20, 55, 55, 71, 71, 88, 88]);

        // Stratified + clustered
        let var_s = taylor_variance(
            &scores,
            Some(&strata_s),
            Some(&psu_s),
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let var_i = taylor_variance(
            &scores,
            Some(&strata_i),
            Some(&psu_i),
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            var_s.to_bits(),
            var_i.to_bits(),
            "strat+cluster: {var_s} vs {var_i}"
        );

        let df_s = degrees_of_freedom(&w, Some(&strata_s), Some(&psu_s)).unwrap();
        let df_i = degrees_of_freedom(&w, Some(&strata_i), Some(&psu_i)).unwrap();
        assert_eq!(df_s, df_i, "strat+cluster df");

        // Unstratified (PSU only): psu codes here are the plain PSU identity.
        let psu_flat_s = scol("p".into(), &["a", "a", "b", "b", "c", "c", "d", "d"]);
        let psu_flat_i = icol("p".into(), &[7, 7, 3, 3, 9, 9, 1, 1]);
        let uv_s =
            taylor_variance(&scores, None, Some(&psu_flat_s), None, None, None, None).unwrap();
        let uv_i =
            taylor_variance(&scores, None, Some(&psu_flat_i), None, None, None, None).unwrap();
        assert_eq!(
            uv_s.to_bits(),
            uv_i.to_bits(),
            "unstratified: {uv_s} vs {uv_i}"
        );

        // Stratified, no PSU
        let sv_s = taylor_variance(&scores, Some(&strata_s), None, None, None, None, None).unwrap();
        let sv_i = taylor_variance(&scores, Some(&strata_i), None, None, None, None, None).unwrap();
        assert_eq!(
            sv_s.to_bits(),
            sv_i.to_bits(),
            "stratified-only: {sv_s} vs {sv_i}"
        );
    }

    /// The covariance diagonal matches the variance, and c'Vc the combined score's variance.
    #[test]
    fn test_taylor_covariance_matches_variance() {
        let a = vec![1.0, -2.0, 3.0, -4.0, 10.0, -12.0, 14.0, -16.0];
        let b = vec![0.5, 2.5, -1.5, 4.0, -3.0, 6.0, -7.0, 8.0];
        let combined: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();

        let strata = scol("s".into(), &["A", "A", "A", "A", "B", "B", "B", "B"]);
        let psu = scol("p".into(), &["1", "1", "2", "2", "1", "1", "2", "2"]);

        let designs = [
            build_taylor_design(Some(&strata), Some(&psu), None, None, None, None).unwrap(),
            build_taylor_design(Some(&strata), None, None, None, None, None).unwrap(),
            build_taylor_design(None, Some(&psu), None, None, None, None).unwrap(),
            build_taylor_design(None, None, None, None, None, None).unwrap(),
        ];
        for (i, d) in designs.iter().enumerate() {
            let cov = taylor_covariance_apply(&[a.clone(), b.clone()], d);
            let var_a = taylor_variance_apply(&a, d);
            let var_b = taylor_variance_apply(&b, d);
            let var_c = taylor_variance_apply(&combined, d);
            assert!(
                relative_eq!(cov[0][0], var_a, epsilon = 1e-10, max_relative = 1e-10),
                "diag a, design {i}"
            );
            assert!(
                relative_eq!(cov[1][1], var_b, epsilon = 1e-10, max_relative = 1e-10),
                "diag b, design {i}"
            );
            assert_eq!(
                cov[0][1].to_bits(),
                cov[1][0].to_bits(),
                "symmetry, design {i}"
            );
            let var_from_cov = cov[0][0] - 2.0 * cov[0][1] + cov[1][1];
            assert!(
                relative_eq!(var_from_cov, var_c, epsilon = 1e-10, max_relative = 1e-10),
                "contrast variance, design {i}: {var_from_cov} vs {var_c}"
            );
        }
    }

    /// The covariance diagonal reproduces the centred variance with a one-PSU stratum.
    #[test]
    fn test_taylor_covariance_singleton_center() {
        let a = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        let b = vec![2.0, 1.0, -1.0, 0.5, -0.5];
        let combined: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        // Stratum B has a single PSU.
        let strata = scol("s".into(), &["A", "A", "A", "A", "B"]);
        let psu = scol("p".into(), &["1", "1", "2", "2", "1"]);
        let d = build_taylor_design(Some(&strata), Some(&psu), None, None, None, Some("center"))
            .unwrap();
        let cov = taylor_covariance_apply(&[a.clone(), b.clone()], &d);
        let var_a = taylor_variance_apply(&a, &d);
        let var_c = taylor_variance_apply(&combined, &d);
        assert_relative_eq!(cov[0][0], var_a, epsilon = 1e-10, max_relative = 1e-10);
        let var_from_cov = cov[0][0] + 2.0 * cov[0][1] + cov[1][1];
        assert_relative_eq!(var_from_cov, var_c, epsilon = 1e-10, max_relative = 1e-10);
    }

    #[test]
    fn test_taylor_covariance_domains_in_disjoint_strata() {
        let a = vec![1.0, -2.0, 3.0, -4.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0, 0.0, 10.0, -12.0, 14.0, -16.0];
        let combined: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
        let strata = scol("s".into(), &["A", "A", "A", "A", "B", "B", "B", "B"]);
        let psu = scol("p".into(), &["1", "1", "2", "2", "1", "1", "2", "2"]);
        let designs = [
            build_taylor_design(Some(&strata), Some(&psu), None, None, None, None).unwrap(),
            build_taylor_design(Some(&strata), None, None, None, None, None).unwrap(),
        ];
        for (i, d) in designs.iter().enumerate() {
            let cov = taylor_covariance_apply(&[a.clone(), b.clone()], d);
            assert_eq!(cov[0][1].to_bits(), 0.0f64.to_bits(), "design {i}");
            assert_eq!(cov[1][0].to_bits(), 0.0f64.to_bits(), "design {i}");
            let var_c = taylor_variance_apply(&combined, d);
            let var_from_cov = cov[0][0] + cov[1][1];
            assert!(
                relative_eq!(var_from_cov, var_c, epsilon = 1e-10, max_relative = 1e-10),
                "design {i}"
            );
        }
    }

    #[test]
    fn test_taylor_covariance_nonfinite_score_reaches_absent_domain() {
        let a = vec![f64::INFINITY, -2.0, 3.0, -4.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.0, 0.0, 10.0, -12.0, 14.0, -16.0];
        let strata = scol("s".into(), &["A", "A", "A", "A", "B", "B", "B", "B"]);
        let psu = scol("p".into(), &["1", "1", "2", "2", "1", "1", "2", "2"]);
        let designs = [
            build_taylor_design(Some(&strata), Some(&psu), None, None, None, None).unwrap(),
            build_taylor_design(Some(&strata), None, None, None, None, None).unwrap(),
        ];
        for (i, d) in designs.iter().enumerate() {
            let cov = taylor_covariance_apply(&[a.clone(), b.clone()], d);
            assert!(cov[0][1].is_nan(), "design {i}: {}", cov[0][1]);
        }
    }

    #[test]
    fn test_weighted_quantile_uniform_weights() {
        // cdf = [0.2, 0.4, 0.6, 0.8, 1.0]; p = 0.5 lies between y = 2 and y = 3.
        // R svyquantile: qrule="math" gives 3, qrule="hf4" gives 2.5.
        let y = [1.0, 2.0, 3.0, 4.0, 5.0];
        let cdf = [0.2, 0.4, 0.6, 0.8, 1.0];
        let median = |method| weighted_quantile(&y, &cdf, 0.5, method);

        assert_eq!(median(SvyQuantileMethod::Lower), 2.0);
        assert_eq!(median(SvyQuantileMethod::Higher), 3.0);
        assert_eq!(median(SvyQuantileMethod::Middle), 2.5);
        assert_eq!(median(SvyQuantileMethod::Nearest), 2.0);
        assert_abs_diff_eq!(median(SvyQuantileMethod::Linear), 2.5, epsilon = 1e-15);
    }

    #[test]
    fn test_weighted_quantile_nonuniform_weights() {
        // cdf = [0.25, 0.75, 1.0]: p = 0.5 is equidistant, so Nearest takes the lower value.
        // R svyquantile: qrule="math" gives 2, qrule="hf4" gives 1.5.
        let y = [1.0, 2.0, 3.0];
        let cdf = [0.25, 0.75, 1.0];
        let median = |method| weighted_quantile(&y, &cdf, 0.5, method);

        assert_eq!(median(SvyQuantileMethod::Lower), 1.0);
        assert_eq!(median(SvyQuantileMethod::Higher), 2.0);
        assert_eq!(median(SvyQuantileMethod::Middle), 1.5);
        assert_eq!(median(SvyQuantileMethod::Nearest), 1.0);
        assert_eq!(median(SvyQuantileMethod::Linear), 1.5);
    }

    #[test]
    fn test_weighted_quantile_lower_boundary() {
        // cdf = [0.8, 0.9, 1.0]: the median is 1.0 for every method; Linear must not extrapolate.
        let y = vec![1.0, 2.0, 3.0];
        let cdf = vec![0.8, 0.9, 1.0];

        assert_eq!(
            weighted_quantile(&y, &cdf, 0.5, SvyQuantileMethod::Higher),
            1.0
        );
        assert_eq!(
            weighted_quantile(&y, &cdf, 0.5, SvyQuantileMethod::Lower),
            1.0
        );
        assert_eq!(
            weighted_quantile(&y, &cdf, 0.5, SvyQuantileMethod::Middle),
            1.0
        );
        assert_eq!(
            weighted_quantile(&y, &cdf, 0.5, SvyQuantileMethod::Nearest),
            1.0
        );
        let linear = weighted_quantile(&y, &cdf, 0.5, SvyQuantileMethod::Linear);
        assert_eq!(linear, 1.0);
        assert!(
            (1.0..=3.0).contains(&linear),
            "Linear must stay in data range"
        );

        // Upper boundary: p >= cdf[n-1] must return the maximum.
        for m in [
            SvyQuantileMethod::Higher,
            SvyQuantileMethod::Lower,
            SvyQuantileMethod::Middle,
            SvyQuantileMethod::Nearest,
            SvyQuantileMethod::Linear,
        ] {
            assert_eq!(weighted_quantile(&y, &cdf, 1.0, m), 3.0);
        }
    }

    #[test]
    fn test_psu_labels_reused_across_strata() {
        // PSU labels reused in both strata must match unique labels; by hand 1.25 (SE 1.118).
        let y =
            Float64Chunked::from_slice("y".into(), &[1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 14.0, 16.0]);
        let w = Float64Chunked::from_slice("w".into(), &[1.0; 8]);
        let strata = scol("s".into(), &["A", "A", "A", "A", "B", "B", "B", "B"]);
        let psu_reused = scol("p".into(), &["1", "1", "2", "2", "1", "1", "2", "2"]);
        let psu_unique = scol(
            "p".into(),
            &["A1", "A1", "A2", "A2", "B1", "B1", "B2", "B2"],
        );

        let scores = scores_mean(&y, &w).unwrap();
        let var_reused = taylor_variance(
            &scores,
            Some(&strata),
            Some(&psu_reused),
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let var_unique = taylor_variance(
            &scores,
            Some(&strata),
            Some(&psu_unique),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        assert!(
            abs_diff_eq!(var_reused, 1.25, epsilon = 1e-12),
            "got {}",
            var_reused
        );
        assert!(
            abs_diff_eq!(var_unique, 1.25, epsilon = 1e-12),
            "got {}",
            var_unique
        );
        assert_abs_diff_eq!(var_reused, var_unique, epsilon = 1e-12);

        // The full covariance matrix path must agree with the scalar path.
        let cov_reused = taylor_variance_matrix(
            std::slice::from_ref(&scores),
            Some(&strata),
            Some(&psu_reused),
            None,
            None,
        )
        .unwrap();
        assert!(
            abs_diff_eq!(cov_reused[0][0], 1.25, epsilon = 1e-12),
            "got {}",
            cov_reused[0][0]
        );
    }

    #[test]
    fn test_unstratified_two_stage_fpc_stage2() {
        // Two-stage with FPC equals one constant stratum; by hand 2.8 = 2.6667 + 0.1333.
        let y = Float64Chunked::from_slice(
            "y".into(),
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        );
        let w = Float64Chunked::from_slice("w".into(), &[1.0; 12]);
        let psu = scol(
            "p".into(),
            &[
                "p1", "p1", "p1", "p1", "p2", "p2", "p2", "p2", "p3", "p3", "p3", "p3",
            ],
        );
        let ssu = scol(
            "u".into(),
            &[
                "s1", "s1", "s2", "s2", "s1", "s1", "s2", "s2", "s1", "s1", "s2", "s2",
            ],
        );
        let fpc = Float64Chunked::from_slice("f".into(), &[0.5; 12]);
        let fpc_ssu = Float64Chunked::from_slice("f2".into(), &[0.8; 12]);
        let strata_const = scol("s".into(), &["S"; 12]);

        let scores = scores_mean(&y, &w).unwrap();
        let var_nostrata = taylor_variance(
            &scores,
            None,
            Some(&psu),
            Some(&ssu),
            Some(&fpc),
            Some(&fpc_ssu),
            None,
        )
        .unwrap();
        let var_conststrata = taylor_variance(
            &scores,
            Some(&strata_const),
            Some(&psu),
            Some(&ssu),
            Some(&fpc),
            Some(&fpc_ssu),
            None,
        )
        .unwrap();

        assert!(
            abs_diff_eq!(var_nostrata, 2.8, epsilon = 1e-12),
            "got {}",
            var_nostrata
        );
        assert_abs_diff_eq!(var_nostrata, var_conststrata, epsilon = 1e-12);
    }

    #[test]
    fn test_srs_variance_ratio_zero_weights() {
        let y = Float64Chunked::from_slice("y".into(), &[1.0, 2.0, 3.0]);
        let x = Float64Chunked::from_slice("x".into(), &[1.0, 1.0, 1.0]);
        let w = Float64Chunked::from_slice("w".into(), &[0.0, 0.0, 0.0]);
        let v =
            srs_variance_ratio(&y, &x, &w, SrsRef::WithoutReplacement { pop_total: None }).unwrap();
        assert!(v.is_nan());
    }

    // ------------------------------------------------------------------
    // SRS reference for the design effect
    // ------------------------------------------------------------------

    fn srs_fixture() -> (Float64Chunked, Float64Chunked) {
        let y = Float64Chunked::from_slice("y".into(), &[10.0, 12.0, 9.0, 15.0, 11.0, 13.0]);
        // sum(w) = 600, n = 6, so the sampling fraction is 1%.
        let w = Float64Chunked::from_slice("w".into(), &[100.0; 6]);
        (y, w)
    }

    #[test]
    fn test_wr_and_wor_differ_by_exactly_the_fpc() {
        let (y, w) = srs_fixture();
        let wr = srs_variance_mean(&y, &w, SrsRef::WithReplacement).unwrap();
        let wor =
            srs_variance_mean(&y, &w, SrsRef::WithoutReplacement { pop_total: None }).unwrap();
        let expected_fpc = 1.0 - 6.0 / 600.0;
        assert!(
            abs_diff_eq!(wor / wr, expected_fpc, epsilon = 1e-12),
            "{wor} / {wr}"
        );
    }

    /// What makes it usable with normalized or calibrated weights.
    #[test]
    fn test_wr_reference_is_scale_invariant() {
        let (y, _) = srs_fixture();
        let mut prev: Option<f64> = None;
        for scale in [1.0, 0.5, 1e-3, 400.0] {
            let w: Vec<f64> = vec![100.0 * scale; 6];
            let v = srs_variance_mean(
                &y,
                &Float64Chunked::from_slice("w".into(), &w),
                SrsRef::WithReplacement,
            )
            .unwrap();
            if let Some(p) = prev {
                assert!(
                    abs_diff_eq!(v, p, epsilon = 1e-12),
                    "scale {scale} changed the reference"
                );
            }
            prev = Some(v);
        }
    }

    #[test]
    fn test_wor_reference_moves_with_the_weight_scale() {
        let (y, w) = srs_fixture();
        let full =
            srs_variance_mean(&y, &w, SrsRef::WithoutReplacement { pop_total: None }).unwrap();
        let halved = Float64Chunked::from_slice("w".into(), &[50.0; 6]);
        let half =
            srs_variance_mean(&y, &halved, SrsRef::WithoutReplacement { pop_total: None }).unwrap();
        assert!(
            (full - half).abs() > 1e-9,
            "expected the FPC to move: {full} vs {half}"
        );
    }

    #[test]
    fn test_declared_pop_total_overrides_the_weight_sum() {
        let (y, _) = srs_fixture();
        // Weights normalized to sum to n, while the design knows N = 600.
        let norm = Float64Chunked::from_slice("w".into(), &[1.0; 6]);
        let with_pop = srs_variance_mean(
            &y,
            &norm,
            SrsRef::WithoutReplacement {
                pop_total: Some(600.0),
            },
        )
        .unwrap();
        let wr = srs_variance_mean(&y, &norm, SrsRef::WithReplacement).unwrap();
        assert_abs_diff_eq!(with_pop / wr, 1.0 - 6.0 / 600.0, epsilon = 1e-12);
    }

    /// Weights normalized to n sit within ~1e-12 of it: that must error, not give a ~1e16 deff.
    #[test]
    fn test_fpc_guard_survives_float_noise_around_n() {
        let (y, _) = srs_fixture();
        for delta in [-1e-11, -1e-13, 0.0, 1e-13, 1e-11] {
            let each = (6.0 + delta) / 6.0;
            let w = Float64Chunked::from_slice("w".into(), &[each; 6]);
            let got =
                srs_variance_mean(&y, &w, SrsRef::WithoutReplacement { pop_total: None }).unwrap();
            assert!(
                got.is_nan(),
                "sum(w) = n{delta:+e} should be degenerate, got {got}"
            );
        }
    }

    #[test]
    fn test_near_census_design_is_still_computable() {
        let y = Float64Chunked::from_slice("y".into(), &[10.0, 12.0, 9.0, 15.0, 11.0, 13.0]);
        // 6 sampled from a population of 6.6 -- FPC ~0.09, extreme but real.
        let w = Float64Chunked::from_slice("w".into(), &[1.1; 6]);
        let v = srs_variance_mean(&y, &w, SrsRef::WithoutReplacement { pop_total: None }).unwrap();
        assert!(v > 0.0 && v.is_finite(), "got {v}");
    }

    #[test]
    fn test_degenerate_fpc_is_nan_for_the_python_layer_to_diagnose() {
        let (y, _) = srs_fixture();
        for w in [vec![1.0; 6], vec![1.0 / 6.0; 6]] {
            let wc = Float64Chunked::from_slice("w".into(), &w);
            let v =
                srs_variance_mean(&y, &wc, SrsRef::WithoutReplacement { pop_total: None }).unwrap();
            assert!(v.is_nan(), "expected a degenerate reference, got {v}");
        }
        // The with-replacement reference is unaffected.
        let wc = Float64Chunked::from_slice("w".into(), &[1.0; 6]);
        assert!(srs_variance_mean(&y, &wc, SrsRef::WithReplacement).unwrap() > 0.0);
    }

    #[test]
    fn test_quantile_method_from_str() {
        assert_eq!(
            SvyQuantileMethod::parse("lower").unwrap(),
            SvyQuantileMethod::Lower
        );
        assert_eq!(
            SvyQuantileMethod::parse("HIGHER").unwrap(),
            SvyQuantileMethod::Higher
        );
        assert_eq!(
            SvyQuantileMethod::parse("Linear").unwrap(),
            SvyQuantileMethod::Linear
        );
        let err = SvyQuantileMethod::parse("unknown").unwrap_err();
        assert_eq!(
            err.to_string(),
            "unknown quantile method 'unknown'; expected 'lower', 'higher', 'middle', 'nearest' or 'linear'"
        );
    }

    /// Must agree on every design shape, including a `where=` filter's zero-weight rows.
    #[test]
    fn test_df_from_design_matches_column_path() {
        let strata_i = icol("s".into(), &[10, 10, 10, 10, 99, 99, 99, 99]);
        let psu_i = icol("p".into(), &[7, 7, 3, 3, 7, 7, 3, 3]);
        let strata_s = scol("s".into(), &["A", "A", "A", "A", "B", "B", "B", "B"]);
        let psu_s = scol("p".into(), &["1", "1", "2", "2", "1", "1", "2", "2"]);

        let w_all = Float64Chunked::from_slice("w".into(), &[1.0; 8]);
        // Zero weights: two PSUs drop out, and stratum B loses one entirely.
        let w_some =
            Float64Chunked::from_slice("w".into(), &[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);

        for (label, strata, psu) in [
            ("integer codes", Some(&strata_i), Some(&psu_i)),
            ("string labels", Some(&strata_s), Some(&psu_s)),
            ("stratified, no psu", Some(&strata_i), None),
            ("psu, no strata", None, Some(&psu_i)),
            ("no design", None, None),
        ] {
            for (wlabel, w) in [("all positive", &w_all), ("some zero", &w_some)] {
                let design = build_taylor_design(strata, psu, None, None, None, None).unwrap();
                let from_design = degrees_of_freedom_from_design(w, &design, None);
                let from_cols = degrees_of_freedom(w, strata, psu).unwrap();
                assert_eq!(
                    from_design, from_cols,
                    "df mismatch for {label} / {wlabel}: design path {from_design} \
                     vs column path {from_cols}"
                );
            }
        }
    }

    /// Must give exactly the round-trip's bits, on the fast path and the null fallback.
    #[test]
    fn test_scores_mean_arr_matches_chunked_roundtrip() {
        let w = Float64Chunked::from_slice("w".into(), &[1.0, 2.5, 0.5, 3.0, 1.25]);

        let dense = Float64Chunked::from_slice("y".into(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let with_nulls = Float64Chunked::from_slice_options(
            "y".into(),
            &[Some(1.0), None, Some(3.0), Some(4.0), None],
        );

        for (label, y) in [("contiguous", &dense), ("with nulls", &with_nulls)] {
            let roundtrip: Vec<f64> = scores_mean(y, &w)
                .unwrap()
                .iter()
                .map(|s| s.unwrap_or(0.0))
                .collect();
            let direct = scores_mean_arr(y, &w).unwrap();
            assert_eq!(
                roundtrip.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                direct.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "scores_mean_arr diverged from the chunked round-trip ({label})"
            );
        }
    }

    #[test]
    fn test_u32_codes_densify_like_the_hash_path() {
        let vals = [
            Some(40u32),
            Some(7),
            None,
            Some(40),
            Some(3),
            Some(7),
            None,
            Some(3),
        ];
        let as_u32 =
            Column::from(UInt32Chunked::from_slice_options("c".into(), &vals).into_series());
        let as_i64: Vec<Option<i64>> = vals.iter().map(|v| v.map(i64::from)).collect();
        let as_i64 =
            Column::from(Int64Chunked::from_slice_options("c".into(), &as_i64).into_series());
        let expected = (vec![0, 1, u32::MAX, 0, 2, 1, u32::MAX, 2], 3);
        assert_eq!(densify_int_codes(&as_u32).unwrap(), expected);
        assert_eq!(densify_int_codes(&as_i64).unwrap(), expected);

        // Too sparse for a table: the hash path, same first-appearance codes.
        let sparse = Column::from(
            UInt32Chunked::from_slice("c".into(), &[u32::MAX - 1, 5, 5]).into_series(),
        );
        assert_eq!(densify_int_codes(&sparse).unwrap(), (vec![0, 1, 1], 2));
    }

    #[test]
    fn test_psu_codes_shared_across_strata_count_in_each() {
        let strata = icol("s".into(), &[10, 10, 10, 10, 99, 99, 99, 99]);
        let psu = icol("p".into(), &[7, 7, 3, 3, 7, 7, 3, 3]);
        let w = Float64Chunked::from_slice("w".into(), &[1.0; 8]);
        let design =
            build_taylor_design(Some(&strata), Some(&psu), None, None, None, None).unwrap();
        assert_eq!(design.psu_per_stratum, Some(vec![vec![0, 1], vec![0, 1]]));
        assert_eq!(degrees_of_freedom_from_design(&w, &design, None), 2);

        let nested = icol("p".into(), &[7, 7, 3, 3, 8, 8, 4, 4]);
        let design =
            build_taylor_design(Some(&strata), Some(&nested), None, None, None, None).unwrap();
        assert_eq!(design.psu_per_stratum, Some(vec![vec![0, 1], vec![2, 3]]));
        assert_eq!(degrees_of_freedom_from_design(&w, &design, None), 2);
    }

    #[test]
    fn test_mask_flags_read_null_as_out() {
        let a = BooleanChunked::from_slice_options("m".into(), &[Some(true), None, Some(false)]);
        let b = BooleanChunked::from_slice("m".into(), &[true, false]);
        let mut chunked = a.clone();
        chunked.append(&b).unwrap();
        assert_eq!(mask_flags(&a), vec![true, false, false]);
        assert_eq!(mask_flags(&chunked), vec![true, false, false, true, false]);
        let w = Float64Chunked::from_slice("w".into(), &[1.0, 1.0, 1.0, 2.0, 0.0]);
        assert_eq!(active_count(&w, Some(&chunked)), 2);
    }

    /// Stratum 1: stage 1 = 8, stage 2 = 0.5 * (2 + 8) = 5. Stratum 2: 27 + 0.25 * 2.
    /// R svytotal over svydesign(ids=~psu+ssu, strata=~st, fpc=~N1+N2) agrees.
    #[test]
    fn two_stage_variance_and_covariance_match_hand_calculation() {
        let scores = [1.0, 3.0, 2.0, 6.0, 1.0, 1.0, 3.0, 5.0];
        let fpc = [0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75];
        let cases = [
            (None, 4, 13.0),
            (Some(&[1, 1, 1, 1, 2, 2, 2, 2][..]), 8, 40.5),
        ];
        for (strata, n, expected) in cases {
            let strata = strata.map(|s| icol("st".into(), s));
            // Integer PSU codes arrive already nested in strata.
            let psu = icol("psu".into(), &[1, 1, 2, 2, 3, 3, 4, 4][..n]);
            let ssu = icol("ssu".into(), &[1, 2, 1, 2, 1, 2, 1, 2][..n]);
            let fpc = Float64Chunked::from_slice("fpc".into(), &fpc[..n]);
            let fpc_ssu = Float64Chunked::from_slice("fpc_ssu".into(), &vec![0.5; n]);
            let d = build_taylor_design(
                strata.as_ref(),
                Some(&psu),
                Some(&ssu),
                Some(&fpc),
                Some(&fpc_ssu),
                None,
            )
            .unwrap();
            let s: Vec<f64> = scores[..n].to_vec();
            let s2: Vec<f64> = s.iter().map(|v| 2.0 * v).collect();

            approx::assert_abs_diff_eq!(taylor_variance_apply(&s, &d), expected, epsilon = 1e-12);
            let cov = taylor_covariance_apply(&[s, s2], &d);
            for (got, want) in cov.iter().flatten().zip([1.0, 2.0, 2.0, 4.0]) {
                approx::assert_abs_diff_eq!(*got, want * expected, epsilon = 1e-12);
            }
        }
    }
}
