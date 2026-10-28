//! Survey GLM by IRLS with a bread-meat-bread sandwich, matching R `svyglm` numerically.
//!
//! - Bread from the converged Fisher information; meat from PSU score totals centred within stratum, scaled m/(m-1).
//! - Out-of-domain rows contribute 0, but strata and PSUs are the full design's, as R's subset().

#![allow(non_snake_case)]

use crate::regression::error::{RResult, RegressionError};
use faer::prelude::Solve;
use faer::{Mat, MatRef, Side};

use polars::prelude::*;

use crate::categorical::ranktest::probit;
use crate::estimation::calib_sweep::CalibSweep;
use crate::estimation::taylor::{
    SingletonMethod, SingletonRule, design_col_codes, design_pair_codes,
};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

// ============================================================================
// Enums & Config
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Family {
    Gaussian,
    Binomial,
    Poisson,
    Gamma,
    InverseGaussian,
    /// Negative binomial at a known theta, Var(mu) = mu + mu^2/theta; `negbin` estimates theta around it.
    NegativeBinomial(f64),
}

impl Family {
    pub(crate) fn from_str(s: &str, theta: Option<f64>) -> RResult<Self> {
        match s.to_lowercase().as_str() {
            "gaussian" => Ok(Family::Gaussian),
            "binomial" => Ok(Family::Binomial),
            "poisson" => Ok(Family::Poisson),
            "gamma" => Ok(Family::Gamma),
            "inversegaussian" | "inverse_gaussian" => Ok(Family::InverseGaussian),
            "negativebinomial" | "negative_binomial" => {
                let th = theta.ok_or(RegressionError::ThetaMissing)?;
                if !(th.is_finite() && th > 0.0) {
                    return Err(RegressionError::ThetaInvalid(th));
                }
                Ok(Family::NegativeBinomial(th))
            }
            _ => Err(RegressionError::UnsupportedFamily(s.to_string())),
        }
    }

    pub(crate) fn variance(&self, mu: f64) -> f64 {
        match self {
            Family::Gaussian => 1.0,
            Family::Binomial => mu * (1.0 - mu),
            Family::Poisson => mu,
            Family::Gamma => mu * mu,
            Family::InverseGaussian => mu * mu * mu,
            Family::NegativeBinomial(theta) => mu + mu * mu / theta,
        }
    }

    /// R's `family$initialize`, on the prior-weight scale: on a flat surface IRLS stops where its start leads.
    fn initial_mu(&self, y: f64, w: f64) -> f64 {
        let eps = 1e-10;
        match self {
            Family::Binomial => (w * y + 0.5) / (w + 1.0),
            Family::Poisson => y + 0.1,
            Family::Gamma | Family::InverseGaussian => y.max(eps),
            Family::Gaussian => y,
            // MASS::negative.binomial's initialize: y + (y == 0)/6.
            Family::NegativeBinomial(_) => {
                if y == 0.0 {
                    1.0 / 6.0
                } else {
                    y
                }
            }
        }
    }

    /// R's `family$dev.resids`, so sum_i w_i d(y_i, mu_i) is svyglm's deviance on the same weight scale.
    fn unit_deviance(&self, y: f64, mu: f64) -> f64 {
        match self {
            Family::Gaussian => (y - mu).powi(2),
            Family::Binomial => {
                let mu_c = mu.clamp(1e-10, 1.0 - 1e-10);
                let t1 = if y > 0.0 { y * (y / mu_c).ln() } else { 0.0 };
                let t2 = if y < 1.0 {
                    (1.0 - y) * ((1.0 - y) / (1.0 - mu_c)).ln()
                } else {
                    0.0
                };
                2.0 * (t1 + t2)
            }
            Family::Poisson => {
                let mu_c = mu.max(1e-10);
                let t = if y > 0.0 { y * (y / mu_c).ln() } else { 0.0 };
                2.0 * (t - (y - mu_c))
            }
            Family::Gamma => {
                let mu_c = mu.max(1e-10);
                let y_c = y.max(1e-10);
                2.0 * (-(y_c / mu_c).ln() + (y - mu_c) / mu_c)
            }
            Family::InverseGaussian => (y - mu).powi(2) / (y.max(1e-10) * mu * mu),
            // MASS::negative.binomial's dev.resids.
            Family::NegativeBinomial(theta) => {
                let mu_c = mu.max(1e-10);
                2.0 * (y * (y.max(1.0) / mu_c).ln()
                    - (y + theta) * ((y + theta) / (mu_c + theta)).ln())
            }
        }
    }
}

/// -qnorm(.Machine$double.eps): the eta bound R's probit linkinv applies.
const PROBIT_ETA_MAX: f64 = 8.125_890_664_701_904;

/// -qcauchy(.Machine$double.eps): the eta bound R's cauchit linkinv applies.
const CAUCHIT_ETA_MAX: f64 = 1.433_540_284_805_664_8e15;

/// Standard Cauchy CDF; outside [-1, 1] it uses `atan(1/x)` as R's pcauchy, since `atan(x)/pi + 0.5` cancels.
fn cauchy_cdf(x: f64) -> f64 {
    use std::f64::consts::PI;
    if x > 1.0 {
        1.0 - (1.0 / x).atan() / PI
    } else if x < -1.0 {
        (-1.0 / x).atan() / PI
    } else {
        0.5 + x.atan() / PI
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Link {
    Identity,
    Logit,
    Probit,
    Cauchit,
    Cloglog,
    Log,
    Sqrt,
    Inverse,
    InverseSquared,
}

// Phi(x) = erfc(-x/sqrt2)/2 on the tail side, so mu near 0 or 1 stays precise; ~5e-15 of R's pnorm.
// erfc: the all-positive confluent series below 1, a modified-Lentz continued fraction above.

const M_1_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
const SQRT_PI: f64 = 1.772_453_850_905_516;

fn norm_pdf(x: f64) -> f64 {
    M_1_SQRT_2PI * (-0.5 * x * x).exp()
}

fn erfc(z: f64) -> f64 {
    if z < 0.0 {
        return 2.0 - erfc(-z);
    }

    if z < 1.0 {
        // erf(z) = (2z/sqrt(pi)) e^{-z^2} sum_n (2z^2)^n / (1*3*...*(2n+1))
        let z2 = z * z;
        let mut term = 2.0 * z / SQRT_PI * (-z2).exp();
        let mut sum = term;
        for n in 1..64 {
            term *= 2.0 * z2 / (2 * n + 1) as f64;
            sum += term;
            if term <= sum * 1e-18 {
                break;
            }
        }
        return 1.0 - sum;
    }

    // erfc(z) = e^{-z^2}/sqrt(pi) * 1/(z + (1/2)/(z + 1/(z + (3/2)/(z + ...))))
    const TINY: f64 = 1e-300;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / z;
    let mut f = d;
    for n in 1..400 {
        let a = n as f64 / 2.0;
        d = z + a * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = z + a / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 1e-17 {
            break;
        }
    }
    (-z * z).exp() / SQRT_PI * f
}

pub(crate) fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x * std::f64::consts::FRAC_1_SQRT_2)
}

impl Link {
    pub(crate) fn from_str(s: &str) -> RResult<Self> {
        match s.to_lowercase().as_str() {
            "identity" => Ok(Link::Identity),
            "logit" => Ok(Link::Logit),
            "probit" => Ok(Link::Probit),
            "cauchit" => Ok(Link::Cauchit),
            "cloglog" => Ok(Link::Cloglog),
            "log" => Ok(Link::Log),
            "sqrt" => Ok(Link::Sqrt),
            "inverse" => Ok(Link::Inverse),
            "inverse_squared" => Ok(Link::InverseSquared),
            _ => Err(RegressionError::UnsupportedLink(s.to_string())),
        }
    }

    pub(crate) fn link(&self, mu: f64) -> f64 {
        match self {
            Link::Identity => mu,
            Link::Logit => (mu / (1.0 - mu)).ln(),
            Link::Probit => probit(mu),
            // As R's qcauchy: tan(pi*(mu - 1/2)) is catastrophic as mu nears 0 or 1.
            Link::Cauchit => -1.0 / (std::f64::consts::PI * mu).tan(),
            Link::Cloglog => (-((1.0 - mu).max(1e-10).ln())).max(1e-10).ln(),
            Link::Log => mu.max(1e-10).ln(),
            Link::Sqrt => mu.max(0.0).sqrt(),
            Link::Inverse => 1.0 / mu,
            Link::InverseSquared => 1.0 / (mu * mu),
        }
    }

    pub(crate) fn inverse(&self, eta: f64) -> f64 {
        match self {
            Link::Identity => eta,
            Link::Logit => {
                if eta >= 0.0 {
                    1.0 / (1.0 + (-eta).exp())
                } else {
                    let e = eta.exp();
                    e / (1.0 + e)
                }
            }

            // R's make.link clamps (eta at +/-qnorm(eps), mu in [eps, 1-eps]) keep the binomial variance positive.
            Link::Probit => norm_cdf(eta.clamp(-PROBIT_ETA_MAX, PROBIT_ETA_MAX)),
            Link::Cauchit => {
                let e = eta.clamp(-CAUCHIT_ETA_MAX, CAUCHIT_ETA_MAX);
                cauchy_cdf(e)
            }
            Link::Cloglog => {
                let m = -(-(eta.min(700.0).exp())).exp_m1();
                m.clamp(f64::EPSILON, 1.0 - f64::EPSILON)
            }
            Link::Log => eta.clamp(-30.0, 30.0).exp(),
            Link::Sqrt => eta * eta,
            Link::Inverse => 1.0 / eta,
            Link::InverseSquared => 1.0 / eta.sqrt(),
        }
    }

    /// d2μ/dη2, for the negative binomial's joint bread, so only for that family's links.
    pub(crate) fn mu_eta2(&self, mu: f64, eta: f64) -> f64 {
        match self {
            Link::Identity => 0.0,
            Link::Logit => mu * (1.0 - mu) * (1.0 - 2.0 * mu),
            Link::Probit => -eta * norm_pdf(eta),
            Link::Cauchit => {
                let d = 1.0 + eta * eta;
                -2.0 * eta / (std::f64::consts::PI * d * d)
            }
            Link::Cloglog => {
                let e = eta.min(700.0).exp();
                e * (-e).exp() * (1.0 - e)
            }
            Link::Log => mu,
            Link::Sqrt => 2.0,
            Link::Inverse => 2.0 * mu * mu * mu,
            Link::InverseSquared => 0.75 * mu.powi(5),
        }
    }

    /// dμ/dη. Probit and cloglog are the arms that need eta rather than mu.
    pub(crate) fn mu_eta(&self, mu: f64, eta: f64) -> f64 {
        match self {
            Link::Identity => 1.0,
            Link::Logit => mu * (1.0 - mu),
            Link::Probit => norm_pdf(eta).max(f64::EPSILON),
            Link::Cauchit => (1.0 / (std::f64::consts::PI * (1.0 + eta * eta))).max(f64::EPSILON),
            Link::Cloglog => {
                let e = eta.min(700.0).exp();
                (e * (-e).exp()).max(f64::EPSILON)
            }
            Link::Log => mu,
            Link::Sqrt => 2.0 * eta,
            Link::Inverse => -(mu * mu),
            Link::InverseSquared => -0.5 * mu.powi(3),
        }
    }
}

// ============================================================================
// Numerics: Kahan summation
// ============================================================================

#[derive(Clone, Copy, Debug, Default)]
struct Kahan {
    sum: f64,
    c: f64,
}

impl Kahan {
    #[inline]
    fn new() -> Self {
        Self { sum: 0.0, c: 0.0 }
    }

    #[inline]
    fn add(&mut self, x: f64) {
        let y = x - self.c;
        let t = self.sum + y;
        self.c = (t - self.sum) - y;
        self.sum = t;
    }

    #[inline]
    fn value(self) -> f64 {
        self.sum
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Column-major `nrows * ncols` copy; `fit_glm_domain` rejects nulls, so each chunk copies wholesale.
pub(crate) fn cols_to_vec(cols: &[&Float64Chunked], nrows: usize) -> RResult<Vec<f64>> {
    let mut out = vec![0.0; nrows * cols.len()];
    for (j, col) in cols.iter().enumerate() {
        if col.len() != nrows {
            return Err(RegressionError::DimensionMismatch {
                what: "column",
                expected: nrows,
                got: col.len(),
            });
        }
        let dst = &mut out[j * nrows..(j + 1) * nrows];
        let mut off = 0usize;
        for arr in col.downcast_iter() {
            let vals = arr.values().as_slice();
            let end = (off + vals.len()).min(nrows);
            dst[off..end].copy_from_slice(&vals[..end - off]);
            off = end;
        }
    }
    Ok(out)
}

/// Every per-row input must have `y`'s length: the fits index them all by row.
pub(crate) fn check_lengths(
    y: &Series,
    x_cols: &[Series],
    weights: &Series,
    design: [Option<&Series>; 4], // strata, psu, fpc, offset
    domain_mask: Option<&[bool]>,
) -> RResult<()> {
    let expected = y.len();
    let named = [
        ("weights", Some(weights.len())),
        ("strata", design[0].map(|s| s.len())),
        ("psu", design[1].map(|s| s.len())),
        ("fpc", design[2].map(|s| s.len())),
        ("offset", design[3].map(|s| s.len())),
        ("domain mask", domain_mask.map(<[bool]>::len)),
    ];
    let predictors = x_cols.iter().map(|s| ("predictor", Some(s.len())));
    for (what, len) in named.into_iter().chain(predictors) {
        if let Some(got) = len
            && got != expected
        {
            return Err(RegressionError::DimensionMismatch {
                what,
                expected,
                got,
            });
        }
    }
    Ok(())
}

/// Dense 0-based codes for `strata` and nested PSUs; nulls (`u32::MAX`) become one extra level.
pub(crate) fn design_codes(
    strata: Option<&Series>,
    psu: Option<&Series>,
    n: usize,
) -> RResult<(Vec<usize>, usize)> {
    let (raw, n_levels) = match (strata, psu) {
        (Some(s), Some(p)) => {
            design_pair_codes(&s.clone().into_column(), &p.clone().into_column())?
        }
        (None, Some(p)) => design_col_codes(&p.clone().into_column())?,
        (Some(s), None) => design_col_codes(&s.clone().into_column())?,
        (None, None) => return Ok(((0..n).collect(), n)),
    };

    let mut total = n_levels as usize;
    let mut null_slot: Option<usize> = None;
    let mut out = Vec::with_capacity(raw.len());
    for &c in &raw {
        if c == u32::MAX {
            let slot = *null_slot.get_or_insert_with(|| {
                let t = total;
                total += 1;
                t
            });
            out.push(slot);
        } else {
            out.push(c as usize);
        }
    }
    Ok((out, total))
}

/// Rows per cache block: a 256-row X block is k*2 KB, within L1/L2 for any realistic k.
const XP_BLOCK: usize = 256;

/// Rows per parallel chunk, fixed so the summation order, hence the result, is the same at any core count.
const XP_CHUNK: usize = 8192;

/// Upper triangle of `out[a*k + b] += sum_i w[i] cols[a*n + i] cols[b*n + i]`, in an order rayon cannot change.
fn accumulate_weighted_crossprod(cols: &[f64], w: &[f64], n: usize, k: usize, out: &mut [f64]) {
    let chunk_of = |lo: usize, hi: usize| -> Vec<f64> {
        let mut acc = vec![0.0f64; k * k];
        let mut wx = [0.0f64; XP_BLOCK];
        let mut b = lo;
        while b < hi {
            let e = (b + XP_BLOCK).min(hi);
            let len = e - b;
            let wb = &w[b..e];
            for a in 0..k {
                let xa = &cols[a * n + b..a * n + e];
                let wxa = &mut wx[..len];
                for t in 0..len {
                    wxa[t] = wb[t] * xa[t];
                }
                for c in a..k {
                    let xc = &cols[c * n + b..c * n + e];
                    let mut s = 0.0f64;
                    for t in 0..len {
                        s += wxa[t] * xc[t];
                    }
                    acc[a * k + c] += s;
                }
            }
            b = e;
        }
        acc
    };

    if n <= XP_CHUNK {
        let acc = chunk_of(0, n);
        for t in 0..k * k {
            out[t] += acc[t];
        }
        return;
    }

    let n_chunks = n.div_ceil(XP_CHUNK);
    let partials: Vec<Vec<f64>> = (0..n_chunks)
        .into_par_iter()
        .map(|c| chunk_of(c * XP_CHUNK, ((c + 1) * XP_CHUNK).min(n)))
        .collect();
    for acc in &partials {
        for t in 0..k * k {
            out[t] += acc[t];
        }
    }
}

/// One IRLS step's XtWX and XtWz, as R's fisherinf: t(D) %*% (w * D / V), D = X * dmu/deta.
///
/// Plain blocked sums as R's `crossprod`: Kahan bought ~1e-16 and cost the vectorization.
#[allow(clippy::too_many_arguments)]
fn build_irls_normal_eqs(
    family: Family,
    link: Link,
    n: usize,
    k: usize,
    y_vals: &[f64],
    x: &[f64],
    w_samp: &[f64],
    eta: &[f64],
    mu: &[f64],
    offset: &[f64],
    domain_mask: Option<&[bool]>,
    z: &mut [f64],
    w_irls: &mut [f64],
    XtWX: &mut Mat<f64>,
    XtWz: &mut Mat<f64>,
) {
    for i in 0..n {
        let in_domain = domain_mask.is_none_or(|m| m[i]);
        let w_i = w_samp[i];
        if !in_domain || w_i <= 0.0 {
            w_irls[i] = 0.0;
            z[i] = 0.0;
            continue;
        }

        let mu_i = mu[i];
        let v = family.variance(mu_i).max(1e-12);
        let d = link.mu_eta(mu_i, eta[i]);
        let wi = w_i * (d * d) / v;

        let safe_d = if d.abs() < 1e-12 { 1e-12 } else { d };
        // The offset leaves z and returns wherever eta is rebuilt: R's z <- (eta - offset) + (y - mu)/mu.eta.
        z[i] = (eta[i] - offset[i]) + (y_vals[i] - mu_i) / safe_d;
        w_irls[i] = if wi < 1e-18 { 0.0 } else { wi };
    }

    let mut acc_wx = vec![0.0f64; k * k];
    accumulate_weighted_crossprod(x, w_irls, n, k, &mut acc_wx);

    for r in 0..k {
        let xr = &x[r * n..(r + 1) * n];
        let mut total = 0.0f64;
        let mut b = 0usize;
        while b < n {
            let e = (b + XP_BLOCK).min(n);
            let mut s = 0.0f64;
            for t in b..e {
                s += w_irls[t] * xr[t] * z[t];
            }
            total += s;
            b = e;
        }
        XtWz[(r, 0)] = total;
    }

    for r in 0..k {
        for c in r..k {
            let v = acc_wx[r * k + c];
            XtWX[(r, c)] = v;
            XtWX[(c, r)] = v;
        }
    }
}

/// Solve A x = b by Cholesky, then LBLT, LU and an SVD pseudoinverse.
fn solve_linear_system(A: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Mat<f64> {
    if let Ok(chol) = A.llt(Side::Lower) {
        return chol.solve(b);
    }

    let lblt = A.lblt(Side::Lower);
    let x = lblt.solve(b);
    let mut ok = true;
    for i in 0..x.nrows() {
        if !x[(i, 0)].is_finite() {
            ok = false;
            break;
        }
    }
    if ok {
        return x;
    }

    let lu = A.partial_piv_lu();
    let x2 = lu.solve(b);
    for i in 0..x2.nrows() {
        if !x2[(i, 0)].is_finite() {
            // If even the SVD fails (e.g. a constant response), the non-finite LU solution becomes a fit error, not a panic.
            return match A.thin_svd() {
                Ok(svd) => svd.pseudoinverse() * b,
                Err(_) => x2,
            };
        }
    }
    x2
}

/// Per-stratum `(1 - f_h) m/(m-1)` and PSU counts; a single-PSU stratum gets 0, which keeps its rows out.
pub(crate) fn stratum_scales(
    strata_obs: &[Vec<usize>],
    psu_idx: Option<&[usize]>,
    n_psu_levels: usize,
    fpc_rows: Option<&[f64]>,
) -> (Vec<f64>, Vec<usize>) {
    let n_strata = strata_obs.len();
    let mut psus_h = vec![0usize; n_strata];

    match psu_idx {
        // No PSU: every row is its own.
        None => {
            for h in 0..n_strata {
                psus_h[h] = strata_obs[h].len();
            }
        }
        Some(psu) => {
            let mut seen = vec![false; n_psu_levels];
            for h in 0..n_strata {
                let mut m = 0usize;
                for &i in &strata_obs[h] {
                    if !seen[psu[i]] {
                        seen[psu[i]] = true;
                        m += 1;
                    }
                }
                for &i in &strata_obs[h] {
                    seen[psu[i]] = false;
                }
                psus_h[h] = m;
            }
        }
    }

    let mut scale_h = vec![0.0f64; n_strata];
    for h in 0..n_strata {
        let m = psus_h[h];
        if m <= 1 {
            continue;
        }
        let f = match fpc_rows {
            Some(f) => strata_obs[h].first().map(|&i| f[i]).unwrap_or(1.0),
            None => 1.0,
        };
        scale_h[h] = f * (m as f64) / ((m - 1) as f64);
    }
    (scale_h, psus_h)
}

/// R's `lonely.psu` "adjust"/"average" on a finished `p * p` meat, over the strata holding `active` rows.
///
/// `domain_rule` "center" re-centres a stratum whose domain rows sit in one PSU, adding `scale_h m (T̄_h - Ḡ)(T̄_h - Ḡ)'`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn singleton_meat(
    meat: &mut [f64],
    p: usize,
    col_at: impl Fn(usize, usize) -> f64,
    strata_obs: &[Vec<usize>],
    psu_idx: Option<&[usize]>,
    psus_h: &[usize],
    fpc_rows: Option<&[f64]>,
    method: SingletonMethod,
    domain_rule: bool,
    active: Option<&[bool]>,
) {
    if method == SingletonMethod::None && !domain_rule {
        return;
    }
    // PSUs of each stratum holding an active row.
    let reached: Vec<usize> = strata_obs
        .iter()
        .zip(psus_h)
        .map(|(obs, &m)| match active {
            None => m,
            Some(a) => {
                let mut units: Vec<usize> = obs
                    .iter()
                    .filter(|&&i| a[i])
                    .map(|&i| psu_idx.map_or(i, |u| u[i]))
                    .collect();
                units.sort_unstable();
                units.dedup();
                units.len()
            }
        })
        .collect();
    let lonely = |h: usize| domain_rule && reached[h] == 1 && psus_h[h] > 1;
    let (mut n_units, mut n_strata, mut n_singletons) = (0usize, 0usize, 0usize);
    for (h, &m) in psus_h.iter().enumerate() {
        if reached[h] > 0 {
            n_units += m;
            n_strata += 1;
            n_singletons += usize::from(m == 1 || lonely(h));
        }
    }
    let fpc_of = |h: usize| fpc_rows.map_or(1.0, |f| f[strata_obs[h][0]]);
    let add_outer = |meat: &mut [f64], w: f64, v: &[f64]| {
        for a in 0..p {
            for b in 0..p {
                meat[a * p + b] += w * v[a] * v[b];
            }
        }
    };
    let stratum_total = |h: usize| -> Vec<f64> {
        (0..p)
            .map(|j| strata_obs[h].iter().map(|&i| col_at(i, j)).sum())
            .collect()
    };

    if method != SingletonMethod::Center {
        // Lonely strata leave: subtract their standard term.
        for h in (0..psus_h.len()).filter(|&h| lonely(h)) {
            let m = psus_h[h];
            let mut totals: FxHashMap<usize, Vec<f64>> = FxHashMap::default();
            for &i in &strata_obs[h] {
                let t = totals
                    .entry(psu_idx.map_or(i, |u| u[i]))
                    .or_insert_with(|| vec![0.0; p]);
                for (j, v) in t.iter_mut().enumerate() {
                    *v += col_at(i, j);
                }
            }
            let mean: Vec<f64> = stratum_total(h).iter().map(|t| t / m as f64).collect();
            let scale_h = fpc_of(h) * m as f64 / (m - 1) as f64;
            for t in totals.values() {
                let dev: Vec<f64> = t.iter().zip(&mean).map(|(t, c)| t - c).collect();
                add_outer(meat, -scale_h, &dev);
            }
            // PSUs without rows here are zero totals.
            let n_empty = m.saturating_sub(totals.len()) as f64;
            add_outer(meat, -scale_h * n_empty, &mean);
        }
        if method == SingletonMethod::Scale {
            let f = if n_strata > n_singletons {
                n_strata as f64 / (n_strata - n_singletons) as f64
            } else {
                f64::NAN
            };
            meat.iter_mut().for_each(|v| *v *= f);
        }
        return;
    }

    if n_units == 0 {
        return;
    }
    let mut grand_mean = vec![0.0f64; p];
    for obs in strata_obs {
        for &i in obs {
            for (j, g) in grand_mean.iter_mut().enumerate() {
                *g += col_at(i, j);
            }
        }
    }
    grand_mean.iter_mut().for_each(|g| *g /= n_units as f64);
    for h in 0..psus_h.len() {
        let m = psus_h[h];
        if reached[h] == 0 || !(m == 1 || lonely(h)) {
            continue;
        }
        let total = stratum_total(h);
        if m == 1 {
            let dev: Vec<f64> = total.iter().zip(&grand_mean).map(|(t, g)| t - g).collect();
            add_outer(meat, fpc_of(h), &dev);
        } else {
            let dev: Vec<f64> = total
                .iter()
                .zip(&grand_mean)
                .map(|(t, g)| t / m as f64 - g)
                .collect();
            let scale_h = fpc_of(h) * m as f64 / (m - 1) as f64;
            add_outer(meat, scale_h * m as f64, &dev);
        }
    }
}

/// R's `svyrecvar`: row-major `p * p` covariance of the totals of column-major, already-weighted `n * p` columns.
#[allow(clippy::too_many_arguments)]
pub(crate) fn design_vcov_of_totals(
    cols: &[f64],
    n: usize,
    p: usize,
    strata_idx: &[usize],
    strata_obs: &[Vec<usize>],
    psu_idx: Option<&[usize]>,
    n_psu_levels: usize,
    fpc_rows: Option<&[f64]>,
    singleton: SingletonRule,
    active: Option<&[bool]>,
) -> Vec<f64> {
    let n_strata = strata_obs.len();
    let (scale_h, psus_h) = stratum_scales(strata_obs, psu_idx, n_psu_levels, fpc_rows);
    let mut upper = vec![0.0f64; p * p];

    match psu_idx {
        None => {
            // Every row is its own PSU, so the stratum sums telescope:
            //   sum_i (t_i - tbar)(t_i - tbar)' = sum_i t_i t_i' - m tbar tbar'.
            let mut row_w = vec![0.0f64; n];
            for i in 0..n {
                row_w[i] = scale_h[strata_idx[i]];
            }
            let mut sums = vec![0.0f64; n_strata * p];
            for j in 0..p {
                let col = &cols[j * n..(j + 1) * n];
                for i in 0..n {
                    sums[strata_idx[i] * p + j] += col[i];
                }
            }
            accumulate_weighted_crossprod(cols, &row_w, n, p, &mut upper);

            for h in 0..n_strata {
                let m = psus_h[h];
                if m <= 1 {
                    continue;
                }
                let c = scale_h[h] / (m as f64);
                for a in 0..p {
                    let sa = sums[h * p + a];
                    for b in a..p {
                        upper[a * p + b] -= c * sa * sums[h * p + b];
                    }
                }
            }
        }
        Some(psu) => {
            let mut slot = vec![usize::MAX; n_psu_levels];
            let mut totals: Vec<f64> = Vec::new();
            let mut used: Vec<usize> = Vec::new();
            let mut mean = vec![0.0f64; p];
            let mut local = vec![0.0f64; p * p];

            for h in 0..n_strata {
                totals.clear();
                used.clear();

                for &i in &strata_obs[h] {
                    let pid = psu[i];
                    let li = if slot[pid] == usize::MAX {
                        let t = used.len();
                        slot[pid] = t;
                        used.push(pid);
                        totals.resize(totals.len() + p, 0.0);
                        t
                    } else {
                        slot[pid]
                    };
                    let base = li * p;
                    for j in 0..p {
                        totals[base + j] += cols[j * n + i];
                    }
                }

                let m = used.len();
                for &pid in &used {
                    slot[pid] = usize::MAX;
                }
                if m <= 1 {
                    continue;
                }

                mean.iter_mut().for_each(|v| *v = 0.0);
                for li in 0..m {
                    for j in 0..p {
                        mean[j] += totals[li * p + j];
                    }
                }
                for v in mean.iter_mut().take(p) {
                    *v /= m as f64;
                }

                local.iter_mut().for_each(|v| *v = 0.0);
                for li in 0..m {
                    let base = li * p;
                    for a in 0..p {
                        let da = totals[base + a] - mean[a];
                        for b in a..p {
                            local[a * p + b] += da * (totals[base + b] - mean[b]);
                        }
                    }
                }
                for a in 0..p {
                    for b in a..p {
                        upper[a * p + b] += scale_h[h] * local[a * p + b];
                    }
                }
            }
        }
    }

    // Mirror the upper triangle: symmetric by construction.
    let mut out = vec![0.0f64; p * p];
    for a in 0..p {
        for b in a..p {
            let v = upper[a * p + b];
            out[a * p + b] = v;
            out[b * p + a] = v;
        }
    }
    singleton_meat(
        &mut out,
        p,
        |i, j| cols[j * n + i],
        strata_obs,
        psu_idx,
        &psus_h,
        fpc_rows,
        singleton.method,
        singleton.domains,
        active,
    );
    out
}

/// Raise on aliased columns by R's "first columns win" pivot-free Cholesky, rather than reporting NA coefficients.
fn check_rank(XtWX: &Mat<f64>, k: usize, cols: &[Series]) -> RResult<()> {
    // An aliased column leaves ~1e-16 of its norm, one at R^2 = 0.999999 leaves 1e-6: five orders either side.
    const ALIAS_TOL: f64 = 1e-11;

    let mut l = vec![0.0f64; k * k]; // column-major
    let mut aliased: Vec<usize> = Vec::new();

    for j in 0..k {
        let mut d = XtWX[(j, j)];
        for p in 0..j {
            d -= l[p * k + j] * l[p * k + j];
        }
        // Negated so a NaN diagonal is aliased rather than accepted.
        #[allow(clippy::neg_cmp_op_on_partial_ord)] // NaN takes this branch on purpose
        if !(d > ALIAS_TOL * XtWX[(j, j)]) {
            // A zero factor column measures later columns against the kept ones only.
            aliased.push(j);
            continue;
        }
        let ljj = d.sqrt();
        l[j * k + j] = ljj;
        for i in (j + 1)..k {
            let mut v = XtWX[(i, j)];
            for p in 0..j {
                v -= l[p * k + i] * l[p * k + j];
            }
            l[j * k + i] = v / ljj;
        }
    }

    if aliased.is_empty() {
        return Ok(());
    }

    let listed: Vec<String> = aliased
        .iter()
        .map(|&j| format!("'{}'", cols[j].name()))
        .collect();
    Err(RegressionError::RankDeficient(listed.join(", ")))
}

/// A^{-1} by the `solve_linear_system` chain; an error, not a panic, when even the SVD fails.
pub(crate) fn invert_matrix(A: MatRef<'_, f64>, k: usize) -> RResult<Mat<f64>> {
    if let Ok(chol) = A.llt(Side::Lower) {
        let mut inv = Mat::<f64>::identity(k, k);
        chol.solve_in_place(inv.as_mut());
        return Ok(inv);
    }

    let lblt = A.lblt(Side::Lower);
    let mut inv = Mat::<f64>::identity(k, k);
    lblt.solve_in_place(inv.as_mut());

    for r in 0..k {
        for c in 0..k {
            if !inv[(r, c)].is_finite() {
                let lu = A.partial_piv_lu();
                let mut inv2 = Mat::<f64>::identity(k, k);
                lu.solve_in_place(inv2.as_mut());

                for rr in 0..k {
                    for cc in 0..k {
                        if !inv2[(rr, cc)].is_finite() {
                            return match A.thin_svd() {
                                Ok(svd) => Ok(svd.pseudoinverse()),
                                Err(_) => Err(RegressionError::InfoNotInvertible),
                            };
                        }
                    }
                }
                return Ok(inv2);
            }
        }
    }

    Ok(inv)
}

// ============================================================================
// Result
// ============================================================================

#[allow(dead_code)]
pub(crate) struct GlmResult {
    pub params: Vec<f64>,
    pub cov_params: Vec<f64>,
    /// R svyglm's naive.cov, (X'WX)^-1 at the final beta, on the fit's mean-1 weight scale.
    pub naive_cov: Vec<f64>,
    pub scale: f64,
    pub df_resid: f64,
    pub deviance: f64,
    pub null_deviance: f64,
    pub iterations: u32,
    pub n_obs: usize,
    /// Whether IRLS met the tolerance rather than exhausting `max_iter`.
    pub converged: bool,
    /// Negative binomial theta, with its SE when estimated rather than supplied.
    pub theta: Option<f64>,
    pub theta_se: Option<f64>,
    /// Set when the fit pushed rows to the edge of the response's range.
    pub separation: Option<Separation>,
}

/// Rows the fit drove to the boundary, and what is left to identify beta.
pub(crate) struct Separation {
    /// Contributing rows whose fitted value the fit cannot tell from y.
    pub boundary_rows: usize,
    /// Row-major `sum_i w_i x_i x_i'` over the other rows; a coefficient outside its row space is not identified.
    pub gram: Vec<f64>,
}

/// In tolerances: separated rows' deviance share stops near 1.6 (logit, log; ~3 cauchit), ordinary rows orders above.
const SEPARATION_RESOLUTION: f64 = 10.0;

// ============================================================================
// Core Algorithm
// ============================================================================

/// Null deviance under an offset: as R's glm(), refit `y ~ 1` with the offset rather than use the weighted mean.
#[allow(clippy::too_many_arguments)]
fn null_deviance_with_offset(
    family: Family,
    link: Link,
    n: usize,
    y_vals: &[f64],
    w_samp: &[f64],
    offset: &[f64],
    domain_mask: Option<&[bool]>,
    tol: f64,
    max_iter: usize,
) -> f64 {
    let contributes = |i: usize| domain_mask.is_none_or(|m| m[i]) && w_samp[i] > 0.0;

    let mut b = 0.0f64;
    let mut deviance = f64::INFINITY;

    for _ in 0..max_iter {
        // One IRLS step for the single intercept: b_new = sum(w z) / sum(w).
        let mut num = Kahan::new();
        let mut den = Kahan::new();

        for i in 0..n {
            if !contributes(i) {
                continue;
            }
            let eta_i = b + offset[i];
            let mu_i = link.inverse(eta_i);
            let d = link.mu_eta(mu_i, eta_i);
            let v = family.variance(mu_i).max(1e-12);
            let wi = w_samp[i] * (d * d) / v;
            if wi.abs() < 1e-18 {
                continue;
            }
            let safe_d = if d.abs() < 1e-12 { 1e-12 } else { d };
            let z_i = (eta_i - offset[i]) + (y_vals[i] - mu_i) / safe_d;
            num.add(wi * z_i);
            den.add(wi);
        }

        let den_v = den.value();
        if den_v <= 0.0 {
            break;
        }
        let b_new = num.value() / den_v;

        let mut dev_new = 0.0;
        for i in 0..n {
            if !contributes(i) {
                continue;
            }
            let mu_i = link.inverse(b_new + offset[i]);
            dev_new += w_samp[i] * family.unit_deviance(y_vals[i], mu_i);
        }

        let converged = deviance.is_finite()
            && ((deviance - dev_new).abs() / (0.1 + dev_new.abs()) < tol
                || (b_new - b).abs() < tol);

        b = b_new;
        deviance = dev_new;

        if converged {
            break;
        }
    }

    if deviance.is_finite() {
        deviance
    } else {
        f64::NAN
    }
}

/// One fit; only an estimated negative binomial theta leaves the plain IRLS path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_one(
    y: &Series,
    x_cols: Vec<Series>,
    weights: &Series,
    strata: Option<&Series>,
    psu: Option<&Series>,
    fpc: Option<&Series>,
    offset: Option<&Series>,
    domain_mask: Option<&[bool]>,
    family_str: &str,
    link_str: &str,
    theta: Option<f64>,
    tol: f64,
    max_iter: usize,
    calib: Option<&CalibSweep>,
    singleton: SingletonRule,
) -> RResult<GlmResult> {
    if matches!(
        family_str.to_lowercase().as_str(),
        "negativebinomial" | "negative_binomial"
    ) {
        return crate::regression::negbin::fit_negbin(
            y,
            x_cols,
            weights,
            strata,
            psu,
            fpc,
            offset,
            domain_mask,
            link_str,
            theta,
            tol,
            max_iter,
            calib,
            singleton,
        );
    }
    fit_glm_domain(
        y,
        x_cols,
        weights,
        strata,
        psu,
        fpc,
        offset,
        domain_mask,
        family_str,
        link_str,
        theta,
        tol,
        max_iter,
        calib,
        singleton,
        true,
    )
}

/// Full-sample fit without singleton handling; the unit tests' entry point.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_glm(
    y: &Series,
    x_cols: Vec<Series>,
    weights: &Series,
    strata: Option<&Series>,
    psu: Option<&Series>,
    fpc: Option<&Series>,
    offset: Option<&Series>,
    family_str: &str,
    link_str: &str,
    theta: Option<f64>,
    tol: f64,
    max_iter: usize,
    calib: Option<&CalibSweep>,
) -> RResult<GlmResult> {
    fit_one(
        y,
        x_cols,
        weights,
        strata,
        psu,
        fpc,
        offset,
        None,
        family_str,
        link_str,
        theta,
        tol,
        max_iter,
        calib,
        SingletonRule::NONE,
    )
}

/// One `where=` domain fit; `fit_glm_by` would also fit the complement, and fail on a degenerate one.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_glm_where(
    y: &Series,
    x_cols: Vec<Series>,
    weights: &Series,
    strata: Option<&Series>,
    psu: Option<&Series>,
    fpc: Option<&Series>,
    offset: Option<&Series>,
    mask: &Series,
    family_str: &str,
    link_str: &str,
    theta: Option<f64>,
    tol: f64,
    max_iter: usize,
    calib: Option<&CalibSweep>,
    singleton: SingletonRule,
) -> RResult<GlmResult> {
    let cast = mask.cast(&DataType::Boolean)?;
    let ca = cast.bool()?;
    let mask_vec: Vec<bool> = ca.iter().map(|v| v.unwrap_or(false)).collect();

    fit_one(
        y,
        x_cols,
        weights,
        strata,
        psu,
        fpc,
        offset,
        Some(&mask_vec),
        family_str,
        link_str,
        theta,
        tol,
        max_iter,
        calib,
        singleton,
    )
}

/// One fit per level of `by_col` over the full design, as R's `svyglm(..., design = subset(d, ...))`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_glm_by(
    y: &Series,
    x_cols: Vec<Series>,
    weights: &Series,
    strata: Option<&Series>,
    psu: Option<&Series>,
    fpc: Option<&Series>,
    offset: Option<&Series>,
    by_col: &Series,
    family_str: &str,
    link_str: &str,
    theta: Option<f64>,
    tol: f64,
    max_iter: usize,
    calib: Option<&CalibSweep>,
    singleton: SingletonRule,
) -> RResult<Vec<(String, GlmResult)>> {
    let by_str_series = by_col.cast(&DataType::String)?;
    let by_str = by_str_series.str()?;
    let unique_groups = crate::estimation::sorted_groups(by_str)?;

    // Collected in level order, so the result does not depend on the thread count.
    let groups: Vec<&str> = unique_groups.iter().flatten().collect();
    let attempts: Vec<(String, RResult<GlmResult>)> = groups
        .par_iter()
        .map(|&group_val| {
            let mask_vec: Vec<bool> = by_str.iter().map(|v| v == Some(group_val)).collect();

            // Series are reference-counted: this copies only the Vec.
            let xs = x_cols.to_vec();
            let res = fit_one(
                y,
                xs,
                weights,
                strata,
                psu,
                fpc,
                offset,
                Some(&mask_vec),
                family_str,
                link_str,
                theta,
                tol,
                max_iter,
                calib,
                singleton,
            );
            (group_val.to_string(), res)
        })
        .collect();

    // A failed level raises, naming the levels, rather than shortening the list unnoticed.
    let mut results = Vec::with_capacity(attempts.len());
    let mut failed: Vec<String> = Vec::new();
    let mut first_err: Option<RegressionError> = None;
    for (level, res) in attempts {
        match res {
            Ok(r) => results.push((level, r)),
            Err(e) => {
                if first_err.is_none() {
                    first_err = Some(e);
                }
                failed.push(level);
            }
        }
    }
    if let Some(source) = first_err {
        return Err(RegressionError::LevelsFailed {
            n_failed: failed.len(),
            total: failed.len() + results.len(),
            by: by_col.name().to_string(),
            levels: failed.join(", "),
            source: Box::new(source),
        });
    }
    Ok(results)
}

/// Parallel replicate refits of the coefficients and separation, skipping the unused sandwich except for negbin.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_glm_reps(
    y: &Series,
    x_cols: &[Series],
    rep_weights: &[Series],
    strata: Option<&Series>,
    psu: Option<&Series>,
    offset: Option<&Series>,
    domain_mask: Option<&[bool]>,
    family_str: &str,
    link_str: &str,
    theta: Option<f64>,
    tol: f64,
    max_iter: usize,
) -> Vec<RResult<GlmResult>> {
    let negbin = matches!(
        family_str.to_lowercase().as_str(),
        "negativebinomial" | "negative_binomial"
    );
    rep_weights
        .par_iter()
        .map(|w| {
            if negbin {
                fit_one(
                    y,
                    x_cols.to_vec(),
                    w,
                    strata,
                    psu,
                    None,
                    offset,
                    domain_mask,
                    family_str,
                    link_str,
                    theta,
                    tol,
                    max_iter,
                    None,
                    SingletonRule::NONE,
                )
            } else {
                fit_glm_domain(
                    y,
                    x_cols.to_vec(),
                    w,
                    strata,
                    psu,
                    None,
                    offset,
                    domain_mask,
                    family_str,
                    link_str,
                    theta,
                    tol,
                    max_iter,
                    None,
                    SingletonRule::NONE,
                    false,
                )
            }
        })
        .collect()
}

/// Core GLM fit: out-of-domain rows contribute 0 to IRLS and the meat, but strata and PSUs stay the full design's.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_glm_domain(
    y: &Series,
    x_cols: Vec<Series>,
    weights: &Series,
    strata: Option<&Series>,
    psu: Option<&Series>,
    fpc: Option<&Series>,
    offset: Option<&Series>,
    domain_mask: Option<&[bool]>,
    family_str: &str,
    link_str: &str,
    theta: Option<f64>,
    tol: f64,
    max_iter: usize,
    calib: Option<&CalibSweep>,
    singleton: SingletonRule,
    want_variance: bool,
) -> RResult<GlmResult> {
    let family = Family::from_str(family_str, theta)?;
    let link = Link::from_str(link_str)?;
    check_lengths(y, &x_cols, weights, [strata, psu, fpc, offset], domain_mask)?;

    let n = y.len();
    let k = x_cols.len();

    // Reject nulls: null-skipping iteration would silently misalign the y/X/w rows.
    let y_cast = y.cast(&DataType::Float64)?;
    let y_ca = y_cast.f64()?;
    let w_cast = weights.cast(&DataType::Float64)?;
    let w_ca = w_cast.f64()?;
    let x_cast: Vec<Series> = x_cols
        .iter()
        .map(|s| s.cast(&DataType::Float64))
        .collect::<PolarsResult<Vec<_>>>()?;
    let mut x_ca_list: Vec<&Float64Chunked> = Vec::with_capacity(x_cast.len());
    for s in &x_cast {
        x_ca_list.push(s.f64()?);
    }

    if y_ca.null_count() > 0 {
        return Err(RegressionError::NullValues {
            what: "response",
            col: y.name().to_string(),
        });
    }
    if w_ca.null_count() > 0 {
        return Err(RegressionError::NullValues {
            what: "weight",
            col: weights.name().to_string(),
        });
    }
    for s in &x_cast {
        if s.null_count() > 0 {
            return Err(RegressionError::NullValues {
                what: "predictor",
                col: s.name().to_string(),
            });
        }
    }

    let y_vals = cols_to_vec(&[y_ca], n)?;
    let x = cols_to_vec(&x_ca_list, n)?;

    // Known link-scale term with coefficient 1; all zero when absent.
    let offset_vals: Vec<f64> = match offset {
        Some(s) => {
            let cast = s.cast(&DataType::Float64)?;
            let ca = cast.f64()?;
            if ca.null_count() > 0 {
                return Err(RegressionError::NullValues {
                    what: "offset",
                    col: s.name().to_string(),
                });
            }
            ca.iter().map(|v| v.unwrap_or(0.0)).collect()
        }
        None => vec![0.0; n],
    };

    let mut w_samp = vec![0.0; n];
    let mut w_sum = 0.0;
    for (i, v) in w_ca.iter().enumerate() {
        let v = v.unwrap_or(0.0);
        w_samp[i] = v;
        w_sum += v;
    }

    // Scale to sum(w) = n for conditioning; the sandwich is invariant to it.
    if w_sum > 0.0 {
        let scale = (n as f64) / w_sum;
        for wi in &mut w_samp {
            *wi *= scale;
        }
        w_sum = n as f64;
    }

    // Without this, no contributing rows gives zero normal equations, whose pseudoinverse yields a finite beta = 0.
    let n_obs = (0..n)
        .filter(|&i| domain_mask.is_none_or(|m| m[i]) && w_samp[i] > 0.0)
        .count();
    if n_obs <= k {
        return Err(RegressionError::TooFewObservations { n_obs, k });
    }

    // `f64::max` drops a NaN, so `.max(1e-12)` on the variance would carry one to a NaN beta.
    for i in 0..n {
        if !(domain_mask.is_none_or(|m| m[i]) && w_samp[i] > 0.0) {
            continue;
        }
        if !y_vals[i].is_finite() {
            return Err(RegressionError::NonFinite {
                what: format!("response column '{}'", y.name()),
                row: i,
            });
        }
        if !offset_vals[i].is_finite() {
            return Err(RegressionError::NonFinite {
                what: "offset".to_string(),
                row: i,
            });
        }
        for j in 0..k {
            if !x[j * n + i].is_finite() {
                return Err(RegressionError::NonFinite {
                    what: format!("predictor column '{}'", x_cast[j].name()),
                    row: i,
                });
            }
        }
    }

    // Out-of-domain rows get a placeholder start: they never contribute.
    let mut beta = Mat::<f64>::zeros(k, 1);
    let mut mu = vec![0.0; n];
    let mut eta = vec![0.0; n];

    for i in 0..n {
        let in_domain = domain_mask.is_none_or(|m| m[i]);
        let y_init = if in_domain { y_vals[i] } else { 0.5 };
        mu[i] = family.initial_mu(y_init, w_samp[i]);
        // As R, eta starts at linkfun(mustart) without the offset; later etas add it back.
        eta[i] = link.link(mu[i]);
    }

    let mut z_work = vec![0.0f64; n];
    let mut w_irls = vec![0.0; n];
    let mut XtWX = Mat::<f64>::zeros(k, k);
    let mut XtWz = Mat::<f64>::zeros(k, 1);

    // eta/mu enter each pass already at the current beta.
    let mut iter_count = 0;
    let mut deviance = 0.0;
    let mut converged = false;

    for iter in 0..max_iter {
        iter_count += 1;

        build_irls_normal_eqs(
            family,
            link,
            n,
            k,
            &y_vals,
            &x,
            &w_samp,
            &eta,
            &mu,
            &offset_vals,
            domain_mask,
            &mut z_work,
            &mut w_irls,
            &mut XtWX,
            &mut XtWz,
        );

        // Aliasing makes every later SE meaningless: raise at the first information matrix.
        if iter == 0 {
            check_rank(&XtWX, k, &x_cast)?;
        }

        let beta_new = solve_linear_system(XtWX.as_ref(), XtWz.as_ref());

        // Column-outer: one contiguous pass over x per term.
        eta[..n].copy_from_slice(&offset_vals[..n]);
        for j in 0..k {
            let bj = beta_new[(j, 0)];
            if bj == 0.0 {
                continue;
            }
            let xj = &x[j * n..(j + 1) * n];
            for i in 0..n {
                eta[i] += xj[i] * bj;
            }
        }

        let mut dev_new = 0.0;
        for i in 0..n {
            let mu_i = link.inverse(eta[i]);
            mu[i] = mu_i;
            if domain_mask.is_none_or(|m| m[i]) && w_samp[i] > 0.0 {
                dev_new += w_samp[i] * family.unit_deviance(y_vals[i], mu_i);
            }
        }

        let mut max_delta = 0.0;
        for j in 0..k {
            let d = (beta_new[(j, 0)] - beta[(j, 0)]).abs();
            if d > max_delta {
                max_delta = d;
            }
        }

        let rel_dev = if iter > 0 {
            (deviance - dev_new).abs() / (0.1 + dev_new.abs())
        } else {
            f64::INFINITY
        };

        beta = beta_new;
        deviance = dev_new;

        // R's glm.fit test, abs(dev - devold) / (abs(dev) + 0.1) < epsilon; `max_delta` only backstops an
        // oscillating deviance, as there is no step-halving here.
        if iter > 0 && (rel_dev < tol || max_delta < tol) {
            converged = true;
            break;
        }
    }

    // A degenerate system (e.g. a constant binomial response) is an error, not NaN coefficients.
    for j in 0..k {
        if !beta[(j, 0)].is_finite() {
            return Err(RegressionError::NonFiniteCoefficients);
        }
    }

    // The theta loop and replicate refits skip the costly sandwich.
    if !want_variance {
        let separation = separation_check(
            family,
            n,
            k,
            &y_vals,
            &x,
            &w_samp,
            &mu,
            domain_mask,
            deviance,
            tol,
        );
        return Ok(GlmResult {
            params: (0..k).map(|i| beta[(i, 0)]).collect(),
            cov_params: Vec::new(),
            naive_cov: Vec::new(),
            scale: 1.0,
            df_resid: 1.0,
            deviance,
            null_deviance: 0.0,
            iterations: iter_count as u32,
            n_obs,
            converged,
            theta: None,
            theta_se: None,
            separation,
        });
    }

    // =========================================================================
    // 4) Sandwich variance (R-alignment: rebuild XtWX at FINAL beta)
    // =========================================================================

    // The bread must be R's fisherinf at the converged beta, where eta/mu already are.
    build_irls_normal_eqs(
        family,
        link,
        n,
        k,
        &y_vals,
        &x,
        &w_samp,
        &eta,
        &mu,
        &offset_vals,
        domain_mask,
        &mut z_work,
        &mut w_irls,
        &mut XtWX,
        &mut XtWz,
    );

    // The full design, whatever the domain.
    let (strata_idx, n_strata) = match strata {
        Some(_) => design_codes(strata, None, n)?,
        None => (vec![0usize; n], 1usize),
    };

    let (psu_idx, n_psu_levels) = match psu {
        Some(_) => design_codes(strata, psu, n)?,
        None => ((0..n).collect::<Vec<_>>(), n),
    };

    let mut strata_obs: Vec<Vec<usize>> = vec![Vec::new(); n_strata];
    for i in 0..n {
        strata_obs[strata_idx[i]].push(i);
    }

    // Per-row (1 - f_h), scaling its stratum's meat as R svyglm with fpc=; missing means no correction.
    let fpc_rows: Option<Vec<f64>> = match fpc {
        Some(s) => {
            let s_cast = s.cast(&DataType::Float64)?;
            let ca = s_cast.f64()?;
            Some(ca.iter().map(|v| v.unwrap_or(1.0)).collect())
        }
        None => None,
    };

    // w (y - mu) (dmu/deta) / V directly: w_irls * working_resid carries the (d + eps) guard, ~1e-7 off in SEs.
    let score_at = |i: usize| -> f64 {
        let w_i = w_samp[i];
        if w_i <= 0.0 || w_irls[i] <= 0.0 {
            return 0.0;
        }
        let mu_i = mu[i];
        let d = link.mu_eta(mu_i, eta[i]);
        let v = family.variance(mu_i).max(1e-12);
        w_i * (y_vals[i] - mu_i) * d / v
    };

    // R sweeps the influence (svy.varcoef); the sweep commutes with the bread, S(E A) = (S E) A, so sweeping here matches.
    // Every row is kept and accumulated: a zero-score row is charged its cell's mean once swept.
    let ef: Option<Vec<f64>> = calib.map(|c| {
        let mut m = vec![0.0; n * k];
        for i in 0..n {
            let s_i = score_at(i);
            if s_i != 0.0 {
                for j in 0..k {
                    m[j * n + i] = s_i * x[j * n + i];
                }
            }
        }
        for j in 0..k {
            c.apply(&mut m[j * n..(j + 1) * n]);
        }
        m
    });

    // Uncalibrated, the n x k scores are not materialised (160 MB at 1e6 x 20): the score folds into the row weight.
    let psu_opt = if psu.is_some() {
        Some(psu_idx.as_slice())
    } else {
        None
    };

    let meat_flat = match &ef {
        Some(e) => design_vcov_of_totals(
            e,
            n,
            k,
            &strata_idx,
            &strata_obs,
            psu_opt,
            n_psu_levels,
            fpc_rows.as_deref(),
            singleton,
            // Calibrated scores are nonzero outside the domain: whole frame.
            None,
        ),
        None => {
            let mut scores = vec![0.0f64; n];
            for (i, s) in scores.iter_mut().enumerate().take(n) {
                *s = score_at(i);
            }
            let (scale_h, psus_h) =
                stratum_scales(&strata_obs, psu_opt, n_psu_levels, fpc_rows.as_deref());
            let mut upper = vec![0.0f64; k * k];

            if psu.is_none() {
                // Every row its own PSU: the centred cross-product telescopes to one pass plus a k-vector per stratum.
                let mut row_w = vec![0.0f64; n];
                for i in 0..n {
                    row_w[i] = scale_h[strata_idx[i]] * scores[i] * scores[i];
                }
                let mut sums = vec![0.0f64; n_strata * k];
                for j in 0..k {
                    let col = &x[j * n..(j + 1) * n];
                    for i in 0..n {
                        sums[strata_idx[i] * k + j] += scores[i] * col[i];
                    }
                }
                accumulate_weighted_crossprod(&x, &row_w, n, k, &mut upper);

                for h in 0..n_strata {
                    let m = psus_h[h];
                    if m <= 1 {
                        continue;
                    }
                    let c = scale_h[h] / (m as f64);
                    for a in 0..k {
                        let sa = sums[h * k + a];
                        for b in a..k {
                            upper[a * k + b] -= c * sa * sums[h * k + b];
                        }
                    }
                }
            } else {
                let mut slot = vec![usize::MAX; n_psu_levels];
                let mut totals: Vec<f64> = Vec::new();
                let mut used: Vec<usize> = Vec::new();
                let mut mean = vec![0.0f64; k];
                let mut local = vec![0.0f64; k * k];

                for h in 0..n_strata {
                    totals.clear();
                    used.clear();

                    for &i in &strata_obs[h] {
                        let pid = psu_idx[i];
                        let li = if slot[pid] == usize::MAX {
                            let t = used.len();
                            slot[pid] = t;
                            used.push(pid);
                            totals.resize(totals.len() + k, 0.0);
                            t
                        } else {
                            slot[pid]
                        };
                        let s_i = scores[i];
                        if s_i != 0.0 {
                            let base = li * k;
                            for j in 0..k {
                                totals[base + j] += s_i * x[j * n + i];
                            }
                        }
                    }

                    let m = used.len();
                    for &pid in &used {
                        slot[pid] = usize::MAX;
                    }
                    if m <= 1 {
                        continue;
                    }

                    mean.iter_mut().for_each(|v| *v = 0.0);
                    for li in 0..m {
                        for j in 0..k {
                            mean[j] += totals[li * k + j];
                        }
                    }
                    for v in mean.iter_mut().take(k) {
                        *v /= m as f64;
                    }

                    local.iter_mut().for_each(|v| *v = 0.0);
                    for li in 0..m {
                        let base = li * k;
                        for a in 0..k {
                            let da = totals[base + a] - mean[a];
                            for b in a..k {
                                local[a * k + b] += da * (totals[base + b] - mean[b]);
                            }
                        }
                    }
                    for a in 0..k {
                        for b in a..k {
                            upper[a * k + b] += scale_h[h] * local[a * k + b];
                        }
                    }
                }
            }

            let mut out = vec![0.0f64; k * k];
            for a in 0..k {
                for b in a..k {
                    let v = upper[a * k + b];
                    out[a * k + b] = v;
                    out[b * k + a] = v;
                }
            }
            // The domain's rows whatever their weight, as R's subset() keeps zero-weight rows.
            let active = (singleton != SingletonRule::NONE).then(|| {
                (0..n)
                    .map(|i| domain_mask.is_none_or(|m| m[i]))
                    .collect::<Vec<_>>()
            });
            singleton_meat(
                &mut out,
                k,
                |i, j| scores[i] * x[j * n + i],
                &strata_obs,
                psu_opt,
                &psus_h,
                fpc_rows.as_deref(),
                singleton.method,
                singleton.domains,
                active.as_deref(),
            );
            out
        }
    };

    let mut meat = Mat::<f64>::zeros(k, k);
    for a in 0..k {
        for b in 0..k {
            meat[(a, b)] = meat_flat[a * k + b];
        }
    }

    let bread = invert_matrix(XtWX.as_ref(), k)?;

    let tmp = &bread * &meat;
    let cov = &tmp * &bread;

    // As R, df counts only PSUs and strata with a positive-weight domain row; NaN when none is left.
    use std::collections::HashSet;
    let df_resid = if psu.is_some() && strata.is_some() {
        let mut total_psus: usize = 0;
        let mut nonempty_strata: usize = 0;
        for obs in strata_obs.iter().take(n_strata) {
            let psus_in_dom: HashSet<usize> = obs
                .iter()
                .filter(|&&i| domain_mask.is_none_or(|m| m[i]) && w_samp[i] > 0.0)
                .map(|&i| psu_idx[i])
                .collect();
            if !psus_in_dom.is_empty() {
                total_psus += psus_in_dom.len();
                nonempty_strata += 1;
            }
        }
        let df = (total_psus as isize) - (nonempty_strata as isize);
        if df <= 0 { f64::NAN } else { df as f64 }
    } else if psu.is_some() {
        let psus_in_dom: HashSet<usize> = (0..n)
            .filter(|&i| domain_mask.is_none_or(|m| m[i]) && w_samp[i] > 0.0)
            .map(|i| psu_idx[i])
            .collect();
        let m = psus_in_dom.len() as isize;
        if m <= 1 { f64::NAN } else { (m - 1) as f64 }
    } else if strata.is_some() {
        let mut n_dom: isize = 0;
        let mut strata_in_dom: HashSet<usize> = HashSet::new();
        for i in 0..n {
            let in_domain = domain_mask.is_none_or(|m| m[i]);
            if in_domain && w_samp[i] > 0.0 {
                n_dom += 1;
                strata_in_dom.insert(strata_idx[i]);
            }
        }
        let df = n_dom - (strata_in_dom.len() as isize);
        if df <= 0 { f64::NAN } else { df as f64 }
    } else {
        let n_dom: isize = (0..n)
            .filter(|&i| domain_mask.is_none_or(|m| m[i]) && w_samp[i] > 0.0)
            .count() as isize;
        if n_dom <= 1 {
            f64::NAN
        } else {
            (n_dom - 1) as f64
        }
    };

    // Reporting-only Pearson phi, R/Stata glm's "(1/df) Pearson" for every family; design SEs never use it.
    // Divided by n_obs - k, not the design df: a moment estimate, while the design df is the t reference's.
    let scale = {
        let mut pearson = 0.0;
        for i in 0..n {
            if !(domain_mask.is_none_or(|m| m[i]) && w_samp[i] > 0.0) {
                continue;
            }
            let mu_i = mu[i];
            let v = family.variance(mu_i).max(1e-12);
            pearson += w_samp[i] * (y_vals[i] - mu_i).powi(2) / v;
        }
        pearson / ((n_obs - k) as f64)
    };

    // Without an offset the intercept-only MLE is the weighted mean for any link; with one it needs IRLS, as R's glm().
    let null_deviance = if offset.is_some() {
        null_deviance_with_offset(
            family,
            link,
            n,
            &y_vals,
            &w_samp,
            &offset_vals,
            domain_mask,
            tol,
            max_iter,
        )
    } else {
        let mut sum_wy = 0.0;
        let mut sum_w = 0.0;
        for i in 0..n {
            let in_domain = domain_mask.is_none_or(|m| m[i]);
            if !in_domain || w_samp[i] <= 0.0 {
                continue;
            }
            sum_wy += y_vals[i] * w_samp[i];
            sum_w += w_samp[i];
        }
        let y_mean = if sum_w > 0.0 { sum_wy / sum_w } else { 0.0 };

        let mut dev0 = 0.0;
        for i in 0..n {
            let in_domain = domain_mask.is_none_or(|m| m[i]);
            if !in_domain {
                continue;
            }
            let w_i = w_samp[i];
            if w_i <= 0.0 {
                continue;
            }
            let y_i = y_vals[i];
            dev0 += w_i * family.unit_deviance(y_i, y_mean);
        }
        dev0
    };

    let _ = w_sum;

    let separation = separation_check(
        family,
        n,
        k,
        &y_vals,
        &x,
        &w_samp,
        &mu,
        domain_mask,
        deviance,
        tol,
    );

    let params: Vec<f64> = (0..k).map(|i| beta[(i, 0)]).collect();
    let mut cov_flat = Vec::with_capacity(k * k);
    let mut naive_flat = Vec::with_capacity(k * k);
    for r in 0..k {
        for c in 0..k {
            cov_flat.push(cov[(r, c)]);
            naive_flat.push(bread[(r, c)]);
        }
    }

    Ok(GlmResult {
        params,
        cov_params: cov_flat,
        naive_cov: naive_flat,
        scale,
        df_resid,
        deviance,
        null_deviance,
        iterations: iter_count as u32,
        n_obs,
        converged,
        theta: None,
        theta_se: None,
        separation,
    })
}

/// Quasi-complete separation: boundary rows whose deviance share is below what convergence can resolve.
///
/// Recession directions move only those rows, so they are the null space of the other rows' Gram matrix, returned.
#[allow(clippy::too_many_arguments)]
fn separation_check(
    family: Family,
    n: usize,
    k: usize,
    y_vals: &[f64],
    x: &[f64],
    w_samp: &[f64],
    mu: &[f64],
    domain_mask: Option<&[bool]>,
    deviance: f64,
    tol: f64,
) -> Option<Separation> {
    // unit_deviance's clamps: past them the deviance no longer moves at all.
    const MU_FLOOR: f64 = 1e-10;
    let binomial = match family {
        Family::Binomial => true,
        Family::Poisson | Family::NegativeBinomial(_) => false,
        _ => return None,
    };
    let resolution = SEPARATION_RESOLUTION * tol * (deviance + 0.1);

    let mut retained = vec![0.0f64; n];
    let mut boundary_rows = 0usize;
    for i in 0..n {
        let w_i = w_samp[i];
        if !(domain_mask.is_none_or(|m| m[i]) && w_i > 0.0) {
            continue;
        }
        let (y_i, mu_i) = (y_vals[i], mu[i]);
        let at_floor = if binomial {
            (y_i == 0.0 && mu_i <= MU_FLOOR) || (y_i == 1.0 && mu_i >= 1.0 - MU_FLOOR)
        } else {
            y_i == 0.0 && mu_i <= MU_FLOOR
        };
        let at_edge = y_i == 0.0 || (binomial && y_i == 1.0);
        // Binomial deviance is -2 ln(u) >= 2 (1 - u): a row this far out skips the log.
        let far = binomial && {
            let mu_c = mu_i.clamp(MU_FLOOR, 1.0 - MU_FLOOR);
            w_i * if y_i == 1.0 { 1.0 - mu_c } else { mu_c } >= resolution
        };
        if at_edge && (at_floor || (!far && w_i * family.unit_deviance(y_i, mu_i) < resolution)) {
            boundary_rows += 1;
        } else {
            retained[i] = w_i;
        }
    }
    if boundary_rows == 0 {
        return None;
    }

    let mut upper = vec![0.0f64; k * k];
    accumulate_weighted_crossprod(x, &retained, n, k, &mut upper);
    let mut gram = vec![0.0f64; k * k];
    for a in 0..k {
        for b in a..k {
            gram[a * k + b] = upper[a * k + b];
            gram[b * k + a] = upper[a * k + b];
        }
    }
    Some(Separation {
        boundary_rows,
        gram,
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// R: pnorm(x) at 17 significant digits.
    #[test]
    #[allow(clippy::excessive_precision)] // as R prints them
    fn norm_cdf_matches_r_pnorm() {
        let cases: [(f64, f64); 13] = [
            (-8.5, 9.4795348222033192e-18),
            (-5.0, 2.8665157187919391e-07),
            (-2.5, 0.0062096653257761349),
            (-1.0, 0.15865525393145705),
            (-0.6, 0.27425311775007361),
            (-0.3, 0.38208857781104733),
            (0.0, 0.5),
            (0.3, 0.61791142218895267),
            (0.6, 0.72574688224992645),
            (1.0, 0.84134474606854293),
            (2.5, 0.99379033467422384),
            (5.0, 0.99999971334842808),
            (8.5, 1.0),
        ];
        for (x, expected) in cases {
            let got = norm_cdf(x);
            let tol = 1e-14 * expected.max(1e-300);
            assert!(
                (got - expected).abs() <= tol,
                "pnorm({x}): got {got:e}, want {expected:e}"
            );
        }
    }

    /// Probit's forward link is Acklam's qnorm (~1.15e-9), which only seeds IRLS: hence the looser bound.
    #[test]
    fn link_roundtrips() {
        for (link, tol) in [
            (Link::Logit, 1e-12),
            (Link::Probit, 1e-9),
            (Link::Cauchit, 1e-12),
            (Link::Cloglog, 1e-12),
            (Link::Sqrt, 1e-12),
        ] {
            for mu in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
                let back = link.inverse(link.link(mu));
                assert!(
                    (back - mu).abs() < tol,
                    "{link:?} roundtrip at mu={mu}: got {back}"
                );
            }
        }
    }

    /// dmu/deta against a central difference of the inverse link.
    #[test]
    fn mu_eta_matches_numeric_derivative() {
        let h = 1e-6;
        for link in [Link::Probit, Link::Cauchit, Link::Cloglog, Link::Sqrt] {
            for eta in [-3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0] {
                let numeric = (link.inverse(eta + h) - link.inverse(eta - h)) / (2.0 * h);
                let analytic = link.mu_eta(link.inverse(eta), eta);
                assert!(
                    (analytic - numeric).abs() < 1e-7,
                    "{link:?} mu_eta at eta={eta}: analytic {analytic}, numeric {numeric}"
                );
            }
        }
    }

    /// Level 'c' (every third row) has no event but row 2's `y_c`; at 0 its dummy drifts to -infinity.
    fn separated_fit(y_c: f64) -> GlmResult {
        let n = 90;
        let ones = vec![1.0; n];
        let x: Vec<f64> = (0..n).map(|i| ((i as f64) * 1.7).sin()).collect();
        let c: Vec<f64> = (0..n).map(|i| if i % 3 == 2 { 1.0 } else { 0.0 }).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                if i % 3 == 2 {
                    if i == 2 { y_c } else { 0.0 }
                } else if (x[i] + ((i as f64) * 2.3).cos()) > 0.2 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        fit_glm(
            &Series::new("y".into(), y),
            vec![
                Series::new("one".into(), ones.clone()),
                Series::new("x".into(), x),
                Series::new("c".into(), c),
            ],
            &Series::new("w".into(), ones),
            None,
            None,
            None,
            None,
            "binomial",
            "logit",
            None,
            1e-8,
            100,
            None,
        )
        .unwrap()
    }

    #[test]
    fn separation_reports_boundary_rows_and_retained_gram() {
        let r = separated_fit(0.0);
        let sep = r.separation.expect("separated fit");
        assert_eq!(sep.boundary_rows, 30);
        // The separated dummy is zero on every retained row.
        for j in 0..3 {
            assert_eq!(sep.gram[2 * 3 + j], 0.0);
            assert_eq!(sep.gram[j * 3 + 2], 0.0);
        }
        assert_eq!(sep.gram[0], 60.0);
    }

    #[test]
    fn no_separation_record_on_an_identified_fit() {
        // One event in level 'c' makes the MLE finite.
        assert!(separated_fit(1.0).separation.is_none());
    }

    #[test]
    fn short_or_long_inputs_raise_dimension_mismatch() {
        use crate::errors::Coded;
        let y = Series::new("y".into(), vec![0.0, 1.0, 2.0, 1.0, 3.0, 2.0]);
        let inputs = [
            "weights",
            "strata",
            "psu",
            "fpc",
            "offset",
            "domain mask",
            "predictor",
        ];
        for family in ["poisson", "negativebinomial"] {
            for (what, got) in inputs.iter().flat_map(|&w| [(w, 5), (w, 7)]) {
                let len = |name: &str| if name == what { got } else { 6 };
                let col = |name: &str| Series::new(name.into(), vec![1.0; len(name)]);
                let mask = vec![true; len("domain mask")];
                let err = fit_one(
                    &y,
                    vec![col("predictor")],
                    &col("weights"),
                    Some(&col("strata")),
                    Some(&col("psu")),
                    Some(&col("fpc")),
                    Some(&col("offset")),
                    Some(&mask),
                    family,
                    "log",
                    None,
                    1e-8,
                    100,
                    None,
                    SingletonRule::NONE,
                )
                .err()
                .expect("mismatch must fail");
                assert!(
                    matches!(
                        err,
                        RegressionError::DimensionMismatch { what: w, expected: 6, got: g }
                            if w == what && g == got
                    ),
                    "{family} {what}: {err:?}"
                );
                assert_eq!(
                    err.to_string(),
                    format!("{what} has length {got}, expected 6")
                );
                assert_eq!(err.code(), "SHAPE_MISMATCH");
                assert!(err.invalid_input());
            }
        }
    }

    #[test]
    fn cols_to_vec_copies_every_chunk_and_refuses_other_lengths() {
        let mut chunked = Float64Chunked::from_slice("a".into(), &[1.0, 2.0]);
        chunked
            .append(&Float64Chunked::from_slice("a".into(), &[3.0]))
            .unwrap();
        let other = Float64Chunked::from_slice("b".into(), &[4.0, 5.0, 6.0]);
        assert_eq!(
            cols_to_vec(&[&chunked, &other], 3).unwrap(),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );

        for (vals, got) in [(&[4.0, 5.0][..], 2), (&[4.0, 5.0, 6.0, 7.0][..], 4)] {
            let bad = Float64Chunked::from_slice("b".into(), vals);
            let err = cols_to_vec(&[&chunked, &bad], 3).unwrap_err();
            assert_eq!(
                err.to_string(),
                format!("column has length {got}, expected 3")
            );
        }
    }
}
