// src/regression/glm.rs
//
// Survey-robust GLM via IRLS + sandwich variance
// Goal: match R survey::svyglm (linearization/sandwich) numerically.
//
// Key alignment points:
// - Normalize weights to sum(w)=n for conditioning (sandwich invariant to global scaling).
// - Build bread from FINAL (converged) Fisher information (XtWX) at final eta/mu.
// - Meat: PSU totals of per-row score contributions, centered within stratum, scaled m/(m-1).
//
// Domain estimation (by_col / where):
// - Out-of-domain rows contribute 0 to both IRLS normal equations and sandwich meat.
// - Strata/PSU enumeration uses the full design — domain only affects which rows
//   contribute, not the m/(m-1) centering structure. This matches R's
//   subset(design, ...) semantics and the estimation namespace's by_col pattern.
//
// NOTE: This implements the classic "bread %*% meat %*% bread" route.

// #![allow(non_upper_case_globals)]
#![allow(non_snake_case)]

use faer::prelude::Solve;
use faer::{Mat, MatRef, Side};

use polars::prelude::*;

use crate::categorical::ranktest::probit;
use crate::estimation::calib_sweep::CalibSweep;
use crate::estimation::taylor::{design_col_codes, design_pair_codes};
use rayon::prelude::*;

// ============================================================================
// Enums & Config
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Family {
    Gaussian,
    Binomial,
    Poisson,
    Gamma,
    InverseGaussian,
    /// Negative binomial with a KNOWN dispersion: Var(mu) = mu + mu^2/theta.
    ///
    /// Estimating theta is the Python side's outer loop. It needs digamma and
    /// trigamma, and this crate carries no math dependency; scipy is already a
    /// dependency of `svy`, so the profile likelihood lives there and the
    /// kernel only ever sees a fixed theta.
    NegativeBinomial(f64),
}

impl Family {
    pub fn from_str(s: &str, theta: Option<f64>) -> PolarsResult<Self> {
        match s.to_lowercase().as_str() {
            "gaussian" => Ok(Family::Gaussian),
            "binomial" => Ok(Family::Binomial),
            "poisson" => Ok(Family::Poisson),
            "gamma" => Ok(Family::Gamma),
            "inversegaussian" | "inverse_gaussian" => Ok(Family::InverseGaussian),
            "negativebinomial" | "negative_binomial" => {
                let th = theta.ok_or_else(|| {
                    PolarsError::ComputeError(
                        "negative binomial needs its dispersion: pass theta".into(),
                    )
                })?;
                if !(th.is_finite() && th > 0.0) {
                    return Err(PolarsError::ComputeError(
                        format!("negative binomial theta must be finite and positive, got {th}")
                            .into(),
                    ));
                }
                Ok(Family::NegativeBinomial(th))
            }
            _ => Err(PolarsError::ComputeError(
                format!("Unsupported family: {}", s).into(),
            )),
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

    /// R's `family$initialize` starting value, on the prior-weight scale.
    ///
    /// IRLS stops on a relative change in the deviance, so on a flat surface
    /// where it starts decides where it stops: seeding binomial at
    /// `(y + 1/2)/2` instead of R's `(w y + 1/2)/(w + 1)` left cloglog's
    /// coefficients 3e-5 from R's on apistrat, with the deviances agreeing to
    /// 14 digits.
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

    /// Unit deviance d(y, mu) — matches R's family$dev.resids (squared,
    /// per-observation) so that sum_i w_i * d(y_i, mu_i) reproduces R's
    /// glm/svyglm deviance on the same weight scale.
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

/// pcauchy: the standard Cauchy CDF.
///
/// `atan(x)/pi + 0.5` loses digits in the tails — at eta = -1000 the two terms
/// cancel to three and a half digits, which showed up as ~1e-7 in the fitted
/// coefficients against R. Outside [-1, 1] the tail is computed from
/// `atan(1/x)`, which is what R's pcauchy does.
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
pub enum Link {
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

// Standard normal CDF/PDF backing the probit link.
//
// Phi(x) = erfc(-x/sqrt2)/2, always evaluated on the tail side so that a mu
// near 0 or 1 keeps full relative precision — a cheap erf approximation is
// visible directly in the fitted probit coefficients. erfc uses the
// all-positive confluent series below 1 (no cancellation, ~19 terms) and a
// modified-Lentz continued fraction above, where the series would cancel.
// Agrees with R's pnorm to ~5e-15 relative across the range (see tests).

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

fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x * std::f64::consts::FRAC_1_SQRT_2)
}

impl Link {
    pub fn from_str(s: &str) -> PolarsResult<Self> {
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
            _ => Err(PolarsError::ComputeError(
                format!("Unsupported link: {}", s).into(),
            )),
        }
    }

    pub(crate) fn link(&self, mu: f64) -> f64 {
        match self {
            Link::Identity => mu,
            Link::Logit => (mu / (1.0 - mu)).ln(),
            Link::Probit => probit(mu),
            // qcauchy: -1/tan(pi*mu) rather than tan(pi*(mu - 1/2)), which
            // is catastrophic as mu approaches 0 or 1 (the argument
            // approaches +/-pi/2). R's qcauchy takes the same cotangent form.
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

            // Clamps mirror R's make.link("probit"/"cloglog"): eta is bounded at
            // +/-qnorm(eps) and mu at [eps, 1-eps], so mu never reaches 0 or 1
            // exactly and the binomial variance stays positive.
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

    /// d2μ/dη2 — only the negative binomial's joint (beta, theta) bread needs
    /// it, and only for the links that family admits.
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

/// Materialise columns into one column-major `nrows * ncols` buffer.
///
/// Nulls are rejected upstream in `fit_glm_domain`, so each chunk's value
/// slice is dense and copies wholesale.
pub(crate) fn cols_to_vec(cols: &[&Float64Chunked], nrows: usize) -> Vec<f64> {
    let mut out = vec![0.0; nrows * cols.len()];
    for (j, col) in cols.iter().enumerate() {
        let dst = &mut out[j * nrows..(j + 1) * nrows];
        let mut off = 0usize;
        for arr in col.downcast_iter() {
            let vals = arr.values().as_slice();
            let end = (off + vals.len()).min(nrows);
            dst[off..end].copy_from_slice(&vals[..end - off]);
            off = end;
        }
    }
    out
}

/// Dense 0-based design codes for `strata` (and, when given, PSUs nested
/// within them), reusing the estimation namespace's factorizers: the Python
/// layer already hands the kernel integer code columns, and hashing those is
/// ~10x cheaper than casting to String and hashing `(&str, &str)` per row.
///
/// Nulls arrive as `u32::MAX`; they are collected into one extra level so the
/// code can index a `Vec` directly.
pub(crate) fn design_codes(
    strata: Option<&Series>,
    psu: Option<&Series>,
    n: usize,
) -> PolarsResult<(Vec<usize>, usize)> {
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

/// Rows per cache block in the cross-product accumulations. The X block of a
/// 256-row window is k*2 KB, which stays in L1/L2 for every realistic k.
const XP_BLOCK: usize = 256;

/// Rows per parallel chunk. Fixed rather than derived from the thread count,
/// so the summation order — and therefore the result — is identical on every
/// machine and at every core count.
const XP_CHUNK: usize = 8192;

/// `out[a*k + b] += sum_i w[i] * cols[a*n + i] * cols[b*n + i]`, upper
/// triangle only, over the rows `[0, n)` of a column-major `cols`.
///
/// Columns-outer with the row block innermost: the inner loop is a contiguous
/// dot product the compiler can vectorize, and `w * cols[a]` is formed once
/// per (block, a) instead of once per (block, a, b). Blocks are summed in
/// index order within a chunk and chunks in index order, so the result does
/// not depend on how rayon schedules them.
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

/// Build XtWX and XtWz from the current eta/mu (one IRLS step), mirroring
/// fisherinf: t(D) %*% (w * D / V), D = X * d, d = dmu/deta.
///
/// When `domain_mask` is `Some`, rows where mask[i] is false contribute 0 to
/// both XtWX and XtWz (and have `w_irls[i]` set to 0). When `None`, every row
/// contributes.
///
/// Summation is plain blocked f64, as R's `crossprod` is; Kahan compensation
/// bought ~1e-16 on a quantity the sandwich is insensitive to and cost the
/// vectorization of the whole accumulation.
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
    // Pass 1 — per-row IRLS weight and working response. O(n), no k factor.
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
        let d = link.mu_eta(mu_i, eta[i]); // dμ/dη
        let wi = w_i * (d * d) / v;

        let safe_d = if d.abs() < 1e-12 { 1e-12 } else { d };
        // Working response on the X-only scale: the offset is a known part of
        // eta with no parameter, so it comes out here and goes back in wherever
        // eta is rebuilt (R's glm.fit: z <- (eta - offset) + (y - mu)/mu.eta).
        z[i] = (eta[i] - offset[i]) + (y_vals[i] - mu_i) / safe_d;
        w_irls[i] = if wi < 1e-18 { 0.0 } else { wi };
    }

    // Pass 2 — XtWX (blocked, columns-outer) and XtWz.
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

/// Solve A x = b with deterministic fallback chain.
fn solve_linear_system(A: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Mat<f64> {
    if let Ok(chol) = A.llt(Side::Lower) {
        return chol.solve(b);
    }

    // Symmetric indefinite
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

    // LU fallback
    let lu = A.partial_piv_lu();
    let x2 = lu.solve(b);
    for i in 0..x2.nrows() {
        if !x2[(i, 0)].is_finite() {
            // Last resort: SVD pseudoinverse. If even the SVD fails to
            // converge (degenerate system, e.g. a constant response in a
            // complement domain), return the non-finite LU solution — the
            // caller surfaces it as a fit error instead of a panic.
            return match A.thin_svd() {
                Ok(svd) => svd.pseudoinverse() * b,
                Err(_) => x2,
            };
        }
    }
    x2
}

/// Per-stratum `(1 - f_h) * m/(m-1)`, zero where the stratum cannot contribute
/// (a single PSU), which is how those rows are kept out of the accumulation.
///
/// Returns `(scale_h, psus_h)`.
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

/// Design-based variance-covariance of the totals of `p` already-weighted
/// columns — R survey's `svyrecvar`.
///
/// `cols` is column-major, `n * p`. PSU totals are centred within stratum and
/// scaled by `stratum_scales`. Returns the full symmetric `p * p` matrix,
/// row-major.
///
/// The caller passes influence functions, not raw variables: the weights are
/// already folded into the columns.
pub(crate) fn design_vcov_of_totals(
    cols: &[f64],
    n: usize,
    p: usize,
    strata_idx: &[usize],
    strata_obs: &[Vec<usize>],
    psu_idx: Option<&[usize]>,
    n_psu_levels: usize,
    fpc_rows: Option<&[f64]>,
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
                for j in 0..p {
                    mean[j] /= m as f64;
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
    out
}

/// Refuse an aliased design matrix.
///
/// A pivot-free Cholesky of XtWX in the model's own column order: column j is
/// aliased when the variance it still has after projecting out the columns
/// before it is a vanishing fraction of its own. That is R's "first columns
/// win" rule, but raising instead of reporting NA coefficients — consistent
/// with the rest of svy, and better than today's alternative, a fit that
/// "succeeds" with SE 0 or NaN and an F statistic around 1e30.
fn check_rank(XtWX: &Mat<f64>, k: usize, cols: &[Series]) -> PolarsResult<()> {
    // An exactly aliased column leaves ~1e-16 of its own norm; a genuinely
    // collinear-but-distinct one (R^2 = 0.999999) leaves 1e-6. This separates
    // them by five orders of magnitude either way.
    const ALIAS_TOL: f64 = 1e-11;

    let mut l = vec![0.0f64; k * k]; // column-major lower factor
    let mut aliased: Vec<usize> = Vec::new();

    for j in 0..k {
        let mut d = XtWX[(j, j)];
        for p in 0..j {
            d -= l[p * k + j] * l[p * k + j];
        }
        // Negated so a NaN diagonal is aliased rather than accepted.
        if !(d > ALIAS_TOL * XtWX[(j, j)]) {
            // Its factor column stays zero, so the columns after it are
            // measured against the ones that were actually kept.
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
    Err(PolarsError::ComputeError(
        format!(
            "GLM design matrix is rank deficient: {} is collinear with the columns \
             before it. Drop it, or one of the columns it duplicates.",
            listed.join(", ")
        )
        .into(),
    ))
}

/// Compute A^{-1} via solving A X = I with same solve strategy.
///
/// Returns an error rather than panicking when even the SVD fails: a single
/// non-finite entry in the information matrix used to reach `thin_svd().unwrap()`.
pub(crate) fn invert_matrix(A: MatRef<'_, f64>, k: usize) -> PolarsResult<Mat<f64>> {
    if let Ok(chol) = A.llt(Side::Lower) {
        let mut inv = Mat::<f64>::identity(k, k);
        chol.solve_in_place(inv.as_mut());
        return Ok(inv);
    }

    let lblt = A.lblt(Side::Lower);
    let mut inv = Mat::<f64>::identity(k, k);
    lblt.solve_in_place(inv.as_mut());

    // sanity: if not finite, LU then SVD
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
                                Err(_) => Err(PolarsError::ComputeError(
                                    "GLM information matrix could not be inverted \
                                     (degenerate or non-finite fit)"
                                        .into(),
                                )),
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
pub struct GlmResult {
    pub params: Vec<f64>,
    pub cov_params: Vec<f64>,
    /// Model-based (inverse-information) covariance (X'WX)^-1 at the final
    /// beta — R svyglm's naive.cov, used for the Rao-Scott/dAIC design
    /// effects. Same weight scale as the fit (weights normalized to mean 1).
    pub naive_cov: Vec<f64>,
    pub scale: f64,
    pub df_resid: f64,
    pub deviance: f64,
    pub null_deviance: f64,
    pub iterations: u32,
    pub n_obs: usize,
    /// Whether IRLS met the tolerance rather than exhausting `max_iter`.
    pub converged: bool,
    /// Negative binomial dispersion, and its design-based standard error when
    /// it was estimated rather than supplied. `None` for every other family.
    pub theta: Option<f64>,
    pub theta_se: Option<f64>,
}

// ============================================================================
// Core Algorithm
// ============================================================================

/// Null deviance under an offset: the deviance of the intercept-only model
/// eta_i = b + offset_i, found by one-parameter IRLS. Mirrors R's glm(), which
/// refits `y ~ 1` carrying the offset rather than using the weighted mean.
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
    let contributes = |i: usize| domain_mask.map_or(true, |m| m[i]) && w_samp[i] > 0.0;

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

    if deviance.is_finite() { deviance } else { 0.0 }
}

/// One fit, dispatching on the family.
///
/// Only the negative binomial has anything to do before IRLS: unless its
/// dispersion was supplied it has to be estimated, and the variance then spans
/// (beta, theta) rather than beta alone. Everything else goes straight to the
/// IRLS solve.
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
) -> PolarsResult<GlmResult> {
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
        true,
    )
}

/// Full-sample GLM fit (no domain restriction).
///
/// Equivalent to `fit_glm_domain(..., domain_mask=None)`. Kept as a thin
/// wrapper to preserve the existing public API and call sites.
pub fn fit_glm(
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
) -> PolarsResult<GlmResult> {
    fit_one(
        y, x_cols, weights, strata, psu, fpc, offset, None, family_str, link_str, theta, tol,
        max_iter, calib,
    )
}

/// Single fit restricted to the rows where `mask` is true.
///
/// A `where=` clause is one domain, not a by-variable: routing it through
/// `fit_glm_by` fitted the complement as well and threw it away, doubling the
/// CPU (rayon only hid it) and turning a degenerate complement into a
/// swallowed error.
#[allow(clippy::too_many_arguments)]
pub fn fit_glm_where(
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
) -> PolarsResult<GlmResult> {
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
    )
}

/// Per-domain GLM fits over the levels of `by_col`.
///
/// Returns one (level, GlmResult) pair per unique level of by_col. The full
/// design (strata, PSU) is preserved across all fits — only the rows
/// contributing to the IRLS and the sandwich meat are restricted to the
/// domain. This produces correct domain-estimation SEs matching R's
/// `svyglm(..., design = subset(d, ...))`.
pub fn fit_glm_by(
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
) -> PolarsResult<Vec<(String, GlmResult)>> {
    // Materialize by_col as strings, enumerate unique levels.
    let by_str_series = by_col.cast(&DataType::String)?;
    let by_str = by_str_series.str()?;
    let unique_groups = by_str.unique()?;

    // Domain fits are independent; fan them out over the rayon pool and collect
    // in level order (deterministic, thread-count-independent — see the policy
    // note in estimation/mod.rs). Each fit is a full, self-contained IRLS solve.
    let groups: Vec<&str> = unique_groups.iter().flatten().collect();
    let attempts: Vec<(String, PolarsResult<GlmResult>)> = groups
        .par_iter()
        .map(|&group_val| {
            let mask_vec: Vec<bool> = by_str
                .iter()
                .map(|v| v.map_or(false, |s| s == group_val))
                .collect();

            // Clone the x_cols Vec for each domain fit. The Series themselves
            // are cheap reference-counted handles in polars; the clone copies
            // only the Vec, not the underlying data.
            let xs = x_cols.iter().cloned().collect();
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
            );
            (group_val.to_string(), res)
        })
        .collect();

    // A level that fails to fit used to be dropped silently, so a caller
    // asking for every level got a short list with no way to tell which one
    // was missing or why.
    let mut results = Vec::with_capacity(attempts.len());
    let mut failed: Vec<String> = Vec::new();
    let mut first_err: Option<PolarsError> = None;
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
    if !failed.is_empty() {
        let detail = first_err
            .map(|e| e.to_string())
            .unwrap_or_else(|| "unknown error".to_string());
        return Err(PolarsError::ComputeError(
            format!(
                "GLM failed on {} of {} levels of '{}' ({}): {detail}",
                failed.len(),
                failed.len() + results.len(),
                by_col.name(),
                failed.join(", ")
            )
            .into(),
        ));
    }
    Ok(results)
}

/// Core GLM fit. When `domain_mask` is `None`, behavior is byte-identical to
/// the original `fit_glm`. When `Some`, out-of-domain rows contribute 0 to
/// both the IRLS loop and the sandwich meat, while the strata/PSU
/// enumeration remains based on the full design.
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
    want_variance: bool,
) -> PolarsResult<GlmResult> {
    let family = Family::from_str(family_str, theta)?;
    let link = Link::from_str(link_str)?;

    // 1) Data prep
    let n = y.len();
    let k = x_cols.len();

    // Cast y/weights/x to Float64 (error, not panic, on incompatible dtypes),
    // then reject nulls: a null anywhere would otherwise silently misalign
    // the y/X/w rows via null-skipping iteration.
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
        return Err(PolarsError::ComputeError(
            format!("GLM response column '{}' contains null values", y.name()).into(),
        ));
    }
    if w_ca.null_count() > 0 {
        return Err(PolarsError::ComputeError(
            format!(
                "GLM weight column '{}' contains null values",
                weights.name()
            )
            .into(),
        ));
    }
    for s in &x_cast {
        if s.null_count() > 0 {
            return Err(PolarsError::ComputeError(
                format!("GLM predictor column '{}' contains null values", s.name()).into(),
            ));
        }
    }

    let y_vals = cols_to_vec(&[y_ca], n);
    let x = cols_to_vec(&x_ca_list, n);

    // Known term on the link scale, coefficient fixed at 1. Absent -> all zero,
    // which makes every offset expression below a no-op on the existing path.
    let offset_vals: Vec<f64> = match offset {
        Some(s) => {
            let cast = s.cast(&DataType::Float64)?;
            let ca = cast.f64()?;
            if ca.null_count() > 0 {
                return Err(PolarsError::ComputeError(
                    format!("GLM offset column '{}' contains null values", s.name()).into(),
                ));
            }
            ca.iter().map(|v| v.unwrap_or(0.0)).collect()
        }
        None => vec![0.0; n],
    };

    // sampling weights
    let mut w_samp = vec![0.0; n];
    let mut w_sum = 0.0;
    for (i, v) in w_ca.iter().enumerate() {
        let v = v.unwrap_or(0.0);
        w_samp[i] = v;
        w_sum += v;
    }

    // Normalize weights: sum(w)=n
    if w_sum > 0.0 {
        let scale = (n as f64) / w_sum;
        for wi in &mut w_samp {
            *wi *= scale;
        }
        w_sum = n as f64;
    }

    // Rows that will actually carry the fit. An empty `where` domain or an
    // all-zero weight column used to return a zero fit as `Ok`: the normal
    // equations are then the zero matrix, whose SVD pseudoinverse is also
    // zero, and beta = 0 passes the finite check below.
    let n_obs = (0..n)
        .filter(|&i| domain_mask.is_none_or(|m| m[i]) && w_samp[i] > 0.0)
        .count();
    if n_obs <= k {
        return Err(PolarsError::ComputeError(
            format!("GLM needs more observations than parameters: n_obs={n_obs}, k={k}").into(),
        ));
    }

    // Nulls are rejected above; NaN/Inf are not, and Rust's `f64::max` returns
    // the other operand for a NaN, so `.max(1e-12)` on the variance would hide
    // one all the way to a NaN beta. Only contributing rows matter: a padded
    // zero-weight row never enters an accumulation.
    for i in 0..n {
        if !(domain_mask.is_none_or(|m| m[i]) && w_samp[i] > 0.0) {
            continue;
        }
        if !y_vals[i].is_finite() {
            return Err(PolarsError::ComputeError(
                format!(
                    "GLM response column '{}' contains a non-finite value at row {i}",
                    y.name()
                )
                .into(),
            ));
        }
        if !offset_vals[i].is_finite() {
            return Err(PolarsError::ComputeError(
                format!("GLM offset contains a non-finite value at row {i}").into(),
            ));
        }
        for j in 0..k {
            if !x[j * n + i].is_finite() {
                return Err(PolarsError::ComputeError(
                    format!(
                        "GLM predictor column '{}' contains a non-finite value at row {i}",
                        x_cast[j].name()
                    )
                    .into(),
                ));
            }
        }
    }

    // 2) IRLS init — only meaningful for in-domain rows; out-of-domain rows
    //    get a neutral placeholder since they won't contribute.
    let mut beta = Mat::<f64>::zeros(k, 1);
    let mut mu = vec![0.0; n];
    let mut eta = vec![0.0; n];

    for i in 0..n {
        let in_domain = domain_mask.map_or(true, |m| m[i]);
        let y_init = if in_domain { y_vals[i] } else { 0.5 };
        mu[i] = family.initial_mu(y_init, w_samp[i]);
        // R seeds eta at linkfun(mustart) without the offset; it enters through
        // the working response, and every later eta adds it back explicitly.
        eta[i] = link.link(mu[i]);
    }

    // work arrays
    let mut z_work = vec![0.0f64; n];
    let mut w_irls = vec![0.0; n];
    let mut XtWX = Mat::<f64>::zeros(k, k);
    let mut XtWz = Mat::<f64>::zeros(k, 1);

    // 3) IRLS loop
    //
    // eta/mu enter each pass already evaluated at the current beta — step 4
    // leaves them there and step 6 commits that beta — so there is no leading
    // recomputation.
    let mut iter_count = 0;
    let mut deviance = 0.0;
    let mut converged = false;

    for iter in 0..max_iter {
        iter_count += 1;

        // 1) build normal equations at current beta
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

        // 2) an aliased column makes every SE that follows meaningless, so
        //    say so once, at the first information matrix, instead of
        //    returning a fit built on a pseudoinverse.
        if iter == 0 {
            check_rank(&XtWX, k, &x_cast)?;
        }

        // 3) solve for beta_new
        let beta_new = solve_linear_system(XtWX.as_ref(), XtWz.as_ref());

        // 4) eta/mu at beta_new, column-outer: each term is one contiguous
        //    pass over x, where a row-outer loop strides by n per term.
        for i in 0..n {
            eta[i] = offset_vals[i];
        }
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

        // 5) convergence check
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

        // 6) commit
        beta = beta_new;
        deviance = dev_new;

        // R's glm.fit convergence test, and only that one:
        //   abs(dev - devold) / (abs(dev) + 0.1) < epsilon.
        // svy also stopped on `max_delta < tol`, which fires an iteration
        // early wherever beta has settled while the deviance is still moving
        // in its last few digits — cloglog on apistrat lands 3e-5 from R's
        // coefficients that way. `max_delta` is only a backstop now, for a
        // deviance that oscillates instead of settling (there is no
        // step-halving here), at a threshold too tight to pre-empt R.
        if iter > 0 && (rel_dev < tol || max_delta < tol) {
            converged = true;
            break;
        }
    }

    // A degenerate system (e.g. constant response under binomial) must
    // surface as an error, not NaN coefficients or a panic downstream.
    for j in 0..k {
        if !beta[(j, 0)].is_finite() {
            return Err(PolarsError::ComputeError(
                "GLM did not produce finite coefficients (degenerate or non-convergent fit)".into(),
            ));
        }
    }

    // The negative binomial's theta loop refits beta several times and needs
    // nothing else; the sandwich below is the most expensive part of a fit.
    if !want_variance {
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
        });
    }

    // =========================================================================
    // 4) Sandwich variance (R-alignment: rebuild XtWX at FINAL beta)
    // =========================================================================

    // eta/mu are already at the converged beta (step 4 of the last pass).
    // rebuild XtWX at final beta (bread must match fisherinf)
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

    // strata/psu indices — FULL design (not affected by domain).
    //
    // The Python layer already hands the kernel dense integer code columns
    // (the PSU code encodes the (stratum, psu) pair), so this reuses the
    // estimation namespace's factorizers instead of casting to String and
    // hashing a (&str, &str) per row.
    let (strata_idx, n_strata) = match strata {
        Some(_) => design_codes(strata, None, n)?,
        None => (vec![0usize; n], 1usize),
    };

    let (psu_idx, n_psu_levels) = match psu {
        Some(_) => design_codes(strata, psu, n)?,
        None => ((0..n).collect::<Vec<_>>(), n),
    };

    // Pre-build strata → obs index
    let mut strata_obs: Vec<Vec<usize>> = vec![Vec::new(); n_strata];
    for i in 0..n {
        strata_obs[strata_idx[i]].push(i);
    }

    // Per-row FPC factor (1 - f_h), constant within a stratum; each
    // stratum's meat contribution is multiplied by its factor (matches R
    // svyglm on a design with fpc=). Missing/absent -> 1.0 (no correction).
    let fpc_rows: Option<Vec<f64>> = match fpc {
        Some(s) => {
            let s_cast = s.cast(&DataType::Float64)?;
            let ca = s_cast.f64()?;
            Some(ca.iter().map(|v| v.unwrap_or(1.0)).collect())
        }
        None => None,
    };

    // Score contribution (estimating function) for row i: w (y - mu) (dmu/deta) / V.
    // Computed directly instead of w_irls * working_resid: the working-residual
    // guard (d + eps) biased scores wherever dmu/deta is small (visible at ~1e-7
    // in SEs), while for canonical links d/V cancels exactly.
    //
    // Domain-aware: w_irls[i] is 0 for out-of-domain rows, which therefore score 0.
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

    // Calibration-aware variance. R centres the INFLUENCE functions --
    // `svyrecvar(estfun %*% Ainv, ..., postStrata=)` in `svy.varcoef` -- but every
    // sweep branch is a left multiplication by an operator fixed by the weights,
    // cells and aux matrix alone, applied identically to each column. It therefore
    // commutes with the bread: S(E A) = (S E) A, so centring the estimating
    // functions here and sandwiching afterwards is the same estimator. Checked
    // against survey 4.5 on apiclus1 (both routes agree to 1.6e-14).
    //
    // Two things the sweep needs that the uncentred path does not. Every row must
    // be materialised, zero-score ones included: their weight carries the cell
    // denominators. And every row must then be ACCUMULATED, because centring
    // leaves a zero-score row nonzero -- it is charged its cell's mean. Skipping
    // them is what makes a nearly-right SE.
    //
    // Column-major so each column hands `apply` a contiguous slice.
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

    // MEAT = sum_h Var_h( PSU totals ) with svytotal-style centering.
    //
    // The calibrated path already has the influence functions materialised, so
    // it hands them straight to the shared svyrecvar. The uncalibrated path
    // does not materialise them: at 1e6 x 20 an n x k score matrix is 160 MB,
    // and the PSU total of row i is just score_at(i) times its X row, so the
    // same accumulation runs off X with the score folded into the row weight.
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
        ),
        None => {
            let mut scores = vec![0.0f64; n];
            for i in 0..n {
                scores[i] = score_at(i);
            }
            let (scale_h, psus_h) =
                stratum_scales(&strata_obs, psu_opt, n_psu_levels, fpc_rows.as_deref());
            let mut upper = vec![0.0f64; k * k];

            if psu.is_none() {
                // Every row is its own PSU: the centred cross-product
                // telescopes into one weighted pass over all rows plus a
                // k-vector per stratum.
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
                    for j in 0..k {
                        mean[j] /= m as f64;
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
            out
        }
    };

    let mut meat = Mat::<f64>::zeros(k, k);
    for a in 0..k {
        for b in 0..k {
            meat[(a, b)] = meat_flat[a * k + b];
        }
    }

    // BREAD = (XtWX)^-1 at final beta
    let bread = invert_matrix(XtWX.as_ref(), k)?;

    // Cov = bread * meat * bread
    let tmp = &bread * &meat;
    let cov = &tmp * &bread;

    // df_resid — domain-aware. We restrict the PSU and stratum counts to
    // those with at least one in-domain row with positive weight. This
    // matches R's behavior: domains shrink the effective df when entire
    // PSUs/strata fall outside the domain.
    use std::collections::HashSet;
    let df_resid = if psu.is_some() && strata.is_some() {
        let mut total_psus: usize = 0;
        let mut nonempty_strata: usize = 0;
        for h in 0..n_strata {
            let psus_in_dom: HashSet<usize> = strata_obs[h]
                .iter()
                .filter(|&&i| domain_mask.map_or(true, |m| m[i]) && w_samp[i] > 0.0)
                .map(|&i| psu_idx[i])
                .collect();
            if !psus_in_dom.is_empty() {
                total_psus += psus_in_dom.len();
                nonempty_strata += 1;
            }
        }
        let df = (total_psus as isize) - (nonempty_strata as isize);
        if df <= 0 { 1.0 } else { df as f64 }
    } else if psu.is_some() {
        let psus_in_dom: HashSet<usize> = (0..n)
            .filter(|&i| domain_mask.map_or(true, |m| m[i]) && w_samp[i] > 0.0)
            .map(|i| psu_idx[i])
            .collect();
        let m = psus_in_dom.len() as isize;
        if m <= 1 { 1.0 } else { (m - 1) as f64 }
    } else if strata.is_some() {
        // Count rows in domain and strata containing in-domain rows.
        let mut n_dom: isize = 0;
        let mut strata_in_dom: HashSet<usize> = HashSet::new();
        for i in 0..n {
            let in_domain = domain_mask.map_or(true, |m| m[i]);
            if in_domain && w_samp[i] > 0.0 {
                n_dom += 1;
                strata_in_dom.insert(strata_idx[i]);
            }
        }
        let df = n_dom - (strata_in_dom.len() as isize);
        if df <= 0 { 1.0 } else { df as f64 }
    } else {
        let n_dom: isize = (0..n)
            .filter(|&i| domain_mask.map_or(true, |m| m[i]) && w_samp[i] > 0.0)
            .count() as isize;
        if n_dom <= 1 { 1.0 } else { (n_dom - 1) as f64 }
    };

    // Dispersion (phi), reporting only: the Pearson estimate
    //   sum_i w_i (y_i - mu_i)^2 / V(mu_i) / (n_obs - k),
    // over in-domain positive-weight rows, on the fit's own weight scale.
    // This is R glm()'s and Stata glm's "(1/df) Pearson" for every family,
    // binomial and poisson included: reporting the estimate is the
    // overdispersion diagnostic a count model needs, and design-based SEs
    // never use it, so nothing else moves. The divisor is n_obs - k, not the
    // design df_resid: the dispersion is a moment estimate, while the design
    // df belongs to the t reference distribution.
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

    // Null deviance — family unit deviance at the intercept-only fit. For a
    // GLM with only an intercept the MLE of mu is the weighted mean of y
    // regardless of link (the constant score sum_i w_i (y_i - mu) c = 0), so
    // this reproduces R's null.deviance on the same weight scale.
    //
    // That shortcut dies with an offset: mu then varies by row and the null fit
    // is a genuine one-parameter IRLS, which is why R's glm() refits the
    // intercept-only model with the offset instead of taking a mean.
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
        // domain-restricted weighted mean
        let mut sum_wy = 0.0;
        let mut sum_w = 0.0;
        for i in 0..n {
            let in_domain = domain_mask.map_or(true, |m| m[i]);
            if !in_domain || w_samp[i] <= 0.0 {
                continue;
            }
            sum_wy += y_vals[i] * w_samp[i];
            sum_w += w_samp[i];
        }
        let y_mean = if sum_w > 0.0 { sum_wy / sum_w } else { 0.0 };

        let mut dev0 = 0.0;
        for i in 0..n {
            let in_domain = domain_mask.map_or(true, |m| m[i]);
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

    // Suppress unused-var warning when no domain restriction is active.
    let _ = w_sum;

    // flatten
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
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// R: pnorm(x) at 17 significant digits.
    /// The erfc route agrees to ~1e-14 relative across the whole range.
    #[test]
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

    /// The links must invert each other. Probit's forward link is Acklam's
    /// qnorm (~1.15e-9), which only ever seeds IRLS, hence the looser bound.
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
}
