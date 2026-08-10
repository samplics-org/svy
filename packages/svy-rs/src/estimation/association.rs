// src/estimation/association.rs
//
// Bivariate association estimators: covariance and Pearson correlation.
//
// Both are smooth functions of the same six weighted totals, so they share one
// moment kernel and differ only in the final combination and in the
// linearization handed to the variance machinery.
//
// # Estimands
//
// With W = sum(w), ybar = sum(w y)/W, a = y - ybar, b = x - xbar:
//
//   m_yx = sum(w a b) / W          (and m_yy, m_xx likewise)
//   cov  = m_yx * n/(n-1)
//   corr = m_yx / sqrt(m_yy * m_xx)
//
// The n/(n-1) factor on the covariance matches R's `svyvar`, which builds the
// estimate as `svymean(a*b*n/(n-1))` (survey 4.5, R/survey.R:703-723) with the
// source comment "Kish uses the n-1 divisor, so it affects design effects".
// `n` there is the count of rows with NONZERO SAMPLING WEIGHT and no missing
// value -- not the frame height -- so `where=`/`drop_nulls` rows, which reach
// this kernel zero-weighted rather than deleted, are excluded exactly as R
// excludes them. The correlation needs no such choice: any common divisor
// cancels between numerator and denominator, so corr is divisor-invariant.
//
// # Linearization
//
// For the covariance, R centers by the ESTIMATED mean and then treats the
// cross-product as a fixed analysis variable. That looks like it drops the
// contribution of having estimated ybar and xbar, but those terms cancel
// identically:
//
//   IF = (yx - mu_yx) - xbar(y - ybar) - ybar(x - xbar) = a*b - m_yx
//
// so the shortcut is exactly the delta-method influence function, and our
// covariance SE agrees with `svyvar` rather than merely approximating it.
//
// For the correlation the terms do NOT cancel, because the standard deviations
// in the denominator are themselves estimated. Writing rho as a function of the
// six totals and applying the chain rule collapses to
//
//   IF = yt*xt - (rho/2)(yt^2 + xt^2),   yt = a/s_y,  xt = b/s_x
//
// which is the classic Pearson influence function. Feeding the covariance-style
// score to a correlation would understate its variance.
//
// In both cases the score handed to the variance kernel is z_i = (w_i/W) * IF_i,
// matching the convention already used by `scores_mean` and `scores_ratio`.
//
// # Numerical stability
//
// Moments are accumulated in two passes (means, then centered sums) rather than
// from raw sums of squares. The textbook one-pass form
// `sum(w y^2)/W - ybar^2` cancels catastrophically whenever the mean is large
// relative to the spread -- for income-like data (mean 5e4, sd 1e4) it discards
// roughly seven significant digits, and for a covariance that error lands
// directly in the reported estimate. Two passes cost one extra sweep of memory
// that the Taylor path pays anyway, since the scores cannot be formed until
// rho, the means and the standard deviations are known.
//
// Accumulation is strictly row-ordered and never parallelized within a single
// estimate, per the determinism policy in `mod.rs`: output must stay
// bit-identical across rayon thread counts.

use polars::prelude::*;

/// Weighted bivariate moments shared by the covariance and correlation paths.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BivarMoments {
    /// Sum of weights over contributing rows.
    pub sum_w: f64,
    /// Weighted mean of y.
    pub mean_y: f64,
    /// Weighted mean of x.
    pub mean_x: f64,
    /// sum(w (y-ybar)^2) / W
    pub m_yy: f64,
    /// sum(w (x-xbar)^2) / W
    pub m_xx: f64,
    /// sum(w (y-ybar)(x-xbar)) / W
    pub m_yx: f64,
    /// Count of rows with nonzero weight and no missing value, matching the
    /// `n` R's `svyvar` uses for its n/(n-1) factor.
    pub n: usize,
}

impl BivarMoments {
    /// Kish's n/(n-1) inflation, or NaN when fewer than two rows contribute.
    #[inline]
    fn kish_factor(&self) -> f64 {
        if self.n < 2 {
            f64::NAN
        } else {
            self.n as f64 / (self.n as f64 - 1.0)
        }
    }

    /// Design-based covariance, on R's `svyvar` scale.
    #[inline]
    pub fn cov(&self) -> f64 {
        self.m_yx * self.kish_factor()
    }

    /// Design-based Pearson correlation. NaN when either variable is constant,
    /// which is the same answer `cov2cor` gives a zero-variance diagonal.
    #[inline]
    pub fn corr(&self) -> f64 {
        let denom = (self.m_yy * self.m_xx).sqrt();
        if denom > 0.0 { self.m_yx / denom } else { f64::NAN }
    }
}

/// True when both slices are contiguous and null-free, enabling the fast path.
fn cont_triple<'a>(
    y: &'a Float64Chunked,
    x: &'a Float64Chunked,
    w: &'a Float64Chunked,
) -> Option<(&'a [f64], &'a [f64], &'a [f64])> {
    if y.null_count() == 0
        && x.null_count() == 0
        && w.null_count() == 0
        && let (Ok(sy), Ok(sx), Ok(sw)) = (y.cont_slice(), x.cont_slice(), w.cont_slice())
    {
        return Some((sy, sx, sw));
    }
    None
}

/// Accumulate weighted bivariate moments over the rows selected by `domain`
/// (all rows when `None`).
///
/// Rows that are null in y, x or w, and rows outside the domain, contribute
/// nothing and are not counted in `n`. Zero-weight rows contribute nothing to
/// the sums and are likewise excluded from `n` -- that is how `where=` and
/// `drop_nulls` filtering, which arrive here as zeroed weights, stay consistent
/// with R's definition of `n`.
pub fn bivar_moments(
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked,
    domain: Option<&BooleanChunked>,
) -> PolarsResult<BivarMoments> {
    let mut sum_w = 0.0f64;
    let mut sum_wy = 0.0f64;
    let mut sum_wx = 0.0f64;
    let mut n = 0usize;

    // ---- pass 1: weights, means, and R's n -------------------------------
    match (cont_triple(y, x, weights), domain) {
        (Some((ys, xs, ws)), None) => {
            for i in 0..ys.len() {
                let w = ws[i];
                sum_w += w;
                sum_wy += w * ys[i];
                sum_wx += w * xs[i];
                if w != 0.0 {
                    n += 1;
                }
            }
        }
        _ => {
            for (((yi, xi), wi), mi) in y
                .iter()
                .zip(x.iter())
                .zip(weights.iter())
                .zip(mask_iter(domain, y.len()))
            {
                if let (Some(yv), Some(xv), Some(wv), true) = (yi, xi, wi, mi) {
                    sum_w += wv;
                    sum_wy += wv * yv;
                    sum_wx += wv * xv;
                    if wv != 0.0 {
                        n += 1;
                    }
                }
            }
        }
    }

    if sum_w == 0.0 {
        return Ok(BivarMoments {
            sum_w: 0.0,
            mean_y: f64::NAN,
            mean_x: f64::NAN,
            m_yy: f64::NAN,
            m_xx: f64::NAN,
            m_yx: f64::NAN,
            n,
        });
    }

    let mean_y = sum_wy / sum_w;
    let mean_x = sum_wx / sum_w;

    // ---- pass 2: centered second moments ---------------------------------
    let mut s_yy = 0.0f64;
    let mut s_xx = 0.0f64;
    let mut s_yx = 0.0f64;

    match (cont_triple(y, x, weights), domain) {
        (Some((ys, xs, ws)), None) => {
            for i in 0..ys.len() {
                let w = ws[i];
                let a = ys[i] - mean_y;
                let b = xs[i] - mean_x;
                s_yy += w * a * a;
                s_xx += w * b * b;
                s_yx += w * a * b;
            }
        }
        _ => {
            for (((yi, xi), wi), mi) in y
                .iter()
                .zip(x.iter())
                .zip(weights.iter())
                .zip(mask_iter(domain, y.len()))
            {
                if let (Some(yv), Some(xv), Some(wv), true) = (yi, xi, wi, mi) {
                    let a = yv - mean_y;
                    let b = xv - mean_x;
                    s_yy += wv * a * a;
                    s_xx += wv * b * b;
                    s_yx += wv * a * b;
                }
            }
        }
    }

    Ok(BivarMoments {
        sum_w,
        mean_y,
        mean_x,
        m_yy: s_yy / sum_w,
        m_xx: s_xx / sum_w,
        m_yx: s_yx / sum_w,
        n,
    })
}

/// Iterate a domain mask, or an all-true mask of length `len` when absent.
fn mask_iter(domain: Option<&BooleanChunked>, len: usize) -> Box<dyn Iterator<Item = bool> + '_> {
    match domain {
        Some(m) => Box::new(m.iter().map(|v| v == Some(true))),
        None => Box::new(std::iter::repeat_n(true, len)),
    }
}

// ============================================================================
// Point estimates
// ============================================================================

pub fn point_estimate_cov(
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked,
) -> PolarsResult<f64> {
    let m = bivar_moments(y, x, weights, None)?;
    if m.sum_w == 0.0 {
        return Err(PolarsError::ComputeError(
            "Sum of weights is zero; covariance is undefined".into(),
        ));
    }
    Ok(m.cov())
}

pub fn point_estimate_cov_domain(
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
) -> PolarsResult<f64> {
    // Domains degrade to NaN rather than erroring: one empty by-group must not
    // fail the whole grouped call, matching `scores_ratio_domain`.
    let m = bivar_moments(y, x, weights, Some(domain_mask))?;
    Ok(if m.sum_w == 0.0 { f64::NAN } else { m.cov() })
}

pub fn point_estimate_corr(
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked,
) -> PolarsResult<f64> {
    let m = bivar_moments(y, x, weights, None)?;
    if m.sum_w == 0.0 {
        return Err(PolarsError::ComputeError(
            "Sum of weights is zero; correlation is undefined".into(),
        ));
    }
    Ok(m.corr())
}

pub fn point_estimate_corr_domain(
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
) -> PolarsResult<f64> {
    let m = bivar_moments(y, x, weights, Some(domain_mask))?;
    Ok(if m.sum_w == 0.0 { f64::NAN } else { m.corr() })
}

// ============================================================================
// Linearization scores
// ============================================================================

/// Per-unit linearized values for the covariance.
///
/// z_i = (w_i/W)(a_i b_i - m_yx) * n/(n-1)
///
/// The Kish factor is a constant multiplier, so it scales the score -- and
/// hence the SE -- exactly as it scales the estimate, keeping the two on the
/// same footing as R's `svyvar`.
pub fn scores_cov(
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked,
) -> PolarsResult<Float64Chunked> {
    let m = bivar_moments(y, x, weights, None)?;
    if m.sum_w == 0.0 {
        return Err(PolarsError::ComputeError(
            "Sum of weights is zero; covariance is undefined".into(),
        ));
    }
    Ok(assoc_score_values(y, x, weights, &m, None, AssocKind::Cov))
}

pub fn scores_cov_domain(
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
) -> PolarsResult<Float64Chunked> {
    let m = bivar_moments(y, x, weights, Some(domain_mask))?;
    if m.sum_w == 0.0 {
        return Ok(Float64Chunked::from_slice("scores".into(), &vec![0.0; y.len()]));
    }
    Ok(assoc_score_values(y, x, weights, &m, Some(domain_mask), AssocKind::Cov))
}

/// Per-unit linearized values for the Pearson correlation.
///
/// z_i = (w_i/W)[ yt_i xt_i - (rho/2)(yt_i^2 + xt_i^2) ]
///
/// with yt, xt the weighted-standardized residuals. Verified against a numeric
/// delta method over the six underlying totals; the scores sum to zero, as they
/// must for a scale-free statistic.
pub fn scores_corr(
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked,
) -> PolarsResult<Float64Chunked> {
    let m = bivar_moments(y, x, weights, None)?;
    if m.sum_w == 0.0 {
        return Err(PolarsError::ComputeError(
            "Sum of weights is zero; correlation is undefined".into(),
        ));
    }
    if !(m.m_yy > 0.0 && m.m_xx > 0.0) {
        return Err(PolarsError::ComputeError(
            "Zero weighted variance; correlation is undefined for a constant column".into(),
        ));
    }
    Ok(assoc_score_values(y, x, weights, &m, None, AssocKind::Corr))
}

pub fn scores_corr_domain(
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
) -> PolarsResult<Float64Chunked> {
    let m = bivar_moments(y, x, weights, Some(domain_mask))?;
    if m.sum_w == 0.0 || !(m.m_yy > 0.0 && m.m_xx > 0.0) {
        return Ok(Float64Chunked::from_slice("scores".into(), &vec![0.0; y.len()]));
    }
    Ok(assoc_score_values(y, x, weights, &m, Some(domain_mask), AssocKind::Corr))
}

/// Shared score writer: applies `f` to each contributing row, 0.0 elsewhere.
///
/// Out-of-domain rows emit 0.0 rather than null so the score vector stays
/// full-length against the shared design build, exactly as the ratio domain
/// kernels do.
fn score_values(
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked,
    domain: Option<&BooleanChunked>,
    f: impl Fn(f64, f64, f64) -> f64,
) -> Float64Chunked {
    match (cont_triple(y, x, weights), domain) {
        (Some((ys, xs, ws)), None) => {
            let vals: Vec<f64> = (0..ys.len()).map(|i| f(ys[i], xs[i], ws[i])).collect();
            Float64Chunked::from_slice("scores".into(), &vals)
        }
        _ => {
            let vals: Vec<f64> = y
                .iter()
                .zip(x.iter())
                .zip(weights.iter())
                .zip(mask_iter(domain, y.len()))
                .map(|(((yi, xi), wi), mi)| match (yi, xi, wi, mi) {
                    (Some(yv), Some(xv), Some(wv), true) => f(yv, xv, wv),
                    _ => 0.0,
                })
                .collect();
            Float64Chunked::from_slice("scores".into(), &vals)
        }
    }
}

/// Score kernel for either association: `z_i = (w_i/W) * u_i`, with `u` the
/// shared influence function so the SE and the deff denominator can never be
/// built from different linearizations.
fn assoc_score_values(
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked,
    m: &BivarMoments,
    domain: Option<&BooleanChunked>,
    kind: AssocKind,
) -> Float64Chunked {
    let sum_w = m.sum_w;
    let inf = Influence::new(kind, m);
    score_values(y, x, weights, domain, move |yv, xv, wv| {
        (wv / sum_w) * inf.eval(yv, xv)
    })
}

// ============================================================================
// SRS reference variance (the denominator of deff)
// ============================================================================

/// Loop-invariant pieces of an influence function, hoisted out of the row loop.
///
/// This is the single definition of `u`: the bracket the score kernels weight
/// by `w/W`, and the variable whose population variance forms the SRS reference
/// below. Keeping one definition is what stops a deff from being computed
/// against a different linearization than the SE it is compared with.
#[derive(Debug, Clone, Copy)]
struct Influence {
    kind: AssocKind,
    mean_y: f64,
    mean_x: f64,
    m_yx: f64,
    /// 1/s_y and 1/s_x, so the row loop multiplies instead of dividing.
    inv_s_y: f64,
    inv_s_x: f64,
    rho: f64,
    kish: f64,
}

impl Influence {
    fn new(kind: AssocKind, m: &BivarMoments) -> Self {
        Influence {
            kind,
            mean_y: m.mean_y,
            mean_x: m.mean_x,
            m_yx: m.m_yx,
            inv_s_y: 1.0 / m.m_yy.sqrt(),
            inv_s_x: 1.0 / m.m_xx.sqrt(),
            rho: m.corr(),
            kish: m.kish_factor(),
        }
    }

    #[inline]
    fn eval(&self, yv: f64, xv: f64) -> f64 {
        let a = yv - self.mean_y;
        let b = xv - self.mean_x;
        match self.kind {
            AssocKind::Cov => (a * b - self.m_yx) * self.kish,
            AssocKind::Corr => {
                let yt = a * self.inv_s_y;
                let xt = b * self.inv_s_x;
                yt * xt - 0.5 * self.rho * (yt * yt + xt * xt)
            }
        }
    }
}

/// SRS reference variance for a design effect.
///
/// svy computes deff as `V_design / V_srs`, where `V_srs` is a plug-in: the
/// estimated population variance of the estimator's influence function, divided
/// by the nominal `n`, times the finite-population correction `1 - n/N_hat`.
/// That is the rule `srs_variance_mean` and `srs_variance_ratio` already apply
/// (taylor.rs), and it is R's rule too -- `svymean` builds its reference as
/// `svyvar(x, design)/nobs * (psum-nobs)/psum` (R/survey.R:557-562).
///
/// Note R offers no deff for `svyvar`, so there is no upstream behaviour to
/// match for an association; this follows svy's own established pattern rather
/// than porting anything.
///
/// Taking the *empirical* variance of the influence function, rather than the
/// textbook `(1-rho^2)^2/n`, is deliberate. The two agree asymptotically -- the
/// textbook form is exactly what this converges to under bivariate normality --
/// but the empirical version needs no distributional assumption, which is very
/// likely why R declined to define deff here at all.
///
/// For the covariance the Kish factor enters the influence function and the
/// estimate alike, so it cancels out of deff entirely.
fn srs_variance_assoc(
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked,
    domain: Option<&BooleanChunked>,
    kind: AssocKind,
) -> PolarsResult<f64> {
    let m = bivar_moments(y, x, weights, domain)?;
    if m.sum_w == 0.0 || (kind == AssocKind::Corr && !(m.m_yy > 0.0 && m.m_xx > 0.0)) {
        return Ok(f64::NAN);
    }

    // Only positive-weight rows carry information, matching srs_variance_mean:
    // zero-weight rows (out-of-domain, or nulls under domain semantics) must
    // not inflate n and so must not deflate deff.
    let inf = Influence::new(kind, &m);
    let mut uv: Vec<f64> = Vec::new();
    let mut wv: Vec<f64> = Vec::new();
    for i in 0..y.len() {
        if domain.is_some_and(|d| d.get(i) != Some(true)) {
            continue;
        }
        if let (Some(yval), Some(xval), Some(wval)) = (y.get(i), x.get(i), weights.get(i))
            && wval > 0.0
        {
            uv.push(inf.eval(yval, xval));
            wv.push(wval);
        }
    }

    let n = uv.len() as f64;
    if n < 2.0 {
        return Ok(f64::NAN);
    }
    let sum_w: f64 = wv.iter().sum();
    if sum_w <= 0.0 {
        return Ok(f64::NAN);
    }
    let wn: Vec<f64> = wv.iter().map(|w| w / sum_w).collect();
    let s2_u = crate::estimation::taylor::weighted_s2(&uv, &wn);
    Ok((s2_u / n) * (1.0 - (n / sum_w)))
}

pub fn srs_variance_corr(
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked,
) -> PolarsResult<f64> {
    srs_variance_assoc(y, x, weights, None, AssocKind::Corr)
}

pub fn srs_variance_corr_domain(
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
) -> PolarsResult<f64> {
    srs_variance_assoc(y, x, weights, Some(domain_mask), AssocKind::Corr)
}

pub fn srs_variance_cov(
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked,
) -> PolarsResult<f64> {
    srs_variance_assoc(y, x, weights, None, AssocKind::Cov)
}

pub fn srs_variance_cov_domain(
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked,
    domain_mask: &BooleanChunked,
) -> PolarsResult<f64> {
    srs_variance_assoc(y, x, weights, Some(domain_mask), AssocKind::Cov)
}

// ============================================================================
// All-pairs moments: one sweep for every pair
// ============================================================================

/// Centered second moments for every pair among `k` columns.
///
/// Computing `k` columns pairwise runs `k(k+1)/2` independent two-pass kernels,
/// each re-reading its two columns: `2k(k+1)` column reads in total. Building
/// the whole matrix in one sweep reads each column twice regardless of `k`, so
/// traffic falls from O(k^2) to O(k) -- for ten variables, 110 column reads
/// become 20. The arithmetic is unchanged (the same k^2/2 cross-products), but
/// the kernel is memory-bound at scale, so the traffic is what shows up on the
/// clock.
#[derive(Debug, Clone)]
pub struct MomentMatrix {
    pub sum_w: f64,
    pub n: usize,
    pub k: usize,
    pub means: Vec<f64>,
    /// Row-major `k * k`: `m[j*k + l] = sum(w (yj-mj)(yl-ml)) / W`.
    pub m: Vec<f64>,
}

impl MomentMatrix {
    #[inline]
    fn kish_factor(&self) -> f64 {
        if self.n < 2 {
            f64::NAN
        } else {
            self.n as f64 / (self.n as f64 - 1.0)
        }
    }

    #[inline]
    pub fn moment(&self, j: usize, l: usize) -> f64 {
        self.m[j * self.k + l]
    }

    /// Covariance of columns `j` and `l`, on R's `svyvar` scale.
    #[inline]
    pub fn cov(&self, j: usize, l: usize) -> f64 {
        self.moment(j, l) * self.kish_factor()
    }

    /// Pearson correlation of columns `j` and `l`.
    #[inline]
    pub fn corr(&self, j: usize, l: usize) -> f64 {
        let denom = (self.moment(j, j) * self.moment(l, l)).sqrt();
        if denom > 0.0 {
            self.moment(j, l) / denom
        } else {
            f64::NAN
        }
    }
}

/// Accumulate the full moment matrix for `cols` in two sweeps.
///
/// A row contributes only when every column, and the weight, is non-null and
/// the row is in-domain; that is the multivariate reading of the complete-pairs
/// rule the bivariate kernel applies, and it matches R's `svyvar`, which drops
/// a row from the whole matrix when any column is missing.
pub fn multi_moments(
    cols: &[&Float64Chunked],
    weights: &Float64Chunked,
    domain: Option<&BooleanChunked>,
) -> PolarsResult<MomentMatrix> {
    let k = cols.len();
    if k == 0 {
        return Err(PolarsError::ComputeError(
            "multi_moments requires at least one column".into(),
        ));
    }
    let len = weights.len();
    if let Some(bad) = cols.iter().find(|c| c.len() != len) {
        return Err(PolarsError::ComputeError(
            format!("column length {} does not match weights length {len}", bad.len()).into(),
        ));
    }

    // Contiguous, null-free, undomained input takes a raw-slice path. Going
    // through `ChunkedArray::get` per element costs more than the O(k^2)->O(k)
    // traffic saving is worth: measured, the `get`-based loop ran ~1.8x SLOWER
    // than repeated pairwise kernels, which use `cont_slice` throughout.
    let slices: Option<(Vec<&[f64]>, &[f64])> = if domain.is_none()
        && weights.null_count() == 0
        && cols.iter().all(|c| c.null_count() == 0)
    {
        let mut out = Vec::with_capacity(k);
        let mut ok = weights.cont_slice().is_ok();
        for c in cols {
            match c.cont_slice() {
                Ok(s) => out.push(s),
                Err(_) => {
                    ok = false;
                    break;
                }
            }
        }
        if ok { weights.cont_slice().ok().map(|w| (out, w)) } else { None }
    } else {
        None
    };

    let mut sum_w = 0.0f64;
    let mut sums = vec![0.0f64; k];
    let mut n = 0usize;
    let mut row = vec![0.0f64; k];

    // ---- pass 1: weights, per-column means, and n -------------------------
    if let Some((cs, ws)) = slices.as_ref() {
        for i in 0..len {
            let w = ws[i];
            sum_w += w;
            for j in 0..k {
                sums[j] += w * cs[j][i];
            }
            if w != 0.0 {
                n += 1;
            }
        }
    } else {
        for i in 0..len {
            let Some(w) = weights.get(i) else { continue };
            if domain.is_some_and(|d| d.get(i) != Some(true)) {
                continue;
            }
            let mut complete = true;
            for (j, c) in cols.iter().enumerate() {
                match c.get(i) {
                    Some(v) => row[j] = v,
                    None => {
                        complete = false;
                        break;
                    }
                }
            }
            if !complete {
                continue;
            }
            sum_w += w;
            for j in 0..k {
                sums[j] += w * row[j];
            }
            if w != 0.0 {
                n += 1;
            }
        }
    }

    if sum_w == 0.0 {
        return Ok(MomentMatrix {
            sum_w: 0.0,
            n,
            k,
            means: vec![f64::NAN; k],
            m: vec![f64::NAN; k * k],
        });
    }

    let means: Vec<f64> = sums.iter().map(|s| s / sum_w).collect();

    // ---- pass 2: centered cross-products, upper triangle then mirrored ----
    let mut m = vec![0.0f64; k * k];
    let mut dev = vec![0.0f64; k];
    if let Some((cs, ws)) = slices.as_ref() {
        for i in 0..len {
            let w = ws[i];
            for j in 0..k {
                dev[j] = cs[j][i] - means[j];
            }
            for j in 0..k {
                let wj = w * dev[j];
                for l in j..k {
                    m[j * k + l] += wj * dev[l];
                }
            }
        }
    } else {
        for i in 0..len {
            let Some(w) = weights.get(i) else { continue };
            if domain.is_some_and(|d| d.get(i) != Some(true)) {
                continue;
            }
            let mut complete = true;
            for (j, c) in cols.iter().enumerate() {
                match c.get(i) {
                    Some(v) => dev[j] = v - means[j],
                    None => {
                        complete = false;
                        break;
                    }
                }
            }
            if !complete {
                continue;
            }
            for j in 0..k {
                let wj = w * dev[j];
                for l in j..k {
                    m[j * k + l] += wj * dev[l];
                }
            }
        }
    }

    for j in 0..k {
        for l in j..k {
            m[j * k + l] /= sum_w;
            m[l * k + j] = m[j * k + l];
        }
    }

    Ok(MomentMatrix { sum_w, n, k, means, m })
}

// ============================================================================
// Replication: precomputed products, one dot product set per replicate
// ============================================================================

/// Which association is being estimated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssocKind {
    Corr,
    Cov,
}

impl AssocKind {
    /// Parse the selector the Python layer sends. Aliases are accepted the way
    /// `_normalize_ci_method` accepts them: case-insensitive, underscores and
    /// hyphens interchangeable.
    pub fn from_name(s: &str) -> PolarsResult<Self> {
        match s.to_lowercase().replace('_', "-").as_str() {
            "corr" | "correlation" | "pearson" => Ok(AssocKind::Corr),
            "cov" | "covariance" => Ok(AssocKind::Cov),
            other => Err(PolarsError::ComputeError(
                format!("unknown association kind '{other}'; expected 'corr' or 'cov'").into(),
            )),
        }
    }

    /// Label carried into the result frame.
    pub fn as_str(&self) -> &'static str {
        match self {
            AssocKind::Corr => "corr",
            AssocKind::Cov => "cov",
        }
    }
}

// ---------------------------------------------------------------------------
// Kind-dispatching entry points
//
// The API layer works in terms of (kind, optional domain) rather than four
// separately-named functions per statistic, so the dispatch lives here once
// instead of being rebuilt at every call site.
// ---------------------------------------------------------------------------

pub fn point_estimate_assoc(
    kind: AssocKind,
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked,
    domain: Option<&BooleanChunked>,
) -> PolarsResult<f64> {
    match (kind, domain) {
        (AssocKind::Corr, None) => point_estimate_corr(y, x, weights),
        (AssocKind::Corr, Some(d)) => point_estimate_corr_domain(y, x, weights, d),
        (AssocKind::Cov, None) => point_estimate_cov(y, x, weights),
        (AssocKind::Cov, Some(d)) => point_estimate_cov_domain(y, x, weights, d),
    }
}

pub fn scores_assoc(
    kind: AssocKind,
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked,
    domain: Option<&BooleanChunked>,
) -> PolarsResult<Float64Chunked> {
    match (kind, domain) {
        (AssocKind::Corr, None) => scores_corr(y, x, weights),
        (AssocKind::Corr, Some(d)) => scores_corr_domain(y, x, weights, d),
        (AssocKind::Cov, None) => scores_cov(y, x, weights),
        (AssocKind::Cov, Some(d)) => scores_cov_domain(y, x, weights, d),
    }
}

/// SRS reference variance for either association, over an optional domain.
pub fn srs_variance_assoc_of(
    kind: AssocKind,
    y: &Float64Chunked,
    x: &Float64Chunked,
    weights: &Float64Chunked,
    domain: Option<&BooleanChunked>,
) -> PolarsResult<f64> {
    srs_variance_assoc(y, x, weights, domain, kind)
}

/// Full-sample-centered products for one pair, reused across every replicate.
///
/// `y*x`, `y^2` and `x^2` do not depend on the weights, so forming them once
/// turns each replicate into six weighted dot products instead of redoing
/// `R * n` multiplications. Centering by the full-sample mean first is free
/// here and matters: covariance and correlation are translation-invariant, so
/// the estimate is unchanged, but each replicate's `S/W - mean^2` subtraction
/// is then taken between small numbers instead of cancelling two large ones.
///
/// R recomputes the mean inside every replicate (`svrepvar`'s `v(w)` closure,
/// R/surveyrep.R:800-806); that is preserved exactly, because the per-replicate
/// means reappear as the `mean_a`/`mean_b` terms below. What R fixes once,
/// outside the closure, is `n` for the Kish factor -- so this does too.
pub struct PairProducts {
    a: Vec<f64>,
    b: Vec<f64>,
    aa: Vec<f64>,
    bb: Vec<f64>,
    ab: Vec<f64>,
    /// Domain indicator as a 0/1 float, matching the replication layer's
    /// convention. Held rather than baked into the products because the
    /// per-replicate weight sum must also exclude out-of-domain rows.
    mask: Option<Vec<f64>>,
    /// Full-sample n, held fixed across replicates as R holds it.
    n: usize,
}

impl PairProducts {
    pub fn new(
        y: &Float64Chunked,
        x: &Float64Chunked,
        weights: &Float64Chunked,
        domain: Option<&[f64]>,
    ) -> PolarsResult<Self> {
        // Fold the domain into the weights for the full-sample moments: a
        // zeroed weight is exactly how an out-of-domain row drops out, and
        // `bivar_moments` already keeps zero-weight rows out of `n`.
        let masked: Option<Float64Chunked> = domain.map(|m| {
            let v: Vec<f64> = (0..y.len())
                .map(|i| {
                    weights.get(i).unwrap_or(0.0) * m.get(i).copied().unwrap_or(0.0)
                })
                .collect();
            Float64Chunked::from_slice("w".into(), &v)
        });
        let full = bivar_moments(y, x, masked.as_ref().unwrap_or(weights), None)?;
        let (mean_y, mean_x) = if full.sum_w == 0.0 {
            (0.0, 0.0)
        } else {
            (full.mean_y, full.mean_x)
        };

        let len = y.len();
        let mut a = Vec::with_capacity(len);
        let mut b = Vec::with_capacity(len);
        let mut aa = Vec::with_capacity(len);
        let mut bb = Vec::with_capacity(len);
        let mut ab = Vec::with_capacity(len);

        for i in 0..len {
            // A null row contributes zero to every replicate sum, which drops
            // it without disturbing the design structure.
            let (av, bv) = match (y.get(i), x.get(i)) {
                (Some(yv), Some(xv)) => (yv - mean_y, xv - mean_x),
                _ => (0.0, 0.0),
            };
            a.push(av);
            b.push(bv);
            aa.push(av * av);
            bb.push(bv * bv);
            ab.push(av * bv);
        }

        Ok(PairProducts { a, b, aa, bb, ab, mask: domain.map(<[f64]>::to_vec), n: full.n })
    }

    /// Recompute the association under one replicate weight column.
    ///
    /// Six row-ordered dot products; the summation order is a function of the
    /// data alone, so this stays bit-identical across thread counts even though
    /// replicates run concurrently.
    pub fn estimate(&self, rep_w: &[f64], kind: AssocKind) -> f64 {
        let (mut sw, mut sa, mut sb) = (0.0f64, 0.0f64, 0.0f64);
        let (mut saa, mut sbb, mut sab) = (0.0f64, 0.0f64, 0.0f64);

        // Pre-slicing all six arrays to a common length lets the bounds checks
        // fall out of the loop body. The traversal is six-way parallel, so an
        // index walk expresses it more directly than nested zips would. The
        // domain branch is hoisted out of the loop rather than tested per row.
        let n = self.a.len().min(rep_w.len());
        let (w_s, a_s, b_s) = (&rep_w[..n], &self.a[..n], &self.b[..n]);
        let (aa_s, bb_s, ab_s) = (&self.aa[..n], &self.bb[..n], &self.ab[..n]);

        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let w = match &self.mask {
                Some(m) => w_s[i] * m[i],
                None => w_s[i],
            };
            sw += w;
            sa += w * a_s[i];
            sb += w * b_s[i];
            saa += w * aa_s[i];
            sbb += w * bb_s[i];
            sab += w * ab_s[i];
        }

        if sw == 0.0 {
            return f64::NAN;
        }
        let mean_a = sa / sw;
        let mean_b = sb / sw;
        let m_aa = saa / sw - mean_a * mean_a;
        let m_bb = sbb / sw - mean_b * mean_b;
        let m_ab = sab / sw - mean_a * mean_b;

        match kind {
            AssocKind::Cov => {
                if self.n < 2 {
                    f64::NAN
                } else {
                    m_ab * (self.n as f64 / (self.n as f64 - 1.0))
                }
            }
            AssocKind::Corr => {
                let denom = (m_aa * m_bb).sqrt();
                if denom > 0.0 { m_ab / denom } else { f64::NAN }
            }
        }
    }
}

/// Recompute the association across every replicate weight column.
///
/// Parallel over replicates -- an independent axis, so this respects the
/// determinism policy in `mod.rs` while still using every core.
pub fn replicate_association(
    products: &PairProducts,
    rep_cols: &[&[f64]],
    kind: AssocKind,
) -> Vec<f64> {
    use rayon::prelude::*;
    rep_cols.par_iter().map(|col| products.estimate(col, kind)).collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn fc(name: &str, v: &[f64]) -> Float64Chunked {
        Float64Chunked::from_slice(name.into(), v)
    }

    /// y, x and weights from a small fixture whose moments are easy to verify
    /// independently.
    fn fixture() -> (Float64Chunked, Float64Chunked, Float64Chunked) {
        let y = fc("y", &[1.0, 2.0, 3.0, 4.0, 5.0, 7.0]);
        let x = fc("x", &[2.0, 1.0, 4.0, 3.0, 7.0, 6.0]);
        let w = fc("w", &[1.0, 2.0, 1.5, 3.0, 2.5, 1.0]);
        (y, x, w)
    }

    /// Reference moments computed straight from the definitions, independent of
    /// the two-pass kernel under test.
    /// Returns (sum_w, m_yy, m_xx, m_yx, mean_y).
    fn reference(y: &[f64], x: &[f64], w: &[f64]) -> (f64, f64, f64, f64, f64) {
        let sw: f64 = w.iter().sum();
        let my: f64 = y.iter().zip(w).map(|(v, wi)| v * wi).sum::<f64>() / sw;
        let mx: f64 = x.iter().zip(w).map(|(v, wi)| v * wi).sum::<f64>() / sw;
        let mut yy = 0.0;
        let mut xx = 0.0;
        let mut yx = 0.0;
        for i in 0..y.len() {
            yy += w[i] * (y[i] - my).powi(2);
            xx += w[i] * (x[i] - mx).powi(2);
            yx += w[i] * (y[i] - my) * (x[i] - mx);
        }
        (sw, yy / sw, xx / sw, yx / sw, my)
    }

    #[test]
    fn test_moments_match_definition() {
        let (y, x, w) = fixture();
        let m = bivar_moments(&y, &x, &w, None).unwrap();
        let ys = y.cont_slice().unwrap();
        let xs = x.cont_slice().unwrap();
        let ws = w.cont_slice().unwrap();
        let (sw, m_yy, m_xx, m_yx, mean_y) = reference(ys, xs, ws);

        assert!((m.sum_w - sw).abs() < 1e-12);
        assert!((m.mean_y - mean_y).abs() < 1e-12);
        assert!((m.m_yy - m_yy).abs() < 1e-12);
        assert!((m.m_xx - m_xx).abs() < 1e-12);
        assert!((m.m_yx - m_yx).abs() < 1e-12);
        assert_eq!(m.n, 6);
    }

    #[test]
    fn test_corr_of_variable_with_itself_is_one() {
        let (y, _, w) = fixture();
        let r = point_estimate_corr(&y, &y, &w).unwrap();
        assert!((r - 1.0).abs() < 1e-12, "corr(y,y) = {r}");
    }

    #[test]
    fn test_cov_of_variable_with_itself_is_its_variance() {
        let (y, _, w) = fixture();
        let c = point_estimate_cov(&y, &y, &w).unwrap();
        let m = bivar_moments(&y, &y, &w, None).unwrap();
        let expected = m.m_yy * (m.n as f64 / (m.n as f64 - 1.0));
        assert!((c - expected).abs() < 1e-12, "cov(y,y) = {c}, var = {expected}");
    }

    #[test]
    fn test_corr_is_symmetric() {
        let (y, x, w) = fixture();
        let a = point_estimate_corr(&y, &x, &w).unwrap();
        let b = point_estimate_corr(&x, &y, &w).unwrap();
        assert_eq!(a.to_bits(), b.to_bits(), "corr must be order-independent");
    }

    #[test]
    fn test_corr_is_translation_and_scale_invariant() {
        let (y, x, w) = fixture();
        let base = point_estimate_corr(&y, &x, &w).unwrap();

        let ys: Vec<f64> = y.cont_slice().unwrap().iter().map(|v| v * 3.0 + 100.0).collect();
        let xs: Vec<f64> = x.cont_slice().unwrap().iter().map(|v| v * 0.5 - 7.0).collect();
        let shifted = point_estimate_corr(&fc("y", &ys), &fc("x", &xs), &w).unwrap();

        assert!((base - shifted).abs() < 1e-12, "{base} vs {shifted}");
    }

    /// The two-pass kernel must survive a large offset that would destroy the
    /// naive sum-of-squares form. With an offset of 1e9 on data of order 1, the
    /// one-pass formula loses essentially all precision.
    #[test]
    fn test_corr_stable_under_large_offset() {
        let (y, x, w) = fixture();
        let base = point_estimate_corr(&y, &x, &w).unwrap();

        let ys: Vec<f64> = y.cont_slice().unwrap().iter().map(|v| v + 1e9).collect();
        let xs: Vec<f64> = x.cont_slice().unwrap().iter().map(|v| v + 1e9).collect();
        let offset = point_estimate_corr(&fc("y", &ys), &fc("x", &xs), &w).unwrap();

        assert!((base - offset).abs() < 1e-9, "offset broke stability: {base} vs {offset}");
    }

    /// A linearization for a scale-free statistic must have scores summing to
    /// zero; a nonzero sum means the estimator drifts under the design.
    #[test]
    fn test_corr_scores_sum_to_zero() {
        let (y, x, w) = fixture();
        let s = scores_corr(&y, &x, &w).unwrap();
        let total: f64 = s.cont_slice().unwrap().iter().sum();
        assert!(total.abs() < 1e-14, "scores summed to {total}");
    }

    #[test]
    fn test_cov_scores_sum_to_zero() {
        let (y, x, w) = fixture();
        let s = scores_cov(&y, &x, &w).unwrap();
        let total: f64 = s.cont_slice().unwrap().iter().sum();
        assert!(total.abs() < 1e-12, "scores summed to {total}");
    }

    /// Zero-weight rows must not shift the estimate and must not inflate `n`;
    /// this is how `where=` and `drop_nulls` reach the kernel.
    #[test]
    fn test_zero_weight_rows_are_excluded() {
        let (y, x, w) = fixture();
        let mut ys = y.cont_slice().unwrap().to_vec();
        let mut xs = x.cont_slice().unwrap().to_vec();
        let mut ws = w.cont_slice().unwrap().to_vec();
        // Append rows that are wild in value but carry zero weight.
        ys.extend_from_slice(&[1e6, -1e6]);
        xs.extend_from_slice(&[-1e6, 1e6]);
        ws.extend_from_slice(&[0.0, 0.0]);

        let base = bivar_moments(&y, &x, &w, None).unwrap();
        let padded =
            bivar_moments(&fc("y", &ys), &fc("x", &xs), &fc("w", &ws), None).unwrap();

        assert_eq!(base.n, padded.n, "zero-weight rows must not count toward n");
        assert!((base.corr() - padded.corr()).abs() < 1e-12);
        assert!((base.cov() - padded.cov()).abs() < 1e-12);
    }

    /// A domain restricted to a subset must reproduce the estimate computed on
    /// that subset alone.
    #[test]
    fn test_domain_matches_subset() {
        let (y, x, w) = fixture();
        let mask = BooleanChunked::from_slice(
            "m".into(),
            &[true, true, true, false, false, false],
        );
        let dom = point_estimate_corr_domain(&y, &x, &w, &mask).unwrap();

        let sub = point_estimate_corr(
            &fc("y", &y.cont_slice().unwrap()[..3]),
            &fc("x", &x.cont_slice().unwrap()[..3]),
            &fc("w", &w.cont_slice().unwrap()[..3]),
        )
        .unwrap();

        assert!((dom - sub).abs() < 1e-12, "domain {dom} vs subset {sub}");
    }

    /// Out-of-domain rows carry a zero score so the vector stays aligned with
    /// the shared design build.
    #[test]
    fn test_domain_scores_zero_outside() {
        let (y, x, w) = fixture();
        let mask = BooleanChunked::from_slice(
            "m".into(),
            &[true, true, true, false, false, false],
        );
        let s = scores_corr_domain(&y, &x, &w, &mask).unwrap();
        let vals = s.cont_slice().unwrap();
        assert!(vals[3..].iter().all(|v| *v == 0.0), "tail not zeroed: {vals:?}");
        let total: f64 = vals.iter().sum();
        assert!(total.abs() < 1e-14, "in-domain scores summed to {total}");
    }

    /// A constant column has no correlation to report; the ungrouped path says
    /// so explicitly rather than emitting a silently meaningless number.
    #[test]
    fn test_constant_column_is_rejected() {
        let (y, _, w) = fixture();
        let konst = fc("k", &[2.0; 6]);
        assert!(point_estimate_corr(&y, &konst, &w).unwrap().is_nan());
        assert!(scores_corr(&y, &konst, &w).is_err());
    }

    // ------------------------------------------------------------------
    // R golden fixtures
    //
    // survey 4.5, single-stage cluster design (8 PSUs x 3 rows, unequal
    // weights). Covariance targets come from `svyvar(~y+x, d)`; the
    // correlation targets come from `svycontrast` applied to the moment
    // means, which is R computing the same delta method independently of our
    // linearization:
    //
    //   mns <- svymean(~y+x+yy+xx+yx, d)
    //   svycontrast(mns, quote((yx - y*x)/sqrt((yy-y^2)*(xx-x^2))))
    //
    // The SE comparisons are the load-bearing ones: a wrong score still
    // reproduces the point estimate, but cannot reproduce the SE.
    // ------------------------------------------------------------------

    const G_Y: [f64; 24] = [
        12.0, 15.0, 11.0, 22.0, 25.0, 19.0, 31.0, 29.0, 35.0, 41.0, 44.0, 39.0, 18.0, 16.0, 21.0,
        27.0, 30.0, 24.0, 36.0, 33.0, 38.0, 45.0, 49.0, 43.0,
    ];
    const G_X: [f64; 24] = [
        9.0, 4.0, 14.0, 6.0, 17.0, 8.0, 21.0, 11.0, 13.0, 12.0, 26.0, 7.0, 19.0, 5.0, 23.0, 10.0,
        28.0, 15.0, 8.0, 22.0, 16.0, 18.0, 31.0, 20.0,
    ];
    const G_W: [f64; 24] = [
        2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 1.5, 1.5, 1.5, 4.0, 4.0, 4.0, 2.5, 2.5, 2.5, 3.5, 3.5, 3.5,
        1.0, 1.0, 1.0, 5.0, 5.0, 5.0,
    ];

    fn golden_psu() -> Column {
        let v: Vec<i64> = (1..=8i64).flat_map(|c| std::iter::repeat_n(c, 3)).collect();
        Column::from(Int64Chunked::from_slice("psu".into(), &v).into_series())
    }

    fn golden_strata() -> Column {
        let v: Vec<i64> = (1..=2i64).flat_map(|s| std::iter::repeat_n(s, 12)).collect();
        Column::from(Int64Chunked::from_slice("str".into(), &v).into_series())
    }

    fn se_of(scores: &Float64Chunked, strata: Option<&Column>, psu: Option<&Column>) -> f64 {
        crate::estimation::taylor::taylor_variance(scores, strata, psu, None, None, None, None)
            .unwrap()
            .sqrt()
    }

    #[test]
    fn test_golden_cov_matches_r_svyvar() {
        let (y, x, w) = (fc("y", &G_Y), fc("x", &G_X), fc("w", &G_W));
        let est = point_estimate_cov(&y, &x, &w).unwrap();
        assert!((est - 50.138658078368209).abs() < 1e-11, "cov est = {est}");

        let se = se_of(&scores_cov(&y, &x, &w).unwrap(), None, Some(&golden_psu()));
        assert!((se - 21.498361515848689).abs() < 1e-9, "cov se = {se}");
    }

    /// The diagonal of `svyvar` is the variance, so cov(y,y) must reproduce it
    /// through exactly the same code path.
    #[test]
    fn test_golden_variance_matches_r_svyvar_diagonal() {
        let (y, w) = (fc("y", &G_Y), fc("w", &G_W));
        let est = point_estimate_cov(&y, &y, &w).unwrap();
        assert!((est - 139.54576489533011).abs() < 1e-11, "var est = {est}");

        let se = se_of(&scores_cov(&y, &y, &w).unwrap(), None, Some(&golden_psu()));
        assert!((se - 37.342191100127813).abs() < 1e-9, "var se = {se}");
    }

    /// The decisive test for the correlation linearization: R computes this SE
    /// by its own delta method over the five moment means, with no shared code
    /// or algebra with our score.
    #[test]
    fn test_golden_corr_matches_r_svycontrast() {
        let (y, x, w) = (fc("y", &G_Y), fc("x", &G_X), fc("w", &G_W));
        let est = point_estimate_corr(&y, &x, &w).unwrap();
        assert!((est - 0.52860852253793722).abs() < 1e-12, "corr est = {est}");

        let se = se_of(&scores_corr(&y, &x, &w).unwrap(), None, Some(&golden_psu()));
        assert!((se - 0.15921590459764234).abs() < 1e-11, "corr se = {se}");
    }

    /// Stratification is handled by the shared variance kernel, but the score
    /// must stay correct underneath it.
    #[test]
    fn test_golden_cov_stratified_matches_r() {
        let (y, x, w) = (fc("y", &G_Y), fc("x", &G_X), fc("w", &G_W));
        let se = se_of(
            &scores_cov(&y, &x, &w).unwrap(),
            Some(&golden_strata()),
            Some(&golden_psu()),
        );
        assert!((se - 23.178556453025269).abs() < 1e-9, "stratified cov se = {se}");
    }

    // ------------------------------------------------------------------
    // SRS reference / deff
    // ------------------------------------------------------------------

    /// With equal weights and neither strata nor clusters, the design IS simple
    /// random sampling, so the design variance must land on the SRS reference
    /// and deff must be 1. Weights are large enough that the finite-population
    /// correction is negligible, isolating the comparison from the FPC.
    #[test]
    fn test_deff_is_one_under_srs() {
        let (y, x) = (fc("y", &G_Y), fc("x", &G_X));
        let w = fc("w", &[1e6; 24]);

        for kind in [AssocKind::Corr, AssocKind::Cov] {
            let scores = match kind {
                AssocKind::Corr => scores_corr(&y, &x, &w).unwrap(),
                AssocKind::Cov => scores_cov(&y, &x, &w).unwrap(),
            };
            let design = se_of(&scores, None, None).powi(2);
            let srs = match kind {
                AssocKind::Corr => srs_variance_corr(&y, &x, &w).unwrap(),
                AssocKind::Cov => srs_variance_cov(&y, &x, &w).unwrap(),
            };
            let deff = design / srs;
            assert!((deff - 1.0).abs() < 1e-4, "{kind:?}: deff under SRS = {deff}");
        }
    }

    /// The golden fixture is clustered with substantial between-PSU spread, so
    /// its design effect must exceed 1 -- the whole point of reporting one.
    #[test]
    fn test_deff_exceeds_one_under_clustering() {
        let (y, x, w) = (fc("y", &G_Y), fc("x", &G_X), fc("w", &G_W));
        let design = se_of(&scores_corr(&y, &x, &w).unwrap(), None, Some(&golden_psu())).powi(2);
        let srs = srs_variance_corr(&y, &x, &w).unwrap();
        let deff = design / srs;
        assert!(deff > 1.0, "clustered deff should exceed 1, got {deff}");
        assert!(deff.is_finite(), "deff must be finite, got {deff}");
    }

    /// deff is a ratio of two variances of the same estimator, so rescaling the
    /// inputs must leave it untouched. For the covariance this also pins the
    /// claim that the Kish factor cancels: it multiplies the influence function
    /// and the estimate alike, so it cannot survive into deff.
    #[test]
    fn test_deff_is_scale_invariant() {
        let (y, x, w) = (fc("y", &G_Y), fc("x", &G_X), fc("w", &G_W));
        let ys: Vec<f64> = G_Y.iter().map(|v| v * 1000.0 + 5.0).collect();
        let xs: Vec<f64> = G_X.iter().map(|v| v * 0.01 - 3.0).collect();
        let (y2, x2) = (fc("y", &ys), fc("x", &xs));

        let deff = |yy: &Float64Chunked, xx: &Float64Chunked| {
            let d = se_of(&scores_cov(yy, xx, &w).unwrap(), None, Some(&golden_psu())).powi(2);
            d / srs_variance_cov(yy, xx, &w).unwrap()
        };

        let base = deff(&y, &x);
        let scaled = deff(&y2, &x2);
        assert!(
            (base - scaled).abs() < 1e-9,
            "deff moved under rescaling: {base} vs {scaled}"
        );
    }

    /// The empirical SRS reference is claimed to converge to the textbook
    /// `(1-rho^2)^2/n` under bivariate normality. Check that on generated
    /// near-normal data; the tolerance is loose because this is an asymptotic
    /// identity being checked at finite n, not an exact algebraic one.
    #[test]
    fn test_srs_corr_approaches_normal_theory() {
        let n = 20_000usize;
        let mut ys = Vec::with_capacity(n);
        let mut xs = Vec::with_capacity(n);
        let mut s = 987_654_321u64;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let rho_true = 0.6f64;
        for _ in 0..n {
            // Box-Muller for a standard normal pair, then induce correlation.
            let (u1, u2) = (next().max(1e-12), next());
            let r = (-2.0 * u1.ln()).sqrt();
            let (z1, z2) = (r * (2.0 * std::f64::consts::PI * u2).cos(), r * (2.0 * std::f64::consts::PI * u2).sin());
            ys.push(z1);
            xs.push(rho_true * z1 + (1.0 - rho_true * rho_true).sqrt() * z2);
        }
        let y = fc("y", &ys);
        let x = fc("x", &xs);
        // Large weights so the FPC does not enter the comparison.
        let w = fc("w", &vec![1e7; n]);

        let rho = point_estimate_corr(&y, &x, &w).unwrap();
        let srs = srs_variance_corr(&y, &x, &w).unwrap();
        let normal_theory = (1.0 - rho * rho).powi(2) / n as f64;
        let ratio = srs / normal_theory;

        assert!(
            (ratio - 1.0).abs() < 0.10,
            "empirical SRS variance {srs:.3e} vs normal theory {normal_theory:.3e} (ratio {ratio:.4})"
        );
    }

    /// Zero-weight rows must not enter n in the SRS reference either, or deff
    /// would shrink simply because a filter left rows behind.
    #[test]
    fn test_srs_excludes_zero_weight_rows() {
        let (y, x, w) = (fc("y", &G_Y), fc("x", &G_X), fc("w", &G_W));
        let base = srs_variance_corr(&y, &x, &w).unwrap();

        let mut ys = G_Y.to_vec();
        let mut xs = G_X.to_vec();
        let mut ws = G_W.to_vec();
        ys.extend_from_slice(&[1e6, -1e6]);
        xs.extend_from_slice(&[-1e6, 1e6]);
        ws.extend_from_slice(&[0.0, 0.0]);
        let padded = srs_variance_corr(&fc("y", &ys), &fc("x", &xs), &fc("w", &ws)).unwrap();

        assert!((base - padded).abs() < 1e-12, "{base} vs {padded}");
    }

    // ------------------------------------------------------------------
    // All-pairs kernel
    // ------------------------------------------------------------------

    /// The one-sweep matrix must agree with the bivariate kernel pair for pair.
    /// That kernel is the R-validated one, so agreement carries the validation
    /// across without needing separate R fixtures per pair.
    #[test]
    fn test_multi_moments_matches_pairwise() {
        let y = fc("y", &G_Y);
        let x = fc("x", &G_X);
        let z = fc("z", &G_W); // any third column with real spread
        let w = fc("w", &G_W);

        let mm = multi_moments(&[&y, &x, &z], &w, None).unwrap();
        assert_eq!(mm.k, 3);
        assert_eq!(mm.n, 24);

        let cols = [&y, &x, &z];
        for j in 0..3 {
            for l in 0..3 {
                let pair = bivar_moments(cols[j], cols[l], &w, None).unwrap();
                assert!(
                    (mm.corr(j, l) - pair.corr()).abs() < 1e-12,
                    "corr({j},{l}): {} vs {}",
                    mm.corr(j, l),
                    pair.corr()
                );
                assert!(
                    (mm.cov(j, l) - pair.cov()).abs() < 1e-10,
                    "cov({j},{l}): {} vs {}",
                    mm.cov(j, l),
                    pair.cov()
                );
            }
        }
    }

    /// The matrix must be symmetric and carry unit correlations on the diagonal.
    #[test]
    fn test_multi_moments_symmetry_and_diagonal() {
        let (y, x, w) = (fc("y", &G_Y), fc("x", &G_X), fc("w", &G_W));
        let mm = multi_moments(&[&y, &x], &w, None).unwrap();
        assert_eq!(mm.moment(0, 1).to_bits(), mm.moment(1, 0).to_bits());
        for j in 0..2 {
            assert!((mm.corr(j, j) - 1.0).abs() < 1e-12);
        }
        // Diagonal covariance is the variance R reports on svyvar's diagonal.
        assert!((mm.cov(0, 0) - 139.54576489533011).abs() < 1e-10);
        assert!((mm.cov(0, 1) - 50.138658078368209).abs() < 1e-10);
    }

    /// A row missing in ANY column leaves the whole matrix, matching how
    /// `svyvar` drops an incomplete row from every cell.
    #[test]
    fn test_multi_moments_drops_row_missing_any_column() {
        let y = Float64Chunked::from_slice_options(
            "y".into(),
            &[Some(1.0), Some(2.0), Some(3.0), Some(4.0)],
        );
        let x = Float64Chunked::from_slice_options(
            "x".into(),
            &[Some(2.0), Some(1.0), None, Some(3.0)],
        );
        let w = fc("w", &[1.0, 1.0, 1.0, 1.0]);
        let mm = multi_moments(&[&y, &x], &w, None).unwrap();
        assert_eq!(mm.n, 3, "row with a null in x must leave the matrix");
        assert!((mm.sum_w - 3.0).abs() < 1e-12);
    }

    // ------------------------------------------------------------------
    // Replication kernel
    // ------------------------------------------------------------------

    /// JK1 over the 8 golden PSUs: replicate r drops PSU r and inflates the
    /// survivors by 8/7, which is what `as.svrepdesign(type="JK1")` produced.
    fn jk1_rep_weights() -> Vec<Vec<f64>> {
        (0..8usize)
            .map(|r| {
                (0..24usize)
                    .map(|i| if i / 3 == r { 0.0 } else { G_W[i] * 8.0 / 7.0 })
                    .collect()
            })
            .collect()
    }

    /// R's per-replicate covariances, from
    /// `svyvar(~y+x, as.svrepdesign(d, type="JK1"), return.replicates=TRUE)`.
    const R_COV_REPS: [f64; 8] = [
        40.722413332298757,
        46.226109824772017,
        53.735975549640145,
        60.335842305289987,
        53.635362318840571,
        58.450198723353004,
        52.940861100007069,
        23.329713102632358,
    ];

    /// Correlations derived from each replicate's own variance matrix.
    const R_CORR_REPS: [f64; 8] = [
        0.48055776528106858,
        0.48070293973143952,
        0.53540826755142246,
        0.63514953996442003,
        0.58236081385363858,
        0.57516710027925266,
        0.54192817928138626,
        0.31366849757047521,
    ];

    #[test]
    fn test_replicate_cov_matches_r_jk1() {
        let (y, x, w) = (fc("y", &G_Y), fc("x", &G_X), fc("w", &G_W));
        let p = PairProducts::new(&y, &x, &w, None).unwrap();
        let reps = jk1_rep_weights();
        let cols: Vec<&[f64]> = reps.iter().map(|v| v.as_slice()).collect();
        let got = replicate_association(&p, &cols, AssocKind::Cov);

        for (r, (g, e)) in got.iter().zip(R_COV_REPS.iter()).enumerate() {
            assert!((g - e).abs() < 1e-10, "replicate {r}: {g} vs R {e}");
        }
    }

    #[test]
    fn test_replicate_corr_matches_r_jk1() {
        let (y, x, w) = (fc("y", &G_Y), fc("x", &G_X), fc("w", &G_W));
        let p = PairProducts::new(&y, &x, &w, None).unwrap();
        let reps = jk1_rep_weights();
        let cols: Vec<&[f64]> = reps.iter().map(|v| v.as_slice()).collect();
        let got = replicate_association(&p, &cols, AssocKind::Corr);

        for (r, (g, e)) in got.iter().zip(R_CORR_REPS.iter()).enumerate() {
            assert!((g - e).abs() < 1e-12, "replicate {r}: {g} vs R {e}");
        }
    }

    /// The precomputed-products path must agree with recomputing from scratch
    /// under each replicate's weights. Correlation is divisor-free, so this is
    /// an exact cross-check of the optimization against the direct kernel.
    #[test]
    fn test_precomputed_products_match_direct_recompute() {
        let (y, x, w) = (fc("y", &G_Y), fc("x", &G_X), fc("w", &G_W));
        let p = PairProducts::new(&y, &x, &w, None).unwrap();

        for rep in jk1_rep_weights() {
            let rw = fc("rw", &rep);
            let direct = point_estimate_corr(&y, &x, &rw).unwrap();
            let fast = p.estimate(&rep, AssocKind::Corr);
            assert!((direct - fast).abs() < 1e-12, "{direct} vs {fast}");
        }
    }

    /// R fixes `n` once outside its replicate closure, so a replicate that
    /// zeroes a PSU keeps the full-sample Kish factor rather than shrinking it.
    /// Recomputing `n` per replicate would silently rescale every covariance.
    #[test]
    fn test_replicate_cov_holds_n_fixed() {
        let (y, x, w) = (fc("y", &G_Y), fc("x", &G_X), fc("w", &G_W));
        let p = PairProducts::new(&y, &x, &w, None).unwrap();
        let rep = &jk1_rep_weights()[0];

        let fast = p.estimate(rep, AssocKind::Cov);
        // Recomputing n from the replicate would use 21 rows, not 24.
        let rw = fc("rw", rep);
        let per_replicate_n = point_estimate_cov(&y, &x, &rw).unwrap();
        let ratio = per_replicate_n / fast;
        let expected_drift = (21.0 / 20.0) / (24.0 / 23.0);

        assert!((fast - R_COV_REPS[0]).abs() < 1e-10, "fixed-n must match R");
        assert!(
            (ratio - expected_drift).abs() < 1e-9,
            "per-replicate n drifts by {ratio}, confirming why n is held fixed"
        );
    }

    /// Nulls in either column drop that row from both the sums and `n`.
    #[test]
    fn test_nulls_are_skipped() {
        let y = Float64Chunked::from_slice_options(
            "y".into(),
            &[Some(1.0), Some(2.0), None, Some(4.0)],
        );
        let x = Float64Chunked::from_slice_options(
            "x".into(),
            &[Some(2.0), Some(1.0), Some(9.0), Some(3.0)],
        );
        let w = fc("w", &[1.0, 1.0, 5.0, 1.0]);
        let m = bivar_moments(&y, &x, &w, None).unwrap();
        assert_eq!(m.n, 3, "null row must not count toward n");
        assert!((m.sum_w - 3.0).abs() < 1e-12, "null row must not add weight");
    }
}
