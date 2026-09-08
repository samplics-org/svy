// src/regression/negbin.rs
//
// The negative binomial's dispersion, and the joint (beta, theta) variance.
//
// The family arm in glm.rs fits beta by IRLS at a KNOWN theta, which is all a
// family can do — theta is a second parameter, not a property of the mean.
// This module adds the two things that need:
//
//   * `theta_ml`, Newton on the weighted profile likelihood (MASS::theta.ml),
//     driven by an outer loop that alternates with the IRLS the way
//     MASS::glm.nb does. The pair it converges to is the joint MLE.
//
//   * the design-based variance over BOTH parameters, which is R survey's
//     `svymle` route — the method Lumley gives for this model in *Complex
//     Surveys* (Appendix E) and what `sjstats::svyglm.nb` implements.
//     Conditioning on theta-hat instead is the cheaper composition of
//     `glm.nb` with `svyglm`, and it is a different number: on apistrat the
//     two run from 25% under to 13% over each other. The orthogonality that
//     makes them agree asymptotically is a property of the model, and a
//     design-based sandwich is the thing that declines to assume it.
//
// The log-likelihood, dropping terms free of (mu, theta):
//
//   l = lgamma(theta + y) - lgamma(theta) - lgamma(y + 1)
//       + theta log theta + y log mu - (theta + y) log(theta + mu)

use faer::Mat;
use polars::prelude::*;

use crate::estimation::calib_sweep::CalibSweep;
use crate::regression::glm::{
    GlmResult, Link, design_codes, design_vcov_of_totals, fit_glm_domain, invert_matrix,
};
use crate::regression::special::{digamma, lgamma, trigamma};

/// The identity and sqrt links can push mu to zero, and y/mu is the first
/// thing the score does.
const MU_FLOOR: f64 = 1e-10;

fn contributing(w: f64) -> bool {
    w > 0.0
}

/// The weighted log-likelihood.
fn loglik(y: &[f64], mu: &[f64], theta: f64, w: &[f64]) -> f64 {
    let mut total = 0.0;
    for i in 0..y.len() {
        if !contributing(w[i]) {
            continue;
        }
        let m = mu[i].max(MU_FLOOR);
        let yi = y[i];
        total += w[i]
            * (lgamma(theta + yi) - lgamma(theta) - lgamma(yi + 1.0)
                + theta * theta.ln()
                + yi * (m + if yi == 0.0 { 1.0 } else { 0.0 }).ln()
                - (theta + yi) * (theta + m).ln());
    }
    total
}

/// (score, observed information) of the profile likelihood in theta.
fn theta_score_info(y: &[f64], mu: &[f64], theta: f64, w: &[f64]) -> (f64, f64) {
    let dg_theta = digamma(theta);
    let tg_theta = trigamma(theta);
    let mut score = 0.0;
    let mut info = 0.0;
    for i in 0..y.len() {
        if !contributing(w[i]) {
            continue;
        }
        let m = mu[i].max(MU_FLOOR);
        let yi = y[i];
        let tm = theta + m;
        score += w[i]
            * (digamma(theta + yi) - dg_theta + theta.ln() + 1.0 - tm.ln() - (yi + theta) / tm);
        info += w[i]
            * (-trigamma(theta + yi) + tg_theta - 1.0 / theta + 2.0 / tm
                - (yi + theta) / (tm * tm));
    }
    (score, info)
}

/// Maximum-likelihood theta at fixed mu — `MASS::theta.ml`.
///
/// Newton from the method-of-moments start `sum(w) / sum(w (y/mu - 1)^2)`.
/// Scale-invariant in the weights: a global factor cancels out of the root.
fn theta_ml(y: &[f64], mu: &[f64], w: &[f64], eps: f64, limit: usize) -> PolarsResult<f64> {
    let mut w_sum = 0.0;
    let mut denom = 0.0;
    for i in 0..y.len() {
        if !contributing(w[i]) {
            continue;
        }
        let r = y[i] / mu[i].max(MU_FLOOR) - 1.0;
        w_sum += w[i];
        denom += w[i] * r * r;
    }
    if !(denom > 0.0) {
        return Err(PolarsError::ComputeError(
            "negative binomial: the Pearson statistic is zero, so theta has no \
             moment estimate — the response does not vary around the fitted mean"
                .into(),
        ));
    }

    let mut t = w_sum / denom;
    let mut delta: f64 = 1.0;
    for _ in 0..limit {
        if delta.abs() <= eps {
            break;
        }
        t = t.abs();
        let (score, info) = theta_score_info(y, mu, t, w);
        if info == 0.0 || !info.is_finite() {
            break;
        }
        delta = score / info;
        t += delta;
    }

    if !(t.is_finite() && t > 0.0) {
        return Err(PolarsError::ComputeError(
            format!(
                "negative binomial: the profile likelihood in theta reached {t}. \
                 The counts are probably not overdispersed — fit family='poisson', \
                 or pass theta to fit at a known value."
            )
            .into(),
        ));
    }
    Ok(t)
}

/// Fit the negative binomial, estimating theta unless it is given.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_negbin(
    y: &Series,
    x_cols: Vec<Series>,
    weights: &Series,
    strata: Option<&Series>,
    psu: Option<&Series>,
    fpc: Option<&Series>,
    offset: Option<&Series>,
    domain_mask: Option<&[bool]>,
    link_str: &str,
    theta: Option<f64>,
    tol: f64,
    max_iter: usize,
    calib: Option<&CalibSweep>,
) -> PolarsResult<GlmResult> {
    let fit_at = |family: &str, th: Option<f64>, want_variance: bool| {
        fit_glm_domain(
            y,
            x_cols.clone(),
            weights,
            strata,
            psu,
            fpc,
            offset,
            domain_mask,
            family,
            link_str,
            th,
            tol,
            max_iter,
            calib,
            want_variance,
        )
    };

    // A theta the caller knows is not a parameter: the ordinary sandwich,
    // which holds it fixed, is the right variance and there is no row to
    // report a standard error for.
    if let Some(th) = theta {
        let mut result = fit_at("negativebinomial", Some(th), true)?;
        result.theta = Some(th);
        return Ok(result);
    }

    let link = Link::from_str(link_str)?;
    let n = y.len();
    let k = x_cols.len();

    // The same materialisation fit_glm_domain does, needed again here for the
    // profile likelihood and the joint bread.
    let (y_vals, x, w, offset_vals) = materialise(y, &x_cols, weights, offset, domain_mask, n, k)?;

    let eps = tol.max(1e-14);
    let mu_of = |beta: &[f64]| -> Vec<f64> {
        let mut mu = vec![0.0; n];
        for i in 0..n {
            let mut eta = offset_vals[i];
            for (j, &b) in beta.iter().enumerate() {
                eta += x[j * n + i] * b;
            }
            mu[i] = link.inverse(eta);
        }
        mu
    };

    // R seeds theta from a Poisson fit, the theta -> infinity limit.
    let mut mu = mu_of(&fit_at("poisson", None, false)?.params);
    let mut theta = theta_ml(&y_vals, &mu, &w, eps, max_iter)?;

    // MASS::glm.nb's alternation, with its convergence test made relative —
    // the weighted log-likelihood here is O(1e4), so its own rounding is
    // ~1e-10 and an absolute test could never reach a tol of 1e-12 however
    // converged theta is.
    let mut delta: f64 = 1.0;
    let mut ll = loglik(&y_vals, &mu, theta, &w);
    let mut ll_prev = ll + 2.0 * ll.abs().max(1.0);
    let mut settled = false;

    for _ in 0..max_iter {
        if (ll_prev - ll).abs() / (0.1 + ll.abs()) + delta.abs() / (0.1 + theta.abs()) <= tol {
            settled = true;
            break;
        }
        mu = mu_of(&fit_at("negativebinomial", Some(theta), false)?.params);
        let prev = theta;
        theta = theta_ml(&y_vals, &mu, &w, eps, max_iter)?;
        delta = prev - theta;
        ll_prev = ll;
        ll = loglik(&y_vals, &mu, theta, &w);
    }

    let mut result = fit_at("negativebinomial", Some(theta), true)?;

    let (cov, theta_se) = joint_vcov(
        &y_vals,
        &x,
        &w,
        &offset_vals,
        &result.params,
        theta,
        link,
        n,
        k,
        strata,
        psu,
        fpc,
    )?;

    // The kernel's own sandwich conditions on theta; this one does not.
    result.cov_params = cov;
    result.theta = Some(theta);
    result.theta_se = Some(theta_se);
    result.converged = result.converged && settled;
    Ok(result)
}

/// y, X (column-major), the contributing-row weights, and the offset.
fn materialise(
    y: &Series,
    x_cols: &[Series],
    weights: &Series,
    offset: Option<&Series>,
    domain_mask: Option<&[bool]>,
    n: usize,
    k: usize,
) -> PolarsResult<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> {
    let y_cast = y.cast(&DataType::Float64)?;
    let y_vals: Vec<f64> = y_cast.f64()?.iter().map(|v| v.unwrap_or(0.0)).collect();

    let mut x = vec![0.0; n * k];
    for (j, s) in x_cols.iter().enumerate() {
        let cast = s.cast(&DataType::Float64)?;
        for (i, v) in cast.f64()?.iter().enumerate() {
            x[j * n + i] = v.unwrap_or(0.0);
        }
    }

    // Out-of-domain and non-positive-weight rows carry no information, but
    // stay in the frame: the design structure is the full design's.
    let w_cast = weights.cast(&DataType::Float64)?;
    let w: Vec<f64> = w_cast
        .f64()?
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let wi = v.unwrap_or(0.0);
            if domain_mask.is_none_or(|m| m[i]) && wi > 0.0 {
                wi
            } else {
                0.0
            }
        })
        .collect();

    let offset_vals = match offset {
        Some(s) => {
            let cast = s.cast(&DataType::Float64)?;
            cast.f64()?.iter().map(|v| v.unwrap_or(0.0)).collect()
        }
        None => vec![0.0; n],
    };

    Ok((y_vals, x, w, offset_vals))
}

/// The joint (beta, theta) design variance: `svyrecvar(scores %*% -H^-1)`.
///
/// Returns the beta block, row-major k*k, and the standard error of theta.
#[allow(clippy::too_many_arguments)]
fn joint_vcov(
    y: &[f64],
    x: &[f64],
    w: &[f64],
    offset_vals: &[f64],
    beta: &[f64],
    theta: f64,
    link: Link,
    n: usize,
    k: usize,
    strata: Option<&Series>,
    psu: Option<&Series>,
    fpc: Option<&Series>,
) -> PolarsResult<(Vec<f64>, f64)> {
    let p = k + 1;
    let dg_theta = digamma(theta);
    let tg_theta = trigamma(theta);

    // Per-row score and second-derivative pieces.
    let mut s_eta = vec![0.0; n]; // w dl/deta
    let mut s_theta = vec![0.0; n]; // w dl/dtheta
    let mut h_eta = vec![0.0; n]; // w d2l/deta2
    let mut h_cross = vec![0.0; n]; // w d2l/deta dtheta
    let mut h_theta = 0.0; // sum w d2l/dtheta2

    for i in 0..n {
        if !contributing(w[i]) {
            continue;
        }
        let mut eta = offset_vals[i];
        for (j, &b) in beta.iter().enumerate() {
            eta += x[j * n + i] * b;
        }
        let mu = link.inverse(eta).max(MU_FLOOR);
        let yi = y[i];
        let tm = theta + mu;
        let ty = theta + yi;

        let dl_dmu = yi / mu - ty / tm;
        let d2l_dmu2 = -yi / (mu * mu) + ty / (tm * tm);
        let d2l_dmu_dtheta = (yi - mu) / (tm * tm);

        let mu_eta = link.mu_eta(mu, eta);
        let mu_eta2 = link.mu_eta2(mu, eta);

        s_eta[i] = w[i] * dl_dmu * mu_eta;
        s_theta[i] = w[i] * (digamma(ty) - dg_theta + theta.ln() + 1.0 - tm.ln() - ty / tm);
        h_eta[i] = w[i] * (d2l_dmu2 * mu_eta * mu_eta + dl_dmu * mu_eta2);
        h_cross[i] = w[i] * d2l_dmu_dtheta * mu_eta;

        let d2l_dtheta2 = trigamma(ty) - tg_theta + 1.0 / theta - 2.0 / tm + ty / (tm * tm);
        if d2l_dtheta2.is_finite() {
            h_theta += w[i] * d2l_dtheta2;
        }
    }

    // Bread = -H of the weighted log-likelihood, ordered [beta..., theta].
    let mut bread = Mat::<f64>::zeros(p, p);
    for a in 0..k {
        let xa = &x[a * n..(a + 1) * n];
        for b in a..k {
            let xb = &x[b * n..(b + 1) * n];
            let mut acc = 0.0;
            for i in 0..n {
                acc += h_eta[i] * xa[i] * xb[i];
            }
            bread[(a, b)] = -acc;
            bread[(b, a)] = -acc;
        }
        let mut cross = 0.0;
        for i in 0..n {
            cross += h_cross[i] * xa[i];
        }
        bread[(a, k)] = -cross;
        bread[(k, a)] = -cross;
    }
    bread[(k, k)] = -h_theta;

    let a_inv = invert_matrix(bread.as_ref(), p)?;

    // Influence functions, column-major n x p.
    let mut influence = vec![0.0; n * p];
    for col in 0..p {
        let dst_off = col * n;
        for a in 0..k {
            let coef = a_inv[(a, col)];
            if coef == 0.0 {
                continue;
            }
            let xa = &x[a * n..(a + 1) * n];
            for i in 0..n {
                influence[dst_off + i] += s_eta[i] * xa[i] * coef;
            }
        }
        let coef = a_inv[(k, col)];
        if coef != 0.0 {
            for i in 0..n {
                influence[dst_off + i] += s_theta[i] * coef;
            }
        }
    }

    // Design variance of the column totals.
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
    let fpc_rows: Option<Vec<f64>> = match fpc {
        Some(s) => {
            let cast = s.cast(&DataType::Float64)?;
            Some(cast.f64()?.iter().map(|v| v.unwrap_or(1.0)).collect())
        }
        None => None,
    };
    let psu_opt = if psu.is_some() {
        Some(psu_idx.as_slice())
    } else {
        None
    };

    let vcov = design_vcov_of_totals(
        &influence,
        n,
        p,
        &strata_idx,
        &strata_obs,
        psu_opt,
        n_psu_levels,
        fpc_rows.as_deref(),
    );

    let mut beta_block = vec![0.0; k * k];
    for a in 0..k {
        for b in 0..k {
            beta_block[a * k + b] = vcov[a * p + b];
        }
    }
    Ok((beta_block, vcov[k * p + k].max(0.0).sqrt()))
}
