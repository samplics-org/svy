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
use std::collections::HashMap;

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
}

impl Family {
    pub fn from_str(s: &str) -> PolarsResult<Self> {
        match s.to_lowercase().as_str() {
            "gaussian" => Ok(Family::Gaussian),
            "binomial" => Ok(Family::Binomial),
            "poisson" => Ok(Family::Poisson),
            "gamma" => Ok(Family::Gamma),
            "inversegaussian" | "inverse_gaussian" => Ok(Family::InverseGaussian),
            _ => Err(PolarsError::ComputeError(
                format!("Unsupported family: {}", s).into(),
            )),
        }
    }

    fn variance(&self, mu: f64) -> f64 {
        match self {
            Family::Gaussian => 1.0,
            Family::Binomial => mu * (1.0 - mu),
            Family::Poisson => mu,
            Family::Gamma => mu * mu,
            Family::InverseGaussian => mu * mu * mu,
        }
    }

    fn initial_mu(&self, y: f64) -> f64 {
        let eps = 1e-10;
        match self {
            Family::Binomial => (y + 0.5) / 2.0,
            Family::Poisson | Family::Gamma | Family::InverseGaussian => y.max(eps),
            Family::Gaussian => y,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Link {
    Identity,
    Logit,
    Log,
    Inverse,
    InverseSquared,
}

impl Link {
    pub fn from_str(s: &str) -> PolarsResult<Self> {
        match s.to_lowercase().as_str() {
            "identity" => Ok(Link::Identity),
            "logit" => Ok(Link::Logit),
            "log" => Ok(Link::Log),
            "inverse" => Ok(Link::Inverse),
            "inverse_squared" => Ok(Link::InverseSquared),
            _ => Err(PolarsError::ComputeError(
                format!("Unsupported link: {}", s).into(),
            )),
        }
    }

    fn link(&self, mu: f64) -> f64 {
        match self {
            Link::Identity => mu,
            Link::Logit => (mu / (1.0 - mu)).ln(),
            Link::Log => mu.max(1e-10).ln(),
            Link::Inverse => 1.0 / mu,
            Link::InverseSquared => 1.0 / (mu * mu),
        }
    }

    fn inverse(&self, eta: f64) -> f64 {
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

            Link::Log => eta.clamp(-30.0, 30.0).exp(),
            Link::Inverse => 1.0 / eta,
            Link::InverseSquared => 1.0 / eta.sqrt(),
        }
    }

    /// dμ/dη
    fn mu_eta(&self, mu: f64, _eta: f64) -> f64 {
        match self {
            Link::Identity => 1.0,
            Link::Logit => mu * (1.0 - mu),
            Link::Log => mu,
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

fn cols_to_mat(cols: &[&Float64Chunked], nrows: usize) -> Mat<f64> {
    let ncols = cols.len();
    let mut mat = Mat::<f64>::zeros(nrows, ncols);
    for (j, col) in cols.iter().enumerate() {
        // Null-aware iteration: `into_no_null_iter` would silently compact
        // rows past a null, misaligning y/X/w. Nulls are rejected upstream in
        // fit_glm_domain; any that slip through become 0.0 at the right row.
        for (i, val) in col.iter().enumerate() {
            mat[(i, j)] = val.unwrap_or(0.0);
        }
    }
    mat
}

/// Robust group indexing for String/Categorical/Enum/anything-castable-to-string.
fn index_groups(series: &Series) -> PolarsResult<(Vec<usize>, usize)> {
    let mut map: HashMap<String, usize> = HashMap::new();
    let mut indices = Vec::with_capacity(series.len());
    let mut next_idx = 0;

    match series.dtype() {
        DataType::String => {
            let ca = series.str()?;
            for opt_s in ca.into_iter() {
                let s = opt_s.unwrap_or("__NULL__");
                let idx = *map.entry(s.to_string()).or_insert_with(|| {
                    let i = next_idx;
                    next_idx += 1;
                    i
                });
                indices.push(idx);
            }
        }
        DataType::Categorical(_, _) | DataType::Enum(_, _) => {
            let physical = series.to_physical_repr();
            let ca = physical.u32()?;
            let mut phys_map: HashMap<u32, usize> = HashMap::new();

            for opt_v in ca.into_iter() {
                let v = opt_v.unwrap_or(u32::MAX);
                let idx = *phys_map.entry(v).or_insert_with(|| {
                    let i = next_idx;
                    next_idx += 1;
                    i
                });
                indices.push(idx);
            }
        }
        _ => {
            let s_str = series.cast(&DataType::String)?;
            return index_groups(&s_str);
        }
    }

    Ok((indices, next_idx))
}

/// Build XtWX and XtWz deterministically from current eta/mu (IRLS step),
/// mirroring fisherinf: t(D) %*% (w * D / V), D = X * d, d = dmu/deta
///
/// When `domain_mask` is `Some`, rows where mask[i] is false contribute 0 to
/// both XtWX and XtWz (and have w_irls[i] set to 0). When `None`, every row
/// contributes (original behavior).
fn build_irls_normal_eqs(
    family: Family,
    link: Link,
    n: usize,
    k: usize,
    Y: &Mat<f64>,
    X: &Mat<f64>,
    w_samp: &[f64],
    eta: &[f64],
    mu: &[f64],
    domain_mask: Option<&[bool]>,
    Z: &mut Mat<f64>,
    w_irls: &mut [f64],
    XtWX: &mut Mat<f64>,
    XtWz: &mut Mat<f64>,
) {
    // Kahan accumulators
    let mut acc_wz = vec![Kahan::new(); k];
    let mut acc_wx = vec![Kahan::new(); k * k];

    for i in 0..n {
        let in_domain = domain_mask.map_or(true, |m| m[i]);
        let w_i = w_samp[i];

        if !in_domain || w_i <= 0.0 {
            w_irls[i] = 0.0;
            Z[(i, 0)] = 0.0;
            continue;
        }

        let y_i = Y[(i, 0)];
        let mu_i = mu[i];

        let v = family.variance(mu_i).max(1e-12);
        let d = link.mu_eta(mu_i, eta[i]); // dμ/dη

        // IRLS weight: w * (d^2 / V)
        let wi = w_i * (d * d) / v;
        w_irls[i] = wi;

        let safe_d = if d.abs() < 1e-12 { 1e-12 } else { d };
        let z_i = eta[i] + (y_i - mu_i) / safe_d;
        Z[(i, 0)] = z_i;

        if wi.abs() < 1e-18 {
            continue;
        }

        for r in 0..k {
            let x_ir = X[(i, r)];
            acc_wz[r].add(wi * x_ir * z_i);

            for c in r..k {
                let x_ic = X[(i, c)];
                acc_wx[r * k + c].add(wi * x_ir * x_ic);
            }
        }
    }

    // XtWz
    for r in 0..k {
        XtWz[(r, 0)] = acc_wz[r].value();
    }

    // XtWX symmetric
    for r in 0..k {
        for c in r..k {
            let v = acc_wx[r * k + c].value();
            XtWX[(r, c)] = v;
            XtWX[(c, r)] = v;
        }
    }

    // force symmetry (numerical)
    for r in 0..k {
        for c in 0..k {
            let v = 0.5 * (XtWX[(r, c)] + XtWX[(c, r)]);
            XtWX[(r, c)] = v;
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
            return A.thin_svd().unwrap().pseudoinverse() * b;
        }
    }
    x2
}

/// Compute A^{-1} via solving A X = I with same solve strategy.
fn invert_matrix(A: MatRef<'_, f64>, k: usize) -> Mat<f64> {
    if let Ok(chol) = A.llt(Side::Lower) {
        let mut inv = Mat::<f64>::identity(k, k);
        chol.solve_in_place(inv.as_mut());
        return inv;
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
                            return A.thin_svd().unwrap().pseudoinverse();
                        }
                    }
                }
                return inv2;
            }
        }
    }

    inv
}

// ============================================================================
// Result
// ============================================================================

#[allow(dead_code)]
pub struct GlmResult {
    pub params: Vec<f64>,
    pub cov_params: Vec<f64>,
    pub scale: f64,
    pub df_resid: f64,
    pub deviance: f64,
    pub null_deviance: f64,
    pub iterations: u32,
    pub n_obs: usize,
}

// ============================================================================
// Core Algorithm
// ============================================================================

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
    family_str: &str,
    link_str: &str,
    tol: f64,
    max_iter: usize,
) -> PolarsResult<GlmResult> {
    fit_glm_domain(
        y, x_cols, weights, strata, psu, None, family_str, link_str, tol, max_iter,
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
    by_col: &Series,
    family_str: &str,
    link_str: &str,
    tol: f64,
    max_iter: usize,
) -> PolarsResult<Vec<(String, GlmResult)>> {
    // Materialize by_col as strings, enumerate unique levels.
    let by_str_series = by_col.cast(&DataType::String)?;
    let by_str = by_str_series.str()?;
    let unique_groups = by_str.unique()?;

    let mut results: Vec<(String, GlmResult)> = Vec::new();

    for group_opt in unique_groups.iter() {
        if let Some(group_val) = group_opt {
            let mask_vec: Vec<bool> = by_str
                .iter()
                .map(|v| v.map_or(false, |s| s == group_val))
                .collect();

            // Clone the x_cols Vec for each domain fit. The Series themselves
            // are cheap reference-counted handles in polars; the clone copies
            // only the Vec, not the underlying data.
            let xs = x_cols.iter().cloned().collect();
            let res = fit_glm_domain(
                y,
                xs,
                weights,
                strata,
                psu,
                Some(&mask_vec),
                family_str,
                link_str,
                tol,
                max_iter,
            )?;

            results.push((group_val.to_string(), res));
        }
    }

    Ok(results)
}

/// Core GLM fit. When `domain_mask` is `None`, behavior is byte-identical to
/// the original `fit_glm`. When `Some`, out-of-domain rows contribute 0 to
/// both the IRLS loop and the sandwich meat, while the strata/PSU
/// enumeration remains based on the full design.
fn fit_glm_domain(
    y: &Series,
    x_cols: Vec<Series>,
    weights: &Series,
    strata: Option<&Series>,
    psu: Option<&Series>,
    domain_mask: Option<&[bool]>,
    family_str: &str,
    link_str: &str,
    tol: f64,
    max_iter: usize,
) -> PolarsResult<GlmResult> {
    let family = Family::from_str(family_str)?;
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
            format!("GLM weight column '{}' contains null values", weights.name()).into(),
        ));
    }
    for s in &x_cast {
        if s.null_count() > 0 {
            return Err(PolarsError::ComputeError(
                format!("GLM predictor column '{}' contains null values", s.name()).into(),
            ));
        }
    }

    let Y = cols_to_mat(&[y_ca], n);
    let X = cols_to_mat(&x_ca_list, n);

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

    // 2) IRLS init — only meaningful for in-domain rows; out-of-domain rows
    //    get a neutral placeholder since they won't contribute.
    let mut beta = Mat::<f64>::zeros(k, 1);
    let mut mu = vec![0.0; n];
    let mut eta = vec![0.0; n];

    for i in 0..n {
        let in_domain = domain_mask.map_or(true, |m| m[i]);
        let y_init = if in_domain { Y[(i, 0)] } else { 0.5 };
        mu[i] = family.initial_mu(y_init);
        eta[i] = link.link(mu[i]);
    }

    // work arrays
    let mut Z = Mat::<f64>::zeros(n, 1);
    let mut w_irls = vec![0.0; n];
    let mut XtWX = Mat::<f64>::zeros(k, k);
    let mut XtWz = Mat::<f64>::zeros(k, 1);

    // 3) IRLS loop
    let mut iter_count = 0;
    let mut deviance = 0.0;

    for iter in 0..max_iter {
        iter_count += 1;

        // 1) eta/mu from current beta
        if iter > 0 {
            for i in 0..n {
                let mut s = 0.0f64;
                for j in 0..k {
                    s += X[(i, j)] * beta[(j, 0)];
                }
                eta[i] = s;
                mu[i] = link.inverse(s);
            }
        }

        // 2) build normal equations at current beta
        build_irls_normal_eqs(
            family,
            link,
            n,
            k,
            &Y,
            &X,
            &w_samp,
            &eta,
            &mu,
            domain_mask,
            &mut Z,
            &mut w_irls,
            &mut XtWX,
            &mut XtWz,
        );

        // 3) solve for beta_new
        let beta_new = solve_linear_system(XtWX.as_ref(), XtWz.as_ref());

        // 4) recompute eta/mu at beta_new — direct loop, no N×1 Mat alloc
        let mut dev_new = 0.0;
        {
            for i in 0..n {
                let mut s = 0.0f64;
                for j in 0..k {
                    s += X[(i, j)] * beta_new[(j, 0)];
                }
                let mu_i = link.inverse(s);
                eta[i] = s;
                mu[i] = mu_i;
                let in_domain = domain_mask.map_or(true, |m| m[i]);
                if !in_domain {
                    continue;
                }
                let w_i = w_samp[i];
                if w_i > 0.0 {
                    let y_i = Y[(i, 0)];
                    dev_new += w_i * (y_i - mu_i).powi(2);
                }
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

        if iter > 0 && (rel_dev < tol || max_delta < tol) {
            break;
        }
    }

    // =========================================================================
    // 4) Sandwich variance (R-alignment: rebuild XtWX at FINAL beta)
    // =========================================================================

    // final eta/mu at converged beta — direct loop
    for i in 0..n {
        let mut s = 0.0f64;
        for j in 0..k {
            s += X[(i, j)] * beta[(j, 0)];
        }
        eta[i] = s;
        mu[i] = link.inverse(s);
    }

    // rebuild XtWX at final beta (bread must match fisherinf)
    build_irls_normal_eqs(
        family,
        link,
        n,
        k,
        &Y,
        &X,
        &w_samp,
        &eta,
        &mu,
        domain_mask,
        &mut Z,
        &mut w_irls,
        &mut XtWX,
        &mut XtWz,
    );

    // strata/psu indices — FULL design (not affected by domain)
    let (strata_idx, n_strata) = match strata {
        Some(s) => index_groups(s)?,
        None => (vec![0usize; n], 1usize),
    };

    let (psu_idx, _n_psu_levels) = match (strata, psu) {
        (Some(s), Some(p)) => {
            // nest PSU within strata: id = (stratum, psu)
            let s_str = s.cast(&DataType::String)?;
            let p_str = p.cast(&DataType::String)?;
            let s_ca = s_str.str()?;
            let p_ca = p_str.str()?;
            let s_vals: Vec<&str> = s_ca.into_iter().map(|v| v.unwrap_or("__NULL__")).collect();
            let p_vals: Vec<&str> = p_ca.into_iter().map(|v| v.unwrap_or("__NULL__")).collect();

            let mut map: HashMap<(&str, &str), usize> = HashMap::new();
            let mut idx = Vec::with_capacity(n);
            let mut next = 0usize;

            for i in 0..n {
                let key = (s_vals[i], p_vals[i]);
                let v = *map.entry(key).or_insert_with(|| {
                    let t = next;
                    next += 1;
                    t
                });
                idx.push(v);
            }
            (idx, next)
        }
        (_, Some(p)) => index_groups(p)?,
        _ => ((0..n).collect::<Vec<_>>(), n),
    };

    // Pre-build strata → obs index
    let mut strata_obs: Vec<Vec<usize>> = vec![Vec::new(); n_strata];
    for i in 0..n {
        strata_obs[strata_idx[i]].push(i);
    }

    // MEAT = sum_h Var_h( PSU totals ) with svytotal-style centering.
    // Out-of-domain rows have w_irls[i] == 0 from build_irls_normal_eqs, so
    // their score contributions are naturally 0.
    let mut meat_acc = vec![Kahan::new(); k * k];

    for h in 0..n_strata {
        let mut local_map: HashMap<usize, usize> = HashMap::new();
        let mut totals: Vec<Vec<Kahan>> = Vec::new();

        for &i in &strata_obs[h] {
            let psu_id = psu_idx[i];
            let li = *local_map.entry(psu_id).or_insert_with(|| {
                let new_i = totals.len();
                totals.push(vec![Kahan::new(); k]);
                new_i
            });

            let w_i = w_samp[i];
            let w_irls_i = w_irls[i];

            // Domain-aware: w_irls_i is 0 for out-of-domain rows.
            if w_i <= 0.0 || w_irls_i <= 0.0 {
                continue;
            }

            let y_i = Y[(i, 0)];
            let mu_i = mu[i];
            let d = link.mu_eta(mu_i, eta[i]);

            // Working residual
            let working_resid = (y_i - mu_i) / (d + d.signum() * 1e-12);

            // Score contribution: X * w_irls * working_resid
            for j in 0..k {
                totals[li][j].add(w_irls_i * X[(i, j)] * working_resid);
            }
        }

        let m = totals.len();
        if m <= 1 {
            continue;
        }

        // mean-center PSU totals in stratum
        let mut mean = vec![0.0; k];
        for t in &totals {
            for j in 0..k {
                mean[j] += t[j].value();
            }
        }
        for j in 0..k {
            mean[j] /= m as f64;
        }

        // with-replacement factor m/(m-1)
        let scale_h = (m as f64) / ((m - 1) as f64);

        for a in 0..k {
            for b in 0..k {
                let mut s = Kahan::new();
                for t in &totals {
                    let da = t[a].value() - mean[a];
                    let db = t[b].value() - mean[b];
                    s.add(da * db);
                }
                meat_acc[a * k + b].add(scale_h * s.value());
            }
        }
    }

    // materialize meat
    let mut meat = Mat::<f64>::zeros(k, k);
    for a in 0..k {
        for b in 0..k {
            meat[(a, b)] = meat_acc[a * k + b].value();
        }
    }
    // symmetrize
    for a in 0..k {
        for b in 0..k {
            let v = 0.5 * (meat[(a, b)] + meat[(b, a)]);
            meat[(a, b)] = v;
        }
    }

    // BREAD = (XtWX)^-1 at final beta
    let bread = invert_matrix(XtWX.as_ref(), k);

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

    // n_obs: full sample size when no domain, in-domain count when domain.
    let n_obs = match domain_mask {
        Some(m) => (0..n).filter(|&i| m[i] && w_samp[i] > 0.0).count(),
        None => n,
    };

    // scale (phi) for gaussian/gamma/invgauss (reporting only).
    // Pearson sum is computed over in-domain rows only.
    let scale = if matches!(
        family,
        Family::Gaussian | Family::Gamma | Family::InverseGaussian
    ) {
        let mut pearson = 0.0;
        for i in 0..n {
            let in_domain = domain_mask.map_or(true, |m| m[i]);
            if !in_domain {
                continue;
            }
            let w_i = w_samp[i];
            if w_i <= 0.0 {
                continue;
            }
            let mu_i = mu[i];
            let v = family.variance(mu_i).max(1e-12);
            let y_i = Y[(i, 0)];
            pearson += w_i * (y_i - mu_i).powi(2) / v;
        }
        if df_resid > 0.0 {
            pearson / df_resid
        } else {
            1.0
        }
    } else {
        1.0
    };

    // null deviance proxy — weighted SSE around weighted mean over domain.
    let null_deviance = {
        // domain-restricted weighted mean
        let mut sum_wy = 0.0;
        let mut sum_w = 0.0;
        for i in 0..n {
            let in_domain = domain_mask.map_or(true, |m| m[i]);
            if !in_domain || w_samp[i] <= 0.0 {
                continue;
            }
            sum_wy += Y[(i, 0)] * w_samp[i];
            sum_w += w_samp[i];
        }
        let y_mean = if sum_w > 0.0 { sum_wy / sum_w } else { 0.0 };

        let mut sse = 0.0;
        for i in 0..n {
            let in_domain = domain_mask.map_or(true, |m| m[i]);
            if !in_domain {
                continue;
            }
            let w_i = w_samp[i];
            if w_i <= 0.0 {
                continue;
            }
            let y_i = Y[(i, 0)];
            sse += w_i * (y_i - y_mean).powi(2);
        }
        sse
    };

    // Suppress unused-var warning when no domain restriction is active.
    let _ = w_sum;

    // flatten
    let params: Vec<f64> = (0..k).map(|i| beta[(i, 0)]).collect();
    let mut cov_flat = Vec::with_capacity(k * k);
    for r in 0..k {
        for c in 0..k {
            cov_flat.push(cov[(r, c)]);
        }
    }

    Ok(GlmResult {
        params,
        cov_params: cov_flat,
        scale,
        df_resid,
        deviance,
        null_deviance,
        iterations: iter_count as u32,
        n_obs,
    })
}
