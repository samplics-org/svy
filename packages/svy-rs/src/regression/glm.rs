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
// NOTE: This implements the classic "bread %*% meat %*% bread" route
// (the one you had that was already extremely close to R).

use polars::prelude::*;
use faer::Mat;
use faer::Side;
use faer::prelude::SpSolver;
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
            _ => Err(PolarsError::ComputeError(format!("Unsupported family: {}", s).into())),
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
            _ => Err(PolarsError::ComputeError(format!("Unsupported link: {}", s).into())),
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
        for (i, val) in col.into_no_null_iter().enumerate() {
            mat.write(i, j, val);
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
    Z: &mut Mat<f64>,
    w_irls: &mut [f64],
    XtWX: &mut Mat<f64>,
    XtWz: &mut Mat<f64>,
) {
    // Kahan accumulators
    let mut acc_wz = vec![Kahan::new(); k];
    let mut acc_wx = vec![Kahan::new(); k * k];

    for i in 0..n {
        let w_i = w_samp[i];
        if w_i <= 0.0 {
            w_irls[i] = 0.0;
            Z.write(i, 0, 0.0);
            continue;
        }

        let y_i = Y.read(i, 0);
        let mu_i = mu[i];

        let v = family.variance(mu_i).max(1e-12);
        let d = link.mu_eta(mu_i, eta[i]); // dμ/dη

        // IRLS weight: w * (d^2 / V)
        let wi = w_i * (d * d) / v;
        w_irls[i] = wi;

        let safe_d = if d.abs() < 1e-12 { 1e-12 } else { d };
        let z_i = eta[i] + (y_i - mu_i) / safe_d;
        Z.write(i, 0, z_i);

        if wi.abs() < 1e-18 {
            continue;
        }

        for r in 0..k {
            let x_ir = X.read(i, r);
            acc_wz[r].add(wi * x_ir * z_i);

            for c in r..k {
                let x_ic = X.read(i, c);
                acc_wx[r * k + c].add(wi * x_ir * x_ic);
            }
        }
    }

    // XtWz
    for r in 0..k {
        XtWz.write(r, 0, acc_wz[r].value());
    }

    // XtWX symmetric
    for r in 0..k {
        for c in r..k {
            let v = acc_wx[r * k + c].value();
            XtWX.write(r, c, v);
            XtWX.write(c, r, v);
        }
    }

    // force symmetry (numerical)
    for r in 0..k {
        for c in 0..k {
            let v = 0.5 * (XtWX.read(r, c) + XtWX.read(c, r));
            XtWX.write(r, c, v);
        }
    }
}

/// Solve A x = b with deterministic fallback chain.
fn solve_linear_system(A: &Mat<f64>, b: &Mat<f64>) -> Mat<f64> {
    if let Ok(chol) = A.cholesky(Side::Lower) {
        return chol.solve(b);
    }

    // Symmetric indefinite
    let lblt = A.lblt(Side::Lower);
    let x = lblt.solve(b);
    let mut ok = true;
    for i in 0..x.nrows() {
        if !x.read(i, 0).is_finite() {
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
        if !x2.read(i, 0).is_finite() {
            return A.thin_svd().pseudoinverse() * b;
        }
    }
    x2
}

/// Compute A^{-1} via solving A X = I with same solve strategy.
fn invert_matrix(A: &Mat<f64>, k: usize) -> Mat<f64> {
    if let Ok(chol) = A.cholesky(Side::Lower) {
        let mut inv = Mat::<f64>::identity(k, k);
        chol.solve_in_place(&mut inv);
        return inv;
    }

    let lblt = A.lblt(Side::Lower);
    let mut inv = Mat::<f64>::identity(k, k);
    lblt.solve_in_place(&mut inv);

    // sanity: if not finite, LU then SVD
    for r in 0..k {
        for c in 0..k {
            if !inv.read(r, c).is_finite() {
                let lu = A.partial_piv_lu();
                let mut inv2 = Mat::<f64>::identity(k, k);
                lu.solve_in_place(&mut inv2);

                for rr in 0..k {
                    for cc in 0..k {
                        if !inv2.read(rr, cc).is_finite() {
                            return A.thin_svd().pseudoinverse();
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
    let family = Family::from_str(family_str)?;
    let link = Link::from_str(link_str)?;

    // 1) Data prep
    let n = y.len();
    let k = x_cols.len();

    let y_ca = y.f64()?;
    let w_ca = weights.f64()?;
    let x_ca_list: Vec<&Float64Chunked> = x_cols.iter().map(|s| s.f64().unwrap()).collect();

    let Y = cols_to_mat(&[y_ca], n);
    let X = cols_to_mat(&x_ca_list, n);

    // sampling weights
    let mut w_samp = vec![0.0; n];
    let mut w_sum = 0.0;
    for (i, v) in w_ca.into_no_null_iter().enumerate() {
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

    // 2) IRLS init
    let mut beta = Mat::<f64>::zeros(k, 1);
    let mut mu = vec![0.0; n];
    let mut eta = vec![0.0; n];

    let y_bar = if w_sum > 0.0 {
        let mut sum_wy = 0.0;
        for i in 0..n {
            sum_wy += Y.read(i, 0) * w_samp[i];
        }
        sum_wy / w_sum
    } else {
        0.0
    };

    for i in 0..n {
        mu[i] = family.initial_mu(y_bar);
        eta[i] = 0.0;
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
        {
            let pred = &X * &beta;
            for i in 0..n {
                eta[i] = pred.read(i, 0);
                mu[i] = link.inverse(eta[i]);
            }
        }

        // 2) build normal equations at current beta
        build_irls_normal_eqs(
            family, link, n, k, &Y, &X, &w_samp, &eta, &mu,
            &mut Z, &mut w_irls, &mut XtWX, &mut XtWz,
        );

        // 3) solve for beta_new
        let beta_new = solve_linear_system(&XtWX, &XtWz);

        // 4) recompute eta/mu at beta_new (THIS is what you were missing)
        let mut dev_new = 0.0;
        {
            let pred_new = &X * &beta_new;
            for i in 0..n {
                let eta_i = pred_new.read(i, 0);
                let mu_i = link.inverse(eta_i);

                // update working state to the *new* values
                eta[i] = eta_i;
                mu[i] = mu_i;

                // deviance proxy (weighted SSE) at beta_new
                let w_i = w_samp[i];
                if w_i > 0.0 {
                    let y_i = Y.read(i, 0);
                    dev_new += w_i * (y_i - mu_i).powi(2);
                }
            }
        }

        // 5) convergence check
        let mut max_delta = 0.0;
        for j in 0..k {
            let d = (beta_new.read(j, 0) - beta.read(j, 0)).abs();
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

    // final eta/mu at converged beta
    {
        let pred = &X * &beta;
        for i in 0..n {
            eta[i] = pred.read(i, 0);
            mu[i] = link.inverse(eta[i]);
        }
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
        &mut Z,
        &mut w_irls,
        &mut XtWX,
        &mut XtWz,
    );

    // strata/psu indices
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

            let mut map: HashMap<(String, String), usize> = HashMap::new();
            let mut idx = Vec::with_capacity(n);
            let mut next = 0usize;

            for i in 0..n {
                let ss = s_ca.get(i).unwrap_or("__NULL__").to_string();
                let pp = p_ca.get(i).unwrap_or("__NULL__").to_string();
                let key = (ss, pp);
                let v = *map.entry(key).or_insert_with(|| { let t = next; next += 1; t });
                idx.push(v);
            }
            (idx, next)
        }
        (_, Some(p)) => index_groups(p)?,
        _ => ((0..n).collect::<Vec<_>>(), n),
    };

    // MEAT = sum_h Var_h( PSU totals ) with svytotal-style centering
    let mut meat_acc = vec![Kahan::new(); k * k];

    for h in 0..n_strata {
        // map global PSU id -> local index
        let mut local_map: HashMap<usize, usize> = HashMap::new();
        // PSU totals: Vec[psu][k]
        let mut totals: Vec<Vec<Kahan>> = Vec::new();

        for i in 0..n {
            if strata_idx[i] != h {
                continue;
            }

            let psu_id = psu_idx[i];
            let li = *local_map.entry(psu_id).or_insert_with(|| {
                let new_i = totals.len();
                totals.push(vec![Kahan::new(); k]);
                new_i
            });

            let w_i = w_samp[i];
            if w_i <= 0.0 {
                continue;
            }

            let y_i = Y.read(i, 0);
            let mu_i = mu[i];
            let v = family.variance(mu_i).max(1e-12);
            let d = link.mu_eta(mu_i, eta[i]);
            // IRLS weight (same as used in XtWX)
            let w_irls_i = w_i * (d * d) / v;

            // Working residual
            let working_resid = (y_i - mu_i) / (d + d.signum() * 1e-12);


            // Score contribution: X * w_irls * working_resid
            // This matches R's: model.matrix * weights * resid(, "working")
            for j in 0..k {
                totals[li][j].add(w_irls_i * X.read(i, j) * working_resid);
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
            meat.write(a, b, meat_acc[a * k + b].value());
        }
    }
    // symmetrize
    for a in 0..k {
        for b in 0..k {
            let v = 0.5 * (meat.read(a, b) + meat.read(b, a));
            meat.write(a, b, v);
        }
    }

    // BREAD = (XtWX)^-1 at final beta
    let bread = invert_matrix(&XtWX, k);

    // Cov = bread * meat * bread
    let tmp = &bread * &meat;
    let cov = &tmp * &bread;

    // df_resid (your prior design-based df convention)
    let df_resid = if psu.is_some() && strata.is_some() {
        let mut total_psus = 0usize;
        for h in 0..n_strata {
            let mut set: HashMap<usize, ()> = HashMap::new();
            for i in 0..n {
                if strata_idx[i] == h {
                    set.insert(psu_idx[i], ());
                }
            }
            total_psus += set.len();
        }
        let df = (total_psus as isize) - (n_strata as isize);
        if df <= 0 { 1.0 } else { df as f64 }
    } else if psu.is_some() {
        let mut set: HashMap<usize, ()> = HashMap::new();
        for i in 0..n {
            set.insert(psu_idx[i], ());
        }
        let m = set.len() as isize;
        if m <= 1 { 1.0 } else { (m - 1) as f64 }
    } else if strata.is_some() {
        let df = (n as isize) - (n_strata as isize);
        if df <= 0 { 1.0 } else { df as f64 }
    } else {
        if n <= 1 { 1.0 } else { (n - 1) as f64 }
    };

    // scale (phi) for gaussian/gamma/invgauss (reporting only)
    let scale = if matches!(family, Family::Gaussian | Family::Gamma | Family::InverseGaussian) {
        let mut pearson = 0.0;
        for i in 0..n {
            let w_i = w_samp[i];
            if w_i <= 0.0 {
                continue;
            }
            let mu_i = mu[i];
            let v = family.variance(mu_i).max(1e-12);
            let y_i = Y.read(i, 0);
            pearson += w_i * (y_i - mu_i).powi(2) / v;
        }
        if df_resid > 0.0 { pearson / df_resid } else { 1.0 }
    } else {
        1.0
    };

    // null deviance proxy
    let null_deviance = {
        let y_mean = if w_sum > 0.0 {
            let mut sum_wy = 0.0;
            for i in 0..n {
                sum_wy += Y.read(i, 0) * w_samp[i];
            }
            sum_wy / w_sum
        } else {
            0.0
        };

        let mut sse = 0.0;
        for i in 0..n {
            let w_i = w_samp[i];
            if w_i <= 0.0 {
                continue;
            }
            let y_i = Y.read(i, 0);
            sse += w_i * (y_i - y_mean).powi(2);
        }
        sse
    };

    // flatten
    let params: Vec<f64> = (0..k).map(|i| beta.read(i, 0)).collect();
    let mut cov_flat = Vec::with_capacity(k * k);
    for r in 0..k {
        for c in 0..k {
            cov_flat.push(cov.read(r, c));
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
        n_obs: n,
    })
}
