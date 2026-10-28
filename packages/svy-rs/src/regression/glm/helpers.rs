use super::*;

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
pub(super) fn accumulate_weighted_crossprod(
    cols: &[f64],
    w: &[f64],
    n: usize,
    k: usize,
    out: &mut [f64],
) {
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
pub(super) fn build_irls_normal_eqs(
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
pub(super) fn solve_linear_system(A: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Mat<f64> {
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
pub(super) fn check_rank(XtWX: &Mat<f64>, k: usize, cols: &[Series]) -> RResult<()> {
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
