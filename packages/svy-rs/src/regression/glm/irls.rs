use super::*;

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
