use super::*;

#[allow(clippy::too_many_arguments)]
pub fn compute_assoc(
    df: &DataFrame,
    y_cols: &[String],
    x_cols: &[String],
    kind: AssocKind,
    weight_col: &str,
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_ssu_col: Option<&str>,
    by_col: Option<&str>,
    singleton_method: Option<&str>,
    domain_rows_col: Option<&str>,
    srs: SrsRef,
    calib: Option<CalibSweep>,
) -> EResult<DataFrame> {
    let weights = df.column(weight_col)?.f64()?;
    let rows = rows_column(df, domain_rows_col)?;
    let strata = strata_col.map(|c| df.column(c)).transpose()?;
    let psu = psu_col.map(|c| df.column(c)).transpose()?;
    let ssu = ssu_col.map(|c| df.column(c)).transpose()?;
    let fpc = fpc_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;
    let fpc_ssu = fpc_ssu_col
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .transpose()?;

    // Resolve columns outside the parallel region, as the other multi kernels do.
    let ys: Vec<&Float64Chunked> = y_cols
        .iter()
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .collect::<PolarsResult<Vec<_>>>()?;
    let xs: Vec<&Float64Chunked> = x_cols
        .iter()
        .map(|c| df.column(c).and_then(|s| s.f64()))
        .collect::<PolarsResult<Vec<_>>>()?;

    let design =
        build_taylor_design(strata, psu, ssu, fpc, fpc_ssu, singleton_method)?.with_calib(calib);

    // Ungrouped runs as a single unmasked group, so one fan-out covers both shapes.
    let by_str = by_col
        .map(|c| df.column(c).and_then(|s| s.str()))
        .transpose()?;
    let unique_groups = by_str.map(crate::estimation::sorted_groups).transpose()?;
    let groups: Vec<Option<&str>> = match unique_groups.as_ref() {
        Some(u) => u.iter().flatten().map(Some).collect(),
        None => vec![None],
    };

    // A by-group is a domain: its df counts its own active PSUs/strata, not the frame's.
    let group_dfs: Vec<u32> = groups
        .par_iter()
        .map(|g| match (g, by_str) {
            (Some(gv), Some(bs)) => {
                degrees_of_freedom_from_design(weights, &design, Some(&bs.equal(*gv)))
            }
            _ => degrees_of_freedom_from_design(weights, &design, None),
        })
        .collect();

    let n_pairs = ys.len();
    let combos: Vec<(usize, usize)> = (0..groups.len())
        .flat_map(|gi| (0..n_pairs).map(move |pi| (gi, pi)))
        .collect();

    let rows = combos
        .par_iter()
        .map(
            |&(gi, pi)| -> EResult<(usize, usize, f64, f64, f64, u32, f64)> {
                let mask = match (groups[gi], by_str) {
                    (Some(g), Some(bs)) => Some(bs.equal(g)),
                    _ => None,
                };
                let (y, x) = (ys[pi], xs[pi]);
                let active = design.domain_rows(weights.len(), rows.as_ref(), mask.as_ref());
                let estimate = point_estimate_assoc(kind, y, x, weights, mask.as_ref())?;
                let n = active_count(weights, mask.as_ref());
                // An undefined estimate (a constant column's correlation) has no variance either.
                if estimate.is_nan() {
                    return Ok((gi, pi, estimate, f64::NAN, f64::NAN, n, f64::NAN));
                }
                let scores = scores_assoc(kind, y, x, weights, mask.as_ref())?;
                let scores_arr: Vec<f64> = scores.iter().map(|s| s.unwrap_or(0.0)).collect();
                let variance = taylor_variance_apply_in(&scores_arr, &design, active.as_deref());
                let se = variance.max(0.0).sqrt();
                let srs_var = srs_variance_assoc_of(kind, y, x, weights, mask.as_ref(), srs)?;
                let deff = if srs_var > 0.0 {
                    variance / srs_var
                } else {
                    f64::NAN
                };
                Ok((gi, pi, estimate, se, variance, n, deff))
            },
        )
        .collect::<EResult<Vec<_>>>()?;

    let nv = rows.len();
    let mut by_vals: Vec<&str> = Vec::with_capacity(nv);
    let mut y_names: Vec<&str> = Vec::with_capacity(nv);
    let mut x_names: Vec<&str> = Vec::with_capacity(nv);
    let mut estimates: Vec<f64> = Vec::with_capacity(nv);
    let mut ses: Vec<f64> = Vec::with_capacity(nv);
    let mut variances: Vec<f64> = Vec::with_capacity(nv);
    let mut dfs: Vec<u32> = Vec::with_capacity(nv);
    let mut ns: Vec<u32> = Vec::with_capacity(nv);
    let mut deffs: Vec<f64> = Vec::with_capacity(nv);
    for (gi, pi, est, se, var, n, deff) in rows {
        if let Some(g) = groups[gi] {
            by_vals.push(g);
        }
        y_names.push(y_cols[pi].as_str());
        x_names.push(x_cols[pi].as_str());
        estimates.push(est);
        ses.push(se);
        variances.push(var);
        dfs.push(group_dfs[gi]);
        ns.push(n);
        deffs.push(deff);
    }

    let kinds = vec![kind.as_str(); nv];
    let mut out = df![
        "y" => y_names, "x" => x_names, "kind" => kinds, "est" => estimates,
        "se" => ses, "var" => variances, "df" => dfs, "n" => ns, "deff" => deffs
    ]?;
    if let Some(name) = by_col {
        out.insert_column(0, Column::new(name.into(), by_vals))?;
    }
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::estimation::calib_sweep::{CalibSweep, SweepKind};
    use crate::test_fixtures::{N, ROW, W, X2, Y, floats, psu, strata};
    use approx::assert_relative_eq;

    #[test]
    fn poststratified_cov_se_matches_r() {
        let ps_w: [f64; N] = [
            2.0454545454545454,
            2.1666666666666665,
            1.3636363636363635,
            3.2500000000000000,
            2.7083333333333330,
            1.3636363636363635,
            2.7272727272727271,
            1.6250000000000000,
            4.0909090909090908,
            1.0833333333333333,
            2.1666666666666665,
            3.4090909090909083,
        ];
        let df = DataFrame::new_infer_height(vec![
            strata().into_column(),
            psu().into_column(),
            floats("w", &ps_w).into_column(),
            floats("y", &Y).into_column(),
            floats("x2", &X2).into_column(),
        ])
        .unwrap();
        let calib = CalibSweep::new(
            SweepKind::Cells,
            vec![ROW.to_vec()],
            vec![2],
            vec![],
            W.to_vec(),
            ps_w.to_vec(),
            true,
        );
        let out = compute_assoc(
            &df,
            &["y".into()],
            &["x2".into()],
            AssocKind::Cov,
            "w",
            Some("st"),
            Some("psu"),
            None,
            None,
            None,
            None,
            None,
            None,
            SrsRef::Wor,
            Some(calib),
        )
        .unwrap();
        let est = out.column("est").unwrap().f64().unwrap().get(0).unwrap();
        let se = out.column("se").unwrap().f64().unwrap().get(0).unwrap();
        // R survey::svyvar(~y + x2, postStratify(des, ~r, Freq = c(15, 13)))
        assert_relative_eq!(est, 1.2819023711217206, max_relative = 1e-12);
        assert_relative_eq!(se, 0.61115052028652095, max_relative = 1e-12);
    }
}
