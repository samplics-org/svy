use super::*;

#[pyfunction]
#[pyo3(signature = (
    data, rowvar_col, weight_col,
    colvar_col=None, strata_col=None, psu_col=None, ssu_col=None,
    fpc_col=None, fpc_ssu_col=None, singleton_method=None,
    compute_totals=false,
    calib_kind=None, calib_cells=None, calib_aux=None, calib_prev_wgt=None,
    calib_pins_total=None, calib_new_wgt=None, domain_col=None,
    rep_weight_cols=None, rep_coefs=None, rep_df=None, variance_center="rep_mean",
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn tabulate_rs(
    _py: Python,
    data: PyDataFrame,
    rowvar_col: String,
    weight_col: String,
    colvar_col: Option<String>,
    strata_col: Option<String>,
    psu_col: Option<String>,
    ssu_col: Option<String>,
    fpc_col: Option<String>,
    fpc_ssu_col: Option<String>,
    singleton_method: Option<String>,
    compute_totals: bool,
    calib_kind: Option<String>,
    calib_cells: Option<Vec<String>>,
    calib_aux: Option<Vec<String>>,
    calib_prev_wgt: Option<String>,
    calib_pins_total: Option<bool>,
    calib_new_wgt: Option<String>,
    domain_col: Option<String>,
    rep_weight_cols: Option<Vec<String>>,
    rep_coefs: Option<Vec<f64>>,
    rep_df: Option<f64>,
    variance_center: &str,
) -> PyResult<(PyDataFrame, PyDataFrame)> {
    let df = into_contiguous(data);
    let rep = rep_spec(&df, rep_weight_cols, rep_coefs, rep_df, variance_center)?;
    let calib = make_calib(
        &df,
        &weight_col,
        calib_kind,
        calib_cells,
        calib_aux,
        calib_prev_wgt,
        calib_pins_total,
        calib_new_wgt,
    )
    .map_err(|e| to_py_err(&e))?;
    let result = compute_tabulate(
        &df,
        &rowvar_col,
        &weight_col,
        colvar_col.as_deref(),
        strata_col.as_deref(),
        psu_col.as_deref(),
        ssu_col.as_deref(),
        fpc_col.as_deref(),
        fpc_ssu_col.as_deref(),
        singleton_method.as_deref(),
        compute_totals,
        calib.as_ref(),
        domain_col.as_deref(),
        rep.as_ref(),
    )
    .map_err(|e| to_py_err(&e))?;
    Ok((PyDataFrame(result.0), PyDataFrame(result.1)))
}

#[allow(clippy::too_many_arguments)]
fn compute_tabulate(
    df: &DataFrame,
    rowvar_col: &str,
    weight_col: &str,
    colvar_col: Option<&str>,
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_ssu_col: Option<&str>,
    singleton_method: Option<&str>,
    compute_totals: bool,
    calib: Option<&CalibSweep>,
    domain_col: Option<&str>,
    rep: Option<&RepSpec>,
) -> CResult<(DataFrame, DataFrame)> {
    // As R's subset(): out-of-domain rows get weight 0 but keep the PSU structure; df counts units with an in-domain row.
    let domain: Option<BooleanChunked> = domain_col
        .map(|c| {
            df.column(c).and_then(|s| {
                Ok(s.as_materialized_series()
                    .cast(&DataType::Boolean)?
                    .bool()?
                    .clone())
            })
        })
        .transpose()?;
    let raw_weights = df.column(weight_col)?.as_materialized_series().f64()?;
    let zeroed: Float64Chunked;
    let weights: &Float64Chunked = match domain.as_ref() {
        Some(m) => {
            zeroed = raw_weights
                .iter()
                .enumerate()
                .map(|(i, w)| {
                    if m.get(i).unwrap_or(false) {
                        w
                    } else {
                        Some(0.0)
                    }
                })
                .collect::<Float64Chunked>()
                .with_name(raw_weights.name().clone());
            &zeroed
        }
        None => raw_weights,
    };
    let strata = strata_col.map(|c| df.column(c).cloned()).transpose()?;
    let psu = psu_col.map(|c| df.column(c).cloned()).transpose()?;
    let ssu = ssu_col.map(|c| df.column(c).cloned()).transpose()?;
    let fpc = fpc_col
        .map(|c| {
            df.column(c)
                .and_then(|s| Ok(s.as_materialized_series().f64()?.clone()))
        })
        .transpose()?;
    let fpc_ssu = fpc_ssu_col
        .map(|c| {
            df.column(c)
                .and_then(|s| Ok(s.as_materialized_series().f64()?.clone()))
        })
        .transpose()?;

    let is_two_way = colvar_col.is_some();
    let n_obs = match domain.as_ref() {
        Some(m) => m.iter().filter(|v| v.unwrap_or(false)).count(),
        None => df.height(),
    };

    let rowvar_series = df.column(rowvar_col)?.as_materialized_series();
    let rowvar_str = rowvar_series.cast(&DataType::String)?;
    let rowvar_ca = rowvar_str.str()?;

    let combined_key: StringChunked;
    let y_effective: &StringChunked;

    if let Some(cv_col) = colvar_col {
        let colvar_series = df.column(cv_col)?.as_materialized_series();
        let colvar_str = colvar_series.cast(&DataType::String)?;
        let colvar_ca = colvar_str.str()?;

        let keys: Vec<Option<String>> = rowvar_ca
            .iter()
            .zip(colvar_ca.iter())
            .map(|(a, b)| match (a, b) {
                (Some(va), Some(vb)) => Some(format!("{}__by__{}", va, vb)),
                _ => None,
            })
            .collect();
        combined_key = StringChunked::from_iter(keys);
        y_effective = &combined_key;
    } else {
        combined_key = rowvar_ca.clone();
        y_effective = &combined_key;
    }

    let rep_cells = match rep {
        Some(spec) => {
            let levels = domain_levels(y_effective, domain.as_ref())?;
            let cells = replicate_cells(y_effective, weights, domain.as_ref(), &levels, spec);
            Some((levels, cells))
        }
        None => None,
    };
    let (levels, proportions, ses, cov_matrix, deff_vec, df_val) = match &rep_cells {
        Some((levels, cells)) => {
            let k = levels.len();
            let ses: Vec<f64> = (0..k)
                .map(|j| cell_se(cells.proportions[j], cells.prop_cov[j][j]))
                .collect();
            (
                levels.clone(),
                cells.proportions.clone(),
                ses,
                cells.prop_cov.clone(),
                vec![f64::NAN; k],
                rep.map_or(0, |s| s.df.max(0.0) as u32),
            )
        }
        None => estimate_proportions(
            y_effective,
            weights,
            strata.as_ref(),
            psu.as_ref(),
            ssu.as_ref(),
            fpc.as_ref(),
            fpc_ssu.as_ref(),
            singleton_method,
            calib,
            domain.as_ref(),
        )?,
    };

    let k = levels.len();
    let mut rowvars: Vec<String> = Vec::with_capacity(k);
    let mut colvars: Vec<String> = Vec::with_capacity(k);
    for lvl in &levels {
        if is_two_way {
            match lvl.split_once("__by__") {
                Some((r, c)) => {
                    rowvars.push(r.to_string());
                    colvars.push(c.to_string());
                }
                None => {
                    rowvars.push(lvl.clone());
                    colvars.push(String::new());
                }
            }
        } else {
            rowvars.push(lvl.clone());
            colvars.push(String::new());
        }
    }

    let (est_vals, se_vals) = if let (true, Some((_, cells))) = (compute_totals, &rep_cells) {
        let ses = (0..levels.len())
            .map(|j| cells.total_cov[j][j].max(0.0).sqrt())
            .collect();
        (cells.totals.clone(), ses)
    } else if compute_totals {
        estimate_totals(
            y_effective,
            weights,
            strata.as_ref(),
            psu.as_ref(),
            ssu.as_ref(),
            fpc.as_ref(),
            fpc_ssu.as_ref(),
            singleton_method,
            calib,
            &levels,
            domain.as_ref(),
        )?
    } else {
        (proportions.clone(), ses.clone())
    };

    let cv_vals: Vec<f64> = est_vals
        .iter()
        .zip(se_vals.iter())
        .map(|(&e, &s)| {
            if e.abs() > 0.0 {
                (s / e).abs()
            } else {
                f64::NAN
            }
        })
        .collect();

    let cells_df = df![
        "rowvar" => rowvars,
        "colvar" => colvars,
        "est" => &est_vals,
        "se" => &se_vals,
        "cv" => &cv_vals,
        "deff" => &deff_vec,
        "df" => vec![df_val; k],
        "n" => vec![active_count(weights, None); k],
    ]?;

    let stats_df = if let Some(cv_col) = colvar_col {
        let colvar_series = df.column(cv_col)?.as_materialized_series();
        let colvar_str2 = colvar_series.cast(&DataType::String)?;
        let colvar_ca2 = colvar_str2.str()?;

        let row_levels = domain_levels(rowvar_ca, domain.as_ref())?;
        let col_levels = domain_levels(colvar_ca2, domain.as_ref())?;

        let nr = row_levels.len();
        let nc = col_levels.len();

        let mut prop_ordered = vec![0.0; nr * nc];
        let mut cov_ordered = vec![vec![0.0; nr * nc]; nr * nc];

        let mut level_to_ordered: HashMap<String, usize> = HashMap::new();
        for (ri, rl) in row_levels.iter().enumerate() {
            for (ci, cl) in col_levels.iter().enumerate() {
                level_to_ordered.insert(format!("{}__by__{}", rl, cl), ri * nc + ci);
            }
        }

        for (orig_idx, lvl) in levels.iter().enumerate() {
            if let Some(&ordered_idx) = level_to_ordered.get(lvl) {
                prop_ordered[ordered_idx] = proportions[orig_idx];
                for (orig_idx2, lvl2) in levels.iter().enumerate() {
                    if let Some(&ordered_idx2) = level_to_ordered.get(lvl2) {
                        cov_ordered[ordered_idx][ordered_idx2] = cov_matrix[orig_idx][orig_idx2];
                    }
                }
            }
        }

        let nu = match rep {
            Some(spec) => spec.df,
            None => {
                let (n_strata_count, n_psu_count) =
                    count_strata_psus(strata.as_ref(), psu.as_ref(), n_obs, domain.as_ref());
                n_psu_count as f64 - n_strata_count as f64
            }
        };

        let (p_chisq, p_df, p_p, p_adj_f, p_adj_ndf, p_adj_ddf, p_adj_p) =
            rao_scott(&prop_ordered, &cov_ordered, nr, nc, n_obs, nu);

        df![
            "stat" => vec!["chisq", "f"],
            "value" => vec![p_chisq, p_adj_f],
            "df" => vec![p_df, p_adj_ndf],
            "df2" => vec![f64::NAN, p_adj_ddf],
            "p_value" => vec![p_p, p_adj_p],
        ]?
    } else {
        df![
            "stat" => Vec::<String>::new(),
            "value" => Vec::<f64>::new(),
            "df" => Vec::<f64>::new(),
            "df2" => Vec::<f64>::new(),
            "p_value" => Vec::<f64>::new(),
        ]?
    };

    Ok((cells_df, stats_df))
}
