use super::*;

#[allow(clippy::too_many_arguments)]
pub fn compute_svyranktest(
    df: &DataFrame,
    y_col: &str,
    group_col: &str,
    weight_col: &str,
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_ssu_col: Option<&str>,
    score_method: Option<&str>,
    singleton_method: Option<&str>,
    domain_col: Option<&str>,
    domain_val: Option<&str>,
    by_col: Option<&str>,
    calib: Option<&CalibSweep>,
    rep: Option<&RepSpec>,
    scores_given: bool,
) -> CResult<DataFrame> {
    match by_col {
        None => compute_svyranktest_single(
            df,
            y_col,
            group_col,
            weight_col,
            strata_col,
            psu_col,
            ssu_col,
            fpc_col,
            fpc_ssu_col,
            score_method,
            singleton_method,
            domain_col,
            domain_val,
            calib,
            rep,
            scores_given,
            None,
        ),
        Some(by) => {
            let frame_levels = prepare_two_sample_data(
                df, y_col, weight_col, group_col, strata_col, psu_col, domain_col, domain_val,
            )?
            .3;
            compute_by_groups(df, by, domain_col, domain_val, |temp_df, d_col, d_val| {
                compute_svyranktest_single(
                    temp_df,
                    y_col,
                    group_col,
                    weight_col,
                    strata_col,
                    psu_col,
                    ssu_col,
                    fpc_col,
                    fpc_ssu_col,
                    score_method,
                    singleton_method,
                    d_col,
                    d_val,
                    calib,
                    rep,
                    scores_given,
                    Some(&frame_levels),
                )
            })
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_svyranktest_single(
    df: &DataFrame,
    y_col: &str,
    group_col: &str,
    weight_col: &str,
    strata_col: Option<&str>,
    psu_col: Option<&str>,
    ssu_col: Option<&str>,
    fpc_col: Option<&str>,
    fpc_ssu_col: Option<&str>,
    score_method: Option<&str>,
    singleton_method: Option<&str>,
    domain_col: Option<&str>,
    domain_val: Option<&str>,
    calib: Option<&CalibSweep>,
    rep: Option<&RepSpec>,
    scores_given: bool,
    // By-group only: the frame's levels, so fewer than two weighted groups yields a NaN row, not an error.
    frame_levels: Option<&[String]>,
) -> CResult<DataFrame> {
    let method = score_method
        .map(str::parse::<RankScoreMethod>)
        .transpose()?
        .unwrap_or(RankScoreMethod::Wilcoxon);

    let method_name = if scores_given {
        "custom"
    } else {
        match method {
            RankScoreMethod::Wilcoxon | RankScoreMethod::KruskalWallis => "KruskalWallis",
            RankScoreMethod::VanDerWaerden => "vanderWaerden",
            RankScoreMethod::Median => "median",
        }
    };
    let rep_score = (!scores_given).then_some(method);

    let strata = get_opt_col(df, strata_col)?;
    let psu = get_opt_col(df, psu_col)?;
    let ssu = get_opt_col(df, ssu_col)?;
    let fpc = get_opt_f64(df, fpc_col)?;
    let fpc_ssu = get_opt_f64(df, fpc_ssu_col)?;

    let (y_arr, w_arr, g_arr, levels, n, rows) = prepare_two_sample_data(
        df, y_col, weight_col, group_col, strata_col, psu_col, domain_col, domain_val,
    )?;
    let n_groups = levels.len();

    if let Some(frame) = frame_levels
        && n_groups < 2
        && frame.len() >= 2
    {
        let nan = f64::NAN;
        let n_obs = level_counts(&w_arr, &g_arr, &levels, frame)
            .iter()
            .sum::<u32>();
        return if frame.len() == 2 {
            Ok(df![
                "type" => vec!["two-sample"],
                "method" => vec![method_name],
                "y" => vec![y_col],
                "group" => vec![group_col],
                "level_0" => vec![frame[0].as_str()],
                "level_1" => vec![frame[1].as_str()],
                "delta" => vec![nan],
                "se" => vec![nan],
                "t" => vec![nan],
                "df" => vec![nan],
                "p_value" => vec![nan],
                "n" => vec![n_obs],
            ]?)
        } else {
            Ok(df![
                "type" => vec!["k-sample"],
                "method" => vec![method_name],
                "y" => vec![y_col],
                "group" => vec![group_col],
                "ndf" => vec![(frame.len() - 1) as u32],
                "ddf" => vec![nan],
                "chisq" => vec![nan],
                "f_stat" => vec![nan],
                "p_value" => vec![nan],
                "n" => vec![n_obs],
            ]?)
        };
    }

    if n_groups == 2 {
        let (level_0, level_1) = (levels[0].clone(), levels[1].clone());
        let res = match rep {
            Some(spec) => {
                ranktest_two_sample_rep(&y_arr, &g_arr, &w_arr, n, rep_score, levels, &rows, spec)?
            }
            None => ranktest_two_sample(
                &y_arr,
                &g_arr,
                &w_arr,
                n,
                strata,
                psu,
                ssu,
                fpc,
                fpc_ssu,
                method,
                singleton_method,
                calib,
                levels,
                &rows,
            )?,
        };

        Ok(df![
            "type" => vec!["two-sample"],
            "method" => vec![method_name],
            "y" => vec![y_col],
            "group" => vec![group_col],
            "level_0" => vec![level_0],
            "level_1" => vec![level_1],
            "delta" => vec![res.delta],
            "se" => vec![res.se],
            "t" => vec![res.t_stat],
            "df" => vec![res.df],
            "p_value" => vec![res.p_value],
            "n" => vec![res.n_obs as u32],
        ]?)
    } else {
        let res = match rep {
            Some(spec) => ranktest_k_sample_rep(
                &y_arr, &g_arr, &w_arr, n, n_groups, rep_score, levels, &rows, spec,
            )?,
            None => ranktest_k_sample(
                &y_arr,
                &g_arr,
                &w_arr,
                n,
                n_groups,
                strata,
                psu,
                ssu,
                fpc,
                fpc_ssu,
                method,
                singleton_method,
                calib,
                levels,
                &rows,
            )?,
        };

        Ok(df![
            "type" => vec!["k-sample"],
            "method" => vec![method_name],
            "y" => vec![y_col],
            "group" => vec![group_col],
            "ndf" => vec![res.ndf as u32],
            "ddf" => vec![res.ddf],
            "chisq" => vec![res.chisq],
            "f_stat" => vec![res.f_stat],
            "p_value" => vec![res.p_value],
            "n" => vec![res.n_obs as u32],
        ]?)
    }
}
