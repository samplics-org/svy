use super::*;

/// R: pnorm(x) at 17 significant digits.
#[test]
#[allow(clippy::excessive_precision)] // as R prints them
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

/// Probit's forward link is Acklam's qnorm (~1.15e-9), which only seeds IRLS: hence the looser bound.
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

/// Level 'c' (every third row) has no event but row 2's `y_c`; at 0 its dummy drifts to -infinity.
fn separated_fit(y_c: f64) -> GlmResult {
    let n = 90;
    let ones = vec![1.0; n];
    let x: Vec<f64> = (0..n).map(|i| ((i as f64) * 1.7).sin()).collect();
    let c: Vec<f64> = (0..n).map(|i| if i % 3 == 2 { 1.0 } else { 0.0 }).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            if i % 3 == 2 {
                if i == 2 { y_c } else { 0.0 }
            } else if (x[i] + ((i as f64) * 2.3).cos()) > 0.2 {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    fit_glm(
        &Series::new("y".into(), y),
        vec![
            Series::new("one".into(), ones.clone()),
            Series::new("x".into(), x),
            Series::new("c".into(), c),
        ],
        &Series::new("w".into(), ones),
        None,
        None,
        None,
        None,
        "binomial",
        "logit",
        None,
        1e-8,
        100,
        None,
    )
    .unwrap()
}

#[test]
fn separation_reports_boundary_rows_and_retained_gram() {
    let r = separated_fit(0.0);
    let sep = r.separation.expect("separated fit");
    assert_eq!(sep.boundary_rows, 30);
    // The separated dummy is zero on every retained row.
    for j in 0..3 {
        assert_eq!(sep.gram[2 * 3 + j], 0.0);
        assert_eq!(sep.gram[j * 3 + 2], 0.0);
    }
    assert_eq!(sep.gram[0], 60.0);
}

#[test]
fn no_separation_record_on_an_identified_fit() {
    // One event in level 'c' makes the MLE finite.
    assert!(separated_fit(1.0).separation.is_none());
}

#[test]
fn short_or_long_inputs_raise_dimension_mismatch() {
    use crate::errors::Coded;
    let y = Series::new("y".into(), vec![0.0, 1.0, 2.0, 1.0, 3.0, 2.0]);
    let inputs = [
        "weights",
        "strata",
        "psu",
        "fpc",
        "offset",
        "domain mask",
        "predictor",
    ];
    for family in ["poisson", "negativebinomial"] {
        for (what, got) in inputs.iter().flat_map(|&w| [(w, 5), (w, 7)]) {
            let len = |name: &str| if name == what { got } else { 6 };
            let col = |name: &str| Series::new(name.into(), vec![1.0; len(name)]);
            let mask = vec![true; len("domain mask")];
            let err = fit_one(
                &y,
                vec![col("predictor")],
                &col("weights"),
                Some(&col("strata")),
                Some(&col("psu")),
                Some(&col("fpc")),
                Some(&col("offset")),
                Some(&mask),
                family,
                "log",
                None,
                1e-8,
                100,
                None,
                SingletonRule::NONE,
            )
            .err()
            .expect("mismatch must fail");
            assert!(
                matches!(
                    err,
                    RegressionError::DimensionMismatch { what: w, expected: 6, got: g }
                        if w == what && g == got
                ),
                "{family} {what}: {err:?}"
            );
            assert_eq!(
                err.to_string(),
                format!("{what} has length {got}, expected 6")
            );
            assert_eq!(err.code(), "SHAPE_MISMATCH");
            assert!(err.invalid_input());
        }
    }
}

#[test]
fn cols_to_vec_copies_every_chunk_and_refuses_other_lengths() {
    let mut chunked = Float64Chunked::from_slice("a".into(), &[1.0, 2.0]);
    chunked
        .append(&Float64Chunked::from_slice("a".into(), &[3.0]))
        .unwrap();
    let other = Float64Chunked::from_slice("b".into(), &[4.0, 5.0, 6.0]);
    assert_eq!(
        cols_to_vec(&[&chunked, &other], 3).unwrap(),
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );

    for (vals, got) in [(&[4.0, 5.0][..], 2), (&[4.0, 5.0, 6.0, 7.0][..], 4)] {
        let bad = Float64Chunked::from_slice("b".into(), vals);
        let err = cols_to_vec(&[&chunked, &bad], 3).unwrap_err();
        assert_eq!(
            err.to_string(),
            format!("column has length {got}, expected 3")
        );
    }
}
