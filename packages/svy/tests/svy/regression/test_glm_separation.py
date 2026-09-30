# tests/svy/regression/test_glm_separation.py
"""
Quasi-complete separation in a survey GLM.

When the response is perfectly predicted on some rows, the maximum likelihood
estimates along that direction are infinite. IRLS stops anyway, once the
deviance settles, and every package prints the stopped values: R (silently),
Stata's `svy: poisson` (silently); Stata's `svy: logit` drops the predictor.

svy keeps the stopped estimate, reports NaN for its standard error, tests and
intervals (and for the model Wald test and AIC), and raises GLM_SEPARATION
naming the terms and levels. The identified coefficients are unaffected: they
match R's svyglm (survey 4.5, epsilon = 1e-12) on the same data, and Stata's
`svy: logit` estimates, which refits without the separated rows.
"""

import warnings

import numpy as np
import polars as pl
import pytest

import svy

from svy.core.warnings import WarnCode


SEP = r"ignore:\[GLM_SEPARATION\]:svy.SvyUserWarning"


def _data() -> pl.DataFrame:
    """Level 'c' of g never has an event (binary) or a count (Poisson)."""
    rng = np.random.default_rng(20260926)
    n = 240
    x = rng.normal(size=n)
    g = np.tile(["a", "b", "c"], n // 3)
    eta = -0.3 + 0.8 * x + np.where(g == "b", 0.7, 0.0)
    y = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    cnt = rng.poisson(np.exp(0.2 + 0.4 * x))
    return pl.DataFrame(
        {
            "stratum": np.repeat([1, 2, 3, 4], n // 4),
            "psu": np.repeat(np.arange(24), n // 24),
            "wgt": rng.uniform(1, 3, size=n),
            "x": x,
            "x2": rng.normal(size=n),
            "g": g,
            "y": np.where(g == "c", 0, y),
            "cnt": np.where(g == "c", 0, cnt),
            "y_ok": y,
            "dom": np.arange(n) % 2 == 0,
        }
    )


@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    return _data()


def _sample(data: pl.DataFrame) -> svy.Sample:
    return svy.Sample(data, svy.Design(stratum="stratum", psu="psu", wgt="wgt"))


def _fit(sample: svy.Sample, **kw):
    """Fit, returning the GLMFit and the GLM_SEPARATION warnings raised."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fit = sample.glm.fit(**kw).fitted
    raised = [w for w in rec if "[GLM_SEPARATION]" in str(w.message)]
    others = [w for w in rec if w not in raised]
    assert not others, [str(w.message) for w in others]
    return fit, raised


def _coef(fit, term):
    return next(c for c in fit.coefs if c.term == term)


def _findings(sample):
    return [w for w in sample.warnings if w.code == WarnCode.GLM_SEPARATION]


# ---------------------------------------------------------------------------
# The separated coefficient: NaN statistics, estimate kept
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family, y", [("binomial", "y"), ("poisson", "cnt")])
def test_separated_level_has_nan_statistics(data, family, y):
    fit, raised = _fit(_sample(data), y=y, x=["x", svy.Cat("g")], family=family)
    assert len(raised) == 1
    c = _coef(fit, "g_c")
    assert np.isfinite(c.est) and c.est < -5
    for v in (c.se, c.lci, c.uci, c.wald.value, c.wald.p_value):
        assert np.isnan(v)
    j = fit.feature_names.index("g_c")
    assert np.isnan(fit.cov_matrix[j]).all()
    assert np.isnan(fit.cov_matrix[:, j]).all()
    others = [i for i in range(len(fit.feature_names)) if i != j]
    assert np.isfinite(fit.cov_matrix[np.ix_(others, others)]).all()
    assert np.isnan(fit.stats.wald.value) and np.isnan(fit.stats.wald.p_value)
    assert np.isnan(fit.stats.wald_adj.value) and np.isnan(fit.stats.wald_adj.p_value)
    assert np.isnan(fit.stats.aic)
    assert np.isfinite(fit.stats.deviance)


# survey 4.5, svyglm(y ~ x + g, design, family = quasibinomial / quasipoisson,
# control = glm.control(epsilon = 1e-12)) on _data().
R_IDENTIFIED = {
    "binomial": {
        "_intercept_": (0.02661304456198074, 0.24314353138961420),
        "x": (0.49688598699275499, 0.21597156993153982),
        "g_b": (0.80423535001932556, 0.40459223672517075),
    },
    "poisson": {
        "_intercept_": (0.20652991145178715, 0.13394372763035983),
        "x": (0.39670029173240412, 0.07977201753101444),
        "g_b": (-0.17380497871902303, 0.14915062827395442),
    },
}


@pytest.mark.parametrize("family, y", [("binomial", "y"), ("poisson", "cnt")])
def test_identified_coefficients_match_r(data, family, y):
    fit, _ = _fit(_sample(data), y=y, x=["x", svy.Cat("g")], family=family, tol=1e-12)
    for term, (est, se) in R_IDENTIFIED[family].items():
        c = _coef(fit, term)
        assert c.est == pytest.approx(est, rel=1e-7)
        assert c.se == pytest.approx(se, rel=1e-6)


@pytest.mark.parametrize("tol", [1e-6, 1e-8, 1e-10, 1e-12])
def test_identified_results_do_not_depend_on_tol(data, tol):
    ref, _ = _fit(_sample(data), y="y", x=["x", svy.Cat("g")], family="binomial")
    fit, raised = _fit(_sample(data), y="y", x=["x", svy.Cat("g")], family="binomial", tol=tol)
    assert len(raised) == 1
    for term in ("_intercept_", "x", "g_b"):
        assert _coef(fit, term).est == pytest.approx(_coef(ref, term).est, rel=1e-6, abs=1e-6)
        assert _coef(fit, term).se == pytest.approx(_coef(ref, term).se, rel=1e-5)
    assert np.isnan(_coef(fit, "g_c").se)


@pytest.mark.parametrize("link", ["logit", "probit", "cloglog"])
def test_every_binomial_link(data, link):
    fit, raised = _fit(_sample(data), y="y", x=["x", svy.Cat("g")], family="binomial", link=link)
    assert len(raised) == 1
    assert [c.term for c in fit.coefs if np.isnan(c.se)] == ["g_c"]


@pytest.mark.filterwarnings(r"ignore:\[MAX_ITER_REACHED\]:svy.SvyUserWarning")
def test_cauchit_separation_found_without_convergence(data):
    # Cauchit's heavy tail drifts slowly: IRLS runs out of iterations, and the
    # separated rows are still found at the boundary.
    s = _sample(data)
    with pytest.warns(svy.SvyUserWarning, match=r"\[GLM_SEPARATION\]"):
        fit = s.glm.fit(y="y", x=["x", svy.Cat("g")], family="binomial", link="cauchit").fitted
    assert [c.term for c in fit.coefs if np.isnan(c.se)] == ["g_c"]
    assert {w.code for w in s.warnings} == {"MAX_ITER_REACHED", "GLM_SEPARATION"}


@pytest.mark.parametrize("theta", [None, 2.0])
def test_negative_binomial(data, theta):
    fit, raised = _fit(
        _sample(data), y="cnt", x=["x", svy.Cat("g")], family="negative_binomial", theta=theta
    )
    assert len(raised) == 1
    assert [c.term for c in fit.coefs if np.isnan(c.se)] == ["g_c"]


# ---------------------------------------------------------------------------
# The finding
# ---------------------------------------------------------------------------


def test_finding_names_terms_rows_and_levels(data):
    s = _sample(data)
    _, raised = _fit(s, y="y", x=["x", svy.Cat("g")], family="binomial")
    (w,) = _findings(s)
    assert str(raised[0].message).startswith("[GLM_SEPARATION]")
    assert w.where == "GLM.fit"
    assert w.var == "g"
    n_c = int((data["g"] == "c").sum())
    assert w.got == {"not_identified": ["g_c"], "boundary_rows": n_c}
    assert f"on {n_c} of {data.height} rows" in w.detail
    assert "'g_c'" in w.detail
    assert "the AIC and the model Wald test" in w.detail
    assert "'y' is 0 in every row with g == 'c'" in w.hint
    assert "Collapse those levels of 'g' or drop it from x" in w.hint


def test_finding_raised_at_the_callers_line(data):
    s = _sample(data)
    with pytest.warns(svy.SvyUserWarning, match=r"\[GLM_SEPARATION\]") as rec:
        s.glm.fit(y="y", x=["x", svy.Cat("g")], family="binomial")
    assert rec[0].filename == __file__


def test_refit_on_same_state_raises_once(data):
    s = _sample(data)
    _, first = _fit(s, y="y", x=["x", svy.Cat("g")], family="binomial")
    fit, second = _fit(s, y="y", x=["x", svy.Cat("g")], family="binomial")
    assert len(first) == 1 and second == []
    assert len(_findings(s)) == 1
    # The NaN statistics do not depend on the finding having been raised.
    assert np.isnan(_coef(fit, "g_c").se)


def test_count_hint_names_the_zero_level(data):
    s = _sample(data)
    _fit(s, y="cnt", x=["x", svy.Cat("g")], family="poisson")
    (w,) = _findings(s)
    assert "'cnt' is 0 in every row with g == 'c'" in w.hint
    assert " 1 in every row" not in w.hint


def test_separated_reference_level_reports_identified_combination(data):
    # With 'c' as reference the intercept drifts too, and each other level's
    # dummy with it; only intercept + dummy is pinned down for a level that
    # still varies. With x out of the model that sum is the level's log-odds.
    s = _sample(data)
    fit, _ = _fit(s, y="y", x=[svy.Cat("g", ref="c")], family="binomial")
    assert [c.term for c in fit.coefs if np.isnan(c.se)] == ["_intercept_", "g_a", "g_b"]
    (w,) = _findings(s)
    assert w.got["not_identified"] == ["_intercept_", "g_a", "g_b"]
    # Two varying levels: two combinations, none reported as a sentence.
    assert "is identified:" not in w.detail
    assert "g == 'c'" in w.hint

    ab = data.filter(pl.col("g") != "a")
    s2 = _sample(ab)
    fit2, _ = _fit(s2, y="y", x=[svy.Cat("g", ref="c")], family="binomial")
    (w2,) = _findings(s2)
    b = ab.filter(pl.col("g") == "b")
    p = float((b["y"] * b["wgt"]).sum() / b["wgt"].sum())
    total = _coef(fit2, "_intercept_").est + _coef(fit2, "g_b").est
    assert total == pytest.approx(np.log(p / (1 - p)), abs=1e-6)
    assert f"The combination _intercept_ + g_b is identified: {total:.6g}." in w2.detail
    # Only the intercept is left in the model Wald test's complement.
    assert np.isnan(fit2.stats.wald.value)


def test_complete_separation_by_a_continuous_predictor(data):
    df = data.with_columns((pl.col("x") > 0).cast(pl.Int64).alias("y_x"))
    s = _sample(df)
    fit, raised = _fit(s, y="y_x", x=["x"], family="binomial")
    assert len(raised) == 1
    assert all(np.isnan(c.se) for c in fit.coefs)
    (w,) = _findings(s)
    assert w.got["boundary_rows"] == df.height
    assert w.var == "x"
    # No categorical to point at: the hint names the term.
    assert "given 'x'" in w.hint


def test_slope_separated_intercept_identified():
    # y is 1 wherever x > 0, 0 wherever x < 0, and varies at x == 0: the slope
    # drifts to +infinity while the intercept is the log-odds at x == 0.
    n = 60
    x = np.r_[np.zeros(20), np.tile([-2.0, -1.0, 1.0, 2.0], 10)]
    y = np.r_[np.tile([0, 1, 1, 0, 1], 4), (np.tile([-2.0, -1.0, 1.0, 2.0], 10) > 0)]
    df = pl.DataFrame({"x": x, "y": y.astype(int), "w": np.ones(n), "psu": np.arange(n) // 3})
    s = svy.Sample(df, svy.Design(psu="psu", wgt="w"))
    fit, raised = _fit(s, y="y", x=["x"], family="binomial")
    assert len(raised) == 1
    assert [c.term for c in fit.coefs if np.isnan(c.se)] == ["x"]
    icpt = _coef(fit, "_intercept_")
    assert icpt.est == pytest.approx(np.log(12 / 8), abs=1e-6)
    assert np.isfinite(icpt.se)
    assert np.isnan(fit.stats.wald.value)
    (w,) = _findings(s)
    assert w.got == {"not_identified": ["x"], "boundary_rows": 40}


def test_constant_response_at_one_value_is_not_separation():
    # y is 0 wherever x == 0 but no direction sends those rows to -infinity
    # while keeping the rows at x = 1..4 finite: the MLE exists.
    n = 60
    x = np.r_[np.zeros(20), np.tile([1.0, 2.0, 3.0, 4.0], 10)]
    y = np.r_[np.zeros(20), np.tile([0, 1, 1, 0, 1, 0, 1, 1], 5)]
    df = pl.DataFrame({"x": x, "y": y, "w": np.ones(n), "psu": np.arange(n) // 3})
    s = svy.Sample(df, svy.Design(psu="psu", wgt="w"))
    fit, raised = _fit(s, y="y", x=["x"], family="binomial")
    assert raised == []
    assert all(np.isfinite(c.se) for c in fit.coefs)


# ---------------------------------------------------------------------------
# Scope: where=, replicate designs
# ---------------------------------------------------------------------------


def test_where_domain_separated_only_inside(data):
    # Separation confined to the domain: the full-sample model on y_ok is
    # clean, the domain model on y is separated.
    s = _sample(data)
    fit, raised = _fit(s, y="y", x=["x", svy.Cat("g")], family="binomial", where=svy.col("dom"))
    assert len(raised) == 1
    assert [c.term for c in fit.coefs if np.isnan(c.se)] == ["g_c"]
    (w,) = _findings(s)
    n_c = int(data.filter(pl.col("dom") & (pl.col("g") == "c")).height)
    assert w.got["boundary_rows"] == n_c


def test_where_domain_outside_the_separated_rows_is_clean(data):
    s = _sample(data)
    fit, raised = _fit(s, y="y", x=["x"], family="binomial", where=svy.col("g") != "c")
    assert raised == []
    assert _findings(s) == []
    assert all(np.isfinite(c.se) for c in fit.coefs)


def test_replicate_design_main_fit_separated(data):
    s = _sample(data).weighting.create_jk_wgts()
    fit, raised = _fit(s, y="y", x=["x", svy.Cat("g")], family="binomial")
    assert len(raised) == 1
    assert [c.term for c in fit.coefs if np.isnan(c.se)] == ["g_c"]
    assert all(np.isfinite(c.se) for c in fit.coefs if c.term != "g_c")


def test_replicate_refit_separated_only():
    # Level 'c' has a single event, in one PSU: the full fit is identified,
    # but the jackknife replicate that deletes that PSU is separated.
    df = _data().with_columns(
        pl.when(pl.col("g") != "c")
        .then(pl.col("y"))
        .when(pl.int_range(pl.len()) == 2)
        .then(1)
        .otherwise(0)
        .alias("y1")
    )
    s = _sample(df).weighting.create_jk_wgts()
    fit, raised = _fit(s, y="y1", x=["x", svy.Cat("g")], family="binomial", method="replication")
    assert len(raised) == 1
    assert [c.term for c in fit.coefs if np.isnan(c.se)] == ["g_c"]
    (w,) = _findings(s)
    assert w.got["boundary_rows"] == 0
    assert w.got["replicate_refits"] == {"g_c": 1}
    assert f"'g_c' in 1 of {s.n_reps}" in w.detail
    assert "predicted perfectly on" not in w.detail

    # The Taylor fit of the same model is identified and raises nothing.
    t_fit, t_raised = _fit(_sample(df), y="y1", x=["x", svy.Cat("g")], family="binomial")
    assert t_raised == []
    assert np.isfinite(_coef(t_fit, "g_c").se)


# ---------------------------------------------------------------------------
# No finding on ordinary fits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "family, y, link",
    [
        ("binomial", "y_ok", "logit"),
        ("binomial", "y_ok", "probit"),
        ("binomial", "y_ok", "cloglog"),
        ("poisson", "x2", None),
        ("gaussian", "y", None),
        ("gaussian", "cnt", None),
    ],
)
def test_ordinary_fits_raise_nothing(data, family, y, link):
    df = (
        data.with_columns(pl.col("x2").abs().round(0).cast(pl.Int64))
        if family == "poisson"
        else data
    )
    s = _sample(df)
    fit, raised = _fit(s, y=y, x=["x", svy.Cat("g")], family=family, link=link)
    assert raised == []
    assert _findings(s) == []
    assert all(np.isfinite(c.se) for c in fit.coefs)
    assert np.isfinite(fit.stats.wald.value)


def test_near_collinear_design_is_not_separation(data):
    df = data.with_columns((pl.col("x") + 1e-4 * pl.col("x2")).alias("x_near"))
    fit, raised = _fit(_sample(df), y="y_ok", x=["x", "x_near"], family="binomial")
    assert raised == []
    assert all(np.isfinite(c.se) for c in fit.coefs)


def test_rare_outcome_with_events_in_every_level():
    rng = np.random.default_rng(3)
    n = 20000
    g = rng.choice(["a", "b", "c"], size=n, p=[0.49, 0.49, 0.02])
    x = rng.normal(size=n)
    y = rng.binomial(1, 1 / (1 + np.exp(-(-6.0 + 0.8 * x))))
    y[np.flatnonzero(g == "c")[:2]] = 1
    df = pl.DataFrame(
        {"g": g, "x": x, "y": y, "w": rng.uniform(1, 5, n), "psu": np.arange(n) // 100}
    )
    s = svy.Sample(df, svy.Design(psu="psu", wgt="w"))
    fit, raised = _fit(s, y="y", x=["x", svy.Cat("g")], family="binomial")
    assert raised == []
    assert all(np.isfinite(c.se) for c in fit.coefs)


# ---------------------------------------------------------------------------
# Downstream: tests, contrasts, prediction, export
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sep_model(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _sample(data).glm.fit(y="y", x=["x", svy.Cat("g")], family="binomial")


def test_term_test(sep_model):
    t_g = sep_model.fitted.term_test("g")
    assert np.isnan(t_g.value) and np.isnan(t_g.p_value)
    t_x = sep_model.fitted.term_test("x")
    assert np.isfinite(t_x.value) and np.isfinite(t_x.p_value)
    t_b = sep_model.fitted.term_test("g_b")
    assert np.isfinite(t_b.value)


def test_predict_se_nan_only_on_separated_rows(sep_model, data):
    pred = sep_model.predict(data)
    yhat, se = np.asarray(pred.yhat), np.asarray(pred.se)
    c_rows = (data["g"] == "c").to_numpy()
    assert np.isfinite(yhat).all()
    assert (yhat[c_rows] < 1e-6).all()
    # Those rows' linear predictor is not identified; every other row's
    # does not involve g_c at all.
    assert np.isnan(se[c_rows]).all()
    assert np.isfinite(se[~c_rows]).all()


def test_margins_keep_identified_standard_errors(sep_model):
    # The average marginal effects and predictive margins depend on g_c only
    # through rows at the boundary, where d mu / d eta has collapsed: they are
    # identified, and the NaN covariance row must not leak into them.
    ame = {r.term: r for r in sep_model.margins(variables=["g", "x"])}
    assert np.isfinite(ame["x"].to_polars()["se"].to_numpy()).all()
    g = ame["g"].to_polars()
    assert np.isfinite(g["se"].to_numpy()).all()
    pm = sep_model.margins(at={"g": ["a", "b", "c"]}).to_polars()
    by = dict(zip(pm["value"].to_list(), pm["margin"].to_list()))
    se = dict(zip(pm["value"].to_list(), pm["se"].to_list()))
    assert by["c"] < 1e-6 and se["c"] < 1e-6
    assert np.isfinite(se["a"]) and se["a"] > 1e-3
    # c - a is minus a's margin, with its standard error.
    ca = g.filter(pl.col("value") == "c - a").row(0, named=True)
    assert ca["margin"] == pytest.approx(-by["a"], abs=1e-6)
    assert ca["se"] == pytest.approx(se["a"], rel=1e-6)


def test_delta_var():
    from svy.regression.glm import delta_var

    cov = np.array([[2.0, 0.5, np.nan], [0.5, 1.0, np.nan], [np.nan, np.nan, np.nan]])
    assert delta_var(np.array([1.0, 1.0, 0.0]), cov) == pytest.approx(4.0)
    assert delta_var(np.array([1.0, 0.0, 1e-9]), cov) == pytest.approx(2.0)
    assert np.isnan(delta_var(np.array([1.0, 0.0, 1e-3]), cov))
    out = delta_var(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), cov)
    assert out[0] == pytest.approx(1.0) and np.isnan(out[1])
    clean = np.array([[2.0, 0.5], [0.5, 1.0]])
    assert delta_var(np.array([1.0, -1.0]), clean) == pytest.approx(2.0)


def test_to_polars_carries_nan(sep_model):
    row = sep_model.to_polars().filter(pl.col("term") == "g_c").row(0, named=True)
    assert np.isfinite(row["estimate"])
    for col in ("std_err", "conf_low", "conf_high", "statistic", "p_value"):
        assert np.isnan(row[col])


def test_contrast_touching_the_separated_level_is_nan(sep_model):
    df = sep_model.contrast({"c vs b": {"g_c": 1, "g_b": -1}, "b": {"g_b": 1}}).to_polars()
    se = dict(zip(df["contrast"].to_list(), df["se"].to_list()))
    assert np.isnan(se["c vs b"])
    assert se["b"] == pytest.approx(_coef(sep_model.fitted, "g_b").se)
