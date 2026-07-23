# tests/svy/regression/test_glm_design_gaps.py
"""
Round 8 GLM design-gap fixes (findings R6-R13).

R reference values generated with survey 4.5 (2026-07-23):

  api <- read.csv("apistrat.csv"); api$y_bin <- as.integer(api$api00 > 600)
  d1  <- svydesign(ids=~1, weights=~pw, data=api)
  m1  <- svyglm(y_bin ~ ell + meals + mobility, design=d1, family=quasibinomial())
  deviance(m1); m1$null.deviance; AIC(m1)          # 137.2613 / 253.4408 / 143.1756699
  mg  <- svyglm(api00 ~ ell + meals, design=d1)
  deviance(mg); AIC(mg)                            # 1030475.410961 / 2283.0747437
  mp  <- svyglm(enroll ~ ell + meals, design=d1, family=quasipoisson())
  deviance(mp); AIC(mp)                            # 50539.724280 / 51351.3251693
  dfpc<- svydesign(ids=~1, strata=~stype, weights=~pw, fpc=~fpc, data=api)
  SE(svyglm(api00 ~ ell + meals, design=dfpc))     # 8.75949495 0.38791652 0.27576549
  SE(svyglm(y_bin ~ ell + meals, design=dfpc, family=quasibinomial()))
                                                   # 0.57895049 0.01493313 0.01296360
  dnf <- svydesign(ids=~1, strata=~stype, weights=~pw, data=api)
  rj  <- as.svrepdesign(dnf, type="JKn")
  SE(svyglm(api00 ~ ell + meals, design=rj))       # 9.01172834 0.40647424 0.28786459
  api2 <- api; api2$ell[1:5] <- NA
  m2  <- svyglm(api00 ~ ell + meals, design=svydesign(ids=~1, strata=~stype,
                weights=~pw, data=api2))
  coef(m2); SE(m2)   # 822.39263891 -0.58973797 -3.04680103 / 8.97658993 0.41001108 0.28108227

AIC(svyglm) is the Lumley-Scott dAIC; BIC(svyglm) requires a maximal
model (no plain BIC exists), hence stats.bic is None.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import svy

from svy.core.design import RepWeights
from svy.core.sample import Design, Sample
from svy.core.terms import Cat
from svy.errors.model_errors import ModelError


DATA_DIR = Path(__file__).resolve().parents[2] / "test_data"

RTOL = 1e-6
RTOL_SE = 1e-6


@pytest.fixture
def api_strat():
    return pl.read_csv(DATA_DIR / "apistrat.csv")


@pytest.fixture
def api_binary(api_strat):
    return api_strat.with_columns((pl.col("api00") > 600).cast(pl.Int32).alias("y_bin"))


# =============================================================================
# R6 — family deviance, dAIC, no plain BIC
# =============================================================================


class TestDevianceAIC:
    def test_binomial_deviance_matches_r(self, api_binary):
        s = Sample(api_binary, Design(wgt="pw"))
        m = s.glm.fit(y="y_bin", x=["ell", "meals", "mobility"], family="binomial")
        st = m.fitted.stats
        np.testing.assert_allclose(st.deviance, 137.2613, rtol=1e-5)
        np.testing.assert_allclose(st.aic, 143.1756699, rtol=RTOL)
        assert st.bic is None
        # null deviance via R^2 = 1 - dev/null_dev
        null_dev = st.deviance / (1.0 - st.r_squared)
        np.testing.assert_allclose(null_dev, 253.4408, rtol=1e-5)

    def test_gaussian_deviance_and_daic_match_r(self, api_strat):
        s = Sample(api_strat, Design(wgt="pw"))
        m = s.glm.fit(y="api00", x=["ell", "meals"])
        st = m.fitted.stats
        np.testing.assert_allclose(st.deviance, 1030475.410961, rtol=RTOL)
        np.testing.assert_allclose(st.aic, 2283.0747437, rtol=RTOL)
        assert st.bic is None

    def test_poisson_deviance_matches_r(self, api_strat):
        s = Sample(api_strat, Design(wgt="pw"))
        m = s.glm.fit(y="enroll", x=["ell", "meals"], family="poisson")
        st = m.fitted.stats
        np.testing.assert_allclose(st.deviance, 50539.724280, rtol=RTOL)
        np.testing.assert_allclose(st.aic, 51351.3251693, rtol=RTOL)


# =============================================================================
# R7 — replicate-weight designs use replicate variance
# =============================================================================


def _jkn_columns(df: pl.DataFrame, stratum: str, wgt: str) -> tuple[pl.DataFrame, int]:
    """Deterministic JKn delete-one weights (ids=~1): replicate r zeroes row
    r and scales same-stratum rows by n_h/(n_h-1) — R as.svrepdesign
    type='JKn' construction."""
    strata = df.get_column(stratum).to_list()
    w = df.get_column(wgt).to_numpy().astype(float)
    n = len(w)
    from collections import Counter

    n_h = Counter(strata)
    cols = {}
    for r in range(n):
        h = strata[r]
        f = n_h[h] / (n_h[h] - 1)
        col = np.array(
            [0.0 if i == r else (w[i] * f if strata[i] == h else w[i]) for i in range(n)]
        )
        cols[f"jw{r + 1}"] = col
    return df.with_columns([pl.Series(k, v) for k, v in cols.items()]), n


class TestReplicateGLM:
    def test_jkn_matches_r(self, api_strat):
        df, n = _jkn_columns(api_strat, "stype", "pw")
        strata = api_strat.get_column("stype").to_list()
        from collections import Counter

        n_h = Counter(strata)
        rscales = tuple((n_h[h] - 1.0) / n_h[h] for h in strata)
        rw = RepWeights(method="jackknife", prefix="jw", n_reps=n, rscales=rscales)
        s = Sample(df, Design(stratum="stype", wgt="pw", rep_wgts=rw))
        m = s.glm.fit(y="api00", x=["ell", "meals"])
        got = [c.se for c in m.fitted.coefs]
        np.testing.assert_allclose(
            got, [9.01172834, 0.40647424, 0.28786459], rtol=1e-4
        )

    def test_replicate_variance_differs_from_taylor(self, api_strat):
        """Round 8 R7: a replicate design previously produced SEs identical
        to the plain weighted (Taylor) design."""
        df, n = _jkn_columns(api_strat, "stype", "pw")
        strata = api_strat.get_column("stype").to_list()
        from collections import Counter

        n_h = Counter(strata)
        rscales = tuple((n_h[h] - 1.0) / n_h[h] for h in strata)
        rw = RepWeights(method="jackknife", prefix="jw", n_reps=n, rscales=rscales)
        s_rep = Sample(df, Design(stratum="stype", wgt="pw", rep_wgts=rw))
        s_tay = Sample(api_strat, Design(stratum="stype", wgt="pw"))
        se_rep = [c.se for c in s_rep.glm.fit(y="api00", x=["ell", "meals"]).fitted.coefs]
        se_tay = [c.se for c in s_tay.glm.fit(y="api00", x=["ell", "meals"]).fitted.coefs]
        assert not np.allclose(se_rep, se_tay, rtol=1e-6)

    def test_bootstrap_combination_self_consistent(self):
        """V = sum_r c_r (b_r - mean)(b_r - mean)' with c_r = 1/R."""
        rng = np.random.default_rng(5)
        n = 60
        df = pl.DataFrame(
            {
                "w": rng.uniform(1, 3, n),
                "x": rng.normal(size=n),
                "y": rng.normal(size=n),
            }
        )
        reps = {}
        for r in range(4):
            m = rng.poisson(1.0, n).astype(float)
            reps[f"bw{r + 1}"] = df.get_column("w").to_numpy() * m
        dfr = df.with_columns([pl.Series(k, v) for k, v in reps.items()])
        rw = RepWeights(method="bootstrap", prefix="bw", n_reps=4)
        s = Sample(dfr, Design(wgt="w", rep_wgts=rw))
        fit = s.glm.fit(y="y", x=["x"])

        # Rebuild the expected covariance from per-replicate refits
        betas = []
        for r in range(4):
            sr = Sample(dfr, Design(wgt=f"bw{r + 1}"))
            betas.append([c.est for c in sr.glm.fit(y="y", x=["x"]).fitted.coefs])
        B = np.array(betas)
        Bc = B - B.mean(axis=0)
        V = (Bc * (1.0 / 4.0)).T @ Bc
        np.testing.assert_allclose(
            [c.se for c in fit.fitted.coefs], np.sqrt(np.diag(V)), rtol=1e-8
        )


# =============================================================================
# R8 — FPC applied in the GLM sandwich
# =============================================================================


class TestGLMFpc:
    def test_gaussian_fpc_matches_r(self, api_strat):
        s = Sample(api_strat, Design(stratum="stype", wgt="pw", pop_size="fpc"))
        m = s.glm.fit(y="api00", x=["ell", "meals"])
        got = [c.se for c in m.fitted.coefs]
        np.testing.assert_allclose(got, [8.75949495, 0.38791652, 0.27576549], rtol=RTOL_SE)

    def test_binomial_fpc_matches_r(self, api_binary):
        s = Sample(api_binary, Design(stratum="stype", wgt="pw", pop_size="fpc"))
        m = s.glm.fit(y="y_bin", x=["ell", "meals"], family="binomial")
        got = [c.se for c in m.fitted.coefs]
        np.testing.assert_allclose(got, [0.57895049, 0.01493313, 0.01296360], rtol=1e-5)

    def test_fpc_changes_ses(self, api_strat):
        """Round 8 R8: pop_size was silently ignored (identical SEs)."""
        s_f = Sample(api_strat, Design(stratum="stype", wgt="pw", pop_size="fpc"))
        s_n = Sample(api_strat, Design(stratum="stype", wgt="pw"))
        se_f = [c.se for c in s_f.glm.fit(y="api00", x=["ell", "meals"]).fitted.coefs]
        se_n = [c.se for c in s_n.glm.fit(y="api00", x=["ell", "meals"]).fitted.coefs]
        assert all(f < n for f, n in zip(se_f, se_n))


# =============================================================================
# R9 — absent Cat reference level raises
# =============================================================================


class TestCatRefValidation:
    def test_absent_ref_raises(self, api_strat):
        s = Sample(api_strat, Design(wgt="pw"))
        with pytest.raises(ModelError, match="[Rr]eference"):
            s.glm.fit(y="api00", x=[Cat("stype", ref="Z")])

    def test_dtype_mismatch_ref_raises(self, api_strat):
        data = api_strat.with_columns(
            pl.col("stype").replace_strict({"E": 1, "M": 2, "H": 3}).alias("stype_num")
        )
        s = Sample(data, Design(wgt="pw"))
        with pytest.raises(ModelError, match="[Rr]eference"):
            s.glm.fit(y="api00", x=[Cat("stype_num", ref="1")])  # str vs int

    def test_present_ref_works(self, api_strat):
        s = Sample(api_strat, Design(wgt="pw"))
        m = s.glm.fit(y="api00", x=[Cat("stype", ref="M")])
        terms = [c.term for c in m.fitted.coefs]
        assert "stype_M" not in terms  # M is the reference


# =============================================================================
# R10 — Cat levels enumerated in-domain under where=
# =============================================================================


def test_cat_levels_enumerated_in_domain(api_strat):
    """A level existing only outside the domain must not create a phantom
    all-zero dummy (est 0, se 0) or inflate df."""
    data = api_strat.with_columns(
        pl.when(pl.col("awards") == "No")
        .then(pl.lit("Z"))
        .otherwise(pl.col("stype"))
        .alias("stype2")
    )
    s = Sample(data, Design(wgt="pw"))
    m = s.glm.fit(
        y="api00",
        x=["ell", Cat("stype2")],
        where=svy.col("awards") == "Yes",
    )
    terms = [c.term for c in m.fitted.coefs]
    assert not any(t.endswith("_Z") for t in terms)


# =============================================================================
# R11 — response validated on in-domain rows only
# =============================================================================


def test_response_validated_in_domain(api_binary):
    """Rows excluded by where= must not fail binomial 0/1 validation."""
    data = api_binary.with_columns(
        pl.when(pl.col("awards") == "No")
        .then(pl.lit(2))
        .otherwise(pl.col("y_bin"))
        .alias("y_mixed")
    )
    s = Sample(data, Design(wgt="pw"))
    # y == 2 exists only out-of-domain; R fits this fine
    m = s.glm.fit(
        y="y_mixed",
        x=["ell", "meals"],
        family="binomial",
        where=svy.col("awards") == "Yes",
    )
    assert m.fitted is not None

    # ...but in-domain invalid y still raises
    with pytest.raises(ModelError):
        s.glm.fit(y="y_mixed", x=["ell", "meals"], family="binomial")


# =============================================================================
# R12 — covariate nulls keep-and-zero-weight (design preserved)
# =============================================================================


def test_null_covariate_matches_r(self=None):
    api = pl.read_csv(DATA_DIR / "apistrat.csv")
    api2 = api.with_columns(
        pl.when(pl.int_range(pl.len()) < 5).then(None).otherwise(pl.col("ell")).alias("ell")
    )
    s = Sample(api2, Design(stratum="stype", wgt="pw"))
    m = s.glm.fit(y="api00", x=["ell", "meals"])
    got_c = [c.est for c in m.fitted.coefs]
    got_s = [c.se for c in m.fitted.coefs]
    np.testing.assert_allclose(got_c, [822.39263891, -0.58973797, -3.04680103], rtol=1e-6)
    np.testing.assert_allclose(got_s, [8.97658993, 0.41001108, 0.28108227], rtol=1e-6)


# =============================================================================
# R13 — n/df counted over contributing rows
# =============================================================================


def test_unweighted_null_y_counts(api_strat=None):
    rng = np.random.default_rng(1)
    n = 100
    df = pl.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n)})
    df = df.with_columns(
        pl.when(pl.int_range(pl.len()) < 10).then(None).otherwise(pl.col("y")).alias("y")
    )
    s = Sample(df)  # unweighted design
    m = s.glm.fit(y="y", x=["x"])
    st = m.fitted.stats
    assert st.n == 90
    # df = (n_in - 1) - (k - 1) = 89 - 1 = 88
    assert m.fitted.coefs[0].wald.df == 88
