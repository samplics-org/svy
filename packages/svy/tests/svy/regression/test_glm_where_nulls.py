# tests/svy/regression/test_glm_where_nulls.py
"""
GLM with drop_nulls=False ignores nulls outside the `where` domain, as R's
svyglm on subset() does. See tests/svy/estimation/test_where_nulls.py for the
dataset.

Reference values from R survey 4.5:

    d <- read.csv("tests/test_data/where_nulls_20260929.csv")
    d$g <- NULL   # svyglm.svyrep.design evaluates a local `g` inside the data
    des <- svydesign(id = ~psu, strata = ~stratum, weights = ~wgt, data = d)
    s <- subset(des, dom == 1)
    ctl <- glm.control(epsilon = 1e-12)
    svyglm(y ~ z, s, control = ctl)
    svyglm(ybin ~ z, s, family = quasibinomial(), control = ctl)

    rw <- read.csv("tests/test_data/where_nulls_jkn_20260929.csv")
    R <- as.matrix(rw[, paste0("rep", 1:12)])
    rd <- svrepdesign(data = d, repweights = R, weights = ~wgt, type = "JKn",
                      scale = 1, rscales = rep(0.75, 12), combined.weights = TRUE)
    # the same two fits on subset(rd, dom == 1)
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from svy import Cat, Design, RepWeights, Sample


BASE_DIR = Path(__file__).parents[2]
REL = 1e-6
W = pl.col("dom") == 1

# coefficient -> (estimate, Taylor SE, JKn SE)
GAUSSIAN = [
    (11.87166369862975, 3.560996512841014, 3.69520336829290),
    (2.03240861579416, 0.269722237199704, 0.28300060158381),
]
LOGIT = [
    (-4.598847049438832, 2.021564422967844, 2.309749977785613),
    (0.404181215774666, 0.156828243198137, 0.180642542929127),
]


@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    return pl.read_csv(BASE_DIR / "test_data" / "where_nulls_20260929.csv")


def _taylor(df: pl.DataFrame) -> Sample:
    return Sample(df, Design(stratum="stratum", psu="psu", wgt="wgt"))


def _jkn(df: pl.DataFrame) -> Sample:
    rw = pl.read_csv(BASE_DIR / "test_data" / "where_nulls_jkn_20260929.csv")
    df = df.join(rw, on=["psu", "unit"], how="left", maintain_order="left")
    rep = RepWeights(
        method="jackknife", prefix="rep", n_reps=12, kind="jkn", stratum="stratum", psu="psu"
    )
    return Sample(df, Design(stratum="stratum", psu="psu", wgt="wgt", rep_wgts=rep))


@pytest.fixture(scope="module")
def samples(data) -> dict[str, Sample]:
    return {"taylor": _taylor(data), "jkn": _jkn(data)}


@pytest.mark.parametrize("design,k", [("taylor", 1), ("jkn", 2)])
@pytest.mark.parametrize(
    "y,family,ref",
    [("y", "gaussian", GAUSSIAN), ("ybin", "binomial", LOGIT)],
    ids=["gauss", "logit"],
)
def test_matches_r(samples, design, k, y, family, ref):
    fit = samples[design].glm.fit(y=y, x=["z"], family=family, where=W, drop_nulls=False)
    for c, r in zip(fit.fitted.coefs, ref):
        assert c.est == pytest.approx(r[0], rel=REL)
        assert c.se == pytest.approx(r[k], rel=REL)


@pytest.mark.parametrize("design", ["taylor", "jkn"])
def test_equals_drop_nulls_path(samples, design):
    a = samples[design].glm.fit(y="y", x=["z", Cat("cat")], where=W, drop_nulls=False)
    b = samples[design].glm.fit(y="y", x=["z", Cat("cat")], where=W, drop_nulls=True)
    assert [(c.est, c.se) for c in a.fitted.coefs] == pytest.approx(
        [(c.est, c.se) for c in b.fitted.coefs], rel=1e-12
    )


def test_null_y_inside_domain_raises(data):
    df = data.with_columns(
        pl.when(W & (pl.col("unit") == 2)).then(None).otherwise(pl.col("y")).alias("y")
    )
    with pytest.raises(ValueError, match="inside the `where` domain.*y \\(NULL\\)"):
        _taylor(df).glm.fit(y="y", x=["z"], where=W, drop_nulls=False)


def test_null_covariate_inside_domain_raises(data):
    df = data.with_columns(
        pl.when(W & (pl.col("unit") == 2)).then(None).otherwise(pl.col("z")).alias("z")
    )
    with pytest.raises(ValueError, match="inside the `where` domain.*z \\(NULL\\)"):
        _taylor(df).glm.fit(y="y", x=["z"], where=W, drop_nulls=False)


def test_where_column_also_a_covariate(samples):
    """A covariate named in `where` is still analysed: out-of-domain nulls are
    ignored, the fit matches the plain domain fit."""
    fit = samples["taylor"].glm.fit(
        y="y", x=["z"], where=W & (pl.col("z") > -1e9), drop_nulls=False
    )
    for c, r in zip(fit.fitted.coefs, GAUSSIAN):
        assert c.est == pytest.approx(r[0], rel=REL)
        assert c.se == pytest.approx(r[1], rel=REL)
