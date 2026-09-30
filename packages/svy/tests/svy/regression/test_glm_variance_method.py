# tests/svy/regression/test_glm_variance_method.py
"""glm.fit(method=): Taylor unless method="replication", as in estimation."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import svy

from svy.core.repwgts import BootstrapWgts
from svy.errors.singleton_errors import SingletonError


def _frame() -> pl.DataFrame:
    rng = np.random.default_rng(930)
    n = 120
    st = np.repeat([1, 2, 3], 40)
    psu = np.repeat(np.arange(1, 13), 10)
    w = rng.uniform(1, 4, n)
    x = rng.normal(size=n)
    df = pl.DataFrame(
        {"st": st, "psu": psu, "w": w, "x": x, "y": 1 + 0.5 * x + rng.normal(size=n)}
    )
    return df.with_columns(pl.Series(f"bs{r + 1}", w * rng.poisson(1.0, n)) for r in range(20))


REP = BootstrapWgts(prefix="bs", n_reps=20, df=19)


def _ses(fit) -> list[float]:
    return [c.se for c in fit.fitted.coefs]


def test_default_is_taylor():
    df = _frame()
    rep = svy.Sample(df, svy.Design(stratum="st", psu="psu", wgt="w", rep_wgts=REP))
    plain = svy.Sample(df, svy.Design(stratum="st", psu="psu", wgt="w"))
    default = _ses(rep.glm.fit("y", x=["x"]))
    assert default == _ses(rep.glm.fit("y", x=["x"], method="taylor"))
    assert default == _ses(plain.glm.fit("y", x=["x"]))
    replicated = rep.glm.fit("y", x=["x"], method="replication")
    assert not np.allclose(default, _ses(replicated))
    # Replicate df (n_reps - 1) less the slope.
    assert replicated.fitted.stats.wald.df_den == 18


# R survey: svyglm(y ~ x, svrepdesign(..., type="other", scale=1/20,
# rscales=1, mse=FALSE, degf=19)), whole sample and subset(des, st == 1).
# Columns: estimate, SE, lower, upper, p-value.
R_REP_FULL = [
    (
        0.977287931883351,
        0.0962684188941410,
        0.775035488849493,
        1.17954037491721,
        7.08033808850257e-09,
    ),
    (
        0.468224072707398,
        0.0977476423454725,
        0.262863896522197,
        0.67358424889260,
        1.46520580536670e-04,
    ),
]
R_REP_ST1 = [
    (
        0.801100009685407,
        0.135556231878502,
        0.516306934439838,
        1.085893084930976,
        1.35658118063807e-05,
    ),
    (
        0.515513886329908,
        0.180744022668887,
        0.135784785463016,
        0.895242987196799,
        1.05820661968515e-02,
    ),
]


def _rep_sample(rep: BootstrapWgts) -> svy.Sample:
    return svy.Sample(_frame(), svy.Design(stratum="st", psu="psu", wgt="w", rep_wgts=rep))


def _check_r(fit, golden) -> None:
    for c, (est, se, lci, uci, p) in zip(fit.fitted.coefs, golden, strict=True):
        np.testing.assert_allclose(
            [c.est, c.se, c.lci, c.uci, c.wald.p_value], [est, se, lci, uci, p], rtol=1e-9
        )


def test_replication_df_without_recorded_df_is_n_reps_minus_one():
    # Not the Taylor df of 12 PSUs in 3 strata (9 - 1 = 8).
    s = _rep_sample(BootstrapWgts(prefix="bs", n_reps=20))
    fit = s.glm.fit("y", x=["x"], method="replication")
    assert fit.fitted.stats.wald.df_den == 18
    assert all(c.wald.df == 18 for c in fit.fitted.coefs)
    assert s.estimation.mean("y", method="replication").estimates[0].df == 19
    _check_r(fit, R_REP_FULL)


def test_replication_df_does_not_shrink_on_a_domain():
    # Stratum 1 has 4 PSUs: the Taylor df would be 3 - 1 = 2.
    s = _rep_sample(BootstrapWgts(prefix="bs", n_reps=20))
    fit = s.glm.fit("y", x=["x"], method="replication", where=svy.col("st") == 1)
    assert fit.fitted.stats.wald.df_den == 18
    _check_r(fit, R_REP_ST1)


def test_replication_df_uses_the_recorded_df():
    s = _rep_sample(BootstrapWgts(prefix="bs", n_reps=20, df=12))
    assert s.glm.fit("y", x=["x"], method="replication").fitted.stats.wald.df_den == 11


def test_replication_without_replicate_weights_raises():
    s = svy.Sample(_frame(), svy.Design(stratum="st", psu="psu", wgt="w"))
    with pytest.raises(ValueError, match="Replication requires rep_wgts"):
        s.glm.fit("y", x=["x"], method="replication")


def test_taylor_on_a_replicate_design_needs_the_singleton_rule():
    # Stratum 3 keeps PSU 9 only.
    df = _frame().filter(~pl.col("psu").is_in([10, 11, 12]))
    s = svy.Sample(df, svy.Design(stratum="st", psu="psu", wgt="w", rep_wgts=REP))
    with pytest.raises(SingletonError) as err:
        s.glm.fit("y", x=["x"])
    assert "Pass method='replication'" in err.value.hint
    assert _ses(s.glm.fit("y", x=["x"], method="replication"))[0] > 0
