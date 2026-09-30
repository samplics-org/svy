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
