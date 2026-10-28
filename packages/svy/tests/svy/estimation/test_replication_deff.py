# tests/svy/estimation/test_replication_deff.py
"""deff= on method="replication": the replicate variance over the SRS reference.

Reference values: R survey 4.5 (golden/replication_deff_r.R) on svrepdesign
(type="other") with svy's replicate weights and per-replicate coefficients as
rscales. Mean, total and proportions are svymean/svytotal with deff=TRUE
("wor") and deff="replace" ("wr"); where= is subset() and by= svyby. Ratio,
cov and corr are written out in R: R's replicate svyratio deffs a linearized
total rather than the ratio it reports, and svyvar has no deff.
"""

from __future__ import annotations

import json

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import svy

from svy.core.repwgts import BootstrapWgts, BrrWgts, JackknifeWgts
from svy.errors import EstimationError


HERE = Path(__file__).parent
DATA = HERE.parents[1] / "test_data" / "categorical_rep_30092026.csv"
R = json.loads((HERE / "golden" / "replication_deff_r.json").read_text())
REL = 1e-8

REPS = {
    "brr": BrrWgts(prefix="brr_", n_reps=8, df=6),
    "fay": BrrWgts(prefix="fay_", n_reps=8, fay_coef=0.3, df=6),
    "jkn": JackknifeWgts(prefix="jkn_", n_reps=12, kind="jkn", stratum="st", psu="psu", df=6),
    "jk1": JackknifeWgts(prefix="jk1_", n_reps=12, kind="jk1", df=11),
    "bs": BootstrapWgts(prefix="bs_", n_reps=50, df=49),
}
W = svy.col("dom") == 1
REP = {"method": "replication"}

CASES = {
    "mean": lambda e, d: e.mean("y", deff=d, **REP),
    "mean_where": lambda e, d: e.mean("y", where=W, deff=d, **REP),
    "mean_by": lambda e, d: e.mean("y", by="region", deff=d, **REP),
    "total": lambda e, d: e.total("y", deff=d, **REP),
    "total_by": lambda e, d: e.total("y", by="region", deff=d, **REP),
    "prop": lambda e, d: e.prop("a", deff=d, **REP),
    "prop_where": lambda e, d: e.prop("a", where=W, deff=d, **REP),
    "prop_by": lambda e, d: e.prop("a", by="region", deff=d, **REP),
    "factor_total": lambda e, d: e.total("a", as_factor=True, deff=d, **REP),
    "ratio": lambda e, d: e.ratio("y", "y2", deff=d, **REP),
    "ratio_where": lambda e, d: e.ratio("y", "y2", where=W, deff=d, **REP),
    "ratio_by": lambda e, d: e.ratio("y", "y2", by="region", deff=d, **REP),
    "cov": lambda e, d: e.cov(("y", "y2"), deff=d, **REP),
    "corr": lambda e, d: e.corr(("y", "y2"), deff=d, **REP),
    "corr_by": lambda e, d: e.corr(("y", "y2"), by="region", deff=d, **REP),
}


@pytest.fixture(scope="module")
def frame() -> pl.DataFrame:
    return pl.read_csv(DATA)


def _sample(frame: pl.DataFrame, rep: str) -> svy.Sample:
    return svy.Sample(frame, svy.Design(stratum="st", psu="psu", wgt="w", rep_wgts=REPS[rep]))


def _want(case: dict | list, key: str) -> np.ndarray:
    parts = case if isinstance(case, list) else [case]
    return np.hstack([np.ravel(p[key]) for p in parts]).astype(float)


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("ref", ["wor", "wr"])
@pytest.mark.parametrize("rep", list(REPS))
def test_matches_r(frame, rep, ref, case):
    est = CASES[case](_sample(frame, rep).estimation, ref)
    rows = est.to_dicts()
    want = R[rep][ref][case]
    for key in ("est", "se", "deff"):
        got = np.array([r[key] for r in rows], dtype=float)
        np.testing.assert_allclose(got, _want(want, key), rtol=REL, atol=0, err_msg=key)
    assert est.deff_ref == ref
    if isinstance(want, dict) and "level" in want:
        assert [r["by_level"][0] for r in rows] == want["level"]


def _rescaled(frame: pl.DataFrame) -> pl.DataFrame:
    """Weights normalized to sum to n, replicate weights by the same factor."""
    f = frame.height / frame["w"].sum()
    cols = ["w", *[c for c in frame.columns if c.startswith("jkn_")]]
    return frame.with_columns([pl.col(c) * f for c in cols])


@pytest.mark.parametrize("case", ["mean", "prop", "ratio"])
def test_wor_raises_under_rescaled_weights(frame, case):
    est = _sample(_rescaled(frame), "jkn").estimation
    with pytest.raises(EstimationError) as exc:
        CASES[case](est, "wor")
    message = str(exc.value)
    assert "DEFF_NOT_COMPUTABLE" in message
    assert "rescaled" in message
    assert "deff='wr'" in message


@pytest.mark.parametrize("case", ["mean", "mean_by", "prop", "ratio", "corr"])
def test_wr_is_invariant_to_rescaled_weights(frame, case):
    got = CASES[case](_sample(_rescaled(frame), "jkn").estimation, "wr").to_dicts()
    want = R["jkn"]["wr"][case]
    np.testing.assert_allclose([r["deff"] for r in got], _want(want, "deff"), rtol=REL, atol=0)
