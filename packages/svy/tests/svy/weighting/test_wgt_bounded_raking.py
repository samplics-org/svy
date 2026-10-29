# tests/svy/weighting/test_wgt_bounded_raking.py
"""``rake(bounds=)`` against R survey 4.5 ``calibrate(..., calfun="raking", bounds=)``.

The ``bounded_raking_*_29102026.csv`` fixtures hold R's weights (15 significant digits, solved to
``epsilon=1e-12``): apiclus1 with ``svydesign(id=~dnum, weights=~pw, fpc=~fpc)`` and
``~stype + sch.wide``; apistrat with ``svydesign(id=~1, strata=~stype, weights=~pw, fpc=~fpc)``
and ``~stype + sch.wide + awards``. A JK1 replicate R fails with its own and with a corrected
derivative is NA; a linear program confirms none of those can meet the margins within the bounds.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from numpy.testing import assert_allclose

from svy import Design, RepWeights, Sample, col
from svy.errors import SvyWarning, WeightingError
from svy.weighting.types import TrimConfig


DATA_DIR = Path(__file__).resolve().parents[2] / "test_data"

CLUS = {"stype": {"E": 4421.0, "H": 755.0, "M": 1018.0}, "sch.wide": {"No": 1000.0, "Yes": 5194.0}}
STRAT = {
    "stype": {"E": 4421.0, "H": 755.0, "M": 1018.0},
    "sch.wide": {"No": 1072.0, "Yes": 5122.0},
    "awards": {"No": 2027.0, "Yes": 4167.0},
}
TIGHT = {"tol": 1e-12, "max_iter": 200}
# The CSV keeps 15 significant digits.
RTOL_W = 1e-10
R_FAILED_REPS = [1, 2, 6, 7, 9, 11, 12, 15]


def _read(name: str) -> pl.DataFrame:
    return pl.read_csv(DATA_DIR / name, null_values=["NA"])


@pytest.fixture(scope="module")
def clus():
    return _read("apiclus1.csv").join(
        _read("bounded_raking_apiclus1_29102026.csv"), on="snum", how="left", maintain_order="left"
    )


@pytest.fixture(scope="module")
def strat():
    return _read("apistrat.csv").join(
        _read("bounded_raking_apistrat_29102026.csv"), on="snum", how="left", maintain_order="left"
    )


def _clus(df: pl.DataFrame, **design) -> Sample:
    return Sample(df, Design(wgt="pw", psu="dnum", pop_size="fpc", **design))


def _strat(df: pl.DataFrame) -> Sample:
    return Sample(df, Design(wgt="pw", stratum="stype", pop_size="fpc"))


def _raked(sample: Sample, controls=CLUS, **kw) -> np.ndarray:
    return sample.weighting.rake(controls=controls, **kw).data["rk_wgt"].to_numpy()


# ---------------------------------------------------------------------------
# Weights and SEs match R
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("bounds", "r_col"), [((0.5, 1.6), "w_a"), ((0.85, 1.8), "w_b")])
def test_weights_match_r(clus, bounds, r_col):
    got = _raked(_clus(clus), bounds=bounds, **TIGHT)
    assert_allclose(got, clus[r_col].to_numpy(), rtol=RTOL_W)


def test_three_margins_match_r(strat):
    got = _raked(_strat(strat), controls=STRAT, bounds=(0.6, 1.06), **TIGHT)
    assert_allclose(got, strat["w_s"].to_numpy(), rtol=RTOL_W)


def test_rows_hold_the_bounds_and_margins_are_met(clus):
    w = _raked(_clus(clus), bounds=(0.5, 1.6), **TIGHT)
    g = w / clus["pw"].to_numpy()
    assert g.min() >= 0.5 - 1e-12 and g.max() <= 1.6 + 1e-12
    for var, targets in CLUS.items():
        for level, target in targets.items():
            assert_allclose(w[(clus[var] == level).to_numpy()].sum(), target, rtol=1e-10)


def test_taylor_se_matches_r(clus):
    # svymean/svytotal(~api00) on R's bounded design, which R records as a GREG calibration.
    out = _clus(clus).weighting.rake(controls=CLUS, bounds=(0.5, 1.6), **TIGHT)
    mean = out.estimation.mean("api00").to_polars()
    total = out.estimation.total("api00").to_polars()
    assert_allclose(mean["est"][0], 641.17075710439531, rtol=1e-12)
    assert_allclose(mean["se"][0], 23.864142153721758, rtol=1e-8)
    assert_allclose(total["est"][0], 3971411.6695046178, rtol=1e-12)
    assert_allclose(total["se"][0], 147814.49650015228, rtol=1e-8)
    rec = out.design.wgt_adjustment
    assert rec.kind == "calibration" and rec.prev_wgt == "pw" and len(rec.aux) == 4


def test_bounds_that_r_cannot_reach_still_converge(strat):
    # The unbounded raking factors (0.809 to 1.078) lie inside (0.8, 1.2), but R's cal.raking
    # zeroes its derivative below log(g) = 0.8 - 1 and fails; the answer is the unbounded rake.
    got = _raked(_strat(strat), controls=STRAT, bounds=(0.8, 1.2), **TIGHT)
    assert_allclose(got, strat["w_free"].to_numpy(), rtol=RTOL_W)


def test_bounds_that_never_bind_give_the_unbounded_rake(clus):
    # Unbounded raking factors span [0.891, 1.854] here.
    bounded = _raked(_clus(clus), bounds=(0.5, 2.0), **TIGHT)
    assert_allclose(bounded, _raked(_clus(clus), **TIGHT), rtol=1e-10)


# ---------------------------------------------------------------------------
# Non-convergence
# ---------------------------------------------------------------------------


def test_unreachable_margins_raise_and_leave_the_sample(clus):
    # A linear program shows no g in (0.5, 1.5) meets these margins.
    s = _clus(clus)
    with pytest.raises(WeightingError) as ei:
        s.weighting.rake(controls=CLUS, bounds=(0.5, 1.5))
    err = ei.value
    assert err.code == "WEIGHTING_NOT_CONVERGED" and err.param == "max_iter"
    assert "Bounded raking did not converge after 100 iterations" in err.detail
    assert err.got["steps"] == 100 and err.expected == {"tol": 1e-4}
    assert "rk_wgt" not in s.data.columns and s.design.wgt == "pw"


def test_warn_keeps_a_last_iterate_within_the_bounds(clus):
    with pytest.warns(SvyWarning, match=r"\[WEIGHTING_NOT_CONVERGED\]"):
        out = _clus(clus).weighting.rake(
            controls=CLUS, bounds=(0.5, 1.5), on_nonconvergence="warn"
        )
    g = out.data["rk_wgt"].to_numpy() / clus["pw"].to_numpy()
    assert g.min() >= 0.5 - 1e-12 and g.max() <= 1.5 + 1e-12
    (found,) = out.warnings.list(code="WEIGHTING_NOT_CONVERGED")
    assert "'rk_wgt' holds the last iterate" in found.detail


# ---------------------------------------------------------------------------
# Replicates
# ---------------------------------------------------------------------------


@pytest.fixture
def clus_jk(clus):
    return _clus(clus, rep_wgts=RepWeights("jackknife", prefix="jk", n_reps=15))


def test_replicates_match_r_where_reachable(clus, clus_jk):
    with pytest.warns(SvyWarning, match=r"\[WEIGHTING_NOT_CONVERGED\]"):
        out = clus_jk.weighting.rake(
            controls=CLUS, bounds=(0.5, 1.6), on_nonconvergence="warn", **TIGHT
        )
    for j in set(range(1, 16)) - set(R_FAILED_REPS):
        assert_allclose(
            out.data[f"rk_wgt{j}"].to_numpy(),
            clus[f"ra{j}"].cast(pl.Float64).to_numpy(),
            rtol=RTOL_W,
        )
    (found,) = out.warnings.list(code="WEIGHTING_NOT_CONVERGED")
    assert found.got["replicates"] == [f"rk_wgt{j}" for j in R_FAILED_REPS]


def test_unreachable_replicates_raise_by_default(clus_jk):
    with pytest.raises(WeightingError) as ei:
        clus_jk.weighting.rake(controls=CLUS, bounds=(0.5, 1.6), **TIGHT)
    assert ei.value.got["replicates"] == [f"rk_wgt{j}" for j in R_FAILED_REPS]
    assert "rk_wgt" not in clus_jk.data.columns


# ---------------------------------------------------------------------------
# Shares, scope, arguments
# ---------------------------------------------------------------------------


def test_shares_rake_like_their_controls(clus):
    total = float(clus["pw"].sum())
    shares = {v: {k: x / 6194.0 for k, x in levels.items()} for v, levels in CLUS.items()}
    as_controls = {v: {k: x * total for k, x in levels.items()} for v, levels in shares.items()}
    got = _clus(clus).weighting.rake(shares=shares, bounds=(0.5, 1.6), **TIGHT)
    want = _raked(_clus(clus), controls=as_controls, bounds=(0.5, 1.6), **TIGHT)
    assert_allclose(got.data["rk_wgt"].to_numpy(), want, rtol=1e-10)


def test_controls_pin_the_total_and_shares_do_not(clus):
    df = clus.with_columns(pl.lit(1.0).alias("one"))
    total = float(clus["pw"].sum())
    shares = {v: {k: x / 6194.0 for k, x in levels.items()} for v, levels in CLUS.items()}
    by_controls = _clus(df).weighting.rake(controls=CLUS, bounds=(0.5, 1.6), **TIGHT)
    by_shares = _clus(df).weighting.rake(shares=shares, bounds=(0.5, 1.6), **TIGHT)
    assert by_controls.estimation.total("one").to_polars()["se"][0] < 1e-6 * 6194
    assert_allclose(by_shares.estimation.total("one").to_polars()["est"][0], total, rtol=1e-10)
    assert by_shares.estimation.total("one").to_polars()["se"][0] > 0.01 * total


def test_where_rakes_only_the_scoped_rows(clus):
    scope = (clus["stype"] != "H").to_numpy()
    controls = {"stype": {"E": 4421.0, "M": 1018.0}, "sch.wide": {"No": 800.0, "Yes": 4639.0}}
    out = _raked(
        _clus(clus), controls=controls, where=col("stype") != "H", bounds=(0.5, 1.6), **TIGHT
    )
    alone = _raked(
        _clus(clus.filter(pl.Series(scope))), controls=controls, bounds=(0.5, 1.6), **TIGHT
    )
    assert_allclose(out[scope], alone, rtol=1e-14)
    assert np.array_equal(out[~scope], clus["pw"].to_numpy()[~scope])


def test_bounds_with_trimming_are_refused(clus):
    s = _clus(clus)
    with pytest.raises(WeightingError) as ei:
        s.weighting.rake(controls=CLUS, bounds=(0.5, 1.6), trimming=TrimConfig(upper=100.0))
    assert ei.value.code == "NOT_SUPPORTED" and ei.value.param == "trimming"
    assert "rk_wgt" not in s.data.columns


@pytest.mark.parametrize(
    ("kw", "param", "code"),
    [
        ({"tol": 0}, "tol", "INVALID_RANGE"),
        ({"tol": "small"}, "tol", "INVALID_TYPE"),
        ({"max_iter": 0}, "max_iter", "INVALID_RANGE"),
        ({"max_iter": 2.5}, "max_iter", "INVALID_TYPE"),
    ],
)
def test_bad_tol_and_max_iter_are_refused(clus, kw, param, code):
    with pytest.raises(WeightingError) as ei:
        _clus(clus).weighting.rake(controls=CLUS, bounds=(0.5, 1.6), **kw)
    assert ei.value.code == code and ei.value.param == param
