# tests/svy/weighting/test_wgt_bounded_calibration.py
"""Bounded linear calibration against R survey 4.5 ``calibrate(..., bounds=)``.

The ``bounded_calibration_*_29102026.csv`` fixtures hold R's weights (15 significant digits),
from ``data(api)``: apiclus1 with ``svydesign(id=~dnum, weights=~pw, fpc=~fpc)`` and
``calibrate(d, ~stype+api99, c(6194, 755, 1018, 3914069), bounds=...)``; apistrat with
``svydesign(id=~1, strata=~stype, weights=~pw, fpc=~fpc)`` and ``~stype + api99:stype - 1``;
apisrs with unit weights.
"""

import math

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from numpy.testing import assert_allclose

from svy import Cat, Cross, Design, RepWeights, Sample, col
from svy.errors import SvyWarning, WeightingError
from svy.weighting.types import TrimConfig


DATA_DIR = Path(__file__).resolve().parents[2] / "test_data"

# R's model matrix for ~stype+api99: intercept, stypeH, stypeM, api99.
CLUS_CONTROLS = {"one": 6194.0, "is_H": 755.0, "is_M": 1018.0, "api99": 3914069.0}
# The CSV keeps 15 significant digits; the Newton solution itself is exact to ~1e-13.
RTOL_W = 1e-10


def _read(name: str) -> pl.DataFrame:
    return pl.read_csv(DATA_DIR / name, null_values=["NA"])


@pytest.fixture(scope="module")
def clus():
    df = _read("apiclus1.csv").with_columns(
        pl.lit(1.0).alias("one"),
        (pl.col("stype") == "H").cast(pl.Float64).alias("is_H"),
        (pl.col("stype") == "M").cast(pl.Float64).alias("is_M"),
    )
    return df.join(
        _read("bounded_calibration_apiclus1_29102026.csv"),
        on="snum",
        how="left",
        maintain_order="left",
    )


def _clus_sample(df: pl.DataFrame, **design) -> Sample:
    return Sample(df, Design(wgt="pw", psu="dnum", pop_size="fpc", **design))


def _calib(sample: Sample, **kw) -> np.ndarray:
    return sample.weighting.calibrate(controls=CLUS_CONTROLS, **kw).data["calib_wgt"].to_numpy()


# ---------------------------------------------------------------------------
# Weights match R
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "r_col"),
    [
        ({"bounds": (0.5, 1.7)}, "w_a"),
        ({"bounds": (0.6, None)}, "w_a2"),
        ({"bounds": (0.5, 1.7), "tol": 0.01}, "w_c"),
    ],
    ids=["two-sided", "lower-only", "tol-0.01"],
)
def test_weights_match_r(clus, kwargs, r_col):
    assert_allclose(_calib(_clus_sample(clus), **kwargs), clus[r_col].to_numpy(), rtol=RTOL_W)


def test_scale_matches_r_variance(clus):
    got = _calib(_clus_sample(clus), bounds=(0.5, 1.7), scale=clus["api99"].to_numpy())
    assert_allclose(got, clus["w_d"].to_numpy(), rtol=RTOL_W)


def test_rows_at_the_bounds_hold_them_and_controls_are_met(clus):
    out = _clus_sample(clus).weighting.calibrate(controls=CLUS_CONTROLS, bounds=(0.5, 1.7))
    g = out.data["calib_wgt"].to_numpy() / clus["pw"].to_numpy()
    assert g.min() >= 0.5 - 1e-12 and g.max() <= 1.7 + 1e-12
    # 10 rows sit on a bound in R's solution.
    assert (
        int(
            np.sum(np.isclose(g, 0.5, rtol=0, atol=1e-12) | np.isclose(g, 1.7, rtol=0, atol=1e-12))
        )
        == 10
    )
    w = out.data["calib_wgt"].to_numpy()
    achieved = [float(np.sum(w * clus[c].to_numpy())) for c in CLUS_CONTROLS]
    assert_allclose(achieved, list(CLUS_CONTROLS.values()), rtol=1e-7)


def test_taylor_se_matches_r(clus):
    # svymean/svytotal(~api00) on R's bounded design: the GREG record needs no change.
    out = _clus_sample(clus).weighting.calibrate(controls=CLUS_CONTROLS, bounds=(0.5, 1.7))
    mean = out.estimation.mean("api00").to_polars()
    total = out.estimation.total("api00").to_polars()
    assert_allclose(mean["est"][0], 665.3633419959532, rtol=1e-12)
    assert_allclose(mean["se"][0], 3.437329188144327, rtol=1e-8)
    assert_allclose(total["est"][0], 4121260.540322939, rtol=1e-12)
    assert_allclose(total["se"][0], 21290.81699136605, rtol=1e-8)


@pytest.fixture(scope="module")
def strat():
    df = _read("apistrat.csv").with_columns(pl.lit(1.0).alias("one"))
    return df.join(
        _read("bounded_calibration_apistrat_29102026.csv"),
        on="snum",
        how="left",
        maintain_order="left",
    )


# apipop: N and sum(api99) by stype.
STRAT_N = {"E": 4421.0, "H": 755.0, "M": 1018.0}
STRAT_X = {"E": 2799206.0, "H": 468895.0, "M": 645968.0}


@pytest.mark.parametrize(("tol", "r_col"), [(1e-7, "w_s"), (0.01, "w_s01")])
def test_stratum_interaction_matches_r(strat, tol, r_col):
    out = Sample(strat, Design(wgt="pw", stratum="stype", pop_size="fpc")).weighting.calibrate(
        controls={
            Cat("stype"): STRAT_N,
            Cross(Cat("stype"), "api99"): {(k, "api99"): v for k, v in STRAT_X.items()},
        },
        bounds=(0.75, 1.3),
        tol=tol,
    )
    assert_allclose(out.data["calib_wgt"].to_numpy(), strat[r_col].to_numpy(), rtol=RTOL_W)


def test_by_stratum_gives_the_joint_solution(strat):
    # The stratum constraints separate, so solving each stratum apart reaches R's joint solution.
    out = Sample(strat, Design(wgt="pw", stratum="stype", pop_size="fpc")).weighting.calibrate(
        controls={k: {"one": STRAT_N[k], "api99": STRAT_X[k]} for k in STRAT_N},
        by="stype",
        bounds=(0.75, 1.3),
    )
    assert_allclose(out.data["calib_wgt"].to_numpy(), strat["w_s"].to_numpy(), rtol=RTOL_W)


def test_unit_weights_are_rescaled_before_the_bounds(clus):
    srs = (
        _read("apisrs.csv")
        .with_columns(
            pl.lit(1.0).alias("one"),
            (pl.col("stype") == "H").cast(pl.Float64).alias("is_H"),
            (pl.col("stype") == "M").cast(pl.Float64).alias("is_M"),
        )
        .join(
            _read("bounded_calibration_apisrs_29102026.csv"),
            on="snum",
            how="left",
            maintain_order="left",
        )
    )
    got = _calib(Sample(srs, Design(wgt="one")), bounds=(0.9, 1.1))
    assert_allclose(got, srs["w_p"].to_numpy(), rtol=RTOL_W)


# ---------------------------------------------------------------------------
# Non-convergence
# ---------------------------------------------------------------------------

# R: "Failed to converge: eps=0.121162529597207 in 51 iterations" for bounds=c(0.7, 1.4).
R_TIGHT_EPS = 0.121162529597207


def test_unreachable_controls_raise_and_leave_the_sample(clus):
    s = _clus_sample(clus)
    with pytest.raises(WeightingError) as ei:
        s.weighting.calibrate(controls=CLUS_CONTROLS, bounds=(0.7, 1.4))
    err = ei.value
    assert err.code == "WEIGHTING_NOT_CONVERGED" and err.param == "max_iter"
    assert "Bounded calibration did not converge after 50 iterations" in err.detail
    assert err.expected == {"tol": 1e-7}
    assert err.got["steps"] == 50
    assert_allclose(err.got["achieved"], R_TIGHT_EPS, rtol=1e-12)
    assert "calib_wgt" not in s.data.columns and s.design.wgt == "pw"


def test_warn_keeps_the_last_iterate_as_r_force(clus):
    with pytest.warns(SvyWarning, match=r"\[WEIGHTING_NOT_CONVERGED\]"):
        out = _clus_sample(clus).weighting.calibrate(
            controls=CLUS_CONTROLS, bounds=(0.7, 1.4), on_nonconvergence="warn"
        )
    assert_allclose(out.data["calib_wgt"].to_numpy(), clus["w_b"].to_numpy(), rtol=RTOL_W)
    (found,) = out.warnings.list(code="WEIGHTING_NOT_CONVERGED")
    assert "'calib_wgt' holds the last iterate" in found.detail
    assert out.design.wgt == "calib_wgt"


def test_ignore_records_without_raising(clus, recwarn):
    out = _clus_sample(clus).weighting.calibrate(
        controls=CLUS_CONTROLS, bounds=(0.7, 1.4), on_nonconvergence="ignore"
    )
    assert not [w for w in recwarn.list if issubclass(w.category, SvyWarning)]
    assert len(out.warnings.list(code="WEIGHTING_NOT_CONVERGED")) == 1


def test_max_iter_caps_the_steps(clus):
    with pytest.raises(WeightingError) as ei:
        _clus_sample(clus).weighting.calibrate(
            controls=CLUS_CONTROLS, bounds=(0.5, 1.7), max_iter=1
        )
    assert ei.value.got["steps"] == 1 and ei.value.got["max_iter"] == 1


def test_failing_domain_is_named():
    df = pl.DataFrame({"d": ["a"] * 4 + ["b"] * 4, "x": [1.0, 2, 3, 4] * 2, "w": [1.0] * 8})
    # Domain b can reach at most 1.4 * 10 = 14 < 15.
    with pytest.raises(WeightingError) as ei:
        Sample(df, Design(wgt="w")).weighting.calibrate(
            controls={"a": {"x": 12.0}, "b": {"x": 15.0}}, by="d", bounds=(0.5, 1.4)
        )
    assert ei.value.got["domains"] == ["b"]
    assert "in domain(s) ['b']" in ei.value.detail
    assert_allclose(ei.value.got["achieved"], 1 / 16, rtol=1e-12)


# ---------------------------------------------------------------------------
# Replicates
# ---------------------------------------------------------------------------


@pytest.fixture
def clus_jk(clus):
    return _clus_sample(clus, rep_wgts=RepWeights("jackknife", prefix="jk", n_reps=15))


def test_replicates_match_r_and_failures_are_kept_on_warn(clus, clus_jk):
    with pytest.warns(SvyWarning, match=r"\[WEIGHTING_NOT_CONVERGED\]"):
        out = clus_jk.weighting.calibrate(
            controls=CLUS_CONTROLS, bounds=(0.5, 1.7), on_nonconvergence="warn"
        )
    got = out.data.select([f"calib_wgt{j}" for j in range(1, 16)]).to_numpy()
    want = clus.select([f"ra{j}" for j in range(1, 16)]).to_numpy()
    assert_allclose(got, want, rtol=RTOL_W)
    (found,) = out.warnings.list(code="WEIGHTING_NOT_CONVERGED")
    assert found.got["replicates"] == ["calib_wgt9", "calib_wgt11", "calib_wgt12"]


def test_failed_replicates_raise_by_default(clus_jk):
    with pytest.raises(WeightingError) as ei:
        clus_jk.weighting.calibrate(controls=CLUS_CONTROLS, bounds=(0.5, 1.7))
    assert ei.value.code == "WEIGHTING_NOT_CONVERGED"
    assert ei.value.got["replicates"] == ["calib_wgt9", "calib_wgt11", "calib_wgt12"]
    # R: the worst of the three failed replicates.
    assert_allclose(ei.value.got["achieved"], 0.183202564400971, rtol=1e-12)
    assert "calib_wgt" not in clus_jk.data.columns


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bounds", [None, (None, None)])
def test_unset_bounds_keep_the_closed_form(clus, bounds):
    base = _calib(_clus_sample(clus))
    assert np.array_equal(_calib(_clus_sample(clus), bounds=bounds), base)


def test_bounds_that_never_bind_reach_the_closed_form(clus):
    # Unbounded g spans [0.419, 1.833] here. At the default tol the first Newton step already
    # stops (its miss is ~1e-10), so solve to the precision being compared.
    assert_allclose(
        _calib(_clus_sample(clus), bounds=(0.4, 1.9), tol=1e-12),
        _calib(_clus_sample(clus)),
        rtol=1e-12,
    )


def test_calibrate_matrix_matches_calibrate(clus):
    s = _clus_sample(clus)
    X = clus.select(list(CLUS_CONTROLS)).to_numpy()
    got = s.weighting.calibrate_matrix(
        aux_vars=X, controls=list(CLUS_CONTROLS.values()), bounds=(0.5, 1.7), weights_only=True
    )
    assert_allclose(got, clus["w_a"].to_numpy(), rtol=RTOL_W)


def test_where_bounds_only_the_scoped_rows(clus):
    scope = clus["stype"] == "E"
    controls = {"one": 4421.0, "api99": 2799206.0}
    out = _clus_sample(clus).weighting.calibrate(
        controls=controls, where=col("stype") == "E", bounds=(0.5, 1.7)
    )
    alone = _clus_sample(clus.filter(scope)).weighting.calibrate(
        controls=controls, bounds=(0.5, 1.7)
    )
    w = out.data["calib_wgt"].to_numpy()
    assert_allclose(w[scope.to_numpy()], alone.data["calib_wgt"].to_numpy(), rtol=1e-14)
    assert np.array_equal(w[~scope.to_numpy()], clus.filter(~scope)["pw"].to_numpy())


# ---------------------------------------------------------------------------
# Argument errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kw", "param", "code"),
    [
        ({"tol": 0}, "tol", "INVALID_RANGE"),
        ({"tol": -1e-3}, "tol", "INVALID_RANGE"),
        ({"tol": math.nan}, "tol", "INVALID_RANGE"),
        ({"tol": math.inf}, "tol", "INVALID_RANGE"),
        ({"tol": "small"}, "tol", "INVALID_TYPE"),
        ({"tol": True}, "tol", "INVALID_TYPE"),
        ({"max_iter": 0}, "max_iter", "INVALID_RANGE"),
        ({"max_iter": 2.5}, "max_iter", "INVALID_TYPE"),
        ({"max_iter": True}, "max_iter", "INVALID_TYPE"),
    ],
)
@pytest.mark.parametrize("bounds", [None, (0.5, 1.7)])
def test_bad_tol_and_max_iter_are_refused(clus, kw, param, code, bounds):
    with pytest.raises(WeightingError) as ei:
        _clus_sample(clus).weighting.calibrate(controls=CLUS_CONTROLS, bounds=bounds, **kw)
    assert ei.value.code == code and ei.value.param == param


def test_bounds_with_trimming_are_refused(clus):
    s = _clus_sample(clus)
    with pytest.raises(WeightingError) as ei:
        s.weighting.calibrate(
            controls=CLUS_CONTROLS, bounds=(0.5, 1.7), trimming=TrimConfig(upper=100.0)
        )
    assert ei.value.code == "NOT_SUPPORTED" and ei.value.param == "trimming"
    assert "bounds= or trimming=" in ei.value.hint
    assert "calib_wgt" not in s.data.columns
