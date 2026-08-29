# tests/svy/weighting/test_calibration_variance.py
"""Calibration-aware Taylor variance.

svy's replication path re-adjusts every replicate column, so its SEs already
reflect a calibrating adjustment. The Taylor path has no such record: it treats
adjusted weights as fixed and so estimates the variance of a different
statistic. R closes this in ``svyrecvar`` by centering the linearized scores
within the cells that were pinned::

    e_i = w*_i (z_i - z_c)      z_c = sum_{i in c} w_i z_i / sum_{i in c} w_i

with the cell means taken under the PREVIOUS weights.

This module is written before that sweep exists, and it is deliberately split:

* Unmarked tests pass today and must keep passing. They pin the cases where
  centering is absent by design (provenance-only adjustments) or provably does
  nothing, so they guard against a future sweep being applied too widely.
* ``xfail(strict=True)`` tests are the specification of the missing behavior.
  Strict means an unexpected pass fails the suite, so the moment any of them
  starts working we find out. Implementing the sweep removes markers one at a
  time.

R-parity values: survey 4.5, ``data(api)``, apiclus1 with
``svydesign(id=~dnum, weights=~pw, fpc=~fpc)``.
"""

from pathlib import Path

import polars as pl
import pytest

from numpy.testing import assert_allclose

from svy import Design, Sample


DATA_DIR = Path(__file__).resolve().parents[2] / "test_data"


# --- survey 4.5 on apiclus1 ------------------------------------------------
R_BASELINE_MEAN_SE = 23.542240693781
R_PS_CELLS_MEAN_EST = 642.310788211584
R_PS_CELLS_MEAN_SE = 23.9204864450905
R_PS_TOTAL_MEAN_SE = 23.542240693781  # identical to baseline: the no-op
# rake(..., control=list(epsilon=1e-12)): R's DEFAULT epsilon=1 is an absolute
# margin tolerance of one unit in 6194, which stops early and yields
# 23.7458597039106. Both implementations agree to 12 digits once actually
# converged, so the test pins the converged value and matches tol.
R_RAKE_MEAN_SE = 23.745841047537
R_STD_EST = [605.662913034119, 646.303126782460]
R_STD_SE = [31.9154418732174, 24.9249948644004]
R_GREG_MEAN_EST = 666.717736250273
R_GREG_MEAN_SE = 3.29588091215716

# calibrate(d, ~api99, c(`(Intercept)`=6194, api99=3914069))
GREG_CONTROLS = {"one": 6194.0, "api99": 3914069.0}

STYPE_POP = {"E": 4421.0, "H": 755.0, "M": 1018.0}
SCHWIDE_POP = {"No": 1000.0, "Yes": 5194.0}
POP_TOTAL = 6194.0


@pytest.fixture
def apiclus1():
    df = pl.read_csv(DATA_DIR / "apiclus1.csv", null_values=["NA"])
    return df.with_columns(
        [(pl.col("stype") == lv).cast(pl.Float64).alias(f"is_{lv}") for lv in ("E", "H", "M")]
        + [pl.lit(1.0).alias("one")]
    )


@pytest.fixture
def design(apiclus1):
    def build():
        return Sample(apiclus1, Design(wgt="pw", psu="dnum", pop_size="fpc"))

    return build


def _mean_se(sample, y="api00"):
    return sample.estimation.mean(y).to_polars()["se"][0]


def _total_se(sample, y):
    return sample.estimation.total(y).to_polars()["se"][0]


def _refit(sample):
    """Rebuild a Sample from the adjusted weights, discarding any provenance.

    Any adjustment whose variance treatment is "weights fixed" must give the
    same SE either way; that equivalence is what the provenance-only tests pin.
    """
    return Sample(sample.data, Design(wgt=sample.design.wgt, psu="dnum", pop_size="fpc"))


# ===========================================================================
# Passes today, must keep passing
# ===========================================================================


def test_baseline_matches_r(design):
    """No adjustment: the Taylor path is already right, and stays right."""
    assert_allclose(_mean_se(design()), R_BASELINE_MEAN_SE, rtol=1e-10)


def test_poststratify_leaves_the_point_estimate_alone(design):
    """Centering is a variance operation; it must never move an estimate."""
    ps = design().weighting.poststratify(STYPE_POP, cells="stype")
    got = ps.estimation.mean("api00").to_polars()["est"][0]
    assert_allclose(got, R_PS_CELLS_MEAN_EST, rtol=1e-12)


def test_poststratify_to_grand_total_does_not_change_a_mean_se(design):
    """A uniform rescale is a provable no-op for a mean.

    With one cell the adjustment factor is constant, so it cancels out of the
    weighted cell mean and the centred scores equal the uncentred ones. R agrees
    exactly: postStratify to a known population size returns the baseline SE.
    """
    ps = design().weighting.poststratify(POP_TOTAL)
    assert_allclose(_mean_se(ps), R_PS_TOTAL_MEAN_SE, rtol=1e-10)
    assert_allclose(_mean_se(ps), _mean_se(design()), rtol=1e-12)


def test_normalize_is_variance_transparent(design):
    """normalize's targets are conveniences, not population constraints.

    It is recorded for provenance only, so its SE must equal that of a design
    rebuilt from the same weights with no record at all.
    """
    norm = design().weighting.normalize(POP_TOTAL)
    assert_allclose(_mean_se(norm), _mean_se(_refit(norm)), rtol=1e-12)


def test_adjust_is_variance_transparent(design, apiclus1):
    """Non-response adjustment gets no variance record, matching R."""
    df = apiclus1.with_columns(
        pl.when(pl.col("snum") % 5 == 0).then(pl.lit("nr")).otherwise(pl.lit("rr")).alias("status")
    )
    s = Sample(df, Design(wgt="pw", psu="dnum", pop_size="fpc"))
    adj = s.weighting.adjust("status", "stype", respondents_only=False)
    assert_allclose(_mean_se(adj), _mean_se(_refit(adj)), rtol=1e-12)


def test_trim_is_variance_transparent(design):
    """Trimming breaks the constraints, so it must not claim a calibration."""
    tr = design().weighting.trim(upper=100.0)
    assert_allclose(_mean_se(tr), _mean_se(_refit(tr)), rtol=1e-12)


def test_standardize_matches_r_point_estimates(design):
    std = design().weighting.standardize("stype", shares=STYPE_POP, by="sch.wide")
    got = std.estimation.mean("api00", by="sch.wide").to_polars().sort("sch.wide")
    assert_allclose(got["est"].to_numpy(), R_STD_EST, rtol=1e-12)


def test_greg_leaves_the_point_estimate_alone(design):
    """A continuous auxiliary shifts the estimate a long way -- correctly."""
    cal = design().weighting.calibrate(controls=GREG_CONTROLS)
    got = cal.estimation.mean("api00").to_polars()["est"][0]
    assert_allclose(got, R_GREG_MEAN_EST, rtol=1e-12)


# ===========================================================================
# The specification of the missing sweep
# ===========================================================================


def test_poststratified_margin_total_has_no_sampling_error(design):
    """The decisive case: a pinned margin has no variability left.

    R gives exactly 0 for all three levels. Without the record svy reports
    1209 / 253 / 202 -- the variance of a quantity the design fixed by fiat.
    """
    ps = design().weighting.poststratify(STYPE_POP, cells="stype")
    for lv in ("E", "H", "M"):
        assert _total_se(ps, f"is_{lv}") == pytest.approx(0.0, abs=1e-6)


def test_poststratify_mean_se_matches_r(design):
    ps = design().weighting.poststratify(STYPE_POP, cells="stype")
    assert_allclose(_mean_se(ps), R_PS_CELLS_MEAN_SE, rtol=1e-9)


def test_known_population_size_has_no_sampling_error(design):
    """poststratify(controls=<scalar>) declares the population size known.

    The mean SE is untouched (see the no-op test above) but the total is now a
    constant, so its SE is 0. Same adjustment, opposite answers by estimand.
    """
    ps = design().weighting.poststratify(POP_TOTAL)
    assert _total_se(ps, "one") == pytest.approx(0.0, abs=1e-6)


def test_rake_mean_se_matches_r(design):
    """Exercises the per-margin sweep: R alternates it over margins, 10 times."""
    rk = design().weighting.rake(
        controls={"stype": STYPE_POP, "sch.wide": SCHWIDE_POP}, tol=1e-12, max_iter=200
    )
    assert_allclose(_mean_se(rk), R_RAKE_MEAN_SE, rtol=1e-9)


def test_standardize_se_matches_r(design):
    std = design().weighting.standardize("stype", shares=STYPE_POP, by="sch.wide")
    got = std.estimation.mean("api00", by="sch.wide").to_polars().sort("sch.wide")
    assert_allclose(got["se"].to_numpy(), R_STD_SE, rtol=1e-6)


def test_shares_pin_composition_but_not_the_total(design):
    """shares and controls differ in WHAT they pin, so they differ in variance.

    There is no R analog -- postStratify has no shares form -- so this is svy's
    own semantics: shares remove k-1 constraints (composition), controls remove
    k (composition and level). A cell proportion therefore has no sampling error
    while the population size still does.
    """
    ps = design().weighting.poststratify(shares=STYPE_POP, cells="stype")
    assert _mean_se(ps, "is_E") == pytest.approx(0.0, abs=1e-9)
    assert _total_se(ps, "one") > 1.0


def test_calibrated_auxiliary_total_has_no_sampling_error(design):
    """GREG pins a continuous total as firmly as poststratification pins a cell.

    The calibration hits the target exactly, so the total has no sampling
    variability. Without the sweep svy reports ~880,533.
    """
    cal = design().weighting.calibrate(controls=GREG_CONTROLS)
    assert _total_se(cal, "api99") == pytest.approx(0.0, abs=1e-3)


def test_greg_mean_se_matches_r(design):
    """The widest gap of any case: 21.86 against R's 3.30, a 6.6x overstatement.

    This branch of R's sweep is a WLS residual (`qr.resid`), not a cell mean, so
    it needs the recorded aux columns rather than cell codes -- the reason the
    record carries `aux` at all.
    """
    cal = design().weighting.calibrate(controls=GREG_CONTROLS)
    assert_allclose(_mean_se(cal), R_GREG_MEAN_SE, rtol=1e-6)


# ---------------------------------------------------------------------------
# NHANES age standardization -- the case the method exists for
# ---------------------------------------------------------------------------

POPAGE = {"(0,19]": 55901, "(19,39]": 77670, "(39,59]": 72816, "(59,Inf]": 45364}

# svyby(~HI_CHOL, ~race+RIAGENDR, svymean,
#       design=subset(stdes, agecat != "(0,19]"))   -- NCHS databrief 92 fig 1
R_DB92_SE = {
    (1, 1): 0.00831820386264,
    (2, 1): 0.01018283847756,
    (3, 1): 0.01354767807155,
    (4, 1): 0.04227427142396,
    (1, 2): 0.01341863662245,
    (2, 2): 0.00893213403197,
    (3, 2): 0.01895358562333,
    (4, 2): 0.04009110600664,
}


@pytest.fixture
def nhanes():
    df = pl.read_csv(DATA_DIR / "nhanes.csv", null_values=["NA"])
    return Sample(df, Design(wgt="WTMEC2YR", stratum="SDMVSTRA", psu="SDMVPSU"))


def test_nhanes_standardized_se_matches_r(nhanes):
    """The published case: age-standardized cholesterol by race and sex.

    Point estimates already match R to 3.6e-13 (see the standardization tests);
    these SEs are out by roughly -11% to +14% in both directions, which is the
    signature of a missing sweep rather than a scale error.
    """
    from svy import col as _col

    std = nhanes.weighting.standardize(
        "agecat",
        shares=POPAGE,
        by=["race", "RIAGENDR"],
        where=_col("HI_CHOL").is_not_null(),
    )
    got = (
        std.estimation.mean(
            "HI_CHOL",
            by=["race", "RIAGENDR"],
            where=_col("agecat") != "(0,19]",
            drop_nulls=True,
        )
        .to_polars()
        .sort(["race", "RIAGENDR"])
    )
    for row in got.iter_rows(named=True):
        key = (int(row["race"]), int(row["RIAGENDR"]))
        assert_allclose(row["se"], R_DB92_SE[key], rtol=1e-6)
