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

from svy import Design, Sample, col
from svy.weighting.types import TrimConfig


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


# ---------------------------------------------------------------------------
# Estimators beyond mean/total that share the variance funnel
# ---------------------------------------------------------------------------

R_PS_MEDIAN_SE = 37.0666231974705


def test_poststratified_proportions_have_no_sampling_error(design):
    """A proportion is the mean of an indicator, so the pinned variable's own
    proportions carry no variability. R gives 0 for all three levels."""
    ps = design().weighting.poststratify(STYPE_POP, cells="stype")
    got = ps.estimation.prop("stype").to_polars()
    for se in got["se"].to_list():
        assert se == pytest.approx(0.0, abs=1e-9)


def test_poststratified_tabulate_has_no_sampling_error(design):
    """Same claim through the categorical path, which reaches the variance
    funnel via score columns rather than the estimation API."""
    ps = design().weighting.poststratify(STYPE_POP, cells="stype")
    tab = ps.categorical.tabulate("stype").to_polars()
    col = next(c for c in tab.columns if c.lower() in ("se", "stderr"))
    for se in tab[col].to_list():
        assert se == pytest.approx(0.0, abs=1e-9)


def test_poststratified_median_se_matches_r(design):
    """Woodruff quantiles invert a proportion's CI, so they inherit the sweep
    through `taylor_variance_apply` with no quantile-specific work."""
    ps = design().weighting.poststratify(STYPE_POP, cells="stype")
    got = ps.estimation.median("api00").to_polars()["se"][0]
    assert_allclose(got, R_PS_MEDIAN_SE, rtol=1e-9)


# ---------------------------------------------------------------------------
# Regression paths
#
# glm and the two-sample t-test build variance from influence functions that
# already carry the bread matrix, and an earlier reading of R took that to mean
# the sweep could not be applied there. It can. `svy.varcoef` calls
# `svyrecvar(estfun %*% Ainv, ..., postStrata=)` -- R centres AFTER the bread --
# and every sweep branch is a left multiplication by an operator fixed by the
# weights, cells and aux alone, applied identically to each column. It therefore
# commutes with any right multiplication: S(E A) = (S E) A. glm centres its
# estimating functions and sandwiches after; wols centres `influence * w`, which
# IS R's matrix. Both routes were checked against survey 4.5 and agree to 1.6e-14.
# ---------------------------------------------------------------------------

# svyglm(api00 ~ ell + meals, design=<adjusted>) on apiclus1
R_GLM_PS_COEF = [808.90531325861, -0.423451436759323, -3.13332460445148]
R_GLM_PS_SE = [19.638215620182, 0.301676353781227, 0.332436938999844]
R_GLM_RAKE_COEF = [808.229191981178, -0.397809108649268, -3.14123469449935]
R_GLM_RAKE_SE = [19.136719810137, 0.296963780915734, 0.330481840079923]
R_GLM_GREG_COEF = [825.071694299282, -0.527397095251339, -3.1925326169424]
R_GLM_GREG_SE = [13.8069914557681, 0.345353806000647, 0.308690697513365]

X_COLS = ["ell", "meals"]


def _glm(sample):
    t = sample.glm.fit(y="api00", x=X_COLS).to_polars()
    return t["estimate"].to_numpy(), t["std_err"].to_numpy()


def test_glm_poststratified_matches_r(design):
    """The cells sweep, reaching variance through glm's sandwich meat."""
    ps = design().weighting.poststratify(STYPE_POP, cells="stype")
    coef, se = _glm(ps)
    assert_allclose(coef, R_GLM_PS_COEF, rtol=1e-10)
    assert_allclose(se, R_GLM_PS_SE, rtol=1e-9)


def test_glm_raked_matches_r(design):
    """Per-margin sweep, alternated ten times, under the bread."""
    rk = design().weighting.rake(
        controls={"stype": STYPE_POP, "sch.wide": SCHWIDE_POP}, tol=1e-12, max_iter=200
    )
    coef, se = _glm(rk)
    assert_allclose(coef, R_GLM_RAKE_COEF, rtol=1e-10)
    assert_allclose(se, R_GLM_RAKE_SE, rtol=1e-9)


def test_glm_calibrated_matches_r(design):
    """GREG: the WLS residual sweep, fitted with the PREVIOUS weights."""
    cal = design().weighting.calibrate(controls=GREG_CONTROLS)
    coef, se = _glm(cal)
    assert_allclose(coef, R_GLM_GREG_COEF, rtol=1e-10)
    assert_allclose(se, R_GLM_GREG_SE, rtol=1e-9)


def test_glm_calibration_does_not_move_the_coefficients(design):
    """Centring is a variance operation. It must not touch a point estimate."""
    ps = design().weighting.poststratify(STYPE_POP, cells="stype")
    swept, _ = _glm(ps)
    unswept, _ = _glm(_refit(ps))
    assert_allclose(swept, unswept, rtol=1e-12)


def test_glm_sweep_tightens_the_se(design):
    """Pinning cells in the population removes variability the uncentred
    estimator still charges for, so the swept SE must be the smaller one."""
    ps = design().weighting.poststratify(STYPE_POP, cells="stype")
    _, swept = _glm(ps)
    _, unswept = _glm(_refit(ps))
    assert (swept < unswept).all()


# --- t-tests -----------------------------------------------------------------
#
# First on designs with NO population size, isolating the sweep from the FPC;
# the `pop_size` cases follow below.

R_TT2_PS_NOFPC_T = 1.74431180795363
R_TT2_PS_NOFPC_DIFF = 35.0985134660009
R_TT1_PS_NOFPC_T = 1.7511974720176
R_TT2_RAKE_NOFPC_T = 1.74923396543338


@pytest.fixture
def design_nofpc(apiclus1):
    def build():
        return Sample(apiclus1, Design(wgt="pw", psu="dnum"))

    return build


def test_two_sample_ttest_poststratified_matches_r(design_nofpc):
    ps = design_nofpc().weighting.poststratify(STYPE_POP, cells="stype")
    tt = ps.categorical.ttest("api00", group="sch.wide").to_polars()
    assert_allclose(tt["diff"][0], R_TT2_PS_NOFPC_DIFF, rtol=1e-12)
    assert_allclose(tt["t"][0], R_TT2_PS_NOFPC_T, rtol=1e-10)


def test_two_sample_ttest_raked_matches_r(design_nofpc):
    rk = design_nofpc().weighting.rake(
        controls={"stype": STYPE_POP, "sch.wide": SCHWIDE_POP}, tol=1e-12, max_iter=200
    )
    tt = rk.categorical.ttest("api00", group="sch.wide").to_polars()
    assert_allclose(tt["t"][0], R_TT2_RAKE_NOFPC_T, rtol=1e-10)


def test_one_sample_ttest_poststratified_matches_r(design_nofpc):
    ps = design_nofpc().weighting.poststratify(STYPE_POP, cells="stype")
    tt = ps.categorical.ttest("api00", mean_h0=600).to_polars()
    assert_allclose(tt["t"][0], R_TT1_PS_NOFPC_T, rtol=1e-10)


# On designs WITH a population size. The t-test facade builds the FPC column
# itself, as the estimation and regression facades do; before it did, every
# t-test on a `pop_size` design answered the fpc-free question, and the ratio
# -- exactly 1/sqrt(1 - 15/757), present with or without an adjustment -- was
# once misread as a failure of the calibration sweep.

R_TT2_BASE_T = 2.10898984779292
R_TT1_BASE_T = 1.87617650680004
R_TT2_PS_T = 1.76185477505784
R_TT1_PS_T = 1.76880968991608


def test_ttest_applies_the_fpc(design):
    """No adjustment: the FPC alone, against R."""
    d = design()
    two = d.categorical.ttest("api00", group="sch.wide").to_polars()["t"][0]
    one = d.categorical.ttest("api00", mean_h0=600).to_polars()["t"][0]
    assert_allclose(two, R_TT2_BASE_T, rtol=1e-10)
    assert_allclose(one, R_TT1_BASE_T, rtol=1e-10)


def test_ttest_poststratified_with_fpc_matches_r(design):
    """Both corrections at once -- the sweep and the FPC."""
    ps = design().weighting.poststratify(STYPE_POP, cells="stype")
    two = ps.categorical.ttest("api00", group="sch.wide").to_polars()["t"][0]
    one = ps.categorical.ttest("api00", mean_h0=600).to_polars()["t"][0]
    assert_allclose(two, R_TT2_PS_T, rtol=1e-10)
    assert_allclose(one, R_TT1_PS_T, rtol=1e-10)


def test_ttest_fpc_tightens_the_se(design, design_nofpc):
    """Sampling a known share of a finite population removes variability, so
    the corrected t must be the larger one."""
    ps = design().weighting.poststratify(STYPE_POP, cells="stype")
    ps0 = design_nofpc().weighting.poststratify(STYPE_POP, cells="stype")
    with_pop = ps.categorical.ttest("api00", group="sch.wide").to_polars()["t"][0]
    without = ps0.categorical.ttest("api00", group="sch.wide").to_polars()["t"][0]
    assert with_pop > without


# ---------------------------------------------------------------------------
# Subpopulations and by-groups on a calibrated design
#
# svy keeps every row and zeroes the weight outside the domain; R subsets,
# which zeroes the design weights but leaves `postStrata` at its full-sample
# values. Both keep the rows, and the cell means the sweep subtracts must come
# from the FULL sample either way: the adjustment was performed on the whole
# sample, so the constraint being removed is a whole-sample constraint.
#
# Restricting those means to the domain instead was measured at 9% off R.
# ---------------------------------------------------------------------------

R_BY_SCHWIDE_SE = [27.3731955191233, 24.3970669015281]
R_SUBPOP_SE = 24.3970669015281
R_SCOPED_ADJ_SE = 24.9249948644004


def test_by_group_se_matches_r_when_cells_cross_cut(design):
    """The by-groups here cut ACROSS the poststratification cells.

    Nested cells hide this: if every cell sits inside one group, restricting
    the cell means to that group changes nothing, which is why the NHANES case
    passed while this one did not.
    """
    ps = design().weighting.poststratify(STYPE_POP, cells="stype")
    got = ps.estimation.mean("api00", by="sch.wide").to_polars().sort("sch.wide")
    assert_allclose(got["se"].to_numpy(), R_BY_SCHWIDE_SE, rtol=1e-9)


def test_subpopulation_se_matches_r(design):
    """Estimation-time `where=` on a calibrated design.

    svy zeroes the active weight, which IS the record's new_wgt, so the
    calibrated weights the sweep needs are snapshotted before zeroing --
    svy's analogue of R leaving postStrata untouched under subset().
    """
    ps = design().weighting.poststratify(STYPE_POP, cells="stype")
    got = ps.estimation.mean("api00", where=col("sch.wide") == "Yes").to_polars()
    assert_allclose(got["se"][0], R_SUBPOP_SE, rtol=1e-9)


def test_scoped_adjustment_se_matches_r(design):
    """Weighting-time `where=`: the adjustment itself covers only those rows,
    so R's equivalent is postStratify on an already-subsetted design."""
    ps = design().weighting.poststratify(STYPE_POP, cells="stype", where=col("sch.wide") == "Yes")
    got = ps.estimation.mean("api00", where=col("sch.wide") == "Yes").to_polars()
    assert_allclose(got["se"][0], R_SCOPED_ADJ_SE, rtol=1e-9)


# ---------------------------------------------------------------------------
# Invalidation: the record must not fail silently
# ---------------------------------------------------------------------------


def test_dropping_a_snapshotted_column_warns_and_falls_back(design):
    """The worst available outcome is a silent fallback: the estimate is
    unchanged and the SE quietly stops crediting the calibration."""
    ps = design().weighting.poststratify(STYPE_POP, cells="stype")
    cells_col = ps.design.wgt_adjustment.cells[0]
    stripped = Sample(ps.data.drop(cells_col), ps.design)
    with pytest.warns(UserWarning, match="do not account for the poststratification"):
        se = _mean_se(stripped)
    assert_allclose(se, 23.967322009317, rtol=1e-6)  # the weights-fixed value


def test_use_weight_away_from_the_record_warns(design):
    ps = design().weighting.poststratify(STYPE_POP, cells="stype")
    with pytest.warns(UserWarning, match="active weight is 'pw'"):
        _mean_se(ps.use_weight("pw"))


def test_the_warning_fires_once_per_rebind(design):
    """Guarded on the data/design version: a warning per estimate would be
    noise, and noise gets muted."""
    ps = design().weighting.poststratify(STYPE_POP, cells="stype")
    stripped = Sample(ps.data.drop(ps.design.wgt_adjustment.cells[0]), ps.design)
    with pytest.warns(UserWarning) as rec:
        _mean_se(stripped)
        _mean_se(stripped)
        _total_se(stripped, "api00")
    assert len(rec) == 1


# ---------------------------------------------------------------------------
# The rest of the spec's validation plan
# ---------------------------------------------------------------------------

# rake(epsilon=1e-12) then postStratify, svymean(~api00)
R_STACKED_SE = 23.7630841260046


def test_taylor_and_replication_both_account_for_the_calibration(design):
    """The two paths used to answer different questions on a calibrated design.

    Replication has always re-adjusted each replicate; Taylor treated the
    weights as fixed. Both now respond, and Taylor matches R exactly. The two
    are only asymptotically equivalent and apiclus1 has 15 PSUs, so they are
    not expected to coincide -- R shows the same spread, which is what this
    compares against rather than asserting they agree.
    """
    jk = design().weighting.create_jk_wgts()
    ps = jk.weighting.poststratify(STYPE_POP, cells="stype")

    taylor = _mean_se(ps)
    replication = ps.estimation.mean("api00", method="replication").to_polars()["se"][0]
    assert_allclose(taylor, R_PS_CELLS_MEAN_SE, rtol=1e-9)

    # Both moved off their uncalibrated values, which is the whole point.
    assert taylor != pytest.approx(_mean_se(jk), rel=1e-6)
    assert replication != pytest.approx(
        jk.estimation.mean("api00", method="replication").to_polars()["se"][0], rel=1e-6
    )

    # R: taylor 23.9204864450905, jackknife 26.9345353674275 -> 0.8881.
    # svy's jackknife differs from R's by a documented convention (df fixed at
    # n_reps - 1), so the ratio is compared, not the value.
    assert taylor / replication == pytest.approx(0.8881, rel=0.02)


def test_trim_poststratify_cycle_leaves_the_record_exact(design):
    """The integrated cycle ends on a poststratify step, so the controls hold
    and the record describes the weights exactly -- which is why this, and not
    trimming afterwards, is the supported route to calibrated-and-trimmed."""
    ps = design().weighting.poststratify(
        STYPE_POP,
        cells="stype",
        trimming=TrimConfig(upper=0.995, redistribute=True, min_cell_size=1, max_iter=20),
    )
    got = ps.data.group_by("stype").agg(pl.col("ps_wgt").sum()).sort("stype")
    assert_allclose(got["ps_wgt"].to_numpy(), [4421.0, 755.0, 1018.0], rtol=1e-9)
    assert ps.design.wgt_adjustment.kind == "poststratification"

    d = ps.data.with_columns(
        [(pl.col("stype") == lv).cast(pl.Float64).alias(f"is_{lv}") for lv in ("E", "H", "M")]
    )
    trimmed = Sample(d, ps.design)
    for lv in ("E", "H", "M"):
        assert _total_se(trimmed, f"is_{lv}") == pytest.approx(0.0, abs=1e-9)


def test_clone_carries_the_record_and_its_columns(design):
    """History is descriptive, but the snapshotted columns are load-bearing: a
    clone keeping the record and losing the columns would warn and fall back."""
    ps = design().weighting.poststratify(STYPE_POP, cells="stype")
    cloned = ps.clone()
    rec = cloned.design.wgt_adjustment
    assert rec is not None and rec.kind == "poststratification"
    assert all(c in cloned.data.columns for c in rec.cells)
    assert_allclose(_mean_se(cloned), _mean_se(ps), rtol=0, atol=0)


def test_stacked_calibrations_centre_for_the_last_only(design):
    """A DOCUMENTED divergence from R, measured rather than asserted away.

    R replays every postStrata record in turn; svy keeps one record and
    centres for the last calibration, the conditional linearization that
    treats earlier-adjusted weights as base weights. On apiclus1 that is
    23.8646 against R's 23.7631 -- 0.43% -- with point estimates identical.
    """
    st = (
        design()
        .weighting.rake(
            controls={"stype": STYPE_POP, "sch.wide": SCHWIDE_POP}, tol=1e-12, max_iter=200
        )
        .weighting.poststratify(STYPE_POP, cells="stype")
    )
    assert st.design.wgt_adjustment.kind == "poststratification"
    got = st.estimation.mean("api00").to_polars()
    assert_allclose(got["est"][0], 641.594288477427, rtol=1e-9)
    assert_allclose(got["se"][0], 23.864643020701, rtol=1e-9)
    assert abs(got["se"][0] / R_STACKED_SE - 1.0) < 0.01
