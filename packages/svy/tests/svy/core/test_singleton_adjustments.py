# tests/svy/core/test_singleton_adjustments.py
import math

from pathlib import Path

import polars as pl
import pytest
import svy_rs as ps

import svy

from svy.core.enumerations import SingletonMethod
from svy.core.singleton import _VAR_EXCLUDE_COL
from svy.errors.singleton_errors import SingletonError


def _result(sample):
    """What the design's singleton rule did to the current data (internal)."""
    sample._sync_parts()
    return sample._singleton_result


def _declare(sample, method, **kw):
    """A fork of ``sample`` with the singleton rule declared on its design."""
    from svy.core.design import Singleton as _Rule

    new = sample._fork()
    new.update_design(singleton=_Rule(method, **kw))
    return new


DATA_DIR = Path(__file__).resolve().parents[2] / "test_data"


@pytest.fixture
def adjustment_sample():
    """
    Sample with 50% singletons (2 of 4 strata).
    Strata: A(1), B(2), C(1), D(2).
    """
    rows = [
        (0, "A", "101", 10.0, 1.0),
        (1, "A", "101", 12.0, 1.0),
        (2, "B", "201", 20.0, 1.0),
        (3, "B", "202", 22.0, 1.0),
        (4, "C", "301", 30.0, 1.0),
        (5, "D", "401", 40.0, 1.0),
        (6, "D", "402", 42.0, 1.0),
    ]
    df = pl.DataFrame(
        rows,
        schema=["id", "stratum", "cluster", "income", "weight"],
        orient="row",
    )
    design = svy.Design(stratum="stratum", psu="cluster", wgt="weight")
    return svy.Sample(data=df, design=design)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS FOR SCALE METHOD
# ══════════════════════════════════════════════════════════════════════════════


def test_scale_config_correctness(adjustment_sample):
    sample = _declare(adjustment_sample, "scale")
    result = _result(sample)
    assert result.method == SingletonMethod.SCALE
    assert result.config.singleton_fraction == 0.5
    assert result.config.var_exclude_col == _VAR_EXCLUDE_COL


def test_scale_excludes_singletons_from_data(adjustment_sample):
    sample = _declare(adjustment_sample, "scale")
    df = sample._data
    assert df.filter(pl.col("stratum") == "A").get_column(_VAR_EXCLUDE_COL).all()
    assert not df.filter(pl.col("stratum") == "B").get_column(_VAR_EXCLUDE_COL).any()


def test_scale_inflation_is_uniform(adjustment_sample):
    """R's "average" multiplies the whole variance matrix by 1/(1-f), whatever the statistic."""
    est = _declare(adjustment_sample, "scale").estimation

    mock_result = pl.DataFrame({"est": [50.0], "var": [100.0], "se": [10.0], "deff": [1.5]})
    res, cov = est._apply_scale_adjustment(mock_result, [100.0, -40.0, -40.0, 100.0])

    assert res["var"][0] == pytest.approx(200.0)
    assert res["se"][0] == pytest.approx(200.0**0.5)
    assert res["deff"][0] == pytest.approx(3.0)
    assert cov == pytest.approx([200.0, -80.0, -80.0, 200.0])


def test_scale_estimation_flow_integration(adjustment_sample, monkeypatch):
    """
    Verify integration for MEAN calculation.
    """
    sample = _declare(adjustment_sample, "scale")

    def mock_taylor_mean(*args, **kwargs):
        # Mirrors the kernel contract: (result frame, optional flat covariance).
        frame = pl.DataFrame(
            {
                "y": ["income"],
                "est": [1.0],
                "se": [1.0],
                "var": [1.0],
                "df": [10],
                "n": [100],
                "deff": [1.0],
            }
        )
        return frame, None

    monkeypatch.setattr(ps, "taylor_mean", mock_taylor_mean)

    result = sample.estimation.mean("income")

    # f = 0.5, so the kernel variance of 1.0 is inflated by 1/(1-f) = 2.
    est_var = result.estimates[0].se ** 2
    assert est_var == pytest.approx(2.0)


# ══════════════════════════════════════════════════════════════════════════════
# TESTS FOR CENTER METHOD
# ══════════════════════════════════════════════════════════════════════════════


def test_center_config_correctness(adjustment_sample):
    sample = _declare(adjustment_sample, "center")
    result = _result(sample)
    assert result.method == SingletonMethod.CENTER
    assert result.config.singleton_fraction is None


def test_center_does_not_exclude_rows(adjustment_sample):
    sample = _declare(adjustment_sample, "center")
    df = sample._data
    assert not df.get_column(_VAR_EXCLUDE_COL).any()


def test_center_arg_passing(adjustment_sample, monkeypatch):
    sample = _declare(adjustment_sample, "center")
    captured_kwargs = {}

    def mock_taylor_mean(*args, **kwargs):
        captured_kwargs.update(kwargs)
        frame = pl.DataFrame(
            {
                "y": ["income"],
                "est": [1.0],
                "se": [1.0],
                "var": [1.0],
                "df": [10],
                "n": [100],
                "deff": [1.0],
            }
        )
        return frame, None

    monkeypatch.setattr(ps, "taylor_mean", mock_taylor_mean)
    sample.estimation.mean("income")
    assert captured_kwargs.get("singleton_method") == "center"


def test_center_idempotent_if_not_configured(monkeypatch):
    # A design WITHOUT singletons: without a .center() call the engine must
    # receive singleton_method=None. (A singleton design would instead raise;
    # that fail-fast policy is covered in tests/svy/estimation/test_singletons.py.)
    rows = [
        (0, "B", "201", 20.0, 1.0),
        (1, "B", "202", 22.0, 1.0),
        (2, "D", "401", 40.0, 1.0),
        (3, "D", "402", 42.0, 1.0),
    ]
    df = pl.DataFrame(
        rows,
        schema=["id", "stratum", "cluster", "income", "weight"],
        orient="row",
    )
    design = svy.Design(stratum="stratum", psu="cluster", wgt="weight")
    sample = svy.Sample(data=df, design=design)
    captured_kwargs = {}

    def mock_taylor_mean(*args, **kwargs):
        captured_kwargs.update(kwargs)
        frame = pl.DataFrame(
            {
                "y": ["income"],
                "est": [1.0],
                "se": [1.0],
                "var": [1.0],
                "df": [10],
                "n": [100],
                "deff": [1.0],
            }
        )
        return frame, None

    monkeypatch.setattr(ps, "taylor_mean", mock_taylor_mean)
    sample.estimation.mean("income")
    assert captured_kwargs.get("singleton_method") is None


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICATION TEST (Regression against R)
# ══════════════════════════════════════════════════════════════════════════════


def test_verify_singleton_methods():
    try:
        data = pl.read_csv(DATA_DIR / "singleton_test_20012026.csv")
    except FileNotFoundError:
        pytest.skip("Verification CSV not found. Run generation script first.")

    design = svy.Design(stratum="stratum", psu="psu", wgt="weight")
    sample = svy.Sample(data, design)

    R_EXPECTED = {
        "MEAN": 24.86417079,
        "DF": 2.00000000,
        "SCALE_SE": 0.07411897,
        "CENTER_SE": 3.93156714,
    }

    # SCALE
    s_scale = _declare(sample, "scale")
    res_scale = s_scale.estimation.mean("y")
    est_scale = res_scale.estimates[0]

    assert est_scale.est == pytest.approx(R_EXPECTED["MEAN"], abs=1e-6)
    assert est_scale.se == pytest.approx(R_EXPECTED["SCALE_SE"], abs=1e-6)
    assert est_scale.df == pytest.approx(R_EXPECTED["DF"], abs=1e-6)

    # CENTER
    s_center = _declare(sample, "center")
    res_center = s_center.estimation.mean("y")
    est_center = res_center.estimates[0]

    assert est_center.est == pytest.approx(R_EXPECTED["MEAN"], abs=1e-6)
    assert est_center.se == pytest.approx(R_EXPECTED["CENTER_SE"], abs=1e-6)
    assert est_center.df == pytest.approx(R_EXPECTED["DF"], abs=1e-6)


def test_verify_scale_unequal_weights():
    """R lonely.psu="average" golden values where the singleton weight share differs from f.

    options(survey.lonely.psu = "average")
    d <- svydesign(ids = ~psu, strata = ~stratum, weights = ~wgt, nest = TRUE, data = df)
    svymean(~y, d, deff = TRUE); svytotal(~y, d, deff = TRUE); svyratio(~y, ~x, d)
    svymean(~factor(b), d); svyby(~y, ~g, d, svymean); svyby(~y, ~g, d, svytotal)
    svymean(~y, subset(d, g == 1)); svyvar(~y + x, d)
    """
    # Strata A-D hold 3, 2, 1, 4 PSUs (f = 1/4); x stays integer on purpose.
    data = pl.read_csv(DATA_DIR / "singleton_scale_13092026.csv")
    sample = svy.Sample(data, svy.Design(stratum="stratum", psu="psu", wgt="wgt"))
    est = _declare(sample, "scale").estimation

    mean = est.mean("y", deff="wor").estimates[0]
    assert mean.est == pytest.approx(10.277204, abs=1e-6)
    assert mean.se == pytest.approx(0.2915884087, abs=1e-9)
    assert mean.deff == pytest.approx(2.810019569, abs=1e-8)
    assert mean.df == 6

    total = est.total("y", deff="wor").estimates[0]
    assert total.se == pytest.approx(31.93840344, abs=1e-7)
    assert total.deff == pytest.approx(3.374199501, abs=1e-8)

    assert est.ratio("y", "x").estimates[0].se == pytest.approx(0.3324688156, abs=1e-9)

    prop = est.prop("b")
    assert [e.se for e in prop.estimates] == pytest.approx([0.07697404173] * 2, abs=1e-9)
    assert prop.covariance[0, 1] == pytest.approx(-0.0059250031, abs=1e-9)

    by_g = {e.by_level: e.se for e in est.mean("y", by="g").estimates}
    assert [by_g[(1,)], by_g[(2,)]] == pytest.approx([0.416112587, 0.4253003654], abs=1e-9)
    tot_g = {e.by_level: e.se for e in est.total("y", by="g").estimates}
    assert [tot_g[(1,)], tot_g[(2,)]] == pytest.approx([48.89427676, 42.55674713], abs=1e-7)

    where = est.mean("y", where=svy.col("g") == 1).estimates[0]
    assert where.se == pytest.approx(0.416112587, abs=1e-9)

    cov = est.cov(("y", "x")).estimates[0]
    assert cov.est == pytest.approx(0.100632804, abs=1e-9)
    assert cov.se == pytest.approx(0.2965718978, abs=1e-9)


def test_verify_scale_domain_fraction():
    """R lonely.psu="average" counts nstrat/nokstrat over the strata in the estimate.

    Strata A-E hold 3, 2, 1, 1, 2 PSUs. ``g == 1`` lives in A-C (3 strata, 1
    lonely: 3/2), ``g == 2`` in D-E (2 strata, 1 lonely: 2/1), ``h == 1`` only
    in D, and ``b`` touches every PSU (the design fraction 5/3).

    options(survey.lonely.psu = "average")
    d <- svydesign(ids = ~psu, strata = ~stratum, weights = ~wgt, nest = TRUE, data = df)
    svymean(~y, d, deff = TRUE); svymean(~y, subset(d, g == 1), deff = TRUE)
    svyby(~y, ~g, d, svymean); svyby(~y, ~g, d, svytotal); svyby(~y, ~b, d, svymean)
    svyratio(~y, ~x, subset(d, g == 2)); svymean(~factor(b), subset(d, g == 1))
    svyvar(~y + x, subset(d, g == 1)); svymean(~y, subset(d, h == 1))
    """
    data = pl.read_csv(DATA_DIR / "singleton_scale_domain_13092026.csv")
    sample = svy.Sample(data, svy.Design(stratum="stratum", psu="psu", wgt="wgt"))
    est = _declare(sample, "scale").estimation
    g1 = svy.col("g") == 1

    full = est.mean("y", deff="wor").estimates[0]
    assert full.se == pytest.approx(0.448784342569, abs=1e-9)
    assert full.deff == pytest.approx(2.74589125265, abs=1e-8)

    where = est.mean("y", where=g1, deff="wor").estimates[0]
    assert where.est == pytest.approx(10.2818260901, abs=1e-9)
    assert where.se == pytest.approx(0.517058811743, abs=1e-9)
    assert where.deff == pytest.approx(3.09651284751, abs=1e-8)

    by_g = {e.by_level: e.se for e in est.mean("y", by="g").estimates}
    assert [by_g[(1,)], by_g[(2,)]] == pytest.approx([0.517058811743, 0.947944278639], abs=1e-9)
    tot_g = {e.by_level: e.se for e in est.total("y", by="g").estimates}
    assert [tot_g[(1,)], tot_g[(2,)]] == pytest.approx([30.7462372913, 9.47066169944], abs=1e-8)
    by_b = {e.by_level: e.se for e in est.mean("y", by="b").estimates}
    assert [by_b[(0,)], by_b[(1,)]] == pytest.approx([0.431358491831, 0.524469647606], abs=1e-9)

    ratio = est.ratio("y", "x", where=svy.col("g") == 2).estimates[0]
    assert ratio.se == pytest.approx(0.667460675129, abs=1e-9)

    prop = est.prop("b", where=g1)
    assert [e.se for e in prop.estimates] == pytest.approx([0.0276313547124] * 2, abs=1e-9)
    assert prop.covariance[0, 1] == pytest.approx(-0.00076349176324, abs=1e-9)

    cov = est.cov(("y", "x"), where=g1).estimates[0]
    assert cov.est == pytest.approx(-0.711043927838, abs=1e-9)
    assert cov.se == pytest.approx(0.0663444415875, abs=1e-9)

    # Every stratum in the domain is lonely: no reference variance, NaN as in R.
    assert math.isnan(est.mean("y", where=svy.col("h") == 1).estimates[0].se)

    # The per-domain factor reaches the Woodruff quantile variance too:
    # svyquantile(~y, subset(d, g == 1), 0.5, qrule = "math", ci = TRUE)
    med = {e.by_level: e.se for e in est.median("y", by="g").estimates}
    assert [med[(1,)], med[(2,)]] == pytest.approx([0.830995952896, 0.304437877424], abs=1e-9)


def test_verify_skip_unequal_weights():
    """R lonely.psu="remove" golden values on the same design as the scale test.

    options(survey.lonely.psu = "remove")
    d <- svydesign(ids = ~psu, strata = ~stratum, weights = ~wgt, nest = TRUE, data = df)
    svymean(~y, d, deff = TRUE); svytotal(~y, d, deff = TRUE); svyratio(~y, ~x, d)
    svymean(~factor(b), d); svyby(~y, ~g, d, svymean); svyby(~y, ~g, d, svytotal)
    svymean(~y, subset(d, g == 1)); svyvar(~y + x, d)
    """
    data = pl.read_csv(DATA_DIR / "singleton_scale_13092026.csv")
    sample = svy.Sample(data, svy.Design(stratum="stratum", psu="psu", wgt="wgt"))
    est = _declare(sample, "skip").estimation

    # Every row stays in the estimator: the point estimates are the full-sample ones.
    mean = est.mean("y", deff="wor").estimates[0]
    assert mean.est == pytest.approx(10.277203998815, abs=1e-9)
    assert mean.se == pytest.approx(0.252522969382, abs=1e-9)
    assert mean.deff == pytest.approx(2.107514676615, abs=1e-8)
    assert mean.df == 6

    total = est.total("y", deff="wor").estimates[0]
    assert total.est == pytest.approx(1027.27617896321, abs=1e-8)
    assert total.se == pytest.approx(27.6594687312, abs=1e-7)
    assert total.deff == pytest.approx(2.53064962591, abs=1e-8)

    ratio = est.ratio("y", "x").estimates[0]
    assert ratio.est == pytest.approx(4.66767035847, abs=1e-9)
    assert ratio.se == pytest.approx(0.287926440246, abs=1e-9)

    prop = est.prop("b")
    assert [e.se for e in prop.estimates] == pytest.approx([0.0666614755669] * 2, abs=1e-9)
    assert prop.covariance[0, 1] == pytest.approx(-0.00444375232475, abs=1e-9)

    by_g = {e.by_level: e.se for e in est.mean("y", by="g").estimates}
    assert [by_g[(1,)], by_g[(2,)]] == pytest.approx([0.360364071215, 0.368320920686], abs=1e-9)
    tot_g = {e.by_level: e.se for e in est.total("y", by="g").estimates}
    assert [tot_g[(1,)], tot_g[(2,)]] == pytest.approx([42.3436857741, 36.8552241148], abs=1e-7)

    where = est.mean("y", where=svy.col("g") == 1).estimates[0]
    assert where.se == pytest.approx(0.360364071215, abs=1e-9)

    cov = est.cov(("y", "x")).estimates[0]
    assert cov.est == pytest.approx(0.100632803999, abs=1e-9)
    assert cov.se == pytest.approx(0.256838797506, abs=1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# CENTER IN DOMAINS (Regression against R)
# ══════════════════════════════════════════════════════════════════════════════


def _by_se(result):
    return {e.by_level: e.se for e in result.estimates}


@pytest.fixture
def center_domain():
    """Strata 1-6 hold 3, 2, 2, 3, 1, 1 PSUs; 5 and 6 are singletons.

    ``g1`` misses strata 3 and 6 and reaches one PSU of stratum 2 (whose first
    row has y = 0); ``g2`` misses stratum 5; ``g3`` lies in strata 3 and 6.
    ``y2`` is missing on g2's rows of stratum 3, so na.rm drops that stratum.

    options(survey.lonely.psu = "adjust")
    d <- svydesign(ids = ~psu, strata = ~stratum, weights = ~wgt, nest = TRUE, data = df)
    """
    data = pl.read_csv(DATA_DIR / "singleton_center_domain_29092026.csv")
    return data, svy.Design(stratum="stratum", psu="psu", wgt="wgt")


def test_verify_center_domain_grand_mean(center_domain):
    """R centers a singleton at the grand mean of the PSU totals of the strata
    holding domain rows; a singleton stratum outside the domain drops out.

    svyby(~y, ~dom, d, svytotal); svyby(~y, ~dom, d, svymean)
    svytotal(~y, subset(d, reg == 1)); svyby(~y, ~dom, subset(d, reg == 1), svytotal)
    svytotal(~y + x, subset(d, reg == 1)); svymean(~y + x, subset(d, dom == "g2"))
    svyby(~y, ~dom, d, svyratio, denominator = ~x)
    svymean(~factor(cat), subset(d, dom == "g1")); svyby(~factor(cat), ~dom, d, svytotal)
    svyby(~y2, ~dom, d, svytotal, na.rm = TRUE); svytotal(~y, d)
    """
    data, design = center_domain
    est = _declare(svy.Sample(data, design), "center").estimation
    reg1 = svy.col("reg") == 1
    g = [("g1",), ("g2",), ("g3",)]

    tot = _by_se(est.total("y", by="dom"))
    assert [tot[k] for k in g] == pytest.approx([392.461141804, 272.59136151, 174.583129602])
    mean = _by_se(est.mean("y", by="dom"))
    assert [mean[k] for k in g] == pytest.approx([0.570647427826, 1.02587655816, 0.974555800635])
    assert est.total("y", where=reg1).estimates[0].se == pytest.approx(663.086248419)
    tot_reg = _by_se(est.total("y", by="dom", where=reg1))
    assert [tot_reg[k] for k in g] == pytest.approx([409.960368006, 365.953170579, 116.135682259])
    multi = [r.estimates[0].se for r in est.total(["y", "x"], where=reg1)]
    assert multi == pytest.approx([663.086248419, 1062.71114004])
    multi = [r.estimates[0].se for r in est.mean(["y", "x"], where=svy.col("dom") == "g2")]
    assert multi == pytest.approx([1.02587655816, 0.974877022729])

    ratio = _by_se(est.ratio("y", "x", by="dom"))
    assert [ratio[k] for k in g] == pytest.approx(
        [0.0308233456977, 0.0494154470714, 0.0626878079167]
    )
    prop = est.prop("cat", where=svy.col("dom") == "g1")
    assert [e.se for e in prop.estimates] == pytest.approx(
        [0.0916353153642, 0.144508295716, 0.126554752422]
    )
    assert prop.covariance[0, 1] == pytest.approx(-0.00663178659604)
    cat_tot = {
        (e.by_level, e.y_level): e.se for e in est.total("cat", by="dom", as_factor=True).estimates
    }
    assert [cat_tot[(k, "p")] for k in g] == pytest.approx(
        [46.8597671545, 47.7378291356, 2.55168789453]
    )
    assert [cat_tot[(k, "r")] for k in g] == pytest.approx(
        [46.026479228, 44.8034735224, 30.5179364382]
    )

    narm = _by_se(est.total("y2", by="dom", drop_nulls=True))
    assert [narm[k] for k in g] == pytest.approx([392.461141804, 239.045584028, 174.583129602])
    assert est.total("y").estimates[0].se == pytest.approx(417.192026665)


def test_verify_center_domain_quantile(center_domain):
    """svyquantile(~y, subset(d, dom == "g1"), quantiles = 0.5, ci = TRUE, qrule = "math")
    and the same for g2; R's g1 upper bound is NaN (above the CDF's range)."""
    data, design = center_domain
    est = _declare(svy.Sample(data, design), "center").estimation
    g1 = est.median("y", where=svy.col("dom") == "g1").estimates[0]
    assert (g1.est, g1.lci) == pytest.approx((4.11, 1.77))
    assert math.isnan(g1.uci)
    g2 = est.median("y", where=svy.col("dom") == "g2").estimates[0]
    assert (g2.est, g2.lci, g2.uci) == pytest.approx((5.2, 1.62, 13.29))


def test_verify_center_domain_element_and_fpc(center_domain):
    """Without PSUs each row is a unit; with an FPC the stratum scale carries it.

    d_el <- svydesign(ids = ~1, strata = ~stratum, weights = ~wgt, data = df_el)
    svyby(~y, ~dom, d_el, svytotal)  # df_el keeps one row of stratum 5
    d_fpc <- svydesign(ids = ~psu, strata = ~stratum, weights = ~wgt, fpc = ~fpc,
                       nest = TRUE, data = df)
    svyby(~y, ~dom, d_fpc, svytotal)
    """
    data, design = center_domain
    first_row = pl.int_range(pl.len()).over("stratum", "psu") == 0
    element = data.filter((pl.col("stratum") != 5) | first_row)
    est = _declare(svy.Sample(element, svy.Design(stratum="stratum", wgt="wgt")), "center")
    tot = _by_se(est.estimation.total("y", by="dom"))
    assert [tot[k] for k in [("g1",), ("g2",), ("g3",)]] == pytest.approx(
        [320.087828212, 360.206440533, 163.303263685]
    )

    fpc = svy.Design(stratum="stratum", psu="psu", wgt="wgt", pop_size="fpc")
    est = _declare(svy.Sample(data, fpc), "center")
    tot = _by_se(est.estimation.total("y", by="dom"))
    assert [tot[k] for k in [("g1",), ("g2",), ("g3",)]] == pytest.approx(
        [322.676166692, 228.476644453, 151.084704618]
    )


def test_verify_center_domain_calibrated_keeps_whole_frame(center_domain):
    """A calibrated design's scores are nonzero outside the domain; R keeps
    those rows, so the grand mean runs over every stratum.

    ps <- postStratify(d, ~ps, data.frame(ps = c("u", "v"), Freq = c(700, 400)))
    svyby(~y, ~dom, ps, svytotal)
    """
    data, design = center_domain
    sample = _declare(svy.Sample(data, design), "center")
    ps = sample.weighting.poststratify(controls={"u": 700.0, "v": 400.0}, cells="ps")
    tot = _by_se(ps.estimation.total("y", by="dom"))
    assert [tot[k] for k in [("g1",), ("g2",), ("g3",)]] == pytest.approx(
        [595.400339346, 464.250541171, 277.209927545]
    )


# R: options(survey.lonely.psu = "adjust") and "average" on the center_domain
# design, with b = as.integer(y > 4):
#   svyttest(I(y - 4) ~ 0, d); svyttest(I(y - 4) ~ 0, subset(d, dom == "g1"))
#   svyttest(y ~ factor(reg), d); svyttest(y ~ factor(reg), subset(d, dom == "g2"))
#   svymean(~factor(cat), d); svymean(~factor(cat), subset(d, dom == "g1"))
#   svymean(~interaction(cat, b), d); svychisq(~cat + b, d, statistic = "F")
#   svychisq(~cat + b, subset(d, dom == "g3"), statistic = "F")
#   svyglm(y ~ x + factor(cat), d); svyglm(y ~ x, subset(d, dom == "g2"))
#   svyglm(b ~ x, d, family = quasibinomial()); svyranktest(y ~ factor(reg), d)
R_ANALYSES = {
    "center": {
        "ttest1": 1.53745252067,
        "ttest1_g1": 0.643261763547,
        "ttest2": 0.0814893662899,
        "ttest2_g2": -2.93185822993,
        "tab1": [0.0524243454611, 0.0738848042858, 0.0826685293244],
        "tab1_g1": [0.0916353153642, 0.144508295716, 0.126554752422],
        # cells (cat, b): (p,0) (p,1) (q,0) (q,1) (r,0) (r,1)
        "tab2": [
            0.0302774643284,
            0.053948651639,
            0.0622148508883,
            0.0426679753701,
            0.0735030584894,
            0.0872823037327,
        ],
        "chisq_f": (0.753732823118, 1.99705074253, 11.9823044552),
        "chisq_f_g3": 10.9464852213,
        "glm": [0.539219093393, 0.0369297318375, 0.578771720032, 0.3283547864],
        "glm_g2": [0.612034013818, 0.0627265876046],
        "logit": [1.95132489725, 0.290028822486],
        "rank": 0.297034636741,
    },
    "scale": {
        "ttest1": 1.35172351073,
        "ttest1_g1": 0.586668934101,
        "ttest2": 0.0706173660424,
        "ttest2_g2": -2.65844847869,
        "tab1": [0.0616336456361, 0.0839065744083, 0.0900551703984],
        "tab1_g1": [0.104436497965, 0.151555787673, 0.136254857437],
        "tab2": [
            0.0235809448849,
            0.0632869044554,
            0.0735931972578,
            0.0503097244452,
            0.038121219484,
            0.0992846292473,
        ],
        "chisq_f": (0.735919080177, 1.84573437661, 11.0744062597),
        "chisq_f_g3": 54.4095322734,
        "glm": [0.644720215198, 0.0441446294343, 0.704272525278, 0.386870437075],
        "glm_g2": [0.682152906695, 0.0701164511351],
        "logit": [2.11684898698, 0.29746914234],
        "rank": 0.253717952447,
    },
}


@pytest.mark.parametrize("method", ["center", "scale"])
def test_verify_singleton_rule_in_every_analysis(center_domain, method):
    """t-tests, tables, the Rao-Scott F, GLMs and rank tests apply the rule
    as estimation does (they used to leave singleton strata out, R's "remove")."""
    data, design = center_domain
    data = data.with_columns(b=(pl.col("y") > 4).cast(pl.Int64))
    sample = _declare(svy.Sample(data, design), method)
    cat, r = sample.categorical, R_ANALYSES[method]
    g1, g2, g3 = (svy.col("dom") == g for g in ("g1", "g2", "g3"))

    assert cat.ttest("y", mean_h0=4).stats.t == pytest.approx(r["ttest1"])
    assert cat.ttest("y", mean_h0=4, where=g1).stats.t == pytest.approx(r["ttest1_g1"])
    assert cat.ttest("y", group="reg").stats.t == pytest.approx(r["ttest2"])
    assert cat.ttest("y", group="reg", where=g2).stats.t == pytest.approx(r["ttest2_g2"])

    assert [e.se for e in cat.tabulate("cat").estimates] == pytest.approx(r["tab1"])
    assert [e.se for e in cat.tabulate("cat", where=g1).estimates] == pytest.approx(r["tab1_g1"])
    two_way = cat.tabulate("cat", "b")
    assert [e.se for e in two_way.estimates] == pytest.approx(r["tab2"])
    f = two_way.stats.f
    assert (f.value, f.df_num, f.df_den) == pytest.approx(r["chisq_f"])
    assert cat.tabulate("cat", "b", where=g3).stats.f.value == pytest.approx(r["chisq_f_g3"])

    glm = sample.glm.fit("y", x=["x", svy.Cat("cat")])
    assert [c.se for c in glm.coefs] == pytest.approx(r["glm"])
    glm = sample.glm.fit("y", x=["x"], where=g2)
    assert [c.se for c in glm.coefs] == pytest.approx(r["glm_g2"])
    # The skip baseline already sits ~5e-7 from R on this logit.
    logit = sample.glm.fit("b", x=["x"], family="binomial")
    assert [c.se for c in logit.coefs] == pytest.approx(r["logit"], rel=2e-6)

    rank = cat.ranktest("y", group="reg", method="kruskal-wallis")
    assert rank.stats.value == pytest.approx(r["rank"])


@pytest.mark.parametrize(
    "method", ["self_representing", "skip", "collapse", "pool", "center", "scale"]
)
def test_tabulate_uses_the_singleton_rule_like_estimation(center_domain, method):
    """A one-way table's SEs are the proportions' SEs under every rule,
    including those that recode the strata and PSUs."""
    data, design = center_domain
    sample = _declare(svy.Sample(data, design), method)
    table = sample.categorical.tabulate("cat")
    prop = sample.estimation.prop("cat")
    assert [e.se for e in table.estimates] == pytest.approx([e.se for e in prop.estimates])


@pytest.mark.parametrize(
    "analysis",
    [
        lambda s: s.categorical.ttest("y", mean_h0=4),
        lambda s: s.categorical.ttest("y", group="reg"),
        lambda s: s.categorical.ranktest("y", group="reg", method="kruskal-wallis"),
        lambda s: s.categorical.ranktest("y", group="reg", score_fn=lambda r, n: r / n),
        lambda s: s.categorical.tabulate("cat"),
        lambda s: s.glm.fit("y", x=["x"]),
    ],
    ids=["ttest", "ttest2", "ranktest", "ranktest_custom", "tabulate", "glm"],
)
def test_unhandled_singletons_raise_in_every_taylor_analysis(center_domain, analysis):
    data, design = center_domain
    with pytest.raises(SingletonError, match="singleton"):
        analysis(svy.Sample(data, design))


def test_verify_scale_calibrated_counts_every_stratum(center_domain):
    """R keeps a calibrated design's out-of-domain rows, so nstrat/nokstrat is
    counted over every stratum (6/4 here), not the domain's.

    options(survey.lonely.psu = "average")
    ps <- postStratify(d, ~ps, data.frame(ps = c("u", "v"), Freq = c(700, 400)))
    svyby(~y, ~dom, ps, svytotal); svyby(~y, ~dom, ps, svymean)
    m <- svymean(~I(y - 4), subset(ps, dom == "g3")); coef(m) / SE(m)
    """
    data, design = center_domain
    sample = _declare(svy.Sample(data, design), "scale")
    ps = sample.weighting.poststratify(controls={"u": 700.0, "v": 400.0}, cells="ps")
    g = [("g1",), ("g2",), ("g3",)]
    tot = _by_se(ps.estimation.total("y", by="dom"))
    assert [tot[k] for k in g] == pytest.approx([664.6618508206, 524.4551412330, 257.9743863549])
    mean = _by_se(ps.estimation.mean("y", by="dom"))
    assert [mean[k] for k in g] == pytest.approx([0.6800421585, 1.2859366357, 0.6648057037])
    g3 = svy.col("dom") == "g3"
    assert ps.estimation.total("y", where=g3).estimates[0].se == pytest.approx(257.9743863549)
    assert ps.categorical.ttest("y", mean_h0=4, where=g3).stats.t == pytest.approx(0.0225441528)
