"""
Tests for quantile estimation in svy (issue #112).

Standard errors follow Woodruff (1952): the design-based variance of the
estimated proportion P(Y <= q) is taken on the probability scale, and the
interval comes from inverting the weighted CDF at ``F(q_hat) +/- t*se_p``,
where ``F(q_hat)`` is the estimated CDF at the quantile.

Reference values come from R's ``svyquantile`` (``interval.type="mean"``, the
default), which is the same construction. svy's ``q_method`` maps onto R's
``qrule``:

    q_method="higher"  <->  qrule="math" (hf1)
    q_method="linear"  <->  qrule="hf4"

Centering at ``F(q_hat)`` rather than at ``p`` matters on a step CDF: R's older
``oldsvyquantile(interval.type="Wald")`` centers at ``p`` and gives different
limits whenever ``F(q_hat) != p``. A limit whose probability falls outside
[0, 1] is NaN, and so is the SE, as in R.

Point estimates and both confidence limits agree with R **exactly**. Standard
errors agree to ~1e-15 relative, and the residual is not the survey math:
``se`` is the back-solved half-width ``(uci - lci) / (2t)``, and scipy's
``t.ppf(0.975, 190)`` differs from R's ``qt(0.975, 190)`` by 1.0e-15 relative.
Hence the 1e-12 tolerance below, which still leaves three orders of magnitude
of headroom.

Two fixtures:

* ``quantile_ref_20260805.csv`` stresses the estimator: 5000 records, 10 strata
  x 20 PSUs (df = 190), skewed income spanning 1.9k to 536k, and continuous
  unequal weights spanning a 425x range. The extreme probabilities (0.001,
  0.999) exercise the sparse tails, and 0.0345 / 0.679 are deliberately
  off-grid.
* ``quantile_ties_13092026.csv`` is small and heavily tied: 60 records, 4
  strata x 3 PSUs (df = 8), integer ``y`` with 16 distinct values. Ties make
  ``F(q_hat)`` differ from ``p`` by up to 0.15, and the low df pushes tail
  limits outside [0, 1] on one side at a time.

R Setup Code:
-------------
```r
options(digits = 15)
library(survey)

d <- read.csv("packages/svy/tests/test_data/quantile_ref_20260805.csv")
des <- svydesign(id = ~psu, strata = ~stratum, weights = ~wgt, data = d, nest = TRUE)
degf(des)   # 190

probs <- c(0.001, 0.01, 0.0345, 0.1, 0.25, 0.5, 0.679, 0.75, 0.9, 0.99, 0.999)

# R_HIGHER: qrule="math"   |   R_LINEAR: qrule="hf4"
for (p in probs) {
    q <- svyquantile(~income, des, quantiles = p, ci = TRUE, qrule = "math")
    cat(sprintf("%g %.15g %.15g %.15g %.15g\n",
        p, coef(q), SE(q), confint(q)[1], confint(q)[2]))
}

# R_BY_REGION
for (rg in c("North", "South")) {
    for (p in c(0.1, 0.25, 0.5, 0.75, 0.9)) {
        svyquantile(~income, subset(des, region == rg), quantiles = p, ci = TRUE)
    }
}

# TIES_*: same loops on the ties fixture
t <- read.csv("packages/svy/tests/test_data/quantile_ties_13092026.csv")
dt <- svydesign(id = ~psu, strata = ~stratum, weights = ~wgt, data = t, nest = TRUE)
degf(dt)                                  # 8
degf(subset(dt, region == "East"))        # 7
for (p in c(0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98)) {
    svyquantile(~y, dt, quantiles = p, ci = TRUE, qrule = "math")   # and "hf4"
}
for (rg in c("East", "West")) {
    for (p in c(0.25, 0.5, 0.75)) {
        svyquantile(~y, subset(dt, region == rg), quantiles = p, ci = TRUE)
    }
}
```
"""

import math

from pathlib import Path

import polars as pl
import pytest

from svy import Design, Sample, col
from svy.core.enumerations import PopParam
from svy.estimation import EstimateList


# Point estimates and confidence limits match R exactly; SEs carry only the
# scipy-vs-R t-quantile difference (~1e-15).
REL = 1e-12

DATA_DIR = Path(__file__).resolve().parents[2] / "test_data"
FIXTURE = DATA_DIR / "quantile_ref_20260805.csv"
TIES_FIXTURE = DATA_DIR / "quantile_ties_13092026.csv"
NAN = float("nan")


def approx(expected):
    return pytest.approx(expected, rel=REL, nan_ok=True)


# ============================================================================
# Test Dataset
# ============================================================================


def make_quantile_data() -> pl.DataFrame:
    """The committed fixture R was run against — the single source of truth."""
    return pl.read_csv(FIXTURE)


@pytest.fixture
def sample() -> Sample:
    return Sample(
        make_quantile_data(),
        Design(stratum="stratum", psu="psu", wgt="wgt"),
    )


@pytest.fixture
def ties_sample() -> Sample:
    return Sample(pl.read_csv(TIES_FIXTURE), Design(stratum="stratum", psu="psu", wgt="wgt"))


# R: svyquantile(..., qrule="math")
# (prob, est, se, lci, uci)
R_HIGHER = [
    (0.001, 2453.7738, 153.130283641138, 2120.4955, 2724.6031),
    (0.01, 4400.0454, 309.189448122974, 3746.2698, 4966.0396),
    (0.0345, 6732.2575, 372.563016693827, 5895.4843, 7365.2664),
    (0.1, 10884.6084, 466.970514492444, 10008.7569, 11850.9819),
    (0.25, 18668.346, 640.564054561709, 17542.652, 20069.7133),
    (0.5, 33429.9866, 950.920786387401, 31453.9434, 35205.3795),
    (0.679, 46769.1758, 1280.00883487419, 44568.8496, 49618.5566),
    (0.75, 54394.0689, 1448.501966193, 51534.3152, 57248.7371),
    (0.9, 91935.9034, 4150.910275813, 83424.7868, 99800.3618),
    (0.99, 187573.6841, 12990.0017823841, 175376.1657, 226622.4549),
    (0.999, 329662.0411, NAN, 297789.5347, NAN),
]

# R: svyquantile(..., qrule="hf4"). p = 0.001 is tested separately: R's hf4
# extrapolates its lower limit below the sample minimum.
R_LINEAR = [
    (0.01, 4396.7486276543, 307.497231851433, 3745.73724999402, 4958.8311614227),
    (0.0345, 6724.26363172486, 379.725845248474, 5818.70757283937, 7316.74743521314),
    (0.1, 10881.6789867565, 468.015230948578, 9994.61988448812, 11840.966349792),
    (0.25, 18659.0719519676, 649.235015801994, 17504.967114701, 20066.235845524),
    (0.5, 33427.4974064113, 958.510682003842, 31402.6724551328, 35184.0511211366),
    (0.679, 46763.5698205693, 1275.76247043989, 44555.5223555054, 49588.47720847),
    (0.75, 54391.3922866024, 1444.64585454854, 51531.1696757038, 57230.3789979205),
    (0.9, 91931.3113738885, 4185.36731830946, 82677.3992845479, 99188.9092593332),
    (0.99, 187465.615949273, 12950.4161905614, 174712.965314289, 225803.087123346),
    (0.999, 329485.233713584, NAN, 268602.602633441, NAN),
]

# R: per-domain via subset(des, region == ...), qrule="math"
# (region, prob, est, se, lci, uci)
R_BY_REGION = [
    ("North", 0.1, 11291.8521, 835.116102817864, 9338.4527, 12656.6605),
    ("North", 0.25, 18968.7621, 1127.60452409403, 16839.273, 21319.6394),
    ("North", 0.5, 34150.2656, 1732.3576053785, 30891.9519, 37775.2134),
    ("North", 0.75, 55752.9531, 2776.86072226291, 51610.4256, 62643.8626),
    ("North", 0.9, 95245.4547, 6921.97454502464, 83673.6125, 111177.0337),
    ("South", 0.1, 10501.6883, 740.114860222214, 9297.6129, 12238.3476),
    ("South", 0.25, 18557.5425, 1211.78278106246, 16121.0066, 20935.8426),
    ("South", 0.5, 32525.2997, 1764.29673214534, 29476.4836, 36486.6504),
    ("South", 0.75, 52340.9449, 2380.18772230549, 48057.5476, 57514.8643),
    ("South", 0.9, 85969.1396, 4872.01640313528, 77599.3692, 96957.5911),
]

# R on the ties fixture, qrule="math" and qrule="hf4" (df = 8)
# (prob, est, se, lci, uci)
TIES_HIGHER = [
    (0.02, 7, NAN, NAN, 9),
    (0.05, 7, NAN, NAN, 9),
    (0.1, 8, 0.433650566680993, 7, 9),
    (0.25, 10, 0.433650566680993, 10, 12),
    (0.5, 12, 0.867301133361986, 11, 15),
    (0.75, 15, 1.51777698338348, 13, 20),
    (0.9, 17, NAN, 15, NAN),
    (0.95, 19, NAN, 17, NAN),
    (0.98, 20, NAN, 19, NAN),
]

TIES_LINEAR = [
    (0.02, 7, NAN, NAN, 9),
    (0.05, 7, NAN, NAN, 9),
    (0.1, 8, 0.433650566680993, 7, 9),
    (0.25, 9.29686605609616, 0.23270679076538, 8.92675435640989, 10),
    (0.5, 11.3953959484346, 0.65047585002149, 10, 13),
    (0.75, 14.5441992783791, 1.18564617983684, 12, 17.4682099871855),
    (0.9, 17, NAN, 15, NAN),
    (0.95, 19, NAN, 16.5995283270563, NAN),
    (0.98, 19.7291935953421, NAN, 16.5995283270563, NAN),
]

# R: subset(dt, region == ...), qrule="math"; East loses a PSU, so df = 7
# (region, prob, est, se, lci, uci, df)
TIES_BY_REGION = [
    ("East", 0.25, 9, 0.211450085426133, 9, 10, 7),
    ("East", 0.5, 12, 1.48015059798293, 10, 17, 7),
    ("East", 0.75, 15, 1.48015059798293, 12, 19, 7),
    ("West", 0.25, 10, 0.65047585002149, 10, 13, 8),
    ("West", 0.5, 12, 1.08412641670248, 10, 15, 8),
    ("West", 0.75, 14, 1.51777698338348, 13, 20, 8),
]


def assert_matches(got, est, se, lci, uci):
    assert got.est == approx(est)
    assert got.se == approx(se)
    assert got.lci == approx(lci)
    assert got.uci == approx(uci)


# ============================================================================
# Agreement with R
# ============================================================================


class TestMatchesR:
    @pytest.mark.parametrize("prob, est, se, lci, uci", R_HIGHER)
    def test_higher_matches_r(self, sample, prob, est, se, lci, uci):
        """q_method='higher' reproduces R's qrule='math'."""
        got = sample.estimation.quantile("income", p=prob, q_method="higher").estimates[0]
        assert_matches(got, est, se, lci, uci)
        assert got.df == 190

    @pytest.mark.parametrize("prob, est, se, lci, uci", R_LINEAR)
    def test_linear_matches_r(self, sample, prob, est, se, lci, uci):
        """q_method='linear' reproduces R's qrule='hf4'.

        The interval endpoints must be inverted with the *same* rule as the
        point estimate. Inverting linearly while estimating with 'higher'
        shrinks the interval, badly so in the tails.
        """
        got = sample.estimation.quantile("income", p=prob, q_method="linear").estimates[0]
        assert_matches(got, est, se, lci, uci)

    def test_linear_lower_limit_stops_at_sample_minimum(self, sample):
        """Deliberate deviation from R at p = 0.001.

        The lower Woodruff probability falls below the first CDF step. R's hf4
        then extrapolates past the smallest observation (1732.52 < 1884.07);
        svy stops at the minimum. The estimate and upper limit still match.
        """
        got = sample.estimation.quantile("income", p=0.001, q_method="linear").estimates[0]
        assert got.est == approx(2453.72704364921)
        assert got.uci == approx(2681.77200413227)
        assert got.lci == make_quantile_data()["income"].min()

    def test_domains_match_r(self, sample):
        """by= domains reproduce R's subset() estimates and intervals."""
        results = sample.estimation.quantile("income", p=(0.1, 0.25, 0.5, 0.75, 0.9), by="region")

        got = {
            (e.by_level[0], round(float(e.prob), 2)): e for est in results for e in est.estimates
        }
        assert len(got) == len(R_BY_REGION)

        for region, prob, est, se, lci, uci in R_BY_REGION:
            assert_matches(got[(region, prob)], est, se, lci, uci)

    def test_tail_probabilities(self, sample):
        """Extreme probabilities stay inside the observed support."""
        data = make_quantile_data()
        lo, hi = data["income"].min(), data["income"].max()

        for prob in (0.01, 0.99):
            got = sample.estimation.quantile("income", p=prob).estimates[0]
            assert lo <= got.est <= hi
            assert lo <= got.lci <= got.est <= got.uci <= hi


class TestTiesMatchR:
    """Small, heavily tied design where F(q_hat) and p disagree."""

    @pytest.mark.parametrize("prob, est, se, lci, uci", TIES_HIGHER)
    def test_higher_matches_r(self, ties_sample, prob, est, se, lci, uci):
        got = ties_sample.estimation.quantile("y", p=prob, q_method="higher").estimates[0]
        assert_matches(got, est, se, lci, uci)
        assert got.df == 8

    @pytest.mark.parametrize("prob, est, se, lci, uci", TIES_LINEAR)
    def test_linear_matches_r(self, ties_sample, prob, est, se, lci, uci):
        got = ties_sample.estimation.quantile("y", p=prob, q_method="linear").estimates[0]
        assert_matches(got, est, se, lci, uci)

    def test_domains_match_r(self, ties_sample):
        results = ties_sample.estimation.quantile("y", p=(0.25, 0.5, 0.75), by="region")
        got = {(e.by_level[0], e.prob): e for est in results for e in est.estimates}
        assert len(got) == len(TIES_BY_REGION)

        for region, prob, est, se, lci, uci, df in TIES_BY_REGION:
            row = got[(region, prob)]
            assert_matches(row, est, se, lci, uci)
            assert row.df == df

    def test_where_matches_r_subset(self, ties_sample):
        """where= domain: CDF, center and df all come from the domain."""
        for prob, est, se, lci, uci, df in (
            (r[1], *r[2:]) for r in TIES_BY_REGION if r[0] == "East"
        ):
            got = ties_sample.estimation.quantile(
                "y", p=prob, where=col("region") == "East"
            ).estimates[0]
            assert_matches(got, est, se, lci, uci)
            assert got.df == df


class TestIntervalCenter:
    def test_centered_at_cdf_of_estimate_not_p(self, ties_sample):
        """Regression: the interval is centered at F(q_hat), not at p.

        At p = 0.25 the estimate is 10 and F(10) = 0.3998. Centering at p
        (R's oldsvyquantile Wald) gives (9, 10); centering at F(q_hat) gives
        R svyquantile's (10, 12).
        """
        got = ties_sample.estimation.quantile("y", p=0.25).estimates[0]
        assert got.est == 10
        assert (got.lci, got.uci) == (10, 12)

    def test_median_centered_at_cdf_of_estimate(self, ties_sample):
        got = ties_sample.estimation.median("y").estimates[0]
        assert_matches(got, 12, 0.867301133361986, 11, 15)

    def test_lower_limit_nan_alone(self, ties_sample):
        """Only the side whose probability drops below 0 is undefined."""
        got = ties_sample.estimation.quantile("y", p=0.05).estimates[0]
        assert math.isnan(got.lci)
        assert got.uci == 9
        assert math.isnan(got.se)
        assert got.est == 7

    def test_upper_limit_nan_alone(self, ties_sample):
        """Only the side whose probability rises above 1 is undefined."""
        got = ties_sample.estimation.quantile("y", p=0.9).estimates[0]
        assert got.lci == 15
        assert math.isnan(got.uci)
        assert math.isnan(got.se)
        assert got.est == 17

    @pytest.mark.parametrize("q_method", ["higher", "linear"])
    def test_finite_limits_bracket_the_estimate(self, ties_sample, sample, q_method):
        probs = tuple(round(0.01 * k, 2) for k in range(1, 100))
        for s, y in ((ties_sample, "y"), (sample, "income")):
            for r in s.estimation.quantile(y, p=probs, q_method=q_method):
                e = r.estimates[0]
                if not math.isnan(e.lci):
                    assert e.lci <= e.est, f"{y} p={e.prob}"
                if not math.isnan(e.uci):
                    assert e.est <= e.uci, f"{y} p={e.prob}"


class TestSingletonScale:
    """Woodruff intervals under singleton.scale() (R lonely.psu = "average").

    options(survey.lonely.psu = "average")
    d <- read.csv("packages/svy/tests/test_data/singleton_scale_13092026.csv")
    des <- svydesign(ids = ~psu, strata = ~stratum, weights = ~wgt, nest = TRUE, data = d)
    svyquantile(~y, des, quantiles = p, ci = TRUE)

    At p = 0.5, F(q_hat) = 0.5114: centering at p gave (9.779978, 10.729145).
    """

    # (prob, est, se, lci, uci)
    R_AVERAGE = [
        (0.25, 9.40494631125905, 0.759305995358125, 6.7024252672981, 10.4183349446726),
        (0.5, 10.3991690656942, 0.227382664731022, 9.7799784705775, 10.892749144728),
        (0.75, 11.1976924252693, 0.231619624578124, 10.6656272262761, 11.799132834952),
    ]

    @pytest.fixture
    def scaled(self) -> Sample:
        data = pl.read_csv(DATA_DIR / "singleton_scale_13092026.csv")
        return Sample(data, Design(stratum="stratum", psu="psu", wgt="wgt")).singleton.scale()

    @pytest.mark.parametrize("prob, est, se, lci, uci", R_AVERAGE)
    def test_quantile_matches_r(self, scaled, prob, est, se, lci, uci):
        got = scaled.estimation.quantile("y", p=prob).estimates[0]
        assert_matches(got, est, se, lci, uci)
        assert got.df == 6

    def test_median_matches_r(self, scaled):
        _, est, se, lci, uci = self.R_AVERAGE[1]
        assert_matches(scaled.estimation.median("y").estimates[0], est, se, lci, uci)


# ============================================================================
# Relationship to median()
# ============================================================================


class TestMedianEquivalence:
    def test_quantile_at_half_equals_median(self, sample):
        """median() is the p = 0.5 case and must agree to the bit."""
        med = sample.estimation.median("income").estimates[0]
        qua = sample.estimation.quantile("income", p=0.5).estimates[0]

        assert qua.est == med.est
        assert qua.se == med.se
        assert qua.lci == med.lci
        assert qua.uci == med.uci
        assert qua.df == med.df

    def test_params_differ(self, sample):
        """median() reports as MEDIAN; quantile() as QUANTILE with a prob."""
        assert sample.estimation.median("income").param == PopParam.MEDIAN
        assert sample.estimation.quantile("income", p=0.5).param == PopParam.QUANTILE

        assert sample.estimation.median("income").estimates[0].prob is None
        assert sample.estimation.quantile("income", p=0.5).estimates[0].prob == 0.5

    def test_median_by_domain_equivalence(self, sample):
        med = sample.estimation.median("income", by="region")
        qua = sample.estimation.quantile("income", p=0.5, by="region")

        by_level = lambda est: sorted(est.estimates, key=lambda e: e.by_level)  # noqa: E731
        for m, q in zip(by_level(med), by_level(qua)):
            assert q.est == m.est
            assert q.se == m.se

    def test_median_by_domain_equivalence_on_ties(self, ties_sample):
        med = ties_sample.estimation.median("y", by="region")
        qua = ties_sample.estimation.quantile("y", p=0.5, by="region")

        by_level = lambda est: sorted(est.estimates, key=lambda e: e.by_level)  # noqa: E731
        for m, q in zip(by_level(med), by_level(qua), strict=True):
            assert (q.est, q.se, q.lci, q.uci, q.df) == (m.est, m.se, m.lci, m.uci, m.df)

    def test_batched_median_matches_single(self, ties_sample):
        """The multi-variable median path builds its intervals the same way."""
        data = pl.read_csv(TIES_FIXTURE).with_columns((pl.col("y") * 10).alias("y10"))
        s = Sample(data, Design(stratum="stratum", psu="psu", wgt="wgt"))

        single = s.estimation.median("y").estimates[0]
        batched = [r.estimates[0] for r in s.estimation.median(["y", "y10"])]

        assert (batched[0].est, batched[0].se, batched[0].lci, batched[0].uci) == (
            single.est,
            single.se,
            single.lci,
            single.uci,
        )
        assert_matches(batched[1], 120, 8.67301133361986, 110, 150)


# ============================================================================
# Return shape
# ============================================================================


class TestReturnShape:
    def test_scalar_p_returns_single_estimate(self, sample):
        result = sample.estimation.quantile("income", p=0.5)
        assert not isinstance(result, list)
        assert len(result.estimates) == 1

    def test_sequence_p_returns_estimate_list(self, sample):
        result = sample.estimation.quantile("income", p=(0.25, 0.5, 0.75))
        assert isinstance(result, EstimateList)
        assert isinstance(result, list)  # stays a plain list for consumers
        assert len(result) == 3
        assert [e.estimates[0].prob for e in result] == [0.25, 0.5, 0.75]

    def test_default_is_quartiles(self, sample):
        result = sample.estimation.quantile("income")
        assert [e.estimates[0].prob for e in result] == [0.25, 0.50, 0.75]

    def test_probability_order_is_preserved(self, sample):
        result = sample.estimation.quantile("income", p=(0.9, 0.1, 0.5))
        assert [e.estimates[0].prob for e in result] == [0.9, 0.1, 0.5]

    def test_multi_variable_flattens(self, sample):
        data = make_quantile_data().with_columns((pl.col("income") * 0.4).alias("expend"))
        s = Sample(data, Design(stratum="stratum", psu="psu", wgt="wgt"))

        result = s.estimation.quantile(["income", "expend"], p=(0.25, 0.75))
        assert isinstance(result, EstimateList)
        assert [(e.estimates[0].y, e.estimates[0].prob) for e in result] == [
            ("income", 0.25),
            ("income", 0.75),
            ("expend", 0.25),
            ("expend", 0.75),
        ]

    def test_by_gives_one_row_per_domain(self, sample):
        result = sample.estimation.quantile("income", p=(0.25, 0.75), by="region")
        assert len(result) == 2
        for est in result:
            assert len(est.estimates) == 2  # North, South

    def test_estimate_list_prints_members(self, sample):
        """A bare list would print object reprs; EstimateList must not."""
        text = str(sample.estimation.quantile("income", p=(0.25, 0.75)))
        assert "object at 0x" not in text
        assert "QUANTILE" in text


# ============================================================================
# Validation
# ============================================================================


class TestValidation:
    @pytest.mark.parametrize("prob", [0.0, 1.0, -0.1, 1.5])
    def test_probability_must_be_inside_unit_interval(self, sample, prob):
        with pytest.raises(ValueError, match="strictly in"):
            sample.estimation.quantile("income", p=prob)

    def test_duplicate_probabilities_rejected(self, sample):
        with pytest.raises(ValueError, match="duplicate"):
            sample.estimation.quantile("income", p=(0.25, 0.5, 0.25))

    def test_empty_probabilities_rejected(self, sample):
        with pytest.raises(ValueError, match="at least one"):
            sample.estimation.quantile("income", p=())


# ============================================================================
# Other paths
# ============================================================================


class TestOtherPaths:
    def test_where_clause_is_recorded(self, sample):
        from svy import col

        result = sample.estimation.quantile("income", p=0.5, where=col("region") == "North")
        assert result.where_clause is not None

    def test_where_matches_manual_subset(self, sample):
        from svy import col

        filtered = sample.estimation.quantile("income", p=0.5, where=col("region") == "North")
        subset = Sample(
            make_quantile_data().filter(pl.col("region") == "North"),
            Design(stratum="stratum", psu="psu", wgt="wgt"),
        )
        assert (
            filtered.estimates[0].est
            == subset.estimation.quantile("income", p=0.5).estimates[0].est
        )

    def test_monotone_in_probability(self, sample):
        """Quantiles must not decrease as p increases."""
        result = sample.estimation.quantile("income", p=(0.1, 0.25, 0.5, 0.75, 0.9))
        ests = [e.estimates[0].est for e in result]
        assert ests == sorted(ests)

    def test_serialization_round_trip(self, sample):
        from svy.serialize import serialize, to_json

        result = sample.estimation.quantile("income", p=(0.25, 0.75))
        data = serialize(result)

        assert data.kind == "estimate_list"
        assert [m.estimates[0].prob for m in data.estimates] == [0.25, 0.75]
        assert len(to_json(result)) > 0
