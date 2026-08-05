"""
Tests for quantile estimation in svy (issue #112).

Standard errors follow Woodruff (1952): the design-based variance of the
estimated proportion P(Y <= q) is taken on the probability scale, and the
interval comes from inverting the weighted CDF at ``p +/- t*se_p``.

Reference values come from R's ``oldsvyquantile`` with ``interval.type="Wald"``,
which is the same construction. svy's ``q_method`` maps onto R's ``method``/``f``:

    q_method="higher"  <->  method="constant", f=1
    q_method="linear"  <->  method="linear"

Point estimates and both confidence limits agree with R **exactly** — every
value below is bit-identical. Standard errors agree to ~1e-15 relative, and the
residual is not the survey math: ``se`` is the back-solved half-width
``(uci - lci) / (2t)``, and scipy's ``t.ppf(0.975, 190)`` differs from R's
``qt(0.975, 190)`` by 1.0e-15 relative. Hence the 1e-12 tolerance below, which
still leaves three orders of magnitude of headroom.

The fixture deliberately stresses the estimator: 5000 records, 10 strata x 20
PSUs (df = 190), skewed income spanning 1.9k to 536k, and continuous unequal
weights spanning a 425x range with 4997 distinct values. The extreme
probabilities (0.001, 0.999) exercise the sparse tails where the CDF inversion
rule matters most, and 0.0345 / 0.679 are deliberately off-grid.

R Setup Code:
-------------
```r
options(digits = 15)
library(survey)

d <- read.csv("packages/svy/tests/test_data/quantile_ref_20260805.csv")
des <- svydesign(id = ~psu, strata = ~stratum, weights = ~wgt, data = d, nest = TRUE)
degf(des)   # 190

probs <- c(0.001, 0.01, 0.0345, 0.1, 0.25, 0.5, 0.679, 0.75, 0.9, 0.99, 0.999)

# R_HIGHER: method="constant", f=1   |   R_LINEAR: method="linear"
for (p in probs) {
    q <- oldsvyquantile(~income, des, quantiles = p, ci = TRUE,
                        interval.type = "Wald", method = "constant", f = 1)
    cat(sprintf("%g %.15g %.15g %.15g %.15g\n",
        p, coef(q), SE(q), confint(q)[1], confint(q)[2]))
}

# R_BY_REGION
for (rg in c("North", "South")) {
    for (p in c(0.1, 0.25, 0.5, 0.75, 0.9)) {
        oldsvyquantile(~income, subset(des, region == rg), quantiles = p, ci = TRUE,
                       interval.type = "Wald", method = "constant", f = 1)
    }
}
```
"""

from pathlib import Path

import polars as pl
import pytest

from svy import Design, Sample
from svy.core.enumerations import PopParam
from svy.estimation import EstimateList


# Point estimates and confidence limits match R exactly; SEs carry only the
# scipy-vs-R t-quantile difference (~1e-15).
REL = 1e-12

FIXTURE = Path(__file__).resolve().parents[2] / "test_data" / "quantile_ref_20260805.csv"


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


# R: oldsvyquantile(..., interval.type="Wald", method="constant", f=1)
# (prob, est, se, lci, uci)
R_HIGHER = [
    (0.001, 2453.7738, 213.058781027707, 1884.0742, 2724.6031),
    (0.01, 4400.0454, 309.189448122974, 3746.2698, 4966.0396),
    (0.0345, 6732.2575, 363.915079414323, 5895.4843, 7331.1498),
    (0.1, 10884.6084, 466.970514492444, 10008.7569, 11850.9819),
    (0.25, 18668.346, 647.96767501857, 17512.9969, 20069.2659),
    (0.5, 33429.9866, 951.332516879979, 31452.3191, 35205.3795),
    (0.679, 46769.1758, 1280.00883487419, 44568.8496, 49618.5566),
    (0.75, 54394.0689, 1448.50196619299, 51534.3152, 57248.7371),
    (0.9, 91935.9034, 4311.02884997682, 82793.11, 99800.3618),
    (0.99, 187573.6841, 13035.146663361, 175198.0666, 226622.4549),
    (0.999, 329662.0411, 60382.1082440284, 297789.5347, 536000.3551),
]

# R: oldsvyquantile(..., interval.type="Wald", method="linear")
R_LINEAR = [
    (0.001, 2453.72704364921, 198.621982194827, 1903.09341254061, 2686.66832742913),
    (0.01, 4396.7486276543, 304.082344033955, 3759.85592845707, 4959.47791496906),
    (0.0345, 6724.26363172486, 366.361944026794, 5883.45354369663, 7328.77206250791),
    (0.1, 10881.6789867565, 466.616284466675, 10002.3754831598, 11843.2030257423),
    (0.25, 18659.0719519676, 647.820558592866, 17511.7229625745, 20067.411579983),
    (0.5, 33427.4974064113, 950.170122774319, 31444.011999696, 35192.486689432),
    (0.679, 46763.5698205693, 1277.212283491, 44560.7738211196, 49599.4482682881),
    (0.75, 54391.3922866024, 1444.97993749059, 51531.6430418443, 57232.1703400977),
    (0.9, 91931.3113738885, 4199.47078591178, 82742.5269409024, 99309.6758903069),
    (0.99, 187465.615949273, 13034.1119176631, 175182.646153341, 226602.95232324),
    (0.999, 329485.233713584, 64601.8585499204, 281142.381901239, 536000.3551),
]

# R: per-domain via subset(des, region == ...), method="constant", f=1
# (region, prob, est, se)
R_BY_REGION = [
    ("North", 0.1, 11291.8521, 832.885289511911),
    ("North", 0.25, 18968.7621, 1127.72739273457),
    ("North", 0.5, 34150.2656, 1700.52818455268),
    ("North", 0.75, 55752.9531, 2791.97633349401),
    ("North", 0.9, 95245.4547, 6984.59821460462),
    ("South", 0.1, 10501.6883, 768.451836836439),
    ("South", 0.25, 18557.5425, 1211.78278106246),
    ("South", 0.5, 32525.2997, 1781.16202603899),
    ("South", 0.75, 52340.9449, 2388.09701981651),
    ("South", 0.9, 85969.1396, 4649.84105888204),
]


# ============================================================================
# Agreement with R
# ============================================================================


class TestMatchesR:
    @pytest.mark.parametrize("prob, est, se, lci, uci", R_HIGHER)
    def test_higher_matches_r(self, sample, prob, est, se, lci, uci):
        """q_method='higher' reproduces R's method='constant', f=1."""
        result = sample.estimation.quantile("income", p=prob, q_method="higher")
        got = result.estimates[0]

        assert got.est == pytest.approx(est, rel=REL)
        assert got.se == pytest.approx(se, rel=REL)
        assert got.lci == pytest.approx(lci, rel=REL)
        assert got.uci == pytest.approx(uci, rel=REL)
        assert got.df == 190

    @pytest.mark.parametrize("prob, est, se, lci, uci", R_LINEAR)
    def test_linear_matches_r(self, sample, prob, est, se, lci, uci):
        """q_method='linear' reproduces R's method='linear'.

        The interval endpoints must be inverted with the *same* rule as the
        point estimate — R hands one method/f pair to both its point
        ``approxfun`` and its endpoint ``approx``. Inverting linearly while
        estimating with 'higher' shrinks the interval, badly so in the tails.
        """
        result = sample.estimation.quantile("income", p=prob, q_method="linear")
        got = result.estimates[0]

        assert got.est == pytest.approx(est, rel=REL)
        assert got.se == pytest.approx(se, rel=REL)
        assert got.lci == pytest.approx(lci, rel=REL)
        assert got.uci == pytest.approx(uci, rel=REL)

    def test_domains_match_r(self, sample):
        """by= domains reproduce R's subset() estimates."""
        results = sample.estimation.quantile("income", p=(0.1, 0.25, 0.5, 0.75, 0.9), by="region")

        got = {
            (e.by_level[0], round(float(e.prob), 2)): e for est in results for e in est.estimates
        }
        assert len(got) == len(R_BY_REGION)

        for region, prob, est, se in R_BY_REGION:
            row = got[(region, prob)]
            assert row.est == pytest.approx(est, rel=REL), f"{region} p={prob}"
            assert row.se == pytest.approx(se, rel=REL), f"{region} p={prob}"

    def test_tail_probabilities(self, sample):
        """Extreme probabilities stay inside the observed support."""
        data = make_quantile_data()
        lo, hi = data["income"].min(), data["income"].max()

        for prob in (0.01, 0.99):
            got = sample.estimation.quantile("income", p=prob).estimates[0]
            assert lo <= got.est <= hi
            assert lo <= got.lci <= got.est <= got.uci <= hi


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
