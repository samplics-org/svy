"""
Tests for quantile estimation in svy (issue #112).

Standard errors follow Woodruff (1952): the design-based variance of the
estimated proportion P(Y <= q) is taken on the probability scale, and the
interval comes from inverting the weighted CDF at ``p +/- t*se_p``.

The reference values below come from R's ``oldsvyquantile`` with
``interval.type="Wald"``, which is the same construction. svy's ``q_method``
maps onto R's ``method``/``f`` pair:

    q_method="higher"  <->  method="constant", f=1
    q_method="linear"  <->  method="linear"

R Setup Code:
-------------
```r
library(survey)

d <- read.csv("fixture.csv")   # the frame built by make_quantile_data() below
des <- svydesign(id = ~psu, strata = ~stratum, weights = ~wgt, data = d, nest = TRUE)
degf(des)   # 12

for (p in c(0.1, 0.25, 0.5, 0.75, 0.9)) {
    q <- oldsvyquantile(~income, des, quantiles = p, ci = TRUE,
                        interval.type = "Wald", method = "constant", f = 1)
    print(c(coef(q), SE(q), confint(q)))
}

# Domains
for (rg in c("North", "South")) {
    oldsvyquantile(~income, subset(des, region == rg), quantiles = 0.5, ci = TRUE,
                   interval.type = "Wald", method = "constant", f = 1)
}
```
"""

import polars as pl
import pytest

from svy import Design, Sample
from svy.core.enumerations import PopParam
from svy.estimation import EstimateList


# ============================================================================
# Test Dataset
# ============================================================================


def make_quantile_data() -> pl.DataFrame:
    """Deterministic 80-row design: 4 strata x 4 PSUs x 5 units, df = 12."""
    rows = []
    for si, stratum in enumerate("ABCD"):
        for psu in range(4):
            for unit in range(5):
                rows.append(
                    {
                        "stratum": stratum,
                        "psu": f"{stratum}{psu}",
                        "income": 10000
                        + 2500 * si
                        + 1700 * psu
                        + 900 * unit
                        + 130 * ((si * 7 + psu * 3 + unit * 11) % 13),
                        "wgt": 1.0 + 0.25 * ((si + psu + unit) % 5),
                        "region": "North" if (psu + unit) % 2 == 0 else "South",
                    }
                )
    return pl.DataFrame(rows)


@pytest.fixture
def sample() -> Sample:
    return Sample(
        make_quantile_data(),
        Design(stratum="stratum", psu="psu", wgt="wgt"),
    )


# R: oldsvyquantile(..., interval.type="Wald", method="constant", f=1)
# (prob, est, se, lci, uci)
R_HIGHER = [
    (0.10, 14180.000000, 1252.975916, 10000.000000, 15460.000000),
    (0.25, 15900.000000, 904.162108, 14050.000000, 17990.000000),
    (0.50, 18540.000000, 736.639687, 17420.000000, 20630.000000),
    (0.75, 21470.000000, 1076.274184, 19310.000000, 24000.000000),
    (0.90, 23760.000000, 1177.246602, 22240.000000, 27370.000000),
]

# R: oldsvyquantile(..., interval.type="Wald", method="linear")
R_LINEAR = [
    (0.10, 14136.666667, 1248.405948, 10000.000000, 15440.085792),
    (0.25, 15842.857143, 969.404522, 13737.455091, 17961.757110),
    (0.50, 18517.142857, 743.926890, 17321.477035, 20563.231942),
    (0.75, 21470.000000, 1079.828430, 19280.108457, 23985.596530),
    (0.90, 23760.000000, 1205.897832, 22115.148664, 27370.000000),
]

# R: per-domain via subset(des, region == ...), method="constant", f=1
# (region, prob, est, se)
R_BY_REGION = [
    ("North", 0.25, 15700.000000, 1076.274184),
    ("North", 0.50, 18540.000000, 844.496588),
    ("North", 0.75, 21270.000000, 1076.274184),
    ("South", 0.25, 16270.000000, 727.460376),
    ("South", 0.50, 18740.000000, 858.265554),
    ("South", 0.75, 21910.000000, 1122.170738),
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

        assert got.est == pytest.approx(est, rel=1e-9)
        assert got.se == pytest.approx(se, rel=1e-9)
        assert got.lci == pytest.approx(lci, rel=1e-9)
        assert got.uci == pytest.approx(uci, rel=1e-9)
        assert got.df == 12

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

        assert got.est == pytest.approx(est, rel=1e-9)
        assert got.se == pytest.approx(se, rel=1e-9)
        assert got.lci == pytest.approx(lci, rel=1e-9)
        assert got.uci == pytest.approx(uci, rel=1e-9)

    def test_domains_match_r(self, sample):
        """by= domains reproduce R's subset() estimates."""
        results = sample.estimation.quantile("income", p=(0.25, 0.50, 0.75), by="region")

        got = {
            (e.by_level[0], round(float(e.prob), 2)): e for est in results for e in est.estimates
        }
        assert len(got) == len(R_BY_REGION)

        for region, prob, est, se in R_BY_REGION:
            row = got[(region, prob)]
            assert row.est == pytest.approx(est, rel=1e-9), f"{region} p={prob}"
            assert row.se == pytest.approx(se, rel=1e-9), f"{region} p={prob}"

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
