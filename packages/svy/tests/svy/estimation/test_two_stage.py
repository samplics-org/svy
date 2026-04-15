# tests/test_two_stage_variance.py
"""
Tests for two-stage variance estimation.

Validates against R's survey package results:
```r
library(survey)
data(api)

# Two-stage cluster design
dclus2 <- svydesign(id = ~dnum + snum, weights = ~pw, data = apiclus2)
svymean(~api00, dclus2)
#>         mean     SE
#> api00 670.81 30.099

confint(svymean(~api00, dclus2))
#>          2.5 %   97.5 %
#> api00 609.9285 731.6951
```
"""

from pathlib import Path

import pytest

import svy


BASE_DIR = Path(__file__).parents[2]


@pytest.fixture
def apiclus2_sample():
    """Load apiclus2 dataset and create two-stage sample."""
    apiclus2 = svy.io.read_csv(BASE_DIR / "test_data/apiclus2.csv")
    design = svy.Design(psu="dnum", ssu="snum", wgt="pw")
    return svy.Sample(data=apiclus2, design=design)


@pytest.fixture
def apiclus2_one_stage_sample():
    """Load apiclus2 dataset with one-stage design (for comparison)."""
    apiclus2 = svy.io.read_csv(BASE_DIR / "test_data/apiclus2.csv")
    design = svy.Design(psu="dnum", wgt="pw")
    return svy.Sample(data=apiclus2, design=design)


class TestTwoStageMean:
    """Tests for mean estimation with two-stage designs."""

    # Expected values from R's survey package
    R_MEAN_API00 = 670.8118
    R_SE_API00 = 30.099  # R's two-stage SE
    R_LCI_API00 = 609.9285
    R_UCI_API00 = 731.6951

    def test_mean_estimate(self, apiclus2_sample):
        """Test that point estimate matches R."""
        result = apiclus2_sample.estimation.mean("api00")
        est = result.estimates[0].est

        assert est == pytest.approx(self.R_MEAN_API00, rel=1e-4)

    def test_mean_se_two_stage(self, apiclus2_sample):
        """Test that two-stage SE matches R."""
        result = apiclus2_sample.estimation.mean("api00")
        se = result.estimates[0].se

        # Two-stage SE should be close to R's value
        assert se == pytest.approx(self.R_SE_API00, rel=0.05)

    def test_mean_ci_two_stage(self, apiclus2_sample):
        """Test that confidence interval matches R."""
        result = apiclus2_sample.estimation.mean("api00")
        lci = result.estimates[0].lci
        uci = result.estimates[0].uci

        assert lci == pytest.approx(self.R_LCI_API00, rel=0.05)
        assert uci == pytest.approx(self.R_UCI_API00, rel=0.05)

    def test_two_stage_vs_one_stage_se(self, apiclus2_sample, apiclus2_one_stage_sample):
        """
        Test that two-stage SE equals one-stage SE (ultimate cluster variance).

        Without FPC, R's survey package uses ultimate cluster variance estimation,
        which only uses first-stage PSUs. This is the expected behavior.
        """
        result_two_stage = apiclus2_sample.estimation.mean("api00")
        result_one_stage = apiclus2_one_stage_sample.estimation.mean("api00")

        se_two_stage = result_two_stage.estimates[0].se
        se_one_stage = result_one_stage.estimates[0].se

        # Point estimates should be identical
        assert result_two_stage.estimates[0].est == pytest.approx(
            result_one_stage.estimates[0].est, rel=1e-10
        )

        # SEs should be identical (ultimate cluster approach)
        assert se_two_stage == pytest.approx(se_one_stage, rel=1e-10)


class TestTwoStageTotal:
    """Tests for total estimation with two-stage designs."""

    def test_total_estimate(self, apiclus2_sample):
        """Test total estimation with two-stage design."""
        result = apiclus2_sample.estimation.total("api00")

        # Should produce a valid estimate
        assert result.estimates[0].est > 0
        assert result.estimates[0].se > 0


class TestTwoStageRatio:
    """Tests for ratio estimation with two-stage designs."""

    def test_ratio_estimate(self, apiclus2_sample):
        """Test ratio estimation with two-stage design."""
        result = apiclus2_sample.estimation.ratio(y="api00", x="api99")

        # Should produce a valid estimate close to 1 (api00 ~ api99)
        assert result.estimates[0].est == pytest.approx(1.0, rel=0.1)
        assert result.estimates[0].se > 0


class TestTwoStageProportion:
    """Tests for proportion estimation with two-stage designs."""

    def test_proportion_estimate(self, apiclus2_sample):
        """Test proportion estimation with two-stage design."""
        result = apiclus2_sample.estimation.prop("sch.wide")  # prop, not proportion

        # Should produce valid proportions
        for est in result.estimates:
            assert 0 <= est.est <= 1
            assert est.se > 0


class TestTwoStageByDomain:
    """Tests for domain estimation with two-stage designs."""

    def test_mean_by_domain(self, apiclus2_sample):
        """Test mean estimation by domain with two-stage design."""
        result = apiclus2_sample.estimation.mean("api00", by="stype")

        # Should have estimates for each school type
        assert len(result.estimates) >= 2

        for est in result.estimates:
            assert est.est > 0
            assert est.se > 0
            assert est.by == ("stype",)  # by returns a tuple


class TestDesignWithSSU:
    """Tests for Design class with SSU specification."""

    def test_design_with_ssu(self):
        """Test that Design accepts ssu parameter."""
        design = svy.Design(psu="dnum", ssu="snum", wgt="pw")

        assert design.psu == "dnum"
        assert design.ssu == "snum"
        assert design.wgt == "pw"

    def test_design_without_ssu(self):
        """Test that Design works without ssu (one-stage)."""
        design = svy.Design(psu="dnum", wgt="pw")

        assert design.psu == "dnum"
        assert design.ssu is None
        assert design.wgt == "pw"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
