# tests/svy/estimation/test_contrast.py
"""Between-estimate covariance and post-estimation contrasts.

R reference values from survey 4.5:

```r
library(survey)
apiclus1 <- read.csv("tests/test_data/apiclus1.csv")
dclus1 <- svydesign(id = ~dnum, weights = ~pw, data = apiclus1)
m <- svyby(~api00, ~stype, dclus1, svymean, covmat = TRUE)
vcov(m); degf(dclus1)
svycontrast(m, list(diff = c(E = 1, H = -1), trend = c(E = -1, M = 0, H = 1)))
```

Each test carries its own R snippet.
"""

from pathlib import Path

import numpy as np
import pytest

import svy

from svy import Cat, Design, Sample, estd
from svy.core.repwgts import RepWeights
from svy.errors import MethodError


BASE_DIR = Path(__file__).parents[2]

# Reference values were captured from R at full double precision
# (sprintf %.15g), so agreement is asserted near machine precision — any
# looser tolerance would only be hiding a real methodological difference.
TOL = 1e-12


@pytest.fixture(scope="module")
def apiclus1():
    return svy.io.read_csv(BASE_DIR / "test_data/apiclus1.csv")


@pytest.fixture(scope="module")
def dclus1(apiclus1):
    return Sample(apiclus1, Design(psu="dnum", wgt="pw"))


def _cov(result, key_a, key_b):
    keys = result.keys()
    return result.covariance[keys.index(key_a), keys.index(key_b)]


# =============================================================================
# Taylor: covariance across domains
# =============================================================================


class TestTaylorMeanCovariance:
    """R: vcov(svyby(~api00, ~stype, dclus1, svymean, covmat=TRUE))"""

    R_VCOV = {
        ("E", "E"): 510.18671155248,
        ("E", "H"): 532.531931516402,
        ("E", "M"): 659.913660796955,
        ("H", "H"): 1474.76185517939,
        ("H", "M"): 671.639510204081,
        ("M", "M"): 1019.35690971428,
    }

    def test_full_covariance_matches_r(self, dclus1):
        r = dclus1.estimation.mean("api00", by="stype")
        for (a, b), want in self.R_VCOV.items():
            assert _cov(r, a, b) == pytest.approx(want, rel=TOL)
            assert _cov(r, b, a) == pytest.approx(want, rel=TOL)

    def test_diagonal_equals_se_squared(self, dclus1):
        r = dclus1.estimation.mean("api00", by="stype")
        for p in r.estimates:
            key = r._row_key(p)
            assert _cov(r, key, key) == pytest.approx(p.se**2, rel=1e-12)

    def test_design_df_is_full_design(self, dclus1):
        r = dclus1.estimation.mean("api00", by="stype")
        assert r.design_df == 14  # R degf(dclus1)

    def test_domains_populated(self, dclus1):
        r = dclus1.estimation.mean("api00", by="stype")
        assert sorted(r.domains) == ["E", "H", "M"]
        assert r.keys() == list(r.domains)


class TestTaylorMeanContrast:
    """R: svycontrast(m, list(diff=c(E=1,H=-1), trend=c(E=-1,M=0,H=1)))
    with confint/t on degf(dclus1) = 14."""

    R_DIFF_EST = 30.2966269841261
    R_DIFF_SE = 30.3296011134183
    R_DIFF_LCI = -34.7538977337905
    R_DIFF_UCI = 95.3471517020427
    R_DIFF_T = 0.998912807024104
    R_DIFF_P = 0.334790114360783

    def test_expression_form(self, dclus1):
        r = dclus1.estimation.mean("api00", by="stype")
        c = r.contrast(estd("E") - estd("H"))
        row = c.estimates[0]
        assert row.est == pytest.approx(self.R_DIFF_EST, rel=TOL)
        assert row.se == pytest.approx(self.R_DIFF_SE, rel=TOL)
        assert row.lci == pytest.approx(self.R_DIFF_LCI, rel=TOL)
        assert row.uci == pytest.approx(self.R_DIFF_UCI, rel=TOL)
        assert row.t == pytest.approx(self.R_DIFF_T, rel=TOL)
        assert row.p_value == pytest.approx(self.R_DIFF_P, rel=TOL)
        assert row.df == 14

    def test_dict_form_matches_expression(self, dclus1):
        r = dclus1.estimation.mean("api00", by="stype")
        a = r.contrast(estd("E") - estd("H")).estimates[0]
        b = r.contrast({"E": 1, "H": -1}).estimates[0]
        assert a.est == b.est
        assert a.se == b.se

    def test_named_multiple(self, dclus1):
        r = dclus1.estimation.mean("api00", by="stype")
        c = r.contrast({"diff": estd("E") - estd("H"), "trend": -estd("E") + estd("H")})
        assert [e.contrast for e in c.estimates] == ["diff", "trend"]
        assert c.estimates[0].est == pytest.approx(self.R_DIFF_EST, rel=TOL)
        assert c.estimates[1].est == pytest.approx(-self.R_DIFF_EST, rel=TOL)
        # LVL' off-diagonal: Cov(diff, -diff) = -Var(diff)
        assert c.covariance[0, 1] == pytest.approx(-(self.R_DIFF_SE**2), rel=TOL)

    def test_unknown_key_raises_listing_valid(self, dclus1):
        r = dclus1.estimation.mean("api00", by="stype")
        with pytest.raises(MethodError, match="Valid keys"):
            r.contrast(estd("X") - estd("H"))

    def test_scaling_and_division(self, dclus1):
        r = dclus1.estimation.mean("api00", by="stype")
        avg = (estd("E") + estd("H") + estd("M")) / 3
        c = r.contrast(avg)
        means = [p.est for p in r.estimates]
        assert c.estimates[0].est == pytest.approx(np.mean(means), rel=1e-12)

    def test_nonlinear_operations_raise(self, dclus1):
        r = dclus1.estimation.mean("api00", by="stype")
        with pytest.raises(MethodError, match="[Nn]onlinear"):
            estd("E") * estd("H")
        with pytest.raises(MethodError, match="[Nn]onlinear"):
            estd("E") / estd("H")
        with pytest.raises(MethodError, match="[Nn]onlinear"):
            estd("E") + 1.0
        del r


class TestTaylorMeanFpc:
    """R: dclus1f <- svydesign(id=~dnum, weights=~pw, fpc=~fpc, data=apiclus1)
    vcov(svyby(~api00, ~stype, dclus1f, svymean, covmat=TRUE))
    svycontrast(..., c(E=1, H=-1))"""

    def test_fpc_covariance_and_contrast(self, apiclus1):
        s = Sample(apiclus1, Design(psu="dnum", wgt="pw", pop_size="fpc"))
        r = s.estimation.mean("api00", by="stype")
        assert _cov(r, "E", "H") == pytest.approx(521.979779636949, rel=TOL)
        assert _cov(r, "E", "M") == pytest.approx(646.837432379578, rel=TOL)
        assert _cov(r, "H", "M") == pytest.approx(658.330933383656, rel=TOL)
        c = r.contrast(estd("E") - estd("H"))
        assert c.estimates[0].est == pytest.approx(30.2966269841261, rel=TOL)
        assert c.estimates[0].se == pytest.approx(30.0276061918455, rel=TOL)


class TestTaylorTotalContrast:
    """R: tt <- svyby(~enroll, ~stype, dclus1, svytotal, covmat=TRUE)
    svycontrast(tt, c(E=1, H=-1, M=-1))"""

    def test_total_covariance_and_contrast(self, dclus1):
        r = dclus1.estimation.total("enroll", by="stype")
        assert _cov(r, "E", "H") == pytest.approx(78730968177.1117, rel=TOL)
        assert _cov(r, "E", "M") == pytest.approx(76272913471.0846, rel=TOL)
        assert _cov(r, "H", "M") == pytest.approx(35480356677.0, rel=TOL)
        c = r.contrast(estd("E") - estd("H") - estd("M"))
        assert c.estimates[0].est == pytest.approx(814494.119140626, rel=TOL)
        assert c.estimates[0].se == pytest.approx(516348.003586747, rel=TOL)


class TestPropLevelCovariance:
    """R: p <- svymean(~factor(stype), dclus1); vcov(p)
    svycontrast(p, c(`factor(stype)E`=1, `factor(stype)H`=-1))"""

    def test_level_covariance(self, dclus1):
        r = dclus1.estimation.prop("stype")
        assert _cov(r, "E", "H") == pytest.approx(-0.00101348123237873, rel=TOL)
        assert _cov(r, "E", "M") == pytest.approx(-0.00117699947724718, rel=TOL)
        assert _cov(r, "H", "M") == pytest.approx(0.000280162635233734, rel=TOL)

    def test_level_contrast(self, dclus1):
        r = dclus1.estimation.prop("stype")
        c = r.contrast(estd("E") - estd("H"))
        assert c.estimates[0].est == pytest.approx(0.710382513661203, rel=TOL)
        assert c.estimates[0].se == pytest.approx(0.0703616498636037, rel=TOL)


class TestCalibrationInterplay:
    """Post-stratified design: the score sweep must run before the
    cross-products, or off-diagonals (and contrast SEs) come out wrong.

    R: pop.types <- data.frame(stype=c("E","H","M"), Freq=c(4421,755,1018))
    rclus1 <- postStratify(dclus1, ~stype, pop.types)
    mp <- svyby(~api00, ~comp.imp, rclus1, svymean, covmat=TRUE); vcov(mp)
    svycontrast(mp, c(Yes=1, No=-1))"""

    def test_poststratified_contrast(self, apiclus1):
        s = Sample(apiclus1, Design(psu="dnum", wgt="pw"))
        s = s.weighting.poststratify(controls={"E": 4421, "H": 755, "M": 1018}, cells="stype")
        r = s.estimation.mean("api00", by="comp.imp")
        assert _cov(r, "No", "No") == pytest.approx(804.417956450135, rel=TOL)
        assert _cov(r, "Yes", "Yes") == pytest.approx(561.917214477123, rel=TOL)
        assert _cov(r, "No", "Yes") == pytest.approx(555.69373386016, rel=TOL)
        c = r.contrast(estd("Yes") - estd("No"))
        assert c.estimates[0].est == pytest.approx(12.7920276905844, rel=TOL)
        assert c.estimates[0].se == pytest.approx(15.967081862599, rel=TOL)


# =============================================================================
# Replication (BRR)
# =============================================================================


class TestBrrCovariance:
    """R: brr <- subset(read.csv(...), !is.na(income))
    dbrr <- svrepdesign(weights=~weight, repweights=brr[, brr_cols],
                        type="BRR", data=brr, combined.weights=TRUE)
    mb <- svyby(~income, ~educ, dbrr, svymean, covmat=TRUE); vcov(mb)
    svycontrast(mb, c(High=1, Low=-1))"""

    @pytest.fixture()
    def brr_sample(self):
        import polars as pl

        data = (
            svy.io.read_csv(BASE_DIR / "test_data/fake_survey_brr_24122025.csv")
            .fill_nan(None)
            .drop_nulls()
            .with_columns(pl.col("income").cast(pl.Float64))
        )
        design = Design(
            row_index="id",
            wgt="weight",
            stratum="stratum",
            psu="psu",
            rep_wgts=RepWeights(method="BRR", prefix="brr_", n_reps=8, df=7),
        )
        return Sample(data, design)

    def test_brr_covariance_and_contrast(self, brr_sample):
        r = brr_sample.estimation.mean("income", by="educ", method="replication")
        assert _cov(r, "High", "Low") == pytest.approx(715074.595136997, rel=TOL)
        assert _cov(r, "High", "Med") == pytest.approx(760181.172921706, rel=TOL)
        assert _cov(r, "Low", "Med") == pytest.approx(389433.723528035, rel=TOL)
        c = r.contrast(estd("High") - estd("Low"))
        assert c.estimates[0].est == pytest.approx(8.57644489169616, rel=TOL)
        assert c.estimates[0].se == pytest.approx(1232.05113930133, rel=TOL)
        # Replication df stays the recorded one, not the PSU count.
        assert r.design_df == 7
        assert c.estimates[0].df == 7


# =============================================================================
# GLM: contrast + term test
# =============================================================================


class TestGlmContrast:
    """R: g <- svyglm(api00 ~ api99 + stype, dclus1)
    svycontrast(g, c(stypeH=1, stypeM=-1))
    regTermTest(g, ~stype, method="Wald")"""

    @pytest.fixture()
    def fit(self, apiclus1):
        s = Sample(
            apiclus1.with_row_index("row_id"),
            Design(psu="dnum", wgt="pw", row_index="row_id"),
        )
        return s.glm.fit(y="api00", x=["api99", Cat("stype")])

    def test_coefficient_contrast(self, fit):
        c = fit.contrast(estd("stype_H") - estd("stype_M"))
        assert c.estimates[0].est == pytest.approx(-1.22904588514547, rel=TOL)
        assert c.estimates[0].se == pytest.approx(6.40005680560589, rel=TOL)
        assert c.estimates[0].df == 11  # degf - (k-1) = 14 - 3

    def test_keys_are_coefficient_names(self, fit):
        assert fit.keys() == ["_intercept_", "api99", "stype_H", "stype_M"]

    def test_term_test_wald(self, fit):
        ft = fit.term_test("stype")
        assert ft.value == pytest.approx(9.26672931933068, rel=TOL)
        assert ft.df_num == 2
        assert ft.df_den == 11
        assert ft.p_value == pytest.approx(0.00437452662779401, rel=TOL)

    def test_term_test_accepts_cat(self, fit):
        assert fit.term_test(Cat("stype")).value == pytest.approx(fit.term_test("stype").value)

    def test_term_test_continuous(self, fit):
        # Single-column term: F = t² on (1, design df).
        ft = fit.term_test("api99")
        coef = next(c for c in fit.fitted.coefs if c.term == "api99")
        assert ft.value == pytest.approx(coef.wald.value**2, rel=1e-10)
        assert ft.df_num == 1

    def test_unknown_term_raises(self, fit):
        from svy.errors import ModelError

        with pytest.raises(ModelError, match="Unknown model term"):
            fit.term_test("nope")


# =============================================================================
# Behavior & guardrails
# =============================================================================


class TestContrastGuards:
    def test_quantile_refused(self, dclus1):
        r = dclus1.estimation.median("api00", by="stype")
        with pytest.raises(MethodError, match="quantile"):
            r.contrast(estd("E") - estd("H"))

    def test_multi_variable_prop_refused(self, dclus1):
        rs = dclus1.estimation.prop(["stype", "comp.imp"])
        with pytest.raises(MethodError, match="independently"):
            rs[0].contrast(estd("E") - estd("H"))

    def test_estimate_list_single_member_delegates(self, dclus1):
        lst = svy.EstimateList([dclus1.estimation.mean("api00", by="stype")])
        c = lst.contrast(estd("E") - estd("H"))
        assert c.estimates[0].est == pytest.approx(30.2966269841261, rel=TOL)

    def test_na_propagation(self, dclus1):
        r = dclus1.estimation.mean("api00", by="stype")
        # Poison one estimate: contrasts touching it go NA, others survive.
        import msgspec

        r.estimates[1] = msgspec.structs.replace(r.estimates[1], est=float("nan"))
        keys = r.keys()
        bad_key, good_a, good_b = keys[1], keys[0], keys[2]
        c = r.contrast(
            {
                "touches_na": estd(bad_key) - estd(good_a),
                "clean": estd(good_a) - estd(good_b),
            }
        )
        assert np.isnan(c.estimates[0].est)
        assert np.isfinite(c.estimates[1].est)

    def test_covariance_to_polars_lower_triangle(self, dclus1):
        r = dclus1.estimation.mean("api00", by="stype")
        tidy = r.covariance_to_polars()
        assert set(tidy.columns) == {"key_a", "key_b", "cov"}
        assert tidy.height == 6  # k(k+1)/2 for k=3

    def test_string_int_key_normalization(self, dclus1):
        r = dclus1.estimation.prop("both")  # levels "No"/"Yes"
        c1 = r.contrast(estd("Yes") - estd("No"))
        assert np.isfinite(c1.estimates[0].est)


class TestLabelResolution:
    """Contrast keys accept metadata value labels wherever unambiguous —
    what the printed table shows must be typeable."""

    @pytest.fixture()
    def labeled_result(self, dclus1):
        from svy.metadata import MetadataStore

        r = dclus1.estimation.mean("api00", by="stype")
        store = MetadataStore()
        store.set_value_labels("stype", {"E": "Elementary", "H": "High", "M": "Middle"})
        r.metadata = store
        return r

    def test_keys_with_labels(self, labeled_result):
        # Row order tracks the kernel's group order; the mapping is the contract.
        label_of = {"E": "Elementary", "H": "High", "M": "Middle"}
        raw = labeled_result.keys()
        assert sorted(raw) == ["E", "H", "M"]
        assert labeled_result.keys(labels=True) == [label_of[k] for k in raw]

    def test_label_keys_resolve(self, labeled_result):
        by_code = labeled_result.contrast(estd("E") - estd("H")).estimates[0]
        by_label = labeled_result.contrast(estd("Elementary") - estd("High")).estimates[0]
        assert by_label.est == by_code.est
        assert by_label.se == by_code.se

    def test_unknown_label_still_raises(self, labeled_result):
        with pytest.raises(MethodError, match="Valid keys"):
            labeled_result.contrast(estd("Kindergarten") - estd("High"))


class TestSingletonInterplay:
    """PSU-total retention must respect the active singleton strategy exactly
    as the variances do (spec open question 5)."""

    def test_center_strategy_runs(self, apiclus1):
        import polars as pl

        # Make a singleton PSU in its own stratum.
        data = apiclus1.with_columns(
            pl.when(pl.col("dnum") == pl.col("dnum").first())
            .then(pl.lit("S1"))
            .otherwise(pl.lit("S2"))
            .alias("strat")
        )
        s = Sample(data, Design(stratum="strat", psu="dnum", wgt="pw"))
        s = s.singleton.center()
        r = s.estimation.mean("api00", by="stype")
        # Sum-to-total identity: Var(sum of parts) from the covariance equals
        # itself — the matrix must at least be symmetric PSD-ish here.
        cov = r.covariance
        assert np.allclose(cov, cov.T)
        c = r.contrast(estd("E") - estd("H"))
        assert np.isfinite(c.estimates[0].se)
