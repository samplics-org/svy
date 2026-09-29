# tests/svy/core/test_singleton_domains.py
"""Strata with several PSUs but one inside an estimation domain.

The fixture (see test_singleton_adjustments.center_domain): strata 1-6 hold
3, 2, 2, 3, 1, 1 PSUs. Stratum 2's domain rows sit in PSU 1 for ``dom == "g1"``
(its first row has y = 0) and in PSU 2 for ``g2``; ``reg == 1`` is PSU 1 of
every stratum.

R golden values: options(survey.adjust.domain.lonely = TRUE) with
survey.lonely.psu = "adjust" (center) and "average" (scale), survey 4.5.
"""

import json
import math
import warnings

from pathlib import Path

import msgspec
import polars as pl
import pytest

import svy

from svy.core.design import SingletonSpec
from svy.core.singleton import find_domain_singletons
from svy.core.warnings import Severity
from svy.errors import MethodError
from svy.errors.singleton_errors import SingletonError
from svy.serialize import serialize, to_design


DATA = Path(__file__).resolve().parents[2] / "test_data" / "singleton_center_domain_29092026.csv"
G = [("g1",), ("g2",), ("g3",)]


@pytest.fixture
def data():
    return pl.read_csv(DATA).with_columns(b=(pl.col("y") > 4).cast(pl.Int64))


@pytest.fixture
def design():
    return svy.Design(stratum="stratum", psu="psu", wgt="wgt")


@pytest.fixture
def no_singletons(data, design):
    """Strata 1-4 only: every stratum has several PSUs."""
    return svy.Sample(data.filter(pl.col("stratum") <= 4), design)


def dom(g):
    return svy.col("dom") == g


def by_se(result):
    se = {e.by_level: e.se for e in result.estimates}
    return [se[k] for k in G]


# ══════════════════════════════════════════════════════════════════════════════
# SingletonSpec.domains
# ══════════════════════════════════════════════════════════════════════════════


class TestSpec:
    def test_default_is_warn(self):
        assert SingletonSpec.center([5]).domains == "warn"

    @pytest.mark.parametrize("method", ["center", "scale"])
    def test_apply_on_center_and_scale(self, method):
        spec = getattr(SingletonSpec, method)([5], domains="apply")
        assert spec.domains == "apply"

    @pytest.mark.parametrize("method", ["certainty", "skip", "pool"])
    def test_apply_refused_elsewhere(self, method):
        with pytest.raises(ValueError, match="center and scale define"):
            getattr(SingletonSpec, method)([5], domains="apply")
        with pytest.raises(ValueError, match="center and scale define"):
            SingletonSpec.collapse({5: 1}, domains="apply")

    def test_unknown_domains(self):
        with pytest.raises(ValueError, match="Unknown domains"):
            SingletonSpec.center([5], domains="adjust")

    @pytest.mark.parametrize("domains", ["ignore", "error", "apply"])
    def test_no_strata_only_with_a_domains_setting(self, domains):
        assert SingletonSpec.center(domains=domains).strata == ()
        assert SingletonSpec.collapse(domains="error").mapping == ()
        with pytest.raises(ValueError, match="non-empty strata"):
            SingletonSpec.center()
        with pytest.raises(ValueError, match="non-empty mapping"):
            SingletonSpec.collapse()

    def test_repr_and_code_name_domains_only_when_set(self):
        assert repr(SingletonSpec.center([5])) == "SingletonSpec.center([5])"
        assert repr(SingletonSpec.scale([5], domains="apply")) == (
            "SingletonSpec.scale([5], domains='apply')"
        )
        assert SingletonSpec.collapse({5: 1}, domains="ignore")._to_code() == (
            "svy.SingletonSpec.collapse({5: 1}, domains='ignore')"
        )
        assert SingletonSpec.pool([5, 6], name="p", domains="error")._to_code() == (
            "svy.SingletonSpec.pool([5, 6], name='p', domains='error')"
        )


class TestSerialization:
    def test_round_trip(self, data, design):
        sample = svy.Sample(data, design).singleton.center(domains="apply")
        back = to_design(serialize(sample.design))
        assert back.singleton == sample.design.singleton

    def test_default_is_not_saved(self, data, design):
        saved = msgspec.to_builtins(serialize(svy.Sample(data, design).singleton.skip().design))
        assert "domains" not in saved["singleton"]

    def test_payload_without_domains_reads_as_warn(self, data, design):
        saved = msgspec.to_builtins(serialize(svy.Sample(data, design).singleton.skip().design))
        payload = msgspec.json.encode(saved)
        back = to_design(msgspec.json.decode(payload, type=type(serialize(design))))
        assert back.singleton.domains == "warn"

    def test_to_code_rebuilds_the_rule(self, no_singletons):
        sample = no_singletons.singleton.center(domains="apply")
        assert "domains='apply'" in sample.to_code()


# ══════════════════════════════════════════════════════════════════════════════
# sample.singleton.*(domains=...)
# ══════════════════════════════════════════════════════════════════════════════


class TestFacet:
    @pytest.mark.parametrize(
        "method", ["certainty", "skip", "collapse", "pool", "scale", "center"]
    )
    def test_default_call_without_singletons_is_a_no_op(self, no_singletons, method):
        assert getattr(no_singletons.singleton, method)() is no_singletons

    @pytest.mark.parametrize(
        "method", ["certainty", "skip", "collapse", "pool", "scale", "center"]
    )
    @pytest.mark.parametrize("domains", ["ignore", "error"])
    def test_setting_is_recorded_without_singletons(self, no_singletons, method, domains):
        sample = getattr(no_singletons.singleton, method)(domains=domains)
        spec = sample.design.singleton
        assert (spec.method, spec.domains) == (method, domains)
        assert spec.handled == ()
        assert no_singletons.design.singleton is None

    @pytest.mark.parametrize("method", ["center", "scale"])
    def test_apply_recorded_without_singletons(self, no_singletons, method):
        spec = getattr(no_singletons.singleton, method)(domains="apply").design.singleton
        assert (spec.method, spec.domains) == (method, "apply")

    def test_setting_is_recorded_with_singletons(self, data, design):
        spec = svy.Sample(data, design).singleton.center(domains="apply").design.singleton
        assert set(spec.strata) == {5, 6}
        assert spec.domains == "apply"

    @pytest.mark.parametrize("method", ["certainty", "skip", "collapse", "pool"])
    def test_apply_refused_with_a_guiding_error(self, no_singletons, method):
        with pytest.raises(MethodError, match="domains") as err:
            getattr(no_singletons.singleton, method)(domains="apply")
        assert 'center(domains="apply")' in str(err.value)

    def test_unknown_domains_refused(self, no_singletons):
        with pytest.raises(MethodError, match="domains"):
            no_singletons.singleton.center(domains="yes")

    def test_rule_without_strata_cleared_when_singletons_appear(self, data, design):
        sample = svy.Sample(data.filter(pl.col("stratum") <= 4), design)
        sample = sample.singleton.center(domains="apply")
        sample = sample.wrangling.filter_records(~((pl.col("stratum") == 1) & (pl.col("psu") > 1)))
        # The rule is checked against the data when the sample is next used.
        with pytest.warns(UserWarning, match="singleton strata changed"):
            assert sample.design.singleton is None


# ══════════════════════════════════════════════════════════════════════════════
# Detection
# ══════════════════════════════════════════════════════════════════════════════


def _found(df, **kw):
    kw.setdefault("stratum_cols", ["stratum"])
    return [f.label for f in find_domain_singletons(df, strata_col="stratum", **kw)]


class TestDetection:
    def test_by_levels(self, data):
        assert _found(data, psu_col="psu", weight_col="wgt", by_col="dom", by_cols=["dom"]) == [
            "dom=g1: stratum=2",
            "dom=g2: stratum=2",
        ]

    def test_a_zero_value_row_still_places_its_psu(self, data):
        # g1's only row with y = 0 is in stratum 2's PSU 1: still one PSU.
        g1 = data.with_columns(w=pl.when(pl.col("dom") == "g1").then("wgt").otherwise(0.0))
        assert _found(g1, psu_col="psu", weight_col="w") == ["stratum=2"]

    def test_mask_and_zero_weights(self, data):
        # reg == 1 keeps PSU 1 of every stratum: all multi-PSU strata.
        assert _found(data, psu_col="psu", weight_col="wgt", mask=pl.col("reg") == 1) == [
            f"stratum={h}" for h in (1, 2, 3, 4)
        ]
        w = data.with_columns(w=pl.when(pl.col("reg") == 1).then("wgt").otherwise(0.0))
        assert _found(w, psu_col="psu", weight_col="w") == [f"stratum={h}" for h in (1, 2, 3, 4)]

    def test_null_and_nan_weights_are_not_domain_rows(self, data):
        w = data.with_columns(
            w=pl.when(pl.col("reg") == 1).then("wgt").otherwise(None),
            v=pl.when(pl.col("reg") == 1).then("wgt").otherwise(float("nan")),
        )
        assert len(_found(w, psu_col="psu", weight_col="w")) == 4
        assert len(_found(w, psu_col="psu", weight_col="v")) == 4

    def test_element_design_counts_rows(self, data):
        # Without PSUs a stratum with one domain row among several is lonely.
        one = data.with_columns(
            w=pl.when(pl.int_range(pl.len()).over("stratum") == 0).then("wgt").otherwise(0.0)
        )
        assert _found(one, psu_col=None, weight_col="w") == [
            f"stratum={h}" for h in (1, 2, 3, 4, 5, 6)
        ]

    def test_full_sample_singletons_are_not_domain_singletons(self, data):
        assert _found(data, psu_col="psu", weight_col="wgt") == []

    def test_by_column_that_is_the_stratum(self, data, design):
        # by="stratum" names the same column twice.
        r = (
            svy.Sample(data, design)
            .singleton.center()
            .estimation.mean("y", by="stratum", where=svy.col("reg") == 1)
        )
        assert len(r.findings[0].extra["pairs"]) == 4

    def test_tuple_strata_named_by_their_columns(self, data):
        d = data.with_columns(s2=pl.lit("x"))
        found = find_domain_singletons(
            d,
            strata_col="stratum",
            psu_col="psu",
            weight_col="wgt",
            stratum_cols=["stratum", "s2"],
            by_col="dom",
            by_cols=["dom"],
        )
        assert found[0].label == "dom=g1: stratum=2, s2=x"

    def test_calibrated_design_has_none(self, data, design):
        sample = svy.Sample(data, design).singleton.center()
        ps = sample.weighting.poststratify(controls={"u": 700.0, "v": 400.0}, cells="ps")
        assert ps.estimation.total("y", by="dom").findings == []

    def test_unstratified_design_has_none(self, data):
        sample = svy.Sample(data, svy.Design(psu="psu", wgt="wgt"))
        assert sample.estimation.total("y", by="dom").findings == []


# ══════════════════════════════════════════════════════════════════════════════
# domains= ignore / warn / error / apply
# ══════════════════════════════════════════════════════════════════════════════


class TestPolicy:
    def test_warn_records_a_finding_and_a_note_not_a_python_warning(self, data, design):
        sample = svy.Sample(data, design).singleton.center()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            r = sample.estimation.mean("y", by="dom")
        (f,) = r.findings
        assert f.code == "DOMAIN_SINGLETON_PSU" and f.level == Severity.WARNING
        assert f.extra["pairs"] == ["dom=g1: stratum=2", "dom=g2: stratum=2"]
        text = str(r)
        assert "note: 1 stratum has a single PSU within 2 domains" in " ".join(text.split())
        assert 'sample.singleton.center(domains="apply")' in r.__plain_str__()
        kept = [w for w in sample.warnings if w.code == "DOMAIN_SINGLETON_PSU"]
        assert kept and kept[-1].level == Severity.INFO

    def test_warn_uses_the_standard_variance(self, data, design):
        warn = svy.Sample(data, design).singleton.center().estimation.total("y", by="dom")
        ignore = svy.Sample(data, design).singleton.center(domains="ignore")
        ignored = ignore.estimation.total("y", by="dom")
        assert by_se(warn) == by_se(ignored)
        assert ignored.findings == []
        assert "note:" not in ignored.__plain_str__()

    def test_no_rule_behaves_like_warn(self, no_singletons):
        r = no_singletons.estimation.mean("y", by="dom")
        assert r.findings[0].level == Severity.WARNING
        assert "note:" in r.__plain_str__()

    def test_error_raises_before_estimating(self, data, design):
        sample = svy.Sample(data, design).singleton.center(domains="error")
        with pytest.raises(SingletonError) as err:
            sample.estimation.mean("y", by="dom")
        assert err.value.code == "DOMAIN_SINGLETON"
        assert "dom=g1: stratum=2" in str(err.value)
        assert 'center(domains="apply")' in str(err.value)
        # Nothing to find, nothing raised.
        assert sample.estimation.mean("y").estimates

    @pytest.mark.parametrize(
        "method, how", [("center", "centered at the grand mean"), ("scale", "left out")]
    )
    def test_apply_notes_what_it_did(self, data, design, method, how):
        sample = getattr(svy.Sample(data, design).singleton, method)(domains="apply")
        r = sample.estimation.mean("y", by="dom")
        assert r.findings[0].level == Severity.INFO
        assert how in r.__plain_str__()

    def test_one_note_per_call(self, data, design):
        # y2 is missing on g2's rows of stratum 3: that variable finds one more pair.
        sample = svy.Sample(data, design).singleton.center()
        r = sample.estimation.total(
            ["y", "y2"], by="dom", where=svy.col("reg") == 1, drop_nulls=True
        )
        notes = [ln for ln in r.__plain_str__().splitlines() if ln.startswith("note:")]
        assert len(notes) == 1
        assert "and " in notes[0] and "more" in notes[0]

    def test_findings_are_plain_in_glm_to_dict(self, data, design):
        fit = svy.Sample(data, design).singleton.center().glm.fit("y", x=["x"], where=dom("g1"))
        d = fit.fitted.to_dict()
        json.dumps(d)
        assert d["findings"][0]["code"] == "DOMAIN_SINGLETON_PSU"


@pytest.mark.parametrize("domains", ["warn", "apply"])
def test_every_analysis_reports_the_domain(data, design, domains):
    sample = svy.Sample(data, design).singleton.center(domains=domains)
    c = sample.categorical
    results = [
        sample.estimation.prop("cat", where=dom("g1")),
        sample.estimation.ratio("y", "x", by="dom"),
        sample.estimation.median("y", where=dom("g1")),
        sample.estimation.corr(("y", "x"), where=dom("g1")),
        c.ttest("y", mean_h0=4, where=dom("g1")),
        c.ttest("y", group="reg", by="dom"),
        c.ranktest("y", group="reg", method="kruskal-wallis", where=dom("g1")),
        c.ranktest("y", group="reg", score_fn=lambda r, n: r / n, where=dom("g1")),
        c.tabulate("cat", where=dom("g1")),
        c.tabulate("cat", "b", where=dom("g2")),
        sample.glm.fit("y", x=["x"], where=dom("g1")).fitted,
    ]
    for r in results:
        assert [f.code for f in r.findings] == ["DOMAIN_SINGLETON_PSU"], type(r).__name__
        assert "note:" in r.__plain_str__(), type(r).__name__
        assert "note:" in str(r), type(r).__name__


# ══════════════════════════════════════════════════════════════════════════════
# domains="apply" against R
# ══════════════════════════════════════════════════════════════════════════════

R_APPLY = {
    "center": {
        "total_by": [393.509424845, 325.941690928, 174.583129602],
        "mean_by": [0.574407603207, 1.34152313442, 0.974555800635],
        "total_reg1": 679.196967013,
        "mean_reg1_by": [0.456885382815, 0.436489159937, 0.300438983715],
        "ratio_by": [0.0315571957517, 0.0665225719832, 0.0626878079167],
        "prop_g1": [0.10701251172, 0.14482520717, 0.134551010643],
        # (g1, g2, g3) per category p, q, r
        "cattot_by": {
            "p": [50.4779709929, 48.2287241663, 2.55168789453],
            "q": [50.4592379189, 21.50123579, 10.9480591887],
            "r": [48.2631508337, 46.4974663844, 30.5179364382],
        },
        "median_g2": (5.2, 0.24, math.nan),
        "ttest1_g1": 0.639050856461,
        "ttest2_g2": -2.29664018556,
        "chisq_g2": (9.16148629688, 1.14117004586, 5.70585022928),
        "glm_g1": [1.07969943422, 0.119030868399],
        "rank_g2": -1.9294428952,
    },
    "scale": {
        "total_by": [240.399022748, 331.738708366, 245.295342394],
        "mean_by": [0.76067710675, 0.712412281346, 1.00925501928],
        "total_reg1": math.nan,
        "mean_reg1_by": [math.nan] * 3,
        "ratio_by": [0.0392813087634, 0.0270992773207, 0.0375090276583],
        "prop_g1": [0.101248110478, 0.185122524193, 0.153858184877],
        "cattot_by": {
            "p": [33.1544868758, 51.654138266, 1.97989898732],
            "q": [62.1443480938, 0.0, 12.0208152802],
            "r": [50.5365214474, 55.0976406028, 21.0717820794],
        },
        "median_g2": (5.2, 1.62, 13.29),
        "ttest1_g1": 0.482564372622,
        "ttest2_g2": -3.87573610427,
        "chisq_g2": (6.26912669626, 1.05024962136, 5.2512481068),
        "glm_g1": [0.595272065677, 0.0640923211321],
        "rank_g2": -2.77450717629,
    },
}


def _approx(x):
    return pytest.approx(x, nan_ok=True, abs=1e-9, rel=1e-8)


@pytest.mark.parametrize("method", ["center", "scale"])
def test_verify_apply_against_r(data, design, method):
    """options(survey.lonely.psu = "adjust" | "average", survey.adjust.domain.lonely = TRUE)
    d <- svydesign(ids = ~psu, strata = ~stratum, weights = ~wgt, nest = TRUE, data = df)
    svyby(~y, ~dom, d, svytotal); svyby(~y, ~dom, d, svymean)
    svytotal(~y, subset(d, reg == 1)); svyby(~y, ~dom, subset(d, reg == 1), svymean)
    svyby(~y, ~dom, d, svyratio, denominator = ~x)
    svymean(~factor(cat), subset(d, dom == "g1")); svyby(~factor(cat), ~dom, d, svytotal)
    svyquantile(~y, subset(d, dom == "g2"), 0.5, ci = TRUE, qrule = "math")
    svyttest(I(y - 4) ~ 0, subset(d, dom == "g1")); svyttest(y ~ factor(reg), subset(d, dom == "g2"))
    svychisq(~cat + b, subset(d, dom == "g2"), statistic = "F")
    svyglm(y ~ x, subset(d, dom == "g1")); svyranktest(y ~ factor(reg), subset(d, dom == "g2"))
    """
    r = R_APPLY[method]
    sample = getattr(svy.Sample(data, design).singleton, method)(domains="apply")
    e, c = sample.estimation, sample.categorical
    reg1 = svy.col("reg") == 1

    assert by_se(e.total("y", by="dom")) == _approx(r["total_by"])
    assert by_se(e.mean("y", by="dom")) == _approx(r["mean_by"])
    assert e.total("y", where=reg1).estimates[0].se == _approx(r["total_reg1"])
    assert by_se(e.mean("y", by="dom", where=reg1)) == _approx(r["mean_reg1_by"])
    assert by_se(e.ratio("y", "x", by="dom")) == _approx(r["ratio_by"])
    assert [x.se for x in e.prop("cat", where=dom("g1")).estimates] == _approx(r["prop_g1"])
    cat = {
        (x.by_level, x.y_level): x.se for x in e.total("cat", by="dom", as_factor=True).estimates
    }
    for level, ses in r["cattot_by"].items():
        assert [cat[(k, level)] for k in G] == _approx(ses)
    q = e.median("y", where=dom("g2")).estimates[0]
    assert (q.est, q.lci, q.uci) == _approx(r["median_g2"])

    assert c.ttest("y", mean_h0=4, where=dom("g1")).stats.t == _approx(r["ttest1_g1"])
    assert c.ttest("y", group="reg", where=dom("g2")).stats.t == _approx(r["ttest2_g2"])
    f = c.tabulate("cat", "b", where=dom("g2")).stats.f
    assert (f.value, f.df_num, f.df_den) == _approx(r["chisq_g2"])
    tab = c.tabulate("cat", where=dom("g1"))
    assert [x.se for x in tab.estimates] == _approx(r["prop_g1"])
    glm = sample.glm.fit("y", x=["x"], where=dom("g1")).fitted
    assert [k.se for k in glm.coefs] == _approx(r["glm_g1"])
    rank = c.ranktest("y", group="reg", method="kruskal-wallis", where=dom("g2"))
    assert rank.stats.value == _approx(r["rank_g2"])


def test_apply_without_full_sample_singletons(data, design):
    """center(domains="apply") set preventively on a design without singletons.

    d <- svydesign(..., data = df[df$stratum <= 4, ]); options(survey.lonely.psu = "adjust")
    svyby(~y, ~dom, d, svytotal)  # survey.adjust.domain.lonely FALSE, then TRUE
    """
    sample = svy.Sample(data.filter(pl.col("stratum") <= 4), design)
    standard = sample.estimation.total("y", by="dom")
    assert by_se(standard) == _approx([389.641389363, 265.579019489, 173.45])
    applied = sample.singleton.center(domains="apply").estimation.total("y", by="dom")
    assert by_se(applied) == _approx([391.734795063, 327.11788508, 173.45])
