# tests/svy/core/test_singleton_domains.py
"""Strata with several PSUs but one inside an estimation domain.

The fixture (see test_singleton_adjustments.center_domain): strata 1-6 hold
3, 2, 2, 3, 1, 1 PSUs. Stratum 2's domain rows sit in PSU 1 for ``dom == "g1"``
(its first row has y = 0) and in PSU 2 for ``g2``; ``reg == 1`` is PSU 1 of
every stratum.

``svy.Singleton(..., domains=...)`` picks the variance formula ("standard" or
"apply"), ``on_domain_singletons=`` what svy reports ("ignore", the default,
records the finding; "warn" prints a note; "error" raises).

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

from svy.core.design import Singleton
from svy.core.singleton import find_domain_singletons
from svy.core.warnings import Severity
from svy.errors import MethodError
from svy.errors.singleton_errors import SingletonError
from svy.serialize import serialize, to_design


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
# svy.Singleton(domains=..., on_domain_singletons=...)
# ══════════════════════════════════════════════════════════════════════════════


class TestRule:
    def test_defaults(self):
        rule = Singleton("center")
        assert (rule.domains, rule.on_domain_singletons) == ("standard", "ignore")

    @pytest.mark.parametrize("method", ["center", "scale"])
    def test_apply_on_center_and_scale(self, method):
        assert Singleton(method, domains="apply").domains == "apply"

    @pytest.mark.parametrize("method", ["self_representing", "skip", "pool", "collapse"])
    def test_apply_refused_elsewhere(self, method):
        with pytest.raises(MethodError, match="center and scale define") as err:
            Singleton(method, domains="apply")
        assert err.value.hint == 'svy.Singleton("center", domains="apply")'

    def test_unknown_domains(self):
        with pytest.raises(MethodError, match="domains") as err:
            Singleton("center", domains="adjust")
        assert err.value.code == "INVALID_CHOICE"

    @pytest.mark.parametrize("value", ["warn", "ignore", "error"])
    def test_former_domains_values_point_to_on_domain_singletons(self, value):
        with pytest.raises(MethodError, match="on_domain_singletons") as err:
            Singleton("scale", domains=value)
        assert err.value.hint == f'svy.Singleton("scale", on_domain_singletons="{value}")'

    def test_unknown_on_domain_singletons(self):
        with pytest.raises(MethodError, match="on_domain_singletons"):
            Singleton("center", on_domain_singletons="yes")

    def test_repr_and_code_name_settings_only_when_set(self):
        assert repr(Singleton("center")) == "Singleton('center')"
        assert repr(Singleton("scale", domains="apply")) == "Singleton('scale', domains='apply')"
        assert Singleton("skip", on_domain_singletons="warn")._to_code() == (
            "svy.Singleton('skip', on_domain_singletons='warn')"
        )


class TestSerialization:
    def test_round_trip(self, data, design):
        rule = Singleton("center", domains="apply", on_domain_singletons="error")
        sample = svy.Sample(data, design).update_design(singleton=rule)
        back = to_design(serialize(sample.design))
        assert back.singleton == rule

    def test_defaults_are_not_saved(self, data, design):
        saved = msgspec.to_builtins(serialize(_declare(svy.Sample(data, design), "skip").design))
        assert saved["singleton"] == {"method": "skip"}

    def test_to_code_rebuilds_the_rule(self, no_singletons):
        sample = _declare(no_singletons, "center", domains="apply")
        assert "singleton=svy.Singleton('center', domains='apply')" in sample.to_code()


# ══════════════════════════════════════════════════════════════════════════════
# The rule on a sample with or without singletons
# ══════════════════════════════════════════════════════════════════════════════


class TestDeclared:
    @pytest.mark.parametrize(
        "method", ["self_representing", "skip", "collapse", "pool", "scale", "center"]
    )
    @pytest.mark.parametrize("on", ["ignore", "warn", "error"])
    def test_kept_without_singletons(self, no_singletons, method, on):
        sample = _declare(no_singletons, method, on_domain_singletons=on)
        assert sample.design.singleton == Singleton(method, on_domain_singletons=on)
        assert _result(sample) is None
        assert no_singletons.design.singleton is None

    @pytest.mark.parametrize("method", ["center", "scale"])
    def test_apply_without_singletons_reaches_the_kernels(self, no_singletons, method):
        sample = _declare(no_singletons, method, domains="apply")
        assert _result(sample).method == method

    def test_with_singletons(self, data, design):
        sample = _declare(svy.Sample(data, design), "center", domains="apply")
        handled = {s.stratum_values["stratum"] for s in _result(sample).detected}
        assert handled == {5, 6}

    def test_singletons_appearing_later_are_handled_silently(self, data, design):
        sample = svy.Sample(data.filter(pl.col("stratum") <= 4), design)
        sample = _declare(sample, "center", domains="apply")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sample = sample.wrangling.filter_records(
                ~((pl.col("stratum") == 1) & (pl.col("psu") > 1))
            )
            assert sample.design.singleton == Singleton("center", domains="apply")
            assert sample.estimation.mean("y").estimates
        detected = _result(sample).detected
        assert [s.stratum_values["stratum"] for s in detected] == [1]


class TestDomainSingletons:
    def test_by_levels(self, data, design):
        found = svy.Sample(data, design).domain_singletons(by="dom")
        assert found.columns == ["dom", "stratum", "psu", "n", "n_psus"]
        assert found.select("dom", "stratum", "n_psus").rows() == [("g1", 2, 2), ("g2", 2, 2)]
        assert found["psu"].to_list() == [1, 2]  # the PSU column's own values

    def test_where(self, data, design):
        found = svy.Sample(data, design).domain_singletons(where=svy.col("reg") == 1)
        assert found["stratum"].to_list() == [1, 2, 3, 4]

    def test_matches_the_estimate_finding(self, data, design):
        sample = _declare(svy.Sample(data, design), "center")
        where = svy.col("reg") == 1
        found = sample.domain_singletons(by="dom", where=where)
        r = sample.estimation.total("y", by="dom", where=where)
        pairs = [f"stratum={h} in dom={g}" for g, h in found.select("dom", "stratum").rows()]
        assert pairs == r.findings[0].extra["pairs"]

    def test_none(self, data, design):
        sample = svy.Sample(data, design)
        assert sample.domain_singletons().is_empty()
        assert sample.domain_singletons(by="stratum").is_empty()

    def test_unknown_by_column(self, data, design):
        with pytest.raises(MethodError, match="not in the data"):
            svy.Sample(data, design).domain_singletons(by="nope")


# ══════════════════════════════════════════════════════════════════════════════
# Detection
# ══════════════════════════════════════════════════════════════════════════════


def _found(df, **kw):
    kw.setdefault("stratum_cols", ["stratum"])
    return [f.label for f in find_domain_singletons(df, strata_col="stratum", **kw)]


class TestDetection:
    def test_by_levels(self, data):
        assert _found(data, psu_col="psu", by_col="dom", by_cols=["dom"]) == [
            "stratum=2 in dom=g1",
            "stratum=2 in dom=g2",
        ]

    def test_a_zero_value_row_still_places_its_psu(self, data):
        # g1's only row with y = 0 is in stratum 2's PSU 1: still one PSU.
        assert _found(data, psu_col="psu", mask=pl.col("dom") == "g1") == ["stratum=2"]

    def test_a_zero_weight_row_still_places_its_psu(self, data):
        # R's subset() keeps zero-weight rows: a PSU whose only domain row has
        # weight 0 is present, so stratum 2 has two PSUs in the domain.
        g2_in_psu1 = (pl.col("stratum") == 2) & (pl.col("psu") == 1) & (pl.col("dom") == "g1")
        d = data.with_columns(
            dom=pl.when(g2_in_psu1 & (pl.int_range(pl.len()).over("stratum", "psu") == 0))
            .then(pl.lit("g2"))
            .otherwise("dom"),
        ).with_columns(
            wgt=pl.when(g2_in_psu1 & (pl.col("dom") == "g2")).then(0.0).otherwise("wgt")
        )
        assert _found(d, psu_col="psu", mask=pl.col("dom") == "g2") == []
        sample = svy.Sample(d, svy.Design(stratum="stratum", psu="psu", wgt="wgt"))
        r = _declare(sample, "center").estimation.mean("y", where=dom("g2"))
        assert r.findings == []

    def test_where_mask(self, data):
        # reg == 1 keeps PSU 1 of every stratum: all multi-PSU strata.
        assert _found(data, psu_col="psu", mask=pl.col("reg") == 1) == [
            f"stratum={h}" for h in (1, 2, 3, 4)
        ]

    def test_missing_values_leave_the_domain(self, data, design):
        # g2 has one row in each PSU of stratum 3, and y2 is missing on both:
        # in PSU 1, g2 reaches strata 1 and 4 (lonely) but not 3.
        sample = _declare(svy.Sample(data, design), "center")
        where = dom("g2") & (svy.col("psu") == 1)
        r = sample.estimation.total("y2", where=where, drop_nulls=True)
        assert r.findings[0].extra["pairs"] == ["stratum=1", "stratum=4"]
        r = sample.estimation.total("y", where=where)
        assert r.findings[0].extra["pairs"] == ["stratum=1", "stratum=3", "stratum=4"]

    def test_element_design_counts_rows(self, data):
        # Without PSUs a stratum with one domain row among several is lonely.
        first = pl.int_range(pl.len()).over("stratum") == 0
        assert _found(data, psu_col=None, mask=first) == [
            f"stratum={h}" for h in (1, 2, 3, 4, 5, 6)
        ]

    def test_whole_sample_has_none(self, data):
        assert _found(data, psu_col="psu") == []

    def test_by_column_that_is_the_stratum(self, data, design):
        # by="stratum" names the same column twice.
        r = _declare(svy.Sample(data, design), "center").estimation.mean(
            "y", by="stratum", where=svy.col("reg") == 1
        )
        assert len(r.findings[0].extra["pairs"]) == 4

    def test_tuple_strata_named_by_their_columns(self, data):
        d = data.with_columns(s2=pl.lit("x"))
        found = find_domain_singletons(
            d,
            strata_col="stratum",
            psu_col="psu",
            stratum_cols=["stratum", "s2"],
            by_col="dom",
            by_cols=["dom"],
        )
        assert found[0].label == "stratum=2, s2=x in dom=g1"

    def test_calibrated_design_has_none(self, data, design):
        sample = _declare(svy.Sample(data, design), "center")
        ps = sample.weighting.poststratify(controls={"u": 700.0, "v": 400.0}, cells="ps")
        assert ps.estimation.total("y", by="dom").findings == []

    def test_unstratified_design_has_none(self, data):
        sample = svy.Sample(data, svy.Design(psu="psu", wgt="wgt"))
        assert sample.estimation.total("y", by="dom").findings == []


# ══════════════════════════════════════════════════════════════════════════════
# on_domain_singletons= ignore / warn / error, domains= standard / apply
# ══════════════════════════════════════════════════════════════════════════════


def _notes(result):
    return [ln for ln in result.__plain_str__().splitlines() if ln.startswith("note:")]


class TestPolicy:
    def test_ignore_records_the_finding_and_prints_nothing(self, data, design):
        sample = _declare(svy.Sample(data, design), "center")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            r = sample.estimation.mean("y", by="dom")
        (f,) = r.findings
        assert f.code == "DOMAIN_SINGLETON_PSU" and f.level == Severity.INFO
        assert f.extra["pairs"] == ["stratum=2 in dom=g1", "stratum=2 in dom=g2"]
        assert _notes(r) == [] and "note:" not in str(r)
        kept = [w for w in sample.warnings if w.code == "DOMAIN_SINGLETON_PSU"]
        assert kept and kept[-1].level == Severity.INFO

    def test_warn_prints_one_note_not_a_python_warning(self, data, design):
        sample = _declare(svy.Sample(data, design), "center", on_domain_singletons="warn")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            r = sample.estimation.mean("y", by="dom")
        assert r.findings[0].level == Severity.WARNING
        assert _notes(r) == [
            "note: 2 domain × stratum pairs with one PSU in the domain (stratum=2 in dom=g1; "
            "stratum=2 in dom=g2); standard domain variance used"
        ]
        assert r.findings[0].hint.startswith('svy.Singleton("center", domains="apply")')

    def test_the_setting_does_not_change_the_numbers(self, data, design):
        ses = [
            by_se(
                _declare(
                    svy.Sample(data, design), "center", on_domain_singletons=on
                ).estimation.total("y", by="dom")
            )
            for on in ("ignore", "warn")
        ]
        assert ses[0] == ses[1]

    def test_no_rule_behaves_like_ignore(self, no_singletons):
        r = no_singletons.estimation.mean("y", by="dom")
        assert r.findings[0].level == Severity.INFO
        assert _notes(r) == []

    def test_error_raises_before_estimating(self, data, design):
        sample = _declare(svy.Sample(data, design), "center", on_domain_singletons="error")
        with pytest.raises(SingletonError) as err:
            sample.estimation.mean("y", by="dom")
        assert err.value.code == "DOMAIN_SINGLETON"
        assert "stratum=2 in dom=g1" in str(err.value)
        assert 'svy.Singleton("center", domains="apply")' in str(err.value)
        # Nothing to find, nothing raised.
        assert sample.estimation.mean("y").estimates

    @pytest.mark.parametrize(
        "method, how", [("center", "centered at the grand mean"), ("scale", "left out")]
    )
    def test_apply_reports_what_it_did_as_asked(self, data, design, method, how):
        quiet = _declare(svy.Sample(data, design), method, domains="apply")
        r = quiet.estimation.mean("y", by="dom")
        assert r.findings[0].level == Severity.INFO and _notes(r) == []
        assert how in r.findings[0].detail
        loud = _declare(
            svy.Sample(data, design), method, domains="apply", on_domain_singletons="warn"
        )
        (note,) = _notes(loud.estimation.mean("y", by="dom"))
        assert how in note and 'domains="apply"' in note

    def test_one_note_per_call(self, data, design):
        # y2 is missing on g2's rows of stratum 3: that variable finds one more pair.
        sample = _declare(svy.Sample(data, design), "center", on_domain_singletons="warn")
        r = sample.estimation.total(
            ["y", "y2"], by="dom", where=svy.col("reg") == 1, drop_nulls=True
        )
        (note,) = _notes(r)
        assert note.startswith("note: 7 domain × stratum pairs") and "; 4 more)" in note

    def test_a_where_domain_alone_counts_strata(self, data, design):
        sample = _declare(svy.Sample(data, design), "center", on_domain_singletons="warn")
        (note,) = _notes(sample.estimation.total("y", where=svy.col("reg") == 1))
        assert note.startswith("note: 4 strata with one PSU in the domain (stratum=1; ")

    def test_integer_valued_float_strata_print_as_integers(self, data):
        d = data.with_columns(pl.col("stratum").cast(pl.Float64) + 2000)
        design = svy.Design(
            stratum="stratum",
            psu="psu",
            wgt="wgt",
            singleton=Singleton("center", on_domain_singletons="warn"),
        )
        (note,) = _notes(svy.Sample(d, design).estimation.mean("y", by="dom"))
        assert "stratum=2002 in dom=g1" in note and "2002.0" not in note

    def test_findings_are_plain_in_glm_to_dict(self, data, design):
        fit = _declare(svy.Sample(data, design), "center").glm.fit("y", x=["x"], where=dom("g1"))
        d = fit.fitted.to_dict()
        json.dumps(d)
        assert d["findings"][0]["code"] == "DOMAIN_SINGLETON_PSU"


@pytest.mark.parametrize("domains", ["standard", "apply"])
@pytest.mark.parametrize("on", ["ignore", "warn"])
def test_every_analysis_reports_the_domain(data, design, domains, on):
    sample = _declare(svy.Sample(data, design), "center", domains=domains, on_domain_singletons=on)
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
        shown = on == "warn"
        assert ("note:" in r.__plain_str__()) is shown, type(r).__name__
        assert ("note:" in str(r)) is shown, type(r).__name__


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
    sample = _declare(svy.Sample(data, design), method, domains="apply")
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
    applied = _declare(sample, "center", domains="apply").estimation.total("y", by="dom")
    assert by_se(applied) == _approx([391.734795063, 327.11788508, 173.45])
