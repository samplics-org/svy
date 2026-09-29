# tests/svy/estimation/test_where_nulls.py
"""
Nulls outside the `where` domain are ignored, as in R's subset().

Without drop_nulls, a null in a column read only by `where` makes the row
out-of-domain (R: NA in the condition is FALSE), and a null analysis value
on an out-of-domain row is irrelevant. Nulls inside the domain still raise
unless drop_nulls=True, and design columns must be complete everywhere.
Out-of-domain rows keep their place in the design, so SEs and df equal the
drop_nulls=True path and R.

Dataset: where_nulls_20260929.csv (see tests/test_data/where_nulls_test_data.py),
an events file after a full join: people without an event have a null `dom`
and null analysis values. PSU 14 has no event at all.

Reference values from R survey 4.5:

    options(digits = 15)
    library(survey)
    d <- read.csv("tests/test_data/where_nulls_20260929.csv")
    des <- svydesign(id = ~psu, strata = ~stratum, weights = ~wgt, data = d)
    s <- subset(des, dom == 1)                     # degf(s) = 8
    svymean(~y, s); svytotal(~y, s); svymean(~cat, s); svyratio(~y, ~x, s)
    svymean(~y + y2, s); svyvar(~y + y2, s)
    svyby(~y, ~g, s, svymean); svyby(~y, ~g, s, svytotal); svyby(~cat, ~g, s, svymean)
    svyquantile(~y, s, c(0.25, 0.5), ci = TRUE, qrule = "math")
    svymean(~y, subset(des, y > 30))               # NA in y > 30 is FALSE
    d$gin <- ifelse(d$unit == 1, NA, d$g)          # by NA inside the domain
    svyby(~y, ~gin, subset(des, dom == 1), svymean)    # NA group dropped

    # cov with na.rm: listwise, but the design keeps the rows (PSU 14 stays)
    svyvar(~y + y2, des, na.rm = TRUE)             # 32.495710162104, SE 6.58178918917496
    d2 <- d; d2$y2[!is.na(d2$dom) & d2$dom == 1 & d2$unit == 2] <- NA
    des2 <- svydesign(id = ~psu, strata = ~stratum, weights = ~wgt, data = d2)
    svyvar(~y + y2, subset(des2, dom == 1), na.rm = TRUE)
    # 30.4764593407369, SE 5.41722034017244

    # JKn: replicate weights exported as where_nulls_jkn_20260929.csv
    rdes <- as.svrepdesign(des, type = "JKn")      # rscales 0.75, scale 1
    rw <- weights(rdes, "analysis")
    rs <- subset(rdes, dom == 1)
    # same calls on rs; quantiles with interval.type = "quantile", the
    # replicate-quantile construction svy uses on replicate designs
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import svy

from svy import Design, RepWeights, Sample


BASE_DIR = Path(__file__).parents[2]
REL = 1e-9

W = pl.col("dom") == 1

# (Taylor SE, JKn SE)
MEAN_Y = (36.02331291102032, 1.16942364338535, 1.1731263662892)
TOTAL_Y = (57844.79494, 8125.66156852215, 8125.66156852214)
RATIO = (11.8920719583242, 1.0435145518120, 1.12168848373698)
MEAN_Y2 = (17.9193831705859, 0.651445256766784, 0.653713842201503)
PROP_CAT = {
    "a": (0.326586787564767, 0.0711846232608628, 0.0729245354010006),
    "b": (0.339807941410921, 0.0883185298457674, 0.0889450161777327),
    "c": (0.333605271024312, 0.0988374521596168, 0.1014346137186696),
}
MEAN_BY_G = {
    "g1": (38.3175979589548, 1.86006111065039, 1.93915449003045),
    "g2": (33.7310272898174, 1.92237816468256, 2.07791802234553),
}
TOTAL_BY_G = {
    "g1": (30751.02189, 9762.30108707970, 9762.30108707971),
    "g2": (27093.77305, 6539.02733380602, 6539.02733380602),
}
PROP_BY_G = {
    ("g1", "a"): (0.284699637396733, 0.1020588555443394, 0.1070664687658853),
    ("g1", "b"): (0.402477165962643, 0.119786672030148, 0.134229661128075),
    ("g1", "c"): (0.312823196640624, 0.0779709625313773, 0.0822521778440592),
    ("g2", "a"): (0.368437433860787, 0.0922548409091836, 0.0950575948505196),
    ("g2", "b"): (0.277193331922363, 0.127051633632751, 0.135075536479268),
    ("g2", "c"): (0.354369234216849, 0.1426571740466909, 0.1501413183123547),
}
# p -> (estimate, Taylor SE, JKn SE)
QUANTILES = {
    0.25: (31.421, 2.42128793906333, 1.01890198252825),
    0.5: (37.372, 2.14808808205430, 2.61022943196570),
}
# svyvar(~y + y2): cov(y, y2) and the variance of that element
COV_Y_Y2 = (37.4937356052483, 52.0441447105956**0.5, 53.9255400401248**0.5)

METHODS = [pytest.param(None, 1, id="taylor"), pytest.param("replication", 2, id="jkn")]


@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    return pl.read_csv(BASE_DIR / "test_data" / "where_nulls_20260929.csv")


def _taylor(df: pl.DataFrame) -> Sample:
    return Sample(df, Design(stratum="stratum", psu="psu", wgt="wgt"))


def _jkn(df: pl.DataFrame) -> Sample:
    rw = pl.read_csv(BASE_DIR / "test_data" / "where_nulls_jkn_20260929.csv")
    df = df.join(rw, on=["psu", "unit"], how="left", maintain_order="left")
    rep = RepWeights(
        method="jackknife", prefix="rep", n_reps=12, kind="jkn", stratum="stratum", psu="psu"
    )
    return Sample(df, Design(stratum="stratum", psu="psu", wgt="wgt", rep_wgts=rep))


@pytest.fixture(scope="module")
def samples(data) -> dict[str | None, Sample]:
    return {None: _taylor(data), "replication": _jkn(data)}


def _pairs(r) -> list[tuple[float, float]]:
    return [(e.est, e.se) for e in r.estimates]


def _same(a, b) -> None:
    """drop_nulls=False and drop_nulls=True give the same estimates."""
    assert len(a.estimates) == len(b.estimates)
    for x, y in zip(a.estimates, b.estimates):
        assert x.est == pytest.approx(y.est, rel=1e-12, nan_ok=True)
        assert x.se == pytest.approx(y.se, rel=1e-12, nan_ok=True)
        assert x.df == y.df
        assert x.n == y.n


def test_data_has_nulls_only_outside_the_domain(data):
    """The fixture exercises the bug: nulls everywhere except inside dom == 1."""
    inside = data.filter(W)
    assert data["dom"].null_count() > 0
    assert data.filter(pl.col("dom") == 0)["y"].null_count() > 0
    assert inside.null_count().sum_horizontal().item() == 0
    assert data.filter(pl.col("psu") == 14)["dom"].null_count() == 5


# ── Every estimator, Taylor and JKn, against R ───────────────────────────


@pytest.mark.parametrize("method,k", METHODS)
class TestMatchesR:
    def test_mean(self, samples, method, k):
        r = samples[method].estimation.mean("y", where=W, method=method)
        e = r.estimates[0]
        assert e.est == pytest.approx(MEAN_Y[0], rel=REL)
        assert e.se == pytest.approx(MEAN_Y[k], rel=REL)
        assert e.n == 32

    def test_total(self, samples, method, k):
        e = samples[method].estimation.total("y", where=W, method=method).estimates[0]
        assert e.est == pytest.approx(TOTAL_Y[0], rel=REL)
        assert e.se == pytest.approx(TOTAL_Y[k], rel=REL)

    def test_prop_has_no_null_level(self, samples, method, k):
        r = samples[method].estimation.prop("cat", where=W, method=method)
        got = {e.y_level: (e.est, e.se) for e in r.estimates}
        assert set(got) == set(PROP_CAT)
        for lvl, ref in PROP_CAT.items():
            assert got[lvl][0] == pytest.approx(ref[0], rel=REL)
            assert got[lvl][1] == pytest.approx(ref[k], rel=REL)

    def test_ratio(self, samples, method, k):
        e = samples[method].estimation.ratio("y", "x", where=W, method=method).estimates[0]
        assert e.est == pytest.approx(RATIO[0], rel=REL)
        assert e.se == pytest.approx(RATIO[k], rel=REL)

    def test_multi_y(self, samples, method, k):
        r = samples[method].estimation.mean(["y", "y2"], where=W, method=method)
        (ey,), (ey2,) = r[0].estimates, r[1].estimates
        assert (ey.est, ey.se) == pytest.approx((MEAN_Y[0], MEAN_Y[k]), rel=REL)
        assert (ey2.est, ey2.se) == pytest.approx((MEAN_Y2[0], MEAN_Y2[k]), rel=REL)

    def test_mean_by(self, samples, method, k):
        r = samples[method].estimation.mean("y", by="g", where=W, method=method)
        got = {e.by_level[0]: (e.est, e.se) for e in r.estimates}
        assert set(got) == set(MEAN_BY_G)
        for lvl, ref in MEAN_BY_G.items():
            assert got[lvl] == pytest.approx((ref[0], ref[k]), rel=REL)

    def test_total_by(self, samples, method, k):
        r = samples[method].estimation.total("y", by="g", where=W, method=method)
        got = {e.by_level[0]: (e.est, e.se) for e in r.estimates}
        for lvl, ref in TOTAL_BY_G.items():
            assert got[lvl] == pytest.approx((ref[0], ref[k]), rel=REL)

    def test_prop_by(self, samples, method, k):
        r = samples[method].estimation.prop("cat", by="g", where=W, method=method)
        got = {(e.by_level[0], e.y_level): (e.est, e.se) for e in r.estimates}
        assert set(got) == set(PROP_BY_G)
        for key, ref in PROP_BY_G.items():
            assert got[key] == pytest.approx((ref[0], ref[k]), rel=REL)

    def test_quantile(self, samples, method, k):
        r = samples[method].estimation.quantile("y", p=[0.25, 0.5], where=W, method=method)
        for est, (p, ref) in zip(r, QUANTILES.items()):
            e = est.estimates[0]
            assert e.est == pytest.approx(ref[0], rel=REL), p
            assert e.se == pytest.approx(ref[k], rel=1e-9), p

    def test_median(self, samples, method, k):
        e = samples[method].estimation.median("y", where=W, method=method).estimates[0]
        assert e.est == pytest.approx(QUANTILES[0.5][0], rel=REL)
        assert e.se == pytest.approx(QUANTILES[0.5][k], rel=1e-9)

    def test_cov(self, samples, method, k):
        e = samples[method].estimation.cov(["y", "y2"], where=W, method=method).estimates[0]
        assert e.est == pytest.approx(COV_Y_Y2[0], rel=REL)
        assert e.se == pytest.approx(COV_Y_Y2[k], rel=REL)

    def test_corr_runs(self, samples, method, k):
        e = samples[method].estimation.corr(["y", "y2"], where=W, method=method).estimates[0]
        assert e.est == pytest.approx(
            COV_Y_Y2[0] / (64.4924138067696 * 28.1381893364551) ** 0.5, rel=REL
        )


def test_taylor_domain_df(samples):
    """Domain df counts PSUs with domain rows: PSU 14 is out (R degf = 8)."""
    e = samples[None].estimation.mean("y", where=W).estimates[0]
    assert e.df == 8


# ── Same answer as drop_nulls=True ───────────────────────────────────────


@pytest.mark.parametrize("method", [None, "replication"], ids=["taylor", "jkn"])
@pytest.mark.parametrize(
    "call",
    [
        lambda s, **kw: s.estimation.mean("y", **kw),
        lambda s, **kw: s.estimation.total("y", **kw),
        lambda s, **kw: s.estimation.prop("cat", **kw),
        lambda s, **kw: s.estimation.ratio("y", "x", **kw),
        lambda s, **kw: s.estimation.median("y", **kw),
        lambda s, **kw: s.estimation.mean("y", by="g", **kw),
        lambda s, **kw: s.estimation.total("y", by="g", **kw),
        lambda s, **kw: s.estimation.prop("cat", by="g", **kw),
        lambda s, **kw: s.estimation.ratio("y", "x", by="g", **kw),
        lambda s, **kw: s.estimation.median("y", by="g", **kw),
        lambda s, **kw: s.estimation.mean("cat", as_factor=True, **kw),
        lambda s, **kw: s.estimation.total("cat", as_factor=True, **kw),
        lambda s, **kw: s.estimation.cov(["y", "y2"], **kw),
        lambda s, **kw: s.estimation.corr(["y", "y2"], **kw),
    ],
    ids=[
        "mean",
        "total",
        "prop",
        "ratio",
        "median",
        "mean_by",
        "total_by",
        "prop_by",
        "ratio_by",
        "median_by",
        "mean_factor",
        "total_factor",
        "cov",
        "corr",
    ],
)
def test_equals_drop_nulls_path(samples, method, call):
    s = samples[method]
    _same(
        call(s, where=W, method=method),
        call(s, where=W, method=method, drop_nulls=True),
    )


class TestAssocDropNulls:
    """drop_nulls=True on cov/corr zeroes the weight of a row with a missing
    value in any named column; it no longer drops the row, which deleted
    PSU 14 and understated the Taylor SE (6.6029 instead of 7.2142 here)."""

    def test_cov_no_where(self, samples):
        e = samples[None].estimation.cov(["y", "y2"], drop_nulls=True).estimates[0]
        assert e.est == pytest.approx(32.495710162104, rel=REL)
        assert e.se == pytest.approx(6.58178918917496, rel=REL)

    def test_cov_null_inside_domain(self, data):
        df = data.with_columns(
            pl.when(W & (pl.col("unit") == 2)).then(None).otherwise(pl.col("y2")).alias("y2")
        )
        e = _taylor(df).estimation.cov(["y", "y2"], where=W, drop_nulls=True).estimates[0]
        assert e.est == pytest.approx(30.4764593407369, rel=REL)
        assert e.se == pytest.approx(5.41722034017244, rel=REL)

    def test_cov_where_matches_r(self, samples):
        e = samples[None].estimation.cov(["y", "y2"], where=W, drop_nulls=True).estimates[0]
        assert (e.est, e.se) == pytest.approx(COV_Y_Y2[:2], rel=REL)


# ── What still raises ────────────────────────────────────────────────────


class TestStillRaises:
    def test_null_y_inside_domain(self, data):
        df = data.with_columns(
            pl.when(W & (pl.col("unit") == 2)).then(None).otherwise(pl.col("y")).alias("y")
        )
        s = _taylor(df)
        with pytest.raises(ValueError, match="inside the `where` domain.*y \\(NULL\\)"):
            s.estimation.mean("y", where=W)
        # drop_nulls=True still handles it
        assert s.estimation.mean("y", where=W, drop_nulls=True).estimates[0].n < 32

    def test_nan_y_inside_domain(self, data):
        """Sample turns NaN into null, so it is reported as NULL."""
        df = data.with_columns(
            pl.when(W & (pl.col("unit") == 2)).then(float("nan")).otherwise(pl.col("y")).alias("y")
        )
        with pytest.raises(ValueError, match="inside the `where` domain.*y \\(NULL\\)"):
            _taylor(df).estimation.total("y", where=W)

    def test_inf_y_inside_domain(self, data):
        df = data.with_columns(
            pl.when(W & (pl.col("unit") == 2)).then(float("inf")).otherwise(pl.col("y")).alias("y")
        )
        with pytest.raises(ValueError, match="inside the `where` domain.*y \\(±∞\\)"):
            _taylor(df).estimation.total("y", where=W)

    def test_null_by_inside_domain(self, data):
        """R's svyby silently drops an NA group; svy asks for drop_nulls."""
        df = data.with_columns(
            pl.when(pl.col("unit") == 1).then(None).otherwise(pl.col("g")).alias("gin")
        )
        with pytest.raises(ValueError, match="gin \\(NULL\\)"):
            _taylor(df).estimation.mean("y", by="gin", where=W)

    def test_null_by_inside_domain_with_drop_nulls_matches_r(self, data):
        df = data.with_columns(
            pl.when(pl.col("unit") == 1).then(None).otherwise(pl.col("g")).alias("gin")
        )
        r = _taylor(df).estimation.mean("y", by="gin", where=W, drop_nulls=True)
        got = {e.by_level[0]: (e.est, e.se) for e in r.estimates}
        assert set(got) == {"g1", "g2"}
        assert got["g1"] == pytest.approx((38.2480609270673, 2.42190370116355), rel=REL)
        assert got["g2"] == pytest.approx((33.5331896425109, 2.32693255286737), rel=REL)

    @pytest.mark.parametrize("col", ["psu", "stratum", "wgt"])
    def test_null_design_column_outside_domain(self, data, col):
        df = data.with_columns(
            pl.when(pl.col("dom").is_null() & (pl.col("psu") == 14) & (pl.col("unit") == 1))
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
        )
        with pytest.raises(ValueError, match=f"{col} \\(NULL\\)"):
            _taylor(df).estimation.mean("y", where=W)

    def test_no_where_still_checks_every_row(self, samples):
        with pytest.raises(
            ValueError, match="Missing or invalid values found in required columns;"
        ):
            samples[None].estimation.mean("y")


# ── Edge cases ───────────────────────────────────────────────────────────


class TestEdgeCases:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_y_outside_domain_ignored(self, data, samples, bad):
        df = data.with_columns(
            pl.when(pl.col("dom") == 0).then(bad).otherwise(pl.col("y")).alias("y")
        )
        e = _taylor(df).estimation.mean("y", where=W).estimates[0]
        assert (e.est, e.se) == pytest.approx(MEAN_Y[:2], rel=REL)

    @pytest.mark.parametrize(
        "where",
        [
            {"dom": 1},
            [pl.col("dom") == 1, pl.col("wgt") > 0],
            svy.col("dom") == 1,
        ],
        ids=["dict", "list", "svy_col"],
    )
    def test_where_forms(self, samples, where):
        e = samples[None].estimation.mean("y", where=where).estimates[0]
        assert (e.est, e.se) == pytest.approx(MEAN_Y[:2], rel=REL)

    def test_where_on_y_itself(self, samples):
        """A null y makes `y > 30` null, hence out of the domain (R: FALSE)."""
        e = samples[None].estimation.mean("y", where=pl.col("y") > 30).estimates[0]
        assert e.est == pytest.approx(38.1670807326541, rel=REL)
        assert e.se == pytest.approx(0.79511481451892, rel=REL)
        assert e.df == 8

    def test_where_only_column_null_inside_domain(self, data, samples):
        """Kleene OR: TRUE | NULL is TRUE, so a row can be in the domain with a
        null where-only column; that column is not analysed, so no error."""
        df = data.with_columns(
            pl.when(W & (pl.col("unit") <= 2)).then(None).otherwise(pl.lit(0)).alias("aux")
        )
        e = _taylor(df).estimation.mean("y", where=W | (pl.col("aux") == 1)).estimates[0]
        assert (e.est, e.se) == pytest.approx(MEAN_Y[:2], rel=REL)

    def test_predicate_selecting_null_rows(self, samples):
        """is_null() puts the null-dom rows IN the domain, where y is null."""
        with pytest.raises(ValueError, match="inside the `where` domain"):
            samples[None].estimation.mean("y", where=pl.col("dom").is_null())

    def test_by_column_also_in_where(self, samples):
        r = samples[None].estimation.mean("y", by="g", where=W & (pl.col("g") == "g1"))
        assert [e.by_level[0] for e in r.estimates] == ["g1"]
        assert (r.estimates[0].est, r.estimates[0].se) == pytest.approx(
            MEAN_BY_G["g1"][:2], rel=REL
        )

    def test_by_level_only_outside_domain(self, data):
        df = data.with_columns(
            pl.when(pl.col("dom") == 0).then(pl.lit("g3")).otherwise(pl.col("g")).alias("g")
        )
        r = _taylor(df).estimation.mean("y", by="g", where=W)
        got = {e.by_level[0]: (e.est, e.se) for e in r.estimates}
        assert set(got) == {"g1", "g2"}
        assert got["g1"] == pytest.approx(MEAN_BY_G["g1"][:2], rel=REL)

    @pytest.mark.parametrize("method", [None, "replication"], ids=["taylor", "jkn"])
    def test_zero_weight_domain_level_kept(self, data, method):
        """#201: a by level whose domain rows all have weight 0 is a NaN row."""
        df = data.with_columns(
            pl.when(W & (pl.col("g") == "g2")).then(0.0).otherwise(pl.col("wgt")).alias("wgt")
        )
        s = _taylor(df) if method is None else _jkn(df)
        r = s.estimation.mean("y", by="g", where=W, method=method)
        got = {e.by_level[0]: e for e in r.estimates}
        assert set(got) == {"g1", "g2"}
        assert got["g2"].est != got["g2"].est  # NaN
        assert got["g1"].est == pytest.approx(MEAN_BY_G["g1"][0], rel=REL)

    def test_prop_drop_nulls_true_unchanged(self, samples):
        """#193: with drop_nulls=True a null category is out of the domain."""
        r = samples[None].estimation.prop("cat", where=W, drop_nulls=True)
        assert {e.y_level for e in r.estimates} == set(PROP_CAT)

    def test_null_predicate_with_drop_nulls_replication(self, samples):
        """Ungrouped replication (domain-mask path) with null predicates."""
        e = (
            samples["replication"]
            .estimation.mean("y", where=W, method="replication", drop_nulls=True)
            .estimates[0]
        )
        assert (e.est, e.se) == pytest.approx((MEAN_Y[0], MEAN_Y[2]), rel=REL)
