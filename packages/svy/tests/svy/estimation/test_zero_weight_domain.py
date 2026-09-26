# tests/svy/estimation/test_zero_weight_domain.py
"""
A domain whose weights are all zero keeps its row, and never hides the others.

Taylor ``mean``/``prop``/``ratio`` used to return an Estimate with no rows at
all (by= and where=) when one domain carried no weight: the kernel failed on
the zero denominator and the whole result was replaced by an empty one.

The undefined domain is NaN (est, se, CIs), as in R's ``svymean(subset())``;
``total`` stays 0 with SE 0, as ``svytotal(subset())``. R's ``svyby`` omits
such a level instead (it builds its levels from nonzero-weight rows only); svy
keeps it, as its total, quantile and replication paths already did.

R survey 4.5 reference:

```r
i <- 0:59
d <- data.frame(st = ifelse(i < 30, "a", "b"), psu = i %/% 5,
  w = ifelse(i < 55 & i %% 10 != 9, 1 + i %% 4, 0),
  y = ((i * 37) %% 23) / 4 - 2, x = 1 + ((i * 11) %% 7) / 10,
  b = as.integer((i * 13) %% 3 == 0),
  r = ifelse(i < 55, "z", "q"), g = ifelse(i %% 2 == 0, "u", "v"),
  k = ifelse(i %% 10 == 9, "m", "n"))
des <- svydesign(ids = ~psu, strata = ~st, weights = ~w, data = d, nest = TRUE)
svyby(~y, ~r, des, svymean); svyby(~y, ~r + g, des, svymean)
svyby(~y, ~k, des, svymean); svyby(~factor(b), ~r, des, svymean)
svyby(~y, ~r, denominator = ~x, des, svyratio); svyby(~y, ~r, des, svytotal)
svyby(~y, ~r, des, svyquantile, quantiles = 0.5, ci = TRUE, se = TRUE)
svymean(~y, subset(des, r == "q"))    # NaN, degf 0
svytotal(~y, subset(des, r == "q"))   # 0, SE 0
```

Level ``q`` is one whole PSU with zero weight; level ``m`` is scattered over
PSUs that otherwise carry weight.
"""

import math

import numpy as np
import polars as pl
import pytest

import svy


TOL = 1e-10

# (est, se) of the weighted domains, from R.
R_MEAN = {("z",): (0.547131147540983, 0.141576302329795)}
R_MEAN_RG = {
    ("z", "u"): (0.709821428571429, 0.394023304299924),
    ("z", "v"): (0.409090909090909, 0.407075248698807),
}
R_MEAN_K = {("n",): (0.547131147540983, 0.141576302329795)}
R_PROP = {
    ("z", "0"): (0.655737704918033, 0.057235232106466),
    ("z", "1"): (0.344262295081967, 0.057235232106466),
}
R_RATIO = {("z",): (0.420870113493064, 0.10756276773332)}
R_TOTAL = {("z",): (66.75, 19.3923309583969)}
R_MEDIAN = {("z",): (0.5, 0.331541951343592)}
R_DF = 9


@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    i = np.arange(60)
    return pl.DataFrame(
        {
            "st": np.where(i < 30, "a", "b"),
            "psu": i // 5,
            "w": np.where((i < 55) & (i % 10 != 9), 1.0 + i % 4, 0.0),
            "y": ((i * 37) % 23) / 4 - 2,
            "x": 1 + ((i * 11) % 7) / 10,
            "b": ((i * 13) % 3 == 0).astype(int),
            "r": np.where(i < 55, "z", "q"),
            "g": np.where(i % 2 == 0, "u", "v"),
            "k": np.where(i % 10 == 9, "m", "n"),
        }
    )


@pytest.fixture(scope="module")
def taylor(data) -> svy.Sample:
    return svy.Sample(data, svy.Design(stratum="st", psu="psu", wgt="w"))


@pytest.fixture(scope="module")
def bs(taylor) -> svy.Sample:
    return taylor.weighting.create_bs_wgts(n_reps=40, rstate=7)


@pytest.fixture(scope="module")
def jk(taylor) -> svy.Sample:
    return taylor.weighting.create_jk_wgts()


@pytest.fixture(scope="module", params=["taylor", "bs", "jk"])
def design(request):
    """(sample, method kwargs) for each variance method."""
    s = request.getfixturevalue(request.param)
    kw = {} if request.param == "taylor" else {"method": "replication"}
    return request.param, s, kw


def _rows(res) -> dict:
    """Map (*by_level, y_level) -> ParamEst."""
    out = {}
    for e in res.estimates:
        key = tuple(e.by_level or ()) + ((str(e.y_level),) if e.y_level is not None else ())
        out[key] = e
    return out


def _assert_undefined(e):
    assert math.isnan(e.est)
    assert math.isnan(e.se)
    assert math.isnan(e.lci) and math.isnan(e.uci)


def _assert_r(e, ref):
    est, se = ref
    assert e.est == pytest.approx(est, abs=TOL)
    assert e.se == pytest.approx(se, abs=TOL)
    assert e.df == R_DF


# ---------------------------------------------------------------------------
# Taylor matches R: the weighted domains exactly, the empty one as NaN
# ---------------------------------------------------------------------------


class TestTaylorMatchesR:
    def test_mean_by(self, taylor):
        rows = _rows(taylor.estimation.mean("y", by="r"))
        assert set(rows) == {("q",), ("z",)}
        _assert_r(rows[("z",)], R_MEAN[("z",)])
        _assert_undefined(rows[("q",)])
        assert rows[("q",)].df == 0

    def test_mean_by_two_columns(self, taylor):
        rows = _rows(taylor.estimation.mean("y", by=["r", "g"]))
        assert set(rows) == {("q", "u"), ("q", "v"), ("z", "u"), ("z", "v")}
        for key, ref in R_MEAN_RG.items():
            _assert_r(rows[key], ref)
        _assert_undefined(rows[("q", "u")])
        _assert_undefined(rows[("q", "v")])

    def test_mean_by_scattered_zero_level(self, taylor):
        rows = _rows(taylor.estimation.mean("y", by="k"))
        assert set(rows) == {("m",), ("n",)}
        _assert_r(rows[("n",)], R_MEAN_K[("n",)])
        _assert_undefined(rows[("m",)])

    def test_prop_by(self, taylor):
        rows = _rows(taylor.estimation.prop("b", by="r"))
        assert set(rows) == {("q", "0"), ("q", "1"), ("z", "0"), ("z", "1")}
        for key, ref in R_PROP.items():
            _assert_r(rows[key], ref)
        _assert_undefined(rows[("q", "0")])
        _assert_undefined(rows[("q", "1")])

    def test_mean_as_factor_by(self, taylor):
        rows = _rows(taylor.estimation.mean("b", by="r", as_factor=True))
        for key, ref in R_PROP.items():
            _assert_r(rows[key], ref)
        _assert_undefined(rows[("q", "0")])
        _assert_undefined(rows[("q", "1")])

    def test_ratio_by(self, taylor):
        rows = _rows(taylor.estimation.ratio("y", "x", by="r"))
        assert set(rows) == {("q",), ("z",)}
        _assert_r(rows[("z",)], R_RATIO[("z",)])
        _assert_undefined(rows[("q",)])

    def test_total_by_is_zero(self, taylor):
        rows = _rows(taylor.estimation.total("y", by="r"))
        _assert_r(rows[("z",)], R_TOTAL[("z",)])
        assert rows[("q",)].est == 0.0
        assert rows[("q",)].se == 0.0

    def test_median_by(self, taylor):
        rows = _rows(taylor.estimation.median("y", by="r"))
        _assert_r(rows[("z",)], R_MEDIAN[("z",)])
        _assert_undefined(rows[("q",)])


# ---------------------------------------------------------------------------
# Every variance method: the zero-weight domain keeps its row, NaN
# ---------------------------------------------------------------------------

ESTIMATORS = {
    "mean": lambda e, **kw: e.mean("y", **kw),
    "prop": lambda e, **kw: e.prop("b", **kw),
    "mean_factor": lambda e, **kw: e.mean("b", as_factor=True, **kw),
    "ratio": lambda e, **kw: e.ratio("y", "x", **kw),
    "median": lambda e, **kw: e.median("y", **kw),
}


class TestEveryMethod:
    @pytest.mark.parametrize("name", sorted(ESTIMATORS))
    @pytest.mark.parametrize("by", ["r", "k", ["r", "g"]])
    def test_by_keeps_every_level(self, design, name, by):
        _, s, kw = design
        res = ESTIMATORS[name](s.estimation, by=by, **kw)
        rows = _rows(res)
        empty = {"r": "q", "k": "m"}[by if isinstance(by, str) else by[0]]
        undefined = [e for key, e in rows.items() if key[0] == empty]
        defined = [e for key, e in rows.items() if key[0] != empty]
        assert undefined and defined
        for e in undefined:
            _assert_undefined(e)
        for e in defined:
            assert math.isfinite(e.est) and math.isfinite(e.se) and e.se > 0

    @pytest.mark.parametrize("name", sorted(ESTIMATORS))
    def test_by_row_equals_where(self, design, name):
        """The weighted domain's by= row is its where= estimate."""
        _, s, kw = design
        by_rows = _rows(ESTIMATORS[name](s.estimation, by="r", **kw))
        where_rows = _rows(ESTIMATORS[name](s.estimation, where=pl.col("r") == "z", **kw))
        for key, w in where_rows.items():
            b = by_rows[("z",) + key]
            assert b.est == pytest.approx(w.est, abs=TOL)
            assert b.se == pytest.approx(w.se, abs=TOL)

    @pytest.mark.parametrize("name", sorted(ESTIMATORS))
    @pytest.mark.parametrize(
        "where",
        [pl.col("r") == "q", pl.col("k") == "m", pl.col("r") == "absent"],
        ids=["whole_psu", "scattered", "no_match"],
    )
    def test_where_gives_undefined_rows(self, design, name, where):
        _, s, kw = design
        res = ESTIMATORS[name](s.estimation, where=where, **kw)
        assert res.estimates
        for e in res.estimates:
            _assert_undefined(e)

    @pytest.mark.parametrize(
        "where",
        [pl.col("r") == "q", pl.col("r") == "absent"],
        ids=["zero_weight", "no_match"],
    )
    def test_where_total_is_zero(self, design, where):
        _, s, kw = design
        (e,) = s.estimation.total("y", where=where, **kw).estimates
        assert e.est == 0.0
        assert e.se == 0.0

    def test_quantile_by(self, design):
        _, s, kw = design
        res = s.estimation.quantile("y", p=[0.25, 0.75], by="r", **kw)
        for member in res:
            rows = _rows(member)
            assert set(rows) == {("q",), ("z",)}
            _assert_undefined(rows[("q",)])
            assert math.isfinite(rows[("z",)].est)


# ---------------------------------------------------------------------------
# Batched variables, covariance, contrasts, whole-sample zero weights
# ---------------------------------------------------------------------------


class TestAroundTheRow:
    def test_batched_mean_where(self, design):
        _, s, kw = design
        res = s.estimation.mean(["y", "x"], where=pl.col("r") == "q", **kw)
        assert len(res) == 2
        for member in res:
            (e,) = member.estimates
            _assert_undefined(e)

    def test_batched_mean_by_matches_single(self, design):
        _, s, kw = design
        batched = s.estimation.mean(["y", "x"], by="r", **kw)
        for yy, member in zip(["y", "x"], batched):
            single = _rows(s.estimation.mean(yy, by="r", **kw))
            rows = _rows(member)
            _assert_undefined(rows[("q",)])
            assert rows[("z",)].est == pytest.approx(single[("z",)].est, abs=TOL)
            assert rows[("z",)].se == pytest.approx(single[("z",)].se, abs=TOL)

    @pytest.mark.parametrize(
        "call",
        [
            lambda e, kw: e.mean("y", by="r", **kw),
            lambda e, kw: e.prop("b", by="r", **kw),
            lambda e, kw: e.ratio("y", "x", by="r", **kw),
        ],
        ids=["mean", "prop", "ratio"],
    )
    def test_covariance_masks_undefined_domain(self, design, call):
        _, s, kw = design
        res = call(s.estimation, kw)
        cov = np.asarray(res.covariance)
        undefined = np.array([math.isnan(e.est) for e in res.estimates])
        assert undefined.any() and not undefined.all()
        assert np.isnan(cov[undefined, :]).all()
        assert np.isnan(cov[:, undefined]).all()
        defined = cov[np.ix_(~undefined, ~undefined)]
        assert np.isfinite(defined).all()
        ses = np.array([e.se for e in res.estimates])[~undefined]
        np.testing.assert_allclose(np.sqrt(np.diag(defined)), ses, rtol=1e-12)

    def test_contrast_ignoring_undefined_domain(self, taylor):
        res = taylor.estimation.mean("y", by="r")
        c = res.contrast({"z": 2}).to_polars()
        assert c["est"][0] == pytest.approx(2 * R_MEAN[("z",)][0], abs=TOL)
        assert c["se"][0] == pytest.approx(2 * R_MEAN[("z",)][1], abs=TOL)

    def test_contrast_with_undefined_domain_is_nan(self, taylor):
        res = taylor.estimation.mean("y", by="r")
        c = res.contrast(svy.estd("z") - svy.estd("q")).to_polars()
        assert math.isnan(c["est"][0])
        assert math.isnan(c["se"][0])

    def test_all_weights_zero(self, data):
        s = svy.Sample(
            data.with_columns(w=pl.lit(0.0)), svy.Design(stratum="st", psu="psu", wgt="w")
        )
        (e,) = s.estimation.mean("y").estimates
        _assert_undefined(e)
        assert len(s.estimation.prop("b").estimates) == 2
        (t,) = s.estimation.total("y").estimates
        assert t.est == 0.0

    def test_ratio_zero_denominator_with_weight(self, data):
        """Weighted rows whose x sums to zero: that domain only is undefined."""
        s = svy.Sample(
            data.with_columns(x=pl.when(pl.col("g") == "u").then(0.0).otherwise(pl.col("x"))),
            svy.Design(stratum="st", psu="psu", wgt="w"),
        )
        rows = _rows(s.estimation.ratio("y", "x", by="g"))
        assert not math.isfinite(rows[("u",)].est)
        assert math.isfinite(rows[("v",)].est) and rows[("v",)].se > 0

    def test_deff_and_to_polars(self, taylor):
        res = taylor.estimation.mean("y", by="r", deff="wr")
        rows = _rows(res)
        assert math.isnan(rows[("q",)].deff)
        assert math.isfinite(rows[("z",)].deff)
        df = res.to_polars()
        assert df.height == 2
        assert df.filter(pl.col("r") == "q")["est"].is_nan().all()
