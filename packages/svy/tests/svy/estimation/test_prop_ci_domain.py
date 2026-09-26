# tests/svy/estimation/test_prop_ci_domain.py
"""
Domain sample size and degenerate cases in proportion confidence intervals.

`beta`, `korn-graubard` and `wilson` read a respondent count `n` from the
kernel. It must be the domain count n_d -- rows in the domain (and by-level)
with a nonzero weight -- whatever way the domain is expressed. Under `where=`
out-of-domain rows stay in the frame with zero weight, and the kernels used to
count them, so a `where=` domain got the whole-sample `n`.

Rules pinned here:
  - n_d counts rows in the domain with weight != 0 (zero-weight rows represent
    no population units; negative calibrated weights do).
  - df <= 0: every method returns NaN bounds.
  - p = 0 or 1: logit, beta and wilson return NaN bounds and warn
    PROP_CI_BOUNDARY; korn-graubard returns its one-sided interval.
  - se = 0 with 0 < p < 1: every method returns [p, p].

Data: 2 strata x 4 PSUs x 10 persons. Domain female == 1 is persons 1-5 of
every PSU: n_d = 40 in 8 PSUs, df = 6.

R reference (survey 4.5):
    des <- svydesign(ids=~psu, strata=~stratum, weights=~wgt, data=d, nest=TRUE)
    svyciprop(~I(y==1), subset(des, female==1), method="beta")
    svyciprop(~I(y==1), subset(des, female==1 & stratum==h), method="beta")
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from scipy import stats

from svy import Design, Sample, SvyUserWarning, col
from svy.core.warnings import WarnCode


ALPHA = 0.05
R_TOL = 1e-9
CI_METHODS = ["logit", "beta", "wilson", "korn-graubard"]
NAN_AT_BOUNDARY = ["logit", "beta", "wilson"]

_MID = {1: [1, 2, 0, 3], 2: [2, 1, 4, 1]}
_FLAT = {1: [2, 2, 2, 3], 2: [2, 3, 2, 2]}


def _frame() -> pl.DataFrame:
    rows = []
    for h in (1, 2):
        for k in range(4):
            for j in range(10):
                fem = j < 5
                rows.append(
                    {
                        "stratum": h,
                        "psu": f"{h}-{k + 1}",
                        "female": int(fem),
                        "wgt": 10.0 if h == 1 else 20.0,
                        "y_mid": int(j < _MID[h][k]) if fem else int(j < 7),
                        "y_zero": 0 if fem else int(j < 8),
                        "y_flat": int(j < _FLAT[h][k]) if fem else int(j < 6),
                    }
                )
    return pl.DataFrame(rows).with_columns(
        dom=pl.format("{}_{}", pl.col("female"), pl.col("stratum")),
    )


DESIGNS = {
    "weight_only": dict(wgt="wgt"),
    "stratum": dict(stratum="stratum", wgt="wgt"),
    "psu": dict(psu="psu", wgt="wgt"),
    "stratum_psu": dict(stratum="stratum", psu="psu", wgt="wgt"),
}


def _sample(frame: pl.DataFrame | None = None, design: str = "stratum_psu") -> Sample:
    return Sample(_frame() if frame is None else frame, Design(**DESIGNS[design]))


def _key(v) -> str:
    return str(v)


def _cells(result) -> dict:
    """{(by_level, y_level): ParamEst}, levels stringified."""
    out = {}
    for e in result.estimates:
        by = None if e.by_level is None else tuple(_key(b) for b in e.by_level)
        out[(by, _key(e.y_level))] = e
    return out


def _bounds_equal(a, b, tol=1e-12) -> bool:
    for x, y in ((a.lci, b.lci), (a.uci, b.uci)):
        if math.isnan(x) or math.isnan(y):
            if not (math.isnan(x) and math.isnan(y)):
                return False
        elif abs(x - y) > tol:
            return False
    return abs(a.est - b.est) <= tol and abs(a.se - b.se) <= tol and a.df == b.df


def _kg_boundary_bound(n_d: int, df: int) -> float:
    """KG at p = 0: upper bound 1 - (alpha/2)^(1/n*), n* = min(n_d, n_d * t_adj)."""
    t_adj = (stats.t.ppf(1 - ALPHA / 2, n_d - 1) / stats.t.ppf(1 - ALPHA / 2, df)) ** 2
    n_star = min(n_d, n_d * t_adj)
    return 1 - (ALPHA / 2) ** (1 / n_star)


def _beta_ci(p: float, se: float, n_d: int, df: int) -> tuple[float, float]:
    """R's svyciprop(method="beta") with n = n_d."""
    n_eff = p * (1 - p) / se**2
    n_eff *= (stats.t.ppf(ALPHA / 2, n_d - 1) / stats.t.ppf(ALPHA / 2, df)) ** 2
    return (
        stats.beta.ppf(ALPHA / 2, n_eff * p, n_eff * (1 - p) + 1),
        stats.beta.ppf(1 - ALPHA / 2, n_eff * p + 1, n_eff * (1 - p)),
    )


# y_zero sits at p = 0/1 in a domain: a side effect for tests about other things.
BOUNDARY_EXPECTED = pytest.mark.filterwarnings(r"ignore:\[PROP_CI_BOUNDARY\]:svy.SvyUserWarning")


def _boundary_warnings(sample: Sample) -> list:
    return sample.warnings.list(code=WarnCode.PROP_CI_BOUNDARY)


# =============================================================================
# 1. R reference values (Taylor)
# =============================================================================


R_BETA = {
    # (domain, y): (p, se, lci, uci)
    ("female", "y_mid"): (
        0.366666666666667,
        0.103637545034320,
        0.138846801273085,
        0.649366612413814,
    ),
    ("female", "y_flat"): (0.45, 0.037267799624996, 0.359739743149435, 0.542749458006750),
    ("female&stratum1", "y_mid"): (0.3, 0.129099444873581, 0.026912858673537, 0.772658049163753),
    ("female&stratum2", "y_mid"): (0.4, 0.141421356237310, 0.056027190573667, 0.847107899229998),
    ("female&stratum1", "y_flat"): (0.45, 0.05, 0.297777350798571, 0.609395087641518),
    ("female&stratum2", "y_flat"): (0.45, 0.05, 0.297777350798571, 0.609395087641518),
}


def _assert_r(e, ref):
    p, se, lci, uci = ref
    assert e.est == pytest.approx(p, abs=R_TOL)
    assert e.se == pytest.approx(se, abs=R_TOL)
    assert e.lci == pytest.approx(lci, abs=R_TOL)
    assert e.uci == pytest.approx(uci, abs=R_TOL)


@pytest.mark.parametrize("y", ["y_mid", "y_flat"])
class TestBetaMatchesR:
    def test_where(self, y):
        r = _sample().estimation.prop(y, where=col("female") == 1, ci_method="beta")
        _assert_r(_cells(r)[(None, "1")], R_BETA[("female", y)])

    def test_by(self, y):
        r = _sample().estimation.prop(y, by="female", ci_method="beta")
        _assert_r(_cells(r)[(("1",), "1")], R_BETA[("female", y)])

    def test_where_and_by(self, y):
        r = _sample().estimation.prop(y, where=col("female") == 1, by="stratum", ci_method="beta")
        cells = _cells(r)
        for h in (1, 2):
            _assert_r(cells[((str(h),), "1")], R_BETA[(f"female&stratum{h}", y)])

    def test_multi_condition_where(self, y):
        r = _sample().estimation.prop(
            y, where=[col("female") == 1, col("stratum") == 2], ci_method="beta"
        )
        _assert_r(_cells(r)[(None, "1")], R_BETA[("female&stratum2", y)])


# =============================================================================
# 2. Korn-Graubard closed forms
# =============================================================================


class TestKornGraubardClosedForm:
    @pytest.mark.parametrize(
        "kwargs, by_level, n_d, df",
        [
            (dict(where=col("female") == 1), None, 40, 6),
            (dict(by="female"), ("1",), 40, 6),
            (dict(where=col("female") == 1, by="stratum"), ("1",), 20, 3),
            (dict(where=[col("female") == 1, col("stratum") == 2]), None, 20, 3),
        ],
    )
    def test_boundary(self, kwargs, by_level, n_d, df):
        r = _sample().estimation.prop("y_zero", ci_method="korn-graubard", **kwargs)
        cells = _cells(r)
        bound = _kg_boundary_bound(n_d, df)
        p0, p1 = cells[(by_level, "1")], cells[(by_level, "0")]
        assert (p0.est, p0.df) == (0.0, df)
        assert p0.lci == 0.0
        assert p0.uci == pytest.approx(bound, abs=1e-12)
        assert (p1.est, p1.uci) == (1.0, 1.0)
        assert p1.lci == pytest.approx(1 - bound, abs=1e-12)

    @pytest.mark.parametrize("kwargs", [dict(where=col("female") == 1), dict(by="female")])
    def test_cap_binds_at_domain_n(self, kwargs):
        # y_flat has deff < 1: n_eff * t_adj = 121.8, capped at n_d = 40.
        r = _sample().estimation.prop("y_flat", ci_method="korn-graubard", **kwargs)
        e = next(v for (b, lv), v in _cells(r).items() if lv == "1" and b in (None, ("1",)))
        x = 0.45 * 40
        assert e.lci == pytest.approx(stats.beta.ppf(ALPHA / 2, x, 40 - x + 1), abs=1e-12)
        assert e.uci == pytest.approx(stats.beta.ppf(1 - ALPHA / 2, x + 1, 40 - x), abs=1e-12)


# =============================================================================
# 3. Invariants: the domain count does not depend on how the domain is spelled
# =============================================================================


@pytest.mark.parametrize("design", list(DESIGNS))
@pytest.mark.parametrize("method", CI_METHODS)
class TestDomainSpellingInvariance:
    @BOUNDARY_EXPECTED
    @pytest.mark.parametrize("y", ["y_mid", "y_zero", "y_flat"])
    def test_where_equals_by(self, design, method, y):
        s = _sample(design=design)
        w = _cells(s.estimation.prop(y, where=col("female") == 1, ci_method=method))
        b = _cells(s.estimation.prop(y, by="female", ci_method=method))
        for lv in ("0", "1"):
            assert _bounds_equal(w[(None, lv)], b[(("1",), lv)]), (
                lv,
                w[(None, lv)],
                b[(("1",), lv)],
            )

    @pytest.mark.parametrize("y", ["y_mid", "y_flat"])
    def test_where_and_by_equals_by_on_combined_domain(self, design, method, y):
        s = _sample(design=design)
        wb = _cells(s.estimation.prop(y, where=col("female") == 1, by="stratum", ci_method=method))
        comb = _cells(s.estimation.prop(y, by="dom", ci_method=method))
        for h in ("1", "2"):
            for lv in ("0", "1"):
                assert _bounds_equal(wb[((h,), lv)], comb[((f"1_{h}",), lv)])


@pytest.mark.parametrize("method", CI_METHODS)
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(where=col("female") == 1),
        dict(by="female"),
        dict(where=col("female") == 1, by="stratum"),
    ],
    ids=["where", "by", "where_by"],
)
@pytest.mark.parametrize("y", ["y_mid", "y_zero", "y_flat"])
@BOUNDARY_EXPECTED
def test_zero_weight_rows_change_nothing(method, kwargs, y):
    base = _frame()
    extra = base.filter(pl.col("female") == 1).with_columns(
        wgt=pl.lit(0.0), y_mid=1 - pl.col("y_mid"), y_flat=1 - pl.col("y_flat")
    )
    plain = _cells(_sample(base).estimation.prop(y, ci_method=method, **kwargs))
    padded = _cells(
        _sample(pl.concat([base, extra])).estimation.prop(y, ci_method=method, **kwargs)
    )
    assert plain.keys() == padded.keys()
    for k in plain:
        assert _bounds_equal(plain[k], padded[k]), (k, plain[k], padded[k])


@pytest.mark.parametrize("method", CI_METHODS)
@BOUNDARY_EXPECTED
def test_poststratified_equals_plain(method):
    # Controls equal the sample's own stratum totals: weights, estimates and
    # SEs are unchanged, so no interval may move. (R's subset of a calibrated
    # design keeps every row and reports nrow() = 80 here.)
    s = _sample()
    ps = s.weighting.poststratify(controls={1: 400.0, 2: 800.0}, cells="stratum")
    for y in ("y_mid", "y_zero", "y_flat"):
        a = _cells(s.estimation.prop(y, where=col("female") == 1, ci_method=method))
        b = _cells(ps.estimation.prop(y, where=col("female") == 1, ci_method=method))
        for k in a:
            assert _bounds_equal(a[k], b[k], tol=1e-9), (y, k, a[k], b[k])


@pytest.mark.parametrize("method", CI_METHODS)
@BOUNDARY_EXPECTED
def test_multi_variable_path_matches_single(method):
    s = _sample()
    ys = ["y_mid", "y_zero", "y_flat"]
    multi = s.estimation.prop(ys, where=col("female") == 1, ci_method=method)
    for y, r in zip(ys, multi):
        single = _cells(s.estimation.prop(y, where=col("female") == 1, ci_method=method))
        for k, e in _cells(r).items():
            assert _bounds_equal(e, single[k]), (y, k)


class TestDropNulls:
    def _frame_with_nulls(self):
        f = _frame().with_row_index("rid")
        null_rows = f.filter(pl.col("female") == 1)["rid"].head(3).to_list()
        return f.with_columns(
            y_zero=pl.when(pl.col("rid").is_in(null_rows)).then(None).otherwise(pl.col("y_zero"))
        ).drop("rid")

    @pytest.mark.parametrize(
        "kwargs, by_level",
        [(dict(where=col("female") == 1), None), (dict(by="female"), ("1",))],
    )
    def test_null_rows_leave_the_domain_count(self, kwargs, by_level):
        s = _sample(self._frame_with_nulls())
        r = s.estimation.prop("y_zero", ci_method="korn-graubard", drop_nulls=True, **kwargs)
        e = _cells(r)[(by_level, "1")]
        assert e.uci == pytest.approx(_kg_boundary_bound(37, e.df), abs=1e-12)


# =============================================================================
# 4. Replication: same n_d as Taylor
# =============================================================================


def _rep_sample(kind: str) -> Sample:
    if kind == "brr":
        f = _frame().filter(pl.col("psu").str.ends_with("1") | pl.col("psu").str.ends_with("2"))
        return _sample(f).weighting.create_brr_wgts()
    s = _sample()
    if kind == "jackknife":
        return s.weighting.create_jk_wgts()
    if kind == "bootstrap":
        return s.weighting.create_bs_wgts(20, rstate=3)
    return s.weighting.create_sdr_wgts(8)


REP_N_D = {"jackknife": 40, "bootstrap": 40, "sdr": 40, "brr": 20}


@pytest.mark.parametrize("kind", list(REP_N_D))
class TestReplication:
    @pytest.mark.parametrize(
        "kwargs, by_level, n_d_scale",
        [
            (dict(where=col("female") == 1), None, 1.0),
            (dict(by="female"), ("1",), 1.0),
            (dict(where=col("female") == 1, by="stratum"), ("1",), 0.5),
        ],
        ids=["where", "by", "where_by"],
    )
    def test_kg_boundary_uses_domain_n(self, kind, kwargs, by_level, n_d_scale):
        s = _rep_sample(kind)
        r = s.estimation.prop("y_zero", ci_method="korn-graubard", method="replication", **kwargs)
        e = _cells(r)[(by_level, "1")]
        n_d = int(REP_N_D[kind] * n_d_scale)
        assert e.uci == pytest.approx(_kg_boundary_bound(n_d, e.df), abs=1e-12)

    @BOUNDARY_EXPECTED
    @pytest.mark.parametrize("method", CI_METHODS)
    def test_where_equals_by(self, kind, method):
        s = _rep_sample(kind)
        for y in ("y_mid", "y_zero"):
            w = _cells(
                s.estimation.prop(
                    y, where=col("female") == 1, ci_method=method, method="replication"
                )
            )
            b = _cells(s.estimation.prop(y, by="female", ci_method=method, method="replication"))
            for lv in ("0", "1"):
                assert _bounds_equal(w[(None, lv)], b[(("1",), lv)], tol=1e-10), (y, lv)

    def test_boundary_nan_and_warning(self, kind):
        s = _rep_sample(kind)
        with pytest.warns(SvyUserWarning, match=r"\[PROP_CI_BOUNDARY\]"):
            r = s.estimation.prop(
                "y_zero", where=col("female") == 1, ci_method="beta", method="replication"
            )
        for e in r.estimates:
            assert math.isnan(e.lci) and math.isnan(e.uci)
        assert _boundary_warnings(s)


# =============================================================================
# 5. Zero and negative weights inside the domain
# =============================================================================


class TestNonzeroWeightCount:
    def test_zero_weight_psu_is_not_counted(self):
        # PSU 2-4 carries weight 0: domain rows 40, nonzero-weight rows 35, df 5.
        # R counts nrow() = 40 here; svy counts 35 (documented difference).
        f = _frame().with_columns(
            wgt=pl.when(pl.col("psu") == "2-4").then(0.0).otherwise(pl.col("wgt"))
        )
        s = _sample(f)
        mid = _cells(s.estimation.prop("y_mid", where=col("female") == 1, ci_method="beta"))[
            (None, "1")
        ]
        assert mid.df == 5
        assert mid.se == pytest.approx(0.1131371, abs=1e-7)  # R: same p and se
        lci, uci = _beta_ci(mid.est, mid.se, 35, 5)
        assert (mid.lci, mid.uci) == (pytest.approx(lci, abs=1e-12), pytest.approx(uci, abs=1e-12))

        zero = _cells(s.estimation.prop("y_zero", where=col("female") == 1, ci_method="kg"))[
            (None, "1")
        ]
        assert zero.uci == pytest.approx(_kg_boundary_bound(35, 5), abs=1e-12)

    def test_negative_weight_is_counted(self):
        f = _frame().with_row_index("rid")
        rid = f.filter((pl.col("female") == 1) & (pl.col("psu") == "1-1"))["rid"][0]
        f = f.with_columns(
            wgt=pl.when(pl.col("rid") == rid).then(-5.0).otherwise(pl.col("wgt"))
        ).drop("rid")
        r = _sample(f).estimation.prop("y_zero", where=col("female") == 1, ci_method="kg")
        e = _cells(r)[(None, "1")]
        assert e.uci == pytest.approx(_kg_boundary_bound(40, 6), abs=1e-12)


def test_kernel_n_is_nonzero_weight_count():
    import svy_rs as rs

    df = pl.DataFrame(
        {
            "y": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            "w": [1.0, 0.0, 2.0, -1.0, 0.0, 3.0],
            "g": ["a", "a", "a", "b", "b", "b"],
        }
    )
    ungrouped, _ = rs.taylor_prop(df, value_col="y", weight_col="w")
    assert set(ungrouped["n"].to_list()) == {4}
    grouped, _ = rs.taylor_prop(df, value_col="y", weight_col="w", by_col="g")
    assert dict(zip(grouped["g"], grouped["n"])) == {"a": 2, "b": 2}


# =============================================================================
# 6. Degenerate cases
# =============================================================================


class TestBoundary:
    @pytest.mark.parametrize("method", NAN_AT_BOUNDARY)
    @pytest.mark.parametrize(
        "kwargs, by_level",
        [(dict(where=col("female") == 1), None), (dict(by="female"), ("1",))],
    )
    def test_nan_and_warning(self, method, kwargs, by_level):
        s = _sample()
        with pytest.warns(SvyUserWarning, match=r"\[PROP_CI_BOUNDARY\]"):
            cells = _cells(s.estimation.prop("y_zero", ci_method=method, **kwargs))
        for lv, p in (("1", 0.0), ("0", 1.0)):
            e = cells[(by_level, lv)]
            assert e.est == p
            assert math.isnan(e.lci) and math.isnan(e.uci)
        (w,) = _boundary_warnings(s)
        assert "korn-graubard" in w.hint
        assert w.got == method

    @BOUNDARY_EXPECTED
    def test_non_boundary_cells_keep_their_interval(self):
        # Men's y_zero varies across PSUs here, so their cell has se > 0.
        f = (
            _frame()
            .with_row_index("rid")
            .with_columns(
                y_zero=pl.when(pl.col("female") == 0)
                .then((pl.col("rid") % 10 < 5 + pl.col("rid") // 10 % 3).cast(pl.Int64))
                .otherwise(pl.col("y_zero"))
            )
        )
        s = _sample(f.drop("rid"))
        cells = _cells(s.estimation.prop("y_zero", by="female", ci_method="beta"))
        men = cells[(("0",), "1")]
        assert men.se > 0
        assert 0 < men.lci < men.est < men.uci < 1
        assert math.isnan(cells[(("1",), "1")].lci)

    def test_korn_graubard_has_no_warning(self):
        s = _sample()
        s.estimation.prop("y_zero", where=col("female") == 1, ci_method="korn-graubard")
        assert not _boundary_warnings(s)

    def test_no_warning_without_boundary(self):
        s = _sample()
        s.estimation.prop("y_mid", where=col("female") == 1, ci_method="beta")
        assert not _boundary_warnings(s)


class TestNoResidualDf:
    # The 5 women of PSU 1-1: one PSU, df = 0. R returns NaN for every method.
    WHERE = [col("female") == 1, col("psu") == "1-1"]

    @pytest.mark.parametrize("method", CI_METHODS)
    @pytest.mark.parametrize("y", ["y_mid", "y_zero"])
    def test_all_methods_nan(self, method, y):
        s = _sample()
        r = s.estimation.prop(y, where=self.WHERE, ci_method=method)
        for e in r.estimates:
            assert e.df == 0
            assert math.isnan(e.lci) and math.isnan(e.uci), (method, e)
        assert not _boundary_warnings(s)

    @pytest.mark.parametrize("method", CI_METHODS)
    def test_whole_psu_domain_with_float_noise_se(self, method):
        # A whole-PSU domain gives an SE of ~1e-17 rather than exactly 0.
        s = _sample()
        r = s.estimation.prop("y_mid", where=col("psu") == "1-1", ci_method=method)
        for e in r.estimates:
            assert math.isnan(e.lci) and math.isnan(e.uci), (method, e)


class TestZeroSe:
    @pytest.mark.parametrize("method", CI_METHODS)
    def test_interior_zero_se_is_a_point(self, method):
        # Men: y_mid is 2 of 5 in every PSU, so every PSU total is equal and the
        # design-based variance is zero. Not a boundary: [p, p] for every method.
        s = _sample()
        e = _cells(s.estimation.prop("y_mid", where=col("female") == 0, ci_method=method))[
            (None, "1")
        ]
        assert e.est == pytest.approx(0.4, abs=1e-12)
        assert e.se == pytest.approx(0.0, abs=1e-12)
        assert e.lci == pytest.approx(e.est, abs=1e-12)
        assert e.uci == pytest.approx(e.est, abs=1e-12)
        assert not _boundary_warnings(s)


class TestTolerance:
    est = _sample().estimation

    @pytest.mark.parametrize("method", NAN_AT_BOUNDARY)
    def test_p_within_noise_of_one_is_a_boundary(self, method):
        lci, uci = self.est._compute_prop_ci(1 - 1e-15, 1e-17, ALPHA, 6, 40, method)
        assert math.isnan(lci) and math.isnan(uci)

    def test_kg_p_within_noise_of_one(self):
        lci, uci = self.est._compute_prop_ci(1 - 1e-15, 1e-17, ALPHA, 6, 40, "korn-graubard")
        assert uci == 1.0
        assert lci == pytest.approx(1 - _kg_boundary_bound(40, 6), abs=1e-12)

    @pytest.mark.parametrize("method", CI_METHODS)
    def test_se_within_noise_of_zero_is_a_point(self, method):
        lci, uci = self.est._compute_prop_ci(0.3, 1e-17, ALPHA, 6, 40, method)
        assert (lci, uci) == (pytest.approx(0.3, abs=1e-12), pytest.approx(0.3, abs=1e-12))

    def test_vectorised_logit_matches_scalar(self):
        s = _sample()
        r = s.estimation.prop("y_mid", where=col("psu") == "1-1", ci_method="logit")
        assert all(math.isnan(e.lci) for e in r.estimates)
        assert np.isfinite(
            _cells(s.estimation.prop("y_mid", where=col("female") == 1, ci_method="logit"))[
                (None, "1")
            ].lci
        )
