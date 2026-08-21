# tests/svy/estimation/test_association.py
"""Tests for the design-based ``corr`` and ``cov`` estimators.

Golden values come from R survey 4.5 on a single-stage cluster design (8 PSUs
of 3 rows, unequal weights)::

    d  <- svydesign(id=~psu, weights=~w, data=dat)
    v  <- svyvar(~y+x, d)                        # covariance and its SE
    ct <- svycontrast(svymean(~y+x+yy+xx+yx, d),
                      quote((yx - y*x)/sqrt((yy-y^2)*(xx-x^2))))
    rd <- as.svrepdesign(d, type="JK1"); svyvar(~y+x, rd)

The correlation SE comes from ``svycontrast`` rather than ``svyvar`` because R
has no correlation SE of its own: ``svycontrast`` applies its own delta method
over the moment means, so agreement is an independent check of the
linearization rather than a restatement of the same formula.
"""

from __future__ import annotations

import polars as pl
import pytest

from svy import Sample
from svy.core.design import Design, RepWeights
from svy.core.enumerations import PopParam
from svy.errors import DimensionError, MethodError


R_CORR = 0.52860852253793722
R_CORR_SE = 0.15921590459764234
R_COV = 50.138658078368209
R_COV_SE = 21.498361515848689
R_VAR_Y = 139.54576489533011
R_COV_STRAT_SE = 23.178556453025269
R_COV_JK1_SE = 29.751032168261617

_Y = [
    12,
    15,
    11,
    22,
    25,
    19,
    31,
    29,
    35,
    41,
    44,
    39,
    18,
    16,
    21,
    27,
    30,
    24,
    36,
    33,
    38,
    45,
    49,
    43,
]
_X = [9, 4, 14, 6, 17, 8, 21, 11, 13, 12, 26, 7, 19, 5, 23, 10, 28, 15, 8, 22, 16, 18, 31, 20]
_W = [2, 2, 2, 3, 3, 3, 1.5, 1.5, 1.5, 4, 4, 4, 2.5, 2.5, 2.5, 3.5, 3.5, 3.5, 1, 1, 1, 5, 5, 5]
_PSU = [i // 3 + 1 for i in range(24)]


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": list(range(24)),
            "y": [float(v) for v in _Y],
            "x": [float(v) for v in _X],
            # Deliberately an exact linear function of x, so corr(x, z) must be
            # 1 and corr(y, z) must equal corr(y, x) -- a scale/shift check.
            "z": [float(v) * 1.7 + 3.0 for v in _X],
            "w": [float(v) for v in _W],
            "psu": _PSU,
            "stratum": ["1"] * 12 + ["2"] * 12,
            "grp": ["A"] * 12 + ["B"] * 12,
        }
    )


@pytest.fixture
def sample() -> Sample:
    return Sample(_frame(), Design(row_index="id", wgt="w", psu="psu"))


@pytest.fixture
def strat_sample() -> Sample:
    return Sample(_frame(), Design(row_index="id", wgt="w", psu="psu", stratum="stratum"))


@pytest.fixture
def rep_sample() -> Sample:
    df = _frame()
    reps = {
        f"jk_{r + 1}": [0.0 if _PSU[i] == r + 1 else _W[i] * 8 / 7 for i in range(24)]
        for r in range(8)
    }
    df = df.with_columns([pl.Series(k, v) for k, v in reps.items()])
    design = Design(
        row_index="id",
        wgt="w",
        psu="psu",
        rep_wgts=RepWeights(method="Jackknife", prefix="jk_", n_reps=8, df=7),
    )
    return Sample(df, design)


def _row(estimate, i: int = 0) -> dict:
    return estimate.to_dicts()[i]


# ---------------------------------------------------------------------------
# Agreement with R
# ---------------------------------------------------------------------------


def test_corr_matches_r(sample):
    r = _row(sample.estimation.corr(("y", "x")))
    assert r["est"] == pytest.approx(R_CORR, rel=1e-10)
    assert r["se"] == pytest.approx(R_CORR_SE, rel=1e-9)


def test_cov_matches_r(sample):
    r = _row(sample.estimation.cov(("y", "x")))
    assert r["est"] == pytest.approx(R_COV, rel=1e-10)
    assert r["se"] == pytest.approx(R_COV_SE, rel=1e-9)


def test_cov_stratified_matches_r(strat_sample):
    r = _row(strat_sample.estimation.cov(("y", "x")))
    assert r["se"] == pytest.approx(R_COV_STRAT_SE, rel=1e-9)


def test_cov_self_pair_is_the_variance(sample):
    """An explicit self-pair is the variance on svyvar's diagonal."""
    r = _row(sample.estimation.cov(("y", "y")))
    assert r["est"] == pytest.approx(R_VAR_Y, rel=1e-9)


def test_replication_matches_r(rep_sample):
    est = rep_sample.estimation
    c = _row(est.cov(("y", "x"), method="replication"))
    assert c["est"] == pytest.approx(R_COV, rel=1e-9)
    assert c["se"] == pytest.approx(R_COV_JK1_SE, rel=1e-7)

    r = _row(est.corr(("y", "x"), method="replication"))
    assert r["est"] == pytest.approx(R_CORR, rel=1e-10)


# ---------------------------------------------------------------------------
# Symmetry — the reason there is no y/x argument
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("verb", ["corr", "cov"])
def test_argument_order_does_not_matter(sample, verb):
    fn = getattr(sample.estimation, verb)
    a, b = _row(fn(("y", "x"))), _row(fn(("x", "y")))
    assert a["est"] == pytest.approx(b["est"], rel=1e-14)
    assert a["se"] == pytest.approx(b["se"], rel=1e-14)


def test_corr_is_scale_and_shift_invariant(sample):
    """z is an exact affine function of x, so both must follow from that."""
    d = {(r["y"], r["x"]): r["est"] for r in sample.estimation.corr(["y", "x", "z"]).to_dicts()}
    assert d[("x", "z")] == pytest.approx(1.0, abs=1e-12)
    assert d[("y", "z")] == pytest.approx(d[("y", "x")], rel=1e-12)


# ---------------------------------------------------------------------------
# cols spellings
# ---------------------------------------------------------------------------


def test_pair_returns_one_row(sample):
    assert sample.estimation.corr(("y", "x")).to_dicts().__len__() == 1


def test_two_element_list_is_the_same_as_a_tuple(sample):
    """The two spellings coincide at two columns, so they must agree."""
    a = _row(sample.estimation.corr(("y", "x")))
    b = _row(sample.estimation.corr(["y", "x"]))
    assert a["est"] == pytest.approx(b["est"], rel=1e-14)


def test_flat_list_expands_to_all_unique_pairs(sample):
    rows = sample.estimation.corr(["y", "x", "z"]).to_dicts()
    assert [(r["y"], r["x"]) for r in rows] == [("y", "x"), ("y", "z"), ("x", "z")]


def test_pair_list_selects_exactly_those_pairs(sample):
    """A focal variable against several others, without the others' pairs."""
    rows = sample.estimation.corr([("y", "x"), ("y", "z")]).to_dicts()
    assert [(r["y"], r["x"]) for r in rows] == [("y", "x"), ("y", "z")]


def test_by_group_yields_a_row_per_group_and_pair(sample):
    rows = sample.estimation.corr([("y", "x"), ("y", "z")], by="grp").to_dicts()
    assert len(rows) == 4


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


def test_fisher_interval_is_asymmetric_and_bounded(sample):
    r = _row(sample.estimation.corr(("y", "x")))
    assert -1.0 <= r["lci"] <= r["est"] <= r["uci"] <= 1.0
    lower_arm = r["est"] - r["lci"]
    upper_arm = r["uci"] - r["est"]
    assert abs(upper_arm - lower_arm) > 1e-6, "Fisher arms should not be equal"


def test_wald_interval_is_symmetric(sample):
    r = _row(sample.estimation.corr(("y", "x"), ci_method="wald"))
    assert (r["uci"] - r["est"]) == pytest.approx(r["est"] - r["lci"], rel=1e-12)


def test_fisher_and_wald_differ(sample):
    f = _row(sample.estimation.corr(("y", "x")))
    w = _row(sample.estimation.corr(("y", "x"), ci_method="wald"))
    assert f["lci"] != pytest.approx(w["lci"], rel=1e-9)


def test_cov_interval_is_wald(sample):
    """Covariance is unbounded, so it takes the symmetric interval."""
    r = _row(sample.estimation.cov(("y", "x")))
    assert (r["uci"] - r["est"]) == pytest.approx(r["est"] - r["lci"], rel=1e-12)


# ---------------------------------------------------------------------------
# Metadata and deff
# ---------------------------------------------------------------------------


def test_param_carries_the_full_word(sample):
    assert sample.estimation.corr(("y", "x")).param == PopParam.CORR
    assert str(sample.estimation.cov(("y", "x")).param) == "Covariance"


def test_deff_exceeds_one_under_clustering(sample):
    r = _row(sample.estimation.corr(("y", "x"), deff="wor"))
    assert r["deff"] > 1.0


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_single_column_is_rejected(sample):
    with pytest.raises(DimensionError, match="nothing to pair with"):
        sample.estimation.corr("y")


def test_duplicate_columns_rejected_in_flat_list(sample):
    """Three or more columns expand to all pairs, where a repeat is degenerate."""
    with pytest.raises(DimensionError, match="duplicate"):
        sample.estimation.corr(["y", "x", "y"])


def test_planned_kind_reports_not_implemented(sample):
    """'spearman' is recognised, so say it is unimplemented, not invalid."""
    with pytest.raises(MethodError, match="not supported yet"):
        sample.estimation.corr(("y", "x"), kind="spearman")


def test_unknown_kind_is_rejected(sample):
    with pytest.raises(MethodError, match="Unknown correlation kind"):
        sample.estimation.corr(("y", "x"), kind="nonsense")


def test_pandas_style_method_is_redirected_to_kind(sample):
    """pandas spells the coefficient method=; point the user at kind=."""
    with pytest.raises(MethodError, match="not a variance method"):
        sample.estimation.corr(("y", "x"), method="spearman")


def test_mixed_cols_are_rejected(sample):
    with pytest.raises(DimensionError, match="mixes column names and pairs"):
        sample.estimation.corr(["y", ("x", "z")])
