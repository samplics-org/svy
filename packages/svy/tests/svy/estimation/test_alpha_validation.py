# tests/svy/estimation/test_alpha_validation.py
"""Every estimator validates ``alpha`` the same way: a real number strictly in
(0, 1), refused with INVALID_TYPE or INVALID_RANGE before any work is done."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

import svy

from svy.errors import MethodError


@pytest.fixture(scope="module")
def s() -> svy.Sample:
    rng = np.random.default_rng(7)
    n = 40
    df = pl.DataFrame(
        {
            "stratum": ["s1"] * 20 + ["s2"] * 20,
            "psu": [i // 5 for i in range(n)],
            "wgt": rng.uniform(1.0, 3.0, n),
            "y": rng.normal(10.0, 2.0, n),
            "x": rng.normal(5.0, 1.0, n),
            "b": [i % 2 for i in range(n)],
            "grp": ["g1", "g2"] * 20,
        }
    )
    return svy.Sample(df, svy.Design(stratum="stratum", psu="psu", wgt="wgt"))


@pytest.fixture(scope="module")
def fitted(s):
    return s.glm.fit("y", x=["x"])


CALLS = {
    "mean": lambda s, a: s.estimation.mean("y", alpha=a),
    "total": lambda s, a: s.estimation.total("y", alpha=a),
    "prop": lambda s, a: s.estimation.prop("grp", alpha=a),
    "ratio": lambda s, a: s.estimation.ratio("y", "x", alpha=a),
    "median": lambda s, a: s.estimation.median("y", alpha=a),
    "quantile": lambda s, a: s.estimation.quantile("y", p=0.25, alpha=a),
    "corr": lambda s, a: s.estimation.corr(("y", "x"), alpha=a),
    "cov": lambda s, a: s.estimation.cov(("y", "x"), alpha=a),
    "tabulate": lambda s, a: s.categorical.tabulate("grp", alpha=a),
    "ttest": lambda s, a: s.categorical.ttest("y", mean_h0=10, alpha=a),
    "ranktest": lambda s, a: s.categorical.ranktest(
        "y", group="grp", method="kruskal-wallis", alpha=a
    ),
    "Estimate.contrast": lambda s, a: s.estimation.mean("y", by="grp").contrast(
        _diff(s.estimation.mean("y", by="grp")), alpha=a
    ),
    "EstimateList.contrast": lambda s, a: s.estimation.quantile("y", p=[0.25]).contrast(
        {}, alpha=a
    ),
    "glm.fit": lambda s, a: s.glm.fit("y", x=["x"], alpha=a),
}
GLM_CALLS = {
    "glm.predict": lambda g, a: g.predict(pl.DataFrame({"x": [4.0, 6.0]}), alpha=a),
    "glm.margins": lambda g, a: g.margins(alpha=a),
    "glm.contrast": lambda g, a: g.contrast({"x": 1.0}, alpha=a),
}


def _diff(est) -> dict:
    k1, k2 = est.keys()[:2]
    return {k1: 1.0, k2: -1.0}


BAD_RANGE = [0, 1, 0.0, 1.0, -0.05, 1.5, 100, float("nan"), float("inf"), -float("inf")]
BAD_TYPE = ["0.05", True, False, [0.05], (0.05,), object()]


def _check(call, alpha, code):
    with pytest.raises(MethodError) as ei:
        call(alpha)
    err = ei.value
    assert err.code == code and err.param == "alpha"
    assert "0.05 gives 95%" in err.hint
    json.dumps(err.to_dict(), default=str)


@pytest.mark.parametrize("name", list(CALLS))
@pytest.mark.parametrize("alpha", BAD_RANGE, ids=repr)
def test_out_of_range(s, name, alpha):
    _check(lambda a: CALLS[name](s, a), alpha, "INVALID_RANGE")


@pytest.mark.parametrize("name", list(CALLS))
@pytest.mark.parametrize("alpha", BAD_TYPE, ids=lambda a: type(a).__name__)
def test_wrong_type(s, name, alpha):
    _check(lambda a: CALLS[name](s, a), alpha, "INVALID_TYPE")


@pytest.mark.parametrize("name", [n for n in CALLS if "contrast" not in n])
def test_none_is_refused(s, name):
    _check(lambda a: CALLS[name](s, a), None, "INVALID_TYPE")


@pytest.mark.parametrize("name", list(GLM_CALLS))
@pytest.mark.parametrize("alpha", BAD_RANGE, ids=repr)
def test_glm_out_of_range(fitted, name, alpha):
    _check(lambda a: GLM_CALLS[name](fitted, a), alpha, "INVALID_RANGE")


@pytest.mark.parametrize("name", list(GLM_CALLS))
@pytest.mark.parametrize("alpha", BAD_TYPE, ids=lambda a: type(a).__name__)
def test_glm_wrong_type(fitted, name, alpha):
    _check(lambda a: GLM_CALLS[name](fitted, a), alpha, "INVALID_TYPE")


GOOD = [0.1, np.float64(0.1), np.float32(0.1), 1e-6, 0.999999]


@pytest.mark.parametrize("name", [n for n in CALLS if n != "EstimateList.contrast"])
@pytest.mark.parametrize("alpha", GOOD, ids=repr)
def test_valid_alpha_accepted(s, name, alpha):
    CALLS[name](s, alpha)


@pytest.mark.parametrize("name", list(GLM_CALLS))
@pytest.mark.parametrize("alpha", GOOD, ids=repr)
def test_glm_valid_alpha_accepted(fitted, name, alpha):
    GLM_CALLS[name](fitted, alpha)


def test_numpy_alpha_stored_as_float(s):
    est = s.estimation.mean("y", alpha=np.float64(0.1))
    assert type(est.alpha) is float and est.alpha == pytest.approx(0.1)
    ref = s.estimation.mean("y", alpha=0.1)
    assert est.to_polars().equals(ref.to_polars())


def test_contrast_none_inherits_alpha(s):
    est = s.estimation.mean("y", by="grp", alpha=0.1)
    assert est.contrast(_diff(est)).alpha == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Sample size: alpha scalar or per stratum
# ---------------------------------------------------------------------------

SIZE_CALLS = {
    "estimate_prop": lambda a: svy.SampleSize().estimate_prop(p=0.3, moe=0.05, alpha=a),
    "estimate_mean": lambda a: svy.SampleSize().estimate_mean(sigma=7000, moe=1000, alpha=a),
    "compare_props": lambda a: svy.SampleSize().compare_props(p1=0.3, p2=0.4, alpha=a),
    "compare_means": lambda a: svy.SampleSize().compare_means(
        mu1=10.0, mu2=12.0, sigma1=5.0, alpha=a
    ),
}


@pytest.mark.parametrize("name", list(SIZE_CALLS))
@pytest.mark.parametrize("alpha", BAD_RANGE, ids=repr)
def test_size_out_of_range(name, alpha):
    _check(SIZE_CALLS[name], alpha, "INVALID_RANGE")


@pytest.mark.parametrize("name", list(SIZE_CALLS))
@pytest.mark.parametrize("alpha", [*BAD_TYPE, None], ids=lambda a: type(a).__name__)
def test_size_wrong_type(name, alpha):
    _check(SIZE_CALLS[name], alpha, "INVALID_TYPE")


@pytest.mark.parametrize("name", list(SIZE_CALLS))
@pytest.mark.parametrize("bad", [0.0, 1.0, float("nan"), "0.05", True])
def test_size_per_stratum_alpha_checked(name, bad):
    with pytest.raises(MethodError) as ei:
        SIZE_CALLS[name]({"r1": 0.05, "r2": bad})
    assert ei.value.param == "alpha"
    assert ei.value.code in ("INVALID_RANGE", "INVALID_TYPE")


@pytest.mark.parametrize("name", list(SIZE_CALLS))
def test_size_valid_alpha_forms_agree(name):
    ref = SIZE_CALLS[name](0.1).size.n0
    assert SIZE_CALLS[name](np.float64(0.1)).size.n0 == ref
    per = SIZE_CALLS[name]({"r1": 0.1, "r2": np.float64(0.1)}).size
    assert sorted(z.stratum for z in per) == ["r1", "r2"]
    assert all(z.n0 == ref for z in per)
