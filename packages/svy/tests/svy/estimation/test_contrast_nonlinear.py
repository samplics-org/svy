# tests/svy/estimation/test_contrast_nonlinear.py
"""Delta-method contrasts against R survey 4.5:

```r
options(digits = 15)
library(survey)
api <- read.csv("tests/test_data/apiclus1.csv")
d <- svydesign(id = ~dnum, weights = ~pw, data = api)
m <- svyby(~api00, ~stype, d, svymean, covmat = TRUE)
svycontrast(m, quote(H / E))                 # 0.953308493576269  SE 0.04682352757304
svycontrast(m, quote((H - E) / E))           # -0.0466915064237311 SE 0.04682352757304
svycontrast(m, quote(log(H) - log(E)))       # -0.0478167198650326 SE 0.0491168681371807
svycontrast(m, quote(H * E))                 # 401371.240079365   SE 35264.9638828295
svycontrast(m, quote(exp(log(H) - log(E))))  # 0.953308493576269  SE 0.04682352757304
degf(d)                                      # 14
```
"""

import numpy as np
import pytest

import svy

from svy import Design, Sample, estd
from svy.errors import MethodError


TOL = 1e-11


@pytest.fixture(scope="module")
def by_stype():
    api = svy.io.read_csv("tests/test_data/apiclus1.csv")
    return Sample(api, Design(psu="dnum", wgt="pw")).estimation.mean("api00", by="stype")


R = {
    "ratio": (0.953308493576269, 0.04682352757304),
    "pct": (-0.0466915064237311, 0.04682352757304),
    "logratio": (-0.0478167198650326, 0.0491168681371807),
    "prod": (401371.240079365, 35264.9638828295),
    "exp": (0.953308493576269, 0.04682352757304),
}


def _row(c, name):
    return next(e for e in c.estimates if e.contrast == name)


def test_ratio_percent_log_product_match_r(by_stype):
    c = by_stype.contrast(
        {
            "ratio": estd("H") / estd("E"),
            "pct": (estd("H") - estd("E")) / estd("E"),
            "logratio": estd("H").log() - estd("E").log(),
            "prod": estd("H") * estd("E"),
            "exp": (estd("H").log() - estd("E").log()).exp(),
        }
    )
    for name, (est, se) in R.items():
        row = _row(c, name)
        assert row.est == pytest.approx(est, rel=TOL), name
        assert row.se == pytest.approx(se, rel=TOL), name
        assert row.df == 14


def test_constants_and_reversed_operands(by_stype):
    c = by_stype.contrast(
        {
            "pct": estd("H") / estd("E") - 1,
            "pct_r": -1 + estd("H") / estd("E"),
            "inv": 1 / (estd("E") / estd("H")),
            "twice": 2 * (estd("H") / estd("E")),
        }
    )
    assert _row(c, "pct").est == pytest.approx(R["pct"][0], rel=TOL)
    assert _row(c, "pct").se == pytest.approx(R["pct"][1], rel=TOL)
    assert _row(c, "pct_r").est == pytest.approx(R["pct"][0], rel=TOL)
    assert _row(c, "inv").est == pytest.approx(R["ratio"][0], rel=TOL)
    assert _row(c, "inv").se == pytest.approx(R["ratio"][1], rel=TOL)
    assert _row(c, "twice").se == pytest.approx(2 * R["ratio"][1], rel=TOL)


def test_linear_path_unchanged_and_mixed_covariance(by_stype):
    lin = by_stype.contrast(estd("E") - estd("H")).estimates[0]
    mixed = by_stype.contrast({"diff": estd("E") - estd("H"), "ratio": estd("H") / estd("E")})
    assert _row(mixed, "diff").est == lin.est and _row(mixed, "diff").se == lin.se
    assert mixed.covariance.shape == (2, 2)
    # G V Gᵀ: the off-diagonal is the gradient cross term, not zero
    assert mixed.covariance[0, 1] != 0
    assert np.allclose(np.diag(mixed.covariance), [lin.se**2, R["ratio"][1] ** 2])


def test_is_linear_and_coefs():
    assert (estd("a") - 2 * estd("b")).is_linear()
    assert not (estd("a") / estd("b")).is_linear()
    assert (estd("a") + 1).is_linear() is False
    with pytest.raises(MethodError, match="[Nn]onlinear"):
        (estd("a") / estd("b")).coefs()
    assert (estd("a") / estd("b")).keys() == ["a", "b"]


def test_t_ci_and_p_on_design_df(by_stype):
    from scipy import stats

    row = by_stype.contrast(estd("H") / estd("E") - 1).estimates[0]
    t_crit = stats.t.ppf(0.975, 14)
    assert row.lci == pytest.approx(row.est - t_crit * row.se)
    assert row.uci == pytest.approx(row.est + t_crit * row.se)
    assert row.p_value == pytest.approx(2 * stats.t.sf(abs(row.est / row.se), 14))


def test_division_by_zero_estimate_named(by_stype):
    with pytest.raises(MethodError, match="denominator .*estd\\('E'\\).* evaluates to 0"):
        by_stype.contrast(estd("H") / (estd("E") - estd("E")))
    with pytest.raises(MethodError, match="[Ll]og"):
        by_stype.contrast((estd("E") - estd("H") - estd("M")).log())


def test_unknown_key_in_nonlinear_expression(by_stype):
    with pytest.raises(MethodError, match="Unknown contrast key"):
        by_stype.contrast(estd("H") / estd("Z"))


def test_dict_form_stays_linear(by_stype):
    c = by_stype.contrast({"H": 1.0, "E": -1.0})
    lin = by_stype.contrast(estd("H") - estd("E")).estimates[0]
    assert c.estimates[0].est == lin.est and c.estimates[0].se == lin.se


def test_type_errors():
    with pytest.raises(TypeError):
        estd("a") * "b"  # type: ignore[operator]
    with pytest.raises(TypeError):
        "b" / estd("a")  # type: ignore[operator]


def test_glm_coefficient_ratio():
    api = svy.io.read_csv("tests/test_data/apiclus1.csv")
    fit = Sample(api, Design(psu="dnum", wgt="pw")).glm.fit("api00", x=["api99", "ell"])
    c = fit.contrast(estd("api99") / estd("ell"))
    coefs = {k: v for k, v in zip(fit.keys(), [x.est for x in fit.coefs])}
    assert c.estimates[0].est == pytest.approx(coefs["api99"] / coefs["ell"])
    assert np.isfinite(c.estimates[0].se) and c.estimates[0].se > 0


def test_panel_change_ratio_on_long_sample():
    rng = np.random.default_rng(3)
    n = 40
    y1 = rng.normal(10, 2, n)
    y2 = y1 * 1.1 + rng.normal(0, 0.3, n)
    import polars as pl

    long = pl.DataFrame(
        {
            "id": np.tile(np.arange(n), 2),
            "wave": np.repeat([1, 2], n),
            "y": np.concatenate([y1, y2]),
            "w": np.ones(2 * n),
        }
    )
    s = Sample(long, Design(case_id="id", wave="wave", wgt="w"))
    r = s.estimation.mean("y", by="wave")
    c = r.contrast({"ratio": estd(2) / estd(1), "pct": (estd(2) - estd(1)) / estd(1)})
    assert _row(c, "ratio").est == pytest.approx(y2.mean() / y1.mean())
    assert _row(c, "pct").est == pytest.approx(y2.mean() / y1.mean() - 1)
    assert _row(c, "pct").se == pytest.approx(_row(c, "ratio").se)
    # the case-level pairing makes the ratio far more precise than independence would
    rows = r.to_polars()
    naive = np.sqrt(
        (rows["se"][1] / rows["est"][0]) ** 2
        + (rows["est"][1] * rows["se"][0] / rows["est"][0] ** 2) ** 2
    )
    assert _row(c, "ratio").se < naive / 3
