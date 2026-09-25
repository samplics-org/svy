# tests/svy/regression/test_glm_fit_record.py
"""
What a fitted GLM records about itself: ``to_dict()`` gives plain Python
values (it used to raise on numpy scalars), and a ``where=`` fit keeps its
domain in ``where_clause``, printed and serialized like an ``Estimate``'s.
"""

import json

import numpy as np
import polars as pl
import pytest

import svy

from svy.serialize import from_json, to_json
from svy.ui.printing import format_where_clause


@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    rng = np.random.default_rng(97)
    n = 240
    x = rng.normal(0, 1, n)
    return pl.DataFrame(
        {
            "stratum": np.repeat([1, 2, 3, 4], 60),
            "psu": np.repeat(np.arange(24), 10),
            "wgt": rng.uniform(1, 5, n),
            "x": x,
            "g": rng.choice(["a", "b"], n),
            "age": rng.integers(18, 80, n),
            "y_num": 1.0 + 2.0 * x + rng.normal(0, 1, n),
            "y_bin": rng.binomial(1, 1 / (1 + np.exp(-(0.3 + 0.8 * x)))),
            "y_cnt": rng.poisson(np.exp(0.5 + 0.3 * x)),
        }
    )


@pytest.fixture(scope="module")
def sample(data) -> svy.Sample:
    return svy.Sample(data, svy.Design(stratum="stratum", psu="psu", wgt="wgt"))


@pytest.fixture(scope="module")
def jk_sample(sample) -> svy.Sample:
    return sample.weighting.create_jk_wgts()


FAMILIES = {"gaussian": "y_num", "binomial": "y_bin", "poisson": "y_cnt"}


def _is_plain(obj) -> bool:
    if isinstance(obj, dict):
        return all(isinstance(k, str) and _is_plain(v) for k, v in obj.items())
    if isinstance(obj, list):
        return all(_is_plain(v) for v in obj)
    return obj is None or type(obj) in (str, int, float, bool)


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


class TestToDict:
    @pytest.mark.parametrize("family", sorted(FAMILIES))
    @pytest.mark.parametrize("design", ["taylor", "jk"])
    @pytest.mark.parametrize("domain", [False, True])
    def test_plain_values(self, sample, jk_sample, family, design, domain):
        s = sample if design == "taylor" else jk_sample
        kw = {"where": svy.col("g") == "a"} if domain else {}
        fit = s.glm.fit(FAMILIES[family], x=["x"], family=family, **kw).fitted
        d = fit.to_dict()
        assert _is_plain(d)
        json.dumps(d)  # no custom encoder needed
        assert "cov_matrix" not in d and "term_info" not in d
        assert [c["est"] for c in d["coefs"]] == [float(c.est) for c in fit.coefs]
        assert d["stats"]["n"] == int(fit.stats.n)
        assert d["alpha"] == fit.alpha
        assert d["where_clause"] == fit.where_clause

    def test_categorical_and_negative_binomial(self, sample):
        fit = sample.glm.fit("y_cnt", x=[svy.Cat("g"), "x"], family="negative_binomial").fitted
        d = fit.to_dict()
        assert _is_plain(d)
        assert d["stats"]["theta"] == float(fit.stats.theta)

    def test_coef_and_stats_to_dict(self, sample):
        fit = sample.glm.fit("y_bin", x=["x"], family="binomial").fitted
        assert all(_is_plain(c.to_dict()) for c in fit.coefs)
        assert _is_plain(fit.stats.to_dict())

    def test_does_not_touch_the_fit(self, sample):
        fit = sample.glm.fit("y_num", x=["x"]).fitted
        fit.to_dict()
        assert fit.cov_matrix is not None and fit.term_info is not None


# ---------------------------------------------------------------------------
# where_clause
# ---------------------------------------------------------------------------

WHERES = {
    "expr": svy.col("g") == "a",
    "compound": (svy.col("g") == "b") & (svy.col("age") > 30),
    "mapping": {"g": "a"},
}


class TestWhereClause:
    def test_none_without_where(self, sample):
        fit = sample.glm.fit("y_num", x=["x"]).fitted
        assert fit.where_clause is None
        assert "where" not in fit.__plain_str__()
        assert "where:" not in str(fit)

    @pytest.mark.parametrize("kind", sorted(WHERES))
    @pytest.mark.parametrize("design", ["taylor", "jk"])
    def test_recorded_like_estimate(self, sample, jk_sample, kind, design):
        s = sample if design == "taylor" else jk_sample
        where = WHERES[kind]
        fit = s.glm.fit("y_num", x=["x"], where=where).fitted
        expected = format_where_clause(where)
        assert expected
        assert fit.where_clause == expected
        method = None if design == "taylor" else "replication"
        assert s.estimation.mean("y_num", where=where, method=method).where_clause == expected

    @pytest.mark.parametrize("kind", sorted(WHERES))
    def test_printed(self, sample, kind, capsys):
        fit = sample.glm.fit("y_bin", x=["x"], family="binomial", where=WHERES[kind]).fitted
        clause = fit.where_clause
        plain = fit.__plain_str__()
        assert f"where    : {clause}" in plain.splitlines()[2]
        assert "where: " in str(fit) and clause in str(fit)
        assert clause in fit.__plain_str__(exponentiate=True)
        fit.show(use_rich=True, exponentiate=True)
        out = capsys.readouterr().out
        assert "where: " in out and "Odds ratio" in out

    def test_glm_wrapper_prints_it(self, sample):
        glm = sample.glm.fit("y_num", x=["x"], where=svy.col("g") == "a")
        assert glm.fitted.where_clause in glm.__plain_str__()
        assert glm.fitted.where_clause in str(glm)

    @pytest.mark.parametrize("design", ["taylor", "jk"])
    def test_json_round_trip(self, sample, jk_sample, design):
        s = sample if design == "taylor" else jk_sample
        fit = s.glm.fit("y_num", x=["x"], where=svy.col("g") == "b", alpha=0.1)
        payload = json.loads(to_json(fit))
        assert payload["where_clause"] == fit.fitted.where_clause
        back = from_json(to_json(fit))
        assert back.where_clause == fit.fitted.where_clause
        assert back.alpha == 0.1

    def test_json_without_where(self, sample):
        fit = sample.glm.fit("y_num", x=["x"])
        assert json.loads(to_json(fit))["where_clause"] is None
        assert from_json(to_json(fit)).where_clause is None

    def test_old_payload_without_field_decodes(self, sample):
        payload = json.loads(to_json(sample.glm.fit("y_num", x=["x"], where={"g": "a"})))
        del payload["where_clause"]
        assert from_json(json.dumps(payload).encode()).where_clause is None
