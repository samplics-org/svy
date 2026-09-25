# tests/svy/regression/test_glm_alpha.py
"""
A GLM fit keeps the alpha its coefficient intervals were computed at.

``GLMFit.alpha`` is the value passed to ``fit``; it travels into the
serialized ``GLMFitData`` (0.05 when an older payload lacks it) and sets the
interval headers of the printed table, which used to read ``[0.025 0.975]``
whatever the level.
"""

import json

import numpy as np
import polars as pl
import pytest

from scipy import stats

import svy

from svy.core.containers import FDist
from svy.errors import MethodError
from svy.regression.glm import GLMFit, GLMStats
from svy.serialize import from_json, to_json, to_polars
from svy.serialize.structs import GLMFitData


@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    rng = np.random.default_rng(20260925)
    n = 240
    x = rng.normal(0, 1, n)
    return pl.DataFrame(
        {
            "stratum": np.repeat([1, 2, 3, 4], 60),
            "psu": np.repeat(np.arange(24), 10),
            "wgt": rng.uniform(1, 5, n),
            "x": x,
            "g": rng.choice(["a", "b"], n),
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


def _headers(alpha: str) -> tuple[str, str]:
    return {
        "0.05": ("[0.025", "0.975]"),
        "0.1": ("[0.05", "0.95]"),
        "0.01": ("[0.005", "0.995]"),
        "0.5": ("[0.25", "0.75]"),
        "0.001": ("[0.0005", "0.9995]"),
    }[alpha]


ALPHAS = ["0.05", "0.1", "0.01", "0.5", "0.001"]


def _assert_width(fit: GLMFit, alpha: float) -> None:
    for c in fit.coefs:
        t = stats.t.ppf(1 - alpha / 2, c.wald.df)
        np.testing.assert_allclose(c.uci - c.est, t * c.se, rtol=1e-12)
        np.testing.assert_allclose(c.est - c.lci, t * c.se, rtol=1e-12)


# ---------------------------------------------------------------------------
# The stored value
# ---------------------------------------------------------------------------


class TestStored:
    def test_default(self, sample):
        fit = sample.glm.fit("y_num", x=["x"]).fitted
        assert fit.alpha == 0.05
        _assert_width(fit, 0.05)

    @pytest.mark.parametrize("alpha", ALPHAS)
    def test_non_default(self, sample, alpha):
        fit = sample.glm.fit("y_num", x=["x"], alpha=float(alpha)).fitted
        assert fit.alpha == float(alpha)
        _assert_width(fit, float(alpha))

    def test_width_orders_with_alpha(self, sample):
        widths = [
            sample.glm.fit("y_bin", x=["x"], family="binomial", alpha=a).fitted.coefs[1]
            for a in (0.5, 0.1, 0.05, 0.01, 0.001)
        ]
        spans = [c.uci - c.lci for c in widths]
        assert spans == sorted(spans)
        # The point estimate and SE do not depend on alpha.
        assert len({c.est for c in widths}) == 1
        assert len({c.se for c in widths}) == 1

    def test_where(self, sample):
        fit = sample.glm.fit("y_num", x=["x"], where=svy.col("g") == "a", alpha=0.1).fitted
        assert fit.alpha == 0.1
        _assert_width(fit, 0.1)

    @pytest.mark.parametrize("family", ["gaussian", "binomial", "poisson"])
    def test_replication(self, jk_sample, family):
        y = {"gaussian": "y_num", "binomial": "y_bin", "poisson": "y_cnt"}[family]
        fit = jk_sample.glm.fit(y, x=["x"], family=family, alpha=0.01).fitted
        assert fit.alpha == 0.01
        _assert_width(fit, 0.01)

    def test_replication_with_where(self, jk_sample):
        fit = jk_sample.glm.fit("y_num", x=["x"], where=svy.col("g") == "b", alpha=0.1).fitted
        assert fit.alpha == 0.1
        _assert_width(fit, 0.1)

    def test_direct_construction_defaults(self):
        fit = GLMFit(
            y="y",
            family="Gaussian",
            link="identity",
            stats=GLMStats(
                n=1,
                wald=FDist(1, 1, 0.0, 1.0),
                wald_adj=FDist(1, 1, 0.0, 1.0),
                scale=1.0,
                deviance=0.0,
            ),
        )
        assert fit.alpha == 0.05


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


class TestPrinting:
    @pytest.mark.parametrize("alpha", ALPHAS)
    def test_plain(self, sample, alpha):
        fit = sample.glm.fit("y_num", x=["x"], alpha=float(alpha))
        lo, hi = _headers(alpha)
        for text in (fit.fitted.__plain_str__(), fit.__plain_str__()):
            assert lo in text and hi in text
        if alpha != "0.05":
            assert "[0.025" not in fit.fitted.__plain_str__()

    @pytest.mark.parametrize("alpha", ALPHAS)
    def test_rich(self, sample, alpha):
        fit = sample.glm.fit("y_num", x=["x"], alpha=float(alpha))
        lo, hi = _headers(alpha)
        for text in (str(fit.fitted), str(fit)):
            assert lo in text and hi in text
        if alpha != "0.05":
            assert "[0.025" not in str(fit.fitted)

    @pytest.mark.parametrize("use_rich", [True, False])
    def test_show(self, sample, capsys, use_rich):
        sample.glm.fit("y_num", x=["x"], alpha=0.1).fitted.show(use_rich=use_rich)
        out = capsys.readouterr().out
        assert "[0.05" in out and "0.95]" in out

    @pytest.mark.parametrize(
        ("family", "link", "y", "header"),
        [
            ("binomial", "logit", "y_bin", "Odds ratio"),
            ("poisson", "log", "y_cnt", "Rate ratio"),
            ("binomial", "cloglog", "y_bin", "Hazard ratio"),
        ],
    )
    @pytest.mark.parametrize("use_rich", [True, False])
    def test_exponentiate(self, sample, capsys, family, link, y, header, use_rich):
        fit = sample.glm.fit(y, x=["x"], family=family, link=link, alpha=0.01).fitted
        fit.show(use_rich=use_rich, exponentiate=True)
        out = capsys.readouterr().out
        assert header in out
        assert "[0.005" in out and "0.995]" in out
        assert "[0.025" not in out

        plain = fit.__plain_str__(exponentiate=True)
        assert "[0.005" in plain and "0.995]" in plain

        # The printed bounds are exp() of the link-scale bounds at this alpha.
        frame = fit.to_polars(exponentiate=True)
        np.testing.assert_allclose(
            frame["conf_low"].to_numpy(), np.exp([c.lci for c in fit.coefs]), rtol=1e-12
        )

    def test_repr_unchanged(self, sample):
        fit = sample.glm.fit("y_num", x=["x"], alpha=0.1).fitted
        assert repr(fit).startswith("GLMFit(y='y_num'")


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    @pytest.mark.parametrize("alpha", ALPHAS)
    def test_round_trip(self, sample, alpha):
        fit = sample.glm.fit("y_num", x=["x"], alpha=float(alpha))
        payload = json.loads(to_json(fit))
        assert payload["alpha"] == float(alpha)
        back = from_json(to_json(fit))
        assert isinstance(back, GLMFitData)
        assert back.alpha == float(alpha)
        # The payload's table carries the same bounds as the live fit.
        live = fit.fitted.to_polars()
        np.testing.assert_allclose(
            to_polars(back)["conf_low"].to_numpy(), live["conf_low"].to_numpy(), rtol=1e-12
        )

    def test_replication_round_trip(self, jk_sample):
        fit = jk_sample.glm.fit("y_bin", x=["x"], family="binomial", alpha=0.1)
        assert from_json(to_json(fit)).alpha == 0.1

    def test_payload_without_alpha_decodes_to_default(self, sample):
        payload = json.loads(to_json(sample.glm.fit("y_num", x=["x"], alpha=0.1)))
        del payload["alpha"]
        back = from_json(json.dumps(payload).encode())
        assert back.alpha == 0.05


# ---------------------------------------------------------------------------
# Predictions and margins print their level without truncation
# ---------------------------------------------------------------------------


class TestPredAndMargins:
    @pytest.mark.parametrize(
        ("alpha", "label"),
        [
            (0.05, "95% CI"),
            (0.1, "90% CI"),
            (0.01, "99% CI"),
            (0.001, "99.9% CI"),
            (0.42, "58% CI"),
        ],
    )
    def test_pred_level(self, sample, data, alpha, label):
        pred = sample.glm.fit("y_num", x=["x"]).predict(data.head(5), alpha=alpha)
        assert pred.alpha == alpha
        for text in (repr(pred), str(pred), pred.__plain_str__()):
            assert label in text

    @pytest.mark.parametrize(("alpha", "label"), [(0.05, "95% CI"), (0.001, "99.9% CI")])
    def test_margins_level(self, sample, alpha, label):
        m = sample.glm.fit("y_num", x=["x"]).margins(variables=["x"], alpha=alpha)
        members = m if isinstance(m, list) else [m]
        for mm in members:
            for text in (repr(mm), str(mm), mm.__plain_str__()):
                assert label in text


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize("alpha", [0, 0.0, 1, 1.0, -0.05, 1.5, float("nan"), float("inf")])
    def test_out_of_range(self, sample, alpha):
        with pytest.raises(MethodError) as exc:
            sample.glm.fit("y_num", x=["x"], alpha=alpha)
        assert exc.value.code == "INVALID_RANGE"
        assert exc.value.param == "alpha"
        assert exc.value.where == "GLM.fit"
        assert "0.05 gives 95%" in str(exc.value)

    @pytest.mark.parametrize("alpha", ["0.05", True, False, None, [0.05]])
    def test_wrong_type(self, sample, alpha):
        with pytest.raises(MethodError) as exc:
            sample.glm.fit("y_num", x=["x"], alpha=alpha)
        assert exc.value.code == "INVALID_TYPE"
        assert exc.value.param == "alpha"

    def test_replication_fit_validates_too(self, jk_sample):
        with pytest.raises(MethodError, match="INVALID_RANGE"):
            jk_sample.glm.fit("y_num", x=["x"], alpha=1.0)

    def test_refused_before_fitting(self, data):
        glm = svy.Sample(data, svy.Design(stratum="stratum", psu="psu", wgt="wgt")).glm
        with pytest.raises(MethodError):
            glm.fit("y_num", x=["x"], alpha=0)
        assert glm.fitted is None

    @pytest.mark.parametrize("alpha", [np.float64(0.1), np.float32(0.25), 1e-12, 1 - 1e-12])
    def test_accepted_edges(self, sample, alpha):
        fit = sample.glm.fit("y_num", x=["x"], alpha=alpha).fitted
        assert type(fit.alpha) is float
        assert fit.alpha == float(alpha)
