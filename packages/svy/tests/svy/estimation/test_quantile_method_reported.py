# tests/svy/estimation/test_quantile_method_reported.py
"""
median() and quantile() report the quantile rule they were computed with.

``Estimate.q_method`` used to stay at its constructor default (``LINEAR``)
whatever ``q_method=`` was passed, so the attribute, the printed header and
the serialized ``q_method`` field all misreported it, while the values
themselves used the requested rule.
"""

import json

import numpy as np
import polars as pl
import pytest

import svy

from svy import EstimateList, QuantileMethod
from svy.serialize import from_json, to_json


Q_METHODS = {
    "higher": QuantileMethod.HIGHER,
    "lower": QuantileMethod.LOWER,
    "nearest": QuantileMethod.NEAREST,
    "linear": QuantileMethod.LINEAR,
    "middle": QuantileMethod.MIDDLE,
}


@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    rng = np.random.default_rng(20260925)
    n = 150
    g = rng.choice(["a", "b"], n).astype(object)
    g[0] = "solo"  # a single-row domain
    return pl.DataFrame(
        {
            "stratum": np.repeat([1, 2, 3], 50),
            "psu": np.repeat(np.arange(15), 10),
            "wgt": rng.uniform(1, 5, n),
            # Heavily tied integers: the rules give different answers.
            "y": rng.integers(0, 20, n).astype(float),
            "z": rng.normal(10, 3, n),
            "const": np.full(n, 3.0),
            "g": g.tolist(),
        }
    )


@pytest.fixture(scope="module")
def sample(data) -> svy.Sample:
    return svy.Sample(data, svy.Design(stratum="stratum", psu="psu", wgt="wgt"))


@pytest.fixture(scope="module")
def jk_sample(sample) -> svy.Sample:
    return sample.weighting.create_jk_wgts()


@pytest.fixture(scope="module")
def bs_sample(sample) -> svy.Sample:
    return sample.weighting.create_bs_wgts(n_reps=30, rstate=11)


def _members(res) -> list:
    return list(res) if isinstance(res, list) else [res]


def _assert_reported(res, expected: QuantileMethod) -> None:
    members = _members(res)
    assert members
    for m in members:
        assert m.q_method is expected
        assert json.loads(to_json(m))["q_method"] == expected.value
        assert f"q_method={expected.value.lower()}" in m.__plain_str__().splitlines()[0]


# ---------------------------------------------------------------------------
# Every rule x estimator x variance method x shape
# ---------------------------------------------------------------------------

SHAPES = {
    "plain": {},
    "by": {"by": "g"},
    "where": {"where": svy.col("z") > 8},
    "by_where": {"by": "g", "where": svy.col("z") > 8},
}


@pytest.mark.parametrize("q_method", sorted(Q_METHODS))
@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("variance", ["taylor", "jk", "bs"])
class TestEveryCombination:
    def _sample(self, variance, sample, jk_sample, bs_sample):
        return {"taylor": sample, "jk": jk_sample, "bs": bs_sample}[variance]

    def _method(self, variance):
        return None if variance == "taylor" else "replication"

    def test_median(self, q_method, shape, variance, sample, jk_sample, bs_sample):
        s = self._sample(variance, sample, jk_sample, bs_sample)
        res = s.estimation.median(
            "y", q_method=q_method, method=self._method(variance), **SHAPES[shape]
        )
        _assert_reported(res, Q_METHODS[q_method])

    def test_median_several_y(self, q_method, shape, variance, sample, jk_sample, bs_sample):
        s = self._sample(variance, sample, jk_sample, bs_sample)
        res = s.estimation.median(
            ["y", "z"], q_method=q_method, method=self._method(variance), **SHAPES[shape]
        )
        assert len(res) == 2
        _assert_reported(res, Q_METHODS[q_method])

    def test_quantile_scalar_p(self, q_method, shape, variance, sample, jk_sample, bs_sample):
        s = self._sample(variance, sample, jk_sample, bs_sample)
        res = s.estimation.quantile(
            "y", p=0.3, q_method=q_method, method=self._method(variance), **SHAPES[shape]
        )
        assert not isinstance(res, list)
        _assert_reported(res, Q_METHODS[q_method])

    def test_quantile_several_p_and_y(
        self, q_method, shape, variance, sample, jk_sample, bs_sample
    ):
        s = self._sample(variance, sample, jk_sample, bs_sample)
        res = s.estimation.quantile(
            ["y", "z"],
            p=[0.1, 0.5, 0.9],
            q_method=q_method,
            method=self._method(variance),
            **SHAPES[shape],
        )
        assert isinstance(res, EstimateList)
        assert len(res) == 6
        _assert_reported(res, Q_METHODS[q_method])
        head = res.__plain_str__().splitlines()[0]
        assert f"q_method={q_method}" in head


# ---------------------------------------------------------------------------
# Defaults, input spellings, and that the report matches the computation
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_median_default_is_higher(self, sample, jk_sample):
        _assert_reported(sample.estimation.median("y"), QuantileMethod.HIGHER)
        _assert_reported(
            jk_sample.estimation.median("y", method="replication"), QuantileMethod.HIGHER
        )

    def test_quantile_default_is_higher(self, sample, jk_sample):
        _assert_reported(sample.estimation.quantile("y"), QuantileMethod.HIGHER)
        _assert_reported(
            jk_sample.estimation.quantile("y", method="replication"), QuantileMethod.HIGHER
        )

    @pytest.mark.parametrize("spelling", ["LOWER", " Lower ", "lower"])
    def test_spelling_is_normalized(self, sample, spelling):
        _assert_reported(sample.estimation.median("y", q_method=spelling), QuantileMethod.LOWER)

    def test_unknown_rule_still_raises(self, sample):
        with pytest.raises(ValueError, match="Unknown quantile method"):
            sample.estimation.quantile("y", p=0.3, q_method="hf7")

    def test_reported_rule_is_the_one_used(self, sample):
        """Each reported rule reproduces its own value, and the rules differ here."""
        values = {}
        for name in Q_METHODS:
            r = sample.estimation.quantile("z", p=0.3, q_method=name)
            values[r.q_method] = r.estimates[0].est
        assert len(set(values.values())) > 1
        for qm, est in values.items():
            again = sample.estimation.quantile("z", p=0.3, q_method=qm.value.lower())
            assert again.estimates[0].est == est

    def test_non_quantile_header_has_no_rule(self, sample):
        assert "q_method" not in sample.estimation.mean("y").__plain_str__()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.parametrize("q_method", sorted(Q_METHODS))
    def test_all_equal_values(self, sample, q_method):
        res = sample.estimation.quantile("const", p=[0.25, 0.75], q_method=q_method)
        _assert_reported(res, Q_METHODS[q_method])
        assert all(m.estimates[0].est == 3.0 for m in res)

    @pytest.mark.parametrize("q_method", sorted(Q_METHODS))
    def test_extreme_probabilities(self, sample, jk_sample, q_method):
        for s, method in ((sample, None), (jk_sample, "replication")):
            res = s.estimation.quantile("y", p=[0.001, 0.999], q_method=q_method, method=method)
            _assert_reported(res, Q_METHODS[q_method])

    @pytest.mark.parametrize("p", [0.0, 1.0])
    def test_boundary_probabilities_still_refused(self, sample, p):
        with pytest.raises(ValueError, match="strictly"):
            sample.estimation.quantile("y", p=p, q_method="lower")

    def test_single_row_domain(self, sample):
        res = sample.estimation.median("y", by="g", q_method="nearest")
        assert "solo" in [e.by_level[0] for e in res.estimates]
        _assert_reported(res, QuantileMethod.NEAREST)

    @pytest.mark.parametrize("q_method", sorted(Q_METHODS))
    def test_where_selects_nothing(self, sample, q_method):
        where = svy.col("z") > 1e9
        _assert_reported(
            sample.estimation.median("y", where=where, q_method=q_method), Q_METHODS[q_method]
        )
        _assert_reported(
            sample.estimation.quantile("y", p=[0.2, 0.8], where=where, q_method=q_method),
            Q_METHODS[q_method],
        )

    @pytest.mark.parametrize("param", ["median", "quantile"])
    def test_zero_weight_fallback_reports_rule(self, sample, jk_sample, monkeypatch, param):
        """The empty result built when the kernel finds no weight keeps the rule."""
        import svy.estimation.base as base

        def boom(*args, **kwargs):
            raise RuntimeError("Sum of weights is zero")

        for name in (
            "_taylor_median",
            "_taylor_quantile",
            "_replicate_median",
            "_replicate_quantile",
        ):
            monkeypatch.setattr(base, name, boom)

        for s, method in ((sample, None), (jk_sample, "replication")):
            if param == "median":
                res = s.estimation.median("y", q_method="middle", method=method)
            else:
                res = s.estimation.quantile("y", p=[0.2, 0.8], q_method="middle", method=method)
            for m in _members(res):
                assert m.estimates == []
                assert m.q_method is QuantileMethod.MIDDLE
                assert json.loads(to_json(m))["q_method"] == "Middle"

    @pytest.mark.parametrize("q_method", sorted(Q_METHODS))
    def test_nulls_dropped(self, data, q_method):
        with_nulls = data.with_columns(
            y=pl.when(pl.int_range(pl.len()) < 7).then(None).otherwise(pl.col("y"))
        )
        s = svy.Sample(with_nulls, svy.Design(stratum="stratum", psu="psu", wgt="wgt"))
        _assert_reported(
            s.estimation.median("y", q_method=q_method, drop_nulls=True), Q_METHODS[q_method]
        )
        _assert_reported(
            s.estimation.quantile("y", p=[0.3, 0.6], q_method=q_method, drop_nulls=True),
            Q_METHODS[q_method],
        )


# ---------------------------------------------------------------------------
# Printing and serialization
# ---------------------------------------------------------------------------


class TestPrintingAndSerialization:
    def test_rich_header_shows_rule(self, sample, jk_sample):
        assert "q_method=nearest" in str(sample.estimation.median("y", q_method="nearest"))
        rep = jk_sample.estimation.quantile("y", p=0.4, q_method="middle", method="replication")
        text = str(rep)
        assert "JACKKNIFE" in text
        assert "q_method=middle" in text

    def test_rich_header_of_a_list(self, sample):
        res = sample.estimation.quantile("y", p=[0.2, 0.8], q_method="lower")
        assert "q_method=lower" in str(res)

    def test_header_keeps_deff_reference(self, sample):
        res = sample.estimation.mean("y", deff="wr")
        head = res.__plain_str__().splitlines()[0]
        assert "deff=" in head and "q_method" not in head

    def test_list_mixing_params(self, sample):
        mixed = EstimateList(
            [sample.estimation.mean("y"), sample.estimation.median("y", q_method="lower")]
        )
        head = mixed.__plain_str__().splitlines()[0]
        assert "MIXED" in head and "q_method=lower" in head

    def test_list_mixing_rules(self, sample):
        mixed = EstimateList(
            [
                sample.estimation.median("y", q_method="higher"),
                sample.estimation.median("y", q_method="lower"),
            ]
        )
        assert "q_method=mixed" in mixed.__plain_str__().splitlines()[0]

    def test_list_without_quantiles_has_no_rule(self, sample):
        res = EstimateList([sample.estimation.mean("y"), sample.estimation.total("y")])
        assert "q_method" not in res.__plain_str__()

    @pytest.mark.parametrize("q_method", sorted(Q_METHODS))
    def test_json_round_trip(self, sample, jk_sample, q_method):
        expected = Q_METHODS[q_method].value
        single = from_json(to_json(sample.estimation.median("y", q_method=q_method)))
        assert single.q_method == expected
        many = from_json(
            to_json(
                jk_sample.estimation.quantile(
                    ["y", "z"], p=[0.25, 0.75], q_method=q_method, method="replication"
                )
            )
        )
        assert [e.q_method for e in many.estimates] == [expected] * 4


# ---------------------------------------------------------------------------
# Non-quantile results carry no rule
# ---------------------------------------------------------------------------

NON_QUANTILE = {
    "mean": lambda s, **kw: s.estimation.mean("y", **kw),
    "mean_as_factor": lambda s, **kw: s.estimation.mean("g", as_factor=True, **kw),
    "total": lambda s, **kw: s.estimation.total("y", **kw),
    "total_as_factor": lambda s, **kw: s.estimation.total("g", as_factor=True, **kw),
    "prop": lambda s, **kw: s.estimation.prop("g", **kw),
    "ratio": lambda s, **kw: s.estimation.ratio("y", "z", **kw),
    "corr": lambda s, **kw: s.estimation.corr(("y", "z"), **kw),
    "cov": lambda s, **kw: s.estimation.cov(("y", "z"), **kw),
    "mean_several_y": lambda s, **kw: s.estimation.mean(["y", "z"], **kw),
}


def _assert_no_rule(res) -> None:
    for m in _members(res):
        assert m.q_method is None
        assert json.loads(to_json(m))["q_method"] is None
        assert from_json(to_json(m)).q_method is None
        assert "q_method" not in m.__plain_str__()


class TestNonQuantileHasNoRule:
    # The single-row "solo" domain puts a factor level at p = 0/1.
    @pytest.mark.filterwarnings(r"ignore:\[PROP_CI_BOUNDARY\]:svy.SvyUserWarning")
    @pytest.mark.parametrize("estimator", sorted(NON_QUANTILE))
    @pytest.mark.parametrize("shape", sorted(SHAPES))
    def test_taylor(self, sample, estimator, shape):
        _assert_no_rule(NON_QUANTILE[estimator](sample, **SHAPES[shape]))

    @pytest.mark.parametrize("estimator", ["mean", "total", "prop", "ratio", "mean_several_y"])
    def test_replication(self, jk_sample, estimator):
        _assert_no_rule(NON_QUANTILE[estimator](jk_sample, method="replication"))

    @pytest.mark.parametrize("estimator", ["mean", "total", "ratio"])
    def test_with_deff_and_where(self, sample, estimator):
        res = NON_QUANTILE[estimator](sample, deff="wr", where=svy.col("z") > 8)
        _assert_no_rule(res)
        assert "deff=wr" in res.__plain_str__().splitlines()[0]

    def test_median_with_where_keeps_rule(self, sample):
        res = sample.estimation.median("y", q_method="lower", where=svy.col("z") > 8)
        _assert_reported(res, QuantileMethod.LOWER)
        assert res.where_clause is not None

    def test_list_mixing_median_and_mean(self, sample):
        mixed = EstimateList(
            [
                sample.estimation.mean("y"),
                sample.estimation.median("y", q_method="nearest"),
                sample.estimation.total("y"),
            ]
        )
        assert [m.q_method for m in mixed] == [None, QuantileMethod.NEAREST, None]
        payload = json.loads(to_json(mixed))
        assert [e["q_method"] for e in payload["estimates"]] == [None, "Nearest", None]
        back = from_json(to_json(mixed))
        assert [e.q_method for e in back.estimates] == [None, "Nearest", None]
        assert "q_method=nearest" in mixed.__plain_str__().splitlines()[0]

    def test_old_payload_with_linear_still_decodes(self, sample):
        payload = json.loads(to_json(sample.estimation.mean("y")))
        payload["q_method"] = "Linear"
        assert from_json(json.dumps(payload).encode()).q_method == "Linear"

    def test_payload_without_the_field_decodes_to_none(self, sample):
        payload = json.loads(to_json(sample.estimation.mean("y")))
        del payload["q_method"]
        assert from_json(json.dumps(payload).encode()).q_method is None

    def test_default_on_a_new_estimate(self):
        from svy.core.enumerations import PopParam
        from svy.estimation.estimate import Estimate

        assert Estimate(PopParam.MEAN).q_method is None
        assert Estimate(PopParam.MEDIAN).q_method is None
