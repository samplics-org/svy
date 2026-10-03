"""Estimate.ci_method records how the intervals were built; saved GLMs keep theta and offset."""

import numpy as np
import polars as pl
import pytest

import svy

from svy.serialize import from_json, serialize, to_json


@pytest.fixture(scope="module")
def sample() -> svy.Sample:
    rng = np.random.default_rng(11)
    n = 200
    return svy.Sample(
        pl.DataFrame(
            {
                "stratum": np.repeat([1, 2, 3, 4], n // 4),
                "psu": np.arange(n) // 5,
                "w": rng.uniform(1, 3, n),
                "y": rng.normal(10, 2, n),
                "x": rng.normal(5, 1, n),
                "b": rng.integers(0, 2, n),
                "cat": rng.choice(["a", "b", "c"], n),
                "k": rng.poisson(3, n),
                "expo": np.log(rng.uniform(1, 4, n)),
            }
        ),
        svy.Design(stratum="stratum", psu="psu", wgt="w"),
    )


@pytest.fixture(scope="module")
def rep_sample(sample) -> svy.Sample:
    return sample.weighting.create_jk_wgts()


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        (lambda s: s.estimation.mean("y"), "wald"),
        (lambda s: s.estimation.total("y"), "wald"),
        (lambda s: s.estimation.ratio("y", "x"), "wald"),
        (lambda s: s.estimation.prop("cat"), "logit"),
        (lambda s: s.estimation.prop("cat", ci_method="beta"), "beta"),
        (lambda s: s.estimation.prop("cat", ci_method="kg"), "korn-graubard"),
        (lambda s: s.estimation.prop("cat", ci_method="wilson"), "wilson"),
        (lambda s: s.estimation.mean("cat", as_factor=True), "logit"),
        (lambda s: s.estimation.median("y"), "woodruff"),
        (lambda s: s.estimation.quantile("y", p=0.25), "woodruff"),
        (lambda s: s.estimation.corr(("y", "x")), "fisher"),
        (lambda s: s.estimation.corr(("y", "x"), ci_method="wald"), "wald"),
        (lambda s: s.estimation.cov(("y", "x")), "wald"),
    ],
)
def test_taylor_records_the_interval_method(sample, call, expected):
    est = call(sample)
    assert est.ci_method == expected
    assert serialize(est).ci_method == expected


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        (lambda s: s.estimation.mean("y", method="replication"), "wald"),
        (lambda s: s.estimation.prop("cat", method="replication", ci_method="beta"), "beta"),
        (lambda s: s.estimation.median("y", method="replication"), "wald"),
        (lambda s: s.estimation.corr(("y", "x"), method="replication"), "fisher"),
    ],
)
def test_replication_records_the_interval_method(rep_sample, call, expected):
    assert call(rep_sample).ci_method == expected


def test_by_domains_keep_one_method(sample):
    est = sample.estimation.prop("b", by="cat", ci_method="wilson")
    assert est.ci_method == "wilson"
    assert len(est.estimates) == 6


def test_ci_method_round_trips_through_json(sample):
    est = sample.estimation.prop("cat", ci_method="beta")
    assert from_json(to_json(est)).ci_method == "beta"


def test_old_payload_without_ci_method_decodes(sample):
    import json

    raw = json.loads(to_json(sample.estimation.mean("y")))
    raw.pop("ci_method")
    raw["schema_version"] = "svy-result/0.6"
    assert from_json(json.dumps(raw)).ci_method is None


def test_negative_binomial_theta_and_offset_saved(sample):
    fit = sample.glm.fit("k", x=["x"], family="negative_binomial", offset="expo")
    data = serialize(fit)
    assert data.stats.theta == pytest.approx(fit.stats.theta)
    assert data.stats.theta_se == pytest.approx(fit.stats.theta_se)
    assert data.offset == "expo"
    back = from_json(to_json(fit))
    assert back.stats.theta == pytest.approx(fit.stats.theta)
    assert back.offset == "expo"


def test_other_families_leave_theta_and_offset_empty(sample):
    data = serialize(sample.glm.fit("y", x=["x"]))
    assert data.stats.theta is None
    assert data.stats.theta_se is None
    assert data.offset is None
