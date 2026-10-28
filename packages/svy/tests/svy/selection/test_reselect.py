"""Selecting again from an already-selected sample: taken output names raise, new names chain."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from svy import Sample
from svy.core.constants import SVY_CERTAINTY, SVY_HIT, SVY_PROB
from svy.core.design import Design
from svy.errors import SelectionError, SelectionWarning


N = 1000


def _sample() -> Sample:
    data = pl.DataFrame(
        {"id": range(N), "w": [1.0] * N, "m": np.arange(1, N + 1, dtype=np.float64)}
    )
    return Sample(data, Design(wgt="w", mos="m"))


def _select(sample: Sample, method: str, n: int, **kwargs) -> Sample:
    return getattr(sample.sampling, method)(n=n, rstate=1, **kwargs)


METHODS = ["srs", "pps_sys", "pps_wr", "pps_brewer", "pps_murphy", "pps_rs"]


@pytest.mark.parametrize("method", METHODS)
def test_second_selection_with_default_names_raises(method):
    s = _select(_sample(), method, 2 if method == "pps_murphy" else 20)
    before = s.data
    with pytest.raises(SelectionError) as exc:
        _select(s, method, 2)
    assert exc.value.code == "METHOD_NOT_APPLICABLE"
    msg = str(exc.value)
    for name in (SVY_PROB, SVY_HIT, "'w'", "prob_name=", "wgt_name=", "hit_name=", "add_stage"):
        assert name in msg
    assert s.data.equals(before)


def test_second_selection_without_wgt_name_raises_on_weight():
    s = _select(_sample(), "srs", 20)
    with pytest.raises(SelectionError) as exc:
        _select(s, "srs", 5, prob_name="p2", hit_name="h2")
    assert exc.value.code == "METHOD_NOT_APPLICABLE"
    assert "'w'" in str(exc.value)
    assert "wgt_name=" in str(exc.value)
    assert "prob_name=" not in str(exc.value)


@pytest.mark.parametrize("method", ["pps_sys", "pps_wr"])
def test_second_pps_selection_raises_on_certainty(method):
    s = _select(_sample(), method, 20)
    with pytest.raises(SelectionError) as exc:
        _select(s, method, 5, prob_name="p2", wgt_name="w2", hit_name="h2")
    assert exc.value.code == "METHOD_NOT_APPLICABLE"
    assert SVY_CERTAINTY in str(exc.value)
    assert "add_stage" in str(exc.value)


def test_second_srs_with_new_names_chains_probabilities():
    s = _select(_sample(), "srs", 20)
    s = _select(s, "srs", 5, prob_name="p2", wgt_name="w2", hit_name="h2")
    d = s.data
    assert d.height == 5
    assert not [c for c in d.columns if c.endswith("_right")]
    # pi = 20/1000 * 5/20
    np.testing.assert_allclose(d["p2"].to_numpy(), 0.005, rtol=1e-12)
    np.testing.assert_allclose(d["w2"].to_numpy(), 200.0, rtol=1e-12)
    np.testing.assert_allclose(d[SVY_PROB].to_numpy(), 0.02, rtol=1e-12)
    np.testing.assert_allclose(d["w"].to_numpy(), 50.0, rtol=1e-12)
    assert d["h2"].to_list() == [1] * 5
    assert (s.design.prob, s.design.wgt, s.design.hit) == ("p2", "w2", "h2")


def test_srs_after_pps_with_new_names_chains_probabilities():
    s = _select(_sample(), "pps_sys", 20)
    first = dict(zip(s.data["id"], s.data[SVY_PROB]))
    s = _select(s, "srs", 4, prob_name="p2", wgt_name="w2", hit_name="h2")
    for i, p in zip(s.data["id"], s.data["p2"]):
        assert p == pytest.approx(first[i] * 4 / 20, rel=1e-12)
    assert s.design.prob == "p2"


@pytest.mark.parametrize("method", ["srs", "pps_sys"])
def test_prior_prob_column_is_kept_and_chained_into_a_new_column(method):
    data = pl.DataFrame({"id": range(10), "prev_prob": [0.5] * 10, "m": [1.0] * 10})
    s = Sample(data, Design(prob="prev_prob", mos="m"))
    s = _select(s, method, 2)
    d = s.data
    assert not [c for c in d.columns if c.endswith("_right")]
    assert d["prev_prob"].to_list() == [0.5, 0.5]
    # equal sizes, so pps is 2/10 as well: pi = 0.5 * 2/10
    np.testing.assert_allclose(d[SVY_PROB].to_numpy(), 0.1, rtol=1e-12)
    np.testing.assert_allclose(d[s.design.wgt].to_numpy(), 10.0, rtol=1e-12)
    assert s.design.prob == SVY_PROB


def test_prior_prob_column_survives_where_matching_no_rows():
    data = pl.DataFrame({"id": range(10), "prev_prob": [0.5] * 10})
    s = Sample(data, Design(prob="prev_prob"))
    with pytest.warns(SelectionWarning, match=r"SELECTION_(STRATUM|GROUP)_EMPTY"):
        s = s.sampling.srs(n=2, where=pl.col("id") < 0, rstate=1)
    assert s.data["prev_prob"].to_list() == [0.5] * 10
    assert s.data[SVY_PROB].null_count() == 10


def test_design_hit_column_already_in_data_raises():
    data = pl.DataFrame({"id": range(10), "h": [1] * 10})
    s = Sample(data, Design(hit="h"))
    with pytest.raises(SelectionError) as exc:
        s.sampling.srs(n=2, rstate=1)
    assert exc.value.code == "METHOD_NOT_APPLICABLE"
    assert "'h'" in str(exc.value)
    assert "hit_name=" in str(exc.value)


def test_first_selection_still_writes_the_design_weight_column():
    s = _select(_sample(), "srs", 20)
    np.testing.assert_allclose(s.data["w"].to_numpy(), 50.0, rtol=1e-12)
    assert s.design.wgt == "w"
