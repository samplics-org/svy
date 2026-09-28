# tests/svy/estimation/test_enum_order.py
"""Levels of an Enum column come back in the Enum's order.

Rows are otherwise sorted by level (see test_row_order.py); an Enum states
its own order, which ``estimates``, ``keys()``, ``to_polars()``, printing and
saved payloads follow.
"""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl
import pytest

from polars.testing import assert_frame_equal

from svy import Design, Sample, estd
from svy.serialize import from_json, to_json, to_polars


N = 480

Q_ORDER = ["Jul-Sep 2023", "Oct-Dec 2023", "Jan-Mar 2024", "Apr-Jun 2024"]
R_ORDER = ["South", "North"]
S_ORDER = ["low", "mid", "high"]


@pytest.fixture(scope="module")
def data() -> pl.DataFrame:
    rng = np.random.default_rng(20260928)
    return pl.DataFrame(
        {
            "stratum": np.repeat(["s1", "s2"], N // 2),
            "psu": np.repeat(np.arange(48), N // 48),
            "w": rng.uniform(1, 3, N),
            "y": rng.normal(10, 2, N),
            "x": rng.uniform(1, 2, N),
            "quarter": rng.choice(Q_ORDER, N),
            "region": rng.choice(R_ORDER, N),
            "status": rng.choice(S_ORDER, N),
            "sex": rng.choice(["m", "f"], N),
        }
    ).with_columns(
        pl.col("quarter").cast(pl.Enum(Q_ORDER)),
        pl.col("region").cast(pl.Enum(R_ORDER)),
        pl.col("status").cast(pl.Enum(S_ORDER)),
        quarter_cat=pl.col("quarter").cast(pl.Categorical),
    )


@pytest.fixture(scope="module")
def sample(data) -> Sample:
    return Sample(data, Design(stratum="stratum", psu="psu", wgt="w"))


@pytest.fixture(scope="module")
def rep_sample(sample) -> Sample:
    return sample.weighting.create_jk_wgts()


def _by(result) -> list:
    return [p.by_level[0] if len(p.by_level) == 1 else p.by_level for p in result.estimates]


@pytest.mark.parametrize("which", ["sample", "rep_sample"])
class TestEnumOrder:
    @pytest.mark.parametrize("stat", ["mean", "total", "median"])
    def test_by(self, request, which, stat):
        s = request.getfixturevalue(which)
        r = getattr(s.estimation, stat)("y", by="quarter")
        assert _by(r) == Q_ORDER
        assert r.domains == Q_ORDER

    def test_ratio_by(self, request, which):
        s = request.getfixturevalue(which)
        assert _by(s.estimation.ratio("y", "x", by="quarter")) == Q_ORDER

    def test_prop_levels(self, request, which):
        s = request.getfixturevalue(which)
        assert [p.y_level for p in s.estimation.prop("status").estimates] == S_ORDER

    def test_prop_by(self, request, which):
        s = request.getfixturevalue(which)
        r = s.estimation.prop("status", by="quarter")
        assert [(p.by_level[0], p.y_level) for p in r.estimates] == list(
            itertools.product(Q_ORDER, S_ORDER)
        )

    def test_mean_as_factor(self, request, which):
        s = request.getfixturevalue(which)
        r = s.estimation.mean("status", as_factor=True)
        assert [p.y_level for p in r.estimates] == S_ORDER

    def test_several_by(self, request, which):
        s = request.getfixturevalue(which)
        r = s.estimation.mean("y", by=["quarter", "region"])
        assert _by(r) == list(itertools.product(Q_ORDER, R_ORDER))

    def test_enum_beside_plain_by(self, request, which):
        s = request.getfixturevalue(which)
        r = s.estimation.mean("y", by=["sex", "quarter"])
        assert _by(r) == list(itertools.product(["f", "m"], Q_ORDER))
        r = s.estimation.ratio("y", "x", by=["region", "sex"])
        assert _by(r) == list(itertools.product(R_ORDER, ["f", "m"]))


class TestOtherColumnsKeepSortedOrder:
    def test_categorical_sorts_by_level(self, sample):
        r = sample.estimation.mean("y", by="quarter_cat")
        assert _by(r) == sorted(Q_ORDER)
        assert r.level_orders == {}

    def test_level_orders_lists_enums_only(self, sample):
        r = sample.estimation.prop("status", by=["sex", "quarter"])
        assert r.level_orders == {"quarter": Q_ORDER, "status": S_ORDER}


def test_unobserved_category_keeps_the_rest_in_order(data):
    order = ["Apr-Jun 2023", *Q_ORDER]
    df = data.with_columns(pl.col("quarter").cast(pl.String).cast(pl.Enum(order)))
    s = Sample(df, Design(stratum="stratum", psu="psu", wgt="w"))
    assert _by(s.estimation.mean("y", by="quarter")) == Q_ORDER


def test_covariance_follows_rows(sample):
    r = sample.estimation.prop("status", by="quarter")
    np.testing.assert_allclose(np.diag(r.covariance), [p.se**2 for p in r.estimates], rtol=1e-12)


class TestTables:
    def test_to_polars(self, sample):
        r = sample.estimation.prop("status", by=["quarter", "region"])
        df = r.to_polars(use_labels=False)
        assert df.select("quarter", "region", "status").rows() == list(
            itertools.product(Q_ORDER, R_ORDER, S_ORDER)
        )

    def test_printable_keeps_enum_order_under_labels(self, data):
        s = Sample(data, Design(stratum="stratum", psu="psu", wgt="w"))
        # Labels whose text sorts in the opposite order.
        s.meta.set_value_labels("quarter", dict(zip(Q_ORDER, ["d", "c", "b", "a"])))
        r = s.estimation.mean("y", by="quarter")
        assert r.to_polars()["quarter"].to_list() == Q_ORDER
        assert r.to_polars()["quarter_label"].to_list() == ["d", "c", "b", "a"]
        assert r.to_polars_printable()["quarter"].to_list() == ["d", "c", "b", "a"]

    def test_saved_payload_keeps_order(self, sample):
        r = sample.estimation.prop("status", by="quarter")
        back = from_json(to_json(r))
        assert back.level_orders == {"quarter": Q_ORDER, "status": S_ORDER}
        assert_frame_equal(to_polars(back), r.to_polars())


class TestContrast:
    def test_by_level(self, sample):
        r = sample.estimation.mean("y", by="quarter")
        c = r.contrast(estd("Apr-Jun 2024") - estd("Jul-Sep 2023"))
        est = {p.by_level[0]: p.est for p in r.estimates}
        assert c.estimates[0].est == pytest.approx(est["Apr-Jun 2024"] - est["Jul-Sep 2023"])

        i, j = r.keys().index("Apr-Jun 2024"), r.keys().index("Jul-Sep 2023")
        v = r.covariance
        se = np.sqrt(v[i, i] + v[j, j] - 2 * v[i, j])
        assert c.estimates[0].se == pytest.approx(se, rel=1e-12)

    def test_several_by(self, sample):
        r = sample.estimation.mean("y", by=["quarter", "region"])
        c = r.contrast(estd("Jan-Mar 2024", "North") - estd("Jan-Mar 2024", "South"))
        est = dict(zip(r.keys(), (p.est for p in r.estimates)))
        want = est[("Jan-Mar 2024", "North")] - est[("Jan-Mar 2024", "South")]
        assert c.estimates[0].est == pytest.approx(want)

    def test_prop_cell(self, sample):
        r = sample.estimation.prop("status", by="quarter")
        c = r.contrast(estd("Oct-Dec 2023", "high") - estd("Oct-Dec 2023", "low"))
        est = dict(zip(r.keys(), (p.est for p in r.estimates)))
        want = est[("Oct-Dec 2023", "high")] - est[("Oct-Dec 2023", "low")]
        assert c.estimates[0].est == pytest.approx(want)
