# tests/svy/selection/test_add_stage_multistage.py
"""add_stage chains past two stages: districts, EAs, households, persons."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from svy import Design, Sample
from svy.core.constants import SVY_HIT, SVY_PROB, SVY_PROB_STAGE1, SVY_WEIGHT
from svy.errors import SelectionError


def _p(k: int) -> str:
    return f"{SVY_PROB}_stage{k}"


def _w(k: int) -> str:
    return f"{SVY_WEIGHT}_stage{k}"


def _districts() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "district": [1, 2, 3, 4, 5, 6],
            "region": ["N", "N", "N", "S", "S", "S"],
            "mos": [10.0, 20.0, 25.0, 15.0, 30.0, 20.0],
        }
    )


def _eas() -> pl.DataFrame:
    district = [d for d in range(1, 7) for _ in range(4)]
    return pl.DataFrame(
        {
            "ea": list(range(1, 25)),
            "district": district,
            "ea_mos": [float(i % 3 + 1) for i in range(24)],
            "label": [f"ea{i}" for i in range(1, 25)],
        }
    )


def _hhs() -> pl.DataFrame:
    ea = [e for e in range(1, 25) for _ in range(5)]
    return pl.DataFrame(
        {
            "hid": list(range(1, len(ea) + 1)),
            "ea": ea,
            "y": np.random.default_rng(7).normal(50, 10, len(ea)),
        }
    )


def _persons() -> pl.DataFrame:
    hid = [h for h in range(1, 121) for _ in range(3)]
    return pl.DataFrame(
        {
            "pid": list(range(1, len(hid) + 1)),
            "hid": hid,
            "age": [20 + i % 50 for i in range(len(hid))],
        }
    )


def _within(sample: Sample, frame: pl.DataFrame, key: str, **design) -> Sample:
    """The next frame restricted to the selected units, linked on ``key``."""
    selected = sample.data[key].unique().to_list()
    return Sample(frame.filter(pl.col(key).is_in(selected)), Design(psu=key, **design))


def _stage1(rstate=1) -> Sample:
    return Sample(
        _districts(), Design(stratum="region", psu="district", mos="mos")
    ).sampling.pps_sys(n=2, by="region", rstate=rstate, certainty_threshold=1.0)


def _stage2(rstate=1, pps=False) -> Sample:
    s = _stage1(rstate)
    if pps:
        s = s.sampling.add_stage(_within(s, _eas(), "district"))
        return s.update_design(mos="ea_mos").sampling.pps_sys(
            n=2, by="district", rstate=rstate + 1, certainty_threshold=1.0
        )
    s = s.sampling.add_stage(_within(s, _eas(), "district"))
    return s.sampling.srs(n=2, by="district", rstate=rstate + 1)


def _add3(rstate=1, pps=False) -> Sample:
    s = _stage2(rstate, pps)
    return s.sampling.add_stage(_within(s, _hhs(), "ea"))


def _stage3(rstate=1, pps=False) -> Sample:
    return _add3(rstate, pps).sampling.srs(n=2, by="ea", rstate=rstate + 2)


def _stage4(rstate=1) -> Sample:
    s = _stage3(rstate)
    s = s.sampling.add_stage(_within(s, _persons(), "hid"))
    return s.sampling.srs(n=1, by="hid", rstate=rstate + 3)


def _expected_stage1(df: pl.DataFrame) -> np.ndarray:
    frame = _districts().with_columns(
        (2 * pl.col("mos") / pl.col("mos").sum().over("region")).alias("p1")
    )
    return df.join(frame.select("district", "p1"), on="district", how="left")["p1"].to_numpy()


def _expected_ea_pps(df: pl.DataFrame) -> np.ndarray:
    frame = _eas().with_columns(
        (2 * pl.col("ea_mos") / pl.col("ea_mos").sum().over("district")).alias("p2")
    )
    return df.join(frame.select("ea", "p2"), on="ea", how="left")["p2"].to_numpy()


def _product(df: pl.DataFrame, k: int) -> np.ndarray:
    return np.prod([df[_p(j)].to_numpy() for j in range(1, k + 1)], axis=0)


class TestThreeStages:
    def test_rows_are_the_selected_households(self):
        out = _stage3()
        df = out.data
        assert df.height == 2 * 2 * 2 * 2
        assert df["hid"].n_unique() == df.height
        assert set(df["ea"].to_list()) <= set(_stage2().data["ea"].to_list())

    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_stage_columns_present(self, k):
        df = _stage3().data
        assert _p(k) in df.columns
        assert _w(k) in df.columns

    def test_stage_probabilities(self):
        df = _stage3().data
        np.testing.assert_allclose(df[_p(1)].to_numpy(), _expected_stage1(df), rtol=1e-12)
        np.testing.assert_allclose(df[_p(2)].to_numpy(), 2 / 4, rtol=1e-12)
        np.testing.assert_allclose(df[_p(3)].to_numpy(), 2 / 5, rtol=1e-12)

    def test_pps_second_stage_probabilities(self):
        df = _stage3(pps=True).data
        np.testing.assert_allclose(df[_p(2)].to_numpy(), _expected_ea_pps(df), rtol=1e-12)
        np.testing.assert_allclose(df[SVY_PROB].to_numpy(), _product(df, 3), rtol=1e-12)

    def test_overall_probability_is_product(self):
        df = _stage3().data
        np.testing.assert_allclose(df[SVY_PROB].to_numpy(), _product(df, 3), rtol=1e-12)
        np.testing.assert_allclose(
            df[SVY_PROB].to_numpy(), _expected_stage1(df) * 0.5 * 0.4, rtol=1e-12
        )

    def test_weights_are_inverse_probabilities(self):
        df = _stage3().data
        np.testing.assert_allclose(df[SVY_WEIGHT].to_numpy(), 1 / df[SVY_PROB].to_numpy())
        for k in (1, 2, 3):
            np.testing.assert_allclose(df[_w(k)].to_numpy(), 1 / df[_p(k)].to_numpy())

    def test_hits_and_certainty_per_stage(self):
        df = _stage3(pps=True).data
        for col in (
            "svy_number_of_hits_stage1",
            "svy_number_of_hits_stage2",
            "svy_certainty_stage1",
            "svy_certainty_stage2",
            SVY_HIT,
        ):
            assert col in df.columns

    def test_design_after_selection(self):
        d = _stage3().design
        assert d.stratum == "region"
        assert d.psu == "district"
        assert d.ssu is None
        assert d.prob == SVY_PROB
        assert d.wgt == SVY_WEIGHT
        assert d.hit == SVY_HIT
        assert d.wr is False

    def test_design_before_selection(self):
        out = _add3()
        assert out.design.psu == "district"
        assert out.design.ssu is None
        assert out.design.prob == SVY_PROB_STAGE1
        assert out.design.wgt is None
        # The stale combined columns of stage 2 are gone; the next draw rewrites them.
        assert SVY_PROB not in out.data.columns
        assert SVY_WEIGHT not in out.data.columns

    def test_third_stage_frame_is_population(self):
        out = _add3()
        assert out.data.height == 2 * 2 * 2 * 5
        assert out.data["hid"].n_unique() == out.data.height

    def test_estimation_is_horvitz_thompson(self):
        out = _stage3()
        df = out.data
        est = out.estimation.total("y").to_polars()
        np.testing.assert_allclose(est["est"][0], (df["y"] * df[SVY_WEIGHT]).sum(), rtol=1e-12)
        assert np.isfinite(est["se"][0])
        assert np.isfinite(out.estimation.mean("y").to_polars()["se"][0])

    def test_custom_names_at_third_stage(self):
        s = _stage2()
        out = s.sampling.add_stage(_within(s, _hhs(), "ea"), prob_name="p", wgt_name="w")
        out = out.sampling.srs(n=2, by="ea", rstate=3)
        df = out.data
        assert out.design.prob == "p"
        assert out.design.wgt == "w"
        np.testing.assert_allclose(df["p"].to_numpy(), _product(df, 3), rtol=1e-12)
        np.testing.assert_allclose(df["w"].to_numpy(), 1 / df["p"].to_numpy())

    def test_preselected_third_stage(self):
        s = _stage2()
        hh = _within(s, _hhs(), "ea").sampling.srs(n=2, by="ea", rstate=3)
        out = s.sampling.add_stage(hh)
        df = out.data
        assert df.height == 16
        assert out.design.prob == SVY_PROB
        assert out.design.wgt == SVY_WEIGHT
        assert out.design.ssu is None
        np.testing.assert_allclose(df[_p(3)].to_numpy(), 0.4, rtol=1e-12)
        np.testing.assert_allclose(df[SVY_PROB].to_numpy(), _product(df, 3), rtol=1e-12)
        np.testing.assert_allclose(df[SVY_WEIGHT].to_numpy(), 1 / _product(df, 3))

    def test_colliding_columns_get_last_stage_suffix(self):
        s = _stage2()
        hh = _hhs().join(_eas().select("ea", "district", "label"), on="ea")
        out = s.sampling.add_stage(_within(s, hh, "ea")).sampling.srs(n=2, by="ea", rstate=3)
        df = out.data
        assert "district_stage2" in df.columns
        assert "label_stage2" in df.columns
        assert (df["district"] == df["district_stage2"]).all()
        assert out.design.psu == "district"
        assert np.isfinite(out.estimation.mean("y").to_polars()["se"][0])

    def test_unmatched_units_warn(self):
        s = _stage2()
        keep = s.data["ea"].unique().sort().to_list()[1:]
        hh = Sample(_hhs().filter(pl.col("ea").is_in(keep)), Design(psu="ea"))
        with pytest.warns(UserWarning, match="STAGE_PSUS_UNMATCHED"):
            out = s.sampling.add_stage(hh)
        assert out.data["ea"].n_unique() == len(keep)


class TestFourStages:
    def test_rows_are_the_selected_persons(self):
        df = _stage4().data
        assert df.height == 16
        assert df["pid"].n_unique() == 16
        assert df["hid"].n_unique() == 16

    def test_stage_probabilities(self):
        df = _stage4().data
        np.testing.assert_allclose(df[_p(1)].to_numpy(), _expected_stage1(df), rtol=1e-12)
        np.testing.assert_allclose(df[_p(2)].to_numpy(), 0.5, rtol=1e-12)
        np.testing.assert_allclose(df[_p(3)].to_numpy(), 0.4, rtol=1e-12)
        np.testing.assert_allclose(df[_p(4)].to_numpy(), 1 / 3, rtol=1e-12)

    def test_overall_probability_is_product(self):
        df = _stage4().data
        np.testing.assert_allclose(df[SVY_PROB].to_numpy(), _product(df, 4), rtol=1e-12)
        np.testing.assert_allclose(df[SVY_WEIGHT].to_numpy(), 1 / _product(df, 4))
        for k in (1, 2, 3, 4):
            np.testing.assert_allclose(df[_w(k)].to_numpy(), 1 / df[_p(k)].to_numpy())

    def test_hits_per_stage(self):
        df = _stage4().data
        for k in (1, 2, 3):
            assert f"svy_number_of_hits_stage{k}" in df.columns
        assert SVY_HIT in df.columns

    def test_design(self):
        d = _stage4().design
        assert (d.stratum, d.psu, d.ssu) == ("region", "district", None)
        assert (d.prob, d.wgt, d.hit) == (SVY_PROB, SVY_WEIGHT, SVY_HIT)

    def test_estimation_runs(self):
        out = _stage4()
        est = out.estimation.mean("age").to_polars()
        assert np.isfinite(est["se"][0])


class TestInclusionFrequency:
    def test_empirical_inclusion_matches_probability(self):
        reps = 250
        p1 = _districts().with_columns(
            (pl.col("mos") / pl.col("mos").sum().over("region")).alias("p1")
        )
        expected = (
            _hhs()
            .join(_eas().select("ea", "district"), on="ea")
            .join(p1.select("district", "p1"), on="district")
            .select("hid", (pl.col("p1") * 2 / 4 * 1 / 5).alias("p"))
            .sort("hid")
        )
        counts = dict.fromkeys(expected["hid"].to_list(), 0)
        rng = np.random.default_rng(2026)
        for _ in range(reps):
            seed = int(rng.integers(1 << 31))
            s = Sample(
                _districts(), Design(stratum="region", psu="district", mos="mos")
            ).sampling.pps_sys(n=1, by="region", rstate=seed)
            s = s.sampling.add_stage(_within(s, _eas(), "district"))
            s = s.sampling.srs(n=2, by="district", rstate=seed + 1)
            s = s.sampling.add_stage(_within(s, _hhs(), "ea"))
            s = s.sampling.srs(n=1, by="ea", rstate=seed + 2)
            got = s.data.join(expected, on="hid")
            np.testing.assert_allclose(got[SVY_PROB].to_numpy(), got["p"].to_numpy(), rtol=1e-12)
            for hid in s.data["hid"].to_list():
                counts[hid] += 1
        p = expected["p"].to_numpy()
        freq = np.array(list(counts.values())) / reps
        assert np.all(np.abs(freq - p) < 4.5 * np.sqrt(p * (1 - p) / reps))


class TestErrors:
    def test_bare_frame_past_second_stage_raises(self):
        s = _stage2()
        hh = _hhs().join(_eas().select("ea", "district"), on="ea")
        with pytest.raises(SelectionError, match="does not identify one selected unit") as exc:
            s.sampling.add_stage(hh.filter(pl.col("ea").is_in(s.data["ea"].to_list())))
        assert "psu=" in exc.value.hint

    def test_renamed_key_past_second_stage_raises(self):
        # ea_code is not in the current sample, so the link falls back to the stage-1 PSU.
        s = _stage2()
        hh = _hhs().join(_eas().select("ea", "district"), on="ea").rename({"ea": "ea_code"})
        hh = hh.filter(pl.col("ea_code").is_in(s.data["ea"].to_list()))
        with pytest.raises(SelectionError, match="does not identify one selected unit"):
            s.sampling.add_stage(Sample(hh, Design(psu="ea_code")))

    def test_add_stage_before_selecting_raises(self):
        s = _stage1()
        s = s.sampling.add_stage(_within(s, _eas(), "district"))
        with pytest.raises(SelectionError, match="not been selected yet"):
            s.sampling.add_stage(_hhs())

    def test_unmatched_link_values_raise(self):
        s = _stage2()
        hh = Sample(_hhs(), Design(psu="ea"))  # households of unselected EAs too
        with pytest.raises(SelectionError, match="next_stage PSU values"):
            s.sampling.add_stage(hh)
