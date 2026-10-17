# tests/svy/selection/test_multistage_stages.py
"""sample.sampling.multistage: every stage drawn in one call from a register."""

import numpy as np
import polars as pl
import pytest

import svy

from svy.errors.method_errors import MethodError
from svy.selection import Stage


def _register(seed=0) -> pl.DataFrame:
    """Two regions, six districts each, 2-5 EAs per district, 20-59 dwellings per EA."""
    rng = np.random.default_rng(seed)
    rows = []
    for region in ["north", "south"]:
        for d in range(1, 7):
            for ea in range(1, int(rng.integers(3, 7))):
                hh = int(rng.integers(20, 60))
                for dw in range(1, hh + 1):
                    rows.append((region, d, ea, hh, dw, float(rng.normal(100, 20))))
    return pl.DataFrame(
        rows, schema=["region", "district", "ea", "ea_hh", "dwelling", "y"], orient="row"
    )


FRAME = _register()
STAGES = [Stage("district", n=3), Stage("ea", n=2, mos="ea_hh"), Stage(None, n=10)]


def _draw(stages=STAGES, frame=FRAME, rstate=1, stratum="region"):
    return svy.Sample(frame, svy.Design(stratum=stratum)).sampling.multistage(
        stages, rstate=rstate
    )


class TestResult:
    def test_rows_and_columns(self):
        out = _draw()
        assert out.data.height == 2 * 3 * 2 * 10
        for k in (1, 2, 3):
            assert f"svy_prob_selection_stage{k}" in out.data.columns
            assert f"svy_sample_weight_stage{k}" in out.data.columns

    def test_design(self):
        d = _draw().design
        assert (d.stratum, d.psu, d.ssu) == ("region", "district", "ea")
        assert (d.prob, d.wgt, d.wr) == ("svy_prob_selection", "svy_sample_weight", False)

    def test_probability_is_the_product_and_weight_its_inverse(self):
        df = _draw().data
        prod = (
            df["svy_prob_selection_stage1"]
            * df["svy_prob_selection_stage2"]
            * df["svy_prob_selection_stage3"]
        )
        np.testing.assert_allclose(df["svy_prob_selection"], prod)
        np.testing.assert_allclose(df["svy_sample_weight"] * df["svy_prob_selection"], 1.0)
        np.testing.assert_allclose(
            df["svy_sample_weight_stage2"] * df["svy_prob_selection_stage2"], 1.0
        )

    def test_stage_probabilities(self):
        df = _draw().data
        n_districts = FRAME.group_by("region").agg(pl.col("district").n_unique())
        assert set(df["svy_prob_selection_stage1"].unique()) == {3 / 6}
        eas = FRAME.unique(["region", "district", "ea"])
        for rec in df.unique(["region", "district", "ea"]).iter_rows(named=True):
            parent = eas.filter(
                (pl.col("region") == rec["region"]) & (pl.col("district") == rec["district"])
            )
            # A district with two EAs gives both, with probability 1.
            expected = 1.0 if parent.height <= 2 else 2 * rec["ea_hh"] / parent["ea_hh"].sum()
            assert rec["svy_prob_selection_stage2"] == pytest.approx(expected)
            assert rec["svy_prob_selection_stage3"] == pytest.approx(10 / rec["ea_hh"])
        assert n_districts["district"].to_list() == [6, 6]

    def test_ids_repeat_across_parents(self):
        """EA 1 of district 3 and EA 1 of district 5 are different units."""
        df = _draw().data
        per_district = df.unique(["region", "district", "ea"]).group_by("region", "district").len()
        assert per_district["len"].to_list() == [2] * 6

    def test_estimation_runs_on_the_result(self):
        est = _draw().estimation.total("y")
        assert np.isfinite(est.estimates[0].se)

    def test_no_findings_on_a_plain_draw(self):
        assert list(_draw().warnings) == []

    def test_same_rstate_same_draw(self):
        assert _draw(rstate=7).data.equals(_draw(rstate=7).data)
        assert not _draw(rstate=7).data.equals(_draw(rstate=8).data)

    def test_one_generator_drives_all_stages(self):
        g = np.random.default_rng(3)
        a, b = _draw(rstate=g).data, _draw(rstate=g).data
        assert not a.equals(b)


class TestStageShapes:
    def test_four_stages_and_a_composite_unit(self):
        # Four blocks per EA, each of at least five dwellings (an EA has 20 or more).
        frame = FRAME.with_columns(block=(pl.col("dwelling") - 1) % 4)
        stages = [
            Stage("district", n=2),
            Stage("ea", n=1),
            Stage("block", n=2),
            Stage(("dwelling",), n=3),
        ]
        out = _draw(stages, frame=frame)
        assert out.data.height == 2 * 2 * 1 * 2 * 3
        assert "svy_prob_selection_stage4" in out.data.columns

    def test_tuple_unit_in_the_design(self):
        frame = FRAME.with_columns(zone=pl.col("district") % 2)
        out = _draw([Stage(("zone", "district"), n=2), Stage(None, n=5)], frame=frame)
        assert out.design.psu == ("zone", "district")
        assert out.design.ssu is None

    def test_none_takes_all(self):
        out = _draw([Stage("district", n=None), Stage(None, n=1)])
        assert set(out.data["svy_prob_selection_stage1"].unique()) == {1.0}
        assert out.data.height == FRAME.unique(["region", "district"]).height

    def test_stage_one_sizes_by_stratum(self):
        out = _draw([Stage("district", n={"north": 1, "south": 4}), Stage(None, n=1)])
        counts = dict(out.data.group_by("region").len().iter_rows())
        assert counts == {"north": 1, "south": 4}

    def test_unstratified(self):
        out = svy.Sample(FRAME, svy.Design()).sampling.multistage(
            [Stage(("region", "district"), n=4), Stage(None, n=2)], rstate=1
        )
        assert out.data.height == 8
        assert set(out.data["svy_prob_selection_stage1"].unique()) == {4 / 12}


class TestTakesAll:
    def test_parent_with_too_few_units_gives_all_with_one_finding(self):
        out = _draw([Stage("district", n=2), Stage("ea", n=5), Stage(None, n=1)])
        eas = FRAME.unique(["region", "district", "ea"]).group_by("region", "district").len()
        selected = out.data.unique(["region", "district", "ea"])
        for rec in selected.group_by("region", "district").len().iter_rows(named=True):
            total = eas.filter(
                (pl.col("region") == rec["region"]) & (pl.col("district") == rec["district"])
            )["len"][0]
            assert rec["len"] == min(5, total)
        found = [w for w in out.warnings if w.code == "STAGE_TAKES_ALL"]
        assert len(found) == 1
        assert set(out.data["svy_prob_selection_stage2"].unique()) <= {1.0} | {
            5 / t for t in eas["len"].to_list()
        }

    def test_zero_skips_a_stratum(self):
        out = _draw([Stage("district", n={"north": 0, "south": 2}), Stage(None, n=1)])
        assert set(out.data["region"]) == {"south"}


class TestMonteCarlo:
    def test_total_is_unbiased_and_inclusion_matches_probability(self):
        frame = FRAME.filter(pl.col("region") == "north")
        stages = [Stage("district", n=2), Stage("ea", n=1, mos="ea_hh"), Stage(None, n=8)]
        true_total = frame["y"].sum()
        g = np.random.default_rng(2026)
        reps, totals, hits = 400, [], {}
        for _ in range(reps):
            out = svy.Sample(frame, svy.Design()).sampling.multistage(stages, rstate=g)
            df = out.data
            totals.append((df["y"] * df["svy_sample_weight"]).sum())
            for d in df["district"].unique().to_list():
                hits[d] = hits.get(d, 0) + 1
        totals = np.array(totals)
        assert abs(totals.mean() - true_total) < 4 * totals.std(ddof=1) / np.sqrt(reps)
        for freq in hits.values():
            assert abs(freq / reps - 2 / 6) < 4 * np.sqrt((2 / 6) * (4 / 6) / reps)


class TestErrors:
    @pytest.mark.parametrize(
        "kw, param",
        [
            (dict(unit=3), "unit"),
            (dict(unit=()), "unit"),
            (dict(unit="ea", n=-1), "n"),
            (dict(unit="ea", n=True), "n"),
            (dict(unit="ea", method="sys"), "mos"),
            (dict(unit="ea", mos="ea_hh", method="srs"), "method"),
            (dict(unit="ea", method="wr", mos="ea_hh"), "method"),
            (dict(unit="ea", certainty_threshold=0.9), "certainty_threshold"),
        ],
    )
    def test_invalid_stage(self, kw, param):
        with pytest.raises(MethodError) as exc:
            Stage(**kw)
        assert (exc.value.code, exc.value.param) == ("INVALID_STAGE", param)

    def test_stage_defaults(self):
        assert Stage("ea").method == "srs"
        assert Stage("ea", mos="ea_hh").method == "sys"
        assert Stage(["ea"]).unit == "ea"

    @pytest.mark.parametrize(
        "stages",
        [
            [Stage(None, n=2)],
            [Stage("district", n=2), Stage(None, n=2), Stage("ea", n=1)],
            [Stage("district", n=2), Stage("ea", n={"x": 1})],
            [],
        ],
    )
    def test_invalid_stage_lists(self, stages):
        with pytest.raises(MethodError) as exc:
            _draw(stages)
        assert exc.value.code == "INVALID_STAGE"

    def test_missing_column(self):
        with pytest.raises(MethodError) as exc:
            _draw([Stage("county", n=1)])
        assert exc.value.code == "MULTISTAGE_COLUMN_MISSING"

    def test_output_column_taken(self):
        with pytest.raises(MethodError) as exc:
            _draw(frame=FRAME.with_columns(svy_prob_selection_stage2=pl.lit(1.0)))
        assert exc.value.code == "MULTISTAGE_COLUMN_TAKEN"

    def test_frame_already_selected(self):
        s = svy.Sample(FRAME.with_columns(w=pl.lit(2.0)), svy.Design(stratum="region", wgt="w"))
        with pytest.raises(MethodError) as exc:
            s.sampling.multistage(STAGES, rstate=1)
        assert exc.value.code == "MULTISTAGE_ALREADY_SELECTED"

    def test_mos_varies_within_unit(self):
        frame = FRAME.with_columns(ea_hh=pl.col("ea_hh") + (pl.col("dwelling") == 1).cast(int))
        with pytest.raises(MethodError) as exc:
            _draw(frame=frame)
        assert (exc.value.code, exc.value.param) == ("MULTISTAGE_NOT_CONSTANT", "ea_hh")

    def test_stratum_missing_from_n(self):
        with pytest.raises(MethodError) as exc:
            _draw([Stage("district", n={"north": 1}), Stage(None, n=1)])
        assert exc.value.code == "MULTISTAGE_N_MISSING"


URB = FRAME.with_columns(
    urbrur=pl.when(pl.col("district") <= 3).then(pl.lit("Urban")).otherwise(pl.lit("Rural"))
)
TWO = ("region", "urbrur")


def _psus(out) -> dict:
    df = out.data.select(*TWO, "district").unique().group_by(*TWO).len()
    return {(r, u): n for r, u, n in df.iter_rows()}


class TestStageOneN:
    def test_full_tuple_keys(self):
        n = {("north", "Urban"): 1, ("north", "Rural"): 2, ("south", "Urban"): 3}
        out = _draw([Stage("district", n={**n, ("south", "Rural"): 0})], URB, stratum=TWO)
        assert _psus(out) == n

    def test_one_variable_broadcasts_across_the_other(self):
        out = _draw([Stage("district", n={"Urban": 2, "Rural": 1})], URB, stratum=TWO)
        assert _psus(out) == {
            ("north", "Urban"): 2,
            ("north", "Rural"): 1,
            ("south", "Urban"): 2,
            ("south", "Rural"): 1,
        }

    def test_first_variable_broadcasts_too(self):
        out = _draw([Stage("district", n={"north": 1, "south": 3})], URB, stratum=TWO)
        assert _psus(out) == {
            ("north", "Urban"): 1,
            ("north", "Rural"): 1,
            ("south", "Urban"): 3,
            ("south", "Rural"): 3,
        }

    def test_broadcast_draws_as_its_full_expansion(self):
        stages = [Stage("ea", n=1, mos="ea_hh"), Stage(None, n=5)]
        short = _draw([Stage("district", n={"Urban": 2, "Rural": 1}), *stages], URB, stratum=TWO)
        full = {
            (r, u): 2 if u == "Urban" else 1
            for r in ("north", "south")
            for u in ("Urban", "Rural")
        }
        long = _draw([Stage("district", n=full), *stages], URB, stratum=TWO)
        assert short.data.equals(long.data)

    def test_broadcast_with_pps_and_zero(self):
        out = _draw(
            [Stage("district", n={"Urban": 0, "Rural": 2}, mos="ea_hh"), Stage(None, n=1)],
            URB.with_columns(ea_hh=pl.col("district") * 10),
            stratum=TWO,
        )
        assert _psus(out) == {("north", "Rural"): 2, ("south", "Rural"): 2}

    def test_stage_one_probability_is_n_over_n_per_stratum(self):
        out = _draw([Stage("district", n={"Urban": 2, "Rural": 1})], URB, stratum=TWO)
        p = out.data.select("urbrur", "svy_prob_selection_stage1").unique()
        assert dict(p.iter_rows()) == {"Urban": 2 / 3, "Rural": 1 / 3}

    def test_int_stratum_keys(self):
        n = {d: 1 + d % 2 for d in range(1, 7)}
        out = _draw([Stage("ea", n=n)], stratum=("region", "district"))
        got = out.data.select("region", "district", "ea").unique().group_by("district").len()
        assert dict(got.iter_rows()) == {d: 2 * k for d, k in n.items()}

    @pytest.mark.parametrize(
        "n",
        [
            {"Urban": 2},
            {"north": 1, "south": 1, "east": 1},
            {("north", "Urban"): 1, "Rural": 1},
            {"north": 1, "Urban": 2},
            {("north", "Urban"): 1, ("north", "Rural"): 1, ("south", "Urban"): 1},
            {"Urban": 2, "Rural": 1, "Suburban": 1},
        ],
        ids=["incomplete", "orphan", "mixed", "ambiguous", "missing-cell", "extra-level"],
    )
    def test_keys_that_do_not_match_raise(self, n):
        with pytest.raises(MethodError) as exc:
            _draw([Stage("district", n=n)], URB, stratum=TWO)
        assert (exc.value.code, exc.value.param) == ("MULTISTAGE_N_MISSING", "n")

    def test_mapping_without_stratum_raises(self):
        with pytest.raises(MethodError) as exc:
            svy.Sample(FRAME, svy.Design()).sampling.multistage(
                [Stage("district", n={"north": 1})], rstate=1
            )
        assert exc.value.code == "MULTISTAGE_N_MISSING"
