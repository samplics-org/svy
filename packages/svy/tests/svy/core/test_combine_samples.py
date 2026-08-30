"""combine_samples — stacking repeated cross-sections and panel waves."""

from __future__ import annotations

import warnings

import polars as pl
import pytest

import svy

from svy.core.enumerations import MeasurementType
from svy.errors import MethodError


def _cycle(strat, wgt_scale=1.0, extra=None, wgt_name="w"):
    df = pl.DataFrame(
        {
            "strat": strat,
            "psu": [1, 2, 1, 2, 3, 3],
            wgt_name: [x * wgt_scale for x in [10.0, 12.0, 8.0, 9.0, 11.0, 7.0]],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    if extra:
        df = df.with_columns(**extra)
    return df


def _sample(df, wgt_name="w", stratum="strat", psu="psu"):
    return svy.Sample(df, svy.Design(stratum=stratum, psu=psu, wgt=wgt_name))


@pytest.fixture()
def two_cycles():
    return _sample(_cycle([1, 1, 2, 2, 1, 2])), _sample(_cycle([3, 3, 4, 4, 3, 4], 2.0))


# ── basics ──────────────────────────────────────────────────────────────────


def test_average_default_and_design(two_cycles):
    s1, s2 = two_cycles
    c = svy.combine_samples([s1, s2])
    assert c.design.stratum == ("wave", "strat")
    assert c.design.psu == ("psu",)
    assert c.design.wgt == "combined_wgt"
    pop1, pop2 = s1.data["w"].sum(), s2.data["w"].sum()
    assert c.data["combined_wgt"].sum() == pytest.approx((pop1 + pop2) / 2)
    assert sorted(c.data["wave"].unique().to_list()) == [1, 2]


def test_wave_metadata(two_cycles):
    c = svy.combine_samples(list(two_cycles), wave_labels=["2017-2018", "2019-2020"])
    meta = c.meta.get("wave")
    assert meta.mtype == MeasurementType.ORDINAL
    assert meta.labels == {1: "2017-2018", 2: "2019-2020"}


def test_default_wave_labels(two_cycles):
    c = svy.combine_samples(list(two_cycles))
    assert c.meta.get("wave").labels == {1: "s1", 2: "s2"}


def test_adjust_none_keeps_weight(two_cycles):
    c = svy.combine_samples(list(two_cycles), adjust="none")
    assert c.design.wgt == "w"
    assert "combined_wgt" not in c.data.columns


def test_mean_invariant_to_adjust(two_cycles):
    ca = svy.combine_samples(list(two_cycles))
    cn = svy.combine_samples(list(two_cycles), adjust="none")
    ma = ca.estimation.mean("x").to_polars()
    mn = cn.estimation.mean("x").to_polars()
    assert ma["est"][0] == pytest.approx(mn["est"][0])
    assert ma["se"][0] == pytest.approx(mn["se"][0])


# ── the statistical core: wave-qualified strata ─────────────────────────────


def test_matches_manual_interaction_declaration_with_colliding_codes():
    # DHS-style code reuse: same stratum codes in both rounds.
    df1, df2 = _cycle([1, 1, 2, 2, 1, 2]), _cycle([1, 1, 2, 2, 1, 2], 2.0)
    c = svy.combine_samples([_sample(df1), _sample(df2)])

    manual = pl.concat(
        [df1.with_columns(pl.lit(1).alias("wv")), df2.with_columns(pl.lit(2).alias("wv"))]
    ).with_columns((pl.col("w") / 2).alias("w2"))
    ms = svy.Sample(manual, svy.Design(stratum=["wv", "strat"], psu="psu", wgt="w2"))

    for fn in ("mean", "total"):
        got = getattr(c.estimation, fn)("x").to_polars()
        want = getattr(ms.estimation, fn)("x").to_polars()
        assert got["est"][0] == pytest.approx(want["est"][0])
        assert got["se"][0] == pytest.approx(want["se"][0])

    # naive stacking on colliding codes merges unrelated PSUs: different variance
    naive = svy.Sample(manual, svy.Design(stratum="strat", psu="psu", wgt="w2"))
    assert naive.estimation.mean("x").to_polars()["se"][0] != pytest.approx(
        c.estimation.mean("x").to_polars()["se"][0]
    )


# ── design-role alignment ───────────────────────────────────────────────────


def test_differing_design_names_error_with_rename_hint():
    df1 = _cycle([1, 1, 2, 2, 1, 2])
    df2 = _cycle([3, 3, 4, 4, 3, 4]).rename({"strat": "SDMVSTRA"})
    s2 = _sample(df2, stratum="SDMVSTRA")
    with pytest.raises(MethodError, match="rename_columns"):
        svy.combine_samples([_sample(df1), s2])


def test_upfront_rename_then_combine():
    df1 = _cycle([1, 1, 2, 2, 1, 2])
    df2 = _cycle([3, 3, 4, 4, 3, 4]).rename({"strat": "SDMVSTRA"})
    s2 = _sample(df2, stratum="SDMVSTRA").wrangling.rename_columns({"SDMVSTRA": "strat"})
    c = svy.combine_samples([_sample(df1), s2])
    assert c.design.stratum == ("wave", "strat")
    assert "SDMVSTRA" not in c.data.columns


def test_missing_role_errors(two_cycles):
    s1, _ = two_cycles
    no_stratum = svy.Sample(_cycle([3, 3, 4, 4, 3, 4]), svy.Design(psu="psu", wgt="w"))
    with pytest.raises(MethodError, match="role 'stratum'"):
        svy.combine_samples([s1, no_stratum])


def test_mixed_clustering_errors_by_default(two_cycles):
    s1, _ = two_cycles
    unclustered = svy.Sample(
        _cycle([3, 3, 4, 4, 3, 4], 2.0).drop("psu"), svy.Design(stratum="strat", wgt="w")
    )
    with pytest.raises(MethodError, match="on_mixed_design"):
        svy.combine_samples([s1, unclustered])


def test_mixed_clustering_opt_in(two_cycles):
    # a wave with no PSU is a complete design (element sampling); independent
    # stacking keeps each wave's variance structure self-contained
    s1, _ = two_cycles
    df2 = _cycle([3, 3, 4, 4, 3, 4], 2.0)
    unclustered = svy.Sample(df2.drop("psu"), svy.Design(stratum="strat", wgt="w"))
    with pytest.warns(UserWarning, match="element-sampled"):
        c = svy.combine_samples([s1, unclustered], on_mixed_design="warn")

    manual = _sample(
        df2.drop("psu").with_row_index("psu").with_columns(pl.col("psu").cast(pl.Int64))
    )
    ref = svy.combine_samples([s1, manual])

    got = c.estimation.mean("x").to_polars()
    want = ref.estimation.mean("x").to_polars()
    assert got["est"][0] == pytest.approx(want["est"][0])
    assert got["se"][0] == pytest.approx(want["se"][0])

    quiet = svy.combine_samples([s1, unclustered], on_mixed_design="ignore")
    assert quiet.estimation.mean("x").to_polars()["se"][0] == pytest.approx(want["se"][0])


def test_mixed_stratification_opt_in(two_cycles):
    # stratified vs not: an unstratified wave is one stratum, materialized as a
    # constant column (design columns cannot hold nulls)
    s1, _ = two_cycles
    df2 = _cycle([3, 3, 4, 4, 3, 4], 2.0)
    unstratified = svy.Sample(df2.drop("strat"), svy.Design(psu="psu", wgt="w"))
    with pytest.warns(UserWarning, match="no stratum"):
        c = svy.combine_samples([s1, unstratified], on_mixed_design="warn")

    # self-describing String encoding: real codes as strings, "__single__" fill
    strat_vals = c.data.filter(pl.col("wave") == 2)["combined_strat"].unique().to_list()
    assert strat_vals == ["__single__"]
    assert set(c.data.filter(pl.col("wave") == 1)["combined_strat"].to_list()) == {"1", "2"}

    manual = _sample(df2.with_columns(pl.lit(9, dtype=pl.Int64).alias("strat")))
    ref = svy.combine_samples([s1, manual])

    got = c.estimation.mean("x").to_polars()
    want = ref.estimation.mean("x").to_polars()
    assert got["est"][0] == pytest.approx(want["est"][0])
    assert got["se"][0] == pytest.approx(want["se"][0])


def test_mixed_design_writes_new_columns_never_clobbers(two_cycles):
    # the wave declares no psu but its DATA has a psu column — the combined
    # encoding goes into combined_psu; the user's psu column stays untouched
    s1, _ = two_cycles
    df2 = _cycle([3, 3, 4, 4, 3, 4], 2.0)
    undeclared = svy.Sample(df2, svy.Design(stratum="strat", wgt="w"))
    with pytest.warns(UserWarning, match="combined_psu"):
        c = svy.combine_samples([s1, undeclared], on_mixed_design="warn")
    assert c.design.psu == ("combined_psu",)
    assert c.data.filter(pl.col("wave") == 2)["psu"].to_list() == df2["psu"].to_list()

    # the new name is subject to the usual no-overwrite rule
    colliding = svy.Sample(
        df2.drop("psu").with_columns(pl.lit(0).alias("combined_psu")),
        svy.Design(stratum="strat", wgt="w"),
    )
    with pytest.raises(MethodError, match="already has"):
        svy.combine_samples([s1, colliding], on_mixed_design="warn")


def test_column_playing_two_roles_across_waves_errors():
    # 'psu' is the PSU in wave 1 but a stratum component in wave 2 — one name,
    # two meanings; caught before the combined_<col> names could collide
    df1, df2 = _cycle([1, 1, 2, 2, 1, 2]), _cycle([3, 3, 4, 4, 3, 4], 2.0)
    s1 = svy.Sample(df1, svy.Design(psu="psu", wgt="w"))
    s2 = svy.Sample(df2, svy.Design(stratum=("strat", "psu"), wgt="w"))
    with pytest.raises(MethodError, match="different design roles"):
        svy.combine_samples([s1, s2], on_mixed_design="warn")


def test_missing_psu_shared_mode_errors():
    df1, df2 = _cycle([1, 1, 2, 2, 1, 2]), _cycle([1, 1, 2, 2, 1, 2], 2.0)
    unclustered = svy.Sample(df2.drop("psu"), svy.Design(stratum="strat", wgt="w"))
    with pytest.raises(MethodError, match="role 'psu'"):
        svy.combine_samples([_sample(df1), unclustered], units="shared")


def test_knob_does_not_relax_shared_mode():
    df1, df2 = _cycle([1, 1, 2, 2, 1, 2]), _cycle([1, 1, 2, 2, 1, 2], 2.0)
    unclustered = svy.Sample(df2.drop("psu"), svy.Design(stratum="strat", wgt="w"))
    with pytest.raises(MethodError, match="role 'psu'"):
        svy.combine_samples([_sample(df1), unclustered], units="shared", on_mixed_design="warn")


def test_srs_wave_missing_both_roles(two_cycles):
    # one wave is a plain SRS: no strata, no PSU — both translations at once
    s1, _ = two_cycles
    df2 = _cycle([3, 3, 4, 4, 3, 4], 2.0)
    srs = svy.Sample(df2.drop("strat", "psu"), svy.Design(wgt="w"))
    with pytest.warns(UserWarning, match="no stratum.*no psu"):
        c = svy.combine_samples([s1, srs], on_mixed_design="warn")

    manual = _sample(
        df2.drop("strat", "psu")
        .with_columns(pl.lit(9, dtype=pl.Int64).alias("strat"))
        .with_row_index("psu")
        .with_columns(pl.col("psu").cast(pl.Int64))
    )
    ref = svy.combine_samples([s1, manual])
    got = c.estimation.mean("x").to_polars()
    want = ref.estimation.mean("x").to_polars()
    assert got["est"][0] == pytest.approx(want["est"][0])
    assert got["se"][0] == pytest.approx(want["se"][0])


def test_first_sample_is_the_lacking_wave():
    # canonical names and the donor frame come from a LATER sample
    df1, df2 = _cycle([1, 1, 2, 2, 1, 2]), _cycle([3, 3, 4, 4, 3, 4], 2.0)
    unclustered = svy.Sample(df1.drop("psu"), svy.Design(stratum="strat", wgt="w"))
    with pytest.warns(UserWarning, match=r"sample\(s\) \[1\]"):
        c = svy.combine_samples([unclustered, _sample(df2)], on_mixed_design="warn")
    assert c.design.psu == ("combined_psu",)

    manual = _sample(
        df1.drop("psu").with_row_index("psu").with_columns(pl.col("psu").cast(pl.Int64))
    )
    ref = svy.combine_samples([manual, _sample(df2)])
    assert c.estimation.mean("x").to_polars()["se"][0] == pytest.approx(
        ref.estimation.mean("x").to_polars()["se"][0]
    )


def test_three_waves_lacking_different_roles(two_cycles):
    s1, _ = two_cycles
    unclustered = svy.Sample(
        _cycle([3, 3, 4, 4, 3, 4], 2.0).drop("psu"), svy.Design(stratum="strat", wgt="w")
    )
    unstratified = svy.Sample(
        _cycle([5, 5, 6, 6, 5, 6], 3.0).drop("strat"), svy.Design(psu="psu", wgt="w")
    )
    with pytest.warns(UserWarning, match=r"no stratum.*no psu"):
        c = svy.combine_samples([s1, unclustered, unstratified], on_mixed_design="warn")
    assert sorted(c.data["wave"].unique().to_list()) == [1, 2, 3]
    assert c.estimation.mean("x").to_polars()["se"][0] > 0


def test_string_psu_codes_get_string_element_ids(two_cycles):
    df1 = _cycle([1, 1, 2, 2, 1, 2]).with_columns(pl.col("psu").cast(pl.String))
    df2 = _cycle([3, 3, 4, 4, 3, 4], 2.0)
    unclustered = svy.Sample(df2.drop("psu"), svy.Design(stratum="strat", wgt="w"))
    with pytest.warns(UserWarning, match="element-sampled"):
        c = svy.combine_samples([_sample(df1), unclustered], on_mixed_design="warn")
    assert c.data.schema["combined_psu"] == pl.String


def test_knob_does_not_relax_other_roles(two_cycles):
    s1, _ = two_cycles
    df2 = _cycle([3, 3, 4, 4, 3, 4], 2.0).with_columns(pl.lit(1).alias("ssu"))
    with_ssu = svy.Sample(df2, svy.Design(stratum="strat", psu="psu", ssu="ssu", wgt="w"))
    with pytest.raises(MethodError, match="role 'ssu'"):
        svy.combine_samples([s1, with_ssu], on_mixed_design="warn")

    no_wgt = svy.Sample(_cycle([3, 3, 4, 4, 3, 4]), svy.Design(stratum="strat", psu="psu"))
    with pytest.raises(MethodError, match="role 'wgt'"):
        svy.combine_samples([s1, no_wgt], on_mixed_design="warn")


def test_name_mismatch_among_declaring_waves_still_errors(two_cycles):
    s1, _ = two_cycles
    unclustered = svy.Sample(
        _cycle([3, 3, 4, 4, 3, 4], 2.0).drop("psu"), svy.Design(stratum="strat", wgt="w")
    )
    renamed = _sample(_cycle([5, 5, 6, 6, 5, 6], 3.0).rename({"psu": "cluster"}), psu="cluster")
    with pytest.raises(MethodError, match="rename_columns"):
        svy.combine_samples([s1, unclustered, renamed], on_mixed_design="warn")


def test_multicolumn_stratum_with_lacking_wave():
    df1 = _cycle([1, 1, 2, 2, 1, 2]).with_columns(pl.lit(1, dtype=pl.Int64).alias("region"))
    s1 = svy.Sample(df1, svy.Design(stratum=["region", "strat"], psu="psu", wgt="w"))
    df2 = _cycle([3, 3, 4, 4, 3, 4], 2.0).drop("strat")
    unstratified = svy.Sample(df2, svy.Design(psu="psu", wgt="w"))
    with pytest.warns(UserWarning, match="no stratum"):
        c = svy.combine_samples([s1, unstratified], on_mixed_design="warn")
    assert c.design.stratum == ("wave", "combined_region", "combined_strat")
    assert c.data.filter(pl.col("wave") == 2)["combined_region"].null_count() == 0
    assert c.data.filter(pl.col("wave") == 2)["combined_strat"].null_count() == 0


def test_all_waves_unclustered_no_warning():
    df1, df2 = _cycle([1, 1, 2, 2, 1, 2]), _cycle([3, 3, 4, 4, 3, 4], 2.0)
    a = svy.Sample(df1.drop("psu"), svy.Design(stratum="strat", wgt="w"))
    b = svy.Sample(df2.drop("psu"), svy.Design(stratum="strat", wgt="w"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c = svy.combine_samples([a, b])  # no mixing, no knob needed
    assert c.design.psu is None
    assert c.design.stratum == ("wave", "strat")


def test_no_mismatch_no_false_warning(two_cycles):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        svy.combine_samples(list(two_cycles), on_mixed_design="warn")


def test_invalid_on_mixed_design_value(two_cycles):
    with pytest.raises(MethodError, match="on_mixed_design"):
        svy.combine_samples(list(two_cycles), on_mixed_design="warning")


# ── wave-column reuse ───────────────────────────────────────────────────────


def test_reuses_existing_wave_column():
    s1 = _sample(_cycle([1, 1, 2, 2, 1, 2], extra={"cycle": pl.lit(5)}))
    s2 = _sample(_cycle([3, 3, 4, 4, 3, 4], extra={"cycle": pl.lit(6)}))
    c = svy.combine_samples([s1, s2], wave_name="cycle")
    assert sorted(c.data["cycle"].unique().to_list()) == [5, 6]
    assert c.design.stratum == ("cycle", "strat")
    assert c.meta.get("cycle").mtype == MeasurementType.ORDINAL


def test_reused_wave_column_must_increase_with_caller_order():
    s1 = _sample(_cycle([1, 1, 2, 2, 1, 2], extra={"cycle": pl.lit(6)}))
    s2 = _sample(_cycle([3, 3, 4, 4, 3, 4], extra={"cycle": pl.lit(5)}))
    with pytest.raises(MethodError, match="not increasing"):
        svy.combine_samples([s1, s2], wave_name="cycle")


def test_wave_column_in_some_inputs_errors(two_cycles):
    _, s2 = two_cycles
    s1 = _sample(_cycle([1, 1, 2, 2, 1, 2], extra={"cycle": pl.lit(5)}))
    with pytest.raises(MethodError, match="some inputs"):
        svy.combine_samples([s1, s2], wave_name="cycle")


def test_nonconstant_wave_column_errors():
    df1 = _cycle([1, 1, 2, 2, 1, 2]).with_columns(pl.Series("cycle", [5, 5, 5, 6, 5, 5]))
    s2 = _sample(_cycle([3, 3, 4, 4, 3, 4], extra={"cycle": pl.lit(6)}))
    with pytest.raises(MethodError, match="one non-null value"):
        svy.combine_samples([_sample(df1), s2], wave_name="cycle")


def test_numeric_labels_out_of_order_warn(two_cycles):
    with pytest.warns(UserWarning, match="not increasing"):
        svy.combine_samples(list(two_cycles), wave_labels=["2024", "2023"])


def test_wave_labels_length_mismatch(two_cycles):
    with pytest.raises(MethodError, match="wave_labels"):
        svy.combine_samples(list(two_cycles), wave_labels=["only-one"])


# ── column harmonization ────────────────────────────────────────────────────


def test_diagonal_concat_warns_on_null_filled(two_cycles):
    s1, _ = two_cycles
    s2 = _sample(_cycle([3, 3, 4, 4, 3, 4], extra={"newvar": pl.lit(9)}))
    with pytest.warns(UserWarning, match="null-filled"):
        c = svy.combine_samples([s1, s2])
    assert c.data.filter(pl.col("wave") == 1)["newvar"].null_count() == 6


def test_dtype_conflict_errors(two_cycles):
    s1, _ = two_cycles
    s2 = _sample(_cycle([3, 3, 4, 4, 3, 4], extra={"x": pl.col("x").cast(pl.Int64)}))
    with pytest.raises(MethodError, match="conflicting dtypes"):
        svy.combine_samples([s1, s2])


def test_value_label_conflict_drops_and_warns(two_cycles):
    s1, s2 = two_cycles
    s1.meta.set_value_labels("x", {1: "low", 2: "high"})
    s2.meta.set_value_labels("x", {1: "bottom", 2: "top"})
    with pytest.warns(UserWarning, match="Value labels conflict"):
        c = svy.combine_samples([s1, s2])
    meta = c.meta.get("x")
    assert meta is None or not meta.labels


def test_identical_value_labels_merge(two_cycles):
    s1, s2 = two_cycles
    s1.meta.set_value_labels("x", {1: "low", 2: "high"})
    s2.meta.set_value_labels("x", {1: "low", 2: "high"})
    s1.meta.set_label("x", "The X")
    c = svy.combine_samples([s1, s2])
    assert c.meta.get("x").labels == {1: "low", 2: "high"}
    assert c.meta.get("x").label == "The X"


# ── shared units (panel) mode ───────────────────────────────────────────────


def test_shared_mode_identical_units():
    df1, df2 = _cycle([1, 1, 2, 2, 1, 2]), _cycle([1, 1, 2, 2, 1, 2], 2.0)
    c = svy.combine_samples([_sample(df1), _sample(df2)], units="shared")
    assert c.design.stratum == ("strat",)  # NOT wave-qualified
    assert c.design.wgt == "w"  # adjust resolves to "none"


def test_shared_mode_differing_units_errors(two_cycles):
    with pytest.raises(MethodError, match="identical design units"):
        svy.combine_samples(list(two_cycles), units="shared")


def test_shared_mode_rejects_average(two_cycles):
    with pytest.raises(MethodError, match="half a person"):
        svy.combine_samples(list(two_cycles), units="shared", adjust="average")


# ── guards ──────────────────────────────────────────────────────────────────


def test_needs_two_samples(two_cycles):
    with pytest.raises(MethodError, match="at least 2"):
        svy.combine_samples([two_cycles[0]])


def test_wgt_name_collision(two_cycles):
    s1, _ = two_cycles
    s2 = _sample(_cycle([3, 3, 4, 4, 3, 4], extra={"combined_wgt": pl.lit(1.0)}))
    with pytest.raises(MethodError, match="combined_wgt"):
        svy.combine_samples([s1, s2])


def test_replicate_designs_rejected(two_cycles):
    s1, s2 = two_cycles
    s1r = s1.weighting.create_jk_wgts()
    with pytest.raises(MethodError, match="replicate"):
        svy.combine_samples([s1r, s2])


# ── normalize(factor=) companion ────────────────────────────────────────────


def test_normalize_factor_scales_weights(two_cycles):
    s1, _ = two_cycles
    out = s1.weighting.normalize(factor=4 / 6)
    assert out.data["norm_wgt"].sum() == pytest.approx(s1.data["w"].sum() * 4 / 6)
    assert out.design.wgt == "norm_wgt"
    assert out.design.wgt_adjustment.kind == "normalization"


def test_normalize_factor_exclusive(two_cycles):
    s1, _ = two_cycles
    with pytest.raises(MethodError, match="factor"):
        s1.weighting.normalize(factor=0.5, cells="strat")
    with pytest.raises(MethodError):
        s1.weighting.normalize(factor=0.0)


def test_nchs_multispan_recipe(two_cycles):
    s1, s2 = two_cycles
    a = s1.weighting.normalize(factor=4 / 6)
    b = s2.weighting.normalize(factor=2 / 6)
    c = svy.combine_samples([a, b], adjust="none")
    want = s1.data["w"].sum() * 4 / 6 + s2.data["w"].sum() * 2 / 6
    assert c.data["norm_wgt"].sum() == pytest.approx(want)
