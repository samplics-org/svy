"""wrangling.join — bring variables from another sample onto this one's records."""

import polars as pl
import pytest

import svy

from svy.errors import DimensionError, MethodError


def _persons():
    df = pl.DataFrame(
        {
            "hh": [3, 1, 1, 2, 2, 2, 4],
            "line": [1, 1, 2, 1, 2, 3, 1],
            "age": [40, 35, 8, 60, 58, 20, 33],
            "stratum": ["a", "a", "a", "b", "b", "b", "b"],
            "psu": [1, 1, 5, 2, 2, 6, 3],
            "wgt": [2.0, 1.0, 1.0, 3.0, 3.0, 3.0, 4.0],
        }
    )
    return svy.Sample(df, svy.Design(stratum="stratum", psu="psu", wgt="wgt"))


def _households():
    df = pl.DataFrame(
        {
            "hh_id": [1, 2, 3],
            "region": [10, 20, 10],
            "rooms": [2, 5, 3],
            "wgt": [1.5, 2.5, 3.5],
        }
    )
    s = svy.Sample(df, svy.Design(wgt="wgt"))
    s.meta.set_label("region", "Region of residence")
    s.meta.set_value_labels("region", {10: "North", 20: "South"})
    return s


def test_brings_columns_onto_every_record_in_order():
    p = _persons()
    out = p.wrangling.join(_households(), on={"hh": "hh_id"}, cols=["region", "rooms"])
    assert out.data.height == p.data.height
    assert out.data["hh"].to_list() == p.data["hh"].to_list()
    assert out.data["region"].to_list() == [10, 10, 10, 20, 20, 20, None]
    assert out.data["rooms"].to_list() == [3, 2, 2, 5, 5, 5, None]
    assert "hh_id" not in out.data.columns


def test_design_is_untouched_and_the_other_weight_is_a_plain_column():
    p = _persons()
    out = p.wrangling.join(_households(), on={"hh": "hh_id"}, cols=["wgt"], into={"wgt": "hh_wgt"})
    assert out.design == p.design
    assert out.design.wgt == "wgt"
    assert out.data["wgt"].to_list() == p.data["wgt"].to_list()
    assert out.data["hh_wgt"].to_list()[:3] == [3.5, 1.5, 1.5]


def test_estimates_on_existing_columns_do_not_move():
    p = _persons()
    out = p.wrangling.join(_households(), on={"hh": "hh_id"}, cols=["rooms"])
    before = p.estimation.mean("age")
    after = out.estimation.mean("age")
    assert after.estimates[0].est == pytest.approx(before.estimates[0].est)
    assert after.estimates[0].se == pytest.approx(before.estimates[0].se)


def test_default_brings_every_non_key_column_and_refuses_a_clash():
    with pytest.raises(MethodError) as exc:
        _persons().wrangling.join(_households(), on={"hh": "hh_id"})
    assert exc.value.code == "JOIN_COLUMN_EXISTS"
    assert "wgt" in str(exc.value)


@pytest.mark.parametrize("name", ["wgt", "stratum", "age"])
def test_never_overwrites_a_column(name):
    with pytest.raises(MethodError) as exc:
        _persons().wrangling.join(
            _households(), on={"hh": "hh_id"}, cols=["rooms"], into={"rooms": name}
        )
    assert exc.value.code == "JOIN_COLUMN_EXISTS"


def test_two_columns_cannot_land_on_one_name():
    with pytest.raises(MethodError) as exc:
        _persons().wrangling.join(
            _households(), on={"hh": "hh_id"}, into={"rooms": "x", "region": "x", "wgt": "w2"}
        )
    assert exc.value.code == "JOIN_COLUMN_EXISTS"


def test_repeated_key_on_the_other_side_is_refused_with_examples():
    hh = pl.DataFrame({"hh": [1, 1, 2], "region": [10, 11, 20]})
    with pytest.raises(MethodError) as exc:
        _persons().wrangling.join(hh, on="hh", cols=["region"])
    assert exc.value.code == "JOIN_KEY_NOT_UNIQUE"
    assert "(1,)" in str(exc.value)


def test_one_to_one_checks_this_side_too():
    p = _persons()
    with pytest.raises(MethodError) as exc:
        p.wrangling.join(_households(), on={"hh": "hh_id"}, cols=["rooms"], validate="1:1")
    assert exc.value.code == "JOIN_KEY_NOT_UNIQUE"
    heads = p.wrangling.filter_records(svy.col("line") == 1)
    out = heads.wrangling.join(_households(), on={"hh": "hh_id"}, cols=["rooms"], validate="1:1")
    assert out.data["rooms"].to_list() == [3, 2, 5, None]


def test_multi_column_key():
    roster = pl.DataFrame({"hh": [1, 1, 2], "line": [1, 2, 1], "educ": [3, 1, 2]})
    out = _persons().wrangling.join(roster, on=["hh", "line"], on_unmatched="ignore")
    assert out.data["educ"].to_list() == [None, 3, 1, 2, None, None, None]


def test_unmatched_warns_by_default_with_the_count():
    out = _persons().wrangling.join(_households(), on={"hh": "hh_id"}, cols=["rooms"])
    warns = [w for w in out.warnings if w.code == "JOIN_UNMATCHED"]
    assert len(warns) == 1
    assert warns[0].extra == {"n_unmatched": 1, "n_records": 7}


def test_unmatched_keep_error_and_indicator():
    p, hh = _persons(), _households()
    out = p.wrangling.join(
        hh, on={"hh": "hh_id"}, cols=["rooms"], on_unmatched="ignore", indicator="in_hh"
    )
    assert not [w for w in out.warnings if w.code == "JOIN_UNMATCHED"]
    assert out.data["in_hh"].to_list() == [True] * 6 + [False]
    with pytest.raises(MethodError) as exc:
        p.wrangling.join(hh, on={"hh": "hh_id"}, cols=["rooms"], on_unmatched="error")
    assert exc.value.code == "JOIN_UNMATCHED"


def test_null_keys_never_match_and_do_not_count_as_repeats():
    p = _persons().wrangling.mutate(
        {"hh": pl.when(pl.col("line") == 3).then(None).otherwise(pl.col("hh"))}
    )
    hh = pl.DataFrame({"hh": [1, 2, None, None], "rooms": [2, 5, 9, 9]})
    out = p.wrangling.join(hh, on="hh", on_unmatched="ignore")
    assert out.data["rooms"].to_list() == [None, 2, 2, 5, 5, None, None]


def test_integer_widths_join_and_text_against_numbers_is_refused():
    hh = pl.DataFrame({"hh": pl.Series([1, 2], dtype=pl.Int16), "rooms": [2, 5]})
    out = _persons().wrangling.join(hh, on="hh", on_unmatched="ignore")
    assert out.data["rooms"].to_list()[:4] == [None, 2, 2, 5]
    as_text = pl.DataFrame({"hh": ["1", "2"], "rooms": [2, 5]})
    with pytest.raises(MethodError) as exc:
        _persons().wrangling.join(as_text, on="hh")
    assert exc.value.code == "JOIN_KEY_TYPE_MISMATCH"


def test_text_joins_categorical():
    p = _persons()
    strata = pl.DataFrame(
        {"stratum": pl.Series(["a", "b"], dtype=pl.Categorical), "frame_n": [100, 200]}
    )
    out = p.wrangling.join(strata, on="stratum")
    assert out.data["frame_n"].to_list() == [100, 100, 100, 200, 200, 200, 200]
    assert out.data.schema["stratum"] == pl.String


def test_a_brought_column_may_share_the_other_key_name():
    hh = pl.DataFrame({"hh_id": [1, 2, 3], "code": [7, 8, 9]})
    out = _persons().wrangling.join(
        hh, on={"hh": "hh_id"}, into={"code": "hh_id"}, on_unmatched="ignore"
    )
    assert out.data["hh_id"].to_list()[:4] == [9, 7, 7, 8]


def test_labels_carry_over_from_a_sample_and_follow_a_rename():
    out = _persons().wrangling.join(
        _households(),
        on={"hh": "hh_id"},
        cols=["region"],
        into={"region": "hh_region"},
        on_unmatched="ignore",
    )
    meta = out.meta.get("hh_region")
    assert meta.label == "Region of residence"
    assert meta.labels == {10: "North", 20: "South"}


def test_lazy_frame_and_missing_columns():
    lazy = _households().data.lazy()
    out = _persons().wrangling.join(
        lazy, on={"hh": "hh_id"}, cols=["rooms"], on_unmatched="ignore"
    )
    assert out.data["rooms"].to_list()[0] == 3
    with pytest.raises(DimensionError):
        _persons().wrangling.join(lazy, on={"hh": "nope"}, cols=["rooms"])
    with pytest.raises(DimensionError):
        _persons().wrangling.join(lazy, on={"hh": "hh_id"}, cols=["nope"])
    with pytest.raises(MethodError) as exc:
        _persons().wrangling.join(lazy, on={"hh": "hh_id"}, cols=["hh_id"])
    assert exc.value.code == "JOIN_KEY_IN_COLS"


def test_copy_on_write_and_inplace():
    p = _persons()
    out = p.wrangling.join(
        _households(), on={"hh": "hh_id"}, cols=["rooms"], on_unmatched="ignore"
    )
    assert "rooms" not in p.data.columns and "rooms" in out.data.columns
    p.wrangling.join(
        _households(), on={"hh": "hh_id"}, cols=["rooms"], on_unmatched="ignore", inplace=True
    )
    assert "rooms" in p.data.columns


def test_svy_columns_stay_out_by_default_and_come_in_only_renamed():
    sel = pl.DataFrame(
        {"hh_id": [1, 2, 3], "svy_sample_weight": [5.0, 6.0, 7.0], "rooms": [2, 5, 3]}
    )
    out = _persons().wrangling.join(sel, on={"hh": "hh_id"}, on_unmatched="ignore")
    assert "svy_sample_weight" not in out.data.columns
    out = _persons().wrangling.join(
        sel,
        on={"hh": "hh_id"},
        cols=["svy_sample_weight"],
        into={"svy_sample_weight": "base_wgt"},
        on_unmatched="ignore",
    )
    assert out.data["base_wgt"].to_list()[:2] == [7.0, 5.0]
    with pytest.raises(MethodError):
        _persons().wrangling.join(sel, on={"hh": "hh_id"}, cols=["svy_sample_weight"])


def test_a_column_that_reads_as_a_replicate_weight_is_refused():
    p = _persons()
    reps = {f"rw{i}": [1.0] * 7 for i in range(1, 3)}
    s = svy.Sample(
        p.data.with_columns(**{k: pl.Series(v) for k, v in reps.items()}),
        svy.Design(wgt="wgt", rep_wgts=svy.RepWeights(method="bootstrap", prefix="rw", n_reps=2)),
    )
    hh = pl.DataFrame({"hh": [1, 2, 3], "rw3": [1.0, 1.0, 1.0]})
    with pytest.raises(MethodError) as exc:
        s.wrangling.join(hh, on="hh")
    assert "replicate" in str(exc.value)


def test_a_lazy_other_is_read_for_the_requested_columns_only():
    wide = pl.LazyFrame({"hh": [1, 2, 3], "rooms": [2, 5, 3], "big": ["x", "y", "z"]})
    out = _persons().wrangling.join(wide, on="hh", cols=["rooms"], on_unmatched="ignore")
    assert "big" not in out.data.columns


# --- design variables -------------------------------------------------------


def _same_estimate(a, b):
    assert a.estimates[0].est == pytest.approx(b.estimates[0].est)
    assert a.estimates[0].se == pytest.approx(b.estimates[0].se)


def test_joining_on_a_design_variable_leaves_it_and_its_internal_columns_alone():
    p = _persons()
    frame = pl.DataFrame({"psu": [1, 2, 3, 5, 6], "psu_pop": [10, 20, 30, 50, 60]})
    out = p.wrangling.join(frame, on="psu")
    assert out.design == p.design
    assert out._internal_design == p._internal_design
    for c in p._data.columns:
        assert out._data[c].equals(p._data[c])
    assert out.data["psu_pop"].to_list() == [10, 10, 50, 20, 20, 60, 30]
    _same_estimate(p.estimation.mean("age"), out.estimation.mean("age"))


def test_a_cast_key_does_not_change_the_design_variable_type():
    p = _persons().wrangling.cast("stratum", pl.Categorical)
    frame = pl.DataFrame({"stratum": ["a", "b"], "n_frame": [100, 200]})
    out = p.wrangling.join(frame, on="stratum")
    assert out.data.schema["stratum"] == p.data.schema["stratum"]
    assert out._internal_design == p._internal_design
    _same_estimate(p.estimation.mean("age"), out.estimation.mean("age"))


def test_a_joined_variable_can_become_a_design_variable():
    p = _persons()
    hh = pl.DataFrame({"hh": [1, 2, 3, 4], "region": [10, 20, 10, 20]})
    out = p.wrangling.join(hh, on="hh")
    redesigned = out.update_design(stratum="region")
    assert redesigned.design.stratum == "region"
    assert redesigned.estimation.mean("age").estimates[0].se > 0


def test_singleton_handling_survives_a_join():
    df = _persons().data.with_columns(pl.Series("psu", [1, 1, 1, 2, 2, 6, 3]))
    s = svy.Sample(df, svy.Design(stratum="stratum", psu="psu", wgt="wgt"))
    handled = s.singleton.certainty()
    out = handled.wrangling.join(_households(), on={"hh": "hh_id"}, cols=["rooms"])
    assert out.design == handled.design
    assert out._singletons == handled._singletons
    _same_estimate(handled.estimation.mean("age"), out.estimation.mean("age"))


def test_replicate_design_and_its_estimates_are_unchanged():
    df = _persons().data.with_columns(
        **{f"bs{i}": pl.Series([1.0 + 0.1 * ((i + j) % 3) for j in range(7)]) for i in range(1, 5)}
    )
    s = svy.Sample(
        df,
        svy.Design(wgt="wgt", rep_wgts=svy.RepWeights(method="bootstrap", prefix="bs", n_reps=4)),
    )
    out = s.wrangling.join(_households(), on={"hh": "hh_id"}, cols=["rooms"])
    assert out.design == s.design
    _same_estimate(
        s.estimation.mean("age", method="replication"),
        out.estimation.mean("age", method="replication"),
    )


def test_selection_variables_on_this_sample_are_kept():
    df = _persons().data.with_columns(
        pl.Series("svy_prob_selection", [0.5] * 7), pl.Series("svy_sample_weight", [2.0] * 7)
    )
    s = svy.Sample(df, svy.Design(stratum="stratum", psu="psu", wgt="svy_sample_weight"))
    out = s.wrangling.join(_households(), on={"hh": "hh_id"}, cols=["rooms"])
    assert out.data["svy_sample_weight"].to_list() == [2.0] * 7
    assert out.design.wgt == "svy_sample_weight"


# --- records missing on one side --------------------------------------------


def test_records_only_on_the_other_side_are_never_added():
    hh = pl.DataFrame({"hh": [1, 2, 3, 4, 5, 6], "rooms": [2, 5, 3, 1, 9, 9]})
    out = _persons().wrangling.join(hh, on="hh")
    assert out.data.height == 7
    assert 9 not in out.data["rooms"].to_list()


def test_order_follows_this_sample_whatever_the_other_order():
    p = _persons()
    hh = pl.DataFrame({"hh": [4, 3, 2, 1], "rooms": [1, 3, 5, 2]})
    out = p.wrangling.join(hh, on="hh")
    assert out.data["hh"].to_list() == p.data["hh"].to_list()
    assert out.data["svy_row_index"].to_list() == p.data["svy_row_index"].to_list()
    assert out.data["rooms"].to_list() == [3, 2, 2, 5, 5, 5, 1]


def test_nothing_matches():
    hh = pl.DataFrame({"hh": [90, 91], "rooms": [1, 2]})
    out = _persons().wrangling.join(hh, on="hh", indicator="m")
    assert out.data["rooms"].null_count() == 7
    assert not any(out.data["m"].to_list())
    warns = [w for w in out.warnings if w.code == "JOIN_UNMATCHED"]
    assert warns[0].extra == {"n_unmatched": 7, "n_records": 7}


def test_empty_other_keeps_every_record_with_the_columns_typed():
    hh = pl.DataFrame(
        {"hh": pl.Series([], dtype=pl.Int64), "rooms": pl.Series([], dtype=pl.Int32)}
    )
    out = _persons().wrangling.join(hh, on="hh", on_unmatched="ignore")
    assert out.data.height == 7
    assert out.data.schema["rooms"] == pl.Int32


def test_blanks_brought_in_are_told_apart_from_no_match_by_the_indicator():
    hh = pl.DataFrame({"hh": [1, 2, 3], "rooms": [None, 5, 3]})
    out = _persons().wrangling.join(hh, on="hh", indicator="m", on_unmatched="ignore")
    rows = list(
        zip(out.data["hh"].to_list(), out.data["rooms"].to_list(), out.data["m"].to_list())
    )
    assert (1, None, True) in rows
    assert (4, None, False) in rows


def test_unmatched_records_fall_out_of_a_domain_not_out_of_the_sample():
    out = _persons().wrangling.join(_households(), on={"hh": "hh_id"}, cols=["region"])
    by = out.estimation.mean("age", by="region", drop_nulls=True)
    # region 10: ages 35, 8 (w 1) and 40 (w 2); region 20: 60, 58, 20 (w 3).
    # Household 4 found no match and sits in neither domain.
    assert sorted((e.by_level, e.est) for e in by.estimates) == [(("10",), 30.75), (("20",), 46.0)]
    _same_estimate(_persons().estimation.mean("age"), out.estimation.mean("age"))


# --- nothing leaks, nothing half-done -----------------------------------------


def test_a_refused_join_leaves_the_sample_untouched_even_inplace():
    p = _persons()
    before = p.data.clone()
    with pytest.raises(MethodError):
        p.wrangling.join(
            _households(), on={"hh": "hh_id"}, cols=["rooms"], on_unmatched="error", inplace=True
        )
    assert p.data.equals(before)
    assert not [w for w in p.warnings if w.code == "JOIN_UNMATCHED"]


def test_the_warning_and_labels_stay_on_the_result():
    p = _persons()
    p.meta.set_label("age", "Age in years")
    out = p.wrangling.join(_households(), on={"hh": "hh_id"}, cols=["region"])
    assert not [w for w in p.warnings if w.code == "JOIN_UNMATCHED"]
    assert "region" not in p.meta
    assert out.meta.get("age").label == "Age in years"


def test_float_keys_of_the_same_type_join_exactly():
    p = _persons().wrangling.cast("hh", pl.Float64)
    hh = pl.DataFrame({"hh": [1.0, 2.0, 3.0], "rooms": [2, 5, 3]})
    out = p.wrangling.join(hh, on="hh", on_unmatched="ignore")
    assert out.data["rooms"].to_list()[:4] == [3, 2, 2, 5]


def test_order_is_kept_on_a_file_large_enough_for_a_parallel_join():
    n = 200_000
    persons = pl.DataFrame(
        {
            "hh": pl.int_range(n, eager=True).shuffle(seed=1) // 3,
            "wgt": pl.Series([1.0] * n),
        }
    )
    s = svy.Sample(persons, svy.Design(wgt="wgt"))
    hh = pl.DataFrame({"hh": pl.int_range(n // 3 + 1, eager=True).shuffle(seed=2)}).with_columns(
        rooms=pl.col("hh") * 2
    )
    out = s.wrangling.join(hh, on="hh", on_unmatched="ignore")
    assert out.data["hh"].equals(s.data["hh"])
    assert (out.data["rooms"] == out.data["hh"] * 2).all()


def test_exactly_the_requested_columns_come_in_and_nothing_else():
    p = _persons()
    out = p.wrangling.join(
        _households(),
        on={"hh": "hh_id"},
        cols=["rooms"],
        into={"rooms": "hh_rooms"},
        indicator="m",
    )
    assert out._data.columns == [*p._data.columns, "hh_rooms", "m"]
    assert out.meta.get("region") is None
    out = p.wrangling.join(
        _households().data.drop("wgt"), on={"hh": "hh_id"}, on_unmatched="ignore"
    )
    assert out._data.columns == [*p._data.columns, "region", "rooms"]


# --- names: cols picks, into names, suffix settles the rest ------------


def test_bring_everything_and_resolve_only_the_clash():
    p = _persons()
    out = p.wrangling.join(_households(), on={"hh": "hh_id"}, into={"wgt": "hh_wgt"})
    assert out._data.columns == [*p._data.columns, "region", "rooms", "hh_wgt"]


def test_selected_columns_with_some_renamed_for_intent():
    out = _persons().wrangling.join(
        _households(),
        on={"hh": "hh_id"},
        cols=["region", "rooms"],
        into={"rooms": "hh_rooms"},
    )
    assert "region" in out.data.columns and "hh_rooms" in out.data.columns
    assert "rooms" not in out.data.columns and "hh_wgt" not in out.data.columns


def test_suffix_settles_only_the_clashes():
    p = _persons()
    out = p.wrangling.join(_households(), on={"hh": "hh_id"}, suffix="_hh")
    assert out._data.columns == [*p._data.columns, "region", "rooms", "wgt_hh"]
    assert out.design.wgt == "wgt"


def test_an_explicit_name_wins_over_the_suffix_and_is_still_checked():
    out = _persons().wrangling.join(
        _households(), on={"hh": "hh_id"}, into={"wgt": "base_wgt"}, suffix="_hh"
    )
    assert "base_wgt" in out.data.columns and "wgt_hh" not in out.data.columns
    with pytest.raises(MethodError) as exc:
        _persons().wrangling.join(
            _households(), on={"hh": "hh_id"}, into={"wgt": "age"}, suffix="_hh"
        )
    assert exc.value.code == "JOIN_COLUMN_EXISTS"


def test_a_suffixed_name_that_still_clashes_is_refused():
    p = _persons().wrangling.mutate({"wgt_hh": 1.0})
    with pytest.raises(MethodError) as exc:
        p.wrangling.join(_households(), on={"hh": "hh_id"}, suffix="_hh")
    assert exc.value.code == "JOIN_COLUMN_EXISTS"
    assert "wgt_hh" in str(exc.value)


def test_naming_a_column_that_is_not_brought_is_refused():
    with pytest.raises(MethodError) as exc:
        _persons().wrangling.join(
            _households(), on={"hh": "hh_id"}, cols=["rooms"], into={"wgt": "hh_wgt"}
        )
    assert exc.value.code == "JOIN_INTO_NOT_BROUGHT"


def test_into_as_one_name_needs_one_column():
    out = _persons().wrangling.join(
        _households(), on={"hh": "hh_id"}, cols="rooms", into="hh_rooms", on_unmatched="ignore"
    )
    assert "hh_rooms" in out.data.columns
    with pytest.raises(MethodError):
        _persons().wrangling.join(
            _households(), on={"hh": "hh_id"}, cols=["rooms", "region"], into="x"
        )


def test_into_never_renames_this_sample():
    p = _persons()
    out = p.wrangling.join(_households(), on={"hh": "hh_id"}, into={"wgt": "hh_wgt"})
    assert out.data["wgt"].to_list() == p.data["wgt"].to_list()
    assert out.design.wgt == "wgt"
    with pytest.raises(MethodError) as exc:
        p.wrangling.join(_households(), on={"hh": "hh_id"}, cols=["rooms"], into={"age": "x"})
    assert exc.value.code == "JOIN_INTO_NOT_BROUGHT"


# --- decimal identifiers, as SPSS and Stata store them ------------------------


def test_whole_decimal_key_on_the_other_side_joins_an_integer_key():
    hh = pl.DataFrame({"hh": [1.0, 2.0, 3.0], "rooms": [2, 5, 3]})
    p = _persons()
    out = p.wrangling.join(hh, on="hh", on_unmatched="ignore")
    assert out.data["rooms"].to_list() == [3, 2, 2, 5, 5, 5, None]
    assert out.data.schema["hh"] == p.data.schema["hh"]


def test_whole_decimal_key_on_this_side_joins_and_keeps_its_type():
    p = _persons().wrangling.cast("hh", pl.Float64)
    hh = pl.DataFrame({"hh": pl.Series([1, 2, 3], dtype=pl.Int32), "rooms": [2, 5, 3]})
    out = p.wrangling.join(hh, on="hh", on_unmatched="ignore")
    assert out.data["rooms"].to_list() == [3, 2, 2, 5, 5, 5, None]
    assert out.data.schema["hh"] == pl.Float64


def test_a_decimal_key_with_fractions_is_refused_with_examples():
    hh = pl.DataFrame({"hh": [1.0, 2.5, 3.0], "rooms": [2, 5, 3]})
    with pytest.raises(MethodError) as exc:
        _persons().wrangling.join(hh, on="hh")
    assert exc.value.code == "JOIN_KEY_TYPE_MISMATCH"
    assert "2.5" in str(exc.value)


def test_blank_and_nan_decimal_keys_never_match():
    hh = pl.DataFrame({"hh": [1.0, None, float("nan"), 2.0], "rooms": [2, 8, 9, 5]})
    out = _persons().wrangling.join(hh, on="hh", on_unmatched="ignore")
    assert out.data["rooms"].to_list() == [None, 2, 2, 5, 5, 5, None]


def test_float_widths_join():
    p = _persons().wrangling.cast("hh", pl.Float32)
    hh = pl.DataFrame({"hh": [1.0, 2.0, 3.0], "rooms": [2, 5, 3]})
    out = p.wrangling.join(hh, on="hh", on_unmatched="ignore")
    assert out.data["rooms"].to_list()[:4] == [3, 2, 2, 5]
