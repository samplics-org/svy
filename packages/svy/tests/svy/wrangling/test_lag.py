"""wrangling.lag — the one panel primitive."""

import polars as pl
import pytest

import svy

from svy.errors import DimensionError, MethodError


def _panel(waves=(1, 2, 3)):
    # case 2 skips wave 2; case 3 has no wave 3
    df = pl.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 3, 3],
            "wave": [1, 2, 3, 1, 3, 1, 2],
            "y": [1, 2, 3, 4, 5, 6, 7],
            "w": [1.0] * 7,
        }
    )
    codes = {1: waves[0], 2: waves[1], 3: waves[2]}
    df = df.with_columns(pl.col("wave").replace_strict(codes))
    s = svy.Sample(df, svy.Design(case_id="id", wave="wave", wgt="w"))
    s.meta.set_value_labels("y", {1: "one", 2: "two"})
    s.meta.set_label("y", "Outcome")
    return s


def test_lag_is_previous_wave_not_previous_row():
    out = _panel().wrangling.lag("y")
    assert out.data["y_lag1"].to_list() == [None, 1, 2, None, None, None, 6]


def test_gaps_skip_takes_previous_observed_row():
    out = _panel().wrangling.lag("y", gaps="skip")
    assert out.data["y_lag1"].to_list() == [None, 1, 2, None, 4, None, 6]


def test_producer_wave_codes_step_by_rank():
    out = _panel(waves=(2019, 2021, 2023)).wrangling.lag("y")
    assert out.data["y_lag1"].to_list() == [None, 1, 2, None, None, None, 6]


def test_lag2_and_lead():
    s = _panel()
    assert s.wrangling.lag("y", 2).data["y_lag2"].to_list() == [None, None, 1, None, 4, None, None]
    assert s.wrangling.lag("y", -1).data["y_lead1"].to_list() == [2, 3, None, None, None, 7, None]


def test_row_order_is_kept_regardless_of_input_order():
    s = _panel()
    shuffled = svy.Sample(s.data.sample(fraction=1.0, shuffle=True, seed=3), s.design)
    out = shuffled.wrangling.lag("y")
    expected = {(1, 2): 1, (1, 3): 2, (3, 2): 6}
    for row in out.data.select("id", "wave", "y_lag1").iter_rows():
        assert row[2] == expected.get((row[0], row[1]))


def test_labels_and_type_carry_over():
    out = _panel().wrangling.lag("y")
    meta = out.meta.get("y_lag1")
    assert meta.labels == {1: "one", 2: "two"}
    assert meta.label == "Outcome (lag 1)"
    assert out.meta.get("y_lag1").mtype == out.meta.get("y").mtype


def test_custom_names_and_several_columns():
    s = _panel().wrangling.mutate({"z": svy.col("y") * 10})
    out = s.wrangling.lag(["y", "z"], name=["py", "pz"])
    assert out.data["pz"].to_list() == [None, 10, 20, None, None, None, 60]
    with pytest.raises(DimensionError):
        s.wrangling.lag(["y", "z"], name="only_one")


def test_copy_on_write_and_inplace():
    s = _panel()
    out = s.wrangling.lag("y")
    assert "y_lag1" not in s.data.columns and "y_lag1" in out.data.columns
    s.wrangling.lag("y", inplace=True)
    assert "y_lag1" in s.data.columns


def test_requires_panel():
    df = pl.DataFrame({"id": [1, 2], "y": [1, 2]})
    with pytest.raises(MethodError, match="kind='panel'"):
        svy.Sample(df, svy.Design(case_id="id")).wrangling.lag("y")


def test_guards():
    s = _panel()
    with pytest.raises(MethodError, match="n"):
        s.wrangling.lag("y", 0)
    with pytest.raises(MethodError, match="already exist"):
        s.wrangling.lag("y", name="y")
    with pytest.raises(DimensionError):
        s.wrangling.lag("nope")
    with pytest.raises(MethodError, match="gaps"):
        s.wrangling.lag("y", gaps="fill")  # type: ignore[arg-type]


def test_transition_table_and_paired_change_recipes():
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4] * 2,
            "wave": [1] * 4 + [2] * 4,
            "emp": [1, 1, 0, 0, 1, 0, 0, 1],
            "inc": [10.0, 20.0, 30.0, 40.0, 12.0, 21.0, 33.0, 41.0],
            "w": [1.0] * 8,
        }
    )
    s = svy.Sample(df, svy.Design(case_id="id", wave="wave", wgt="w")).wrangling.lag(
        ["emp", "inc"]
    )
    # transitions = P(emp at t | emp at t-1), a domain-correct conditional table
    tr = s.estimation.prop(
        "emp", by="emp_lag1", where=svy.col("wave") == 2, drop_nulls=True
    ).to_polars()
    cells = {(int(r["emp_lag1"]), int(r["emp"])): r["est"] for r in tr.iter_rows(named=True)}
    assert cells == pytest.approx({(0, 0): 0.5, (0, 1): 0.5, (1, 0): 0.5, (1, 1): 0.5})
    # the joint table after subsetting rows to the target wave
    tab = s.wrangling.filter_records(svy.col("wave") == 2).categorical.tabulate(
        "emp_lag1", "emp", units="count"
    )
    assert tab.to_polars()["est"].sum() == pytest.approx(4.0)
    tt = s.categorical.ttest("inc", y_pair="inc_lag1", where=svy.col("wave") == 2, drop_nulls=True)
    assert tt.estimates[0].est == pytest.approx((2 + 1 + 3 + 1) / 4)
