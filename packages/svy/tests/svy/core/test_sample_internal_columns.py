"""Sample.internal_columns: the __svy_* columns sample.data shows, which the design needs."""

import polars as pl

import svy


def _sample() -> svy.Sample:
    df = pl.DataFrame(
        {
            "g": ["a", "a", "b", "b", "a", "b"],
            "r": ["rr", "nr", "rr", "rr", "rr", "rr"],
            "w": [1.0, 2.0, 3.0, 4.0, 1.5, 2.5],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    return svy.Sample(df, svy.Design(wgt="w"))


def test_none_on_a_plain_sample():
    s = _sample()
    assert s.internal_columns == []
    assert "__svy_row_index__" not in s.internal_columns


def test_adjustment_snapshots_are_listed_and_in_data():
    s = _sample().weighting.poststratify(controls={"a": 10, "b": 20}, cells="g")
    assert s.internal_columns == ["__svy_cells_ps_wgt"]
    assert set(s.internal_columns) <= set(s.data.columns)


def test_raking_and_calibration_snapshots():
    s = _sample().weighting.rake(controls={"g": {"a": 10, "b": 20}})
    assert s.internal_columns and all(c.startswith("__svy_cells_") for c in s.internal_columns)
    c = _sample().weighting.calibrate(controls={svy.Cat("g"): {"a": 10, "b": 20}})
    assert any(col.startswith("__svy_aux_") for col in c.internal_columns)


def test_the_design_needs_them():
    s = _sample().weighting.poststratify(controls={"a": 10, "b": 20}, cells="g")
    assert set(s.internal_columns) <= set(s.design.columns())


def test_user_columns_are_never_listed():
    s = _sample().weighting.poststratify(controls={"a": 10, "b": 20}, cells="g")
    assert not set(s.internal_columns) & {"g", "r", "w", "x", "ps_wgt"}


def test_describe_leaves_them_out():
    s = _sample().weighting.poststratify(controls={"a": 10, "b": 20}, cells="g")
    names = {it.name for it in s.describe().items}
    assert not names & set(s.internal_columns)
    assert "ps_wgt" in names
