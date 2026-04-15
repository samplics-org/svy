# tests/svy/weighting/test_wgt_raking_controls_margins_template.py
import numpy as np
import polars as pl
import pytest
from svy.core.sample import Design, Sample
from svy.errors import DimensionError


@pytest.fixture
def base_df():
    return pl.DataFrame(
        {
            "initial_weight": [10.0] * 8,
            "age_group": ["18-34", "35-54", "55+", "18-34", "35-54", "55+", "18-34", "35-54"],
            "region": ["North", "North", "North", "North", "South", "South", "South", "South"],
        }
    )


@pytest.fixture
def sample(base_df):
    return Sample(data=base_df, design=Design(wgt="initial_weight"))


def test_controls_margins_template_basic(sample):
    margins = {"age": "age_group", "region": "region"}
    tmpl = sample.weighting.controls_margins_template(margins=margins, cat_na="level")
    assert set(tmpl.keys()) == {"age", "region"}
    assert isinstance(tmpl["age"], dict)
    assert isinstance(tmpl["region"], dict)
    assert list(tmpl["age"].keys()) == ["18-34", "35-54", "55+"]
    assert all(np.isnan(v) for v in tmpl["age"].values())
    assert list(tmpl["region"].keys()) == ["North", "South"]


def test_controls_margins_template_includes_na_with_level_policy(sample):
    df2 = sample.data.with_columns(
        pl.when(pl.arange(0, pl.len()) == 0).then(None).otherwise(pl.col("region")).alias("region")
    )
    sample2 = Sample(data=df2, design=Design(wgt="initial_weight"))
    margins = {"region": "region"}
    tmpl = sample2.weighting.controls_margins_template(margins=margins, cat_na="level")
    assert "__NA__" in tmpl["region"]
    assert all(np.isnan(v) for v in tmpl["region"].values())


def test_controls_margins_template_cat_na_error_raises_on_nulls(sample):
    df2 = sample.data.with_columns(
        pl.when(pl.arange(0, pl.len()) == 1)
        .then(None)
        .otherwise(pl.col("age_group"))
        .alias("age_group")
    )
    sample2 = Sample(data=df2, design=Design(wgt="initial_weight"))
    with pytest.raises(DimensionError) as ei:
        sample2.weighting.controls_margins_template(margins={"age": "age_group"}, cat_na="error")
    msg = str(ei.value)
    assert "Missing values in margin column" in msg or "MARGIN_NA" in msg


def test_controls_margins_template_unknown_column_raises(sample):
    with pytest.raises(Exception):
        sample.weighting.controls_margins_template(margins={"bad": "not_a_column"})


def test_controls_margins_template_requires_str_keys_and_values(sample):
    with pytest.raises(Exception):
        sample.weighting.controls_margins_template(margins={1: "region"})  # type: ignore[arg-type]
    with pytest.raises(Exception):
        sample.weighting.controls_margins_template(margins={"region": 123})  # type: ignore[arg-type]


def test_controls_margins_template_natural_sort_for_numeric_like_strings():
    df = pl.DataFrame(
        {
            "initial_weight": [1.0] * 6,
            "hhsize_cat": ["1", "10+", "2", "3", "9", "2"],
        }
    )
    s = Sample(data=df, design=Design(wgt="initial_weight"))
    tmpl = s.weighting.controls_margins_template(margins={"hhsize": "hhsize_cat"}, cat_na="level")
    assert list(tmpl["hhsize"].keys()) == ["1", "2", "3", "9", "10+"]
    assert all(np.isnan(v) for v in tmpl["hhsize"].values())
