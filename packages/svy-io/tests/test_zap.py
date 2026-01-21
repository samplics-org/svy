# tests/test_zap.py
from __future__ import annotations

import copy

import polars as pl
import polars.testing as plt
import pytest

from svy_io.tagged_na import tagged_na

# Assuming you exposed zap helpers from svy_io.zap (or svy_io)
# If you placed them elsewhere, just tweak the imports accordingly.
from svy_io.zap import (
    zap_empty,
    zap_label,
    zap_labels,
    zap_missing,
    zap_missing_with_meta,
    zap_widths,
)


# ---------- fixtures & helpers ----------


def _fake_meta():
    # Mirrors haven-style structures you already produce in meta
    return {
        "vars": [
            # labelled numeric with a variable label
            {"name": "y1", "label": "foo", "label_set": None, "fmt": None},
            # labelled numeric with a variable label
            {"name": "y2", "label": "bar", "label_set": None, "fmt": None},
            # labelled string that points to a label set
            {"name": "gender", "label": None, "label_set": "$GENDER", "fmt": None},
            # plain numeric, no label
            {"name": "x", "label": None, "label_set": None, "fmt": None},
        ],
        "value_labels": [
            {"set_name": "$GENDER", "mapping": {"f": "Female", "m": "Male"}},
        ],
        # Optional: user-missing/tagged missing scaffolding if you emit it
        "user_missing": [],
    }


# ---------- zap_label ----------


def test_zap_label_strips_label_but_keeps_other_meta():
    meta_in = _fake_meta()
    meta_out = zap_label(copy.deepcopy(meta_in))

    # y1/y2 labels removed; other fields untouched
    vmap = {v["name"]: v for v in meta_out["vars"]}
    assert vmap["y1"]["label"] is None
    assert vmap["y2"]["label"] is None

    # label_set stays intact
    assert vmap["gender"]["label_set"] == "$GENDER"
    # value label sets remain unchanged
    assert any(vl["set_name"] == "$GENDER" for vl in meta_out["value_labels"])


def test_zap_label_on_dataframe_applies_per_column():
    df = pl.DataFrame(
        {
            "x": list(range(1, 11)),
            "y1": list(range(10, 0, -1)),
            "y2": list(range(1, 11)),
        }
    )
    meta_in = _fake_meta()
    df2, meta_out = zap_label(df, meta_in)  # assuming your zap returns (df, meta)

    assert df2.shape == df.shape  # same frame
    vmap = {v["name"]: v for v in meta_out["vars"]}
    assert vmap["y1"]["label"] is None
    assert vmap["y2"]["label"] is None


def test_zap_label_leaves_unlabelled_vectors_unmodified():
    df = pl.DataFrame({"x": [1, 98, 99]})
    meta = {
        "vars": [{"name": "x", "label": None, "label_set": None, "fmt": None}],
        "value_labels": [],
        "user_missing": [],
    }
    df2, meta2 = zap_label(df, meta)
    plt.assert_frame_equal(df2, df, check_row_order=True, check_dtypes=True)


# ---------- zap_labels ----------


def test_zap_labels_strips_value_labels():
    meta_in = _fake_meta()
    # Attach a value-label set to y1/y2 (simulating labelled numeric) and ensure removal
    meta_in["vars"][0]["label_set"] = "$DUMMY"
    meta_in["vars"][1]["label_set"] = "$DUMMY"
    meta_in["value_labels"].append(
        {"set_name": "$DUMMY", "mapping": {"1": "good", "2": "bad"}}
    )

    meta_out = zap_labels(copy.deepcopy(meta_in))
    vmap = {v["name"]: v for v in meta_out["vars"]}

    # All columns must have label_set cleared
    assert vmap["y1"]["label_set"] is None
    assert vmap["y2"]["label_set"] is None
    assert vmap["gender"]["label_set"] is None  # clears SAS value labels too

    # And global value_labels should be dropped
    assert meta_out.get("value_labels", []) in ([], None)


def test_zap_labels_dataframe_applied_per_column():
    df = pl.DataFrame({"x": list(range(1, 11)), "y": list(range(10, 0, -1))})
    meta_in = _fake_meta()
    # Attach y to some label set
    meta_in["vars"][1]["label_set"] = "$DUMMY"
    meta_in["value_labels"].append({"set_name": "$DUMMY", "mapping": {"1": "good"}})

    df2, meta_out = zap_labels(df, meta_in)
    assert df2.shape == df.shape
    vmap = {v["name"]: v for v in meta_out["vars"]}
    assert vmap["y"]["label_set"] is None


@pytest.mark.xfail(reason="SPSS user-defined missings not wired yet")
def test_zap_labels_spss_user_na_conversion_default_false():
    # If you later support SPSS user-missings, this should convert user-missings to None/NA
    df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
    meta = {
        "vars": [{"name": "x", "label": None, "label_set": None, "fmt": None}],
        "value_labels": [{"set_name": "$LAB", "mapping": {"1": "a"}}],
        "user_missing": [
            {"col": "x", "type": "spss", "na_values": [2, 4], "na_range": None}
        ],
    }
    df2, meta2 = zap_labels(df, meta, user_na=False)  # expect 2 and 4 -> nulls
    assert df2["x"].to_list() == [1, None, 3, None, 5]


@pytest.mark.xfail(reason="SPSS user-defined missings persistence not wired yet")
def test_zap_labels_spss_user_na_true_keeps_values():
    df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
    meta = {
        "vars": [{"name": "x", "label": None, "label_set": None, "fmt": None}],
        "value_labels": [{"set_name": "$LAB", "mapping": {"1": "a"}}],
        "user_missing": [
            {"col": "x", "type": "spss", "na_values": [2, 4], "na_range": None}
        ],
    }
    df2, _ = zap_labels(df, meta, user_na=True)  # keep 2 and 4 as values
    assert df2["x"].to_list() == [1, 2, 3, 4, 5]


# ---------- zap_missing ----------


# @pytest.mark.xfail(reason="tagged/user missings to regular NA not implemented yet")
def test_zap_missing_converts_special_missings():
    # Once you or native layer emits tagged missings/user-missings in meta,
    # zap_missing should turn them into nulls and adjust labels accordingly.
    df = pl.DataFrame({"x": [1, 2, 99]})
    meta = {
        "vars": [{"name": "x", "label": None, "label_set": "$LAB", "fmt": None}],
        "value_labels": [{"set_name": "$LAB", "mapping": {"99": "missing"}}],
        "user_missing": [
            {"col": "x", "type": "spss", "na_values": [99], "na_range": None}
        ],
    }
    df2, meta2 = zap_missing_with_meta(df, meta)
    assert df2["x"].to_list() == [1, 2, None]
    # Also expect the value label for 99 removed
    # (exact behavior may vary; adjust once implemented)


# ---------- zap_widths ----------


def test_zap_widths_vector_metadata_is_removed():
    # If you propagate display widths via column metadata, simulate it in meta
    meta_in = _fake_meta()
    # Simulate a non-standard attribute in var metadata, e.g., "display_width"
    for v in meta_in["vars"]:
        v["display_width"] = 10

    meta_out = zap_widths(copy.deepcopy(meta_in))
    assert all("display_width" not in v for v in meta_out["vars"])


def test_zap_widths_on_dataframe_leaves_data_intact():
    df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
    meta = {
        "vars": [
            {
                "name": "x",
                "label": None,
                "label_set": None,
                "fmt": None,
                "display_width": 10,
            }
        ],
        "value_labels": [],
        "user_missing": [],
    }
    df2, meta2 = zap_widths(df, meta)
    assert df2.equals(df)
    assert "display_width" not in meta2["vars"][0]


# ---------- zap_empty ----------


def test_zap_empty_replaces_empty_strings_with_none():
    x = ["", "a", None]
    out = zap_empty(x)
    assert out == [None, "a", None]


# ---- Added test
def test_zap_missing_converts_tagged_na_in_series():
    df = pl.DataFrame({"x": [1.0, tagged_na("a"), 3.0]}, strict=False)
    out = zap_missing(df, meta={"vars": [], "value_labels": [], "user_missing": []})
    assert out["x"].to_list() == [1.0, None, 3.0]
