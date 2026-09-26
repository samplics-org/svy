# tests/svy/weighting/test_wgt_templates.py
"""The two target templates: native keys, one missing-value option, round trips."""

from __future__ import annotations

import datetime as dt
import json
import math

import numpy as np
import polars as pl
import pytest

from svy import Cat, Design, Sample
from svy.errors import DimensionError, MethodError, SvyError, WeightingError


LEVELS = {
    "int": [3, 1, 2],
    "neg": [-2, 0, 5],
    "float": [2.5, 1.0, 3.0],
    "bool": [True, False, True],
    "str": ["b", "a", "c"],
    "cat": ["b", "a", "c"],
    "enum": ["b", "a", "c"],
    "date": [dt.date(2024, 3, 1), dt.date(2024, 1, 1), dt.date(2024, 2, 1)],
}
SORTED = {
    "int": [1, 2, 3],
    "neg": [-2, 0, 5],
    "float": [1.0, 2.5, 3.0],
    "bool": [False, True],
    "str": ["a", "b", "c"],
    "cat": ["a", "b", "c"],
    "enum": ["a", "b", "c"],
    "date": [dt.date(2024, 1, 1), dt.date(2024, 2, 1), dt.date(2024, 3, 1)],
}
CASTS = {"cat": pl.Categorical, "enum": pl.Enum(["a", "b", "c"])}
N = 12


def _col(name: str, with_null: bool = False) -> pl.Series:
    vals = [LEVELS[name][i % 3] for i in range(N)]
    if with_null:
        vals[4] = None
    s = pl.Series(name, vals)
    return s.cast(CASTS[name]) if name in CASTS else s


def _sample(with_null: bool = False) -> Sample:
    df = pl.DataFrame([_col(c, with_null) for c in LEVELS]).with_columns(
        pl.Series("w", [1.0 + i % 4 for i in range(N)]),
        pl.Series("g", ["g1", "g2"] * (N // 2)),
        pl.Series("h", [1, 1, 2, 2] * (N // 4)),
    )
    return Sample(df, Design(wgt="w"))


def _json_key(k):
    if isinstance(k, tuple):
        return "_&_".join(str(_json_key(p)) for p in k)
    if isinstance(k, dt.date):
        return k.isoformat()
    return k


def _through_json(d):
    """What a caller that stores the template as JSON gets back."""
    if isinstance(d, dict):
        return {
            str(_json_key(k)) if not isinstance(k, str) else k: _through_json(v)
            for k, v in d.items()
        }
    return d


def _weighted_totals(s: Sample, col: str) -> dict:
    return dict(s.data.group_by(col).agg(pl.col("w").sum() * 2).rows())


# ---------------------------------------------------------------------------
# controls_margins_template
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("col", list(LEVELS))
def test_margins_template_native_keys(col):
    tmpl = _sample().weighting.controls_margins_template(margins={col: col})
    assert list(tmpl) == [col]
    assert list(tmpl[col]) == SORTED[col]
    assert all(type(k) is type(SORTED[col][0]) for k in tmpl[col])
    assert all(math.isnan(v) for v in tmpl[col].values())


@pytest.mark.parametrize("col", list(LEVELS))
def test_margins_template_na_options(col):
    s = _sample(with_null=True)
    with pytest.raises(DimensionError) as ei:
        s.weighting.controls_margins_template(margins={col: col})
    assert ei.value.code == "MARGIN_NA"
    assert ei.value.got == {col: 1}
    assert "na='level'" in ei.value.hint
    json.dumps(ei.value.to_dict())

    level = s.weighting.controls_margins_template(margins={col: col}, na="level")
    assert list(level[col]) == [*SORTED[col], "__NA__"]
    named = s.weighting.controls_margins_template(margins={col: col}, na="level", na_label="NA")
    assert list(named[col])[-1] == "NA"
    drop = s.weighting.controls_margins_template(margins={col: col}, na="drop")
    assert list(drop[col]) == SORTED[col]


def test_margins_template_several_margins_and_natural_sort():
    df = pl.DataFrame(
        {"w": [1.0] * 6, "hh": ["1", "10+", "2", "3", "9", "2"], "z": [2, 1, 2, 1, 2, 1]}
    )
    tmpl = Sample(df, Design(wgt="w")).weighting.controls_margins_template(
        margins={"hh": "hh", "z": "z"}
    )
    assert list(tmpl["hh"]) == ["1", "2", "3", "9", "10+"]
    assert list(tmpl["z"]) == [1, 2]


@pytest.mark.parametrize("col", list(LEVELS))
@pytest.mark.parametrize("via_json", [False, True])
def test_margins_template_round_trip_to_rake(col, via_json):
    s = _sample()
    tmpl = s.weighting.controls_margins_template(margins={col: col, "g": "g"})
    totals = _weighted_totals(s, col)
    tmpl[col] = {k: totals[k] for k in tmpl[col]}
    g_tot = _weighted_totals(s, "g")
    tmpl["g"] = {k: g_tot[k] for k in tmpl["g"]}
    ctl = _through_json(json.loads(json.dumps(_through_json(tmpl)))) if via_json else tmpl
    out = s.weighting.rake(controls=ctl)
    got = dict(out.data.group_by(col).agg(pl.col("rk_wgt").sum()).rows())
    assert got == pytest.approx(totals)


def test_margins_template_legacy_cat_na_names_the_new_one():
    with pytest.raises(WeightingError) as ei:
        _sample().weighting.controls_margins_template(margins={"int": "int"}, cat_na="level")
    err = ei.value
    assert err.code == "PARAM_RENAMED"
    assert err.expected == "na" and err.got == "cat_na"
    assert "na='level'" in err.hint
    assert "default is now 'error'" in err.hint


def test_margins_template_bad_na_value():
    with pytest.raises(MethodError) as ei:
        _sample().weighting.controls_margins_template(margins={"int": "int"}, na="keep")
    assert ei.value.code == "INVALID_CHOICE"
    assert ei.value.expected == ["error", "level", "drop"]


def test_margins_template_unknown_column():
    with pytest.raises(WeightingError) as ei:
        _sample().weighting.controls_margins_template(margins={"m": "nope"})
    assert ei.value.code == "MISSING_COLUMNS"
    assert ei.value.got == ["nope"]


def test_margins_template_unknown_kwarg_is_still_a_typeerror():
    with pytest.raises(TypeError, match="unexpected keyword argument 'bogus'"):
        _sample().weighting.controls_margins_template(margins={"int": "int"}, bogus=1)


# ---------------------------------------------------------------------------
# control_aux_template / build_aux_matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("col", list(LEVELS))
def test_aux_template_native_keys(col):
    s = _sample()
    flat = s.weighting.control_aux_template(x=[Cat(col)])
    assert list(flat) == SORTED[col]
    nested = s.weighting.control_aux_template(x=["w"], by=col)
    assert list(nested) == SORTED[col]
    assert all(list(v) == ["w"] for v in nested.values())


@pytest.mark.parametrize("col", list(LEVELS))
def test_aux_template_na_options(col):
    s = _sample(with_null=True)
    with pytest.raises(DimensionError) as ei:
        s.weighting.control_aux_template(x=["w"], by=col)
    assert ei.value.code == "BY_NA"
    assert ei.value.got == {col: 1}
    json.dumps(ei.value.to_dict())
    level = s.weighting.control_aux_template(x=["w"], by=col, na="level")
    assert list(level) == [*SORTED[col], "__NA__"]
    drop = s.weighting.control_aux_template(x=["w"], by=col, na="drop")
    assert list(drop) == SORTED[col]
    X, _ = s.weighting.build_aux_matrix(x=["w"], by=col, na="drop")
    assert X.shape[0] == N - 1


def test_aux_template_several_by_columns_and_nulls():
    s = _sample(with_null=True)
    tmpl = s.weighting.control_aux_template(x=["w"], by=["g", "int"], na="level")
    assert list(tmpl) == [
        ("g1", 1),
        ("g1", 2),
        ("g1", 3),
        ("g1", "__NA__"),
        ("g2", 1),
        ("g2", 2),
        ("g2", 3),
    ]
    drop = s.weighting.control_aux_template(x=["w"], by=["g", "int"], na="drop")
    assert all(None not in k and "__NA__" not in k for k in drop)


@pytest.mark.parametrize("method", ["control_aux_template", "build_aux_matrix"])
def test_aux_legacy_by_na_names_the_new_one(method):
    with pytest.raises(WeightingError) as ei:
        getattr(_sample().weighting, method)(x=["w"], by="g", by_na="drop")
    err = ei.value
    assert err.code == "PARAM_RENAMED"
    assert (err.expected, err.got) == ("na", "by_na")
    assert "na='drop'" in err.hint
    assert "default" not in err.hint


@pytest.mark.parametrize("col", list(LEVELS))
@pytest.mark.parametrize("via_json", [False, True])
def test_aux_template_round_trip_to_calibrate(col, via_json):
    s = _sample()
    tmpl = s.weighting.control_aux_template(x=[Cat(col)])
    totals = _weighted_totals(s, col)
    filled = {k: totals[k] for k in tmpl}
    ctl = _through_json(json.loads(json.dumps(_through_json(filled)))) if via_json else filled
    out = s.weighting.calibrate(controls={Cat(col): ctl})
    got = dict(out.data.group_by(col).agg(pl.col("calib_wgt").sum()).rows())
    assert got == pytest.approx(totals)


@pytest.mark.parametrize("col", list(LEVELS))
@pytest.mark.parametrize("via_json", [False, True])
def test_aux_template_by_round_trip_to_calibrate_matrix(col, via_json):
    s = _sample()
    X, tmpl = s.weighting.build_aux_matrix(x=[Cat("g"), "h"], by=col)
    labels = list(next(iter(tmpl.values())))
    for dom, inner in tmpl.items():
        rows = s.data.filter(pl.col(col) == dom)
        wsum = dict(rows.group_by("g").agg(pl.col("w").sum()).rows())
        for lab in inner:
            inner[lab] = (rows["w"] * rows["h"]).sum() if lab == "h" else wsum.get(lab, 0.0)
    ctl = _through_json(json.loads(json.dumps(_through_json(tmpl)))) if via_json else tmpl
    out = s.weighting.calibrate_matrix(aux_vars=X, control=ctl, labels=labels, by=col)
    np.testing.assert_allclose(out.data["calib_wgt"].to_numpy(), s.data["w"].to_numpy())


@pytest.mark.parametrize("via_json", [False, True])
def test_aux_template_two_by_columns_round_trip(via_json):
    s = _sample()
    X, tmpl = s.weighting.build_aux_matrix(x=["h"], by=["g", "bool"])
    for (g, b), inner in tmpl.items():
        rows = s.data.filter((pl.col("g") == g) & (pl.col("bool") == b))
        inner["h"] = (rows["w"] * rows["h"]).sum()
    ctl = _through_json(json.loads(json.dumps(_through_json(tmpl)))) if via_json else tmpl
    if via_json:
        assert all("_&_" in k for k in ctl)
    out = s.weighting.calibrate_matrix(aux_vars=X, control=ctl, labels=["h"], by=["g", "bool"])
    np.testing.assert_allclose(out.data["calib_wgt"].to_numpy(), s.data["w"].to_numpy())


def test_template_errors_are_json_ready():
    for call in (
        lambda s: s.weighting.controls_margins_template(margins={"m": "nope"}),
        lambda s: s.weighting.control_aux_template(x=[]),
        lambda s: s.weighting.control_aux_template(x=["w"], by="nope"),
        lambda s: s.weighting.control_aux_template(x=[Cat("int", ref=9)]),
    ):
        with pytest.raises(SvyError) as ei:
            call(_sample())
        payload = json.dumps(ei.value.to_dict())
        assert "np." not in payload
