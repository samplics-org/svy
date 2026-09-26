# tests/svy/weighting/test_wgt_target_keys.py
"""Target keys match the data's values, with a text fallback.

JSON object keys are always strings, so every keyed weighting method has to
find the level ``1`` from ``"1"``, ``True`` from ``"true"`` and a date from its
ISO string, while a key typed in the data's own type keeps working.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import polars as pl
import pytest

from svy import Cat, Cross, Design, Sample, SvyUserWarning
from svy.errors import MethodError, SvyError, WeightingError


N = 24


def _text(v):
    """The key a JSON round trip gives for ``v``."""
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, dt.date):
        return v.isoformat()
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


DTYPES = {
    "int": ([1, 2, 3], None),
    "neg": ([-1, 0, 2], None),
    "float": ([1.0, 2.5, 3.0], None),
    "bool": ([True, False], None),
    "str": (["a", "b", "c"], None),
    "cat": (["a", "b", "c"], pl.Categorical),
    "enum": (["a", "b", "c"], pl.Enum(["a", "b", "c"])),
    "date": ([dt.date(2024, 1, 1), dt.date(2024, 2, 1), dt.date(2024, 3, 1)], None),
}


def _frame() -> pl.DataFrame:
    cols = {}
    for name, (levels, dtype) in DTYPES.items():
        vals = [levels[i % len(levels)] for i in range(N)]
        s = pl.Series(name, vals)
        cols[name] = s.cast(dtype) if dtype is not None else s
    return pl.DataFrame(cols).with_columns(
        pl.Series("w", [1.0 + (i * 7) % 5 for i in range(N)]),
        pl.Series("dom", (["d1"] * 3 + ["d2"] * 3) * (N // 6)),
        pl.Series("x", [float(i % 7) for i in range(N)]),
        pl.Series("psu", [i // 2 for i in range(N)]),
        pl.Series("status_int", [1, 2, 1, 3] * (N // 4)),
        pl.Series("status_bool", [True, False, True, True] * (N // 4)),
        pl.Series("status_float", [1.0, 2.0, 1.0, 1.0] * (N // 4)),
        pl.Series("status_str", ["Resp", "Refused", "resp", "dk"] * (N // 4)),
    )


@pytest.fixture
def sample() -> Sample:
    return Sample(_frame(), Design(wgt="w"))


def _native_controls(col: str) -> dict:
    levels = DTYPES[col][0]
    return {lv: 10.0 * (i + 1) for i, lv in enumerate(levels)}


def _forms(col: str) -> dict[str, dict]:
    nat = _native_controls(col)
    text = {_text(k): v for k, v in nat.items()}
    mixed = {(k if i % 2 == 0 else _text(k)): v for i, (k, v) in enumerate(nat.items())}
    return {"native": nat, "text": text, "mixed": mixed}


def _w(s: Sample, name: str) -> np.ndarray:
    return s.data.get_column(name).to_numpy()


# ---------------------------------------------------------------------------
# Every keyed method x every dtype x native / text / mixed keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("col", list(DTYPES))
@pytest.mark.parametrize("form", ["native", "text", "mixed"])
class TestEveryMethodEveryDtype:
    def test_poststratify_controls(self, sample, col, form):
        ref = sample.weighting.poststratify(_native_controls(col), cells=col)
        out = sample.weighting.poststratify(_forms(col)[form], cells=col)
        np.testing.assert_allclose(_w(out, "ps_wgt"), _w(ref, "ps_wgt"))
        totals = out.data.group_by(col).agg(pl.col("ps_wgt").sum()).rows()
        for lv, tot in totals:
            assert tot == pytest.approx(_native_controls(col)[lv])

    def test_poststratify_shares(self, sample, col, form):
        ref = sample.weighting.poststratify(shares=_native_controls(col), cells=col)
        out = sample.weighting.poststratify(shares=_forms(col)[form], cells=col)
        np.testing.assert_allclose(_w(out, "ps_wgt"), _w(ref, "ps_wgt"))

    def test_normalize_controls_and_shares(self, sample, col, form):
        ref = sample.weighting.normalize(_native_controls(col), cells=col)
        out = sample.weighting.normalize(_forms(col)[form], cells=col)
        np.testing.assert_allclose(_w(out, "norm_wgt"), _w(ref, "norm_wgt"))
        ref = sample.weighting.normalize(shares=_native_controls(col), cells=col)
        out = sample.weighting.normalize(shares=_forms(col)[form], cells=col)
        np.testing.assert_allclose(_w(out, "norm_wgt"), _w(ref, "norm_wgt"))

    def test_rake_controls_and_shares(self, sample, col, form):
        total = sum(_native_controls(col).values())
        other = {"d1": total / 2, "d2": total / 2}
        ref = sample.weighting.rake(controls={col: _native_controls(col), "dom": other})
        out = sample.weighting.rake(controls={col: _forms(col)[form], "dom": other})
        np.testing.assert_allclose(_w(out, "rk_wgt"), _w(ref, "rk_wgt"))
        ref = sample.weighting.rake(shares={col: _native_controls(col)})
        out = sample.weighting.rake(shares={col: _forms(col)[form]})
        np.testing.assert_allclose(_w(out, "rk_wgt"), _w(ref, "rk_wgt"))

    def test_standardize_by_domain(self, sample, col, form):
        ref = sample.weighting.standardize(col, shares=_native_controls(col), by="dom")
        out = sample.weighting.standardize(col, shares=_forms(col)[form], by="dom")
        np.testing.assert_allclose(_w(out, "std_wgt"), _w(ref, "std_wgt"))

    def test_calibrate_cat_levels(self, sample, col, form):
        ref = sample.weighting.calibrate(controls={Cat(col): _native_controls(col)})
        out = sample.weighting.calibrate(controls={Cat(col): _forms(col)[form]})
        np.testing.assert_allclose(_w(out, "calib_wgt"), _w(ref, "calib_wgt"))

    def test_calibrate_by_domain_keys(self, sample, col, form):
        def by_domain(keys):
            return {k: {"x": 20.0 + 5 * i} for i, k in enumerate(keys)}

        nat = list(_native_controls(col))
        keys = list(_forms(col)[form])
        ref = sample.weighting.calibrate(controls=by_domain(nat), by=col)
        out = sample.weighting.calibrate(controls=by_domain(keys), by=col)
        np.testing.assert_allclose(_w(out, "calib_wgt"), _w(ref, "calib_wgt"))

    def test_calibrate_matrix_domain_keys(self, sample, col, form):
        X = sample.data.select(pl.lit(1.0), "x").to_numpy()
        nat = list(_native_controls(col))
        keys = list(_forms(col)[form])
        ref = sample.weighting.calibrate_matrix(
            aux_vars=X, controls={k: [12.0, 30.0 + i] for i, k in enumerate(nat)}, by=col
        )
        out = sample.weighting.calibrate_matrix(
            aux_vars=X, controls={k: [12.0, 30.0 + i] for i, k in enumerate(keys)}, by=col
        )
        np.testing.assert_allclose(_w(out, "calib_wgt"), _w(ref, "calib_wgt"))


# ---------------------------------------------------------------------------
# adjust: resp_mapping values against the status column's own values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "column, native, text",
    [
        ("status_int", {"rr": 1, "nr": 2, "in": 3}, {"rr": "1", "nr": "2", "in": "3"}),
        ("status_bool", {"rr": True, "nr": False}, {"rr": "true", "nr": "False"}),
        ("status_float", {"rr": 1.0, "nr": 2.0}, {"rr": "1", "nr": "2.0"}),
    ],
)
def test_adjust_mapping_text_values(sample, column, native, text):
    ref = sample.weighting.adjust(column, resp_mapping=native, respondents_only=False)
    out = sample.weighting.adjust(column, resp_mapping=text, respondents_only=False)
    np.testing.assert_allclose(_w(out, "nr_wgt"), _w(ref, "nr_wgt"))
    nr = sample.data[column].to_numpy() == native["nr"]
    assert _w(out, "nr_wgt")[nr].sum() == 0
    assert _w(out, "nr_wgt")[~nr].sum() == pytest.approx(_w(sample, "w").sum())


def test_adjust_mapping_stays_case_insensitive(sample):
    # "resp" names both "Resp" and "resp", as it always has.
    out = sample.weighting.adjust(
        "status_str",
        resp_mapping={"rr": "resp", "nr": ["refused", "DK"]},
        respondents_only=False,
    )
    s = sample.data["status_str"].to_numpy()
    w = _w(sample, "w")
    got = _w(out, "nr_wgt")
    assert got[np.isin(s, ["Refused", "dk"])].sum() == 0
    assert got.sum() == pytest.approx(w.sum())


def test_adjust_mapping_list_of_mixed_forms(sample):
    out = sample.weighting.adjust(
        "status_int", resp_mapping={"rr": [1, "3"], "nr": ["2"]}, respondents_only=False
    )
    s = sample.data["status_int"].to_numpy()
    assert _w(out, "nr_wgt")[s == 2].sum() == 0


# ---------------------------------------------------------------------------
# Several cells columns: tuple keys, element-wise, and the joined text form
# ---------------------------------------------------------------------------


@pytest.fixture
def two_way() -> Sample:
    df = pl.DataFrame(
        {
            "r": [1, 1, 2, 2] * 3,
            "s": ["M", "F", "M", "F"] * 3,
            "b": [True, False] * 6,
            "w": [1.0, 2.0, 3.0, 4.0] * 3,
        }
    )
    return Sample(df, Design(wgt="w"))


TWO_WAY_NATIVE = {(1, "M"): 10.0, (1, "F"): 20.0, (2, "M"): 30.0, (2, "F"): 40.0}


@pytest.mark.parametrize(
    "keys",
    [
        TWO_WAY_NATIVE,
        {("1", "M"): 10.0, ("1", "F"): 20.0, ("2", "M"): 30.0, ("2", "F"): 40.0},
        {(1, "M"): 10.0, ("1", "F"): 20.0, (2, "M"): 30.0, ("2", "F"): 40.0},
        {"1_&_M": 10.0, "1_&_F": 20.0, "2_&_M": 30.0, "2_&_F": 40.0},
        {(1, "M"): 10.0, "1_&_F": 20.0, ("2", "M"): 30.0, (2, "F"): 40.0},
    ],
    ids=["native", "text", "mixed-parts", "joined", "mixed-forms"],
)
def test_two_column_keys(two_way, keys):
    ref = two_way.weighting.poststratify(TWO_WAY_NATIVE, cells=["r", "s"])
    out = two_way.weighting.poststratify(keys, cells=["r", "s"])
    np.testing.assert_allclose(_w(out, "ps_wgt"), _w(ref, "ps_wgt"))


def test_three_column_keys_with_bool(two_way):
    levels = two_way.data.select("r", "s", "b").unique().rows()
    nat = {tuple(lv): 5.0 + i for i, lv in enumerate(sorted(levels))}
    text = {tuple(_text(p) for p in k): v for k, v in nat.items()}
    joined = {"_&_".join(_text(p) for p in k): v for k, v in nat.items()}
    ref = two_way.weighting.poststratify(nat, cells=["r", "s", "b"])
    for keys in (text, joined):
        out = two_way.weighting.poststratify(keys, cells=["r", "s", "b"])
        np.testing.assert_allclose(_w(out, "ps_wgt"), _w(ref, "ps_wgt"))


def test_standardize_two_column_composition(two_way):
    df = two_way.data.with_columns(pl.Series("g", ["x"] * 4 + ["y"] * 4 + ["x"] * 4))
    s = Sample(df, Design(wgt="w"))
    ref = s.weighting.standardize(["r", "s"], shares=TWO_WAY_NATIVE, by="g")
    out = s.weighting.standardize(
        ["r", "s"], shares={"_&_".join(map(str, k)): v for k, v in TWO_WAY_NATIVE.items()}, by="g"
    )
    np.testing.assert_allclose(_w(out, "std_wgt"), _w(ref, "std_wgt"))


def test_calibrate_cross_term_text_keys(two_way):
    term = Cross(Cat("r"), Cat("s"))
    ref = two_way.weighting.calibrate(controls={term: TWO_WAY_NATIVE})
    out = two_way.weighting.calibrate(
        controls={term: {tuple(map(str, k)): v for k, v in TWO_WAY_NATIVE.items()}}
    )
    np.testing.assert_allclose(_w(out, "calib_wgt"), _w(ref, "calib_wgt"))


def test_calibrate_by_two_columns_joined_and_tuple(two_way):
    X = np.ones((two_way.data.height, 1))
    ref = two_way.weighting.calibrate_matrix(
        aux_vars=X, controls={k: [v] for k, v in TWO_WAY_NATIVE.items()}, by=["r", "s"]
    )
    out = two_way.weighting.calibrate_matrix(
        aux_vars=X,
        controls={"_&_".join(map(str, k)): [v] for k, v in TWO_WAY_NATIVE.items()},
        by=["r", "s"],
    )
    np.testing.assert_allclose(_w(out, "calib_wgt"), _w(ref, "calib_wgt"))


def test_calibrate_matrix_labels_keyed(sample):
    X = sample.data.select(pl.lit(1.0), "x").to_numpy()
    ref = sample.weighting.calibrate_matrix(aux_vars=X, controls=np.array([30.0, 80.0]))
    out = sample.weighting.calibrate_matrix(
        aux_vars=X, controls={"x": 80.0, "1": 30.0}, labels=[1, "x"]
    )
    np.testing.assert_allclose(_w(out, "calib_wgt"), _w(ref, "calib_wgt"))


def test_calibrate_matrix_labels_keyed_by_domain(sample):
    X = sample.data.select(pl.lit(1.0), "x").to_numpy()
    ref = sample.weighting.calibrate_matrix(
        aux_vars=X, controls={"d1": [10.0, 30.0], "d2": [6, 20]}, by="dom"
    )
    out = sample.weighting.calibrate_matrix(
        aux_vars=X,
        controls={
            "d1": {"x": 30.0, "one": 10.0},
            "d2": {"one": 6, "x": 20},
        },
        by="dom",
        labels=["one", "x"],
    )
    np.testing.assert_allclose(_w(out, "calib_wgt"), _w(ref, "calib_wgt"))


def test_cat_reference_level_by_text(sample):
    ref = sample.weighting.calibrate(controls={Cat("int", ref=1): {2: 50.0, 3: 50.0}, "x": 60})
    out = sample.weighting.calibrate(
        controls={Cat("int", ref="1"): {"1": 999.0, "2": 50.0, "3": 50.0}, "x": 60}
    )
    np.testing.assert_allclose(_w(out, "calib_wgt"), _w(ref, "calib_wgt"))


# ---------------------------------------------------------------------------
# Text forms that must NOT match, and ones that must be refused
# ---------------------------------------------------------------------------


def test_string_column_1_vs_01():
    df = pl.DataFrame({"c": ["1", "01", "1", "01"], "w": [1.0, 2.0, 3.0, 4.0]})
    s = Sample(df, Design(wgt="w"))
    out = s.weighting.poststratify({1: 10.0, "01": 30.0}, cells="c")
    sums = dict(out.data.group_by("c").agg(pl.col("ps_wgt").sum()).rows())
    assert sums == pytest.approx({"1": 10.0, "01": 30.0})


def test_int_column_does_not_read_01_as_1(sample):
    with pytest.raises(WeightingError) as ei:
        sample.weighting.poststratify({"01": 10.0, 2: 20.0, 3: 30.0}, cells="int")
    assert ei.value.code == "CONTROLS_KEYS_MISMATCH"
    assert ei.value.got == {"missing": [1], "extra": ["01"]}


@pytest.mark.parametrize("key", ["1", "1.0", 1, 1.0])
def test_float_column_whole_number_forms(sample, key):
    ctl = {key: 10.0, 2.5: 20.0, "3": 30.0}
    out = sample.weighting.poststratify(ctl, cells="float")
    sums = dict(out.data.group_by("float").agg(pl.col("ps_wgt").sum()).rows())
    assert sums == pytest.approx({1.0: 10.0, 2.5: 20.0, 3.0: 30.0})


def test_float_column_other_spellings_do_not_match(sample):
    with pytest.raises(WeightingError) as ei:
        sample.weighting.poststratify({"1.00": 10.0, 2.5: 20.0, 3.0: 30.0}, cells="float")
    assert ei.value.got == {"missing": [1.0], "extra": ["1.00"]}


@pytest.mark.parametrize("key", ["-1", -1, -1.0])
def test_negative_levels(sample, key):
    out = sample.weighting.poststratify({key: 10.0, "0": 20.0, 2: 30.0}, cells="neg")
    sums = dict(out.data.group_by("neg").agg(pl.col("ps_wgt").sum()).rows())
    assert sums == pytest.approx({-1: 10.0, 0: 20.0, 2: 30.0})


@pytest.mark.parametrize("t, f", [("true", "false"), ("True", "False"), (True, "false")])
def test_bool_spellings(sample, t, f):
    out = sample.weighting.rake(controls={"bool": {t: 60.0, f: 40.0}})
    sums = dict(out.data.group_by("bool").agg(pl.col("rk_wgt").sum()).rows())
    assert sums == pytest.approx({True: 60.0, False: 40.0})


def test_bool_rake_no_longer_raises_numpy_keyerror(sample):
    # Used to fail with KeyError(np.False_).
    out = sample.weighting.rake(controls={"bool": {"true": 3, "false": 3}})
    assert "rk_wgt" in out.data.columns


def test_date_keys_as_iso_strings(sample):
    out = sample.weighting.poststratify(
        {"2024-01-01": 10.0, "2024-02-01": 20.0, dt.date(2024, 3, 1): 30.0}, cells="date"
    )
    assert _w(out, "ps_wgt").sum() == pytest.approx(60.0)


def test_datetime_keys_both_iso_spellings():
    ts = [dt.datetime(2024, 1, 1, 8), dt.datetime(2024, 1, 2, 9, 30)]
    s = Sample(pl.DataFrame({"t": ts * 2, "w": [1.0, 2.0, 3.0, 4.0]}), Design(wgt="w"))
    out = s.weighting.poststratify(
        {"2024-01-01T08:00:00": 5.0, "2024-01-02 09:30:00": 7.0}, cells="t"
    )
    assert _w(out, "ps_wgt").sum() == pytest.approx(12.0)


def test_ambiguous_joined_key():
    df = pl.DataFrame({"a": ["x_&_y", "x"], "b": ["z", "y_&_z"], "w": [1.0, 2.0]})
    s = Sample(df, Design(wgt="w"))
    with pytest.raises(WeightingError) as ei:
        s.weighting.poststratify({"x_&_y_&_z": 1.0, ("x", "y_&_z"): 2.0}, cells=["a", "b"])
    err = ei.value
    assert err.code == "CONTROLS_KEY_AMBIGUOUS"
    assert err.got == "x_&_y_&_z"
    assert sorted(err.expected) == [("x", "y_&_z"), ("x_&_y", "z")]
    assert "instead of 'x_&_y_&_z'" in err.hint
    json.dumps(err.to_dict())
    # Tuples stay unambiguous.
    out = s.weighting.poststratify({("x_&_y", "z"): 1.0, ("x", "y_&_z"): 2.0}, cells=["a", "b"])
    assert _w(out, "ps_wgt").tolist() == [1.0, 2.0]


def test_duplicate_keys_for_one_level(sample):
    with pytest.raises(WeightingError) as ei:
        sample.weighting.poststratify({1: 10.0, "1": 10.0, 2: 1.0, 3: 1.0}, cells="int")
    err = ei.value
    assert err.code == "CONTROLS_KEY_DUPLICATE"
    assert err.expected == 1
    assert err.got == [1, "1"]


def test_bool_key_on_int_column_keeps_equality_match():
    # True == 1 in Python, and such keys were accepted before.
    s = Sample(pl.DataFrame({"c": [0, 1, 0, 1], "w": [1.0] * 4}), Design(wgt="w"))
    out = s.weighting.poststratify({False: 2.0, True: 6.0}, cells="c")
    assert _w(out, "ps_wgt").tolist() == [1.0, 3.0, 1.0, 3.0]


# ---------------------------------------------------------------------------
# Missing, extra, empty, absent levels, where=, nulls
# ---------------------------------------------------------------------------


def test_missing_and_extra_together(sample):
    with pytest.raises(WeightingError) as ei:
        sample.weighting.poststratify({"1": 10.0, "2": 10.0, "9": 5.0, 7: 1.0}, cells="int")
    err = ei.value
    assert err.code == "CONTROLS_KEYS_MISMATCH"
    assert err.expected == [1, 2, 3]
    assert err.got == {"missing": [3], "extra": ["9", 7]}
    assert "{1: ..., 2: ..., 3: ...}" in err.hint


def test_empty_controls_dict(sample):
    with pytest.raises(WeightingError) as ei:
        sample.weighting.poststratify({}, cells="str")
    assert ei.value.code == "CONTROLS_KEYS_MISMATCH"
    assert ei.value.got == {"missing": ["a", "b", "c"], "extra": []}
    with pytest.raises(WeightingError) as ei:
        sample.weighting.rake(controls={})
    assert ei.value.code == "CONTROLS_MISSING"
    with pytest.raises(WeightingError) as ei:
        sample.weighting.rake(controls={"str": {}})
    assert ei.value.code == "CONTROLS_TYPE_INVALID"
    with pytest.raises(WeightingError) as ei:
        sample.weighting.calibrate(controls={})
    assert ei.value.code == "CONTROLS_MISSING"


def test_level_absent_from_data_zero_is_accepted(sample):
    base = sample.weighting.poststratify({"a": 1.0, "b": 2.0, "c": 3.0}, cells="str")
    out = sample.weighting.poststratify({"a": 1.0, "b": 2.0, "c": 3.0, "zz": 0}, cells="str")
    np.testing.assert_allclose(_w(out, "ps_wgt"), _w(base, "ps_wgt"))
    out = sample.weighting.rake(controls={"str": {"a": 1, "b": 2, "c": 3, "zz": 0.0}})
    assert _w(out, "rk_wgt").sum() == pytest.approx(6.0)


@pytest.mark.parametrize("method", ["poststratify", "rake", "calibrate", "standardize"])
def test_level_absent_from_data_nonzero_is_refused(sample, method):
    ctl = {"a": 1.0, "b": 2.0, "c": 3.0, "zz": 4.0}
    with pytest.raises(WeightingError) as ei:
        if method == "poststratify":
            sample.weighting.poststratify(ctl, cells="str")
        elif method == "rake":
            sample.weighting.rake(controls={"str": ctl})
        elif method == "calibrate":
            sample.weighting.calibrate(controls={Cat("str"): ctl})
        else:
            sample.weighting.standardize("str", shares=ctl)
    assert ei.value.code == "CONTROLS_KEYS_MISMATCH"
    assert ei.value.got["extra"] == ["zz"]


def test_where_scopes_the_levels(sample):
    scoped = pl.col("str") != "c"
    out = sample.weighting.poststratify({"a": 10.0, "b": 20.0}, cells="str", where=scoped)
    w0, w1 = _w(sample, "w"), _w(out, "ps_wgt")
    c = sample.data["str"].to_numpy() == "c"
    np.testing.assert_allclose(w1[c], w0[c])
    with pytest.raises(WeightingError) as ei:
        sample.weighting.poststratify({"a": 1.0, "b": 2.0, "c": 3.0}, cells="str", where=scoped)
    assert ei.value.got == {"missing": [], "extra": ["c"]}
    assert "outside `where`" in ei.value.hint
    out = sample.weighting.rake(controls={"str": {"a": 10.0, "b": 20.0}}, where=scoped)
    np.testing.assert_allclose(_w(out, "rk_wgt")[c], w0[c])


def test_null_cells_keep_todays_behaviour():
    df = pl.DataFrame({"c": [1, 2, None, 1, 2], "w": [1.0, 2.0, 3.0, 4.0, 5.0]})
    s = Sample(df, Design(wgt="w"))
    with pytest.warns(SvyUserWarning, match=r"\[CELLS_NULL_UNADJUSTED\]"):
        out = s.weighting.poststratify({"1": 10.0, "2": 20.0}, cells="c")
    assert _w(out, "ps_wgt")[2] == 3.0
    with pytest.raises(WeightingError) as ei:
        s.weighting.poststratify({1: 10.0, 2: 20.0, None: 3.0}, cells="c")
    assert ei.value.got == {"missing": [], "extra": [None]}


def test_null_level_in_cat_term_takes_a_none_key():
    df = pl.DataFrame({"c": ["a", None, "a", "b"], "w": [1.0, 2.0, 3.0, 4.0]})
    s = Sample(df, Design(wgt="w"))
    out = s.weighting.calibrate(controls={Cat("c"): {"a": 5.0, "b": 5.0, None: 5.0}})
    assert _w(out, "calib_wgt").sum() == pytest.approx(15.0)


def test_rake_null_margin_still_refused():
    df = pl.DataFrame({"c": [1, None, 2, 1], "w": [1.0] * 4})
    s = Sample(df, Design(wgt="w"))
    with pytest.raises(SvyError) as ei:
        s.weighting.rake(controls={"c": {1: 2.0, 2: 2.0}})
    assert ei.value.code == "MARGIN_NA"
    assert ei.value.got == {"c": 1}


def test_replicate_weights_follow_text_keys(sample):
    s = Sample(sample.data, Design(wgt="w", psu="psu")).weighting.create_jk_wgts()
    ref = s.weighting.poststratify(_native_controls("int"), cells="int")
    out = s.weighting.poststratify(_forms("int")["text"], cells="int")
    reps = out.design.rep_wgts.columns
    np.testing.assert_allclose(out.data.select(reps).to_numpy(), ref.data.select(reps).to_numpy())
    for c in reps:
        sums = dict(out.data.group_by("int").agg(pl.col(c).sum()).rows())
        assert sums == pytest.approx(_native_controls("int"))


def test_keyed_errors_are_method_errors(sample):
    # Callers catching MethodError keep working.
    with pytest.raises(MethodError):
        sample.weighting.poststratify({"x": 1.0}, cells="int")
