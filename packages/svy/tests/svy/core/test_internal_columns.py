# tests/svy/core/test_internal_columns.py
"""`__svy_` columns are svy's; everything in `sample.data` is your data or your design's.

svy keeps its derived bookkeeping (row index, design keys, singleton variance
columns) in ``sample._data`` only. ``sample.data`` and ``svy.write_parquet``
show the user's columns plus ``design.columns()``, the weight-adjustment
record's snapshots included, and nothing else. A user column named like the
bookkeeping is refused with a guiding error.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

import svy

from svy import col
from svy.core.constants import (
    BOOKKEEPING_COLUMNS,
    SVY_ROW_INDEX,
    SVY_VAR_EXCLUDE,
    SVY_VAR_PSU,
    SVY_VAR_STRATUM,
    key_col,
)
from svy.errors import MethodError
from svy.serialize import from_json, to_design, to_json


def _declare(sample, method, **kw):
    """A fork of ``sample`` with the singleton rule declared on its design."""
    from svy.core.design import Singleton as _Rule

    new = sample._fork()
    new.update_design(singleton=_Rule(method, **kw))
    return new


STRATUM_KEY = key_col("stratum")
PSU_KEY = key_col("psu")
SSU_KEY = key_col("ssu")


def make_frame(singleton: bool = False) -> pl.DataFrame:
    """Four strata (region x urban) of 3 PSUs x 4 rows; optionally one singleton stratum."""
    rng = np.random.default_rng(11)
    rows = []
    i = 0
    for reg in ("N", "S"):
        for urb in ("u", "r"):
            n_psu = 1 if singleton and (reg, urb) == ("S", "r") else 3
            for p in range(n_psu):
                for h in range(4):
                    i += 1
                    rows.append(
                        {
                            "id": i,
                            "reg": reg,
                            "urb": urb,
                            "ea": f"{reg}{urb}{p}",
                            "hh": h % 2,
                            "w": 1.0 + (i % 5) * 0.5,
                            "y": float(rng.normal(50 + 5 * p, 4)),
                            "x": float(rng.normal(10, 2)),
                            "g": ("a", "b", "c")[i % 3],
                            "resp": 1 if i % 7 else 2,
                            "one": 1.0,
                        }
                    )
    return pl.DataFrame(rows)


DATA = make_frame()
USER = DATA.columns

DESIGNS = {
    "none": None,
    "wgt": svy.Design(wgt="w"),
    "single": svy.Design(stratum="reg", psu="ea", wgt="w"),
    "tuple": svy.Design(stratum=("reg", "urb"), psu=("ea", "hh"), wgt="w"),
    "ssu": svy.Design(stratum=("reg", "urb"), psu="ea", ssu="hh", wgt="w"),
    "strata": svy.Design(stratum=("reg", "urb"), psu="ea", wgt="w"),
}


def sample(kind: str = "tuple", data: pl.DataFrame = DATA) -> svy.Sample:
    return svy.Sample(data, DESIGNS[kind])


def assert_shown(s: svy.Sample) -> None:
    """sample.data is the user's columns plus the design's, nothing of svy's bookkeeping."""
    shown = s.data.columns
    kept = s._data.columns
    assert not set(shown) & BOOKKEEPING_COLUMNS, shown
    assert set(shown) == set(kept) - BOOKKEEPING_COLUMNS
    assert [c for c in kept if c not in BOOKKEEPING_COLUMNS] == shown, "order is kept"
    assert set(s.design.columns(data_columns=shown)) <= set(shown)
    # svy's own columns shown are design data: a record snapshot of this design
    # or of one in its history.
    own = {c for c in shown if c.startswith("__svy_")}
    assert own <= {c for d in s.design_history for c in d.columns(data_columns=shown)}, own
    assert SVY_ROW_INDEX in kept
    assert s._data[SVY_ROW_INDEX].n_unique() == s._data.height
    assert s.n_columns == len(shown)


def ests(s: svy.Sample) -> list:
    out = []
    for e in (
        s.estimation.mean("y"),
        s.estimation.total("y", by="g"),
        s.estimation.prop("g"),
    ):
        out.append(sorted((str(p.by_level), str(p.y_level), p.est, p.se) for p in e.estimates))
    return out


def assert_same_estimates(a: svy.Sample, b: svy.Sample) -> None:
    for ra, rb in zip(ests(a), ests(b), strict=True):
        assert len(ra) == len(rb)
        for x, y in zip(ra, rb):
            assert x[:2] == y[:2]
            assert x[2] == pytest.approx(y[2], rel=1e-12)
            assert x[3] == pytest.approx(y[3], rel=1e-12)


def roundtrip(s: svy.Sample, tmp_path) -> svy.Sample:
    p = tmp_path / "s.parquet"
    svy.write_parquet(s, p)
    back = svy.read_parquet(p)
    assert back.columns == s.data.columns
    assert back.equals(s.data)
    return svy.Sample(back, s.design)


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------


def test_bookkeeping_names_are_svy_dunders():
    assert SVY_ROW_INDEX == "__svy_row_index__"
    assert (STRATUM_KEY, PSU_KEY, SSU_KEY) == (
        "__svy_stratum_key__",
        "__svy_psu_key__",
        "__svy_ssu_key__",
    )
    assert all(c.startswith("__svy_") and c.endswith("__") for c in BOOKKEEPING_COLUMNS)


def test_selection_outputs_keep_their_svy_names():
    s = svy.Sample(DATA, svy.Design(stratum=("reg", "urb"))).sampling.srs(n=2, rstate=1)
    assert {"svy_prob_selection", "svy_sample_weight", "svy_number_of_hits"} <= set(s.data.columns)
    assert_shown(s)
    p = svy.Sample(DATA, svy.Design(stratum=("reg", "urb"), mos="x")).sampling.pps_sys(
        n=2, rstate=np.random.default_rng(1)
    )
    assert {"svy_prob_selection", "svy_sample_weight", "svy_certainty"} <= set(p.data.columns)
    assert_shown(p)


# ---------------------------------------------------------------------------
# What sample.data shows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(DESIGNS))
def test_data_is_user_columns_for_every_design(kind):
    s = sample(kind)
    assert s.data.columns == USER
    assert_shown(s)


@pytest.mark.parametrize(
    "kind, keys",
    [
        ("none", set()),
        ("wgt", set()),
        ("single", {STRATUM_KEY, PSU_KEY}),
        ("tuple", {STRATUM_KEY, PSU_KEY}),
        ("ssu", {STRATUM_KEY, PSU_KEY, SSU_KEY}),
    ],
)
def test_keys_live_in_the_private_frame_only(kind, keys):
    s = sample(kind)
    present = set(s._data.columns) & {STRATUM_KEY, PSU_KEY, SSU_KEY}
    assert present == keys
    assert not present & set(s.data.columns)


def test_data_is_a_copy():
    s = sample()
    d = s.data
    d = d.with_columns(pl.lit(0.0).alias("y"))
    assert s.data["y"].sum() != 0.0


def test_describe_dtypes_meta_and_print_hide_bookkeeping():
    s = sample()
    assert set(s.dtypes) == set(USER)
    assert set(s.meta) == set(USER)
    assert f"Columns  : {len(USER)}" in s.__plain_str__()
    described = str(s.describe())
    assert "__svy_" not in described


def test_show_data_hides_bookkeeping():
    out = sample().show_data(n=None)
    assert out.columns == USER


# ---------------------------------------------------------------------------
# Singleton handling
# ---------------------------------------------------------------------------

SINGLE_DATA = make_frame(singleton=True)
SINGLETON_RULES = {
    "certainty": lambda s: _declare(s, "self_representing"),
    "skip": lambda s: _declare(s, "skip"),
    "scale": lambda s: _declare(s, "scale"),
    "center": lambda s: _declare(s, "center"),
    "pool": lambda s: _declare(s, "pool"),
    "collapse": lambda s: _declare(s, "collapse"),
}


@pytest.mark.parametrize("rule", list(SINGLETON_RULES))
def test_singleton_variance_columns_are_bookkeeping(rule, tmp_path):
    s = SINGLETON_RULES[rule](svy.Sample(SINGLE_DATA, DESIGNS["strata"]))
    assert s.design.singleton is not None
    assert {SVY_VAR_STRATUM, SVY_VAR_PSU, SVY_VAR_EXCLUDE} <= set(s._data.columns)
    assert_shown(s)
    back = roundtrip(s, tmp_path)
    assert_same_estimates(s, back)
    assert_shown(back)


def test_singleton_rule_survives_a_second_round_trip(tmp_path):
    s = _declare(svy.Sample(SINGLE_DATA, DESIGNS["strata"]), "collapse")
    once = roundtrip(s, tmp_path)
    twice = roundtrip(once, tmp_path)
    assert_same_estimates(s, twice)


# ---------------------------------------------------------------------------
# Weighting records and replicate weights
# ---------------------------------------------------------------------------

G_TOTALS = {"a": 40.0, "b": 50.0, "c": 60.0}
WEIGHTING = {
    "poststratify": lambda s: s.weighting.poststratify(G_TOTALS, cells="g", wgt_name="nw"),
    "rake": lambda s: s.weighting.rake(
        controls={"g": G_TOTALS, "urb": {"u": 70.0, "r": 80.0}}, wgt_name="nw"
    ),
    "calibrate": lambda s: s.weighting.calibrate(
        controls={"one": 150.0, "x": 1500.0}, wgt_name="nw"
    ),
    "normalize": lambda s: s.weighting.normalize(controls=G_TOTALS, cells="g", wgt_name="nw"),
    "standardize": lambda s: s.weighting.standardize(
        "g", shares={"a": 0.3, "b": 0.3, "c": 0.4}, by="reg", wgt_name="nw"
    ),
    "adjust": lambda s: s.weighting.adjust(
        "resp", cells="g", resp_mapping={"rr": 1, "nr": 2}, wgt_name="nw"
    ),
    "trim": lambda s: s.weighting.trim(upper=2.5, wgt_name="nw"),
}
REPLICATES = {
    "taylor": lambda s: s,
    "jk": lambda s: s.weighting.create_jk_wgts(rep_prefix="jk"),
    "bs": lambda s: s.weighting.create_bs_wgts(n_reps=12, rep_prefix="bs", rstate=3),
}


@pytest.mark.parametrize("reps", list(REPLICATES))
@pytest.mark.parametrize("method", list(WEIGHTING))
def test_weighting_record_is_shown_and_saved(method, reps, tmp_path):
    base = REPLICATES[reps](sample("ssu"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = WEIGHTING[method](base)
    assert s.design.wgt == "nw"
    assert_shown(s)
    record = set(s.design.columns(data_columns=s.data.columns)) - set(base.data.columns)
    assert record <= set(s.data.columns)
    back = roundtrip(s, tmp_path)
    assert back.design == s.design
    assert_same_estimates(s, back)


def test_record_snapshot_columns_stay_visible():
    s = sample("tuple").weighting.poststratify(G_TOTALS, cells="g", wgt_name="nw")
    snaps = [c for c in s.data.columns if c.startswith("__svy_cells_")]
    assert snaps
    assert set(snaps) <= set(s.design.columns(data_columns=s.data.columns))


@pytest.mark.parametrize(
    "make",
    [
        lambda s: s.weighting.create_jk_wgts(rep_prefix="jk"),
        lambda s: s.weighting.create_bs_wgts(n_reps=10, rep_prefix="bs", rstate=2),
        lambda s: s.weighting.create_brr_wgts(rep_prefix="brr"),
        lambda s: s.weighting.create_sdr_wgts(n_reps=8, rep_prefix="sdr"),
    ],
)
def test_replicate_weights_round_trip(make, tmp_path):
    s = make(sample("single"))
    assert set(s.rep_columns) <= set(s.data.columns)
    assert_shown(s)
    back = roundtrip(s, tmp_path)
    a = s.estimation.mean("y", method="replication").estimates[0]
    b = back.estimation.mean("y", method="replication").estimates[0]
    assert (a.est, a.se) == (pytest.approx(b.est, rel=1e-12), pytest.approx(b.se, rel=1e-12))


# ---------------------------------------------------------------------------
# Wrangling, data and design changes
# ---------------------------------------------------------------------------

WRANGLING = {
    "filter_records": lambda s, ip: s.wrangling.filter_records(col("id") > 5, inplace=ip),
    "mutate": lambda s, ip: s.wrangling.mutate({"z": pl.col("y") * 2}, inplace=ip),
    "mutate_design_col": lambda s, ip: s.wrangling.mutate(
        {"urb": pl.col("urb").str.to_uppercase()}, inplace=ip
    ),
    "rename": lambda s, ip: s.wrangling.rename_columns({"reg": "region"}, inplace=ip),
    "remove_force": lambda s, ip: s.wrangling.remove_columns(["urb"], force=True, inplace=ip),
    "remove": lambda s, ip: s.wrangling.remove_columns(["x"], inplace=ip),
    "keep": lambda s, ip: s.wrangling.keep_columns(
        ["reg", "urb", "ea", "hh", "w", "y", "g"], inplace=ip
    ),
    "cast": lambda s, ip: s.wrangling.cast("hh", pl.Int64, inplace=ip),
    "order_by": lambda s, ip: s.wrangling.order_by("y", inplace=ip),
    "recode": lambda s, ip: s.wrangling.recode("g", {"ab": ["a", "b"]}, into="g2", inplace=ip),
    "clean_names": lambda s, ip: s.wrangling.clean_names(letter_case="upper", inplace=ip),
    "join": lambda s, ip: s.wrangling.join(
        pl.DataFrame({"reg": ["N", "S"], "area": [1.0, 2.0]}), on="reg", inplace=ip
    ),
    "with_row_index": lambda s, ip: s.wrangling.with_row_index("rid", inplace=ip),
}


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("op", list(WRANGLING))
def test_every_wrangling_path_keeps_the_rule(op, inplace):
    s = sample("tuple")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = WRANGLING[op](s, inplace)
    assert (out is s) == inplace
    assert_shown(out)
    if not inplace:
        assert_shown(s)
        assert s.data.columns == USER
    out.estimation.mean("y" if "y" in out.data.columns else "Y")


def test_keys_follow_a_design_column_rewrite():
    s = sample("tuple").wrangling.mutate({"urb": pl.col("urb").str.to_uppercase()})
    assert set(s._data[STRATUM_KEY].cast(pl.String).to_list()) == {
        "N__by__U",
        "N__by__R",
        "S__by__U",
        "S__by__R",
    }


def test_keys_follow_a_rename():
    s = sample("tuple").wrangling.rename_columns({"reg": "region"})
    assert s.design.stratum == ("region", "urb")
    assert STRATUM_KEY in s._data.columns
    assert_same_estimates(s, sample("tuple"))


@pytest.mark.parametrize("method", ["set_data", "update_data"])
def test_set_data_from_sample_data(method):
    s = sample("tuple").wrangling.filter_records(col("id") > 4)
    getattr(s, method)(s.data.with_columns(z=pl.col("y") + 1))
    assert "z" in s.data.columns
    assert_shown(s)
    assert_same_estimates(s, svy.Sample(s.data, s.design))


def test_update_design_and_history_restore():
    s = sample("tuple").weighting.poststratify(G_TOTALS, cells="g", wgt_name="nw")
    s = s.weighting.rake(controls={"urb": {"u": 70.0, "r": 80.0}}, wgt_name="rk")
    assert_shown(s)
    s.update_design(stratum="reg")
    assert_shown(s)
    assert STRATUM_KEY in s._data.columns
    s.update_design(wgt="nw")
    assert s.design.wgt_adjustment is not None
    assert_shown(s)
    s.update_design(stratum=None, psu=None)
    assert not {STRATUM_KEY, PSU_KEY} & set(s._data.columns)
    assert_shown(s)


def test_clone_without_data_carries_no_stale_bookkeeping():
    s = sample("tuple").wrangling.filter_records(col("id") > 10)
    c = s.clone()
    assert c.data.equals(s.data)
    assert c._data[SVY_ROW_INDEX].to_list() == list(range(c.n_records))
    assert_same_estimates(s, c)


def test_add_stage_drops_the_next_stages_keys():
    ea = pl.DataFrame(
        {
            "ea": list(range(1, 9)),
            "region": ["N"] * 4 + ["S"] * 4,
            "mos": [float(i) for i in range(1, 9)],
        }
    )
    hh = pl.DataFrame(
        {
            "hid": list(range(32)),
            "ea": [1 + i // 4 for i in range(32)],
            "urban": ["u", "r"] * 16,
            "y": [float(i % 7) for i in range(32)],
        }
    )
    s1 = svy.Sample(ea, svy.Design(mos="mos", stratum="region", psu="ea")).sampling.pps_sys(
        n=2, rstate=np.random.default_rng(1)
    )
    s2 = svy.Sample(
        hh.filter(pl.col("ea").is_in(s1.data["ea"].to_list())),
        svy.Design(stratum="urban", psu="ea"),
    ).sampling.srs(n=2, by="ea", rstate=1)
    assert STRATUM_KEY in s2._data.columns
    c = s1.sampling.add_stage(s2)
    assert c.design.stratum == "region"
    # The keys are rebuilt for the combined design: region, not stage 2's urban.
    keys = c._data.select(pl.col(STRATUM_KEY).cast(pl.Utf8), pl.col("region").cast(pl.Utf8))
    assert keys[STRATUM_KEY].to_list() == keys["region"].to_list()
    assert not set(c.data.columns) & BOOKKEEPING_COLUMNS


def test_panel_design():
    long = pl.concat([DATA.with_columns(wave=pl.lit(1)), DATA.with_columns(wave=pl.lit(2))])
    s = svy.Sample(long, svy.Design(case_id="id", wave="wave", wgt="w"))
    assert PSU_KEY in s._data.columns
    assert s.data.columns == long.columns
    assert_shown(s)


def test_combine_samples():
    a = sample("tuple")
    b = sample("tuple").wrangling.filter_records(col("id") > 20)
    c = svy.combine_samples([a, b])
    assert_shown(c)
    assert c.n_records == a.n_records + b.n_records


# ---------------------------------------------------------------------------
# Writing, to_code, serialize
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", list(DESIGNS))
def test_write_parquet_writes_what_is_shown(kind, tmp_path):
    s = sample(kind)
    back = roundtrip(s, tmp_path)
    assert back.data.columns == USER
    if kind != "none":
        assert_same_estimates(s, back)


def test_write_csv_writes_what_is_shown(tmp_path):
    s = sample("tuple").weighting.poststratify(G_TOTALS, cells="g", wgt_name="nw")
    p = tmp_path / "s.csv"
    svy.write_csv(s, p)
    assert pl.read_csv(p).columns == s.data.columns


def test_to_code_runs_on_the_saved_file(tmp_path):
    # skip: scale has no replicate analogue (SINGLETON_REPLICATES).
    s = _declare(svy.Sample(SINGLE_DATA, DESIGNS["strata"]), "skip")
    s = s.weighting.create_jk_wgts(rep_prefix="jk")
    s = s.weighting.poststratify(G_TOTALS, cells="g", wgt_name="nw")
    p = tmp_path / "s.parquet"
    svy.write_parquet(s, p)
    scope: dict = {}
    exec(s.to_code(data=p), scope)
    rebuilt = scope["sample"]
    assert rebuilt.design == s.design
    assert_shown(rebuilt)
    a = s.estimation.mean("y", method="replication").estimates[0]
    b = rebuilt.estimation.mean("y", method="replication").estimates[0]
    assert (a.est, a.se) == (pytest.approx(b.est, rel=1e-12), pytest.approx(b.se, rel=1e-12))


def test_serialize_design_round_trip_is_unchanged():
    s = _declare(svy.Sample(SINGLE_DATA, DESIGNS["strata"]), "collapse")
    s = s.weighting.poststratify(G_TOTALS, cells="g", wgt_name="nw")
    text = to_json(s.design)
    text = text.decode() if isinstance(text, bytes) else text
    assert not any(c in text for c in BOOKKEEPING_COLUMNS)
    assert to_design(from_json(text)) == s.design


# ---------------------------------------------------------------------------
# Reserved names
# ---------------------------------------------------------------------------


def _reserved(fn) -> MethodError:
    with pytest.raises(MethodError) as ei:
        fn()
    assert ei.value.code == "RESERVED_COLUMN"
    return ei.value


@pytest.mark.parametrize("name", sorted(BOOKKEEPING_COLUMNS))
def test_a_user_column_with_a_reserved_name_is_refused(name):
    df = DATA.with_columns(pl.lit(1).alias(name))
    err = _reserved(lambda: svy.Sample(df, DESIGNS["tuple"]))
    assert err.param == name
    assert name in err.hint
    assert name.removeprefix("__svy_").strip("_") in err.hint


def test_the_private_frame_is_refused_with_a_hint_to_drop():
    s = sample("tuple")
    err = _reserved(lambda: svy.Sample(s._data, s.design))
    for c in (SVY_ROW_INDEX, STRATUM_KEY, PSU_KEY):
        assert c in err.got
        assert c in err.hint
    fixed = svy.Sample(s._data.drop(err.got), s.design)
    assert_same_estimates(s, fixed)


def test_a_frame_saved_by_an_earlier_version_is_refused():
    legacy = sample("tuple").data.with_columns(
        pl.lit("x").alias(SVY_VAR_STRATUM), pl.lit(1).alias("svy_row_index")
    )
    err = _reserved(lambda: svy.Sample(legacy, DESIGNS["tuple"]))
    assert err.got == [SVY_VAR_STRATUM]
    s = svy.Sample(legacy.drop(err.got), DESIGNS["tuple"])
    assert "svy_row_index" in s.data.columns, "the old name is an ordinary column now"


@pytest.mark.parametrize("method", ["set_data", "update_data"])
def test_set_data_refuses_reserved_names(method):
    s = sample("tuple")
    before = s.data
    _reserved(lambda: getattr(s, method)(s.data.with_columns(pl.lit(0).alias(SVY_ROW_INDEX))))
    assert s.data.equals(before)


def test_clone_refuses_reserved_names():
    s = sample("tuple")
    _reserved(lambda: s.clone(data=s._data))


@pytest.mark.parametrize("name", [SVY_ROW_INDEX, SVY_VAR_STRATUM, STRATUM_KEY])
def test_mutate_refuses_reserved_names(name):
    s = sample("tuple")
    err = _reserved(lambda: s.wrangling.mutate({name: pl.lit(1)}))
    assert err.where == "wrangling.mutate"
    assert s.data.columns == USER


def test_value_writers_refuse_reserved_names():
    s = sample("tuple")
    _reserved(lambda: s.wrangling.recode("g", {"ab": ["a", "b"]}, into=SVY_VAR_PSU))
    _reserved(lambda: s.wrangling.top_code({"y": 60.0}, into=SVY_VAR_EXCLUDE))
    _reserved(lambda: s.wrangling.cast(SVY_ROW_INDEX, pl.Int64))
    _reserved(lambda: s.wrangling.with_row_index(SVY_ROW_INDEX))


def test_rename_refuses_reserved_names():
    s = sample("tuple")
    for pair in ({"reg": STRATUM_KEY}, {SVY_ROW_INDEX: "rid"}):
        with pytest.raises(MethodError) as ei:
            s.wrangling.rename_columns(pair)
        assert ei.value.code == "RENAME_FORBIDDEN"


def test_join_refuses_reserved_names():
    s = sample("tuple")
    other = pl.DataFrame({"reg": ["N", "S"], "v": [1, 2]})
    with pytest.raises(MethodError) as ei:
        s.wrangling.join(other, on="reg", into={"v": SVY_ROW_INDEX})
    assert ei.value.code == "JOIN_COLUMN_EXISTS"


def test_join_never_brings_the_other_samples_bookkeeping():
    s = sample("tuple")
    other = (
        sample("tuple")
        .wrangling.keep_columns(["id", "reg", "urb", "ea", "hh", "w", "x"])
        .wrangling.rename_columns({"x": "x2"})
    )
    out = s.wrangling.join(other, on="id", cols=None, suffix="_o")
    assert_shown(out)


# ---------------------------------------------------------------------------
# Findings name the user's columns
# ---------------------------------------------------------------------------


def _boundary_message(by) -> str:
    rng = np.random.default_rng(1)
    n = 240
    df = pl.DataFrame(
        {
            "st": np.repeat(["s1", "s2", "s3", "s4"], n // 4),
            "psu": np.repeat(np.arange(40), n // 40),
            "w": rng.uniform(1, 3, n),
            "quint": np.tile(["richest", "poorest", "middle", "second", "fourth", "2nd"], n // 6),
            "zone": np.tile(["x", "y", "z"], n // 3),
            "pov": rng.integers(0, 2, n),
        }
    ).with_columns(
        pov=pl.when(pl.col("quint").is_in(["poorest", "2nd"])).then(1).otherwise(pl.col("pov"))
    )
    s = svy.Sample(df, svy.Design(stratum="st", psu="psu", wgt="w"))
    with pytest.warns(svy.SvyUserWarning, match=r"\[PROP_CI_BOUNDARY\]") as rec:
        s.estimation.prop("pov", by=by, ci_method="logit")
    (w,) = [r for r in rec if "PROP_CI_BOUNDARY" in str(r.message)]
    return str(w.message)


def test_prop_boundary_warning_names_the_by_column_in_domain_order():
    msg = _boundary_message("quint")
    assert "__svy_" not in msg and "concatenated" not in msg
    assert "in quint=poorest" in msg
    assert msg.index("quint=2nd") < msg.index("quint=poorest")
    assert msg.index("pov=0 in quint=2nd") < msg.index("pov=1 in quint=2nd")


def test_prop_boundary_warning_splits_several_by_columns():
    msg = _boundary_message(["quint", "zone"])
    assert "__svy_" not in msg and "\x00" not in msg and "__by__" not in msg
    assert "quint=poorest, zone=y" in msg


def test_error_listings_leave_out_bookkeeping():
    s = sample("tuple")
    with pytest.raises(Exception) as ei:
        s.weighting.calibrate_matrix(
            aux_vars=np.ones((s.n_records, 1)), controls=[1.0], by="nope", wgt_name="c"
        )
    assert "__svy_" not in str(ei.value)
