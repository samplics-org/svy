# tests/svy/core/test_design_record_consistency.py
"""The design stays consistent with its weight, its record and its data.

Covers ``Design.columns()``, the ``RepWgts.wgt`` pairing, the ``update(wgt=X)``
rule (restore from history, drop the record, reset replicates with a warning),
``Sample`` validation of every design column, wrangling protection and renames
of record columns, and that estimates after each state change reflect it.
"""

import warnings

from contextlib import contextmanager
from pathlib import Path

import msgspec
import numpy as np
import polars as pl
import pytest

from numpy.testing import assert_allclose

from svy import Design, Sample, SvyUserWarning
from svy.core.design import PopSize, WgtAdjustment
from svy.core.repwgts import BootstrapWgts, BrrWgts, JackknifeWgts, SdrWgts
from svy.errors import MethodError


DATA_DIR = Path(__file__).resolve().parents[2] / "test_data"

STYPE_POP = {"E": 4421.0, "H": 755.0, "M": 1018.0}
SCHWIDE_POP = {"No": 1000.0, "Yes": 5194.0}
RAKE_CONTROLS = {"stype": STYPE_POP, "sch.wide": SCHWIDE_POP}
GREG_CONTROLS = {"one": 6194.0, "api99": 3914069.0}


@contextmanager
def no_warnings(allow: str | None = None):
    """Every warning is an error, except svy's finding ``allow`` (by code)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        if allow is not None:
            warnings.filterwarnings("ignore", message=rf"\[{allow}\]", category=SvyUserWarning)
        yield


@pytest.fixture(scope="module")
def apiclus1():
    df = pl.read_csv(DATA_DIR / "apiclus1.csv", null_values=["NA"])
    return df.with_columns(pl.lit(1.0).alias("one"))


@pytest.fixture
def api(apiclus1):
    def build():
        return Sample(apiclus1, Design(wgt="pw", psu="dnum", pop_size="fpc"))

    return build


@pytest.fixture
def api_bs(api):
    """apiclus1 with bootstrap replicates of ``pw``."""

    def build():
        return api().weighting.create_bs_wgts(n_reps=20, rstate=np.random.default_rng(7))

    return build


@pytest.fixture
def small():
    """A stratified frame with received replicate weights of ``w``."""
    rng = np.random.default_rng(3)
    n = 24
    df = pl.DataFrame(
        {
            "id": list(range(n)),
            "s": [1] * 8 + [2] * 8 + [3] * 8,
            "p": [i // 2 for i in range(n)],
            "w": rng.uniform(1, 3, n),
            "y": rng.normal(10, 2, n),
            "g": ["a", "b"] * 12,
            "other_w": rng.uniform(1, 3, n),
        }
    )
    reps = {f"r{i}": df["w"].to_numpy() * rng.uniform(0.5, 1.5, n) for i in range(1, 5)}
    return df.with_columns([pl.Series(k, v) for k, v in reps.items()])


def _se(sample, y="api00", kind="mean"):
    return getattr(sample.estimation, kind)(y).to_polars()["se"][0]


def _est(sample, y="api00", kind="mean"):
    return getattr(sample.estimation, kind)(y).to_polars()["est"][0]


def _rake(s, name="rk"):
    return s.weighting.rake(controls=RAKE_CONTROLS, wgt_name=name)


def _taylor(sample):
    d = sample.design
    return Sample(sample.data, Design(wgt=d.wgt, psu=d.psu, pop_size=d.pop_size))


def _fixed_weight_se(sample, y="api00"):
    """SE of the same weights with no record: the weights-fixed value."""
    d = sample.design
    return _se(Sample(sample.data, Design(wgt=d.wgt, psu=d.psu, pop_size=d.pop_size)), y)


# ---------------------------------------------------------------------------
# Equality and hash
# ---------------------------------------------------------------------------


def test_eq_and_hash_include_the_record():
    rec = WgtAdjustment(kind="raking", prev_wgt="w0", new_wgt="w")
    a = Design(wgt="w", psu="p")
    b = Design(wgt="w", psu="p", wgt_adjustment=rec)
    assert a != b
    assert b == Design(wgt="w", psu="p", wgt_adjustment=rec)
    assert hash(b) == hash(Design(wgt="w", psu="p", wgt_adjustment=rec))
    assert len({a, b}) == 2


def test_eq_sees_filled_pairing():
    explicit = Design(wgt="w", rep_wgts=BrrWgts(prefix="r", n_reps=4, wgt="w"))
    filled = Design(wgt="w", rep_wgts=BrrWgts(prefix="r", n_reps=4))
    assert explicit == filled
    assert hash(explicit) == hash(filled)


# ---------------------------------------------------------------------------
# RepWgts.wgt pairing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", [BootstrapWgts, JackknifeWgts, BrrWgts, SdrWgts])
def test_design_fills_the_pairing(variant):
    d = Design(wgt="w", rep_wgts=variant(prefix="r", n_reps=4))
    assert d.rep_wgts.wgt == "w"


def test_no_weight_leaves_replicates_unpaired_until_one_is_set():
    d = Design(rep_wgts=BrrWgts(prefix="r", n_reps=4))
    assert d.rep_wgts.wgt is None
    with no_warnings():
        d2 = d.update(wgt="w")
    assert d2.rep_wgts.wgt == "w"
    assert d.fill_missing(wgt="w").rep_wgts.wgt == "w"


def test_received_replicates_pair_with_the_design_weight(small):
    s = Sample(small, Design(wgt="w", rep_wgts=BrrWgts(prefix="r", n_reps=4)))
    assert s.design.rep_wgts.wgt == "w"
    assert s.rep_wgts.wgt == "w"


@pytest.mark.parametrize(
    "create",
    [
        lambda s: s.weighting.create_bs_wgts(n_reps=6, rstate=np.random.default_rng(1)),
        lambda s: s.weighting.create_jk_wgts(),
        lambda s: s.weighting.create_brr_wgts(),
        lambda s: s.weighting.create_sdr_wgts(n_reps=4, order_col="id"),
    ],
    ids=["bootstrap", "jackknife", "brr", "sdr"],
)
def test_created_replicates_pair_with_their_weight(create):
    df = pl.DataFrame(
        {
            "id": list(range(12)),
            "s": [1] * 4 + [2] * 4 + [3] * 4,
            "p": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            "w": [1.0 + i / 10 for i in range(12)],
            "y": [float(i) for i in range(12)],
        }
    )
    with no_warnings():
        s = create(Sample(df, Design(wgt="w", stratum="s", psu="p")))
    assert s.design.rep_wgts.wgt == "w"


@pytest.mark.parametrize(
    "method",
    ["poststratify", "rake", "calibrate", "normalize", "trim", "standardize", "adjust"],
)
def test_weighting_with_replicates_repairs_them_to_the_new_weight(api_bs, method):
    s = api_bs().weighting
    with no_warnings():
        out = {
            "poststratify": lambda: s.poststratify(STYPE_POP, cells="stype", wgt_name="nw"),
            "rake": lambda: s.rake(controls=RAKE_CONTROLS, wgt_name="nw"),
            "calibrate": lambda: s.calibrate(controls=GREG_CONTROLS, wgt_name="nw"),
            "normalize": lambda: s.normalize(controls=6194.0, wgt_name="nw"),
            "trim": lambda: s.trim(upper=40.0, wgt_name="nw"),
            "standardize": lambda: s.standardize(
                "stype", shares=STYPE_POP, by="sch.wide", wgt_name="nw"
            ),
        }
        if method == "adjust":
            base = api_bs()
            base = base.wrangling.mutate(
                {"resp": pl.Series([1] * (base.n_records - 10) + [2] * 10)}
            )
            res = base.weighting.adjust(
                resp_status="resp",
                resp_mapping={"rr": 1, "nr": 2},
                cells="stype",
                wgt_name="nw",
                respondents_only=False,
            )
        else:
            res = out[method]()
    assert res.design.wgt == "nw"
    assert res.design.rep_wgts.wgt == "nw"
    assert res.design.rep_wgts.prefix == "nw"


def _with_resp(sample):
    return sample.wrangling.mutate({"resp": pl.Series([1] * (sample.n_records - 10) + [2] * 10)})


IGNORE_REPS = {
    "poststratify": lambda s: s.weighting.poststratify(
        STYPE_POP, cells="stype", wgt_name="nw", ignore_reps=True
    ),
    "rake": lambda s: s.weighting.rake(controls=RAKE_CONTROLS, wgt_name="nw", ignore_reps=True),
    "calibrate": lambda s: s.weighting.calibrate(
        controls=GREG_CONTROLS, wgt_name="nw", ignore_reps=True
    ),
    "normalize": lambda s: s.weighting.normalize(controls=6194.0, wgt_name="nw", ignore_reps=True),
    "standardize": lambda s: s.weighting.standardize(
        "stype", shares=STYPE_POP, by="sch.wide", wgt_name="nw", ignore_reps=True
    ),
    "adjust": lambda s: _with_resp(s).weighting.adjust(
        resp_status="resp",
        resp_mapping={"rr": 1, "nr": 2},
        cells="stype",
        wgt_name="nw",
        respondents_only=False,
        ignore_reps=True,
    ),
}


@pytest.mark.parametrize("method", list(IGNORE_REPS))
def test_ignore_reps_leaves_the_new_weight_without_replicates(api, api_bs, method):
    s = api_bs()
    rw = s.design.rep_wgts
    se_pw = _se(s)
    with no_warnings():
        out = IGNORE_REPS[method](s)
        se_nw = _se(out)
    assert out.design.wgt == "nw"
    assert out.design.rep_wgts is None
    assert all(c in out.data.columns for c in rw.columns)
    assert not any(c.startswith("nw") and c[2:].isdigit() for c in out.data.columns)
    # Taylor, and with the record credited: the same as on a design that never
    # had replicates.
    assert_allclose(se_nw, _se(IGNORE_REPS[method](api())))
    # The previous design keeps its replicates and comes back with them.
    assert out.design_history[-2].rep_wgts == rw
    with no_warnings():
        out.update_design(wgt="pw")
        se_back = _se(out)
    assert out.design.rep_wgts == rw
    assert out.design.wgt_adjustment is None
    assert_allclose(se_back, se_pw)


def test_ignore_reps_on_the_record_kinds_keeps_the_record(api_bs):
    with no_warnings():
        rk = api_bs().weighting.rake(controls=RAKE_CONTROLS, wgt_name="rk", ignore_reps=True)
        se = _se(rk)
    assert rk.design.wgt_adjustment.kind == "raking"
    taylor = _rake(_taylor(api_bs()))
    assert_allclose(se, _se(taylor))


def test_update_with_explicit_replicates_repairs_them():
    rw = BrrWgts(prefix="r", n_reps=4, wgt="old")
    d = Design(wgt="w").update(rep_wgts=rw)
    assert d.rep_wgts.wgt == "w"
    d2 = Design(wgt="w", rep_wgts=BrrWgts(prefix="r", n_reps=4)).update(wgt="v", rep_wgts=rw)
    assert d2.rep_wgts.wgt == "v"


def test_clone_with_replicates_pairs_them(small):
    s = Sample(small, Design(wgt="w"))
    c = s.clone(rep_wgts=BrrWgts(prefix="r", n_reps=4, wgt="elsewhere"))
    assert c.design.rep_wgts.wgt == "w"


# ---------------------------------------------------------------------------
# Construction errors
# ---------------------------------------------------------------------------


def test_design_rejects_replicates_of_another_weight():
    with pytest.raises(ValueError, match="rep_wgts go with weight 'v'"):
        Design(wgt="w", rep_wgts=BrrWgts(prefix="r", n_reps=4, wgt="v"))
    with pytest.raises(ValueError, match="rep_wgts go with weight 'v'"):
        Design(rep_wgts=BrrWgts(prefix="r", n_reps=4, wgt="v"))


def test_design_rejects_a_record_of_another_weight():
    rec = WgtAdjustment(kind="raking", prev_wgt="w0", new_wgt="v")
    with pytest.raises(ValueError, match="wgt_adjustment describes weight 'v'"):
        Design(wgt="w", wgt_adjustment=rec)
    with pytest.raises(ValueError, match="wgt_adjustment describes weight 'v'"):
        Design(wgt_adjustment=rec)
    with pytest.raises(ValueError, match="wgt_adjustment describes weight 'v'"):
        Design(wgt="w").update(wgt_adjustment=rec)


def test_repwgts_rejects_an_empty_weight():
    with pytest.raises(ValueError, match="'wgt'"):
        BrrWgts(prefix="r", n_reps=4, wgt="")


# ---------------------------------------------------------------------------
# columns()
# ---------------------------------------------------------------------------


def test_columns_taylor_order_and_tuples():
    d = Design(
        case_id="id",
        stratum=("s1", "s2"),
        wgt="w",
        prob="pr",
        psu=("p1", "p2"),
        ssu="u",
        pop_size="fpc",
    )
    assert d.columns() == ["id", "s1", "s2", "w", "pr", "p1", "p2", "u", "fpc"]
    assert d.columns() == d.specified_fields()


def test_columns_popsize():
    assert Design(wgt="w", pop_size=PopSize(psu="N1", ssu="N2")).columns() == ["w", "N1", "N2"]
    assert Design(wgt="w", pop_size=PopSize(psu="N1")).columns() == ["w", "N1"]


def test_columns_degenerate_designs():
    assert Design().columns() == []
    assert Design(wgt="w").columns() == ["w"]
    # One column in two roles is listed once, where it first appears.
    assert Design(stratum="s", psu="s", wgt="w", mos="w").columns() == ["s", "w"]


def test_columns_replicates_units_and_padding():
    rw = JackknifeWgts(prefix="r", n_reps=3, stratum="vs", psu=("vp", "s"))
    d = Design(stratum="s", wgt="w", rep_wgts=rw)
    assert d.columns() == ["s", "w", "r1", "r2", "r3", "vs", "vp"]
    data_cols = ["s", "w", "r01", "r02", "r03", "vs", "vp"]
    assert d.columns(data_columns=data_cols) == ["s", "w", "r01", "r02", "r03", "vs", "vp"]
    explicit = Design(wgt="w", rep_wgts=BrrWgts(prefix="r", n_reps=2, padding=3))
    assert explicit.columns() == ["w", "r001", "r002"]
    assert explicit.columns(data_columns=["w", "r1", "r2"]) == ["w", "r001", "r002"]


@pytest.mark.parametrize(
    "rec",
    [
        WgtAdjustment(kind="poststratification", prev_wgt="w0", new_wgt="w", cells=("c1",)),
        WgtAdjustment(kind="raking", prev_wgt="w0", new_wgt="w", cells=("c1", "c2")),
        WgtAdjustment(kind="calibration", prev_wgt="w0", new_wgt="w", aux=("a1", "a2")),
        WgtAdjustment(kind="standardization", prev_wgt="w0", new_wgt="w", cells=("c1",)),
        WgtAdjustment(kind="trimming", prev_wgt="w0", new_wgt="w"),
        WgtAdjustment(kind="normalization", prev_wgt="w0", new_wgt="w"),
        WgtAdjustment(kind="nonresponse", prev_wgt="w0", new_wgt="w"),
    ],
    ids=lambda r: r.kind,
)
def test_columns_every_record_kind(rec):
    d = Design(psu="p", wgt="w", wgt_adjustment=rec)
    expected = ["w", "p", "w0", *(rec.cells or ()), *(rec.aux or ())]
    assert d.columns() == expected
    # specified_fields is estimation's selection helper and does not grow.
    assert d.specified_fields() == ["w", "p"]


def test_columns_record_column_in_a_design_role():
    rec = WgtAdjustment(kind="trimming", prev_wgt="pw", new_wgt="w")
    d = Design(wgt="w", mos="pw", rep_wgts=BrrWgts(prefix="r", n_reps=2), wgt_adjustment=rec)
    assert d.columns() == ["w", "pw", "r1", "r2"]


def test_columns_of_a_live_sample_are_all_in_the_data(api_bs):
    cal = api_bs().weighting.calibrate(controls=GREG_CONTROLS, wgt_name="cal")
    cols = cal.design.columns(data_columns=cal.data.columns)
    assert set(cols) <= set(cal.data.columns)
    assert set(cal.design.wgt_adjustment.aux) <= set(cols)
    assert "pw" in cols and "cal" in cols and "cal1" in cols


# ---------------------------------------------------------------------------
# Sample(data, design) and set_design validate every design column
# ---------------------------------------------------------------------------


def test_sample_rejects_a_frame_missing_a_record_column(api):
    rk = _rake(api())
    for col in (rk.design.wgt_adjustment.cells[1], "pw"):
        with pytest.raises(ValueError, match="Design references columns not found in data") as ei:
            Sample(rk.data.drop(col), rk.design)
        assert col in str(ei.value)
        assert "wgt_adjustment (raking)" in str(ei.value)


def test_sample_rejects_a_frame_missing_an_aux_column(api):
    cal = api().weighting.calibrate(controls=GREG_CONTROLS, wgt_name="cal")
    with pytest.raises(ValueError, match="not found in data"):
        Sample(cal.data.drop(cal.design.wgt_adjustment.aux[0]), cal.design)


def test_sample_rejects_missing_replicate_units_and_columns(small):
    d = Design(wgt="w", rep_wgts=JackknifeWgts(prefix="r", n_reps=4, stratum="s", psu="p"))
    with pytest.raises(ValueError, match="separate question from the Design's stratum/psu"):
        Sample(small.drop("p"), d)
    with pytest.raises(ValueError, match=r"Design references columns not found in data: \['r3'\]"):
        Sample(small.drop("r3"), d)


def test_set_and_update_design_validate_record_columns(api):
    s = api()
    rec = WgtAdjustment(kind="poststratification", prev_wgt="pw", new_wgt="pw2", cells=("nope",))
    s2 = s.wrangling.mutate({"pw2": pl.col("pw") * 1.0})
    with pytest.raises(ValueError, match="nope"):
        s2.set_design(Design(wgt="pw2", psu="dnum", wgt_adjustment=rec))
    with pytest.raises(ValueError, match="nope"):
        s2.update_design(wgt="pw2", wgt_adjustment=rec)


# ---------------------------------------------------------------------------
# Wrangling: protection, force, rename
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["cells", "prev_wgt"])
def test_remove_record_column_needs_force(api, which):
    rk = _rake(api())
    col = rk.design.wgt_adjustment.cells[0] if which == "cells" else "pw"
    with pytest.raises(MethodError) as ei:
        rk.wrangling.remove_columns(col)
    assert ei.value.code == "DROP_PROTECTED_COLUMNS"
    assert col in str(ei.value)


def test_remove_aux_column_needs_force(api):
    cal = api().weighting.calibrate(controls=GREG_CONTROLS, wgt_name="cal")
    with pytest.raises(MethodError) as ei:
        cal.wrangling.drop(cal.design.wgt_adjustment.aux[1])
    assert ei.value.code == "DROP_PROTECTED_COLUMNS"


@pytest.mark.parametrize("op", ["keep_columns", "select"])
def test_keep_without_record_columns_needs_force(api, op):
    rk = _rake(api())
    with pytest.raises(MethodError) as ei:
        getattr(rk.wrangling, op)(["api00", "rk", "pw", "dnum", "fpc"])
    assert ei.value.code == "KEEP_DROPS_PROTECTED"
    assert rk.design.wgt_adjustment.cells[0] in str(ei.value)


def test_force_removing_a_cells_column_drops_the_record(api):
    rk = _rake(api())
    cells = rk.design.wgt_adjustment.cells[0]
    with pytest.warns(UserWarning, match=r"wgt_adjustment \(raking\)") as rec:
        out = rk.wrangling.remove_columns(cells, force=True)
    assert len(rec) == 1
    assert out.design.wgt == "rk"
    assert out.design.wgt_adjustment is None
    assert rk.design.wgt_adjustment is not None  # the parent is untouched
    with no_warnings():
        se = _se(out)
    assert_allclose(se, _fixed_weight_se(rk))


def test_force_keep_drops_the_record(api):
    rk = _rake(api())
    with pytest.warns(UserWarning, match="wgt_adjustment"):
        out = rk.wrangling.keep_columns(["api00", "rk", "pw", "dnum", "fpc"], force=True)
    assert out.design.wgt_adjustment is None
    with no_warnings():
        _se(out)


def test_force_removing_the_active_weight_cleans_everything_on_it(api_bs):
    rk = _rake(api_bs())
    with pytest.warns(UserWarning) as rec:
        out = rk.wrangling.remove_columns("rk", force=True)
    assert len(rec) == 1
    msg = str(rec[0].message)
    assert "wgt='rk'" in msg and "wgt_adjustment (raking)" in msg and "rep_wgts (" in msg
    assert out.design.wgt is None
    assert out.design.wgt_adjustment is None
    # The replicates went with 'rk': they leave the design, their columns stay.
    assert out.design.rep_wgts is None
    assert set(rk.design.rep_wgts.columns) <= set(out.data.columns)


def test_force_removing_a_replicate_column_drops_the_replicates(small):
    s = Sample(small, Design(wgt="w", rep_wgts=BrrWgts(prefix="r", n_reps=4)))
    with pytest.warns(UserWarning, match="rep_wgts"):
        out = s.wrangling.remove_columns("r2", force=True)
    assert out.design.rep_wgts is None


def test_force_removing_replicate_units(small):
    rw = JackknifeWgts(prefix="r", n_reps=4, stratum="s", psu="p", scale=0.75)
    s = Sample(small, Design(wgt="w", rep_wgts=rw))
    with pytest.raises(MethodError):
        s.wrangling.remove_columns("p")
    with pytest.warns(UserWarning, match=r"rep_wgts.psu='p'"):
        out = s.wrangling.remove_columns("p", force=True)
    assert out.design.rep_wgts.psu is None
    assert out.design.rep_wgts.stratum == "s"
    assert out.design.rep_wgts.wgt == "w"


def test_force_removing_popsize_columns(small):
    df = small.with_columns(pl.lit(100.0).alias("N1"), pl.lit(10.0).alias("N2"))
    s = Sample(df, Design(wgt="w", stratum="s", psu="p", pop_size=PopSize(psu="N1", ssu="N2")))
    with pytest.raises(MethodError):
        s.wrangling.remove_columns("N2")
    with pytest.warns(UserWarning, match="pop_size.ssu"):
        out = s.wrangling.remove_columns("N2", force=True)
    assert out.design.pop_size == PopSize(psu="N1")
    with pytest.warns(UserWarning, match="pop_size"):
        out2 = s.wrangling.remove_columns("N1", force=True)
    assert out2.design.pop_size is None


def test_removing_unprotected_columns_does_not_warn(api):
    rk = _rake(api())
    with no_warnings():
        out = rk.wrangling.remove_columns("api99")
    assert out.design == rk.design


def test_force_removing_stratum_rebuilds_internal_state(small):
    s = Sample(small, Design(wgt="w", stratum="s", psu="p"))
    with pytest.warns(UserWarning, match="stratum='s'"):
        out = s.wrangling.remove_columns("s", force=True)
    assert out.design.stratum is None
    assert out._internal_design["stratum"] is None
    assert not any(c.startswith("stratum_svy_internal") for c in out._data.columns)
    ref = Sample(small.drop("s"), Design(wgt="w", psu="p"))
    assert_allclose(_se(out, "y"), _se(ref, "y"))


@pytest.mark.parametrize("inplace", [False, True])
def test_rename_record_columns(api, inplace):
    rk = _rake(api())
    se_before = _se(rk)
    c1, c2 = rk.design.wgt_adjustment.cells
    renames = {"pw": "base", "rk": "raked", c1: "cell_a"}
    with no_warnings():
        out = rk.wrangling.rename_columns(renames, inplace=inplace)
        se_after = _se(out)
    rec = out.design.wgt_adjustment
    assert out.design.wgt == "raked"
    assert (rec.prev_wgt, rec.new_wgt, rec.cells) == ("base", "raked", ("cell_a", c2))
    assert_allclose(se_after, se_before)
    assert (out is rk) is inplace


def test_rename_aux_and_replicate_weight(api_bs):
    cal = api_bs().weighting.calibrate(controls=GREG_CONTROLS, wgt_name="cal")
    a0 = cal.design.wgt_adjustment.aux[0]
    se_before = _se(cal)
    with no_warnings():
        out = cal.wrangling.rename_columns({"cal": "w_cal", a0: "aux_zero"})
        se_after = _se(out)
    assert out.design.wgt_adjustment.aux[0] == "aux_zero"
    assert out.design.wgt_adjustment.new_wgt == "w_cal"
    assert out.design.rep_wgts.wgt == "w_cal"
    assert out.design.rep_wgts.prefix == "cal"
    assert_allclose(se_after, se_before)


def test_rename_replicate_units_and_popsize(small):
    df = small.with_columns(pl.lit(100.0).alias("N1"))
    rw = JackknifeWgts(prefix="r", n_reps=4, stratum="s", psu=("p", "s"), scale=0.75)
    s = Sample(df, Design(wgt="w", stratum="s", psu="p", pop_size=PopSize(psu="N1"), rep_wgts=rw))
    out = s.wrangling.rename_columns({"s": "S", "p": "P", "N1": "NN"})
    assert out.design.rep_wgts.stratum == "S"
    assert out.design.rep_wgts.psu == ("P", "S")
    assert out.design.pop_size == PopSize(psu="NN")


def test_clean_names_renames_the_record(api):
    rk = _rake(api())
    with no_warnings():
        out = rk.wrangling.clean_names()
    rec = out.design.wgt_adjustment
    assert set(out.design.columns(data_columns=out.data.columns)) <= set(out.data.columns)
    assert rec is not None and rec.kind == "raking"


def test_failed_inplace_rename_leaves_the_sample_untouched(small):
    s = Sample(small, Design(wgt="w", rep_wgts=BrrWgts(prefix="r", n_reps=4)))
    before = s.data.columns
    with pytest.raises(ValueError, match="Partial replicate-weight rename"):
        s.wrangling.rename_columns({"r1": "x1"}, inplace=True)
    assert s.data.columns == before
    assert s.design.rep_wgts.prefix == "r"


# ---------------------------------------------------------------------------
# update(wgt=X): the rule
# ---------------------------------------------------------------------------


def test_same_weight_and_structural_edits_keep_record_and_replicates(api_bs):
    rk = _rake(api_bs())
    rec, rw = rk.design.wgt_adjustment, rk.design.rep_wgts
    with no_warnings():
        rk.update_design(wgt="rk")
        rk.update_design(psu="dnum", pop_size="fpc", wr=True)
        rk.update_design(wr=False, stratum="stype")
    assert rk.design.wgt_adjustment == rec
    assert rk.design.rep_wgts == rw


def test_restore_after_trim_brings_back_record_and_replicates(api_bs):
    rk = _rake(api_bs())
    se_rk = _se(rk)
    rep_se_rk = _se(rk.use_weight("rk"))
    tr = rk.weighting.trim(upper=40.0, wgt_name="tr")
    assert tr.design.wgt_adjustment.kind == "trimming"
    with no_warnings():
        tr.update_design(wgt="rk")
        se_back = _se(tr)
    assert tr.design.wgt_adjustment == rk.design.wgt_adjustment
    assert tr.design.rep_wgts == rk.design.rep_wgts
    assert_allclose(se_back, se_rk)
    assert_allclose(se_back, rep_se_rk)


def test_restore_taylor_se_equals_before_trim(api):
    rk = _rake(api())
    se_rk, est_rk = _se(rk), _est(rk)
    tr = rk.weighting.trim(upper=40.0, wgt_name="tr")
    with no_warnings():
        tr.update_design(wgt="rk")
        assert_allclose(_se(tr), se_rk)
        assert_allclose(_est(tr), est_rk)


@pytest.mark.parametrize("reps", [False, True], ids=["taylor", "bootstrap"])
def test_chain_restores_each_earlier_weight_in_turn(api, api_bs, reps):
    s = api_bs() if reps else api()
    active = {"pw": (s.design, _se(s), _se(s, kind="total"))}
    s = _rake(s)
    active["rk"] = (s.design, _se(s), _se(s, kind="total"))
    s = s.weighting.trim(upper=40.0, wgt_name="tr")
    active["tr"] = (s.design, _se(s), _se(s, kind="total"))
    s = s.weighting.poststratify(STYPE_POP, cells="stype", wgt_name="ps")
    active["ps"] = (s.design, _se(s), _se(s, kind="total"))
    with no_warnings():
        for w in ("tr", "rk", "pw", "ps", "rk", "tr", "ps"):
            s.update_design(wgt=w)
            design, se_mean, se_total = active[w]
            assert s.design == design
            assert s.design.wgt_adjustment == design.wgt_adjustment
            assert s.design.rep_wgts == design.rep_wgts
            assert_allclose(_se(s), se_mean)
            assert_allclose(_se(s, kind="total"), se_total)
    if reps:
        assert s.design.rep_wgts.prefix == "ps"


def test_restore_picks_the_most_recent_design_on_that_weight(api):
    s = api()
    s = s.weighting.poststratify(STYPE_POP, cells="stype", wgt_name="ps")
    s.update_design(wgt="pw", psu="dnum", wr=True)
    s.update_design(wgt="ps")
    s.update_design(wgt="pw")
    assert s.design.wr is True


def test_unknown_weight_drops_record_silently_on_taylor(api):
    ps = api().weighting.poststratify(STYPE_POP, cells="stype", wgt_name="ps")
    ps = ps.wrangling.mutate({"mine": pl.col("ps") * 1.0})
    with no_warnings():
        ps.update_design(wgt="mine")
        se = _se(ps)
    assert ps.design.wgt_adjustment is None
    assert_allclose(se, _fixed_weight_se(ps))


def test_unknown_weight_resets_replicates_with_one_warning(small):
    s = Sample(
        small, Design(wgt="w", stratum="s", psu="p", rep_wgts=BrrWgts(prefix="r", n_reps=4))
    )
    with pytest.warns(UserWarning, match=r"go with weight 'w', not 'other_w'.*rep_wgts=") as rec:
        s.update_design(wgt="other_w")
    assert len(rec) == 1
    assert s.design.rep_wgts is None
    assert s.design.wgt == "other_w"


def test_unknown_weight_with_replicates_passed_keeps_them(small):
    rw = BrrWgts(prefix="r", n_reps=4)
    s = Sample(small, Design(wgt="w", rep_wgts=rw))
    # No replicate-reset warning; the Taylor-on-replicates finding is recorded
    # and raised once.
    with pytest.warns(SvyUserWarning) as rec:
        s.update_design(wgt="other_w", rep_wgts=rw)
        _se(s, "y")
    assert [str(r.message).split("]")[0] for r in rec] == ["[TAYLOR_WITHOUT_DESIGN"]
    assert rec[0].filename == __file__
    assert len(s.warnings.list(code="TAYLOR_WITHOUT_DESIGN")) == 1
    assert s.design.rep_wgts.wgt == "other_w"
    assert s.design.rep_wgts.prefix == "r"


def test_replicates_only_design_on_padded_columns(small):
    df = small.rename({f"r{i}": f"r0{i}" for i in range(1, 5)})
    s = Sample(df, Design(wgt="w", rep_wgts=BrrWgts(prefix="r", n_reps=4)))
    assert s.design.columns(data_columns=df.columns)[-4:] == ["r01", "r02", "r03", "r04"]
    with pytest.raises(MethodError):
        s.wrangling.remove_columns("r03")
    with pytest.warns(UserWarning, match="rep_wgts"):
        s.update_design(wgt="other_w")


def test_explicit_record_and_replicates_win_over_history(api_bs):
    rk = _rake(api_bs())
    tr = rk.weighting.trim(upper=40.0, wgt_name="tr")
    own = WgtAdjustment(kind="trimming", prev_wgt="pw", new_wgt="rk")
    with no_warnings():
        tr.update_design(wgt="rk", wgt_adjustment=own)
    assert tr.design.wgt_adjustment == own
    assert tr.design.rep_wgts == rk.design.rep_wgts  # still from history
    rw = BootstrapWgts(prefix="pw", n_reps=20)
    with no_warnings():
        tr.update_design(wgt="tr", rep_wgts=rw)
    assert tr.design.rep_wgts.prefix == "pw"
    assert tr.design.wgt_adjustment.kind == "trimming"  # from history


def test_history_entry_missing_columns_is_not_restored(api_bs):
    rk = _rake(api_bs())
    tr = rk.weighting.trim(upper=40.0, wgt_name="tr")
    # The raking snapshot is not the current design's, so it is not protected.
    tr = tr.wrangling.remove_columns(rk.design.wgt_adjustment.cells[0])
    with pytest.warns(UserWarning, match="rep_wgts="):
        tr.update_design(wgt="rk")
    assert tr.design.wgt_adjustment is None
    assert tr.design.rep_wgts is None


def test_force_remove_then_update_back(api):
    rk = _rake(api())
    tr = rk.weighting.trim(upper=40.0, wgt_name="tr")
    with pytest.warns(UserWarning, match="wgt_adjustment"):
        tr = tr.wrangling.remove_columns("rk", force=True)
    assert tr.design.wgt == "tr" and tr.design.wgt_adjustment is None
    with no_warnings():
        tr.update_design(wgt="pw")
        _se(tr)
    assert tr.design.wgt_adjustment is None
    with no_warnings():
        tr.update_design(wgt="tr")
    # The trimming record read 'rk', which is gone.
    assert tr.design.wgt_adjustment is None


def test_rename_carries_into_history(api):
    rk = _rake(api())
    se_rk = _se(rk)
    tr = rk.weighting.trim(upper=40.0, wgt_name="tr")
    c1 = rk.design.wgt_adjustment.cells[0]
    with no_warnings():
        tr = tr.wrangling.rename_columns({"rk": "raked", c1: "rake_cell"})
        tr.update_design(wgt="raked")
        se_back = _se(tr)
    rec = tr.design.wgt_adjustment
    assert rec.kind == "raking"
    assert rec.new_wgt == "raked" and rec.cells[0] == "rake_cell"
    assert all(c in tr.data.columns for c in tr.design.columns(data_columns=tr.data.columns))
    assert_allclose(se_back, se_rk)


def test_history_entry_a_rename_cannot_reach_is_not_restored(api_bs):
    s = api_bs()
    s = s.wrangling.mutate({"mine": pl.col("pw") * 1.0})
    with pytest.warns(UserWarning, match="rep_wgts="):
        s.update_design(wgt="mine")
    assert s.design.rep_wgts is None
    # A partial rename of the earlier design's replicates cannot be carried
    # into history; that entry keeps its old names and is not restored.
    with no_warnings():
        s = s.wrangling.rename_columns({"pw1": "x1"})
        s.update_design(wgt="pw")
    assert s.design.rep_wgts is None
    assert s.design_history[0].rep_wgts.prefix == "pw"


def test_bare_design_update_rule():
    rec = WgtAdjustment(kind="raking", prev_wgt="w0", new_wgt="w", cells=("c",))
    d = Design(wgt="w", psu="p", rep_wgts=BrrWgts(prefix="r", n_reps=4), wgt_adjustment=rec)
    with no_warnings():
        assert d.update(wgt="w").wgt_adjustment == rec
        assert d.update(psu="q").rep_wgts == d.rep_wgts
    with pytest.warns(UserWarning, match="go with weight 'w'") as caught:
        moved = d.update(wgt="w0")
    assert len(caught) == 1
    assert moved.wgt_adjustment is None and moved.rep_wgts is None
    with no_warnings():
        kept = d.update(wgt="w0", rep_wgts=d.rep_wgts)
    assert kept.rep_wgts.wgt == "w0" and kept.wgt_adjustment is None


def test_use_weight_follows_the_history_rule(api):
    rk = _rake(api())
    se_rk = _se(rk)
    tr = rk.weighting.trim(upper=40.0, wgt_name="tr")
    with no_warnings():
        back = tr.use_weight("rk")
        base = tr.use_weight("pw")
        assert_allclose(_se(back), se_rk)
        _se(base)
    assert back.design.wgt_adjustment.kind == "raking"
    assert base.design.wgt_adjustment is None
    assert tr.design.wgt == "tr"


def test_weighting_on_a_restored_weight(api_bs):
    s = _rake(api_bs())
    s = s.weighting.trim(upper=40.0, wgt_name="tr")
    with no_warnings():
        s.update_design(wgt="rk")
        s = s.weighting.poststratify(STYPE_POP, cells="stype", wgt_name="ps")
    assert s.design.wgt_adjustment.prev_wgt == "rk"
    assert s.design.rep_wgts.wgt == "ps"
    assert s.design.rep_wgts.prefix == "ps"


# ---------------------------------------------------------------------------
# State changes between estimates
# ---------------------------------------------------------------------------


def test_update_design_between_estimates(api):
    ps = api().weighting.poststratify(STYPE_POP, cells="stype", wgt_name="ps")
    se_ps = _se(ps)
    se_pw = _se(api())
    with no_warnings():
        ps.update_design(wgt="pw")
        assert_allclose(_se(ps), se_pw)
        ps.update_design(wgt="ps")
        assert_allclose(_se(ps), se_ps)


def test_parent_and_fork_do_not_share_state(api_bs):
    rk = _rake(api_bs())
    se_rk = _se(rk)
    c1 = rk.design.wgt_adjustment.cells[0]
    with pytest.warns(UserWarning):
        fork = rk.wrangling.remove_columns(c1, force=True)
    with no_warnings():
        se_fork = _se(fork)
        assert_allclose(_se(rk), se_rk)
        renamed = rk.wrangling.rename_columns({"rk": "raked"})
        assert_allclose(_se(renamed), se_rk)
        assert_allclose(_se(fork), se_fork)
        assert_allclose(_se(rk), se_rk)
    assert rk.design.wgt == "rk" and rk.design.wgt_adjustment.cells[0] == c1
    assert fork.design.wgt_adjustment is None
    assert rk.design_history[-2].wgt == "pw"
    assert renamed.design_history[-1].wgt == "raked"


def test_inplace_wrangling_between_estimates(api):
    rk = _rake(api())
    se_rk = _se(rk)
    v0 = rk._data_version
    with no_warnings():
        rk.wrangling.rename_columns({"rk": "raked"}, inplace=True)
        assert rk._data_version != v0
        assert_allclose(_se(rk), se_rk)
        rk.wrangling.filter_records(pl.col("api00") > 0, inplace=True)
        assert_allclose(_se(rk), se_rk)
    with pytest.warns(UserWarning, match="wgt_adjustment"):
        rk.wrangling.remove_columns("pw", force=True, inplace=True)
    with no_warnings():
        assert_allclose(_se(rk), _fixed_weight_se(rk))


def test_mutate_cast_recode_keep_the_record(api):
    rk = _rake(api())
    se_rk = _se(rk)
    with no_warnings():
        out = rk.wrangling.mutate({"api00_x": pl.col("api00") * 2})
        out = out.wrangling.cast("api99", pl.Float64)
        out = out.wrangling.recode("stype", {"Elem": ["E"]})
        assert_allclose(_se(out), se_rk)
    assert out.design.wgt_adjustment == rk.design.wgt_adjustment


def test_join_keeps_the_record(api):
    rk = _rake(api())
    se_rk = _se(rk)
    extra = rk.data.select("snum", (pl.col("api00") * 0 + 1).alias("joined_flag"))
    with no_warnings():
        out = rk.wrangling.join(extra, on="snum")
        assert_allclose(_se(out), se_rk)
    assert out.design.wgt_adjustment == rk.design.wgt_adjustment


def test_record_reset_is_seen_by_the_next_estimate(api):
    """A record dropped between two estimates is not served from a cache."""
    rk = _rake(api())
    se_rk = _se(rk)
    rk.update_design(wgt_adjustment=None)
    se_fixed = _se(rk)
    assert not np.isclose(se_fixed, se_rk)
    assert_allclose(se_fixed, _fixed_weight_se(rk))
    rec = rk.design_history[-2].wgt_adjustment
    rk.update_design(wgt_adjustment=rec)
    assert_allclose(_se(rk), se_rk)


def test_replicate_se_after_restore(small):
    rw = BrrWgts(prefix="r", n_reps=4)
    s = Sample(small, Design(wgt="w", rep_wgts=rw))
    se_w = _se(s, "y")
    s2 = s.weighting.poststratify({"a": 30.0, "b": 20.0}, cells="g", wgt_name="ps")
    se_ps = _se(s2, "y")
    # Each update is a new state, so the Taylor-on-replicates finding is new too.
    with no_warnings(allow="TAYLOR_WITHOUT_DESIGN"):
        s2.update_design(wgt="w")
        assert_allclose(_se(s2, "y"), se_w)
        s2.update_design(wgt="ps")
        assert_allclose(_se(s2, "y"), se_ps)
    assert s2.design.rep_wgts.prefix == "ps"
    assert s2.warnings.list(code="TAYLOR_WITHOUT_DESIGN")


def test_combine_panel_replicates_pair_with_the_combined_weight():
    import svy

    base = pl.DataFrame(
        {
            "id": list(range(12)),
            "s": [1] * 4 + [2] * 4 + [3] * 4,
            "p": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            "w": [1.0 + i / 10 for i in range(12)],
            "y": [float(i) for i in range(12)],
        }
    )
    s1 = Sample(base, Design(case_id="id", wgt="w", stratum="s", psu="p"))
    s1 = s1.weighting.create_jk_wgts()
    reps = s1.data.select(["id", *s1.design.rep_wgts.columns])
    df2 = base.with_columns(pl.col("y") + 1).join(reps, on="id")
    s2 = Sample(df2, s1.design)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = svy.combine_samples([s1, s2], kind="panel", case_id="id")
    assert c.design.rep_wgts.wgt == c.design.wgt
    assert msgspec.structs.replace(c.design.rep_wgts, wgt="w") == s1.design.rep_wgts


# ---------------------------------------------------------------------------
# More edge cases
# ---------------------------------------------------------------------------


def test_rename_record_column_then_restore_under_the_new_name(api):
    ps = api().weighting.poststratify(STYPE_POP, cells="stype", wgt_name="ps")
    se_ps = _se(ps)
    cell = ps.design.wgt_adjustment.cells[0]
    with no_warnings():
        ps.update_design(wgt="pw")
        ps = ps.wrangling.rename_columns({cell: "ps_cells", "ps": "post"})
        ps.update_design(wgt="post")
        se_back = _se(ps)
    rec = ps.design.wgt_adjustment
    assert (rec.new_wgt, rec.cells) == ("post", ("ps_cells",))
    assert_allclose(se_back, se_ps)


def test_remove_history_only_column_then_restore_is_first_time_like(api):
    rk = _rake(api())
    tr = rk.weighting.trim(upper=40.0, wgt_name="tr")
    cells = rk.design.wgt_adjustment.cells
    with no_warnings():
        tr = tr.wrangling.remove_columns(list(cells))
        tr.update_design(wgt="rk")
        se = _se(tr)
    assert tr.design.wgt == "rk"
    assert tr.design.wgt_adjustment is None
    assert_allclose(se, _fixed_weight_se(tr))


def test_force_remove_active_weight_then_update_back(api_bs):
    s = api_bs()
    se_pw = _se(s)
    rw = s.design.rep_wgts
    rk = _rake(s)
    with pytest.warns(UserWarning, match="wgt='rk'"):
        rk = rk.wrangling.remove_columns("rk", force=True)
    assert rk.design.wgt is None and rk.design.rep_wgts is None
    with no_warnings():
        rk.update_design(wgt="pw")
        se_back = _se(rk)
    assert rk.design.rep_wgts == rw
    assert rk.design.wgt_adjustment is None
    assert_allclose(se_back, se_pw)
    before, n_hist = rk.design, len(rk.design_history)
    with no_warnings(), pytest.raises(ValueError, match="not found in data"):
        rk.update_design(wgt="rk")
    assert rk.design == before and len(rk.design_history) == n_hist


def test_failed_design_replacement_leaves_the_sample_untouched(api):
    rk = _rake(api())
    before, data_cols, n_hist = rk.design, rk.data.columns, len(rk.design_history)
    bad = WgtAdjustment(kind="raking", prev_wgt="pw", new_wgt="rk", cells=("nope",))
    with pytest.raises(ValueError, match="nope"):
        rk.update_design(wgt_adjustment=bad)
    with pytest.raises((ValueError, KeyError), match="nope"):
        rk.set_design(Design(wgt="rk", stratum="nope"))
    assert rk.design == before
    assert rk.data.columns == data_cols
    assert len(rk.design_history) == n_hist
    with no_warnings():
        assert_allclose(_se(rk), _se(_rake(api())))


def test_replicates_built_on_the_design_units_dedupe_in_columns(api):
    s = api().weighting.create_jk_wgts()
    rw = s.design.rep_wgts
    assert rw.psu == "dnum"
    cols = s.design.columns(data_columns=s.data.columns)
    assert cols.count("dnum") == 1
    assert cols == ["pw", "dnum", "fpc", *rw.columns_from_data(s.data.columns)]


def test_record_with_prev_equal_to_new_weight(api):
    """An in-place trim (adjust(trimming=...)) records prev_wgt == new_wgt."""
    rec = WgtAdjustment(kind="trimming", prev_wgt="pw", new_wgt="pw")
    d = Design(wgt="pw", psu="dnum", pop_size="fpc", wgt_adjustment=rec)
    assert d.columns() == ["pw", "dnum", "fpc"]
    s = _with_resp(api())
    with no_warnings():
        out = s.weighting.adjust(
            resp_status="resp",
            resp_mapping={"rr": 1, "nr": 2},
            cells="stype",
            respondents_only=False,
            trimming=__import__("svy").weighting.types.TrimConfig(upper=100.0),
        )
    rec = out.design.wgt_adjustment
    assert rec.prev_wgt == rec.new_wgt == out.design.wgt
    assert out.design.columns().count(out.design.wgt) == 1


PROVENANCE = {
    "trimming": lambda s: s.weighting.trim(upper=40.0, wgt_name="nw"),
    "normalization": lambda s: s.weighting.normalize(controls=6194.0, wgt_name="nw"),
    "nonresponse": lambda s: _with_resp(s).weighting.adjust(
        resp_status="resp",
        resp_mapping={"rr": 1, "nr": 2},
        cells="stype",
        wgt_name="nw",
        respondents_only=False,
    ),
}


@pytest.mark.parametrize("kind", list(PROVENANCE))
def test_provenance_only_records(api, kind):
    with no_warnings():
        out = PROVENANCE[kind](api())
        se_nw = _se(out)
    rec = out.design.wgt_adjustment
    assert rec.kind == kind and not rec.is_variance_consumed
    assert out.design.columns() == ["nw", "dnum", "fpc", "pw"]
    # Provenance-only: estimation treats the weights as fixed.
    assert_allclose(se_nw, _fixed_weight_se(out))
    with pytest.raises(MethodError) as ei:
        out.wrangling.remove_columns("pw")
    assert ei.value.code == "DROP_PROTECTED_COLUMNS"
    with no_warnings():
        out.update_design(wgt="pw")
        assert out.design.wgt_adjustment is None
        out.update_design(wgt="nw")
        assert_allclose(_se(out), se_nw)
    assert out.design.wgt_adjustment == rec
    with pytest.warns(UserWarning, match=f"wgt_adjustment \\({kind}\\)"):
        forced = out.wrangling.remove_columns("pw", force=True)
    assert forced.design.wgt_adjustment is None and forced.design.wgt == "nw"


def test_replicates_only_design_unknown_weight(small):
    rw = BrrWgts(prefix="r", n_reps=4)
    s = Sample(small, Design(wgt="w", rep_wgts=rw))
    se_w = _se(s, "y")
    with pytest.warns(UserWarning, match="go with weight 'w'") as caught:
        s.update_design(wgt="other_w")
    assert len(caught) == 1
    assert s.design.rep_wgts is None
    with no_warnings(allow="TAYLOR_WITHOUT_DESIGN"):
        se_other = _se(s, "y")
        s.update_design(wgt="w")
        assert s.design.rep_wgts.wgt == "w"
        assert_allclose(_se(s, "y"), se_w)
        s.update_design(wgt="other_w", rep_wgts=rw)
        assert s.design.rep_wgts.wgt == "other_w"
    ref = Sample(small, Design(wgt="other_w"))
    assert_allclose(se_other, _se(ref, "y"))


@pytest.mark.parametrize("padding", [None, 2])
def test_padded_replicates_columns_and_pairing(small, padding):
    df = small.rename({f"r{i}": f"r0{i}" for i in range(1, 5)})
    rw = BrrWgts(prefix="r", n_reps=4, padding=padding)
    s = Sample(df, Design(wgt="w", rep_wgts=rw))
    assert s.design.rep_wgts.wgt == "w"
    assert s.design.columns(data_columns=df.columns) == ["w", "r01", "r02", "r03", "r04"]
    if padding is not None:
        assert s.design.columns() == ["w", "r01", "r02", "r03", "r04"]
    else:
        assert s.design.columns() == ["w", "r1", "r2", "r3", "r4"]
    with pytest.raises(MethodError):
        s.wrangling.remove_columns("r02")
    with no_warnings():
        out = s.wrangling.rename_columns({"w": "base"})
    assert out.design.rep_wgts.wgt == "base"
    with pytest.raises(ValueError, match="not found in data"):
        Sample(df.drop("r04"), Design(wgt="w", rep_wgts=rw))


def test_reset_warning_points_at_the_callers_line(small):
    d = Design(wgt="w", rep_wgts=BrrWgts(prefix="r", n_reps=4))
    s = Sample(small, d)
    with pytest.warns(UserWarning, match="go with weight") as caught:
        d.update(wgt="other_w")
        s.update_design(wgt="other_w")
        Sample(small, d).use_weight("other_w")
    assert len(caught) == 3
    assert all(w.filename == __file__ for w in caught)


def test_force_removal_warning_points_at_the_callers_line(api):
    rk = _rake(api())
    with pytest.warns(UserWarning, match="Removed from the design") as caught:
        rk.wrangling.remove_columns(rk.design.wgt_adjustment.cells[0], force=True)
    assert caught[0].filename == __file__


def test_replicates_do_not_attach_to_a_new_weight_after_their_weight_is_removed(api_bs):
    with pytest.warns(UserWarning, match="rep_wgts"):
        rk = _rake(api_bs()).wrangling.remove_columns("rk", force=True)
    rk.wrangling.mutate({"w_new": pl.col("pw") * 2}, inplace=True)
    with no_warnings():
        rk.update_design(wgt="w_new")
    assert rk.design.rep_wgts is None
