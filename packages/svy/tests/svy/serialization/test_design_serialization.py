# tests/svy/serialization/test_design_serialization.py
"""
svy.serialize for designs: DesignData, to_design, lossless round trips.

The acceptance test is on estimates: a Sample rebuilt from the saved design
and its data gives the live estimates exactly, SEs included, after every
adjustment svy records.
"""

from __future__ import annotations

import json
import warnings

from pathlib import Path

import msgspec
import polars as pl
import pytest

import svy

from svy.core.design import PopSize, WgtAdjustment
from svy.core.repwgts import BootstrapWgts, BrrWgts, JackknifeWgts, SdrWgts
from svy.errors import SerializationError
from svy.serialize import (
    DESIGN_SCHEMA_VERSION,
    BootstrapWgtsData,
    BrrWgtsData,
    DesignData,
    JackknifeWgtsData,
    PopSizeData,
    SdrWgtsData,
    WgtAdjustmentData,
    from_json,
    serialize,
    to_design,
    to_dict,
    to_json,
)


DATA_DIR = Path(__file__).resolve().parents[2] / "test_data"
STYPE_POP = {"E": 4421.0, "H": 755.0, "M": 1018.0}
SCHWIDE_POP = {"No": 1000.0, "Yes": 5194.0}
GREG_CONTROLS = {"one": 6194.0, "api99": 3914069.0}


def _round_trip(design: svy.Design) -> svy.Design:
    return to_design(from_json(to_json(design)))


# ---------------------------------------------------------------------------
# Every field, every variant
# ---------------------------------------------------------------------------

DESIGNS = {
    "empty": svy.Design(),
    "weight_only": svy.Design(wgt="w"),
    "full_taylor": svy.Design(
        case_id="id",
        wave="wave",
        stratum=("region", "urban"),
        wgt="w",
        prob="p",
        hit="h",
        mos="size",
        psu=("district", "village"),
        ssu="hh",
        pop_size=PopSize(psu="N_psu", ssu="N_ssu"),
        wr=True,
    ),
    "str_pop_size": svy.Design(stratum="s", psu="p", wgt="w", pop_size="fpc"),
    "bootstrap": svy.Design(
        wgt="w",
        rep_wgts=BootstrapWgts(prefix="bw", n_reps=4, kind="poisson", df=3.5, padding=2),
    ),
    "jackknife_jkn": svy.Design(
        wgt="w",
        stratum="s",
        psu="p",
        rep_wgts=JackknifeWgts(
            prefix="jk",
            n_reps=4,
            kind="jkn",
            rep_coefs=(0.5, 0.5, 0.75, 0.75),
            stratum="s",
            psu=("p", "q"),
        ),
    ),
    "jackknife_no_kind": svy.Design(wgt="w", rep_wgts=JackknifeWgts(prefix="jk", n_reps=3)),
    "brr_fay": svy.Design(
        wgt="w", rep_wgts=BrrWgts(prefix="r", n_reps=4, fay_coef=0.3, scale=0.2, padding=0)
    ),
    "sdr": svy.Design(
        wgt="w", rep_wgts=SdrWgts(prefix="sd", n_reps=4, scale=(1.0, 1.0, 2.0, 2.0))
    ),
    "record_raking": svy.Design(
        wgt="rk",
        wgt_adjustment=WgtAdjustment(
            kind="raking", prev_wgt="w", new_wgt="rk", cells=("__svy_cells_a", "__svy_cells_b")
        ),
    ),
    "record_calibration": svy.Design(
        wgt="cal",
        wgt_adjustment=WgtAdjustment(
            kind="calibration", prev_wgt="w", new_wgt="cal", aux=("one", "x"), pins_total=False
        ),
    ),
    "record_and_replicates": svy.Design(
        wgt="ps",
        psu="p",
        rep_wgts=BrrWgts(prefix="ps", n_reps=4, wgt="ps"),
        wgt_adjustment=WgtAdjustment(
            kind="poststratification", prev_wgt="w", new_wgt="ps", cells=("__svy_cells_ps",)
        ),
    ),
}


@pytest.mark.parametrize("design", DESIGNS.values(), ids=DESIGNS.keys())
def test_round_trip_is_lossless(design):
    back = _round_trip(design)
    assert back == design
    assert hash(back) == hash(design)
    assert back.columns() == design.columns()
    # Types survive, not only equality: tuples stay tuples, PopSize stays PopSize.
    for field in ("stratum", "psu", "ssu", "pop_size", "rep_wgts", "wgt_adjustment"):
        assert type(getattr(back, field)) is type(getattr(design, field))


@pytest.mark.parametrize("design", DESIGNS.values(), ids=DESIGNS.keys())
def test_struct_round_trip_and_to_dict(design):
    data = serialize(design)
    assert isinstance(data, DesignData)
    assert to_design(data) == design
    assert json.loads(to_json(design)) == to_dict(design)


def test_payload_is_tagged_and_versioned():
    raw = json.loads(to_json(DESIGNS["brr_fay"]))
    assert raw["kind"] == "design"
    assert raw["schema_version"] == DESIGN_SCHEMA_VERSION == "svy-design/0.1"
    assert raw["rep_wgts"]["method"] == "BRR"
    assert raw["rep_wgts"]["fay_coef"] == 0.3
    assert raw["rep_wgts"]["wgt"] == "w"


def test_scale_and_coefficients_are_stored_per_replicate():
    raw = json.loads(to_json(DESIGNS["brr_fay"]))
    assert raw["rep_wgts"]["scale"] == [0.2] * 4


@pytest.mark.parametrize(
    "live,data",
    [
        (BootstrapWgts, BootstrapWgtsData),
        (JackknifeWgts, JackknifeWgtsData),
        (BrrWgts, BrrWgtsData),
        (SdrWgts, SdrWgtsData),
        (WgtAdjustment, WgtAdjustmentData),
    ],
)
def test_saved_structs_mirror_every_live_field(live, data):
    """A field added to the live class must be added to the saved form too."""
    assert set(live.__struct_fields__) == set(data.__struct_fields__)


def test_pop_size_mirrors_every_field():
    assert set(PopSize._fields) == set(PopSizeData.__struct_fields__)


def test_design_fields_are_all_saved():
    from svy.core.design import _FIELDS

    saved = set(DesignData.__struct_fields__) - {"kind", "schema_version"}
    assert saved == {*_FIELDS, "rep_wgts", "wgt_adjustment", "singleton", "parts"}


def test_unknown_fields_are_ignored():
    """A payload from a later 0.x with an added field still loads."""
    raw = json.loads(to_json(DESIGNS["weight_only"]))
    raw["added_later"] = {"x": 1}
    assert to_design(from_json(json.dumps(raw).encode())) == DESIGNS["weight_only"]


def test_to_design_rejects_other_payloads():
    with pytest.raises(SerializationError) as exc:
        to_design({"wgt": "w"})  # type: ignore[arg-type]
    assert exc.value.code == "PAYLOAD_NOT_A_DESIGN"


def test_a_mismatched_payload_fails_like_a_mismatched_design():
    raw = json.loads(to_json(DESIGNS["record_raking"]))
    raw["wgt"] = "other"
    with pytest.raises(ValueError, match="wgt_adjustment describes"):
        to_design(from_json(json.dumps(raw).encode()))


def test_designs_are_not_results():
    """No table, and not among the result serializers."""
    from svy.serialize.serializers import _SERIALIZERS

    assert svy.Design not in _SERIALIZERS


# ---------------------------------------------------------------------------
# Acceptance: estimates on a restored design equal the live ones
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def apiclus1() -> pl.DataFrame:
    df = pl.read_csv(DATA_DIR / "apiclus1.csv", null_values=["NA"])
    return df.with_columns(pl.lit(1.0).alias("one"))


@pytest.fixture(scope="module")
def base(apiclus1) -> svy.Sample:
    return svy.Sample(apiclus1, svy.Design(wgt="pw", psu="dnum", pop_size="fpc"))


def _adjusted(base: svy.Sample) -> dict[str, svy.Sample]:
    w = base.weighting
    jk = w.create_jk_wgts()
    return {
        "poststratify": w.poststratify(STYPE_POP, cells="stype"),
        "rake": w.rake(controls={"stype": STYPE_POP, "sch.wide": SCHWIDE_POP}, tol=1e-12),
        "calibrate": w.calibrate(controls=GREG_CONTROLS),
        "standardize": w.standardize("stype", shares=STYPE_POP, by="sch.wide"),
        "trim": w.trim(upper=40.0),
        "rake_then_trim": w.rake(
            controls={"stype": STYPE_POP, "sch.wide": SCHWIDE_POP}, tol=1e-12
        ).weighting.trim(upper=40.0),
        "jackknife_poststratify": jk.weighting.poststratify(STYPE_POP, cells="stype"),
        "bootstrap_rake": w.create_bs_wgts(n_reps=20, rstate=1).weighting.rake(
            controls={"stype": STYPE_POP, "sch.wide": SCHWIDE_POP}, tol=1e-12
        ),
    }


def _estimates(s: svy.Sample, method: str | None = None) -> list[tuple]:
    e = s.estimation
    out = []
    for r in (
        e.mean("api00", method=method),
        e.total("enroll", method=method),
        e.mean("api00", by="sch.wide", method=method),
        e.prop("stype", method=method),
    ):
        # Domain rows come back in no fixed order; compare them by key.
        rows = [(p.by_level, p.y_level, p.est, p.se, p.lci, p.uci) for p in r.estimates]
        out += sorted(rows, key=lambda row: (str(row[0]), str(row[1])))
    return out


@pytest.mark.parametrize(
    "case",
    [
        "poststratify",
        "rake",
        "calibrate",
        "standardize",
        "trim",
        "rake_then_trim",
        "jackknife_poststratify",
        "bootstrap_rake",
    ],
)
def test_restored_design_gives_the_live_estimates(base, case):
    live = _adjusted(base)[case]
    restored = svy.Sample(live.data, _round_trip(live.design))
    assert restored.design == live.design
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _estimates(restored) == _estimates(live)
        if live.design.rep_wgts is not None:
            assert _estimates(restored, "replication") == _estimates(live, "replication")


def test_the_restored_design_still_needs_its_columns(base):
    ps = base.weighting.poststratify(STYPE_POP, cells="stype")
    design = _round_trip(ps.design)
    cells = design.wgt_adjustment.cells[0]
    with pytest.raises(ValueError, match="not found in data"):
        svy.Sample(ps.data.drop(cells), design)


def test_round_trip_after_renames_and_restore(base):
    rk = base.weighting.rake(controls={"stype": STYPE_POP, "sch.wide": SCHWIDE_POP}, tol=1e-12)
    rk = rk.weighting.create_jk_wgts() if rk.design.rep_wgts is None else rk
    renamed = rk.wrangling.rename_columns({"pw": "pw_base"})
    renamed = renamed.wrangling.rename_rep_wgts({renamed.design.rep_wgts.prefix: "rk_rep"})
    restored = svy.Sample(renamed.data, _round_trip(renamed.design))
    assert restored.design == renamed.design
    assert _estimates(restored) == _estimates(renamed)
    assert _estimates(restored, "replication") == _estimates(renamed, "replication")


def test_the_history_is_not_part_of_the_design(base):
    """A saved design is one design; lineage is the caller's to keep."""
    ps = base.weighting.poststratify(STYPE_POP, cells="stype")
    restored = svy.Sample(ps.data, _round_trip(ps.design))
    assert len(ps.design_history) > 1
    assert restored.design_history == (restored.design,)


def test_payload_decodes_with_msgspec_directly():
    """The saved form is plain msgspec: typed decoding needs no svy helper."""
    js = to_json(DESIGNS["record_and_replicates"])
    assert msgspec.json.decode(js, type=DesignData) == serialize(DESIGNS["record_and_replicates"])


def test_a_design_payload_has_no_table():
    from svy.serialize import to_polars

    with pytest.raises(SerializationError) as exc:
        to_polars(serialize(DESIGNS["brr_fay"]))
    assert exc.value.code == "PAYLOAD_NO_TABLE"
