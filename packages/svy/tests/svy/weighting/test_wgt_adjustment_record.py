# tests/svy/weighting/test_wgt_adjustment_record.py
"""The weight-adjustment record and the design-snapshot lineage.

``Design.wgt_adjustment`` holds ONE record describing how the current weights
were last produced; each weighting method replaces it. Full lineage is not a
weight log but the chain of Design snapshots on the Sample, and since every
snapshot carries its own record, that chain is the chain of adjustments -- which
is what makes keeping a single record lossless rather than lossy.
"""

import polars as pl
import pytest

from svy import Design, Sample, col
from svy.core.design import WgtAdjustment
from svy.core.terms import Cat


@pytest.fixture
def sample():
    df = pl.DataFrame(
        {
            "w": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "g": ["a", "a", "b", "b", "c", "c"],
            "status": ["rr", "rr", "rr", "nr", "rr", "rr"],
        }
    )
    return Sample(df, Design(wgt="w"))


CONTROLS = {"a": 10.0, "b": 20.0, "c": 30.0}


def _run(sample, method):
    return {
        "poststratify": lambda: sample.weighting.poststratify(CONTROLS, cells="g"),
        "rake": lambda: sample.weighting.rake(controls={"g": CONTROLS}),
        "calibrate": lambda: sample.weighting.calibrate(controls={Cat("g"): CONTROLS}),
        "standardize": lambda: sample.weighting.standardize("g", shares={"a": 1, "b": 1, "c": 1}),
        "normalize": lambda: sample.weighting.normalize(100.0),
        "adjust": lambda: sample.weighting.adjust("status", "g", respondents_only=False),
        "trim": lambda: sample.weighting.trim(upper=5.0, min_cell_size=1),
    }[method]()


@pytest.mark.parametrize(
    ("method", "kind"),
    [
        ("poststratify", "poststratification"),
        ("rake", "raking"),
        ("calibrate", "calibration"),
        ("standardize", "standardization"),
        ("normalize", "normalization"),
        ("adjust", "nonresponse"),
        ("trim", "trimming"),
    ],
)
def test_every_method_records_its_technique(sample, method, kind):
    """kind names the technique, not the producing method.

    Matches the RepWgts tags, where create_jk_wgts yields method="jackknife".
    """
    rec = _run(sample, method).design.wgt_adjustment
    assert rec.kind == kind
    assert rec.prev_wgt == "w"
    assert rec.new_wgt == _run(sample, method).design.wgt


@pytest.mark.parametrize(
    ("method", "consumed"),
    [
        ("poststratify", True),
        ("rake", True),
        ("calibrate", True),
        ("standardize", True),
        ("normalize", False),
        ("adjust", False),
        ("trim", False),
    ],
)
def test_variance_consumed_split(sample, method, consumed):
    """Only the four that pin population quantities are centred for."""
    rec = _run(sample, method).design.wgt_adjustment
    assert rec.is_variance_consumed is consumed
    # Consumed kinds must carry the structure the sweep needs.
    assert bool(rec.cells or rec.aux) is consumed


def test_controls_pin_the_total_but_shares_do_not(sample):
    """The distinction that decides how many constraints the sweep removes."""
    by_controls = sample.weighting.poststratify(CONTROLS, cells="g")
    by_shares = sample.weighting.poststratify(shares=CONTROLS, cells="g")
    assert by_controls.design.wgt_adjustment.pins_total is True
    assert by_shares.design.wgt_adjustment.pins_total is False


def test_standardize_does_not_pin_domain_totals(sample):
    """Domain totals are held at their current estimates, not asserted as known."""
    std = sample.weighting.standardize("g", shares={"a": 1, "b": 1, "c": 1})
    assert std.design.wgt_adjustment.pins_total is False


def test_rake_records_one_cells_column_per_margin(sample):
    """A single concatenated column would encode a stronger calibration."""
    df = sample.data.with_columns(pl.Series("h", ["x", "y"] * 3))
    s = Sample(df, Design(wgt="w"))
    out = s.weighting.rake(controls={"g": CONTROLS, "h": {"x": 30.0, "y": 30.0}}, strict=False)
    assert len(out.design.wgt_adjustment.cells) == 2


def test_scoped_cells_are_null_outside_the_adjustment(sample):
    """Null marks a row that got factor 1 and must not be centred."""
    out = sample.weighting.poststratify({"a": 10.0, "b": 20.0}, cells="g", where=col("g") != "c")
    cells = out.data[out.design.wgt_adjustment.cells[0]].to_list()
    assert cells[:4] == [0, 0, 1, 1]
    assert cells[4:] == [None, None]


@pytest.mark.parametrize("method", ["poststratify", "rake", "calibrate", "standardize"])
def test_snapshot_columns_are_hidden(sample, method):
    out = _run(sample, method)
    rec = out.design.wgt_adjustment
    hidden = out._hidden_columns_for_ui()
    for name in (rec.cells or ()) + (rec.aux or ()):
        assert name in hidden


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------


def test_the_record_is_replaced_not_accumulated(sample):
    """One record: it describes the LAST adjustment only."""
    out = sample.weighting.poststratify(CONTROLS, cells="g").weighting.trim(
        upper=15.0, min_cell_size=1
    )
    assert out.design.wgt_adjustment.kind == "trimming"
    assert out.design.wgt_adjustment.prev_wgt == "ps_wgt"


def test_lineage_is_the_chain_of_designs(sample):
    """What the single record gives up, the design history keeps."""
    out = sample.weighting.poststratify(CONTROLS, cells="g").weighting.trim(
        upper=15.0, min_cell_size=1
    )
    kinds = [getattr(d.wgt_adjustment, "kind", None) for d in out.design_history]
    assert kinds == [None, "poststratification", "trimming"]
    assert [d.wgt for d in out.design_history] == ["w", "ps_wgt", "trim_wgt"]


def test_history_does_not_leak_back_to_the_caller(sample):
    """inplace=False forks; an immutable tuple diverges on its own."""
    before = len(sample.design_history)
    sample.weighting.poststratify(CONTROLS, cells="g")
    assert len(sample.design_history) == before


def test_internal_design_rebinds_are_not_lineage(sample):
    """poststratify writes _design twice; that is one user-visible step."""
    df = sample.data.with_columns([pl.Series(f"r{i}", [1.0] * 6) for i in (1, 2)])
    s = Sample(df, Design(wgt="w"))
    s._design = s.design.update_rep_weights(method="BRR", prefix="r", n_reps=2)
    out = s.weighting.poststratify(CONTROLS, cells="g")
    assert len(out.design_history) == 2


def test_record_survives_an_unrelated_design_update(sample):
    rec = WgtAdjustment(kind="poststratification", prev_wgt="w", new_wgt="p", cells=("c",))
    d = Design(wgt="p", wgt_adjustment=rec)
    assert d.update(stratum="g").wgt_adjustment is rec
