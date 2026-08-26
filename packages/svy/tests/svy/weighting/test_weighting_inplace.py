# tests/svy/weighting/test_weighting_inplace.py
"""The weighting namespace's copy-on-write contract.

``inplace=False`` (the default) leaves the caller's Sample untouched;
``inplace=True`` rewrites it and returns it. Both return a Sample, so chaining
is identical under either. This mirrors ``wrangling``, where all 19 methods
already take ``inplace: bool = False``.

Before this contract every weighting method mutated its receiver and returned
self, so two generators branching off one Sample produced an object whose
columns came from one method and whose design came from the other.
"""

import inspect

import polars as pl
import pytest

from svy import Design, Sample
from svy.core.warnings import WarnCode
from svy.weighting import Weighting


# Methods that transform a Sample, and therefore carry `inplace`. The three
# that do not -- controls_margins_template, control_aux_template,
# build_aux_matrix -- return templates and matrices, not a Sample.
TRANSFORMS = [
    "create_brr_wgts",
    "create_jk_wgts",
    "create_bs_wgts",
    "create_sdr_wgts",
    "adjust",
    "normalize",
    "poststratify",
    "rake",
    "calibrate",
    "calibrate_matrix",
    "trim",
]

NON_TRANSFORMS = [
    "controls_margins_template",
    "control_aux_template",
    "build_aux_matrix",
]


@pytest.fixture
def frame():
    # 4 strata x 2 PSUs, so BRR/JK2 pairing and the Rao-Wu bootstrap all run.
    return pl.DataFrame(
        {
            "stratum": ["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["d"] * 4,
            "psu": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
            "wgt": [10.0] * 16,
            "y": [1.0, 2.0, 3.0, 4.0] * 4,
            "urb": ["U", "R"] * 8,
            "status": ["rr"] * 16,
        }
    )


@pytest.fixture
def sample(frame):
    return Sample(data=frame, design=Design(stratum="stratum", psu="psu", wgt="wgt"))


def _snapshot(s: Sample) -> tuple:
    return (s._data.width, s._data.height, repr(s._design))


# ---------------------------------------------------------------------------
# Signature contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", TRANSFORMS)
def test_transform_has_keyword_only_inplace_defaulting_false(name):
    param = inspect.signature(getattr(Weighting, name)).parameters.get("inplace")
    assert param is not None, f"{name} is missing the inplace parameter"
    assert param.kind is inspect.Parameter.KEYWORD_ONLY, f"{name}.inplace must be keyword-only"
    assert param.default is False, f"{name}.inplace must default to False"


@pytest.mark.parametrize("name", NON_TRANSFORMS)
def test_non_transform_has_no_inplace(name):
    """These return templates/matrices, so there is no Sample to place."""
    assert "inplace" not in inspect.signature(getattr(Weighting, name)).parameters


# ---------------------------------------------------------------------------
# Default (inplace=False) leaves the caller alone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda s: s.weighting.create_bs_wgts(n_reps=4, rstate=1), id="bootstrap"),
        pytest.param(lambda s: s.weighting.create_jk_wgts(), id="jackknife"),
        pytest.param(lambda s: s.weighting.create_jk_wgts(paired=True), id="jackknife-paired"),
        pytest.param(lambda s: s.weighting.create_brr_wgts(), id="brr"),
        pytest.param(lambda s: s.weighting.create_sdr_wgts(n_reps=4), id="sdr"),
        pytest.param(lambda s: s.weighting.adjust(resp_status="status", by="urb"), id="adjust"),
        pytest.param(lambda s: s.weighting.normalize(controls=100.0), id="normalize"),
        pytest.param(
            lambda s: s.weighting.rake(controls={"urb": {"U": 90.0, "R": 90.0}}), id="rake"
        ),
        pytest.param(lambda s: s.weighting.trim(upper=15.0), id="trim"),
    ],
)
def test_default_does_not_touch_the_caller(sample, call):
    before = _snapshot(sample)
    out = call(sample)
    assert _snapshot(sample) == before, "the caller's Sample was modified"
    assert out is not sample, "the default must return a distinct Sample"


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda s: s.weighting.create_bs_wgts(n_reps=4, rstate=1, inplace=True), id="bootstrap"
        ),
        pytest.param(lambda s: s.weighting.create_jk_wgts(inplace=True), id="jackknife"),
        pytest.param(
            lambda s: s.weighting.normalize(controls=100.0, inplace=True), id="normalize"
        ),
    ],
)
def test_inplace_true_rewrites_the_caller_and_returns_it(sample, call):
    before = _snapshot(sample)
    out = call(sample)
    assert out is sample
    assert _snapshot(sample) != before


# ---------------------------------------------------------------------------
# The bug this contract exists to prevent
# ---------------------------------------------------------------------------


def test_two_generators_can_branch_off_one_sample(sample):
    boot = sample.weighting.create_bs_wgts(n_reps=4, rep_prefix="bs", rstate=1)
    jack = sample.weighting.create_jk_wgts(rep_prefix="jk")

    assert sample.design.rep_wgts is None
    assert len({id(sample), id(boot), id(jack)}) == 3

    # each branch's design describes its own columns
    assert boot.design.rep_wgts.method == "Bootstrap"
    assert boot.design.rep_wgts.prefix == "bs"
    assert jack.design.rep_wgts.method == "Jackknife"
    assert jack.design.rep_wgts.prefix == "jk"

    assert {"bs1", "bs4"} <= set(boot._data.columns)
    assert "jk1" not in boot._data.columns
    assert {"jk1", "jk8"} <= set(jack._data.columns)
    assert "bs1" not in jack._data.columns


def test_chaining_is_unaffected_by_the_mode(sample):
    controls = {"urb": {"U": 90.0, "R": 90.0}}

    forked = (
        sample.weighting.create_bs_wgts(n_reps=4, rep_prefix="bw", rstate=1)
        .weighting.adjust(resp_status="status", by="urb", wgt_name="nr_wgt")
        .weighting.rake(controls=controls, wgt_name="final_wgt")
    )
    mutated = (
        Sample(data=sample._data, design=sample._design)
        .weighting.create_bs_wgts(n_reps=4, rep_prefix="bw", rstate=1, inplace=True)
        .weighting.adjust(resp_status="status", by="urb", wgt_name="nr_wgt", inplace=True)
        .weighting.rake(controls=controls, wgt_name="final_wgt", inplace=True)
    )

    assert forked.design.wgt == mutated.design.wgt == "final_wgt"
    assert sample.design.wgt == "wgt", "the forked chain leaked back to its source"
    assert forked._data["final_wgt"].to_list() == pytest.approx(
        mutated._data["final_wgt"].to_list()
    )


def test_fork_isolates_the_warning_store(sample):
    """A failed default-mode call leaves the caller's warnings alone too.

    The isolation has to cover diagnostics, not just data: "this call had no
    effect on my Sample" is only true if the warning store is untouched as
    well. The error is still raised, and still emitted to the log.
    """
    negative = sample.wrangling.mutate({"wgt": pl.lit(-1.0)})
    before = len(negative._warnings.list(code=WarnCode.NEGATIVE_WEIGHT))

    with pytest.raises(Exception, match="negative weight|Negative weights"):
        negative.weighting.trim(upper=15.0)
    assert len(negative._warnings.list(code=WarnCode.NEGATIVE_WEIGHT)) == before

    # inplace=True is how you ask for the diagnostic to land on your Sample
    with pytest.raises(Exception, match="negative weight|Negative weights"):
        negative.weighting.trim(upper=15.0, inplace=True)
    assert len(negative._warnings.list(code=WarnCode.NEGATIVE_WEIGHT)) == before + 1
