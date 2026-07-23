# tests/svy/weighting/test_wgt_round8_fixes.py
"""
Round 8 weighting marshalling fixes (findings W1-W8).

W1  bounded= calibration raises instead of being silently ignored
W2  unmatched response statuses raise instead of encoding as respondents
W3  respondents_only filters from the encoded codes (case-insensitive)
W4  adjust(trimming=..., update_design_wgts=False) trims the new weight
W5  calibrate handles overlapping level labels across terms
W6  strict trim-calibrate failure leaves the sample untouched
W7  calibrate's trim cycle honors TrimConfig.by / min_cell_size
"""

import numpy as np
import polars as pl
import pytest

from svy.core.sample import Design, Sample
from svy.core.terms import Cat
from svy.errors import MethodError
from svy.weighting.types import TrimConfig


# ---------------------------------------------------------------------------
# W1 — bounded calibration blocked
# ---------------------------------------------------------------------------


def _calib_sample(n=40, seed=7):
    rng = np.random.default_rng(seed)
    return Sample(
        pl.DataFrame(
            {
                "wgt": rng.uniform(1.0, 3.0, n),
                "grp": ["a", "b"] * (n // 2),
                "z": rng.normal(10.0, 2.0, n),
            }
        ),
        Design(wgt="wgt"),
    )


def test_bounded_calibration_raises():
    s = _calib_sample()
    with pytest.raises(NotImplementedError, match="bounded"):
        s.weighting.calibrate(
            controls={Cat("grp"): {"a": 40.0, "b": 40.0}},
            bounded=True,
        )


# ---------------------------------------------------------------------------
# W2 — unmatched response statuses raise
# ---------------------------------------------------------------------------


def test_adjust_unmatched_status_raises():
    """'noncontact' outside the mapping used to be encoded as a respondent
    and receive an INFLATED weight."""
    s = Sample(
        pl.DataFrame(
            {
                "wgt": [1.0] * 4,
                "status": ["resp", "resp", "refusal", "noncontact"],
            }
        ),
        Design(wgt="wgt"),
    )
    with pytest.raises(MethodError, match="noncontact"):
        s.weighting.adjust(
            resp_status="status",
            by=None,
            resp_mapping={"rr": "resp", "nr": "refusal"},
        )


def test_adjust_null_status_raises():
    s = Sample(
        pl.DataFrame({"wgt": [1.0] * 3, "status": ["rr", None, "nr"]}),
        Design(wgt="wgt"),
    )
    with pytest.raises(MethodError):
        s.weighting.adjust(resp_status="status", by=None)


def test_adjust_collection_valued_mapping():
    """{"nr": ["refusal", "noncontact"]} matches both labels."""
    s = Sample(
        pl.DataFrame(
            {
                "wgt": [1.0] * 4,
                "status": ["resp", "resp", "refusal", "noncontact"],
            }
        ),
        Design(wgt="wgt"),
    )
    out = s.weighting.adjust(
        resp_status="status",
        by=None,
        resp_mapping={"rr": "resp", "nr": ["refusal", "noncontact"]},
    )
    # Both nonrespondents' weight moved to the two respondents: 4/2 = 2.0
    w = out.data.get_column("nr_wgt").to_numpy()
    np.testing.assert_allclose(w, [2.0, 2.0])


# ---------------------------------------------------------------------------
# W3 — respondents_only uses the encoded codes
# ---------------------------------------------------------------------------


def test_adjust_case_insensitive_statuses_keep_respondents():
    """Upper-case statuses used to encode fine but filter to a 0-row Sample."""
    s = Sample(
        pl.DataFrame({"wgt": [1.0] * 4, "status": ["RR", "RR", "NR", "IN"]}),
        Design(wgt="wgt"),
    )
    out = s.weighting.adjust(resp_status="status", by=None)
    assert out.data.height == 2
    w = out.data.get_column("nr_wgt").to_numpy()
    # One NR's weight redistributed over two RRs (IN weight is not): 1.5 each
    np.testing.assert_allclose(w, [1.5, 1.5])


# ---------------------------------------------------------------------------
# W4 — adjust(trimming=..., update_design_wgts=False)
# ---------------------------------------------------------------------------


def test_adjust_trim_without_design_update_preserves_original_weights():
    """The trim used to overwrite the ORIGINAL design weight column in place
    (leaving the new nr_wgt untrimmed)."""
    wgts = [1.0] * 11 + [100.0]
    statuses = ["rr"] * 11 + ["rr"]
    s = Sample(
        pl.DataFrame({"wgt": wgts, "status": statuses}),
        Design(wgt="wgt"),
    )
    out = s.weighting.adjust(
        resp_status="status",
        by=None,
        update_design_wgts=False,
        trimming=TrimConfig(upper=5.0, redistribute=False),
    )
    # Original design weight column untouched
    np.testing.assert_allclose(out.data.get_column("wgt").to_numpy(), wgts)
    # The new adjusted weight is the trimmed one
    nr = out.data.get_column("nr_wgt").to_numpy()
    assert nr.max() <= 5.0 + 1e-9
    # Design still points at the original weight
    assert out._design.wgt == "wgt"


# ---------------------------------------------------------------------------
# W5 — overlapping level labels across terms
# ---------------------------------------------------------------------------


def test_calibrate_overlapping_level_labels():
    """Two Cats sharing numeric-style codes used to crash with
    'Internal error: Design matrix label alignment mismatch'."""
    rng = np.random.default_rng(11)
    n = 40
    s = Sample(
        pl.DataFrame(
            {
                "wgt": rng.uniform(1.0, 2.0, n),
                "A": ([1, 2] * (n // 2)),
                "B": ([1] * (n // 2) + [2] * (n // 2)),
            }
        ),
        Design(wgt="wgt"),
    )
    out = s.weighting.calibrate(
        controls={
            Cat("A", ref=1): {2: 30.0},
            Cat("B", ref=1): {2: 25.0},
        },
        wgt_name="cw",
    )
    w = out.data.get_column("cw").to_numpy()
    a = (out.data.get_column("A") == 2).cast(pl.Float64).to_numpy()
    b = (out.data.get_column("B") == 2).cast(pl.Float64).to_numpy()
    np.testing.assert_allclose(float(a @ w), 30.0, rtol=1e-6)
    np.testing.assert_allclose(float(b @ w), 25.0, rtol=1e-6)


# ---------------------------------------------------------------------------
# W6 — strict trim-calibrate failure leaves the sample untouched
# ---------------------------------------------------------------------------


def test_strict_trim_calibrate_failure_mutates_nothing():
    s = _calib_sample()
    cols_before = list(s.data.columns)
    wgt_before = s.data.get_column("wgt").to_numpy().copy()
    design_wgt_before = s._design.wgt

    # Absolute cap (floats > 1) far below what the totals require: the final
    # calibrate always pushes weights back above the cap -> cannot converge
    with pytest.raises(MethodError, match="did not converge"):
        s.weighting.calibrate(
            controls={Cat("grp"): {"a": 500.0, "b": 500.0}},
            wgt_name="cw",
            strict=True,
            trimming=TrimConfig(upper=1.5, min_cell_size=1, max_iter=2),
        )

    assert list(s.data.columns) == cols_before  # no cw column
    assert s._design.wgt == design_wgt_before
    np.testing.assert_allclose(s.data.get_column("wgt").to_numpy(), wgt_before)


# ---------------------------------------------------------------------------
# W7 — trim cycle honors TrimConfig.by and min_cell_size
# ---------------------------------------------------------------------------


def test_calibrate_trim_cycle_unknown_by_column_raises():
    s = _calib_sample()
    with pytest.raises(MethodError, match="trimming.by|by"):
        s.weighting.calibrate(
            controls={Cat("grp"): {"a": 40.0, "b": 40.0}},
            wgt_name="cw",
            trimming=TrimConfig(upper=10.0, by="no_such_col"),
        )


def test_calibrate_trim_cycle_min_cell_size_skips_with_warning():
    """min_cell_size larger than the sample -> trim skipped, warning stored,
    plain calibration result kept."""
    s = _calib_sample()
    plain = s.weighting.calibrate(
        controls={Cat("grp"): {"a": 40.0, "b": 40.0}},
        wgt_name="cw_plain",
        update_design_wgts=False,
    )
    plain_w = plain.data.get_column("cw_plain").to_numpy()

    s2 = _calib_sample()
    out = s2.weighting.calibrate(
        controls={Cat("grp"): {"a": 40.0, "b": 40.0}},
        wgt_name="cw",
        update_design_wgts=False,
        trimming=TrimConfig(upper=1.01, min_cell_size=999),
    )
    from svy.core.warnings import WarnCode

    w = out.data.get_column("cw").to_numpy()
    np.testing.assert_allclose(w, plain_w)
    assert out._warnings.list(code=WarnCode.DOMAIN_SKIPPED)


def test_calibrate_trim_cycle_by_domain_thresholds():
    """Per-domain trim: each group's weights end at or below its own cap."""
    rng = np.random.default_rng(3)
    n = 40
    df = pl.DataFrame(
        {
            "wgt": np.concatenate(
                [rng.uniform(1.0, 2.0, n // 2), rng.uniform(5.0, 10.0, n // 2)]
            ),
            "grp": ["a"] * (n // 2) + ["b"] * (n // 2),
        }
    )
    s = Sample(df, Design(wgt="wgt"))
    total_a = float(df.filter(pl.col("grp") == "a")["wgt"].sum())
    total_b = float(df.filter(pl.col("grp") == "b")["wgt"].sum())
    out = s.weighting.calibrate(
        controls={Cat("grp"): {"a": total_a, "b": total_b}},
        wgt_name="cw",
        strict=False,
        trimming=TrimConfig(upper=3.0, by="grp", min_cell_size=1, max_iter=5),
    )
    w = out.data.get_column("cw").to_numpy()
    grp = out.data.get_column("grp").to_numpy()
    # Group a was never above the cap; group b must have been trimmed toward it
    assert w[grp == "a"].max() <= 3.0 + 1e-6
    assert w[grp == "b"].max() < 10.0
