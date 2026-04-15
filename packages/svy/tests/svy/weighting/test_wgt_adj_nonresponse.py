# tests/svy/core/test_adjustment_nr.py
import numpy as np
import polars as pl
import pytest

from svy import EstimationMethod
from svy.core.sample import Design, Sample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Default auto-generated weight name for nonresponse adjustment
NR_WGT = "nr_wgt"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_data_basic():
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "weight": [1.0] * 10,
            "status": ["rr", "rr", "nr", "in", "uk", "rr", "nr", "in", "rr", "uk"],
            "adj_class": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
            "some_val": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        }
    )


@pytest.fixture
def sample_data_custom_codes():
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "weight": [2.0] * 6,
            "status_code": ["R", "R", "N", "I", "U", "R"],
            "adj_class_num": [1, 1, 1, 2, 2, 2],
        }
    )


@pytest.fixture
def sample_data_no_respondents():
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "weight": [5.0] * 4,
            "status": ["nr", "in", "uk", "nr"],
            "adj_class": ["X", "X", "Y", "Y"],
        }
    )


@pytest.fixture
def sample_data_all_respondents():
    return pl.DataFrame(
        {
            "id": [1, 2, 3],
            "weight": [10.0] * 3,
            "status": ["rr", "rr", "rr"],
            "adj_class": ["Group1", "Group1", "Group2"],
        }
    )


@pytest.fixture
def sample_data_single_class():
    return pl.DataFrame(
        {
            "id": list(range(1, 11)),
            "weight": [1.0] * 10,
            "status": ["rr", "rr", "nr", "in", "uk", "rr", "nr", "in", "rr", "uk"],
            "single_class": ["all"] * 10,
        }
    )


# ---------------------------------------------------------------------------
# Core adjustment correctness
# ---------------------------------------------------------------------------


def test_adjust_single_class_standard_codes(sample_data_single_class):
    sample = Sample(data=sample_data_single_class, design=Design(wgt="weight"))
    sample.weighting.adjust(
        by="single_class", resp_status="status", unknown_to_inelig=True, respondents_only=False
    )
    adjusted = sample.data[NR_WGT].to_numpy()
    expected = np.zeros(10)
    expected[[0, 1, 5, 8]] = 1.875  # rr units
    expected[[3, 7]] = 1.25  # in units
    np.testing.assert_allclose(adjusted, expected, atol=1e-9)
    assert NR_WGT in sample.data.columns


def test_adjust_stratified_custom_codes(sample_data_custom_codes):
    sample = Sample(data=sample_data_custom_codes, design=Design(wgt="weight"))
    mapping = {"rr": "R", "nr": "N", "in": "I", "uk": "U"}
    sample.weighting.adjust(
        by="adj_class_num",
        resp_status="status_code",
        resp_mapping=mapping,
        unknown_to_inelig=True,
        respondents_only=False,
    )
    adjusted = sample.data[NR_WGT].to_numpy()
    expected = np.array([3.0, 3.0, 0.0, 3.0, 0.0, 3.0])
    np.testing.assert_allclose(adjusted, expected, atol=1e-9)


def test_adjust_stratified_unknown_to_inelig_true(sample_data_basic):
    sample = Sample(data=sample_data_basic, design=Design(wgt="weight"))
    sample.weighting.adjust(
        by="adj_class", resp_status="status", unknown_to_inelig=True, respondents_only=False
    )
    adjusted = sample.data[NR_WGT].to_numpy()
    expected = np.array(
        [
            1.875,
            1.875,
            0.0,
            1.25,
            0.0,  # A: rr,rr,nr,in,uk
            1.875,
            0.0,
            1.25,
            1.875,
            0.0,  # B: rr,nr,in,rr,uk
        ]
    )
    np.testing.assert_allclose(adjusted, expected, atol=1e-9)


def test_adjust_stratified_unknown_to_inelig_false(sample_data_basic):
    sample = Sample(data=sample_data_basic, design=Design(wgt="weight"))
    sample.weighting.adjust(
        by="adj_class", resp_status="status", unknown_to_inelig=False, respondents_only=False
    )
    adjusted = sample.data[NR_WGT].to_numpy()
    expected = np.array(
        [
            2.0,
            2.0,
            0.0,
            1.0,
            0.0,  # A
            2.0,
            0.0,
            1.0,
            2.0,
            0.0,  # B
        ]
    )
    np.testing.assert_allclose(adjusted, expected, atol=1e-9)


def test_adjust_zero_respondents_in_class(sample_data_no_respondents):
    sample = Sample(data=sample_data_no_respondents, design=Design(wgt="weight"))
    sample.weighting.adjust(
        by="adj_class", resp_status="status", unknown_to_inelig=True, respondents_only=False
    )
    adjusted = sample.data[NR_WGT].to_numpy()
    expected = np.array([0.0, 5.0, 0.0, 0.0])  # X: nr,in  Y: uk,nr
    np.testing.assert_allclose(adjusted, expected, atol=1e-9)


def test_adjust_all_respondents(sample_data_all_respondents):
    sample = Sample(data=sample_data_all_respondents, design=Design(wgt="weight"))
    sample.weighting.adjust(by="adj_class", resp_status="status", unknown_to_inelig=True)
    adjusted = sample.data[NR_WGT].to_numpy()
    expected = np.array([10.0, 10.0, 10.0])
    np.testing.assert_allclose(adjusted, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# Auto-generated column name
# ---------------------------------------------------------------------------


def test_adjust_auto_col_name_uses_nr_wgt(sample_data_basic):
    """Auto-generated adjusted weight column must be nr_wgt."""
    sample = Sample(data=sample_data_basic, design=Design(wgt="weight"))
    sample.weighting.adjust(by="adj_class", resp_status="status")
    assert NR_WGT in sample.data.columns
    assert "svy_adj_weight" not in sample.data.columns


def test_adjust_wgt_name_overrides_default(sample_data_basic):
    """Explicit wgt_name overrides auto-generated name."""
    sample = Sample(data=sample_data_basic, design=Design(wgt="weight"))
    sample.weighting.adjust(by="adj_class", resp_status="status", wgt_name="my_adj_wgt")
    assert "my_adj_wgt" in sample.data.columns
    assert NR_WGT not in sample.data.columns
    assert sample.design.wgt == "my_adj_wgt"


# ---------------------------------------------------------------------------
# Column name collision guard
# ---------------------------------------------------------------------------


def test_adjust_wgt_name_collision_raises(sample_data_basic):
    """Using an existing column name raises an error."""
    sample = Sample(data=sample_data_basic, design=Design(wgt="weight"))
    with pytest.raises(Exception, match="already exists"):
        sample.weighting.adjust(
            by="adj_class",
            resp_status="status",
            wgt_name="weight",
        )


# ---------------------------------------------------------------------------
# Replicate weights
# ---------------------------------------------------------------------------


def test_adjust_replicate_weights_auto_prefix(sample_data_basic):
    """Replicate weight prefix matches the full-sample weight name."""
    df = sample_data_basic.with_columns(
        pl.Series("rw1", np.ones(sample_data_basic.height)),
        pl.Series("rw2", np.ones(sample_data_basic.height)),
    )
    sample = Sample(data=df, design=Design(wgt="weight"))
    sample._design = sample.design.update_rep_weights(
        method=EstimationMethod.BRR,
        prefix="rw",
        n_reps=2,
    )
    sample.weighting.adjust(
        by="adj_class",
        resp_status="status",
        update_design_wgts=True,
        respondents_only=False,
    )
    assert f"{NR_WGT}1" in sample.data.columns
    assert f"{NR_WGT}2" in sample.data.columns
    assert sample.design.rep_wgts.columns == [f"{NR_WGT}1", f"{NR_WGT}2"]
    assert sample.design.rep_wgts.prefix == NR_WGT
    assert sample.design.rep_wgts.n_reps == 2


def test_adjust_replicate_weights_custom_wgt_name(sample_data_basic):
    """Custom wgt_name propagates to replicate weight prefix."""
    df = sample_data_basic.with_columns(
        pl.Series("rw1", np.ones(sample_data_basic.height)),
        pl.Series("rw2", np.ones(sample_data_basic.height)),
    )
    sample = Sample(data=df, design=Design(wgt="weight"))
    sample._design = sample.design.update_rep_weights(
        method=EstimationMethod.BRR,
        prefix="rw",
        n_reps=2,
    )
    sample.weighting.adjust(
        by="adj_class",
        resp_status="status",
        wgt_name="my_wgt",
        respondents_only=False,
    )
    assert "my_wgt" in sample.data.columns
    assert "my_wgt1" in sample.data.columns
    assert "my_wgt2" in sample.data.columns
    assert sample.design.rep_wgts.prefix == "my_wgt"


def test_adjust_replicate_weights_ignored_when_flag_set(sample_data_basic):
    df = sample_data_basic.with_columns(
        pl.Series("rw1", np.ones(sample_data_basic.height)),
        pl.Series("rw2", np.ones(sample_data_basic.height)),
    )
    sample = Sample(data=df, design=Design(wgt="weight"))
    sample._design = sample.design.update_rep_weights(
        method=EstimationMethod.BRR,
        prefix="rw",
        n_reps=2,
    )
    sample.weighting.adjust(
        by="adj_class",
        resp_status="status",
        ignore_reps=True,
        respondents_only=False,
    )
    assert f"{NR_WGT}1" not in sample.data.columns
    assert sample.design.rep_wgts.columns == ["rw1", "rw2"]


def test_adjust_no_design_update(sample_data_basic):
    df = sample_data_basic.with_columns(
        pl.Series("rw1", np.ones(sample_data_basic.height)),
        pl.Series("rw2", np.ones(sample_data_basic.height)),
    )
    sample = Sample(data=df, design=Design(wgt="weight"))
    sample._design = sample.design.update_rep_weights(
        method=EstimationMethod.BRR,
        prefix="rw",
        n_reps=2,
    )
    sample.weighting.adjust(
        by="adj_class",
        resp_status="status",
        update_design_wgts=False,
        respondents_only=False,
    )
    assert NR_WGT in sample.data.columns
    assert f"{NR_WGT}1" in sample.data.columns
    # Design unchanged
    assert sample.design.wgt == "weight"
    assert sample.design.rep_wgts.columns == ["rw1", "rw2"]


# ---------------------------------------------------------------------------
# by= multi-column (tuple / list)
# ---------------------------------------------------------------------------


def _make_twoway_sample():
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "weight": [1.0] * 6,
            "status": ["rr", "nr", "rr", "in", "rr", "uk"],
            "region": ["North", "North", "North", "North", "South", "South"],
            "sex": ["M", "M", "F", "F", "M", "F"],
        }
    )
    expected = np.array([2.0, 0.0, 1.0, 1.0, 1.0, 0.0])
    return df, expected


def test_adjust_by_tuple():
    """Tuple of strings crosses the columns into adjustment classes."""
    df, expected = _make_twoway_sample()
    sample = Sample(data=df, design=Design(wgt="weight"))
    sample.weighting.adjust(
        resp_status="status",
        by=("region", "sex"),
        unknown_to_inelig=True,
        respondents_only=False,
    )
    out = sample.data[NR_WGT].to_numpy()
    np.testing.assert_allclose(out, expected, atol=1e-9)


def test_adjust_by_list_accepted():
    """Lists of strings are now accepted (equivalent to tuple)."""
    df, expected = _make_twoway_sample()
    sample = Sample(data=df, design=Design(wgt="weight"))
    sample.weighting.adjust(
        resp_status="status",
        by=["region", "sex"],
        unknown_to_inelig=True,
        respondents_only=False,
    )
    out = sample.data[NR_WGT].to_numpy()
    np.testing.assert_allclose(out, expected, atol=1e-9)


def test_adjust_by_missing_column_raises():
    """Missing column in by= raises an error."""
    df, _ = _make_twoway_sample()
    sample = Sample(data=df, design=Design(wgt="weight"))
    with pytest.raises(Exception, match="All `by` columns must exist"):
        sample.weighting.adjust(resp_status="status", by=("region", "not_here"))


# ---------------------------------------------------------------------------
# respondents_only
# ---------------------------------------------------------------------------


def test_adjust_respondents_only_standard(sample_data_basic):
    sample = Sample(data=sample_data_basic, design=Design(wgt="weight"))
    out = sample.weighting.adjust(
        by="adj_class",
        resp_status="status",
        unknown_to_inelig=True,
        respondents_only=True,
    )
    assert set(out.data["status"].unique().to_list()) == {"rr"}


def test_adjust_respondents_only_custom_mapping(sample_data_custom_codes):
    sample = Sample(data=sample_data_custom_codes, design=Design(wgt="weight"))
    mapping = {"rr": "R", "nr": "N", "in": "I", "uk": "U"}
    out = sample.weighting.adjust(
        by="adj_class_num",
        resp_status="status_code",
        resp_mapping=mapping,
        respondents_only=True,
    )
    assert set(out.data["status_code"].unique().to_list()) == {"R"}


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_adjust_invalid_weight_column_raises():
    df = pl.DataFrame({"w": [1], "s": ["rr"], "c": ["A"]})
    sample = Sample(data=df)
    with pytest.raises(Exception, match="Sample weight is None"):
        sample.weighting.adjust(by="c", resp_status="s")


def test_adjust_non_existent_column_raises():
    df = pl.DataFrame({"w_col": [1], "s_col": ["rr"], "c_col": ["A"]})
    sample = Sample(data=df, design=Design(wgt="w_col"))
    with pytest.raises(Exception, match="All `by` columns must exist"):
        sample.weighting.adjust(resp_status="s_col", by="non_existent_c")
    with pytest.raises(Exception, match="resp_status"):
        sample.weighting.adjust(by="c_col", resp_status="non_existent_s")


# ---------------------------------------------------------------------------
# Trimming (one-shot — no iteration)
# ---------------------------------------------------------------------------


def _skewed_adj_sample():
    """Dataset where adjustment produces one extreme weight.

    Class A: 1 respondent, 4 non-respondents → adjustment factor = 5x
    Class B: 4 respondents, 1 non-respondent → adjustment factor = 1.25x
    Base weight = 1.0 for all.
    """
    return pl.DataFrame(
        {
            "weight": [1.0] * 10,
            "status": [
                "rr",
                "nr",
                "nr",
                "nr",
                "nr",  # class A: 1 rr, 4 nr
                "rr",
                "rr",
                "rr",
                "rr",
                "nr",
            ],  # class B: 4 rr, 1 nr
            "cls": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
        }
    )


def test_adjust_trim_caps_extreme_weight():
    """Trimming after adjustment caps the extreme weight."""
    from svy.weighting.types import TrimConfig

    sample = Sample(data=_skewed_adj_sample(), design=Design(wgt="weight"))
    out = sample.weighting.adjust(
        by="cls",
        resp_status="status",
        respondents_only=False,
        trimming=TrimConfig(upper=3.0, redistribute=True, min_cell_size=1),
    )
    weights = out.data[NR_WGT].to_numpy()
    # Class A respondent gets weight=5 before trim → should be capped at 3
    assert weights.max() <= 3.0 * 1.01


def test_adjust_trim_none_unchanged():
    """trimming=None gives identical result to plain adjust()."""
    s1 = Sample(data=_skewed_adj_sample(), design=Design(wgt="weight"))
    s2 = Sample(data=_skewed_adj_sample(), design=Design(wgt="weight"))

    out1 = s1.weighting.adjust(by="cls", resp_status="status", respondents_only=False)
    out2 = s2.weighting.adjust(
        by="cls", resp_status="status", respondents_only=False, trimming=None
    )
    np.testing.assert_allclose(
        out1.data[NR_WGT].to_numpy(),
        out2.data[NR_WGT].to_numpy(),
    )


def test_adjust_trim_no_iteration():
    """Adjustment trimming is one-shot — TrimConfig.max_iter controls
    only the internal trim redistribution, not an outer cycle."""
    from svy.weighting.types import TrimConfig

    sample = Sample(data=_skewed_adj_sample(), design=Design(wgt="weight"))
    out = sample.weighting.adjust(
        by="cls",
        resp_status="status",
        respondents_only=False,
        trimming=TrimConfig(upper=3.0, redistribute=True, max_iter=1),
    )
    assert NR_WGT in out.data.columns
    assert len(out.data[NR_WGT].to_numpy()) == 10


def test_adjust_trim_design_updated():
    """After trim, design.wgt points to the trimmed weight column."""
    from svy.weighting.types import TrimConfig

    sample = Sample(data=_skewed_adj_sample(), design=Design(wgt="weight"))
    out = sample.weighting.adjust(
        by="cls",
        resp_status="status",
        respondents_only=False,
        trimming=TrimConfig(upper=3.0, redistribute=True),
    )
    assert out.design.wgt == NR_WGT
