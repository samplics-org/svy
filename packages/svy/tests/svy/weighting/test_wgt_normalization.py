# tests/svy/weighting/test_wgt_normalization.py
import numpy as np
import polars as pl
import pytest

from numpy.testing import assert_allclose

from svy import Design, EstimationMethod, Sample


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

NORM_WGT = "norm_wgt"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_design():
    return Design(wgt="samp_weight")


@pytest.fixture
def sample_data():
    return pl.DataFrame(
        {
            "samp_weight": [10.0, 10.0, 20.0, 20.0, 30.0, 30.0],
            "domain": ["A", "A", "A", "B", "B", "B"],
        }
    )


# ---------------------------------------------------------------------------
# Core normalization correctness
# ---------------------------------------------------------------------------


def test_normalize_no_domain_or_control(sample_data, mock_design):
    """Weights normalized to sum to number of samples (n=6)."""
    sample = Sample(data=sample_data, design=mock_design)
    sample.weighting.normalize()

    expected = np.array([0.5, 0.5, 1.0, 1.0, 1.5, 1.5])
    result = sample.data.get_column(NORM_WGT).to_numpy()

    assert NORM_WGT in sample.data.columns
    assert_allclose(result, expected)
    assert_allclose(result.sum(), len(sample_data))


def test_normalize_with_numeric_control(sample_data, mock_design):
    """With a scalar control total, weights sum to that total."""
    sample = Sample(data=sample_data, design=mock_design)
    control_total = 240
    sample.weighting.normalize(controls=control_total)

    expected = np.array([20.0, 20.0, 40.0, 40.0, 60.0, 60.0])
    result = sample.data.get_column(NORM_WGT).to_numpy()

    assert_allclose(result, expected)
    assert_allclose(result.sum(), control_total)


def test_normalize_with_domain_and_dict_control(sample_data, mock_design):
    """Normalization within domains using a dict of control totals."""
    sample = Sample(data=sample_data, design=mock_design)
    controls = {"A": 60, "B": 180}
    sample.weighting.normalize(controls=controls, by="domain")

    expected = np.array([15.0, 15.0, 30.0, 45.0, 67.5, 67.5])
    result = sample.data.get_column(NORM_WGT).to_numpy()

    assert_allclose(result, expected)
    a_sum = sample.data.filter(pl.col("domain") == "A").get_column(NORM_WGT).sum()
    b_sum = sample.data.filter(pl.col("domain") == "B").get_column(NORM_WGT).sum()
    assert_allclose(a_sum, controls["A"])
    assert_allclose(b_sum, controls["B"])


def test_normalize_with_domain_no_control(sample_data, mock_design):
    """Normalization within domains, each domain sums to its count."""
    sample = Sample(data=sample_data, design=mock_design)
    sample.weighting.normalize(by="domain")

    expected = np.array([0.75, 0.75, 1.5, 0.75, 1.125, 1.125])
    result = sample.data.get_column(NORM_WGT).to_numpy()

    assert_allclose(result, expected)
    a_sum = sample.data.filter(pl.col("domain") == "A").get_column(NORM_WGT).sum()
    b_sum = sample.data.filter(pl.col("domain") == "B").get_column(NORM_WGT).sum()
    assert_allclose(a_sum, 3)
    assert_allclose(b_sum, 3)


# ---------------------------------------------------------------------------
# Auto-generated column name
# ---------------------------------------------------------------------------


def test_normalize_auto_col_name_uses_norm_wgt(sample_data, mock_design):
    """Auto-generated normalized weight column must be norm_wgt."""
    sample = Sample(data=sample_data, design=mock_design)
    sample.weighting.normalize()
    assert NORM_WGT in sample.data.columns
    assert "svy_norm_samp_weight" not in sample.data.columns


def test_normalize_wgt_name_overrides_default(sample_data, mock_design):
    """Explicit wgt_name overrides auto-generated name."""
    sample = Sample(data=sample_data, design=mock_design)
    sample.weighting.normalize(wgt_name="my_norm_wgt")
    assert "my_norm_wgt" in sample.data.columns
    assert NORM_WGT not in sample.data.columns
    assert sample.design.wgt == "my_norm_wgt"


# ---------------------------------------------------------------------------
# Column name collision guard
# ---------------------------------------------------------------------------


def test_normalize_wgt_name_collision_raises(sample_data, mock_design):
    """Using an existing column name raises an error."""
    sample = Sample(data=sample_data, design=mock_design)
    with pytest.raises(Exception, match="already exists"):
        sample.weighting.normalize(wgt_name="samp_weight")


# ---------------------------------------------------------------------------
# Replicate weights
# ---------------------------------------------------------------------------


def test_normalize_replicate_weights_auto_prefix(sample_data, mock_design):
    """Replicate weight prefix matches the full-sample weight name."""
    df = sample_data.with_columns(
        pl.Series("rw1", np.ones(sample_data.height)),
        pl.Series("rw2", np.ones(sample_data.height)),
    )
    sample = Sample(data=df, design=mock_design)
    sample._design = sample.design.update_rep_weights(
        method=EstimationMethod.BRR, prefix="rw", n_reps=2
    )
    sample.weighting.normalize(update_design_wgts=True)
    assert f"{NORM_WGT}1" in sample.data.columns
    assert f"{NORM_WGT}2" in sample.data.columns
    assert sample.design.rep_wgts.columns == [f"{NORM_WGT}1", f"{NORM_WGT}2"]
    assert sample.design.rep_wgts.prefix == NORM_WGT


def test_normalize_replicate_weights_custom_wgt_name(sample_data, mock_design):
    """Custom wgt_name propagates to replicate weight prefix."""
    df = sample_data.with_columns(
        pl.Series("rw1", np.ones(sample_data.height)),
        pl.Series("rw2", np.ones(sample_data.height)),
    )
    sample = Sample(data=df, design=mock_design)
    sample._design = sample.design.update_rep_weights(
        method=EstimationMethod.BRR, prefix="rw", n_reps=2
    )
    sample.weighting.normalize(wgt_name="my_norm")
    assert "my_norm" in sample.data.columns
    assert "my_norm1" in sample.data.columns
    assert "my_norm2" in sample.data.columns
    assert sample.design.rep_wgts.prefix == "my_norm"


def test_normalize_replicate_weights_ignored_when_flag_set(sample_data, mock_design):
    df = sample_data.with_columns(
        pl.Series("rw1", np.ones(sample_data.height)),
        pl.Series("rw2", np.ones(sample_data.height)),
    )
    sample = Sample(data=df, design=mock_design)
    sample._design = sample.design.update_rep_weights(
        method=EstimationMethod.BRR, prefix="rw", n_reps=2
    )
    sample.weighting.normalize(ignore_reps=True)
    assert f"{NORM_WGT}1" not in sample.data.columns
    assert sample.design.rep_wgts.columns == ["rw1", "rw2"]


def test_normalize_no_design_update(sample_data, mock_design):
    df = sample_data.with_columns(
        pl.Series("rw1", np.ones(sample_data.height)),
        pl.Series("rw2", np.ones(sample_data.height)),
    )
    sample = Sample(data=df, design=mock_design)
    sample._design = sample.design.update_rep_weights(
        method=EstimationMethod.BRR, prefix="rw", n_reps=2
    )
    sample.weighting.normalize(update_design_wgts=False)
    assert NORM_WGT in sample.data.columns
    assert f"{NORM_WGT}1" in sample.data.columns
    assert sample.design.wgt == "samp_weight"
    assert sample.design.rep_wgts.columns == ["rw1", "rw2"]


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_normalize_raises_for_invalid_weight_or_by(sample_data):
    with pytest.raises(ValueError, match="Design references columns not found in data"):
        Sample(data=sample_data, design=Design(wgt="not_a_column"))

    sample = Sample(data=sample_data, design=Design(wgt="samp_weight"))
    with pytest.raises(Exception, match="All `by` columns must exist"):
        sample.weighting.normalize(by="non_existent_domain")


# ---------------------------------------------------------------------------
# Multi-column by (tuple / list)
# ---------------------------------------------------------------------------


def _two_way_sample_norm():
    return pl.DataFrame(
        {
            "samp_weight": [10.0] * 8,
            "region": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "sex": ["M", "F", "M", "F", "M", "F", "M", "F"],
        }
    )


def test_normalize_by_tuple_with_control_dict(mock_design):
    df = _two_way_sample_norm()
    sample = Sample(data=df, design=mock_design)
    controls = {
        ("A", "M"): 30.0,
        ("A", "F"): 10.0,
        ("B", "M"): 50.0,
        ("B", "F"): 30.0,
    }
    out = sample.weighting.normalize(controls=controls, by=("region", "sex"))
    assert NORM_WGT in out.data.columns
    for (r, s), target in controls.items():
        got = (
            out.data.filter((pl.col("region") == r) & (pl.col("sex") == s))
            .get_column(NORM_WGT)
            .sum()
        )
        assert_allclose(got, target)


def test_normalize_by_list_accepted(mock_design):
    """Lists of strings are now accepted (equivalent to tuple)."""
    df = _two_way_sample_norm()
    sample = Sample(data=df, design=mock_design)
    controls = {
        ("A", "M"): 30.0,
        ("A", "F"): 10.0,
        ("B", "M"): 50.0,
        ("B", "F"): 30.0,
    }
    out = sample.weighting.normalize(controls=controls, by=["region", "sex"])
    assert NORM_WGT in out.data.columns
    for (r, s), target in controls.items():
        got = (
            out.data.filter((pl.col("region") == r) & (pl.col("sex") == s))
            .get_column(NORM_WGT)
            .sum()
        )
        assert_allclose(got, target)


def test_normalize_by_tuple_without_control(mock_design):
    """Each cell should sum to its count (2)."""
    df = _two_way_sample_norm()
    sample = Sample(data=df, design=mock_design)
    out = sample.weighting.normalize(by=("region", "sex"))
    for r in ("A", "B"):
        for s in ("M", "F"):
            got = (
                out.data.filter((pl.col("region") == r) & (pl.col("sex") == s))
                .get_column(NORM_WGT)
                .sum()
            )
            assert_allclose(got, 2.0)


def test_normalize_by_missing_column_raises(mock_design):
    df = _two_way_sample_norm()
    sample = Sample(data=df, design=mock_design)
    with pytest.raises(Exception, match="All `by` columns must exist"):
        sample.weighting.normalize(by=("region", "age"))  # 'age' missing
