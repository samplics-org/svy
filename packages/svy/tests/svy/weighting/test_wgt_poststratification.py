# tests/svy/weighting/test_wgt_poststratification.py
import numpy as np
import polars as pl
import pytest

from numpy.testing import assert_allclose

from svy import Design, Sample


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

# Default auto-generated weight name for post-stratification
PS_WGT = "ps_wgt"

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
# Core behavior
# ---------------------------------------------------------------------------


def test_postratify_with_control_dict(sample_data, mock_design):
    sample = Sample(data=sample_data, design=mock_design)
    controls = {"A": 60, "B": 180}
    sample = sample.weighting.poststratify(controls=controls, cells="domain")

    expected = np.array([15.0, 15.0, 30.0, 45.0, 67.5, 67.5])
    got = sample.data.get_column(PS_WGT).to_numpy()

    assert PS_WGT in sample.data.columns
    assert_allclose(got, expected)


def test_postratify_with_shares_dict(sample_data, mock_design):
    """shares pin composition and preserve the grand total.

    The old ``factors=`` spelling multiplied each cell by the grand total without
    normalizing, so a vector summing to 2.0 silently doubled the population.
    ``shares`` normalizes first: A:B is still 1:3, and the total stays 120.
    """
    sample = Sample(data=sample_data, design=mock_design)
    shares = {"A": 0.5, "B": 1.5}
    sample = sample.weighting.poststratify(shares=shares, cells="domain")

    expected = np.array([7.5, 7.5, 15.0, 22.5, 33.75, 33.75])
    got = sample.data.get_column(PS_WGT).to_numpy()
    assert_allclose(got, expected)
    assert_allclose(got.sum(), 120.0)


def test_postratify_with_numeric_control(sample_data, mock_design):
    sample = Sample(data=sample_data, design=mock_design)
    control_total = 300
    sample = sample.weighting.poststratify(controls=control_total)

    expected = np.array([25.0, 25.0, 50.0, 50.0, 75.0, 75.0])
    got = sample.data.get_column(PS_WGT).to_numpy()
    assert_allclose(got, expected)
    assert_allclose(got.sum(), control_total)


def test_postratify_scalar_shares_rejected(sample_data, mock_design):
    """A scalar is one cell: it cannot describe a composition across cells."""
    sample = Sample(data=sample_data, design=mock_design)
    with pytest.raises(Exception, match="shares.*must be a dict"):
        sample.weighting.poststratify(shares=2.5, cells="domain")


def test_postratify_scalar_controls_with_cells_rejected(sample_data, mock_design):
    """Scalar controls name one total, so pairing them with cells is a mistake.

    Previously this silently ignored ``cells`` and rescaled to the grand total.
    """
    sample = Sample(data=sample_data, design=mock_design)
    with pytest.raises(Exception, match="single number but `cells`"):
        sample.weighting.poststratify(controls=300, cells="domain")


def test_postratify_raises_assertion_error(sample_data, mock_design):
    sample = Sample(data=sample_data, design=mock_design)
    with pytest.raises(Exception, match="controls.*shares|Either controls"):
        sample.weighting.poststratify()


def test_postratify_raises_value_error_for_mismatched_keys(sample_data, mock_design):
    sample = Sample(data=sample_data, design=mock_design)
    with pytest.raises(Exception, match="mismatch|missing|Mapping keys"):
        sample.weighting.poststratify(controls={"A": 60, "C": 180}, cells="domain")


# ---------------------------------------------------------------------------
# Auto-generated column name
# ---------------------------------------------------------------------------


def test_poststratify_auto_col_name_uses_ps_wgt(sample_data, mock_design):
    """Auto-generated column must be ps_wgt."""
    sample = Sample(data=sample_data, design=mock_design)
    sample = sample.weighting.poststratify(controls=300)
    assert PS_WGT in sample.data.columns
    assert "svy_ps_samp_weight" not in sample.data.columns


def test_poststratify_wgt_name_overrides_default(sample_data, mock_design):
    """Explicit wgt_name overrides auto-generated name."""
    sample = Sample(data=sample_data, design=mock_design)
    sample = sample.weighting.poststratify(controls=300, wgt_name="my_ps_wgt")
    assert "my_ps_wgt" in sample.data.columns
    assert PS_WGT not in sample.data.columns
    assert sample.design.wgt == "my_ps_wgt"


# ---------------------------------------------------------------------------
# Column name collision guard
# ---------------------------------------------------------------------------


def test_poststratify_wgt_name_collision_raises(sample_data, mock_design):
    """Using an existing column name raises an error."""
    sample = Sample(data=sample_data, design=mock_design)
    with pytest.raises(Exception, match="already exists"):
        sample.weighting.poststratify(controls=300, wgt_name="samp_weight")


def test_poststratify_no_design_update(sample_data, mock_design):
    sample = Sample(data=sample_data, design=mock_design)
    out = sample.weighting.poststratify(controls=300, update_design_wgts=False)
    assert out.design.wgt == "samp_weight"
    assert PS_WGT in out.data.columns


# ---------------------------------------------------------------------------
# Replicate weights
# ---------------------------------------------------------------------------


def test_poststratify_replicate_weights_auto_prefix(sample_data, mock_design):
    """Replicate weight prefix matches the full-sample weight name."""
    df = sample_data.with_columns(
        pl.Series("rw1", np.ones(sample_data.height)),
        pl.Series("rw2", np.ones(sample_data.height)),
    )
    sample = Sample(data=df, design=mock_design)
    sample._design = sample.design.update_rep_weights(method="BRR", prefix="rw", n_reps=2)
    out = sample.weighting.poststratify(controls=300, update_design_wgts=True)
    assert f"{PS_WGT}1" in out.data.columns
    assert f"{PS_WGT}2" in out.data.columns
    assert out.design.rep_wgts.columns == [f"{PS_WGT}1", f"{PS_WGT}2"]
    assert out.design.rep_wgts.prefix == PS_WGT


def test_poststratify_replicate_weights_custom_wgt_name(sample_data, mock_design):
    """Custom wgt_name propagates to replicate weight prefix."""
    df = sample_data.with_columns(
        pl.Series("rw1", np.ones(sample_data.height)),
        pl.Series("rw2", np.ones(sample_data.height)),
    )
    sample = Sample(data=df, design=mock_design)
    sample._design = sample.design.update_rep_weights(method="BRR", prefix="rw", n_reps=2)
    out = sample.weighting.poststratify(controls=300, wgt_name="my_ps")
    assert "my_ps" in out.data.columns
    assert "my_ps1" in out.data.columns
    assert "my_ps2" in out.data.columns
    assert out.design.rep_wgts.prefix == "my_ps"


def test_poststratify_replicate_weights_ignored_when_flag_set(sample_data, mock_design):
    df = sample_data.with_columns(
        pl.Series("rw1", np.ones(sample_data.height)),
        pl.Series("rw2", np.ones(sample_data.height)),
    )
    sample = Sample(data=df, design=mock_design)
    sample._design = sample.design.update_rep_weights(method="BRR", prefix="rw", n_reps=2)
    out = sample.weighting.poststratify(controls=300, ignore_reps=True)
    assert f"{PS_WGT}1" not in out.data.columns
    assert out.design.rep_wgts.columns == ["rw1", "rw2"]


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_poststratify_raises_for_invalid_weight_or_by(sample_data):
    with pytest.raises(ValueError, match="Design references columns not found in data"):
        Sample(data=sample_data, design=Design(wgt="not_a_column"))

    sample = Sample(data=sample_data, design=Design(wgt="samp_weight"))
    with pytest.raises(Exception, match="All `cells` columns must exist"):
        sample.weighting.poststratify(controls=300, cells="non_existent_domain")


# ---------------------------------------------------------------------------
# Multi-column by (tuple / list)
# ---------------------------------------------------------------------------


def _two_way_sample():
    return pl.DataFrame(
        {
            "samp_weight": [10.0] * 8,
            "region": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "sex": ["M", "F", "M", "F", "M", "F", "M", "F"],
        }
    )


def test_poststratify_by_tuple_with_control_dict(mock_design):
    df = _two_way_sample()
    sample = Sample(data=df, design=mock_design)
    controls = {
        ("A", "M"): 30.0,
        ("A", "F"): 10.0,
        ("B", "M"): 50.0,
        ("B", "F"): 30.0,
    }
    out = sample.weighting.poststratify(controls=controls, cells=("region", "sex"))

    totals = (
        out.data.group_by(["region", "sex"])
        .agg(pl.col(PS_WGT).sum().alias("sum_w"))
        .sort(["region", "sex"])
    )
    assert_allclose(
        totals["sum_w"].to_numpy(),
        [controls[("A", "F")], controls[("A", "M")], controls[("B", "F")], controls[("B", "M")]],
        rtol=1e-6,
    )


def test_poststratify_by_tuple_with_factor_dict(mock_design):
    df = _two_way_sample()
    sample = Sample(data=df, design=mock_design)
    factors = {
        ("A", "M"): 0.25,
        ("A", "F"): 0.125,
        ("B", "M"): 0.5,
        ("B", "F"): 0.125,
    }
    out = sample.weighting.poststratify(shares=factors, cells=("region", "sex"))

    totals = (
        out.data.group_by(["region", "sex"])
        .agg(pl.col(PS_WGT).sum().alias("sum_w"))
        .sort(["region", "sex"])
    )
    expected_by_cell = {k: 80 * v for k, v in factors.items()}
    assert_allclose(
        totals["sum_w"].to_numpy(),
        [
            expected_by_cell[("A", "F")],
            expected_by_cell[("A", "M")],
            expected_by_cell[("B", "F")],
            expected_by_cell[("B", "M")],
        ],
        rtol=1e-6,
    )


def test_poststratify_by_missing_column_raises(mock_design):
    sample = Sample(data=_two_way_sample(), design=mock_design)
    with pytest.raises(Exception, match="All `cells` columns must exist"):
        sample.weighting.poststratify(controls=100.0, cells=("region", "DOES_NOT_EXIST"))


def test_poststratify_by_list_accepted(mock_design):
    """Lists of strings are now accepted (equivalent to tuple)."""
    df = _two_way_sample()
    sample = Sample(data=df, design=mock_design)
    controls = {
        ("A", "M"): 30.0,
        ("A", "F"): 10.0,
        ("B", "M"): 50.0,
        ("B", "F"): 30.0,
    }
    out = sample.weighting.poststratify(controls=controls, cells=["region", "sex"])

    totals = (
        out.data.group_by(["region", "sex"])
        .agg(pl.col(PS_WGT).sum().alias("sum_w"))
        .sort(["region", "sex"])
    )
    assert_allclose(
        totals["sum_w"].to_numpy(),
        [controls[("A", "F")], controls[("A", "M")], controls[("B", "F")], controls[("B", "M")]],
        rtol=1e-6,
    )


def test_poststratify_tuple_key_controls_mismatch_raises(mock_design):
    sample = Sample(data=_two_way_sample(), design=mock_design)
    bad_controls = {
        ("A", "M"): 30.0,
        ("A", "F"): 10.0,
        ("B", "M"): 50.0,
        ("B", "X"): 30.0,  # 'X' not present
    }
    with pytest.raises(Exception, match="mismatch|missing|Mapping keys"):
        sample.weighting.poststratify(controls=bad_controls, cells=("region", "sex"))


def test_poststratify_tuple_key_factors_mismatch_raises(mock_design):
    sample = Sample(data=_two_way_sample(), design=mock_design)
    bad_factors = {
        ("A", "M"): 1.0,
        ("A", "F"): 1.0,
        ("B", "M"): 1.0,
        ("B", "X"): 1.0,  # 'X' not present
    }
    with pytest.raises(Exception, match="mismatch|missing|Mapping keys"):
        sample.weighting.poststratify(shares=bad_factors, cells=("region", "sex"))
