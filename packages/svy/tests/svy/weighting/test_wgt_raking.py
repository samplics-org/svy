# tests/svy/weighting/test_wgt_raking.py
import polars as pl
import pytest

from numpy.testing import assert_allclose

from svy import EstimationMethod
from svy.core.sample import Design, Sample


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

RK_WGT = "rk_wgt"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_design():
    return Design(wgt="initial_weight")


@pytest.fixture
def sample_data_for_raking():
    """Simple dataset with two margins summing to total weight 80."""
    return pl.DataFrame(
        {
            "initial_weight": [10.0] * 8,
            "age_group": ["18-34", "35-54", "55+", "18-34", "35-54", "55+", "18-34", "35-54"],
            "region": ["North", "North", "North", "North", "South", "South", "South", "South"],
        }
    )


# ---------------------------------------------------------------------------
# Core raking correctness
# ---------------------------------------------------------------------------


def test_rake_successful_convergence_with_controls(sample_data_for_raking, mock_design):
    sample = Sample(data=sample_data_for_raking, design=mock_design)
    controls = {
        "age_group": {"18-34": 35.0, "35-54": 30.0, "55+": 15.0},
        "region": {"North": 50.0, "South": 30.0},
    }
    sample = sample.weighting.rake(controls=controls)
    raked = sample.data
    col = RK_WGT

    assert col in raked.columns

    age_sums = raked.group_by("age_group").agg(pl.col(col).sum()).sort("age_group")
    region_sums = raked.group_by("region").agg(pl.col(col).sum()).sort("region")

    assert_allclose(age_sums[col].to_numpy(), [35.0, 30.0, 15.0], rtol=1e-3)
    assert_allclose(region_sums[col].to_numpy(), [50.0, 30.0], rtol=1e-3)


def test_rake_successful_convergence_with_factors_empty_dict(sample_data_for_raking, mock_design):
    sample = Sample(data=sample_data_for_raking, design=mock_design)
    factors = {
        "age_group": {"18-34": 0.8, "35-54": 1.0, "55+": 1.3},
        "region": {"North": 1.25, "South": 0.75},
    }
    sample.weighting.rake(controls={}, factors=factors)
    col = RK_WGT
    raked = sample.data

    age_sums = raked.group_by("age_group").agg(pl.col(col).sum()).sort("age_group")
    region_sums = raked.group_by("region").agg(pl.col(col).sum()).sort("region")

    assert_allclose(age_sums[col].to_numpy(), [24.0, 30.0, 26.0], rtol=1e-3)
    assert_allclose(region_sums[col].to_numpy(), [50.0, 30.0], rtol=1e-3)


def test_rake_successful_convergence_with_factors(sample_data_for_raking, mock_design):
    sample = Sample(data=sample_data_for_raking, design=mock_design)
    factors = {
        "age_group": {"18-34": 0.8, "35-54": 1.0, "55+": 1.3},
        "region": {"North": 1.25, "South": 0.75},
    }
    sample.weighting.rake(controls=None, factors=factors)
    col = RK_WGT
    raked = sample.data

    age_sums = raked.group_by("age_group").agg(pl.col(col).sum()).sort("age_group")
    region_sums = raked.group_by("region").agg(pl.col(col).sum()).sort("region")

    assert_allclose(age_sums[col].to_numpy(), [24.0, 30.0, 26.0], rtol=1e-3)
    assert_allclose(region_sums[col].to_numpy(), [50.0, 30.0], rtol=1e-3)


# ---------------------------------------------------------------------------
# Auto-generated column name
# ---------------------------------------------------------------------------


def test_rake_auto_col_name_uses_rk_wgt(sample_data_for_raking, mock_design):
    """Auto-generated raked weight column must be rk_wgt."""
    sample = Sample(data=sample_data_for_raking, design=mock_design)
    controls = {
        "age_group": {"18-34": 35.0, "35-54": 30.0, "55+": 15.0},
        "region": {"North": 50.0, "South": 30.0},
    }
    sample = sample.weighting.rake(controls=controls)
    assert RK_WGT in sample.data.columns
    assert sample.design.wgt == RK_WGT


def test_rake_wgt_name_overrides_default(sample_data_for_raking, mock_design):
    """Explicit wgt_name overrides auto-generated name."""
    sample = Sample(data=sample_data_for_raking, design=mock_design)
    controls = {
        "age_group": {"18-34": 35.0, "35-54": 30.0, "55+": 15.0},
        "region": {"North": 50.0, "South": 30.0},
    }
    sample = sample.weighting.rake(controls=controls, wgt_name="my_raked_wgt")
    assert "my_raked_wgt" in sample.data.columns
    assert RK_WGT not in sample.data.columns
    assert sample.design.wgt == "my_raked_wgt"


# ---------------------------------------------------------------------------
# Replicate weights
# ---------------------------------------------------------------------------


def test_rake_replicate_weights_auto_prefix(sample_data_for_raking, mock_design):
    """Replicate weight prefix matches the full-sample weight name."""
    import numpy as np
    import polars as pl

    df = sample_data_for_raking.with_columns(
        pl.Series("rw1", [1.0] * sample_data_for_raking.height),
        pl.Series("rw2", [1.0] * sample_data_for_raking.height),
    )
    sample = Sample(data=df, design=mock_design)
    sample._design = sample.design.update_rep_weights(
        method=EstimationMethod.BOOTSTRAP, prefix="rw", n_reps=2
    )
    controls = {
        "age_group": {"18-34": 35.0, "35-54": 30.0, "55+": 15.0},
        "region": {"North": 50.0, "South": 30.0},
    }
    sample = sample.weighting.rake(controls=controls)
    assert f"{RK_WGT}1" in sample.data.columns
    assert f"{RK_WGT}2" in sample.data.columns
    assert sample.design.rep_wgts.prefix == RK_WGT


def test_rake_replicate_weights_custom_wgt_name(sample_data_for_raking, mock_design):
    """Custom wgt_name propagates to replicate weight prefix."""
    import polars as pl

    df = sample_data_for_raking.with_columns(
        pl.Series("rw1", [1.0] * sample_data_for_raking.height),
        pl.Series("rw2", [1.0] * sample_data_for_raking.height),
    )
    sample = Sample(data=df, design=mock_design)
    sample._design = sample.design.update_rep_weights(
        method=EstimationMethod.BOOTSTRAP, prefix="rw", n_reps=2
    )
    controls = {
        "age_group": {"18-34": 35.0, "35-54": 30.0, "55+": 15.0},
        "region": {"North": 50.0, "South": 30.0},
    }
    sample = sample.weighting.rake(controls=controls, wgt_name="my_rk")
    assert "my_rk" in sample.data.columns
    assert "my_rk1" in sample.data.columns
    assert "my_rk2" in sample.data.columns
    assert sample.design.rep_wgts.prefix == "my_rk"


# ---------------------------------------------------------------------------
# Convergence / error cases
# ---------------------------------------------------------------------------


def test_rake_stops_at_max_iter_on_no_convergence(sample_data_for_raking, mock_design, capsys):
    sample = Sample(data=sample_data_for_raking, design=mock_design)
    controls = {
        "age_group": {"18-34": 35.0, "35-54": 30.0, "55+": 15.0},
        "region": {"North": 50.0, "South": 30.0},
    }
    # strict=True (default): raises on non-convergence, design unchanged
    original_wgt = sample.design.wgt
    with pytest.raises(Exception, match="did not converge|Raking did not converge"):
        sample.weighting.rake(controls=controls, max_iter=3, tol=1e-20)
    assert sample.design.wgt == original_wgt  # design not mutated

    # strict=False: stores partial weights, prints warning
    sample.weighting.rake(controls=controls, max_iter=3, tol=1e-20, strict=False)
    captured = capsys.readouterr()
    assert "Warning: Raking did not converge after 3 iterations" in captured.out


def test_rake_stops_when_bounds_exceeded(sample_data_for_raking, mock_design):
    sample = Sample(data=sample_data_for_raking, design=mock_design)
    controls = {"region": {"North": 50.0, "South": 30.0}}
    with pytest.raises(
        ValueError, match="Raking failed: Weight ratios exceeded specified bounds."
    ):
        sample.weighting.rake(controls=controls, up_bound=1.2)


def test_rake_raises_error_unknown_control_column(sample_data_for_raking, mock_design):
    sample = Sample(data=sample_data_for_raking, design=mock_design)
    with pytest.raises(Exception, match="not_a_column|not found"):
        sample.weighting.rake(controls={"not_a_column": {"X": 10.0}})


def test_rake_raises_error_no_control_or_factor(sample_data_for_raking, mock_design):
    sample = Sample(data=sample_data_for_raking, design=mock_design)
    with pytest.raises(Exception, match="Either.*control.*factor|controls.*factors"):
        sample.weighting.rake(controls=None)


# ---------------------------------------------------------------------------
# Trim-rake cycle
# ---------------------------------------------------------------------------


def _skewed_raking_sample():
    """Dataset with one extreme weight — raking + trimming scenario.

    8 units, equal base weights of 10 except unit 0 which is 80 (8x extreme).
    Two margins: age_group and region.
    Controls are set so raking converges easily.
    """
    import numpy as np

    return pl.DataFrame(
        {
            "initial_weight": [80.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            "age_group": ["18-34", "35-54", "55+", "18-34", "35-54", "55+", "18-34", "35-54"],
            "region": ["North", "North", "North", "North", "South", "South", "South", "South"],
        }
    )


def test_trim_rake_produces_raked_result(mock_design):
    """With trimming=None, rake() behaves exactly as before."""
    from svy.weighting.types import TrimConfig

    sample = Sample(data=_skewed_raking_sample(), design=mock_design)
    controls = {
        "age_group": {"18-34": 35.0, "35-54": 30.0, "55+": 15.0},
        "region": {"North": 50.0, "South": 30.0},
    }
    out = sample.weighting.rake(controls=controls, trimming=None)
    col = RK_WGT
    assert col in out.data.columns

    age_sums = out.data.group_by("age_group").agg(pl.col(col).sum()).sort("age_group")
    region_sums = out.data.group_by("region").agg(pl.col(col).sum()).sort("region")
    assert_allclose(age_sums[col].to_numpy(), [35.0, 30.0, 15.0], rtol=1e-3)
    assert_allclose(region_sums[col].to_numpy(), [50.0, 30.0], rtol=1e-3)


def test_trim_rake_final_step_is_rake(mock_design):
    """After trim-rake cycle, margins must be satisfied (final step = rake)."""
    from svy.weighting.types import TrimConfig

    sample = Sample(data=_skewed_raking_sample(), design=mock_design)
    controls = {
        "age_group": {"18-34": 35.0, "35-54": 30.0, "55+": 15.0},
        "region": {"North": 50.0, "South": 30.0},
    }
    out = sample.weighting.rake(
        controls=controls,
        trimming=TrimConfig(upper=30.0, redistribute=True),
    )
    col = RK_WGT

    age_sums = out.data.group_by("age_group").agg(pl.col(col).sum()).sort("age_group")
    region_sums = out.data.group_by("region").agg(pl.col(col).sum()).sort("region")

    # Margins must be satisfied after trim-rake
    assert_allclose(age_sums[col].to_numpy(), [35.0, 30.0, 15.0], rtol=1e-3)
    assert_allclose(region_sums[col].to_numpy(), [50.0, 30.0], rtol=1e-3)


def test_trim_rake_caps_extreme_weight(mock_design):
    """Trimming should cap the extreme weight even after cycling."""
    from svy.weighting.types import TrimConfig

    sample = Sample(data=_skewed_raking_sample(), design=mock_design)
    controls = {
        "age_group": {"18-34": 35.0, "35-54": 30.0, "55+": 15.0},
        "region": {"North": 50.0, "South": 30.0},
    }
    out = sample.weighting.rake(
        controls=controls,
        trimming=TrimConfig(upper=30.0, redistribute=True),
    )
    col = RK_WGT
    weights = out.data[col].to_numpy()

    # No weight should exceed the cap (with small tolerance for redistribution)
    assert weights.max() <= 30.0 * 1.01


def test_trim_rake_single_cycle_max_iter_1(mock_design):
    """max_iter=1 means one rake + one trim + one final rake — three total Rust calls."""
    from svy.weighting.types import TrimConfig

    sample = Sample(data=_skewed_raking_sample(), design=mock_design)
    controls = {
        "age_group": {"18-34": 35.0, "35-54": 30.0, "55+": 15.0},
        "region": {"North": 50.0, "South": 30.0},
    }
    # max_iter=1: one cycle, should still return a result (strict=False in case
    # one cycle isn't enough for full convergence)
    out = sample.weighting.rake(
        controls=controls,
        trimming=TrimConfig(upper=30.0, redistribute=True),
        max_iter=1,
        strict=False,
    )
    col = RK_WGT
    assert col in out.data.columns
    assert len(out.data[col].to_numpy()) == 8


def test_trim_rake_without_trimming_unchanged(mock_design):
    """rake() with no trimming gives identical result to previous behaviour."""
    import numpy as np
    from svy.weighting.types import TrimConfig

    controls = {
        "age_group": {"18-34": 35.0, "35-54": 30.0, "55+": 15.0},
        "region": {"North": 50.0, "South": 30.0},
    }
    col = RK_WGT

    s1 = Sample(data=_skewed_raking_sample(), design=Design(wgt="initial_weight"))
    s2 = Sample(data=_skewed_raking_sample(), design=Design(wgt="initial_weight"))

    out1 = s1.weighting.rake(controls=controls)
    out2 = s2.weighting.rake(controls=controls, trimming=None)

    assert_allclose(
        out1.data[col].to_numpy(),
        out2.data[col].to_numpy(),
        rtol=1e-10,
    )


def test_trim_rake_strict_raises_on_non_convergence(mock_design):
    """strict=True raises if trim-rake cycle doesn't fully converge."""
    from svy.weighting.types import TrimConfig

    sample = Sample(data=_skewed_raking_sample(), design=mock_design)
    controls = {
        "age_group": {"18-34": 35.0, "35-54": 30.0, "55+": 15.0},
        "region": {"North": 50.0, "South": 30.0},
    }
    original_wgt = sample.design.wgt

    with pytest.raises(Exception, match="converge"):
        sample.weighting.rake(
            controls=controls,
            trimming=TrimConfig(upper=30.0, redistribute=True),
            max_iter=1,
            tol=1e-20,  # effectively impossible to satisfy
            strict=True,
        )

    # Design must not have been mutated
    assert sample.design.wgt == original_wgt
