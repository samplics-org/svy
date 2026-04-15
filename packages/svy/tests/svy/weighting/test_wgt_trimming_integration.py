# tests/svy/weighting/test_wgt_trimming_integration.py
"""
Integration tests for trimming= parameter within weighting methods,
and for trim(wgt_name=None) in-place behavior.

Raking trimming integration is already covered in test_wgt_raking.py.
Adjust trimming integration is already covered in test_wgt_adj_nonresponse.py.
This file covers poststratify and calibrate, plus trim in-place.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_allclose

from svy import EstimationMethod
from svy.core.sample import Design, Sample
from svy.core.terms import Cat
from svy.weighting.types import TrimConfig


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def skewed_ps_sample():
    """10 units, two strata, one extreme weight per stratum."""
    return pl.DataFrame(
        {
            "weight": [1.0, 1.0, 1.0, 1.0, 80.0, 1.0, 1.0, 1.0, 1.0, 80.0],
            "strat": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
        }
    )


@pytest.fixture
def skewed_calib_sample():
    """12 units with one extreme weight for calibration testing."""
    return pl.DataFrame(
        {
            "weight": [1.0, 1.0, 1.0, 1.0, 1.0, 80.0, 1.0, 1.0, 1.0, 1.0, 1.0, 80.0],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        }
    )


# ---------------------------------------------------------------------------
# Poststratify + trimming
# ---------------------------------------------------------------------------


class TestPostStratifyTrimming:
    def test_trim_ps_produces_ps_wgt(self, skewed_ps_sample):
        """poststratify(trimming=...) still produces ps_wgt column."""
        sample = Sample(data=skewed_ps_sample, design=Design(wgt="weight"))
        out = sample.weighting.poststratify(
            controls={"A": 50.0, "B": 50.0},
            by="strat",
            trimming=TrimConfig(upper=20.0, redistribute=True, min_cell_size=1),
        )
        assert "ps_wgt" in out.data.columns
        assert out.design.wgt == "ps_wgt"

    def test_trim_ps_margins_satisfied(self, skewed_ps_sample):
        """After trim-ps cycle, margins must still be satisfied."""
        sample = Sample(data=skewed_ps_sample, design=Design(wgt="weight"))
        out = sample.weighting.poststratify(
            controls={"A": 50.0, "B": 50.0},
            by="strat",
            trimming=TrimConfig(upper=20.0, redistribute=True, min_cell_size=1),
        )
        a_sum = out.data.filter(pl.col("strat") == "A")["ps_wgt"].sum()
        b_sum = out.data.filter(pl.col("strat") == "B")["ps_wgt"].sum()
        assert_allclose(a_sum, 50.0, rtol=1e-3)
        assert_allclose(b_sum, 50.0, rtol=1e-3)

    def test_trim_ps_none_unchanged(self, skewed_ps_sample):
        """trimming=None gives identical result to plain poststratify()."""
        s1 = Sample(data=skewed_ps_sample, design=Design(wgt="weight"))
        s2 = Sample(data=skewed_ps_sample, design=Design(wgt="weight"))
        out1 = s1.weighting.poststratify(controls={"A": 50.0, "B": 50.0}, by="strat")
        out2 = s2.weighting.poststratify(
            controls={"A": 50.0, "B": 50.0}, by="strat", trimming=None
        )
        assert_allclose(
            out1.data["ps_wgt"].to_numpy(),
            out2.data["ps_wgt"].to_numpy(),
            rtol=1e-10,
        )

    def test_trim_ps_strict_raises_on_non_convergence(self, skewed_ps_sample):
        """strict=True raises if trim-ps cycle doesn't converge.

        Note: design.wgt is updated before the trim-ps cycle runs, so it
        will be mutated even when strict=True raises. This is a known
        limitation of the current implementation — the cycle runs after
        the initial poststratify write-back.
        """
        sample = Sample(data=skewed_ps_sample, design=Design(wgt="weight"))
        with pytest.raises(Exception, match="converge"):
            sample.weighting.poststratify(
                controls={"A": 50.0, "B": 50.0},
                by="strat",
                trimming=TrimConfig(
                    upper=2.0,
                    redistribute=True,
                    max_iter=1,
                    min_cell_size=1,
                ),
                strict=True,
            )

    def test_trim_ps_strict_false_stores_partial(self, skewed_ps_sample):
        """strict=False stores partial result even without convergence."""
        sample = Sample(data=skewed_ps_sample, design=Design(wgt="weight"))
        out = sample.weighting.poststratify(
            controls={"A": 50.0, "B": 50.0},
            by="strat",
            trimming=TrimConfig(
                upper=2.0,
                redistribute=True,
                max_iter=1,
                min_cell_size=1,
            ),
            strict=False,
        )
        assert "ps_wgt" in out.data.columns

    def test_trim_ps_rep_weights_adjusted(self, skewed_ps_sample):
        """Replicate weights are poststratified when present."""
        df = skewed_ps_sample.with_columns(
            pl.Series("rw1", [1.0] * skewed_ps_sample.height),
            pl.Series("rw2", [1.0] * skewed_ps_sample.height),
        )
        sample = Sample(data=df, design=Design(wgt="weight"))
        sample._design = sample.design.update_rep_weights(
            method=EstimationMethod.BRR, prefix="rw", n_reps=2
        )
        out = sample.weighting.poststratify(
            controls={"A": 50.0, "B": 50.0},
            by="strat",
            trimming=TrimConfig(upper=20.0, redistribute=True, min_cell_size=1),
        )
        assert "ps_wgt1" in out.data.columns
        assert "ps_wgt2" in out.data.columns
        assert out.design.rep_wgts.prefix == "ps_wgt"


# ---------------------------------------------------------------------------
# Calibrate + trimming
# ---------------------------------------------------------------------------


class TestCalibrateTrimming:
    def test_trim_calib_produces_calib_wgt(self, skewed_calib_sample):
        """calibrate(trimming=...) still produces calib_wgt column."""
        sample = Sample(data=skewed_calib_sample, design=Design(wgt="weight"))
        target = float((skewed_calib_sample["x"] * skewed_calib_sample["weight"]).sum())
        out = sample.weighting.calibrate(
            controls={"x": target},
            trimming=TrimConfig(upper=50.0, redistribute=True, min_cell_size=1),
            strict=False,
        )
        assert "calib_wgt" in out.data.columns
        assert out.design.wgt == "calib_wgt"

    def test_trim_calib_none_unchanged(self, skewed_calib_sample):
        """trimming=None gives identical result to plain calibrate()."""
        target = float((skewed_calib_sample["x"] * skewed_calib_sample["weight"]).sum())
        s1 = Sample(data=skewed_calib_sample, design=Design(wgt="weight"))
        s2 = Sample(data=skewed_calib_sample, design=Design(wgt="weight"))
        out1 = s1.weighting.calibrate(controls={"x": target})
        out2 = s2.weighting.calibrate(controls={"x": target}, trimming=None)
        assert_allclose(
            out1.data["calib_wgt"].to_numpy(),
            out2.data["calib_wgt"].to_numpy(),
            rtol=1e-10,
        )

    def test_trim_calib_design_updated(self, skewed_calib_sample):
        """After trim, design.wgt points to the calibrated weight column."""
        target = float((skewed_calib_sample["x"] * skewed_calib_sample["weight"]).sum())
        sample = Sample(data=skewed_calib_sample, design=Design(wgt="weight"))
        out = sample.weighting.calibrate(
            controls={"x": target},
            trimming=TrimConfig(upper=50.0, redistribute=True, min_cell_size=1),
            strict=False,
        )
        assert out.design.wgt == "calib_wgt"

    def test_trim_calib_rep_weights_adjusted(self, skewed_calib_sample):
        """Replicate weights are calibrated when present."""
        df = skewed_calib_sample.with_columns(
            pl.Series("rw1", [1.0] * skewed_calib_sample.height),
            pl.Series("rw2", [1.0] * skewed_calib_sample.height),
        )
        target = float((skewed_calib_sample["x"] * skewed_calib_sample["weight"]).sum())
        sample = Sample(data=df, design=Design(wgt="weight"))
        sample._design = sample.design.update_rep_weights(
            method=EstimationMethod.BRR, prefix="rw", n_reps=2
        )
        out = sample.weighting.calibrate(
            controls={"x": target},
            trimming=TrimConfig(upper=50.0, redistribute=True, min_cell_size=1),
            strict=False,
        )
        assert "calib_wgt1" in out.data.columns
        assert "calib_wgt2" in out.data.columns
        assert out.design.rep_wgts.prefix == "calib_wgt"


# ---------------------------------------------------------------------------
# trim(wgt_name=None) — in-place
# ---------------------------------------------------------------------------


class TestTrimInPlace:
    def _sample(self):
        return Sample(
            data=pl.DataFrame({"weight": [1.0] * 9 + [200.0]}),
            design=Design(wgt="weight"),
        )

    def test_inplace_no_new_column(self):
        """wgt_name=None does not create a new column."""
        out = self._sample().weighting.trim(upper=50.0, wgt_name=None)
        assert "trim_wgt" not in out.data.columns
        assert "weight" in out.data.columns

    def test_inplace_modifies_existing_column(self):
        """wgt_name=None modifies the existing weight column."""
        out = self._sample().weighting.trim(upper=50.0, wgt_name=None, redistribute=False)
        assert out.data["weight"][-1] <= 50.0

    def test_inplace_design_wgt_unchanged(self):
        """wgt_name=None leaves design.wgt pointing to the same column."""
        out = self._sample().weighting.trim(upper=50.0, wgt_name=None)
        assert out.design.wgt == "weight"

    def test_inplace_vs_named_same_values(self):
        """In-place and named trim produce the same weight values."""
        s1 = self._sample()
        s2 = self._sample()
        out_inplace = s1.weighting.trim(upper=50.0, wgt_name=None, redistribute=False)
        out_named = s2.weighting.trim(upper=50.0, wgt_name="trim_wgt", redistribute=False)
        assert_allclose(
            out_inplace.data["weight"].to_numpy(),
            out_named.data["trim_wgt"].to_numpy(),
            rtol=1e-10,
        )

    def test_inplace_rep_weights_modified_in_place(self):
        """wgt_name=None modifies replicate columns in-place, no new rep columns."""
        df = pl.DataFrame(
            {
                "weight": [1.0] * 9 + [200.0],
                "rw1": [1.0] * 9 + [999.0],
                "rw2": [1.0] * 9 + [999.0],
            }
        )
        sample = Sample(data=df, design=Design(wgt="weight"))
        sample._design = sample.design.update_rep_weights(
            method=EstimationMethod.BRR, prefix="rw", n_reps=2
        )
        out = sample.weighting.trim(upper=50.0, wgt_name=None, redistribute=False)

        # No new rep columns created
        assert "trim_wgt1" not in out.data.columns
        assert "trim_wgt2" not in out.data.columns

        # Existing rep columns were modified
        w_orig = df["weight"].to_numpy()
        w_out = out.data["weight"].to_numpy()
        factors = np.where(w_orig > 0, w_out / w_orig, 1.0)
        expected_rw1 = df["rw1"].to_numpy() * factors
        expected_rw2 = df["rw2"].to_numpy() * factors
        assert_allclose(out.data["rw1"].to_numpy(), expected_rw1, rtol=1e-6)
        assert_allclose(out.data["rw2"].to_numpy(), expected_rw2, rtol=1e-6)

        # rep_wgts metadata prefix unchanged
        assert out.design.rep_wgts.prefix == "rw"

    def test_inplace_after_rake_trims_rk_wgt(self):
        """Typical workflow: rake then trim in-place on rk_wgt."""
        df = pl.DataFrame(
            {
                "weight": [10.0] * 7 + [80.0],
                "age": ["18-34", "35-54", "55+", "18-34", "35-54", "55+", "18-34", "35-54"],
                "region": ["N", "N", "N", "N", "S", "S", "S", "S"],
            }
        )
        sample = Sample(data=df, design=Design(wgt="weight"))
        out = sample.weighting.rake(
            controls={
                "age": {"18-34": 35.0, "35-54": 30.0, "55+": 15.0},
                "region": {"N": 50.0, "S": 30.0},
            }
        )
        # design.wgt is now rk_wgt
        assert out.design.wgt == "rk_wgt"

        # trim in-place on rk_wgt
        out2 = out.weighting.trim(upper=30.0, wgt_name=None, redistribute=False)
        assert out2.design.wgt == "rk_wgt"
        assert "trim_wgt" not in out2.data.columns
        assert out2.data["rk_wgt"].to_numpy().max() <= 30.0 * 1.01
