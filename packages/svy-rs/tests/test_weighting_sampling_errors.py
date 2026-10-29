"""Weighting, sampling and calibration-sweep kernels refuse input they used to replace silently."""

import numpy as np
import polars as pl
import pytest

import svy_rs as ps

FRAME = np.arange(6, dtype=np.int64)
STRATUM = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
MOS = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
WGT = np.ones(6)
PSU = np.arange(6, dtype=np.int64)
X = np.column_stack([np.ones(6), np.arange(1.0, 7.0)])


@pytest.fixture
def df():
    return pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "w": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            "w0": [1.0] * 6,
            "cell": pl.Series([0, 0, 0, 1, 1, 1], dtype=pl.Int32),
        }
    )


INVALID_INPUT = [
    (
        lambda: ps.select_srs_rs(FRAME, n_scalar=2),
        "INVALID_VALUE",
        "a seed is required",
    ),
    (
        lambda: ps.select_pps_rs(FRAME, MOS, n_scalar=2),
        "INVALID_VALUE",
        "a seed is required",
    ),
    (
        lambda: ps.select_srs_rs(FRAME, n_map={0: 1}, stratum=STRATUM, seed=1),
        "INVALID_VALUE",
        "n_map has no entry for stratum 1",
    ),
    (
        lambda: ps.select_pps_rs(FRAME, MOS, n_map={0: 1}, stratum=STRATUM, seed=1),
        "INVALID_VALUE",
        "n_map has no entry for stratum 1",
    ),
    (
        lambda: ps.select_srs_rs(
            FRAME, n_map={0: 1, 1: 1, 5: 1}, stratum=STRATUM, seed=1
        ),
        "INVALID_CHOICE",
        "n_map names stratum 5, which is not in the frame",
    ),
    (
        lambda: ps.select_pps_rs(
            FRAME, MOS, n_map={0: 1, 1: 1, 5: 1}, stratum=STRATUM, seed=1
        ),
        "INVALID_CHOICE",
        "n_map names stratum 5, which is not in the frame",
    ),
    (
        lambda: ps.select_pps_rs(FRAME, MOS, n_scalar=2, method="zzz", seed=1),
        "INVALID_CHOICE",
        'Unknown PPS method "zzz". Use: sys, wr, brewer, murphy, rs',
    ),
    (
        lambda: ps.select_srs_rs(FRAME, n_scalar=2, n_map={0: 1}, seed=1),
        "INVALID_VALUE",
        "Provide either n_scalar or n_map, not both",
    ),
    (
        lambda: ps.select_srs_rs(FRAME, n_scalar=7, seed=1),
        "SELECTION_N_EXCEEDS_GROUP",
        "Cannot draw n=7 without replacement from population of 6",
    ),
    (
        lambda: ps.create_bootstrap_wgts(WGT, PSU, 5),
        "INVALID_VALUE",
        "a seed is required",
    ),
    (
        lambda: ps._internal.create_poisson_bootstrap_wgts(WGT, 5),
        "INVALID_VALUE",
        "a seed is required",
    ),
    (
        lambda: ps.create_jk_wgts(WGT, PSU, STRATUM, paired=True),
        "INVALID_VALUE",
        "a seed is required",
    ),
    (
        lambda: ps.calibrate(
            WGT[:, None], X, np.array([6.0, 21.0]), np.array([1, 0, 1, 1, 1, 1.0])
        ),
        "INVALID_RANGE",
        "calibration scale must be finite and positive, got 0 at row 1",
    ),
    (
        lambda: ps.calibrate(
            WGT[:, None], X, np.array([6.0, 21.0]), np.array([1, 1, 1, -2, 1, 1.0])
        ),
        "INVALID_RANGE",
        "got -2 at row 3",
    ),
    (
        lambda: ps.calibrate(
            WGT[:, None], X, np.array([6.0, 21.0]), np.array([1, 1, np.nan, 1, 1, 1.0])
        ),
        "INVALID_RANGE",
        "got NaN at row 2",
    ),
    (
        lambda: ps.calibrate_by_domain(
            WGT[:, None], X, STRATUM, {0: np.array([3.0, 6.0])}
        ),
        "INVALID_VALUE",
        "calibration domain 1 has no control totals",
    ),
]


@pytest.mark.parametrize(("call", "code", "message"), INVALID_INPUT)
def test_invalid_input_is_a_coded_value_error(call, code, message):
    with pytest.raises(ps.KernelValueError) as info:
        call()
    assert info.value.code == code
    assert message in str(info.value)


def test_unpaired_jackknife_needs_no_seed():
    reps, _, _ = ps.create_jk_wgts(WGT, PSU)
    assert reps.shape == (6, 6)


SWEEP_INPUT = [
    (
        {"calib_kind": "trimming", "calib_prev_wgt": "w0"},
        "INVALID_CHOICE",
        "unknown calibration kind 'trimming'",
    ),
    (
        {"calib_kind": "calibration", "calib_prev_wgt": "w0"},
        "INVALID_VALUE",
        "a calibration sweep needs calib_aux",
    ),
    (
        {"calib_kind": "raking", "calib_prev_wgt": "w0"},
        "INVALID_VALUE",
        "a cell sweep needs calib_cells",
    ),
    (
        {"calib_kind": "poststratification", "calib_cells": ["cell"]},
        "INVALID_VALUE",
        "calib_kind needs calib_prev_wgt",
    ),
]


@pytest.mark.parametrize(("calib", "code", "message"), SWEEP_INPUT)
def test_bad_sweep_input_is_a_coded_value_error(df, calib, code, message):
    with pytest.raises(ps.KernelValueError) as info:
        ps.taylor_mean(df, "y", "w", **calib)
    assert info.value.code == code
    assert message in str(info.value)


def test_sweep_column_missing_is_an_internal_error(df):
    with pytest.raises(ps.KernelRuntimeError) as info:
        ps.taylor_mean(
            df,
            "y",
            "w",
            calib_kind="poststratification",
            calib_cells=["cell"],
            calib_prev_wgt="gone",
        )
    assert info.value.code == "KERNEL_INTERNAL"
    assert "gone" in str(info.value)
