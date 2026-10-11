"""Kernels give the same answer on a chunked frame as on a contiguous one."""

import numpy as np
import polars as pl
import svy_rs as ps

R = 8
rng = np.random.default_rng(3)
N = 400
FRAME = pl.DataFrame(
    {
        "y": rng.normal(10, 2, N),
        "w": rng.uniform(1, 3, N),
        "c": rng.integers(0, 3, N),
        **{f"rw{r}": rng.uniform(0, 3, N) for r in range(R)},
    }
)
CHUNKED = pl.concat([FRAME.slice(0, 150), FRAME.slice(150, 100), FRAME.slice(250)], rechunk=False)
COLS = [f"rw{r}" for r in range(R)]
COEFS = [1 / R] * R


def test_frame_is_chunked():
    assert max(CHUNKED.n_chunks("all")) > 1


def test_replicate_mean_chunked_matches_contiguous():
    want, _ = ps.replicate_mean(FRAME, "y", "w", COLS, COEFS)
    got, _ = ps.replicate_mean(CHUNKED, "y", "w", COLS, COEFS)
    assert got.equals(want)


def test_replicate_prop_chunked_matches_contiguous():
    want, want_cov = ps.replicate_prop(FRAME, "c", "w", COLS, COEFS)
    got, got_cov = ps.replicate_prop(CHUNKED, "c", "w", COLS, COEFS)
    assert got.equals(want)
    assert got_cov == want_cov
