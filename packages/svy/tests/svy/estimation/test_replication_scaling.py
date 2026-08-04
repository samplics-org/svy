# tests/svy/estimation/test_replication_scaling.py
"""Guard against O(B^2) cost in replication estimation, B = replicate count.

The replicate kernels are O(n*B), but the Python prep layer used to test
membership against ``df.columns`` once per replicate weight column. Each access
rebuilds the entire column-name list across the FFI boundary, so B lookups cost
O(B^2) Python-string constructions. At bootstrap replicate counts this
dominated: with n*B held constant at 2e7 cells, B=800 ran ~10x slower than
B=100 despite doing identical arithmetic on an identically sized matrix.

These tests assert the *structural* property that produced the blowup --
per-call accesses to column metadata must not grow with B -- rather than
wall-clock time, which is too noisy to assert on in CI.
"""

import numpy as np
import polars as pl
import pytest

import svy


N_ROWS = 2_000
SMALL_REPS = 16
LARGE_REPS = 256


def _make_sample(n_reps: int, n_rows: int = N_ROWS) -> svy.Sample:
    """A tiny replication design with ``n_reps`` bootstrap replicate weights."""
    rng = np.random.default_rng(20260804)
    data = {
        "psu": np.repeat(np.arange(40, dtype=np.int64), n_rows // 40),
        "wgt": rng.uniform(5.0, 20.0, size=n_rows),
        "y": rng.normal(10.0, 3.0, size=n_rows),
        "x": rng.normal(20.0, 4.0, size=n_rows),
    }
    for r in range(n_reps):
        data[f"bsrw{r + 1}"] = data["wgt"] * rng.integers(0, 3, size=n_rows)
    df = pl.DataFrame(data)

    design = svy.Design(
        wgt="wgt",
        rep_wgts=svy.RepWeights(method="bootstrap", prefix="bsrw", n_reps=n_reps),
    )
    return svy.Sample(data=df, design=design)


@pytest.fixture
def count_columns_access(monkeypatch):
    """Count ``pl.DataFrame.columns`` property accesses inside a with-block."""
    counter = {"n": 0}
    original = pl.DataFrame.columns

    def counting_fget(self):
        counter["n"] += 1
        return original.fget(self)

    monkeypatch.setattr(pl.DataFrame, "columns", property(counting_fget))
    return counter


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda s: s.estimation.mean("y", method="replication"), id="mean"),
        pytest.param(lambda s: s.estimation.total("y", method="replication"), id="total"),
        pytest.param(lambda s: s.estimation.ratio("y", "x", method="replication"), id="ratio"),
    ],
)
def test_column_metadata_access_does_not_scale_with_replicates(count_columns_access, call):
    """A 16x larger replicate count must not mean 16x more `.columns` accesses.

    Pre-fix this grew as ~2*B (two call sites each looping over every replicate
    column), which is what made total cost quadratic.
    """
    small = _make_sample(SMALL_REPS)
    large = _make_sample(LARGE_REPS)

    # Warm any per-sample caches so we measure steady-state estimation cost,
    # not one-off design resolution.
    call(small)
    call(large)

    count_columns_access["n"] = 0
    call(small)
    small_accesses = count_columns_access["n"]

    count_columns_access["n"] = 0
    call(large)
    large_accesses = count_columns_access["n"]

    # Allow a small constant slack for incidental per-call metadata lookups,
    # but nothing proportional to the 240 extra replicate columns.
    assert large_accesses <= small_accesses + 10, (
        f"`.columns` accesses grew with replicate count "
        f"({small_accesses} at B={SMALL_REPS} -> {large_accesses} at B={LARGE_REPS}); "
        f"an O(B) access pattern makes replication estimation O(B^2)."
    )


def test_replicate_estimates_are_unchanged_by_column_lookup_shape():
    """The optimisation must be numerically inert.

    `_ensure_float64` now reads dtypes from a single schema snapshot rather than
    per-column `data[c].dtype`; non-Float64 replicate weights must still be cast.
    """
    sample = _make_sample(SMALL_REPS)
    ref = sample.estimation.mean("y", method="replication").estimates[0]

    # Same data, but replicate weights stored as Float32 so the cast path runs.
    df = sample._data
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    cast_df = df.with_columns([pl.col(f"bsrw{r + 1}").cast(pl.Float32) for r in range(SMALL_REPS)])
    design = svy.Design(
        wgt="wgt",
        rep_wgts=svy.RepWeights(method="bootstrap", prefix="bsrw", n_reps=SMALL_REPS),
    )
    recast = svy.Sample(data=cast_df, design=design)
    got = recast.estimation.mean("y", method="replication").estimates[0]

    assert got.est == pytest.approx(ref.est, rel=1e-6)
    assert got.se == pytest.approx(ref.se, rel=1e-6)
