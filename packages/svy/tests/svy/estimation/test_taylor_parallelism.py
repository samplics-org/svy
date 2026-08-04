# tests/svy/estimation/test_taylor_parallelism.py
"""Guard the work-elimination that lets the Taylor kernels use more than one core.

Single-variable Taylor estimation used ~1.15 cores of 10 at 1M rows, and the
8-variable batched call reached only ~1.8 of a possible 8. Neither was a rayon
width problem. Three pieces of redundant *serial* work dominated, and each one
is a structural property that can be asserted without timing anything:

1. The design was indexed twice per call — once in ``build_taylor_design`` and
   again inside ``degrees_of_freedom``, which re-densified the same stratum and
   PSU columns. That is Rust-side and is covered by ``svy-rs``'s own tests; what
   is observable here is the consequence, below.

2. ``sample.estimation`` returned a *new* ``Estimation`` on every attribute
   access, so its ``_data_version``-keyed design cache was discarded before it
   could ever be reused and every call re-derived the factorized design.

3. The reporting metadata on each ``Estimate`` (unique strata, PSU count) was
   computed with ``np.unique`` over the full-length design arrays *per
   estimate*. A batched call produces one ``Estimate`` per variable, so an
   8-variable mean did 16 full-length ``np.unique`` passes — 63% of that call's
   wall time at 1M rows, all of it serial.

These tests assert the structural invariants (work must not repeat per call, and
must not grow with the number of variables) rather than wall-clock time, which is
too noisy for CI. Timing lives in ``benchmarks/bench_taylor_scaling.py``.
"""

import numpy as np
import polars as pl
import pytest

import svy


N_ROWS = 20_000
N_STRATA = 10
PSU_PER_STRATUM = 20
Y_VARS = [f"y{j}" for j in range(8)]


def _make_sample(n_rows: int = N_ROWS) -> svy.Sample:
    """A stratified two-stage sample with integer design columns."""
    rng = np.random.default_rng(20260804)
    n_psu = N_STRATA * PSU_PER_STRATUM
    psu = (np.arange(n_rows, dtype=np.int64) * n_psu) // n_rows
    data = {
        "stratum": psu // PSU_PER_STRATUM,
        "psu": psu,
        "wgt": rng.uniform(5.0, 20.0, size=n_rows),
        "x": rng.normal(20.0, 4.0, size=n_rows),
        "grp": rng.integers(0, 4, size=n_rows),
    }
    for j in range(len(Y_VARS)):
        data[f"y{j}"] = rng.normal(10.0 + j, 3.0, size=n_rows)
    df = pl.DataFrame(data)
    return svy.Sample(df, svy.Design(stratum="stratum", psu="psu", wgt="wgt"))


@pytest.fixture
def count_unique(monkeypatch):
    """Count ``np.unique`` calls made on full-length arrays."""
    counter = {"n": 0}
    original = np.unique

    def counting_unique(arr, *args, **kwargs):
        if getattr(arr, "size", 0) >= N_ROWS:
            counter["n"] += 1
        return original(arr, *args, **kwargs)

    monkeypatch.setattr("svy.estimation.base.np.unique", counting_unique, raising=True)
    return counter


# ── 2. The estimation accessor must not throw away its design cache ──────────


def test_estimation_accessor_is_retained():
    """Repeated access returns the same object, so its caches survive."""
    sample = _make_sample()
    assert sample.estimation is sample.estimation


def test_design_is_factorized_once_across_calls(monkeypatch):
    """The factorized design is derived once per sample, not once per call."""
    sample = _make_sample()
    est = sample.estimation

    calls = {"n": 0}
    original = type(est)._get_factorized_design

    def counting(self):
        # Count only the cache *misses* — the rebuilds are the cost.
        if self._design_cache is None or (
            self._design_cache["_data_version"] != self._sample._data_version
        ):
            calls["n"] += 1
        return original(self)

    monkeypatch.setattr(type(est), "_get_factorized_design", counting)

    for _ in range(5):
        sample.estimation.mean("y0")

    assert calls["n"] == 1, (
        f"factorized design rebuilt {calls['n']}x over 5 calls; it is keyed on "
        "_data_version and the data never changed, so it must be built once"
    )


def test_derived_sample_does_not_inherit_parent_estimation():
    """A fork must not answer with an Estimation still bound to its parent.

    ``_replace_data`` builds derived samples with ``copy.copy``, which carries
    the cached accessor across verbatim. Without an identity check the fork
    would estimate the parent's data.
    """
    sample = _make_sample()
    parent_est = sample.estimation
    assert parent_est._sample is sample

    derived = sample._replace_data(sample._data.head(N_ROWS // 2))
    assert derived.estimation is not parent_est
    assert derived.estimation._sample is derived

    # And it must actually see the smaller frame: half the rows is half the PSUs,
    # which is exactly the memoised metadata that would go stale if it did not.
    n_psu = N_STRATA * PSU_PER_STRATUM
    assert sample.estimation.mean("y0").n_psus == n_psu
    assert derived.estimation.mean("y0").n_psus == n_psu // 2


# ── 3. Per-estimate metadata must not scale with the variable count ──────────


def test_metadata_unique_does_not_grow_with_variable_count(count_unique):
    """Unique strata/PSU labels are design properties, not per-variable ones."""
    sample = _make_sample()

    sample.estimation.mean("y0")
    one_var = count_unique["n"]

    # Fresh sample so the memo starts cold again for the batched call.
    sample2 = _make_sample()
    count_unique["n"] = 0
    sample2.estimation.mean(Y_VARS)
    eight_vars = count_unique["n"]

    assert eight_vars == one_var, (
        f"{eight_vars} full-length np.unique passes for 8 variables vs "
        f"{one_var} for 1; the design's unique strata and PSU count do not "
        "depend on the variable being estimated"
    )
    assert one_var <= 2, (
        f"{one_var} full-length np.unique passes for a single estimate; expected "
        "at most one for strata and one for PSUs"
    )


def test_metadata_is_not_recomputed_across_calls(count_unique):
    """The memo survives between calls on the same sample."""
    sample = _make_sample()
    sample.estimation.mean("y0")
    after_first = count_unique["n"]

    for _ in range(4):
        sample.estimation.mean("y0")

    assert count_unique["n"] == after_first, (
        f"{count_unique['n'] - after_first} extra np.unique passes over 4 repeat "
        "calls; the design did not change, so the metadata must be memoised"
    )


def test_metadata_invalidates_when_data_changes():
    """A memo must never outlive the design arrays it was derived from."""
    sample = _make_sample()
    first = sample.estimation.mean("y0")
    assert first.n_strata == N_STRATA

    # Keep only the first two strata; the memo must not report the old count.
    smaller = sample._replace_data(sample._data.filter(pl.col("stratum") < 2))
    second = smaller.estimation.mean("y0")
    assert second.n_strata == 2
    assert sorted(second.strata) == [0, 1]


def test_estimate_strata_lists_are_independent():
    """Each Estimate owns its strata list; the shared memo must not be aliased."""
    sample = _make_sample()
    estimates = sample.estimation.mean(Y_VARS)

    first, second = estimates[0], estimates[1]
    assert first.strata == second.strata
    assert first.strata is not second.strata

    first.strata.append("mutated")
    assert "mutated" not in second.strata
    assert "mutated" not in sample.estimation.mean("y0").strata
