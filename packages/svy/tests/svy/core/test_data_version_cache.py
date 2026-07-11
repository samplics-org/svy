# tests/svy/core/test_data_version_cache.py
"""
Regression tests for the Sample data-version counter and the caches keyed on
it (estimation design caches + the prepared-fields cache).

Background: the estimation design caches used to be keyed on ``id(sample._data)``
without holding a reference to the frame. After an in-place mutation freed and
reallocated the underlying DataFrame, CPython id-reuse could make a stale cache
entry *look* current and serve design arrays for the old data — a silent wrong
result. The fix is a globally-monotonic ``_data_version`` stamp that changes on
every rebind of ``_data``/``_design`` (enforced in ``Sample.__setattr__``); every
version-keyed cache compares against it and rebuilds on mismatch.

These tests pin the invariant "the version changes whenever data or design
changes, and a held estimation facade never serves a stale cached result."
"""

import polars as pl
import pytest

import svy


@pytest.fixture
def df():
    return pl.DataFrame(
        {
            "stratum": ["a", "a", "b", "b", "a", "b"],
            "psu": ["1", "1", "2", "2", "3", "3"],
            "w": [1.0, 2.0, 1.5, 0.5, 1.0, 2.0],
            "w2": [2.0, 1.0, 0.5, 1.5, 2.0, 1.0],
            "y": [10.0, 12.0, 20.0, 22.0, 11.0, 21.0],
        }
    )


@pytest.fixture
def sample(df):
    design = svy.Design(stratum="stratum", psu="psu", wgt="w")
    return svy.Sample(df, design)


# ── Basic invariants ────────────────────────────────────────────────────────


def test_version_is_int_and_present(sample):
    assert isinstance(sample._data_version, int)


def test_distinct_samples_have_distinct_versions(df):
    d = svy.Design(stratum="stratum", psu="psu", wgt="w")
    s1 = svy.Sample(df, d)
    s2 = svy.Sample(df, d)
    assert s1._data_version != s2._data_version


# ── Mutation bumps the version ──────────────────────────────────────────────


def test_set_data_bumps_version(sample, df):
    v0 = sample._data_version
    sample.set_data(df)
    assert sample._data_version != v0


def test_update_data_bumps_version(sample, df):
    v0 = sample._data_version
    sample.update_data(df)
    assert sample._data_version != v0


def test_update_design_bumps_version(sample):
    v0 = sample._data_version
    sample.update_design(wgt="w2")
    assert sample._data_version != v0


def test_inplace_wrangling_bumps_version(sample):
    v0 = sample._data_version
    sample.wrangling.filter_records(svy.col("stratum") == "a", inplace=True)
    assert sample._data_version != v0


def test_non_inplace_wrangling_fork_has_distinct_version(sample):
    v0 = sample._data_version
    forked = sample.wrangling.filter_records(svy.col("stratum") == "a", inplace=False)
    assert forked._data_version != v0
    # Original is untouched → its version is unchanged.
    assert sample._data_version == v0


def test_use_weight_fork_has_distinct_version(sample):
    forked = sample.use_weight("w2")
    assert forked._data_version != sample._data_version


# ── The catastrophic path: held facade must not serve a stale cache ─────────


def test_held_facade_rebuilds_after_set_data(sample, df):
    """A held Estimation facade must reflect new data after set_data, not a
    stale cached design/estimate."""
    est = sample.estimation
    mean_before = est.mean("y").estimates[0].est
    assert mean_before == pytest.approx(16.0)

    # Replace data with y scaled 10x → the correct mean is 160.
    sample.set_data(df.with_columns((pl.col("y") * 10).alias("y")))

    mean_after = est.mean("y").estimates[0].est
    assert mean_after == pytest.approx(160.0)
    # The cache that produced it must be stamped with the current version.
    assert est._design_cache["_data_version"] == sample._data_version


def test_held_facade_rebuilds_after_inplace_wrangling(sample):
    """In-place row filtering under a held facade must change the estimate."""
    est = sample.estimation
    full_mean = est.mean("y").estimates[0].est

    # Keep only stratum 'a' rows in place; the mean must change.
    sample.wrangling.filter_records(svy.col("stratum") == "a", inplace=True)
    filtered_mean = est.mean("y").estimates[0].est

    assert filtered_mean != pytest.approx(full_mean)
    assert est._design_cache["_data_version"] == sample._data_version


def test_polars_design_cache_also_rebuilds(sample, df):
    """The second design cache (_get_polars_design_info) is version-keyed too."""
    est = sample.estimation
    info1 = est._get_polars_design_info()
    assert info1["_data_version"] == sample._data_version

    sample.set_data(df)
    info2 = est._get_polars_design_info()
    assert info2["_data_version"] == sample._data_version
    assert info2["_data_version"] != info1["_data_version"]


# ── Phase C: the cached integer design-code columns ─────────────────────────


def test_design_codes_cache_builds_and_reuses(sample):
    """The Phase C design-code columns are cached per data version and reused."""
    from svy.core.data_prep import _get_design_codes

    codes1 = _get_design_codes(sample, sample._design)
    assert codes1 is not None and "stratum" in codes1 and "psu" in codes1
    assert codes1["stratum"].dtype == pl.UInt32
    # Second call at the same version returns the identical cached object.
    codes2 = _get_design_codes(sample, sample._design)
    assert codes2 is codes1
    assert sample._design_codes_cache[0] == sample._data_version


def test_design_codes_rebuild_after_mutation(sample, df):
    """Codes are rebuilt (not served stale) after the data mutates."""
    from svy.core.data_prep import _get_design_codes

    codes_before = _get_design_codes(sample, sample._design)
    ver_before = sample._data_version

    # Mutate the design source values in place → version bumps.
    sample.set_data(df.with_columns((pl.col("stratum") + "_x").alias("stratum")))
    assert sample._data_version != ver_before

    codes_after = _get_design_codes(sample, sample._design)
    assert codes_after is not codes_before
    assert sample._design_codes_cache[0] == sample._data_version


def test_domain_estimate_uses_codes(sample):
    """A held facade with a domain filter still produces a correct estimate via
    the integer code path (codes survive the where-mask zeroing)."""
    est = sample.estimation
    full = est.mean("y").estimates[0].est
    dom = est.mean("y", where=svy.col("stratum") == "a").estimates[0].est
    assert dom != pytest.approx(full)
    # stratum 'a' rows: y = 10, 12, 11 with w = 1, 2, 1 → (10 + 24 + 11) / 4.
    assert dom == pytest.approx(11.25)
