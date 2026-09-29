# tests/svy/estimation/test_small_int_dtypes.py
"""
Small integer and Float32 columns reach the Rust kernel (regression).

svy-rs built polars without the Int8/Int16/UInt8/UInt16 dtypes, so the kernel
could not import a frame holding such a column: `cannot create series from
Int8 while processing 'data'`. The response is cast to Float64 before the
call, but where/by/group columns and the stratum/psu/pop_size columns are
passed as they are, so a Stata byte (read as Int8) in any of those roles
failed every estimator.

Every case compares against the same data held as Int64 (Float32: the same
values held as Float64), which must give the identical table.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from polars.testing import assert_frame_equal

from svy import Design, Sample


SMALL_DTYPES = [pl.Int8, pl.Int16, pl.UInt8, pl.UInt16, pl.Float32]
METHODS = ["taylor", "replication"]


@pytest.fixture(scope="module")
def base() -> pl.DataFrame:
    rng = np.random.default_rng(20260929)
    n_psu, per_psu = 12, 8
    n = n_psu * per_psu
    psu = np.repeat(np.arange(1, n_psu + 1), per_psu)
    return pl.DataFrame(
        {
            "stratum": np.where(psu <= n_psu // 2, 1, 2),
            "psu": psu,
            "N": np.where(psu <= n_psu // 2, 40, 60),
            "w": rng.uniform(1.0, 4.0, n),
            "y": rng.integers(0, 2, n),
            "x": rng.integers(1, 5, n),
            "z": rng.integers(0, 100, n),
            "grp": rng.integers(1, 4, n),
            "flag": rng.integers(0, 2, n),
        }
    )


@pytest.fixture(scope="module")
def rep_base(base) -> tuple[pl.DataFrame, Design]:
    s = Sample(base, Design(stratum="stratum", psu="psu", wgt="w")).weighting.create_jk_wgts()
    return s.data, s.design


ROLE_COLS = {
    "response": ["y", "x", "z"],
    "where": ["flag"],
    "by": ["grp"],
    "design": ["stratum", "psu"],
    "pop_size": ["N"],
}

WHERE = pl.col("flag") == 1
ROLE_DOMAINS = {
    "response": [{}, {"by": "grp"}, {"where": WHERE}, {"by": "grp", "where": WHERE}],
    "where": [{"where": WHERE}, {"by": "grp", "where": WHERE}],
    "by": [{"by": "grp"}, {"by": "grp", "where": WHERE}],
    "design": [{}, {"by": "grp", "where": WHERE}],
    "pop_size": [{}, {"by": "grp", "where": WHERE}],
}

ESTIMATORS = {
    "mean": lambda s, **k: s.estimation.mean("z", **k),
    "total": lambda s, **k: s.estimation.total("y", **k),
    "prop": lambda s, **k: s.estimation.prop("y", **k),
    "ratio": lambda s, **k: s.estimation.ratio("z", "x", **k),
    "quantile": lambda s, **k: s.estimation.quantile("z", p=(0.25, 0.5), **k),
    "median": lambda s, **k: s.estimation.median("z", **k),
    "mean_multi": lambda s, **k: s.estimation.mean(["y", "z"], **k),
    "total_multi": lambda s, **k: s.estimation.total(["y", "z"], **k),
}


def _cast(df: pl.DataFrame, cols: list[str], dtype) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(data with `cols` as `dtype`, the same values in the reference dtype)."""
    ref_dtype = pl.Float64 if dtype == pl.Float32 else pl.Int64
    small = df.with_columns(pl.col(cols).cast(dtype))
    return small, small.with_columns(pl.col(cols).cast(ref_dtype))


def _samples(base, rep_base, method, cols, dtype) -> tuple[Sample, Sample]:
    if method == "taylor":
        pop = "N" in cols
        design = Design(stratum="stratum", psu="psu", wgt="w", pop_size="N" if pop else None)
        small, ref = _cast(base, cols, dtype)
    else:
        data, design = rep_base
        small, ref = _cast(data, cols, dtype)
    return Sample(small, design), Sample(ref, design)


def _table(result) -> pl.DataFrame:
    if isinstance(result, list):
        return pl.concat([r.to_polars() for r in result], how="diagonal_relaxed")
    return result.to_polars()


def _assert_same(got, expected) -> None:
    assert_frame_equal(_table(got), _table(expected), check_dtypes=False)


# pop_size is a Taylor design field.
ROLE_METHODS = [
    (r, m) for r in ROLE_COLS for m in METHODS if (r, m) != ("pop_size", "replication")
]


@pytest.mark.parametrize("dtype", SMALL_DTYPES, ids=str)
@pytest.mark.parametrize("estimator", list(ESTIMATORS))
@pytest.mark.parametrize(("role", "method"), ROLE_METHODS)
def test_small_dtype_column_matches_reference(base, rep_base, role, method, estimator, dtype):
    small, ref = _samples(base, rep_base, method, ROLE_COLS[role], dtype)
    fn = ESTIMATORS[estimator]
    for kwargs in ROLE_DOMAINS[role]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _assert_same(
                fn(small, method=method, **kwargs),
                fn(ref, method=method, **kwargs),
            )


@pytest.mark.parametrize("dtype", SMALL_DTYPES, ids=str)
@pytest.mark.parametrize("role", ["group", "where", "by", "response", "design"])
def test_ttest_small_dtype_column_matches_reference(base, role, dtype):
    cols = {
        "group": ["flag"],
        "where": ["flag"],
        "by": ["grp"],
        "response": ["z"],
        "design": ["stratum", "psu"],
    }[role]
    call = {
        "group": lambda s: s.categorical.ttest("z", group="flag"),
        "where": lambda s: s.categorical.ttest("z", mean_h0=50, where=WHERE),
        "by": lambda s: s.categorical.ttest("z", mean_h0=50, by="grp"),
        "response": lambda s: s.categorical.ttest("z", group="flag", where=pl.col("grp") > 1),
        "design": lambda s: s.categorical.ttest("z", group="flag"),
    }[role]
    design = Design(stratum="stratum", psu="psu", wgt="w")
    small, ref = _cast(base, cols, dtype)
    _assert_same(call(Sample(small, design)), call(Sample(ref, design)))


@pytest.mark.parametrize("dtype", [pl.Int8, pl.UInt8], ids=str)
@pytest.mark.parametrize("method", METHODS)
def test_all_integer_columns_small(base, rep_base, method, dtype):
    """Every integer column a byte, as a Stata file read with svy-io gives."""
    cols = ["stratum", "psu", "y", "x", "z", "grp", "flag"]
    small, ref = _samples(base, rep_base, method, cols, dtype)
    for fn in ESTIMATORS.values():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _assert_same(
                fn(small, method=method, by="grp", where=WHERE),
                fn(ref, method=method, by="grp", where=WHERE),
            )


def test_int8_indicator_total_by_where(base):
    """The reported case: an Int8 indicator built from two columns, totalled
    with other variables by a domain within a subpopulation."""
    data = base.with_columns(
        ((pl.col("y") == 1) | (pl.col("flag") == 1)).cast(pl.Int8).alias("any_int8"),
        (pl.col("x") >= 3).cast(pl.Int8).alias("sub_int8"),
    )
    design = Design(stratum="stratum", psu="psu", wgt="w")
    small = Sample(data, design)
    ref = Sample(data.with_columns(pl.col("any_int8", "sub_int8").cast(pl.Int64)), design)

    def call(s: Sample):
        return s.estimation.total(["any_int8", "z"], by="grp", where=pl.col("sub_int8") == 1)

    _assert_same(call(small), call(ref))
