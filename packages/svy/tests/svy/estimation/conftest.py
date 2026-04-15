# tests/svy/estimation/conftest.py
from pathlib import Path

import polars as pl
import pytest

from svy import Design, RepWeights


@pytest.fixture
def load_survey_data():
    """
    Factory fixture to load specific survey data files.
    Returns a function: loader(filename) -> pl.DataFrame
    """

    def _loader(filename):
        # Assumes directory structure: tests/svy/estimation/conftest.py
        # Data is in: tests/data/
        base_dir = Path(__file__).parents[2]
        data_path = base_dir / "test_data" / filename

        if not data_path.exists():
            pytest.skip(f"Data file not found: {filename}")

        df = pl.read_csv(data_path)

        # Standard transformations used in all replication tests
        return df.with_columns(
            pl.when(pl.col("income").is_null() | pl.col("income").is_nan())
            .then(pl.lit(None))
            .when(pl.col("income") < 40_000)
            .then(1)
            .otherwise(0)
            .alias("low_income"),
            # Deterministic denominator (household size 1 to 4)
            ((pl.col("id") % 4) + 1).alias("hh_size"),
        )

    return _loader


@pytest.fixture
def make_design():
    """
    Factory fixture to create designs dynamically.
    Returns a function: maker(...) -> Design
    """

    def _maker(method, rep_prefix, n_reps, df=None, psu_col="psu"):
        rep_weights = RepWeights(method=method, prefix=rep_prefix, n_reps=n_reps, df=df)
        return Design(
            row_index="id", wgt="weight", stratum="stratum", psu=psu_col, rep_wgts=rep_weights
        )

    return _maker
