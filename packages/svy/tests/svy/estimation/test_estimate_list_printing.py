# tests/svy/estimation/test_estimate_list_printing.py
"""
How an EstimateList stacks its members into one table.

Each proportion names its level column after its own variable, which reads
well for a single result but collides when several are stacked: a diagonal
concat unions them into one sparse column per variable. Eight conditions
produced eight mostly-empty columns, a ~118-character box, and numbers
truncated to ellipses. Stacking has to put them under one shared name.
"""

import re

from pathlib import Path

import polars as pl
import pytest

from svy.core.sample import Design, Sample
from svy.estimation import EstimateList


DATA_DIR = Path(__file__).resolve().parents[2] / "test_data"


def visible(text: str) -> list[str]:
    """Lines with the ANSI styling removed, so widths are what a reader sees."""
    return [re.sub(r"\x1b\[[0-9;]*m", "", line) for line in text.splitlines()]


@pytest.fixture
def sample():
    df = pl.read_csv(DATA_DIR / "apistrat.csv").with_columns(
        (pl.col("api00") > 600).alias("hi_api"),
        (pl.col("ell") > 20).alias("hi_ell"),
        (pl.col("meals") > 50).alias("hi_meals"),
    )
    return Sample(df, Design(wgt="pw", stratum="stype"))


BINARY = ["hi_api", "hi_ell", "hi_meals"]


class TestPropListSharesOneLevelColumn:
    def test_one_level_column_not_one_per_variable(self, sample):
        est = sample.estimation.prop(BINARY, drop_nulls=True)
        assert isinstance(est, EstimateList)
        cols = est._combined().columns

        assert "level" in cols
        for var in BINARY:
            assert var not in cols, f"{var} still has its own level column"

    def test_every_row_is_populated(self, sample):
        """The staircase left each row blank in all but one level column."""
        df = sample.estimation.prop(BINARY, drop_nulls=True)._combined()

        assert df.height == 6  # 3 variables x false/true
        assert df["level"].null_count() == 0
        assert df["y"].null_count() == 0
        assert sorted(set(df["y"])) == sorted(BINARY)

    def test_numbers_are_not_truncated(self, sample):
        lines = visible(str(sample.estimation.prop(BINARY, drop_nulls=True)))

        assert not any("…" in line for line in lines)
        # The staircase pushed this past 110; three variables need nowhere near it.
        assert max(len(line) for line in lines) < 80

    def test_plain_text_path_too(self, sample):
        text = sample.estimation.prop(BINARY, drop_nulls=True).__plain_str__()

        assert "…" not in text
        assert "level" in text


class TestNeighbouringCasesUnchanged:
    """The level column is shared only where sharing is what was missing."""

    def test_mean_over_a_list_is_untouched(self, sample):
        cols = sample.estimation.mean(["api00", "ell"])._combined().columns

        assert cols[:2] == ["y", "est"]
        assert "level" not in cols

    def test_single_prop_keeps_the_variable_named_column(self, sample):
        """Alone, the variable name is the better header — and it is in no danger."""
        cols = sample.estimation.prop("hi_api", drop_nulls=True).to_polars_printable().columns

        assert "hi_api" in cols
        assert "level" not in cols

    def test_one_element_list_keeps_the_variable_name(self, sample):
        cols = sample.estimation.prop(["hi_api"], drop_nulls=True)._combined().columns

        assert "hi_api" in cols
        assert "level" not in cols

    def test_quantile_list_keeps_prob(self, sample):
        cols = sample.estimation.quantile("api00", p=(0.25, 0.75))._combined().columns

        assert cols[0] == "prob"
        assert "level" not in cols

    def test_by_column_stays_distinct_from_the_level_column(self, sample):
        cols = sample.estimation.prop(BINARY, by="stype", drop_nulls=True)._combined().columns

        assert "stype" in cols and "level" in cols
        assert cols.index("stype") < cols.index("level")


class TestLevelNameCollision:
    def test_a_variable_actually_named_level_does_not_collide(self):
        """`level` is taken here, so the shared column has to pick another name."""
        df = pl.read_csv(DATA_DIR / "apistrat.csv").with_columns(
            (pl.col("api00") > 600).alias("hi_api"),
            pl.col("stype").alias("level"),
        )
        sample = Sample(df, Design(wgt="pw"))

        cols = (
            sample.estimation.prop(["hi_api", "awards"], by="level", drop_nulls=True)
            ._combined()
            .columns
        )

        assert "level" in cols  # the by-variable keeps its own name
        assert "y_level" in cols  # the shared level column moved aside
        assert len(cols) == len(set(cols))
