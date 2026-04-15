from pathlib import Path

import polars as pl
import pytest

from svy.metadata import Label, LabellingCatalog


@pytest.fixture
def synthetic_sample_df():
    """Load and prepare the synthetic sample dataset."""
    base_dir = Path(__file__).parent.parent.parent
    df = pl.read_csv(base_dir / "test_data/svy_synthetic_sample_07082025.csv")

    return df.with_columns(
        pl.when(pl.col("resp2") == 1)
        .then(1)
        .when(pl.col("resp2") == 2)
        .then(0)
        .otherwise(None)
        .alias("resp2_new")
    )


def categories_from_polars(df: pl.DataFrame, col: str, *, sort: bool = True) -> dict:
    """
    Build a code->text mapping from unique values in a column (good for strings like 'educ'
    or when you just want code-as-text as a placeholder).
    """
    s = df[col]
    u = s.unique().drop_nulls()
    if sort:
        try:
            u = u.sort()
        except Exception:
            pass
    return {v: str(v) for v in u.to_list()}


def test_create_labels_from_scratch(synthetic_sample_df: pl.DataFrame):
    df = synthetic_sample_df

    # 1) Create a catalog with reusable schemes (chainable API, locale inferred)
    catalog = (
        LabellingCatalog(locale="en")
        .add_scheme(
            concept="yes_no01",
            mapping={1: "Yes", 0: "No"},
            title="Yes/No (1/0)",
        )
        .add_scheme(
            concept="yes_no02",
            mapping={1: "Yes", 2: "No"},
            title="Yes/No (1/2)",
        )
        .add_scheme(
            concept="likert5",
            mapping={
                1: "Strongly disagree",
                2: "Disagree",
                3: "Neutral",
                4: "Agree",
                5: "Strongly agree",
            },
            title="Agreement (5-point)",
            ordered=True,
        )
        .add_scheme(
            concept="likert3",
            mapping={1: "Positive", 2: "Neutral", 3: "Negative"},
            title="Sentiment (3-point)",
            ordered=True,
        )
        .add_scheme(
            concept="sex",
            mapping={1: "Male", 2: "Female"},
            title="Sex",
        )
    )

    # 2) Build per-variable labels (mix of reusable schemes + auto from data)
    labels: dict[str, Label] = {
        "resp2_new": catalog.to_label_by_concept("Consent (derived)", concept="yes_no01"),
        "resp2": catalog.to_label_by_concept("Consent (Q2)", concept="yes_no02"),
        "resp3": catalog.to_label_by_concept("Satisfaction (Q3)", concept="likert5"),
        "resp5": catalog.to_label_by_concept("Trust (Q5)", concept="likert5"),
        "sex": catalog.to_label_by_concept("Respondent sex", concept="sex"),
        # Auto-build from the dataset when you don’t have a predefined scheme
        "educ": Label(label="Education", categories=categories_from_polars(df, "educ")),
        "region": Label(label="Region", categories=categories_from_polars(df, "region")),
    }

    # 3) Quick sanity checks
    assert labels["resp2"].categories[1] == "Yes"
    assert labels["resp2_new"].categories[0] == "No"
    assert "High School" in labels["educ"].categories.values()
