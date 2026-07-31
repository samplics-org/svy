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


# ---------------------------------------------------------------------------
# CategoryScheme.parents — hierarchical schemes
#
# A cascading list (region -> district -> ward -> EA) referenced by concept
# rather than inlined into a survey specification, because a geography is
# revised on its own cycle and a census redraw should not read as a change to
# the questionnaire.
# ---------------------------------------------------------------------------


GM_DISTRICT = dict(
    concept="gm_district",
    locale="en",
    mapping={101: "Banjul", 102: "Kanifing", 201: "Lower Saloum", 202: "Nianija"},
    parents=((101, 1), (102, 1), (201, 2), (202, 2)),
)


def _scheme(**overrides):
    from svy.metadata import CategoryScheme

    return CategoryScheme(**{**GM_DISTRICT, **overrides})


def test_parent_and_children_lookups():
    scheme = _scheme()
    assert scheme.parent_of(201) == 2
    assert scheme.children_of(1) == (101, 102)


def test_a_code_with_no_parent_returns_none():
    assert _scheme().parent_of(999) is None


def test_a_scheme_without_a_hierarchy_answers_emptily():
    assert _scheme(parents=None).parent_of(101) is None
    assert _scheme(parents=None).children_of(1) == ()


def test_children_keep_declaration_order():
    scheme = _scheme(parents=((102, 1), (101, 1)))
    assert scheme.children_of(1) == (102, 101)


def test_parents_survive_the_catalog_round_trip():
    catalog = LabellingCatalog().register(_scheme())
    back = LabellingCatalog.from_bytes(catalog.to_bytes()).pick("gm_district", locale="en")
    assert back.parents == GM_DISTRICT["parents"]
    assert all(isinstance(child, int) for child, _ in back.parents)


def test_parents_survive_plain_msgspec_too():
    """The reason this field is pairs and not a dict.

    `mapping` is a Category-keyed dict and only survives through the catalog's
    custom encoder — plain msgspec turns {101: "Banjul"} into {"101": "Banjul"}
    silently. Pairs are correct by any route, so a scheme serialized by some
    future path that forgets the encoder still holds its codes.
    """
    import msgspec

    from svy.metadata import CategoryScheme

    back = msgspec.json.decode(msgspec.json.encode(_scheme()), type=CategoryScheme)
    assert back.parents == GM_DISTRICT["parents"]
    assert all(isinstance(child, int) for child, _ in back.parents)


def test_a_child_code_absent_from_the_mapping_is_rejected():
    from svy.errors import LabelError
    from svy.metadata.labels import validate_scheme_missing

    with pytest.raises(LabelError):
        validate_scheme_missing(_scheme(parents=((999, 1),)), strict=True)


def test_a_parent_code_is_not_validated_against_this_mapping():
    # parents belong to a *different* scheme — the region list — which this one
    # cannot see, so validating them here would reject every real hierarchy
    from svy.metadata.labels import validate_scheme_missing

    validate_scheme_missing(_scheme(), strict=True)


def test_a_hierarchy_coexists_with_missing_semantics():
    from svy.core.enumerations import MissingKind

    scheme = _scheme(
        mapping={**GM_DISTRICT["mapping"], 99: "Refused"},
        missing={99},
        missing_kinds={99: MissingKind.REFUSED},
    )
    catalog = LabellingCatalog().register(scheme)
    back = LabellingCatalog.from_bytes(catalog.to_bytes()).pick("gm_district", locale="en")
    assert back.parents == GM_DISTRICT["parents"]
    assert back.missing == {99}
    assert back.missing_kinds == {99: MissingKind.REFUSED}
