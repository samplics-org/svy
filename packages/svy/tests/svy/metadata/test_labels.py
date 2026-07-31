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
    assert labels["resp2"].label_map[1] == "Yes"
    assert labels["resp2_new"].label_map[0] == "No"
    assert "High School" in labels["educ"].label_map.values()


# ---------------------------------------------------------------------------
# SchemeEntry — one entry per code, rather than parallel collections
#
# The scheme used to hold mapping, missing, missing_kinds and a hierarchy as
# four collections keyed by code. Three serialized badly, and because they could
# disagree, seventy-odd lines existed to check that they did not. One entry per
# code makes those states unrepresentable.
# ---------------------------------------------------------------------------


def _scheme(**overrides):
    from svy.core.enumerations import MissingKind
    from svy.metadata.labels import CategoryScheme, SchemeEntry

    base = dict(
        concept="gm_district",
        locale="en",
        entries=[
            SchemeEntry(code=101, label="Banjul", parent=1),
            SchemeEntry(code=102, label="Kanifing", parent=1),
            SchemeEntry(code=201, label="Lower Saloum", parent=2),
            SchemeEntry(code=99, label="Refused", missing=MissingKind.REFUSED),
        ],
    )
    return CategoryScheme(**{**base, **overrides})


def test_a_dict_is_still_accepted_when_constructing():
    from svy.metadata.labels import CategoryScheme

    scheme = CategoryScheme(concept="sex", entries={1: "Male", 2: "Female"})
    assert scheme.labels == {1: "Male", 2: "Female"}
    assert all(e.missing is None for e in scheme.entries)


def test_labels_covers_every_code_and_substantive_does_not():
    scheme = _scheme()
    assert set(scheme.labels) == {101, 102, 201, 99}
    assert [e.code for e in scheme.substantive] == [101, 102, 201]


def test_missing_semantics_live_on_the_entry():
    from svy.core.enumerations import MissingKind

    scheme = _scheme()
    assert scheme.missing_codes == frozenset({99})
    assert scheme.kind_of(99) is MissingKind.REFUSED
    assert scheme.kind_of(101) is None
    assert scheme.codes_of_kind(MissingKind.REFUSED) == frozenset({99})


def test_hierarchy_lookups():
    scheme = _scheme()
    assert scheme.parent_of(201) == 2
    assert scheme.children_of(1) == (101, 102)
    assert scheme.parent_of(99) is None


def test_children_keep_declaration_order():
    from svy.metadata.labels import SchemeEntry

    scheme = _scheme(
        entries=[
            SchemeEntry(code=102, label="Kanifing", parent=1),
            SchemeEntry(code=101, label="Banjul", parent=1),
        ]
    )
    assert scheme.children_of(1) == (102, 101)


def test_a_code_that_is_not_there_answers_none():
    scheme = _scheme()
    assert scheme.entry(999) is None
    assert scheme.parent_of(999) is None
    assert scheme.kind_of(999) is None


def test_a_scheme_round_trips_through_plain_msgspec():
    """The whole reason for the shape.

    Every field is JSON-native now, so no bespoke encoder is involved and codes
    keep their types by any route. The old shape returned {"101": "Banjul"} for
    {101: "Banjul"} unless it went through the catalog's hand-written pairs.
    """
    import msgspec

    from svy.metadata.labels import CategoryScheme

    scheme = _scheme()
    back = msgspec.json.decode(msgspec.json.encode(scheme), type=CategoryScheme)
    assert back == scheme
    assert all(isinstance(c, int) for c in back.codes)


def test_a_scheme_round_trips_through_the_catalog():
    scheme = _scheme()
    catalog = LabellingCatalog().register(scheme)
    back = LabellingCatalog.from_bytes(catalog.to_bytes()).pick("gm_district", locale="en")
    assert back.entries == scheme.entries


def test_a_duplicate_code_is_rejected():
    from svy.errors import LabelError
    from svy.metadata.labels import SchemeEntry

    with pytest.raises(LabelError):
        _scheme(
            entries=[
                SchemeEntry(code=101, label="Banjul"),
                SchemeEntry(code=101, label="Banjul again"),
            ]
        )


def test_a_missing_code_outside_the_scheme_is_unrepresentable():
    """What validate_scheme_missing used to check, now impossible to express.

    A missing code had to be added to `missing` *and* to `mapping`, and the two
    could disagree. There is one list now, so a code that is missing is by
    construction a code that exists.
    """
    scheme = _scheme()
    assert scheme.missing_codes <= set(scheme.codes)
