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
        LabellingCatalog()
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
        )
        .add_scheme(
            concept="likert3",
            mapping={1: "Positive", 2: "Neutral", 3: "Negative"},
            title="Sentiment (3-point)",
        )
        .add_scheme(
            concept="sex",
            mapping={1: "Male", 2: "Female"},
            title="Sex",
        )
    )

    # 2) Build per-variable labels (mix of reusable schemes + auto from data)
    labels: dict[str, Label] = {
        "resp2_new": catalog.to_label("Consent (derived)", "yes_no01"),
        "resp2": catalog.to_label("Consent (Q2)", "yes_no02"),
        "resp3": catalog.to_label("Satisfaction (Q3)", "likert5"),
        "resp5": catalog.to_label("Trust (Q5)", "likert5"),
        "sex": catalog.to_label("Respondent sex", "sex"),
        # Auto-build from the dataset when you don’t have a predefined scheme
        "educ": Label(label="Education", categories=categories_from_polars(df, "educ")),
        "region": Label(label="Region", categories=categories_from_polars(df, "region")),
    }

    # 3) Quick sanity checks
    assert labels["resp2"].label_map[1] == "Yes"
    assert labels["resp2_new"].label_map[0] == "No"
    assert "High School" in labels["educ"].label_map.values()


# ---------------------------------------------------------------------------
# A scheme is a code→label map
#
# svy labels values so results print nicely, and that is the whole job (design
# 001 §2.2). Everything a scheme used to carry beyond code and label — a
# hierarchy, why a code is a non-answer, a locale, an ordering flag — has moved
# to svy-spec, which is the only place anything could act on it.
# ---------------------------------------------------------------------------


def _scheme(**overrides):
    from svy.metadata.labels import CategoryScheme, SchemeEntry

    base = dict(
        concept="gm_district",
        entries=[
            SchemeEntry(code=101, label="Banjul"),
            SchemeEntry(code=102, label="Kanifing"),
            SchemeEntry(code=201, label="Lower Saloum"),
            SchemeEntry(code=99, label="Refused"),
        ],
    )
    return CategoryScheme(**{**base, **overrides})


def test_a_dict_is_accepted_when_constructing():
    from svy.metadata.labels import CategoryScheme

    scheme = CategoryScheme(concept="sex", entries={1: "Male", 2: "Female"})
    assert scheme.labels == {1: "Male", 2: "Female"}
    assert scheme.codes == (1, 2)


def test_labels_covers_every_code():
    # Including 99. A declared refusal is an ordinary value with an ordinary
    # label; svy prints "Refused" and forms no opinion about it.
    assert _scheme().labels == {
        101: "Banjul",
        102: "Kanifing",
        201: "Lower Saloum",
        99: "Refused",
    }


def test_a_code_that_is_not_there_answers_none():
    assert _scheme().entry(404) is None


def test_a_scheme_carries_nothing_beyond_code_and_label():
    from svy.metadata.labels import SchemeEntry

    entry = SchemeEntry(code=1, label="Yes")
    assert (entry.code, entry.label) == (1, "Yes")
    # A parent belongs to a cascading choice list and a missing kind is a
    # questionnaire fact. Both are svy-spec's; neither is expressible here.
    for gone in ("parent", "missing", "is_missing"):
        assert not hasattr(entry, gone), f"SchemeEntry should not carry {gone!r}"
    for gone in ("locale", "ordered", "id", "parent_of", "children_of", "missing_codes"):
        assert not hasattr(_scheme(), gone), f"CategoryScheme should not carry {gone!r}"


def test_a_scheme_round_trips_through_plain_msgspec():
    import msgspec

    from svy.metadata.labels import CategoryScheme

    scheme = _scheme()
    back = msgspec.json.decode(msgspec.json.encode(scheme), type=CategoryScheme)
    assert back == scheme
    # The codes keep their types. JSON object keys are always strings, which is
    # why entries are a tuple of structs rather than a dict keyed by code.
    assert [type(c) for c in back.codes] == [int, int, int, int]


def test_a_duplicate_code_is_rejected():
    import pytest

    from svy.errors.label_errors import LabelError
    from svy.metadata.labels import CategoryScheme, SchemeEntry

    with pytest.raises(LabelError):
        CategoryScheme(
            concept="x",
            entries=[SchemeEntry(code=1, label="A"), SchemeEntry(code=1, label="B")],
        )


# ---------------------------------------------------------------------------
# The catalogue is keyed by concept
# ---------------------------------------------------------------------------


def test_a_scheme_round_trips_through_the_catalog():
    from svy.metadata.labels import LabellingCatalog

    catalog = LabellingCatalog().register(_scheme())
    back = LabellingCatalog.from_bytes(catalog.to_bytes())
    assert back.pick("gm_district").labels == _scheme().labels


def test_pick_and_get_are_the_same_lookup():
    # One concept, one scheme. There is no id to disambiguate locales, because
    # there are no locales.
    from svy.metadata.labels import LabellingCatalog

    catalog = LabellingCatalog().register(_scheme())
    assert catalog.pick("gm_district") is catalog.get("gm_district")


def test_a_concept_is_normalised_on_the_way_in_and_out():
    from svy.metadata.labels import LabellingCatalog, make_scheme

    catalog = LabellingCatalog().register(make_scheme(concept="Yes No", mapping={1: "Yes"}))
    assert catalog.pick("yes_no").labels == {1: "Yes"}
    assert catalog.pick("Yes No").labels == {1: "Yes"}


def test_registering_one_concept_twice_needs_overwrite():
    import pytest

    from svy.errors.label_errors import LabelError
    from svy.metadata.labels import LabellingCatalog, make_scheme

    catalog = LabellingCatalog().register(make_scheme(concept="sex", mapping={1: "M"}))
    with pytest.raises(LabelError):
        catalog.register(make_scheme(concept="sex", mapping={1: "Male"}))
    catalog.register(make_scheme(concept="sex", mapping={1: "Male"}), overwrite=True)
    assert catalog.pick("sex").labels == {1: "Male"}


def test_two_languages_are_two_concepts():
    # What `locale` used to do, without a matching algorithm: svy holds one set
    # of strings, and choosing which set is svy-spec's job (§2.2).
    from svy.metadata.labels import LabellingCatalog, make_scheme

    catalog = (
        LabellingCatalog()
        .register(make_scheme(concept="sex_en", mapping={1: "Male", 2: "Female"}))
        .register(make_scheme(concept="sex_fr", mapping={1: "Homme", 2: "Femme"}))
    )
    assert catalog.pick("sex_fr").labels[2] == "Femme"
    assert catalog.pick("sex_en").labels[2] == "Female"


def test_a_catalog_labelled_variable_prints_every_code():
    """A 99 labelled "Refusal" is the integer 99 with the label "Refusal".

    This replaces a test that asserted resolve_labels recovered *missing codes*
    from a scheme. Schemes no longer carry missingness, so there is nothing to
    recover — and nothing to lose, which is what the bug it guarded did.
    """
    from svy.metadata import LabellingCatalog, MetadataStore, VariableMeta
    from svy.metadata.variable_meta import SchemeRef

    catalog = LabellingCatalog().register(_scheme())
    store = MetadataStore(catalog=catalog)
    store.set(
        "district", VariableMeta(name="district", scheme_ref=SchemeRef(concept="gm_district"))
    )

    resolved = store.resolve_labels("district")
    assert resolved.labels[99] == "Refused"
    assert resolved.labels[101] == "Banjul"
