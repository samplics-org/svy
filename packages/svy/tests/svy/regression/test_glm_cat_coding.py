# tests/svy/regression/test_glm_cat_coding.py
"""
A Cat predictor must give the same predictions and margins whatever dtype it
is coded in.

Prediction used to slice the level back out of the dummy's *name* as a string
and compare it to the raw column, so an int/float/bool-coded Cat matched
nothing: `predict()` silently returned the reference-level prediction for
every row and every categorical AME came back exactly 0.0 with SE 0.0. The
fitted coefficients were right the whole time, which is what made it quiet.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from svy.core.sample import Design, Sample
from svy.core.terms import Cat, Cross
from svy.errors.model_errors import ModelError


DATA_DIR = Path(__file__).resolve().parents[2] / "test_data"

# Three-level codings of the same partition, in the same level order.
CODINGS = {
    "str": {"E": "0", "H": "1", "M": "2"},
    "int": {"E": 0, "H": 1, "M": 2},
    "float": {"E": 0.0, "H": 1.0, "M": 2.0},
}

# Two-level codings, for the Boolean case.
CODINGS2 = {
    "str": {False: "F", True: "T"},
    "bool": {False: False, True: True},
    "int": {False: 0, True: 1},
}


@pytest.fixture
def api():
    return pl.read_csv(DATA_DIR / "apistrat.csv").with_columns(
        (pl.col("api00") > 600).cast(pl.Int32).alias("y")
    )


def _coded(api: pl.DataFrame, mapping: dict, source: pl.Expr) -> pl.DataFrame:
    return api.with_columns(source.replace_strict(mapping).alias("g"))


def _fit(df: pl.DataFrame, **kw):
    return Sample(df, Design(wgt="pw")).glm.fit(
        y="y", x=["ell", Cat("g")], family="binomial", tol=1e-12, **kw
    )


@pytest.fixture(params=sorted(CODINGS))
def coded_pair(request, api):
    """(reference string-coded fit+data, fit+data under the coding under test)."""
    src = pl.col("stype")
    ref_df = _coded(api, CODINGS["str"], src)
    alt_df = _coded(api, CODINGS[request.param], src)
    return (_fit(ref_df), ref_df), (_fit(alt_df), alt_df), request.param


class TestCodingIndependence:
    def test_coefficients_are_identical(self, coded_pair):
        (ref, _), (alt, _), _ = coded_pair
        np.testing.assert_allclose(
            [c.est for c in alt.fitted.coefs],
            [c.est for c in ref.fitted.coefs],
            rtol=0,
            atol=1e-12,
        )

    def test_predictions_are_identical(self, coded_pair):
        (ref, ref_df), (alt, alt_df), _ = coded_pair
        p_ref = ref.predict(ref_df)
        p_alt = alt.predict(alt_df)

        np.testing.assert_allclose(p_alt.yhat, p_ref.yhat, rtol=0, atol=1e-12)
        np.testing.assert_allclose(p_alt.se, p_ref.se, rtol=0, atol=1e-12)

    def test_predictions_actually_vary_by_level(self, coded_pair):
        """The old path returned the reference prediction for every row."""
        (_, _), (alt, alt_df), _ = coded_pair
        yhat = alt.predict(alt_df).yhat

        assert len(np.unique(np.round(yhat, 12))) > 1

    def test_categorical_ame_is_identical(self, coded_pair):
        (ref, _), (alt, _), _ = coded_pair
        m_ref = next(m for m in ref.margins() if m.term == "g")
        m_alt = next(m for m in alt.margins() if m.term == "g")

        np.testing.assert_allclose(m_alt.margin, m_ref.margin, rtol=0, atol=1e-12)
        np.testing.assert_allclose(m_alt.se, m_ref.se, rtol=0, atol=1e-12)

    def test_categorical_ame_is_not_degenerate(self, coded_pair):
        """Every non-string coding used to give AME 0.0 with SE 0.0."""
        (_, _), (alt, _), _ = coded_pair
        m = next(mm for mm in alt.margins() if mm.term == "g")

        assert np.all(np.abs(m.margin) > 1e-6)
        assert np.all(m.se > 1e-6)

    def test_continuous_ame_is_identical(self, coded_pair):
        (ref, _), (alt, _), _ = coded_pair
        m_ref = next(m for m in ref.margins() if m.term == "ell")
        m_alt = next(m for m in alt.margins() if m.term == "ell")

        np.testing.assert_allclose(m_alt.margin, m_ref.margin, rtol=0, atol=1e-12)
        np.testing.assert_allclose(m_alt.se, m_ref.se, rtol=0, atol=1e-12)

    def test_predictive_margins_at_a_level_are_identical(self, coded_pair):
        (ref, _), (alt, _), coding = coded_pair
        m_ref = ref.margins(at={"g": list(CODINGS["str"].values())})
        m_alt = alt.margins(at={"g": list(CODINGS[coding].values())})

        np.testing.assert_allclose(m_alt.margin, m_ref.margin, rtol=0, atol=1e-12)
        np.testing.assert_allclose(m_alt.se, m_ref.se, rtol=0, atol=1e-12)


class TestBooleanCoding:
    """A Boolean Cat column, against the same two-level partition as strings."""

    @pytest.fixture(params=sorted(CODINGS2))
    def pair(self, request, api):
        src = pl.col("stype") == "E"
        ref_df = _coded(api, CODINGS2["str"], src)
        alt_df = _coded(api, CODINGS2[request.param], src)
        return (_fit(ref_df), ref_df), (_fit(alt_df), alt_df)

    def test_predictions_and_margins_match_the_string_coding(self, pair):
        (ref, ref_df), (alt, alt_df) = pair

        np.testing.assert_allclose(
            alt.predict(alt_df).yhat, ref.predict(ref_df).yhat, rtol=0, atol=1e-12
        )
        m_ref = next(m for m in ref.margins() if m.term == "g")
        m_alt = next(m for m in alt.margins() if m.term == "g")
        np.testing.assert_allclose(m_alt.margin, m_ref.margin, rtol=0, atol=1e-12)
        np.testing.assert_allclose(m_alt.se, m_ref.se, rtol=0, atol=1e-12)


class TestInteractionCoding:
    """The same, through an interaction, where the dummy is one factor."""

    @pytest.fixture(params=sorted(CODINGS))
    def pair(self, request, api):
        src = pl.col("stype")
        ref_df = _coded(api, CODINGS["str"], src)
        alt_df = _coded(api, CODINGS[request.param], src)

        def fit(df):
            return Sample(df, Design(wgt="pw")).glm.fit(
                y="y",
                x=[Cat("g"), "ell", Cross(Cat("g"), "ell")],
                family="binomial",
                tol=1e-12,
            )

        return (fit(ref_df), ref_df), (fit(alt_df), alt_df)

    def test_predictions_match(self, pair):
        (ref, ref_df), (alt, alt_df) = pair
        np.testing.assert_allclose(
            alt.predict(alt_df).yhat, ref.predict(ref_df).yhat, rtol=0, atol=1e-12
        )

    def test_continuous_ame_through_the_interaction_matches(self, pair):
        (ref, _), (alt, _) = pair
        m_ref = next(m for m in ref.margins() if m.term == "ell")
        m_alt = next(m for m in alt.margins() if m.term == "ell")

        np.testing.assert_allclose(m_alt.margin, m_ref.margin, rtol=0, atol=1e-12)
        np.testing.assert_allclose(m_alt.se, m_ref.se, rtol=0, atol=1e-12)


class TestPredictionDataValidation:
    """Levels the fit never saw are refused, not coded as the reference."""

    @pytest.fixture
    def model(self, api):
        return _fit(_coded(api, CODINGS["int"], pl.col("stype")))

    def test_unseen_level_raises(self, model, api):
        new_data = _coded(api.head(5), CODINGS["int"], pl.col("stype")).with_columns(
            pl.lit(9, dtype=pl.Int64).alias("g")
        )

        with pytest.raises(ModelError, match="never saw"):
            model.predict(new_data)

    def test_null_level_raises(self, model, api):
        new_data = _coded(api.head(5), CODINGS["int"], pl.col("stype")).with_columns(
            pl.lit(None, dtype=pl.Int64).alias("g")
        )

        with pytest.raises(ModelError, match="never saw"):
            model.predict(new_data)

    def test_missing_categorical_column_names_it(self, model, api):
        new_data = _coded(api.head(5), CODINGS["int"], pl.col("stype")).drop("g")

        with pytest.raises(ModelError, match="'g'"):
            model.predict(new_data)

    def test_missing_continuous_column_names_it(self, model, api):
        new_data = _coded(api.head(5), CODINGS["int"], pl.col("stype")).drop("ell")

        with pytest.raises(ModelError, match="'ell'"):
            model.predict(new_data)

    def test_margins_at_an_unseen_level_raises(self, model):
        with pytest.raises(ModelError, match="never saw"):
            model.margins(at={"g": [9]})

    def test_wrong_dtype_is_reported_as_unknown_levels(self, model, api):
        """String '0'/'1'/'2' against an int-coded fit is a level error."""
        new_data = _coded(api.head(5), CODINGS["str"], pl.col("stype"))

        with pytest.raises(ModelError, match="never saw"):
            model.predict(new_data)
