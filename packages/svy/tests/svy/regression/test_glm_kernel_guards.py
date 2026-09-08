# tests/svy/regression/test_glm_kernel_guards.py
"""
What the GLM kernel refuses, and what it warns about.

Each case below used to produce a result rather than a complaint: an empty
domain returned a zero fit as a success, `x` and `2 * x` "fitted" with SE 0 and
an F around 1e30, k >= n came back through an SVD pseudoinverse, and a model
that ran out of iterations was reported like a converged one.
"""

import warnings

from pathlib import Path

import polars as pl
import pytest

import svy

from svy.core.sample import Design, Sample
from svy.core.terms import Cat
from svy.errors.model_errors import ModelError


DATA_DIR = Path(__file__).resolve().parents[2] / "test_data"


@pytest.fixture
def api():
    return pl.read_csv(DATA_DIR / "apistrat.csv")


def _sample(df: pl.DataFrame) -> Sample:
    return Sample(df, Design(wgt="pw"))


class TestRefusals:
    def test_collinear_column_is_named(self, api):
        df = api.with_columns((pl.col("ell") * 2).alias("ell2"))

        with pytest.raises(ModelError, match="ell2"):
            _sample(df).glm.fit(y="api00", x=["ell", "ell2"])

    def test_collinear_error_points_at_the_later_column(self, api):
        """R's rule: the columns that came first are the ones that are kept."""
        df = api.with_columns((pl.col("ell") * 2).alias("ell2"))

        with pytest.raises(ModelError) as exc:
            _sample(df).glm.fit(y="api00", x=["ell", "ell2"])

        assert "'ell2'" in str(exc.value)
        assert "'ell'," not in str(exc.value)

    def test_a_duplicated_categorical_is_caught(self, api):
        """`stype` and a copy of it: the second set of dummies is aliased."""
        df = api.with_columns(pl.col("stype").alias("stype2"))

        with pytest.raises(ModelError, match="rank deficient"):
            _sample(df).glm.fit(y="api00", x=[Cat("stype"), Cat("stype2")])

    def test_more_parameters_than_observations(self, api):
        with pytest.raises(ModelError, match="more observations than parameters"):
            _sample(api.head(3)).glm.fit(y="api00", x=["ell", "meals", "mobility"])

    def test_empty_where_domain(self, api):
        with pytest.raises(ModelError, match="No row has a positive weight"):
            _sample(api).glm.fit(y="api00", x=["ell"], where=svy.col("stype") == "ZZ")

    def test_all_weights_zero(self, api):
        df = api.with_columns(pl.lit(0.0).alias("pw"))

        with pytest.raises(ModelError, match="No row has a positive weight"):
            _sample(df).glm.fit(y="api00", x=["ell"])

    def test_repeated_predictor(self, api):
        with pytest.raises(ModelError, match="engineered twice"):
            _sample(api).glm.fit(y="api00", x=["ell", "ell"])

    def test_non_finite_predictor_reaching_the_kernel(self):
        """
        The Python path never gets here — `prepare_data` treats an infinite
        covariate as missing and zero-weights the row — so this drives the
        kernel directly, which is what other callers of `fit_glm_rs` do.
        """
        from svy_rs import _internal as rs

        df = pl.DataFrame(
            {
                "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "_intercept_": [1.0] * 6,
                "x": [1.0, 2.0, float("inf"), 4.0, 5.0, 6.0],
                "w": [1.0] * 6,
            }
        )

        with pytest.raises(RuntimeError, match="non-finite value"):
            rs.fit_glm_rs(
                y_name="y",
                x_names=["_intercept_", "x"],
                weight_name="w",
                data=df,
                family="gaussian",
                link="identity",
            )


class TestWarnings:
    def test_non_convergence_is_reported(self, api):
        with pytest.warns(UserWarning, match="did not converge"):
            _sample(api).glm.fit(y="api00", x=["ell"], family="gamma", link="log", max_iter=1)

    def test_a_converged_fit_is_silent(self, api):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            _sample(api).glm.fit(y="api00", x=["ell", "meals"])

    def test_invalid_weights_are_reported_when_dropped(self, api):
        df = api.with_columns(
            pl.when(pl.int_range(pl.len()) == 0).then(-1.0).otherwise(pl.col("pw")).alias("pw")
        )

        with pytest.warns(UserWarning, match="null, non-finite or negative"):
            _sample(df).glm.fit(y="api00", x=["ell"])

    def test_single_level_categorical_is_reported_when_dropped(self, api):
        df = api.with_columns(pl.lit("A").alias("g"))

        with pytest.warns(UserWarning, match="fewer than 2 levels"):
            model = _sample(df).glm.fit(y="api00", x=["ell", Cat("g")])

        assert [c.term for c in model.fitted.coefs] == ["_intercept_", "ell"]
