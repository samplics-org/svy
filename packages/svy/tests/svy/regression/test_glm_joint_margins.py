# tests/svy/regression/test_glm_joint_margins.py
"""
Joint predictive margins (tuple keys in at=) and marginal effects at fixed
values (variables= with at=).

References from R survey 4.5 + marginaleffects 0.32.0:

    api$y_bin <- as.integer(api$api00 > 600); api$yr_rnd <- api$yr.rnd
    d <- svydesign(ids = ~1, strata = ~stype, weights = ~pw, data = api)
    m <- svyglm(y_bin ~ awards * yr_rnd + meals, design = d,
                family = quasibinomial(), epsilon = 1e-12, maxit = 100)
    avg_predictions(m, variables = list(awards = c("No", "Yes"),
                    yr_rnd = c("No", "Yes")), by = c("awards", "yr_rnd"),
                    wts = weights(m))
    avg_predictions(m, variables = list(awards = "Yes", meals = c(20, 80)),
                    by = c("awards", "meals"), wts = weights(m))
    # cf(v, vals): api stacked once per value with v set to it
    avg_comparisons(m, variables = "awards",
                    newdata = cf("yr_rnd", c("No", "Yes")), by = "yr_rnd", wts = "pw")
    avg_slopes(m, variables = "meals",
               newdata = cf("awards", c("No", "Yes")), by = "awards", wts = "pw")
    # domain: the same model on subset(d, stype != "H")
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from svy.core.sample import Design, Sample
from svy.core.terms import Cat, Cross
from svy.errors._api import UnknownNameError
from svy.errors._model import ModelError
from svy.regression import GLMMargins, GLMMarginsList


DATA_DIR = Path(__file__).resolve().parents[2] / "test_data"

RTOL = 1e-6
# marginaleffects takes the delta-method jacobian, and a slope, numerically.
RTOL_SE = 1e-5
RTOL_SLOPE = 1e-4

R_JOINT = {
    ("No", "No"): (0.571981456610, 0.0432216228949),
    ("No", "Yes"): (0.498483430768, 0.0730608932192),
    ("Yes", "No"): (0.746636906892, 0.0303010158755),
    ("Yes", "Yes"): (0.652139446008, 0.0770989861709),
}
R_JOINT_CONTINUOUS = {
    ("Yes", 20): (0.987764072901, 0.00654808188705),
    ("Yes", 80): (0.358467881461, 0.07958365228489),
}
R_AWARDS_EFFECT_BY_YR_RND = {
    "No": (0.174655450281, 0.0523025177007),
    "Yes": (0.153656015240, 0.1073311749746),
}
R_MEALS_SLOPE_BY_AWARDS = {
    "No": (-0.0090548332973, 0.000508343025783),
    "Yes": (-0.0078224688665, 0.000494904256380),
}
R_DOMAIN_JOINT = {
    ("No", "No"): (0.631967326450, 0.0438410917430),
    ("No", "Yes"): (0.487109203901, 0.0577705298956),
    ("Yes", "No"): (0.730441823782, 0.0326815493971),
    ("Yes", "Yes"): (0.641177001716, 0.0779440947896),
}

GRID = list(R_JOINT)


@pytest.fixture(scope="module")
def api():
    return pl.read_csv(DATA_DIR / "apistrat.csv", null_values="NA").with_columns(
        (pl.col("api00") > 600).cast(pl.Int32).alias("y_bin"),
        pl.col("yr.rnd").alias("yr_rnd"),
    )


def _fit(api, where=None, x=None):
    x = x or [
        Cat("awards", ref="No"),
        Cat("yr_rnd", ref="No"),
        Cross(Cat("awards", ref="No"), Cat("yr_rnd", ref="No")),
        "meals",
    ]
    return Sample(api, Design(stratum="stype", wgt="pw")).glm.fit(
        y="y_bin", x=x, family="binomial", tol=1e-12, where=where
    )


@pytest.fixture(scope="module")
def model(api):
    return _fit(api)


def _by_cell(res: GLMMargins) -> dict[tuple, tuple[float, float]]:
    cols = list(res.at.values())
    return {tuple(c[i].item() for c in cols): (res.margin[i], res.se[i]) for i in range(len(res))}


def _assert_matches(
    got: dict, expected: dict, rtol: float = RTOL, rtol_se: float = RTOL_SE
) -> None:
    assert list(got) == list(expected)
    for cell, (m, se) in expected.items():
        np.testing.assert_allclose(got[cell][0], m, rtol=rtol)
        np.testing.assert_allclose(got[cell][1], se, rtol=rtol_se)


class TestJointPredictiveMargins:
    def test_full_grid_vs_r(self, model):
        res = model.margins(at={("awards", "yr_rnd"): None})
        _assert_matches(_by_cell(res), R_JOINT)

    def test_listed_cells_in_given_order(self, model):
        cells = [("No", "No"), ("Yes", "Yes"), ("No", "Yes"), ("Yes", "No")]
        res = model.margins(at={("awards", "yr_rnd"): cells})
        _assert_matches(_by_cell(res), {c: R_JOINT[c] for c in cells})

    def test_categorical_with_continuous_vs_r(self, model):
        res = model.margins(at={("awards", "meals"): [("Yes", 20), ("Yes", 80)]})
        _assert_matches(_by_cell(res), R_JOINT_CONTINUOUS)

    def test_domain_fit_vs_r(self, api):
        res = _fit(api, where=pl.col("stype") != "H").margins(at={("awards", "yr_rnd"): None})
        _assert_matches(_by_cell(res), R_DOMAIN_JOINT)

    def test_matches_a_combined_categorical(self, api, model):
        """Main effects plus their interaction span the same space as one 4-level factor."""
        combined = api.with_columns((pl.col("awards") + "#" + pl.col("yr_rnd")).alias("cell"))
        alt = _fit(combined, x=[Cat("cell", ref="No#No"), "meals"])
        labels = [f"{a}#{y}" for a, y in GRID]
        expected = alt.margins(at={"cell": labels})

        res = model.margins(at={("awards", "yr_rnd"): GRID})
        np.testing.assert_allclose(res.margin, expected.margin, rtol=1e-10)
        np.testing.assert_allclose(res.se, expected.se, rtol=1e-8)

    def test_one_element_tuple_equals_string_key(self, model):
        joint = model.margins(at={("awards",): None})
        single = model.margins(at={"awards": ["No", "Yes"]})
        np.testing.assert_allclose(joint.margin, single.margin, rtol=1e-12)
        np.testing.assert_allclose(joint.se, single.se, rtol=1e-12)

    def test_none_on_a_string_key_uses_fitted_levels(self, model):
        res = model.margins(at={"awards": None})
        assert res.at["awards"].tolist() == ["No", "Yes"]

    def test_mixed_keys_give_one_result_each(self, model):
        res = model.margins(at={"awards": None, ("awards", "yr_rnd"): None})
        assert [r.term for r in res] == ["awards", "awards#yr_rnd"]
        assert list(res[0].at) == ["awards"]
        assert list(res[1].at) == ["awards", "yr_rnd"]


class TestMarginalEffectsAtValues:
    def test_categorical_effect_vs_r(self, model):
        (res,) = model.margins(variables=["awards"], at={"yr_rnd": None})
        assert res.contrast.tolist() == ["Yes - No", "Yes - No"]
        _assert_matches(
            {(c,): v for c, v in zip(res.at["yr_rnd"], zip(res.margin, res.se))},
            {(k,): v for k, v in R_AWARDS_EFFECT_BY_YR_RND.items()},
        )

    def test_continuous_slope_vs_r(self, model):
        (res,) = model.margins(variables=["meals"], at={"awards": ["No", "Yes"]})
        assert res.contrast is None
        _assert_matches(
            {(c,): v for c, v in zip(res.at["awards"], zip(res.margin, res.se))},
            {(k,): v for k, v in R_MEALS_SLOPE_BY_AWARDS.items()},
            rtol=RTOL_SLOPE,
            rtol_se=RTOL_SLOPE,
        )

    def test_effect_is_the_difference_of_joint_margins(self, model):
        (eff,) = model.margins(variables=["awards"], at={"yr_rnd": None})
        joint = _by_cell(model.margins(at={("awards", "yr_rnd"): None}))
        for lvl, m in zip(eff.at["yr_rnd"], eff.margin):
            np.testing.assert_allclose(m, joint[("Yes", lvl)][0] - joint[("No", lvl)][0])

    def test_one_result_per_variable_and_key(self, model):
        res = model.margins(
            variables=["awards", "meals"],
            at={"yr_rnd": None, ("yr_rnd",): [("No",), ("Yes",)]},
        )
        assert [r.term for r in res] == ["awards", "awards", "meals", "meals"]
        assert all(r.margin_type == "ame" for r in res)
        assert len(res[0]) == 2

    def test_many_levels_repeat_each_cell_per_contrast(self, api):
        model = _fit(api, x=[Cat("stype", ref="E"), Cat("yr_rnd", ref="No"), "meals"])
        (res,) = model.margins(variables=["stype"], at={"yr_rnd": None})
        assert res.contrast.tolist() == ["H - E", "M - E"] * 2
        assert res.at["yr_rnd"].tolist() == ["No", "No", "Yes", "Yes"]

    def test_joint_cell_conditioning(self, model):
        (res,) = model.margins(
            variables=["meals"], at={("awards", "yr_rnd"): [("No", "No"), ("Yes", "Yes")]}
        )
        assert res.at["awards"].tolist() == ["No", "Yes"]
        assert res.at["yr_rnd"].tolist() == ["No", "Yes"]
        assert np.all(np.isfinite(res.se))


class TestJointMarginsOutput:
    def test_fields(self, model):
        res = model.margins(at={("awards", "yr_rnd"): None})
        assert res.term == "awards#yr_rnd"
        assert res.contrast is None
        assert res.margin_type == "predictive"
        assert len(res) == 4

    def test_to_polars_has_one_column_per_variable(self, model):
        df = model.margins(at={("awards", "meals"): [("Yes", 20), ("No", 80)]}).to_polars()
        assert df.columns == [
            "term",
            "awards",
            "meals",
            "est",
            "se",
            "lci",
            "uci",
            "df",
            "margin_type",
            "variance",
            "alpha",
        ]
        assert df["awards"].to_list() == ["Yes", "No"]
        assert df["meals"].to_list() == [20, 80]

    def test_to_polars_single_variable_is_named(self, model):
        df = model.margins(at={"meals": [20, 80]}).to_polars()
        assert df.columns == [
            "term",
            "meals",
            "est",
            "se",
            "lci",
            "uci",
            "df",
            "margin_type",
            "variance",
            "alpha",
        ]
        assert df["meals"].to_list() == [20, 80]

    def test_to_polars_effect_has_contrast_column(self, model):
        (res,) = model.margins(variables=["awards"], at={"yr_rnd": None})
        assert res.to_polars().columns == [
            "term",
            "yr_rnd",
            "contrast",
            "est",
            "se",
            "lci",
            "uci",
            "df",
            "margin_type",
            "variance",
            "alpha",
        ]

    def test_values_is_guided_to_at_and_contrast(self, model):
        res = model.margins(at={"meals": [20]})
        with pytest.raises(UnknownNameError, match="contrast"):
            res.values

    def test_to_dict(self, model):
        d = model.margins(at={("awards", "yr_rnd"): None}).to_dict()
        assert d["at"] == {"awards": ["No", "No", "Yes", "Yes"], "yr_rnd": ["No", "Yes"] * 2}
        assert "contrast" not in d

    def test_repr(self, model):
        r = repr(model.margins(at={("awards", "yr_rnd"): None}))
        assert r == "GLMMargins(term='awards#yr_rnd', n=4, 95% CI, type=predictive)"

    def test_plain_table_has_variable_headers(self, model):
        text = model.margins(variables=["awards"], at={"yr_rnd": None})[0].__plain_str__()
        assert "yr_rnd" in text
        assert "Contrast" in text
        assert "Yes - No" in text


class TestGLMMarginsList:
    def test_several_results_come_as_a_list(self, model):
        for res in (
            model.margins(),
            model.margins(at={"meals": [20], ("awards", "yr_rnd"): None}),
            model.margins(variables=["meals"], at={"yr_rnd": None}),
        ):
            assert isinstance(res, GLMMarginsList)
            assert isinstance(res, list)
            assert all(isinstance(m, GLMMargins) for m in res)

    def test_one_at_key_is_still_a_single_result(self, model):
        assert type(model.margins(at={("awards", "yr_rnd"): None})) is GLMMargins

    def test_print_shows_every_table(self, model):
        res = model.margins(at={"meals": [20, 80], ("awards", "yr_rnd"): None})
        text = str(res)
        assert "GLMMargins(" not in text
        assert "meals" in text and "awards#yr_rnd" in text
        assert repr(res) == text

    def test_plain_text_joins_every_table(self, model):
        text = model.margins(at={"meals": [20], ("awards", "yr_rnd"): None}).__plain_str__()
        assert "GLM Margins: meals" in text
        assert "GLM Margins: awards#yr_rnd" in text

    def test_empty_list_prints_as_a_list(self, model):
        res = model.margins(variables=[])
        assert isinstance(res, GLMMarginsList)
        assert str(res) == "[]"
        assert res.to_polars().is_empty()

    def test_to_polars_stacks_with_null_at_columns(self, model):
        df = model.margins(at={"meals": [20, 80], ("awards", "yr_rnd"): None}).to_polars()
        assert df.columns == [
            "term",
            "meals",
            "awards",
            "yr_rnd",
            "est",
            "se",
            "lci",
            "uci",
            "df",
            "margin_type",
            "variance",
            "alpha",
        ]
        assert df["term"].to_list() == ["meals"] * 2 + ["awards#yr_rnd"] * 4
        assert df["meals"].to_list() == [20, 80, None, None, None, None]
        assert df["awards"].null_count() == 2

    def test_to_polars_keeps_each_variable_its_type(self, model):
        df = model.margins(at={"meals": [20, 80], "awards": None}).to_polars()
        assert df["meals"].dtype == pl.Int64
        assert df["awards"].to_list() == [None, None, "No", "Yes"]

    def test_to_polars_effects_stack_contrast_labels(self, model):
        df = model.margins(variables=["meals", "awards"], at={"yr_rnd": None}).to_polars()
        assert df.columns == [
            "term",
            "yr_rnd",
            "contrast",
            "est",
            "se",
            "lci",
            "uci",
            "df",
            "margin_type",
            "variance",
            "alpha",
        ]
        assert df["contrast"].to_list() == [None, None, "Yes - No", "Yes - No"]

    def test_stacked_matches_single_result_columns(self, model):
        single = model.margins(at={"meals": [20, 80]}).to_polars()
        stacked = model.margins(at={"meals": [20, 80], "awards": None}).to_polars()
        assert stacked.filter(pl.col("term") == "meals").select(single.columns).equals(single)

    def test_unknown_attribute_is_guided(self, model):
        with pytest.raises(UnknownNameError, match="GLMMarginsList"):
            model.margins().margin


class TestAtValidation:
    def _code(self, model, **kwargs) -> str:
        with pytest.raises(ModelError) as exc:
            model.margins(**kwargs)
        return exc.value.code

    def test_none_on_a_continuous_variable(self, model):
        assert self._code(model, at={("awards", "meals"): None}) == "MARGINS_AT_LEVELS"
        assert self._code(model, at={"meals": None}) == "MARGINS_AT_LEVELS"

    @pytest.mark.parametrize("cell", [("No",), ("No", "No", "No"), "No", 1])
    def test_cell_must_match_key(self, model, cell):
        assert self._code(model, at={("awards", "yr_rnd"): [cell]}) == "MARGINS_AT_CELL"

    @pytest.mark.parametrize("key", [(), ("awards", "awards"), ("awards", 1), 3])
    def test_invalid_key(self, model, key):
        assert self._code(model, at={key: None}) == "MARGINS_AT_KEY"

    def test_at_must_be_a_dict(self, model):
        assert self._code(model, at=[("awards", "yr_rnd")]) == "MARGINS_AT_TYPE"

    def test_effect_variable_cannot_be_fixed(self, model):
        code = self._code(model, variables=["awards"], at={("awards", "yr_rnd"): None})
        assert code == "MARGINS_AT_CONFLICT"

    def test_unknown_variable_in_tuple(self, model):
        with pytest.raises(ValueError, match="not found in model"):
            model.margins(at={("awards", "nope"): [("No", 1)]})

    def test_unknown_level_in_cell(self, model):
        with pytest.raises(ModelError, match="never saw"):
            model.margins(at={("awards", "yr_rnd"): [("Maybe", "No")]})
