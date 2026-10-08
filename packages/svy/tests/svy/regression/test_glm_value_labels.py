import numpy as np
import polars as pl
import pytest

import svy

from svy.errors.model_errors import ModelError


@pytest.fixture
def sample():
    rng = np.random.default_rng(3)
    n = 600
    df = pl.DataFrame(
        {
            "str": rng.integers(1, 5, n),
            "psu": np.arange(n) // 10,
            "w": rng.uniform(1, 3, n),
            "area": rng.integers(1, 3, n).astype(float),
            "renda": rng.integers(1, 4, n),
            "sex": rng.integers(1, 3, n),
        }
    )
    noise = rng.normal(0, 1, n)
    df = df.with_columns(
        y=((pl.col("area") == 2).cast(int) + pl.col("renda") * 0.3 + noise > 1.5).cast(int)
    )
    s = svy.Sample(df, svy.Design(stratum="str", psu="psu", wgt="w"))
    s.set_value_labels("area", {1: "URBANA", 2: "RURAL"})
    s.set_value_labels("renda", {1: "Até 1 SM", 2: "Mais de 1 SM até 2 SM", 3: "2+ SM"})
    return s


def _fit(s, *x, **kw):
    return s.glm.fit("y", x=list(x), family="binomial", **kw).fitted


def test_terms_print_value_labels(sample):
    fit = _fit(sample, svy.Cat("area"), svy.Cat("renda"))
    out = fit.__plain_str__()
    assert "area_RURAL" in out and "renda_Mais de 1 SM até 2 SM" in out
    assert "area_RURAL" in str(fit)


def test_term_keeps_the_code_and_to_polars_adds_the_label(sample):
    fit = _fit(sample, svy.Cat("area"))
    frame = fit.to_polars()
    assert frame["term"].to_list() == ["_intercept_", "area_2.0"]
    assert frame["term_label"].to_list() == ["_intercept_", "area_RURAL"]


def test_levels_stay_in_code_order(sample):
    fit = _fit(sample, svy.Cat("renda"))
    assert [c.label for c in fit.coefs[1:]] == ["renda_Mais de 1 SM até 2 SM", "renda_2+ SM"]


def test_ref_by_label_equals_ref_by_code(sample):
    by_label = _fit(sample, svy.Cat("area", ref="RURAL"))
    by_code = _fit(sample, svy.Cat("area", ref=2))
    assert [c.est for c in by_label.coefs] == [c.est for c in by_code.coefs]
    assert by_label.coefs[1].label == "area_URBANA"


def test_unknown_ref_still_raises(sample):
    with pytest.raises(ModelError) as e:
        _fit(sample, svy.Cat("area", ref="NOPE"))
    assert e.value.code == "CAT_REF_NOT_FOUND"


def test_ref_label_shared_by_two_levels_raises(sample):
    sample.set_value_labels("renda", {1: "A", 2: "B", 3: "B"})
    with pytest.raises(ModelError) as e:
        _fit(sample, svy.Cat("renda", ref="B"))
    assert e.value.code == "CAT_REF_AMBIGUOUS"


def test_shared_label_names_carry_the_code(sample):
    sample.set_value_labels("renda", {1: "A", 2: "B", 3: "B"})
    fit = _fit(sample, svy.Cat("renda"))
    assert [c.label for c in fit.coefs[1:]] == ["renda_B (2)", "renda_B (3)"]


def test_cross_terms_use_labels(sample):
    fit = _fit(sample, svy.Cross(svy.Cat("area"), svy.Cat("sex")))
    assert fit.coefs[1].label == "area_RURAL:sex_2"


def test_use_labels_false_prints_codes(sample):
    fit = _fit(sample, svy.Cat("area"), use_labels=False)
    assert all(c.label is None for c in fit.coefs)
    assert "term_label" not in fit.to_polars().columns
    assert "area_2.0" in fit.__plain_str__()


def test_ref_by_label_works_with_use_labels_false(sample):
    fit = _fit(sample, svy.Cat("area", ref="RURAL"), use_labels=False)
    assert fit.coefs[1].term == "area_1.0"


def test_unlabelled_column_unchanged(sample):
    fit = _fit(sample, svy.Cat("sex"))
    assert fit.coefs[1].label is None


def test_margins_contrast_uses_labels(sample):
    glm = sample.glm.fit("y", x=[svy.Cat("area")], family="binomial")
    m = glm.margins(variables=["area"])
    m = m[0] if isinstance(m, list) else m
    assert m.to_polars()["value"].to_list() == ["RURAL - URBANA"]


def test_predict_unaffected(sample):
    glm = sample.glm.fit("y", x=[svy.Cat("area")], family="binomial")
    unlabelled = sample.glm.fit("y", x=[svy.Cat("area")], family="binomial", use_labels=False)
    new = sample.data.head(5)
    assert glm.predict(new).yhat.tolist() == unlabelled.predict(new).yhat.tolist()


def test_saved_fit_keeps_the_codes(sample):
    fit = _fit(sample, svy.Cat("area"))
    back = svy.serialize.from_json(svy.serialize.to_json(fit))
    assert back.coefs[1].term == "area_2.0"
