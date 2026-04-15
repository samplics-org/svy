# tests/test_apply_labels.py
import math

import polars as pl
import pytest

from svy.core.enumerations import MeasurementType
from svy.errors.dimension_errors import DimensionError
from svy.errors.label_errors import LabelError
from svy.errors.method_errors import MethodError
from svy.metadata import Label


class SampleHarness:
    """
    Minimal harness to test apply_labels():
    - mirrors the attributes your method expects
    - _sample points to self (to support your chaining/return)
    - measurement_of(var) returns a MeasurementType when provided
    """

    def __init__(self, df: pl.DataFrame, measurements: dict[str, MeasurementType] | None = None):
        self._sample = self
        self._data = df
        self._labels: dict[str, Label] = {}
        self._measurements = measurements or {}

    # Optional hook the method probes for
    def measurement_of(self, var: str):
        return self._measurements.get(var)

    # ---- paste your apply_labels() implementation here or import from your Sample ----
    def apply_labels(
        self,
        cols: str | list[str],
        labels: str | list[str],
        categories: dict | None = None,
        *,
        strict: bool = True,
        overwrite: bool = True,
    ):
        # --- normalize inputs
        vs = [cols] if isinstance(cols, str) else list(cols)
        ls = [labels] if isinstance(labels, str) else list(labels)

        if len(vs) != len(ls):
            raise MethodError(
                title="Label assignment mismatch",
                detail=f"Lengths differ: vars={len(vs)} labels={len(ls)}",
                code="LABELS_LENGTH_MISMATCH",
                where="wrangling.apply_labels",
                expected="len(vars) == len(labels)",
                got={"vars": len(vs), "labels": len(ls)},
                hint="Provide one label per variable name.",
            )

        # --- resolve available columns
        df = self._sample._data
        col_names = set(df.columns)
        missing = [v for v in vs if v not in col_names]
        if missing:
            if strict:
                raise DimensionError.missing_columns(
                    where="wrangling.apply_labels",
                    param="vars",
                    missing=missing,
                    available=df.columns,
                )
            keep = [i for i, v in enumerate(vs) if v in col_names]
            vs = [vs[i] for i in keep]
            ls = [ls[i] for i in keep]
            if not vs:
                return self._sample  # nothing to do

        # --- detect categories style
        per_var_categories: dict[str, dict] | None = None
        global_categories: dict | None = None

        if categories is not None:
            if not isinstance(categories, dict):
                raise MethodError.invalid_type(
                    where="wrangling.apply_labels",
                    param="categories",
                    got=categories,
                    expected="dict[Category, str] or dict[str, dict[Category, str]]",
                    hint="Use {'var': {0:'No',1:'Yes'}} or {0:'No',1:'Yes'}.",
                )
            if any(isinstance(v, dict) for v in categories.values()):
                per_var_categories = categories  # type: ignore[assignment]
            else:
                global_categories = categories

            def _check_nan_keys(mapping: dict, *, where: str, var: str | None = None) -> None:
                for k in mapping.keys():
                    try:
                        if isinstance(k, float) and math.isnan(k):
                            where_tag = f"{where} ({var})" if var else where
                            raise LabelError.nan_key_forbidden(where=where_tag)
                    except TypeError:
                        pass

            if global_categories is not None:
                _check_nan_keys(global_categories, where="wrangling.apply_labels:global")
            else:
                for var_name, m in per_var_categories.items():  # type: ignore[union-attr]
                    if m is not None:
                        if not isinstance(m, dict):
                            raise MethodError.invalid_type(
                                where="wrangling.apply_labels",
                                param=f"categories[{var_name}]",
                                got=type(m).__name__,
                                expected="dict[Category, str] or None",
                            )
                        _check_nan_keys(m, where="wrangling.apply_labels", var=var_name)

        def _measurement_of(var: str):
            # harness exposes measurement_of
            return (
                self._sample.measurement_of(var)
                if hasattr(self._sample, "measurement_of")
                else None
            )

        allowed_for_value_labels = {
            MeasurementType.NOMINAL,
            MeasurementType.ORDINAL,
            MeasurementType.BOOLEAN,
        }

        for v, lb in zip(vs, ls):
            if not overwrite and v in self._sample._labels:
                raise MethodError.not_applicable(
                    where="wrangling.apply_labels",
                    method="apply_labels",
                    reason=f"variable '{v}' already has labels; set overwrite=True to replace",
                    param=v,
                    hint="Pass overwrite=True to replace existing label mapping.",
                )

            mapping_for_v = None
            if per_var_categories is not None:
                mapping_for_v = per_var_categories.get(v)
            elif global_categories is not None:
                mapping_for_v = global_categories

            if mapping_for_v:
                meas = _measurement_of(v)
                if meas is not None and meas not in allowed_for_value_labels:
                    raise MethodError.not_applicable(
                        where="wrangling.apply_labels",
                        method="value labels",
                        reason=f"variable '{v}' is measured as {getattr(meas, 'name', meas)}",
                        param=v,
                        hint="Value labels are supported for NOMINAL/ORDINAL/BOOLEAN only. "
                        "Remove `categories` or convert the measurement type.",
                    )

                for key, val in mapping_for_v.items():
                    if not isinstance(val, str):
                        raise MethodError.invalid_type(
                            where="wrangling.apply_labels",
                            param=f"categories[{v}][{key}]",
                            got=type(val).__name__,
                            expected="str",
                            hint="Value labels must be strings.",
                        )

                self._sample._labels[v] = Label(label=lb, categories=dict(mapping_for_v))
            else:
                self._sample._labels[v] = Label(label=lb)

        return self._sample


# -----------------------
# Fixtures
# -----------------------


@pytest.fixture
def df():
    return pl.DataFrame(
        {
            "sex": [1, 2, 1],
            "edu": [0, 1, 2],
            "age": [34, 28, 45],
            "flag": [0, 1, 0],
        }
    )


@pytest.fixture
def measurements():
    return {
        "sex": MeasurementType.NOMINAL,
        "edu": MeasurementType.ORDINAL,
        "flag": MeasurementType.BOOLEAN,
        "age": MeasurementType.CONTINUOUS,
    }


@pytest.fixture
def sample(df, measurements):
    return SampleHarness(df, measurements)


# -----------------------
# Tests
# -----------------------


def test_apply_labels_length_mismatch_raises(sample):
    with pytest.raises(MethodError) as ei:
        sample.apply_labels(["sex", "edu"], ["Sex"])  # length mismatch
    assert "LABELS_LENGTH_MISMATCH" in str(ei.value)


def test_apply_labels_missing_columns_strict(sample):
    with pytest.raises(DimensionError) as ei:
        sample.apply_labels(["sex", "missing"], ["Sex", "Missing"])
    assert "MISSING_COLUMNS" in str(ei.value)


def test_apply_labels_missing_columns_non_strict_skips(sample):
    out = sample.apply_labels(["sex", "missing"], ["Sex", "Missing"], strict=False)
    assert out is sample
    assert "sex" in sample._labels and "missing" not in sample._labels
    assert sample._labels["sex"].label == "Sex"


def test_apply_labels_variable_labels_only_ok(sample):
    sample.apply_labels(["sex", "age"], ["Sex", "Age"])
    assert sample._labels["sex"].label == "Sex"
    assert sample._labels["age"].label == "Age"
    assert sample._labels["sex"].categories is None


def test_apply_labels_with_global_categories_to_categorical(sample):
    cats = {1: "Male", 2: "Female"}
    sample.apply_labels(["sex"], ["Sex"], categories=cats)
    assert sample._labels["sex"].categories == cats


def test_apply_labels_with_per_var_categories(sample):
    per = {"sex": {1: "M", 2: "F"}, "edu": {0: "None", 1: "HS", 2: "College"}}
    sample.apply_labels(["sex", "edu"], ["Sex", "Education"], categories=per)
    assert sample._labels["sex"].categories == {1: "M", 2: "F"}
    assert sample._labels["edu"].categories == {0: "None", 1: "HS", 2: "College"}


def test_apply_labels_rejects_nan_key_in_categories_global(sample):
    cats = {float("nan"): "Missing", 1: "Yes", 0: "No"}
    with pytest.raises(LabelError) as ei:
        sample.apply_labels("flag", "Flag", categories=cats)
    assert "LABEL_NAN_KEY" in str(ei.value)


def test_apply_labels_blocks_value_labels_on_non_categorical_when_measured(sample):
    # 'age' is CONTINUOUS in fixture measurements
    with pytest.raises(MethodError) as ei:
        sample.apply_labels("age", "Age", categories={0: "Zero", 1: "One"})
    s = ei.value.summary()
    assert "value labels" in s or "METHOD_NOT_APPLICABLE" in str(ei.value)


def test_apply_labels_overwrite_false_raises(sample):
    sample.apply_labels("sex", "Sex")
    with pytest.raises(MethodError) as ei:
        sample.apply_labels("sex", "Sex v2", overwrite=False)
    assert "already has labels" in str(ei.value)


def test_apply_labels_mapping_values_must_be_strings(sample):
    # Bad: mapping value is int
    with pytest.raises(MethodError) as ei:
        sample.apply_labels("sex", "Sex", categories={1: 100})
    assert "INVALID_TYPE" in str(ei.value) or "Invalid type" in str(ei.value)


def test_apply_labels_non_strict_filters_all_out_returns_immediately(df, measurements):
    s = SampleHarness(df, measurements)
    out = s.apply_labels(["nope"], ["Nope"], strict=False)
    assert out is s
    assert s._labels == {}
