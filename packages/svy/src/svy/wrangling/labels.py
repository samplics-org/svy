# src/svy/wrangling/labels.py
"""
Label management: apply variable labels and value labels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from svy.core.enumerations import MeasurementType
from svy.core.warnings import Severity, WarnCode
from svy.errors import DimensionError, MethodError
from svy.wrangling._helpers import _fork
from svy.wrangling._naming import _check_nan_keys

if TYPE_CHECKING:
    from svy.core.sample import Sample


# -------------------------------------------------------------------
# Measurement-type resolution
# -------------------------------------------------------------------


def _measurement_of(sample: "Sample", var: str):
    """Resolve measurement type for a variable."""
    meas = None
    schema = getattr(sample, "_schema", None)
    if schema is not None:
        try:
            meas = getattr(schema[var], "measurement", None)
        except Exception:
            pass
    if meas is None:
        mget = getattr(sample, "measurement_of", None)
        if callable(mget):
            try:
                meas = mget(var)
            except Exception:
                pass
    return meas


# -------------------------------------------------------------------
# Public function
# -------------------------------------------------------------------


def apply_labels(
    sample: "Sample",
    labels: dict[str, str] | None = None,
    categories: dict[str, dict[Any, str]] | None = None,
    *,
    strict: bool = True,
    overwrite: bool = True,
    inplace: bool = False,
) -> "Sample":
    """Apply variable labels and/or value labels."""
    if labels is None and categories is None:
        raise MethodError(
            title="No labels provided",
            detail=(
                "At least one of 'labels' or 'categories' must be provided."
            ),
            code="MISSING_LABELS",
            where="wrangling.apply_labels",
            param="labels / categories",
            hint=(
                "Pass labels={'var': 'Label'} and/or "
                "categories={'var': {val: 'text'}}."
            ),
        )

    # apply_labels only modifies metadata, not data, so we fork with
    # the same data but an independent metadata store.
    if inplace:
        target = sample
    else:
        target = _fork(sample, sample._data)

    meta = target._metadata
    local_data = target._data
    col_names = set(local_data.columns)

    allowed_for_value_labels = {
        MeasurementType.NOMINAL,
        MeasurementType.ORDINAL,
        MeasurementType.BOOLEAN,
    }

    # Collect all referenced variables
    all_vars: set[str] = set()
    if labels is not None:
        if not isinstance(labels, dict):
            raise MethodError.invalid_type(
                where="wrangling.apply_labels",
                param="labels",
                got=type(labels).__name__,
                expected="dict[str, str]",
            )
        all_vars.update(labels.keys())

    if categories is not None:
        if not isinstance(categories, dict):
            raise MethodError.invalid_type(
                where="wrangling.apply_labels",
                param="categories",
                got=type(categories).__name__,
                expected="dict[str, dict[Any, str]]",
            )
        all_vars.update(categories.keys())

    # -- Validate columns exist ------------------------------------------
    missing = [v for v in all_vars if v not in col_names]
    if missing:
        if strict:
            raise DimensionError.missing_columns(
                where="wrangling.apply_labels",
                param="labels / categories",
                missing=missing,
                available=local_data.columns,
            )
        all_vars = all_vars & col_names

    # -- Apply variable labels -------------------------------------------
    if labels is not None:
        for var, lbl in labels.items():
            if var not in all_vars:
                continue

            if not isinstance(lbl, str):
                raise MethodError.invalid_type(
                    where="wrangling.apply_labels",
                    param=f"labels[{var!r}]",
                    got=type(lbl).__name__,
                    expected="str",
                )

            if not overwrite:
                existing = meta.get(var)
                if existing is not None and existing.label is not None:
                    raise MethodError.not_applicable(
                        where="wrangling.apply_labels",
                        method="apply_labels",
                        reason=(
                            f"Variable {var!r} already has a label; "
                            "set overwrite=True to replace."
                        ),
                        param=var,
                        hint="Pass overwrite=True to replace existing labels.",
                    )

            meta.set_label(var, lbl)

    # -- Apply value labels ----------------------------------------------
    if categories is not None:
        for var, val_labels in categories.items():
            if var not in all_vars:
                continue

            if not isinstance(val_labels, dict):
                raise MethodError.invalid_type(
                    where="wrangling.apply_labels",
                    param=f"categories[{var!r}]",
                    got=type(val_labels).__name__,
                    expected="dict[value, str]",
                )

            _check_nan_keys(
                val_labels, where="wrangling.apply_labels", var=var
            )

            for key, val in val_labels.items():
                if not isinstance(val, str):
                    raise MethodError.invalid_type(
                        where="wrangling.apply_labels",
                        param=f"categories[{var!r}][{key!r}]",
                        got=type(val).__name__,
                        expected="str",
                        hint="Value labels must be strings.",
                    )

            # Check measurement type compatibility
            meas = _measurement_of(sample, var)
            if meas is not None and meas not in allowed_for_value_labels:
                raise MethodError.not_applicable(
                    where="wrangling.apply_labels",
                    method="value labels",
                    reason=(
                        f"Variable {var!r} is measured as "
                        f"{getattr(meas, 'name', meas)}."
                    ),
                    param=var,
                    hint=(
                        "Value labels are supported for "
                        "NOMINAL/ORDINAL/BOOLEAN only."
                    ),
                )

            if not overwrite:
                existing = meta.get(var)
                if (
                    existing is not None
                    and existing.value_labels is not None
                ):
                    raise MethodError.not_applicable(
                        where="wrangling.apply_labels",
                        method="apply_labels",
                        reason=(
                            f"Variable {var!r} already has value labels; "
                            "set overwrite=True to replace."
                        ),
                        param=var,
                        hint="Pass overwrite=True to replace existing value labels.",
                    )

            # Warn: label keys not matching data values
            actual_values = set(
                local_data[var].drop_nulls().unique().to_list()
            )
            label_keys = set(val_labels.keys())

            extra_label_keys = label_keys - actual_values
            if extra_label_keys:
                target.warn(
                    code=WarnCode.LABEL_KEY_NOT_IN_DATA,
                    title="Category labels for values not in data",
                    detail=(
                        f"Variable {var!r}: label keys "
                        f"{extra_label_keys} do not match any values "
                        f"in the data."
                    ),
                    where="wrangling.apply_labels",
                    level=Severity.INFO,
                    var=var,
                    got=sorted(str(k) for k in extra_label_keys),
                    hint=(
                        "These labels will be stored but won't match "
                        "any current data values."
                    ),
                )

            unlabeled_values = actual_values - label_keys
            if unlabeled_values:
                target.warn(
                    code=WarnCode.DATA_VALUE_NOT_LABELED,
                    title="Data values without labels",
                    detail=(
                        f"Variable {var!r}: values {unlabeled_values} "
                        f"have no corresponding label."
                    ),
                    where="wrangling.apply_labels",
                    level=Severity.INFO,
                    var=var,
                    got=sorted(str(v) for v in unlabeled_values),
                    hint=(
                        "Add labels for these values or ignore this "
                        "warning."
                    ),
                )

            meta.set_value_labels(var, dict(val_labels))

    return target
