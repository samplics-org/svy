# src/svy/utils/helpers.py
import logging

from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from svy.core.enumerations import TableUnits
from svy.core.types import DF, Category
from svy.errors import MethodError


log = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Normalization Helpers
# ------------------------------------------------------------------ #


def _normalize_name_seq(
    names: str | Sequence[str] | None,
    *,
    param: str,
    where: str,
) -> list[str] | None:
    """
    Validate and normalize a sequence of column names.
    Accepts: None, single str, or Sequence[str].
    Returns: list[str] or None.
    Raises: MethodError if inputs contain non-strings.
    """
    if names is None:
        return None
    if isinstance(names, str):
        return [names]

    # Fast path for common list/tuple of strings
    if isinstance(names, Sequence) and not isinstance(names, (bytes, bytearray)):
        out: list[str] = []
        for i, x in enumerate(names):
            if x is None or x == "":
                continue
            if not isinstance(x, str):
                raise MethodError.invalid_type(
                    where=where,
                    param=f"{param}[{i}]",
                    got=type(x).__name__,
                    expected="str",
                    hint=f"Item at index {i} must be a string column name.",
                )
            out.append(x)
        return out

    # Fallback for invalid types passed as the container itself
    raise MethodError.invalid_type(
        where=where,
        param=param,
        got=type(names).__name__,
        expected="str | Sequence[str] | None",
        hint="Pass a column name or a list/tuple of names.",
    )


def _colspec_to_list(spec: str | Sequence[str] | None) -> list[str]:
    """
    Flatten a potentially nested column specification into a flat list of strings.
    Handles deeply nested structures iteratively (e.g., stratum=('a', ('b', 'c'))).
    """
    if spec is None:
        return []

    if isinstance(spec, str):
        return [spec]

    if not isinstance(spec, Sequence) or isinstance(spec, (bytes, bytearray)):
        raise TypeError(f"Column spec must be str or Sequence[str], got {type(spec).__name__}")

    # Iterative flattening using a deque as a stack
    out: list[str] = []
    stack = deque(spec)

    while stack:
        item = stack.popleft()

        if item is None or item == "":
            continue

        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, Sequence) and not isinstance(item, (bytes, bytearray)):
            # Prepend nested items to the stack to maintain order
            # extendleft reverses the input, so we reverse it back
            stack.extendleft(reversed(item))
        else:
            raise TypeError(f"Column names must be strings; got {type(item).__name__}")

    return out


def _normalize_columns_arg(data: DF, columns: str | Sequence[str] | None) -> list[str]:
    """
    Resolves column selection arguments.
    '*' or None -> all columns.
    str -> single column list.
    Sequence -> list of columns.
    """
    if columns is None or columns == "*":
        return list(data.columns)

    if isinstance(columns, str):
        return [columns]

    if isinstance(columns, Sequence) and not isinstance(columns, (bytes, bytearray)):
        return list(columns)

    raise TypeError(f"columns must be str | Sequence[str] | None, got {type(columns).__name__}")


# ------------------------------------------------------------------ #
# Mapping Helpers
# ------------------------------------------------------------------ #


def _get_keys_from_maps(**kwargs: Mapping[Any, Any]) -> list[Category]:
    """
    Ensure all provided mappings share the exact same set of keys.
    Returns the keys from the first mapping as a list.
    """
    # Filter for mapping arguments only
    mappings = [(k, v) for k, v in kwargs.items() if isinstance(v, Mapping)]

    if not mappings:
        raise ValueError("No dict-like arguments were provided.")

    # Canonical reference is the first mapping
    ref_name, ref_map = mappings[0]
    if not ref_map:
        raise ValueError(f"Argument '{ref_name}' is an empty mapping.")

    ref_keys = list(ref_map.keys())
    ref_set = set(ref_keys)

    # Verify consistency
    for name, m in mappings[1:]:
        current_set = set(m.keys())
        if current_set != ref_set:
            missing = ref_set - current_set
            extra = current_set - ref_set
            parts = []
            if missing:
                parts.append(f"missing: {sorted(missing)!r}")
            if extra:
                parts.append(f"extra: {sorted(extra)!r}")

            detail = "; ".join(parts)
            raise ValueError(f"Keys of '{name}' do not match '{ref_name}'. {detail}")

    return ref_keys


def _to_map(keys: Sequence[Category], val: Any) -> dict[Category, Any]:
    """
    Broadcast a scalar value to a dict, or convert an existing mapping.
    """
    if isinstance(val, Mapping):
        return dict(val)
    return {k: val for k in keys}


# ------------------------------------------------------------------ #
# Weight Scaling
# ------------------------------------------------------------------ #


def _scale_weights_for_units(
    w: np.ndarray,
    *,
    units: TableUnits,
    count_total: float | int | None,
) -> np.ndarray:
    """
    Scale weights in-place or return new array based on display units.
    """
    sum_w = float(w.sum())

    if sum_w == 0:
        return w  # Return as-is to avoid NaNs, let downstream handle zero-sum

    if count_total is not None:
        # Scale to match a specific population total
        factor = float(count_total) / sum_w
        return w * factor

    if units is TableUnits.PROPORTION:
        return w / sum_w

    if units is TableUnits.PERCENT:
        return w * (100.0 / sum_w)

    if units is TableUnits.COUNT:
        return w

    raise ValueError(f"Unsupported unit type: {units}")
