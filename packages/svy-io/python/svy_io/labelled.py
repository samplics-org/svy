# python/svy_io/labelled.py
from __future__ import annotations

import numbers
import statistics
import warnings

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


Value = Union[int, float, str, None]


# ---------- helpers: typing & validation ----------


def _is_bool(x: Any) -> bool:
    # In Python, bool is a subclass of int; exclude explicitly.
    return isinstance(x, bool)


def _is_numeric_scalar(x: Any) -> bool:
    return isinstance(x, numbers.Number) and not _is_bool(x)


def _is_char_scalar(x: Any) -> bool:
    return isinstance(x, str)


def _is_numeric_seq(seq: Sequence[Any]) -> bool:
    return all((_is_numeric_scalar(v) or v is None) for v in seq)


def _is_string_seq(seq: Sequence[Any]) -> bool:
    return all((_is_char_scalar(v) or v is None) for v in seq)


def _ensure_seq(x: Any) -> List[Value]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    # allow a scalar (e.g. 1 -> [1])
    return [x]  # type: ignore[list-item]


def _normalize_labels(
    labels: Optional[Dict[Any, str] | Sequence[Tuple[Any, str]]],
) -> Dict[Any, str]:
    if labels is None:
        return {}

    # Accept dict or sequence of (code, label) pairs
    if isinstance(labels, dict):
        items = list(labels.items())
    elif isinstance(labels, Sequence) and not isinstance(labels, (str, bytes)):
        items = list(labels)
        for pair in items:
            if not (isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[1], str)):
                raise TypeError(
                    "labels must be dict[value->str] or sequence of (value, str) pairs"
                )
    else:
        raise TypeError("labels must be dict[value->str] or sequence of (value, str) pairs")

    # Haven parity: coded values must be unique (ignoring None)
    codes = [k for k, _ in items if k is not None]
    if len(set(codes)) != len(codes):
        raise ValueError("label codes must be unique")

    # Optional: warn (not error) on duplicate label strings
    names = [v for _, v in items if v is not None]
    if len(set(names)) != len(names):
        warnings.warn("duplicate label strings detected; proceeding (haven allows this)")

    return dict(items)


def _validate_labels_match_data_type(values: List[Value], labels: Dict[Any, str] | None):
    if labels is None:
        return
    if not isinstance(labels, dict):
        raise TypeError("labels must be a dict mapping value -> label (string)")
    if not all(isinstance(v, str) for v in labels.values()):
        raise TypeError("labels must have names (string values)")

    if _is_numeric_seq(values):
        if not all((_is_numeric_scalar(k) or k is None) for k in labels.keys()):
            raise TypeError("labels must be the same type as data (numeric)")
    elif _is_string_seq(values):
        if not all((_is_char_scalar(k) or k is None) for k in labels.keys()):
            raise TypeError("labels must be the same type as data (character)")
    else:
        raise TypeError("x must be a numeric or a character vector.")

    # keys (coded values) must be unique, ignoring None  (dict guarantees this;
    # we keep the explicit check for symmetry with haven)
    keys = [k for k in labels.keys() if k is not None]
    if len(set(keys)) != len(keys):
        raise ValueError("labels must be unique")

    # In our orientation (value -> label_string), also ensure label strings
    # themselves are unique, mirroring haven's "no duplicated codes" constraint.
    label_names = [nm for nm in labels.values() if nm is not None]
    if len(set(label_names)) != len(label_names):
        raise ValueError("labels must be unique")


def _validate_label(label: Optional[str]):
    if label is None:
        return
    if not isinstance(label, str):
        raise TypeError("label must be a character vector of length one")


def _cast_named(values: Optional[Sequence[Value]], target_type: type) -> Optional[List[Value]]:
    """Cast values to target type (mimics vec_cast_named from R)"""
    if values is None:
        return None
    result = []
    for v in values:
        if v is None:
            result.append(None)
        elif target_type in (int, float) and _is_numeric_scalar(v):
            result.append(target_type(v))
        elif target_type is str and _is_char_scalar(v):
            result.append(v)
        else:
            # Type mismatch
            raise TypeError(f"Cannot cast {type(v).__name__} to {target_type.__name__}")
    return result


def _combine_labels(
    x_labels: Dict[Any, str],
    y_labels: Dict[Any, str],
    x_arg: str = "",
    y_arg: str = "",
) -> Dict[Any, str]:
    """Combine label sets, preferring LHS and warning on conflicts"""
    if not y_labels:
        return x_labels
    if not x_labels:
        return y_labels

    # Check for conflicts
    conflicts = []
    for code, x_label in x_labels.items():
        if code in y_labels and y_labels[code] != x_label:
            conflicts.append(code)

    if conflicts:
        # Format conflict message
        if len(conflicts) <= 3:
            conflict_str = ", ".join(str(c) for c in conflicts)
        else:
            conflict_str = f"{conflicts[0]}, {conflicts[1]}, ... ({len(conflicts)} total)"

        warnings.warn(
            f"Conflicting labels for values: {conflict_str}. "
            f"Using labels from '{x_arg or 'left'}' argument.",
            UserWarning,
        )

    # Prefer x_labels (LHS)
    return x_labels


# ---------- core classes ----------


@dataclass
class Labelled:
    """
    Lightweight haven-like labelled vector.

    data:   sequence of numbers or strings (None allowed for missing)
    labels: mapping from *value* -> *label string* (e.g., {1: "Good"})
    label:  optional variable label string
    """

    data: Any = field(default_factory=list)
    labels: Optional[Dict[Any, str]] = None
    label: Optional[str] = None

    # ---------- validation ----------
    def __post_init__(self):
        self.data = _ensure_seq(self.data)
        if not (_is_numeric_seq(self.data) or _is_string_seq(self.data)):
            # This rejects bools (TRUE/FALSE) and mixed types.
            raise TypeError("x must be a numeric or a character vector.")
        _validate_label(self.label)
        _validate_labels_match_data_type(self.data, self.labels)

        # normalize labels dict to a copy to avoid external mutation
        self.labels = _normalize_labels(self.labels)

    # ---------- basic API ----------
    def as_list(self) -> List[Value]:
        return list(self.data)

    def as_character(self) -> List[str]:
        return ["" if v is None else str(v) for v in self.data]

    def levels(self):
        # parity with haven: levels.haven_labelled() -> NULL
        return None

    # ---------- numeric helpers ----------
    def _numeric(self) -> List[float]:
        if not _is_numeric_seq(self.data):
            raise TypeError("Can't compute on labelled<character>.")
        out: List[float] = []
        for v in self.data:
            if v is None:
                out.append(float("nan"))
            else:
                # bools are rejected at validation time; no need to special-case here
                out.append(float(v))  # type: ignore[arg-type]
        return out

    # ---------- arithmetic (strip class) ----------
    def __add__(self, other):
        if isinstance(other, Labelled):
            a = self._numeric()
            b = other._numeric()
            return [x + y for x, y in zip(a, b)]
        if isinstance(other, numbers.Number) and not _is_bool(other):
            a = self._numeric()
            return [x + float(other) for x in a]
        raise TypeError("incompatible types for addition")

    def __radd__(self, other):
        return self.__add__(other)

    # ---------- equality ----------
    def __eq__(self, other):
        if not isinstance(other, Labelled):
            return False
        return (
            self.data == other.data and self.labels == other.labels and self.label == other.label
        )

    def __ne__(self, other):
        return not self.__eq__(other)

    # ---------- python sequence protocol ----------
    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Return new Labelled with sliced data but same metadata
            return self.__class__(data=self.data[idx], labels=self.labels, label=self.label)
        return self.data[idx]

    # ---------- repr ----------
    def __repr__(self):
        class_name = self.__class__.__name__
        data_repr = repr(self.data[:10]) if len(self.data) > 10 else repr(self.data)
        if len(self.data) > 10:
            data_repr = data_repr[:-1] + ", ...]"

        parts = [f"data={data_repr}"]
        if self.labels:
            parts.append(f"labels={self.labels}")
        if self.label:
            parts.append(f"label={self.label!r}")

        return f"{class_name}({', '.join(parts)})"

    # ---------- stats ----------
    def median(self):
        vals = [v for v in self._numeric() if v == v]  # filter NaN
        if not vals:
            return float("nan")
        return float(statistics.median(vals))

    def quantile(self, q: float):
        if not (0 <= q <= 1):
            raise ValueError("q must be in [0, 1]")
        vals = [v for v in self._numeric() if v == v]
        if not vals:
            return float("nan")
        # R test expects Q1 of [1,2,3] = 1.5 -> use 'inclusive' method
        qs = statistics.quantiles(vals, n=4, method="inclusive")
        if q == 0.25:
            return float(qs[0])
        if q == 0.5:
            return float(statistics.median(vals))
        if q == 0.75:
            return float(qs[2])
        # generic linear interpolation across sorted data
        vals.sort()
        idx = q * (len(vals) - 1)
        lo = int(idx)
        hi = min(lo + 1, len(vals) - 1)
        frac = idx - lo
        return float(vals[lo] * (1 - frac) + vals[hi] * frac)

    def summary(self) -> Dict[str, float] | Dict[str, int]:
        if _is_numeric_seq(self.data):
            vals = [v for v in self._numeric() if v == v]
            if not vals:
                return {
                    "min": float("nan"),
                    "1st_qu.": float("nan"),
                    "median": float("nan"),
                    "mean": float("nan"),
                    "3rd_qu.": float("nan"),
                    "max": float("nan"),
                }
            return {
                "min": float(min(vals)),
                "1st_qu.": float(self.quantile(0.25)),
                "median": float(statistics.median(vals)),
                "mean": float(sum(vals) / len(vals)),
                "3rd_qu.": float(self.quantile(0.75)),
                "max": float(max(vals)),
            }
        else:
            return {"length": len(self.data), "na": sum(v is None for v in self.data)}


@dataclass
class LabelledSPSS(Labelled):
    """
    SPSS-specific labelled vector with user-defined missing values.

    Extends Labelled with:
    - na_values: list of specific values that should be treated as missing
    - na_range: tuple (lo, hi) defining an inclusive range of missing values
    """

    na_values: Optional[List[Value]] = None
    na_range: Optional[Tuple[Value, Value]] = None

    def __post_init__(self):
        super().__post_init__()

        # Validate na_values
        if self.na_values is not None:
            if any(v is None for v in self.na_values):
                raise ValueError("na_values cannot contain missing values (None)")

            if _is_numeric_seq(self.data):
                if not all(_is_numeric_scalar(v) for v in self.na_values):
                    raise TypeError("na_values must match data type (numeric)")
            else:
                if not all(_is_char_scalar(v) for v in self.na_values):
                    raise TypeError("na_values must match data type (character)")

        # Validate na_range
        if self.na_range is not None:
            if len(self.na_range) != 2:
                raise ValueError("na_range must be a vector of length two")

            lo, hi = self.na_range
            if lo is None or hi is None:
                raise ValueError("na_range cannot contain missing values (None)")

            if _is_numeric_seq(self.data):
                if not (_is_numeric_scalar(lo) and _is_numeric_scalar(hi)):
                    raise TypeError("na_range must match data type (numeric)")
            else:
                if not (_is_char_scalar(lo) and _is_char_scalar(hi)):
                    raise TypeError("na_range must match data type (character)")

            if not (lo < hi):
                raise ValueError("na_range must be in ascending order")

    def is_na(self) -> List[bool]:
        """Return boolean list indicating which values are missing"""
        # Start with regular None/NA
        miss = [v is None for v in self.data]

        # Check na_values
        if self.na_values is not None:
            for i, v in enumerate(self.data):
                if v in self.na_values:
                    miss[i] = True

        # Check na_range
        if self.na_range is not None:
            lo, hi = self.na_range
            for i, v in enumerate(self.data):
                if v is not None and lo <= v <= hi:
                    miss[i] = True

        return miss

    def __eq__(self, other):
        if not isinstance(other, LabelledSPSS):
            return False
        return (
            super().__eq__(other)
            and self.na_values == other.na_values
            and self.na_range == other.na_range
        )

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Return new LabelledSPSS with sliced data but same metadata
            return self.__class__(
                data=self.data[idx],
                labels=self.labels,
                label=self.label,
                na_values=self.na_values,
                na_range=self.na_range,
            )
        return self.data[idx]

    def __repr__(self):
        class_name = self.__class__.__name__
        data_repr = repr(self.data[:10]) if len(self.data) > 10 else repr(self.data)
        if len(self.data) > 10:
            data_repr = data_repr[:-1] + ", ...]"

        parts = [f"data={data_repr}"]
        if self.labels:
            parts.append(f"labels={self.labels}")
        if self.na_values:
            parts.append(f"na_values={self.na_values}")
        if self.na_range:
            parts.append(f"na_range={self.na_range}")
        if self.label:
            parts.append(f"label={self.label!r}")

        return f"{class_name}({', '.join(parts)})"

    @classmethod
    def concat(cls, vectors: List[LabelledSPSS]) -> Union[LabelledSPSS, Labelled]:
        """
        Concatenate multiple LabelledSPSS vectors.

        Returns LabelledSPSS if all vectors have compatible missing specs,
        otherwise downgrades to regular Labelled.
        """
        if not vectors:
            return cls()

        # Concatenate data
        all_data = []
        for v in vectors:
            if isinstance(v, LabelledSPSS):
                all_data.extend(v.data)
            elif isinstance(v, list):
                all_data.extend(v)
            else:
                raise TypeError(f"Cannot concatenate {type(v)}")

        # Use first vector's metadata as template
        first = vectors[0] if isinstance(vectors[0], LabelledSPSS) else cls(vectors[0])

        # Combine labels (prefer LHS)
        combined_labels = first.labels or {}
        for v in vectors[1:]:
            if isinstance(v, LabelledSPSS) and v.labels:
                combined_labels = _combine_labels(
                    combined_labels, v.labels, x_arg="left", y_arg="right"
                )

        # Check if na specs match
        na_values_match = all(
            (isinstance(v, LabelledSPSS) and v.na_values == first.na_values)
            or not isinstance(v, LabelledSPSS)
            for v in vectors
        )
        na_range_match = all(
            (isinstance(v, LabelledSPSS) and v.na_range == first.na_range)
            or not isinstance(v, LabelledSPSS)
            for v in vectors
        )

        # Variable label from first vector
        label = first.label

        # If na specs don't match, downgrade to regular Labelled
        if not na_values_match or not na_range_match:
            return Labelled(data=all_data, labels=combined_labels, label=label)

        return cls(
            data=all_data,
            labels=combined_labels,
            na_values=first.na_values,
            na_range=first.na_range,
            label=label,
        )

    @classmethod
    def from_values(cls, values: List[Value], like: LabelledSPSS) -> LabelledSPSS:
        """Create a LabelledSPSS from values, using metadata from 'like'"""
        # Type check
        if _is_numeric_seq(like.data) and not _is_numeric_seq(values):
            raise TypeError("Cannot cast non-numeric to numeric labelled")
        if _is_string_seq(like.data) and not _is_string_seq(values):
            raise TypeError("Cannot cast non-string to string labelled")

        return cls(
            data=values,
            labels=like.labels,
            na_values=like.na_values,
            na_range=like.na_range,
            label=like.label,
        )

    def cast_to(self, template: LabelledSPSS) -> LabelledSPSS:
        """
        Cast this vector to match the type/metadata of template.
        Raises ValueError if cast would lose information.
        """

        # Helper function to check if a value is considered missing
        def is_missing_in(val, na_values, na_range):
            if val is None:
                return False  # None is always missing, handled separately
            if na_values and val in na_values:
                return True
            if na_range:
                lo, hi = na_range
                if lo <= val <= hi:
                    return True
            return False

        # Check for lossy cast conditions

        # 1. Check if removing used labels
        if template.labels is not None and self.labels:
            removed_labels = set(self.labels.keys()) - set(template.labels.keys())
            if removed_labels:
                # Check if any data values use the removed labels
                for val in self.data:
                    if val in removed_labels:
                        raise ValueError(
                            f"Lossy cast: value {val} is labeled in source but not in target"
                        )

        # 2. Check if removing used missing value specifications
        # A value that's missing in source must also be missing in target
        for val in self.data:
            if val is None:
                continue  # None is always missing

            source_missing = is_missing_in(val, self.na_values, self.na_range)
            target_missing = is_missing_in(val, template.na_values, template.na_range)

            if source_missing and not target_missing:
                raise ValueError(
                    f"Lossy cast: value {val} is user-missing in source but not in target"
                )

        # Cast is safe, create new vector
        return LabelledSPSS(
            data=list(self.data),
            labels=template.labels if template.labels is not None else self.labels,
            na_values=template.na_values,
            na_range=template.na_range,
            label=self.label or template.label,
        )

    def to_int(self) -> List[int]:
        """Convert to integer list"""
        if not _is_numeric_seq(self.data):
            raise TypeError("Cannot convert string labelled to int")
        return [int(v) if v is not None else 0 for v in self.data]

    def to_float(self) -> List[float]:
        """Convert to float list"""
        if not _is_numeric_seq(self.data):
            raise TypeError("Cannot convert string labelled to float")
        return [float(v) if v is not None else float("nan") for v in self.data]

    def to_str(self) -> List[str]:
        """Convert to string list"""
        if _is_numeric_seq(self.data):
            raise TypeError("Cannot convert numeric labelled to str")
        return [str(v) if v is not None else "" for v in self.data]

    def cast(self, values: List[Value]) -> LabelledSPSS:
        """Cast values to this labelled type"""
        return self.from_values(values, like=self)


# ---- convenience factories / predicates ----


def labelled(
    x: Any = None,
    labels: Optional[Dict[Any, str]] = None,
    label: Optional[str] = None,
) -> Labelled:
    return Labelled(data=x, labels=labels, label=label)


def labelled_spss(
    x: Any = None,
    labels: Optional[Dict[Any, str]] = None,
    *,
    na_values: Optional[List[Value]] = None,
    na_range: Optional[Tuple[Value, Value]] = None,
    label: Optional[str] = None,
) -> LabelledSPSS:
    return LabelledSPSS(data=x, labels=labels, label=label, na_values=na_values, na_range=na_range)


def is_labelled(x: Any) -> bool:
    return isinstance(x, Labelled)


def is_labelled_spss(x: Any) -> bool:
    return isinstance(x, LabelledSPSS)
