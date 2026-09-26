# src/svy/estimation/estimate.py
from __future__ import annotations

import logging

from typing import TYPE_CHECKING, Any, Literal, Mapping, NamedTuple, Sequence

import msgspec
import numpy as np
import polars as pl

from svy.core.enumerations import PopParam, QuantileMethod
from svy.core.types import Category, Number, RandomState

# Import central UI helpers
from svy.ui.printing import (
    make_panel,
    render_plain_table,
    render_rich_to_str,
    resolve_width,
    row_sort_key,
    sort_display_rows,
)


if TYPE_CHECKING:
    from svy.estimation.contrast import Contrast, ContrastExpr
    from svy.metadata import MetadataStore


log = logging.getLogger(__name__)

_DECIMAL_KEYS = ("est", "se", "lci", "uci", "cv", "deff")

_QUANTILE_PARAMS = (PopParam.QUANTILE, PopParam.MEDIAN)

# Carried by to_polars() but kept out of the printed table: df is a per-row
# value that is constant for most results, so a column would repeat one number
# down the page and widen every table. Reach for it via to_polars().
_HIDDEN_DISPLAY_COLS = ("df", "n")


def _display_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _HIDDEN_DISPLAY_COLS]


_ROW_POS = "__svy_row_pos"


class VarLabels(NamedTuple):
    """A variable's label and the value label of each of its levels in a result."""

    var_label: str
    values: dict[Any, str]


def label_vars(estimates: Sequence[Any], *, param: str, as_factor: bool = False) -> list[str]:
    """Variables whose levels an estimate's rows carry: ``by``, then ``y`` when categorical."""
    first = estimates[0]
    names = list(first.by) if first.by else []
    if param == PopParam.PROP or as_factor:
        names.append(first.y)
    return names


def _row_levels(est: Any, names: Sequence[str], n_by: int) -> list[tuple[str, Any]]:
    levels = est.by_level or (None,) * n_by
    out = [(col, levels[i] if i < len(levels) else None) for i, col in enumerate(names[:n_by])]
    if len(names) > n_by:
        out.append((names[n_by], est.y_level))
    return out


def row_counts(result_df: pl.DataFrame) -> list[int | None]:
    """Per-row record counts from a kernel result frame (its ``n`` column)."""
    if "n" not in result_df.columns:
        return [None] * result_df.height
    return [None if v is None else int(v) for v in result_df["n"].to_list()]


def row_order(estimates: Sequence[Any], *, param: str, as_factor: bool = False) -> list[int]:
    """Positions of ``estimates`` in the order ``to_polars()`` lists them.

    By domain, then by category, each compared with :func:`row_sort_key`.
    The sort is stable, so rows sharing their levels (correlation pairs) keep
    their order.
    """
    if len(estimates) < 2:
        return list(range(len(estimates)))
    first = estimates[0]
    n_by = len(first.by) if first.by else 0
    names = label_vars(estimates, param=param, as_factor=as_factor)
    levels = [[raw for _, raw in _row_levels(p, names, n_by)] for p in estimates]
    try:
        keys = [tuple(row_sort_key(v) for v in lv) for lv in levels]
        return sorted(range(len(estimates)), key=keys.__getitem__)
    except TypeError:
        # A level the data lookup could not restore stays a string beside
        # native values; compare everything as text rather than fail.
        keys = [tuple(row_sort_key(str(v)) for v in lv) for lv in levels]
        return sorted(range(len(estimates)), key=keys.__getitem__)


def _label_of(lab: VarLabels | None, raw: Any) -> str | None:
    if raw is None:
        return None
    return lab.values.get(raw, _display_level(raw)) if lab is not None else _display_level(raw)


def estimate_frame(
    estimates: Sequence[Any] | None,
    *,
    param: str,
    as_factor: bool = False,
    tidy: bool = True,
    labels: Mapping[str, VarLabels] | None = None,
    display: bool = False,
    row_index: str | None = None,
) -> pl.DataFrame:
    """The table of an estimate's rows, shared by ``Estimate`` and its serialized form.

    ``estimates`` are ``ParamEst`` or ``ParamEstData`` rows, which share their
    field names. The data view (``display=False``) keeps the codes under the
    variable names and adds a ``<var>_label`` column for each variable with
    value labels in ``labels``; rows sort by code. The display view, which
    printing uses, puts the labels in place of the codes and the variable
    labels in place of the names, and sorts on what is shown. ``row_index``
    adds a first column holding each row's position in ``estimates``.
    """
    if not estimates:
        return pl.DataFrame()
    labels = labels or {}
    first = estimates[0]
    n_by = len(first.by) if first.by else 0
    names = label_vars(estimates, param=param, as_factor=as_factor)
    labelled = {v for v in names if v in labels and labels[v].values}

    if not tidy:
        recs = []
        for p in estimates:
            rec: dict[str, Any] = {}
            for f in p.__struct_fields__:
                rec[f] = getattr(p, f)
                if f == "by_level":
                    # A struct keyed by variable, since levels of several
                    # variables differ in type and a list holds one type.
                    by_levels = _row_levels(p, names, n_by)[:n_by]
                    rec[f] = dict(by_levels) if p.by else None
                    if labelled & set(names[:n_by]):
                        rec["by_label"] = {
                            c: _label_of(labels.get(c), raw) for c, raw in by_levels
                        }
                elif f == "y_level" and len(names) > n_by and names[n_by] in labelled:
                    rec["y_level_label"] = _label_of(labels[names[n_by]], p.y_level)
            recs.append(rec)
        df = pl.from_dicts(recs)
        return df.with_row_index(row_index) if row_index else df

    show_prob = param == PopParam.QUANTILE
    rows = []
    for pos, est in enumerate(estimates):
        r: dict[str, Any] = {_ROW_POS: pos}
        if show_prob and est.prob is not None:
            # Leading column, so quantiles read p → estimate left to right.
            r["prob"] = est.prob
        for col, raw in _row_levels(est, names, n_by):
            lab = labels.get(col)
            if display:
                r[(lab.var_label if lab else "") or col] = _label_of(lab, raw)
                continue
            r[col] = raw
            if col in labelled:
                name = f"{col}_label"
                if name in r or name in names:
                    from svy.errors import MethodError

                    raise MethodError(
                        title="Label column clashes with a variable",
                        detail=f"The labels of '{col}' would go in '{name}', which is also a variable here.",
                        code="LABEL_COLUMN_CLASH",
                        where="Estimate.to_polars",
                        hint="Rename the variable, or pass use_labels=False.",
                    )
                r[name] = _label_of(lab, raw)
        for key in _DECIMAL_KEYS:
            val = getattr(est, key, None)
            if val is not None:
                r[key] = val
        if est.df is not None:
            r["df"] = est.df
        if est.n is not None:
            r["n"] = est.n
        rows.append(r)

    # Display rows sort on what is shown, after label resolution: raw codes
    # ("Rural", "Urban") and their labels ("2. Rural", "1. Urban") order
    # differently. Data rows sort on the codes.
    skip = {*_DECIMAL_KEYS, _ROW_POS, "n"}
    if not display:
        skip |= {"df"} | {f"{c}_label" for c in labelled}
    sort_display_rows(rows, numeric_keys=skip)
    order = [r.pop(_ROW_POS) for r in rows]
    df = pl.from_dicts(rows)
    if row_index:
        df = df.insert_column(0, pl.Series(row_index, order, dtype=pl.UInt32))
    return df


def stack_estimate_frames(members: Sequence[tuple[str, str | None, pl.DataFrame]]) -> pl.DataFrame:
    """Stack an estimate list's ``(y, x, frame)`` members, one row per estimate.

    A leading ``y`` column is added when the members' variables differ, and an
    ``x`` column when their denominators do, so every row stays identifiable.
    """
    frames = [(y, x, f) for y, x, f in members if not f.is_empty()]
    if not frames:
        return pl.DataFrame()
    show_y = len({y for y, _, _ in frames}) > 1
    show_x = len({x for _, x, _ in frames}) > 1
    out = []
    for y, x, f in frames:
        lead = []
        if show_y and "y" not in f.columns:
            lead.append(pl.lit(y, dtype=pl.String).alias("y"))
        if show_x and "x" not in f.columns:
            lead.append(pl.lit(x, dtype=pl.String).alias("x"))
        out.append(f.select(*lead, pl.all()) if lead else f)
    return pl.concat(out, how="diagonal_relaxed")


def _display_level(value: Any) -> str:
    """A level as printed: bools lowercase, as polars prints them."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _kernel_level_str(value: Any, *, as_float: bool = False) -> str:
    """``value`` spelled the way polars casts it to text (``True`` → ``"true"``)."""
    if value is None:
        return "__Null__"
    if isinstance(value, str):
        return value
    if as_float and isinstance(value, (bool, int, float)):
        value = float(value)
    try:
        return pl.Series([value]).cast(pl.Utf8).item()
    except Exception:
        return str(value)


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


class ParamEst(msgspec.Struct, frozen=True):
    y: str
    est: Number
    se: Number
    cv: Number
    lci: Number
    uci: Number
    by: tuple[str, ...] | None = None
    by_level: tuple[Category, ...] | None = None
    y_level: Category | None = None
    x: str | None = None
    x_level: Category | None = None
    deff: Number | None = None
    df: int | None = None
    #: Target probability, set only for quantile estimates (0.5 for the median).
    prob: Number | None = None
    #: Records behind this row: in its domain, with a nonzero weight and the
    #: variables present (after ``where=`` and null handling).
    n: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {f: getattr(self, f) for f in self.__struct_fields__}


class Estimate:
    """
    Container for estimation results.

    ``estimates`` are sorted by domain level, then by category level, in the
    order ``to_polars()`` lists them: numbers numerically, strings naturally
    (``"a2"`` before ``"a10"``). ``keys()``, ``domains`` and the rows and
    columns of ``covariance`` follow the same order.
    """

    DECIMALS: int | dict[str, int] | None = None
    PRINT_WIDTH: int | None = None
    USE_LABELS: bool = True

    __slots__ = (
        "param",
        "q_method",
        "alpha",
        "estimates",
        "covariance",
        "strata",
        "singletons",
        "domains",
        "method",
        "deff_ref",
        "n_strata",
        "n_psus",
        "as_factor",
        "where_clause",
        "design_df",
        "_cov_filled",
        "_decimals",
        "_layout",
        "_print_width",
        "_use_labels",
        "_metadata",
    )

    def __init__(
        self,
        param: PopParam,
        *,
        alpha: float = 0.05,
        rstate: RandomState = None,
        metadata: "MetadataStore | None" = None,
    ):
        self.param = param
        self.alpha = alpha
        self.estimates: list[ParamEst] = []
        self.covariance: np.ndarray = np.zeros((0, 0))
        self.strata: Sequence[Category] = []
        self.singletons: Sequence[Category] = []
        self.domains: Sequence[Category] = []
        self.method: str = "Taylor"
        #: Which SRS reference the design effect was measured against, or None
        #: when no design effect was requested. Recorded because a deff is
        #: ambiguous without it: the two references differ by 1 - n/N.
        self.deff_ref: str | None = None
        self.n_strata: int = 0
        self.n_psus: int = 0
        self.as_factor: bool = False
        #: Quantile rule of a median or quantile; None for every other parameter.
        self.q_method: QuantileMethod | None = None
        self.where_clause: str | None = None
        #: Full design degrees of freedom (R's ``degf``), as opposed to the
        #: per-row domain-aware df. Cross-domain contrasts are referred to
        #: this value.
        self.design_df: int | None = None
        #: Whether the off-diagonals of ``covariance`` were actually computed.
        #: Multi-variable convenience calls estimate each variable
        #: independently and leave them zeroed; ``contrast()`` refuses those.
        self._cov_filled: bool = False

        self._decimals = None
        self._layout = "auto"
        self._print_width = None
        self._use_labels = None
        self._metadata = metadata

    # =========================================================================
    # Label resolution helpers
    # =========================================================================

    def _resolve_use_labels(self) -> bool:
        """Resolve whether to use labels: instance -> class -> True."""
        if self._use_labels is not None:
            return self._use_labels
        return getattr(type(self), "USE_LABELS", True)

    def _get_var_label(self, var: str, use_labels: bool | None = None) -> str:
        """Get variable label or fall back to variable name."""
        resolve = use_labels if use_labels is not None else self._resolve_use_labels()
        if not resolve or self._metadata is None:
            return var
        resolved = self._metadata.resolve_labels(var)
        return resolved.var_label if resolved.has_var_label else var

    def _get_value_label(self, var: str, value: Category, use_labels: bool | None = None) -> str:
        """Get value label or fall back to string representation."""
        resolve = use_labels if use_labels is not None else self._resolve_use_labels()
        if not resolve or self._metadata is None:
            return _display_level(value)
        resolved = self._metadata.resolve_labels(var)
        # Try the value as-is first
        label = resolved.display(value)
        # If we got back the string representation, try converting to int
        if label == str(value) and isinstance(value, str):
            try:
                int_value = int(value)
                label = resolved.display(int_value)
            except (ValueError, TypeError):
                pass
        return _display_level(value) if label == str(value) else label

    # =========================================================================
    # Label configuration
    # =========================================================================

    @property
    def use_labels(self) -> bool:
        """Whether to display labels (True) or raw codes (False)."""
        return self._resolve_use_labels()

    @use_labels.setter
    def use_labels(self, value: bool | None) -> None:
        """Set per-instance label usage preference."""
        self._use_labels = value

    def _context(self) -> str:
        """Variance method, plus the quantile rule and design-effect reference when they apply.

        The reference belongs in the header rather than the frame: a deff is
        ambiguous without it, since the two references differ by 1 - n/N, but
        `to_polars` deliberately carries no provenance at all.
        """
        parts = [self.method.upper()]
        if self.param in _QUANTILE_PARAMS and self.q_method is not None:
            parts.append(f"q_method={self.q_method.value.lower()}")
        if self.deff_ref:
            parts.append(f"deff={self.deff_ref}")
        return ", ".join(parts)

    @classmethod
    def set_default_use_labels(cls, use: bool) -> None:
        """Set the default label usage for all Estimate instances."""
        cls.USE_LABELS = bool(use)

    @property
    def metadata(self) -> "MetadataStore | None":
        """Get the metadata store."""
        return self._metadata

    @metadata.setter
    def metadata(self, value: "MetadataStore | None") -> None:
        """Set the metadata store."""
        self._metadata = value

    # --- Configuration ---

    def set_print_width(self, width: int | None) -> "Estimate":
        if width is None:
            self._print_width = None
            return self
        try:
            w = int(width)
        except Exception as ex:
            raise TypeError(f"print width must be int or None; got {width!r}") from ex
        if w <= 20:
            raise ValueError("print width must be > 20 characters.")
        self._print_width = w
        return self

    @classmethod
    def set_default_print_width(cls, width: int | None) -> None:
        if width is None:
            cls.PRINT_WIDTH = None
            return
        try:
            w = int(width)
        except Exception as ex:
            raise TypeError(f"class print width must be int or None; got {width!r}") from ex
        if w <= 20:
            raise ValueError("class print width must be > 20 characters.")
        cls.PRINT_WIDTH = w

    # --- Properties ---

    @property
    def decimals(self) -> int | dict[str, int] | None:
        return self._decimals

    @decimals.setter
    def decimals(self, value: int | dict[str, int] | None) -> None:
        if value is None or isinstance(value, int):
            self._decimals = value
        elif isinstance(value, dict):
            if any(k not in _DECIMAL_KEYS for k in value):
                raise ValueError(f"Unknown decimals key; allowed: {_DECIMAL_KEYS}")
            self._decimals = dict(value)
        else:
            raise TypeError("decimals must be int | dict[str,int] | None")

    @property
    def print_width(self) -> int | None:
        return self._print_width

    @print_width.setter
    def print_width(self, value: int | None) -> None:
        self.set_print_width(value)

    @property
    def layout(self) -> Literal["auto", "horizontal", "vertical"]:
        return self._layout

    @layout.setter
    def layout(self, value: Literal["auto", "horizontal", "vertical"]) -> None:
        if value not in ("auto", "horizontal", "vertical"):
            raise ValueError("layout must be 'auto', 'horizontal', or 'vertical'")
        self._layout = value

    # --- Export ---

    def to_dicts(self) -> list[dict[str, Any]]:
        return [p.to_dict() for p in self.estimates] if self.estimates else []

    def to_polars(self, *, tidy: bool = True, use_labels: bool | None = None) -> pl.DataFrame:
        """
        The estimates as a DataFrame, one row per estimate.

        Levels keep their codes under the variable's name. With labels on,
        each variable that has value labels gets a ``<var>_label`` column
        next to it. With ``tidy=False``, ``by_level`` (and ``by_label``) is a
        struct keyed by variable, and ``y_level_label`` follows ``y_level``.
        Rows sort by code. Printing uses the labelled view instead, see
        ``to_polars_printable()``.

        Parameters
        ----------
        tidy : bool
            One column per variable (default), or the raw rows with ``by``
            and ``by_level`` as list columns.
        use_labels : bool | None
            Add the label columns. None uses the instance/class default.
        """
        resolve = use_labels if use_labels is not None else self._resolve_use_labels()
        return estimate_frame(
            self.estimates,
            param=self.param,
            as_factor=self.as_factor,
            tidy=tidy,
            labels=self._labels() if resolve else None,
        )

    def to_polars_printable(self, *, use_labels: bool | None = None) -> pl.DataFrame:
        """
        Convert estimates to a printable DataFrame.

        Parameters
        ----------
        use_labels : bool | None
            If True, apply value labels to by_level and y_level columns.
            If None, uses instance/class default.

        Returns
        -------
        pl.DataFrame
            DataFrame formatted for display.
        """
        resolve = use_labels if use_labels is not None else self._resolve_use_labels()
        return estimate_frame(
            self.estimates,
            param=self.param,
            as_factor=self.as_factor,
            labels=self._labels() if resolve else None,
            display=True,
        )

    def _labels(self) -> dict[str, VarLabels]:
        """The label of each variable whose levels the rows carry, and of each level present.

        Empty without metadata. Only levels that appear in the result are
        included, so this is what a serialized estimate stores.
        """
        if self._metadata is None or not self.estimates:
            return {}
        names = label_vars(self.estimates, param=self.param, as_factor=self.as_factor)
        n_by = len(self.estimates[0].by or ())
        out: dict[str, VarLabels] = {}
        for col in names:
            resolved = self._metadata.resolve_labels(col)
            values: dict[Any, str] = {}
            if resolved.has_value_labels:
                for p in self.estimates:
                    raw = dict(_row_levels(p, names, n_by))[col]
                    if raw is not None and raw not in values:
                        values[raw] = self._get_value_label(col, raw, use_labels=True)
            out[col] = VarLabels(resolved.var_label if resolved.has_var_label else "", values)
        return out

    def _y_level_column(self, *, use_labels: bool | None = None) -> str | None:
        """Name of the column holding this estimate's category levels, if any.

        Proportions (and ``as_factor`` means) put the level in a column named
        after the variable itself, which reads well alone but collides when
        several are stacked -- see ``EstimateList._combined``.
        """
        if not self.estimates:
            return None
        if not (self.param == PopParam.PROP or self.as_factor):
            return None
        resolve = use_labels if use_labels is not None else self._resolve_use_labels()
        return self._get_var_label(self.estimates[0].y, use_labels=resolve)

    # --- Contrasts & covariance ---

    def _row_key(self, p: ParamEst) -> Any:
        """The contrast key identifying one estimate row.

        Domain estimates key on their by-level (a tuple when several ``by``
        variables), categorical proportions on the y-level, and combined
        cases on the (by-levels..., y-level) tuple. A single ungrouped row
        keys on the variable name itself.
        """
        parts: list = []
        if p.by_level:
            parts.extend(p.by_level)
        if (self.param == PopParam.PROP or self.as_factor) and p.y_level is not None:
            parts.append(p.y_level)
        if not parts:
            return p.y
        return parts[0] if len(parts) == 1 else tuple(parts)

    def keys(self, *, labels: bool = False) -> list:
        """Contrast keys, one per estimate row, in row (and covariance) order.

        With ``labels=True``, each key component is replaced by its metadata
        value label where one exists — the same rendering the printed table
        uses. Both forms are accepted by :meth:`contrast`.
        """
        raw = [self._row_key(p) for p in self.estimates]
        if not labels:
            return raw
        return [self._labeled_key(p, k) for p, k in zip(self.estimates, raw)]

    def _labeled_key(self, p: ParamEst, key: Any) -> Any:
        """The key with every component swapped for its value label."""
        if self._metadata is None:
            return key
        labeled: list = []
        if p.by_level:
            for var, val in zip(p.by or (), p.by_level):
                labeled.append(self._get_value_label(var, val, use_labels=True))
        if (self.param == PopParam.PROP or self.as_factor) and p.y_level is not None:
            labeled.append(self._get_value_label(p.y, p.y_level, use_labels=True))
        if not labeled:
            return key
        return labeled[0] if len(labeled) == 1 else tuple(labeled)

    def _label_aliases(self) -> dict:
        """Label-form aliases for contrast keys (label → row index)."""
        if self._metadata is None:
            return {}
        aliases: dict = {}
        raw = [self._row_key(p) for p in self.estimates]
        for i, (p, key) in enumerate(zip(self.estimates, raw)):
            lk = self._labeled_key(p, key)
            if lk != key:
                aliases[lk] = i
        return aliases

    def _string_level_aliases(self) -> list[tuple[Any, int]]:
        """Keys as they were spelled when levels came back as strings.

        Levels used to be the kernel's text (``"false"``, and ``"1.0"`` for an
        ``as_factor`` mean), so contrasts written against those keys still
        resolve. Pairs rather than a dict, so a spelling shared by two rows
        is dropped as ambiguous instead of silently picking one.
        """
        float_y = self.as_factor and self.param != PopParam.PROP
        pairs: list[tuple[Any, int]] = []
        for i, p in enumerate(self.estimates):
            parts = [_kernel_level_str(v) for v in (p.by_level or ())]
            if (self.param == PopParam.PROP or self.as_factor) and p.y_level is not None:
                parts.append(_kernel_level_str(p.y_level, as_float=float_y))
            if parts:
                pairs.append((parts[0] if len(parts) == 1 else tuple(parts), i))
        return pairs

    def contrast(
        self,
        contrasts: "Mapping[Any, Any] | ContrastExpr",
        *,
        alpha: float | None = None,
    ) -> "Contrast":
        """Estimate linear contrasts between this result's estimands.

        ``contrasts`` is a contrast expression built from :func:`svy.estd`
        references (``estd("E") - estd("H")``), a sparse ``{key: coef}``
        dict, or several named contrasts ``{"name": expression-or-dict}``.
        Keys are the row identities listed by :meth:`keys` — metadata value
        labels are accepted wherever they are unambiguous. Unmentioned
        estimands get coefficient 0; unknown keys raise.

        Inference is t-based on the full design degrees of freedom
        (:attr:`design_df`, R's ``degf`` convention), not the per-row
        domain-aware df.
        """
        from svy.errors import MethodError
        from svy.estimation.contrast import linear_contrast
        from svy.utils.checks import validate_alpha

        if alpha is not None:
            alpha = validate_alpha(alpha, where="Estimate.contrast")

        if self.param in (PopParam.QUANTILE, PopParam.MEDIAN):
            raise MethodError(
                title="Contrasts are not defined for quantiles",
                detail=(
                    "Woodruff quantile intervals do not arise from a "
                    "linearized score column, so no between-quantile "
                    "covariance exists to combine them with."
                ),
                code="CONTRAST_UNSUPPORTED_PARAM",
                where="Estimate.contrast",
                param="param",
                got=self.param.name,
            )
        if not self.estimates:
            raise MethodError(
                title="Nothing to contrast",
                detail="This result carries no estimates.",
                code="CONTRAST_EMPTY",
                where="Estimate.contrast",
            )
        k = len(self.estimates)
        if k > 1 and not self._cov_filled:
            raise MethodError(
                title="No between-estimate covariance on this result",
                detail=(
                    "Multi-variable convenience calls estimate each variable "
                    "independently, so no covariance between their rows was "
                    "computed. Re-run the single-variable form (e.g. "
                    "prop('y') instead of prop(['y', ...])) and contrast "
                    "that result."
                ),
                code="CONTRAST_NO_COVARIANCE",
                where="Estimate.contrast",
            )

        df = self.design_df
        if df is None:
            row_dfs = [p.df for p in self.estimates if p.df is not None]
            if not row_dfs:
                raise MethodError(
                    title="No degrees of freedom on this result",
                    detail="Neither a design df nor per-row df is available.",
                    code="CONTRAST_NO_DF",
                    where="Estimate.contrast",
                )
            df = max(row_dfs)

        values = np.array([p.est for p in self.estimates], dtype=float)
        return linear_contrast(
            self.keys(),
            values,
            self.covariance,
            contrasts,
            df=float(df),
            alpha=alpha if alpha is not None else self.alpha,
            method=self.method,
            aliases=[*self._string_level_aliases(), *self._label_aliases().items()],
        )

    def covariance_to_polars(self) -> pl.DataFrame:
        """Tidy lower-triangle view of the between-estimate covariance.

        One row per (key_a, key_b) pair including the diagonal; the dense
        matrix stays on :attr:`covariance` in :meth:`keys` order.
        """
        keys = [str(k) for k in self.keys()]
        cov = np.asarray(self.covariance)
        rows = [
            {"key_a": keys[i], "key_b": keys[j], "cov": float(cov[i, j])}
            for i in range(len(keys))
            for j in range(i + 1)
        ]
        return pl.from_dicts(rows) if rows else pl.DataFrame()

    # --- Formatting ---

    def _get_precision(self, col: str) -> int:
        conf = self._decimals or self.DECIMALS
        defaults = {"cv": 2, "est": 4, "se": 4, "lci": 4, "uci": 4, "deff": 4}
        if conf is None:
            return defaults.get(col, 4)
        if isinstance(conf, int):
            return conf
        return conf.get(col, defaults.get(col, 4))

    def _format_val(self, col: str, v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        if isinstance(v, (float, np.floating)):
            prec = self._get_precision(col)
            val = float(v)
            if col == "cv":
                return f"{val * 100:.{prec}f}"
            if col in ("est", "se", "lci", "uci"):
                return f"{val:,.{prec}f}"
            return f"{val:.{prec}f}"
        return str(v)

    # --- Rendering ---

    def __plain_str__(self) -> str:
        """
        Plain-text fallback used when rich is not installed.
        Called by printing.plain_text_fallback(); must NOT call str(self).
        """
        df = self.to_polars_printable()
        if df.is_empty():
            return f"Estimate: {self.param.name} ({self._context()}) — <no estimates>"

        lines = [f"Estimate: {self.param.name} ({self._context()})"]
        if self.where_clause:
            lines.append(f"  where: {self.where_clause}")
        lines.append("")

        shown = _display_columns(df)
        headers = [f"{c} (%)" if c == "cv" else c for c in shown]
        rows = [[self._format_val(c, row[c]) for c in shown] for row in df.iter_rows(named=True)]
        lines.append(render_plain_table(headers, rows))

        return "\n".join(lines)

    def __str__(self) -> str:
        """
        Return string representation for standard print().
        Delegates completely to svy.ui.printing for consistency.
        """
        try:
            # 1. Resolve width (checks instance -> class -> env -> default)
            w = resolve_width(self)
            # 2. Render to string using the centralized console config
            return render_rich_to_str(self, width=w)
        except Exception:
            return self.__repr__()

    def __rich_console__(self, console, options):
        """Integration with the Rich library for pretty printing."""
        from rich import box
        from rich.table import Table
        from rich.text import Text

        df = self.to_polars_printable()

        if df.is_empty():
            yield Text("<no estimates>", style="italic dim")
            return

        # Build content list for the panel
        content = []

        # Add where clause as first item if present
        if self.where_clause:
            where_text = Text()
            where_text.append("where: ", style="dim")
            where_text.append(self.where_clause)
            content.append(where_text)
            content.append(Text(""))  # Empty line for spacing

        # TABLE CONFIGURATION
        table = Table(
            show_header=True,
            header_style="bold",
            box=box.SIMPLE_HEAVY,
            show_edge=True,
            show_lines=False,
            pad_edge=False,
            expand=False,
        )

        shown = _display_columns(df)

        for col in shown:
            justify = "right" if col in _DECIMAL_KEYS else "left"
            header = f"{col} (%)" if col == "cv" else col
            table.add_column(header, justify=justify)

        for row in df.iter_rows(named=True):
            vals = [self._format_val(col, row[col]) for col in shown]
            table.add_row(*vals)

        content.append(table)

        title = f"Estimate: [bold]{self.param.name}[/bold] ({self._context()})"

        # PANEL CONFIGURATION
        yield make_panel(content, title=title, obj=self, kind="estimate")

    def style(
        self,
        *,
        decimals: int | dict[str, int] | None = None,
        print_width: int | None = None,
        use_labels: bool | None = None,
        layout: Literal["auto", "horizontal", "vertical"] | None = None,
    ) -> "Estimate":
        """
        Fluent convenience to set presentation options.

        Parameters
        ----------
        decimals : int | dict[str, int] | None
            Decimal places for numeric columns.
        print_width : int | None
            Print width in characters.
        use_labels : bool | None
            Whether to display labels instead of codes.
        layout : {"auto", "horizontal", "vertical"} | None
            Table layout mode.

        Returns
        -------
        Estimate
            Self for method chaining.
        """
        if decimals is not None:
            self.decimals = decimals
        if print_width is not None:
            self.print_width = print_width
        if use_labels is not None:
            self.use_labels = use_labels
        if layout is not None:
            self.layout = layout
        return self


class EstimateList(list):
    """A list of :class:`Estimate` results that prints as one table.

    Returned wherever a call estimates several things at once — a sequence of
    variables (``mean(["a", "b"])``) or a sequence of probabilities
    (``quantile("x", p=(0.25, 0.75))``).

    This is a plain ``list`` subclass, so indexing, iteration, ``len()``, and
    ``isinstance(result, list)`` all behave exactly as before; the only addition
    is rendering. Printing stacks the members into a single table with a leading
    column for whatever differs between them (``y``, ``prob``, or both), because
    a bare list would otherwise print object reprs.
    """

    __slots__ = ()

    def _members(self) -> list["Estimate"]:
        return [e for e in self if isinstance(e, Estimate) and e.estimates]

    def contrast(
        self,
        contrasts: "Mapping[Any, Any] | ContrastExpr",
        *,
        alpha: float | None = None,
    ) -> "Contrast":
        """Contrast the single member of this list; raise otherwise.

        Covariance exists only within one estimation (batched variables are
        computed independently), so a multi-member list has no joint
        covariance to contrast over — index the member instead.
        """
        from svy.utils.checks import validate_alpha

        if alpha is not None:
            alpha = validate_alpha(alpha, where="EstimateList.contrast")
        members = self._members()
        if len(members) == 1:
            return members[0].contrast(contrasts, alpha=alpha)
        from svy.errors import MethodError

        raise MethodError(
            title="contrast() needs a single result",
            detail=(
                f"This list holds {len(members)} results, estimated "
                "independently — there is no covariance between them. "
                "Contrast one member, e.g. result[0].contrast(...)."
            ),
            code="CONTRAST_LIST_AMBIGUOUS",
            where="EstimateList.contrast",
        )

    def to_polars(self, *, use_labels: bool | None = None) -> pl.DataFrame:
        """Concatenate the members into one frame, one row per estimate.

        Leads with ``y`` when the members' variables differ and ``x`` when
        their denominators do, so each row names the estimate it came from.
        """
        return stack_estimate_frames(
            [
                (m.estimates[0].y, m.estimates[0].x, m.to_polars(use_labels=use_labels))
                for m in self._members()
            ]
        )

    def _combined(self, *, use_labels: bool | None = None) -> pl.DataFrame:
        """The printable frame, with a ``y`` column added when variables differ.

        ``prob`` already arrives as a column on quantile members, so only the
        variable needs re-attaching: a single-variable result keeps ``y`` out of
        the table (it is in the title), while a multi-variable one needs it to
        stay readable.
        """
        members = self._members()
        if not members:
            return pl.DataFrame()

        ys = [m.estimates[0].y for m in members]
        show_y = len(set(ys)) > 1

        # Each proportion names its level column after its own variable, so a
        # diagonal concat of several would union them into one sparse column
        # per variable -- a staircase of blanks that also squeezes the numbers
        # into ellipses. Stacking needs them under one shared name instead.
        level_cols = {}
        if show_y:
            for m in members:
                col = m._y_level_column(use_labels=use_labels)
                if col:
                    level_cols[id(m)] = col
        level_name = ""
        if level_cols:
            taken = {
                c for m in members for c in m.to_polars_printable(use_labels=use_labels).columns
            }
            taken -= set(level_cols.values())
            level_name = next(n for n in ("level", "y_level", "category") if n not in taken)

        frames = []
        for m, y in zip(members, ys):
            f = m.to_polars_printable(use_labels=use_labels)
            if f.is_empty():
                continue
            own = level_cols.get(id(m))
            if own and own in f.columns:
                f = f.rename({own: level_name})
            if show_y and "y" not in f.columns:
                f = f.select(pl.lit(y).alias("y"), pl.all())
            frames.append(f)

        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="diagonal_relaxed")

    def _title(self) -> str:
        members = self._members()
        if not members:
            return "Estimates"
        params = {m.param.name for m in members}
        methods = {m.method.upper() for m in members}
        param = params.pop() if len(params) == 1 else "MIXED"
        method = methods.pop() if len(methods) == 1 else "MIXED"
        q_methods = {
            m.q_method.value.lower()
            for m in members
            if m.param in _QUANTILE_PARAMS and m.q_method is not None
        }
        if q_methods:
            method += f", q_method={q_methods.pop() if len(q_methods) == 1 else 'mixed'}"
        ys = {m.estimates[0].y for m in members}
        suffix = f": {next(iter(ys))}" if len(ys) == 1 else ""
        return f"Estimate: [bold]{param}[/bold] ({method}){suffix}"

    def __plain_str__(self) -> str:
        """Plain-text fallback used when rich is not installed."""
        members = self._members()
        if not members:
            return "Estimates — <no estimates>"

        df = self._combined()
        if df.is_empty():
            return "Estimates — <no estimates>"

        # Strip the rich markup the panel title carries.
        title = self._title().replace("[bold]", "").replace("[/bold]", "")
        lines = [title, ""]
        shown = _display_columns(df)
        headers = [f"{c} (%)" if c == "cv" else c for c in shown]
        fmt = members[0]._format_val
        rows = [[fmt(c, row[c]) for c in shown] for row in df.iter_rows(named=True)]
        lines.append(render_plain_table(headers, rows))
        return "\n".join(lines)

    def __str__(self) -> str:
        if not self._members():
            return super().__repr__()
        try:
            return render_rich_to_str(self, width=resolve_width(self))
        except Exception:
            return self.__plain_str__()

    # A list's repr is what `print([...])` and the REPL both reach for, so the
    # table has to be the repr, not only __str__.
    __repr__ = __str__

    def __rich_console__(self, console, options):
        from rich import box
        from rich.table import Table
        from rich.text import Text

        members = self._members()
        df = self._combined() if members else pl.DataFrame()
        if df.is_empty():
            yield Text("<no estimates>", style="italic dim")
            return

        table = Table(
            show_header=True,
            header_style="bold",
            box=box.SIMPLE_HEAVY,
            show_edge=True,
            show_lines=False,
            pad_edge=False,
            expand=False,
        )

        shown = _display_columns(df)
        for col in shown:
            justify = "right" if col in _DECIMAL_KEYS else "left"
            table.add_column(f"{col} (%)" if col == "cv" else col, justify=justify)

        fmt = members[0]._format_val
        for row in df.iter_rows(named=True):
            table.add_row(*[fmt(col, row[col]) for col in shown])

        content: list = []
        wheres = {m.where_clause for m in members if m.where_clause}
        if len(wheres) == 1:
            where_text = Text()
            where_text.append("where: ", style="dim")
            where_text.append(wheres.pop())
            content.append(where_text)
            content.append(Text(""))
        content.append(table)

        yield make_panel(content, title=self._title(), obj=self, kind="estimate")
