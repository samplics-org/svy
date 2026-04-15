# src/svy/core/sample.py
from __future__ import annotations

import copy
import logging

from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping, Self, Sequence, cast

import numpy as np
import polars as pl

from svy.core.constants import (
    _INTERNAL_CONCAT_SUFFIX,
    SVY_ROW_INDEX,
)
from svy.core.describe import DescribeResult
from svy.core.describe_runtime import run_describe
from svy.core.design import Design, PopSize, RepWeights
from svy.core.enumerations import MeasurementType
from svy.core.expr import to_polars_expr
from svy.core.types import (
    _MISSING,
    DF,
    Category,
    ColumnsArg,
    DomainScalarMap,
    Number,
    OrderByArg,
    WhereArg,
    _MissingType,
)
from svy.core.warnings import Severity, SvyWarning, WarnCode, WarningStore
from svy.errors import DimensionError, MethodError, SvyError
from svy.metadata import MetadataStore
from svy.utils.helpers import _colspec_to_list, _normalize_columns_arg
from svy.utils.random_state import seed_from_random_state
from svy.utils.trace import log_step


if TYPE_CHECKING:
    from svy.categorical import Categorical
    from svy.core.singleton import Singleton
    from svy.estimation import Estimation
    from svy.metadata import LabellingCatalog
    from svy.questionnaire import Questionnaire
    from svy.regression import GLM
    from svy.selection import Selection
    from svy.weighting import Weighting
    from svy.wrangling import Wrangling

log = logging.getLogger(__name__)

INTEGER_DTYPES = {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}
FLOAT_DTYPES = {pl.Float32, pl.Float64}


class Sample:
    """A sample class for survey data."""

    # ════════════════════════════════════════════════════════════════════════
    # Class-level printing style (private; runtime-tunable)
    # ════════════════════════════════════════════════════════════════════════
    _PRINT_PANEL: bool = True
    _PANEL_TITLE: str = "Sample"
    _PANEL_BORDER: str = "cyan"

    # Optional class-level width override.
    PRINT_WIDTH: int | None = None

    # ════════════════════════════════════════════════════════════════════════
    # Slots (private state)
    # ════════════════════════════════════════════════════════════════════════
    __slots__ = (
        "_data",
        "_fpc",
        "_design",
        "_metadata",
        "_schema",
        "_singletons",
        "_internal_design",
        "_warnings",
        "_print_width",
        "_singleton_result",
        "__dict__",
    )

    # ════════════════════════════════════════════════════════════════════════
    # DUNDER METHODS (lifecycle & identity)
    # ════════════════════════════════════════════════════════════════════════

    def __init__(
        self,
        data: pl.DataFrame,
        design: Design | None = None,
        *,
        catalog: LabellingCatalog | None = None,
        questionnaire: Questionnaire | None = None,
    ) -> None:
        if design is None:
            local_design = Design(row_index=SVY_ROW_INDEX)
            self._fpc = 1
        else:
            local_design = copy.deepcopy(design)
            if getattr(local_design, "row_index", None) is None:
                local_design = local_design.update(row_index=SVY_ROW_INDEX)

            # FPC is computed in the estimation layer from pop_size column(s).
            # _fpc is kept for backward compatibility but is not used by the
            # Rust-backed estimation path.
            self._fpc = 1

        self._design = local_design

        local_data = data.clone().fill_nan(None)
        if SVY_ROW_INDEX not in local_data.columns:
            local_data = local_data.with_row_index(name=SVY_ROW_INDEX)

        if design is not None:
            local_data, (_, stratum_cols, psu_cols, ssu_cols) = (
                self._create_concatenated_cols_from_lists(
                    data=local_data,
                    design=design,
                    by=None,
                    null_token="__Null__",
                    suffix=_INTERNAL_CONCAT_SUFFIX,
                )
            )
            self._internal_design = {
                "stratum": f"stratum{_INTERNAL_CONCAT_SUFFIX}" if stratum_cols else None,
                "psu": f"psu{_INTERNAL_CONCAT_SUFFIX}" if psu_cols else None,
                "ssu": f"ssu{_INTERNAL_CONCAT_SUFFIX}" if ssu_cols else None,
                "suffix": _INTERNAL_CONCAT_SUFFIX,
            }
        else:
            self._internal_design = {
                "stratum": None,
                "psu": None,
                "ssu": None,
                "suffix": _INTERNAL_CONCAT_SUFFIX,
            }

        self._warnings: WarningStore = WarningStore()
        self._data = local_data

        # Initialize MetadataStore (replaces _labels)
        self._metadata = MetadataStore(catalog=catalog)
        self._metadata.infer_from_dataframe(cast(pl.DataFrame, self._data))

        if questionnaire is not None:
            self._metadata.import_from_questionnaire(questionnaire, catalog=catalog)

        self._print_width = None

        self._check_for_singletons()
        self._validate_design()

    def __hash__(self) -> int:
        _d = (
            cast(pl.DataFrame, self._data)
            if not isinstance(self._data, pl.LazyFrame)
            else cast(pl.DataFrame, self._data.collect())
        )
        sorted_data = _d.select(sorted(_d.columns))
        row_hashes = sorted_data.hash_rows().to_list()
        design_hash = hash(self._design) if self._design else hash(None)
        return hash(tuple([design_hash] + row_hashes))

    # ════════════════════════════════════════════════════════════════════════
    # INITIALIZATION HELPERS
    # ════════════════════════════════════════════════════════════════════════

    def _calculate_fpc(
        self, pop_size: dict[Category, Number] | Number
    ) -> dict[Category, Number] | Number:
        if isinstance(pop_size, Number):
            return int(1 - cast(pl.DataFrame, self._data).height / pop_size)
        elif isinstance(pop_size, dict):
            fpc: dict[Category, Number] = {}
            for key, value in pop_size.items():
                if not isinstance(key, str) or not isinstance(value, Number):
                    raise TypeError("pop_size must be a Number or a dict[Categoryber, Number]")
                fpc[key] = int(
                    1 - cast(pl.DataFrame, self._data).shape[0] / sum(pop_size.values())
                )
            return fpc
        else:
            raise TypeError("pop_size must be a Number or a dict[Categoryber, Number]")

    # ════════════════════════════════════════════════════════════════════════
    # PRIVATE HELPERS FOR DUNDERS
    # ════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _pad_and_surround(text: str, *, indent: int = 2, surround: bool = False) -> str:
        if text is None:
            return ""
        text = str(text).rstrip("\n")
        if indent > 0:
            pad_str = " " * indent
            text = "\n".join(pad_str + line if line else pad_str for line in text.splitlines())
        if surround:
            return f"\n{text}\n"
        return text

    @staticmethod
    def _fmt_tuple_names(x) -> str:
        if x is None:
            return "None"
        if isinstance(x, (tuple, list)):
            inner = ", ".join(str(v) for v in x)
            if len(x) == 1:
                inner += ","
            return f"({inner})"
        return str(x)

    @staticmethod
    def _fmt_ident(x) -> str:
        if x is None:
            return "None"
        name = getattr(x, "name", None)
        if isinstance(name, str):
            return name
        if isinstance(x, str):
            return x
        return repr(x)

    def _design_summary(self) -> list[tuple[str, str]]:
        design = self._design
        wgt = getattr(design, "wgt", None) or getattr(design, "weight", None)
        # Use RepWeights.__plain_str__ so the sub-fields render consistently
        # with how Design.__plain_str__ displays them.
        fn = getattr(design.rep_wgts, "__plain_str__", None)
        rw_lines = fn().splitlines() if callable(fn) else None

        rows = [
            ("Row index", str(getattr(design, "row_index", None))),
            ("Stratum", self._fmt_tuple_names(getattr(design, "stratum", None))),
            ("PSU", self._fmt_tuple_names(getattr(design, "psu", None))),
            ("SSU", self._fmt_tuple_names(getattr(design, "ssu", None))),
            ("Weight", str(wgt)),
            ("With replacement", str(bool(getattr(design, "wr", False)))),
            ("Prob", str(getattr(design, "prob", None))),
            ("Hit", str(getattr(design, "hit", None))),
            ("MOS", str(getattr(design, "mos", None))),
            ("Population size", str(getattr(design, "pop_size", None))),
        ]
        if rw_lines:
            rows.append(("Replicate weights", ""))
            for sub_line in rw_lines[1:]:
                rows.append((None, f"    {sub_line}"))
        else:
            rows.append(("Replicate weights", "None"))
        return rows

    # ════════════════════════════════════════════════════════════════════════
    # PRETTY STRING REPRESENTATION
    # ════════════════════════════════════════════════════════════════════════
    def __rich_console__(self, console, options):
        from rich.console import Group
        from rich.padding import Padding
        from rich.table import Table
        from rich.text import Text

        from svy.ui.printing import make_panel

        try:
            n_rows = int(self._data.height)
            n_cols = int(self._data.width)
        except Exception:
            n_rows, n_cols = 0, 0

        try:
            strata_df = self.strata
            n_strata = None if strata_df.is_empty() else strata_df.height
        except Exception:
            n_strata = None

        try:
            psus_df = self.psus
            n_psus = None if psus_df.is_empty() else psus_df.height
        except Exception:
            n_psus = None

        design_rows = self._design_summary()

        # Design table — no header, field names not bold
        t = Table(
            show_header=False,
            box=None,
            show_edge=False,
            show_lines=False,
            pad_edge=False,
            expand=False,
        )
        t.add_column("Field", justify="left", no_wrap=True)
        t.add_column("Value", justify="left", no_wrap=False, overflow="fold")
        for k, v in design_rows:
            if k is None:
                t.add_row(v, "")
            else:
                t.add_row(k, v)

        content = Group(
            Text("Survey Data", style="bold"),
            Text(
                f"  Rows     : {n_rows}\n"
                f"  Columns  : {n_cols}\n"
                f"  Strata   : {n_strata if n_strata is not None else 'None'}\n"
                f"  PSUs     : {n_psus if n_psus is not None else 'None'}"
            ),
            Text(""),
            Text("Survey Design", style="bold"),
            Padding(t, (0, 0, 0, 2)),
        )
        yield make_panel([content], title="Sample", obj=self, kind="sample")

    def __plain_str__(self) -> str:
        """Plain-text fallback when rich is not installed."""
        try:
            n_rows = int(self._data.height)
            n_cols = int(self._data.width)
        except Exception:
            n_rows, n_cols = 0, 0

        try:
            strata_df = self.strata
            n_strata = None if strata_df.is_empty() else strata_df.height
        except Exception:
            n_strata = None

        try:
            psus_df = self.psus
            n_psus = None if psus_df.is_empty() else psus_df.height
        except Exception:
            n_psus = None

        design_rows = self._design_summary()

        lines = [
            "Survey Data",
            f"  Rows     : {n_rows}",
            f"  Columns  : {n_cols}",
            f"  Strata   : {n_strata if n_strata is not None else 'None'}",
            f"  PSUs     : {n_psus if n_psus is not None else 'None'}",
            "",
            "Survey Design",
        ]

        # Align field labels
        k_width = max((len(k) for k, v in design_rows if k), default=0)
        for k, v in design_rows:
            if k is None:
                # Sub-field row (rep weight details)
                lines.append(f"  {v}")
            elif v:
                lines.append(f"  {k.ljust(k_width)} : {v}".rstrip())
            else:
                lines.append(f"  {k}")
        return "\n".join(lines)

    def __str__(self) -> str:
        from svy.ui.printing import render_rich_to_str, resolve_width

        try:
            w = resolve_width(self, default=65)
            result = render_rich_to_str(self, width=w)
        except Exception:
            result = self.__plain_str__()

        return self._pad_and_surround(result, indent=2, surround=False)

    # ───────────────────────────────
    # Print-width configuration
    # ───────────────────────────────
    def set_print_width(self, width: int | None) -> "Sample":
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

    # ════════════════════════════════════════════════════════════════════════
    # DESIGN DIAGNOSTICS
    # ════════════════════════════════════════════════════════════════════════
    def _check_for_singletons(self):
        stratum = self._internal_design["stratum"]
        psu = self._internal_design["psu"]
        stratum_cols = [stratum] if isinstance(stratum, str) else (stratum if stratum else None)
        psu_cols = [psu] if isinstance(psu, str) else (psu if psu else None)

        singletons = pl.DataFrame()

        if stratum is not None and psu is not None:
            singletons = (
                cast(pl.DataFrame, self._data)
                .select(cast(list, stratum_cols) + cast(list, psu_cols))
                .unique()
                .group_by(stratum_cols)
                .agg(n_psus=pl.len())
                .filter(pl.col("n_psus") == 1)
            )
        elif psu is not None:
            singletons = (
                cast(pl.DataFrame, self._data)
                .select(cast(list, psu_cols) + [SVY_ROW_INDEX])
                .unique()
                .group_by(psu_cols)
                .agg(n_psus=pl.len())
                .filter(pl.col("n_psus") == 1)
            )
        elif stratum is not None:
            singletons = (
                cast(pl.DataFrame, self._data)
                .select(cast(list, stratum_cols) + [SVY_ROW_INDEX])
                .unique()
                .group_by(stratum_cols)
                .agg(n_psus=pl.len())
                .filter(pl.col("n_psus") == 1)
            )

        if not singletons.is_empty():
            if stratum_cols is not None:
                self._singletons = singletons.select(stratum_cols).to_dicts()
            elif psu_cols is not None:
                self._singletons = singletons.select(psu_cols).to_dicts()
            else:
                self._singletons = None
        else:
            self._singletons = None

    # ════════════════════════════════════════════════════════════════════════
    # SCHEMA & COLUMN HELPERS
    # ════════════════════════════════════════════════════════════════════════
    @staticmethod
    def _schema_names(data: DF) -> set[str]:
        if isinstance(data, pl.DataFrame):
            return set(data.columns)
        return set(data.collect_schema().names())

    def _clean_subset(self, data: DF, cols: Iterable[str | None]) -> list[str]:
        names = self._schema_names(data)
        out = [c for c in cols if c and c in names]
        return out

    # ════════════════════════════════════════════════════════════════════════
    # DESIGN CONCATENATION HELPERS
    # ════════════════════════════════════════════════════════════════════════
    def _concatenate_cols(
        self,
        data: DF,
        *,
        sep: str = "__by__",
        null_token: str = "∅",
        categorical: bool = True,
        drop_original: bool = False,
        rename_suffix: str = "_key",
        **groups: Sequence[str],
    ) -> DF:
        if not groups:
            return data

        names = self._schema_names(data)
        key_exprs: list[pl.Expr] = []
        used_inputs: set[str] = set()

        for group_name, cols in groups.items():
            if not cols:
                continue
            missing = [c for c in cols if c not in names]
            if missing:
                raise KeyError(
                    f"group '{group_name}' references missing columns {missing}; available: {sorted(names)}"
                )
            parts = [pl.col(c).cast(pl.Utf8).fill_null(null_token) for c in cols]
            expr = pl.concat_str(parts, separator=sep)
            if categorical:
                expr = expr.cast(pl.Categorical)
            key_exprs.append(expr.alias(f"{group_name}{rename_suffix}"))
            used_inputs.update(cols)

        out = data.with_columns(key_exprs)
        if drop_original and used_inputs:
            drop_cols = [c for c in used_inputs if c in self._schema_names(out)]
            if drop_cols:
                out = out.drop(drop_cols)
        return out

    @staticmethod
    def _to_cols(spec: str | Sequence[str] | None) -> list[str]:
        if spec is None:
            return []
        if isinstance(spec, str):
            if spec == "":
                raise ValueError("column names must not be empty strings")
            return [spec]
        if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes, bytearray)):
            out: list[str] = []
            for i, x in enumerate(spec):
                if not isinstance(x, str):
                    raise TypeError(
                        f"column names must be str; item {i} has type {type(x).__name__}"
                    )
                if x == "":
                    raise ValueError("column names must not be empty strings")
                out.append(x)
            return out
        raise TypeError(f"invalid column spec: {spec!r}")

    def _create_concatenated_cols_from_lists(
        self,
        data: DF,
        design: Design,
        *,
        by: str | tuple[str, ...] | None = None,
        null_token: str = "__Null__",
        suffix: str = _INTERNAL_CONCAT_SUFFIX,
        categorical: bool = True,
        drop_original: bool = False,
    ) -> tuple[DF, tuple[list[str], list[str], list[str], list[str]]]:
        by_cols = self._to_cols(by)
        stratum_cols = self._to_cols(design.stratum)
        psu_cols = self._to_cols(design.psu)
        ssu_cols = self._to_cols(design.ssu)

        out = self._concatenate_cols(
            data=data,
            null_token=null_token,
            categorical=categorical,
            drop_original=drop_original,
            rename_suffix=_INTERNAL_CONCAT_SUFFIX,
            by=by_cols,
            stratum=stratum_cols,
            psu=psu_cols,
            ssu=ssu_cols,
        )
        concat_data = cast(pl.DataFrame, out.collect() if isinstance(out, pl.LazyFrame) else out)
        return concat_data, (by_cols, stratum_cols, psu_cols, ssu_cols)

    @staticmethod
    def _dedup_preserve_order(cols: list[str]) -> list[str]:
        return list(dict.fromkeys(cols))

    def _design_arrays_from_concat(
        self,
        *,
        data: pl.DataFrame,
        design: Design,
        by: str | tuple[str, ...] | None = None,
        null_token: str = "__Null__",
        suffix: str = _INTERNAL_CONCAT_SUFFIX,
    ) -> tuple[pl.DataFrame, dict[str, np.ndarray | None]]:
        _raw_concat, (_by_cols, _stratum_cols, _psu_cols, _ssu_cols) = (
            self._create_concatenated_cols_from_lists(
                data=data,
                design=design,
                by=by,
                null_token=null_token,
                suffix=suffix,
                categorical=True,
                drop_original=False,
            )
        )
        concat_data = cast(pl.DataFrame, _raw_concat)
        arrays: dict[str, np.ndarray | None] = {
            "stratum": (concat_data[f"stratum{suffix}"].to_numpy() if design.stratum else None),
            "psu": (
                concat_data[f"psu{suffix}"].to_numpy()
                if design.psu
                else np.arange(concat_data.height, dtype=int)
            ),
            "ssu": (concat_data[f"ssu{suffix}"].to_numpy() if design.ssu else None),
            "by": (concat_data[f"by{suffix}"].to_numpy() if by is not None else None),
        }
        return concat_data, arrays

    # ════════════════════════════════════════════════════════════════════════
    # VALIDATION OF DESIGN
    # ════════════════════════════════════════════════════════════════════════
    def _validate_design(self) -> None:
        data: pl.DataFrame = (
            cast(pl.DataFrame, self._data)
            if not isinstance(self._data, pl.LazyFrame)
            else cast(pl.DataFrame, self._data.collect())
        )
        design = cast("Design", self._design)

        if (
            design.row_index is None
            or not isinstance(design.row_index, str)
            or not design.row_index
        ):
            raise ValueError("Design.row_index must be a non-empty string.")
        if design.row_index not in data.columns:
            raise ValueError(f"Design.row_index {design.row_index!r} not found in data columns.")

        schema: dict[str, pl.DataType] = data.schema

        def _is_integer_dtype(dt: pl.DataType) -> bool:
            return dt in INTEGER_DTYPES

        def _is_numeric_dtype(dt: pl.DataType) -> bool:
            return dt in INTEGER_DTYPES or dt in FLOAT_DTYPES

        # 1. Validate explicit design fields (stratum, psu, wgt, etc.)
        # FIX: Pass data_columns to enable auto-detection of replicate weight padding
        needed_cols = design.specified_fields(
            ignore_cols=("wr",),
            data_columns=data.columns,
        )
        missing = [c for c in needed_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Design references columns not found in data: {missing}")

        # 1b. Validate pop_size columns are numeric
        if design.pop_size is not None:
            if isinstance(design.pop_size, PopSize):
                for col_name in (design.pop_size.psu, design.pop_size.ssu):
                    if col_name in schema and not _is_numeric_dtype(schema[col_name]):
                        raise TypeError(f"Population size column {col_name!r} must be numeric.")
            elif isinstance(design.pop_size, str):
                if design.pop_size in schema and not _is_numeric_dtype(schema[design.pop_size]):
                    raise TypeError(f"Population size column {design.pop_size!r} must be numeric.")

        # 2. Validate Replicate Weights
        if design.rep_wgts is not None:
            rw = design.rep_wgts

            # Use auto-detection from actual data columns
            expected_rep_cols = rw.columns_from_data(
                data.columns
            )  # ← KEY FIX: Use columns_from_data

            # Check that all expected columns exist
            missing_rep = [c for c in expected_rep_cols if c not in data.columns]
            if missing_rep:
                raise ValueError(
                    f"Expected replicate weight columns not found in data: {missing_rep[:10]}"
                    + ("..." if len(missing_rep) > 10 else "")
                )

            # Check Count Consistency
            if rw.n_reps > 0 and rw.n_reps != len(expected_rep_cols):
                raise ValueError(
                    f"RepWeights.n_reps ({rw.n_reps}) does not match number of columns found ({len(expected_rep_cols)})."
                )

            # Check Numeric Type
            non_numeric_rep = [c for c in expected_rep_cols if not _is_numeric_dtype(schema[c])]
            if non_numeric_rep:
                sample_bad = non_numeric_rep[:3]
                suffix = "..." if len(non_numeric_rep) > 3 else ""
                raise TypeError(
                    f"Replicate weight columns must be numeric; got non-numeric types for: {sample_bad}{suffix}"
                )

        # 3. Validate types for standard design variables
        if design.wgt is not None:
            if design.wgt not in schema:
                raise ValueError(f"Weight column {design.wgt!r} not found in data.")
            if not _is_numeric_dtype(schema[design.wgt]):
                raise TypeError(f"Weight column {design.wgt!r} must be numeric.")

        if design.prob is not None:
            if not _is_numeric_dtype(schema[design.prob]):
                raise TypeError(f"'prob' column {design.prob!r} must be numeric.")
        if design.mos is not None:
            if not _is_numeric_dtype(schema[design.mos]):
                raise TypeError(f"'mos' column {design.mos!r} must be numeric.")
        if design.hit is not None:
            if not _is_integer_dtype(schema[design.hit]):
                raise TypeError(f"'hit' column {design.hit!r} must be an integer dtype.")

        # 4. Check Row Index Integrity
        if data.select(pl.col(design.row_index).is_null().any()).item():
            raise ValueError("row_index contains nulls.")
        if data.select(pl.col(design.row_index).n_unique() != pl.len()).item():
            raise ValueError("row_index must be unique.")

    def _check_rep_wgts_against_df(self, rw: RepWeights | None) -> None:
        if rw is None:
            return
        if rw.n_reps and rw.wgts and rw.n_reps != len(rw.wgts):  # type: ignore[attr-defined]
            raise ValueError(f"RepWeights.n_reps ({rw.n_reps}) != len(rw.wgts) ({len(rw.wgts)})")  # type: ignore[attr-defined]
        missing = [c for c in rw.wgts if c not in cast(pl.DataFrame, self._data).columns]  # type: ignore[attr-defined]
        if missing:
            raise ValueError(f"Replicate weight columns not in data: {missing}")

    # ════════════════════════════════════════════════════════════════════════
    # COPYING / REPLACEMENT (internal)
    # ════════════════════════════════════════════════════════════════════════
    def _replace_data(self, data: pl.DataFrame) -> "Sample":
        """Return a shallow copy of this Sample with new data."""
        new = copy.copy(self)
        new._data = data
        return new

    def _remove_invalid_weight(
        self, *, df: pl.DataFrame | None = None, wgt_col: str | None = None
    ) -> tuple[pl.DataFrame, dict[str, int]]:
        # NOTE: 'df' argument name preserved for API compatibility
        if df is None:
            df = (
                cast(pl.DataFrame, self._data)
                if not isinstance(self._data, pl.LazyFrame)
                else cast(pl.DataFrame, self._data.collect())
            )
        if wgt_col is None:
            n = int(df.height)
            return df, {"n_in": n, "n_out": n, "n_removed": 0}
        if wgt_col not in df.collect_schema().names():
            raise ValueError(f"Weight column {wgt_col!r} not found in data.")

        n_in = int(df.height)
        w_expr = pl.col(wgt_col).cast(pl.Float64, strict=False)
        cond = w_expr.is_finite() & (w_expr > 0)
        df_clean = df.filter(cond)
        n_out = int(df_clean.height)
        n_removed = n_in - n_out

        if n_removed > 0:
            try:
                self.warn(
                    code="INVALID_WEIGHT_ROWS_REMOVED",
                    title="Rows removed due to invalid weights",
                    detail=(f"Removed {n_removed} row(s) where {wgt_col!r} was NaN/inf/<=0."),
                    where="sample._remove_invalid_weight",
                    level=Severity.INFO,
                    param=wgt_col,
                    extra={"n_in": n_in, "n_out": n_out, "n_removed": n_removed},
                )
            except Exception:
                pass
        return df_clean, {"n_in": n_in, "n_out": n_out, "n_removed": n_removed}

    # ════════════════════════════════════════════════════════════════════════
    # FACET ACCESSORS (subsystems)
    # ════════════════════════════════════════════════════════════════════════
    @property
    def weighting(self) -> Weighting:
        from svy.weighting import Weighting

        return Weighting(self)

    @property
    def wrangling(self) -> Wrangling:
        from svy.wrangling import Wrangling

        return Wrangling(self)

    @property
    def sampling(self) -> Selection:
        from svy.selection import Selection

        return Selection(self)

    @property
    def singleton(self) -> Singleton:
        from svy.core.singleton import Singleton

        return Singleton(self)

    @property
    def estimation(self) -> Estimation:
        from svy.estimation import Estimation

        return Estimation(self)

    @property
    def categorical(self) -> Categorical:
        from svy.categorical import Categorical

        return Categorical(self)

    @property
    def glm(self) -> GLM:
        from svy.regression import GLM

        return GLM(self)

    # ════════════════════════════════════════════════════════════════════════
    # CORE PROPERTIES (data, design, metadata)
    # ════════════════════════════════════════════════════════════════════════
    @property
    def data(self) -> DF:
        """Return a defensive copy to prevent external mutation."""
        local_data = self._data.clone()
        stratum = [self._internal_design["stratum"]] if self._internal_design["stratum"] else []
        psu = [self._internal_design["psu"]] if self._internal_design["psu"] else []
        ssu = [self._internal_design["ssu"]] if self._internal_design["ssu"] else []
        cols_to_drop = [c for c in stratum + psu + ssu if c in local_data.columns]
        if len(cols_to_drop) > 0:
            local_data = local_data.drop(cols_to_drop)
        return local_data

    @property
    def design(self) -> Design:
        """Return a defensive copy to avoid external mutation of internal design."""
        return copy.deepcopy(self._design)

    @property
    def rep_wgts(self) -> RepWeights | None:
        """Return a defensive copy to avoid external mutation of internal rep weights."""
        if self._design.rep_wgts is None:
            return None
        return copy.deepcopy(self._design.rep_wgts)

    @property
    def fpc(self) -> dict[Category, Number] | Number:
        return self._fpc

    @property
    def n_records(self) -> int:
        return cast(pl.DataFrame, self._data).shape[0]

    @property
    def n_columns(self) -> int:
        return cast(pl.DataFrame, self._data).shape[1]

    @property
    def n_strata(self) -> int | None:
        if self._design.stratum is None:
            return None
        return len(self.strata)

    @property
    def n_psus(self) -> int:
        if self._design.psu is None:
            return 0
        return len(self.psus)

    @property
    def deff_w(self) -> DomainScalarMap | Number:
        def deff_due_to_weighting(w: np.ndarray) -> Number:
            if w is None:
                raise ValueError("Sample weight is None")
            mean_w = np.mean(w)
            relvar_w = np.power(w - mean_w, 2) / mean_w**2
            return float(1 + np.mean(relvar_w))

        _dw = (
            cast(pl.DataFrame, self._data)
            if not isinstance(self._data, pl.LazyFrame)
            else cast(pl.DataFrame, self._data.collect())
        )
        if self._design.wgt is None:
            w = np.ones(_dw.shape[0])
        else:
            w = _dw[self._design.wgt].to_numpy()
        return deff_due_to_weighting(w)

    @property
    def strata(self):
        if self._design.stratum is None:
            return pl.DataFrame()
        else:
            strata = _colspec_to_list(self._design.stratum)
            return self._data.select(strata).unique().sort(by=strata)

    @property
    def psus(self):
        if self._design.psu is None:
            return pl.DataFrame()
        else:
            strata = _colspec_to_list(self._design.stratum)
            psus = _colspec_to_list(self._design.psu)
            strata_psus = strata + psus
            return self._data.select(strata_psus).unique().sort(by=strata_psus)

    @property
    def ssus(self):
        if self._design.ssu is None:
            return pl.DataFrame()
        else:
            strata = _colspec_to_list(self._design.stratum)
            psus = _colspec_to_list(self._design.psu)
            ssus = _colspec_to_list(self._design.ssu)
            strata_psus_ssus = strata + psus + ssus
            return self._data.select(strata_psus_ssus).unique().sort(by=strata_psus_ssus)

    @property
    def dtypes(self) -> dict[str, str]:
        """
        Column data types as simplified strings.

        Returns a dict mapping column names to type strings:
        - "int" for integers
        - "float" for floats
        - "str" for strings
        - "bool" for booleans
        - "datetime" for temporal types
        - "categorical" for categorical/enum types

        Examples
        --------
        >>> sample.dtypes
        {'age': 'int', 'income': 'float', 'name': 'str', 'employed': 'bool'}
        """
        import polars as pl

        type_map = {}
        for col, dtype in self._data.schema.items():
            base = dtype.base_type()
            if base.is_integer():
                type_map[col] = "int"
            elif base.is_float():
                type_map[col] = "float"
            elif base == pl.Boolean:
                type_map[col] = "bool"
            elif base == pl.String:
                type_map[col] = "str"
            elif base == pl.Categorical or base == pl.Enum:
                type_map[col] = "categorical"
            elif base.is_temporal():
                type_map[col] = "datetime"
            else:
                type_map[col] = str(dtype)
        return type_map

    @property
    def warnings(self) -> WarningStore:
        return self._warnings

    # ════════════════════════════════════════════════════════════════════════
    # METADATA ACCESS (MetadataStore)
    # ════════════════════════════════════════════════════════════════════════

    @property
    def meta(self) -> MetadataStore:
        """Access variable metadata registry."""
        return self._metadata

    @property
    def labels(self) -> dict[str, dict[Category, str]]:
        """
        Return value labels for all variables that have them.

        Returns a dict mapping variable name to its value labels dict.
        This is a convenience property for backward compatibility.
        """
        result: dict[str, dict[Category, str]] = {}
        for var in self._metadata:
            resolved = self._metadata.resolve_labels(var)
            if resolved.has_value_labels:
                result[var] = dict(resolved.value_labels)
        return result

    # ════════════════════════════════════════════════════════════════════════
    # METADATA CONVENIENCE METHODS
    # ════════════════════════════════════════════════════════════════════════

    def set_var_label(self, var: str, label: str) -> Self:
        """
        Set the variable label (question text) for a variable.

        Parameters
        ----------
        var : str
            Variable name.
        label : str
            The label text.

        Returns
        -------
        Self
            For method chaining.

        Examples
        --------
        >>> svy.set_var_label("q1", "How satisfied are you with our service?")
        """
        self._metadata.set_label(var, label)
        return self

    def set_var_labels(self, **labels: str) -> Self:
        """
        Set variable labels for multiple variables.

        Parameters
        ----------
        **labels : str
            Mapping of variable name to label text.

        Returns
        -------
        Self
            For method chaining.

        Examples
        --------
        >>> svy.set_var_labels(
        ...     q1="How satisfied are you?",
        ...     q2="Would you recommend us?",
        ...     age="What is your age?",
        ... )
        """
        self._metadata.set_labels(**labels)
        return self

    def set_value_labels(self, var: str, labels: dict[Category, str]) -> Self:
        """
        Set value labels (code → display text) for a variable.

        Parameters
        ----------
        var : str
            Variable name.
        labels : dict[Category, str]
            Mapping of codes to display labels.

        Returns
        -------
        Self
            For method chaining.

        Examples
        --------
        >>> svy.set_value_labels("gender", {1: "Male", 2: "Female", 3: "Other"})
        """
        self._metadata.set_value_labels(var, labels)
        return self

    def use_scheme(self, var: str, concept: str, locale: str | None = None) -> Self:
        """
        Link a variable to a label scheme in the catalog.

        Parameters
        ----------
        var : str
            Variable name.
        concept : str
            The concept identifier in the catalog (e.g., "agreement", "yes_no").
        locale : str | None
            Optional locale override.

        Returns
        -------
        Self
            For method chaining.

        Examples
        --------
        >>> svy.use_scheme("q1", "satisfaction", locale="en")
        """
        self._metadata.set_scheme(var, concept, locale)
        return self

    def set_missing(
        self,
        var: str,
        codes: Iterable[Category] | None = None,
        *,
        dont_know: Iterable[Category] | None = None,
        refused: Iterable[Category] | None = None,
        no_answer: Iterable[Category] | None = None,
        skipped: Iterable[Category] | None = None,
        not_applicable: Iterable[Category] | None = None,
        system: Iterable[Category] | None = None,
        structural: Iterable[Category] | None = None,
        na_is_missing: bool = True,
        nan_is_missing: bool = True,
    ) -> Self:
        """
        Define missing values for a variable.

        Can specify either simple codes or codes with semantic kinds.
        User-friendly parameter names are mapped to underlying mechanisms:

        - dont_know → DONT_KNOW (typically MNAR)
        - refused → REFUSED (typically MNAR)
        - no_answer → NO_ANSWER (ambiguous)
        - skipped → STRUCTURAL (design-driven, typically MAR)
        - not_applicable → STRUCTURAL (design-driven, typically MAR)
        - system → SYSTEM (typically MCAR)
        - structural → STRUCTURAL (design-driven, typically MAR)

        Parameters
        ----------
        var : str
            Variable name.
        codes : Iterable[Category] | None
            Simple missing codes (no kind attached).
        dont_know : Iterable[Category] | None
            Codes meaning "don't know".
        refused : Iterable[Category] | None
            Codes meaning "refused to answer".
        no_answer : Iterable[Category] | None
            Codes for no answer provided.
        skipped : Iterable[Category] | None
            Codes for skipped questions (routing/skip logic).
        not_applicable : Iterable[Category] | None
            Codes meaning "not applicable".
        system : Iterable[Category] | None
            System-generated missing codes.
        structural : Iterable[Category] | None
            Codes for values missing by study design.
        na_is_missing : bool
            Whether None is treated as missing.
        nan_is_missing : bool
            Whether NaN is treated as missing.

        Returns
        -------
        Self
            For method chaining.

        Examples
        --------
        >>> # Simple codes
        >>> svy.set_missing("q1", codes=[-99, -98])

        >>> # With semantic kinds
        >>> svy.set_missing(
        ...     "q1",
        ...     dont_know=[-99],
        ...     refused=[-98],
        ...     not_applicable=[-97],
        ... )
        """
        self._metadata.set_missing(
            var,
            codes,
            dont_know=dont_know,
            refused=refused,
            no_answer=no_answer,
            skipped=skipped,
            not_applicable=not_applicable,
            system=system,
            structural=structural,
            na_is_missing=na_is_missing,
            nan_is_missing=nan_is_missing,
        )
        return self

    def resolve_labels(self, var: str):
        """
        Get fully resolved labels for a variable.

        Parameters
        ----------
        var : str
            Variable name.

        Returns
        -------
        ResolvedLabels
            The resolved labels (ready for display).

        Examples
        --------
        >>> labels = svy.resolve_labels("q1")
        >>> labels.display(5)  # "Very satisfied"
        """
        return self._metadata.resolve_labels(var)

    # ════════════════════════════════════════════════════════════════════════
    # MUTATOR / ACCESSOR METHODS (schema, data, design)
    # ════════════════════════════════════════════════════════════════════════
    def set_type(self, col: str, mtype: MeasurementType) -> "Sample":
        """Set the measurement type for a column."""
        self._metadata.set_type(col, mtype)
        return self

    def set_categories(
        self, col: str, categories: Sequence, *, ordered: bool | None = None
    ) -> "Sample":
        """Set valid categories for a column."""
        self._metadata.set_categories(col, categories, ordered=ordered)
        return self

    def set_na_as_level(self, col: str, flag: bool = True) -> "Sample":
        """Set whether NA should be treated as a category level."""
        self._metadata.set_na_as_level(col, flag)
        return self

    def set_data(self, data: pl.DataFrame) -> Self:
        self._data = data
        if "svy_row_index" not in self._data.columns:
            self._data = self._data.with_row_index(name=SVY_ROW_INDEX)
        # Infer metadata for new columns (don't overwrite existing)
        self._metadata.infer_from_dataframe(self._data, overwrite=False)
        return self

    def update_data(self, data: pl.DataFrame) -> Self:
        self._data = data
        if "svy_row_index" not in self._data.columns:
            self._data = self._data.with_row_index(name=SVY_ROW_INDEX)
        self._metadata.align_to_dataframe(self._data)
        return self

    def set_design(self, design: Design) -> Self:
        """Replace the entire Design."""
        self._design = design
        return self

    def update_design(self, **kwargs) -> Self:
        """Update selected Design fields in place."""
        self._design = self._design.update(**kwargs)
        return self

    # ════════════════════════════════════════════════════════════════════════
    # USE (public) to facilitate experimentation
    # ════════════════════════════════════════════════════════════════════════

    def use_weight(self, wgt: str) -> "Sample":
        """
        Return a new Sample instance using the specified weight column.

        This method creates a lightweight copy (view) of the Sample. The underlying
        data is shared, but the active design is updated to use the new weight.
        This is useful for sensitivity analysis or comparing different weights
        without mutating the original sample.

        Args:
            weight_column: The name of the column to use as the new weight.

        Returns:
            A new Sample instance with the updated weight.

        Raises:
            ValueError: If the weight column is not found in the data.
            TypeError: If the weight column is not numeric.
        """
        # 1. Validate the new weight column
        if wgt not in self._data.columns:
            raise ValueError(f"Weight column {wgt!r} not found in data.")

        # Check if it is numeric (using the schema or dtypes)
        dtype = self._data.schema[wgt]
        if dtype not in INTEGER_DTYPES and dtype not in FLOAT_DTYPES:
            raise TypeError(f"Weight column {wgt!r} must be numeric; got {dtype}.")

        # 2. Create a shallow copy of the current sample
        # We use copy.copy() to duplicate the Sample shell (slots/attributes)
        # without duplicating the heavy _data DataFrame (which Polars handles efficiently).
        new_sample = copy.copy(self)

        # 3. Update the design in the new sample
        # We must deepcopy the design so we don't mutate the original sample's design
        if new_sample._design is not None:
            new_sample._design = new_sample._design.update(wgt=wgt)
        else:
            # If no design existed, create a minimal one with the weight
            new_sample._design = Design(row_index=SVY_ROW_INDEX, wgt=wgt)

        # 4. Return the new instance
        return new_sample

    # ════════════════════════════════════════════════════════════════════════
    # CLONE (public)
    # ════════════════════════════════════════════════════════════════════════

    def clone(
        self: Self,
        *,
        data: pl.DataFrame | None | _MissingType = _MISSING,
        design: Design | None | _MissingType = _MISSING,
        rep_wgts: RepWeights | None | _MissingType = _MISSING,
        catalog: LabellingCatalog | None | _MissingType = _MISSING,
    ) -> Sample:
        if data is _MISSING:
            src_data: pl.DataFrame | None = (
                cast(pl.DataFrame, self._data)
                if not isinstance(self._data, pl.LazyFrame)
                else cast(pl.DataFrame, self._data.collect())
            )
        else:
            src_data = cast(pl.DataFrame | None, data)

        if design is _MISSING:
            src_design: Design | None = self._design
        else:
            src_design = cast(Design | None, design)

        if catalog is _MISSING:
            src_catalog = self._metadata.catalog
        else:
            src_catalog = cast("LabellingCatalog | None", catalog)

        _fallback = (
            cast(pl.DataFrame, self._data)
            if not isinstance(self._data, pl.LazyFrame)
            else cast(pl.DataFrame, self._data.collect())
        )
        new_data: pl.DataFrame = (src_data if src_data is not None else _fallback).clone()
        new_design: Design | None = copy.deepcopy(src_design) if src_design is not None else None

        s = Sample(new_data, new_design, catalog=src_catalog)

        if rep_wgts is not _MISSING:
            s._design = s._design.update(rep_wgts=rep_wgts)  # type: ignore[arg-type]

        # Copy metadata from original
        for var in self._metadata:
            meta = self._metadata.get(var)
            if meta is not None and var in s._data.columns:
                s._metadata.set(var, meta)

        s._check_rep_wgts_against_df(s._design.rep_wgts)
        s._fpc = copy.deepcopy(self._fpc)
        return s

    # ════════════════════════════════════════════════════════════════════════
    # DISPLAY UTILITIES (data/records)
    # ════════════════════════════════════════════════════════════════════════
    def show_data(
        self,
        columns: ColumnsArg = None,
        *,
        where: WhereArg = None,
        n: int | None = 5,
        offset: int = 0,
        order_by: OrderByArg = None,
        order_type: Literal["ascending", "descending", "random"] = "ascending",
        nulls_last: bool = False,
        rstate: object | None = None,
    ) -> pl.DataFrame:
        """
        Return a slice of the sample data as a plain Polars DataFrame.

        Parameters
        ----------
        columns : str | Sequence[str] | None
            Columns to include. None returns all columns.
        where : filter expression, optional
            Row filter: dict, list of expressions, Polars Expr, or None.
            When provided, only matching rows are considered before
            ordering and slicing.
        n : int | None
            Maximum number of rows to return. None returns all rows.
        offset : int
            Number of rows to skip before returning results.
        order_by : str | Sequence[str] | None
            Column(s) to sort by before slicing.
        order_type : "ascending" | "descending" | "random"
            Controls output order:

            * ``"ascending"`` / ``"descending"`` — sort by ``order_by``.
              Ignored when ``order_by`` is None.
            * ``"random"`` without ``order_by`` — full random shuffle.
            * ``"random"`` with ``order_by`` — sort by ``order_by``, then
              shuffle within each group of equal values.
        nulls_last : bool
            Place null values at the end when sorting. Ignored for "random".
        rstate : random state
            Seed for reproducibility when ``order_type="random"``.

        Returns
        -------
        pl.DataFrame

        Examples
        --------
        >>> sample.show_data(columns=["hid", "tot_exp"], n=10)
        >>> sample.show_data(where=svy.col("urbrur") == "Urban", n=5)
        >>> sample.show_data(order_by="tot_exp", order_type="descending", n=10)
        >>> sample.show_data(order_type="random", n=7, rstate=42)
        """
        with log_step(
            log,
            "show_data",
            n=n,
            offset=offset,
            order_by=order_by,
            order_type=order_type,
            nulls_last=nulls_last,
            columns=columns if isinstance(columns, str) else (len(columns) if columns else None),
        ):
            local_data = self.data

            # ── Validate n and offset ──────────────────────────────────────
            if n is not None and n < 0:
                raise DimensionError(
                    title="Invalid row count",
                    detail="Parameter 'n' must be a non-negative integer.",
                    code="INVALID_N",
                    where="sample.show_data",
                    param="n",
                    got=n,
                )
            if offset < 0:
                raise DimensionError(
                    title="Invalid offset",
                    detail="Offset must be non-negative.",
                    code="INVALID_OFFSET",
                    where="sample.show_data",
                    param="offset",
                    got=offset,
                )

            # ── Validate order_type ────────────────────────────────────────
            allowed_order = ("ascending", "descending", "random")
            if order_type not in allowed_order:
                raise MethodError(
                    title="Invalid order type",
                    detail=f"'order_type' must be one of {allowed_order}.",
                    code="INVALID_ORDER_TYPE",
                    where="sample.show_data",
                    param="order_type",
                    expected=allowed_order,
                    got=order_type,
                )

            # ── Build filter predicate ─────────────────────────────────────
            predicate_pl: pl.Expr | None = None
            try:
                if where is None:
                    predicate_pl = None
                elif isinstance(where, Mapping):
                    preds = []
                    for k, v in where.items():
                        if k not in local_data.columns:
                            raise DimensionError(
                                title="Filter column not found",
                                detail=f"Column {k!r} not in data.",
                                code="MISSING_FILTER_COLUMN",
                                where="sample.show_data",
                                got=k,
                            )
                        if isinstance(v, (list, tuple, set)) and not isinstance(v, (str, bytes)):
                            if len(v) == 0:
                                raise MethodError(
                                    title="Empty membership filter",
                                    detail=f"Value for {k!r} is empty.",
                                    code="EMPTY_IN_VALUES",
                                    where="sample.show_data",
                                    param=str(k),
                                )
                            preds.append(pl.col(k).is_in(list(v)))
                        else:
                            preds.append(pl.col(k) == v)
                    if preds:
                        acc = preds[0]
                        for p in preds[1:]:
                            acc = acc & p
                        predicate_pl = acc
                elif isinstance(where, (list, tuple)) and not isinstance(where, (str, bytes)):
                    if where:
                        compiled = []
                        for e in where:
                            if hasattr(e, "_e"):
                                compiled.append(to_polars_expr(e))
                            elif isinstance(e, pl.Expr):
                                compiled.append(e)
                            else:
                                raise MethodError(
                                    title="Unsupported filter element",
                                    detail=f"Got {type(e)}.",
                                    code="UNSUPPORTED_FILTER_EXPR",
                                    where="sample.show_data",
                                )
                        acc2 = compiled[0]
                        for p in compiled[1:]:
                            acc2 = acc2 & p
                        predicate_pl = acc2
                elif hasattr(where, "_e") or isinstance(where, pl.Expr):
                    predicate_pl = to_polars_expr(where) if hasattr(where, "_e") else where
                else:
                    raise MethodError(
                        title="Unsupported filter",
                        detail=f"Type {type(where)} not supported.",
                        code="UNSUPPORTED_WHERE",
                        where="sample.show_data",
                    )
            except SvyError:
                raise
            except Exception as ex:
                raise MethodError(
                    title="Failed to build filter",
                    detail=str(ex),
                    code="FILTER_BUILD_FAILED",
                    where="sample.show_data",
                ) from ex

            # ── Apply filter ───────────────────────────────────────────────
            out: pl.DataFrame = local_data
            if predicate_pl is not None:
                out = out.filter(predicate_pl)

            # ── Select columns ─────────────────────────────────────────────
            try:
                want_cols = list(_normalize_columns_arg(data=out, columns=columns))
            except ValueError as ex:
                raise DimensionError(
                    title="Column(s) not found",
                    detail=str(ex),
                    code="MISSING_COLUMNS",
                    where="sample.show_data",
                    param="columns",
                    got=columns,
                ) from ex

            missing = [c for c in want_cols if c not in out.columns]
            if missing:
                raise DimensionError(
                    title="Column(s) not found",
                    detail=f"Missing: {', '.join(map(repr, missing))}.",
                    code="MISSING_COLUMNS",
                    where="sample.show_data",
                    got=missing,
                )

            out = cast(pl.DataFrame, out.select(want_cols))

            # ── Validate order_by columns ──────────────────────────────────
            sort_cols: list[str] | None = None
            if order_by is not None:
                sort_cols = [order_by] if isinstance(order_by, str) else list(order_by)
                miss_sort = [c for c in sort_cols if c not in out.columns]
                if miss_sort:
                    raise DimensionError(
                        title="Sort column(s) not found",
                        detail=f"Missing in selection: {', '.join(map(repr, miss_sort))}.",
                        code="MISSING_SORT_COLUMNS",
                        where="sample.show_data",
                        got=miss_sort,
                    )

            if n is None:
                n = out.height

            seed = seed_from_random_state(rstate)  # type: ignore[arg-type]

            # ── Apply ordering ─────────────────────────────────────────────
            if order_type == "random":
                if sort_cols is not None:
                    # Sort by order_by, then shuffle within each group
                    out = (
                        out.with_columns(
                            pl.arange(0, pl.len())
                            .shuffle(seed=seed)
                            .over(sort_cols)
                            .alias("_shuffle_col")
                        )
                        .sort(*sort_cols, "_shuffle_col")
                        .drop("_shuffle_col")
                    )
                else:
                    out = out.sample(fraction=1.0, shuffle=True, seed=seed)
            elif sort_cols is not None:
                out = out.sort(
                    by=sort_cols,
                    descending=(order_type == "descending"),
                    nulls_last=nulls_last,
                )

            # ── Apply offset and limit ─────────────────────────────────────
            try:
                if offset:
                    out = out.slice(offset, n)
                else:
                    out = out.head(n)
            except Exception as e:
                msg = str(e).lower()
                if "with_replacement=false" in msg or "not enough rows" in msg:
                    raise DimensionError(
                        title="Slicing failed",
                        detail=str(e),
                        code="SLICE_FAILED",
                        where="sample.show_data",
                        param="n",
                        got=n,
                        hint=("Requested more rows than available after filtering and offset."),
                    ) from e
                raise

            return cast(pl.DataFrame, out)

    # ════════════════════════════════════════════════════════════════════════
    # UI HELPERS (printing/describe hygiene)
    # ════════════════════════════════════════════════════════════════════════
    def _hidden_columns_for_ui(self) -> set[str]:
        """Columns that should never appear in user-facing prints (describe, etc.)."""
        hidden: set[str] = set()

        # Synthetic row index we add internally
        if SVY_ROW_INDEX in cast(pl.DataFrame, self._data).columns:
            hidden.add(SVY_ROW_INDEX)

        # Concatenated design helpers (created in __init__)
        idict = getattr(self, "_internal_design", {}) or {}
        for key in ("stratum", "psu", "ssu"):
            col = idict.get(key)
            if isinstance(col, str) and col and col in self._data.columns:
                hidden.add(col)

        # (Optional) if you ever add more auto-concatenated design bits:
        suf = idict.get("suffix")
        if suf:
            for c in self._data.columns:
                if c.endswith(suf) and c.startswith(("stratum", "psu", "ssu", "by")):
                    hidden.add(c)

        return hidden

    # ════════════════════════════════════════════════════════════════════════
    # DESCRIPTIVE STATISTICS (schema-aware)
    # ════════════════════════════════════════════════════════════════════════
    def describe(
        self,
        columns: Sequence[str] | None = None,
        *,
        weighted: bool = False,
        weight_col: str | None = None,
        drop_nulls: bool = True,
        top_k: int = 10,
        percentiles: Sequence[float] = (0.05, 0.25, 0.50, 0.75, 0.95),
    ) -> DescribeResult:
        """
        Compute a typed description of columns based on the current Schema.

        Internal, concatenated design columns (e.g., stratum/psu/ssu with the internal
        suffix) and the synthetic row index are excluded from the description.
        """
        # Resolve default weight column from the active design if requested
        if weighted and weight_col is None:
            w = getattr(self._design, "wgt", None)
            weight_col = w if isinstance(w, str) else None

        # Build the hidden/internal columns set using the UI helper
        hidden = self._hidden_columns_for_ui()

        # Resolve the final column list, excluding hidden/internal ones
        if columns is None:
            cols = [c for c in self._data.columns if c not in hidden]
        else:
            cols = [c for c in columns if c in self._data.columns and c not in hidden]

        if not cols:
            raise ValueError(
                "No visible columns to describe after excluding internal design columns."
            )

        # Keep schema aligned just in case data changed
        _desc_data = (
            cast(pl.DataFrame, self._data)
            if not isinstance(self._data, pl.LazyFrame)
            else cast(pl.DataFrame, self._data.collect())
        )
        self._metadata.infer_from_dataframe(_desc_data, overwrite=False)

        return run_describe(
            df=_desc_data,
            metadata=self._metadata,
            columns=cols,
            weighted=weighted,
            weight_col=weight_col,
            drop_nulls=drop_nulls,
            top_k=top_k,
            percentiles=percentiles,
        )

    # ════════════════════════════════════════════════════════════════════════
    # WARNING API (record non-fatal issues)
    # ════════════════════════════════════════════════════════════════════════
    def warn(
        self,
        *,
        code: WarnCode | str,
        title: str,
        detail: str,
        where: str,
        level: Severity = Severity.WARNING,
        param: str | None = None,
        expected: Any = None,
        got: Any = None,
        hint: str | None = None,
        docs_url: str | None = None,
        extra: dict[str, Any] | None = None,
        var: str | None = None,
        rows: Sequence[int] | None = None,
    ) -> SvyWarning:
        w = SvyWarning(
            code=code,
            title=title,
            detail=detail,
            where=where,
            level=level,
            param=param,
            expected=expected,
            got=got,
            hint=hint,
            docs_url=docs_url,
            extra=extra,
            var=var,
            rows=None if rows is None else tuple(rows),
        )
        self._warnings.add(w)
        return w
