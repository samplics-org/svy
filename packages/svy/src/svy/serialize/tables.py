# src/svy/serialize/tables.py
"""
The table of a serialized result: the frame its live result's ``to_polars()``
returns, built by the same code from the payload's fields.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

import polars as pl

from svy.categorical.table import table_frame
from svy.categorical.ttest import ttest_one_group_frame, ttest_two_groups_frame
from svy.errors.serialization_errors import SerializationError
from svy.estimation.estimate import concat_estimate_frames, estimate_frame
from svy.regression.glm import glm_frame
from svy.regression.prediction import glm_pred_frame
from svy.serialize.structs import (
    EstimateData,
    EstimateListData,
    GLMFitData,
    GLMPredData,
    ResultData,
    TableData,
    TTestOneGroupData,
    TTestTwoGroupsData,
)


def _indexed(df: pl.DataFrame, row_index: str | None) -> pl.DataFrame:
    return df.with_row_index(row_index) if row_index else df


def _estimate(d: EstimateData, *, row_index: str | None = None, tidy: bool = True) -> pl.DataFrame:
    return estimate_frame(
        d.estimates, param=d.param, as_factor=d.as_factor, tidy=tidy, row_index=row_index
    )


def _estimate_list(d: EstimateListData, *, row_index: str | None = None) -> pl.DataFrame:
    frames, offset = [], 0
    for m in d.estimates:
        f = _estimate(m, row_index=row_index)
        if row_index and not f.is_empty():
            f = f.with_columns(pl.col(row_index) + offset)
        frames.append(f)
        offset += len(m.estimates)
    return concat_estimate_frames(frames)


def _ttest_one_group(
    d: TTestOneGroupData,
    *,
    row_index: str | None = None,
    component: Literal["test", "estimates"] = "test",
    tidy: bool = True,
) -> pl.DataFrame:
    return _indexed(ttest_one_group_frame(d, component, tidy=tidy), row_index)


def _ttest_two_groups(
    d: TTestTwoGroupsData,
    *,
    row_index: str | None = None,
    component: Literal["test", "estimates"] = "test",
    tidy: bool = True,
) -> pl.DataFrame:
    return _indexed(ttest_two_groups_frame(d, component, tidy=tidy), row_index)


def _table(d: TableData, *, row_index: str | None = None, tidy: bool = True) -> pl.DataFrame:
    return _indexed(table_frame(d, tidy=tidy), row_index)


def _glm_fit(
    d: GLMFitData, *, row_index: str | None = None, exponentiate: bool = False
) -> pl.DataFrame:
    return _indexed(glm_frame(d, exponentiate=exponentiate), row_index)


def _glm_pred(d: GLMPredData, *, row_index: str | None = None) -> pl.DataFrame:
    return _indexed(glm_pred_frame(d), row_index)


_TABLES: dict[type, Callable[..., pl.DataFrame]] = {
    EstimateData: _estimate,
    EstimateListData: _estimate_list,
    TTestOneGroupData: _ttest_one_group,
    TTestTwoGroupsData: _ttest_two_groups,
    TableData: _table,
    GLMFitData: _glm_fit,
    GLMPredData: _glm_pred,
}


def to_polars(data: ResultData, *, row_index: str | None = None, **options: Any) -> pl.DataFrame:
    """
    The table of a serialized result, as its live result's ``to_polars()`` returns it.

    Parameters
    ----------
    data
        A payload from ``serialize()`` or ``from_json()``.
    row_index
        Name of a first column (``UInt32``) holding each table row's position
        in the payload's rows: ``estimates`` for an estimate, a table and a
        t-test's ``component="estimates"``, ``diff`` for a t-test's
        ``component="test"``, ``coefs`` for a GLM fit, and the prediction
        arrays for a GLM prediction. For an estimate list it counts through
        the members' estimates in order. Estimate tables are sorted for
        display, so this column is the way back to a payload row.
    **options
        The live ``to_polars()`` options that need nothing beyond the payload:
        ``tidy`` (estimate, table, t-tests), ``component`` (t-tests) and
        ``exponentiate`` (GLM fit). Payloads carry no metadata, so levels
        and column names are the raw codes and variable names, as a live
        result gives with ``use_labels=False``.

    Raises
    ------
    SerializationError
        If the payload's kind has no table (code ``PAYLOAD_NO_TABLE``).
    """
    builder = _TABLES.get(type(data))
    if builder is None:
        raise SerializationError.no_table(
            got_type=type(data).__name__, supported=[c.__name__ for c in _TABLES]
        )
    return builder(data, row_index=row_index, **options)
