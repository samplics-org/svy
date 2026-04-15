import logging

import numpy as np
import polars as pl


log = logging.getLogger(__name__)


def _remove_nans(data: pl.DataFrame, cols_null: list[str | None]) -> pl.DataFrame:
    cols_null2 = [var for var in cols_null if var is not None]
    # cols_nan2 = [var for var in cols_nan if var is not None]
    return data.fill_nan(None).drop_nulls(subset=cols_null2)  # .drop_nans(subset=cols_nan2)


def _get_single_psu_strata(data: pl.DataFrame, stratum: str, psu: str) -> np.ndarray:
    df = (
        data.select([stratum, psu])
        .group_by(stratum)
        .agg(pl.col(psu).count())
        .filter(pl.col(psu) == 1)
    )

    if df.shape[0] == 0:
        return df.to_numpy()
    else:
        return df[stratum].to_numpy()
