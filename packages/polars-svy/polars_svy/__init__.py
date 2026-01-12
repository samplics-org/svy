# polars_svy/__init__.py
"""Survey statistics extension for Polars."""

import polars as pl

from polars_svy.taylor import (
    taylor_mean,
    taylor_total,
    taylor_ratio,
    taylor_prop,
)
from polars_svy.replication import (
    replicate_mean,
    replicate_total,
    replicate_ratio,
    replicate_prop,
)

__version__ = "0.1.0"


# Estimator classes for fluent API
class MeanEstimator:
    def __init__(self, col: str):
        self.col = col


class TotalEstimator:
    def __init__(self, col: str):
        self.col = col


class RatioEstimator:
    def __init__(self, numerator: str, denominator: str):
        self.numerator = numerator
        self.denominator = denominator


class PropEstimator:
    def __init__(self, col: str):
        self.col = col

    def __repr__(self) -> str:
        return f"PropEstimator({self.col!r})"


# Convenience functions
def mean(col: str) -> MeanEstimator:
    """Survey-weighted mean estimator."""
    return MeanEstimator(col)


def total(col: str) -> TotalEstimator:
    """Survey-weighted total estimator."""
    return TotalEstimator(col)


def ratio(numerator: str, denominator: str) -> RatioEstimator:
    """Survey-weighted ratio estimator."""
    return RatioEstimator(numerator, denominator)


def prop(col: str) -> PropEstimator:
    """Proportion estimator for the specified column."""
    return PropEstimator(col)


# Survey design query builder
class SurveyQuery:
    def __init__(
        self,
        df: pl.DataFrame,
        weight: str,
        strata: str | None,
        psu: str | None,
        fpc: str | None,
    ):
        self.df = df
        self.weight = weight
        self.strata = strata
        self.psu = psu
        self.fpc = fpc
        self._by_col: str | None = None

    def _prepare_df(self, value_cols: list[str]) -> pl.DataFrame:
        """Cast columns to appropriate types for Rust compatibility."""
        casts = []

        float_cols = value_cols + [self.weight]
        if self.fpc:
            float_cols.append(self.fpc)

        for col in float_cols:
            if col in self.df.columns:
                dtype = self.df[col].dtype
                if dtype != pl.Float64 and dtype.is_numeric():
                    casts.append(pl.col(col).cast(pl.Float64))

        str_cols = []
        if self.strata:
            str_cols.append(self.strata)
        if self.psu:
            str_cols.append(self.psu)
        if self._by_col:
            str_cols.append(self._by_col)

        for col in str_cols:
            if col in self.df.columns:
                dtype = self.df[col].dtype
                if dtype != pl.String and dtype != pl.Utf8:
                    casts.append(pl.col(col).cast(pl.String))

        return self.df.with_columns(casts) if casts else self.df

    def group_by(self, col: str) -> "SurveyQuery":
        self._by_col = col
        return self

    def agg(self, **estimators) -> pl.DataFrame:
        """Execute aggregation with the specified estimators."""
        results = []

        for name, estimator in estimators.items():
            if isinstance(estimator, MeanEstimator):
                df_prepared = self._prepare_df([estimator.col])
                result = taylor_mean(
                    df_prepared,
                    value_col=estimator.col,
                    weight_col=self.weight,
                    strata_col=self.strata,
                    psu_col=self.psu,
                    fpc_col=self.fpc,
                    by_col=self._by_col,
                )
            elif isinstance(estimator, TotalEstimator):
                df_prepared = self._prepare_df([estimator.col])
                result = taylor_total(
                    df_prepared,
                    value_col=estimator.col,
                    weight_col=self.weight,
                    strata_col=self.strata,
                    psu_col=self.psu,
                    fpc_col=self.fpc,
                    by_col=self._by_col,
                )
            elif isinstance(estimator, RatioEstimator):
                df_prepared = self._prepare_df([estimator.numerator, estimator.denominator])
                result = taylor_ratio(
                    df_prepared,
                    numerator_col=estimator.numerator,
                    denominator_col=estimator.denominator,
                    weight_col=self.weight,
                    strata_col=self.strata,
                    psu_col=self.psu,
                    fpc_col=self.fpc,
                    by_col=self._by_col,
                )
            elif isinstance(estimator, PropEstimator):
                df_prepared = self._prepare_df([estimator.col])
                result = taylor_prop(
                    df_prepared,
                    value_col=estimator.col,
                    weight_col=self.weight,
                    strata_col=self.strata,
                    psu_col=self.psu,
                    fpc_col=self.fpc,
                    by_col=self._by_col,
                )
            else:
                raise TypeError(f"Unknown estimator type: {type(estimator)}")

            result = result.rename({"est": name})
            results.append(result)

        return results[0] if len(results) == 1 else results[0]


@pl.api.register_dataframe_namespace("svy")
class SurveyNamespace:
    def __init__(self, df: pl.DataFrame):
        self._df = df

    def design(
        self,
        weight: str,
        strata: str | None = None,
        psu: str | None = None,
        fpc: str | None = None,
    ) -> SurveyQuery:
        """Specify survey design parameters."""
        return SurveyQuery(self._df, weight, strata, psu, fpc)


__all__ = [
    # Convenience functions
    "mean",
    "total",
    "ratio",
    "prop",
    # Taylor functions
    "taylor_mean",
    "taylor_total",
    "taylor_ratio",
    "taylor_prop",
    # Replication functions
    "replicate_mean",
    "replicate_total",
    "replicate_ratio",
    "replicate_prop",
    # Classes
    "SurveyQuery",
    "SurveyNamespace",
]
