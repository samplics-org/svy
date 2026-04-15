# src/svy/evaluation/compare.py
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import polars as pl

from rich.table import Table as RTable
from rich.text import Text

from svy.regression.glm import GLM
from svy.regression.glm_pred import GLMPred  # type: ignore[unresolved-import]
from svy.utils.formats import _fmt_fixed


def compare(
    objects: Sequence[Any],
    names: Sequence[str] | None = None,
    **kwargs,
) -> Any:
    """
    Universal comparison function.
    Dispatches to specific comparison logic based on input types.

    Parameters
    ----------
    objects : Sequence[Any]
        A list of objects to compare (GLMs, Predictions, etc).
    names : Sequence[str] | None
        Optional labels for the columns/rows.
    **kwargs :
        Additional arguments passed to the specific comparison function
        (e.g. metrics=["aic", "bic"] for models, metrics=["mse"] for preds).
    """
    if not objects:
        raise ValueError("No objects provided to compare.")

    # 1. Identify Type of the first element
    first = objects[0]

    # 2. Dispatch: GLM Models
    if isinstance(first, GLM):
        # Validation: Ensure all are GLMs
        if not all(isinstance(x, GLM) for x in objects):
            raise TypeError("Cannot compare mixed types. All objects must be GLM instances.")
        return compare_models(objects, names=names, **kwargs)

    # 3. Dispatch: Predictions (GLMPred class)
    elif isinstance(first, GLMPred):
        if not all(isinstance(x, GLMPred) for x in objects):
            raise TypeError("All objects must be GLMPred instances.")
        return compare_predictions(objects, names=names, **kwargs)

    # 4. Dispatch: Raw Arrays (e.g. comparing y_true vs multiple y_pred)
    elif isinstance(first, (np.ndarray, pl.Series, list)):
        return compare_arrays(objects, names=names, **kwargs)

    else:
        obj_type = type(first).__name__
        raise NotImplementedError(f"No comparison logic implemented for type '{obj_type}'")


# =============================================================================
# Model Comparison Implementation
# =============================================================================


class ModelComparison:
    """Container for model comparison results with publication-quality printing."""

    def __init__(self, coefs: pl.DataFrame, stats: pl.DataFrame, names: list[str]):
        self.coefs = coefs
        self.stats = stats
        self.names = names

    def __repr__(self) -> str:
        return f"<ModelComparison: {len(self.names)} models>"

    def __rich_console__(self, console, options):
        # Create a "Stata/Academic-style" table
        tbl = RTable(show_header=True, header_style="bold magenta", box=None, padding=(0, 2))

        # Columns: Term, Model 1, Model 2...
        tbl.add_column("Term", style="bold")
        for name in self.names:
            tbl.add_column(name, justify="center")

        # 1. Coefficients
        # We iterate through Polars rows
        for row in self.coefs.iter_rows(named=True):
            cells = [row["term"]]

            for name in self.names:
                est = row[f"{name}_est"]
                se = row[f"{name}_se"]

                if est is None:
                    cells.append("-")
                else:
                    # Format:
                    #  0.123
                    #  (0.05)
                    est_str = _fmt_fixed(est)
                    se_str = f"({_fmt_fixed(se)})"

                    # Combine into one cell with newline, dim the SE
                    cells.append(Text.assemble(est_str, "\n", (se_str, "dim")))

            tbl.add_row(*cells)

        # 2. Divider (Visual separation between coefs and stats)
        tbl.add_section()

        # 3. Statistics
        for row in self.stats.iter_rows(named=True):
            stat_label = row["stat"].replace("_", " ").title()
            cells = [Text(stat_label, style="italic")]

            for name in self.names:
                val = row[name]
                if val is None:
                    cells.append(Text("-"))
                elif isinstance(val, int) or (isinstance(val, float) and val.is_integer()):
                    cells.append(Text(str(int(val))))
                else:
                    cells.append(Text(_fmt_fixed(val)))
            tbl.add_row(*cells)

        yield tbl


def compare_models(
    models: Sequence[GLM],
    names: Sequence[str] | None,
    metrics: Sequence[str] = ("n", "aic", "bic", "r_squared_adj"),
) -> ModelComparison:
    """
    Compare multiple fitted GLM models side-by-side.
    """
    # 1. Resolve Names
    if names is None:
        names = [f"Model {i + 1}" for i in range(len(models))]

    if len(names) != len(models):
        raise ValueError(
            f"Length of 'names' ({len(names)}) must match length of 'models' ({len(models)})."
        )

    # 2. Extract Coefficients
    # We want to align coefficients by term name across all models
    term_map: dict[str, dict[str, tuple[float, float]]] = {}
    all_terms_order = []

    for name, model in zip(names, models):
        if model.fitted is None:
            continue

        for c in model.coefs:
            if c.term not in term_map:
                term_map[c.term] = {}
                all_terms_order.append(c.term)

            term_map[c.term][name] = (c.est, c.se)

    # 3. Build Polars DataFrame for Coefs
    # Structure: [term, m1_est, m1_se, m2_est, m2_se, ...]
    data = {"term": all_terms_order}
    for name in names:
        ests = []
        ses = []
        for term in all_terms_order:
            val = term_map[term].get(name)
            if val:
                ests.append(val[0])
                ses.append(val[1])
            else:
                ests.append(None)
                ses.append(None)

        data[f"{name}_est"] = ests
        data[f"{name}_se"] = ses

    df_coefs = pl.DataFrame(data)

    # 4. Extract Model-Level Stats
    stats_data = []
    for m in metrics:
        row: dict[str, Any] = {"stat": m}
        for name, model in zip(names, models):
            if model.fitted is None:
                row[name] = None
                continue

            # Use getattr to fetch stats dynamically from GLMStats struct
            # GLM.stats property returns GLMStats object
            val = getattr(model.stats, m, None)
            row[name] = val
        stats_data.append(row)

    df_stats = pl.DataFrame(stats_data)

    return ModelComparison(df_coefs, df_stats, list(names))


# =============================================================================
# Other Implementations (Stubs)
# =============================================================================


def compare_predictions(
    preds: Sequence[GLMPred], names: Sequence[str] | None, metrics: Sequence[str] = ("rmse", "mae")
):
    # TODO: Implement prediction comparison table
    print(f"Comparing {len(preds)} Predictions (Not implemented yet)...")


def compare_arrays(arrays: Sequence[Any], names: Sequence[str] | None, **kwargs):
    # TODO: Implement raw array comparison
    pass
