# tests/test_data/domain_nulls_test_data.py
"""
Generate domain_nulls_20260722.csv: a two-stratum cluster sample with a
standard skip pattern (income is null for every non-employed respondent),
plus two missing-at-random null incomes among the employed.

Key property: cluster 104 (stratum A) is entirely non-employed, so its
income values are all null. Physically dropping null-income rows deletes
that whole PSU from the design (7 PSUs instead of 8, df 5 instead of 6),
which is the subpopulation-estimation error that domain weight-zeroing
exists to avoid. Reference values for the correct behavior come from R
survey::svymean on subset()/na.rm designs (see test_domain_nulls.py).

Run from packages/svy:
    uv run python tests/test_data/domain_nulls_test_data.py
"""

from pathlib import Path

import numpy as np
import polars as pl


def main() -> None:
    rng = np.random.default_rng(20260722)

    rows = []
    clusters = {
        "A": [101, 102, 103, 104],
        "B": [201, 202, 203, 204],
    }
    for stratum, cluster_ids in clusters.items():
        for cluster in cluster_ids:
            for unit in range(6):
                if cluster == 104:
                    employed = 0  # whole PSU out of the employed domain
                else:
                    employed = int(rng.random() < 0.7)
                base = 30_000 if stratum == "A" else 42_000
                income = (
                    round(float(base + rng.normal(0.0, 8_000.0) + 2_500.0 * unit), 2)
                    if employed
                    else None
                )
                rows.append(
                    {
                        "stratum": stratum,
                        "cluster": cluster,
                        "unit": unit + 1,
                        "employed": employed,
                        "income": income,
                        "wgt": round(float(rng.uniform(50.0, 150.0)), 2),
                    }
                )

    df = pl.DataFrame(rows)

    # Two missing-at-random incomes among the employed, in different clusters
    employed_idx = df.with_row_index().filter(pl.col("employed") == 1)["index"].to_list()
    mar = {employed_idx[3], employed_idx[17]}
    df = (
        df.with_row_index()
        .with_columns(
            pl.when(pl.col("index").is_in(list(mar)))
            .then(None)
            .otherwise(pl.col("income"))
            .alias("income")
        )
        .drop("index")
    )

    out = Path(__file__).parent / "domain_nulls_20260722.csv"
    df.write_csv(out)
    n_employed = df.filter(pl.col("employed") == 1).height
    n_null = df["income"].null_count()
    print(f"wrote {out.name}: {df.height} rows, {n_employed} employed, {n_null} null incomes")


if __name__ == "__main__":
    main()
