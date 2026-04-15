# tests/data/svy_test_data_jackknife.py
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl


def generate_and_save_jackknife():
    # 1. Lock the Bit Generator (Same Seed as previous tests)
    bg = np.random.PCG64(seed=99)
    rng = np.random.Generator(bg)

    N = 200

    # 2. Design: 4 Strata, 2 PSUs per stratum
    strata = np.repeat(["S1", "S2", "S3", "S4"], N // 4)
    psu_local = np.tile(np.repeat([1, 2], N // 8), 4)
    # Unique PSU ID: S1-1, S1-2, etc.
    psu_global = [f"{s}-{p}" for s, p in zip(strata, psu_local)]

    # 3. Weights
    w_base = rng.uniform(0.5, 1.5, N)
    w_psu = np.where(psu_local == 1, 1.5, 1.0)
    weights = w_base * w_psu

    # 4. Variables (Standard logic)
    inc_base = rng.normal(50000, 10000, N)
    inc_shift = np.array([0 if s in ("S1", "S2") else 10000 for s in strata])
    income = np.abs(inc_base + inc_shift)
    income[0] = None

    sex = rng.choice(["Male", "Female"], size=N)
    sex[1] = None

    educ = rng.choice(["Low", "Med", "High"], size=N, p=[0.3, 0.5, 0.2])

    df = pl.DataFrame(
        {
            "id": np.arange(N),
            "stratum": strata,
            "psu": psu_global,
            "weight": weights,
            "income": income,
            "sex": sex,
            "educ": educ,
        }
    )

    # 5. Generate Jackknife (JKn) Replicates
    # For Stratified JKn:
    # One replicate per PSU.
    # In replicate 'r' (dropping PSU 'j' in Stratum 'h'):
    #   - Weights in Stratum 'h':
    #       - PSU 'j': 0
    #       - Other PSUs: w * (n_h / n_h - 1)
    #   - Weights in other Strata: Unchanged

    unique_psus = np.unique(psu_global)  # Should be 8 PSUs
    rep_cols = {}

    for i, target_psu in enumerate(unique_psus):
        # Identify the stratum of this target PSU
        # We grab the first row matching this PSU to find its stratum
        target_stratum = df.filter(pl.col("psu") == target_psu)["stratum"][0]

        # Calculate scaling factor for this stratum
        # n_h = count of unique PSUs in this stratum
        stratum_psus = df.filter(pl.col("stratum") == target_stratum)["psu"].unique()
        n_h = len(stratum_psus)

        # JKn Multiplier: n_h / (n_h - 1)
        # Note: If n_h=1, this would be infinite (JK doesn't work with singletons without corrections)
        # Here n_h is 2, so factor is 2/1 = 2.
        factor = n_h / (n_h - 1)

        # Apply weights
        # Logic:
        # If row is in target_psu -> 0
        # If row is in target_stratum BUT NOT target_psu -> weight * factor
        # If row is in different stratum -> weight (unchanged)

        replicate_weights = df.with_columns(
            pl.when(pl.col("psu") == target_psu)
            .then(0.0)
            .when(pl.col("stratum") == target_stratum)
            .then(pl.col("weight") * factor)
            .otherwise(pl.col("weight"))
            .alias("w_rep")
        )["w_rep"]

        rep_cols[f"jk_{i + 1}"] = replicate_weights

    # Attach replicates
    df = df.with_columns([pl.Series(k, v) for k, v in rep_cols.items()])

    # 6. Save
    today_str = date.today().strftime("%d%m%Y")
    filename = f"fake_survey_jackknife_{today_str}.csv"
    out_path = Path(f"tests/data/{filename}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out_path)

    print(f"Generated {N} rows with {len(unique_psus)} JKn weights.")
    print(f"Saved to: {out_path.absolute()}")


if __name__ == "__main__":
    generate_and_save_jackknife()
