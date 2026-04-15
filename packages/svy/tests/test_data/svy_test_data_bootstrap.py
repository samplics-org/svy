# tests/data/svy_test_data_bootstrap.py
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl


def generate_and_save_bootstrap():
    # 1. Lock the Bit Generator for permanent stability (Same Seed as BRR)
    bg = np.random.PCG64(seed=99)
    rng = np.random.Generator(bg)

    N = 200
    N_REPS = 20

    # 2. Design: 4 Strata, 2 PSUs per stratum (Balanced)
    strata = np.repeat(["S1", "S2", "S3", "S4"], N // 4)
    psu_local = np.tile(np.repeat([1, 2], N // 8), 4)
    psu_global = [f"{s}-{p}" for s, p in zip(strata, psu_local)]

    # 3. Weights: Correlated with PSU
    w_base = rng.uniform(0.5, 1.5, N)
    w_psu = np.where(psu_local == 1, 1.5, 1.0)
    weights = w_base * w_psu

    # 4. Variables (Identical logic to BRR script)
    inc_base = rng.normal(50000, 10000, N)
    inc_shift = np.array([0 if s in ("S1", "S2") else 10000 for s in strata])
    income = np.abs(inc_base + inc_shift)
    income[0] = None  # Explicit NULL

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

    # 5. Generate 20 Bootstrap Replicate Weights
    unique_strata = np.unique(strata)
    rep_cols = {}

    for r in range(N_REPS):
        rep_col_values = np.zeros(N)

        for s in unique_strata:
            # Filter data for this stratum
            s_mask = (df["stratum"] == s).to_numpy()
            s_psus = df.filter(pl.col("stratum") == s)["psu"].unique().to_numpy()

            n_h = len(s_psus)  # Number of PSUs in stratum (should be 2)
            m_h = n_h - 1  # Resample size

            # Resample PSUs with replacement
            resampled_psus = rng.choice(s_psus, size=m_h, replace=True)

            # Count how many times each PSU was selected (k_hi)
            psu_counts = {p: 0 for p in s_psus}
            unique_selected, counts = np.unique(resampled_psus, return_counts=True)
            for p, c in zip(unique_selected, counts):
                psu_counts[p] = c

            # Calculate Bootstrap Multiplier
            factor = n_h / m_h

            # Assign weights to rows
            s_indices = np.where(s_mask)[0]

            for idx in s_indices:
                # FIX: Explicitly cast numpy int64 to python int
                row_psu = df["psu"][int(idx)]
                k_hi = psu_counts[row_psu]  # How many times this PSU was picked

                # New weight = Old Weight * Factor * Times_Picked
                rep_col_values[int(idx)] = df["weight"][int(idx)] * factor * k_hi

        rep_cols[f"bs_{r + 1}"] = rep_col_values

    # Attach replicates to DataFrame
    df = df.with_columns([pl.Series(k, v) for k, v in rep_cols.items()])

    # 6. Save with Date
    today_str = date.today().strftime("%d%m%Y")
    filename = f"fake_survey_bootstrap_{today_str}.csv"
    out_path = Path(f"tests/data/{filename}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out_path)

    print(f"Generated {N} rows with {N_REPS} Bootstrap weights.")
    print(f"Saved to: {out_path.absolute()} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    generate_and_save_bootstrap()
