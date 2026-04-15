# tests/data/svy_test_data_bbr.py
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl


def generate_and_save():
    # 1. Lock the Bit Generator for permanent stability
    bg = np.random.PCG64(seed=99)
    rng = np.random.Generator(bg)

    N = 200

    # 2. Design: 4 Strata, 2 PSUs per stratum (Balanced for BRR)
    strata = np.repeat(["S1", "S2", "S3", "S4"], N // 4)

    # PSUs: 1, 2 nested within strata
    psu_local = np.tile(np.repeat([1, 2], N // 8), 4)
    psu_global = [f"{s}-{p}" for s, p in zip(strata, psu_local)]

    # 3. Weights: Correlated with PSU (design effect)
    w_base = rng.uniform(0.5, 1.5, N)
    w_psu = np.where(psu_local == 1, 1.5, 1.0)
    weights = w_base * w_psu

    # 4. Variables
    # Income: Continuous, correlated with Stratum
    inc_base = rng.normal(50000, 10000, N)
    inc_shift = np.array([0 if s in ("S1", "S2") else 10000 for s in strata])
    income = np.abs(inc_base + inc_shift)

    # Add explicit NULLs to test drop_nulls behavior
    income[0] = None

    # Sex: Binary (0/1 or M/F)
    sex = rng.choice(["Male", "Female"], size=N)
    sex[1] = None

    # Education: Categorical (3 levels)
    educ = rng.choice(["Low", "Med", "High"], size=N, p=[0.3, 0.5, 0.2])

    df = pl.DataFrame(
        {
            "id": np.arange(N),
            "stratum": strata,
            "psu_local": psu_local,  # Kept temporarily for BRR calculation
            "psu": psu_global,
            "weight": weights,
            "income": income,
            "sex": sex,
            "educ": educ,
        }
    )

    # 5. Generate 8 BRR Replicate Weights (Hadamard 8)
    # We have 4 strata. We use the first 4 columns of an H8 matrix.
    # Rows = Replicates (8), Cols = Strata (4 used)
    # Rule: 1 -> Keep PSU 1; -1 -> Keep PSU 2
    H8 = np.array(
        [
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 1, -1, -1],
            [1, -1, -1, 1],
            [1, 1, 1, 1],
            [1, -1, 1, -1],
            [1, 1, -1, -1],
            [1, -1, -1, 1],
        ]
    )

    # Map Strata Labels "S1".."S4" to column indices 0..3 of the H matrix
    s_map = {"S1": 0, "S2": 1, "S3": 2, "S4": 3}

    rep_cols = {}

    for r in range(8):  # 8 Replicates
        # Get column index for each row's stratum
        s_idx = df["stratum"].replace(s_map, default=None).cast(pl.Int8).to_numpy()

        # Get H value (+1 or -1) for this row's stratum in replicate r
        h_vals = H8[r, s_idx]

        # PSU local ID (1 or 2)
        p_local = df["psu_local"].to_numpy()

        # BRR Logic:
        # Keep if (PSU=1 and H=1) OR (PSU=2 and H=-1)
        # Kept units get 2x weight, others get 0
        mask_keep = ((p_local == 1) & (h_vals == 1)) | ((p_local == 2) & (h_vals == -1))

        mult = np.zeros(N)
        mult[mask_keep] = 2.0

        rep_cols[f"brr_{r + 1}"] = df["weight"].to_numpy() * mult

    # Attach replicates to DataFrame and remove helper column
    df = df.with_columns([pl.Series(k, v) for k, v in rep_cols.items()]).drop("psu_local")

    # 6. Save with Date
    # Format: DDMMYYYY (e.g. 24122025)
    today_str = date.today().strftime("%d%m%Y")
    filename = f"fake_survey_brr_{today_str}.csv"

    # Construct path relative to this script or project root
    # Adjust 'tests/data' if your folder structure differs
    out_path = Path(f"tests/data/{filename}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out_path)

    print(f"Generated {N} rows with 8 BRR weights.")
    print(f"Saved to: {out_path.absolute()} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    generate_and_save()
