"""
Generate a synthetic multi-stage survey dataset for testing FPC.

Design:
    2 strata
    Stratum 1: 3 sampled PSUs out of 15 (sampling fraction = 0.20)
    Stratum 2: 4 sampled PSUs out of 10 (sampling fraction = 0.40)

    Within each PSU: variable number of SSUs sampled from a larger population
    Within each SSU: variable number of units sampled from a larger population

    Weights = product of inverse selection probabilities at each stage.

Output columns:
    stratum     - stratum identifier
    psu         - PSU identifier (unique across strata)
    ssu         - SSU identifier (unique across PSUs)
    unit_id     - unit identifier (unique across SSUs)
    y           - continuous outcome variable
    x           - continuous covariate (correlated with y)
    binary_y    - binary outcome (0/1) for proportion estimation
    wgt         - final sampling weight
    fpc_psu     - population number of PSUs in this stratum (N_h)
    fpc_ssu     - population number of SSUs in this PSU (M_hi)
    fpc_unit    - population number of units in this SSU (K_hij)
"""

import numpy as np
import polars as pl

np.random.seed(202503)

# =============================================================================
# Population structure
# =============================================================================

# fmt: off
design = {
    # stratum: (N_psu, sampled_psus: [(psu_id, N_ssu, sampled_ssus: [(ssu_id, N_unit, n_unit)])])
    "A": {
        "N_psu": 15,
        "psus": [
            {"psu": "A01", "N_ssu": 12, "ssus": [
                {"ssu": "A01_s1", "N_unit": 80, "n_unit": 8},
                {"ssu": "A01_s2", "N_unit": 120, "n_unit": 10},
                {"ssu": "A01_s3", "N_unit": 50, "n_unit": 6},
            ]},
            {"psu": "A02", "N_ssu": 8, "ssus": [
                {"ssu": "A02_s1", "N_unit": 60, "n_unit": 7},
                {"ssu": "A02_s2", "N_unit": 90, "n_unit": 9},
            ]},
            {"psu": "A03", "N_ssu": 10, "ssus": [
                {"ssu": "A03_s1", "N_unit": 100, "n_unit": 12},
                {"ssu": "A03_s2", "N_unit": 70, "n_unit": 8},
                {"ssu": "A03_s3", "N_unit": 40, "n_unit": 5},
                {"ssu": "A03_s4", "N_unit": 55, "n_unit": 6},
            ]},
        ],
    },
    "B": {
        "N_psu": 10,
        "psus": [
            {"psu": "B01", "N_ssu": 6, "ssus": [
                {"ssu": "B01_s1", "N_unit": 45, "n_unit": 5},
                {"ssu": "B01_s2", "N_unit": 110, "n_unit": 11},
            ]},
            {"psu": "B02", "N_ssu": 15, "ssus": [
                {"ssu": "B02_s1", "N_unit": 75, "n_unit": 9},
                {"ssu": "B02_s2", "N_unit": 65, "n_unit": 7},
                {"ssu": "B02_s3", "N_unit": 130, "n_unit": 13},
            ]},
            {"psu": "B03", "N_ssu": 9, "ssus": [
                {"ssu": "B03_s1", "N_unit": 85, "n_unit": 10},
                {"ssu": "B03_s2", "N_unit": 35, "n_unit": 4},
            ]},
            {"psu": "B04", "N_ssu": 11, "ssus": [
                {"ssu": "B04_s1", "N_unit": 95, "n_unit": 10},
                {"ssu": "B04_s2", "N_unit": 50, "n_unit": 6},
                {"ssu": "B04_s3", "N_unit": 70, "n_unit": 8},
                {"ssu": "B04_s4", "N_unit": 40, "n_unit": 5},
                {"ssu": "B04_s5", "N_unit": 60, "n_unit": 7},
            ]},
        ],
    },
}
# fmt: on

# =============================================================================
# Generate rows
# =============================================================================

rows: list[dict] = []

# Stratum-level means for generating realistic data
stratum_means = {"A": 50.0, "B": 75.0}

for stratum_id, stratum_info in design.items():
    N_psu = stratum_info["N_psu"]
    n_psu = len(stratum_info["psus"])

    for psu_info in stratum_info["psus"]:
        psu_id = psu_info["psu"]
        N_ssu = psu_info["N_ssu"]
        n_ssu = len(psu_info["ssus"])

        # PSU-level random effect
        psu_effect = np.random.normal(0, 8)

        for ssu_info in psu_info["ssus"]:
            ssu_id = ssu_info["ssu"]
            N_unit = ssu_info["N_unit"]
            n_unit = ssu_info["n_unit"]

            # SSU-level random effect
            ssu_effect = np.random.normal(0, 4)

            # Weight = product of inverse selection probabilities
            # P(PSU selected) = n_psu / N_psu
            # P(SSU selected | PSU) = n_ssu / N_ssu
            # P(unit selected | SSU) = n_unit / N_unit
            prob = (n_psu / N_psu) * (n_ssu / N_ssu) * (n_unit / N_unit)
            wgt = 1.0 / prob

            base_mean = stratum_means[stratum_id] + psu_effect + ssu_effect

            for u in range(1, n_unit + 1):
                unit_id = f"{ssu_id}_u{u}"

                # Generate correlated y and x
                y_val = base_mean + np.random.normal(0, 10)
                x_val = y_val * 0.8 + np.random.normal(0, 5)

                # Binary outcome (probability depends on y)
                p_binary = 1.0 / (1.0 + np.exp(-(y_val - 60) / 15))
                binary_y = int(np.random.random() < p_binary)

                rows.append(
                    {
                        "stratum": stratum_id,
                        "psu": psu_id,
                        "ssu": ssu_id,
                        "unit_id": unit_id,
                        "y": round(y_val, 2),
                        "x": round(x_val, 2),
                        "binary_y": binary_y,
                        "wgt": round(wgt, 4),
                        "fpc_psu": N_psu,
                        "fpc_ssu": N_ssu,
                        "fpc_unit": N_unit,
                    }
                )

# =============================================================================
# Create DataFrame and save
# =============================================================================

df = pl.DataFrame(rows)

# Print summary
print(f"Total rows: {df.height}")
print(f"Strata: {df['stratum'].n_unique()}")
print(f"PSUs: {df['psu'].n_unique()}")
print(f"SSUs: {df['ssu'].n_unique()}")
print()

# Units per SSU
print("Units per SSU:")
units_per_ssu = df.group_by("ssu").agg(pl.len().alias("n_units"))
print(units_per_ssu.sort("ssu"))
print()

# SSUs per PSU
print("SSUs per PSU:")
ssus_per_psu = df.group_by("psu").agg(pl.col("ssu").n_unique().alias("n_ssus"))
print(ssus_per_psu.sort("psu"))
print()

# Sampling fractions
print("Sampling fractions:")
for stratum_id, stratum_info in design.items():
    N_psu = stratum_info["N_psu"]
    n_psu = len(stratum_info["psus"])
    print(f"  Stratum {stratum_id}: PSU fraction = {n_psu}/{N_psu} = {n_psu / N_psu:.2f}")
    for psu_info in stratum_info["psus"]:
        N_ssu = psu_info["N_ssu"]
        n_ssu = len(psu_info["ssus"])
        print(f"    {psu_info['psu']}: SSU fraction = {n_ssu}/{N_ssu} = {n_ssu / N_ssu:.2f}")
        for ssu_info in psu_info["ssus"]:
            N_unit = ssu_info["N_unit"]
            n_unit = ssu_info["n_unit"]
            print(
                f"      {ssu_info['ssu']}: unit fraction = {n_unit}/{N_unit} = {n_unit / N_unit:.2f}"
            )

print()
print("Weight range:", df["wgt"].min(), "-", df["wgt"].max())
print()
print("First 10 rows:")
print(df.head(10))

# Save to CSV
output_path = "tests/test_data/synthetic_multistage.csv"
df.write_csv(output_path)
print(f"\nSaved to {output_path}")
