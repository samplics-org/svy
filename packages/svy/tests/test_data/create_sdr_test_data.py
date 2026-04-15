# scripts/create_sdr_test_data.py
"""
Create test dataset for SDR (Successive Difference Replication) validation.
This dataset mimics ACS-style data with 80 replicate weights.
"""

import numpy as np
import polars as pl


def create_sdr_test_data(seed: int = 42) -> pl.DataFrame:
    """
    Create reproducible test data for SDR validation.

    Structure mimics ACS PUMS:
    - pwgtp: Person weight
    - repwtp1-repwtp80: Replicate weights
    - income: Continuous variable
    - age: Continuous variable
    - employed: Binary variable (0/1)
    - region: Categorical variable (4 levels)
    """
    np.random.seed(seed)
    n = 5000
    n_reps = 80

    # Base weights (simulate complex survey weights)
    pwgtp = np.random.uniform(50, 500, n)

    # Demographics
    age = np.random.normal(45, 15, n).clip(18, 85).astype(int)
    employed = np.random.binomial(1, 0.65, n)
    region = np.random.choice(
        ["Northeast", "Midwest", "South", "West"], n, p=[0.18, 0.21, 0.38, 0.23]
    )

    # Income (correlated with age and employment)
    base_income = 30000 + age * 500 + employed * 25000
    income = (base_income * np.random.lognormal(0, 0.5, n)).clip(0, 500000)

    # Health expenditure (for ratio estimation)
    health_exp = (income * 0.08 * np.random.uniform(0.5, 1.5, n)).clip(0, 50000)

    # Create replicate weights using successive difference pattern
    # In real ACS, these are constructed via Hadamard matrix
    # Here we simulate the structure
    rep_weights = {}
    for r in range(1, n_reps + 1):
        # Perturbation factor (mimics SDR structure)
        perturbation = 1 + 0.5 * np.sin(2 * np.pi * r / n_reps + np.arange(n) * 0.01)
        perturbation *= np.random.uniform(0.9, 1.1, n)
        rep_weights[f"repwtp{r}"] = pwgtp * perturbation

    # Build DataFrame
    data = {
        "id": np.arange(1, n + 1),
        "pwgtp": pwgtp,
        "age": age,
        "employed": employed,
        "region": region,
        "income": income,
        "health_exp": health_exp,
        **rep_weights,
    }

    return pl.DataFrame(data)


if __name__ == "__main__":
    df = create_sdr_test_data()

    # Save as CSV for R compatibility
    df.write_csv("tests/data/sdr_test_data.csv")

    # Save as parquet for Python
    df.write_parquet("tests/data/sdr_test_data.parquet")

    print(f"Created SDR test data: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Columns: id, pwgtp, age, employed, region, income, health_exp, repwtp1-repwtp80")
