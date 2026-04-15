# tests/svy/data/svy_synthetic_sample.py
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl


BASE_DIR = Path(__file__).resolve().parent.parent


# Set random seed for reproducibility
np.random.seed(99)

# Generate data
n_records = 1000

# ID from 1 to 1000
ids = list(range(1, n_records + 1))

# Region with 4 values
# 1=North, 2=South, 3=West, 4=East
regions = np.random.choice(["1", "2", "3", "4"], size=n_records)

# Generate clusters (10-15 per region)
clusters = []
for region in regions:
    if region == "1":
        cluster = np.random.randint(1, 16)  # 1-15
    elif region == "2":
        cluster = np.random.randint(16, 31)  # 16-30
    elif region == "3":
        cluster = np.random.randint(31, 46)  # 31-45
    else:  # East
        cluster = np.random.randint(46, 61)  # 46-60
    clusters.append(cluster)

# Education levels
education_levels = [
    "Less than High School",
    "High School",
    "Undergraduate",
    "Postgraduate",
    "Other Training",
]
education = np.random.choice(education_levels, p=[0.1, 0.3, 0.3, 0.2, 0.1], size=n_records)

# Add missing data to education (~1%)
education_missing_indices = np.random.choice(n_records, size=int(n_records * 0.010), replace=False)
education[education_missing_indices] = None

# Sex with two categories
# 1=Male, 2=Female
sex = np.random.choice([1, 2], size=n_records).astype(float)
sex_missing_indices = np.random.choice(n_records, size=int(n_records * 0.010), replace=False)
sex[sex_missing_indices] = np.nan

# Response with two categories
# 1=low, 2=high
resp2 = np.random.choice([1, 2], size=n_records).astype(float)
resp2_missing_indices = np.random.choice(n_records, size=int(n_records * 0.020), replace=False)
resp2[resp2_missing_indices] = np.nan

# Response with three categories
# 1=low, 2=medium, 3=high
resp3 = np.random.choice([1, 2, 3], size=n_records).astype(float)
resp3_missing_indices = np.random.choice(n_records, size=int(n_records * 0.003), replace=False)
resp3[resp3_missing_indices] = np.nan

# Response with five categories
# 1="very low", 2="low", 3="medium", 4="high", 5="very high"
resp5 = np.random.choice([1, 2, 3, 4, 5], size=n_records).astype(float)
resp5_missing_indices = np.random.choice(n_records, size=int(n_records * 0.015), replace=False)
resp5[resp5_missing_indices] = np.nan

# Income as float64
income = np.random.normal(50000, 15000, size=n_records).astype(np.float64)
income = np.maximum(income, 0)  # Ensure no negative income
income_missing_indices = np.random.choice(n_records, size=int(n_records * 0.010), replace=False)
income[income_missing_indices] = np.nan

# Family size as integer
family_size = []
for region in regions:
    if region == "1":
        size = np.random.poisson(2)
    elif region == "2":
        size = np.random.poisson(3)
    elif region == "3":
        size = np.random.poisson(4)
    else:  # East
        size = np.random.poisson(2)
    size = max(1, size)
    family_size.append(size)
family_size = np.array(family_size)
family_size_missing_indices = np.random.choice(
    n_records, size=int(n_records * 0.01), replace=False
)
family_size = family_size.astype(float)
family_size[family_size_missing_indices] = None

# Sample weights as float64
# Sample weights as float64 - set by cluster within stratum (region)
sample_weights = []
for i in range(n_records):
    cluster = clusters[i]
    region = regions[i]

    # Different distributions by cluster
    if region == "1":  # clusters 1-15
        if cluster <= 5:
            weight = np.random.gamma(2, 500)  # Gamma distribution
        elif cluster <= 10:
            weight = np.random.exponential(800)  # Exponential distribution
        else:
            weight = np.random.lognormal(6, 0.5)  # Log-normal distribution
    elif region == "2":  # clusters 16-30
        if cluster <= 20:
            weight = np.random.weibull(1.5) * 1000 + 100  # Weibull distribution
        elif cluster <= 25:
            weight = np.random.beta(2, 5) * 1500 + 200  # Beta distribution
        else:
            weight = np.random.normal(1200, 300)  # Normal distribution
    elif region == "3":  # clusters 31-45
        if cluster <= 35:
            weight = np.random.pareto(1.16) * 400 + 150  # Pareto distribution
        elif cluster <= 40:
            weight = np.random.triangular(100, 800, 1500)  # Triangular distribution
        else:
            weight = np.random.uniform(300, 1800)  # Uniform distribution
    else:  # East, clusters 46-60
        if cluster <= 50:
            weight = np.random.chisquare(3) * 200 + 100  # Chi-square distribution
        elif cluster <= 55:
            weight = np.random.gumbel(800, 200)  # Gumbel distribution
        else:
            weight = np.random.laplace(1000, 150)  # Laplace distribution
    weight = max(50, min(weight, 3000))
    sample_weights.append(weight)
sample_weights = np.array(sample_weights).astype(np.float64)
sample_weights_missing_indices = np.random.choice(
    n_records, size=int(n_records * 0.01), replace=False
)
sample_weights[sample_weights_missing_indices] = np.nan

# Create polars DataFrame
df = pl.DataFrame(
    {
        "id": ids,
        "region": regions,
        "cluster": clusters,
        "educ": education,
        "sex": sex,
        "income": income,
        "resp2": resp2,
        "resp3": resp3,
        "resp5": resp5,
        "fam_size": family_size,
        "samp_wgt": sample_weights,
    }
)


df = df.with_columns(
    [
        pl.col("id").cast(pl.Int32),
        pl.col("region").cast(pl.String),
        pl.col("cluster").cast(pl.Int64),
        pl.col("educ").cast(pl.String),
        pl.col("sex").cast(pl.Int8, strict=False),
        pl.col("income").cast(pl.Float64),
        pl.col("resp2").cast(pl.Int8, strict=False),
        pl.col("resp3").cast(pl.Int8, strict=False),
        pl.col("resp5").cast(pl.Int8, strict=False),
        pl.col("fam_size").cast(pl.Int16, strict=False),
        pl.col("samp_wgt").cast(pl.Float64),
    ]
)

# Format date as YYYYMMDD
today_str = date.today().strftime("%d%m%Y")

df.write_csv(file=BASE_DIR / f"datasets/svy_synthetic_sample_{today_str}.csv")
