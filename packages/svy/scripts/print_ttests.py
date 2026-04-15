from pathlib import Path

import polars as pl
import svy


BASE_DIR = Path("tests/test_data")
DATA_PATH = BASE_DIR / "svy_synthetic_sample_07082025.csv"

data = svy.read_csv(DATA_PATH).fill_nan(None).drop_nulls()
design = svy.Design(stratum="region", psu="cluster", wgt="samp_wgt")
sample = svy.Sample(data=data, design=design)

print(f"Rows: {data.height}")
tt1 = sample.categorical.ttest(
    y="income",
    mean_h0=60000,
    drop_nulls=True,
    where=pl.col("income") > 50000,
)
print(tt1)


tt2 = sample.categorical.ttest(
    y="income",
    mean_h0=60000,
    drop_nulls=True,
    by="sex",
)
print(tt2)

tt3 = sample.categorical.ttest(
    y="income",
    mean_h0=60000,
    drop_nulls=True,
    by=("educ", "sex"),
)
print(tt3)

tt4 = sample.categorical.ttest(
    y="income",
    mean_h0=60000,
    drop_nulls=True,
    by="educ",
    where=pl.col("sex") == 2,
)
print(tt4)
