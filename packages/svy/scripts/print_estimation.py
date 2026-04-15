from pathlib import Path
import polars as pl
import svy

BASE_DIR = Path("tests/test_data")
DATA_PATH = BASE_DIR / "svy_synthetic_sample_07082025.csv"

data = svy.read_csv(DATA_PATH).fill_nan(None).drop_nulls()
design = svy.Design(stratum="region", psu="cluster", wgt="samp_wgt")
sample = svy.Sample(data=data, design=design)

print(f"Rows: {data.height}")

# E0: baseline
print(sample.estimation.mean("income"))

# E1: mean with where (pl.Expr)
print(sample.estimation.mean("income", where=pl.col("income") > 50000))

# E2: mean with by
print(sample.estimation.mean("income", by="sex"))

# E3: mean with where + by
print(sample.estimation.mean("income", where=pl.col("educ") == "Postgraduate", by="sex"))

# E4: dict where
print(sample.estimation.mean("income", where={"sex": 2}))

# E5: list of exprs
print(sample.estimation.mean("income", where=[pl.col("income") > 50000, pl.col("sex") == 1]))

# E6: total with where
print(sample.estimation.total("income", where=pl.col("income") > 50000))

# E7: prop with where
print(sample.estimation.prop("sex", where=pl.col("income") > 50000))

# E8: ratio with where
print(sample.estimation.ratio("income", "fam_size", where=pl.col("sex") == 1))

# E9: multi-by
print(sample.estimation.mean("income", by=("region", "sex")))

# E10: where + multi-by
print(
    sample.estimation.mean("income", where=pl.col("educ") == "Postgraduate", by=("region", "sex"))
)
