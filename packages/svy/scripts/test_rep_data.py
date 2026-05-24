import polars as pl


data = pl.read_csv("tests/test_data/fake_survey_brr_24122025.csv")
print(f"Total rows: {data.height}")
print(f"NAs in income: {data['income'].null_count()}")
print(f"NaNs in income: {data['income'].is_nan().sum()}")
print(f"NAs in weight: {data['weight'].null_count()}")
print(f"NAs in psu: {data['psu'].null_count()}")

clean = data.filter(pl.col("income").is_not_null() & pl.col("income").is_not_nan())
print(f"After removing income NA/NaN: {clean.height} rows")
weighted_mean = (clean["income"] * clean["weight"]).sum() / clean["weight"].sum()
print(f"Weighted mean: {weighted_mean}")
