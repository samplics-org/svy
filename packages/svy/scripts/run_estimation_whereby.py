import svy
import polars as pl


# BASE_DIR = Path("tests/test_data")
# DATA_PATH = BASE_DIR / "svy_synthetic_sample_07082025.csv"

# data = pl.read_csv(DATA_PATH).fill_nan(None).drop_nulls()
# print(f"Rows: {data.height}")

# design = svy.Design(stratum="region", psu="cluster", wgt="samp_wgt")
# design = svy.Design(wgt="samp_wgt")

# # Print what by_col name is produced
# from svy.core.data_prep import prepare_data

# prep = prepare_data(
#     sample,
#     y="income",
#     by=["_svy_domain_", "gender"],
#     drop_nulls=True,
#     cast_y_float=True,
#     apply_singleton_filter=False,
# )
# print("by_col:", prep.by_col)
# print("columns:", [c for c in prep.df.columns if "by" in c or "domain" in c])


test_data = pl.DataFrame(
    {
        "id": list(range(1, 21)),
        "stratum": ["A"] * 10 + ["B"] * 10,
        "psu": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
        "weight": [1.5] * 5 + [2.0] * 5 + [1.8] * 5 + [2.2] * 5,
        "income": [
            25000,
            30000,
            35000,
            40000,
            45000,
            50000,
            55000,
            60000,
            65000,
            70000,
            28000,
            33000,
            38000,
            43000,
            48000,
            52000,
            57000,
            62000,
            67000,
            72000,
        ],
        "region": [
            "North",
            "North",
            "South",
            "South",
            "North",
            "South",
            "North",
            "South",
            "North",
            "South",
            "North",
            "South",
            "North",
            "South",
            "North",
            "South",
            "North",
            "South",
            "North",
            "South",
        ],
        "age_group": [
            "Young",
            "Young",
            "Old",
            "Old",
            "Young",
            "Old",
            "Young",
            "Old",
            "Young",
            "Old",
            "Young",
            "Old",
            "Young",
            "Old",
            "Young",
            "Old",
            "Young",
            "Old",
            "Young",
            "Old",
        ],
    }
)


design = svy.Design(stratum="stratum", psu="psu", wgt="weight")
sample = svy.Sample(data=test_data, design=design)


result = sample.estimation.median("income", by="region", where=svy.col("stratum") == "A")


breakpoint()
