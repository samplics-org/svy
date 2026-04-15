from pathlib import Path
import svy


HOME = Path.home()
# DATA_PATH = home / "Publications/2026/jss-svy/data"

data = (
    svy.read_csv(HOME / "Publications/2026/jss-svy/data" / "hh_sample_for_r.csv")
    # .fill_nan(None)
    # .drop_nulls()
    # .select(
    #     [
    #         "rake_wgt",
    #         "total_exp",
    #         "food_exp",
    #         "area_type",
    #         "hh_size",
    #         "rooms",
    #         "ea",
    #         "province",
    #     ]
    # )
)

design = svy.Design(
    stratum=("province", "area_type"),
    psu="ea",
    wgt="rake_wgt",
    rep_wgts=svy.RepWeights(
        method=svy.EstimationMethod.BOOTSTRAP,
        prefix="rake_wgt",
        n_reps=500,
        df=499,
    ),
)

hh_sample = svy.Sample(data=data, design=design)

# hh_sample = hh_sample.wrangling.mutate({"log_total_exp": svy.col("total_exp").log()})
# breakpoint()

lin_model = hh_sample.glm.fit(
    y="food_exp",
    x=["hh_size", "rooms", "total_exp", svy.Cat("area_type")],
    family="Gaussian",
)

print(lin_model)


# lin_model2 = hh_sample.glm.fit(
#     y="is_poor",
#     x=["hh_size", svy.Cat("area_type")],
#     family=svy.core.enumerations.DistFamily.BINOMIAL,
# )

# print(lin_model2.fitted)

breakpoint()
