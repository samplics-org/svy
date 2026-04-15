import svy


hld_frame = svy.load_dataset(
    name="hld_pop_wb_2023",
    limit=None,
    force_local=True,
)


hh_sample = svy.Sample(data=hld_frame).sampling.srs(n=10000)

hh_sample = hh_sample.wrangling.apply_labels(
    labels={
        "urbrur": "Area type",
        "statocc": "Tenure",
    },
    categories={
        "urbrur": {"Urban": "1. Urban", "Rural": "2. Rural"},
        "statocc": {"Occupied for free": "Free use"},
    },
)

mean_exp_ur = hh_sample.estimation.mean(
    y="tot_exp",
    by="urbrur",
)
print(mean_exp_ur)


tab_tenure_count = hh_sample.categorical.tabulate(
    rowvar="statocc",
    units=svy.TableUnits.COUNT,
    drop_nulls=True,
)

print(tab_tenure_count)

breakpoint()
