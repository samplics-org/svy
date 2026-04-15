# tests/svy/core/test_sample_select_srs.py
import numpy as np
import polars as pl

from svy import Design, Sample


DF = pl.DataFrame(
    {
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "name": [
            "Alice",
            "Bob",
            "Charlie",
            "David",
            "Eve",
            "Frank",
            "Grace",
            "Hannah",
            "Isaac",
            "Jack",
            "Kate",
            "Liam",
            "Mia",
            "Nora",
            "Oliver",
        ],
        "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
        "education": [
            "Less than HS",
            "HS or higher",
            "HS or higher",
            "HS or higher",
            "Less than HS",
            "Less than HS",
            "Less than HS",
            "HS or higher",
            "HS or higher",
            "Less than HS",
            "HS or higher",
            "HS or higher",
            "HS or higher",
            "Less than HS",
            "Less than HS",
        ],
        "income": [
            50000,
            60000,
            70000,
            80000,
            90000,
            100000,
            11000,
            120000,
            130000,
            140000,
            150000,
            16000,
            17000,
            18000,
            19000,
        ],
    }
)

# Convert to numpy arrays for sampling
frame = np.linspace(0, DF.shape[0] - 1, DF.shape[0], dtype=int)
mos = DF["income"].to_numpy()
stratum = DF["education"].to_numpy()


def test_sample_select_srswor_n():
    samp = Sample(DF)
    samp2 = samp.sampling.srs(n=2, wr=False, drop_nulls=True)
    assert samp2.data["svy_number_of_hits"].sum() == 2
    assert samp2.data["svy_prob_selection"].min() == 2 / 15
    assert samp2.data["svy_prob_selection"].max() == 2 / 15
    assert samp2.data.shape[0] <= 2
    assert samp2.data.shape[0] >= 1
    assert samp2.data.shape[1] == 9
    assert samp2.data["svy_number_of_hits"].min() == 1
    assert samp2.data["svy_number_of_hits"].max() == 1
    assert samp2.data.columns == [
        "svy_row_index",
        "id",
        "name",
        "age",
        "education",
        "income",
        "svy_prob_selection",
        "svy_number_of_hits",
        "svy_sample_weight",
    ]


def test_sample_select_srswr_n_stratified():
    design = Design(stratum="education")
    samp = Sample(DF, design)
    samp2 = samp.sampling.srs(n=2, wr=False, drop_nulls=True)

    assert samp2.data["svy_number_of_hits"].sum() == 4
    assert (
        samp2.data.filter(pl.col("education") == "Less than HS")["svy_prob_selection"].min()
        == 2 / 7
    )
    assert (
        samp2.data.filter(pl.col("education") == "Less than HS")["svy_prob_selection"].max()
        == 2 / 7
    )
    assert (
        samp2.data.filter(pl.col("education") == "HS or higher")["svy_prob_selection"].min()
        == 2 / 8
    )
    assert (
        samp2.data.filter(pl.col("education") == "HS or higher")["svy_prob_selection"].max()
        == 2 / 8
    )
    assert samp2.data.shape[0] <= 4
    assert samp2.data.shape[0] >= 2
    assert samp2.data.shape[1] == 9
    assert samp2.data["education"].unique().shape[0] == 2
    assert samp2.data.columns == [
        "svy_row_index",
        "id",
        "name",
        "age",
        "education",
        "income",
        "svy_prob_selection",
        "svy_number_of_hits",
        "svy_sample_weight",
    ]


def test_sample_select_srswr_by():
    samp = Sample(DF).sampling.srs(n={"Less than HS": 2, "HS or higher": 3}, by="education")


# === Additions start here ===


def _with_region(df: pl.DataFrame) -> pl.DataFrame:
    """Attach a simple alternating region column so each education level appears in both regions."""
    n = df.height
    regions = (["North", "South"] * ((n + 1) // 2))[:n]
    return df.with_columns(pl.Series("region", regions))


def _group_counts(df: pl.DataFrame, *cols: str) -> dict[tuple, int]:
    out: dict[tuple, int] = {}
    for row in df.group_by(list(cols)).len().iter_rows(named=True):
        key = tuple(row[c] for c in cols)
        out[key] = row["len"]
    return out


def test_sample_select_srs_by_only_mapping_matches_by_keys():
    # by-only selection; no design.stratum
    samp = Sample(DF)
    # n per education group
    n_map = {"Less than HS": 2, "HS or higher": 3}
    samp2 = samp.sampling.srs(n=n_map, by="education", wr=False, drop_nulls=True)

    # total hits = sum of mapping
    assert samp2.data["svy_number_of_hits"].sum() == 5
    # check each group's probability equals n_g / size_g
    sizes = _group_counts(DF, "education")
    for ed, n_g in n_map.items():
        # any row in that group has same prob
        prob = samp2.data.filter(pl.col("education") == ed)["svy_prob_selection"]
        if sizes[(ed,)] > 0:
            assert (prob.min() == n_g / sizes[(ed,)]) and (prob.max() == n_g / sizes[(ed,)])
    # rows count and columns sanity
    assert 1 <= samp2.data.height <= 5
    assert "svy_prob_selection" in samp2.data.columns
    assert "svy_number_of_hits" in samp2.data.columns
    assert "svy_sample_weight" in samp2.data.columns


def test_sample_select_srs_stratum_by_scalar_n_broadcasts_to_all_cells():
    # design stratified by region, and by education; scalar n → apply to each (region×education)
    DF2 = _with_region(DF)
    design = Design(stratum="region")
    samp = Sample(DF2, design)
    samp2 = samp.sampling.srs(n=1, by="education", wr=False, drop_nulls=True)

    # how many non-empty (region, education) cells exist?
    cell_sizes = _group_counts(DF2, "region", "education")
    non_empty = sum(1 for v in cell_sizes.values() if v > 0)

    # hits should equal the number of non-empty cells (n=1 per cell)
    assert samp2.data["svy_number_of_hits"].sum() == non_empty

    # Each cell's prob = 1 / size(cell)
    for (reg, ed), size in cell_sizes.items():
        if size == 0:
            continue
        prob_col = samp2.data.filter((pl.col("region") == reg) & (pl.col("education") == ed))[
            "svy_prob_selection"
        ]
        if prob_col.len() > 0:
            assert prob_col.min() == 1 / size
            assert prob_col.max() == 1 / size


def test_sample_select_srs_stratum_by_mapping_matches_by_keys_broadcast_within_each_stratum():
    # design stratified by region, n provided per education → broadcast within each region
    DF2 = _with_region(DF)
    design = Design(stratum="region")
    samp = Sample(DF2, design)
    n_by = {"Less than HS": 1, "HS or higher": 2}  # apply in each region
    samp2 = samp.sampling.srs(n=n_by, by="education", wr=False, drop_nulls=True)

    # total hits = sum(n_by) * number_of_regions
    n_regions = DF2["region"].n_unique()
    assert samp2.data["svy_number_of_hits"].sum() == (sum(n_by.values()) * n_regions)

    # Within each (region, education): prob = n_by[education] / size(region, education)
    cell_sizes = _group_counts(DF2, "region", "education")
    for (reg, ed), size in cell_sizes.items():
        n_g = n_by.get(ed, 0)
        if size == 0 or n_g == 0:
            # if n=0 for this cell we may have no sampled rows
            continue
        p = samp2.data.filter((pl.col("region") == reg) & (pl.col("education") == ed))[
            "svy_prob_selection"
        ]
        if p.len() > 0:
            assert p.min() == n_g / size
            assert p.max() == n_g / size


def test_sample_select_srs_stratum_by_mapping_matches_stratum_keys_broadcast_over_by():
    # n per region (stratum-only) → broadcast across each education level
    DF2 = _with_region(DF)
    design = Design(stratum="region")
    samp = Sample(DF2, design)
    n_by_stratum = {"North": 1, "South": 2}
    samp2 = samp.sampling.srs(n=n_by_stratum, by="education", wr=False, drop_nulls=True)

    # Total hits = sum(n_by_stratum[s]) * number_of_by_levels
    n_by_levels = DF2["education"].n_unique()
    assert samp2.data["svy_number_of_hits"].sum() == sum(n_by_stratum.values()) * n_by_levels

    # For each (region, education): prob = n_by_stratum[region] / size(region, education)
    cell_sizes = _group_counts(DF2, "region", "education")
    for (reg, ed), size in cell_sizes.items():
        n_cell = n_by_stratum.get(reg, 0)
        if size == 0 or n_cell == 0:
            continue
        p = samp2.data.filter((pl.col("region") == reg) & (pl.col("education") == ed))[
            "svy_prob_selection"
        ]
        if p.len() > 0:
            assert p.min() == n_cell / size
            assert p.max() == n_cell / size


def test_sample_select_srs_stratum_by_subset_by_keys_fill_missing_zero():
    # Provide only one by-key → other by levels effectively get n=0 (no selections)
    DF2 = _with_region(DF)
    design = Design(stratum="region")
    samp = Sample(DF2, design)
    n_partial = {"Less than HS": 2}  # HS or higher → 0
    samp2 = samp.sampling.srs(n=n_partial, by="education", wr=False, drop_nulls=True)

    # Should only sample from "Less than HS" (across both regions): 2 per region
    n_regions = DF2["region"].n_unique()
    assert samp2.data["svy_number_of_hits"].sum() == 2 * n_regions
    assert samp2.data["education"].unique().to_list() == ["Less than HS"]


def test_sample_select_srs_stratum_by_subset_stratum_keys_fill_missing_zero():
    # Provide only one stratum key → missing strata get n=0
    DF2 = _with_region(DF)
    design = Design(stratum="region")
    samp = Sample(DF2, design)
    n_partial_stratum = {"North": 1}  # South → 0
    samp2 = samp.sampling.srs(n=n_partial_stratum, by="education", wr=False, drop_nulls=True)

    # Only North contributes; across both education levels (present in North)
    n_by_levels_north = DF2.filter(pl.col("region") == "North")["education"].n_unique()
    assert samp2.data["svy_number_of_hits"].sum() == 1 * n_by_levels_north
    assert samp2.data["region"].unique().to_list() == ["North"]


def test_sample_select_srs_error_on_unmatched_keys():
    # Keys match neither combined groups, nor by-only, nor stratum-only ⇒ error
    DF2 = _with_region(DF)
    design = Design(stratum="region")
    samp = Sample(DF2, design)

    # Provide keys that don't exist anywhere
    n_bad = {"ZZZ": 2}
    try:
        _ = samp.sampling.srs(n=n_bad, by="education", wr=False, drop_nulls=True)
        assert False, "Expected ValueError for unmatched keys in n"
    except ValueError as e:
        # basic sanity: message mentions keys
        assert "keys" in str(e)


def test_sample_select_srs_chains_previous_probabilities_when_present():
    # If the Design already has a prob column, ensure chaining: pi_new = pi_prev * pi_srs
    DF2 = DF.with_columns(pl.lit(0.5).alias("prev_prob"))
    design = Design(wgt=None, prob="prev_prob")  # prev prob supplied
    samp = Sample(DF2, design)
    samp2 = samp.sampling.srs(n=2, wr=False, drop_nulls=True)

    # The probability column name should be whatever the design advertises
    prob_col = samp2.design.prob or "svy_prob_selection"
    wgt_col = samp2.design.wgt or "svy_sample_weight"
    hit_col = samp2.design.hit or "svy_number_of_hits"

    # Sanity: columns exist
    assert prob_col in samp2.data.columns
    assert wgt_col in samp2.data.columns
    assert hit_col in samp2.data.columns

    # New probs must be <= old (0.5) because pi_new = pi_prev * pi_srs and pi_srs in (0,1]
    assert (samp2.data[prob_col] <= 0.5).all()

    # Weights are 1 / (pi_prev * pi_srs) >= 1 / pi_prev == 2.0
    assert (samp2.data[wgt_col] >= 2.0).all()

    # And hits are integers >= 1 for selected rows
    assert (samp2.data[hit_col] >= 1).all()


def test_sample_select_srs_sublevel_mapping_broadcasts_by_component():
    """
    Keys match sub-components of combined group keys.
    {"Less than HS": 2, "HS or higher": 3} should broadcast across all
    regions because each combined key contains exactly one education level.
    Probabilities must equal n_education / size(region, education).
    """
    DF2 = _with_region(DF)
    design = Design(stratum="region")
    samp = Sample(DF2, design)
    n_map = {"Less than HS": 2, "HS or higher": 3}
    samp2 = samp.sampling.srs(n=n_map, by="education", wr=False, drop_nulls=True)

    # total hits = sum(n_map) * n_regions
    n_regions = DF2["region"].n_unique()
    assert samp2.data["svy_number_of_hits"].sum() == sum(n_map.values()) * n_regions

    # per-cell probability = n_education / size(region, education)
    cell_sizes = _group_counts(DF2, "region", "education")
    for (reg, ed), size in cell_sizes.items():
        n_g = n_map.get(ed, 0)
        if size == 0 or n_g == 0:
            continue
        p = samp2.data.filter((pl.col("region") == reg) & (pl.col("education") == ed))[
            "svy_prob_selection"
        ]
        if p.len() > 0:
            assert np.isclose(p.min(), n_g / size, atol=1e-8)
            assert np.isclose(p.max(), n_g / size, atol=1e-8)


def test_sample_select_srs_sublevel_ambiguous_keys_raises():
    """
    Keys that match multiple components of the same combined key are
    ambiguous and should raise a ValueError.
    e.g. {"North": 1, "Less than HS": 2} — each combined key like
    "North__by__Less than HS" contains both keys, making it ambiguous.
    """
    DF2 = _with_region(DF)
    design = Design(stratum="region")
    samp = Sample(DF2, design)
    n_ambiguous = {"North": 1, "Less than HS": 2}
    try:
        _ = samp.sampling.srs(n=n_ambiguous, by="education", wr=False, drop_nulls=True)
        assert False, "Expected ValueError for ambiguous sub-level keys"
    except ValueError as e:
        assert "keys" in str(e)


def test_sample_select_srs_sublevel_unrecognized_keys_raises():
    """
    Keys that match no component of any combined key should raise ValueError.
    This ensures the existing error behavior is preserved.
    """
    DF2 = _with_region(DF)
    design = Design(stratum="region")
    samp = Sample(DF2, design)
    n_bad = {"ZZZ": 2}
    try:
        _ = samp.sampling.srs(n=n_bad, by="education", wr=False, drop_nulls=True)
        assert False, "Expected ValueError for unrecognized keys in n"
    except ValueError as e:
        assert "keys" in str(e)
