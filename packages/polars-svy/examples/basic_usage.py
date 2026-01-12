"""
Example: Using polars-svy for survey estimation
"""

import polars as pl
import polars_svy as ps

# Create sample household survey data
df = pl.DataFrame(
    {
        "household_id": range(1, 11),
        "region": [
            "North",
            "North",
            "South",
            "South",
            "East",
            "East",
            "West",
            "West",
            "Central",
            "Central",
        ],
        "income": [
            45000,
            52000,
            38000,
            41000,
            55000,
            60000,
            48000,
            51000,
            42000,
            46000,
        ],
        "hours_worked": [2000, 2200, 1800, 1900, 2400, 2500, 2100, 2200, 1950, 2050],
        "employed": [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        "weight": [1.5, 1.8, 2.0, 1.9, 1.6, 1.7, 2.1, 1.9, 1.8, 2.0],
        "stratum": ["1", "1", "2", "2", "3", "3", "4", "4", "5", "5"],
        "psu": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
    }
).with_columns(
    [
        pl.col("income").cast(pl.Float64),
        pl.col("hours_worked").cast(pl.Float64),
        pl.col("employed").cast(pl.Float64),
    ]
)

print("Sample Data:")
print(df)
print("\n" + "=" * 80 + "\n")

# Example 1: Simple mean estimation
print("Example 1: Mean household income (no grouping)")
result1 = df.svy.design(
    weight="weight",
    strata="stratum",
    psu="psu",
).agg(est=ps.mean("income"))
print(result1)
print("\n" + "=" * 80 + "\n")

# Example 2: Domain estimation (by region)
print("Example 2: Mean household income by region")
result2 = (
    df.svy.design(
        weight="weight",
        strata="stratum",
        psu="psu",
    )
    .group_by("region")
    .agg(est=ps.mean("income"))
)
print(result2)
print("\n" + "=" * 80 + "\n")

# Example 3: Total estimation
print("Example 3: Total hours worked (population total)")
result3 = df.svy.design(
    weight="weight",
    strata="stratum",
    psu="psu",
).agg(est=ps.total("hours_worked"))
print(result3)
print("\n" + "=" * 80 + "\n")

# Example 4: Ratio estimation
print("Example 4: Hourly wage (ratio of income to hours)")
result4 = df.svy.design(
    weight="weight",
    strata="stratum",
    psu="psu",
).agg(est=ps.ratio("income", "hours_worked"))
print(result4)
print("\n" + "=" * 80 + "\n")

# Example 5: Multiple domains
print("Example 5: Hourly wage by region")
result5 = (
    df.svy.design(
        weight="weight",
        strata="stratum",
        psu="psu",
    )
    .group_by("region")
    .agg(est=ps.ratio("income", "hours_worked"))
)
print(result5)
print("\n" + "=" * 80 + "\n")

# Example 6: With FPC (finite population correction)
# Add population size column (for demonstration)
df_fpc = df.with_columns(
    pl.when(pl.col("stratum") == "1")
    .then(1000.0)
    .when(pl.col("stratum") == "2")
    .then(1500.0)
    .when(pl.col("stratum") == "3")
    .then(1200.0)
    .when(pl.col("stratum") == "4")
    .then(1800.0)
    .when(pl.col("stratum") == "5")
    .then(1100.0)
    .alias("pop_size")
)

print("Example 6: Mean income with FPC")
result6 = df_fpc.svy.design(
    weight="weight",
    strata="stratum",
    psu="psu",
    fpc="pop_size",
).agg(est=ps.mean("income"))
print(result6)
print("\n" + "=" * 80 + "\n")

# Example 7: Compare with and without stratification
print("Example 7: Effect of stratification on standard errors")

# Without stratification
result_no_strata = df.svy.design(weight="weight").agg(est=ps.mean("income"))

# With stratification
result_with_strata = df.svy.design(
    weight="weight",
    strata="stratum",
    psu="psu",
).agg(est=ps.mean("income"))

print("Without stratification:")
print(f"  Estimate: {result_no_strata['est'][0]:.2f}")
print(f"  SE: {result_no_strata['se'][0]:.2f}")
print(f"  DF: {result_no_strata['df'][0]}")

print("\nWith stratification:")
print(f"  Estimate: {result_with_strata['est'][0]:.2f}")
print(f"  SE: {result_with_strata['se'][0]:.2f}")
print(f"  DF: {result_with_strata['df'][0]}")

# Calculate design effect
deff = (result_with_strata["se"][0] ** 2) / (result_no_strata["se"][0] ** 2)
print(f"\nDesign Effect (DEFF): {deff:.3f}")
print("\n" + "=" * 80 + "\n")

# Example 8: How to use with svy library
print("Example 8: Integration pattern with svy library")
print("""
In your svy library, you would use it like this:

class Estimation:
    def mean(self, y: str, by: str | None = None):
        import polars_svy as ps

        # Prepare data (already in Polars)
        query = self.sample.data.svy.design(
            weight=self.design.wgt,
            strata=self.design.stratum,  # Already concatenated
            psu=self.design.psu,
            fpc=self.design.pop_size,
        )

        if by is not None:
            query = query.group_by(by)

        # Execute
        result_df = query.agg(est=ps.mean(y))

        # Convert to your Estimate object
        return self._df_to_estimate(result_df, PopParam.MEAN, y, by)
""")
