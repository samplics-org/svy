# svy-rs

Efficient survey statistics for Polars DataFrames.

## Installation

```bash
pip install maturin
maturin develop --release
```

## Usage

```python
import polars as pl
import svy_rs as ps

# Create sample data
df = pl.DataFrame({
    "income": [50000, 60000, 70000, 80000, 45000, 55000],
    "region": ["A", "A", "B", "B", "C", "C"],
    "weight": [1.5, 2.0, 1.8, 2.2, 1.6, 1.9],
    "stratum": ["1", "1", "2", "2", "3", "3"],
    "psu": ["1", "2", "3", "4", "5", "6"],
})

# Estimate mean with Taylor series variance
result = (
    df
    .svy.design(
        weight="weight",
        strata="stratum",
        psu="psu",
    )
    .group_by("region")
    .agg(est=ps.mean("income"))
)

print(result)
# ┌────────┬────────┬──────────┬────────┬──────────┬─────┬──────┐
# │ region ┆ y      ┆ est      ┆ se     ┆ var      ┆ df  ┆ n    │
# ├────────┼────────┼──────────┼────────┼──────────┼─────┼──────┤
# │ A      ┆ income ┆ 55714.29 ┆ 3571.4 ┆ 12755102 ┆ 1   ┆ 2    │
# │ B      ┆ income ┆ 75555.56 ┆ 4444.4 ┆ 19753086 ┆ 1   ┆ 2    │
# │ C      ┆ income ┆ 50588.24 ┆ 2941.2 ┆ 8650602  ┆ 1   ┆ 2    │
# └────────┴────────┴──────────┴────────┴──────────┴─────┴──────┘
```

## Features

### Implemented

- ✅ Taylor series linearization variance
- ✅ Stratified designs
- ✅ Clustered (PSU) designs
- ✅ Finite population correction (FPC)
- ✅ Domain estimation (groupby)
- ✅ Mean estimation
- ✅ Total estimation
- ✅ Ratio estimation

### Coming Soon

- ⏳ Replicate weight methods (BRR, Jackknife, Bootstrap)
- ⏳ Proportion estimation with categorical variables
- ⏳ Quantile estimation
- ⏳ Singleton PSU handling options
- ⏳ Calibration and raking

## API

### Design Specification

```python
df.svy.design(
    weight="weight_col",      # Required: sampling weights
    strata="stratum_col",     # Optional: stratification variable
    psu="psu_col",           # Optional: primary sampling unit
    fpc="fpc_col",           # Optional: finite population correction
    singleton_method="certainty"  # How to handle singleton PSUs
)
```

**Note**: If you have multiple strata columns (e.g., `["geo1", "urbrur"]`), concatenate them first:

```python
df = df.with_columns(
    pl.concat_str(["geo1", "urbrur"], separator="||").alias("stratum")
)
```

### Estimators

```python
# Mean
ps.mean("column_name")

# Total
ps.total("column_name")

# Ratio
ps.ratio(numerator="income", denominator="hours")
```

### Domain Estimation

```python
# Group by domain variable
result = (
    df.svy.design(weight="wgt", strata="stratum", psu="psu")
    .group_by("region")  # Domain variable
    .agg(est=ps.mean("income"))
)
```

## How It Works

`svy-rs` implements survey variance estimation using Taylor series linearization:

1. **Calculate linearization variables** (scores):
   - Mean: `z_i = (w_i / N) * (y_i - ȳ)`
   - Total: `z_i = w_i * y_i`
   - Ratio: `z_i = (w_i / X̄) * (y_i - R̂ * x_i)`

2. **Compute variance** using stratified cluster sampling formulas:
   - Sum variance contributions from each stratum
   - Account for clustering within PSUs
   - Apply finite population correction if specified

3. **All computation happens in Rust** for maximum performance

## Performance

Compared to pure Python/NumPy implementations, `svy-rs` provides:

- **Zero-copy operations** - data stays in Apache Arrow format
- **Parallel processing** - leverages Rayon for multi-threading
- **Efficient groupby** - native Polars operations
- **SIMD vectorization** - optimized linear algebra operations

## Integration with svy

`svy-rs` is designed to be used as a computational backend for the `svy` library:

```python
# In svy library
class Estimation:
    def mean(self, y: str, by: str | None = None):
        import svy_rs as ps

        # Prepare composite columns if needed
        data = self._prepare_data()

        # Call svy-rs
        result = (
            data.svy.design(
                weight=self.design.wgt,
                strata=self.design.stratum_combined,
                psu=self.design.psu,
            )
            .group_by(by) if by else lambda x: x
            .agg(est=ps.mean(y))
        )

        # Convert to Estimate object
        return self._df_to_estimate(result)
```

## Development

```bash
# Build in development mode
maturin develop

# Build optimized release
maturin develop --release

# Run tests
pytest tests/

# Format code
cargo fmt
black svy_rs/
```

## License

MIT
