---
name: svy
description: Read before writing, running, reviewing or debugging any Python code that uses svy, and before analysing survey data with weights, strata, clusters (PSUs) or replicate weights in Python. Covers svy's API for reading data, declaring the design, data preparation, estimates, tables, tests, regressions, weighting, sample selection and sample size, and how to get full-precision results out.
---

# svy

## Files: open the one for each namespace you call

Before writing code that calls a namespace below, read its file from this
skill's folder; this page alone is not enough to use svy correctly. Most
analyses need [io.md](io.md), [design.md](design.md) and the file of the
method, e.g. [estimation.md](estimation.md).

| Namespace                           | What it does                                                                                 | File                             |
| ----------------------------------- | -------------------------------------------------------------------------------------------- | -------------------------------- |
| `svy.read_*`, `svy.datasets`        | Read CSV, Parquet, Stata, SPSS, SAS into polars; combine files; example data                 | [io.md](io.md)                   |
| `svy.Design`, `svy.Sample`          | Declare the design; replicate weights; singleton PSUs; `sample.check()`                      | [design.md](design.md)           |
| `sample.wrangling`                  | Create, recode, bin, rename, join and filter columns while keeping the design in step        | [wrangling.md](wrangling.md)     |
| `sample.estimation`                 | Means, totals, proportions, ratios, quantiles, correlations; domains; contrasts              | [estimation.md](estimation.md)   |
| `sample.categorical`                | One- and two-way tables with Rao-Scott tests, t-tests, rank tests                            | [categorical.md](categorical.md) |
| `sample.glm`                        | Survey regression: linear, logistic, Poisson and other GLMs                                  | [glm.md](glm.md)                 |
| `sample.weighting`                  | Nonresponse adjustment, poststratification, raking, calibration, trimming, replicate weights | [weighting.md](weighting.md)     |
| `sample.sampling`, `svy.SampleSize` | Select a sample from a frame; sample size and allocation                                     | [sampling.md](sampling.md)       |

`sample.meta` holds variable and value labels (see [wrangling.md](wrangling.md)).
Methods that change a sample (`wrangling`, `weighting`, `sampling`) return a
new `Sample`; rebind it: `sample = sample.wrangling.mutate(...)`.

## The two objects

svy analyses complex survey data. Two objects carry everything:

- `svy.Design` names the design columns: weight, strata, PSUs (clusters) or
  replicate weights.
- `svy.Sample(data, design)` binds a polars DataFrame to its design. Every
  analysis is a method on a sample namespace, and every estimate has a
  design-based SE, confidence interval and degrees of freedom.

There is no `svy.Survey`, `svy.SurveyDesign`, `svy.svydesign` or
`Design(data, ...)`: the design never holds the data.

## A complete analysis

```python
import svy

data = svy.read_csv("survey.csv")  # a polars DataFrame
sample = svy.Sample(data, svy.Design(stratum="stratum", psu="cluster", wgt="weight"))

result = sample.estimation.mean("income", by="sex")
print(result)  # a rounded display, for people

rows = result.to_polars()  # full precision, one row per sex
for row in rows.iter_rows(named=True):
    print(row["sex"], row["est"], row["se"], row["df"])

older = sample.estimation.prop("employed", where=svy.col("age") >= 65)
older.to_polars()  # one row per level of employed
```

## Getting numbers out

Results print as rounded tables. Never copy numbers from the printed table.
Every result has `.to_polars()` with full-precision columns:

| Result of                            | Columns                                                                              |
| ------------------------------------ | ------------------------------------------------------------------------------------ |
| `estimation.mean/total/ratio/median` | by columns, `est`, `se`, `lci`, `uci`, `cv`, `df`, `n`                               |
| `estimation.prop`                    | by columns, the level of `y`, then as above                                          |
| `estimation.quantile`                | adds the quantile probability                                                        |
| `categorical.tabulate`               | row and column variables, `est`, `se`, `lci`, `uci`, `n`                             |
| `categorical.ttest`                  | `diff`, `se`, `lci`, `uci`, `t`, `df`, `p_value`                                     |
| `glm.fit`                            | `term`, `estimate`, `std_err`, `conf_low`, `conf_high`, `statistic`, `p_value`, `df` |

A list of columns as `y` returns a list of results, one per column. Results
are not DataFrames: no `.est`, `.head()`, `[...]` or `.to_pandas()`; call
`.to_polars()` first.

## Rules

1. Declare the design from the survey documentation before any estimate.
   Read the documentation in full first, not a preview: the weight, stratum
   and PSU can be listed anywhere in it. Never compute a survey statistic from
   the raw frame. If the design columns are unknown, ask the user.
2. Estimate a subpopulation with `by=` or `where=`, not by filtering rows:
   filtering drops PSUs and understates the variance.
3. Report each estimate with its SE or CI and its degrees of freedom.
4. Report svy errors and warnings to the user. Do not silence them or change
   the design (singleton rule, weights) to make them go away.
5. Use svy for variances and weight adjustments, not hand-written formulas.
6. svy works on polars. Read with `svy.read_*`; convert a pandas frame with
   `pl.from_pandas(df)` before `svy.Sample`.
7. Nulls in an analysis column raise. Pass `drop_nulls=True` to exclude them
   (the design is kept) and say so.

## Common wrong names

| Not this                                               | This                                        |
| ------------------------------------------------------ | ------------------------------------------- |
| `svy.SurveyDesign(data, ...)`, `svy.Design(data, ...)` | `svy.Sample(data, svy.Design(...))`         |
| `Design(strata=, weights=, cluster=)`                  | `Design(stratum=, wgt=, psu=)`              |
| `estimation.proportion`, `estimation.percent`          | `estimation.prop`                           |
| `estimation.tabulate`, `estimation.ttest`              | `categorical.tabulate`, `categorical.ttest` |
| `result.est`, `result.estimate`, `result.head()`       | `result.to_polars()`                        |
| `sample.data.filter(...)` then estimate                | `where=svy.col(...) ...`                    |

These files match the installed svy. Where memory disagrees, trust them, then
the docstrings: `help(svy.Sample)`, `help(sample.estimation.mean)`.
