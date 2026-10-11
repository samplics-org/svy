# `sample.estimation`: point estimates

If you have not read [SKILL.md](SKILL.md), read it first: it maps the namespaces
to files, shows how to get numbers out of results and states the rules.

The examples assume a polars DataFrame `data` with columns `region`,
`stratum`, `cluster`, `weight`, `sex`, `age`, `income`, `educ`, `employed`
(0/1) and `hours` (null when not employed). Replace them with the survey's
own names.

```python
import svy

sample = svy.Sample(data, svy.Design(stratum="stratum", psu="cluster", wgt="weight"))
```

## Methods

| Method                                | Estimates                                        |
| ------------------------------------- | ------------------------------------------------ |
| `mean(y)`                             | population mean; on a 0/1 variable, a proportion |
| `total(y)`                            | population total; on a 0/1 variable, a count     |
| `prop(y)`                             | share of each level of a categorical `y`         |
| `ratio(y, x)`                         | ratio of totals, y over x                        |
| `median(y)`, `quantile(y, p=...)`     | quantiles, Woodruff intervals                    |
| `corr(("a", "b"))`, `cov(("a", "b"))` | correlation, covariance                          |

All take `by=`, `where=`, `method=`, `alpha=` and `drop_nulls=`. `y` may be a
list of columns, which returns one result per column.

```python
mean_income = sample.estimation.mean("income")
mean_income.to_polars()  # est, se, lci, uci, cv, df, n

sample.estimation.total("employed").to_polars()   # number employed
sample.estimation.prop("educ").to_polars()        # one row per educ level
sample.estimation.quantile("income", p=(0.1, 0.5, 0.9)).to_polars()
sample.estimation.corr(("income", "age")).to_polars()
```

- `prop` CIs default to `ci_method="logit"`; `"korn-graubard"` and `"wilson"`
  are available. A `prop` on a 0/1 column returns both levels.
- `deff="wor"` adds design effects; use `deff="wr"` when the weights were
  rescaled or normalised.
- `alpha=0.1` gives 90% intervals.

## Domains

```python
by_sex = sample.estimation.mean("income", by="sex")
by_two = sample.estimation.prop("employed", by=("region", "sex"))
seniors = sample.estimation.mean("income", where=svy.col("age") >= 65)
women_by_region = sample.estimation.mean(
    "income", by="region", where=(svy.col("sex") == "Female") & (svy.col("age") >= 15)
)
```

`by=` gives one row per level (or combination of levels). `where=` restricts
to a subpopulation; it takes an expression built with `svy.col`, or a list of
them combined with and. Both keep the full design for the variance and set the
degrees of freedom from the PSUs that hold domain members. Do not filter the
data instead.

## Missing values

```python
mean_hours = sample.estimation.mean("hours", drop_nulls=True)
hourly = sample.estimation.ratio("income", "hours", drop_nulls=True)
```

Without `drop_nulls=True`, a null in `y`, `x` or a `by` column raises. With
it, those rows leave the estimate but stay in the design; `n` in the result
is the number of rows used.

## Replicate-weight variance

```python
rep_sample = sample.weighting.create_jk_wgts()
rep_mean = rep_sample.estimation.mean("income", by="sex", method="replication")
```

`method` defaults to Taylor linearisation whenever the design has strata or
PSUs, even if it also has replicate weights; pass `method="replication"` to
use them. A design with only replicate weights uses replication. The
replication df is `sample.design.rep_wgts.df`.

## Comparing estimates

Compare domains with a contrast on the result, not by subtracting estimates
and guessing the SE: domain estimates are correlated through the design.

```python
by_sex.keys()  # the names estd() accepts
gap = by_sex.contrast(svy.estd("Male") - svy.estd("Female"))
gap.to_polars()  # contrast, est, se, lci, uci, t, p_value, df

by_sex.contrast({"male/female": svy.estd("Male") / svy.estd("Female")})
```

Linear combinations are exact; ratios, `log()` and `exp()` use the delta
method. `estd` takes a domain level, a level of `y` for `prop`, or a
(domain, level) pair.
