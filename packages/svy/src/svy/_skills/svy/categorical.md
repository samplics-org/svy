# `sample.categorical`: tables and tests

If you have not read [SKILL.md](SKILL.md), read it first: it maps the namespaces
to files, shows how to get numbers out of results and states the rules.

The examples assume a polars DataFrame `data` with columns `region`,
`stratum`, `cluster`, `weight`, `sex`, `age`, `income`, `educ` and
`employed` (0/1). Replace them with the survey's own names.

```python
import svy

sample = svy.Sample(data, svy.Design(stratum="stratum", psu="cluster", wgt="weight"))
```

## Methods

| Method | Purpose |
| ------ | ------- |
| `tabulate(rowvar, colvar=None)` | one-way distribution or two-way table, with Rao-Scott tests |
| `ttest(y, mean_h0=)`, `ttest(y, group=)`, `ttest(y, y_pair=)` | one-sample, two-group and paired t-tests |
| `ranktest(y, group=, score=)` | design-based rank tests (Wilcoxon, Kruskal-Wallis, van der Waerden, median) |

All take `where=`, `method="replication"`, `alpha=` and `drop_nulls=`.

## Tables

```python
one_way = sample.categorical.tabulate("educ", units="percent")
one_way.to_polars()  # educ, est, se, lci, uci, n

two_way = sample.categorical.tabulate("region", "sex", units="percent", share_of="row")
print(two_way)       # cells, then the Rao-Scott F and chi-square tests
two_way.to_polars()  # region, sex, est, se, lci, uci, n
two_way.crosstab(stats=("est", "se"))  # wide layout
two_way.stats.f.p_value  # Rao-Scott F test of independence
```

- `units`: `"proportion"` (default), `"percent"` or `"count"` (weighted
  counts).
- `share_of`: `"total"` (default, cells sum to 1), `"row"` or `"col"`.
- For a distribution within a subpopulation, use `where=`, e.g.
  `tabulate("educ", where=svy.col("age") >= 25)`.

## t-tests

```python
one_sample = sample.categorical.ttest("employed", mean_h0=0.5)
two_groups = sample.categorical.ttest("income", group="sex")
two_groups.to_polars()  # y, group_var, diff, se, lci, uci, t, df, p_value
by_region = sample.categorical.ttest("income", group="sex", by="region")
```

`alternative="less"` or `"greater"` gives one-sided tests.

## Rank tests

```python
ranks = sample.categorical.ranktest("income", group="region", score="kruskal-wallis")
ranks.to_polars()
```

`score`: `"kruskal-wallis"` (Wilcoxon for two groups), `"vander-waerden"`,
`"median"`; or `score_fn=` for a custom score.
