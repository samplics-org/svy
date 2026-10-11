# `sample.wrangling`: preparing variables

If you have not read [SKILL.md](SKILL.md), read it first: it maps the namespaces
to files, shows how to get numbers out of results and states the rules.

Wrangling methods change the sample's data and keep the design in step: a
renamed weight column stays the design weight, a dropped design column is
refused. Each returns a new `Sample`; rebind it.

The examples assume a polars DataFrame `data` with columns `region`,
`stratum`, `cluster`, `hh_id`, `weight`, `sex`, `age`, `income`, `educ`,
`employed` and `hours`. Replace them with the survey's own names.

```python
import polars as pl
import svy

sample = svy.Sample(data, svy.Design(stratum="stratum", psu="cluster", wgt="weight"))
```

## Methods

| Method | Purpose |
| ------ | ------- |
| `mutate({name: expr})` | create or replace columns from expressions, scalars or arrays |
| `recode(cols, {new: [old, ...]})` | map old values to new ones |
| `categorize(col, bins=, labels=)` | bin a numeric column |
| `cast(cols, dtype)` | change dtypes |
| `fill_null(cols, value=)` | fill nulls |
| `top_code`, `bottom_code`, `bottom_and_top_code` | cap values |
| `rename_columns({old: new})`, `clean_names()` | rename |
| `keep_columns`, `select`, `remove_columns`, `drop` | choose columns |
| `join(other, on=)` | left-join columns from another frame or sample |
| `filter_records(where)` | keep rows; not for subpopulation estimates |
| `distinct`, `order_by`, `with_row_index` | deduplicate, sort, index rows |
| `apply_labels(labels=, categories=)` | variable and value labels |
| `rename_rep_wgts`, `lag` | rename replicate weight sets; panel lags |

## Derived variables

Expressions use `svy.col`, `svy.when` and `svy.lit` (polars expressions).

```python
sample = sample.wrangling.mutate(
    {
        "income_k": svy.col("income") / 1000,
        "senior": svy.when(svy.col("age") >= 65).then(1).otherwise(0),
        "female": (svy.col("sex") == "Female").cast(pl.Int8),
    }
)
```

A 0/1 indicator makes `estimation.mean` a proportion and is what
`glm.fit(family="binomial")` needs. Keep missing values missing:
`svy.when(svy.col("income").is_null()).then(None).otherwise(...)`.

## Recoding and binning

```python
sample = sample.wrangling.recode(
    "educ",
    {"Basic": ["None", "Primary"], "Higher": ["Secondary", "Tertiary"]},
    into="educ2",
)
sample = sample.wrangling.categorize(
    "age", bins=[0, 15, 25, 45, 65, 120], labels=["0-14", "15-24", "25-44", "45-64", "65+"],
    right=False, into="age_group",
)
sample = sample.wrangling.top_code({"income": 50_000}, into="income_tc")
```

Name the output with `into=`; without it, `recode` and `categorize` write a
new `svy_<col>_recoded` or `svy_<col>_categorized` column, and
`replace=True` overwrites the original.

## Joining

```python
regions = pl.DataFrame({"region": ["North", "South", "East", "West"], "zone": ["A", "A", "B", "B"]})
sample = sample.wrangling.join(regions, on="region")
```

A left join: every record stays, in order; unmatched records get nulls and a
warning. A key that repeats in `other` is refused.

## Filtering

```python
adults = sample.wrangling.filter_records(svy.col("age") >= 18)
```

Filtering removes records from the design. Use it only for records outside
the survey's population altogether. For "among adults", "in the North" and
similar, estimate with `where=` or `by=` (see [estimation.md](estimation.md)).

## Labels

```python
sample = sample.wrangling.apply_labels(
    labels={"employed": "Employed last week"},
    categories={"employed": {1: "Yes", 0: "No"}},
)
sample.meta.var_labels["employed"]
```

Value labels show in printed results; `to_polars()` keeps the codes.
`svy.read_stata` and `svy.read_spss` bring the file's labels in.
