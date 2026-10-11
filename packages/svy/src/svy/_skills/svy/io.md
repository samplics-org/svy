# Reading and combining data

If you have not read [SKILL.md](SKILL.md), read it first: it maps the namespaces
to files, shows how to get numbers out of results and states the rules.

svy works on polars DataFrames. Read files with svy's readers; each returns a
`pl.DataFrame`.

| Reader | Files |
| ------ | ----- |
| `svy.read_csv(path)` | CSV |
| `svy.read_parquet(path)` | Parquet |
| `svy.read_stata(path)`, `svy.read_dta(path)` | Stata `.dta`, with value labels |
| `svy.read_spss(path)`, `svy.read_sav(path)` | SPSS `.sav` |
| `svy.read_sas(path)`, `svy.read_xpt(path)` | SAS `.sas7bdat`, `.xpt` |

`svy.write_*` mirror them. `svy.create_from_csv(path)` (and `_parquet`,
`_stata`, ...) reads straight into a `Sample` with no design yet; prefer
reading, then `svy.Sample(data, design)`.

## One file

```python
import polars as pl
import svy

data = svy.read_csv("survey.csv")
sample = svy.Sample(data, svy.Design(stratum="stratum", psu="cluster", wgt="weight"))
```

## From pandas

`svy.Sample` needs polars. Convert, do not round-trip:

```python
import pandas as pd

pdf = pd.read_csv("survey.csv")
data = pl.from_pandas(pdf)
```

## Several files

Household surveys often ship a household file (with the design columns) and a
person file. Join the design columns onto the level you analyse, before
building the sample. The household id is often unique only within a cluster,
so join on both.

```python
households = svy.read_csv("households.csv")  # cluster, hh_id, region, stratum, weight
persons = svy.read_csv("persons.csv")        # cluster, hh_id, person_id, sex, age, ...

people = persons.join(households, on=["cluster", "hh_id"], how="left", validate="m:1")
assert people.height == persons.height

person_sample = svy.Sample(people, svy.Design(stratum="stratum", psu="cluster", wgt="weight"))
```

`validate="m:1"` makes polars raise if a key repeats in the household file.
To add columns to an existing sample, use `sample.wrangling.join` (see
[wrangling.md](wrangling.md)). Use the weight that belongs to the unit
analysed: a person weight for person estimates where the survey provides one.

## Example data

`svy.datasets.load(name, source="bundled")` loads a bundled example dataset
and `svy.datasets.describe(name, source="bundled").design` gives its design
as keyword arguments for `svy.Design(**...)`.
