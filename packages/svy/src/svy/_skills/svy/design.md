# Designs and samples

If you have not read [SKILL.md](SKILL.md), read it first: it maps the namespaces
to files, shows how to get numbers out of results and states the rules.

The examples assume the user's survey is a polars DataFrame `data` with
columns `region`, `stratum`, `cluster`, `weight`, `sex`, `age` and `income`.
Replace them with the survey's own names.

## Declare a design

```python
import svy

design = svy.Design(stratum="stratum", psu="cluster", wgt="weight")
sample = svy.Sample(data, design)
print(sample)
```

- `stratum` and `psu` take a column name or a tuple of names; a tuple means the
  combination defines the stratum or PSU, e.g. `stratum=("region", "area")`.
- PSU codes are taken within stratum (R's `nest=TRUE`): the same code in two
  strata is two PSUs.
- `ssu=` names the second-stage unit, `pop_size=` the population size column
  for a finite population correction, `wr=True` declares with-replacement
  sampling.
- A design with only `wgt=` is a weighted, unclustered, unstratified sample.
- `svy.Sample` accepts a polars `DataFrame` or `LazyFrame`, not pandas.
  Reading files and combining a household and a person file is in
  [io.md](io.md).
- `svy.Sample(data)` with no design treats the data as an unweighted simple
  random sample. Use it for data preparation, not for survey estimates.

## Check the data against the design

```python
report = sample.check()
print(report)
```

`check()` reports null, zero or negative weights, PSU codes reused across
strata, singleton strata and duplicated case ids. `sample.n_strata`,
`sample.n_psus` and `sample.n_records` give the counts.

## Replicate weights supplied with the data

Many public files ship replicate weights instead of strata and PSUs. Declare
them with the method the documentation names; take `df` and `fay_coef` from
the documentation when it gives them.

```python
rep_design = svy.Design(
    wgt="weight",
    rep_wgts=svy.BrrWgts(prefix="repwgt", n_reps=80, fay_coef=0.5, df=39),
)
```

`svy.JackknifeWgts`, `svy.BootstrapWgts` and `svy.SdrWgts` follow the same
pattern. The replicate columns are `prefix` followed by 1..n_reps. Estimate
with `method="replication"` (see [estimation.md](estimation.md)); creating
replicate weights from strata and PSUs is in [weighting.md](weighting.md).

## Singleton PSUs

A stratum with one PSU has no within-stratum variance, so svy raises instead of
guessing. The fix is a methodological choice: report the singleton strata to
the user and let them choose a rule.

```python
singleton_rules = [
    "center",                                   # R lonely.psu="adjust"
    "scale",                                    # R "average"
    "skip",                                     # R "remove" and "certainty"
    "self_representing",                        # the PSU becomes a stratum
    svy.Singleton("collapse", within="region"), # merge into a stratum of the same region
    "pool",                                     # singletons pooled into one stratum
]
centered = svy.Sample(
    data, svy.Design(stratum="stratum", psu="cluster", wgt="weight", singleton="center")
)
```

On an existing sample: `sample.update_design(singleton="center")`.
`sample.singletons` lists them; `sample.domain_singletons(...)` finds strata
that become singletons inside a domain.

## Samples are immutable by default

Methods that change a sample return a new `Sample`; the original is
unchanged. Chain them, or rebind the name. `inplace=True` exists but chains
read better.

```python
sample = sample.wrangling.mutate({"income_k": svy.col("income") / 1000})
```

Change the design with `sample.update_design(...)`, never by assigning to
private attributes.
