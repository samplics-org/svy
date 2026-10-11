# `sample.weighting`: adjusting weights

If you have not read [SKILL.md](SKILL.md), read it first: it maps the namespaces
to files, shows how to get numbers out of results and states the rules.

Each method returns a new `Sample` with a new weight column (`wgt_name=`) and,
by default, makes that column the design weight. Steps therefore chain: each
one adjusts the weight the previous step produced. Replicate weights, when the
design has them, are adjusted the same way unless `ignore_reps=True`.

The examples assume the user's survey is a polars DataFrame `data` with
columns `region`, `stratum`, `cluster`, `weight`, `sex`, `age` and `resp`
(`"respondent"`, `"nonrespondent"`, `"ineligible"`). Replace them with the
survey's own names and the population figures with real ones.

```python
import svy

sample = svy.Sample(data, svy.Design(stratum="stratum", psu="cluster", wgt="weight"))
```

## Methods

| Method | Purpose |
| ------ | ------- |
| `adjust(resp_status, cells=)` | nonresponse adjustment within cells |
| `poststratify(controls, cells=)` | match population counts of one classification |
| `rake(controls=)` | match several margins iteratively |
| `calibrate(controls=)` | GREG calibration to categorical and numeric totals |
| `calibrate_matrix(aux_vars=, controls=)` | GREG against an auxiliary matrix you built |
| `build_aux_matrix(x=)`, `control_aux_template(x=)` | the matrix and the empty `controls` that `calibrate` uses |
| `controls_margins_template(margins=)` | the empty `controls` that `rake` uses |
| `standardize(cells, shares=)` | reweight to a standard composition |
| `normalize(controls)` | rescale weights to a chosen total |
| `trim(upper=, lower=)` | cap extreme weights and redistribute the excess |
| `create_jk_wgts`, `create_bs_wgts`, `create_brr_wgts`, `create_sdr_wgts` | replicate weights |

## Order of steps

The usual order: base weights, replicate weights (if you create them), then
nonresponse adjustment, then poststratification, calibration or raking to
population figures, then trimming. Create replicate weights before the
adjustments so every replicate goes through the same steps.

```python
sample = sample.weighting.create_bs_wgts(n_reps=200, rstate=42)
```

## Nonresponse

```python
sample = sample.weighting.adjust(
    resp_status="resp",
    cells="region",
    resp_mapping={"rr": "respondent", "nr": "nonrespondent", "in": "ineligible"},
    wgt_name="nr_wgt",
)
```

`resp_mapping` maps svy's statuses (`"rr"` respondent, `"nr"` nonrespondent,
`"in"` ineligible, `"uk"` unknown) to the codes in the data. Weight is moved
from nonrespondents to respondents within each cell. With
`respondents_only=True` (the default) only respondents are kept afterwards.

## Population figures

```python
sample = sample.weighting.poststratify(
    controls={"North": 14_000, "South": 15_000, "East": 13_000, "West": 12_000},
    cells="region",
    wgt_name="ps_wgt",
)

sample = sample.weighting.rake(
    controls={
        "region": {"North": 14_000, "South": 15_000, "East": 13_000, "West": 12_000},
        "sex": {"Female": 28_000, "Male": 26_000},
    },
    wgt_name="rk_wgt",
)

sample = sample.weighting.calibrate(
    controls={svy.Cat("sex"): {"Female": 28_000, "Male": 26_000}, "age": 2_500_000},
    wgt_name="cal_wgt",
)
```

- `poststratify` matches one classification exactly; `shares=` takes
  proportions instead of counts.
- `rake` matches several margins iteratively; `bounds=(low, high)` limits the
  adjustment factors.
- `calibrate` (GREG) matches totals of categorical (`svy.Cat`) and numeric
  auxiliaries at once; `bounds=` gives bounded calibration.
- Control keys must match the data's levels exactly. A level present in the
  data but missing from the controls raises; fix the controls, not the data.
- Non-convergence raises by default (`on_nonconvergence="error"`). Report it;
  do not switch to `"ignore"` without the user's agreement.

## Trimming

```python
sample = sample.weighting.trim(
    upper=svy.Threshold("median") + 3.5 * svy.Threshold("iqr"),
    by="region",
    wgt_name="trim_wgt",
)
```

`svy.Threshold` composes statistics of the weights (`"median"`, `"mean"`,
`"sd"`, `"iqr"`, `svy.Threshold.quantile(0.99)`). The trimmed excess is
redistributed to the other weights (`redistribute=True`). To trim while
raking or calibrating, pass `trimming=svy.TrimConfig(...)` to that step.

## Rescaling

```python
normalized = sample.weighting.normalize(controls=1_000, wgt_name="norm_wgt")
```

Normalising changes totals. Use it only when the user asks for it, and use
`deff="wr"` for design effects afterwards.

## Creating replicate weights

| Method | Use |
| ------ | --- |
| `create_jk_wgts()` | delete-one-PSU jackknife |
| `create_bs_wgts(n_reps=..., rstate=...)` | Rao-Wu bootstrap |
| `create_brr_wgts()` | BRR; strata with more than two PSUs are paired into variance strata |
| `create_sdr_wgts(n_reps=...)` | successive difference, for systematic samples |

Pass `rstate=` to any random method so the replicates are reproducible.

## After weighting

```python
print(sample.check())
print(sample.design.wgt)
```

Report the final weight column, its range and the Kish design effect from
`check()`, and estimate with that sample.
