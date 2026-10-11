# `sample.glm`: survey regression

If you have not read [SKILL.md](SKILL.md), read it first: it maps the namespaces
to files, shows how to get numbers out of results and states the rules.

The examples assume a polars DataFrame `data` with columns `region`,
`stratum`, `cluster`, `weight`, `sex`, `age`, `income`, `educ` and
`employed` (0/1). Replace them with the survey's own names.

```python
import polars as pl
import svy

sample = svy.Sample(data, svy.Design(stratum="stratum", psu="cluster", wgt="weight"))
```

## Methods

| Method | Purpose |
| ------ | ------- |
| `fit(y, x=, family=, link=)` | fit the model; returns the fitted `GLM` |
| `to_polars()` | coefficient table |
| `term_test(term)` | design-based F test of one (multi-level) term |
| `contrast(...)` | linear combinations of coefficients, with `svy.estd` |
| `margins(variables=)`, `margins(at=)` | average marginal effects, predictive margins |
| `predict(new_data)` | predictions with intervals |
| `keys()` | coefficient names for `contrast` |

## Fitting

```python
model = sample.glm.fit(
    y="employed",
    x=["age", svy.Cat("sex", ref="Male"), svy.Cat("educ", ref="None")],
    family="binomial",
)
print(model)
model.to_polars()  # term, estimate, std_err, conf_low, conf_high, statistic, p_value, df

linear = sample.glm.fit(y="income", x=["age", svy.Cat("region")], family="gaussian")
```

- Plain names are numeric predictors. Wrap categorical ones in
  `svy.Cat(column, ref=level)`; without `ref`, the first level is the
  reference.
- `family`: `"gaussian"`, `"binomial"`, `"poisson"`, `"gamma"`,
  `"inverse_gaussian"`, `"negative_binomial"`. `link=` overrides the
  canonical link (`"logit"`, `"probit"`, `"log"`, ...).
- A binomial `y` is a 0/1 column; build it with `wrangling.mutate` first.
- `offset=` names an offset column (log exposure for Poisson rates).
- `where=` fits on a subpopulation with the full design. Rows with nulls in
  the model's columns are dropped (`drop_nulls=True` by default).
- Coefficients of a binomial or Poisson model are on the link scale;
  odds ratios are `exp(estimate)`.

## After fitting

```python
model.term_test(svy.Cat("educ")).p_value
model.keys()
model.contrast(svy.estd("age") * 10)
model.margins(variables=["age"])
model.margins(at={"age": [25, 45, 65]})
new = pl.DataFrame({"age": [30, 50], "sex": ["Female", "Male"], "educ": ["Primary", "Tertiary"]})
model.predict(new)
```

If a coefficient shows separation, svy keeps the estimate, reports NaN
statistics and records a finding. Report it; do not drop the variable
silently.
