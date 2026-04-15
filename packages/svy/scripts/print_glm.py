from pathlib import Path

import polars as pl

from svy.core.design import Design
from svy.core.enumerations import DistFamily
from svy.core.sample import Sample
from svy.regression.glm import GLMFit

DATA_PATH = Path("tests/test_data/apistrat.csv")

data = pl.read_csv(DATA_PATH)
data = data.with_columns((pl.col("api00") > 600).cast(pl.Int32).alias("y_bin"))

sample = Sample(data, Design(wgt="pw"))

print(f"Rows: {data.height}")

# ── G0: Linear GLM — baseline ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("G0: linear GLM, api00 ~ ell + meals + mobility")
print("=" * 70)
model_linear = sample.glm.fit(y="api00", x=["ell", "meals", "mobility"])
print(model_linear.fitted)

# ── G1: Logistic GLM ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("G1: logistic GLM, y_bin ~ ell + meals + mobility")
print("=" * 70)
model_logistic = sample.glm.fit(
    y="y_bin",
    x=["ell", "meals", "mobility"],
    family=DistFamily.BINOMIAL,
)
print(model_logistic.fitted)

# ── G2: Predictions — linear ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("G2: predictions from linear model")
print("=" * 70)
pred_linear = model_linear.predict(data)
print(pred_linear)

# ── G3: Predictions — logistic ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("G3: predictions from logistic model (probabilities)")
print("=" * 70)
pred_logistic = model_logistic.predict(data)
print(pred_logistic)

# ── G4: Predictions with residuals ───────────────────────────────────────────
print("\n" + "=" * 70)
print("G4: predictions with residuals")
print("=" * 70)
pred_resid = model_linear.predict(data, y_col="api00")
print(pred_resid)

# ── G5: AME — linear (should equal coefficients) ─────────────────────────────
print("\n" + "=" * 70)
print("G5: AME from linear model")
print("=" * 70)
ame_linear = model_linear.margins()
for m in ame_linear:
    print(m)

# ── G6: AME — logistic ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("G6: AME from logistic model")
print("=" * 70)
ame_logistic = model_logistic.margins()
for m in ame_logistic:
    print(m)

# ── G7: AME specific variable ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("G7: AME for meals only")
print("=" * 70)
ame_meals = model_logistic.margins(variables=["meals"])
print(ame_meals[0])

# ── G8: Predictive margins ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("G8: predictive margins at meals = 20, 50, 80")
print("=" * 70)
pm = model_logistic.margins(at={"meals": [20, 50, 80]})
print(pm)

# ── G9: set_default_print_width ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("G9: narrow print width (60 chars)")
print("=" * 70)
GLMFit.set_default_print_width(60)
print(model_linear.fitted)
GLMFit.set_default_print_width(None)  # reset
