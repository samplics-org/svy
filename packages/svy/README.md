# svy

Modern Python tools for **complex survey analysis**, built for real-world statistical workflows.

**svy** is a rigorously design-based yet production-oriented library for survey design, weighting, and estimation — without sacrificing transparency or scalability.

🌐 Website: [svylab.com](https://svylab.com)
📘 Documentation: [svylab.com/docs/svy](https://svylab.com/docs/svy/)
📦 Source: [github.com/samplics-org/svy/packages](https://github.com/samplics-org/svy/packages/svy)

---

## What is svy?

svy is designed for people who **actually work with complex survey data**, including national statistical offices, public health and development programs, survey methodologists, and data scientists working with complex samples.

> **Correct inference first — without hiding assumptions or sacrificing usability.**

### Validation

svy has been validated against R's `survey` package, producing numerically identical results across Taylor linearization, replication methods, and complex survey designs. See the [full comparison](https://svylab.com/learn/notes/posts/svy-vs-r-comparison/).

---

## Installation

```bash
pip install svy
```

---

## Quick Start

```python
import svy

# Load example data
hld_data = svy.datasets.load("hld_sample_wb_2023")

# Define the survey design
hld_design = svy.Design(stratum=("geo1", "urbrur"), psu="ea", wgt="hhweight")

# Create a sample object
hld_sample = svy.Sample(data=hld_data, design=hld_design)

# Estimate the mean of total expenditure
tot_exp_mean = hld_sample.estimation.mean(y="tot_exp")

print(tot_exp_mean)
```

---

## Capabilities

- **Complex survey design** — strata, clusters, weights
- **Design-based estimation** with valid standard errors
- **Replication methods** — BRR, bootstrap, jackknife, SDR
- **Categorical data analysis** — tabulation, crosstabulation, t-test, Rao-Scott test
- **Generalized linear models** — logistic, Poisson, Gamma with survey weights
- **Explicit, inspectable, reproducible outputs**
- **Built on Polars, NumPy, SciPy, msgspec and Rust**

---

## Related Packages

| Package                                     | Purpose                         | Install               |
| ------------------------------------------- | ------------------------------- | --------------------- |
| **svy**                                     | Core survey design & estimation | `pip install svy`     |
| [svy-sae](https://svylab.com/docs/svy-sae/) | Small Area Estimation           | `pip install svy-sae` |
| [svy-io](https://svylab.com/docs/svy-io/)   | SPSS / Stata / SAS I/O          | `pip install svy-io`  |

---

## Documentation

Full documentation, tutorials, and methodological notes:
👉 [svylab.com/docs/svy](https://svylab.com/docs/svy/)

---

## Feedback

- Issues: [github.com/samplics/svy/issues](https://github.com/samplics-org/svy/issues)
- Discussions: [github.com/samplics-org/svy/discussions](https://github.com/samplics-org/svy/discussions)

---

## License

MIT License — Copyright © 2026 Samplics LLC
