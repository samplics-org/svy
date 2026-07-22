# svy

Modern Python tools for **complex survey analysis**, built for real-world statistical workflows.

**svy** is a rigorously design-based, production-oriented ecosystem for survey design, weighting, estimation, and small area estimation, without sacrificing transparency or scalability.

🌐 Website: https://svylab.com  
📘 Documentation: https://svylab.com/docs

---

> [!TIP]
> **Validation**: Want to assess the correctness of svy?  
> See our [comparison with R’s survey package](https://svylab.com/learn/notes/posts/svy-vs-r-comparison/), showing numerically identical results across Taylor linearization, replication methods, and complex survey designs.

## What is svy?

svy is designed for people who **actually work with complex survey data**, including:

- National statistical offices
- Public health and development programs
- Opinion polling and market research organizations
- Survey methodologists and social science researchers
- Data scientists working with complex samples

The guiding principle is:

> **Correct inference first, without hiding assumptions or sacrificing usability.**

svy prioritizes statistical validity while remaining compatible with modern Python workflows.

---

## Installation

```bash
pip install svy # svy[report] for rich outputs
```

or

```bash
uv add svy
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

- **Sample size calculation** for estimation and comparison objectives
- **Sample selection** including SRS, systematic, PPS, and multi-stage designs
- **Weight adjustments**: nonresponse, poststratification, calibration (GREG), raking, trimming
- **Design-based estimation** with valid standard errors (Taylor linearization)
- **Replication methods**: BRR, bootstrap, jackknife, SDR
- **Categorical data analysis**: tabulation, crosstabulation, t-test, rank tests, Rao-Scott test
- **Generalized linear models**: linear, logistic, Poisson, Gamma with survey weights
- **Explicit, inspectable, reproducible outputs**
- **Built on Polars, NumPy, SciPy, and msgspec**, with a Rust computational engine

All methods are grounded in established survey methodology.

---

## Ecosystem Packages

| Package                                     | Purpose                                 | Install                  |
| ------------------------------------------- | --------------------------------------- | ------------------------ |
| **svy**                                     | Core survey design & estimation         | `pip install svy`        |
| [svy-sae](https://svylab.com/docs/svy-sae/) | Small Area Estimation                   | `pip install svy-sae`    |
| [svy-io](https://svylab.com/docs/svy-io/)   | SPSS / Stata / SAS I/O                  | `pip install svy-io`     |
| [svy-rs](https://pypi.org/project/svy-rs/)  | Rust computational engine used by svy   | installed automatically  |

This repository is a monorepo: the Python package lives in [`packages/svy`](packages/svy), the Rust engine in [`packages/svy-rs`](packages/svy-rs), and the I/O library in [`packages/svy-io`](packages/svy-io).

---

## Documentation

Full documentation, tutorials, and methodological notes:
👉 https://svylab.com/docs

---

## Feedback

- Issues: https://github.com/samplics-org/svy/issues
- Discussions: https://github.com/samplics-org/svy/discussions

If you work with complex surveys and want to influence the design of a modern Python survey stack, this is the right place to engage.

---

## License

MIT License  
Copyright © 2026 Samplics LLC

---

**svy is built for practitioners who need statistical rigor that survives contact with reality.**
