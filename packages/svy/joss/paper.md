---
title: "svy: A Python package for end-to-end design and analysis of complex survey data"
tags:
  - Python
  - survey sampling
  - complex surveys
  - design-based inference
  - official statistics
  - sampling weights
  - calibration
  - variance estimation
  - Rust
authors:
  - name: Mamadou S. Diallo
    orcid: 0000-0002-0376-3631
    corresponding: true
    affiliation: 1
affiliations:
  - name: Samplics, United States
    index: 1
date: 22 July 2026
bibliography: paper.bib
---

# Summary

Probability-based surveys are one of the main tools used by governments, international organizations, and researchers to produce statistics that guide decision-making. For example, national household surveys inform policy on poverty, health, education, and employment for hundreds of millions of people each year. Similarly, opinion polling and market research surveys inform corporations, media, and other organizations on populations' views and preferences, and social science researchers rely on complex samples to study behaviors and living conditions. To be efficient and operationally feasible, these surveys rely on stratification, multi-stage cluster sampling, unequal probabilities of selection, and post-collection weight adjustments. As a result, standard statistical methods, which assume independent and identically distributed observations, produce incorrect variance estimates and invalid confidence intervals when applied directly to survey data [@kish1965; @lohr2021]. Design-based inference, which uses the known sampling mechanism, is the standard framework for analyzing such data.

`svy` is an open-source Python package for the design and analysis of complex survey samples. It covers the full survey workflow: sample size calculation, sample selection (simple random, systematic, and probability proportional to size methods, including multi-stage designs), weight adjustment (nonresponse adjustment, poststratification, GREG calibration, raking, and trimming), estimation of means, totals, proportions, ratios, and medians with Taylor linearization or replication-based variance (BRR, Fay-BRR, jackknife, bootstrap, and SDR), design-adjusted categorical analysis (tabulation, t-tests, rank tests, and Rao-Scott tests), and generalized linear models with design-adjusted standard errors. The package is built on Polars [@polars2025], NumPy [@harris2020], and SciPy [@virtanen2020], and its performance-critical routines are implemented in Rust.

`svy` is the successor of `samplics` [@diallo2021; @diallo2022], an earlier Python package for complex survey analysis by the same author. `samplics` has been retired, and `svy`, a ground-up redesign informed by several years of `samplics` usage, is its replacement and the recommended migration target.

# Statement of need

Software support for design-based inference is mature in several environments: the `survey` package for R [@lumley2004; @lumley2010], the survey procedures in SAS, the `svy` prefix in Stata, the SPSS Complex Samples module, and specialized tools such as SUDAAN. However, Python, despite being a leading language for data science and increasingly the primary language of analysts at national statistical offices, public health programs, and research organizations, has lacked a comprehensive and production-oriented equivalent. In practice, analysts working in Python had to move their data to R or Stata for design-based inference, or apply weighted but design-naive methods that understate uncertainty.

`samplics` [@diallo2021] was the first Python package to offer a broad set of design-based methods, and it demonstrated the demand for survey methodology tools in the Python ecosystem. Experience with `samplics` in production settings also revealed structural limitations that could not be addressed incrementally. For example, design information had to be passed manually to each function, weight adjustments were disconnected from the replicate weights, and performance was a constraint on large surveys. `svy` is a ground-up redesign built around five goals:

1. Coverage of the full survey lifecycle. A single coherent API spans planning (sample size calculation for estimation and comparison objectives), selection (including multi-stage designs where inclusion probabilities are chained across stages automatically), weighting, estimation, testing, and modeling.

2. Correctness by construction. The central `Sample` object binds the data to a `Design` object (strata, sampling units, weights, and replicate weights) once, validates the combination, and is immutable; every transformation returns a new `Sample` with the design metadata propagated automatically. Furthermore, subpopulation analysis via the `where` argument retains the full design structure for variance estimation, matching the behavior of `subset()` on a survey design object in R. This matters because filtering the rows before analysis, a common practice, generally understates the standard errors.

3. Replicate-weight fidelity. When replicate weights are created from the design weights, every subsequent adjustment (nonresponse, poststratification, calibration, raking, and trimming) is applied to the replicate weights in the same pass as the full-sample weights. Hence, replication-based standard errors reflect the variability of the entire weighting process, not just sampling variability, without additional effort from the analyst [@valliant2018; @wolter2007].

4. Performance. Variance estimation and other computational bottlenecks are implemented in Rust and distributed as the companion wheel `svy-rs`, while data operations use Polars. A full pipeline consisting of selection, weighting with 500 bootstrap replicates, and estimation runs in under a quarter of a second on a standard laptop. This makes the package practical for national-scale surveys and for simulation studies.

5. Reproducibility and automation. An `svy` analysis is expressed entirely in code: data transformations are performed through the built-in `wrangling` namespace rather than ad hoc outside the package, random operations accept explicit seeds, and the `Sample` object manages its metadata internally, i.e., the design roles, variable and category labels, and the full weight history travel with the data instead of living in external documentation. Every analytical method returns a typed result object (e.g., `Estimate`, `Table`, `TTest`, and the GLM fit objects), and the `serialize` module converts these results to stable, versioned, machine-readable structures. Hence, `svy` pipelines can be rerun, audited, and integrated into automated production systems, and the typed, validated interfaces provide a reliable contract for emerging AI-assisted workflows, where generated analysis code must be checked and reproduced rather than trusted.

The primary target audiences are survey statisticians, national statistical offices, public health and development programs, opinion polling and market research organizations, and social science researchers and data scientists working with complex samples from large survey programs such as ACS and NHANES in the United States, EU-SILC in Europe, and DHS and LSMS internationally.

# State of the field

The `survey` package for R [@lumley2004] is the reference open-source implementation of design-based inference, and it served as the primary benchmark for `svy`. Across design specification, Taylor linearization, replication methods, calibration, categorical analysis, and generalized linear models, `svy` and `survey` produce numerically identical results to at least six significant digits. The few intentional differences are documented and align with the conventions used in Stata. The full side-by-side comparison, with reproducible R and Python code, is available at <https://svylab.com/learn/notes/posts/svy-vs-r-comparison/>.

Within Python, `samplics` [@diallo2021; @diallo2022] was previously the most comprehensive option; it is now retired in favor of `svy`. Other Python tools address individual pieces of the workflow, e.g., weighting adjustments or weighted descriptive statistics, but not design-based variance estimation across the full analysis lifecycle. Relative to the R `survey` package itself, `svy` also integrates the pre-collection stages (sample size calculation and multi-stage sample selection) into the same object model used for analysis. Hence, a survey can be planned, drawn, weighted, and analyzed without leaving the package.

The statistical methodology follows the established literature: calibration and model-assisted estimation [@deville1992; @sarndal1992], replication variance estimation [@wolter2007], Rao-Scott corrections for categorical tests [@rao1981; @rao1984], and design-adjusted GLMs via linearization of the estimating equations [@binder1983].

# Software design

`svy` exposes two primary objects. `Design` is a lightweight, frozen descriptor that maps dataset columns to their roles in the sampling design, i.e., strata, primary and secondary sampling units, weights, selection probabilities, measures of size, and the replicate weight configuration. `Sample` binds a Polars `DataFrame` to a `Design`, validates the combination at construction, and organizes the functionality into focused namespaces: `sampling` for selection, `weighting`, `estimation`, `categorical`, `glm`, and `wrangling` for design-aware data transformations. The `wrangling` methods keep the design, weights, and replicate weights synchronized; for example, when a design column is renamed, the `Design` object is updated automatically.

Both objects are immutable, so analytical pipelines are auditable and reproducible by construction. Each weighting step adds a named weight column and preserves the full weight history, and there is no risk of silently overwriting the design or losing track of the active weights. Every estimation call returns a typed result object that renders as a formatted table and can be exported to a Polars `DataFrame`.

The following example gives a taste of the API. It loads one of the bundled datasets, binds the data to its sampling design, creates a literacy indicator and a simulated response status, adjusts the weights for nonresponse, trims extreme weights, and estimates the adult literacy rate by urban and rural areas with design-based standard errors. Each step returns a new `Sample` object, hence the chained style:

```python
import svy
import numpy as np

rng = np.random.default_rng(147)

# A bundled example survey (runs offline)
data = svy.datasets.load("ind_sample_wb_2023", source="bundled")
info = svy.datasets.describe("ind_sample_wb_2023", source="bundled")

# Bind the data to its sampling design
sample = svy.Sample(data, svy.Design(**info.design))

literacy_rate = (
    sample
    # 1. Wrangle: a 1/0 literacy indicator, plus a response status
    .wrangling.mutate({
        "literate": svy.when(svy.col("literacy") == "Yes").then(1)
                       .when(svy.col("literacy") == "No").then(0)
                       .otherwise(None),
        # simulated for illustration; the example data has 100% response
        "resp_status": rng.choice(
            ["respondent", "non-respondent"], p=[0.85, 0.15],
            size=sample.n_records,
        ),
    })
    # 2. Adjust the weights for nonresponse, within urban/rural classes
    .weighting.adjust(
        resp_status="resp_status",
        by="urbrur",
        resp_mapping={"rr": "respondent", "nr": "non-respondent"},
        wgt_name="nr_wgt",
    )
    # 3. Trim extreme weights to reduce variance
    .weighting.trim(upper=3.0)
    # 4. Adult (15+) literacy rate by area, with design-based SEs
    .estimation.mean(
        "literate",
        by="urbrur",
        where=svy.col("age") >= 15,
        drop_nulls=True,
    )
)

print(literacy_rate)
```

```
Estimate: MEAN (TAYLOR)
where: age >= 15

urbrur      est       se      lci      uci   cv (%)
Rural    0.7516   0.0466   0.6558   0.8474     6.20
Urban    0.9183   0.0188   0.8797   0.9570     2.05
```

The result reports, for each domain, the point estimate, its standard error, the 95% confidence interval, and the coefficient of variation, all computed under the survey design that was declared once at the start of the pipeline. The `by` argument produces domain estimates with standard errors that follow the sample domain estimation theory, and the `where` argument restricts the estimation to a subpopulation while keeping the full sample for variance estimation, rather than filtering the rows.

The package is distributed on PyPI (`pip install svy`) under the MIT license, with the Rust engine `svy-rs` shipped as a separate wheel. `svy` is also the core of a growing ecosystem of survey packages that share its design conventions: `svy-io` reads and writes SPSS, Stata, and SAS files, and `svy-sae` provides small area estimation. Additional extensions are under development. For documentation and teaching, `svy` bundles an offline-complete synthetic country derived from the World Bank synthetic census and survey data [@worldbank2023-cen; @worldbank2023-svy; @solatorio2023]. Hence, every example in the documentation is reproducible without network access.

# Research impact statement

`svy` inherits and extends the user base of `samplics`, which has been used since 2020 in applied survey work and research across public health, official statistics, and international development settings. `samplics` has been downloaded more than 278,000 times from PyPI, including an estimated 33,000 direct user installs after excluding known mirrors and continuous integration systems, and its usage has kept growing through 2026. `svy` was first released on PyPI in July 2025 and has been downloaded nearly 50,000 times in its first thirteen months, with direct user installs increasing month over month and exceeding 1,500 in July 2026 alone. With the retirement of `samplics`, `svy` is the maintained, comprehensive package for design-based survey inference in Python. The published numerical validation against the R `survey` package provides the correctness evidence that organizations require before adopting a new tool for the production of official statistics. The ambition, first stated for `samplics` [@diallo2022], remains the same: a robust, comprehensive, and easy to use Python ecosystem for survey sampling and the production of official statistics, with `svy` at its core. Documentation, tutorials, and methodological notes are maintained at <https://svylab.com/docs/svy>.

# AI usage disclosure

`samplics` was developed before the mainstream use of generative AI for software development, and none of its code was written by AI. The development of `svy` began by porting the `samplics` implementation to the new API design described in this paper: the author fully designed the software and used the `samplics` code base to build an initial working prototype of `svy`. At that point, generative AI tools were introduced to polish the code, fix bugs, and improve performance, and they played a critical role in the development of the Rust engine `svy-rs`. All AI-assisted code was reviewed and tested by the author, and the numerical validation against the R `survey` package applies to the final code regardless of how it was produced.

This manuscript was prepared with the assistance of Claude Fable 5 (Anthropic), drawing on the author's previously written manuscripts and documentation, and was revised and finalized through several rounds of review and editing by the author.

# Acknowledgements

The author thanks the World Bank Development Data Group for making the synthetic household survey data publicly available, and Thomas Lumley, whose `survey` package for R has set the standard for design-based inference in statistical software and served as the reference implementation for validating `svy`. The author is also grateful to the users of `samplics` and `svy` who reported issues, suggested features, and contributed in many ways to the development of both packages over the years.

# References
