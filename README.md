# svy

Modern Python tools for **complex survey analysis**, built for real-world statistical workflows.

**svy** is a rigorously design-based yet production-oriented ecosystem for survey design, weighting, estimation, and small area estimation — without sacrificing transparency or scalability.

🌐 Website: https://svylab.com  
📘 Documentation: https://svylab.com/docs

---

## ⚠️ Current Status (Read This First)

**The svy libraries are not yet publicly downloadable.**

This repository is intentionally public **before the code release** so that early users can:

- ask questions,
- report documentation gaps,
- suggest features,
- discuss real-world survey use cases,
- help shape stable APIs.

📘 **Documentation is live**  
🧪 **Code is under finalization**  
🐞 **Issues & discussions are open**

When the first public releases are ready, this repository will become the main code home.

---

## What is svy?

svy is designed for people who **actually work with complex survey data**, including:

- National statistical offices
- Public health and development programs
- Survey methodologists
- Data scientists working with complex samples

The guiding principle is:

> **Correct inference first — without hiding assumptions or sacrificing usability.**

svy prioritizes statistical validity while remaining compatible with modern Python workflows.

---

## Planned Capabilities

The svy ecosystem is being built to support:

- Complex survey design (strata, clusters, weights)
- Design-based estimation with valid standard errors
- Replication methods (BRR, bootstrap, jackknife)
- Small Area Estimation (area- and unit-level models)
- Explicit, inspectable, reproducible outputs
- Integration with pandas, NumPy, SciPy, and JAX-based tooling

All methods are grounded in established survey methodology.

---

## Example (Illustrative API)

The example below shows the **intended public API**.
It reflects the current design but **cannot yet be run** until the first release.

```python
import svy

design = svy.SurveyDesign(
    data=data,
    strata="stratum",
    cluster="psu",
    weights="weight"
)

result = design.mean("income")

print(result)
```

```python
import svy_sae as sae

milk = svy.load_dataset(name="milk", limit=None)

model = sae.AreaLevel(milk)

predictions = model.fh(
    y="yi",
    x=svy.Cat("MajorArea", ref=1),
    variance="variance",
    area="SmallArea",
    method="REML",
    mse="prasad_rao",
)

print(predictions)
```

No shortcuts.  
No hidden assumptions.  
Just correct survey inference.

---

## Ecosystem Packages (Upcoming)

| Package | Purpose                         | Status      |
| ------- | ------------------------------- | ----------- |
| svy     | Core survey design & estimation | In progress |
| svy-sae | Small Area Estimation           | In progress |
| svy-io  | SPSS / Stata / SAS I/O          | In progress |

Installation instructions will be added once packages are published.

---

## Documentation (Available Now)

👉 https://svylab.com/docs

Includes conceptual guides, tutorials, and methodological notes reflecting the intended stable APIs.

---

## Feedback & Early Engagement

Early feedback is strongly encouraged.

- Issues: https://github.com/samplics-org/svy/issues
- Discussions: https://github.com/samplics-org/svy/discussions

If you work with complex surveys and want to influence the design of a modern Python survey stack, this is the right place to engage.

---

## License

MIT License  
Copyright © 2025 Samplics LLC

---

**svy is built for practitioners who need statistical rigor that survives contact with reality.**
