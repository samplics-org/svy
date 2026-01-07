# svy

Modern Python tools for **complex survey analysis**, built for real-world statistical workflows.

**svy** provides a rigorously design-based yet production-ready ecosystem for survey design, weighting, estimation, and small area estimation — without sacrificing transparency or scalability.

🌐 Website: https://svylab.com  
📘 Documentation: https://svylab.com/docs

---

## What is svy?

svy is designed for people who _actually use survey data_:

- National statistical offices
- Public health and development programs
- Survey methodologists
- Data scientists working with complex samples

It emphasizes **correct inference first**, while remaining usable in modern Python pipelines.

---

## Key Capabilities

- Complex survey design (strata, clusters, weights)
- Design-based estimation with valid standard errors
- Replication methods (BRR, bootstrap, jackknife)
- Small Area Estimation (area- and unit-level models)
- Clean, explicit, inspectable outputs
- Integration with pandas, NumPy, SciPy, and JAX-based tooling

---

## Packages

| Package | Purpose                           | Install             |
| ------- | --------------------------------- | ------------------- |
| svy     | Core survey design and estimation | pip install svy     |
| svy-sae | Small Area Estimation             | pip install svy-sae |
| svy-io  | SPSS / Stata / SAS I/O            | pip install svy-io  |

Extras:

pip install svy[sae]  
pip install svy[io]  
pip install svy[all]

---

## Minimal Example

```python
import svy

design = svy.SurveyDesign(
    data=data,
    strata="stratum",
    cluster="psu",
    weights="weight"
)

result = design.mean("income")

print(result.estimate)
print(result.se)
```

No shortcuts.  
No hidden assumptions.  
Just correct survey inference.

---

## Documentation

Full documentation, tutorials, and theory notes are available at:

👉 https://svylab.com/docs

---

## Project Status

- Core APIs are stable and used in production
- Advanced features are evolving
- Backward compatibility is taken seriously

svy supersedes earlier survey tooling efforts and is the focus of all new development.

---

## Contributing

Feedback, issues, and discussions are welcome:

- Issues: https://github.com/samplics-org/svy/issues
- Discussions: https://github.com/samplics-org/svy/discussions

---

## License

MIT License  
Copyright © 2025 Samplics LLC

---

svy is built for practitioners who need **statistical rigor that survives contact with reality**.
