# svy-io

Lightweight, Pythonic file I/O for **SAS**, **SPSS**, and **Stata** powered by the ReadStat C library. Returns **Polars** DataFrames and preserves useful metadata — with a tiny, consistent API.

---

## Installation

```bash
pip install svy-io
```

**Requires:** Python, [Polars](https://pola-rs.github.io/polars/), and [pyarrow](https://arrow.apache.org/docs/python/) (installed automatically as a dependency).

---

## What you get

- **SAS**: read `.sas7bdat` and `.xpt`; write `.xpt`
- **SPSS**: read `.sav`, `.zsav`, `.por`; write `.sav` (incl. `.zsav` via `compress="zsav"`)
- **Stata**: read & write `.dta` (v8–15)
- **Polars-first**: fast DataFrames; each `read_*` returns `(df, meta)`

---

## Usage

Below are the only things you need: `read_*` and `write_*`.

### SAS

```python
# Read
from svy_io.sas import read_sas, read_xpt

df, meta = read_sas("data.sas7bdat")              # optional: catalog_path="formats.sas7bcat"
df_xpt, meta_xpt = read_xpt("transport.xpt")

# Write (XPT v8 recommended)
from svy_io.sas import write_xpt
import polars as pl

df_out = pl.DataFrame({"id": [1, 2, 3], "score": [10.5, 9.3, 8.8]})
write_xpt(df_out, "output.xpt", version=8, label="Study Data")
```

---

### SPSS

```python
# Read
from svy_io.spss import read_sav, read_por, read_spss

df_sav, meta_sav = read_sav("survey.sav")         # .zsav handled automatically
df_por, meta_por = read_por("legacy.por")
df_auto, meta_auto = read_spss("data.sav")        # auto by extension

# Write
from svy_io.spss import write_sav
import polars as pl

df = pl.DataFrame({"subject_id": [1, 2, 3], "age": [25, 30, 35], "gender": [1, 2, 1]})

write_sav(
    df,
    "out.sav",                 # or "out.zsav" with compress="zsav"
    compress="byte",
    var_labels={"age": "Age (years)", "gender": "Gender"},
    value_labels=[{"col": "gender", "labels": {"1": "Male", "2": "Female"}}],
    user_missing=[{"col": "age", "values": [-99]}]
)
```

---

### Stata

```python
# Read
from svy_io.stata import read_dta

df, meta = read_dta("data.dta")

# Write
from svy_io.stata import write_dta
import polars as pl

df_out = pl.DataFrame({"id": [1, 2, 3], "income": [50000, 62000, 58000]})

write_dta(
    df_out,
    "output.dta",
    version=15,                              # 8–15 supported
    file_label="Survey Data 2024",
    var_labels={"income": "Annual income (USD)"}
)
```

---

## Notes & tips

- All readers return a tuple: `(df: polars.DataFrame, meta: dict)`.
- If you work with dates, many `read_*` functions support `coerce_temporals=True`.
- For large files, consider `cols_skip=[...]` and/or `n_max=...` while exploring.
- Stata limits strings to 2045 bytes; SAS XPT v5 has stricter name/length limits — prefer XPT v8 unless you need legacy compatibility.

---

## Help & links

- Source & issues: https://github.com/samplics-org/svy-io
- ReadStat (upstream): https://github.com/WizardMac/ReadStat
- Polars docs: https://pola-rs.github.io/polars/
