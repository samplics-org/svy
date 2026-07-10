# svy.serialize — Design Document

**Status:** Accepted 2026-07-10
**Scope:** svy roadmap step 1 — typed result serialization.
**Companions:** [svyLab decisions/002](../../../../../../svylab/docs/decisions/002-plan-driven-analysis.md)
(§Prerequisites), [backlog_and_roadmap.md](../../../../../../svylab/docs/backlog_and_roadmap.md) step 1.

---

## 1. Problem

`svy-agents` executes generated Python code that calls `svy` and captures the
output via `print(res)`. The stdout text is then parsed for JSON or stored as a
raw string blob. This means:

- Results are **untyped** — svyLab receives formatted text, not structured data.
- Report templates can't bind to specific fields (est, se, lci, uci, deff).
- Cross-wave comparisons can't diff on real numbers.
- QA rules (CV thresholds, singleton warnings) can't operate on parsed values.
- Caching/diffing is meaningless when results are opaque strings.

The fix: `svy` owns a stable serialization contract for its result classes.
`svy-agents` carries the typed payloads instead of parsing stdout. svyLab
renders, compares, and reports from the typed data.

## 2. Design decisions

### 2.1 Central module, not per-class methods

The roadmap specifies: *"a central serialization module (payload structs +
kind-tagged union + dispatch registry), not per-class methods."* The structs
live in `svy.serialize.structs`; the serializer functions live in
`svy.serialize.serializers`. If svy's internal classes change, only the
serializer functions update — the struct contract stays stable.

### 2.2 No existing classes modified

The serialize subpackage is a **pure external consumer**. It reads public
attributes from result objects without adding methods, decorators, or fields to
them. The only existing file touched is `svy/__init__.py` (one-line import).

### 2.3 msgspec.Struct for all structs

Consistent with svy's existing patterns. Gives us `msgspec.json.encode/decode`
for free and native tagged-union support via `tag`/`tag_field`.

### 2.4 `kind`-tagged discriminated union

Each top-level struct has a `kind` field (via msgspec's `tag_field="kind"`) so
consumers can switch on the result type without introspection.

### 2.5 `schema_version` on top-level structs

Every top-level struct carries `schema_version: str = SCHEMA_VERSION` where
`SCHEMA_VERSION = "svy-result/0.1"`. Consumers can check the version to know
what fields to expect.

**Versioning policy:** bump the minor version (0.1 → 0.2) when fields are added
(additive, backward-compatible). Bump the major version (0.x → 1.0) when fields
are removed or renamed (breaking). Existing encoded JSON should always decode
into the current version if only additive changes were made; consumers should
ignore unknown fields.

### 2.6 Sub-structs are untagged

Nested data structs (`ParamEstData`, `GLMCoefData`, etc.) are plain
msgspec.Structs without `kind`/`schema_version`. Only top-level result types
carry the discriminator.

### 2.7 Excluded fields

Internal or large fields not needed for rendering, comparison, or reporting:

| Source class   | Excluded field  | Reason                          |
| -------------- | --------------- | ------------------------------- |
| `Estimate`     | `covariance`    | Large ndarray; internal         |
| `Estimate`     | `strata`        | Internal list of strata labels  |
| `Estimate`     | `singletons`    | Internal singleton tracking     |
| `Estimate`     | `domains`       | Internal domain tracking        |
| `Estimate`     | `as_factor`     | Presentation flag               |
| `GLMFit`       | `cov_matrix`    | Large ndarray; internal         |
| `GLMFit`       | `term_info`     | Internal dict                   |

### 2.8 GLM wrapper support

`GLM.fit()` returns a `GLM` object (not `GLMFit`). `serialize()` detects `GLM`
and delegates to `GLMFit` via the `.fitted` attribute.

### 2.9 Round-trip is struct → JSON → struct

The struct IS the stable representation. Consumers do not reconstruct svy
objects from structs. svyLab renders from structs; svy-agents carries them.

### 2.10 DescribeResult.items stays as list[dict]

The `DescribeItem` union has 7 variants (`DescribeContinuous`,
`DescribeDiscrete`, `DescribeDatetime`, `DescribeNominal`, `DescribeOrdinal`,
`DescribeBoolean`, `DescribeString`). Full typed sub-structs are deferred. Each
dict carries `mtype` (a StrEnum value) as an implicit discriminator.

## 3. Type conversion rules

| Source type                    | Target type           | How                          |
| ------------------------------ | --------------------- | ---------------------------- |
| `StrEnum` (PopParam, etc.)    | `str`                 | `.value`                     |
| `np.ndarray`                   | `list`                | `.tolist()`                  |
| `tuple`                        | `list`                | `list(...)`                  |
| `np.floating` (np.float64)    | `float`               | `float(...)`                 |
| `np.integer` (np.int64)       | `int`                 | `int(...)`                   |
| `Number` (= `int \| float`)   | `float`               | `float(...)` (normalize)     |
| `Category` (= `str\|int\|float\|bool`) | same  | passthrough                  |
| `dt.datetime`                  | `str`                 | `.isoformat()`               |
| `None`                         | `None`                | passthrough                  |

## 4. Public API

```python
from svy.serialize import serialize, to_json, to_dict, from_json

# Serialize a svy result object to a stable struct
data = serialize(result)          # -> ResultData

# Serialize to JSON bytes
json_bytes = to_json(result)      # -> bytes

# Serialize to a JSON-safe dict
d = to_dict(result)               # -> dict[str, Any]

# Decode JSON bytes back to a struct
data = from_json(json_bytes)      # -> ResultData
```

## 5. Struct reference

### 5.1 Top-level structs (kind-tagged, versioned)

All top-level structs are `msgspec.Struct, tag_field="kind", kw_only=True,
frozen=True` with `schema_version: str = SCHEMA_VERSION`.

#### EstimateData (`kind = "estimate"`)

Source: `svy.estimation.estimate.Estimate` (not msgspec; `__slots__`)

| Field             | Type                    | Source attribute       |
| ----------------- | ----------------------- | ---------------------- |
| `schema_version`  | `str`                   | constant               |
| `param`           | `str`                   | `param` (PopParam)     |
| `method`          | `str`                   | `method` (EstMethod)   |
| `alpha`           | `float`                 | `alpha`                |
| `estimates`       | `list[ParamEstData]`    | `estimates`            |
| `n_strata`        | `int`                   | `n_strata`             |
| `n_psus`          | `int`                   | `n_psus`               |
| `degrees_freedom` | `int`                   | `degrees_freedom`      |
| `where_clause`    | `str \| None`           | `where_clause`         |
| `q_method`        | `str`                   | `q_method` (QuantMethod)|

Excluded: `covariance`, `strata`, `singletons`, `domains`, `as_factor`.

#### TTestOneGroupData (`kind = "ttest_one_group"`)

Source: `svy.categorical.ttest.TTestOneGroup` (msgspec.Struct)

| Field          | Type                        | Source attribute |
| -------------- | --------------------------- | ---------------- |
| `schema_version` | `str`                     | constant         |
| `y`            | `str`                       | `y`              |
| `mean_h0`      | `float`                     | `mean_h0`        |
| `alternative`  | `str`                       | `alternative`    |
| `alpha`        | `float`                     | `alpha`          |
| `diff`         | `list[DiffEstData]`         | `diff`           |
| `estimates`    | `list[TtestEstData]`        | `estimates`      |
| `stats`        | `TTestStatsData \| None`    | `stats`          |

#### TTestTwoGroupsData (`kind = "ttest_two_groups"`)

Source: `svy.categorical.ttest.TTestTwoGroups` (msgspec.Struct)

| Field          | Type                        | Source attribute |
| -------------- | --------------------------- | ---------------- |
| `schema_version` | `str`                     | constant         |
| `y`            | `str`                       | `y`              |
| `groups`       | `GroupLevelsData`           | `groups`         |
| `paired`       | `bool`                      | `paired`         |
| `alternative`  | `str`                       | `alternative`    |
| `alpha`        | `float`                     | `alpha`          |
| `diff`         | `list[DiffEstData]`         | `diff`           |
| `estimates`    | `list[TtestEstData]`        | `estimates`      |
| `stats`        | `TTestStatsData \| None`    | `stats`          |

#### ChiSquareData (`kind = "chi_square"`)

Source: `svy.core.containers.ChiSquare` (msgspec.Struct)

| Field          | Type   | Source attribute |
| -------------- | ------ | ---------------- |
| `schema_version` | `str` | constant       |
| `df`           | `float` | `df`            |
| `value`        | `float` | `value`         |
| `p_value`      | `float` | `p_value`       |

#### TableData (`kind = "table"`)

Source: `svy.categorical.table.Table` (not msgspec; `__slots__`)

| Field          | Type                         | Source attribute |
| -------------- | ---------------------------- | ---------------- |
| `schema_version` | `str`                      | constant         |
| `type`         | `str`                        | `type` (TableType) |
| `rowvar`       | `str`                        | `rowvar`         |
| `colvar`       | `str \| None`                | `colvar`         |
| `alpha`        | `float`                      | `alpha`          |
| `estimates`    | `list[CellEstData]`          | `estimates`      |
| `stats`        | `TableStatsData \| None`     | `stats`          |
| `rowvals`      | `list[str \| int \| float \| bool] \| None` | `rowvals` |
| `colvals`      | `list[str \| int \| float \| bool] \| None` | `colvals` |

#### GLMFitData (`kind = "glm_fit"`)

Source: `svy.regression.glm.GLMFit` (msgspec.Struct)

| Field           | Type                  | Source attribute   |
| --------------- | --------------------- | ------------------ |
| `schema_version` | `str`                | constant           |
| `y`             | `str`                 | `y`                |
| `family`        | `str`                 | `family`           |
| `link`          | `str`                 | `link`             |
| `stats`         | `GLMStatsData`        | `stats`            |
| `coefs`         | `list[GLMCoefData]`   | `coefs`            |
| `feature_names` | `list[str]`           | `feature_names`    |

Excluded: `cov_matrix`, `term_info`.

#### GLMPredData (`kind = "glm_pred"`)

Source: `svy.regression.prediction.GLMPred` (msgspec.Struct)

| Field          | Type              | Source attribute |
| -------------- | ----------------- | ---------------- |
| `schema_version` | `str`           | constant         |
| `df`           | `float`           | `df`             |
| `alpha`        | `float`           | `alpha`          |
| `yhat`         | `list[float]`     | `yhat` (ndarray) |
| `se`           | `list[float]`     | `se` (ndarray)   |
| `lci`          | `list[float]`     | `lci` (ndarray)  |
| `uci`          | `list[float]`     | `uci` (ndarray)  |
| `residuals`    | `list[float] \| None` | `residuals` (ndarray) |

#### DescribeResultData (`kind = "describe"`)

Source: `svy.core.describe.DescribeResult` (msgspec.Struct)

| Field           | Type                | Source attribute     |
| --------------- | ------------------- | -------------------- |
| `schema_version` | `str`              | constant             |
| `weighted`      | `bool`              | `weighted`           |
| `weight_col`    | `str \| None`       | `weight_col`         |
| `drop_nulls`    | `bool`              | `drop_nulls`         |
| `top_k`         | `int`               | `top_k`              |
| `percentiles`   | `list[float]`       | `percentiles` (tuple)|
| `generated_at`  | `str`               | `generated_at` (datetime → isoformat) |
| `notes`         | `str \| None`       | `notes`              |
| `items`         | `list[dict[str, Any]]` | `items` (tuple of DescribeItem union) |

### 5.2 Sub-structs (untagged, JSON-safe)

All sub-structs are `msgspec.Struct, kw_only=True, frozen=True`.

#### ParamEstData

Source: `svy.estimation.estimate.ParamEst`

| Field      | Type                  |
| ---------- | --------------------- |
| `y`        | `str`                 |
| `est`      | `float`               |
| `se`       | `float`               |
| `cv`       | `float`               |
| `lci`      | `float`               |
| `uci`      | `float`               |
| `by`       | `list[str] \| None`   |
| `by_level` | `list[str \| int \| float \| bool] \| None` |
| `y_level`  | `str \| int \| float \| bool \| None` |
| `x`        | `str \| None`         |
| `x_level`  | `str \| int \| float \| bool \| None` |
| `deff`     | `float \| None`       |

#### DiffEstData

Source: `svy.categorical.ttest.DiffEst`

| Field      | Type                                       |
| ---------- | ------------------------------------------ |
| `y`        | `str`                                      |
| `diff`     | `float`                                    |
| `se`       | `float`                                    |
| `lci`      | `float`                                    |
| `uci`      | `float`                                    |
| `by`       | `str \| None`                              |
| `by_level` | `str \| int \| float \| bool \| None`     |

#### TtestEstData

Source: `svy.categorical.ttest.TtestEst`

| Field         | Type                                       |
| ------------- | ------------------------------------------ |
| `by`          | `str \| None`                              |
| `by_level`    | `str \| int \| float \| bool \| None`     |
| `group`       | `str \| None`                              |
| `group_level` | `str \| int \| float \| bool \| None`     |
| `y`           | `str`                                      |
| `y_level`     | `str \| int \| float \| bool \| None`     |
| `est`         | `float`                                    |
| `se`          | `float`                                    |
| `cv`          | `float`                                    |
| `lci`         | `float`                                    |
| `uci`         | `float`                                    |

#### TTestStatsData

Source: `svy.categorical.ttest.TTestStats`

| Field     | Type    |
| --------- | ------- |
| `t`       | `float` |
| `df`      | `float` |
| `p_value` | `float` |

#### GroupLevelsData

Source: `svy.categorical.ttest.GroupLevels`

| Field   | Type                                       |
| ------- | ------------------------------------------ |
| `var`   | `str`                                      |
| `levels` | `list[str \| int \| float \| bool]`      |

#### CellEstData

Source: `svy.categorical.table.CellEst`

| Field    | Type    |
| -------- | ------- |
| `rowvar` | `str`   |
| `colvar` | `str`   |
| `est`    | `float` |
| `se`     | `float` |
| `cv`     | `float` |
| `lci`    | `float` |
| `uci`    | `float` |

#### TableStatsData

Source: `svy.categorical.table.TableStats`

| Field   | Type                    |
| ------- | ----------------------- |
| `chisq` | `ChiSquareData`         |
| `f`     | `FDistData \| None`     |

Note: `ChiSquareData` is both a top-level struct (kind-tagged) and used here as
a sub-struct. When used as a sub-struct, the `kind` and `schema_version` fields
are still present but are ignored by the consumer. Alternatively, a separate
untagged `ChiSquareStatsData` could be used — but the data is identical, so we
reuse the same struct. (See §6 Open questions.)

#### FDistData

Source: `svy.core.containers.FDist`

| Field    | Type    |
| -------- | ------- |
| `df_num` | `float` |
| `df_den` | `float` |
| `value`  | `float` |
| `p_value` | `float` |

#### TDistData

Source: `svy.core.containers.TDist`

| Field     | Type    |
| --------- | ------- |
| `df`      | `float` |
| `value`   | `float` |
| `p_value` | `float` |

#### GLMCoefData

Source: `svy.regression.glm.GLMCoef`

| Field      | Type                 |
| ---------- | -------------------- |
| `term`     | `str`                |
| `est`      | `float`              |
| `se`       | `float`              |
| `lci`      | `float`              |
| `uci`      | `float`              |
| `wald`     | `TDistData \| None`  |
| `wald_adj` | `TDistData \| None`  |

#### GLMStatsData

Source: `svy.regression.glm.GLMStats`

| Field           | Type                |
| --------------- | ------------------- |
| `n`             | `int`               |
| `wald`          | `FDistData`         |
| `wald_adj`      | `FDistData`         |
| `scale`         | `float`             |
| `deviance`      | `float`             |
| `aic`           | `float \| None`     |
| `bic`           | `float \| None`     |
| `r_squared`     | `float \| None`     |
| `r_squared_adj` | `float \| None`     |
| `iterations`    | `int \| None`       |

### 5.3 Discriminated union

```python
ResultData = (
    EstimateData | TTestOneGroupData | TTestTwoGroupsData
    | ChiSquareData | TableData | GLMFitData
    | GLMPredData | DescribeResultData
)
```

## 6. Open questions

- **ChiSquareData dual use.** `ChiSquareData` is both a top-level result
  (kind-tagged, standalone) and a sub-struct inside `TableStatsData`. The
  `kind`/`schema_version` fields are redundant when nested. This is acceptable
  (small overhead, simpler code) but could be split into separate tagged and
  untagged structs if it causes confusion.

## 7. Deferred

| Item                                | Reason                                    |
| ----------------------------------- | ----------------------------------------- |
| `TTestByResult`                     | Wraps multiple TTest results by group     |
| `RankTestTwoSample` / `RankTestKSample` / `RankTestByResult` | Not in roadmap's explicit list |
| Covariance / matrix serialization   | Excluded — not needed for rendering       |
| `DescribeItem` typed sub-structs    | 7-variant union; deferred until needed    |
| svy-agents integration (step 3a)    | Separate step after this lands            |
| svyLab `SavedAnalysis.results_json` | Separate step after svy-agents            |
