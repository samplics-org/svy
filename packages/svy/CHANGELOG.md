# Changelog

All notable changes to **svy**, the Python package for design-based analysis of complex survey data — means, totals, ratios, proportions, regression, weighting, and sample selection — are recorded here. Releases follow [Semantic Versioning](https://semver.org/); the layout follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Companion packages track their own changes: [`svy-io`](../svy-io/CHANGELOG.md) (SAS/SPSS/Stata I/O) and [`svy-rs`](../svy-rs/CHANGELOG.md) (internal Rust extension).

## [Unreleased]

<!-- ### Added, ### Changed, ### Fixed, ### Deprecated, ### Removed, ### Security -->

### Added

- **`corr` and `cov`: design-based correlation and covariance** ([#124](https://github.com/samplics-org/svy/pull/124)). Both are available on `sample.estimation` with Taylor and replication variance, `by=`, `where=`, domains and `deff`.

  Neither statistic has a direction, so neither takes `y`/`x`. A call names a set of columns through one symmetric `cols` argument, and every requested pair is returned as its own row:

  ```python
  sample.estimation.corr(("income", "age"))                        # one pair
  sample.estimation.corr(["income", "age", "educ"])                # every unique pair
  sample.estimation.corr([("income", "age"), ("income", "educ")])  # exactly these
  ```

  The three spellings are disambiguated by element type and agree wherever they overlap, so a two-element list and a two-element tuple mean the same thing. A flat list yields off-diagonal pairs only — for covariance as much as correlation — so a variance is requested explicitly as `cov(("a", "a"))`.

  `kind=` selects the coefficient (`"pearson"` today) while `method=` remains the variance estimator. Since pandas spells the coefficient `method=`, that mix-up is caught by name and redirected; a recognised but unimplemented coefficient reports that it is *not supported yet* rather than *invalid*.

  Correlation is bounded, so its interval is built on Fisher's z scale and transformed back — the same move `prop` makes with a logit, and for the same reason: a symmetric Wald interval can otherwise report bounds outside [-1, 1]. `ci_method="wald"` opts out. Covariance is unbounded and takes Wald.

  Validated against R survey 4.5: covariance and its SE against `svyvar`, correlation and its SE against `svycontrast`'s own delta method over the moment means — R has no correlation SE of its own, so that agreement is an independent check of the linearization rather than a restatement of it — plus stratified and JK1 replicate fixtures. Adds `PopParam.CORR` and `PopParam.COV`.

### Changed

- **BREAKING: `deff` now names the SRS reference instead of taking a boolean.** `deff=True` and `deff=False` are rejected with a structured `MethodError` that names the replacement; the argument takes `"wor"`, `"wr"` or `None`.

  | before | now |
  | --- | --- |
  | `deff=True` | `deff="wor"` — without replacement, Kish's design effect. Same numbers as before. |
  | `deff=False` | omit the argument, or `deff=None` |
  | — | `deff="wr"` — with replacement, the square of Kish's "deft" |

  A boolean never said *which* reference it meant. Accepting it silently would leave two spellings for one thing indefinitely, and ignoring it would be worse: `Literal` is not enforced at runtime, so a string-only implementation would read `deff=True` as "off" and quietly stop reporting a design effect the caller asked for. Rejecting it loudly is the only option that cannot mislead. `"replace"` is accepted as an alias for `"wr"`, since that is R's spelling.

  The reference describes the *denominator* — what the design variance is compared against — and says nothing about how the sample was drawn; `deff="wr"` is not a claim of with-replacement sampling. The two differ by exactly the finite-population correction `1 - n/N`, so they agree closely at small sampling fractions and diverge sharply otherwise: at f = 0.8, an evaluation-study shape, they differ five-fold (5.027 against 1.005, both matching R).

  `"wor"` infers `N` from the sum of the weights, so it is meaningful only while those weights remain reciprocals of selection probabilities. After `normalize`, or to a lesser degree raking or calibration, that sum is no longer a population count and the design effect is silently wrong — svy cannot detect this in general, since weights normalized to twice the sample size pass every check and yield a plausible wrong answer. `"wr"` has no `N` in it and is unaffected. The one provable case now raises rather than returning a column of NaN: when the weights sum to no more than the sample size, the correction is zero or negative, which means either rescaled weights or a census with no sampling variance to compare against.

  The reference is recorded on the result. It appears in the printed header beside the variance method — `Estimate: MEAN (TAYLOR, deff=wr)` — is available as `Estimate.deff_ref`, and round-trips through serialization as an optional `deff_ref` field. `to_polars()` is deliberately unchanged: that frame carries no method, param or design either, and a deff column there was already provenance-free.


- **Error codes default per class.** `SingletonError` and `SvyWarningsError` fell back to the base `SvyError` code when raised without an explicit one, so two distinct failures could surface under the same identifier. Each now carries its own default ([#123](https://github.com/samplics-org/svy/pull/123)).

- **`svy.serialize` raises svy's structured errors instead of bare built-ins.** The serialize module was the last public surface still raising `TypeError`/`ValueError` where the rest of the library uses the `SvyError` hierarchy — structured errors with a stable `code`, `expected`/`got` context, and a hint. The new `SerializationError` (exported as `svy.SerializationError`) covers the serialize-specific failures, and the unfitted-model case now reuses the same `ModelError` that `GLM.predict()` already raises:

  | call | before | now | code |
  | --- | --- | --- | --- |
  | `serialize(<unsupported type>)` | `TypeError` | `SerializationError` | `UNSUPPORTED_RESULT_TYPE` |
  | `serialize(<unfitted GLM>)` | `ValueError` | `ModelError` | `MODEL_NOT_FITTED` |
  | `from_json(<no "kind">)` | `ValueError` | `SerializationError` | `PAYLOAD_MISSING_KIND` |
  | `from_json(<unknown "kind">)` | `ValueError` | `SerializationError` | `PAYLOAD_UNKNOWN_KIND` |

  **Breaking** for callers that catch the old built-ins: `SvyError` subclasses `Exception`, not `TypeError`/`ValueError`. Catch `SerializationError`/`ModelError` (or `svy.SvyError` to cover every svy failure) instead, or match on the `code` field, which is the stable contract.

## [0.23.0] — 2026-08-05

### Added

- **`sample.estimation.quantile()` — design-based quantiles with standard errors** ([#112](https://github.com/samplics-org/svy/issues/112)). Previously only the median carried a standard error; every other quantile was available as a point estimate through `describe(percentiles=)`, with no variance. `quantile()` estimates any set of probabilities under Taylor linearization or replicate weights, with `by=` domains and `where=` filters:

  ```python
  sample.estimation.quantile("income")                       # quartiles, the default
  sample.estimation.quantile("income", p=0.9)                # a single Estimate
  sample.estimation.quantile("income", p=(0.1, 0.5, 0.9), by="region")
  ```

  Standard errors follow Woodruff (1952), the construction behind R's `svyquantile`: the design-based variance of the estimated proportion `P(Y <= q)` is taken on the probability scale, and the interval comes from inverting the weighted CDF at `p ± t·se_p`. The reported `se` is the back-solved half-width.

  `p` follows the same rule as `y`: a scalar returns one `Estimate`, a sequence returns one per probability. `median()` is unchanged and remains the `p = 0.5` case reported as `PopParam.MEDIAN`.

- **`EstimateList`** — the `list` subclass now returned wherever a call estimates several things at once (`mean(["a", "b"])`, `quantile(y, p=(...))`). Printing a bare list previously showed object reprs (`[<svy.estimation.estimate.Estimate object at 0x…>, …]`); an `EstimateList` renders its members as one table and adds `to_polars()`. It *is* a `list`, so indexing, iteration, unpacking, and `isinstance(result, list)` are unaffected.

  Serializing a multi-estimate result also works for the first time — `serialize()` dispatches on exact type and previously raised `TypeError: No serializer registered for list`. The new `estimate_list` kind wraps the members, each serialized exactly as a standalone `Estimate`.

### Fixed

- **Quantile and median confidence limits now invert the CDF with the rule that located the point estimate.** The Woodruff interval endpoints were always interpolated linearly, regardless of `q_method`. R's `oldsvyquantile` hands one `method`/`f` pair to both its point `approxfun` and its endpoint `approx`; interpolating linearly while estimating with, say, `"higher"` pulls both endpoints inward and **understates** the standard error. The error grows where consecutive order statistics are far apart: on a 2000-row stratified design it was 2.0% at the median and **10.3% at `p = 0.99`**, always in the anti-conservative direction. Estimates, standard errors and both confidence limits now agree with R to machine precision at every probability tested from 0.01 to 0.99, including domains.

  This changes `median()` standard errors and confidence limits. Point estimates are unaffected.

- **The Woodruff linearization is centered on its own weighted mean**, matching how R computes the variance (`svymean(U, design)`). Only the `linear`, `middle` and `nearest` tie rules moved — up to 4e-4 relative on the confidence limits — because the `higher`/`lower` inversion snaps to an order statistic and absorbed the difference. `median()` and default-`q_method` results are unchanged by this one. See [`svy-rs`](../svy-rs/CHANGELOG.md).

## [0.22.1] — 2026-08-04

### Changed

- **Taylor estimation uses the cores it is given.** On a 10-core machine at 1M rows a single-variable mean used 1.15 cores and an 8-variable batched mean reached 1.8 of a possible 8. None of it was a rayon width problem — three pieces of redundant *serial* work sat around the fan-out, and removing them is what freed the parallelism:

  - `sample.estimation` returned a **new `Estimation` on every attribute access**, so the `_data_version`-keyed caches it carries — the factorized design arrays and prepared design info — were discarded before they could ever be reused, and every `sample.estimation.mean(...)` re-derived the whole design. The accessor is now retained per `Sample`. Derived samples are handled by an identity check: `_replace_data` forks with `copy.copy`, which carries the cached accessor over verbatim, and without the check a fork would answer with an `Estimation` still bound to its parent's data.
  - The reporting metadata on each `Estimate` (unique stratum labels, PSU count) was computed with `np.unique` over the **full-length** design arrays *per estimate*. A batched call produces one `Estimate` per variable, so an 8-variable mean did 16 full-length passes — **63% of that call's wall time** at 1M rows, all serial. These are properties of the design, not of the variable, so they are memoised on the design cache and invalidate with it.
  - The Rust kernels stopped indexing the design twice per estimate and now overlap their independent halves — see [`svy-rs`](../svy-rs/CHANGELOG.md).

  Measured on a 10-core M1 Max at 1M rows, 50 strata, 2000 PSUs:

  | case | before | after | speedup | cores used |
  | --- | ---: | ---: | ---: | --- |
  | mean, 1 variable | 70.9 ms | 22.4 ms | 3.2× | 1.15 → 1.60 |
  | total, 1 variable | 83.8 ms | 27.4 ms | 3.1× | 1.12 → 1.86 |
  | mean, 8 batched | 163.2 ms | 38.6 ms | 4.2× | 1.78 → 3.47 |
  | mean by group | 144.4 ms | 131.9 ms | 1.1× | 3.45 → 3.46 |

  Thread scaling from 1 to 10 threads went from 1.11× to 1.54× for a single variable and 1.58× to 3.03× for the batched call. A single estimate overlaps two halves rather than fanning out, so its ceiling is ~2× by construction.

  **Estimates, standard errors and degrees of freedom are unchanged** — bit-for-bit, and identical at 1, 2 and 10 threads.

  One trade-off worth knowing: retaining the accessor means its cached design arrays (~31 B/row) stay alive as long as the `Sample` rather than being freed between calls. **Peak memory is unchanged** — those arrays were rebuilt on every call before, so the high-water mark was always there (2433.8 MB before vs 2434.5 MB after, over six calls at 10M rows). What is new is that a long-lived process holding many large `Sample` objects idle now holds their design caches too.

### Fixed

- **Replication variance no longer costs O(B²) in the replicate count B.** The replicate kernels themselves were never at fault — they already do a single O(n·B) pass — but two sites in the Python prep layer tested column membership once per replicate weight column: `prepare_data` with `[c for c in rep_weight_cols if c in df.columns]`, and `Estimation._ensure_float64` with `c in data.columns and data[c].dtype != ...`. `df.columns` is a property that rebuilds the entire column-name list across the FFI boundary on every access, so B lookups cost O(B²) Python-string constructions; `_ensure_float64` also materialised a Series per column. A cProfile run at n=25,000 / B=800 put `PyDataFrame.columns` at 57% of total runtime, called ~2·B times per estimate.

  With total work n·B held constant at 20M cells — an identical 160 MB replicate weight matrix in every case — the sweep spanned 36× across cases doing identical arithmetic. Column names are now snapshotted into a set once, and dtypes read from a single `data.schema` snapshot that answers both existence and dtype:

  | n | B | before | after | speedup |
  | ---: | ---: | ---: | ---: | ---: |
  | 200,000 | 100 | 0.0067 s | 0.0059 s | 1.1× |
  | 50,000 | 400 | 0.0219 s | 0.0077 s | 2.8× |
  | 25,000 | 800 | 0.0704 s | 0.0116 s | 6.1× |
  | 12,500 | 1600 | 0.2442 s | 0.0201 s | 12.1× |

  Sweep spread drops from 36.2× to 3.4×. This matters most for bootstrap designs at B=1000+; it is negligible for BRR (32–64) and SDR/ACS (80). **Results are numerically inert** — mean, total, ratio, prop and mean-by-domain are bit-for-bit identical to the previous build.

## [0.22.0] — 2026-08-02

**svy labels values so results print nicely. That is the whole job.** Everything that made
a label list *shareable* — concepts across locales, hierarchies, the semantics of why an
answer is absent — moves to **svy-spec**, where a questionnaire gives it meaning. A
catalogue without one is machinery with no job.

If you set labels by hand, read them from a `.sav`/`.dta`, or print them on estimates,
nothing you do changes. If you used missing-value definitions, see **Removed**.

> **0.21.1 was never published.** It was numbered and its notes written on 2026-07-27,
> then further work landed before a tag was cut. Its changes ship here, and are kept
> below under their own headings so nothing is lost — but no `svy==0.21.1` exists to
> install, which is why there is no section for it.

### Added

- **`MetadataStore.update(other, *, overwrite=False)`** — merge one store into another, field by field.

  Metadata for a variable arrives from several places that each know a different part of it: measurement types inferred from the data, missing-value codes declared by the analyst, question wording carried by an instrument spec. The only way to combine two stores was `set`, which replaces a whole `VariableMeta` — so applying a spec silently cleared missing codes, because a questionnaire has no concept of them and its record carries `missing=None`. That loss is invisible until an export drops the declarations.

  `update` merges per field, which means a source can only ever *add* what it knows and can never clear what it has no opinion about. `overwrite=False` (the default) fills only gaps, keeping labels you have already chosen; `overwrite=True` lets `other` win where both are set — for a spec whose question wording should be definitive. A field `other` has not set is left alone in either mode, which is the property that makes the merge safe.

  ```python
  store.update(other)                    # fill gaps only
  store.update(other, overwrite=True)    # `other` wins on conflicts
  ```

### Changed

- **`VariableMeta.value_labels`, `ResolvedLabels.value_labels` and `Label.categories` hold
  `(code, label)` pairs**, with a mapping accepted when constructing and a dict view for
  lookup — `.labels`, `.labels`, `.label_map`. JSON object keys are always strings, so a
  `dict[Category, str]` wrote `{101: "Banjul"}` and read it back with a *string* key,
  silently, after which every join against an integer-coded column missed.

  `Label.categories` also dropped `_MissingType` from its union: msgspec refuses to decode
  any union containing a custom type, so the struct raised regardless — and nothing ever
  set the field to the sentinel, since `None` already meant "no value labels".

- **`CategoryScheme` holds one entry per code**, `SchemeEntry(code, label)`, replacing the
  `mapping` / `missing` / `missing_kinds` collections keyed by code. Three of those were
  `Category`-keyed dicts or sets that survived only through a hand-written encoder; every
  field is JSON-native now, so **the bespoke encoder is gone** — `to_bytes` is a single
  `msgspec.json.encode` — and a scheme keeps its code types by any route.

### Removed

- **`MissingDef`, and every API keyed on it**: `VariableMeta.missing`, `.na_as_level`,
  `.has_missing`, `.with_missing`; `ResolvedLabels.missing_codes`, `.is_missing`,
  `.non_missing_labels`; `MetadataStore.set_missing`, `.set_na_as_level`; the same two on
  `Sample`; the `has_missing` columns in `summary()` and `coverage()`. `svy.metadata` no
  longer exports `MissingDef` or `MissingKind` (the enum stays in `svy.core.enumerations`).

  **A 99 labelled "Refusal" is the integer 99 with the label "Refusal".** svy reads it,
  prints it, and forms no opinion. Absence is a polars null, which needs no metadata.

  The evidence for removing rather than slimming: **import never populated it.** svy-io
  surfaces `MissingRule` and `TaggedNA`, and svy's importer reads variable labels and value
  labels only — so the field's only source was a hand-set value or svy-spec's bridge, and
  its only consumer was writing it back out. Nothing ever acted on it: a declared code did
  not change an estimate then, and does not now.

  ```python
  # before
  store.set_missing("age", dont_know=[98], refused=[99])
  # after — the code is a value, and a value needs a label
  store.set_value_labels("age", {98: "Don't know", 99: "Refused"})
  ```

  To declare user-missing in an exported `.sav`, pass it at the boundary that has the
  concept: `svy_io.write_sav(..., user_missing=[...])`.

- **`locale`, everywhere.** `CategoryScheme.locale`, `SchemeRef.locale`,
  `LabellingCatalog(locale=)`/`.locale`/`.set_locale`, the `locale=` argument on `pick`,
  `list`, `add_scheme`, `make_scheme`, `MetadataStore(default_locale=)`,
  `MetadataStore.set_scheme`, and `Sample.use_scheme`.

  svy does not translate. All `locale` did was choose between two schemes registered under
  one concept — and two concept names do that with no matching algorithm. **A label is a
  string:** write `"Femme"` and svy prints `"Femme"`. svy holds one set of strings;
  choosing which set is svy-spec's job.

  ```python
  catalog.add_scheme(concept="sex_en", mapping={1: "Male", 2: "Female"})
  catalog.add_scheme(concept="sex_fr", mapping={1: "Homme", 2: "Femme"})
  ```

- **`CategoryScheme.id`** — it only ever meant `concept:locale`. The catalogue is keyed by
  concept now, one concept holds one scheme, and `pick()` is a lookup. `get`, `remove` and
  `to_label` take a concept where they took a scheme id.

- **`SchemeEntry.parent`, `.missing`, `.is_missing`**, and the lookups over them:
  `parent_of`, `children_of`, `codes_of_kind`, `kind_of`, `substantive`, `missing_codes`.
  A scheme is a code→label map. svy has no cascading selects, and why an answer is absent
  is a questionnaire fact.

- **`CategoryScheme.ordered`.** Order lives in the codes; "is this ordinal" is
  `VariableMeta.mtype`.

- **`to_label_by_concept`**, folded into `to_label` now that concept is the key.

- **171 lines with no caller anywhere** in the monorepo, in svy-spec, or in any test:
  `is_missing_value`, `recode_for_analysis`, `display_text`, `polars_mask`,
  `polars_to_analysis`, `polars_to_display` (none exported), `SchemeCatalogView`, and the
  no-op seams `validate_scheme_missing`, `normalize_scheme_missing`,
  `missing_codes_by_kind`. `labels.py` no longer imports polars.

- **`svy.questionnaire`, `MetadataStore.import_from_questionnaire`, and the `Sample(questionnaire=)` parameter.** Describing an instrument is a different job from analysing the data it produced, and svy had come to own a small piece of it: a flat question model with no notion of rosters, ordered scales, or analysis units. That work now lives in **svy-spec**, which inverts the dependency — svy no longer needs to know what a questionnaire is.

  This is a removal without a deprecation cycle, which the version number alone does not convey. `Questionnaire` was exported from `svy.questionnaire`, but never from the top-level `svy` namespace, never documented, and never used anywhere in svy beyond the one `Sample(questionnaire=)` hook — which only forwarded to `import_from_questionnaire`. A patch bump reflects a path with no known consumer; if you were importing it, pin `svy==0.21.0` and migrate at your convenience.

  To attach instrument metadata, resolve a spec, project it, and merge it in:

  ```python
  from svy_spec.bridge import to_metadata_store
  from svy_spec.resolve import resolve

  sample = svy.Sample(data, design, catalog=catalog)
  sample.meta.update(to_metadata_store(resolve(spec), catalog=catalog), overwrite=True)
  ```

  Use `update` rather than a loop over `set`: it merges per field, so a field the spec does not model — missing codes you declared, notes you added — is never cleared by applying it.

  Pass `overwrite=True` when the spec is the authority, which it is here. `Sample.__init__` runs `infer_from_dataframe`, which sets `mtype` by guessing from each column's storage type; under the default fill-only merge that guess wins and the spec's declared level never lands, so an ordered single-select stays *Numerical Discrete* instead of becoming *Categorical Ordinal*. Apply the spec first, then any adjustments of your own.

  `MetadataSource.QUESTIONNAIRE` stays — it is what the bridge sets, and it remains the right provenance for a field-collected variable.

### Fixed

- **The SPSS and SAS writers could not run at all.** `_write_spss` called
  `svy_io.write_spss` and `_write_sas` called `svy_io.write_sas`; neither name has ever
  existed. Both calls carried `# type: ignore[attr-defined]` — the type checker had said
  so and been silenced.

  The real API is `write_sav(df, path, *, var_labels, value_labels, user_missing, ...)`,
  taking labels as separate arguments rather than one `metadata` dict.

  SAS is more than a rename: ReadStat writes **SAS Transport (XPT) only** — there is no
  `sas7bdat` writer — and XPT carries no variable or value labels. `_write_sas` now writes
  XPT and **warns** that the labels did not travel, pointing at `write_spss` or
  `write_stata`. `format=` and `encoding=` are reported as ignored, and `_write_spss` loses
  its `encoding` parameter, which `write_sav` does not take.

  It survived because the only test used a stub that **defined `write_spss` and
  `write_sas` itself**. A stub that invents the interface it stands in for cannot catch a
  call to a function that is not there.

- **A labelled `Table.crosstab()` returned `None` for every estimate.** The frame's rows
  were replaced with labels while the skeleton they are joined against stayed as codes.

- **A value label did not apply when the code and the value disagreed in type.** SPSS
  stores value-label keys as strings, so a `.sav` read back gives `{"1": "Yes"}` against a
  `Float64` column, and `ResolvedLabels.display` returned the bare number. `display` now
  bridges both directions. `display_series` was never affected.

- **`ttest_to_markdown()` raised `NameError`** on any call — it referenced a
  `_stats_summary_line` that does not exist. The docstring promised a summary line the code
  never had; both are gone.

- **`SingletonResult.config` named a class that does not exist**
  (`_SingletonHandlingConfig` for `SingletonHandlingConfig`). Latent only because
  `from __future__ import annotations` left it unresolved; it would have broken on
  `get_type_hints` or a typed decode.

- **Two tests shared a name**, so the second replaced the first and one never ran. Three
  further tests had no assertions at all.

- **`ruff check` passes on `packages/svy/{src,tests}`** for the first time.

## [0.21.0] — 2026-07-24

Requires [`svy-rs`](../svy-rs/CHANGELOG.md) 0.12.0, which carries the two variance-estimation fixes below; [`svy-io`](../svy-io/CHANGELOG.md) 0.2.0 is unchanged. Grouped confidence intervals and domain design effects change in this release — point estimates and standard errors do not.

### Fixed

- **`by=` groups now use their own degrees of freedom.** A by-group is a domain, so its df must be counted on the PSUs and strata that group covers. It was instead given the df of the surrounding analysis — the full design with no filter, or the `where=` mask with one — so the same subpopulation got a different interval depending on whether it was reached through `by=` or `where=`. Confidence intervals for grouped means, totals, ratios, proportions and medians were consistently **too narrow**; the effect is negligible for groups spanning most of the sample and large for small ones (22% on a 10-record domain with 6 df rather than 56). `est`, `se`, `cv` are unaffected. Verified against R `survey` 4.5 `degf(subset(design, ...))`.
- **Design effects no longer count zero-weight rows.** Under `drop_nulls`, rows with a missing response are kept and zero-weighted rather than dropped; they were still counted in the domain SRS variance's `n`, inflating `deff` for any group containing them (~1–2% on the synthetic fixtures). Only `deff` is affected.

### Removed

- **`Estimate.degrees_freedom`.** Degrees of freedom are a per-row property — a domain or by-group is counted on its own active PSUs and strata, so grouped results legitimately carry a different df per cell. The scalar could not represent that: it was `min()` across rows, so a grouped estimate reported its *smallest* group, and for a by-group inside a domain that meant a headline df of 0. Use `ParamEst.df` (also a `df` column in `to_polars()`) for the per-row value, and `n_psus - n_strata` for the full-design df, which stays at design level under a domain filter.
- **`EstimateData.degrees_freedom`** leaves the serialized payload; `SCHEMA_VERSION` moves to `svy-result/0.2`. Strictly a field removal warrants a major bump under the policy in `serialize/DESIGN.md`; 0.2 was chosen deliberately because no known consumer binds to the field, and the reasoning is recorded there.

### Added

- **`ParamEst.df`** — the design df backing each row's t-quantile, carried through `to_polars()` and the serialized payload. It is deliberately not shown in the printed table: it is constant for most results, so a column would repeat one number down the page and widen every table.

## [0.20.1] — 2026-07-23

Patch release on top of 0.20.0; [`svy-rs`](../svy-rs/CHANGELOG.md) (0.11.0) and [`svy-io`](../svy-io/CHANGELOG.md) (0.2.0) are unchanged.

### Fixed

- **`tabulate` percent and `count_total` cells used an un-centered variance.** A cell percentage is a ratio of two estimated totals, so its variance needs the centered (Hájek) linearization. Because the internal totals flag was inferred from `sum(weights) != 1`, scaling weights to sum to 100 (`units="percent"`) or to a caller-supplied `count_total` routed the standard error through the un-centered total path, dropping the numerator/denominator covariance term. Cell SEs were inflated by a `p`-dependent amount (up to ~12% on high-proportion cells) and the confidence interval fell back to Wald, which could dip below zero. `units="proportion"`, `units="percent"`, and `count_total=N` are now the same estimator scaled by a constant and agree exactly; they match `estimation.prop` and R `survey`'s `svymean(~interaction(...))`. Bare `units="count"` is unchanged and still matches R's `svytotal`, and the Rao-Scott chi-square/F test was never affected.

## [0.20.0] — 2026-07-23

Builds on [`svy-rs`](../svy-rs/CHANGELOG.md) 0.11.0 and [`svy-io`](../svy-io/CHANGELOG.md) 0.2.0. This release lands the round 7–8 review: correctness fixes across estimation, regression, weighting, size/power, categorical, and the dataset downloader, several of which shift standard errors closer to R `survey` 4.5.

### Added

- **`RepWeights.rscales` — exact stratified-JKn variance.** `RepWeights` gains an optional `rscales` tuple (per-replicate variance coefficients, R's `scale`×`rscales` combined); `create_jk_wgts` fills it from the design's strata and estimation threads it to the Rust kernels. svy-generated JKn weights now reproduce R's `as.svrepdesign(type="JKn")` mean/total SEs and `mse=TRUE` centering to 13+ digits (df = degf). Absent `rscales`, each method keeps its global default, so user-supplied replicate weights behave exactly as before unless the file's documented `rscales` are provided.

### Fixed

- **`drop_nulls` zeroes weights instead of dropping rows** (R `na.rm=TRUE` / `subset()` semantics). `prepare_data` physically removed any row with a missing analysis value before the domain machinery ran; under standard skip patterns (`y` null outside the domain) this deleted whole PSUs and strata, understating domain SEs — 15% on the reference dataset — and corrupting df. Missing analysis values now keep their rows with main and replicate weights zeroed. Verified against R `survey` 4.5 to 13+ digits; **the R-calibrated ttest and ratio fixtures were regenerated with these semantics** (the old expectations matched R only on physically-filtered complete-case data).
- **Float-typed stratum/PSU columns are accepted.** Numeric design codes from CSVs (e.g. MEPS `VARSTR`/`VARPSU`) frequently arrive as `Float64`; the factorized-design cache cast them straight to `Categorical`, which polars forbids for floats, crashing estimation with "conversion from f64 to cat failed". Non-string, non-integer dtypes now route through `Utf8` first (float- and int-coded designs produce identical results).
- **SSU-level FPC is grouped by `(stratum, PSU)`, not PSU alone.** PSU labels are commonly reused across strata, so `build_fpc_ssu_column` merged distinct PSUs — valid designs raised `FPC_NOT_CONSTANT`, and matching `M_hi` values pooled SSU counts across strata, understating the two-stage SSU FPC.
- **`method=None` auto-detects** as documented — replication when the only variance information is replicate weights (no strata/PSU), Taylor otherwise. Previously `None` always meant Taylor, silently giving replication-only designs an SRS-like variance.
- **Core polish and API consistency (review round 7).** Replicate-weight prefix matching is strict `^prefix\d+$` (a loose `startswith` absorbed columns like `repwt_flag`; a count/`n_reps` mismatch is a typed `DimensionError`); `set_data`/`update_data`/`set_design`/`update_design` rebuild internal concat columns and re-run singleton detection + design validation instead of leaving stale state; `describe()` reports weighted std/var/quantiles (aweight convention) and computes categorical proportions over all levels; `SingletonHandling` enum values are accepted by `singleton.handle()`; `PopSize(psu=..., ssu=None)` is accepted for PSU-only FPC; `polars_mask()` is null-safe; the design-fields cache is bounded (512 entries); importing `svy` no longer replaces the host's `sys.excepthook` (Rich tracebacks install only on `SVY_RICH=1`). Deleted the unused content-based `Sample.__hash__` and the dead `_calculate_fpc`.
- **GLM design gaps (round 8).** Family-specific unit deviance (matching R `family$dev.resids`) and null deviance at the intercept-only fit; deviance/AIC follow R `survey` exactly (Lumley–Scott dAIC; `bic` is `None`); replicate-weight designs get true replicate variance instead of silently falling back to Taylor SEs; `design.pop_size` feeds per-stratum FPC into the sandwich; `Cat(ref=...)` with an absent reference level raises a typed error listing observed levels; Cat levels, the response, and the invalid-weight filter are evaluated on in-domain rows under `where=`, eliminating phantom all-zero dummies; covariate/`where`-column nulls keep-and-zero-weight (preserving PSUs in stratum centering). Validated against R `survey` 4.5 to ~1e-6 or better.
- **GLM margins rewritten on the fitted frame with delta-method SEs (round 8).** `margins` recomputed from raw sample data with ad-hoc SE formulas; it now averages over exactly the fitted rows (post null-drop, post weight filter, with the domain column), rebuilds interaction columns from counterfactual data, differentiates the full linear predictor for AME, and uses full delta-method SEs `g'V(β)g` over the design-based covariance (Stata `vce(delta)` convention). Validated against R `survey` + `marginaleffects`: points to ~1e-8, SEs to ~1e-4.
- **Weighting adjustment/calibration/trimming marshalling (round 8).** `adjust` raises a typed error on unmatched response statuses (was silently encoding them as respondents and inflating weights) and derives `respondents_only` from the encoded codes (case-insensitive); `adjust(trimming=..., update_design_wgts=False)` trims the freshly created adjusted weight instead of the caller's original; `calibrate(bounded=True)` raises `NotImplementedError` instead of being silently ignored; calibration targets are assembled as ordered per-term lists (fixing a "Design matrix label alignment mismatch" on shared numeric codes); the trim-calibrate cycle runs on arrays before writing (a strict non-convergence failure leaves data/design/replicates untouched) and honors `TrimConfig.by`/`min_cell_size`; `build_aux_matrix` raises on nulls in a continuous auxiliary instead of filling `0.0`.
- **Weighting typed errors and sorted control order** for the svy-rs 0.11.0 changes: `create_brr_wgts` pre-validates `n_reps` against the Hadamard order (`MethodError.invalid_range`); raking-bounds violations surface as `MethodError` at all four kernel call sites; `normalize()` orders control values by sorted group id, matching the kernel and `poststratify`.
- **Wrangling edge cases (round 8).** `categorize()` closes the outer bin edge (R `cut(include.lowest=TRUE)`) so boundary values no longer vanish from tabulations; `remove_columns(force=True)` cleans `design.pop_size`; a partial replicate-weight `rename_columns` raises instead of corrupting the `RepWeights` prefix; `mutate()` specs see same-call redefinitions (dependents no longer read stale values); `clean_names()` preserves internal concat columns; `filter_records()` counts and reports Kleene-null-dropped rows; `fill_null(strategy="mean")` casts integer columns to `Float64` for an exact mean; `cast(strict=True)` raises on lossy float-to-integer casts.
- **Size and power formulas (round 8).** `compare_means` is implemented (was a no-op stub returning `None`); non-inferiority sizing keeps `epsilon` signed (the old `|eps|` collapse under-sized NI designs ~5×); the one-mean two-sided clamp that produced astronomically wrong `n` is removed; one-sided power follows `sign(delta)`; pooled two-proportion variance and the optimal allocation ratio are un-inverted; the adjustment pipeline is reordered to `n0 → DEFF → FPC → nonresponse` so the FPC caps the deff-inflated size toward `pop_size`; parameter validation (p/moe/sigma/power/deff/resp_rate) raises typed `MethodError` instead of silently clipping.
- **`tabulate` count CIs use the design-df t** instead of the normal critical value; with few PSUs (df = 6) count CIs were ~20% too narrow, now matching Stata `svy: tabulate` and svy's own `estimation.total`.
- **`ranktest` with a custom `score_fn` honors `by=`** (each by-level is its own domain, returning one result per level) and **group labels reflect the levels actually tested** under `where=`/`by=` (estimates were always correct; only the reported labels were wrong).

### Security

- **Dataset downloader hardened against a hostile catalog.** Slugs from registry JSON flowed unvalidated into cache paths, glob patterns, and tempfile prefixes (a slug like `../../foo` wrote outside `~/.svy/datasets`); slugs are now allowlisted at the registry boundary and defensively in `path_for`/`clear`. Downloads without a catalog hash pin the first-seen sha256 (trust-on-first-use) and enforce it thereafter. Plain-http URLs and https→http redirect downgrades are rejected (localhost exempt for development). New error codes `DATASET_INVALID_SLUG` and `DATASET_INSECURE_URL`.

## [0.19.1] — 2026-07-21

### Added

- **Bundled offline example datasets.** `svy.datasets.load` / `catalog` / `describe` now take a `source=` argument — `"bundled"`, `"remote"`, or `"auto"` (default: remote if reachable, else bundled). A small, self-consistent synthetic survey — a sampling frame, its household census, and a two-stage sample drawn from that census (design weights sum to the census) — ships inside the wheel, so the docs and your own experiments run fully offline and reproducibly. `SVYLAB_OFFLINE=1` forces the bundled path.
- **`DatasetCatalog` type and richer `Dataset` metadata.** `catalog()` returns a `DatasetCatalog` that prints as a compact table and drills into any entry's full metadata with `.get(slug)` (also `.slugs`, `.to_polars()`). `Dataset` prints as a branded panel and gained a `notes` field documenting how a bundled subset was derived from its remote counterpart.

### Changed

- **All dataset failures route through the `DatasetError` taxonomy** with actionable messages and codes: `DATASET_NOT_BUNDLED` (lists the available bundled slugs), `DATASET_DOWNLOAD_FAILED`, and `BUNDLED_UNAVAILABLE`, alongside the existing not-found, catalog, and integrity errors.

### Fixed

- **`SvyError` panels render again.** The Rich panel path imported its renderers from a module that had since been renamed, so every error silently fell back to plain text; it now renders the branded panel. The panel also stays aligned in HTML/notebook output — the status marker is a width-1 glyph instead of a two-cell emoji — and the title, body, and metadata are spaced for readability.

## [0.19.0] — 2026-07-12

### Added

- **Batched multi-variable estimation.** `estimation.mean`, `total`, `ratio`, `prop`, and `median` now accept a list of columns and return a `list[Estimate]` (one per variable; `ratio` pairs numerator/denominator element-wise and broadcasts a scalar side). A single string still returns a single `Estimate`. For ungrouped Taylor estimation the list form shares one design build across variables and runs them in parallel — 4–13× faster than a manual loop at 1M rows depending on the estimator. `by=`, replication, `drop_nulls`, and the singleton scale double-pass transparently fall back to independent per-variable calls (identical results).
- **A variable may now appear in both `by=` and `where=`.** `where` is domain estimation (out-of-domain weights zeroed) and `by` groups on the original values — the two are orthogonal, so the previous guard forbidding overlap is removed. When a `where` predicate excludes an entire `by` level (e.g. a "don't know" code), that level is correctly absent from the results — matching R's `filter(...) %>% group_by(...)` — while every row still contributes to the shared design and degrees of freedom, so surviving groups' estimates, standard errors, and df are byte-identical. Covers Taylor and replication, all estimands, and multi-`by`.
- **Serialization for result objects.** New `svy.serialize` module provides stable, versioned serialization of every result type (estimates, t-tests, chi-square, tables, GLM fits/predictions, describe): `serialize(result)` returns a kind-tagged struct, `to_json` / `to_dict` export, and `from_json` round-trips. Payloads carry a `SCHEMA_VERSION` for forward compatibility.
- **Single-stage designs and explicit population sizes.** The design's `ssu` (second-stage unit) is now optional, so single-stage designs no longer need a placeholder. A `PopSize` type is exported for specifying finite-population sizes (FPC).

### Changed

- **Estimation now fails fast on unhandled singleton PSUs** instead of silently under-reporting the variance. Taylor estimation (`mean`, `total`, `prop`, `ratio`, `median`) raises `SingletonError` when a design has single-PSU strata and no handling strategy was chosen — matching R's `options(survey.lonely.psu = "fail")`. Pick a strategy explicitly with `sample.singleton.skip()` / `.certainty()` / `.center()` / `.scale()` / `.collapse()` / `.pool()`. Previously such strata were dropped from the variance with no error or warning.

### Fixed

- **Taylor standard errors are now bit-reproducible.** The stratified variance summed each stratum's PSUs in the iteration order of a randomized hash set, so a repeated estimate on identical data could differ in its last digits run-to-run (far below reporting precision, but not reproducible). PSUs are now summed in a canonical order, so `mean`/`total`/`ratio`/`prop`/`median` return identical standard errors across runs.
- **Stale design cache could return silently wrong results.** Estimation design caches were keyed on the identity of the data frame without holding a reference to it; after an in-place mutation freed and reallocated the frame, identity reuse could make a stale entry look current and serve design arrays for the old data. Caches are now keyed on a monotonic per-`Sample` data version bumped on every rebind, so every mutation, weighting, selection, and fork path invalidates correctly.
- **Replication-design crashes and related correctness fixes.** Clone, column keep/remove/rename, and singleton handling now work on replication designs (previously hit stale replicate-weight API usage and could crash). `Expr` now raises `TypeError` on boolean use (`and`/`or`/`not`/chained comparisons) so a malformed `where=` predicate fails loudly instead of silently filtering wrong, and derived samples deep-copy metadata/warnings/design so they no longer share mutable state with the original.

## [0.18.2] — 2026-05-20

First release tracked in this changelog. For the history prior to 0.18.2, see the [Git tags](https://github.com/samplics-org/svy/tags) and [GitHub Releases](https://github.com/samplics-org/svy/releases).

[Unreleased]: https://github.com/samplics-org/svy/compare/svy-v0.23.0...HEAD
[0.23.0]: https://github.com/samplics-org/svy/releases/tag/svy-v0.23.0
[0.22.1]: https://github.com/samplics-org/svy/releases/tag/svy-v0.22.1
[0.22.0]: https://github.com/samplics-org/svy/releases/tag/svy-v0.22.0
[0.21.0]: https://github.com/samplics-org/svy/releases/tag/svy-v0.21.0
[0.20.1]: https://github.com/samplics-org/svy/releases/tag/svy-v0.20.1
[0.20.0]: https://github.com/samplics-org/svy/releases/tag/svy-v0.20.0
[0.19.1]: https://github.com/samplics-org/svy/releases/tag/svy-v0.19.1
[0.19.0]: https://github.com/samplics-org/svy/releases/tag/svy-v0.19.0
[0.18.2]: https://github.com/samplics-org/svy/releases/tag/svy-v0.18.2
