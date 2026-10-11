# `sample.sampling` and `svy.SampleSize`: designing a survey

If you have not read [SKILL.md](SKILL.md), read it first: it maps the namespaces
to files, shows how to get numbers out of results and states the rules.

## Selecting a sample from a frame

The examples assume a cluster frame `frame` (polars DataFrame with `region`,
`cluster` and `n_households`, a size measure) and a household listing
`listing` (`cluster`, `hh_id`). Replace them with the frame's own names.

```python
import polars as pl
import svy

frame_sample = svy.Sample(frame, svy.Design(stratum="region", psu="cluster", mos="n_households"))
clusters = frame_sample.sampling.pps_sys(n=3, rstate=42)  # 3 clusters per region
```

| Method | Selection |
| ------ | --------- |
| `srs(n)` | simple random sampling, with `wr=True` for with replacement |
| `pps_sys(n)` | PPS systematic (design `mos=` is the size measure) |
| `pps_brewer(n)`, `pps_rs(n)`, `pps_murphy(n)` | PPS without replacement: Brewer, Rao-Sampford, Murphy (n=2) |
| `pps_wr(n)` | PPS with replacement |
| `add_stage(next_stage)` | chain the next stage's frame onto the selected units |

- `n` is per stratum: an int for every stratum, or `{stratum: n}`.
  `by=` selects within each level of another column.
- Pass `rstate=` (a seed or `numpy` Generator) so the selection is
  reproducible.
- The result is a new `Sample` of the selected units with
  `svy_prob_selection` and `svy_sample_weight` columns, which the design now
  uses. `prob_name=` and `wgt_name=` rename them.
- `certainty_threshold=` takes units with probability at or above it with
  certainty.

Second stage: add the listing of the selected clusters, then select within
each cluster. The final weight is the product of the stage weights.

```python
selected = clusters.data["cluster"].implode()
households = clusters.sampling.add_stage(
    next_stage=listing.filter(pl.col("cluster").is_in(selected))
).sampling.srs(n=10, by="cluster", rstate=42)
households.data.select("cluster", "hh_id", "svy_sample_weight")
```

## Sample size

```python
size = svy.SampleSize().estimate_prop(p=0.3, moe=0.05, deff=1.5, resp_rate=0.85)
size.n           # overall n, after design effect and response rate
size.to_polars()  # each step: n0, n1_deff, n2_fpc, n

allocated = size.allocate(pop_size={"North": 5000, "South": 7000, "East": 3000, "West": 4000})
allocated.to_polars()  # stratum, pop_size, n
```

| Method | Goal |
| ------ | ---- |
| `estimate_prop(p, moe)` | estimate a proportion within a margin of error |
| `estimate_mean(sigma, moe)` | estimate a mean within a margin of error |
| `compare_props(p1, p2)` | detect a difference in proportions with given `power` |
| `compare_means(mu1, mu2, sigma1)` | detect a difference in means |
| `allocate(n, pop_size=, method=)` | split n across strata: `"proportional"`, `"equal"` or `"neyman"` (needs `sigma=`) |

Every argument can be a `{stratum: value}` dict for per-stratum sizes.
`pop_size=` adds the finite population correction.
