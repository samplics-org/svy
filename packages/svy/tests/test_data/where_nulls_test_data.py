# tests/test_data/where_nulls_test_data.py
"""
Generate where_nulls_20260929.csv: a three-stratum cluster sample shaped like
an events file after a full join. People without an event have a null domain
flag (`dom`) and null analysis values (y, y2, x, z, ybin, cat, g, grp).

Key properties:
  * PSU 14 (stratum S1) has no events at all: every row is null, so dropping
    null rows would delete the PSU and change the domain SE and df.
  * Some rows have dom == 0 with y present, others dom == 0 with y null.
  * Inside the domain (dom == 1) nothing is missing.

Reference values come from R survey on subset(design, dom == 1), where NA in
the condition is FALSE and NA outside the subset is irrelevant (see
tests/svy/estimation/test_where_nulls.py). The JKn replicate weights are
exported by R into where_nulls_jkn_20260929.csv.

Run from packages/svy:
    .venv/bin/python tests/test_data/where_nulls_test_data.py
"""

from pathlib import Path

import numpy as np
import polars as pl


def main() -> None:
    rng = np.random.default_rng(20260929)

    rows = []
    for s in (1, 2, 3):
        for p in range(1, 5):
            psu = 10 * s + p
            for unit in range(5):
                u = rng.random()
                if psu == 14 or u < 0.15:
                    dom = None
                elif u < 0.40:
                    dom = 0
                else:
                    dom = 1
                has_values = dom == 1 or (dom == 0 and rng.random() < 0.5)
                z = round(float(rng.normal(10.0 + s, 3.0)), 3)
                y = round(float(5.0 + 2.0 * z + rng.normal(0.0, 4.0) + 3.0 * s), 3)
                rows.append(
                    {
                        "stratum": f"S{s}",
                        "psu": psu,
                        "unit": unit + 1,
                        "wgt": round(float(rng.uniform(20.0, 80.0)), 2),
                        "dom": dom,
                        "y": y if has_values else None,
                        "y2": round(float(0.5 * y + rng.normal(0.0, 3.0)), 3)
                        if has_values
                        else None,
                        "x": round(float(rng.uniform(1.0, 5.0)), 3) if has_values else None,
                        "z": z if has_values else None,
                        "ybin": int(rng.random() < 1.0 / (1.0 + np.exp(-(z - 12.0) / 3.0)))
                        if has_values
                        else None,
                        "cat": str(rng.choice(["a", "b", "c"])) if has_values else None,
                        "g": str(rng.choice(["g1", "g2"])) if has_values else None,
                        "grp": int(rng.choice([1, 2])) if has_values else None,
                    }
                )

    df = pl.DataFrame(rows)
    out = Path(__file__).parent / "where_nulls_20260929.csv"
    df.write_csv(out)
    n_dom = df.filter(pl.col("dom") == 1).height
    print(
        f"wrote {out.name}: {df.height} rows, {n_dom} in domain, "
        f"{df['dom'].null_count()} null dom, {df['y'].null_count()} null y"
    )


if __name__ == "__main__":
    main()
