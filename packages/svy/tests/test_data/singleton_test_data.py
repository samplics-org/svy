from datetime import date

import numpy as np
import polars as pl


# 1. Generate Data (N=200)
# Stratum 1: 1 PSU (Singleton)
# Stratum 2: 1 PSU (Singleton)
# Stratum 3: 2 PSUs (Normal)
# Stratum 4: 2 PSUs (Normal)

data = pl.DataFrame(
    {
        "stratum": ([1] * 50) + ([2] * 50) + ([3] * 50) + ([4] * 50),
        # PSU structure:
        # 1: 101
        # 2: 201
        # 3: 301, 302
        # 4: 401, 402
        "psu": ([101] * 50 + [201] * 50 + ([301] * 25 + [302] * 25) + ([401] * 25 + [402] * 25)),
        "weight": [1.0] * 200,
        # Variable y correlated with stratum to make variance interesting
        # Stratum 1 (Singleton) has mean 10
        # Stratum 2 (Singleton) has mean 20
        # Stratum 3 has mean 30
        # Stratum 4 has mean 40
        "y": (
            np.random.normal(10, 2, 50).tolist()
            + np.random.normal(20, 2, 50).tolist()
            + np.random.normal(30, 2, 50).tolist()
            + np.random.normal(40, 2, 50).tolist()
        ),
    }
)

# Format date as YYYYMMDD
today_str = date.today().strftime("%d%m%Y")

data.write_csv(file=f"tests/data/singleton_test_{today_str}.csv")
