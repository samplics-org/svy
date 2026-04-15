import numpy as np
import polars as pl

from svy.utils.wrangling import _get_single_psu_strata, _remove_nans


data = pl.DataFrame(
    {
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        "cluster": [1, 2, 3, 1, 1, 2, 3, 2, 1, 2, 3, 3, 1, 2, 3, 1],
        "region": [
            1,
            1,
            None,
            1,
            1,
            2,
            2,
            None,
            2,
            2,
            3,
            3,
            3,
            3,
            3,
            4,
        ],
        "educ": [
            "Primary",
            "Secondary",
            "Tertiary",
            "Primary",
            "Secondary",
            "Tertiary",
            "Primary",
            "Tertiary",
            "Primary",
            "Secondary",
            "Tertiary",
            "Primary",
            "Secondary",
            "Tertiary",
            "Primary",
            "Secondary",
        ],
        "income": [
            25000.5,
            30000,
            35000,
            40000.5,
            None,
            None,
            45000,
            50000,
            55000,
            60000,
            105000,
            None,
            115000,
            120000.5,
            np.nan,
            37000,
        ],
        "age": [25, 45, 76, 22, 32, 42, None, 52, 62, 72, 44, 55, 66, None, 27, 35],
        "weight": [15.3, 11.2, 17.9, 34, 12, 13, 14, 55, 43, 12, 31.5, 23.9, 12, 11, 29, 30.0],
    }
)


def test_remove_nans():
    columns = ["income", "age", None]
    df = _remove_nans(data=data, cols_null=columns)
    assert df.shape[0] == 10


def test_get_single_psu_strata1():
    strata = _get_single_psu_strata(data=data, stratum="region", psu="cluster")
    assert strata.shape[0] == 1


def test_get_single_psu_strata2():
    data2 = pl.DataFrame(
        {
            "id": 17,
            "cluster": 3,
            "region": 15,
            "educ": "Tertiary",
            "income": 33000.3,
            "age": 35,
            "weight": 10.1,
        }
    )
    strata = _get_single_psu_strata(data=pl.concat([data, data2]), stratum="region", psu="cluster")
    assert strata.shape[0] == 2
