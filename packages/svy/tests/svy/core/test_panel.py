"""Panel declarations on Design/Sample: validation and the case as fallback PSU."""

import numpy as np
import polars as pl
import pytest

from svy import Design, Sample, col, estd


def _long(n_cases=60, seed=7):
    rng = np.random.default_rng(seed)
    y1 = rng.normal(10, 2, n_cases)
    y2 = y1 + 0.5 + rng.normal(0, 0.3, n_cases)
    ids = np.arange(1, n_cases + 1)
    wide = pl.DataFrame({"id": ids, "y1": y1, "y2": y2, "w": np.ones(n_cases)})
    long = pl.DataFrame(
        {
            "id": np.concatenate([ids, ids]),
            "wave": np.repeat([1, 2], n_cases),
            "y": np.concatenate([y1, y2]),
            "w": np.ones(2 * n_cases),
        }
    )
    return wide, long


class TestFallbackPsu:
    def test_variance_psu_resolves_to_case_on_a_panel_only(self):
        assert Design(case_id="id", wave="wave").variance_psu == "id"
        assert Design(case_id="id").variance_psu is None
        assert Design(case_id="id", wave="wave", psu="psu").variance_psu == "psu"
        assert Design(wave="wave").variance_psu is None
        assert Design(case_id="id", wave="wave").is_panel

    def test_long_change_matches_wide_and_not_the_naive_se(self):
        wide, long = _long()
        wide_s = Sample(
            wide.with_columns((pl.col("y2") - pl.col("y1")).alias("d")), Design(wgt="w")
        )
        truth = wide_s.estimation.mean("d").to_polars()
        s = Sample(long, Design(case_id="id", wave="wave", wgt="w"))
        by = s.estimation.mean("y", by="wave")
        c = by.contrast(estd(2) - estd(1)).estimates[0]
        assert c.est == pytest.approx(truth["est"][0], rel=1e-12)
        assert c.se == pytest.approx(truth["se"][0], rel=1e-9)
        assert c.df == wide_s.n_records - 1
        rows = by.to_polars()
        naive = float(np.sqrt(rows["se"][0] ** 2 + rows["se"][1] ** 2))
        assert naive > 5 * c.se

    def test_bootstrap_agrees_with_taylor(self):
        _, long = _long()
        s = Sample(long, Design(case_id="id", wave="wave", wgt="w")).weighting.create_bs_wgts(
            n_reps=300, rstate=1
        )
        taylor = Sample(long, Design(case_id="id", wave="wave", wgt="w"))
        t = taylor.estimation.mean("y", by="wave").contrast(estd(2) - estd(1)).estimates[0]
        b = (
            s.estimation.mean("y", by="wave", method="replication")
            .contrast(estd(2) - estd(1))
            .estimates[0]
        )
        assert b.est == pytest.approx(t.est)
        assert b.se == pytest.approx(t.se, rel=0.15)

    def test_n_psus_counts_cases(self):
        _, long = _long(n_cases=10)
        s = Sample(long, Design(case_id="id", wave="wave", wgt="w"))
        assert s.n_psus == 10
        assert "None (variance: id)" in str(s.design)


class TestValidation:
    def test_unique_within_wave(self):
        df = pl.DataFrame({"id": [1, 1, 2], "wave": [1, 1, 2], "w": [1.0] * 3})
        with pytest.raises(ValueError, match="unique within wave"):
            Sample(df, Design(case_id="id", wave="wave", wgt="w"))
        df2 = pl.DataFrame({"id": [1, 1, 2], "wave": [1, 2, 2], "w": [1.0] * 3})
        Sample(df2, Design(case_id="id", wave="wave", wgt="w"))

    def test_unique_overall_on_a_cross_section(self):
        df = pl.DataFrame({"id": [1, 1, 2], "w": [1.0] * 3})
        with pytest.raises(ValueError, match="unique across rows"):
            Sample(df, Design(case_id="id", wgt="w"))

    def test_case_id_nulls_and_missing_columns(self):
        df = pl.DataFrame({"id": [1, None, 2], "w": [1.0] * 3})
        with pytest.raises(ValueError, match="nulls"):
            Sample(df, Design(case_id="id", wgt="w"))
        with pytest.raises(ValueError, match="not found"):
            Sample(df, Design(case_id="id", wave="nope", wgt="w"))

    def test_design_constant_within_case(self):
        df = pl.DataFrame(
            {"id": [1, 1, 2, 2], "wave": [1, 2, 1, 2], "psu": [1, 2, 3, 3], "w": [1.0] * 4}
        )
        with pytest.raises(ValueError, match="constant within case_id"):
            Sample(df, Design(case_id="id", wave="wave", psu="psu", wgt="w"))

    def test_empty_overlap_errors_when_some_case_repeats(self):
        df = pl.DataFrame({"id": [1, 1, 2, 3], "wave": [1, 2, 2, 3], "w": [1.0] * 4})
        with pytest.raises(ValueError, match="No case of wave 2"):
            Sample(df, Design(case_id="id", wave="wave", wgt="w"))

    def test_stacked_cross_section_with_unique_ids_is_not_paired(self):
        df = pl.DataFrame({"id": [1, 2, 3, 4], "wave": [1, 1, 2, 2], "w": [1.0] * 4})
        Sample(df, Design(case_id="id", wave="wave", wgt="w"))

    def test_row_index_keyword_is_gone(self):
        with pytest.raises(TypeError):
            Design(row_index="id")  # type: ignore[call-arg]

    def test_wave_alone_is_allowed(self):
        df = pl.DataFrame({"wave": [1, 2], "w": [1.0, 1.0]})
        s = Sample(df, Design(wave="wave", wgt="w"))
        assert s.design.wave == "wave" and not s.design.is_panel


class TestPlumbing:
    def test_rename_follows_case_id_and_wave(self):
        _, long = _long(n_cases=5)
        s = Sample(long, Design(case_id="id", wave="wave", wgt="w"))
        out = s.wrangling.rename_columns({"id": "pid", "wave": "t"})
        assert out.design.case_id == "pid" and out.design.wave == "t"
        out2 = s.wrangling.clean_names(letter_case="upper")
        assert out2.design.case_id == "ID"

    def test_case_id_column_protected_from_removal(self):
        _, long = _long(n_cases=5)
        s = Sample(long, Design(case_id="id", wave="wave", wgt="w"))
        with pytest.raises(Exception):
            s.wrangling.remove_columns("id")
        out = s.wrangling.remove_columns("id", force=True)
        assert out.design.case_id is None

    def test_where_on_a_wave_uses_that_wave_only(self):
        _, long = _long(n_cases=8)
        s = Sample(long, Design(case_id="id", wave="wave", wgt="w"))
        m = s.estimation.mean("y", where=col("wave") == 2).to_polars()
        assert m["est"][0] == pytest.approx(long.filter(pl.col("wave") == 2)["y"].mean())
