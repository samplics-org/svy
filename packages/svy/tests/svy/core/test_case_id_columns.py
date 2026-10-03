"""Design.case_id on several columns (cluster, household, line), as a CSPro
record id: every use must agree with the same id built as one column."""

import numpy as np
import polars as pl
import pytest

import svy

from svy import Design, Sample, SvyUserWarning, col, estd
from svy.errors import MethodError
from svy.serialize import from_json, serialize, to_design, to_json


ID = ("clu", "hh", "ln")


def _people(n_clu=6, n_hh=3, n_ln=2, seed=3) -> pl.DataFrame:
    """One wave of people; no single id column is unique on its own."""
    rng = np.random.default_rng(seed)
    rows = [(c, h, ln) for c in range(1, n_clu + 1) for h in range(1, n_hh + 1) for ln in (1, 2)]
    clu, hh, ln = (list(x) for x in zip(*rows[: n_clu * n_hh * n_ln]))
    n = len(clu)
    return pl.DataFrame(
        {
            "clu": clu,
            "hh": hh,
            "ln": ln,
            "y": rng.normal(10, 2, n),
            "g": rng.choice(["A", "B"], n),
            "resp": ["rr"] * n,
            "w": rng.uniform(1, 3, n),
        }
    )


def _panel() -> pl.DataFrame:
    w1 = _people().with_columns(wave=pl.lit(1))
    w2 = w1.filter(~((pl.col("clu") == 6) & (pl.col("hh") == 3))).with_columns(
        wave=pl.lit(2),
        y=pl.col("y") + 0.5,
        resp=pl.when((pl.col("clu") == 1) & (pl.col("ln") == 2))
        .then(pl.lit("nr"))
        .otherwise(pl.lit("rr")),
    )
    return pl.concat([w1, w2])


def _with_pid(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pid=pl.concat_str(ID, separator="-"))


class TestDesign:
    def test_list_normalized_to_tuple(self):
        d = Design(case_id=["clu", "hh", "ln"])
        assert d.case_id == ID
        assert Design(case_id="id").case_id == "id"
        assert d.update(case_id=["clu", "ln"]).case_id == ("clu", "ln")

    @pytest.mark.parametrize("bad", [[], ["clu", ""], ["clu", 1]])
    def test_rejects_bad_lists(self, bad):
        with pytest.raises((TypeError, ValueError), match="case_id"):
            Design(case_id=bad)

    def test_variance_psu_falls_back_to_the_columns(self):
        d = Design(case_id=list(ID), wave="wave")
        assert d.variance_psu == ID
        assert d.is_panel

    def test_columns_repr_and_code(self):
        d = Design(case_id=list(ID), wgt="w")
        assert d.columns()[:3] == list(ID)
        assert "case_id=('clu', 'hh', 'ln')" in repr(d)
        assert eval(d._to_code(), {"svy": svy}) == d
        assert "(clu, hh, ln)" in str(d)

    def test_serialization_round_trip(self):
        d = Design(case_id=list(ID), wave="wave", wgt="w")
        data = serialize(d)
        assert data.case_id == list(ID)
        assert to_design(from_json(to_json(d))) == d


class TestValidation:
    def test_unique_on_the_columns_together(self):
        s = Sample(_people(), Design(case_id=list(ID), wgt="w"))
        assert s.design.case_id == ID
        with pytest.raises(ValueError, match="must be unique"):
            Sample(_people(), Design(case_id=["clu", "hh"], wgt="w"))

    def test_duplicates_reported_as_tuples(self):
        df = pl.concat([_people(), _people().head(1)])
        with pytest.raises(ValueError, match=r"duplicated: \[\(1, 1, 1\)\]"):
            Sample(df, Design(case_id=list(ID), wgt="w"))

    def test_missing_column_and_nulls_named(self):
        with pytest.raises(ValueError, match=r"not found in data: \['line'\]"):
            Sample(_people(), Design(case_id=["clu", "hh", "line"], wgt="w"))
        df = _people().with_columns(hh=pl.when(pl.col("clu") == 2).then(None).otherwise("hh"))
        with pytest.raises(ValueError, match=r"contains nulls in \['hh'\]"):
            Sample(df, Design(case_id=list(ID), wgt="w"))

    def test_design_constant_within_case_on_a_panel(self):
        df = _panel().with_columns(
            g=pl.when((pl.col("wave") == 2) & (pl.col("clu") == 1))
            .then(pl.lit("Z"))
            .otherwise("g")
        )
        with pytest.raises(ValueError, match=r"constant within case_id.*\(1, 1, 1\)"):
            Sample(df, Design(case_id=list(ID), wave="wave", stratum="g", wgt="w"))


class TestSameAsOneColumn:
    def test_panel_change_variance(self):
        df = _with_pid(_panel())
        multi = Sample(df, Design(case_id=list(ID), wave="wave", wgt="w"))
        single = Sample(df, Design(case_id="pid", wave="wave", wgt="w"))
        assert multi.n_psus == single.n_psus == 36
        a = multi.estimation.mean("y", by="wave").contrast(estd(2) - estd(1)).estimates[0]
        b = single.estimation.mean("y", by="wave").contrast(estd(2) - estd(1)).estimates[0]
        assert (a.est, a.se, a.df) == pytest.approx((b.est, b.se, b.df), rel=1e-12)

    def test_combine_samples_panel(self):
        df = _with_pid(_panel())
        waves = [df.filter(pl.col("wave") == k).drop("wave") for k in (1, 2)]

        def stack(case_id):
            return svy.combine_samples(
                [Sample(w, Design(wgt="w")) for w in waves], kind="panel", case_id=case_id
            )

        multi, single = stack(list(ID)), stack("pid")
        assert multi.design.case_id == ID
        m = multi.estimation.mean("y", by="wave").contrast(estd(2) - estd(1)).estimates[0]
        s = single.estimation.mean("y", by="wave").contrast(estd(2) - estd(1)).estimates[0]
        assert (m.est, m.se) == pytest.approx((s.est, s.se), rel=1e-12)

    def test_combine_samples_takes_the_declared_columns(self):
        waves = [_people().with_columns(y=pl.col("y") + k) for k in (0, 1)]
        out = svy.combine_samples(
            [Sample(w, Design(case_id=list(ID), wgt="w")) for w in waves], kind="panel"
        )
        assert out.design.case_id == ID

    def test_combine_samples_checks_every_column(self):
        waves = [_people(), _people().drop("ln")]
        with pytest.raises(MethodError, match=r"\['clu', 'hh', 'ln'\] is missing from sample"):
            svy.combine_samples(
                [Sample(w, Design(wgt="w")) for w in waves], kind="panel", case_id=list(ID)
            )

    def test_combine_samples_rejects_a_bad_case_id(self):
        with pytest.raises(MethodError, match="case_id"):
            svy.combine_samples(
                [Sample(_people(), Design(wgt="w"))] * 2, kind="panel", case_id=["clu", 3]
            )

    def test_lag(self):
        df = _with_pid(_panel())
        multi = Sample(df, Design(case_id=list(ID), wave="wave", wgt="w")).wrangling.lag("y")
        single = Sample(df, Design(case_id="pid", wave="wave", wgt="w")).wrangling.lag("y")
        assert multi.data["y_lag1"].to_list() == single.data["y_lag1"].to_list()
        assert multi.data["y_lag1"].null_count() == _people().height

    def test_panel_adjust(self):
        df = _with_pid(_panel())

        def adjusted(case_id):
            s = Sample(df, Design(case_id=case_id, wave="wave", wgt="w"))
            return s.weighting.adjust(
                "resp", cells="g", where=col("wave") == 2, respondents_only=False
            ).data["nr_wgt"]

        assert adjusted(list(ID)).to_list() == pytest.approx(adjusted("pid").to_list())


class TestPlumbing:
    def test_rename_follows_every_column(self):
        s = Sample(_people(), Design(case_id=list(ID), wgt="w"))
        out = s.wrangling.rename_columns({"hh": "household"})
        assert out.design.case_id == ("clu", "household", "ln")
        assert s.wrangling.clean_names(letter_case="upper").design.case_id == ("CLU", "HH", "LN")

    def test_removing_one_column_drops_the_whole_id(self):
        s = Sample(_people(), Design(case_id=list(ID), wgt="w"))
        with pytest.warns(SvyUserWarning, match=r"\[DESIGN_FIELDS_REMOVED\]"):
            out = s.wrangling.remove_columns("ln", force=True)
        assert out.design.case_id is None

    def test_to_code_rebuilds_the_design(self):
        s = Sample(_people(), Design(case_id=list(ID), wgt="w"))
        assert "case_id=('clu', 'hh', 'ln')" in s.to_code()
