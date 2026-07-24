# tests/svy/estimation/test_degrees_of_freedom.py
"""Design degrees of freedom, case by case (issue #3).

Every expected value here is countable by hand from a 12-row toy design, and
every one was confirmed against R survey 4.5 before being written down::

    des <- svydesign(ids=~psu, strata=~stratum, weights=~w, data=d, nest=TRUE)
    degf(des); degf(subset(des, <domain>))

R's rule ([surveyrep.R] ``degf.survey.design2``) is::

    inset <- weights(design, "sampling") != 0
    length(unique(cluster[inset])) - length(unique(strata[inset]))

i.e. active PSUs minus active strata, counted only over rows that carry weight.
svy implements the equivalent ``sum_h (n_PSU_h - 1)`` in
``svy-rs/src/estimation/taylor.rs::degrees_of_freedom_in_domain``.

The toy design: 2 strata x 3 PSUs x 2 rows.

    stratum A: A1 A2 A3      stratum B: B1 B2 B3

so the full design has 6 - 2 = 4 df. The domain indicator columns each isolate
one edge case, and ``grp`` cuts across PSUs to exercise by-groups:

    grp g1 = {A1, A2, B1}    grp g2 = {A3, B2, B3}

These assertions read ``ParamEst.df`` directly rather than inverting a
t-quantile from the CI, which is what made this rule untestable before.
"""

import polars as pl
import pytest

from svy import Design, RepWeights, Sample, col


N_REPS = 16
JKN_FIXTURE = "rep_domain_df_jkn_20260724.csv"


@pytest.fixture
def toy() -> pl.DataFrame:
    rows = []
    for stratum in ("A", "B"):
        for k in (1, 2, 3):
            for row in (1, 2):
                rows.append(
                    {
                        "stratum": stratum,
                        "psu": f"{stratum}{k}",
                        "row": row,
                        "w": 10.0 + k,
                        "y": float(10 * k + row + (0 if stratum == "A" else 5)),
                    }
                )
    return pl.DataFrame(rows).with_columns(
        # whole PSU A3 outside the domain
        pl.when(pl.col("psu") == "A3").then(0).otherwise(1).alias("d_drop_psu"),
        # only one ROW of A3 outside — the PSU itself survives
        pl.when((pl.col("psu") == "A3") & (pl.col("row") == 1))
        .then(0)
        .otherwise(1)
        .alias("d_partial_psu"),
        # stratum B reduced to a single PSU
        pl.when((pl.col("stratum") == "B") & (pl.col("psu") != "B1"))
        .then(0)
        .otherwise(1)
        .alias("d_one_psu_in_b"),
        # stratum B entirely outside the domain
        pl.when(pl.col("stratum") == "B").then(0).otherwise(1).alias("d_no_stratum_b"),
        # a grouping that cuts across PSUs and strata
        pl.when(pl.col("psu").is_in(["A1", "A2", "B1"]))
        .then(pl.lit("g1"))
        .otherwise(pl.lit("g2"))
        .alias("grp"),
    )


def _df_of(sample: Sample, **kwargs) -> int:
    est = sample.estimation.mean("y", **kwargs)
    return est.estimates[0].df


def _df_by_group(sample: Sample, **kwargs) -> dict[str, int]:
    est = sample.estimation.mean("y", **kwargs)
    return {p.by_level[0]: p.df for p in est.estimates}


# ---------------------------------------------------------------------------
# Design variants — what the df is counted from
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "design_kwargs, expected, rationale",
    [
        (dict(wgt="w"), 11, "no design info: every row is its own PSU, n - 1"),
        (dict(wgt="w", stratum="stratum"), 10, "strata only: n - n_strata"),
        (dict(wgt="w", psu="psu"), 5, "clusters only: n_PSU - 1"),
        (dict(wgt="w", stratum="stratum", psu="psu"), 4, "n_PSU - n_strata"),
    ],
)
def test_full_design_df(toy, design_kwargs, expected, rationale):
    """df counts sampling units, not observations — n is 12 in every case."""
    assert _df_of(Sample(toy, Design(**design_kwargs))) == expected, rationale


def test_df_is_unchanged_by_fpc(toy):
    """An FPC changes the variance, not the df."""
    d = toy.with_columns(pl.lit(1000).alias("N"))
    sample = Sample(d, Design(wgt="w", stratum="stratum", psu="psu", pop_size="N"))
    assert _df_of(sample) == 4


# ---------------------------------------------------------------------------
# Domains via where= — R's subset() semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "domain, expected, rationale",
    [
        ("d_drop_psu", 3, "A3 gone entirely: 5 PSUs - 2 strata"),
        ("d_partial_psu", 4, "A3 keeps one row, so it still counts: 6 - 2"),
        ("d_one_psu_in_b", 2, "stratum B down to one PSU, contributing 0: 4 - 2"),
        ("d_no_stratum_b", 2, "stratum B gone, so it stops being counted too: 3 - 1"),
    ],
)
def test_where_domain_df(toy, domain, expected, rationale):
    """A domain that removes whole PSUs loses the df they carried.

    ``d_partial_psu`` is the control: dropping rows from a PSU without emptying
    it must NOT change the df, which is what distinguishes counting PSUs from
    counting rows.
    """
    sample = Sample(toy, Design(wgt="w", stratum="stratum", psu="psu"))
    assert _df_of(sample, where=col(domain) == 1) == expected, rationale


@pytest.mark.parametrize(
    "null_psu_rows, expected",
    [(("A3", None), 3), (("A3", 1), 4)],
)
def test_zero_weight_rows_do_not_count(toy, null_psu_rows, expected):
    """drop_nulls zero-weights rows; those rows must not prop up the df.

    Nulling all of A3 drops it to 3 df exactly as excluding it by ``where=``
    does; nulling one of its two rows leaves 4. Matches R's degf() under
    ``subset(des, !is.na(y))``.
    """
    psu, row = null_psu_rows
    cond = pl.col("psu") == psu
    if row is not None:
        cond = cond & (pl.col("row") == row)
    d = toy.with_columns(pl.when(cond).then(None).otherwise(pl.col("y")).alias("y"))
    sample = Sample(d, Design(wgt="w", stratum="stratum", psu="psu"))
    est = sample.estimation.mean("y", drop_nulls=True)
    assert est.estimates[0].df == expected


# ---------------------------------------------------------------------------
# Domains via by= — must agree with where=
# ---------------------------------------------------------------------------


def test_by_group_df_is_per_group(toy):
    """Each by-group is a domain and gets its own df.

    g1 = {A1, A2, B1} and g2 = {A3, B2, B3} both span 3 PSUs across 2 strata,
    so both are 1 — not the full design's 4.
    """
    sample = Sample(toy, Design(wgt="w", stratum="stratum", psu="psu"))
    assert _df_by_group(sample, by="grp") == {"g1": 1, "g2": 1}


def test_by_group_df_respects_an_outer_where(toy):
    """by= and where= compose: the cell is the intersection.

    Within ``d_drop_psu`` (A3 gone), g2 collapses to {B2, B3} — 2 PSUs in a
    single stratum, so still 1 df.
    """
    sample = Sample(toy, Design(wgt="w", stratum="stratum", psu="psu"))
    got = _df_by_group(sample, by="grp", where=col("d_drop_psu") == 1)
    assert got == {"g1": 1, "g2": 1}


@pytest.mark.parametrize("group", ["g1", "g2"])
def test_by_and_where_agree_cell_for_cell(toy, group):
    """The invariant that makes this whole class of bug impossible.

    A cell reached through ``by=`` and the same cell reached through an
    equivalent ``where=`` must agree on df, estimate, and CI. This is what
    ``by=`` failed before it threaded the group mask into the df computation:
    it reported the surrounding analysis df for every group.
    """
    sample = Sample(toy, Design(wgt="w", stratum="stratum", psu="psu"))
    by_cell = {p.by_level[0]: p for p in sample.estimation.mean("y", by="grp").estimates}[group]
    where_cell = sample.estimation.mean("y", where=col("grp") == group).estimates[0]

    assert by_cell.df == where_cell.df
    assert by_cell.est == pytest.approx(where_cell.est, rel=1e-12)
    assert by_cell.se == pytest.approx(where_cell.se, rel=1e-12)
    assert by_cell.lci == pytest.approx(where_cell.lci, rel=1e-12)
    assert by_cell.uci == pytest.approx(where_cell.uci, rel=1e-12)


@pytest.mark.parametrize("estimator", ["mean", "total", "ratio"])
def test_by_and_where_agree_across_estimators(toy, estimator):
    """The per-group df reaches total and ratio, not just mean."""
    sample = Sample(toy, Design(wgt="w", stratum="stratum", psu="psu"))
    kw = {"x": "w"} if estimator == "ratio" else {}
    by_est = getattr(sample.estimation, estimator)("y", by="grp", **kw)
    by_df = {p.by_level[0]: p.df for p in by_est.estimates}
    for group in ("g1", "g2"):
        where_est = getattr(sample.estimation, estimator)("y", where=col("grp") == group, **kw)
        assert by_df[group] == where_est.estimates[0].df == 1


def test_deff_ignores_zero_weight_rows(toy):
    """deff's n must count contributing rows only — same rule as df.

    A zero-weight row carries no information, so including it in the SRS
    variance's n understates that variance and inflates deff.
    """
    d = toy.with_columns(
        pl.when((pl.col("psu") == "A3") & (pl.col("row") == 1))
        .then(None)
        .otherwise(pl.col("y"))
        .alias("y")
    )
    sample = Sample(d, Design(wgt="w", stratum="stratum", psu="psu"))
    with_null = sample.estimation.mean("y", deff=True, drop_nulls=True).estimates[0]

    # The same 11 contributing rows, with the null row physically absent.
    physical = Sample(
        d.filter(pl.col("y").is_not_null()), Design(wgt="w", stratum="stratum", psu="psu")
    )
    dropped = physical.estimation.mean("y", deff=True).estimates[0]

    assert with_null.deff == pytest.approx(dropped.deff, rel=1e-12)


# ---------------------------------------------------------------------------
# Replication — a different rule, deliberately
# ---------------------------------------------------------------------------


@pytest.fixture
def jkn() -> pl.DataFrame:
    from pathlib import Path

    path = Path(__file__).parents[2] / "test_data" / JKN_FIXTURE
    if not path.exists():
        pytest.skip(f"Fixture {JKN_FIXTURE} not found")
    return pl.read_csv(path)


def _rep_sample(data: pl.DataFrame, df: int | None = None) -> Sample:
    rep = RepWeights(
        method="jackknife", prefix="rw", n_reps=N_REPS, df=df, rscales=tuple([0.75] * N_REPS)
    )
    return Sample(data, Design(wgt="w", stratum="stratum", psu="psu_id", rep_wgts=rep))


def test_replicate_df_is_n_reps_minus_one_everywhere(jkn):
    """The replicate df is a property of the weight set, not of the domain.

    Unlike Taylor, it does not shrink for a ``where=`` domain or a by-group.
    R instead uses qr(analysis weights)$rank - 1 recomputed on the subset rows;
    svy's convention is deliberate — see test_rep_domain_df.py.
    """
    sample = _rep_sample(jkn)
    assert _df_of(sample, method="replication") == N_REPS - 1
    assert _df_of(sample, method="replication", where=col("dom") == 1) == N_REPS - 1
    assert set(_df_by_group(sample, by="dom", method="replication").values()) == {N_REPS - 1}


def test_user_supplied_rep_df_overrides_everywhere(jkn):
    """RepWeights.df wins, domain or not — matching R's set-by-user degf."""
    sample = _rep_sample(jkn, df=9)
    assert _df_of(sample, method="replication") == 9
    assert _df_of(sample, method="replication", where=col("dom") == 1) == 9
