# tests/svy/estimation/test_where_replication.py
"""
Tests for the ``where`` parameter combined with replication-based variance
estimation (BRR, Bootstrap, Jackknife, SDR).

These tests guard against the class of bug where ``where`` zeros only the
main design weight but leaves replicate weights untouched, producing
point estimates that reflect the domain but standard errors that reflect
the full sample (identical SEs for filtered vs. unfiltered runs).

Convention: the where clause restricts on ``educ``, and the ``by`` variable
is usually ``sex``. A variable may appear in both ``where`` and ``by`` on the
same column (issue #9); that overlap case is covered by
``test_where_and_by_same_column``.

The file is organized in five sections:

  1. **Behavioral / regression tests** (no R reference needed). These run
     immediately and would have caught the original bug. The most important
     is ``test_se_differs_from_unfiltered`` — a one-line diagnostic that
     fails the moment the rep-weight zeroing is missing.

  2. **Smoke tests** verifying every estimand × method runs cleanly with
     ``where``.

  3. **Cross-consistency tests** comparing ``where`` against an equivalent
     pre-filtered dataset run unfiltered. Point estimates must match
     exactly; SEs must match to within the small-domain replication
     approximation.

  4. **Edge cases** (empty domain, all-rows domain, etc.).

  5. **Golden value stubs** for rigorous validation against R's
     ``srvyr::filter(svrepdesign(...))`` output. Add the reference
     numbers and the tests will activate.
"""

from __future__ import annotations

import polars as pl
import pytest

from svy import EstimationMethod, Sample, col

from . import data_golden as golden


# ─────────────────────────────────────────────────────────────────────────
# Configuration mirroring test_replication.py
# ─────────────────────────────────────────────────────────────────────────

TOL = 1e-4

# (Method, CSV, Prefix, N_Reps, DF, PSU_Column, Golden_Dict)
SCENARIOS = [
    (EstimationMethod.BRR, "fake_survey_brr_24122025.csv", "brr_", 8, 7, "psu", golden.BRR),
    (
        EstimationMethod.BOOTSTRAP,
        "fake_survey_bootstrap_25122025.csv",
        "bs_",
        20,
        None,
        "psu",
        golden.BOOTSTRAP,
    ),
    (
        EstimationMethod.JACKKNIFE,
        "fake_survey_jackknife_25122025.csv",
        "jk_",
        8,
        7,
        "psu",
        golden.JACKKNIFE,
    ),
]


def assert_est(result, expected):
    """Helper matching the one in test_replication.py."""
    assert result.est == pytest.approx(expected["est"], rel=TOL)
    assert result.se == pytest.approx(expected["se"], rel=TOL)
    assert result.lci == pytest.approx(expected["lci"], rel=TOL)
    assert result.uci == pytest.approx(expected["uci"], rel=TOL)


def assert_est_prop(result, expected):
    """Compare only `est` and `se` for proportion estimates.

    Background: both Python (svy) and R (survey::svyciprop with method="logit")
    construct the logit CI from the design-based SE via the textbook formula
        plogis(qlogis(p) ± t(df) * se / (p * (1 - p)))
    and both produce identical `est` and `se`. They DISAGREE on which `df` to
    use for the t-quantile when the design has been filtered:

      - Python uses df = n_reps - 1 returned by the Rust replication layer,
        independent of any domain restriction. For BRR with 8 reps this is 7.
      - R's svyciprop defaults to degf(svymean(...)), which for a filtered
        svrepdesign reflects the reduced effective degrees of freedom of the
        domain. For the same BRR design after filter() this is ~3 to 5
        depending on internal accounting.

    The result is the same point estimate and SE but different CI bounds.

    TODO(svy-domain-df): the right convention for domain-level df is unsettled
    in this library. Three things to investigate before pinning CIs in goldens:

      1. Does Python's Taylor `where` use domain df or full-design df for the
         CI t-quantile? Existing Taylor goldens use TOL=1e-7 with `by=` and
         pass, but no Taylor `where` test compares CIs against an R reference.
         Likely Python is inconsistent: `by=` matches R (domain df, via the
         Rust Taylor backend computing per-group df), but `where=` uses full-
         design df (because `n_psus`/`n_strata` in `_build_estimate_result_light`
         read from the unfiltered sample via `_get_factorized_design`).

      2. If (1) confirms an inconsistency, decide library convention:
           A. Match Stata: full-design df everywhere (current Python).
           B. Match R/survey: domain df everywhere.
           C. Expose as `df_method` parameter on prop()/mean()/etc.

      3. Once decided, write a Taylor `where` test at TOL=1e-7 against an R
         reference and remove this skip on CI bounds.

    For now we validate `est` and `se` (the underlying point and variance
    estimates) and leave CI-construction conventions to library-level
    documentation rather than pinning to one choice.
    """
    assert result.est == pytest.approx(expected["est"], rel=TOL)
    assert result.se == pytest.approx(expected["se"], rel=TOL)


# Domain: educ values that exist in the test CSVs. `educ` is used for `where`
# and `sex` for `by` in most tests here; a variable may also appear in both
# (see test_where_and_by_same_column, issue #9).
def _default_where():
    return col("educ").is_in(["Med", "High"])


# ═════════════════════════════════════════════════════════════════════════
# Section 1: Behavioral / regression tests
# ═════════════════════════════════════════════════════════════════════════


class TestWhereReplicationRegressions:
    """Cheap regression tests that don't require R reference values.

    The original bug produced identical SEs for filtered and unfiltered
    runs. These tests assert the opposite: when ``where`` selects a real
    subset, both point estimates and SEs must differ from the unfiltered
    versions.
    """

    @pytest.mark.parametrize("method, csv, prefix, reps, df, psu_col, _gold", SCENARIOS)
    def test_se_differs_from_unfiltered(
        self, load_survey_data, make_design, method, csv, prefix, reps, df, psu_col, _gold
    ):
        """Filtered and unfiltered SEs must differ for a real subset.

        This is the diagnostic that would have caught the original bug
        instantly. If the rep weights aren't zeroed alongside the main
        weight, the SEs come back identical.
        """
        data = load_survey_data(csv)
        design = make_design(method, prefix, reps, df, psu_col)
        sample = Sample(data, design)

        filtered = sample.estimation.mean(
            y="income", method="replication", where=_default_where(), drop_nulls=True
        )
        unfiltered = sample.estimation.mean(y="income", method="replication", drop_nulls=True)

        se_f = filtered.estimates[0].se
        se_u = unfiltered.estimates[0].se

        assert se_f > 0
        assert se_u > 0

        rel_diff = abs(se_f - se_u) / max(se_u, 1e-12)
        assert rel_diff > 1e-3, (
            f"Filtered SE ({se_f}) suspiciously close to unfiltered SE ({se_u}). "
            f"This is the symptom of replicate weights not being zeroed by "
            f"the where clause."
        )

    @pytest.mark.parametrize("method, csv, prefix, reps, df, psu_col, _gold", SCENARIOS)
    def test_point_estimate_differs_from_unfiltered(
        self, load_survey_data, make_design, method, csv, prefix, reps, df, psu_col, _gold
    ):
        """Sanity: filtered point estimate must differ from unfiltered for a real subset.

        If this fails, the where clause isn't being applied at all.
        """
        data = load_survey_data(csv)
        design = make_design(method, prefix, reps, df, psu_col)
        sample = Sample(data, design)

        filtered = sample.estimation.mean(
            y="income", method="replication", where=_default_where(), drop_nulls=True
        )
        unfiltered = sample.estimation.mean(y="income", method="replication", drop_nulls=True)

        est_f = filtered.estimates[0].est
        est_u = unfiltered.estimates[0].est

        rel_diff = abs(est_f - est_u) / max(abs(est_u), 1e-12)
        assert rel_diff > 1e-3, (
            f"Filtered estimate ({est_f}) suspiciously close to unfiltered "
            f"({est_u}). The where clause may not be applied."
        )

    @pytest.mark.parametrize("method, csv, prefix, reps, df, psu_col, _gold", SCENARIOS)
    def test_where_with_by_se_differs_from_unfiltered_by(
        self, load_survey_data, make_design, method, csv, prefix, reps, df, psu_col, _gold
    ):
        """Per-domain SEs must differ between where+by and pure by.

        This is the combination that surfaced the original bug in
        production (PUMA-level estimates for a demographic subset).
        """
        data = load_survey_data(csv)
        design = make_design(method, prefix, reps, df, psu_col)
        sample = Sample(data, design)

        filtered = sample.estimation.mean(
            y="income",
            by="sex",
            method="replication",
            where=_default_where(),
            drop_nulls=True,
        )
        unfiltered = sample.estimation.mean(
            y="income", by="sex", method="replication", drop_nulls=True
        )

        # Build a lookup of unfiltered SEs by domain so order-independence
        # is preserved.
        unf_se_by_level = {r.by_level[0]: r.se for r in unfiltered.estimates}

        differing_count = 0
        for r in filtered.estimates:
            level = r.by_level[0]
            if level not in unf_se_by_level:
                continue
            se_u = unf_se_by_level[level]
            # Skip degenerate single-row groups where SE is zero in both.
            if se_u == 0 and r.se == 0:
                continue
            rel_diff = abs(r.se - se_u) / max(se_u, 1e-12)
            if rel_diff > 1e-3:
                differing_count += 1

        assert differing_count > 0, (
            "No domain showed a meaningful SE difference between where+by "
            "and unfiltered by. Likely indicates rep weights not being "
            "zeroed by the where clause."
        )


# ═════════════════════════════════════════════════════════════════════════
# Section 2: Parametrized core matrix (mean / total / prop / ratio)
# ═════════════════════════════════════════════════════════════════════════


class TestWhereReplicationSmoke:
    """Smoke tests that ``where`` runs cleanly for every estimand × method.

    These don't compare to golden values — they verify the call succeeds,
    returns sensible bounds, and produces non-zero SEs. Golden comparisons
    live in Section 5.
    """

    @pytest.mark.parametrize("method, csv, prefix, reps, df, psu_col, _gold", SCENARIOS)
    def test_mean_with_where_runs(
        self, load_survey_data, make_design, method, csv, prefix, reps, df, psu_col, _gold
    ):
        data = load_survey_data(csv)
        design = make_design(method, prefix, reps, df, psu_col)
        sample = Sample(data, design)

        result = sample.estimation.mean(
            y="income", method="replication", where=_default_where(), drop_nulls=True
        )

        assert len(result.estimates) == 1
        est = result.estimates[0]
        assert est.est > 0
        assert est.se > 0
        assert est.lci < est.uci
        assert est.lci <= est.est <= est.uci

    @pytest.mark.parametrize("method, csv, prefix, reps, df, psu_col, _gold", SCENARIOS)
    def test_total_with_where_runs(
        self, load_survey_data, make_design, method, csv, prefix, reps, df, psu_col, _gold
    ):
        data = load_survey_data(csv)
        design = make_design(method, prefix, reps, df, psu_col)
        sample = Sample(data, design)

        result = sample.estimation.total(
            y="low_income", method="replication", where=_default_where(), drop_nulls=True
        )

        assert len(result.estimates) == 1
        est = result.estimates[0]
        assert est.est >= 0
        assert est.se > 0
        assert est.lci < est.uci

    @pytest.mark.parametrize("method, csv, prefix, reps, df, psu_col, _gold", SCENARIOS)
    def test_prop_with_where_runs(
        self, load_survey_data, make_design, method, csv, prefix, reps, df, psu_col, _gold
    ):
        data = load_survey_data(csv)
        design = make_design(method, prefix, reps, df, psu_col)
        sample = Sample(data, design)

        result = sample.estimation.prop(
            y="low_income", method="replication", where=_default_where(), drop_nulls=True
        )

        assert len(result.estimates) > 0
        total_prop = 0.0
        for est in result.estimates:
            assert 0 <= est.est <= 1, f"Proportion out of range: {est.est}"
            assert est.se >= 0
            total_prop += est.est
        assert total_prop == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("method, csv, prefix, reps, df, psu_col, _gold", SCENARIOS)
    def test_ratio_with_where_runs(
        self, load_survey_data, make_design, method, csv, prefix, reps, df, psu_col, _gold
    ):
        data = load_survey_data(csv)
        design = make_design(method, prefix, reps, df, psu_col)
        sample = Sample(data, design)

        result = sample.estimation.ratio(
            y="income",
            x="hh_size",
            method="replication",
            where=_default_where(),
            drop_nulls=True,
        )

        assert len(result.estimates) == 1
        est = result.estimates[0]
        assert est.est > 0
        assert est.se > 0
        assert est.lci < est.uci

    @pytest.mark.parametrize("method, csv, prefix, reps, df, psu_col, _gold", SCENARIOS)
    def test_mean_with_where_and_by_runs(
        self, load_survey_data, make_design, method, csv, prefix, reps, df, psu_col, _gold
    ):
        """The where+by combination that surfaced the original bug.

        ``by`` is on ``sex`` (a different column from the where clause's
        ``educ``). Overlap of the two on the same column is covered separately
        by ``test_where_and_by_same_column``.

        Note: single-observation by-groups (e.g. ``sex="None"`` in the test
        fixtures) produce NaN SEs under Bootstrap/Jackknife because some
        replicates resample the only row out, leaving the variance
        undefined. We tolerate NaN here; the math is correctly undefined,
        not wrong.
        """
        import math

        data = load_survey_data(csv)
        design = make_design(method, prefix, reps, df, psu_col)
        sample = Sample(data, design)

        result = sample.estimation.mean(
            y="income",
            by="sex",
            method="replication",
            where=_default_where(),
            drop_nulls=True,
        )

        assert len(result.estimates) >= 1
        for est in result.estimates:
            assert est.est > 0
            assert math.isnan(est.se) or est.se >= 0
            if not (math.isnan(est.lci) or math.isnan(est.uci)):
                assert est.lci <= est.uci

    @pytest.mark.parametrize("method, csv, prefix, reps, df, psu_col, _gold", SCENARIOS)
    def test_where_and_by_same_column(
        self, load_survey_data, make_design, method, csv, prefix, reps, df, psu_col, _gold
    ):
        """A column may appear in both `where` and `by` (issue #9).

        Grouping ``by="educ"`` while restricting ``where=educ in {Med, High}``
        keeps the domain estimate for each surviving level and drops the
        excluded ones (e.g. ``Low``) entirely — no zero-weight/NaN rows for
        levels the ``where`` clause filtered out.
        """
        import math

        data = load_survey_data(csv)
        design = make_design(method, prefix, reps, df, psu_col)
        sample = Sample(data, design)

        result = sample.estimation.mean(
            y="income",
            by="educ",  # same column as the where clause
            method="replication",
            where=_default_where(),
            drop_nulls=True,
        )

        by_levels = {est.by_level[0] for est in result.estimates}
        # Only the domain's levels survive; excluded educ levels are absent.
        assert by_levels <= {"Med", "High"}
        assert by_levels  # at least one surviving level
        for est in result.estimates:
            assert est.est > 0
            assert math.isnan(est.se) or est.se >= 0


# ═════════════════════════════════════════════════════════════════════════
# Section 3: Cross-consistency (where vs. pre-filtered data)
# ═════════════════════════════════════════════════════════════════════════


class TestWhereVsPreFiltered:
    """Compare ``where`` against running the estimation on a pre-filtered
    dataset.

    Point estimates must match exactly (both compute the same weighted mean
    over the same rows). SEs computed via replication on a pre-filtered
    dataset are not the textbook-correct domain SEs, but in practice for
    typical domain sizes the two are very close.
    """

    @pytest.mark.parametrize("method, csv, prefix, reps, df, psu_col, _gold", SCENARIOS)
    def test_point_estimate_matches_prefiltered(
        self, load_survey_data, make_design, method, csv, prefix, reps, df, psu_col, _gold
    ):
        data = load_survey_data(csv)
        design = make_design(method, prefix, reps, df, psu_col)
        sample = Sample(data, design)

        with_where = sample.estimation.mean(
            y="income", method="replication", where=_default_where(), drop_nulls=True
        )

        filtered_data = data.filter(pl.col("educ").is_in(["Med", "High"]))
        prefiltered_design = make_design(method, prefix, reps, df, psu_col)
        prefiltered_sample = Sample(filtered_data, prefiltered_design)
        prefiltered = prefiltered_sample.estimation.mean(
            y="income", method="replication", drop_nulls=True
        )

        est_w = with_where.estimates[0].est
        est_p = prefiltered.estimates[0].est
        assert est_w == pytest.approx(est_p, rel=1e-10), (
            f"Where estimate {est_w} != pre-filtered {est_p}"
        )

        se_w = with_where.estimates[0].se
        se_p = prefiltered.estimates[0].se
        ratio = max(se_w, se_p) / min(se_w, se_p)
        assert ratio < 1.5, (
            f"Where SE {se_w} and pre-filtered SE {se_p} differ by more than "
            f"50% (ratio={ratio:.3f}). Cross-check expected them to be close."
        )


# ═════════════════════════════════════════════════════════════════════════
# Section 4: Edge cases
# ═════════════════════════════════════════════════════════════════════════


class TestWhereReplicationEdgeCases:
    """Edge cases that don't fit the parametrized matrix."""

    def test_where_matching_no_rows_returns_empty_or_nan(self, load_survey_data, make_design):
        """A where clause matching zero rows should not crash."""
        data = load_survey_data("fake_survey_brr_24122025.csv")
        design = make_design(EstimationMethod.BRR, "brr_", 8, 7, "psu")
        sample = Sample(data, design)

        result = sample.estimation.mean(
            y="income",
            method="replication",
            where=col("educ") == "__nonexistent_category__",
            drop_nulls=True,
        )

        if len(result.estimates) > 0:
            import math

            est = result.estimates[0]
            assert math.isnan(est.est) or est.est == 0

    def test_where_matching_all_rows_matches_unfiltered(self, load_survey_data, make_design):
        """A where clause matching every row must equal the unfiltered estimate exactly."""
        data = load_survey_data("fake_survey_brr_24122025.csv")
        design = make_design(EstimationMethod.BRR, "brr_", 8, 7, "psu")
        sample = Sample(data, design)

        always_true = col("income") >= -1e18

        filtered = sample.estimation.mean(
            y="income", method="replication", where=always_true, drop_nulls=True
        )
        unfiltered = sample.estimation.mean(y="income", method="replication", drop_nulls=True)

        assert filtered.estimates[0].est == pytest.approx(unfiltered.estimates[0].est, rel=1e-10)
        assert filtered.estimates[0].se == pytest.approx(unfiltered.estimates[0].se, rel=1e-6)

    def test_multiple_where_conditions_list_form(self, load_survey_data, make_design):
        """List form of where should produce same result as combined &-expression."""
        data = load_survey_data("fake_survey_brr_24122025.csv")
        design = make_design(EstimationMethod.BRR, "brr_", 8, 7, "psu")
        sample = Sample(data, design)

        list_form = sample.estimation.mean(
            y="income",
            method="replication",
            where=[col("educ").is_in(["Med", "High"]), col("hh_size") >= 2],
            drop_nulls=True,
        )
        combined_form = sample.estimation.mean(
            y="income",
            method="replication",
            where=(col("educ").is_in(["Med", "High"])) & (col("hh_size") >= 2),
            drop_nulls=True,
        )

        assert list_form.estimates[0].est == pytest.approx(
            combined_form.estimates[0].est, rel=1e-10
        )
        assert list_form.estimates[0].se == pytest.approx(combined_form.estimates[0].se, rel=1e-10)

    def test_repeated_where_calls_independent(self, load_survey_data, make_design):
        """Calling estimation with where multiple times shouldn't have side effects.

        Guards against accidental in-place mutation of replicate weight
        columns on the sample's underlying data.
        """
        data = load_survey_data("fake_survey_brr_24122025.csv")
        design = make_design(EstimationMethod.BRR, "brr_", 8, 7, "psu")
        sample = Sample(data, design)

        first = sample.estimation.mean(
            y="income", method="replication", where=_default_where(), drop_nulls=True
        )
        unfiltered = sample.estimation.mean(y="income", method="replication", drop_nulls=True)
        third = sample.estimation.mean(
            y="income", method="replication", where=_default_where(), drop_nulls=True
        )

        assert first.estimates[0].est == pytest.approx(third.estimates[0].est, rel=1e-10)
        assert first.estimates[0].se == pytest.approx(third.estimates[0].se, rel=1e-10)
        assert first.estimates[0].est != pytest.approx(unfiltered.estimates[0].est, rel=1e-3)


# ═════════════════════════════════════════════════════════════════════════
# Section 5: Golden-value tests (stubs — fill in from R)
# ═════════════════════════════════════════════════════════════════════════

# TODO: we need to do two things
# First: Re-run R and drop df <- df[df$sex != "None", ] or leave it and update Python affected tests
# Second: Sort out the DF calculation for domain (by and where) when replication is used.


WHERE_GOLDEN_AVAILABLE = all(
    hasattr(golden, name) for name in ("WHERE_BRR", "WHERE_BOOTSTRAP", "WHERE_JACKKNIFE")
)


@pytest.mark.skipif(
    not WHERE_GOLDEN_AVAILABLE,
    reason="Where-clause golden values not yet added to data_golden.py. "
    "See scripts/verify_where_*.R to compute them.",
)
class TestWhereReplicationGolden:
    """Validate against R's srvyr::filter() output. Activated once goldens are added."""

    WHERE_SCENARIOS = [
        (
            EstimationMethod.BRR,
            "fake_survey_brr_24122025.csv",
            "brr_",
            8,
            7,
            "psu",
            getattr(golden, "WHERE_BRR", {}),
        ),
        (
            EstimationMethod.BOOTSTRAP,
            "fake_survey_bootstrap_25122025.csv",
            "bs_",
            20,
            None,
            "psu",
            getattr(golden, "WHERE_BOOTSTRAP", {}),
        ),
        (
            EstimationMethod.JACKKNIFE,
            "fake_survey_jackknife_25122025.csv",
            "jk_",
            8,
            7,
            "psu",
            getattr(golden, "WHERE_JACKKNIFE", {}),
        ),
    ]

    @pytest.mark.parametrize("method, csv, prefix, reps, df, psu_col, gold", WHERE_SCENARIOS)
    def test_mean_overall_with_where(
        self, load_survey_data, make_design, method, csv, prefix, reps, df, psu_col, gold
    ):
        data = load_survey_data(csv).filter(pl.col("sex") != "None")
        design = make_design(method, prefix, reps, df, psu_col)
        sample = Sample(data, design)

        result = sample.estimation.mean(
            y="income", method="replication", where=_default_where(), drop_nulls=True
        )
        assert_est(result.estimates[0], gold["mean_overall"])

    @pytest.mark.parametrize("method, csv, prefix, reps, df, psu_col, gold", WHERE_SCENARIOS)
    def test_mean_by_sex_with_where(
        self, load_survey_data, make_design, method, csv, prefix, reps, df, psu_col, gold
    ):
        """By=sex on the educ-restricted domain. Only Male/Female checked.

        The ``"None"`` sex value is a single-row degenerate case in the
        fixture (SE = 0); excluded from the assertions.
        """
        data = load_survey_data(csv).filter(pl.col("sex") != "None")
        design = make_design(method, prefix, reps, df, psu_col)
        sample = Sample(data, design)

        result = sample.estimation.mean(
            y="income",
            by="sex",
            method="replication",
            where=_default_where(),
            drop_nulls=True,
        )
        checked = 0
        for res in result.estimates:
            level = res.by_level[0]
            if level not in ("Male", "Female"):
                continue
            assert_est(res, gold["mean_by_sex"][level])
            checked += 1
        assert checked == len(gold["mean_by_sex"]), (
            f"Expected golden levels {set(gold['mean_by_sex'])} but only "
            f"checked {checked} of them in the result."
        )

    @pytest.mark.parametrize("method, csv, prefix, reps, df, psu_col, gold", WHERE_SCENARIOS)
    def test_total_overall_with_where(
        self, load_survey_data, make_design, method, csv, prefix, reps, df, psu_col, gold
    ):
        data = load_survey_data(csv).filter(pl.col("sex") != "None")
        design = make_design(method, prefix, reps, df, psu_col)
        sample = Sample(data, design)

        result = sample.estimation.total(
            y="low_income", method="replication", where=_default_where(), drop_nulls=True
        )
        assert_est(result.estimates[0], gold["total_overall"])

    @pytest.mark.parametrize("method, csv, prefix, reps, df, psu_col, gold", WHERE_SCENARIOS)
    def test_prop_overall_with_where(
        self, load_survey_data, make_design, method, csv, prefix, reps, df, psu_col, gold
    ):
        data = load_survey_data(csv).filter(pl.col("sex") != "None")
        design = make_design(method, prefix, reps, df, psu_col)
        sample = Sample(data, design)

        result = sample.estimation.prop(
            y="low_income", method="replication", where=_default_where(), drop_nulls=True
        )
        for res in result.estimates:
            level = res.y_level
            assert_est_prop(res, gold["prop_overall"][level])

    @pytest.mark.parametrize("method, csv, prefix, reps, df, psu_col, gold", WHERE_SCENARIOS)
    def test_ratio_overall_with_where(
        self, load_survey_data, make_design, method, csv, prefix, reps, df, psu_col, gold
    ):
        data = load_survey_data(csv).filter(pl.col("sex") != "None")
        design = make_design(method, prefix, reps, df, psu_col)
        sample = Sample(data, design)

        result = sample.estimation.ratio(
            y="income",
            x="hh_size",
            method="replication",
            where=_default_where(),
            drop_nulls=True,
        )
        assert_est(result.estimates[0], gold["ratio_overall"])
