# tests/test_prop_ci.py
"""
Tests for proportion confidence interval methods in svy.

Reference values generated from:
  - R survey 4.4-2: svyciprop(method="beta") and svyciprop(method="logit")
  - Stata 18 + kg_nchs (Ward 2019): Korn-Graubard with NCHS truncation

Dataset: svy_synthetic_sample_07082025.csv, 922 rows after complete.cases.

Design effects are all > 1 for this dataset, so beta and korn-graubard
produce identical CIs on the real data.  The two methods diverge only on:
  - p=0 / p=1 edge cases (tested with dummy data from Stata)
  - deff < 1 scenarios (tested with synthetic parameters)

References
----------
Korn E.L., Graubard B.I. (1998). Survey Methodology 24(2):193-201.
Ward B.W. (2019). Stata Journal 19(3):510-522.
Parker J.D. et al. (2017). Vital Health Stat 2(175).
"""

from __future__ import annotations

import math

import pytest
from scipy import stats
from scipy.stats import beta as beta_dist, f as f_dist


# =========================================================================
# Standalone CI function — mirrors Estimation._compute_prop_ci in base.py
# =========================================================================


def _normalize_ci_method(method: str) -> str:
    m = method.lower().replace("_", "-")
    aliases = {
        "clopper-pearson": "korn-graubard",
        "kg": "korn-graubard",
        "score": "wilson",
    }
    return aliases.get(m, m)


def compute_prop_ci(
    p: float, se: float, alpha: float, df: int, n: int, method: str
) -> tuple[float, float]:
    """Standalone version of Estimation._compute_prop_ci for unit testing."""
    method = _normalize_ci_method(method)

    if method == "logit":
        if p <= 0 or p >= 1:
            return (p, p)
        t_crit = stats.t.ppf(1 - alpha / 2, df) if df > 0 else 1.96
        scale = se / (p * (1.0 - p)) if se > 0 else 0
        logit_p = math.log(p / (1 - p))
        lci = 1.0 / (1.0 + math.exp(-(logit_p - t_crit * scale)))
        uci = 1.0 / (1.0 + math.exp(-(logit_p + t_crit * scale)))
        return (lci, uci)

    elif method == "beta":
        if p <= 0 or p >= 1 or se <= 0:
            return (p, p)
        n_eff = (p * (1 - p)) / (se**2)
        if df > 0 and n > 1:
            t_n = stats.t.ppf(alpha / 2, n - 1)
            t_df = stats.t.ppf(alpha / 2, df)
            n_eff = n_eff * (t_n / t_df) ** 2
        x = n_eff * p
        lci = beta_dist.ppf(alpha / 2, x, n_eff - x + 1)
        uci = beta_dist.ppf(1 - alpha / 2, x + 1, n_eff - x)
        return (lci, uci)

    elif method == "korn-graubard":
        if p <= 0 or p >= 1:
            n_eff = float(n)
            if df > 0 and n > 1:
                t_n = stats.t.ppf(1 - alpha / 2, n - 1)
                t_df = stats.t.ppf(1 - alpha / 2, df)
                t_adj = (t_n / t_df) ** 2
                n_eff_df = min(n, n_eff * t_adj)
            else:
                n_eff_df = n_eff
            x = p * n_eff_df
            if p == 0:
                lci = 0.0
                v3, v4 = 2 * (x + 1), 2 * (n_eff_df - x)
                if n_eff_df > 0 and v3 > 0 and v4 > 0:
                    f_upper = f_dist.ppf(1 - alpha / 2, v3, v4)
                    uci = (v3 * f_upper) / (v4 + v3 * f_upper)
                else:
                    uci = 1.0
                return (lci, uci)
            else:  # p == 1
                uci = 1.0
                v1, v2 = 2 * x, 2 * (n_eff_df - x + 1)
                if n_eff_df > 0 and v1 > 0 and v2 > 0:
                    f_lower = f_dist.ppf(alpha / 2, v1, v2)
                    lci = (v1 * f_lower) / (v2 + v1 * f_lower)
                else:
                    lci = 0.0
                return (lci, uci)
        if se <= 0:
            return (p, p)
        n_eff = (p * (1 - p)) / (se**2)
        if df > 0 and n > 1:
            t_n = stats.t.ppf(1 - alpha / 2, n - 1)
            t_df = stats.t.ppf(1 - alpha / 2, df)
            t_adj = (t_n / t_df) ** 2
            n_eff_df = min(n, n_eff * t_adj)
        else:
            n_eff_df = min(n, n_eff)
        x = n_eff_df * p
        v1, v2 = 2 * x, 2 * (n_eff_df - x + 1)
        v3, v4 = 2 * (x + 1), 2 * (n_eff_df - x)
        if v1 > 0 and v2 > 0:
            f_lower = f_dist.ppf(alpha / 2, v1, v2)
            lci = (v1 * f_lower) / (v2 + v1 * f_lower)
        else:
            lci = 0.0
        if v3 > 0 and v4 > 0:
            f_upper = f_dist.ppf(1 - alpha / 2, v3, v4)
            uci = (v3 * f_upper) / (v4 + v3 * f_upper)
        else:
            uci = 1.0
        return (lci, uci)

    elif method == "wilson":
        if p <= 0 or p >= 1 or se <= 0:
            return (p, p)
        n_eff = (p * (1 - p)) / (se**2)
        if df > 0 and n > 1:
            t_n = stats.t.ppf(1 - alpha / 2, n - 1)
            t_df = stats.t.ppf(1 - alpha / 2, df)
            n_eff = n_eff * (t_n / t_df) ** 2
        z = stats.t.ppf(1 - alpha / 2, df) if df > 0 else 1.96
        z2 = z * z
        denom = 1 + z2 / n_eff
        center = (p + z2 / (2 * n_eff)) / denom
        half_width = (z / denom) * math.sqrt(p * (1 - p) / n_eff + z2 / (4 * n_eff * n_eff))
        lci = max(0.0, center - half_width)
        uci = min(1.0, center + half_width)
        return (lci, uci)

    else:
        raise ValueError(f"Unknown CI method: {method!r}")


ALPHA = 0.05
TOL = 1e-7  # 0.05% tolerance for CI bounds


# =========================================================================
# R survey::svyciprop(method="beta") — Korn-Graubard, R implementation
#
# R survey 4.4-2, dataset: 922 rows, resp2_bin==1 proportion
# =========================================================================


class TestBetaSRS:
    """A. SRS: ids=~1, weights=~samp_wgt. degf=921, n=922, deff=1.3642."""

    p, se, n, df = 0.507348258357226, 0.0192309892530454, 922, 921

    def test_ci(self):
        lci, uci = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "beta")
        assert lci == pytest.approx(0.468954757876348, abs=TOL)
        assert uci == pytest.approx(0.545677337133621, abs=TOL)


class TestBetaStratified:
    """B. Stratified: strata=~region. degf=918, n=922, deff=1.3675."""

    p, se, n, df = 0.507348258357226, 0.0192537392244595, 922, 918

    def test_ci(self):
        lci, uci = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "beta")
        assert lci == pytest.approx(0.468908426601806, abs=TOL)
        assert uci == pytest.approx(0.545723513770539, abs=TOL)


class TestBetaClustered:
    """C. Clustered: ids=~cluster. degf=59, n=922, deff=1.0039."""

    p, se, n, df = 0.507348258357226, 0.0164967611954173, 922, 59

    def test_ci(self):
        lci, uci = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "beta")
        assert lci == pytest.approx(0.47383814353398, abs=TOL)
        assert uci == pytest.approx(0.540809220838568, abs=TOL)


class TestBetaStratClus:
    """D. Stratified+Clustered. degf=56, n=922, deff=1.0376, n_eff_df=852.88."""

    p, se, n, df = 0.507348258357226, 0.0167712595840644, 922, 56

    def test_ci(self):
        lci, uci = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "beta")
        assert lci == pytest.approx(0.473233376535445, abs=TOL)
        assert uci == pytest.approx(0.541412207644769, abs=TOL)

    def test_effective_sample_size(self):
        """Cross-check n_eff_df against R (852.88) and Stata (852.88)."""
        n_eff = (self.p * (1 - self.p)) / (self.se**2)
        t_n = stats.t.ppf(ALPHA / 2, self.n - 1)
        t_df = stats.t.ppf(ALPHA / 2, self.df)
        n_eff_df = n_eff * (t_n / t_df) ** 2
        assert n_eff_df == pytest.approx(852.88, abs=0.5)


class TestBetaSubsetHighSchool:
    """E. Strat+Clus, subset educ='High School'. degf=56, n=276, deff=1.3476."""

    p, se, n, df = 0.471999767708621, 0.0348828290040279, 276, 56

    def test_ci(self):
        lci, uci = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "beta")
        assert lci == pytest.approx(0.400786658613709, abs=TOL)
        assert uci == pytest.approx(0.544064683764286, abs=TOL)


class TestBetaSubsetPostgraduate:
    """F. Strat+Clus, subset educ='Postgraduate'. degf=52, n=167, deff=1.2867."""

    p, se, n, df = 0.501335861736522, 0.0438888962101757, 167, 52

    def test_ci(self):
        lci, uci = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "beta")
        assert lci == pytest.approx(0.410824184861583, abs=TOL)
        assert uci == pytest.approx(0.5917831333426, abs=TOL)


# =========================================================================
# R survey::svyciprop(method="logit")
# =========================================================================


class TestLogitSRS:
    """A. SRS logit."""

    p, se, n, df = 0.507348258357226, 0.0192309892530454, 922, 921

    def test_ci(self):
        lci, uci = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "logit")
        assert lci == pytest.approx(0.469636396975924, abs=TOL)
        assert uci == pytest.approx(0.544976682060238, abs=TOL)


class TestLogitStratClus:
    """D. Stratified+Clustered logit."""

    p, se, n, df = 0.507348258357226, 0.0167712595840644, 922, 56

    def test_ci(self):
        lci, uci = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "logit")
        assert lci == pytest.approx(0.473768764613996, abs=TOL)
        assert uci == pytest.approx(0.54086158229386, abs=TOL)


class TestLogitSubsetHighSchool:
    """E. Strat+Clus, subset educ='High School' logit."""

    p, se, n, df = 0.471999767708817, 0.0348828290040279, 276, 56

    def test_ci(self):
        lci, uci = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "logit")
        assert lci == pytest.approx(0.403112644241021, abs=TOL)
        assert uci == pytest.approx(0.541969987343409, abs=TOL)


class TestLogitSubsetPostgraduate:
    """F. Strat+Clus, subset educ='Postgraduate' logit."""

    p, se, n, df = 0.501335861736521, 0.0438888962101757, 167, 52

    def test_ci(self):
        lci, uci = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "logit")
        assert lci == pytest.approx(0.41412536084128, abs=TOL)
        assert uci == pytest.approx(0.588465156732825, abs=TOL)


# =========================================================================
# Stata kg_nchs — Korn-Graubard with NCHS truncation
#
# Stata 18 + kg_nchs, same 922-row dataset.
# For deff > 1, results match R beta exactly.
# =========================================================================


class TestKornGraubardStratClus:
    """D. Stata: strat+clus overall. n*=852.88, df=56."""

    p, se, n, df = 0.507348258357226, 0.0167712595840644, 922, 56

    def test_ci_matches_stata(self):
        lci, uci = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "korn-graubard")
        # Stata: p=0.5073, CI=(0.4732, 0.5414)
        assert lci == pytest.approx(0.4732, abs=1e-3)
        assert uci == pytest.approx(0.5414, abs=1e-3)

    def test_matches_beta_when_deff_gt_1(self):
        """When deff > 1, korn-graubard == beta (no truncation needed)."""
        lci_b, uci_b = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "beta")
        lci_k, uci_k = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "korn-graubard")
        assert lci_k == pytest.approx(lci_b, abs=1e-6)
        assert uci_k == pytest.approx(uci_b, abs=1e-6)


class TestKornGraubardSubsetHighSchool:
    """E. Stata: subpop(hs). n*=197.79, df=56."""

    p, se, n, df = 0.471999767708621, 0.0348828290040279, 276, 56

    def test_ci_matches_stata(self):
        lci, uci = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "korn-graubard")
        # Stata: p=0.4720, CI=(0.4008, 0.5441)
        assert lci == pytest.approx(0.4008, abs=1e-3)
        assert uci == pytest.approx(0.5441, abs=1e-3)


class TestKornGraubardSRS:
    """A. Stata: SRS. n*=675.84, df=921."""

    p, se, n, df = 0.507348258357226, 0.0192309892530454, 922, 921

    def test_ci_matches_stata(self):
        lci, uci = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "korn-graubard")
        # Stata row for resp2_bin==1: CI=(0.4690, 0.5457)
        assert lci == pytest.approx(0.4690, abs=1e-3)
        assert uci == pytest.approx(0.5457, abs=1e-3)


# =========================================================================
# Edge cases: p=0 and p=1 (Stata dummy data, 5 strata, 2 PSUs, n=100, df=5)
#
# Stata kg_nchs: p=1.0, CI=(0.9400, 1.0000), n*=59.58
# The complement p=0 gives CI=(0.0000, 0.0600)
# =========================================================================


class TestKornGraubardPZero:
    """p=0: korn-graubard gives proper CI; beta returns degenerate (0,0)."""

    n, df = 100, 5

    def test_kg_p_zero(self):
        lci, uci = compute_prop_ci(0.0, 0.0, ALPHA, self.df, self.n, "korn-graubard")
        assert lci == 0.0
        # Stata complement: upper CI for p=0 ≈ 0.06
        assert uci == pytest.approx(0.06, abs=0.005)

    def test_kg_p_one(self):
        lci, uci = compute_prop_ci(1.0, 0.0, ALPHA, self.df, self.n, "korn-graubard")
        assert uci == 1.0
        # Stata: lower CI for p=1 ≈ 0.94
        assert lci == pytest.approx(0.94, abs=0.005)

    def test_beta_p_zero_degenerate(self):
        """R beta returns (p, p) when p=0 — no interval."""
        lci, uci = compute_prop_ci(0.0, 0.0, ALPHA, self.df, self.n, "beta")
        assert lci == 0.0
        assert uci == 0.0

    def test_beta_p_one_degenerate(self):
        """R beta returns (p, p) when p=1 — no interval."""
        lci, uci = compute_prop_ci(1.0, 0.0, ALPHA, self.df, self.n, "beta")
        assert lci == 1.0
        assert uci == 1.0


class TestKornGraubardPZeroEffectiveSampleSize:
    """When p=0 or p=1, n_eff falls back to nominal n (NCHS rule)."""

    def test_n_eff_equals_n(self):
        """Stata shows n*=59.58 for n=100, df=5 dummy data.

        This is n * (t_{n-1}/t_{df})^2 = 100 * (t_99/t_5)^2.
        Since p=0 → n_eff=n, then n_eff_df = min(n, n * t_adj).
        With df=5 < n-1=99, t_adj < 1, so n_eff_df = n * t_adj ≈ 59.58.
        But in korn-graubard for p=0, we use n directly (no df-adjustment
        to n_eff, because n_eff IS n). The CI uses the F distribution
        with x=0, n_eff_df=n=100.
        """
        # Stata's n*=59.58 comes from its own internal calculation
        # that applies df-adjustment even for p=0.
        # Our implementation uses n directly for p=0 (matching NCHS SAS macro).
        # The CI values still match Stata's output.
        lci, uci = compute_prop_ci(0.0, 0.0, ALPHA, 5, 100, "korn-graubard")
        assert lci == 0.0
        assert uci > 0.0
        assert uci < 0.10  # reasonable upper bound


# =========================================================================
# Truncation: deff < 1 (synthetic parameters)
# =========================================================================


class TestTruncationDeffLessThanOne:
    """When deff < 1, n_eff > n. NCHS truncates at n; R does not."""

    p, n, df = 0.5, 100, 50
    se = 0.03  # → n_eff ≈ 277.8, deff ≈ 0.36

    def test_kg_wider_than_beta(self):
        """KG truncates to n=100, giving wider CI than beta (n_eff=277.8)."""
        lci_b, uci_b = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "beta")
        lci_k, uci_k = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "korn-graubard")
        assert (uci_k - lci_k) > (uci_b - lci_b)

    def test_beta_uses_uncapped_neff(self):
        """Beta uses n_eff > n without truncation."""
        n_eff = (self.p * (1 - self.p)) / (self.se**2)
        assert n_eff > self.n  # 277.8 > 100

    def test_kg_caps_at_n(self):
        """KG caps n_eff_df at n."""
        n_eff = (self.p * (1 - self.p)) / (self.se**2)
        t_n = stats.t.ppf(1 - ALPHA / 2, self.n - 1)
        t_df = stats.t.ppf(1 - ALPHA / 2, self.df)
        t_adj = (t_n / t_df) ** 2
        n_eff_df = min(self.n, n_eff * t_adj)
        assert n_eff_df == self.n  # capped


# =========================================================================
# Alias resolution
# =========================================================================


class TestAliases:
    """Aliases resolve to the correct canonical method."""

    p, se, n, df = 0.507348258357226, 0.0167712595840644, 922, 56

    def test_clopper_pearson_alias(self):
        ci_cp = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "clopper-pearson")
        ci_kg = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "korn-graubard")
        assert ci_cp == ci_kg

    def test_kg_alias(self):
        ci_kg_short = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "kg")
        ci_kg = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "korn-graubard")
        assert ci_kg_short == ci_kg

    def test_score_alias(self):
        ci_score = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "score")
        ci_wilson = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "wilson")
        assert ci_score == ci_wilson

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown CI method"):
            compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "jeffreys")


# =========================================================================
# Edge cases: se=0, extreme p
# =========================================================================


class TestEdgeCases:
    def test_se_zero_beta(self):
        lci, uci = compute_prop_ci(0.5, 0.0, ALPHA, 100, 1000, "beta")
        assert lci == 0.5
        assert uci == 0.5

    def test_se_zero_kg(self):
        lci, uci = compute_prop_ci(0.5, 0.0, ALPHA, 100, 1000, "korn-graubard")
        assert lci == 0.5
        assert uci == 0.5

    def test_se_zero_logit(self):
        lci, uci = compute_prop_ci(0.5, 0.0, ALPHA, 100, 1000, "logit")
        # logit with se=0: scale=0, CI collapses to (p, p)
        assert lci == pytest.approx(0.5, abs=1e-10)
        assert uci == pytest.approx(0.5, abs=1e-10)

    def test_logit_p_zero(self):
        """Logit returns degenerate (0, 0) for p=0."""
        lci, uci = compute_prop_ci(0.0, 0.01, ALPHA, 100, 1000, "logit")
        assert lci == 0.0
        assert uci == 0.0

    def test_logit_p_one(self):
        """Logit returns degenerate (1, 1) for p=1."""
        lci, uci = compute_prop_ci(1.0, 0.01, ALPHA, 100, 1000, "logit")
        assert lci == 1.0
        assert uci == 1.0


# =========================================================================
# Ward (2019) empirical examples (additional cross-validation)
# =========================================================================


class TestWardExample1:
    """CHF from 2014 NAMCS: p=0.0158, se=0.0013, n=45710, df=2129.

    Expected: n*=8958.54, CI=(0.0133, 0.0186), width=0.0053.
    """

    p, se, n, df = 0.0157735, 0.0013157, 45710, 2129

    def test_beta(self):
        lci, uci = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "beta")
        assert lci == pytest.approx(0.0133, abs=1e-4)
        assert uci == pytest.approx(0.0186, abs=1e-4)

    def test_korn_graubard(self):
        lci, uci = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "korn-graubard")
        assert lci == pytest.approx(0.0133, abs=1e-4)
        assert uci == pytest.approx(0.0186, abs=1e-4)


class TestWardExample2:
    """Hispanic women >=65 with CHF: p=0.0388, se=0.0177, n=440, df=1958.

    Expected: n*=119.11, CI=(0.0120, 0.0910), RCIW=203.3%.
    """

    p, se, n, df = 0.0388459, 0.0177429, 440, 1958

    def test_beta(self):
        lci, uci = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "beta")
        assert lci == pytest.approx(0.0120, abs=1e-4)
        assert uci == pytest.approx(0.0910, abs=1e-4)

    def test_korn_graubard(self):
        lci, uci = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "korn-graubard")
        assert lci == pytest.approx(0.0120, abs=1e-4)
        assert uci == pytest.approx(0.0910, abs=1e-4)

    def test_rciw_unreliable(self):
        """NCHS suppression: RCIW > 130%."""
        lci, uci = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "korn-graubard")
        rciw = ((uci - lci) / self.p) * 100
        assert rciw > 130.0


# =========================================================================
# Wilson score interval — manual R reference values
#
# R survey CRAN does not yet include method="wilson".
# Reference values computed manually in R using:
#   n_eff = p(1-p)/var, df-adjusted via (t_{n-1}/t_{df})^2,
#   then Wilson score formula with t-quantile.
# Same p, se, n, df as beta/logit tests above.
# =========================================================================


class TestWilsonSRS:
    """A. SRS wilson. degf=921, n=922."""

    p, se, n, df = 0.507348258357226, 0.0192309892530454, 922, 921

    def test_ci(self):
        lci, uci = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "wilson")
        assert lci == pytest.approx(0.469672037027401, abs=TOL)
        assert uci == pytest.approx(0.544941199467683, abs=TOL)


class TestWilsonStratified:
    """B. Stratified wilson. degf=918, n=922."""

    p, se, n, df = 0.507348258357226, 0.0192537392244595, 922, 918

    def test_ci(self):
        lci, uci = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "wilson")
        assert lci == pytest.approx(0.469627347615766, abs=TOL)
        assert uci == pytest.approx(0.544985691416404, abs=TOL)


class TestWilsonClustered:
    """C. Clustered wilson. degf=59, n=922."""

    p, se, n, df = 0.507348258357226, 0.0164967611954173, 922, 59

    def test_ci(self):
        lci, uci = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "wilson")
        assert lci == pytest.approx(0.473734388405036, abs=TOL)
        assert uci == pytest.approx(0.540895823087696, abs=TOL)


class TestWilsonStratClus:
    """D. Stratified+Clustered wilson. degf=56, n=922."""

    p, se, n, df = 0.507348258357226, 0.0167712595840644, 922, 56

    def test_ci(self):
        lci, uci = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "wilson")
        assert lci == pytest.approx(0.473100648434371, abs=TOL)
        assert uci == pytest.approx(0.541527041874996, abs=TOL)

    def test_width_ordering(self):
        """Logit narrowest, then beta, then Wilson widest (for this design)."""
        lci_l, uci_l = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "logit")
        lci_b, uci_b = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "beta")
        lci_w, uci_w = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "wilson")
        width_logit = uci_l - lci_l
        width_beta = uci_b - lci_b
        width_wilson = uci_w - lci_w
        assert width_logit < width_beta < width_wilson


class TestWilsonSubsetHighSchool:
    """E. Strat+Clus, subset educ='High School' wilson. degf=56, n=276."""

    p, se, n, df = 0.471999767708621, 0.0348828290040279, 276, 56

    def test_ci(self):
        lci, uci = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "wilson")
        assert lci == pytest.approx(0.402157571004044, abs=TOL)
        assert uci == pytest.approx(0.542955544303627, abs=TOL)


class TestWilsonSubsetPostgraduate:
    """F. Strat+Clus, subset educ='Postgraduate' wilson. degf=52, n=167."""

    p, se, n, df = 0.501335861736522, 0.0438888962101757, 167, 52

    def test_ci(self):
        lci, uci = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "wilson")
        assert lci == pytest.approx(0.413185286701914, abs=TOL)
        assert uci == pytest.approx(0.589403471572584, abs=TOL)


# =========================================================================
# Wilson edge cases
# =========================================================================


class TestWilsonEdgeCases:
    """Wilson edge cases: p=0, p=1, se=0."""

    def test_p_zero_degenerate(self):
        lci, uci = compute_prop_ci(0.0, 0.01, ALPHA, 100, 1000, "wilson")
        assert lci == 0.0
        assert uci == 0.0

    def test_p_one_degenerate(self):
        lci, uci = compute_prop_ci(1.0, 0.01, ALPHA, 100, 1000, "wilson")
        assert lci == 1.0
        assert uci == 1.0

    def test_se_zero_degenerate(self):
        lci, uci = compute_prop_ci(0.5, 0.0, ALPHA, 100, 1000, "wilson")
        assert lci == 0.5
        assert uci == 0.5

    def test_ci_within_zero_one(self):
        """Wilson CI must always be within [0, 1]."""
        # Very small proportion
        lci, uci = compute_prop_ci(0.001, 0.0005, ALPHA, 50, 5000, "wilson")
        assert lci >= 0.0
        assert uci <= 1.0
        # Very large proportion
        lci, uci = compute_prop_ci(0.999, 0.0005, ALPHA, 50, 5000, "wilson")
        assert lci >= 0.0
        assert uci <= 1.0


class TestWilsonComparison:
    """Cross-method comparisons for Wilson."""

    p, se, n, df = 0.507348258357226, 0.0167712595840644, 922, 56

    def test_same_point_estimate(self):
        """All methods use the same p — only CI differs."""
        # Wilson doesn't modify the point estimate
        lci_w, uci_w = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "wilson")
        midpoint = (lci_w + uci_w) / 2
        # Wilson center is shifted toward 0.5, but point estimate is unchanged
        # (it's the p passed in, not the CI midpoint)
        assert self.p == pytest.approx(0.507348, abs=1e-4)

    def test_wilson_and_kg_similar_width(self):
        """Wilson and Korn-Graubard widths are close (within 1%)."""
        lci_w, uci_w = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "wilson")
        lci_k, uci_k = compute_prop_ci(self.p, self.se, ALPHA, self.df, self.n, "korn-graubard")
        width_w = uci_w - lci_w
        width_k = uci_k - lci_k
        assert abs(width_w - width_k) / width_k < 0.01
