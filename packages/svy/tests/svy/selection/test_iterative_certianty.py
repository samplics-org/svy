# tests/svy/selection/test_extract_certainty_rs.py
#
# Tests for certainty extraction behaviour, exercised through the public
# PPS selection API which calls the Rust engine under the hood.
#
# We can't call _extract_certainty directly (it's internal to the Rust
# binary), so we verify its behaviour through observable outputs:
#   - selected rows whose prob == 1.0  → certainty units
#   - n_selected == expected total
#   - n_rem == n - n_certain units selected

import numpy as np
import polars as pl
import pytest
import svy

RNG = 42


def _make_sample(mos_values: list[float], weights: list[float] | None = None) -> svy.Sample:
    """Build a minimal Sample with the given MOS values."""
    n = len(mos_values)
    if weights is None:
        weights = [1.0] * n
    df = pl.DataFrame(
        {
            "y": list(range(n)),
            "mos": mos_values,
            "wgt": weights,
        }
    )
    design = svy.Design(wgt="wgt", mos="mos")
    return svy.Sample(data=df, design=design)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: run pps_sys and return (cert_flags, probs)
# ─────────────────────────────────────────────────────────────────────────────


def _run_pps(sample, n, certainty_threshold=1.0):
    result = sample.sampling.pps_sys(
        n=n,
        certainty_threshold=certainty_threshold,
        rstate=RNG,
    )
    data = result._data
    prob_col = [c for c in data.columns if "prob" in c.lower()][0]
    hit_col = [c for c in data.columns if "hit" in c.lower()][0]
    probs = data[prob_col].to_list()
    hits = data[hit_col].to_list()
    cert_mask = [p is not None and abs(p - 1.0) < 1e-9 for p in probs]
    selected = [h is not None and h > 0 for h in hits]
    return cert_mask, probs, selected


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractCertaintyViaRust:
    """
    Verify certainty extraction by observing which units get prob=1.0 after
    pps_sys() — the Rust engine runs extract_certainty internally.
    """

    def test_cascade(self):
        """
        MOS=[100,60,20,20], n=3, threshold=1.0
        Pass 1: pi=[1.5, 0.9, 0.3, 0.3] → unit 0 certain
        Pass 2: pi=[1.2, 0.4, 0.4]       → unit 1 certain
        Pass 3: pi=[0.5, 0.5]             → done
        Expected: 2 certainty units (idx 0,1), n_rem=1
        """
        samp = _make_sample([100.0, 60.0, 20.0, 20.0])
        cert_mask, probs, selected = _run_pps(samp, n=3, certainty_threshold=1.0)

        # The first two units (largest MOS) must be marked as certain
        n_certain = sum(cert_mask)
        assert n_certain == 2, f"Expected 2 certainty units, got {n_certain}"

        # Exactly 3 units selected total (2 certain + 1 probabilistic)
        n_selected = sum(selected)
        assert n_selected == 3, f"Expected 3 selected, got {n_selected}"

        # The two certain units must be the ones with MOS 100 and 60
        certain_indices = [i for i, c in enumerate(cert_mask) if c]
        assert 0 in certain_indices, "Unit with MOS=100 should be certain"
        assert 1 in certain_indices, "Unit with MOS=60 should be certain"

    def test_no_cascade(self):
        """
        MOS=[30,30,20,20], n=2, threshold=1.0
        All pi < 1 → no certainty units, n_rem=2
        """
        samp = _make_sample([30.0, 30.0, 20.0, 20.0])
        cert_mask, probs, selected = _run_pps(samp, n=2, certainty_threshold=1.0)

        n_certain = sum(cert_mask)
        assert n_certain == 0, f"Expected 0 certainty units, got {n_certain}"
        assert sum(selected) == 2

    def test_custom_threshold(self):
        """
        MOS=[45,30,15,10], n=2, threshold=0.8
        Pass 1: pi=[0.9, 0.6, 0.3, 0.2] → unit 0 exceeds 0.8, marked certain
        n_rem=1, no further cascade
        """
        samp = _make_sample([45.0, 30.0, 15.0, 10.0])
        cert_mask, probs, selected = _run_pps(samp, n=2, certainty_threshold=0.8)

        assert cert_mask[0] is True, "Unit 0 (MOS=45) should be certain at threshold=0.8"
        assert sum(cert_mask) == 1, f"Expected 1 certainty unit, got {sum(cert_mask)}"
        assert sum(selected) == 2

    def test_all_certain(self):
        """
        MOS=[40,30,30], n=3, threshold=1.0
        n >= N so all units are selected with certainty.
        """
        samp = _make_sample([40.0, 30.0, 30.0])
        cert_mask, probs, selected = _run_pps(samp, n=3, certainty_threshold=1.0)

        assert all(cert_mask), f"All units should be certain, got {cert_mask}"
        assert sum(selected) == 3

    def test_n_equals_population(self):
        """
        When n == population size the entire frame is selected.
        """
        samp = _make_sample([25.0, 25.0, 25.0, 25.0])
        cert_mask, probs, selected = _run_pps(samp, n=4, certainty_threshold=1.0)

        assert sum(selected) == 4

    def test_certainty_probs_are_exactly_one(self):
        """
        Certain units must have inclusion probability == 1.0 exactly.
        """
        samp = _make_sample([80.0, 10.0, 5.0, 5.0])
        cert_mask, probs, selected = _run_pps(samp, n=2, certainty_threshold=1.0)

        for i, (c, p) in enumerate(zip(cert_mask, probs)):
            if c:
                assert p is not None and abs(p - 1.0) < 1e-9, (
                    f"Unit {i} marked certain but prob={p}"
                )

    def test_non_certainty_probs_below_one(self):
        """
        Non-certain selected units must have prob < 1.0.
        """
        samp = _make_sample([30.0, 30.0, 20.0, 20.0])
        cert_mask, probs, selected = _run_pps(samp, n=2, certainty_threshold=1.0)

        for i, (c, s, p) in enumerate(zip(cert_mask, selected, probs)):
            if s and not c:
                assert p is not None and p < 1.0 - 1e-9, (
                    f"Unit {i} selected non-certainly but prob={p}"
                )
