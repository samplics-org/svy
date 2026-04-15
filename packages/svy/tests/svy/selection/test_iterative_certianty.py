def test_extract_certainty_iterative_cascade():
    """
    Test that _extract_certainty correctly handles the cascade case where
    removing one certainty unit pushes another unit over the threshold.

    Setup (n=3, threshold=1.0):
      MOS = [100, 60, 20, 20]  total = 200
      pi  = [1.5, 0.9, 0.3, 0.3]
      -> unit 0 is certain (pi=1.5 >= 1.0)

    After removing unit 0 (n_rem=2, total_rem=100):
      pi  = [1.2, 0.4, 0.4]
      -> unit 1 is now certain (pi=1.2 >= 1.0)

    After removing unit 1 (n_rem=1, total_rem=40):
      pi  = [0.5, 0.5]
      -> no more certainty units

    Expected: cert_mask = [True, True, False, False], n_rem = 1
    """
    from svy.engine.sampling.pps import _extract_certainty
    import numpy as np

    p0 = np.array([100, 60, 20, 20], dtype=np.float64) / 200.0
    cert_mask, n_rem = _extract_certainty(p0, n=3, threshold=1.0)

    assert cert_mask.tolist() == [True, True, False, False]
    assert n_rem == 1


def test_extract_certainty_no_cascade():
    """
    When no cascading occurs, only the units that exceed the threshold
    in the first pass are marked as certain.
    """
    from svy.engine.sampling.pps import _extract_certainty
    import numpy as np

    # n=2, all pi < 1 -> no certainty units
    p0 = np.array([30, 30, 20, 20], dtype=np.float64) / 100.0
    cert_mask, n_rem = _extract_certainty(p0, n=2, threshold=1.0)

    assert not cert_mask.any()
    assert n_rem == 2


def test_extract_certainty_custom_threshold():
    """
    With threshold=0.8, units with pi >= 0.8 are treated as certain
    even though pi < 1.0.
    """
    from svy.engine.sampling.pps import _extract_certainty
    import numpy as np

    # n=2: pi = [0.9, 0.6, 0.3, 0.2] -> unit 0 exceeds 0.8
    p0 = np.array([45, 30, 15, 10], dtype=np.float64) / 100.0
    cert_mask, n_rem = _extract_certainty(p0, n=2, threshold=0.8)

    assert cert_mask[0] == True
    assert cert_mask[1:].sum() == 0
    assert n_rem == 1


def test_extract_certainty_all_certain():
    """
    When n >= N, all units should be marked as certain and n_rem == 0.
    """
    from svy.engine.sampling.pps import _extract_certainty
    import numpy as np

    p0 = np.array([40, 30, 30], dtype=np.float64) / 100.0
    cert_mask, n_rem = _extract_certainty(p0, n=3, threshold=1.0)

    assert cert_mask.all()
    assert n_rem == 0
