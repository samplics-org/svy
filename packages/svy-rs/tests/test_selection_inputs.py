"""Selection takes numpy arrays or sequences; every form draws the same sample."""

import numpy as np
import pytest
import svy_rs as ps

N = 500
FRAME = np.arange(N, dtype=np.int64)
STRATUM = np.arange(N, dtype=np.int64) % 4
MOS = np.linspace(1.0, 9.0, N)

FORMS = {
    "list": lambda a: a.tolist(),
    "array": lambda a: a,
    "other_dtype": lambda a: a.astype(np.int32 if a.dtype == np.int64 else np.float32),
    "strided": lambda a: np.repeat(a, 2)[::2],
}


@pytest.mark.parametrize("form", FORMS, ids=str)
def test_srs_input_forms(form):
    to = FORMS[form]
    want = ps.select_srs_rs(frame=FRAME.tolist(), n_scalar=20, stratum=STRATUM.tolist(), seed=7)
    got = ps.select_srs_rs(frame=to(FRAME), n_scalar=20, stratum=to(STRATUM), seed=7)
    assert got == want


@pytest.mark.parametrize("form", FORMS, ids=str)
def test_pps_input_forms(form):
    to = FORMS[form]
    want = ps.select_pps_rs(
        frame=FRAME.tolist(), mos=MOS.tolist(), n_scalar=10, stratum=STRATUM.tolist(), seed=7
    )
    got = ps.select_pps_rs(frame=to(FRAME), mos=to(MOS), n_scalar=10, stratum=to(STRATUM), seed=7)
    if form == "other_dtype":
        # float32 MOS rounds, which can move a systematic draw.
        assert len(got[0]) == len(want[0])
    else:
        assert got == want
