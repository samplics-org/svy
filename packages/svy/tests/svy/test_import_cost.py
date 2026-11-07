# tests/svy/test_import_cost.py
"""``import svy`` stays cheap: heavy scipy submodules load only when a function needs them."""

from __future__ import annotations

import subprocess
import sys


# A fresh interpreter: this test process has imported scipy already.
_SCRIPT = (
    "import sys; import svy; "
    "print(','.join(m for m in ('scipy.stats', 'scipy.optimize') if m in sys.modules))"
)


def test_import_svy_loads_neither_scipy_stats_nor_optimize():
    r = subprocess.run([sys.executable, "-c", _SCRIPT], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == ""
