"""Every Python fence in the READMEs must run, and the outputs shown must match."""

import importlib.util
import sys

from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_readme.py"

pytestmark = pytest.mark.optional


def _load_renderer():
    spec = importlib.util.spec_from_file_location("render_readme", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve postponed annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not _SCRIPT.exists(), reason="scripts/ not shipped in sdist")
@pytest.mark.skipif(
    "rich" not in sys.modules and importlib.util.find_spec("rich") is None,
    reason="rich not installed",
)
def test_readme_snippets_run_and_outputs_are_current():
    renderer = _load_renderer()
    readmes = [p for p in renderer.README_FILES if p.exists()]
    assert readmes, "no README found"
    problems = [p for readme in readmes for p in renderer.refresh(readme, check=True)]
    assert not problems, "stale README outputs, run scripts/render_readme.py:\n" + "\n".join(
        problems
    )
