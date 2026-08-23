"""The public surface is what callers import; pin it.

This exists because the failure it guards against has already happened. svy's
`test_io_roundtrip.py` records that `_write_spss` called `svy_io.write_spss`
and `_write_sas` called `svy_io.write_sas`, and that "neither name has ever
existed" -- both calls shipped with a `# type: ignore[attr-defined]` on them,
which was the type checker saying so and being overruled. Nothing in either
package failed until someone tried to run the writer.

A name in `__all__` that does not resolve, or an alias quietly pointing
somewhere else, is invisible to every other test in this suite: they import
what they need directly and never look at the surface as a whole.
"""

import pathlib

import pytest
import svy_io


# Aliases the package promises. Each pair must be the *same object*: a name
# that resolves to a different function is the drift this file exists to catch.
ALIASES = [
    ("read_sav", "read_sav"),
    ("write_stata", "write_dta"),
    ("read_stata", "read_dta"),
]


def test_every_exported_name_resolves():
    """`from svy_io import *` must not raise, and __all__ must not lie."""
    missing = [n for n in svy_io.__all__ if not hasattr(svy_io, n)]

    assert missing == [], f"names in __all__ that do not exist: {missing}"


def test_all_is_free_of_duplicates():
    dupes = {n for n in svy_io.__all__ if svy_io.__all__.count(n) > 1}

    assert dupes == set()


@pytest.mark.parametrize("name", sorted(svy_io.__all__))
def test_exported_name_is_callable(name):
    """Every current export is a function or class -- no stray constants."""
    assert callable(getattr(svy_io, name)), f"{name} is exported but not callable"


@pytest.mark.parametrize("alias,target", ALIASES)
def test_alias_points_where_it_claims(alias, target):
    assert getattr(svy_io, alias) is getattr(svy_io, target)


def test_star_import_matches_all():
    """A name reachable via * but absent from __all__ is an accident."""
    ns: dict = {}
    exec("from svy_io import *", ns)  # noqa: S102 - that is the thing under test
    ns.pop("__builtins__", None)

    assert sorted(ns) == sorted(svy_io.__all__)


def test_no_submodule_is_orphaned():
    """
    Every module under svy_io is imported by something.

    utils.py sat here for releases with no importer, no export and no
    coverage: not undertested, unreachable. Nothing pointed that out, because
    a module nobody imports cannot fail a test.
    """
    import pkgutil

    pkg_dir = pathlib.Path(svy_io.__file__).parent
    sources = {p.stem: p.read_text() for p in pkg_dir.glob("*.py")}

    orphans = []
    for mod in pkgutil.iter_modules(svy_io.__path__):
        name = mod.name
        if name.startswith("_") or name == "svyreadstat_rs":
            continue
        importers = [
            other
            for other, src in sources.items()
            if other != name
            and (f"from .{name} import" in src or f"from svy_io.{name} import" in src)
        ]
        if not importers:
            orphans.append(name)

    assert orphans == [], f"modules nothing imports: {orphans}"
