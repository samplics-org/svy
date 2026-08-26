import ast
import pathlib

import svy.weighting.replication as _mod


def test_every_rust_import_has_an_importerror_fallback():
    """The Poisson kernel was imported in the try block but never assigned in
    the except, so against an older svy-rs the name was simply unbound and the
    guard raised NameError instead of its message. Nothing exercises that
    branch -- it needs svy_rs absent -- so the invariant is checked here: every
    alias the try block binds, the handler must bind too.
    """
    src = pathlib.Path(_mod.__file__).read_text()
    tree = ast.parse(src)
    tries = [n for n in tree.body if isinstance(n, ast.Try)]
    assert tries, "expected a try/except ImportError around the svy_rs imports"
    node = tries[0]

    imported = {
        (alias.asname or alias.name)
        for stmt in node.body
        if isinstance(stmt, ast.ImportFrom)
        for alias in stmt.names
    }
    handler = next(
        h for h in node.handlers
        if h.type is not None and getattr(h.type, "id", None) == "ImportError"
    )
    assigned = {
        t.id
        for stmt in handler.body
        if isinstance(stmt, ast.Assign)
        for t in stmt.targets
        if isinstance(t, ast.Name)
    }
    missing = sorted(imported - assigned)
    assert not missing, (
        f"imported from svy_rs but not bound in the ImportError fallback: {missing}. "
        f"Against an older svy-rs these raise NameError instead of their guard."
    )
