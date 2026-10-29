# src/svy/weighting/_bounded.py
"""Bounded solves shared by calibrate and rake: g = new/old weight held in (lo, hi)."""

from __future__ import annotations

import math

from typing import TYPE_CHECKING, Any, Literal, Sequence

import msgspec
import numpy as np

from svy.errors import WeightingError, WeightingWarning
from svy.errors._findings import finding_level
from svy.errors._weighting import show, show_list
from svy.weighting._engine import resolve_bounds


try:
    from svy_rs._internal import (
        calibrate_bounded as rust_calibrate_bounded,  # type: ignore[import-untyped]
    )
    from svy_rs._internal import (
        calibrate_bounded_by_domain as rust_calibrate_bounded_by_domain,  # type: ignore[import-untyped]
    )
except ImportError:  # pragma: no cover
    rust_calibrate_bounded = None
    rust_calibrate_bounded_by_domain = None

if TYPE_CHECKING:
    from svy.core.sample import Sample
    from svy.weighting.types import TrimConfig


class BoundedSolve(msgspec.Struct, frozen=True):
    lower: float  # -inf when open
    upper: float  # inf when open
    tol: float
    max_iter: int
    bounds: tuple[float | None, float | None]  # as given, for messages
    distance: Literal["linear", "raking"]


def bounded_solve(
    bounds: Any,
    tol: Any,
    max_iter: Any,
    trimming: TrimConfig | None,
    *,
    where: str,
    distance: Literal["linear", "raking"],
) -> BoundedSolve | None:
    """Validate ``bounds``, ``tol`` and ``max_iter``; None when no side of ``bounds`` is set,
    which keeps the unbounded solve, as R does for ``bounds=c(-Inf, Inf)``."""
    lo, hi = resolve_bounds(bounds, where=where)
    if isinstance(tol, bool) or not isinstance(tol, (int, float)):
        raise WeightingError.invalid_type(
            where=where, param="tol", got=tol, expected="a positive number"
        )
    if not (math.isfinite(tol) and tol > 0):
        raise WeightingError.invalid_range(
            where=where,
            param="tol",
            got=tol,
            detail="Parameter 'tol' must be finite and greater than 0.",
        )
    if isinstance(max_iter, bool) or not isinstance(max_iter, int):
        raise WeightingError.invalid_type(
            where=where, param="max_iter", got=max_iter, expected="an integer >= 1"
        )
    if max_iter < 1:
        raise WeightingError.invalid_range(
            where=where,
            param="max_iter",
            got=max_iter,
            detail="Parameter 'max_iter' must be at least 1.",
        )
    if lo is None and hi is None:
        return None
    if trimming is not None:
        raise WeightingError.not_supported_yet(
            where=where,
            param="trimming",
            got=show(trimming),
            hint="bounds= already holds g within (lo, hi). Use bounds= or trimming=, not both.",
        )
    return BoundedSolve(
        lower=-math.inf if lo is None else float(lo),
        upper=math.inf if hi is None else float(hi),
        tol=float(tol),
        max_iter=max_iter,
        bounds=(lo, hi),
        distance=distance,
    )


def solve_bounded(
    W: np.ndarray,
    X: np.ndarray,
    solve: BoundedSolve,
    *,
    scale: np.ndarray,
    totals: np.ndarray,
    domain_indices: np.ndarray | None = None,
    controls_by_domain: dict[int, list[float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bounded weights for each column of ``W``, and per domain (rows) and column the largest
    ``|misfit| / (1 + |t|)`` and the Newton steps taken; a single row without domains."""
    args = (solve.lower, solve.upper, solve.tol, solve.max_iter, scale, solve.distance)
    if domain_indices is None:
        assert rust_calibrate_bounded is not None  # noqa: S101
        out, achieved, steps = rust_calibrate_bounded(W, X, totals, *args)
        return out, achieved[None, :], steps[None, :]
    assert rust_calibrate_bounded_by_domain is not None  # noqa: S101
    out, doms, achieved_d, steps_d = rust_calibrate_bounded_by_domain(
        W, X, domain_indices, controls_by_domain, *args
    )
    achieved = np.empty_like(achieved_d)
    steps = np.empty_like(steps_d)
    achieved[doms] = achieved_d
    steps[doms] = steps_d
    return out, achieved, steps


def not_converged(
    solve: BoundedSolve,
    achieved: np.ndarray,
    steps: np.ndarray,
    *,
    where: str,
    method: str,
    what: str,
    domains: Sequence[Any] = (),
    replicates: Sequence[str] = (),
) -> WeightingError | None:
    """WEIGHTING_NOT_CONVERGED naming the domains (or replicates) whose solve stopped short."""
    failed = ~(achieved < solve.tol)  # NaN counts as failed
    if not failed.any():
        return None
    got: dict[str, Any] = {
        "max_iter": solve.max_iter,
        "achieved": float(np.nanmax(np.where(failed, achieved, np.nan))),
    }
    if replicates:
        names = [replicates[j] for j in np.flatnonzero(failed.any(axis=0))]
        what += f" of replicate weight(s) {show_list(names)}"
        got["replicates"] = names
    elif domains:
        doms = [domains[d] for d in np.flatnonzero(failed[:, 0])]
        what += f" in domain(s) {show_list(doms)}"
        got["domains"] = doms
    else:
        got["steps"] = int(steps[0, 0])
    return WeightingError.not_converged(
        where=where,
        method=method,
        what=what,
        max_iter=solve.max_iter,
        got=got,
        expected={"tol": solve.tol},
        hint=(
            f"Widen bounds (now {show(solve.bounds)}), increase max_iter (now "
            f"{solve.max_iter}) or relax tol (now {solve.tol:g}). The largest miss is "
            "|achieved - target| / (1 + |target|) over the controls."
        ),
    )


def record_not_converged(
    sample: Sample, err: WeightingError, *, where: str, wgt_name: str, on_nonconvergence: str
) -> None:
    """Record a kept non-converged solve as the finding ``on_nonconvergence`` asks for."""
    sample.warn(
        WeightingWarning(
            code=err.code,
            title=err.title,
            detail=err.detail.split(" The sample has NOT")[0]
            + f" {wgt_name!r} holds the last iterate.",
            where=where,
            level=finding_level(on_nonconvergence),
            param=err.param,
            expected=err.expected,
            got=err.got,
            hint=err.hint,
        )
    )
