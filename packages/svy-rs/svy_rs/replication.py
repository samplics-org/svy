"""Replication-based variance estimation functions."""

from svy_rs import _internal

replicate_mean = _internal.replicate_mean
replicate_total = _internal.replicate_total
replicate_ratio = _internal.replicate_ratio
replicate_prop = _internal.replicate_prop

__all__ = [
    "replicate_mean",
    "replicate_total",
    "replicate_ratio",
    "replicate_prop",
]
