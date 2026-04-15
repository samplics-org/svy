# svy/utils/random_state.py
from __future__ import annotations

import hashlib
import logging

from typing import Hashable, Sequence, TypeAlias

import numpy as np
import numpy.typing as npt


log = logging.getLogger(__name__)

# Public type alias (scikit-learn style)
RandomState: TypeAlias = int | np.integer | np.random.Generator | np.random.SeedSequence | None

__all__ = [
    "RandomState",
    "resolve_random_state",
    "resolve_seed_sequence",
    "spawn_child_rngs",
    "seed_from_random_state",
]


# ------------------------
# Helpers
# ------------------------


def _hash_to_words(
    *parts: str | bytes | int | float,
    salt: bytes = b"svy",
    digest_size: int = 16,  # 16 bytes = 4x uint32 (128 bits), sufficient for SeedSequence
) -> npt.NDArray[np.uint32]:
    """
    Stable hash of inputs into uint32 words for SeedSequence entropy.
    Uses BLAKE2b for speed and security.
    """
    # BLAKE2b 'person' must be <= 16 bytes.
    # We use it to domain-separate our hashes.
    h = hashlib.blake2b(digest_size=digest_size, person=salt[:16])

    for p in parts:
        if isinstance(p, str):
            h.update(p.encode("utf-8"))
        elif isinstance(p, (bytes, bytearray)):
            h.update(p)
        elif isinstance(p, (int, float)):
            h.update(str(p).encode("ascii"))
        else:
            # Fallback for hashable objects
            h.update(str(p).encode("utf-8"))

    # Buffer is guaranteed to be a multiple of 4 bytes if digest_size is.
    return np.frombuffer(h.digest(), dtype=np.uint32)


# ------------------------
# Public API
# ------------------------


def resolve_random_state(random_state: RandomState) -> np.random.Generator:
    """
    Resolve a flexible 'random_state' into a np.random.Generator.

    - int / np.integer -> seeded Generator (default_rng)
    - SeedSequence     -> Generator from that seed sequence
    - Generator        -> returned as-is
    - None             -> fresh unseeded Generator
    """
    if isinstance(random_state, np.random.Generator):
        return random_state

    if isinstance(random_state, np.random.SeedSequence):
        return np.random.Generator(np.random.PCG64(random_state))

    # Integers or None are handled by default_rng directly
    return np.random.default_rng(random_state)


def resolve_seed_sequence(random_state: RandomState) -> np.random.SeedSequence:
    """
    Resolve to a SeedSequence for reproducible, order-invariant derivations.
    """
    if isinstance(random_state, np.random.SeedSequence):
        return random_state

    if isinstance(random_state, (int, np.integer)):
        return np.random.SeedSequence(random_state)

    if isinstance(random_state, np.random.Generator):
        # Extract the state from the generator to create a stable seed.
        # Note: This is a heavy operation but ensures we don't disturb the generator.
        state = random_state.bit_generator.state
        # We hash the repr of the state to get deterministic entropy
        words = _hash_to_words(repr(state), salt=b"svy_rng_state")
        return np.random.SeedSequence(words)

    # None -> Entropy from OS
    return np.random.SeedSequence()


def spawn_child_rngs(
    rng_or_state: RandomState,
    keys: Sequence[Hashable] | npt.NDArray,
    *,
    salt: bytes = b"svy_child",
) -> dict[Hashable, np.random.Generator]:
    """
    Create independent child Generators for a list of keys (e.g. strata).

    Optimization:
    Uses `SeedSequence.spawn()` which is the idiomatic and fastest way
    to create independent streams from a root seed.
    """
    # FIX: Use len() check which works for both list and numpy arrays
    if len(keys) == 0:
        return {}

    # 1. Resolve root entropy
    root_ss = resolve_seed_sequence(rng_or_state)

    # 2. Mix in the salt to separate this operation from others using same seed
    # We create a new root specific to this salt/operation
    salted_root = np.random.SeedSequence(
        entropy=root_ss.entropy,
        spawn_key=root_ss.spawn_key + (int.from_bytes(salt, "big"),),
        pool_size=root_ss.pool_size,
    )

    # 3. Spawn N children efficiently
    # This uses the underlying bit generator's jump/split capabilities if available,
    # or hashing if not. It's much faster than manual hashing loops.
    children_ss = salted_root.spawn(len(keys))

    return {k: np.random.Generator(np.random.PCG64(ss)) for k, ss in zip(keys, children_ss)}


def seed_from_random_state(random_state: RandomState) -> int | None:
    """
    Convert RandomState to a single 32-bit integer seed.
    Useful for interoperability with libraries requiring a simple int seed (e.g. Polars).
    """
    if random_state is None:
        return None

    if isinstance(random_state, (int, np.integer)):
        return int(random_state)

    # Derive a deterministic integer from the complex state
    ss = resolve_seed_sequence(random_state)

    # generate_state(1) returns a uint array, we take the first element
    return int(ss.generate_state(1, dtype=np.uint32)[0])
