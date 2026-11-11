"""Neighbourhood counting and spacetime-rule-tensor (SRT) indexing for N-state STICA.

Rule ordering (unchanged from the original NSTICA definition)
------------------------------------------------------------
A cell in state ``s`` (0..n-1) with Moore neighbourhood counts ``(a_1, ..., a_{n-1})`` -- the number of its 8
neighbours in states 1..n-1 (the rest are in state 0; cells outside the grid count as state 0) -- uses rule

    index = s * L + rank(a_1, ..., a_{n-1}),        L = C(7 + n, 8),

where ``rank`` is the position of the tuple when all tuples with ``a_1 + ... + a_{n-1} <= 8`` are sorted in
descending lexicographic order. For n = 3 the order starts (8,0), (7,1), (7,0), (6,2), (6,1), (6,0), ... and has
L = 45 entries, so an SRT has R = n * L = 135 rules per cell and time step.

The rank is computed exactly with integer binomial coefficients:

    rank = sum_{i=1}^{n-1} C(8 - P_i + n - 1 - i, n - i),        P_i = a_1 + ... + a_i,

using C(k, k + 1) = 0, which covers cells whose 8 neighbours are all non-zero.
"""
from __future__ import annotations

import itertools
import math
from functools import lru_cache

import jax.numpy as jnp
import numpy as np
from jax import Array

MAX_STATES = 127  # SRT entries are stored as int8

_MOORE = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def rules_per_state(n: int) -> int:
    """Number of neighbourhood tuples L = C(7 + n, 8)."""
    return math.comb(7 + n, 8)


def num_rules(n: int) -> int:
    """Rules per SRT cell and time step, R = n * C(7 + n, 8)."""
    _check_states(n)
    return n * rules_per_state(n)


def neighbor_tuples(n: int) -> list[tuple[int, ...]]:
    """All neighbourhood tuples (a_1..a_{n-1}) in rule order (pure Python; for documentation and tests)."""
    return sorted((t for t in itertools.product(range(9), repeat=n - 1) if sum(t) <= 8), reverse=True)


@lru_cache(maxsize=None)
def _binomial_table(size: int) -> np.ndarray:
    table = np.zeros((size + 1, size + 1), np.int32)
    for a in range(size + 1):
        for b in range(a + 1):
            table[a, b] = math.comb(a, b)
    return table


def neighbor_counts(cells: Array, n: int) -> Array:
    """Per-state Moore neighbour counts.

    Args:
        cells: integer array (H, W) of states 0..n-1.
        n: number of states.

    Returns:
        int32 array (H, W, n-1); entry [..., k-1] counts neighbours in state k. Out-of-grid cells count as state 0.
    """
    H, W = cells.shape
    onehot = (cells[..., None] == jnp.arange(1, n, dtype=cells.dtype)).astype(jnp.int32)
    padded = jnp.pad(onehot, ((1, 1), (1, 1), (0, 0)))
    return sum(padded[1 + dy:1 + dy + H, 1 + dx:1 + dx + W] for dy, dx in _MOORE)


def rule_index(cells: Array, counts: Array, n: int) -> Array:
    """Rule index of every cell (int32 array of the same spatial shape as ``cells``)."""
    m = n - 1
    i = jnp.arange(1, m + 1)
    prefix = jnp.cumsum(counts, axis=-1)
    top = 8 - prefix + (m - i)
    bottom = jnp.broadcast_to(m - i + 1, top.shape)
    rank = jnp.asarray(_binomial_table(8 + m))[top, bottom].sum(axis=-1)
    return cells.astype(jnp.int32) * rules_per_state(n) + rank


def reorder_two_state_rules(srt) -> np.ndarray:
    """Convert a 2-state SRT between the legacy STICA2S order (index 9*s + count) and the NSTICA/``stica`` order
    (index 9*s + 8 - count). The map is its own inverse."""
    srt = np.asarray(srt)
    if srt.shape[-1] != 18:
        raise ValueError(f"expected 18 rules in the last axis for 2 states, got {srt.shape[-1]}")
    return np.ascontiguousarray(srt.reshape(*srt.shape[:-1], 2, 9)[..., ::-1].reshape(srt.shape))


def validate_srt(srt, n: int, grid_shape: tuple[int, int] | None = None, *, check_values: bool = True) -> None:
    """Raise ValueError unless ``srt`` is a valid (T|1, H|1, W|1, R) rule tensor for ``n`` states."""
    _check_states(n)
    shape = tuple(srt.shape)
    if len(shape) != 4:
        raise ValueError(f"SRT must have 4 axes (time, row, col, rule), got shape {shape}")
    if shape[3] != num_rules(n):
        raise ValueError(f"SRT last axis must have n*C(7+n,8) = {num_rules(n)} rules for n={n}, got {shape[3]}")
    if grid_shape is not None:
        for axis, (s, g) in enumerate(zip(shape[1:3], grid_shape), start=1):
            if s not in (1, g):
                raise ValueError(f"SRT axis {axis} has size {s}; expected 1 (shared) or the grid size {g}")
    if check_values:
        arr = np.asarray(srt)
        if arr.size and (arr.min() < 0 or arr.max() >= n):
            raise ValueError(f"SRT entries must lie in [0, {n - 1}], found [{arr.min()}, {arr.max()}]")


def _check_states(n: int) -> None:
    if not isinstance(n, (int, np.integer)) or not 2 <= n <= MAX_STATES:
        raise ValueError(f"number of states must be an integer in [2, {MAX_STATES}], got {n!r}")
