"""Objectives on a STICA trajectory read as a 3-D voxel lattice (time = first axis). All are *minimised*.

Any non-zero state counts as solid. The outside of the bounding box counts as empty.
"""
from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array


def surface_area(grid: Array) -> Array:
    """Number of exposed voxel faces (solid next to empty or to the outside), int32."""
    solid = jnp.pad((grid != 0).astype(jnp.int32), 1)
    return sum(jnp.abs(jnp.diff(solid, axis=a)).sum() for a in range(3))


def neg_surface(grid: Array) -> Array:
    """-surface area (maximises surface area). Same values as the legacy `surfacecalc`."""
    return -surface_area(grid).astype(jnp.float32)


def neg_surface_to_volume(grid: Array) -> Array:
    """-(surface area / solid volume). Same values as the legacy `surface_to_vol`, except that an empty lattice
    scores +inf (never accepted) instead of NaN."""
    vol = jnp.count_nonzero(grid)
    ratio = surface_area(grid) / jnp.maximum(vol, 1)
    return jnp.where(vol > 0, -ratio, jnp.inf).astype(jnp.float32)


OBJECTIVES: dict[str, Callable[[Array], Array]] = {
    "surface": neg_surface,
    "surface_to_volume": neg_surface_to_volume,
}


def get_objective(objective: str | Callable[[Array], Array]) -> Callable[[Array], Array]:
    if callable(objective):
        return objective
    try:
        return OBJECTIVES[objective]
    except KeyError:
        raise ValueError(f"unknown objective {objective!r}; choose from {sorted(OBJECTIVES)} or pass a callable") from None
