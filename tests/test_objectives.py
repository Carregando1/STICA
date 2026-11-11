import numpy as np
import jax.numpy as jnp
import pytest

from stica import get_objective, neg_surface, neg_surface_to_volume, surface_area
from reference import surface_area_reference


def legacy_surface(grid):
    """The legacy `surfacecalc` formula (legacy/STICA-Main/NSTICA/algorithm/objectives/surface.py)."""
    return (jnp.count_nonzero(grid[0:-1] * (grid[1:] == 0)) + jnp.count_nonzero(grid[-1]) +
            jnp.count_nonzero(grid[1:] * (grid[0:-1] == 0)) + jnp.count_nonzero(grid[0]) +
            jnp.count_nonzero(grid[:, 0:-1] * (grid[:, 1:] == 0)) + jnp.count_nonzero(grid[:, -1]) +
            jnp.count_nonzero(grid[:, 1:] * (grid[:, 0:-1] == 0)) + jnp.count_nonzero(grid[:, 0]) +
            jnp.count_nonzero(grid[:, :, 0:-1] * (grid[:, :, 1:] == 0)) + jnp.count_nonzero(grid[:, :, -1]) +
            jnp.count_nonzero(grid[:, :, 1:] * (grid[:, :, 0:-1] == 0)) + jnp.count_nonzero(grid[:, :, 0]))


@pytest.mark.parametrize("n", [2, 3])
def test_surface_area(n):
    rng = np.random.default_rng(n)
    for _ in range(4):
        g = rng.integers(0, n, (6, 5, 7))
        assert int(surface_area(jnp.asarray(g))) == surface_area_reference(g) == int(legacy_surface(jnp.asarray(g)))
        assert float(neg_surface(jnp.asarray(g))) == -surface_area_reference(g)


def test_surface_to_volume():
    one = np.zeros((3, 3, 3), int)
    one[1, 1, 1] = 1
    assert float(neg_surface_to_volume(jnp.asarray(one))) == -6.0
    assert float(neg_surface_to_volume(jnp.zeros((3, 3, 3), jnp.int8))) == np.inf   # legacy returned NaN


def test_get_objective():
    assert get_objective("surface") is neg_surface
    assert get_objective(neg_surface) is neg_surface
    with pytest.raises(ValueError, match="unknown objective"):
        get_objective("volume")
