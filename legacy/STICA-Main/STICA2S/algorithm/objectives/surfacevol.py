import jax.numpy as jnp

DX = [
    (+1, 0, 0),
    (-1, 0, 0),
    (0, +1, 0),
    (0, -1, 0),
    (0, 0, +1),
    (0, 0, -1),
]

def surface_to_vol(grid):
    """
    calculate the surface area to volume ratio for a 3x3 lattice. Optimized using numpy.
    """
    vol = jnp.count_nonzero(grid)
    surface = (jnp.count_nonzero(grid[0:-1,:,:] * (grid[1:,:,:] == 0)) + jnp.count_nonzero(grid[-1,:,:]) +
                jnp.count_nonzero(grid[1:,:,:] * (grid[0:-1:,:,:] == 0)) + jnp.count_nonzero(grid[0,:,:]) +
                jnp.count_nonzero(grid[:,0:-1,:] * (grid[:,1:,:] == 0)) + jnp.count_nonzero(grid[:,-1,:]) +
                jnp.count_nonzero(grid[:,1:,:] * (grid[:,0:-1,:] == 0)) + jnp.count_nonzero(grid[:,0,:]) +
                jnp.count_nonzero(grid[:,:,0:-1] * (grid[:,:,1:] == 0)) + jnp.count_nonzero(grid[:,:,-1]) +
                jnp.count_nonzero(grid[:,:,1:] * (grid[:,:,0:-1] == 0)) + jnp.count_nonzero(grid[:,:,0]))
    return - surface / vol
