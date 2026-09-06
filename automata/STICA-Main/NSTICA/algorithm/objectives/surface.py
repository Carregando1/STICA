import jax.numpy as jnp

def surfacecalc(grid):
    """
    calculate the surface area for a 3D lattice. Optimized using numpy.
    """
    surface = (jnp.count_nonzero(grid[0:-1,:,:] * (grid[1:,:,:] == 0)) + jnp.count_nonzero(grid[-1,:,:]) +
                jnp.count_nonzero(grid[1:,:,:] * (grid[0:-1:,:,:] == 0)) + jnp.count_nonzero(grid[0,:,:]) +
                jnp.count_nonzero(grid[:,0:-1,:] * (grid[:,1:,:] == 0)) + jnp.count_nonzero(grid[:,-1,:]) +
                jnp.count_nonzero(grid[:,1:,:] * (grid[:,0:-1,:] == 0)) + jnp.count_nonzero(grid[:,0,:]) +
                jnp.count_nonzero(grid[:,:,0:-1] * (grid[:,:,1:] == 0)) + jnp.count_nonzero(grid[:,:,-1]) +
                jnp.count_nonzero(grid[:,:,1:] * (grid[:,:,0:-1] == 0)) + jnp.count_nonzero(grid[:,:,0]))
    return -surface
