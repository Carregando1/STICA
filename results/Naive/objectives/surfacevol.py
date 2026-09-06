from numpy import int32
from numpy._typing import NDArray


DX = [
    (+1, 0, 0),
    (-1, 0, 0),
    (0, +1, 0),
    (0, -1, 0),
    (0, 0, +1),
    (0, 0, -1),
]

def surface_to_vol(grid: NDArray[int32]):
    """
    calculate the surface area to volume ratio for a 3x3 lattice.
    """
    surface, vol = 0, 0
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            for z in range(grid.shape[2]):
                if grid[x][y][z] == 0:
                    continue
                vol += 1
                for dx in DX:
                    xp, yp, zp = x+dx[0], y+dx[1], z+dx[2]
                    if not (0 <= xp < grid.shape[0]) or not (0 <= yp < grid.shape[1]) or not (0 <= zp < grid.shape[2]) or grid[xp][yp][zp] == 0:
                        surface += 1
    return - surface / vol
