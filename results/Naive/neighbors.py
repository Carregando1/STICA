"""
neighbors.py

module to calculate state to apply ruleset

rewritten for clarity
"""

import numpy as np
from numpy.typing import NDArray

#set of neighbors; surroundings

surr = [
    (1, 1),
    (1, 0),
    (1, -1),
    (0, 1),
    (0, -1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
]


def neighbors(states: NDArray[np.int32], num_states=None) -> NDArray[np.int32]:
    """
    Neighbors: returns width * height * states+1 dim NDArray: states+1 being
    the number of neighbors of a cell in each state followed by the state of the actual
    cell

    states: IC array.
    """
    width, height = states.shape

    #Automatically scans states

    if num_states is None:
        num_states = np.max(states) + 1
    result = np.zeros(states.shape + (num_states + 1,), dtype=np.int32)

    #creates array; state of squares out of bounds is 0

    for x in range(width):
        for y in range(height):
            result[x, y, num_states] = states[x, y]
            for dx in surr:
                xp = (x+dx[0],y+dx[1])
                # if requested square out of bounds value is 0
                if (not  (0 <= xp[0] < width and 0 <= xp[1] < height)): neighbor_state = 0 
                else: neighbor_state = states[xp]
                result[x, y, neighbor_state] += 1

    return result