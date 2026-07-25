import numpy as np

def sampleobj(grid):
    """
    Calculate the sum of all values in grid. Test objective for n-state SICA generalization.
    """
    return -(np.sum(grid))