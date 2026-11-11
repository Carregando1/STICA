"""Plain-NumPy reference implementations used as the specification in the tests."""
import numpy as np

from stica.rules import neighbor_tuples


def moore_neighbours(cur, r, c):
    pad = np.pad(cur, 1)          # outside the grid = state 0
    nb = pad[r:r + 3, c:c + 3].ravel().tolist()
    del nb[4]
    return nb


def simulate_reference(ic, srt, n, num_steps=None):
    """Loop-based STICA with the documented descending rule order. Supports size-1 (shared) SRT axes."""
    ic = np.asarray(ic, np.int64)
    srt = np.asarray(srt, np.int64)
    H, W = ic.shape
    T = srt.shape[0] if num_steps is None else num_steps
    order = {t: i for i, t in enumerate(neighbor_tuples(n))}
    L = len(order)
    out, cur = [ic.copy()], ic.copy()
    for t in range(T):
        rules = srt[min(t, srt.shape[0] - 1)]
        nxt = np.zeros_like(cur)
        for r in range(H):
            for c in range(W):
                nb = moore_neighbours(cur, r, c)
                tup = tuple(nb.count(k) for k in range(1, n))
                nxt[r, c] = rules[r if rules.shape[0] > 1 else 0, c if rules.shape[1] > 1 else 0,
                                  cur[r, c] * L + order[tup]]
        out.append(nxt)
        cur = nxt
    return np.stack(out)


def simulate_reference_ascending_2state(ic, srt):
    """2-state STICA with the legacy STICA2S order (index 9*self + live-neighbour count)."""
    ic = np.asarray(ic, np.int64)
    srt = np.asarray(srt, np.int64)
    H, W = ic.shape
    out, cur = [ic.copy()], ic.copy()
    for t in range(srt.shape[0]):
        nxt = np.zeros_like(cur)
        for r in range(H):
            for c in range(W):
                nxt[r, c] = srt[t, r, c, 9 * cur[r, c] + sum(moore_neighbours(cur, r, c))]
        out.append(nxt)
        cur = nxt
    return np.stack(out)


def surface_area_reference(grid):
    """Count exposed faces voxel by voxel."""
    g = np.asarray(grid) != 0
    total = 0
    for idx in np.argwhere(g):
        for axis in range(3):
            for step in (-1, 1):
                nb = idx.copy()
                nb[axis] += step
                if nb[axis] < 0 or nb[axis] >= g.shape[axis] or not g[tuple(nb)]:
                    total += 1
    return total
