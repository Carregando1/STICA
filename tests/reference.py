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


def wolfram_rule_array_reference(initial, rule_numbers, radius=1, neighborhood="Alternating"):
    """Line-by-line port of the RuleArray function Wolfram published with the essay's figures (k = 2, cyclic
    lattice, orientation "Both"). `rule_numbers[i - 1][j - 1]` is the rule number at 1-indexed row i, column j."""
    state = [int(v) for v in initial]
    states = [state]
    width = len(state)
    for i in range(1, len(rule_numbers) + 1):
        new = []
        for j in range(1, width + 1):
            element = int(rule_numbers[i - 1][j - 1])
            left = state[-1] if j == 1 else state[j - 2]
            center = state[j - 1]
            right = state[0] if j == width else state[j]
            if radius == 1:
                code = (left << 2) | (center << 1) | right
            elif neighborhood == "LeftCenter":
                code = (left << 1) | center
            elif neighborhood == "RightCenter":
                code = (center << 1) | right
            else:  # "Alternating"
                code = ((center << 1) | right) if i % 2 == 1 else ((left << 1) | center)
            new.append((element & (1 << code)) >> code)
        state = new
        states.append(state)
    return np.array(states)


def wolfram_lifetime_reference(states):
    """Wolfram's lifetime: 0 for a blank start, the first all-zero row, or len(states) if the pattern survives."""
    if sum(states[0]) == 0:
        return 0
    for i in range(1, len(states)):
        if sum(states[i]) == 0:
            return i
    return len(states)


def rule_array_reference(ic, rule_numbers, k, radius, boundary):
    """k-colour rule array with integer radius, by explicit digit extraction from the rule numbers."""
    state = [int(v) for v in ic]
    width = len(state)
    states = [state]
    for row in rule_numbers:
        new = []
        for x in range(width):
            code = 0
            for d in range(-radius, radius + 1):
                y = x + d
                if boundary == "periodic":
                    v = state[y % width]
                else:
                    v = state[y] if 0 <= y < width else 0
                code = code * k + v
            new.append((int(row[x]) // k ** code) % k)
        state = new
        states.append(state)
    return np.array(states)
