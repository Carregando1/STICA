"""CPU baseline implementations for the benchmarks.

Each is checked against the stica package in tests/test_baselines.py, so all implementations compute the same thing.

- numpy:     vectorised over the lattice, one Python iteration per time step (the idiomatic NumPy approach).
- numba:     compiled scalar loops, single-threaded. This is the closest analogue to Wolfram's own implementation
             (a `FunctionCompile`d loop over cells and steps).
- numba-par: the same loops parallelised with numba's prange over independent runs (batches, evolution) or over
             rows / cells within a step (single trajectories), on all cores (NUMBA_NUM_THREADS).
- packed:    binary rule arrays bit-packed 64 cells per uint64 word and evaluated with bitwise logic, the CPU
             counterpart of stica.packed (two rules, radius 1, cyclic lattice); one core or all cores.
- cellpylib: CellPyLib [Antunes 2021] with a rule callable that receives the cell index and time step; the generic
             way to run an inhomogeneous CA in an existing Python CA library. One Python call per cell and step.

Models:
- 1D rule arrays (Wolfram): k = 2, radius 1, cyclic lattice; `tables` (K, 8) from stica.rule_tables, `ra` (T, W).
- 2D STICA: Moore neighbourhood, n states, cells outside the grid are 0; `srt` (T, H, W, R) with the package's rule
  order (stica.neighbor_tuples).
- Wolfram's lifetime evolution: the (1+1) loop with single-point mutations, ties accepted, stopping at zero loss.
"""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------------------------------------- 1D rule arrays


def rule_array_numpy(ic, ra, tables):
    ic, ra, tables = np.asarray(ic, np.int8), np.asarray(ra, np.intp), np.asarray(tables, np.int8)
    T, W = ra.shape
    out = np.empty((T + 1, W), np.int8)
    out[0] = cur = ic
    for t in range(T):
        code = (np.roll(cur, 1).astype(np.intp) << 2) | (cur.astype(np.intp) << 1) | np.roll(cur, -1)
        cur = tables[ra[t], code]
        out[t + 1] = cur
    return out


def rule_array_numpy_batch(ics, ras, tables):
    """Many trajectories at once: every step is vectorised over the batch and the lattice."""
    ics, ras, tables = np.asarray(ics, np.int8), np.asarray(ras, np.intp), np.asarray(tables, np.int8)
    B, T, W = ras.shape
    out = np.empty((B, T + 1, W), np.int8)
    out[:, 0] = cur = ics
    for t in range(T):
        code = (np.roll(cur, 1, -1).astype(np.intp) << 2) | (cur.astype(np.intp) << 1) | np.roll(cur, -1, -1)
        cur = tables[ras[:, t], code]
        out[:, t + 1] = cur
    return out


def rule_array_cellpylib(ic, ra, tables):
    import cellpylib as cpl
    ra, tables = np.asarray(ra), np.asarray(tables)

    def apply_rule(n, c, t):          # t = 1, 2, ...: the step that produces row t
        return tables[ra[t - 1, c], (n[0] << 2) | (n[1] << 1) | n[2]]

    return cpl.evolve(np.asarray(ic, int)[None], timesteps=ra.shape[0] + 1, apply_rule=apply_rule, r=1)


_numba_cache = {}


def _numba():
    """Compile the numba kernels on first use (numba is an optional dependency)."""
    if _numba_cache:
        return _numba_cache
    from numba import njit

    @njit(cache=True)
    def rule_array(ic, ra, tables):
        T, W = ra.shape
        out = np.empty((T + 1, W), np.int8)
        out[0] = ic
        for t in range(T):
            for x in range(W):
                left = out[t, x - 1] if x > 0 else out[t, W - 1]
                right = out[t, x + 1] if x < W - 1 else out[t, 0]
                out[t + 1, x] = tables[ra[t, x], (left << 2) | (out[t, x] << 1) | right]
        return out

    @njit(cache=True)
    def lifetime_loss(ic, ra, tables, target):
        """Wolfram's LifetimeLossValue, stopping the simulation as soon as the pattern dies."""
        T, W = ra.shape
        cur = ic.copy()
        nxt = np.empty_like(cur)
        if cur.sum() == 0:
            return float(abs(0 - target))
        for t in range(T):
            alive = 0
            for x in range(W):
                left = cur[x - 1] if x > 0 else cur[W - 1]
                right = cur[x + 1] if x < W - 1 else cur[0]
                v = tables[ra[t, x], (left << 2) | (cur[x] << 1) | right]
                nxt[x] = v
                alive += v
            if alive == 0:
                return float(abs(t + 1 - target))
            cur, nxt = nxt, cur
        return np.inf

    @njit(cache=True)
    def evolve_lifetime(ic, ra0, tables, target, iterations, seed):
        """One run of Wolfram's adaptive evolution. Returns the rule array, its loss and the mutations used."""
        np.random.seed(seed)
        ra = ra0.copy()
        H, W = ra.shape
        K = tables.shape[0]
        best = lifetime_loss(ic, ra, tables, target)
        for it in range(iterations):
            if best == 0.0:
                return ra, best, it
            i, j = np.random.randint(H), np.random.randint(W)
            old = ra[i, j]
            ra[i, j] = (old + np.random.randint(1, K)) % K
            value = lifetime_loss(ic, ra, tables, target)
            if value <= best:
                best = value
            else:
                ra[i, j] = old
        return ra, best, iterations

    from numba import prange

    @njit(cache=True, parallel=True)
    def rule_array_par(ic, ra, tables):
        T, W = ra.shape
        out = np.empty((T + 1, W), np.int8)
        out[0] = ic
        for t in range(T):
            for x in prange(W):
                left = out[t, x - 1] if x > 0 else out[t, W - 1]
                right = out[t, x + 1] if x < W - 1 else out[t, 0]
                out[t + 1, x] = tables[ra[t, x], (left << 2) | (out[t, x] << 1) | right]
        return out

    @njit(cache=True, parallel=True)
    def rule_array_batch_par(ics, ras, tables):
        B, T, W = ras.shape
        out = np.empty((B, T + 1, W), np.int8)
        for b in prange(B):
            out[b] = rule_array(ics[b], ras[b], tables)
        return out

    @njit(cache=True, parallel=True)
    def evolve_lifetime_par(ic, ra0, tables, target, iterations, seeds):
        n = seeds.shape[0]
        best = np.empty(n)
        used = np.empty(n, np.int64)
        for r in prange(n):
            res = evolve_lifetime(ic, ra0, tables, target, iterations, seeds[r])
            best[r] = res[1]
            used[r] = res[2]
        return best, used

    _numba_cache.update(rule_array=rule_array, lifetime_loss=lifetime_loss, evolve_lifetime=evolve_lifetime,
                        rule_array_par=rule_array_par, rule_array_batch_par=rule_array_batch_par,
                        evolve_lifetime_par=evolve_lifetime_par)
    _numba_cache["stica"] = _stica_numba(njit, parallel=False)
    _numba_cache["stica_par"] = _stica_numba(njit, parallel=True)
    return _numba_cache


def rule_array_numba(ic, ra, tables):
    return _numba()["rule_array"](np.asarray(ic, np.int8), np.asarray(ra, np.int8), np.asarray(tables, np.int8))


def evolve_lifetime_numba(ic, ra0, tables, target, iterations, seed):
    return _numba()["evolve_lifetime"](np.asarray(ic, np.int8), np.asarray(ra0, np.int8),
                                       np.asarray(tables, np.int8), float(target), int(iterations), int(seed))


def rule_array_numba_par(ic, ra, tables):
    return _numba()["rule_array_par"](np.asarray(ic, np.int8), np.asarray(ra, np.int8), np.asarray(tables, np.int8))


def rule_array_numba_batch_par(ics, ras, tables):
    return _numba()["rule_array_batch_par"](np.asarray(ics, np.int8), np.asarray(ras, np.int8),
                                            np.asarray(tables, np.int8))


def evolve_lifetime_numba_par(ic, ra0, tables, target, iterations, seeds):
    """Independent runs in parallel threads (one seed each). Returns the best losses and the mutations used."""
    return _numba()["evolve_lifetime_par"](np.asarray(ic, np.int8), np.asarray(ra0, np.int8),
                                           np.asarray(tables, np.int8), float(target), int(iterations),
                                           np.asarray(seeds, np.int64))


# -------------------------------------------------------------------------------------------------------- 2D STICA

def _rank_table(n):
    """Rule offset for every neighbour-count tuple, indexed by the base-9 code sum_k a_k * 9^(k-1), where a_k is
    the number of neighbours in state k; also the per-state weights of that code (0 for state 0)."""
    from stica import neighbor_tuples
    table = np.full(9 ** (n - 1), -1, np.int64)
    for i, tup in enumerate(neighbor_tuples(n)):
        table[sum(a * 9 ** k for k, a in enumerate(tup))] = i
    weights = np.array([0] + [9 ** (s - 1) for s in range(1, n)], np.int64)
    return table, len(neighbor_tuples(n)), weights


def stica_numpy(ic, srt, n):
    """Full (T, H, W, R) or spatially shared rule tensors."""
    ic, srt = np.asarray(ic, np.int8), np.asarray(srt, np.int8)
    T, Hs, Ws, _ = srt.shape
    H, W = ic.shape
    rank, L, weights = _rank_table(n)
    rows = (np.arange(H) if Hs > 1 else np.zeros(H, np.intp))[:, None]
    cols = (np.arange(W) if Ws > 1 else np.zeros(W, np.intp))[None, :]
    out = np.empty((T + 1, H, W), np.int8)
    out[0] = cur = ic
    for t in range(T):
        pad = np.pad(cur, 1)
        code = np.zeros((H, W), np.int64)
        for dy in range(3):
            for dx in range(3):
                if dy == dx == 1:
                    continue
                code += weights[pad[dy:dy + H, dx:dx + W]]
        cur = srt[t, rows, cols, cur.astype(np.int64) * L + rank[code]]
        out[t + 1] = cur
    return out


def _stica_numba(njit, parallel):
    from numba import prange     # behaves like range when parallel=False

    # numba's on-disk cache keys on the function's name and source location, which both variants share: cache only
    # the serial one, or the parallel variant would silently load the serial machine code
    @njit(cache=not parallel, parallel=parallel)
    def kernel(ic, srt, rank, L, weights):
        T, Hs, Ws, _ = srt.shape
        H, W = ic.shape
        out = np.empty((T + 1, H, W), np.int8)
        out[0] = ic
        for t in range(T):
            for y in prange(H):
                for x in range(W):
                    code = 0
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            if dy == 0 and dx == 0:
                                continue
                            yy, xx = y + dy, x + dx
                            if 0 <= yy < H and 0 <= xx < W:
                                code += weights[out[t, yy, xx]]
                    r = np.int64(y) if Hs > 1 else np.int64(0)   # prange indices are unsigned: keep int64
                    c = np.int64(x) if Ws > 1 else np.int64(0)
                    out[t + 1, y, x] = srt[t, r, c, np.int64(out[t, y, x]) * L + rank[code]]
        return out
    return kernel


def stica_numba(ic, srt, n):
    """Full (T, H, W, R) or spatially shared (T, 1, 1, R) rule tensors."""
    rank, L, weights = _rank_table(n)
    return _numba()["stica"](np.asarray(ic, np.int8), np.asarray(srt, np.int8), rank, L, weights)


def stica_numba_par(ic, srt, n):
    """As stica_numba, with the rows of each step processed in parallel."""
    rank, L, weights = _rank_table(n)
    return _numba()["stica_par"](np.asarray(ic, np.int8), np.asarray(srt, np.int8), rank, L, weights)


def stica_cellpylib(ic, srt, n):
    """CellPyLib's 2D lattices are periodic; a ring of cells kept at 0 reproduces the zero boundary."""
    import cellpylib as cpl
    srt = np.asarray(srt)
    T, Hs, Ws, _ = srt.shape
    H, W = np.shape(ic)
    rank, L, weights = _rank_table(n)

    def apply_rule(nbhd, c, t):
        y, x = c
        if y == 0 or x == 0 or y == H + 1 or x == W + 1:
            return 0
        centre = nbhd[1, 1]
        code = weights[nbhd].sum() - weights[centre]
        return srt[t - 1, y - 1 if Hs > 1 else 0, x - 1 if Ws > 1 else 0, centre * L + rank[code]]

    padded = np.pad(np.asarray(ic, int), 1)[None]
    out = cpl.evolve2d(padded, timesteps=T + 1, apply_rule=apply_rule, r=1, neighbourhood="Moore")
    return out[:, 1:-1, 1:-1]


# ------------------------------------------------------------------------- bit-packed binary rule arrays (uint64)

_packed_cache = {}


def _packed():
    if _packed_cache:
        return _packed_cache
    from numba import njit, prange
    ONE = np.uint64(1)

    @njit(cache=True, inline="always")
    def step(cur, nxt, sel1, t0m, t1m, W, nw, lastmask):
        """One step on packed words; returns the OR of the new words (0 once the pattern has died)."""
        top = np.uint64((W - 1) % 64)
        alive = np.uint64(0)
        for w in range(nw):
            c = cur[w]
            lin = (cur[w - 1] >> np.uint64(63)) if w > 0 else ((cur[nw - 1] >> top) & ONE)
            rin = (cur[w + 1] << np.uint64(63)) if w < nw - 1 else ((cur[0] & ONE) << top)
            L = (c << ONE) | lin
            R = (c >> ONE) | rin
            s1 = sel1[w]
            s0 = ~s1
            new = np.uint64(0)
            for code in range(8):
                m = (L if code & 4 else ~L) & (c if code & 2 else ~c) & (R if code & 1 else ~R)
                new |= m & ((s1 & t1m[code]) | (s0 & t0m[code]))
            if w == nw - 1:
                new &= lastmask
            nxt[w] = new
            alive |= new
        return alive

    @njit(cache=True)
    def lifetime_loss(ic, masks, t0m, t1m, W, target):
        T, nw = masks.shape
        lastmask = np.uint64(0xFFFFFFFFFFFFFFFF) if W % 64 == 0 else (ONE << np.uint64(W % 64)) - ONE
        cur = ic.copy()
        nxt = np.empty_like(cur)
        alive = np.uint64(0)
        for w in range(nw):
            alive |= cur[w]
        if alive == 0:
            return float(abs(0 - target))
        for t in range(T):
            if step(cur, nxt, masks[t], t0m, t1m, W, nw, lastmask) == 0:
                return float(abs(t + 1 - target))
            cur, nxt = nxt, cur
        return np.inf

    @njit(cache=True)
    def evolve(ic, masks0, t0m, t1m, W, target, iterations, seed):
        np.random.seed(seed)
        masks = masks0.copy()
        T = masks.shape[0]
        best = lifetime_loss(ic, masks, t0m, t1m, W, target)
        for it in range(iterations):
            if best == 0.0:
                return masks, best, it
            t, x = np.random.randint(T), np.random.randint(W)
            bit = ONE << np.uint64(x % 64)
            masks[t, x // 64] ^= bit                       # two rules: switching = flipping the bit
            value = lifetime_loss(ic, masks, t0m, t1m, W, target)
            if value <= best:
                best = value
            else:
                masks[t, x // 64] ^= bit
        return masks, best, iterations

    @njit(cache=True, parallel=True)
    def evolve_par(ic, masks0, t0m, t1m, W, target, iterations, seeds):
        n = seeds.shape[0]
        best = np.empty(n)
        used = np.empty(n, np.int64)
        for r in prange(n):
            res = evolve(ic, masks0, t0m, t1m, W, target, iterations, seeds[r])
            best[r] = res[1]
            used[r] = res[2]
        return best, used

    _packed_cache.update(lifetime_loss=lifetime_loss, evolve=evolve, evolve_par=evolve_par)
    return _packed_cache


def pack64(cells):
    """0/1 cells (..., W) -> uint64 words (..., ceil(W / 64))."""
    cells = np.asarray(cells, np.uint64)
    W = cells.shape[-1]
    nw = -(-W // 64)
    c = np.zeros(cells.shape[:-1] + (nw * 64,), np.uint64)
    c[..., :W] = cells
    return np.bitwise_or.reduce(c.reshape(cells.shape[:-1] + (nw, 64)) << np.arange(64, dtype=np.uint64), axis=-1)


def _packed_args(ic, ra, tables):
    tables = np.asarray(tables)
    if tables.shape != (2, 8):
        raise ValueError("the packed CPU baseline handles two radius-1 rules")
    full = lambda t: np.where(np.asarray(t) == 1, np.uint64(0xFFFFFFFFFFFFFFFF), np.uint64(0))
    return pack64(ic), pack64(ra), full(tables[0]), full(tables[1]), np.asarray(ic).shape[-1]


def lifetime_loss_packed(ic, ra, tables, target):
    ic_w, masks, t0, t1, W = _packed_args(ic, ra, tables)
    return _packed()["lifetime_loss"](ic_w, masks, t0, t1, W, float(target))


def evolve_lifetime_packed(ic, ra0, tables, target, iterations, seed):
    """Single run (as evolve_lifetime_numba). Returns the rule array as 0/1 cells, its loss and the mutations used."""
    ic_w, masks, t0, t1, W = _packed_args(ic, ra0, tables)
    masks, best, used = _packed()["evolve"](ic_w, masks, t0, t1, W, float(target), int(iterations), int(seed))
    ra = ((masks[..., None] >> np.arange(64, dtype=np.uint64)) & np.uint64(1)).reshape(masks.shape[0], -1)[:, :W]
    return ra.astype(np.int8), best, used


def evolve_lifetime_packed_par(ic, ra0, tables, target, iterations, seeds):
    ic_w, masks, t0, t1, W = _packed_args(ic, ra0, tables)
    return _packed()["evolve_par"](ic_w, masks, t0, t1, W, float(target), int(iterations),
                                   np.asarray(seeds, np.int64))
