"""Bit-packed rule arrays: binary (k = 2) one-dimensional rule arrays, 32 cells per machine word.

The lattice is stored as uint32 words, bit j of word w being cell 32 w + j, and every rule is applied to all 32 cells
of a word at once with bitwise logic. For the neighbourhood inputs X_1..X_m (leftmost first) and a rule with lookup
table T, the new state is the sum of minterms

    f(X) = OR over codes c with T[c] = 1 of  AND_i (X_i if bit i of c is set, else NOT X_i),

and every cell takes the f of the rule it uses. The rule array (which of K rules each cell uses at each step) is
stored as ceil(log2 K) bit-planes: a cell uses rule k where its bits in the planes spell k.

Conventions are those of `stica.rulearray` (Wolfram's): rule numbers as in `CellularAutomaton`, a cyclic or fixed
lattice, radius r or 1/2 (alternating or fixed pairs). The tests check the packed simulator against the unpacked one
bit for bit.

`evolve_packed_batch` runs Wolfram's adaptive evolution (one point mutation per iteration) on this representation,
many runs in parallel, optionally restarting every run that reaches its target.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .ca import SCAN_UNROLL
from .rulearray import _as_tables, _offset_patterns, _resolve_steps, num_inputs, rule_table

WORD = 32
_U = jnp.uint32
_ONES = 0xFFFFFFFF


def num_words(width: int) -> int:
    return -(-width // WORD)


def _valid_mask(width: int) -> np.ndarray:
    """Words with the bits of real cells set (the padding bits of the last word clear)."""
    mask = np.full(num_words(width), _ONES, np.uint32)
    if width % WORD:
        mask[-1] = (1 << (width % WORD)) - 1
    return mask


def pack_cells(cells) -> Array:
    """0/1 cells (..., W) -> uint32 words (..., ceil(W / 32)); padding bits are 0."""
    cells = jnp.asarray(cells)
    width = cells.shape[-1]
    nw = num_words(width)
    c = jnp.pad(cells.astype(_U), [(0, 0)] * (cells.ndim - 1) + [(0, nw * WORD - width)])
    c = c.reshape(cells.shape[:-1] + (nw, WORD))
    return jnp.sum(c << jnp.arange(WORD, dtype=_U), axis=-1, dtype=_U)   # distinct bits: the sum is an OR


@partial(jax.jit, static_argnames=("width",))
def _unpack(words, width):
    bits = ((words[..., None] >> jnp.arange(WORD, dtype=_U)) & 1).astype(jnp.int8)   # fused: no uint32 copy
    return bits.reshape(words.shape[:-1] + (-1,))[..., :width]


def unpack_cells(words, width: int) -> Array:
    """uint32 words (..., nw) -> int8 cells (..., width)."""
    return _unpack(jnp.asarray(words, _U), width)


def _planes(num_rules: int) -> int:
    return max(1, math.ceil(math.log2(num_rules)))


def pack_rule_array(ra, num_rules: int) -> Array:
    """Rule indices (T, W) in 0..num_rules-1 -> bit-planes (P, T, nw), P = ceil(log2 num_rules)."""
    ra = jnp.asarray(ra).astype(jnp.int32)
    return jnp.stack([pack_cells((ra >> j) & 1) for j in range(_planes(num_rules))])


def unpack_rule_array(planes, width: int) -> Array:
    """Bit-planes (P, T, nw) -> rule indices (T, width), int8 (computed in int8: no wide intermediate)."""
    bits = unpack_cells(planes, width)
    if planes.shape[0] == 1:
        return bits[0]
    return jnp.sum(bits << jnp.arange(planes.shape[0], dtype=jnp.int8)[:, None, None], axis=0, dtype=jnp.int8)


# --------------------------------------------------------------------------------------------- bitwise primitives

def _shl(c: Array, s: int) -> Array:
    """Shift the multi-word bit string left by s >= 0 bits (towards higher cell indices), dropping overflow."""
    nw = c.shape[-1]
    q, b = divmod(s, WORD)
    if q >= nw:
        return jnp.zeros_like(c)
    zeros = jnp.zeros(c.shape[:-1] + (q,), _U)
    cq = jnp.concatenate([zeros, c[..., :nw - q]], -1)                    # cq[w] = c[w - q]
    if b == 0:
        return cq
    prev = jnp.concatenate([jnp.zeros(c.shape[:-1] + (1,), _U), cq[..., :-1]], -1)
    return (cq << b) | (prev >> (WORD - b))


def _shr(c: Array, s: int) -> Array:
    """Shift right by s >= 0 bits (towards lower cell indices)."""
    nw = c.shape[-1]
    q, b = divmod(s, WORD)
    if q >= nw:
        return jnp.zeros_like(c)
    zeros = jnp.zeros(c.shape[:-1] + (q,), _U)
    cq = jnp.concatenate([c[..., q:], zeros], -1)                         # cq[w] = c[w + q]
    if b == 0:
        return cq
    nxt = jnp.concatenate([cq[..., 1:], jnp.zeros(c.shape[:-1] + (1,), _U)], -1)
    return (cq >> b) | (nxt << (WORD - b))


def _neighbor(c: Array, d: int, width: int, boundary: str, mask: Array) -> Array:
    """Words whose bit x holds cell x + d. `c` must have clear padding bits."""
    if d == 0:
        return c
    if d > 0:
        out = _shr(c, d)
        wrap = _shl(c, width - d)
    else:
        out = _shl(c, -d)
        wrap = _shr(c, width + d)
    if boundary == "periodic":
        out = out | wrap
    return out & mask


def _code_bits(m: int) -> np.ndarray:
    """(2^m, m) bits of every neighbourhood code, leftmost input most significant."""
    return np.array([[(c >> (m - 1 - i)) & 1 for i in range(m)] for c in range(2 ** m)], bool)


class _Model(NamedTuple):
    """Static description of a packed model (hashable, for jit)."""
    width: int
    num_rules: int
    radius: float
    neighborhood: str
    boundary: str


def _step(cells: Array, t, planes: Array, tmask: Array, model: _Model) -> Array:
    """One packed step: cells (nw,) at step t -> t + 1. planes (P, Ts, nw); tmask (K, 2^m) all-ones/zero words."""
    width, K = model.width, model.num_rules
    mask = jnp.asarray(_valid_mask(width))
    Ts = planes.shape[1]
    P = planes[:, jnp.minimum(t, Ts - 1)] if Ts > 1 else planes[:, 0]
    kbits = np.array([[(k >> j) & 1 for j in range(P.shape[0])] for k in range(K)], bool)
    sel = jnp.full((K, P.shape[-1]), _ONES, _U)                       # sel[k]: cells using rule k
    for j in range(P.shape[0]):
        sel = sel & jnp.where(kbits[:, j, None], P[j][None], ~P[j][None])
    # G[c]: cells whose rule maps code c to 1 (each cell uses exactly one rule, so the sum is an OR)
    G = jnp.sum(sel[:, None, :] & tmask[:, :, None], axis=0, dtype=_U)
    patterns = _offset_patterns(model.radius, model.neighborhood)
    shifted = {d: _neighbor(cells, d, width, model.boundary, mask) for p in patterns for d in p}
    outs = []
    for pattern in patterns:
        bits = _code_bits(len(pattern))
        M = jnp.full(G.shape, _ONES, _U)                                 # minterm of every code
        for i, d in enumerate(pattern):
            X = shifted[d][None]
            M = M & jnp.where(bits[:, i, None], X, ~X)
        outs.append(jnp.sum(M & G, axis=0, dtype=_U))                   # minterms are disjoint: sum = OR
    new = outs[0] if len(outs) == 1 else jnp.where(t % 2 == 0, outs[0], outs[1])
    new = new & mask
    return jnp.where(t < Ts, new, cells) if Ts > 1 else new


def _rollout(ic_words: Array, planes: Array, tmask: Array, model: _Model, num_steps: int) -> Array:
    def body(cells, t):
        new = jax.lax.optimization_barrier(_step(cells, t, planes, tmask, model))
        return new, new

    _, traj = jax.lax.scan(body, ic_words, jnp.arange(num_steps), unroll=SCAN_UNROLL)
    return jnp.concatenate([ic_words[None], traj], 0)


def _tmask(rules, radius) -> Array:
    tables = _as_tables(rules, 2, radius)
    return jnp.where(tables == 1, jnp.uint32(_ONES), jnp.uint32(0))


@partial(jax.jit, static_argnames=("model", "num_steps"))
def _simulate_words(ic_words, planes, tmask, model, num_steps):
    return _rollout(ic_words, planes, tmask, model, num_steps)


def simulate_packed(ic, ra, rules, *, radius: float = 1, neighborhood: str = "alternating",
                    boundary: str = "periodic", num_steps: int | None = None, packed_output: bool = False) -> Array:
    """Binary rule array simulated on packed words. Same arguments and result as `simulate_rule_array` with k = 2:
    the trajectory (num_steps + 1, W) int8, or its packed words (num_steps + 1, nw) uint32 with packed_output."""
    ic = jnp.asarray(ic)
    ra = jnp.broadcast_to(jnp.asarray(ra), (jnp.shape(ra)[0], ic.shape[-1]))       # spatial sharing: replicate
    tmask = _tmask(rules, radius)
    num_inputs(radius)
    model = _Model(ic.shape[-1], tmask.shape[0], radius, neighborhood, boundary)
    num_steps = _resolve_steps(ra.shape[0], num_steps)
    words = _simulate_words(pack_cells(ic), pack_rule_array(ra, model.num_rules), tmask, model, num_steps)
    return words if packed_output else unpack_cells(words, model.width)


# ------------------------------------------------------------------------------------------------------- evolution

class PackedEvolution(NamedTuple):
    ra: Array               # (runs, T, W) each lane's rule array at the end (with restart: the run in progress)
    best: Array             # (runs,) its loss
    initial: Array          # () loss of the starting array
    solved_at: Array        # (runs,) mutations until the lane's first solution, -1 if none
    solves: Array           # (runs,) solutions found (restart) or 1/0 (no restart)
    solve_steps: Array      # (runs,) total mutations those solutions took
    last_solution: Array    # (runs, T, W) the lane's most recent solution (meaningful where solves > 0)
    best_trace: Array | None   # (runs, iters) best loss before any restart, if record_trace
    accepted: Array | None     # (runs, iters), if record_trace


def _evolve_lane(key, ic_words, planes0, tmask, *, model, objective, iterations, accept_ties, stop_at, restart,
                 record_trace):
    T = planes0.shape[1]
    K, width = model.num_rules, model.width
    p = planes0.shape[0]
    loss = lambda planes: objective(_rollout(ic_words, planes, tmask, model, T)).astype(jnp.float32)
    best0 = loss(planes0)
    jbits = jnp.arange(p, dtype=_U)

    def iterate(carry, xs):
        key, i = xs
        planes, best, steps, solves, solve_steps, first, last = carry
        k_t, k_x, k_v = jax.random.split(key, 3)
        t = jax.random.randint(k_t, (), 0, T)
        x = jax.random.randint(k_x, (), 0, width)
        w, b = x // WORD, (x % WORD).astype(_U)
        old = planes[:, t, w]                                            # (P,) words holding the cell's bits
        old_idx = jnp.sum(((old >> b) & 1) << jbits)
        new_idx = (old_idx + jax.random.randint(k_v, (), 1, K).astype(_U)) % jnp.uint32(K)
        bit = jnp.left_shift(jnp.uint32(1), b)
        new = (old & ~bit) | (((new_idx >> jbits) & 1) << b)
        planes = planes.at[:, t, w].set(new)            # mutate in place (the old words are kept for the revert)
        value = loss(planes)
        accept = (value <= best) if accept_ties else (value < best)
        if stop_at is not None and not restart:
            accept = accept & (best > stop_at)
        planes = planes.at[:, t, w].set(jnp.where(accept, new, old))
        best = jnp.where(accept, value, best)
        steps = steps + 1
        out = (best, accept) if record_trace else None
        if stop_at is not None:
            solved = best <= stop_at
            first = jnp.where((first < 0) & solved, i + 1, first)
            if restart:
                solves, solve_steps = solves + solved, solve_steps + jnp.where(solved, steps, 0)
                last = jnp.where(solved, planes, last)
                planes, best = jnp.where(solved, planes0, planes), jnp.where(solved, best0, best)
                steps = jnp.where(solved, 0, steps)
        return (planes, best, steps, solves, solve_steps, first, last), out

    zero = jnp.zeros((), jnp.int32)
    first0 = jnp.where(best0 <= stop_at, 0, -1).astype(jnp.int32) if stop_at is not None else jnp.int32(-1)
    carry0 = (planes0, best0, zero, zero, zero, first0, planes0)
    xs = (jax.random.split(key, iterations), jnp.arange(iterations, dtype=jnp.int32))
    (planes, best, _, solves, solve_steps, first, last), out = jax.lax.scan(iterate, carry0, xs)
    if not restart:
        solves = (first > 0).astype(jnp.int32)
        solve_steps = jnp.maximum(first, 0)
        last = planes
    return planes, best, best0, first, solves, solve_steps, last, out


@partial(jax.jit, static_argnames=("model", "objective", "iterations", "accept_ties", "stop_at", "restart",
                                   "record_trace"))
def _evolve_batch(keys, ic_words, planes0, tmask, **static):
    return jax.vmap(lambda k: _evolve_lane(k, ic_words, planes0, tmask, **static))(keys)


def evolve_packed_batch(key: Array, num_runs: int, ic, ra, rules, *, objective: Callable, iterations: int,
                        radius: float = 1, neighborhood: str = "alternating", boundary: str = "periodic",
                        accept_ties: bool = True, stop_at: float | None = None, restart: bool = False,
                        record_trace: bool = False, backend: str = "auto",
                        keep_last_solution: bool = True) -> PackedEvolution:
    """Wolfram's adaptive evolution of a binary rule array on packed words: `num_runs` independent runs in
    parallel from the same start, one point mutation (a cell switched to a different rule) per iteration, kept
    unless the loss increases (`accept_ties=False`: unless it does not decrease).

    `objective` receives the packed trajectory (T + 1, nw) uint32; it must be hashable (e.g. `LifetimeLoss`, which
    only needs to know which rows are empty and so works on packed words unchanged). For objectives on the cells, use
    `unpack_cells`. An objective with the attribute `rows_only = True` declares that it only depends on which rows
    are non-empty; the fused kernel then passes it one word per row (the OR of the row's words). With `stop_at`, a run stops changing once its loss is <= stop_at; with `restart` too, it counts
    the solution and starts again from `ra`, so the whole budget keeps producing solutions.

    Equivalent in distribution to `evolve_rule_array_batch(..., mutations=1)` (different random streams); the
    tests check every solution it reports with the unpacked simulator.

    `backend`: "pallas" evaluates every candidate with one fused GPU kernel for the whole batch (two rules, radius 1
    or 1/2); "xla" uses the vmapped XLA path, which runs anywhere; "auto" picks "pallas" on a GPU when the model is
    supported. With `restart`, `keep_last_solution=False` saves one pass over the rule arrays per iteration."""
    ic = jnp.asarray(ic)
    ra = jnp.asarray(ra)
    if ra.ndim != 2 or ra.shape[1] != ic.shape[-1]:
        raise ValueError("the packed evolution needs a full (T, W) rule array")
    tmask = _tmask(rules, radius)
    if tmask.shape[0] < 2:
        raise ValueError("evolution needs at least two rules to switch between")
    if restart and stop_at is None:
        raise ValueError("restart needs stop_at")
    model = _Model(ic.shape[-1], tmask.shape[0], radius, neighborhood, boundary)
    if backend == "auto":
        backend = "pallas" if (jax.default_backend() == "gpu" and np.ndim(rules) == 1
                               and _fused_supported(model.num_rules, radius)) else "xla"
    if backend == "pallas":
        if np.ndim(rules) != 1 or not _fused_supported(model.num_rules, radius):
            raise ValueError("the fused kernel handles two rules, given as rule numbers, with radius 1 or 1/2")
        lanes = -(-num_runs // _BLOCK) * _BLOCK
        M, best, best0, first, solves, solve_steps, last, out = _evolve_fused(
            key, pack_cells(ic), pack_cells(ra.astype(jnp.int32) == 1), rules=tuple(int(r) for r in rules),
            width=model.width, radius=radius, neighborhood=neighborhood, boundary=boundary, objective=objective,
            num_lanes=lanes, iterations=iterations, accept_ties=accept_ties, stop_at=stop_at, restart=restart,
            record_trace=record_trace, keep_last=keep_last_solution or not restart)
        unpack = lambda m: unpack_cells(jnp.moveaxis(m, -1, 0), model.width)[:num_runs]
        trace, accepted = (o.T[:num_runs] for o in out) if record_trace else (None, None)
        return PackedEvolution(unpack(M), best[:num_runs], best0[0], first[:num_runs], solves[:num_runs],
                               solve_steps[:num_runs], unpack(last) if last.ndim == 3 else None, trace, accepted)
    if backend != "xla":
        raise ValueError(f"backend must be 'auto', 'pallas' or 'xla', got {backend!r}")
    planes, best, best0, first, solves, solve_steps, last, out = _evolve_batch(
        jax.random.split(key, num_runs), pack_cells(ic), pack_rule_array(ra, model.num_rules), tmask, model=model,
        objective=objective, iterations=iterations, accept_ties=accept_ties, stop_at=stop_at, restart=restart,
        record_trace=record_trace)
    unpack = jax.vmap(lambda pl: unpack_rule_array(pl, model.width))
    trace, accepted = out if record_trace else (None, None)
    return PackedEvolution(unpack(planes), best, best0[0], first, solves, solve_steps, unpack(last), trace, accepted)


# ------------------------------------------------------------------------------------- fused GPU kernel (Pallas)
#
# The XLA path above runs every time step as a few small kernels, so on a GPU each step costs a fixed few
# microseconds however little work it holds. For two-rule binary arrays this kernel instead advances a block of
# runs through all T steps in registers: one launch per candidate evaluation for the whole batch. The rules are
# compiled into the bitwise logic. Layout is lane-major (runs last) so that every load and store is coalesced.

_BLOCK = 128
_RESET_COND = lambda words: words * 4 > 64 * 2 ** 20     # batches of rule arrays larger than 64 MB


def _fused_supported(num_rules: int, radius: float) -> bool:
    return num_rules == 2 and radius in (0.5, 1)


def _shift_words(c: list, d: int, width: int, boundary: str) -> list:
    """Word list (each a vector over runs) whose bit x holds cell x + d, for d in {-1, 0, 1}."""
    nw, top = len(c), (width - 1) % WORD
    if d == 0:
        return c
    out = []
    for w in range(nw):
        if d == 1:
            inc = (c[w + 1] << (WORD - 1)) if w < nw - 1 else (
                ((c[0] & 1) << top) if boundary == "periodic" else jnp.zeros_like(c[w]))
            out.append((c[w] >> 1) | inc)
        else:
            inc = (c[w - 1] >> (WORD - 1)) if w > 0 else (
                ((c[nw - 1] >> top) & 1) if boundary == "periodic" else jnp.zeros_like(c[w]))
            out.append((c[w] << 1) | inc)
    return out


def _fused_step_logic(cells: list, s1: list, t, tables: np.ndarray, patterns, width: int, boundary: str,
                      mask: np.ndarray) -> list:
    """New words from the current words and the 'uses rule 1' words, with the two rule tables baked in."""
    shifted = {d: _shift_words(cells, d, width, boundary) for p in patterns for d in p}
    results = []
    for pattern in patterns:
        bits = _code_bits(len(pattern))
        words = []
        for w in range(len(cells)):
            new = None
            for code in range(2 ** len(pattern)):
                t0, t1 = int(tables[0, code]), int(tables[1, code])
                if not (t0 or t1):
                    continue
                m = None
                for i, d in enumerate(pattern):
                    x = shifted[d][w] if bits[code, i] else ~shifted[d][w]
                    m = x if m is None else m & x
                g = None if (t0 and t1) else (s1[w] if t1 else ~s1[w])
                term = m if g is None else m & g
                new = term if new is None else new | term
            words.append(jnp.zeros_like(cells[w]) if new is None else new)
        results.append(words)
    out = results[0] if len(results) == 1 else [jnp.where((t & 1) == 0, a, b) for a, b in zip(*results)]
    out[-1] = out[-1] & jnp.uint32(int(mask[-1]))
    return out


@partial(jax.jit, static_argnames=("rules", "width", "radius", "neighborhood", "boundary", "rows_only"))
def fused_rollout(ic_words: Array, masks: Array, *, rules: tuple, width: int, radius: float = 1,
                  neighborhood: str = "alternating", boundary: str = "periodic", rows_only: bool = False) -> Array:
    """Packed trajectories of many runs with one fused GPU kernel. ic_words (nw, B) and masks (T, nw, B), uint32,
    where a set bit in masks means the cell uses rules[1]; B must be a multiple of 128. Returns (T + 1, nw, B), or
    with rows_only (T + 1, 1, B) holding the OR of each row's words (non-zero exactly where the row has a live
    cell), which is all a lifetime needs and a fraction of the memory traffic."""
    from jax.experimental import pallas as pl

    T, nw, B = masks.shape
    tables = np.stack([rule_table(r, 2, radius) for r in rules])
    patterns = _offset_patterns(radius, neighborhood)
    mask = _valid_mask(width)

    def store(out_ref, row, words):
        if rows_only:
            acc = words[0]
            for x in words[1:]:
                acc = acc | x
            out_ref[row, 0, :] = acc
        else:
            for w in range(nw):
                out_ref[row, w, :] = words[w]

    def kernel(ic_ref, masks_ref, out_ref):
        cells = [ic_ref[w, :] for w in range(nw)]
        store(out_ref, 0, cells)

        def body(t, cells):
            s1 = [masks_ref[t, w, :] for w in range(nw)]
            new = _fused_step_logic(list(cells), s1, t, tables, patterns, width, boundary, mask)
            store(out_ref, t + 1, new)
            return tuple(new)

        jax.lax.fori_loop(0, T, body, tuple(cells))

    out_words = 1 if rows_only else nw
    return pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((T + 1, out_words, B), _U), grid=(B // _BLOCK,),
        in_specs=[pl.BlockSpec((nw, _BLOCK), lambda i: (0, i)), pl.BlockSpec((T, nw, _BLOCK), lambda i: (0, 0, i))],
        out_specs=pl.BlockSpec((T + 1, out_words, _BLOCK), lambda i: (0, 0, i)))(ic_words, masks)


@partial(jax.jit, static_argnames=("rules", "width", "radius", "neighborhood", "boundary", "objective", "num_lanes",
                                   "iterations", "accept_ties", "stop_at", "restart", "record_trace", "keep_last"))
def _evolve_fused(key, ic_words, masks0, *, rules, width, radius, neighborhood, boundary, objective, num_lanes,
                  iterations, accept_ties, stop_at, restart, record_trace, keep_last):
    """All runs advance together in one lane-major array (T, nw, lanes); every candidate evaluation is one fused
    kernel launch for the whole batch. Two rules, so a mutation flips one bit."""
    T, nw = masks0.shape
    lanes = jnp.arange(num_lanes)
    ic_b = jnp.broadcast_to(ic_words[:, None], (nw, num_lanes))
    M0 = jnp.broadcast_to(masks0[:, :, None], (T, nw, num_lanes))
    roll = partial(fused_rollout, rules=rules, width=width, radius=radius, neighborhood=neighborhood,
                   boundary=boundary, rows_only=getattr(objective, "rows_only", False))
    loss = lambda M: jax.vmap(objective, in_axes=2)(roll(ic_b, M)).astype(jnp.float32)
    best0 = loss(M0)

    def iterate(carry, xs):
        key, i = xs
        M, best, steps, solves, solve_steps, first, last = carry
        k_t, k_x = jax.random.split(key)
        t = jax.random.randint(k_t, (num_lanes,), 0, T)
        x = jax.random.randint(k_x, (num_lanes,), 0, width)
        w = x // WORD
        old = M[t, w, lanes]
        new = old ^ jnp.left_shift(jnp.uint32(1), (x % WORD).astype(_U))
        M = M.at[t, w, lanes].set(new)                  # mutate in place; referencing the old M would copy it
        value = loss(M)
        accept = (value <= best) if accept_ties else (value < best)
        if stop_at is not None and not restart:
            accept = accept & (best > stop_at)
        M = M.at[t, w, lanes].set(jnp.where(accept, new, old))
        best = jnp.where(accept, value, best)
        steps = steps + 1
        out = (best, accept) if record_trace else None
        if stop_at is not None:
            solved = best <= stop_at
            first = jnp.where((first < 0) & solved, i + 1, first)
            if restart:
                solves, solve_steps = solves + solved, solve_steps + jnp.where(solved, steps, 0)
                # A reset touches the whole batch. For large batches skip it on iterations where no run finished;
                # for small ones the branch (a device-host round trip on GPUs) costs more than the pass itself.
                if _RESET_COND(M.size):
                    if keep_last:
                        last = jax.lax.cond(solved.any(), lambda: jnp.where(solved, M, last), lambda: last)
                    M = jax.lax.cond(solved.any(), lambda: jnp.where(solved, M0, M), lambda: M)
                else:
                    if keep_last:
                        last = jnp.where(solved, M, last)
                    M = jnp.where(solved, M0, M)
                best, steps = jnp.where(solved, best0, best), jnp.where(solved, 0, steps)
        return (M, best, steps, solves, solve_steps, first, last), out

    zero = jnp.zeros(num_lanes, jnp.int32)
    first0 = jnp.where(best0 <= stop_at, 0, -1).astype(jnp.int32) if stop_at is not None else zero - 1
    carry0 = (M0, best0, zero, zero, zero, first0, M0 if keep_last else jnp.zeros((), _U))
    xs = (jax.random.split(key, iterations), jnp.arange(iterations, dtype=jnp.int32))
    (M, best, _, solves, solve_steps, first, last), out = jax.lax.scan(iterate, carry0, xs)
    if not restart:
        solves = (first > 0).astype(jnp.int32)
        solve_steps = jnp.maximum(first, 0)
        last = M
    return M, best, best0, first, solves, solve_steps, last, out
