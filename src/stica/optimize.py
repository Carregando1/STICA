"""(1+1) evolutionary optimisation of STICA initial conditions and rule tensors, entirely on the accelerator.

Algorithm: each iteration picks exactly `ic_count` distinct IC cells and `srt_count` distinct SRT entries, moves
each to a uniformly chosen *different* state, simulates, and keeps the candidate if the objective does not increase.
Accepting ties lets neutral mutations drift, as in Wolfram's adaptive evolution of rule arrays (2024); with
`accept_ties=False` only strict improvements are kept, which is the operator of the legacy `Optimizer` +
`ArbitraryRulesetMutator(num_strict=True)`. This is a (1+1)-EA / stochastic hill climber (one parent, one child,
no crossover), not a population GA.

Everything runs inside one `jax.lax.scan`; only the mutated entries are written (and reverted on rejection), so an
iteration moves no rule tensor between host and device. `run_ga_batch` runs independent chains in parallel with vmap.
Results are reproducible from the PRNG key.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from .ca import rollout
from .objectives import get_objective
from .rules import num_rules

INHOMOGENEITY = ("spacetime", "time", "space", "none")


class Mutations(NamedTuple):
    """Per-iteration mutation record (leading axis = iteration)."""
    ic_pos: Array    # (iters, ic_count, 2)
    ic_old: Array
    ic_new: Array
    srt_pos: Array   # (iters, srt_count, 4)
    srt_old: Array
    srt_new: Array


class GAResult(NamedTuple):
    ic: Array            # best IC found
    srt: Array           # best SRT found
    best: Array          # its objective value
    accepted: Array      # (iters,) bool, whether each candidate was accepted
    best_trace: Array    # (iters,) best objective after each iteration
    mutations: Mutations | None


def srt_shape(grid_size: int, n: int, inhomogeneity: str = "spacetime", steps: int | None = None) -> tuple:
    """SRT shape for a square grid: rules vary over time and space ("spacetime"), time only ("time"),
    space only ("space"), or not at all ("none", a homogeneous CA)."""
    T = grid_size - 1 if steps is None else steps
    G, R = grid_size, num_rules(n)
    shapes = {"spacetime": (T, G, G, R), "time": (T, 1, 1, R), "space": (1, G, G, R), "none": (1, 1, 1, R)}
    if inhomogeneity not in shapes:
        raise ValueError(f"inhomogeneity must be one of {INHOMOGENEITY}, got {inhomogeneity!r}")
    return shapes[inhomogeneity]


def init_state(key: Array, grid_size: int, n: int, *, inhomogeneity: str = "spacetime",
               steps: int | None = None, state_probs=None) -> tuple[Array, Array]:
    """Random IC (int8, states drawn with `state_probs`, uniform by default) and an all-zero SRT (int8)."""
    ic = jax.random.choice(key, n, (grid_size, grid_size), p=None if state_probs is None else jnp.asarray(state_probs))
    return ic.astype(jnp.int8), jnp.zeros(srt_shape(grid_size, n, inhomogeneity, steps), jnp.int8)


def sample_distinct(key: Array, shape: tuple[int, ...], k: int) -> Array:
    """k distinct coordinates of an array of `shape`, uniform over all k-subsets. Returns (k, len(shape)) int32."""
    size = math.prod(shape)
    if not 0 <= k <= size:
        raise ValueError(f"cannot pick {k} distinct entries from an array with {size} entries")
    if k == 0:
        return jnp.zeros((0, len(shape)), jnp.int32)
    if 2 * k > size:  # dense case: a permutation is cheaper than rejection
        flat = jax.random.permutation(key, size)[:k]
        return jnp.stack(jnp.unravel_index(flat, shape), axis=1).astype(jnp.int32)

    def draw(key):
        keys = jax.random.split(key, len(shape))
        return jnp.stack([jax.random.randint(kk, (k,), 0, d) for kk, d in zip(keys, shape)], axis=1)

    def duplicates(c):
        order = jnp.lexsort(c.T[::-1])
        cs = c[order]
        dup_sorted = jnp.concatenate([jnp.zeros(1, bool), jnp.all(cs[1:] == cs[:-1], axis=1)])
        return jnp.zeros(k, bool).at[order].set(dup_sorted)

    def redraw(carry):
        key, c, dup = carry
        key, sub = jax.random.split(key)
        c = jnp.where(dup[:, None], draw(sub), c)
        return key, c, duplicates(c)

    key, sub = jax.random.split(key)
    c = draw(sub)
    _, c, _ = jax.lax.while_loop(lambda carry: carry[2].any(), redraw, (key, c, duplicates(c)))
    return c.astype(jnp.int32)


def _shift(key: Array, old: Array, n: int) -> Array:
    """Move each entry to a uniformly chosen different state."""
    return ((old.astype(jnp.int32) + jax.random.randint(key, old.shape, 1, n)) % n).astype(old.dtype)


def _ga_core(key, ic, srt, *, n, num_steps, iterations, ic_count, srt_count, objective_fn, record, accept_ties):
    best0 = objective_fn(rollout(ic, srt, n, num_steps)).astype(jnp.float32)

    def iterate(carry, key):
        ic, srt, best = carry
        k_ic, k_icv, k_srt, k_srtv = jax.random.split(key, 4)
        ic_pos, srt_pos = sample_distinct(k_ic, ic.shape, ic_count), sample_distinct(k_srt, srt.shape, srt_count)
        ii, si = tuple(ic_pos.T), tuple(srt_pos.T)
        ic_old, srt_old = ic[ii], srt[si]
        ic_new, srt_new = _shift(k_icv, ic_old, n), _shift(k_srtv, srt_old, n)
        ic_c, srt_c = ic.at[ii].set(ic_new), srt.at[si].set(srt_new)
        value = objective_fn(rollout(ic_c, srt_c, n, num_steps)).astype(jnp.float32)
        accept = (value <= best) if accept_ties else (value < best)
        # keep or revert only the touched entries: O(count) work and vmap-friendly
        ic = ic_c.at[ii].set(jnp.where(accept, ic_new, ic_old))
        srt = srt_c.at[si].set(jnp.where(accept, srt_new, srt_old))
        best = jnp.where(accept, value, best)
        out = (accept, best, Mutations(ic_pos, ic_old, ic_new, srt_pos, srt_old, srt_new) if record else None)
        return (ic, srt, best), out

    (ic, srt, best), (accepted, trace, muts) = jax.lax.scan(iterate, (ic, srt, best0), jax.random.split(key, iterations))
    return GAResult(ic, srt, best, accepted, trace, muts)


_STATIC = ("n", "num_steps", "iterations", "ic_count", "srt_count", "objective_fn", "record", "accept_ties")
_ga_jit = jax.jit(_ga_core, static_argnames=_STATIC)


def _ga_batch_core(keys, ic, srt, **static):
    return jax.vmap(lambda k, i, s: _ga_core(k, i, s, **static))(keys, ic, srt)


_ga_batch_jit = jax.jit(_ga_batch_core, static_argnames=_STATIC)


def _check_counts(ic_shape, srt_shape_, ic_count, srt_count):
    for name, count, shape in (("ic_count", ic_count, ic_shape), ("srt_count", srt_count, srt_shape_)):
        if not 0 <= count <= math.prod(shape):
            raise ValueError(f"{name}={count} must be between 0 and the number of entries ({math.prod(shape)})")


def run_ga(key: Array, ic: Array, srt: Array, *, n: int, iterations: int, ic_count: int, srt_count: int,
           objective: str | Callable = "surface", num_steps: int | None = None, accept_ties: bool = True,
           record_mutations: bool = False) -> GAResult:
    """Optimise one (IC, SRT) pair. `num_steps` defaults to the SRT's time length (required if it is shared over time).
    `accept_ties=True` keeps candidates whose objective equals the current best; False keeps strict improvements only."""
    ic, srt = jnp.asarray(ic, jnp.int8), jnp.asarray(srt, jnp.int8)
    _check_counts(ic.shape, srt.shape, ic_count, srt_count)
    return _ga_jit(key, ic, srt, n=n, num_steps=num_steps, iterations=iterations, ic_count=ic_count,
                   srt_count=srt_count, objective_fn=get_objective(objective), record=record_mutations,
                   accept_ties=accept_ties)


def run_ga_batch(key: Array, num_runs: int, *, grid_size: int, n: int, iterations: int, ic_count: int,
                 srt_count: int, objective: str | Callable = "surface", inhomogeneity: str = "spacetime",
                 steps: int | None = None, state_probs=None, accept_ties: bool = True,
                 record_mutations: bool = False) -> GAResult:
    """`num_runs` independent chains from random starts, in parallel. Every field of the result gains a leading
    run axis. `steps` (trajectory length) defaults to grid_size - 1, so trajectories are grid_size^3 cubes."""
    steps = grid_size - 1 if steps is None else steps
    k_init, k_run = jax.random.split(key)
    ics, srts = jax.vmap(lambda k: init_state(k, grid_size, n, inhomogeneity=inhomogeneity, steps=steps,
                                              state_probs=state_probs))(jax.random.split(k_init, num_runs))
    _check_counts(ics.shape[1:], srts.shape[1:], ic_count, srt_count)
    return _ga_batch_jit(jax.random.split(k_run, num_runs), ics, srts, n=n, num_steps=steps, iterations=iterations,
                         ic_count=ic_count, srt_count=srt_count, objective_fn=get_objective(objective),
                         record=record_mutations, accept_ties=accept_ties)
