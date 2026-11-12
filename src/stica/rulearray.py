"""Wolfram's rule arrays: one-dimensional spacetime-inhomogeneous cellular automata.

A rule array (Wolfram, 2024) is a 1D cellular automaton in which the cell at position x and step t applies its own
rule, chosen from a finite set of ordinary CA rules. Here `ra[t, x]` indexes a table of rules, so step t does

    cell'(x) = rules[ra[t, x], code_t(x)]

where code_t(x) is the neighbourhood of x read as a base-k number. The conventions follow the `RuleArray` code that
Wolfram published with the essay's figures:

- Rule numbers are as in `CellularAutomaton`. The neighbourhood (left, ..., right) is read as a base-k number with
  the leftmost cell most significant, and digit `code` of the rule number (base k, least significant first) is the
  new state.
- The lattice is cyclic. `boundary="fixed"` treats cells outside the lattice as state 0 instead.
- Radius-1/2 rules read two cells. "alternating" (Wolfram's default) reads (centre, right) on steps t = 0, 2, 4, ...
  and (left, centre) on odd steps (his rows are 1-indexed). "left_center" and "center_right" always read one pair.
- Row t of the rule array takes the state at step t to step t + 1.

Rule sharing works as in the 2D model. `ra` has shape (T, W), (T, 1) (one rule per step, Wolfram's "layered"
array), (1, W) (one rule per position, "vertically layered") or (1, 1) (an ordinary homogeneous CA). Past the end
of a time-varying rule array (t >= T) the cells keep their state.

`table_array_rollout` is the generalisation in which every cell and step owns a full rule table, tables[t, x, code].

Evolution (`evolve_rule_array`) is the same on-device (1+1) loop as the 2D optimiser: random point mutations that
switch a cell's rule to a different one from the set, kept unless the loss increases.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .ca import SCAN_UNROLL
from .optimize import Mutations, _check_counts, hill_climb, hill_climb_batch, steps_to_target

NEIGHBORHOODS = ("alternating", "left_center", "center_right")
BOUNDARIES = ("periodic", "fixed")
_STATIC = ("k", "radius", "neighborhood", "boundary", "num_steps")


# ----------------------------------------------------------------------------------------------- rules and lattices

def num_inputs(radius: float) -> int:
    """Cells a rule reads: 2r + 1 for integer radius r >= 0, and 2 for radius 1/2."""
    if radius == 0.5:
        return 2
    if float(radius).is_integer() and radius >= 0:
        return 2 * int(radius) + 1
    raise ValueError(f"radius must be 1/2 or a non-negative integer, got {radius!r}")


def rule_table(rule: int, k: int = 2, radius: float = 1) -> np.ndarray:
    """Lookup table (k**inputs,) of a rule number: entry `code` is digit `code` of the rule in base k."""
    size = k ** num_inputs(radius)
    rule = int(rule)
    if not 0 <= rule < k ** size:
        raise ValueError(f"rule {rule} is outside 0..{k ** size - 1} for k={k}, radius={radius}")
    digits = np.zeros(size, np.int8)
    for i in range(size):
        rule, digits[i] = divmod(rule, k)
    return digits


def rule_tables(rules: Sequence[int], k: int = 2, radius: float = 1) -> Array:
    """Stacked lookup tables (len(rules), k**inputs), int8, for a list of rule numbers."""
    return jnp.asarray(np.stack([rule_table(r, k, radius) for r in rules]))


def _as_tables(rules, k, radius) -> Array:
    """Rule numbers (1D) become tables; a 2D array is taken to be tables already."""
    if np.ndim(rules) == 1:
        return rule_tables([int(r) for r in np.asarray(rules)], k, radius)
    tables = jnp.asarray(rules, jnp.int8)
    if tables.ndim != 2 or tables.shape[1] != k ** num_inputs(radius):
        raise ValueError(f"rule tables must have shape (K, {k ** num_inputs(radius)}), got {tables.shape}")
    return tables


def single_cell(width: int, value: int = 1) -> Array:
    """A lattice of zeros with one cell set, at position (width - 1) // 2 (Wolfram's CenterArray[1, width])."""
    return jnp.zeros(width, jnp.int8).at[(width - 1) // 2].set(value)


def _offset_patterns(radius, neighborhood):
    if radius == 0.5:
        patterns = {"alternating": ((0, 1), (-1, 0)), "left_center": ((-1, 0),), "center_right": ((0, 1),)}
        if neighborhood not in patterns:
            raise ValueError(f"neighborhood must be one of {NEIGHBORHOODS}, got {neighborhood!r}")
        return patterns[neighborhood]
    r = num_inputs(radius) // 2
    return (tuple(range(-r, r + 1)),)


def _shifted(cells: Array, d: int, boundary: str) -> Array:
    """cells[..., x + d] for every x."""
    if d == 0:
        return cells
    if boundary == "periodic":
        return jnp.roll(cells, -d, axis=-1)
    zeros = jnp.zeros(cells.shape[:-1] + (abs(d),), cells.dtype)
    return jnp.concatenate([cells[..., d:], zeros], -1) if d > 0 else jnp.concatenate([zeros, cells[..., :d]], -1)


def neighborhood_codes(cells: Array, t, *, k: int = 2, radius: float = 1, neighborhood: str = "alternating",
                       boundary: str = "periodic") -> Array:
    """Neighbourhood of every cell at step t as a base-k number, leftmost cell most significant. int32."""
    if boundary not in BOUNDARIES:
        raise ValueError(f"boundary must be one of {BOUNDARIES}, got {boundary!r}")

    def code(offsets):
        c = jnp.zeros(cells.shape, jnp.int32)
        for d in offsets:
            c = c * k + _shifted(cells, d, boundary).astype(jnp.int32)
        return c

    codes = [code(p) for p in _offset_patterns(radius, neighborhood)]
    return codes[0] if len(codes) == 1 else jnp.where(t % 2 == 0, codes[0], codes[1])


# ------------------------------------------------------------------------------------------------------ simulation

def _resolve_steps(Ts: int, num_steps: int | None) -> int:
    if num_steps is None:
        if Ts == 1:
            raise ValueError("num_steps is required when the rule array is shared across time (time axis of size 1)")
        return Ts
    if num_steps < 0 or (Ts > 1 and num_steps > Ts):
        raise ValueError(f"cannot run {num_steps} steps with a rule array of {Ts} time steps")
    return num_steps


def _columns(Ws: int, W: int) -> Array:
    if Ws not in (1, W):
        raise ValueError(f"rule array width {Ws} does not match the lattice width {W}")
    return jnp.arange(W) if Ws > 1 else jnp.zeros(W, jnp.int32)


def _run(step, ic: Array, num_steps: int) -> Array:
    def body(cells, t):
        new = jax.lax.optimization_barrier(step(cells, t))     # see stica.ca.SCAN_UNROLL
        return new, new

    _, traj = jax.lax.scan(body, ic, jnp.arange(num_steps), unroll=SCAN_UNROLL)
    return jnp.concatenate([ic[None], traj], 0)


def rule_array_step(cells: Array, t, ra: Array, rules: Array, *, k: int = 2, radius: float = 1,
                    neighborhood: str = "alternating", boundary: str = "periodic") -> Array:
    """One step of a rule array: cells (W,) at step t -> step t + 1. `rules` is a (K, k**inputs) table array."""
    Ts, Ws = ra.shape
    code = neighborhood_codes(cells, t, k=k, radius=radius, neighborhood=neighborhood, boundary=boundary)
    t_idx = jnp.minimum(t, Ts - 1) if Ts > 1 else 0
    new = rules[ra[t_idx, _columns(Ws, cells.shape[-1])].astype(jnp.int32), code].astype(cells.dtype)
    return jnp.where(t < Ts, new, cells) if Ts > 1 else new


def rule_array_rollout(ic, ra, rules, *, k: int = 2, radius: float = 1, neighborhood: str = "alternating",
                       boundary: str = "periodic", num_steps: int | None = None) -> Array:
    """Trajectory (num_steps + 1, W), int8, including the initial condition. `rules` is a list of rule numbers or a
    (K, k**inputs) table array; `ra` (Ts, Ws) holds indices into it. Use `simulate_rule_array`, the jitted version,
    unless you are already inside a jax transform."""
    ic, ra = jnp.asarray(ic, jnp.int8), jnp.asarray(ra)
    tables = _as_tables(rules, k, radius)
    num_steps = _resolve_steps(ra.shape[0], num_steps)
    step = lambda cells, t: rule_array_step(cells, t, ra, tables, k=k, radius=radius, neighborhood=neighborhood,
                                            boundary=boundary)
    return _run(step, ic, num_steps)


def table_array_rollout(ic, tables, *, k: int = 2, radius: float = 1, neighborhood: str = "alternating",
                        boundary: str = "periodic", num_steps: int | None = None) -> Array:
    """Generalised rule array in which every cell and step has its own full lookup table: tables (Ts, Ws, k**inputs),
    with Ts in {1, T} and Ws in {1, W}. Returns the trajectory (num_steps + 1, W), int8."""
    ic, tables = jnp.asarray(ic, jnp.int8), jnp.asarray(tables, jnp.int8)
    Ts, Ws, size = tables.shape
    if size != k ** num_inputs(radius):
        raise ValueError(f"tables must have {k ** num_inputs(radius)} entries per cell, got {size}")
    num_steps = _resolve_steps(Ts, num_steps)
    cols = _columns(Ws, ic.shape[-1])

    def step(cells, t):
        code = neighborhood_codes(cells, t, k=k, radius=radius, neighborhood=neighborhood, boundary=boundary)
        t_idx = jnp.minimum(t, Ts - 1) if Ts > 1 else 0
        new = tables[t_idx, cols, code]
        return jnp.where(t < Ts, new, cells) if Ts > 1 else new

    return _run(step, ic, num_steps)


_rule_array_jit = jax.jit(rule_array_rollout, static_argnames=_STATIC)
simulate_table_array = jax.jit(table_array_rollout, static_argnames=_STATIC)


def simulate_rule_array(ic, ra, rules, *, k: int = 2, radius: float = 1, neighborhood: str = "alternating",
                        boundary: str = "periodic", num_steps: int | None = None) -> Array:
    """Jitted `rule_array_rollout`. Rule numbers are turned into lookup tables before compilation, so they may be
    arbitrarily large integers."""
    return _rule_array_jit(ic, ra, _as_tables(rules, k, radius), k=k, radius=radius, neighborhood=neighborhood,
                           boundary=boundary, num_steps=num_steps)


# ------------------------------------------------------------------------------------------------------ objectives

def lifetime(traj: Array) -> Array:
    """Wolfram's lifetime of a trajectory (T + 1, W): 0 if the initial state is blank, otherwise the first step
    t >= 1 at which every cell is 0, or T + 1 if the pattern survives all T steps."""
    alive = jnp.any(traj != 0, axis=-1)
    if traj.shape[0] == 1:
        return jnp.where(alive[0], 1, 0).astype(jnp.int32)
    died = ~alive[1:]
    first = jnp.where(died.any(), jnp.argmax(died) + 1, traj.shape[0])
    return jnp.where(alive[0], first, 0).astype(jnp.int32)


@dataclasses.dataclass(frozen=True)
class LifetimeLoss:
    """|lifetime - target|, or +inf if the pattern outlives the rule array (Wolfram's LifetimeLossValue)."""
    target: int

    def __call__(self, traj: Array) -> Array:
        life = lifetime(traj)
        return jnp.where(life == traj.shape[0], jnp.inf, jnp.abs(life - self.target)).astype(jnp.float32)


# ------------------------------------------------------------------------------------------------------- evolution

class RuleArrayResult(NamedTuple):
    ra: Array             # best rule array found
    best: Array           # its loss
    initial: Array        # loss of the starting rule array
    accepted: Array       # (iters,) whether each candidate was kept
    best_trace: Array     # (iters,) best loss after each iteration
    solved_at: Array      # mutations until the loss first reached stop_at (0: at the start, -1: never or no stop_at)
    mutations: Mutations | None


@dataclasses.dataclass(frozen=True)
class _RuleArrayLoss:
    """objective(trajectory) of a rule array, given the IC and the rule tables; hashable for jit."""
    k: int
    radius: float
    neighborhood: str
    boundary: str
    num_steps: int | None
    objective_fn: Callable

    def __call__(self, ic, ra, tables):
        return self.objective_fn(rule_array_rollout(ic, ra, tables, k=self.k, radius=self.radius,
                                                    neighborhood=self.neighborhood, boundary=self.boundary,
                                                    num_steps=self.num_steps))


def _evolve_args(ic, ra, rules, *, objective, k, radius, neighborhood, boundary, num_steps, mutations):
    ic, ra = jnp.asarray(ic, jnp.int8), jnp.asarray(ra, jnp.int8)
    tables = _as_tables(rules, k, radius)
    if tables.shape[0] < 2:
        raise ValueError("evolution needs at least two rules to switch between")
    _check_counts(ic.shape, ra.shape, 0, mutations)
    _columns(ra.shape[1], ic.shape[-1])
    _resolve_steps(ra.shape[0], num_steps)
    loss_fn = _RuleArrayLoss(k, radius, neighborhood, boundary, num_steps, objective)
    return ic, ra, tables, dict(loss_fn=loss_fn, ic_states=k, param_states=tables.shape[0], ic_count=0,
                                param_count=mutations)


def _result(out, stop_at) -> RuleArrayResult:
    _, ra, best, best0, accepted, trace, muts = out
    solved = steps_to_target(best0, trace, stop_at) if stop_at is not None else jnp.full_like(best0, -1, jnp.int32)
    return RuleArrayResult(ra, best, best0, accepted, trace, solved, muts)


def evolve_rule_array(key: Array, ic, ra, rules, *, objective: Callable, iterations: int, k: int = 2,
                      radius: float = 1, neighborhood: str = "alternating", boundary: str = "periodic",
                      num_steps: int | None = None, mutations: int = 1, accept_ties: bool = True,
                      stop_at: float | None = None, record_mutations: bool = False) -> RuleArrayResult:
    """Adaptive evolution of a rule array (Wolfram's procedure). Each iteration switches the rule of `mutations`
    distinct cells to a uniformly chosen different rule from `rules` and keeps the candidate if `objective`
    (trajectory -> scalar, minimised) does not increase (`accept_ties=False`: only if it decreases). With `stop_at`,
    the array stops changing once the loss is <= stop_at, as Wolfram's runs stop at zero loss. The initial condition
    is fixed. `objective` must be hashable and stable (a module-level function or a frozen dataclass such as
    `LifetimeLoss`), because it is a static argument of the compiled loop."""
    ic, ra, tables, static = _evolve_args(ic, ra, rules, objective=objective, k=k, radius=radius,
                                          neighborhood=neighborhood, boundary=boundary, num_steps=num_steps,
                                          mutations=mutations)
    out = hill_climb(key, ic, ra, tables, iterations=iterations, record=record_mutations, accept_ties=accept_ties,
                     stop_at=stop_at, **static)
    return _result(out, stop_at)


def evolve_rule_array_batch(key: Array, num_runs: int, ic, ra, rules, **kwargs) -> RuleArrayResult:
    """`num_runs` independent evolutions from the same starting rule array, in parallel (one PRNG key each).
    Takes the keyword arguments of `evolve_rule_array`; every field of the result gains a leading run axis."""
    iterations, record = kwargs.pop("iterations"), kwargs.pop("record_mutations", False)
    accept_ties, stop_at = kwargs.pop("accept_ties", True), kwargs.pop("stop_at", None)
    ic, ra, tables, static = _evolve_args(ic, ra, rules, **{"k": 2, "radius": 1, "neighborhood": "alternating",
                                                             "boundary": "periodic", "num_steps": None,
                                                             "mutations": 1, **kwargs})
    ics = jnp.broadcast_to(ic, (num_runs,) + ic.shape)
    ras = jnp.broadcast_to(ra, (num_runs,) + ra.shape)
    out = hill_climb_batch(jax.random.split(key, num_runs), ics, ras, tables, iterations=iterations, record=record,
                           accept_ties=accept_ties, stop_at=stop_at, **static)
    return _result(out, stop_at)
