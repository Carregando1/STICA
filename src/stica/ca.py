"""STICA as a CAX cellular automaton, with time carried in a state channel.

State layout: an int32 array (H, W, 2). Channel 0 holds the cell states (0..n-1); channel 1 holds the time step t,
identical in every cell. One update step does

    cell'(x, y) = srt[t, x, y, rule_index(cell(x, y), neighbour counts)]
    t'          = t + 1                       (always, whatever the cells do)

so STICA becomes an ordinary, autonomous CA on this augmented state: the next state depends only on the current state,
`ca.step(state)` needs no external input, and a run can be paused and resumed. The spacetime rule tensor (SRT) is a
constant parameter of the update module.

Rule sharing: the SRT has shape (Ts, Hs, Ws, R) with Ts in {1, T}, Hs in {1, H}, Ws in {1, W}. A size-1 axis shares
rules along it: (T, H, W, R) is full STICA, (T, 1, 1, R) varies rules only in time, (1, H, W, R) only in space, and
(1, 1, 1, R) is a homogeneous outer-totalistic CA. With Ts = T > 1 the rules are defined for t < T; cells are frozen
(state unchanged) for t >= T. With Ts = 1 the rules apply at every step.
"""
from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from cax.core.ca import CA
from cax.core.perceive.perceive import Perceive
from cax.core.update.update import Update
from flax import nnx
from jax import Array

from .rules import neighbor_counts, rule_index, validate_srt

CELL, TIME = 0, 1

# state 0 is white; further states cycle through this palette when rendering
_PALETTE = np.array(
    [[255, 255, 255], [31, 56, 100], [230, 120, 30], [60, 150, 70], [130, 80, 160],
     [200, 50, 50], [40, 160, 170], [210, 180, 40], [120, 120, 120], [0, 0, 0]], np.uint8)


def initial_state(ic, t0: int = 0) -> Array:
    """Augmented state (H, W, 2) from an initial condition (H, W)."""
    ic = jnp.asarray(ic, jnp.int32)
    return jnp.stack([ic, jnp.full_like(ic, t0)], axis=-1)


def stica_update(state: Array, counts: Array, srt: Array, n: int) -> Array:
    """One STICA update on the augmented state, given precomputed neighbour counts."""
    cell = state[..., CELL]
    t = state[0, 0, TIME]
    Ts, Hs, Ws, _ = srt.shape
    H, W = cell.shape
    t_idx = jnp.minimum(t, Ts - 1) if Ts > 1 else 0
    rows = jnp.arange(H)[:, None] if Hs > 1 else jnp.zeros((H, 1), jnp.int32)
    cols = jnp.arange(W)[None, :] if Ws > 1 else jnp.zeros((1, W), jnp.int32)
    new = srt[t_idx, rows, cols, rule_index(cell, counts, n)].astype(state.dtype)
    if Ts > 1:
        new = jnp.where(t < Ts, new, cell)       # rules undefined past the end of the SRT: freeze
    return jnp.stack([new, state[..., TIME] + 1], axis=-1)


def stica_step(state: Array, srt: Array, n: int) -> Array:
    return stica_update(state, neighbor_counts(state[..., CELL], n), srt, n)


def _resolve_steps(Ts: int, num_steps: int | None, t0: int) -> int:
    if num_steps is None:
        if Ts == 1:
            raise ValueError("num_steps is required when the SRT is shared across time (time axis of size 1)")
        num_steps = Ts - t0
    if num_steps < 0 or (Ts > 1 and t0 + num_steps > Ts):
        raise ValueError(f"cannot run {num_steps} steps from t0={t0} with an SRT of {Ts} time steps")
    return num_steps


def rollout(ic: Array, srt: Array, n: int, num_steps: int | None = None, t0: int = 0) -> Array:
    """Pure-function simulation. Returns the cell trajectory (num_steps + 1, H, W), int32, including the IC.

    Same update as `STICA`; use `simulate` (the jitted version) unless you are already inside a jax transform.
    """
    num_steps = _resolve_steps(srt.shape[0], num_steps, t0)
    s0 = initial_state(ic, t0)

    def body(state, _):
        state = stica_step(state, srt, n)
        return state, state[..., CELL]

    _, cells = jax.lax.scan(body, s0, None, length=num_steps)
    return jnp.concatenate([s0[None, ..., CELL], cells], axis=0)


simulate = jax.jit(rollout, static_argnames=("n", "num_steps", "t0"))


class SpacetimeRules(nnx.Variable):
    """The SRT, held as an nnx.Variable so that replacing it never triggers recompilation."""


class STICAPerceive(Perceive):
    """Perceives per-state Moore neighbour counts of the cell channel (the time channel is not perceived)."""

    def __init__(self, num_states: int):
        self.num_states = num_states

    def __call__(self, state: Array) -> Array:
        return neighbor_counts(state[..., CELL], self.num_states)


class STICAUpdate(Update):
    """Looks up each cell's next state in the SRT at the current time and advances the time channel by one."""

    def __init__(self, srt, num_states: int):
        self.num_states = num_states
        self.srt = SpacetimeRules(jnp.asarray(srt, jnp.int8))

    def __call__(self, state: Array, perception: Array, input: Array | None = None) -> Array:
        return stica_update(state, perception, self.srt.value, self.num_states)


def _cell_channel(next_state, state, perception, input):
    return next_state[..., CELL]


class STICA(CA):
    """Spacetime-inhomogeneous cellular automaton as a CAX `CA`.

    Example:
        >>> ca = STICA(srt, num_states=2)
        >>> traj = ca.run(ic)                     # (T + 1, H, W)
        >>> state, _ = ca.step(initial_state(ic))  # one step at a time
    """

    def __init__(self, srt, *, num_states: int, metrics_fn: Callable = _cell_channel, check_values: bool = True):
        validate_srt(srt, num_states, check_values=check_values)
        super().__init__(STICAPerceive(num_states), STICAUpdate(srt, num_states), metrics_fn=metrics_fn)
        self.num_states = num_states

    @property
    def srt(self) -> Array:
        return self.update.srt.value

    def set_srt(self, srt, *, check_values: bool = True) -> None:
        """Replace the SRT. Keeping the same shape avoids recompilation."""
        validate_srt(srt, self.num_states, check_values=check_values)
        self.update.srt.value = jnp.asarray(srt, jnp.int8)

    def run(self, ic, num_steps: int | None = None, t0: int = 0) -> Array:
        """Trajectory of the cell channel (num_steps + 1, H, W), including the IC."""
        Ts, Hs, Ws, _ = self.srt.shape
        ic = jnp.asarray(ic)
        if Hs not in (1, ic.shape[0]) or Ws not in (1, ic.shape[1]):
            raise ValueError(f"IC shape {ic.shape} does not match SRT spatial shape {(Hs, Ws)}")
        num_steps = _resolve_steps(Ts, num_steps, t0)
        s0 = initial_state(ic, t0)
        if num_steps == 0:
            return s0[None, ..., CELL]
        _, cells = self(s0, num_steps=num_steps)
        return jnp.concatenate([s0[None, ..., CELL], cells], axis=0)

    def render(self, state: Array) -> Array:
        """RGB uint8 image (H, W, 3) of an augmented state (H, W, 2) or a cell array (H, W)."""
        cells = state[..., CELL] if state.ndim == 3 else state
        return jnp.asarray(_PALETTE)[cells % len(_PALETTE)]
