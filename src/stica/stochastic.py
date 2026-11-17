"""Stochastic rule arrays: probabilistic rules, asynchronous updates and noisy trajectories.

Three ways to make a binary two-rule array stochastic, alone or together. Every run draws its own randomness:

- probabilistic rules: cell x at step t uses the second rule with probability p[t, x] and the first otherwise;
- asynchronous updates: at each step each cell updates with probability `alpha` and otherwise keeps its state
  (alpha-asynchronous cellular automata, Fates);
- noisy trajectories: after each step each cell's new state flips with probability `epsilon`.

Step t of a run is therefore

    new(x) = f_{M[t,x]}(neighbourhood_t(x))   if U[t, x] else old(x),      then new(x) ^= N[t, x],

with independent Bernoulli masks M (probability p), U (alpha) and N (epsilon). Deterministic STICA is the case
p in {0, 1}, alpha = 1, epsilon = 0, which these functions reproduce bit for bit.

Probabilities are dyadic: p = q / 2**bits with integer q in [0, 2**bits] (`quantize`). This is what makes sampling
cheap on bit-packed words: a mask word whose 32 bits are each 1 with probability q / 2**b is the bitwise comparison
R < q of b random words R read as b-bit numbers (one bit of each random word per cell), so it costs b random words
and about 3b bitwise operations; a per-cell q is held as b + 1 bit-planes. For a probability shared by every cell only
the significant bits of q need random words: epsilon = 2**-10 costs 10.

`stochastic_rollout` is the deterministic core given explicit masks (tested against a NumPy reference);
`simulate_stochastic` samples the masks and runs many independent runs; `evolve_stochastic_batch` is Wolfram's
adaptive evolution with a stochastic fitness: a candidate's score is its mean loss over `samples` runs, and parent and
child are scored on the same random draws (common random numbers), so that most of the sampling noise cancels in the
comparison. The genome is either a deterministic rule array under stochastic dynamics, or a probabilistic one.
"""
from __future__ import annotations

import dataclasses
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .ca import SCAN_UNROLL
from .packed import _BLOCK, _Model, _fused_supported, _step, _tmask, _valid_mask, fused_rollout, num_words, pack_cells
from .rulearray import _resolve_steps, num_inputs

_U = jnp.uint32
_ONES = 0xFFFFFFFF


# ------------------------------------------------------------------------------------------------------- sampling

def quantize(p, bits: int) -> np.ndarray:
    """Probabilities -> integers q = round(p * 2**bits) in [0, 2**bits]."""
    return np.clip(np.rint(np.asarray(p, np.float64) * 2 ** bits), 0, 2 ** bits).astype(np.int64)


def threshold_planes(q, bits: int) -> Array:
    """Per-cell thresholds q (..., W) in [0, 2**bits] -> packed bit-planes (bits + 1, ..., nw); plane j holds bit j."""
    q = jnp.asarray(q, jnp.int32)
    return jnp.stack([pack_cells((q >> j) & 1) for j in range(bits + 1)])


def _below(random_words, planes, bits: int) -> Array:
    """Bitwise R < q for the b-bit numbers R whose bit j is random_words[j], against q in [0, 2**b] given as planes
    (b + 1 of them; plane b set means q = 2**b, probability 1)."""
    lt = jnp.zeros_like(random_words[0]) if bits else jnp.zeros_like(planes[0])
    eq = ~lt
    for j in range(bits - 1, -1, -1):
        r, qj = random_words[j], planes[j]
        lt = lt | (eq & ~r & qj)
        eq = eq & ~(r ^ qj)
    return lt | planes[bits]


def _scalar_planes(q: int, bits: int) -> list:
    return [_U(_ONES) if (q >> j) & 1 else _U(0) for j in range(bits + 1)]


def bernoulli_words(key: Array, q, bits: int, shape: tuple) -> Array:
    """uint32 words of `shape` whose bits are independently 1 with probability q / 2**bits. `q` is an int shared by
    every bit, or planes from `threshold_planes` broadcastable to `shape` (one threshold per cell). Padding bits are
    not cleared (callers mask them)."""
    if isinstance(q, (int, np.integer)):
        q = int(q)
        if q <= 0:
            return jnp.zeros(shape, _U)
        if q >= 2 ** bits:
            return jnp.full(shape, _ONES, _U)
        z = (q & -q).bit_length() - 1                     # trailing zero bits of q: R's low bits do not matter
        b, qq = bits - z, q >> z
        return _below(jax.random.bits(key, (b,) + tuple(shape), _U), _scalar_planes(qq, b), b)
    return jnp.broadcast_to(_below(jax.random.bits(key, (bits,) + tuple(shape), _U), q, bits), shape)


@dataclasses.dataclass(frozen=True)
class Dynamics:
    """Stochastic dynamics: `alpha` = probability that a cell updates at a step (1: synchronous), `epsilon` =
    probability that a cell's new state flips (0: noiseless); both in multiples of 2**-bits."""
    alpha: float = 1.0
    epsilon: float = 0.0
    bits: int = 16

    @property
    def q_alpha(self) -> int:
        return int(quantize(self.alpha, self.bits))

    @property
    def q_epsilon(self) -> int:
        return int(quantize(self.epsilon, self.bits))

    @property
    def synchronous(self) -> bool:
        return self.q_alpha >= 2 ** self.bits

    @property
    def noiseless(self) -> bool:
        return self.q_epsilon == 0


def _dynamics_masks(key, dyn: Dynamics, shape, valid):
    """(update or None, noise or None) words of `shape` (..., nw, lanes), padding bits cleared."""
    ku, kn = jax.random.split(key)
    upd = None if dyn.synchronous else bernoulli_words(ku, dyn.q_alpha, dyn.bits, shape) & valid
    noise = None if dyn.noiseless else bernoulli_words(kn, dyn.q_epsilon, dyn.bits, shape) & valid
    return upd, noise


# ----------------------------------------------------------------------------------------------------- simulation

def _xla_rollout(ic_words, rule_words, upd, noise, tmask, model: _Model, num_steps: int):
    """One run on packed words (any radius, XLA): rule_words (T, nw) = 'uses the second rule', upd / noise (T, nw)."""
    mask = jnp.asarray(_valid_mask(model.width))
    planes = rule_words[None]

    def body(cells, t):
        new = _step(cells, t, planes, tmask, model)
        if upd is not None:
            new = (new & upd[t]) | (cells & ~upd[t])
        if noise is not None:
            new = (new ^ noise[t]) & mask
        new = jax.lax.optimization_barrier(new)
        return new, new

    _, traj = jax.lax.scan(body, ic_words, jnp.arange(num_steps), unroll=SCAN_UNROLL)
    return jnp.concatenate([ic_words[None], traj], 0)


@partial(jax.jit, static_argnames=("model", "num_steps", "rules", "backend", "rows_only"))
def _rollout_lanes(ic_words, rule_masks, upd, noise, *, model, num_steps, rules, backend, rows_only=False):
    """Lane-major masks (T, nw, L) -> packed trajectories (T + 1, nw or 1, L)."""
    L = rule_masks.shape[-1]
    if backend == "pallas":
        ic_b = jnp.broadcast_to(ic_words[:, None], (ic_words.shape[0], L))
        return fused_rollout(ic_b, rule_masks, upd, noise, rules=rules, width=model.width, radius=model.radius,
                             neighborhood=model.neighborhood, boundary=model.boundary, rows_only=rows_only)
    tmask = _tmask(list(rules), model.radius)
    lane = lambda a: None if a is None else jnp.moveaxis(a, -1, 0)
    traj = jax.vmap(lambda r, u, n: _xla_rollout(ic_words, r, u, n, tmask, model, num_steps),
                    in_axes=(0, None if upd is None else 0, None if noise is None else 0))(
        lane(rule_masks), lane(upd), lane(noise))                                    # (L, T + 1, nw)
    if rows_only:
        traj = jnp.any(traj != 0, axis=-1, keepdims=True).astype(_U)                 # (L, T + 1, 1)
    return jnp.moveaxis(traj, 0, -1)


def _backend(backend: str, radius: float) -> str:
    if backend == "auto":
        return "pallas" if jax.default_backend() == "gpu" and _fused_supported(2, radius) else "xla"
    if backend == "pallas" and not _fused_supported(2, radius):
        raise ValueError("the fused kernel handles radius 1 or 1/2")
    if backend not in ("pallas", "xla"):
        raise ValueError(f"backend must be 'auto', 'pallas' or 'xla', got {backend!r}")
    return backend


def _lanes(n: int, backend: str) -> int:
    return -(-n // _BLOCK) * _BLOCK if backend == "pallas" else n


def stochastic_rollout(ic, rule_mask, rules, *, update_mask=None, noise_mask=None, radius: float = 1,
                       neighborhood: str = "alternating", boundary: str = "periodic", backend: str = "auto") -> Array:
    """One run with explicit 0/1 masks of shape (T, W): rule_mask (1 = the second rule), update_mask (1 = the cell
    updates; None: all do), noise_mask (1 = the new state flips; None: none). Returns the trajectory (T + 1, W) int8.
    The deterministic core of `simulate_stochastic`."""
    from .packed import unpack_cells
    ic = jnp.asarray(ic)
    T, W = np.shape(rule_mask)
    backend = _backend(backend, radius)
    num_inputs(radius)
    model = _Model(W, 2, radius, neighborhood, boundary)
    L = _lanes(1, backend)
    lanes = lambda m: (None if m is None else
                       jnp.broadcast_to(pack_cells(jnp.asarray(m))[..., None], (T, num_words(W), L)))
    out = _rollout_lanes(pack_cells(ic), lanes(rule_mask), lanes(update_mask), lanes(noise_mask), model=model,
                         num_steps=T, rules=tuple(int(r) for r in rules), backend=backend)
    return unpack_cells(out[..., 0], W)


def simulate_stochastic(key: Array, ic, rules, *, ra=None, prob=None, alpha: float = 1.0, epsilon: float = 0.0,
                        bits: int = 16, rule_bits: int = 8, num_samples: int = 1, radius: float = 1,
                        neighborhood: str = "alternating", boundary: str = "periodic", num_steps: int | None = None,
                        backend: str = "auto", packed_output: bool = False) -> Array:
    """`num_samples` independent runs of a two-rule binary array. Give either `ra` (T, W) of 0/1 (a deterministic rule
    array) or `prob` (T, W), each cell's probability of using rules[1] (quantised to multiples of 2**-rule_bits).
    `alpha`, `epsilon` (multiples of 2**-bits) as in `Dynamics`. Returns (num_samples, T + 1, W) int8, or packed
    words (num_samples, T + 1, nw) uint32 with packed_output."""
    from .packed import unpack_cells
    if (ra is None) == (prob is None):
        raise ValueError("give exactly one of ra (a rule array) and prob (rule probabilities)")
    if len(rules) != 2:
        raise ValueError("stochastic rule arrays have two rules")
    ic = jnp.asarray(ic)
    arr = jnp.asarray(ra if ra is not None else prob)
    T = _resolve_steps(arr.shape[0], num_steps if num_steps is not None else arr.shape[0])
    W = ic.shape[-1]
    arr = jnp.broadcast_to(arr, (T, W))
    backend = _backend(backend, radius)
    model = _Model(W, 2, radius, neighborhood, boundary)
    nw, L = num_words(W), _lanes(num_samples, backend)
    valid = jnp.asarray(_valid_mask(W))[:, None]
    kr, kd = jax.random.split(key)
    if ra is not None:
        rule_masks = jnp.broadcast_to(pack_cells(arr.astype(jnp.int32) == 1)[..., None], (T, nw, L))
    else:
        planes = threshold_planes(quantize(np.asarray(arr), rule_bits), rule_bits)[..., None]   # (b+1, T, nw, 1)
        rule_masks = bernoulli_words(kr, planes, rule_bits, (T, nw, L)) & valid
    upd, noise = _dynamics_masks(kd, Dynamics(alpha, epsilon, bits), (T, nw, L), valid)
    out = _rollout_lanes(pack_cells(ic), rule_masks, upd, noise, model=model, num_steps=T,
                         rules=tuple(int(r) for r in rules), backend=backend)
    out = jnp.moveaxis(out, -1, 0)[:num_samples]
    return out if packed_output else unpack_cells(out, W)


# ------------------------------------------------------------------------------------------------------ evolution

class StochasticEvolution(NamedTuple):
    genome: Array             # (runs, T, W): q levels (probabilistic genome) or 0/1 (deterministic), at the end
    score: Array              # (runs,) mean loss of that genome on the last iteration's samples (inf counted as T + 1)
    success: Array            # (runs,) fraction of those samples whose loss was 0
    reliable_at: Array        # (runs,) first iteration whose kept genome succeeded on every sample, -1 if none
    trace_score: Array | None     # (runs, iterations) score of the kept genome, if record_trace
    trace_success: Array | None   # (runs, iterations)


def evolve_stochastic_batch(key: Array, num_runs: int, ic, rules, *, target: int, iterations: int, num_steps: int,
                            samples: int = 64, genome: str = "probabilistic", start=None, alpha: float = 1.0,
                            epsilon: float = 0.0, bits: int = 16, rule_bits: int = 4, radius: float = 1,
                            neighborhood: str = "alternating", boundary: str = "periodic", backend: str = "auto",
                            record_trace: bool = False) -> StochasticEvolution:
    """Wolfram's adaptive evolution of `num_runs` independent two-rule arrays (T = num_steps, W = len(ic)) towards
    lifetime `target`, with a stochastic fitness: each iteration mutates one cell of every run and scores parent and
    child on the same `samples` random runs (common random numbers); the child is kept unless its mean loss is higher
    (ties kept, as in Wolfram's rule). Loss = |lifetime - target|, with a pattern that outlives the array counted as
    T + 1 (larger than any finite loss, so the order of losses is Wolfram's).

    genome "deterministic": a 0/1 rule array (a mutation switches a cell's rule); "probabilistic": each cell holds
    q in 0..2**rule_bits, its probability of the second rule being q / 2**rule_bits (a mutation moves q to another
    level, uniformly). `start`: the starting genome (T, W) (default all zeros: the first rule everywhere). With
    genome="deterministic", samples=1, alpha=1 and epsilon=0 this is Wolfram's algorithm. The parent is rescored on
    fresh samples every iteration, so a lucky estimate does not persist."""
    from .packed import unpack_cells
    if genome not in ("probabilistic", "deterministic"):
        raise ValueError("genome must be 'probabilistic' or 'deterministic'")
    if len(rules) != 2:
        raise ValueError("stochastic rule arrays have two rules")
    ic = jnp.asarray(ic)
    W, T = ic.shape[-1], int(num_steps)
    backend = _backend(backend, radius)
    model = _Model(W, 2, radius, neighborhood, boundary)
    start = jnp.zeros((T, W), jnp.int32) if start is None else jnp.asarray(start, jnp.int32)
    planes0 = threshold_planes(start, rule_bits) if genome == "probabilistic" else pack_cells(start == 1)[None]
    P, reliable, score, success, trace = _run_evolution(
        key, pack_cells(ic), planes0, jnp.int32(target), model=model, rules=tuple(int(r) for r in rules),
        genome=genome, iterations=int(iterations), samples=int(samples), rule_bits=int(rule_bits),
        dyn=Dynamics(alpha, epsilon, bits), backend=backend, record_trace=record_trace, num_runs=int(num_runs))
    cells = unpack_cells(jnp.moveaxis(P, -1, 0), W).astype(jnp.int32)                 # (R, nplanes, T, W)
    g = jnp.sum(cells << jnp.arange(P.shape[0])[None, :, None, None], axis=1)
    ts, tsu = trace if record_trace else (None, None)
    return StochasticEvolution(g, score, success, reliable, ts, tsu)


def _lifetime_loss_rows(rows, target, T: int):
    """Loss per lane from row-OR words (T + 1, 1, L): |lifetime - target|, T + 1 for a pattern that survives."""
    alive = rows[:, 0, :] != 0
    died = ~alive[1:]
    life = jnp.where(died.any(0), jnp.argmax(died, 0) + 1, T + 1)
    life = jnp.where(alive[0], life, 0)
    return jnp.where(life == T + 1, jnp.float32(T + 1), jnp.abs(life - target).astype(jnp.float32))


@partial(jax.jit, static_argnames=("model", "rules", "genome", "iterations", "samples", "rule_bits", "dyn",
                                   "backend", "record_trace", "num_runs"))
def _run_evolution(key, ic_words, planes0, target, *, model, rules, genome, iterations, samples, rule_bits, dyn,
                   backend, record_trace, num_runs):
    """All runs advance together; planes (nplanes, T, nw, R) hold every run's genome (threshold planes of q, or the
    single rule plane). One iteration = 2 * R * samples rollouts in one launch (parents, then children)."""
    nplanes, T, nw = planes0.shape
    S, R = samples, num_runs
    vmask = jnp.asarray(_valid_mask(model.width))[:, None, None]
    levels = 2 ** rule_bits + 1
    runs = jnp.arange(R)
    L = 2 * R * S
    pad = _lanes(L, backend) - L
    lanes = lambda a: jnp.pad(a, ((0, 0), (0, 0), (0, pad)))

    def scores(key, parent, child):
        """Mean loss and success fraction (2, R) of parent and child on common random draws."""
        kr, kd = jax.random.split(key)
        shape = (T, nw, R, S)
        if genome == "probabilistic":
            rw = jax.random.bits(kr, (rule_bits, T, nw, R, S), _U)
            masks = [jnp.broadcast_to(_below(rw, p[..., None], rule_bits) & vmask, shape) for p in (parent, child)]
        else:
            masks = [jnp.broadcast_to(p[0][..., None], shape) for p in (parent, child)]
        upd, noise = _dynamics_masks(kd, dyn, shape, vmask)
        both = lambda a: None if a is None else lanes(jnp.concatenate([a.reshape(T, nw, R * S)] * 2, -1))
        rule_masks = lanes(jnp.concatenate([m.reshape(T, nw, R * S) for m in masks], -1))
        rows = _rollout_lanes(ic_words, rule_masks, both(upd), both(noise), model=model, num_steps=T, rules=rules,
                              backend=backend, rows_only=True)
        loss = _lifetime_loss_rows(rows[..., :L], target, T).reshape(2, R, S)
        return loss.mean(-1), (loss == 0).mean(-1)

    def iterate(carry, xs):
        P, reliable, _, _ = carry
        key, i = xs
        kt, kx, kv, ks = jax.random.split(key, 4)
        t = jax.random.randint(kt, (R,), 0, T)
        x = jax.random.randint(kx, (R,), 0, model.width)
        w, b = x // 32, (x % 32).astype(_U)
        old_words = P[:, t, w, runs]                                                 # (nplanes, R)
        if genome == "probabilistic":
            cur = jnp.sum(((old_words >> b) & 1).astype(jnp.int32) << jnp.arange(nplanes)[:, None], axis=0)
            new = (cur + jax.random.randint(kv, (R,), 1, levels)) % levels
            new_bits = ((new[None, :] >> jnp.arange(nplanes)[:, None]) & 1).astype(_U)
            new_words = (old_words & ~jnp.left_shift(_U(1), b)) | (new_bits << b)
        else:
            new_words = old_words ^ jnp.left_shift(_U(1), b)
        child = P.at[:, t, w, runs].set(new_words)
        sc, su = scores(ks, P, child)
        accept = sc[1] <= sc[0]
        P = P.at[:, t, w, runs].set(jnp.where(accept[None], new_words, old_words))
        score, success = jnp.where(accept, sc[1], sc[0]), jnp.where(accept, su[1], su[0])
        reliable = jnp.where((reliable < 0) & (success >= 1.0), i + 1, reliable)
        return (P, reliable, score, success), ((score, success) if record_trace else None)

    P0 = jnp.broadcast_to(planes0[..., None], planes0.shape + (R,))
    zero = jnp.zeros(R, jnp.float32)
    xs = (jax.random.split(key, iterations), jnp.arange(iterations, dtype=jnp.int32))
    (P, reliable, score, success), trace = jax.lax.scan(iterate, (P0, jnp.full(R, -1, jnp.int32), zero, zero), xs)
    trace = tuple(a.T for a in trace) if record_trace else None
    return P, reliable, score, success, trace


# ---------------------------------------------------------------------------------------------------- evaluation

class Reliability(NamedTuple):
    success: Array            # (N,) fraction of samples whose lifetime hit the target exactly
    mean_loss: Array          # (N,) mean |lifetime - target| (a surviving pattern counted as T + 1)
    lifetimes: Array | None   # (N, samples) int32, if keep_lifetimes


def success_probability(key: Array, ic, rules, genomes, *, target: int, genome: str = "deterministic",
                        rule_bits: int = 4, alpha: float = 1.0, epsilon: float = 0.0, flip: float = 0.0,
                        bits: int = 16, samples: int = 256, radius: float = 1, neighborhood: str = "alternating",
                        boundary: str = "periodic", backend: str = "auto", max_lanes: int = 65536,
                        keep_lifetimes: bool = False) -> Reliability:
    """How reliably each of N genomes (N, T, W) reaches lifetime `target` under stochastic conditions, estimated from
    `samples` independent runs each: `alpha`, `epsilon` as in `Dynamics`; `flip` = probability that a cell uses the
    other rule than the genome says (deterministic genomes; a rule-level perturbation). Probabilistic genomes hold
    levels q (probability q / 2**rule_bits of the second rule)."""
    genomes = np.asarray(genomes)
    N, T, W = genomes.shape
    backend = _backend(backend, radius)
    model = _Model(W, 2, radius, neighborhood, boundary)
    nw, S = num_words(W), int(samples)
    dyn = Dynamics(alpha, epsilon, bits)
    q_flip = int(quantize(flip, bits))
    chunk = max(1, min(N, max_lanes // S))
    ic_words = pack_cells(jnp.asarray(ic))
    rules = tuple(int(r) for r in rules)

    @partial(jax.jit, static_argnames=())
    def run(key, g):
        n = g.shape[0]
        vmask = jnp.asarray(_valid_mask(W))[:, None, None]
        shape = (T, nw, n, S)
        kr, kf, kd = jax.random.split(key, 3)
        if genome == "probabilistic":
            planes = jnp.moveaxis(threshold_planes(g, rule_bits), 1, -1)[..., None]  # (b+1, T, nw, n, 1)
            masks = jnp.broadcast_to(bernoulli_words(kr, planes, rule_bits, shape) & vmask, shape)
        else:
            masks = jnp.broadcast_to(jnp.moveaxis(pack_cells(g == 1), 0, -1)[..., None], shape)
            if q_flip:
                masks = masks ^ (bernoulli_words(kf, q_flip, bits, shape) & vmask)
        upd, noise = _dynamics_masks(kd, dyn, shape, vmask)
        L = n * S
        pad = _lanes(L, backend) - L
        flat = lambda a: None if a is None else jnp.pad(a.reshape(T, nw, L), ((0, 0), (0, 0), (0, pad)))
        rows = _rollout_lanes(ic_words, flat(masks), flat(upd), flat(noise), model=model, num_steps=T, rules=rules,
                              backend=backend, rows_only=True)[..., :L]
        loss = _lifetime_loss_rows(rows, target, T).reshape(n, S)
        alive = rows[:, 0, :] != 0
        died = ~alive[1:]
        life = jnp.where(died.any(0), jnp.argmax(died, 0) + 1, T + 1).reshape(n, S)
        return (loss == 0).mean(-1), loss.mean(-1), life

    out = [run(jax.random.fold_in(key, i), jnp.asarray(genomes[i:i + chunk], jnp.int32))
           for i in range(0, N, chunk)]
    cat = lambda j: jnp.concatenate([o[j] for o in out])
    return Reliability(cat(0), cat(1), cat(2) if keep_lifetimes else None)
