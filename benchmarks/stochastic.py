"""Stochastic against deterministic STICA (stica.stochastic) on Wolfram's lifetime task (rules 4 and 146).

    python benchmarks/stochastic.py cost|robustness|evolve [--out DIR] [--quick]

cost        rollouts per second of deterministic and stochastic runs (probabilistic rules, asynchronous updates,
            noise, all three), at Wolfram's size (64 x 31) and at 640 x 311, on the fused kernel and the XLA path;
            evolution iterations per second with a stochastic fitness
robustness  deterministic solutions (evolved without noise) run under noise, asynchrony and rule flips: the
            probability that each still reaches its target lifetime, against how many mutations it took and how many
            rule-146 cells it has
evolve      Wolfram's adaptive evolution with a stochastic fitness (parent and child on the same random runs): a
            deterministic genome under noise or asynchrony, and probabilistic genomes at several resolutions; how
            often and how fast runs reach solutions that succeed on every sample, and how the evolved genomes of each
            setting fare under every other setting (cross-evaluation)

Writes <out>/stochastic_<part>.json (default benchmarks/results/stochastic).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import socket
import subprocess
import time
import zlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import stica
from stica.packed import _Model, num_words, pack_cells, unpack_cells
from stica.stochastic import (Dynamics, _dynamics_masks, _lanes, _lifetime_loss_rows, _rollout_lanes, _valid_mask,
                              bernoulli_words, evolve_stochastic_batch, success_probability, threshold_planes)

ROOT = Path(__file__).resolve().parents[1]
RULES = (4, 146)
SMALL = dict(height=64, width=31, target=50)
LARGE = dict(height=640, width=311, target=500)
# test conditions: name -> keyword arguments of success_probability / the cost model
CONDITIONS = {"none": {}}
for k in (16, 14, 12, 10, 8, 6):
    CONDITIONS[f"noise 2^-{k}"] = dict(epsilon=2.0 ** -k)
for k in (12, 10, 8, 6, 4):
    CONDITIONS[f"async 1-2^-{k}"] = dict(alpha=1 - 2.0 ** -k)
for k in (12, 10, 8, 6, 4):
    CONDITIONS[f"flip 2^-{k}"] = dict(flip=2.0 ** -k)


def metadata() -> dict:
    gpu = None
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=20).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return dict(date=dt.datetime.now().isoformat(timespec="seconds"), host=socket.gethostname(), gpu=gpu,
                jax=jax.__version__, backend=jax.default_backend(), python=platform.python_version(),
                commit=subprocess.run(["git", "-C", str(ROOT), "describe", "--always", "--dirty"],
                                      capture_output=True, text=True).stdout.strip() or os.environ.get("STICA_COMMIT"),
                slurm_job=os.environ.get("SLURM_JOB_ID"))


def timed(fn, min_s=1.0):
    """Median seconds per call after a compile call and a warm-up."""
    jax.block_until_ready(fn(0))
    t0 = time.perf_counter()
    jax.block_until_ready(fn(1))
    one = time.perf_counter() - t0
    calls = max(3, int(min_s / max(one, 1e-6)))
    reps = []
    for r in range(5):
        t0 = time.perf_counter()
        for c in range(calls):
            out = fn(2 + r * calls + c)
        jax.block_until_ready(out)
        reps.append((time.perf_counter() - t0) / calls)
    return float(np.median(reps))


# --------------------------------------------------------------------------------------------------------- cost

def part_cost(args) -> dict:
    rows = []
    modes = {"deterministic": dict(), "probabilistic rules (16 levels)": dict(prob=True),
             "asynchronous (1-2^-8)": dict(alpha=1 - 2 ** -8), "noise (2^-10)": dict(epsilon=2 ** -10),
             "all three": dict(prob=True, alpha=1 - 2 ** -8, epsilon=2 ** -10)}
    backends = ["pallas", "xla"] if jax.default_backend() == "gpu" else ["xla"]
    for setup, lanes in ((SMALL, 32768 if not args.quick else 1024), (LARGE, 8192 if not args.quick else 256)):
        T, W = setup["height"], setup["width"]
        nw = num_words(W)
        model = _Model(W, 2, 1, "alternating", "periodic")
        ic = pack_cells(stica.single_cell(W))
        rng = np.random.default_rng(0)
        ra = (rng.random((T, W)) < 0.3).astype(np.int32)
        q = rng.integers(0, 17, (T, W))
        vmask = jnp.asarray(_valid_mask(W))[:, None]
        for backend in backends:
            L = lanes if backend == "pallas" else max(128, lanes // 32)    # the XLA path at a size it can manage
            for name, m in modes.items():
                dyn = Dynamics(m.get("alpha", 1.0), m.get("epsilon", 0.0), 16)
                fixed = jnp.broadcast_to(pack_cells(ra == 1)[..., None], (T, nw, L))
                planes = threshold_planes(q, 4)[..., None]

                @jax.jit
                def run(key):
                    kr, kd = jax.random.split(key)
                    masks = (bernoulli_words(kr, planes, 4, (T, nw, L)) & vmask) if m.get("prob") else fixed
                    upd, noise = _dynamics_masks(kd, dyn, (T, nw, L), vmask)
                    rows = _rollout_lanes(ic, masks, upd, noise, model=model, num_steps=T, rules=RULES,
                                          backend=backend, rows_only=True)
                    return _lifetime_loss_rows(rows, setup["target"], T).sum()

                sec = timed(lambda i: run(jax.random.PRNGKey(i)), 0.5 if args.quick else 2.0)
                rows.append(dict(size=f"{T}x{W}", backend=backend, mode=name, lanes=L, seconds_per_call=sec,
                                 rollouts_per_s=L / sec, cell_updates_per_s=L * T * W / sec))
                print(f"  cost {T}x{W} {backend:6s} {name:34s} {L:6d} lanes: {L / sec:.3e} rollouts/s", flush=True)
    # evolution with a stochastic fitness, Wolfram's size
    evo = []
    R, S = (256, 64) if not args.quick else (32, 8)
    its = 200 if not args.quick else 20
    for name, kw in (("deterministic genome, 1 sample", dict(genome="deterministic", samples=1)),
                     ("deterministic genome, noise 2^-10", dict(genome="deterministic", samples=S, epsilon=2 ** -10)),
                     ("probabilistic genome, 16 levels", dict(genome="probabilistic", samples=S, rule_bits=4))):
        fn = lambda i: evolve_stochastic_batch(jax.random.PRNGKey(i), R, stica.single_cell(31), RULES, target=50,
                                               iterations=its, num_steps=64, **kw).score
        sec = timed(fn, 0.5)
        evo.append(dict(setting=name, runs=R, samples=kw["samples"], iterations=its, seconds=sec,
                        iterations_per_s=its / sec, candidates_per_s=R * its / sec,
                        rollouts_per_s=2 * R * kw["samples"] * its / sec))
        print(f"  evolve {name:36s}: {R * its / sec:.3e} candidates/s, {2 * R * kw['samples'] * its / sec:.3e} "
              f"rollouts/s", flush=True)
    return dict(rows=rows, evolution=evo)


# --------------------------------------------------------------------------------------------------- robustness

def deterministic_solutions(n: int, seed: int, setup=SMALL):
    """n runs of Wolfram's evolution (no noise, his budget); the solved ones, their mutations and rule-146 counts."""
    res = stica.evolve_packed_batch(jax.random.PRNGKey(seed), n, stica.single_cell(setup["width"]),
                                    np.zeros((setup["height"], setup["width"]), np.int8), list(RULES),
                                    objective=stica.LifetimeLoss(setup["target"]), iterations=10_000, stop_at=0.0)
    first = np.asarray(res.solved_at)
    ok = first > 0
    ras = np.asarray(res.ra)[ok]
    return ras, first[ok], ras.reshape(len(ras), -1).sum(1)


def part_robustness(args) -> dict:
    n = 2048 if not args.quick else 64
    S = 256 if not args.quick else 32
    ras, muts, n146 = deterministic_solutions(n, 11)
    out = dict(solutions=int(len(ras)), samples=S, mutations=muts.tolist(), rule146=n146.tolist(), conditions={})
    for name, cond in CONDITIONS.items():
        t0 = time.perf_counter()
        rel = success_probability(jax.random.PRNGKey(zlib.crc32(name.encode())), stica.single_cell(31), RULES, ras,
                                  target=50, samples=S, **cond)
        succ, loss = np.asarray(rel.success), np.asarray(rel.mean_loss)
        out["conditions"][name] = dict(kwargs=cond, success=succ.round(4).tolist(), mean_loss=loss.round(3).tolist(),
                                       seconds=time.perf_counter() - t0)
        rho = (float(np.corrcoef(succ, np.log(muts))[0, 1]) if succ.std() > 0 else None)
        print(f"  robustness {name:14s}: mean success {succ.mean():.3f}, median {np.median(succ):.3f}, "
              f"P(>=0.9) {(succ >= 0.9).mean():.3f}, corr(success, log mutations) {rho}", flush=True)
    return out


# ------------------------------------------------------------------------------------------------------- evolve

EVOLVE = {
    "deterministic": dict(genome="deterministic", samples=1),
    "probabilistic 2 levels": dict(genome="probabilistic", rule_bits=0),
    "probabilistic 3 levels": dict(genome="probabilistic", rule_bits=1),
    "probabilistic 5 levels": dict(genome="probabilistic", rule_bits=2),
    "probabilistic 17 levels": dict(genome="probabilistic", rule_bits=4),
    "deterministic, noise 2^-12": dict(genome="deterministic", epsilon=2 ** -12),
    "deterministic, noise 2^-10": dict(genome="deterministic", epsilon=2 ** -10),
    "deterministic, async 1-2^-10": dict(genome="deterministic", alpha=1 - 2 ** -10),
    "deterministic, async 1-2^-8": dict(genome="deterministic", alpha=1 - 2 ** -8),
    "deterministic, noise 2^-12 + async 1-2^-10": dict(genome="deterministic", epsilon=2 ** -12, alpha=1 - 2 ** -10),
}
TESTS = ["none", "noise 2^-12", "noise 2^-10", "async 1-2^-10", "async 1-2^-8", "flip 2^-10", "flip 2^-8"]


def part_evolve(args) -> dict:
    R, S, its = (256, 64, args.iterations) if not args.quick else (32, 8, 300)
    out = dict(runs=R, samples=S, iterations=its, settings={}, tests=TESTS)
    for name, kw in EVOLVE.items():
        kw = dict(kw)
        kw.setdefault("samples", S)
        t0 = time.perf_counter()
        res = evolve_stochastic_batch(jax.random.PRNGKey(zlib.crc32(name.encode())), R, stica.single_cell(31), RULES,
                                      target=50, iterations=its, num_steps=64, record_trace=True, **kw)
        jax.block_until_ready(res.score)
        sec = time.perf_counter() - t0
        rel = np.asarray(res.reliable_at)
        trace = np.asarray(res.trace_success).mean(0)
        step = max(1, its // 200)
        genomes = np.asarray(res.genome)
        cross = {}
        for test in TESTS:
            cond = CONDITIONS[test]
            if kw["genome"] == "probabilistic" and "flip" in cond:
                continue
            ev = success_probability(jax.random.PRNGKey(7), stica.single_cell(31), RULES, genomes, target=50,
                                     genome=kw["genome"], rule_bits=kw.get("rule_bits", 4), samples=256, **cond)
            cross[test] = dict(mean_success=float(np.asarray(ev.success).mean()),
                               frac_reliable=float((np.asarray(ev.success) >= 0.99).mean()))
        out["settings"][name] = dict(kwargs={k: v for k, v in kw.items()}, seconds=sec,
                                     reliable_fraction=float((rel > 0).mean()), reliable_at=rel.tolist(),
                                     final_success=np.asarray(res.success).round(4).tolist(),
                                     final_score=np.asarray(res.score).round(3).tolist(),
                                     mean_success_trace=trace[::step].round(4).tolist(), trace_step=step,
                                     cross_evaluation=cross)
        med = np.median(rel[rel > 0]) if (rel > 0).any() else None
        print(f"  evolve {name:40s}: reliable {np.mean(rel > 0):.2f} (median at {med}), final success "
              f"{np.asarray(res.success).mean():.2f}, under none/noise 2^-10/async 1-2^-8: "
              f"{cross['none']['mean_success']:.2f}/{cross['noise 2^-10']['mean_success']:.2f}/"
              f"{cross['async 1-2^-8']['mean_success']:.2f}  ({sec:.0f} s)", flush=True)
    return out


PARTS = dict(cost=part_cost, robustness=part_robustness, evolve=part_evolve)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=list(PARTS))
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results" / "stochastic"))
    ap.add_argument("--iterations", type=int, default=20_000, help="evolve: mutations per run")
    ap.add_argument("--quick", action="store_true", help="small sizes, for a smoke test")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    print(f"{args.part} on {socket.gethostname()} ({jax.default_backend()})", flush=True)
    t0 = time.time()
    result = PARTS[args.part](args)
    path = Path(args.out) / f"stochastic_{args.part}.json"
    path.write_text(json.dumps(dict(part=args.part, metadata=metadata(), settings=vars(args), wall_start=t0,
                                    wall_end=time.time(), result=result), indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
